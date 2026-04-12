import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from utils.chem import BOND_TYPES
from .diffusion import get_timestep_embedding, get_beta_schedule
from ..common import MultiLayerPerceptron, _extend_to_radius_graph
from ..encoder import SchNetEncoder, get_edge_encoder
from ..encoder.painn import PaiNN
from ..geometry import get_distance


# ============================================================
# Utilities
# ============================================================

def _cosine_cutoff_weight(d: torch.Tensor, r0: float, r1: float, w_min: float = 1e-4) -> torch.Tensor:
    w = torch.zeros_like(d)
    if r1 <= r0:
        return torch.clamp(torch.ones_like(d), min=w_min, max=1.0)

    mask1 = d <= r0
    mask2 = (d > r0) & (d < r1)

    w[mask1] = 1.0
    w[mask2] = 0.5 * (1.0 + torch.cos(torch.pi * (d[mask2] - r0) / (r1 - r0)))
    return torch.clamp(w, min=w_min, max=1.0)


def _make_edge_weight(d: torch.Tensor, config) -> torch.Tensor:
    scheme = getattr(config, "w_scheme", "cosine")
    r0 = float(getattr(config, "w_r0", 2.0))
    r1 = float(getattr(config, "w_r1", 10.0))
    w_min = float(getattr(config, "w_min", 1.0e-4))

    if scheme == "cosine":
        return _cosine_cutoff_weight(d, r0, r1, w_min=w_min)

    return torch.clamp(torch.ones_like(d), min=w_min, max=1.0)


def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=p, keepdim=True)
    denom = torch.where(norm > limit, limit / (norm + 1e-12), torch.ones_like(norm))
    return vec * denom


# ============================================================
# Canonical PaiNN-based MDM (final recommended version)
# ============================================================

class MDMFullDP(nn.Module):
    """
    Final recommended design:

    - Global PaiNN branch (radius/cutoff graph):
        predicts direct equivariant coordinate epsilon + global node noise.

    - Local PaiNN branch (bond graph only):
        predicts node noise only.
        It does NOT predict 3D epsilon.

    This design is more compatible with canonical PaiNN:
    the vector branch is used where geometric neighborhoods are rich enough,
    while the chemically sparse local branch focuses on scalar semantics.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.atom_type_input_dim = config.num_atom if 'num_atom' in config else 6
        self.atom_out_dim = config.num_atom if 'num_atom' in config else 6
        self.time_emb = True
        self.vae_context = config.vae_context if 'vae_context' in config else False
        self.context = config.context if 'context' in config else []
        self.model_type = config.type

        # -------------------------------------------------
        # timestep embedding
        # -------------------------------------------------
        if self.time_emb:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim * 4),
            ])
            self.temb_proj = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        # kept only for compatibility with original project structure
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)

        # ============================================================
        # Ablation switches
        # ============================================================
        self.use_edge_weighting = bool(getattr(config, "use_edge_weighting", True))
        self.local_graph_type = getattr(config, "local_graph_type", "bond")   # "bond" or "radius"
        self.predict_local_pos = bool(getattr(config, "predict_local_pos", False))
        self.use_zero_init = bool(getattr(config, "use_zero_init", True))

        # radius-local graph cutoff for ablation
        self.local_radius_cutoff = float(getattr(config, "local_radius_cutoff", 2.2))
        self.local_pos_weight = float(getattr(config, "local_pos_weight", 1.0))

        # -------------------------------------------------
        # optional VAE context
        # -------------------------------------------------
        if self.vae_context:
            self.context_encoder = SchNetEncoder(
                hidden_channels=config.hidden_dim,
                num_filters=config.hidden_dim,
                num_interactions=config.num_convs,
                edge_channels=self.edge_encoder_global.out_channels,
                cutoff=10,
                smooth=config.smooth_conv,
                input_dim=self.atom_type_input_dim,
                time_emb=False,
                context=True,
            )
            self.atom_type_input_dim = self.atom_type_input_dim * 2

        if self.context is not None:
            self.atom_type_input_dim = self.atom_type_input_dim + len(self.context)

        self.atom_proj = nn.Linear(self.atom_type_input_dim, config.hidden_dim)

        # -------------------------------------------------
        # PaiNN backbones
        # -------------------------------------------------
        self.encoder_global = PaiNN(
            num_feat=config.hidden_dim,
            out_channels=config.hidden_dim,
            num_nodes=0,
            cut_off=float(getattr(config, "painn_cutoff_global", getattr(config, "cutoff", 5.0))),
            n_rbf=int(getattr(config, "painn_rbf_dim", getattr(config, "n_rbf", 20))),
            num_interactions=int(getattr(config, "num_convs", 3)),
        )
        self.encoder_local = PaiNN(
            num_feat=config.hidden_dim,
            out_channels=config.hidden_dim,
            num_nodes=0,
            cut_off=float(getattr(config, "painn_cutoff_local", getattr(config, "cutoff", 5.0))),
            n_rbf=int(getattr(config, "painn_rbf_dim", getattr(config, "n_rbf", 20))),
            num_interactions=int(getattr(config, "num_convs_local", 3)),
        )

        # Global branch keeps the only coordinate head
        self.pos_global_head = nn.Linear(config.hidden_dim, 1, bias=False)

        # Node-noise heads
        self.grad_global_node_mlp = MultiLayerPerceptron(
            config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, self.atom_out_dim],
            activation=config.mlp_act,
        )
        self.grad_local_node_mlp = MultiLayerPerceptron(
            config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, self.atom_out_dim],
            activation=config.mlp_act,
        )

        self.model_global = nn.ModuleList([
            self.encoder_global,
            self.pos_global_head,
            self.grad_global_node_mlp,
        ])
        self.model_local = nn.ModuleList([
            self.encoder_local,
            self.grad_local_node_mlp,
        ])

        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        alphas = (1.0 - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)
        
        # ============================================================
        # 【教授的终极防线 1】：全局输出头零初始化
        # ============================================================
        with torch.no_grad():
            # 1. 坐标预测头零初始化（最关键的一步）
            nn.init.zeros_(self.pos_global_head.weight)
            
            # 2. 节点特征预测头 (MLP) 最后一层零初始化
            # 反向遍历 MLP，找到最后一层 Linear 并清零
            for m in reversed(list(self.grad_global_node_mlp.modules())):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    break
                    
            for m in reversed(list(self.grad_local_node_mlp.modules())):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    break

    # =====================================================
    # Internal helpers
    # =====================================================
    def _build_temb(self, time_step, batch):
        if not self.time_emb:
            return None
        nonlinearity = nn.ReLU()
        temb = get_timestep_embedding(time_step, self.config.hidden_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        temb = self.temb_proj(nonlinearity(temb))
        return temb.index_select(0, batch)

    def _vector_head(self, mu, head):
        if mu.dim() != 3 or mu.size(1) != 3:
            raise ValueError(f"Expected mu with shape (N,3,H), got {tuple(mu.shape)}")
        return head(mu).squeeze(-1)

    def _center_eps(self, eps, batch):
        return center_pos(eps, batch)

    # =====================================================
    # Core network
    # =====================================================
    def net(
        self,
        atom_type,
        pos,
        bond_index,
        bond_type,
        batch,
        time_step,
        edge_index=None,
        edge_type=None,
        edge_length=None,
        context=None,
        vae_noise=None,
    ):
        # -------------------------------------------------
        # 1) optional VAE context
        # -------------------------------------------------
        if self.vae_context:
            if self.training:
                edge_length_ctx = get_distance(pos, bond_index).unsqueeze(-1)
                m, log_var = self.context_encoder(
                    z=atom_type,
                    edge_index=bond_index,
                    edge_length=edge_length_ctx,
                    edge_attr=None,
                    embed_node=False,
                )
                std = torch.exp(log_var * 0.5)
                z = torch.randn_like(log_var)
                ctx = m + std * z
                atom_type = torch.cat([atom_type, ctx], dim=1)
                kl_loss = 0.5 * torch.sum(torch.exp(log_var) + m ** 2 - 1.0 - log_var)
            else:
                ctx = torch.randn_like(atom_type) if vae_noise is None else vae_noise
                atom_type = torch.cat([atom_type, ctx], dim=1)
                kl_loss = 0
        else:
            kl_loss = 0

        # -------------------------------------------------
        # 2) optional property context
        # -------------------------------------------------
        if self.context is not None and len(self.context) > 0:
            atom_type = torch.cat([atom_type, context], dim=1)

        # -------------------------------------------------
        # 3) time embedding
        # -------------------------------------------------
        t_emb_node = self._build_temb(time_step, batch)

        # -------------------------------------------------
        # 4) graph build
        # -------------------------------------------------
        if edge_index is None or edge_type is None or edge_length is None:
            bond_type_ones = torch.ones(bond_index.size(1), dtype=torch.long, device=bond_index.device)
            edge_index, edge_type = _extend_to_radius_graph(
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type_ones,
                batch=batch,
                cutoff=self.config.cutoff,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)

        # Global branch: full radius graph
        d_all = edge_length.squeeze(-1)

        if self.use_edge_weighting:
            w_global = _make_edge_weight(d_all, self.config)
        else:
            w_global = torch.ones_like(d_all)

        # Local branch: bond graph only (strict chemistry-local graph)
        local_edge_mask = is_bond(edge_type)
        if local_edge_mask.sum() == 0:
            # rare safeguard for odd batches
            local_edge_mask = torch.logical_and(is_radius_edge(edge_type), d_all < 2.2)
        if local_edge_mask.sum() == 0:
            local_edge_mask = torch.ones_like(edge_type, dtype=torch.bool)

        w_local = w_global[local_edge_mask]

        # -------------------------------------------------
        # 5) scalar / vector init
        # -------------------------------------------------
        s0 = self.atom_proj(atom_type)
        v0 = torch.zeros(
            s0.size(0), 3, self.config.hidden_dim,
            device=s0.device, dtype=s0.dtype
        )

        # -------------------------------------------------
        # 6) edge vectors
        # -------------------------------------------------
        edge_vec_all = pos.index_select(0, edge_index[0]) - pos.index_select(0, edge_index[1])

        edge_index_local = edge_index[:, local_edge_mask]
        edge_vec_local = pos.index_select(0, edge_index_local[0]) - pos.index_select(0, edge_index_local[1])

        # -------------------------------------------------
        # 7) Global PaiNN: predicts coordinate epsilon + node noise
        # -------------------------------------------------
        mu_g, q_g = self.encoder_global(
            s0, v0, edge_index, edge_vec_all, t_emb=t_emb_node, return_s=True
        )
        eps_global = self._vector_head(mu_g, self.pos_global_head)
        eps_global = self._center_eps(eps_global, batch)

        if self.use_edge_weighting:
            node_w_global = scatter_mean(
                w_global.unsqueeze(-1), edge_index[0], dim=0, dim_size=pos.size(0)
            ).clamp_min(1e-4)
        else:
            node_w_global = torch.ones(pos.size(0), 1, device=pos.device, dtype=pos.dtype)

        eps_global = eps_global * node_w_global

        node_score_global = self.grad_global_node_mlp(q_g)

        # -------------------------------------------------
        # 8) Local PaiNN: predicts node noise only
        # -------------------------------------------------
        mu_l, q_l = self.encoder_local(
            s0, v0, edge_index_local, edge_vec_local, t_emb=t_emb_node, return_s=True
        )
        node_score_local = self.grad_local_node_mlp(q_l)

        if self.vae_context:
            return (
                eps_global,
                node_score_global,
                node_score_local,
                edge_index,
                edge_type,
                edge_length,
                local_edge_mask,
                w_global,
                w_local,
                kl_loss,
            )
        else:
            return (
                eps_global,
                node_score_global,
                node_score_local,
                edge_index,
                edge_type,
                edge_length,
                local_edge_mask,
                w_global,
                w_local,
            )

    # =====================================================
    # Training forward
    # =====================================================
    def forward(
        self,
        batch,
        context=None,
        return_unreduced_loss=False,
        return_unreduced_edge_loss=False,
        extend_order=True,
        extend_radius=True,
        is_sidechain=None, ):

        atom_type = batch.atom_feat_full.float()
        pos = batch.pos
        bond_index = batch.edge_index
        bond_type = batch.edge_type
        batch_idx = batch.batch

        node2graph = batch_idx
        num_graphs = node2graph[-1] + 1

        time_step = torch.randint(
            0, self.num_timesteps,
            size=(num_graphs // 2 + 1,),
            device=pos.device
        )
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1],
            dim=0
        )[:num_graphs]

        a = self.alphas.index_select(0, time_step)
        a_node = a.index_select(0, node2graph).unsqueeze(-1)

        geom_a_threshold = float(getattr(self.config, "geom_a_threshold", 0.25))
        geom_graph_mask = (a > geom_a_threshold).float()
        geom_weight = geom_graph_mask.mean()

        # strict DDPM perturbation for positions
        pos_centered = center_pos(pos, batch_idx)
        pos_noise = center_pos(torch.randn_like(pos_centered), batch_idx)
        pos_perturbed = a_node.sqrt() * pos_centered + (1.0 - a_node).sqrt() * pos_noise

        # atom perturbation
        atom_noise = torch.randn_like(atom_type)
        atom_type_scaled = torch.cat(
            [atom_type[:, :-1] / 4, atom_type[:, -1:] / 10],
            dim=1
        )
        atom_perturbed = a_node.sqrt() * atom_type_scaled + (1.0 - a_node).sqrt() * atom_noise

        vae_noise = torch.randn_like(atom_type_scaled)

        net_out = self.net(
            atom_type=atom_perturbed,
            pos=pos_perturbed,
            bond_index=bond_index,
            bond_type=bond_type,
            batch=batch_idx,
            time_step=time_step,
            context=context,
            vae_noise=vae_noise,
        )

        if self.vae_context:
            eps_global, node_score_global, node_score_local, \
            edge_index, edge_type, edge_length, local_edge_mask, w_global, w_local = net_out[:-1]
        else:
            eps_global, node_score_global, node_score_local, \
            edge_index, edge_type, edge_length, local_edge_mask, w_global, w_local = net_out

        # -------------------------------------------------
        # 1) position epsilon loss: GLOBAL ONLY
        # -------------------------------------------------
        target_eps = pos_noise
        loss_global = ((eps_global - target_eps) ** 2).sum(dim=-1, keepdim=True)

        # -------------------------------------------------
        # 2) node-noise losses: global + local
        # -------------------------------------------------
        loss_node_global = ((node_score_global - atom_noise) ** 2).sum(dim=-1, keepdim=True)
        loss_node_local = ((node_score_local - atom_noise) ** 2).sum(dim=-1, keepdim=True)

        # -------------------------------------------------
        # 3) reconstruct x0 from global epsilon only
        # -------------------------------------------------
        sqrt_a = a_node.sqrt().clamp_min(1e-8)
        sqrt_one_minus_a = (1.0 - a_node).sqrt().clamp_min(1e-8)

        x0_hat_global = (pos_perturbed - sqrt_one_minus_a * eps_global) / sqrt_a
        x0_hat_global = center_pos(x0_hat_global, batch_idx)

        # -------------------------------------------------
        # 4) bond-local geometry consistency based on global reconstruction
        # -------------------------------------------------
        edge_index_local = edge_index[:, local_edge_mask]
        if edge_index_local.size(1) > 0:
            d0_local_gt = get_distance(pos_centered, edge_index_local).unsqueeze(-1)
            d0_local_pred = get_distance(x0_hat_global, edge_index_local).unsqueeze(-1)
            local_dist_loss = (d0_local_pred - d0_local_gt) ** 2
            local_w = w_local.unsqueeze(-1)
            local_dist_loss = (local_dist_loss * local_w).sum() / (local_w.sum() + 1e-8)
        else:
            local_dist_loss = torch.tensor(0.0, device=pos.device)

        # -------------------------------------------------
        # 5) bond length consistency from global reconstruction
        # -------------------------------------------------
        bond_mask = is_bond(edge_type)
        edge_index_bond = edge_index[:, bond_mask]
        if edge_index_bond.size(1) > 0:
            d0_bond_gt = get_distance(pos_centered, edge_index_bond).unsqueeze(-1)
            d0_bond_pred = get_distance(x0_hat_global, edge_index_bond).unsqueeze(-1)
            bond_loss = ((d0_bond_pred - d0_bond_gt) ** 2).mean()
        else:
            bond_loss = torch.tensor(0.0, device=pos.device)

        # -------------------------------------------------
        # 6) global geometry loss on bond + radius edges only
        # -------------------------------------------------
        global_geom_mask = torch.logical_or(is_bond(edge_type), is_radius_edge(edge_type))
        if global_geom_mask.sum() > 0:
            edge_index_global_geom = edge_index[:, global_geom_mask]
            w_global_geom = w_global[global_geom_mask]
            d0_global_gt = get_distance(pos_centered, edge_index_global_geom).unsqueeze(-1)
            d0_global_pred = get_distance(x0_hat_global, edge_index_global_geom).unsqueeze(-1)
            global_dist_loss = (d0_global_pred - d0_global_gt) ** 2
            global_w = w_global_geom.unsqueeze(-1)
            global_dist_loss = (global_dist_loss * global_w).sum() / (global_w.sum() + 1e-8)
        else:
            global_dist_loss = torch.tensor(0.0, device=pos.device)

        # -------------------------------------------------
        # 7) repulsion from global reconstruction
        # -------------------------------------------------
        d_pred_all = get_distance(x0_hat_global, edge_index).clamp_min(1e-6)
        nonbond_mask = ~is_bond(edge_type)
        if nonbond_mask.sum() > 0:
            d_rep = d_pred_all[nonbond_mask]
            repulsion_loss = torch.where(
                d_rep < 0.85,
                0.1 / (d_rep ** 2 + 1e-7),
                torch.zeros_like(d_rep)
            ).mean()
        else:
            repulsion_loss = torch.tensor(0.0, device=pos.device)

        # -------------------------------------------------
        # 8) weighted total loss
        # -------------------------------------------------
        lambda_local = getattr(self.config, "lambda_local_dist", 0.50)
        lambda_bond = getattr(self.config, "lambda_bond_dist", 1.00)
        lambda_global = getattr(self.config, "lambda_global_dist", 0.10)
        lambda_rep = getattr(self.config, "lambda_repulsion", 0.02)

        main_loss = loss_global + loss_node_global + loss_node_local
        geom_loss_scalar = geom_weight * (
            lambda_local * local_dist_loss
            + lambda_bond * bond_loss
            + lambda_global * global_dist_loss
            + lambda_rep * repulsion_loss
        )
        geom_loss = torch.ones_like(main_loss) * geom_loss_scalar

        if self.vae_context:
            vae_kl_loss = net_out[-1]
            loss = main_loss + geom_loss + vae_kl_loss
        else:
            loss = main_loss + geom_loss

        if return_unreduced_edge_loss:
            return None
        elif return_unreduced_loss:
            # keep train.py-compatible slot count by returning a zero placeholder
            zero_like_pos = torch.zeros_like(loss_global)
            if self.vae_context:
                return (
                    loss,
                    loss_global,
                    zero_like_pos,
                    loss_node_global,
                    loss_node_local,
                    local_dist_loss,
                    bond_loss,
                    global_dist_loss,
                    repulsion_loss,
                    vae_kl_loss,
                )
            return (
                loss,
                loss_global,
                zero_like_pos,
                loss_node_global,
                loss_node_local,
                local_dist_loss,
                bond_loss,
                global_dist_loss,
                repulsion_loss,
            )
        else:
            return loss

    # =====================================================
    # Sampling
    # =====================================================
    def langevin_dynamics_sample(
        self,
        atom_type,
        pos_init,
        bond_index,
        bond_type,
        batch,
        num_graphs,
        context,
        n_steps=100,
        step_lr=0.0000010,
        clip=1000,
        clip_node=3.0,
        clip_local=None,
        clip_pos=None,
        clip_atom_type=8.0,
        global_start_sigma=float('inf'),
        local_start_sigma=float('inf'),
        w_global_pos=1,
        w_global_node=1,
        w_local_pos=1,
        w_local_node=1,
        **kwargs,
    ):
        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        with torch.no_grad():
            seq = range(self.num_timesteps - n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            pos = center_pos(torch.randn_like(pos_init), batch)
            atom_type = torch.randn_like(atom_type)
            vae_noise = torch.zeros_like(atom_type).uniform_(-1, +1)

            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)

                net_out = self.net(
                    atom_type=atom_type,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    context=context,
                    vae_noise=vae_noise,
                )

                if self.vae_context:
                    eps_global, node_score_global, node_score_local, \
                    edge_index, edge_type, edge_length, local_edge_mask, w_global, w_local = net_out[:-1]
                else:
                    eps_global, node_score_global, node_score_local, \
                    edge_index, edge_type, edge_length, local_edge_mask, w_global, w_local = net_out

                # only global branch predicts coordinate epsilon
                if sigmas[i] < global_start_sigma:
                    eps_pos = w_global_pos * clip_norm(eps_global, limit=clip)
                else:
                    eps_pos = torch.zeros_like(pos)
                    node_score_global = torch.zeros_like(node_score_global)

                if sigmas[i] < local_start_sigma:
                    eps_node_local = node_score_local
                else:
                    eps_node_local = torch.zeros_like(node_score_local)

                eps_node = w_global_node * node_score_global + w_local_node * eps_node_local
                eps_node = clip_norm(eps_node, limit=clip_node)

                sampling_type = kwargs.get("sampling_type", 'ddpm_noisy')
                noise = center_pos(torch.randn_like(pos), batch)
                noise_node = torch.randn_like(atom_type)

                b = self.betas
                t_scalar = t[0]
                next_t = (torch.ones(1) * j).to(pos.device)
                at = compute_alpha(b, t_scalar.long())
                at_next = compute_alpha(b, next_t.long())

                if sampling_type == 'generalized':
                    eta = kwargs.get("eta", 1.)
                    c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()

                    x0_pos = (pos - (1 - at).sqrt() * eps_pos) / at.sqrt()
                    pos_next = at_next.sqrt() * x0_pos + c1 * noise + c2 * eps_pos

                    x0_node = (atom_type - (1 - at).sqrt() * eps_node) / at.sqrt()
                    atom_next = at_next.sqrt() * x0_node + c1 * noise_node + c2 * eps_node

                elif sampling_type == 'ddpm_noisy':
                    beta_t = 1 - at / at_next
                    alpha_t = 1 - beta_t

                    coef1 = 1 / alpha_t.sqrt()
                    coef2 = beta_t / (1 - at).sqrt()

                    mean_pos = coef1 * (pos - coef2 * eps_pos)
                    mask = 1 - (t_scalar == 0).float()
                    logvar = beta_t.log()
                    pos_next = mean_pos + mask * torch.exp(0.5 * logvar) * noise

                    mean_node = coef1 * (atom_type - coef2 * eps_node)
                    atom_next = mean_node + mask * torch.exp(0.5 * logvar) * noise_node

                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    score_pos = -eps_pos / (1 - at).sqrt()
                    score_node = -eps_node / (1 - at).sqrt()
                    pos_next = pos + step_size * score_pos / sigmas[i] + noise * torch.sqrt(step_size * 2)
                    atom_next = atom_type + step_size * score_node / sigmas[i] + noise_node * torch.sqrt(step_size * 2)
                else:
                    raise ValueError(f"Unknown sampling_type: {sampling_type}")

                pos = pos_next
                atom_type = atom_next

                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()

                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)

                pos_traj.append(pos.clone().cpu())
                atom_traj.append(atom_type.clone().cpu())

            atom_type = torch.cat([atom_type[:, :-1] * 4, atom_type[:, -1:] * 10], dim=1)

        return pos, pos_traj, atom_type, atom_traj
