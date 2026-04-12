import torch
import torch.nn as nn
from torch_scatter import scatter_add


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=20):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((offset[1] - offset[0]) ** 2 + 1e-12)

    def forward(self, dist):
        dist = dist.view(-1, 1)
        diff = dist - self.offset.view(1, -1)
        return torch.exp(self.coeff * diff.pow(2))


class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, dist):
        """
        dist: (E,)
        return: (E,)
        """
        out = 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0)
        out = out * (dist <= self.cutoff).float()
        return out


class PaiNNInteraction(nn.Module):
    """
    Stable PaiNN interaction block.

    Inputs:
        q          : (N, D)
        mu         : (N, 3, D)
        edge_index : (2, E)
        edge_vec   : (E, 3)
        rbf        : (E, R)
        cutoff     : (E,)

    Outputs:
        q, mu
    """
    def __init__(self, hidden_dim, rbf_dim, residual_scale=0.1, dir_eps=1e-2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual_scale = residual_scale
        self.dir_eps = dir_eps

        self.filter_net = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )

        self.scalar_proj = nn.Linear(hidden_dim, 3 * hidden_dim)

    def forward(self, q, mu, edge_index, edge_vec, rbf, cutoff):
        src, dst = edge_index

        # safer direction normalization
        dist = torch.norm(edge_vec, dim=-1, keepdim=True)                    # (E,1)
        dist_safe = torch.clamp(dist, min=self.dir_eps)                      # (E,1)
        dir_ij = edge_vec / dist_safe                                        # (E,3)

        # filter
        W = self.filter_net(rbf) * cutoff.unsqueeze(-1)                      # (E,3D)

        # scalar source
        x = self.scalar_proj(q[src]) * W                                     # (E,3D)
        dq_ij, dmu_r_ij, dmu_mu_ij = torch.chunk(x, 3, dim=-1)               # each (E,D)

        # degree-normalized aggregation (very important for stability)
        num_nodes = q.size(0)
        deg = scatter_add(
            torch.ones_like(dst, dtype=q.dtype),
            dst,
            dim=0,
            dim_size=num_nodes
        ).clamp_min(1.0)                                                     # (N,)

        # scalar aggregation
        dq = scatter_add(dq_ij, dst, dim=0, dim_size=num_nodes)              # (N,D)
        dq = dq / deg.unsqueeze(-1)

        # vector aggregation
        dmu_from_r = dir_ij.unsqueeze(-1) * dmu_r_ij.unsqueeze(1)            # (E,3,D)
        dmu_from_mu = mu[src] * dmu_mu_ij.unsqueeze(1)                       # (E,3,D)
        dmu = scatter_add(
            dmu_from_r + dmu_from_mu,
            dst,
            dim=0,
            dim_size=num_nodes
        )                                                                    # (N,3,D)
        dmu = dmu / deg.view(-1, 1, 1)

        # residual scaling
        q = q + self.residual_scale * dq
        mu = mu + self.residual_scale * dmu
        return q, mu


class PaiNNMixing(nn.Module):
    """
    Stable PaiNN mixing block.

    Inputs:
        q  : (N, D)
        mu : (N, 3, D)

    Outputs:
        q, mu
    """
    def __init__(
        self,
        hidden_dim,
        residual_scale=0.1,
        mu_limit=5.0,
        q_limit=20.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual_scale = residual_scale
        self.mu_limit = mu_limit
        self.q_limit = q_limit

        self.mu_channel_proj = nn.Linear(hidden_dim, 2 * hidden_dim)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )

        self.q_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _norm_limit_vector(mu, limit, eps=1e-6):
        """
        Equivariant vector limiter.
        mu: (N,3,D)
        """
        norm = torch.norm(mu, dim=1, keepdim=True) + eps                     # (N,1,D)
        scale = torch.clamp(limit / norm, max=1.0)
        return mu * scale

    @staticmethod
    def _norm_limit_scalar(q, limit, eps=1e-6):
        """
        Soft scalar feature limiter.
        q: (N,D)
        """
        norm = torch.norm(q, dim=-1, keepdim=True) + eps                     # (N,1)
        scale = torch.clamp(limit / norm, max=1.0)
        return q * scale

    def forward(self, q, mu):
        mu_mix = self.mu_channel_proj(mu)                                    # (N,3,2D)
        mu_v, mu_w = torch.chunk(mu_mix, 2, dim=-1)                          # each (N,3,D)

        mu_v_norm = torch.sqrt((mu_v ** 2).sum(dim=1) + 1e-6)                # (N,D)

        ctx = torch.cat([self.q_norm(q), mu_v_norm], dim=-1)                 # (N,2D)
        x = self.scalar_mlp(ctx)                                             # (N,3D)
        dq_intra, dmu_intra, gate = torch.chunk(x, 3, dim=-1)                # each (N,D)

        # bound the dangerous multiplicative terms
        gate = torch.tanh(gate)
        dmu_intra = torch.tanh(dmu_intra)

        # scalar update
        dq = dq_intra + gate * (mu_v * mu_w).sum(dim=1)                      # (N,D)

        # vector update
        dmu = mu_w * dmu_intra.unsqueeze(1)                                  # (N,3,D)

        # residual update
        q_new = q + self.residual_scale * dq
        mu_new = mu + self.residual_scale * dmu

        # limit both scalar and vector branches
        q = self._norm_limit_scalar(q_new, self.q_limit)
        mu = self._norm_limit_vector(mu_new, self.mu_limit)

        return q, mu


class PaiNN(nn.Module):
    """
    Stable PaiNN encoder for diffusion backbones.

    Expected call:
        v_out, s_out = model(s, v, edge_index, edge_vec, t_emb=t_emb, return_s=True)

    Inputs:
        s          : (N, D)
        v          : (N, 3, D) or (N, D, 3)
        edge_index : (2, E)
        edge_vec   : (E, 3)
        t_emb      : (N, D), optional
    """
    def __init__(
        self,
        num_feat,
        cut_off=5.0,
        n_rbf=20,
        num_interactions=3,
        out_channels=None,
        num_nodes=0,
        residual_scale=0.1,
        mu_limit=5.0,
        q_limit=20.0,
        **kwargs
    ):
        super().__init__()
        self.num_feat = num_feat
        self.cut_off = cut_off
        self.n_rbf = n_rbf
        self.num_interactions = num_interactions
        self.out_channels = out_channels if out_channels is not None else num_feat

        self.rbf = GaussianSmearing(0.0, cut_off, n_rbf)
        self.cutoff_fn = CosineCutoff(cut_off)

        self.interactions = nn.ModuleList([
            PaiNNInteraction(
                hidden_dim=num_feat,
                rbf_dim=n_rbf,
                residual_scale=residual_scale,
                dir_eps=1e-2,
            )
            for _ in range(num_interactions)
        ])

        self.mixings = nn.ModuleList([
            PaiNNMixing(
                hidden_dim=num_feat,
                residual_scale=residual_scale,
                mu_limit=mu_limit,
                q_limit=q_limit,
            )
            for _ in range(num_interactions)
        ])

        self.t_proj = nn.Sequential(
            nn.Linear(num_feat, num_feat),
            nn.SiLU(),
            nn.Linear(num_feat, num_feat)
        )

        if self.out_channels != self.num_feat:
            self.s_out_proj = nn.Linear(self.num_feat, self.out_channels)
            self.v_out_proj = nn.Linear(self.num_feat, self.out_channels, bias=False)
        else:
            self.s_out_proj = nn.Identity()
            self.v_out_proj = nn.Identity()

        self.out_norm_q = nn.LayerNorm(
            self.out_channels if self.out_channels != self.num_feat else self.num_feat
        )

        self._stable_init()

    def _stable_init(self):
        """
        Conservative initialization for diffusion stability.
        """
        with torch.no_grad():
            # timestep projection: small init
            for m in self.t_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # interaction / mixing last layers: tiny init
            for interaction in self.interactions:
                nn.init.xavier_uniform_(interaction.filter_net[-1].weight, gain=1e-3)
                nn.init.zeros_(interaction.filter_net[-1].bias)

                nn.init.xavier_uniform_(interaction.scalar_proj.weight, gain=0.5)
                if interaction.scalar_proj.bias is not None:
                    nn.init.zeros_(interaction.scalar_proj.bias)

            for mixing in self.mixings:
                nn.init.xavier_uniform_(mixing.scalar_mlp[-1].weight, gain=1e-3)
                nn.init.zeros_(mixing.scalar_mlp[-1].bias)

                nn.init.xavier_uniform_(mixing.mu_channel_proj.weight, gain=0.5)
                if mixing.mu_channel_proj.bias is not None:
                    nn.init.zeros_(mixing.mu_channel_proj.bias)

            # output heads: near zero init
            if not isinstance(self.s_out_proj, nn.Identity):
                nn.init.zeros_(self.s_out_proj.weight)
                if self.s_out_proj.bias is not None:
                    nn.init.zeros_(self.s_out_proj.bias)

            if not isinstance(self.v_out_proj, nn.Identity):
                nn.init.zeros_(self.v_out_proj.weight)

    def _ensure_v_shape(self, v):
        """
        Canonical vector layout: (N, 3, D)
        Accept:
            (N, 3, D)
            (N, D, 3)
        """
        if v.dim() != 3:
            raise ValueError(f"PaiNN expects 3D vector feature, got shape {tuple(v.shape)}")

        if v.size(1) == 3 and v.size(2) == self.num_feat:
            return v

        if v.size(1) == self.num_feat and v.size(2) == 3:
            return v.transpose(1, 2).contiguous()

        raise ValueError(
            f"Unsupported vector shape {tuple(v.shape)} for num_feat={self.num_feat}. "
            f"Expected (N, 3, D) or (N, D, 3)."
        )

    def forward(self, s, v, edge_index, edge_vec, t_emb=None, return_s=False):
        mu = self._ensure_v_shape(v)                                          # (N,3,D)
        q = s                                                                  # (N,D)

        if t_emb is not None:
            q = q + self.t_proj(t_emb)

        dist = torch.norm(edge_vec, dim=-1)                                    # (E,)
        rbf = self.rbf(dist)                                                   # (E,R)
        cutoff = self.cutoff_fn(dist)                                          # (E,)

        for interaction, mixing in zip(self.interactions, self.mixings):
            q, mu = interaction(q, mu, edge_index, edge_vec, rbf, cutoff)
            q, mu = mixing(q, mu)

        q = self.s_out_proj(q)

        if not isinstance(self.v_out_proj, nn.Identity):
            mu = self.v_out_proj(mu)

        q = self.out_norm_q(q)

        if return_s:
            return mu, q
        return mu