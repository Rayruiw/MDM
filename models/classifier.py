# ----------------------- 修复后的导入区 -----------------------
from typing import List
import torch
from torch import nn
from einops import rearrange

# 用 torch_scatter 的 scatter 做聚合，避免命名冲突
from torch_scatter import scatter as scatter_reduce

# pytorch geometric
try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing, radius_graph
    from torch_geometric.nn.norm import LayerNorm as PygLayerNorm
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
    PYG_AVAILABLE = True
except Exception:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PYG_AVAILABLE = False
    # 为了类型提示不报错
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

# 你项目里提供的公用模块（包含 SiLU / exists / CoorsNorm / fourier_encode_dist / embedd_token / GlobalLinearAttention 等）
#from .egnn_pytorch import *


# ----------------------- 稀疏注意力（修复） -----------------------
class AttentionSparse(Attention):
    def __init__(self, **kwargs):
        """Wrap Attention 以适配稀疏（按 batch 分块）"""
        super().__init__(**kwargs)

    def sparse_forward(self, x, context, batch=None, batch_uniques=None, mask=None):
        assert (batch is not None) or (batch_uniques is not None), "block-sparse 需要 batch 或 (uniques, counts)"
        if batch_uniques is None:
            uniques, counts = torch.unique(batch, return_counts=True)
        else:
            uniques, counts = batch_uniques

        # 单图：走致密
        if uniques.numel() == 1:
            x_, ctx_ = map(lambda t: rearrange(t, 'n d -> () n d'), (x, context))
            out = self.forward(x_, ctx_, mask=None).squeeze(0)
            return out

        # 多图：逐图块 dense 运行，再拼接
        outs = []
        start = 0
        for c in counts.tolist():
            x_blk = x[start:start + c]
            c_blk = context[start:start + c]
            x_b, c_b = map(lambda t: rearrange(t, 'n d -> () n d'), (x_blk, c_blk))
            o_b = self.forward(x_b, c_b, mask=None).squeeze(0)
            outs.append(o_b)
            start += c
        return torch.cat(outs, dim=0)


# ----------------------- 稀疏版全局线性注意力（小修） -----------------------
class GlobalLinearAttentionSparse(nn.Module):
    def __init__(self, *, dim, heads=8, dim_head=64):
        super().__init__()
        # 修正拼写：torch_geometric
        self.norm_seq = PygLayerNorm(dim)
        self.norm_queries = PygLayerNorm(dim)
        self.attn1 = AttentionSparse(dim=dim, heads=heads, dim_head=dim_head)
        self.attn2 = AttentionSparse(dim=dim, heads=heads, dim_head=dim_head)

        self.ff_norm = PygLayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, batch=None, batch_uniques=None, mask=None):
        res_x, res_q = x, queries
        x = self.norm_seq(x, batch=batch) if PYG_AVAILABLE else x
        queries = self.norm_queries(queries, batch=batch) if PYG_AVAILABLE else queries

        induced = self.attn1.sparse_forward(queries, x, batch=batch, batch_uniques=batch_uniques, mask=mask)
        out = self.attn2.sparse_forward(x, induced, batch=batch, batch_uniques=batch_uniques)

        x = out + res_x
        queries = induced + res_q

        x_norm = self.ff_norm(x, batch=batch) if PYG_AVAILABLE else x
        x = self.ff(x_norm) + x_norm
        return x, queries


# ----------------------- PyG 等变 GNN（小修） -----------------------
class EGNNSparse(MessagePassing):
    """
    与密集版不同，这里将边分配与计算解耦，适合局部/稀疏邻接。
    aggr: one of ["add", "mean", "max"]
    """

    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim=0,
        m_dim=16,
        fourier_features=0,
        soft_edge=0,
        norm_feats=False,
        norm_coors=False,
        norm_coors_scale_init=1e-2,
        update_feats=True,
        update_coors=True,
        dropout=0.,
        cutoff=10,
        coor_weights_clamp_value=None,
        aggr="add",
        **kwargs
    ):
        assert aggr in {'add', 'sum', 'max', 'mean'}
        assert update_feats or update_coors
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)

        # 参数
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = None
        self.cutoff = cutoff

        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 边 MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )
        self.edge_weight = (nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid())
                            if soft_edge else None)

        # 节点/坐标归一化与更新
        self.node_norm = PygLayerNorm(feats_dim) if (norm_feats and PYG_AVAILABLE) else None
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()

        self.node_mlp = (nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None)

        self.coors_mlp = (nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(m_dim * 4, 1)
        ) if update_coors else None)

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj = None,
                edge_attr: OptTensor = None, batch: Adj = None,
                angle_data: List = None, size: Size = None) -> Tensor:

        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]

        if edge_index is None:
            edge_index = radius_graph(coors, self.cutoff, batch=batch, loop=False)

        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1) if exists(edge_attr) else rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                               coors=coors, rel_coors=rel_coors,
                                               batch=batch)
        return torch.cat([coors_out, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        if self.soft_edge:
            m_ij = m_ij * self.edge_weight(m_ij)
        return m_ij

    # 兼容 PyG 2.4：手写 propagate，统一用 scatter_reduce
    def propagate(self, edge_index, size=None, **kwargs):
        if 'x' not in kwargs:
            raise ValueError("propagate() requires 'x' in kwargs")
        x        = kwargs['x']                    # (N, F)
        coors    = kwargs.get('coors', None)      # (N, 3) or None
        rel      = kwargs.get('rel_coors', None)  # (E, 3) or None
        edge_attr= kwargs.get('edge_attr', None)  # (E, Fe) or None
        batch    = kwargs.get('batch', None)      # (N,) or None

        device = x.device
        num_nodes = x.size(0)

        if not torch.is_tensor(edge_index):
            raise TypeError("edge_index must be a tensor of shape [2, E]")
        edge_index = edge_index.to(device=device, dtype=torch.long)
        i, j = edge_index[0], edge_index[1]

        x_i, x_j = x[i], x[j]

        if coors is not None and coors.size(0) != num_nodes:
            Nc = coors.size(0)
            if Nc > num_nodes:
                coors = coors[:num_nodes]
            else:
                pad = coors.new_zeros(num_nodes - Nc, *coors.shape[1:])
                coors = torch.cat([coors, pad], dim=0)

        if rel is None and coors is not None:
            rel = coors[i] - coors[j]
        if rel is not None and hasattr(self, "coors_norm") and self.coors_norm:
            rel = self.coors_norm(rel)

        if edge_attr is not None:
            m_ij = self.message(x_i=x_i, x_j=x_j, edge_attr=edge_attr)
        else:
            m_ij = self.message(x_i=x_i, x_j=x_j)

        if getattr(self, "update_coors", False):
            if not hasattr(self, "coors_mlp"):
                raise AttributeError("update_coors=True 但未定义 self.coors_mlp")
            if rel is None:
                raise ValueError("update_coors=True 需要提供 rel_coors 或 coors 以构造相对坐标")

            coor_wij = self.coors_mlp(m_ij)
            mhat_i = scatter_reduce(coor_wij * rel, i, dim=0, dim_size=num_nodes, reduce='sum')  # (N, 3)
            coors_out = (coors + mhat_i) if coors is not None else mhat_i
        else:
            coors_out = coors

        if getattr(self, "update_feats", True):
            m_i = scatter_reduce(m_ij, i, dim=0, dim_size=num_nodes, reduce='sum')

            if hasattr(self, "node_norm") and self.node_norm:
                hidden_feats = self.node_norm(x, batch) if batch is not None else self.node_norm(x)
            else:
                hidden_feats = x

            if not hasattr(self, "node_mlp"):
                raise AttributeError("update_feats=True 但未定义 self.node_mlp")
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
            hidden_out = x + hidden_out
        else:
            hidden_out = x

        return self.update((hidden_out, coors_out))

    def __repr__(self):
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__)


# ----------------------- 网络封装（关键修复 temb 兼容） -----------------------
class EGNNSparseNetwork(nn.Module):
    """
    用 EGNNSparse 叠层的网络。
    - 既兼容把时间嵌入 `temb` 拼在输入 z 的最后（z = [feats | temb]），
    - 也兼容把 temb 作为第 6 个**位置参数**传入 forward(..., batch, temb)。
    """

    def __init__(self, n_layers, feats_input_dim, feats_dim,
                 pos_dim=3, edge_attr_dim=0, m_dim=16, fourier_features=0, soft_edge=0,
                 embedding_nums=None, embedding_dims=None,
                 edge_embedding_nums=None, edge_embedding_dims=None,
                 update_coors=True, update_feats=True, norm_feats=True,
                 norm_coors=False, norm_coors_scale_init=1e-2,
                 dropout=0., coor_weights_clamp_value=None, aggr="add",
                 global_linear_attn_every=0, global_linear_attn_heads=8,
                 global_linear_attn_dim_head=64, num_global_tokens=4, recalc=0):
        super().__init__()

        embedding_nums = [] if embedding_nums is None else embedding_nums
        embedding_dims = [] if embedding_dims is None else embedding_dims
        edge_embedding_nums = [] if edge_embedding_nums is None else edge_embedding_nums
        edge_embedding_dims = [] if edge_embedding_dims is None else edge_embedding_dims

        self.n_layers = n_layers
        self.embedding_nums = embedding_nums
        self.embedding_dims = embedding_dims
        self.emb_layers = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers = nn.ModuleList()

        # embeddings
        for i in range(len(self.embedding_dims)):
            self.emb_layers.append(nn.Embedding(num_embeddings=embedding_nums[i],
                                                embedding_dim=embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1

        for i in range(len(self.edge_embedding_dims)):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings=edge_embedding_nums[i],
                                                     embedding_dim=edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1

        self.mpnn_layers = nn.ModuleList()
        self.feats_input_dim = feats_input_dim  # 仅节点原始特征宽度（不含时间）
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.edge_attr_dim = edge_attr_dim
        self.m_dim = m_dim
        self.fourier_features = fourier_features
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.norm_coors_scale_init = norm_coors_scale_init
        self.update_feats = update_feats
        self.update_coors = update_coors
        self.dropout = dropout
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.recalc = recalc

        self.has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        self.global_linear_attn_every = global_linear_attn_every
        if self.has_global_attn:
            # 修复 dim 未定义 -> 用 feats_dim
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, feats_dim))

        # 叠层
        for i in range(n_layers):
            layer = EGNNSparse(
                feats_dim=feats_dim,
                pos_dim=pos_dim,
                edge_attr_dim=edge_attr_dim,
                m_dim=m_dim,
                fourier_features=fourier_features,
                soft_edge=soft_edge,
                norm_feats=norm_feats,
                norm_coors=norm_coors,
                norm_coors_scale_init=norm_coors_scale_init,
                update_feats=update_feats,
                update_coors=update_coors,
                dropout=dropout,
                coor_weights_clamp_value=coor_weights_clamp_value
            )
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if is_global_layer:
                # 这里的 GlobalLinearAttention 通常来自 egnn_pytorch（保持原设计）
                attn_layer = GlobalLinearAttention(dim=self.feats_dim,
                                                   heads=global_linear_attn_heads,
                                                   dim_head=global_linear_attn_dim_head)
                self.mpnn_layers.append(nn.ModuleList([layer, attn_layer]))
            else:
                self.mpnn_layers.append(layer)

        self.emblin = nn.Linear(self.feats_input_dim, feats_dim)

    def forward(self, z, pos, edge_index, edge_attr, batch, *args,
                bsize=None, recalc_edge=None, verbose=0):
        """
        z: [N, feats] 或 [N, feats + temb]；也可把 temb 作为第 6 个位置参数传进来
        """
        # 1) 取可选的 temb（优先来自 *args，第 1 个张量）
        temb_arg = None
        if len(args) > 0 and torch.is_tensor(args[0]):
            temb_arg = args[0]

        # 2) 如果 z 里拼了 temb，则拆出来
        if z.size(1) > self.feats_input_dim:
            z_feats = z[:, :self.feats_input_dim]
            temb_cat = z[:, self.feats_input_dim:]
        else:
            z_feats = z
            temb_cat = None

        # 3) 先做特征线性映射
        h = self.emblin(z_feats)

        # 4) 选择 temb 源，并做“安全兜底 + 对齐”
        temb = temb_arg if temb_arg is not None else temb_cat
        if isinstance(temb, torch.Tensor):
            if temb.numel() == 0 or temb.dim() != 2 or temb.size(0) != h.size(0) or temb.size(1) == 0:
                temb = None
            elif temb.size(1) != h.size(1):
                # 宽度不等，对齐到 h 的通道数（只在需要时创建）
                if not hasattr(self, "_temb_align"):
                    self._temb_align = nn.Linear(temb.size(1), h.size(1), bias=False).to(h.device)
                temb = self._temb_align(temb)

        if temb is not None:
            h = h + temb

        # 5) 拼回坐标，进入后续 MPNN
        x = torch.cat([pos, h], dim=1)
        x = embedd_token(x, self.embedding_dims, self.emb_layers)

        edges_need_embedding = False

        for i, layer in enumerate(self.mpnn_layers):
            # 边嵌入（如需重算）
            if edges_need_embedding:
                edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)
                edges_need_embedding = False

            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if not is_global_layer:
                x = layer(x, edge_index, edge_attr, batch=batch, size=bsize)
            else:
                # 仅将 feats 过全局注意力，再与坐标合并
                x_attn = layer[0](x[:, self.pos_dim:], edge_index, edge_attr, batch=batch, size=bsize)
                x = torch.cat((x[:, :self.pos_dim], x_attn), dim=-1)
                x = layer[-1](x, edge_index, edge_attr, batch=batch, size=bsize)

            if self.recalc and ((i % self.recalc == 0) and (i != len(self.mpnn_layers) - 1)):
                edge_index, edge_attr, _ = recalc_edge(x)
                edges_need_embedding = True

        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        return feats, coors

    def __repr__(self):
        return f'EGNNSparseNetwork of: {len(self.mpnn_layers)} layers'

    # 兼容接口：用 torch_scatter 的 scatter 作为聚合
    def aggregate(self, inputs, index, dim_size=None, reduce='add'):
        return scatter_reduce(inputs, index, dim=0, dim_size=dim_size, reduce=reduce)