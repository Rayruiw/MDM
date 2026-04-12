# types
from torch_scatter import scatter
from typing import List
import inspect
import torch
from einops import rearrange

# types
from torch_geometric.utils import scatter
# pytorch geometric

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing, radius_graph
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
except:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PYG_AVAILABLE = False

    # to stop throwing errors from type suggestions
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

from .egnn_pytorch import *


# global linear attention

class AttentionSparse(Attention):
    def __init__(self, **kwargs):
        """ Wraps the attention class to operate with pytorch-geometric inputs. """
        super(AttentionSparse, self).__init__(**kwargs)

    def sparse_forward(self, x, context, batch=None, batch_uniques=None, mask=None):
        assert batch is not None or batch_uniques is not None, "Batch/(uniques) must be passed for block_sparse_attn"
        if batch_uniques is None:
            batch_uniques = torch.unique(batch, return_counts=True)
        # only one example in batch - do dense - faster
        if batch_uniques[0].shape[0] == 1:
            x, context = map(lambda t: rearrange(t, 'h d -> () h d'), (x, context))
            return self.forward(x, context, mask=None).squeeze()  #  get rid of batch dim
        # multiple examples in batch - do block-sparse by dense loop
        else:
            x_list = []
            aux_count = 0
            for bi, n_idxs in zip(*batch_uniques):
                x_list.append(
                    self.sparse_forward(
                        x[aux_count:aux_count + n_i],
                        context[aux_count:aux_count + n_idxs],
                        batch_uniques=(bi.unsqueeze(-1), n_idxs.unsqueeze(-1))
                    )
                )
            return torch.cat(x_list, dim=0)


class GlobalLinearAttentionSparse(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64
    ):
        super().__init__()
        self.norm_seq = torch_geomtric.nn.norm.LayerNorm(dim)
        self.norm_queries = torch_geomtric.nn.norm.LayerNorm(dim)
        self.attn1 = AttentionSparse(dim, heads, dim_head)
        self.attn2 = AttentionSparse(dim, heads, dim_head)

        # can't concat pyg norms with torch sequentials
        self.ff_norm = torch_geomtric.nn.norm.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, batch=None, batch_uniques=None, mask=None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x, batch=batch), self.norm_queries(queries, batch=batch)

        induced = self.attn1.sparse_forward(queries, x, batch=batch, batch_uniques=batch_uniques, mask=mask)
        out = self.attn2.sparse_forward(x, induced, batch=batch, batch_uniques=batch_uniques)

        x = out + res_x
        queries = induced + res_queries

        x_norm = self.ff_norm(x, batch=batch)
        x = self.ff(x_norm) + x_norm
        return x, queries


#  define pytorch-geometric equivalents

class EGNNSparse(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and
        computations when the graph is locally or sparse connected).
        * aggr: one of ["add", "mean", "max"]
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
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(EGNNSparse, self).__init__(**kwargs)
        # model params
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

        #  EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1),
                                         nn.Sigmoid()
                                         ) if soft_edge else None

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1.
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None

        #  COORS
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1)
        ) if update_coors else None

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj = None,
                edge_attr: OptTensor = None, batch: Adj = None,
                angle_data: List = None, size: Size = None) -> Tensor:
        """ Inputs:
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_index: (n_edges, 2)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
            * batch: (n_points,) long tensor. specifies xloud belonging for each point
            * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
            * size: None
        """

        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        # print(edge_index.size())
        # print(edge_attr.size())
        # coors, feats = coors, h
        if edge_index is None:
            edge_index = radius_graph(coors, self.cutoff, batch=batch, loop=False)
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)
        # print(rel_dist.size())

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        # print(rel_dist.size())
        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                               coors=coors, rel_coors=rel_coors,
                                               batch=batch)
        return torch.cat([coors_out, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        if self.soft_edge:
            m_ij = m_ij * self.edge_weight(m_ij)
            # print('*************************')
        return m_ij

    def propagate(self, edge_index, size=None, **kwargs):
        """
        一版兼容 PyG 2.4 的手写 propagate：
        - 不使用 Inspector/__collect__/__check_input__
        - 以 x 的行数为权威 num_nodes，所有 scatter(..., dim_size=num_nodes)
        - 自动回退构造 rel_coors（若未提供且给了 coors）
        - 对 coors/x 做长度对齐防御（极端情况下）
        期望 kwargs:
            x:        (N, F)
            coors:    (N, 3)   可选
            rel_coors:(E, 3)   可选；若缺省且给了 coors，则用 coors[i]-coors[j]
            edge_attr:(E, Fe)  可选
            batch:    (N,)     可选
        返回：
            self.update((hidden_out, coors_out), **update_kwargs)
            其中 update_kwargs 默认为空；如果你的 update(...) 需要别的键，可以在此补充。
        """
        # ---------- 基础输入 ----------
        if 'x' not in kwargs:
            raise ValueError("propagate() requires 'x' in kwargs")
        x        = kwargs['x']                    # (N, F)
        coors    = kwargs.get('coors', None)      # (N, 3) or None
        rel      = kwargs.get('rel_coors', None)  # (E, 3) or None
        edge_attr= kwargs.get('edge_attr', None)  # (E, Fe) or None
        batch    = kwargs.get('batch', None)      # (N,) or None

        device = x.device
        # 以特征长度为权威节点数，避免 edge_index 推断丢孤立点
        num_nodes = x.size(0)
        size = (num_nodes, num_nodes)

        # ---------- 边索引与索引切片 ----------
        # 确保 long / device 一致
        if not torch.is_tensor(edge_index):
            raise TypeError("edge_index must be a tensor of shape [2, E]")
        edge_index = edge_index.to(device=device, dtype=torch.long)
        i, j = edge_index[0], edge_index[1]       # 目标/源（与 PyG MessagePassing 约定一致）

        # 取对应端点特征
        x_i, x_j = x[i], x[j]                     # (E, F)

        # ---------- 长度防御：coors 与 x 对齐 ----------
        if coors is not None and coors.size(0) != num_nodes:
            Nc = coors.size(0)
            if Nc > num_nodes:
                coors = coors[:num_nodes]
            else:
                pad = coors.new_zeros(num_nodes - Nc, *coors.shape[1:])
                coors = torch.cat([coors, pad], dim=0)

        # ---------- rel_coors 构造 / 归一 ----------
        if rel is None and coors is not None:
            # 回退：用坐标差作为相对坐标
            rel = coors[i] - coors[j]             # (E, 3)
        if rel is not None and hasattr(self, "coors_norm") and self.coors_norm:
            # 有些实现把 coors_norm 设计成一个 nn.Module / 函数
            rel = self.coors_norm(rel)

        # ---------- 计算消息 ----------
        # 注意：EGNNSparse.message(...) 的签名中若没有 coors/rel_coors，就不要传它们
        # 常见签名：message(self, x_i, x_j, edge_attr=None, **kwargs)
        if edge_attr is not None:
            m_ij = self.message(x_i=x_i, x_j=x_j, edge_attr=edge_attr)
        else:
            m_ij = self.message(x_i=x_i, x_j=x_j)  # (E, hidden)

        # ---------- 坐标更新 ----------
        if getattr(self, "update_coors", False):
            if not hasattr(self, "coors_mlp"):
                raise AttributeError("update_coors=True 但未定义 self.coors_mlp")
            if rel is None:
                raise ValueError("update_coors=True 需要提供 rel_coors 或 coors 以构造相对坐标")

            coor_wij = self.coors_mlp(m_ij)       # (E, 3) or (E, C) 与 rel 做逐元素相乘
            # 聚合到目标结点 i，强制 dim_size=num_nodes 避免长度不一致
            mhat_i = scatter(coor_wij * rel, i, dim=0, dim_size=num_nodes, reduce='sum')  # (N, 3)
            coors_out = (coors + mhat_i) if coors is not None else mhat_i
        else:
            coors_out = coors

        # ---------- 特征更新 ----------
        if getattr(self, "update_feats", True):
            # 边权/软边等在此处乘到 m_ij 上（如果你的实现需要）
            m_i = scatter(m_ij, i, dim=0, dim_size=num_nodes, reduce='sum')  # (N, hidden)

            if hasattr(self, "node_norm") and self.node_norm:
                # 一些实现把 node_norm 作为 Module，接受 (x, batch)
                hidden_feats = self.node_norm(x, batch) if batch is not None else self.node_norm(x)
            else:
                hidden_feats = x

            if not hasattr(self, "node_mlp"):
                raise AttributeError("update_feats=True 但未定义 self.node_mlp")

            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))  # (N, F)
            hidden_out = x + hidden_out
        else:
            hidden_out = x

        # ---------- 返回 update() ----------
        # 若你的 update(...) 需要更多参数，可在此补充 update_kwargs
        return self.update((hidden_out, coors_out))


    def __repr__(self):
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__)


class EGNNSparseNetwork(nn.Module):
    r"""Sample GNN model architecture that uses the EGNN-Sparse
        message passing layer to learn over point clouds.
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * ... : same interpretation as the base layer.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed.
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed.
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed.
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed.
        * recalc: int. Recalculate edge feats every `recalc` MPNN layers. 0 for no recalc
        * verbose: bool. verbosity level.
        -----
        Diff with normal layer: one has to do preprocessing before (radius, global token, ...)
    """

    def __init__(self, n_layers, feats_input_dim, feats_dim,
                 pos_dim=3,
                 edge_attr_dim=0,
                 m_dim=16,
                 fourier_features=0,
                 soft_edge=0,
                 embedding_nums=None,
                 embedding_dims=None,
                 edge_embedding_nums=None,
                 edge_embedding_dims=None,
                 update_coors=True,
                 update_feats=True,
                 norm_feats=True,
                 norm_coors=False,
                 norm_coors_scale_init=1e-2,
                 dropout=0.,
                 coor_weights_clamp_value=None,
                 aggr="add",
                 global_linear_attn_every=0,
                 global_linear_attn_heads=8,
                 global_linear_attn_dim_head=64,
                 num_global_tokens=4,
                 recalc=0, ):
        super().__init__()

        if embedding_nums is None:
            embedding_nums = []
        if embedding_dims is None:
            embedding_dims = []
        if edge_embedding_nums is None:
            edge_embedding_nums = []
        if edge_embedding_dims is None:
            edge_embedding_dims = []

        self.n_layers = n_layers

        # Embeddings? solve here
        self.embedding_nums = embedding_nums
        self.embedding_dims = embedding_dims
        self.emb_layers = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers = nn.ModuleList()

        # instantiate point and edge embedding layers

        for i in range(len(self.embedding_dims)):
            self.emb_layers.append(nn.Embedding(num_embeddings=embedding_nums[i],
                                                embedding_dim=embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1

        for i in range(len(self.edge_embedding_dims)):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings=edge_embedding_nums[i],
                                                     embedding_dim=edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1
        # rest
        self.mpnn_layers = nn.ModuleList()
        self.feats_input_dim = feats_input_dim
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
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))

        # instantiate layers
        for i in range(n_layers):
            layer = EGNNSparse(feats_dim=feats_dim,
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
                                coor_weights_clamp_value=coor_weights_clamp_value)

            # global attention case
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if is_global_layer:
                attn_layer = GlobalLinearAttention(dim=self.feats_dim,
                                                   heads=global_linear_attn_heads,
                                                   dim_head=global_linear_attn_dim_head)
                self.mpnn_layers.append(nn.ModuleList([layer, attn_layer]))
            # normal case
            else:
                self.mpnn_layers.append(layer)
        # self.node_mlp = nn.Linear(feats_dim, m_dim)
        self.emblin = nn.Linear(feats_input_dim, feats_dim)

    def forward(self, z, pos, edge_index, edge_attr, batch,
                bsize=None, recalc_edge=None, verbose=0):
        """ Recalculate edge features every `self.recalc_edge` with the
            `recalc_edge` function if self.recalc_edge is set.

            * x: (N, pos_dim+feats_dim) will be unpacked into coors, feats.
        """
        # NODES - Embedd each dim to its target dimensions:
        z, temb = z[:, :self.feats_input_dim], z[:, self.feats_input_dim:]
        h = self.emblin(z) + temb
        x = torch.cat([pos, h], dim=1)
        x = embedd_token(x, self.embedding_dims, self.emb_layers)

        #  regulates wether to embedd edges each layer
        edges_need_embedding = False
        # print(len(self.mpnn_layers))
        for i, layer in enumerate(self.mpnn_layers):

            # EDGES - Embedd each dim to its target dimensions:
            if edges_need_embedding:
                edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)
                edges_need_embedding = False

            #  attn tokens
            global_tokens = None
            if exists(self.global_tokens):
                _, amounts = torch.unique(batch, return_counts)
                num_idxs = torch.cat([torch.arange(num_idxs_i) for num_idxs_i in amounts], dim=-1)
                global_tokens = self.global_tokens[num_idxs]

            #  pass layers
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if not is_global_layer:
                x = layer(x, edge_index, edge_attr, batch=batch, size=bsize)
            else:
                # only pass feats to the attn layer
                x_attn = layer[0](x[:, self.pos_dim:], global_tokens)
                #  merge attn-ed feats and coords
                x = torch.cat((x[:, :self.pos_dim], x_attn), dim=-1)
                x = layer[-1](x, edge_index, edge_attr, batch=batch, size=bsize)

            # recalculate edge info - not needed if last layer
            if self.recalc and ((i % self.recalc == 0) and not (i == len(self.mpnn_layers) - 1)):
                edge_index, edge_attr, _ = recalc_edge(x)  #  returns attr, idx, any_other_info
                edges_need_embedding = True

        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        # node_embs = self.node_mlp(feats)
        return feats, coors

    def __repr__(self):
        return 'EGNNSparseNetwork of: {0} layers'.format(len(self.mpnn_layers))

    def aggregate(self, inputs, index, dim_size=None, reduce='add'):
    # 自定义聚合函数，兼容 PyG 2.4 不再自动提供 scatter 聚合
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce=reduce)