import copy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.transforms import Compose
from torch_sparse import coalesce


from .chem import BOND_TYPES, BOND_NAMES, get_atom_symbol


class AddHigherOrderEdges(object):

    def __init__(self, order, num_types=len(BOND_TYPES)):
        super().__init__()
        self.order = order
        self.num_types = num_types

    def binarize(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order + 1):
            adj_mats.append(self.binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    def __call__(self, data: Data):
        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = self.get_higher_order_adj_matrix(adj, self.order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)  # (N, N)
        type_highorder = torch.where(adj_order > 1, self.num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)  # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N)  # modify data
        data.is_bond = (data.edge_type < self.num_types)
        assert (data.edge_index == edge_index_1).all()

        return data


class AddEdgeLength(object):

    def __call__(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1)  # (num_edge, 1)
        data.edge_length = d
        return data

    # Add attribute placeholder for data object, so that we can use batch.to_data_list


class AddPlaceHolder(object):
    def __call__(self, data: Data):
        data.pos_gen = -1. * torch.ones_like(data.pos)
        data.d_gen = -1. * torch.ones_like(data.edge_length)
        data.d_recover = -1. * torch.ones_like(data.edge_length)
        return data


class AddEdgeName(object):

    def __init__(self, asymmetric=True):
        super().__init__()
        self.bonds = copy.deepcopy(BOND_NAMES)
        self.bonds[len(BOND_NAMES) + 1] = 'Angle'
        self.bonds[len(BOND_NAMES) + 2] = 'Dihedral'
        self.asymmetric = asymmetric

    def __call__(self, data: Data):
        data.edge_name = []
        for i in range(data.edge_index.size(1)):
            tail = data.edge_index[0, i]
            head = data.edge_index[1, i]
            if self.asymmetric and tail >= head:
                data.edge_name.append('')
                continue
            tail_name = get_atom_symbol(data.atom_type[tail].item())
            head_name = get_atom_symbol(data.atom_type[head].item())
            name = '%s_%s_%s_%d_%d' % (
                self.bonds[data.edge_type[i].item()] if data.edge_type[i].item() in self.bonds else 'E' + str(
                    data.edge_type[i].item()),
                tail_name,
                head_name,
                tail,
                head,
            )
            if hasattr(data, 'edge_length'):
                name += '_%.3f' % (data.edge_length[i].item())
            data.edge_name.append(name)
        return data


class AddAngleDihedral(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def iter_angle_triplet(bond_mat):
        n_atoms = bond_mat.size(0)
        for j in range(n_atoms):
            for k in range(n_atoms):
                for l in range(n_atoms):
                    if bond_mat[j, k].item() == 0 or bond_mat[k, l].item() == 0:
                        continue
                    if (j == k) or (k == l) or (j >= l):
                        continue
                    yield (j, k, l)

    @staticmethod
    def iter_dihedral_quartet(bond_mat):
        n_atoms = bond_mat.size(0)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i >= j:
                    continue
                if bond_mat[i, j].item() == 0:
                    continue
                for k in range(n_atoms):
                    for l in range(n_atoms):
                        if (k in (i, j)) or (l in (i, j)):
                            continue
                        if bond_mat[k, i].item() == 0 or bond_mat[l, j].item() == 0:
                            continue
                        yield (k, i, j, l)

    def __call__(self, data: Data):
        # N = data.num_nodes
        if 'is_bond' in data:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.is_bond).long().squeeze(0) > 0
        else:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).long().squeeze(0) > 0

        # Note: if the name of attribute contains `index`, it will automatically
        #       increases during batching.
        data.angle_index = torch.LongTensor(list(self.iter_angle_triplet(bond_mat))).t()
        data.dihedral_index = torch.LongTensor(list(self.iter_dihedral_quartet(bond_mat))).t()

        return data


class CountNodesPerGraph(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data


class GetAdj(object):
    def __init__(self, radius=None) -> None:
        super().__init__()
        self.radius = radius  # ✅ 可选，不影响原有逻辑

    def __call__(self, data):
        """
        fully connected edges
        (如果将来想用半径筛选，可以利用 self.radius)
        """
        n_particles = data.num_nodes
        coors = data.pos
        rows, cols = [], []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                # 若定义了 cutoff，可在此筛选距离
                if self.radius is not None:
                    dist = torch.norm(coors[i] - coors[j])
                    if dist > self.radius:
                        continue
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)

        rows = torch.LongTensor(rows).unsqueeze(0)
        cols = torch.LongTensor(cols).unsqueeze(0)
        adj = torch.cat([rows, cols], dim=0)

        rel_coors = coors[adj[0]] - coors[adj[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True).sqrt()  # ✅ 距离

        # ✅ 保留你当前版本的输出
        data.edge_index = adj
        data.edge_length = rel_dist.squeeze(1)   # [E]
        data.edge_attr = rel_coors               # [E, 3]
        data.edge_type = data.edge_length        # [E]
        return data

class AtomFeat(object):
    def __init__(self, atom_index) -> None:
        super().__init__()
        self.atom_index = atom_index  # 期望是 Z->idx

    def __call__(self, data):
        """
        one-hot feature
        data.atom_type: 通常是原子序 Z（如 1,6,7,8,9）
        将其通过 Z->idx 映射到类别索引，再做 one-hot
        """
        z_vals = data.atom_type.tolist()
        idx_list = [self.atom_index.get(int(i), 0) for i in z_vals]  # ✅ 容错
        idx = torch.tensor(idx_list, dtype=torch.long)

        num_classes = max(self.atom_index.values()) + 1  # 例如 5
        data.atom_feat = F.one_hot(idx, num_classes=num_classes).float()

        # 维持你原来的拼接：one-hot + 原子序本身
        data.atom_feat_full = torch.cat([data.atom_feat, data.atom_type.unsqueeze(1)], dim=1)

        # 可同步提供 data.x / data.z，方便下游统一使用
        data.x = data.atom_feat
        data.z = idx
        return data
