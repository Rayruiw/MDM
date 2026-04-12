import numpy as np
from torch_geometric.transforms import Compose
from utils.datasets import QM93D
from utils.transforms import CountNodesPerGraph, GetAdj, AtomFeat
from configs.datasets_config import get_dataset_info

dataset_info = get_dataset_info('qm9', False)
transforms = Compose([
    CountNodesPerGraph(),
    GetAdj(),
    AtomFeat(dataset_info['atom_index'])
])

train_set = QM93D('train', condition=None, pre_transform=transforms)
split_idx = train_set.get_half_split(len(train_set), 'qm9_second_half')
train_set = train_set[split_idx]

vals = []
for i in range(len(train_set)):
    data = train_set[i]
    vals.append(data.gap.item())   # gap is intensive; no heavy-atom normalization needed

vals = np.array(vals)

print("count:", len(vals))
print("min:", vals.min())
print("max:", vals.max())
print("mean:", vals.mean())
print("std:", vals.std())

qs = [10, 25, 50, 75, 90]
targets = np.percentile(vals, qs)

print("percentiles:")
for q, t in zip(qs, targets):
    print(f"P{q}: {t:.4f}")

print("gap targets:", [round(float(x), 4) for x in targets])