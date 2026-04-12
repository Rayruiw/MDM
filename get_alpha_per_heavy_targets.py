import numpy as np
from torch_geometric.transforms import Compose

from utils.datasets import QM93D
from utils.transforms import CountNodesPerGraph, GetAdj, AtomFeat
from configs.datasets_config import get_dataset_info

# 1. 构建和训练时一致的 dataset
dataset_info = get_dataset_info('qm9', False)
transforms = Compose([
    CountNodesPerGraph(),
    GetAdj(),
    AtomFeat(dataset_info['atom_index'])
])

train_set = QM93D('train', condition=None, pre_transform=transforms)

# 如果你训练时用了 second-half，这里也要保持一致
split_idx = train_set.get_half_split(len(train_set), 'qm9_second_half')
train_set = train_set[split_idx]

# 2. 收集 alpha_per_heavy
vals = []
for i in range(len(train_set)):
    data = train_set[i]
    alpha = data.alpha.item()
    n_heavy = int(data.num_heavy_atoms.item())
    vals.append(alpha / n_heavy)

vals = np.array(vals)

# 3. 输出统计信息
print("count:", len(vals))
print("min:", vals.min())
print("max:", vals.max())
print("mean:", vals.mean())
print("std:", vals.std())

# 4. 输出推荐分位点
qs = [10, 25, 50, 75, 90]
targets = np.percentile(vals, qs)

print("percentiles:")
for q, t in zip(qs, targets):
    print(f"P{q}: {t:.4f}")

print("recommended visualization targets:")
print([round(float(x), 2) for x in targets])