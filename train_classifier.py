# MDM/train_classifier.py
import os
import math
import yaml
import argparse

import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from easydict import EasyDict as edict

from MDM.configs.datasets_config import get_dataset_info
from MDM.utils.datasets import QM93D, Geom
from MDM.utils.transforms import Compose, CountNodesPerGraph, GetAdj, AtomFeat
from MDM.utils.misc import seed_all
from MDM.models.classifier import NoisyPropertyClassifier


# ---------- 数据集与目标属性 ----------
def build_dataset(args):
    """
    返回：train_set, val_set, test_set, atom_map, target_map
    - atom_map：原子Z值到索引的映射，供 AtomFeat 使用
    - target_map：性质名到 y 中列索引的映射（用于 extract_y）
    """
    dataset_info = get_dataset_info(args.dataset, remove_h=False)

    # 1) 原子映射
    atom_map = dataset_info.get('atom_index') or dataset_info.get('atom_encoder')
    if not isinstance(atom_map, dict):
        raise ValueError("dataset_info['atom_index'/'atom_encoder'] should be a dict")

    # 兼容“值是Z值”的写法：把 {idx: Z} 转为 {Z: idx}
    qm9_Z = {1, 6, 7, 8, 9}
    keys, vals = set(atom_map.keys()), set(atom_map.values())
    if qm9_Z.issubset(vals) and not qm9_Z.intersection(keys):
        atom_map = {int(Z): int(idx) for idx, Z in atom_map.items()}

    # 2) 目标属性映射
    target_map = (
        dataset_info.get('target_map')
        or dataset_info.get('target2idx')
        or dataset_info.get('target_index')
        or {}
    )

    # 3) 变换管线（半径图 + 节点计数 + 原子特征）
    transform = Compose([
        AtomFeat(atom_index=atom_map),
        CountNodesPerGraph(),
        GetAdj(radius=args.cutoff),
    ])

    # 4) 构造数据集
    if args.dataset == 'qm9':
        train_set = QM93D(split='train', transform=transform)
        val_set   = QM93D(split='valid', transform=transform)
        test_set  = QM93D(split='test',  transform=transform)
    elif args.dataset == 'geom':
        train_set = Geom(split='train', transform=transform)
        val_set   = Geom(split='valid', transform=transform)
        test_set  = Geom(split='test',  transform=transform)
    else:
        raise ValueError("Unsupported dataset ('qm9' or 'geom').")

    return train_set, val_set, test_set, atom_map, target_map


def extract_y(batch, prop, target_map):
    """
    严格按 dataset_info 的目标映射取 y 的列，避免误用 y[...,0]
    """
    if target_map and prop in target_map:
        idx = int(target_map[prop])
        return batch.y[..., idx].view(-1).to(batch.pos.device)
    if hasattr(batch, prop):  # 次选：若数据本身带有独立字段
        return getattr(batch, prop).view(-1).to(batch.pos.device)
    raise KeyError(f"Cannot find target '{prop}'. Check dataset_info target_map/target2idx.")


# ---------- 训练主程序 ----------
def main():
    parser = argparse.ArgumentParser()

    # —— 基本训练参数 ——
    parser.add_argument('--dataset', type=str, default='qm9', help='qm9 | geom')
    parser.add_argument('--prop', type=str, required=True, help='目标性质名，如 alpha/homo/lumo/...')
    parser.add_argument('--regression', action='store_true', help='回归任务则开启该开关')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--logdir', type=str, default='./logs_clf/alpha_h256_egnn')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=1e-4)

    # —— 与扩散模型一致的 EGNN 编码器参数（只定义一次！） ——
    # ✅ 与您命令行保持一致的默认值：feats_dim=6, edge_attr_dim=5
    parser.add_argument('--hidden_dim', type=int, default=256)   # feats_dim of EGNN内部通道
    parser.add_argument('--num_convs', type=int, default=6)      # n_layers
    parser.add_argument('--feats_dim', type=int, default=6)      # 节点原始特征宽度（不含时间）
    parser.add_argument('--m_dim', type=int, default=128)
    parser.add_argument('--edge_attr_dim', type=int, default=5)  # [dx,dy,dz,dist,extra]
    parser.add_argument('--pos_dim', type=int, default=3)
    parser.add_argument('--soft_edge', action='store_true', default=True)
    parser.add_argument('--norm_coors', action='store_true', default=True)
    parser.add_argument('--cutoff', type=float, default=2.0)
    parser.add_argument('--edge_encoder', type=str, default='mlp')
    parser.add_argument('--num_atom', type=int, default=6)
    parser.add_argument('--edge_order', type=int, default=3)     # 仅记录/打印

    # —— 扩散超参（用于 q_sample 一致性） ——
    parser.add_argument('--beta_schedule', type=str, default='sigmoid')
    parser.add_argument('--beta_start', type=float, default=1e-7)
    parser.add_argument('--beta_end', type=float, default=2e-3)
    parser.add_argument('--num_diffusion_timesteps', type=int, default=1000)

    args = parser.parse_args()

    # CFG 收口，便于传给 NoisyPropertyClassifier
    cfg = edict({
        'hidden_dim': args.hidden_dim,
        'num_convs': args.num_convs,
        'feats_dim': args.feats_dim,
        'm_dim': args.m_dim,
        'edge_attr_dim': args.edge_attr_dim,
        'pos_dim': args.pos_dim,
        'soft_edge': bool(args.soft_edge),
        'norm_coors': bool(args.norm_coors),

        'beta_schedule': args.beta_schedule,
        'beta_start': args.beta_start,
        'beta_end': args.beta_end,
        'num_diffusion_timesteps': args.num_diffusion_timesteps,

        'cutoff': args.cutoff,
        'edge_encoder': args.edge_encoder,
        'num_atom': args.num_atom,

        'edge_order': args.edge_order,
    })

    # 复现实验环境
    seed_all(42)
    os.makedirs(args.logdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据与加载器
    train_set, val_set, test_set, atom_map, target_map = build_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)

    # 模型、优化器、损失
    out_dim = 1 if args.regression else 2
    clf = NoisyPropertyClassifier(cfg, out_dim=out_dim).to(device)
    optimizer = AdamW(clf.parameters(), lr=args.lr)
    loss_fn = MSELoss() if args.regression else CrossEntropyLoss()

    print(f"✅ Using device: {device}")
    print("✅ Encoder cfg:",
          f"hidden_dim={cfg.hidden_dim}, num_convs={cfg.num_convs}, feats_dim={cfg.feats_dim}, "
          f"m_dim={cfg.m_dim}, edge_order={cfg.edge_order}, soft_edge={cfg.soft_edge}, "
          f"norm_coors={cfg.norm_coors}, cutoff={cfg.cutoff}, edge_encoder={cfg.edge_encoder}, "
          f"num_atom={cfg.num_atom}, edge_attr_dim={cfg.edge_attr_dim}")

    # 保存元信息（便于复现实验）
    meta_path = os.path.join(args.logdir, 'clf_config.yaml')
    with open(meta_path, 'w') as f:
        yaml.safe_dump({
            'dataset': args.dataset,
            'prop': args.prop,
            'regression': bool(args.regression),
            'cfg': dict(cfg),
            'atom_map': {int(k): int(v) if isinstance(v, int) else v for k, v in atom_map.items()}
        }, f)
    print(f"📝 Dumped classifier config/meta to: {meta_path}")

    best_val = math.inf
    best_path = os.path.join(args.logdir, 'best.pt')
    patience_left = args.patience

    # ---------- 训练 ----------
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        clf.train()
        total_loss, count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Train {epoch}"):
            batch = batch.to(device)

            # 每个图采一个时间步，做 q_sample 加噪位置
            B = int(batch.batch.max().item() + 1)
            t = torch.randint(cfg.num_diffusion_timesteps, (B,), device=device, dtype=torch.long)

            pos_t = clf.q_sample(batch.pos, t, batch.batch)
            batch_t = batch.clone()
            batch_t.pos = pos_t

            # 兜底 z
            if getattr(batch_t, 'z', None) is None:
                if getattr(batch_t, 'atom_type', None) is not None:
                    batch_t.z = batch_t.atom_type.long()
                elif getattr(batch_t, 'x', None) is not None:
                    batch_t.z = batch_t.x[:, 0].long()
                else:
                    raise RuntimeError("Cannot infer atom types (z).")

            pred = clf(batch_t, t)  # [B, out_dim]
            y = extract_y(batch, args.prop, target_map)  # [B]

            loss = loss_fn(pred.view(-1), y) if args.regression else loss_fn(pred, y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            count += B

        print(f"[Epoch {epoch}] Train Loss: {total_loss / max(count,1):.6f}")

        # ---- Val ----
        clf.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                B = int(batch.batch.max().item() + 1)
                t = torch.randint(cfg.num_diffusion_timesteps, (B,), device=device, dtype=torch.long)

                pos_t = clf.q_sample(batch.pos, t, batch.batch)
                batch_t = batch.clone()
                batch_t.pos = pos_t

                if getattr(batch_t, 'z', None) is None:
                    if getattr(batch_t, 'atom_type', None) is not None:
                        batch_t.z = batch_t.atom_type.long()
                    elif getattr(batch_t, 'x', None) is not None:
                        batch_t.z = batch_t.x[:, 0].long()

                pred = clf(batch_t, t)
                y = extract_y(batch, args.prop, target_map)
                loss = loss_fn(pred.view(-1), y) if args.regression else loss_fn(pred, y.long())

                val_loss += loss.item() * B
                val_count += B

        val_avg = val_loss / max(val_count, 1)
        print(f"[Epoch {epoch}] Val  Loss: {val_avg:.6f}")

        # 早停 + 保存最好
        if val_avg + args.min_delta < best_val:
            best_val = val_avg
            patience_left = args.patience
            torch.save(clf.state_dict(), best_path)
            print(f"✅ Saved best model to {best_path} (val={best_val:.6f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"⏹️ Early stopped at epoch {epoch}. Best val={best_val:.6f}")
                break

    print("✅ Training complete.")


if __name__ == '__main__':
    main()