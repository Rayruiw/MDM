import argparse
import shutil
import os
import time

import torch
import torch.distributed as dist
import torch.utils.tensorboard
import yaml
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm

from configs.datasets_config import get_dataset_info
from models.epsnet import get_model
from qm9.utils import prepare_context, compute_mean_mad
from utils.common import get_optimizer, get_scheduler
from utils.datasets import QM93D, Geom
from utils.misc import *
from utils.transforms import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='qm9', help='qm9, geom')
parser.add_argument('--config', type=str)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--use_mixed_precision', type=bool, default=False)
parser.add_argument('--dp', type=bool, default=True)
parser.add_argument('--resume_iter', type=int, default=None)
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument(
    "--context",
    nargs='+',
    default=[],
    help='arguments : homo | lumo | alpha | gap | mu | Cv'
)


def _to_item(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.mean().item()
    return float(x)


def train(it):
    model.train()

    sum_loss, sum_n = 0.0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0.0, 0.0
    sum_loss_node_global, sum_loss_node_local = 0.0, 0.0
    sum_local_dist, sum_bond_loss, sum_global_dist = 0.0, 0.0, 0.0
    sum_repulsion, sum_loss_vae_kl = 0.0, 0.0
    last_grad_norm = 0.0

    with tqdm(total=len(train_loader), desc='Training') as pbar:
        for batch in train_loader:
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()

            batch = batch.to(device)

            if len(args.context) > 0:
                context = prepare_context(args.context, batch, property_norms)
            else:
                context = None

            out = model(
                batch,
                context=context,
                return_unreduced_loss=True
            )

            # -------------------------------------------------
            # strict PaiNN forward() return format:
            # non-VAE:
            #   loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local,
            #   local_dist_loss, bond_loss, global_dist_loss, repulsion_loss
            #
            # VAE:
            #   loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local,
            #   local_dist_loss, bond_loss, global_dist_loss, repulsion_loss, vae_kl_loss
            # -------------------------------------------------
            if config.model.vae_context:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, \
                local_dist_loss, bond_loss, global_dist_loss, repulsion_loss, loss_vae_kl = out
                current_loss_vae_kl = _to_item(loss_vae_kl)
            else:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, \
                local_dist_loss, bond_loss, global_dist_loss, repulsion_loss = out
                current_loss_vae_kl = 0.0

            # 主 loss 聚合
            loss = loss.mean()

            # 数值保护
            if not torch.isfinite(loss):
                logger.warning(f'[Train] Iter {it:05d} | Non-finite loss detected, skipping batch.')
                pbar.set_postfix({'loss': 'nan/inf'})
                pbar.update(1)
                continue

            loss.backward()

            last_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

            optimizer_global.step()
            optimizer_local.step()

            # 累加统计
            sum_loss += loss.item()
            sum_n += 1

            sum_loss_pos_global += loss_pos_global.mean().item()
            sum_loss_pos_local += loss_pos_local.mean().item()
            sum_loss_node_global += loss_node_global.mean().item()
            sum_loss_node_local += loss_node_local.mean().item()

            sum_local_dist += _to_item(local_dist_loss)
            sum_bond_loss += _to_item(bond_loss)
            sum_global_dist += _to_item(global_dist_loss)
            sum_repulsion += _to_item(repulsion_loss)
            sum_loss_vae_kl += current_loss_vae_kl

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update(1)

    # 避免整轮都被 skip
    if sum_n == 0:
        logger.warning(f'[Train] Iter {it:05d} | No valid batches processed.')
        return

    avg_loss = sum_loss / sum_n
    avg_loss_pos_global = sum_loss_pos_global / sum_n
    avg_loss_pos_local = sum_loss_pos_local / sum_n
    avg_loss_node_global = sum_loss_node_global / sum_n
    avg_loss_node_local = sum_loss_node_local / sum_n
    avg_local_dist = sum_local_dist / sum_n
    avg_bond_loss = sum_bond_loss / sum_n
    avg_global_dist = sum_global_dist / sum_n
    avg_repulsion = sum_repulsion / sum_n
    avg_loss_vae_kl = sum_loss_vae_kl / sum_n

    logger.info(
        f'[Train] Iter {it:05d} | '
        f'Loss {avg_loss:.4f} | '
        f'PosG {avg_loss_pos_global:.4f} | PosL {avg_loss_pos_local:.4f} | '
        f'NodeG {avg_loss_node_global:.4f} | NodeL {avg_loss_node_local:.4f} | '
        f'LocalDist {avg_local_dist:.4f} | Bond {avg_bond_loss:.4f} | '
        f'GlobalDist {avg_global_dist:.4f} | Repulsion {avg_repulsion:.4f} | '
        f'VAE_KL {avg_loss_vae_kl:.4f} | '
        f'Grad {float(last_grad_norm):.4f} | '
        f'LR {optimizer_global.param_groups[0]["lr"]:.6f}'
    )

    writer.add_scalar('train/loss', avg_loss, it)
    writer.add_scalar('train/loss_pos_global', avg_loss_pos_global, it)
    writer.add_scalar('train/loss_pos_local', avg_loss_pos_local, it)
    writer.add_scalar('train/loss_node_global', avg_loss_node_global, it)
    writer.add_scalar('train/loss_node_local', avg_loss_node_local, it)
    writer.add_scalar('train/local_dist_loss', avg_local_dist, it)
    writer.add_scalar('train/bond_loss', avg_bond_loss, it)
    writer.add_scalar('train/global_dist_loss', avg_global_dist, it)
    writer.add_scalar('train/repulsion_loss', avg_repulsion, it)
    writer.add_scalar('train/loss_vae_KL', avg_loss_vae_kl, it)
    writer.add_scalar('train/lr', optimizer_global.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', float(last_grad_norm), it)
    writer.flush()


def validate(it):
    model.eval()

    sum_loss, sum_n = 0.0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0.0, 0.0
    sum_loss_node_global, sum_loss_node_local = 0.0, 0.0
    sum_local_dist, sum_bond_loss, sum_global_dist = 0.0, 0.0, 0.0
    sum_repulsion, sum_loss_vae_kl = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)

            if len(args.context) > 0:
                context = prepare_context(args.context, batch, property_norms_val)
            else:
                context = None

            out = model(
                batch,
                context=context,
                return_unreduced_loss=True
            )

            if config.model.vae_context:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, \
                local_dist_loss, bond_loss, global_dist_loss, repulsion_loss, loss_vae_kl = out
                current_loss_vae_kl = _to_item(loss_vae_kl)
            else:
                loss, loss_pos_global, loss_pos_local, loss_node_global, loss_node_local, \
                local_dist_loss, bond_loss, global_dist_loss, repulsion_loss = out
                current_loss_vae_kl = 0.0

            # loss 是 unreduced 的 node-level / sample-level 张量
            sum_loss += loss.sum().item()
            sum_n += loss.size(0)

            sum_loss_pos_global += loss_pos_global.mean().item()
            sum_loss_pos_local += loss_pos_local.mean().item()
            sum_loss_node_global += loss_node_global.mean().item()
            sum_loss_node_local += loss_node_local.mean().item()

            sum_local_dist += _to_item(local_dist_loss)
            sum_bond_loss += _to_item(bond_loss)
            sum_global_dist += _to_item(global_dist_loss)
            sum_repulsion += _to_item(repulsion_loss)
            sum_loss_vae_kl += current_loss_vae_kl

    avg_loss = sum_loss / sum_n
    avg_loss_pos_global = sum_loss_pos_global / len(val_loader)
    avg_loss_pos_local = sum_loss_pos_local / len(val_loader)
    avg_loss_node_global = sum_loss_node_global / len(val_loader)
    avg_loss_node_local = sum_loss_node_local / len(val_loader)
    avg_local_dist = sum_local_dist / len(val_loader)
    avg_bond_loss = sum_bond_loss / len(val_loader)
    avg_global_dist = sum_global_dist / len(val_loader)
    avg_repulsion = sum_repulsion / len(val_loader)
    avg_loss_vae_kl = sum_loss_vae_kl / len(val_loader)

    if config.train.scheduler.type == 'plateau':
        scheduler_global.step(avg_loss_pos_global + avg_loss_node_global)
        scheduler_local.step(avg_loss_pos_local + avg_loss_node_local)
    else:
        scheduler_global.step()
        if 'global' not in config.model.network:
            scheduler_local.step()

    logger.info(
        f'[Validate] Iter {it:05d} | '
        f'Loss {avg_loss:.6f} | '
        f'PosG {avg_loss_pos_global:.4f} | PosL {avg_loss_pos_local:.4f} | '
        f'NodeG {avg_loss_node_global:.4f} | NodeL {avg_loss_node_local:.4f} | '
        f'LocalDist {avg_local_dist:.4f} | Bond {avg_bond_loss:.4f} | '
        f'GlobalDist {avg_global_dist:.4f} | Repulsion {avg_repulsion:.4f} | '
        f'VAE_KL {avg_loss_vae_kl:.4f}'
    )

    writer.add_scalar('val/loss', avg_loss, it)
    writer.add_scalar('val/loss_pos_global', avg_loss_pos_global, it)
    writer.add_scalar('val/loss_pos_local', avg_loss_pos_local, it)
    writer.add_scalar('val/loss_node_global', avg_loss_node_global, it)
    writer.add_scalar('val/loss_node_local', avg_loss_node_local, it)
    writer.add_scalar('val/local_dist_loss', avg_local_dist, it)
    writer.add_scalar('val/bond_loss', avg_bond_loss, it)
    writer.add_scalar('val/global_dist_loss', avg_global_dist, it)
    writer.add_scalar('val/repulsion_loss', avg_repulsion, it)
    writer.add_scalar('val/loss_vae_KL', avg_loss_vae_kl, it)
    writer.flush()

    return avg_loss


# ------------------------------------------------------------------------------
# Training file not in ddp mode
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'

    args.dataset = 'qm9' if 'qm9' in args.config else 'geom'

    # 只要指定了 --resume_iter 就认为要恢复训练
    resume = (args.resume_iter is not None)

    config_path = args.config
    resume_from = args.logdir if resume else None

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    config_name = '%s_full_ddpm_2losses' % args.dataset
    seed_all(config.train.seed)

    # 恢复训练时不新建 log_dir，直接沿用原目录
    if resume:
        log_dir = resume_from
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        if not os.path.exists(os.path.join(log_dir, 'models')):
            shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = get_logger('train', log_dir)

    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)

    # 非恢复训练时，备份配置和脚本
    if not resume:
        shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
        shutil.copyfile('./train.py', os.path.join(log_dir, 'train.py'))

    # --------------------------------------------------------------------------
    # Datasets and loaders
    # --------------------------------------------------------------------------
    logger.info('Loading %s datasets...' % args.dataset)

    dataset_info = get_dataset_info(args.dataset, remove_h=False)
    transforms = Compose([
        CountNodesPerGraph(),
        GetAdj(),
        AtomFeat(dataset_info['atom_index'])
    ])

    if args.dataset == 'qm9':
        train_set = QM93D('train', pre_transform=transforms)
        val_set = QM93D('valid', pre_transform=transforms)
        val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)
    elif args.dataset == 'geom':
        train_set = Geom(pre_transform=transforms)
        val_set = None
        val_loader = None
    else:
        raise Exception('Wrong dataset name')

    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True)

    # --------------------------------------------------------------------------
    # Context
    # --------------------------------------------------------------------------
    if len(args.context) > 0:
        print(f'Conditioning on {args.context}')
        property_norms = compute_mean_mad(train_set, args.context, args.dataset)
        property_norms_val = property_norms
    else:
        property_norms = None
        property_norms_val = None

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------
    logger.info('Building model...')
    config.model.context = args.context
    config.model.num_atom = len(dataset_info['atom_decoder']) + 1

    model = get_model(config.model).to(device)

    # --------------------------------------------------------------------------
    # Optimizer / Scheduler
    # --------------------------------------------------------------------------
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)

    # --------------------------------------------------------------------------
    # Resume from checkpoint
    # --------------------------------------------------------------------------
    start_iter = 0
    if resume:
        ckpt_path = os.path.join(log_dir, 'checkpoints', f'{args.resume_iter}.pt')
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt['model'])
        optimizer_global.load_state_dict(ckpt['optimizer_global'])
        optimizer_local.load_state_dict(ckpt['optimizer_local'])
        scheduler_global.load_state_dict(ckpt['scheduler_global'])
        scheduler_local.load_state_dict(ckpt['scheduler_local'])

        start_iter = int(ckpt.get('iteration', args.resume_iter)) + 1
        logger.info(f"Start iteration: {start_iter}")

    # --------------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------------
    best_val_loss = float('inf')
    for it in range(start_iter, config.train.max_iters + 1):
        start_time = time.time()

        train(it)

        elapsed = time.time() - start_time
        print(f'each iteration requires {elapsed} s')

        avg_val_loss = validate(it)

        if it % config.train.val_freq == 0 or it == config.train.max_iters:
            if avg_val_loss < best_val_loss:
                save_path = os.path.join(ckpt_dir, f'{it}.pt')
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'scheduler_global': scheduler_global.state_dict(),
                    'optimizer_local': optimizer_local.state_dict(),
                    'scheduler_local': scheduler_local.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, save_path)
                print('Successfully saved the model!')
                best_val_loss = avg_val_loss