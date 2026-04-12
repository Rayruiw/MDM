import argparse
import os
import shutil
import time

import torch
import torch.utils.tensorboard
import yaml
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from tqdm.auto import tqdm

from configs.datasets_config import get_dataset_info
from models.epsnet import get_model
from qm9.utils import prepare_context, compute_mean_mad
from utils.common import get_optimizer, get_scheduler
from utils.datasets import QM93D, Geom
from utils.Macrolides_dataset import MacrolidesDataset
from utils.misc import *
from utils.transforms import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='qm9', help='qm9, geom, macrolides')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=torch.cuda.is_available())
parser.add_argument('--use_mixed_precision', action='store_true')
parser.add_argument('--dp', action='store_true')
parser.add_argument('--resume_iter', type=int, default=None)
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument(
    "--context",
    nargs='*',
    default=[],
    help='arguments: homo | lumo | alpha | gap | mu | Cv'
)


def _to_item(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.mean().item()
    return float(x)


def _scheduler_step(train_loss=None, val_metrics=None):
    """
    val_metrics: tuple(pos_global, node_global, pos_local, node_local) or None
    """
    if config.train.scheduler.type == 'plateau':
        if val_metrics is not None:
            pos_g, node_g, pos_l, node_l = val_metrics
            scheduler_global.step(pos_g + node_g)
            scheduler_local.step(pos_l + node_l)
        elif train_loss is not None:
            scheduler_global.step(train_loss)
            scheduler_local.step(train_loss)
    else:
        scheduler_global.step()
        if 'global' not in config.model.network:
            scheduler_local.step()


def train(it):
    model.train()

    sum_loss, sum_n = 0.0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0.0, 0.0
    sum_loss_node_global, sum_loss_node_local = 0.0, 0.0
    sum_local_dist, sum_bond_loss, sum_global_dist = 0.0, 0.0, 0.0
    sum_repulsion, sum_loss_vae_kl = 0.0, 0.0
    last_grad_norm = 0.0

    with tqdm(total=len(train_loader), desc=f'Training [{it:05d}]') as pbar:
        for batch in train_loader:
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()

            batch = batch.to(device)

            if len(args.context) > 0 and property_norms is not None:
                context = prepare_context(args.context, batch, property_norms)
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

            loss = loss.mean()

            if not torch.isfinite(loss):
                logger.warning(f'[Train] Iter {it:05d} | Non-finite loss detected, skipping batch.')
                pbar.set_postfix({'loss': 'nan/inf'})
                pbar.update(1)
                continue

            loss.backward()

            last_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

            optimizer_global.step()
            optimizer_local.step()

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

    if sum_n == 0:
        logger.warning(f'[Train] Iter {it:05d} | No valid batches processed.')
        return None

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

    return avg_loss


def validate(it):
    if val_loader is None:
        return None, None

    model.eval()

    sum_loss, sum_n = 0.0, 0
    sum_loss_pos_global, sum_loss_pos_local = 0.0, 0.0
    sum_loss_node_global, sum_loss_node_local = 0.0, 0.0
    sum_local_dist, sum_bond_loss, sum_global_dist = 0.0, 0.0, 0.0
    sum_repulsion, sum_loss_vae_kl = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation [{it:05d}]'):
            batch = batch.to(device)

            if len(args.context) > 0 and property_norms_val is not None:
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

    if sum_n == 0:
        logger.warning(f'[Validate] Iter {it:05d} | No valid validation batches processed.')
        return None, None

    denom = max(len(val_loader), 1)
    avg_loss = sum_loss / sum_n
    avg_loss_pos_global = sum_loss_pos_global / denom
    avg_loss_pos_local = sum_loss_pos_local / denom
    avg_loss_node_global = sum_loss_node_global / denom
    avg_loss_node_local = sum_loss_node_local / denom
    avg_local_dist = sum_local_dist / denom
    avg_bond_loss = sum_bond_loss / denom
    avg_global_dist = sum_global_dist / denom
    avg_repulsion = sum_repulsion / denom
    avg_loss_vae_kl = sum_loss_vae_kl / denom

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

    val_metrics = (avg_loss_pos_global, avg_loss_node_global, avg_loss_pos_local, avg_loss_node_local)
    return avg_loss, val_metrics


if __name__ == '__main__':
    args = parser.parse_args()

    args.cuda = bool(args.cuda) and torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'

    resume = (args.resume_iter is not None)
    config_path = args.config
    resume_from = args.logdir if resume else None

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    if hasattr(config, 'dataset') and hasattr(config.dataset, 'name'):
        args.dataset = config.dataset.name
    else:
        args.dataset = 'qm9' if 'qm9' in args.config else 'geom'

    config_name = f'{args.dataset}_full_ddpm_2losses'
    seed_all(config.train.seed)

    if resume:
        log_dir = resume_from
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        models_dst = os.path.join(log_dir, 'models')
        if not os.path.exists(models_dst):
            shutil.copytree('./models', models_dst, dirs_exist_ok=True)

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)

    if not resume:
        shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
        shutil.copyfile(os.path.abspath(__file__), os.path.join(log_dir, os.path.basename(__file__)))

    logger.info(f'Loading {args.dataset} dataset...')

    dataset_info = get_dataset_info(args.dataset, remove_h=False)

    if args.dataset in ['qm9', 'geom']:
        transforms = Compose([
            CountNodesPerGraph(),
            GetAdj(),
            AtomFeat(dataset_info['atom_index'])
        ])
    elif args.dataset == 'macrolides':
        transforms = None
    else:
        raise Exception(f'Wrong dataset name: {args.dataset}')

    if args.dataset == 'qm9':
        train_set = QM93D('train', pre_transform=transforms)
        val_set = QM93D('valid', pre_transform=transforms)
        val_loader = DataLoader(
            val_set,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers
        )

    elif args.dataset == 'geom':
        train_set = Geom(pre_transform=transforms)
        val_set = None
        val_loader = None

    elif args.dataset == 'macrolides':
        full_set = MacrolidesDataset(
            root=config.dataset.root,
            csv_name=getattr(config.dataset, 'csv_name', 'Macrolides.csv'),
            transform=None,
            pre_transform=None,
            add_hs=getattr(config.dataset, 'add_hs', True),
            min_largest_ring_size=getattr(config.dataset, 'min_largest_ring_size', 12),
            max_atoms=getattr(config.dataset, 'max_atoms', 100),
            add_radius_edges=getattr(config.dataset, 'add_radius_edges', False),
            radius=getattr(config.dataset, 'radius', 5.0),
            max_num_neighbors=getattr(config.dataset, 'max_num_neighbors', 32),
            force_reprocess=getattr(config.dataset, 'force_reprocess', False),
        )

        splits = full_set.splits
        train_set = Subset(full_set, splits['train'])
        val_set = Subset(full_set, splits['val'])

        val_loader = DataLoader(
            val_set,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers
        )
    else:
        raise Exception(f'Wrong dataset name: {args.dataset}')

    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers
    )

    if len(args.context) > 0 and args.dataset in ['qm9', 'geom']:
        logger.info(f'Conditioning on {args.context}')
        property_norms = compute_mean_mad(train_set, args.context, args.dataset)
        property_norms_val = compute_mean_mad(val_set, args.context, args.dataset) if val_set is not None else None
    else:
        property_norms = None
        property_norms_val = None

    logger.info('Building model...')
    config.model.context = args.context
    config.model.num_atom = len(dataset_info['atom_decoder']) + 1

    model = get_model(config.model).to(device)

    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)

    start_iter = 0
    best_val_loss = float('inf')

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
        best_val_loss = float(ckpt.get('best_val_loss', float('inf')))
        logger.info(f"Start iteration: {start_iter}")
        logger.info(f"Recovered best_val_loss: {best_val_loss:.6f}")

    for it in range(start_iter, config.train.max_iters + 1):
        start_time = time.time()

        avg_train_loss = train(it)
        elapsed = time.time() - start_time
        logger.info(f'[Timing] Iter {it:05d} took {elapsed:.2f}s')

        should_validate = val_loader is not None and (
            it % config.train.val_freq == 0 or it == config.train.max_iters
        )

        avg_val_loss = None
        val_metrics = None
        if should_validate:
            avg_val_loss, val_metrics = validate(it)
            if avg_val_loss is not None:
                _scheduler_step(val_metrics=val_metrics)
        else:
            if avg_train_loss is not None:
                _scheduler_step(train_loss=avg_train_loss)

        should_save = (it % config.train.val_freq == 0) or (it == config.train.max_iters)

        if should_save:
            metric_to_compare = avg_val_loss if avg_val_loss is not None else avg_train_loss
            if metric_to_compare is not None and metric_to_compare < best_val_loss:
                best_val_loss = float(metric_to_compare)
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
                    'avg_train_loss': avg_train_loss,
                    'best_val_loss': best_val_loss,
                }, save_path)
                logger.info(f'Successfully saved the model to {save_path}')
