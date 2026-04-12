import argparse
import math
import os
import time
import pickle
import copy
import random
from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

from configs.datasets_config import get_dataset_info
from models.epsnet import *
from utils.misc import *
from utils.reconstruct import set_positions
from utils.sample import *
from utils.Macrolides_dataset import MacrolidesDataset


def RMSD(probe, ref):
    rmsd = 0.0
    assert len(probe) == len(ref)
    atom_num = len(probe)
    for i in range(len(probe)):
        posp = probe[i]
        posf = ref[i]
        rmsd += dist_2(posp, posf)
    rmsd = math.sqrt(rmsd / atom_num)
    return rmsd


def dist_2(atoma_xyz, atomb_xyz):
    dis2 = 0.0
    for i, j in zip(atoma_xyz, atomb_xyz):
        dis2 += (i - j) ** 2
    return dis2


def infer_dataset_name(args, ckpt):
    if getattr(args, 'dataset', None) and args.dataset != 'auto':
        return args.dataset
    cfg = ckpt.get('config', None)
    if cfg is not None and hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'name'):
        return cfg.dataset.name
    ckpt_path = str(args.ckpt).lower()
    if 'macrolides' in ckpt_path:
        return 'macrolides'
    if 'qm9' in ckpt_path:
        return 'qm9'
    return 'geom'


def get_split_ids(dataset, split_name='test'):
    if split_name not in dataset.splits:
        raise RuntimeError(f'Unknown split "{split_name}". Available: {list(dataset.splits.keys())}')
    return [int(x) for x in dataset.splits[split_name]]


def build_macrolides_conformation_batches(dataset, num_samples, batch_size, split_name='test', seed=2021):
    """
    Fixed-topology macrocycle conformation generation:
    Use real molecular graphs from dataset split.
    """
    rng = random.Random(seed)
    split_ids = get_split_ids(dataset, split_name)
    if len(split_ids) == 0:
        raise RuntimeError(f'Split "{split_name}" is empty.')

    chosen_ids = [rng.choice(split_ids) for _ in range(num_samples)]
    data_list = []

    for idx in chosen_ids:
        d = dataset[idx]

        item = Data(
            pos=d.pos.clone(),
            edge_index=d.edge_index.clone(),
            edge_type=d.edge_type.clone() if hasattr(d, 'edge_type') else torch.zeros((d.edge_index.size(1),), dtype=torch.long),
            atom_feat_full=d.atom_feat_full.clone(),
            x=d.x.clone() if hasattr(d, 'x') and d.x is not None else None,
            atom_type=d.atom_type.clone() if hasattr(d, 'atom_type') else d.z.clone(),
            z=d.z.clone(),
            num_nodes=int(d.num_nodes),
        )

        if hasattr(d, 'smiles'):
            item.smiles = d.smiles
        if hasattr(d, 'mol') and d.mol is not None:
            item.mol = copy.deepcopy(d.mol)
        if hasattr(d, 'mol_id'):
            item.mol_id = d.mol_id

        data_list.append(item)

    batched = []
    for i in range(0, len(data_list), batch_size):
        batched.append(data_list[i:i + batch_size])
    return batched


def center_positions(pos: torch.Tensor) -> torch.Tensor:
    return pos - pos.mean(dim=0, keepdim=True)


def kabsch_align_and_rmsd(pred: torch.Tensor, ref: torch.Tensor) -> float:
    pred = pred.detach().cpu().float()
    ref = ref.detach().cpu().float()

    if pred.size(0) != ref.size(0):
        raise ValueError('pred/ref atom count mismatch')

    pred_cent = pred - pred.mean(dim=0, keepdim=True)
    ref_cent = ref - ref.mean(dim=0, keepdim=True)

    h = pred_cent.t().mm(ref_cent)
    u, s, vT = torch.linalg.svd(h)
    r = vT.t().mm(u.t())

    if torch.det(r) < 0:
        vT[-1, :] *= -1
        r = vT.t().mm(u.t())

    pred_aligned = pred_cent.mm(r)
    diff = pred_aligned - ref_cent
    rmsd = torch.sqrt((diff.pow(2).sum(dim=1).mean())).item()
    return rmsd


def unique_undirected_edges(edge_index: torch.Tensor):
    edge_index = edge_index.detach().cpu()
    edges = set()
    for k in range(edge_index.size(1)):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])
        if i == j:
            continue
        if i < j:
            edges.add((i, j))
        else:
            edges.add((j, i))
    return sorted(list(edges))


def bond_length_mae(pred: torch.Tensor, ref: torch.Tensor, edge_index: torch.Tensor) -> float:
    edges = unique_undirected_edges(edge_index)
    if len(edges) == 0:
        return float('nan')

    pred = pred.detach().cpu().float()
    ref = ref.detach().cpu().float()
    errs = []
    for i, j in edges:
        pred_len = torch.norm(pred[i] - pred[j]).item()
        ref_len = torch.norm(ref[i] - ref[j]).item()
        errs.append(abs(pred_len - ref_len))
    return float(np.mean(errs))


def safe_mmff_energy_from_smiles(smiles, positions):
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if mol.GetNumAtoms() != int(positions.size(0)):
            return None
        mol = set_positions(mol, positions.detach().cpu().numpy())
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            return None
        return float(ff.CalcEnergy())
    except Exception:
        return None


def summarize_metric(values):
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(clean) == 0:
        return {'mean': float('nan'), 'median': float('nan'), 'std': float('nan'), 'n': 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std(ddof=0)),
        'n': int(arr.size),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='auto', help='auto, qm9, geom, macrolides')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--ckpt', type=str, required=True, help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true', help='whether store the whole trajectory for sampling')
    parser.add_argument('--save_sdf', action='store_true', help='whether save generated conformers as sdf format')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--context', nargs='+', default=[], help='conditional properties if supported')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)

    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--global_start_sigma', type=float, default=float('inf'))
    parser.add_argument('--local_start_sigma', type=float, default=float('inf'))

    parser.add_argument('--w_global_pos', type=float, default=1.0)
    parser.add_argument('--w_local_pos', type=float, default=1.0)
    parser.add_argument('--w_global_node', type=float, default=1.0)
    parser.add_argument('--w_local_node', type=float, default=1.0)

    parser.add_argument('--sampling_type', type=str, default='generalized', help='generalized, ddpm_noisy, ld')
    parser.add_argument('--eta', type=float, default=1.0)

    parser.add_argument('--max_retry', type=int, default=3)
    parser.add_argument('--skip_failed_batch', action='store_true')

    # Macrolides dataset args
    parser.add_argument('--macrolides_root', type=str, default='./data/macrolides')
    parser.add_argument('--macrolides_csv', type=str, default='Macrolides.csv')
    parser.add_argument('--split', type=str, default='test', help='train, val, test')

    # Conformation-generation specific args
    parser.add_argument('--pos_noise_std', type=float, default=1.0, help='std of Gaussian noise added to coordinates')
    parser.add_argument('--atom_noise_std', type=float, default=0.05, help='std of Gaussian noise added to atom_feat_full')
    parser.add_argument('--fix_atom_types', action='store_true', help='use ground-truth atom types for reconstruction/metrics')
    parser.add_argument('--save_results', action='store_true', help='save per-sample metrics to pickle')

    args = parser.parse_args()

    import easydict
    torch.serialization.add_safe_globals([easydict.EasyDict])
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    args.dataset = infer_dataset_name(args, ckpt)
    config = ckpt['config']

    if args.dataset != 'macrolides':
        raise ValueError(
            f'This revised script is specifically for macrocycle conformation generation. Got dataset={args.dataset}.'
        )

    seed_all(args.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))
    args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    dataset_info = get_dataset_info(args.dataset, False)
    num_samples = args.num_samples
    batch_size = args.batch_size

    macrolides_dataset = MacrolidesDataset(root=args.macrolides_root, csv_name=args.macrolides_csv)
    data_list = build_macrolides_conformation_batches(
        dataset=macrolides_dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        split_name=args.split,
        seed=args.seed,
    )

    tag = 'confgen'
    output_dir = get_new_log_dir(
        log_dir,
        args.sampling_type + "_{}_sigma{}_".format(args.seed, args.global_start_sigma) + tag,
        tag=tag
    )
    logger = get_logger('test', output_dir)
    logger.info(args)

    logger.info('Building model...')
    logger.info(ckpt['config'].model['network'])
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(ckpt['config'].model)
    model.eval()

    logger.info('dataset:%s' % args.dataset)
    logger.info('task: macrocycle conformation generation / refinement')
    logger.info('sample num:%d' % num_samples)
    logger.info('split:%s' % args.split)
    logger.info('sample method:%s' % args.sampling_type)
    logger.info('w_global_pos: %.2f' % args.w_global_pos)
    logger.info('w_global_node: %.2f' % args.w_global_node)
    logger.info('w_local_pos: %.2f' % args.w_local_pos)
    logger.info('w_local_node: %.2f' % args.w_local_node)
    logger.info('pos_noise_std: %.3f' % args.pos_noise_std)
    logger.info('atom_noise_std: %.3f' % args.atom_noise_std)
    logger.info('fix_atom_types: %s' % str(args.fix_atom_types))

    rmsd_list = []
    bond_mae_list = []
    energy_list = []
    results = []
    generated_total = 0

    for n, datas in enumerate(tqdm(data_list)):
        with torch.no_grad():
            start_time = time.time()
            batch = Batch.from_data_list(datas).to(args.device)

            retry_count = 0
            batch_finished = False
            current_clip = float(args.clip)

            while (not batch_finished) and (retry_count < args.max_retry):
                try:
                    # ====================================================
                    # fixed-topology conformation generation:
                    # keep real graph, perturb coordinates, denoise back
                    # ====================================================
                    pos_init = batch.pos.clone() + torch.randn_like(batch.pos) * args.pos_noise_std
                    pos_init = pos_init - scatter_mean(pos_init, batch.batch, dim=0)[batch.batch]

                    atom_template = batch.atom_feat_full.float()
                    atom_init = atom_template + torch.randn_like(atom_template) * args.atom_noise_std

                    pos_gen, pos_gen_traj, atom_type_pred, atom_traj = model.langevin_dynamics_sample(
                        atom_type=atom_init,
                        pos_init=pos_init,
                        bond_index=batch.edge_index,
                        bond_type=getattr(batch, 'edge_type', None),
                        batch=batch.batch,
                        num_graphs=batch.num_graphs,
                        extend_order=False,
                        n_steps=args.n_steps,
                        step_lr=1e-6,
                        w_global_pos=args.w_global_pos,
                        w_global_node=args.w_global_node,
                        w_local_pos=args.w_local_pos,
                        w_local_node=args.w_local_node,
                        global_start_sigma=args.global_start_sigma,
                        local_start_sigma=args.local_start_sigma,
                        clip=current_clip,
                        clip_local=None,
                        clip_node=5.0,
                        sampling_type=args.sampling_type,
                        eta=args.eta,
                        context=None,
                        debug_nan=True,
                    )

                    pos_list = unbatch(pos_gen, batch.batch)
                    atom_pred_list = unbatch(atom_type_pred, batch.batch)
                    ref_pos_list = [d.pos.to(args.device) for d in datas]

                    current_batch_size = len(pos_list)
                    generated_total += current_batch_size
                    secs_per_sample = (time.time() - start_time) / max(1, current_batch_size)
                    print('\t %d/%d Conformations generated at %.2f secs/sample' % (
                        min(generated_total, num_samples), num_samples, secs_per_sample))

                    for m in range(current_batch_size):
                        pred_pos = center_positions(pos_list[m].detach())
                        ref_pos = center_positions(ref_pos_list[m].detach())

                        if args.fix_atom_types:
                            atom_type = datas[m].atom_type.clone()
                        else:
                            atom_logits = atom_pred_list[m][:, :-1]
                            atom_type = torch.argmax(atom_logits, dim=1).cpu()

                        sample_rmsd = kabsch_align_and_rmsd(pred_pos, ref_pos)
                        sample_bond_mae = bond_length_mae(pred_pos, ref_pos, datas[m].edge_index)
                        sample_energy = safe_mmff_energy_from_smiles(
                            getattr(datas[m], 'smiles', None),
                            pred_pos.detach().cpu()
                        )

                        rmsd_list.append(sample_rmsd)
                        bond_mae_list.append(sample_bond_mae)
                        energy_list.append(sample_energy)

                        result = {
                            'pred_pos': pred_pos.detach().cpu(),
                            'ref_pos': ref_pos.detach().cpu(),
                            'atom_type': atom_type,
                            'rmsd': sample_rmsd,
                            'bond_mae': sample_bond_mae,
                            'mmff_energy': sample_energy,
                            'smiles': getattr(datas[m], 'smiles', None),
                        }
                        results.append(result)

                        if args.save_sdf:
                            try:
                                mol = None
                                if hasattr(datas[m], 'mol') and datas[m].mol is not None:
                                    mol = copy.deepcopy(datas[m].mol)
                                elif hasattr(datas[m], 'smiles') and datas[m].smiles is not None:
                                    mol = Chem.MolFromSmiles(str(datas[m].smiles))
                                    if mol is not None:
                                        mol = Chem.AddHs(mol)

                                if mol is not None and mol.GetNumAtoms() == int(pred_pos.size(0)):
                                    mol = set_positions(mol, pred_pos.detach().cpu().numpy())
                                    sdf_dir = './results/macrolides_confgen_sdfs'
                                    os.makedirs(sdf_dir, exist_ok=True)
                                    writer = Chem.SDWriter(os.path.join(sdf_dir, f'conf_{n}_{m}.sdf'))
                                    writer.write(mol)
                                    writer.close()
                            except Exception:
                                pass

                    batch_finished = True

                except FloatingPointError:
                    retry_count += 1
                    if current_clip > 5.0:
                        current_clip = 5.0
                    elif current_clip > 2.5:
                        current_clip = 2.5
                    else:
                        current_clip = 2.0

                    logger.warning(
                        f'Batch {n} failed with NaN during sampling. '
                        f'Retry {retry_count}/{args.max_retry}, new global clip={current_clip}.'
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not batch_finished:
                msg = f'Batch {n} failed after {args.max_retry} retries.'
                if args.skip_failed_batch:
                    logger.warning(msg + ' Skipping this batch.')
                    continue
                else:
                    raise RuntimeError(msg)

            rmsd_stats = summarize_metric(rmsd_list)
            bond_stats = summarize_metric(bond_mae_list)
            logger.info(
                '[Batch %d] cumulative RMSD mean=%.4f | bond_MAE mean=%.4f | processed=%d' %
                (n + 1, rmsd_stats['mean'], bond_stats['mean'], generated_total)
            )

    rmsd_stats = summarize_metric(rmsd_list)
    bond_stats = summarize_metric(bond_mae_list)
    energy_stats = summarize_metric(energy_list)

    print('Final RMSD stats:', rmsd_stats)
    print('Final bond MAE stats:', bond_stats)
    print('Final MMFF energy stats:', energy_stats)

    logger.info('Final RMSD mean: %.6f | median: %.6f | std: %.6f | n=%d' % (
        rmsd_stats['mean'], rmsd_stats['median'], rmsd_stats['std'], rmsd_stats['n']
    ))
    logger.info('Final bond MAE mean: %.6f | median: %.6f | std: %.6f | n=%d' % (
        bond_stats['mean'], bond_stats['median'], bond_stats['std'], bond_stats['n']
    ))
    logger.info('Final MMFF energy mean: %.6f | median: %.6f | std: %.6f | n=%d' % (
        energy_stats['mean'], energy_stats['median'], energy_stats['std'], energy_stats['n']
    ))

    if args.save_results:
        save_path = os.path.join(output_dir, 'macrocycle_conformation_results.pkl')
        logger.info('Saving conformation results to: %s' % save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)