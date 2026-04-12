import argparse
import math
import os
import pickle
import time
from statistics import mean

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from eval import retrieve_qm9_smiles, analyze_stability_for_molecules, retrieve_geom_smiles, compute_prop
from configs.datasets_config import get_dataset_info
from models.epsnet import *
from qm9.utils import compute_mean_mad
from utils.datasets import *
from utils.misc import *
from utils.reconstruct import *
from utils.sample import *
from utils.transforms import *

import faulthandler
faulthandler.enable()


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


def get_visualization_targets(property_name: str):
    """
    Targets for Figure-4-style conditional visualization.
    These are raw targets in the SAME space as the conditioning variable.
    """
    if property_name == 'alpha':
        # Match the spirit of MDM Figure 4
        return torch.tensor([70., 76., 83., 90., 94., 99., 102.], dtype=torch.float32)

    elif property_name == 'alpha_per_heavy':
        # Heavy-atom-normalized alpha targets
        return torch.tensor([7.59, 8.03, 8.54, 9.06, 9.55], dtype=torch.float32)

    elif property_name == 'gap':
        return torch.tensor([0.1925, 0.2170, 0.2502, 0.2893, 0.3162], dtype=torch.float32)
    elif property_name == 'homo':
        return torch.tensor([-0.40, -0.36, -0.32, -0.28, -0.24, -0.20], dtype=torch.float32)
    elif property_name == 'lumo':
        return torch.tensor([0.00, 0.03, 0.06, 0.09, 0.12, 0.15], dtype=torch.float32)
    elif property_name == 'mu':
        return torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32)
    elif property_name == 'Cv':
        return torch.tensor([20., 25., 30., 35., 40., 45.], dtype=torch.float32)
    elif property_name == 'Cv_per_heavy':
        return torch.tensor([3.0751, 3.3003, 3.5754, 3.8679, 4.1479], dtype=torch.float32)
    else:
        raise ValueError(f'No predefined visualization targets for property: {property_name}')


def build_nodewise_context(batch, raw_value, mean_value, mad_value, device):
    """
    Convert a graph-level raw target (in the same space as args.context[0])
    to normalized node-wise context.
    For example:
      - alpha            -> raw alpha
      - alpha_per_heavy  -> raw alpha / N_heavy
      - Cv               -> raw Cv
      - Cv_per_heavy     -> raw Cv / N_heavy
    """
    c_norm = (raw_value - mean_value) / mad_value
    num_nodes = batch.num_nodes
    context = torch.full((num_nodes, 1), fill_value=float(c_norm), device=device)
    return context


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm9', help='qm9, geom')
    parser.add_argument('--ckpt', type=str, required=True, help='path for loading the checkpoint')

    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        "--context",
        nargs='+',
        default=[],
        help='arguments : homo | lumo | alpha | alpha_per_heavy | gap | mu | Cv | Cv_per_heavy'
    )
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_sdf', type=bool, default=True)
    parser.add_argument(
        '--quality_sampling',
        action='store_true',
        help='quality sampling for visualization in Figure 4, else validity sampling for bulk sampling'
    )
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=None,
                        help='sampling num steps; if None, will use checkpoint diffusion steps')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                        help='enable global gradients only when noise is low')
    parser.add_argument('--local_start_sigma', type=float, default=float('inf'),
                        help='enable local gradients only when noise is low')
    parser.add_argument('--w_global_pos', type=float, default=1.0,
                        help='weight for global gradients')
    parser.add_argument('--w_local_pos', type=float, default=1.0,
                        help='weight for local gradients')
    parser.add_argument('--w_global_node', type=float, default=0.5,
                        help='weight for global gradients')
    parser.add_argument('--w_local_node', type=float, default=3.0,
                        help='weight for local gradients')
    parser.add_argument('--sampling_type', type=str, default='generalized',
                        help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Load checkpoint and recover configuration
    # --------------------------------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    args.dataset = 'qm9' if 'qm9' in args.ckpt.lower() else 'geom'
    config = ckpt['config']
    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # Recover condition from checkpoint if available
    if 'context' in ckpt and ckpt['context'] is not None:
        args.context = ckpt['context']
    else:
        args.context = ckpt['config'].model.context
        if isinstance(args.context, str):
            args.context = [args.context]

    if args.context is None or len(args.context) == 0:
        raise ValueError('No condition context found. Please provide --context or save context in the checkpoint.')

    if args.n_steps is None:
        args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    # --------------------------------------------------------------------------
    # Dataset construction for evaluation sampling
    # --------------------------------------------------------------------------
    dataset_info = get_dataset_info(args.dataset, False)
    num_samples = args.num_samples
    batch_size = args.batch_size
    data_list, nodesxsample_list = construct_dataset(num_samples, batch_size, dataset_info)

    transforms = Compose([CountNodesPerGraph(), GetAdj(), AtomFeat(dataset_info['atom_index'])])

    if args.dataset == 'qm9':
        train_set = QM93D('train', condition=None, pre_transform=transforms)
        split_idx = train_set.get_half_split(len(train_set), 'qm9_second_half')
        train_set = train_set[split_idx]

        val_set = QM93D('valid', condition=None, pre_transform=transforms)
        val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)

    elif args.dataset == 'geom':
        train_set = Geom(pre_transform=transforms)
        val_loader = None
    else:
        raise Exception('Wrong dataset name')

    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    # --------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------
    if num_samples < 10000:
        local_tag = 'test'
    else:
        local_tag = 'result'

    output_dir = get_new_log_dir(log_dir, args.sampling_type + "_cond_eval_" + local_tag, tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------
    logger.info('Building model...')
    logger.info(f"Network type: {ckpt['config'].model['network']}")
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])
    print(ckpt['config'].model)
    model.eval()

    # --------------------------------------------------------------------------
    # Global stats containers
    # --------------------------------------------------------------------------
    sa_list = []
    global_valid = 0
    global_stable = 0
    global_smile_list = []
    results = []
    sum_rmsd = 0
    clip_local = None
    show_detail = False

    logger.info('dataset:%s' % args.dataset)
    logger.info('sample num:%d' % num_samples)
    logger.info('sample method:%s' % args.sampling_type)
    logger.info('w_global_pos:%.1f' % args.w_global_pos)
    logger.info('w_global_node:%.1f' % args.w_global_node)
    logger.info('w_local_pos:%.1f' % args.w_local_pos)
    logger.info('w_local_node:%.1f' % args.w_local_node)
    logger.info('context:%s' % args.context)

    position_list = []
    atom_type_list = []
    mols_dict = []

    # --------------------------------------------------------------------------
    # Property normalization / context distribution
    # --------------------------------------------------------------------------
    context_value = ['validity sampling']
    property_norms = None
    prop_dist = None
    mean_v = None
    mad_v = None

    if len(args.context) > 0:
        property_norms = compute_mean_mad(train_set, args.context, args.dataset)
        mean_v = property_norms[args.context[0]]['mean']
        mad_v = property_norms[args.context[0]]['mad']

        if args.quality_sampling:
            context_value = get_visualization_targets(args.context[0])
        else:
            prop_dist = DistributionProperty(train_loader, args.context)
            if prop_dist is not None:
                prop_dist.set_normalizer(property_norms)

    # --------------------------------------------------------------------------
    # Main sampling loop
    # --------------------------------------------------------------------------
    for c in context_value:
        print(f"\n===== sampling target {args.context[0]} = {float(c):.3f} =====")

        target_valid = 0
        target_stable = 0
        target_smile_list = []

        target_mol_ok = 0
        target_mol_fail = 0
        target_sdf_saved = 0

        for n, datas in enumerate(tqdm(data_list, desc=f"target={float(c):.3f}")):
            with torch.no_grad():
                start_time = time.time()
                batch = Batch.from_data_list(datas).to(args.device)

                # Build node-wise context
                if args.quality_sampling:
                    context = build_nodewise_context(batch, c, mean_v, mad_v, args.device)
                else:
                    graph_context = prop_dist.sample_batch(nodesxsample_list[n]).to(args.device)
                    context = graph_context.index_select(0, batch.batch)

                try:
                    pos_gen, pos_gen_traj, atom_type, atom_traj = model.langevin_dynamics_sample(
                        atom_type=batch.x,
                        pos_init=batch.pos,
                        bond_index=batch.edge_index,
                        bond_type=None,
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
                        clip=args.clip,
                        clip_local=clip_local,
                        sampling_type=args.sampling_type,
                        eta=args.eta,
                        context=context
                    )

                    pos_list = unbatch(pos_gen, batch.batch)
                    atom_list = unbatch(atom_type, batch.batch)

                    current_num_samples = (n + 1) * batch_size
                    secs_per_sample = (time.time() - start_time) / max(current_num_samples, 1)
                    print('\t %d/%d Molecules generated at %.2f secs/sample' % (
                        current_num_samples, num_samples, secs_per_sample))

                    for m in range(len(pos_list)):
                        pos = pos_list[m]
                        atom_type_logits = atom_list[m]

                        # Remove the extra channel if present
                        atom_type_logits = atom_type_logits[:, :-1]
                        atom_type = torch.argmax(atom_type_logits, dim=1)

                        position_list.append(pos.cpu().detach())
                        atom_type_list.append(atom_type.cpu().detach())

                        mol = build_molecule(pos, atom_type, dataset_info)

                        if mol is None:
                            target_mol_fail += 1
                            smile = None
                        else:
                            target_mol_ok += 1
                            smile = mol2smiles(mol)

                        ptable = Chem.GetPeriodicTable()
                        atom_decoder = dataset_info['atom_decoder']
                        atom_symbols = [atom_decoder[i] for i in atom_type]
                        atom_numbers = [ptable.GetAtomicNumber(i.capitalize()) for i in atom_symbols]

                        # Visualization stage: do not run expensive property computation here
                        prop_value = None

                        result = {
                            'atom_type': atom_numbers,
                            'pos': pos,
                            'smile': smile,
                            'target_property': float(c) if args.quality_sampling else None,
                            'predicted_property': prop_value,
                            'property_name': args.context[0],
                        }

                        if args.context[0] == 'alpha_per_heavy':
                            result['predicted_alpha_raw'] = None

                        if args.context[0] == 'Cv_per_heavy':
                            result['predicted_Cv_raw'] = None

                        results.append(result)

                        if smile is not None:
                            target_valid += 1
                            global_valid += 1

                            stable_flag = False
                            if "." not in smile:
                                target_stable += 1
                                global_stable += 1
                                stable_flag = True

                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                            largest_mol = max(mol_frags, default=mol, key=lambda mm: mm.GetNumAtoms())
                            largest_smile = mol2smiles(largest_mol)

                            target_smile_list.append(largest_smile)
                            global_smile_list.append(largest_smile)

                        if args.save_sdf and mol is not None:
                            sdf_dir = os.path.join('./results/conditioned', args.context[0], 'molecule_with_value')
                            os.makedirs(sdf_dir, exist_ok=True)

                            if mol.GetNumConformers() == 0:
                                conf = Chem.Conformer(mol.GetNumAtoms())
                                for atom_idx in range(mol.GetNumAtoms()):
                                    conf.SetAtomPosition(
                                        atom_idx,
                                        (float(pos[atom_idx][0]), float(pos[atom_idx][1]), float(pos[atom_idx][2]))
                                    )
                                mol.AddConformer(conf)

                            if args.quality_sampling:
                                sdf_name = f'full_{args.context[0]}_{float(c):.3f}_{n}_{m}.sdf'
                            else:
                                sdf_name = f'full_{args.context[0]}_sampled_{n}_{m}.sdf'

                            sdf_path = os.path.join(sdf_dir, sdf_name)
                            writer = Chem.SDWriter(sdf_path)
                            writer.write(mol)
                            writer.close()
                            target_sdf_saved += 1

                    print('----------------------------')
                    logger.info(
                        "Target %.3f | validity: %.4f | stable: %.4f | uniq: %.4f"
                        % (
                            float(c),
                            target_valid / ((n + 1) * batch_size),
                            target_stable / ((n + 1) * batch_size),
                            len(set(target_smile_list)) / max(((n + 1) * batch_size), 1),
                        )
                    )
                    print(
                        f"Target {float(c):.3f} summary | "
                        f"mol_ok={target_mol_ok}, mol_fail={target_mol_fail}, sdf_saved={target_sdf_saved}"
                    )
                    print('----------------------------')

                except FloatingPointError:
                    clip_local = 10
                    logger.warning('Retrying with local clipping.')
                    raise Exception('Nan in position')

            # For quality_sampling / Figure-4-style visualization, only keep one batch for each target value
            if args.quality_sampling:
                break

    # --------------------------------------------------------------------------
    # Final quality metrics
    # --------------------------------------------------------------------------
    validity_dict = analyze_stability_for_molecules(position_list, atom_type_list, dataset_info)
    print(validity_dict)

    if args.quality_sampling:
        total_generated = len(context_value) * num_samples
    else:
        total_generated = num_samples

    print("Final validity:", global_valid / total_generated)
    print("Final stable:", global_stable / total_generated)
    print("Final unique:", len(set(global_smile_list)) / total_generated)

    if global_valid > 0:
        print("Unique among valid:", len(set(global_smile_list)) / global_valid)
    else:
        print("Unique among valid: 0.0")

    logger.info("Final validity:%.4f" % (global_valid / total_generated))
    logger.info("Final stable:%.4f" % (global_stable / total_generated))
    logger.info("Final unique:%.4f" % (len(set(global_smile_list)) / total_generated))

    uniq = list(set(global_smile_list))

    if args.dataset == 'qm9':
        dataset_smile_list = retrieve_qm9_smiles()
    else:
        dataset_smile_list = retrieve_geom_smiles()

    novel = []
    for smile in uniq:
        if smile not in dataset_smile_list:
            novel.append(smile)

    print(len(novel))
    novelty = len(novel) / len(uniq) if len(uniq) > 0 else 0.0
    logger.info("Final novelty:%.4f" % novelty)

    # --------------------------------------------------------------------------
    # Save samples
    # --------------------------------------------------------------------------
    save = True
    if save:
        save_path = os.path.join(output_dir, 'samples_all.pkl')
        logger.info('Saving samples to: %s' % save_path)

        save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        with open(save_smile_path, 'wb') as f:
            pickle.dump(global_smile_list, f)