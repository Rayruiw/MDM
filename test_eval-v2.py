import argparse
import math
import os
import time
import pickle

import torch
from tqdm.auto import tqdm
from rdkit import Chem
from torch_geometric.data import Batch

from eval import retrieve_qm9_smiles, analyze_stability_for_molecules, retrieve_geom_smiles
from torch_scatter import scatter_mean
from configs.datasets_config import get_dataset_info
from models.epsnet import *
from utils.datasets import *
from utils.misc import *
from utils.reconstruct import *
from utils.sample import *


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='qm9, geom')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true',
                        help='whether store the whole trajectory for sampling')
    parser.add_argument('--save_sdf', action='store_true',
                        help='whether save the molecule as sdf format (just add --save_sdf in command to enable)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--context", nargs='+', default=[],
                        help='arguments : homo | lumo | alpha | gap | mu | Cv')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)

    # Parameters for model sampling
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=float('inf'),
                        help='enable global gradients only when noise is low')
    parser.add_argument('--local_start_sigma', type=float, default=float('inf'),
                        help='enable local gradients only when noise is low')

    # Current architecture:
    # global branch predicts position epsilon
    # local branch only predicts node denoising
    parser.add_argument('--w_global_pos', type=float, default=1.0,
                        help='weight for global position epsilon')
    parser.add_argument('--w_local_pos', type=float, default=0.0,
                        help='deprecated: local branch no longer predicts position')
    parser.add_argument('--w_global_node', type=float, default=1.0,
                        help='weight for global node gradients')
    parser.add_argument('--w_local_node', type=float, default=1.0,
                        help='weight for local node gradients')

    # Parameters for DDPM / DDIM / LD
    parser.add_argument('--sampling_type', type=str, default='ddpm_noisy',
                        help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')

    # Robustness / retry
    parser.add_argument('--max_retry', type=int, default=3,
                        help='maximum retry times for a failed batch')
    parser.add_argument('--skip_failed_batch', action='store_true',
                        help='skip batch after repeated NaN failures instead of raising error')

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Load checkpoint safely on CPU first to avoid GPU OOM during deserialization
    # --------------------------------------------------------------------------
    import easydict
    torch.serialization.add_safe_globals([easydict.EasyDict])
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    args.dataset = 'qm9' if 'qm9' in args.ckpt else 'geom'
    config = ckpt['config']

    seed_all(args.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # use checkpoint's diffusion steps
    args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    # --------------------------------------------------------------------------
    # Dataset construction
    # --------------------------------------------------------------------------
    dataset_info = get_dataset_info(args.dataset, False)
    num_samples = args.num_samples
    batch_size = args.batch_size
    data_list, _ = construct_dataset(num_samples, batch_size, dataset_info)

    # --------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------
    tag = 'result'
    if num_samples < 10000:
        tag = 'test'

    output_dir = get_new_log_dir(
        log_dir,
        args.sampling_type + "_{}_sigma{}_build_all_bonds_".format(args.seed, args.global_start_sigma) + tag,
        tag=tag
    )
    logger = get_logger('test', output_dir)
    logger.info(args)

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------
    logger.info('Building model...')
    logger.info(ckpt['config'].model['network'])
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])
    print(ckpt['config'].model)

    model.eval()

    # --------------------------------------------------------------------------
    # Metrics / containers
    # --------------------------------------------------------------------------
    sa_list = []
    valid = 0
    stable = 0
    smile_list = []
    results = []
    sum_rmsd = 0

    logger.info('dataset:%s' % args.dataset)
    logger.info('sample num:%d' % num_samples)
    logger.info('sample method:%s' % args.sampling_type)
    logger.info('w_global_pos: %.2f' % args.w_global_pos)
    logger.info('w_global_node: %.2f' % args.w_global_node)
    logger.info('w_local_pos (disabled): %.2f' % args.w_local_pos)
    logger.info('w_local_node: %.2f' % args.w_local_node)

    show_detail = False

    position_list = []
    atom_type_list = []
    mols_dict = []

    generated_total = 0

    # --------------------------------------------------------------------------
    # Sampling loop
    # --------------------------------------------------------------------------
    for n, datas in enumerate(tqdm(data_list)):
        with torch.no_grad():
            context = None
            start_time = time.time()
            batch = Batch.from_data_list(datas).to(args.device)

            retry_count = 0
            batch_finished = False
            current_clip = float(args.clip)

            while (not batch_finished) and (retry_count < args.max_retry):
                try:
                    pos_init = torch.randn_like(batch.pos)
                    pos_init = pos_init - scatter_mean(pos_init, batch.batch, dim=0)[batch.batch]

                    if hasattr(batch, "atom_feat_full"):
                        atom_template = batch.atom_feat_full.float()
                    else:
                        atom_template = batch.x.float()

                    atom_init = torch.randn_like(atom_template)

                    pos_gen, pos_gen_traj, atom_type, atom_traj = model.langevin_dynamics_sample(
                        atom_type=atom_init,
                        pos_init=pos_init,
                        bond_index=batch.edge_index,
                        bond_type=None,
                        batch=batch.batch,
                        num_graphs=batch.num_graphs,
                        extend_order=False,
                        n_steps=args.n_steps,
                        step_lr=1e-6,
                        w_global_pos=args.w_global_pos,
                        w_global_node=args.w_global_node,
                        w_local_pos=0.0,
                        w_local_node=args.w_local_node,
                        global_start_sigma=args.global_start_sigma,
                        local_start_sigma=args.local_start_sigma,
                        clip=current_clip,
                        clip_local=None,
                        clip_node=5.0,
                        sampling_type=args.sampling_type,
                        eta=args.eta,
                        context=context,
                        debug_nan=True,
                    )

                    pos_list = unbatch(pos_gen, batch.batch)
                    atom_list = unbatch(atom_type, batch.batch)

                    current_batch_size = len(pos_list)
                    generated_total += current_batch_size

                    secs_per_sample = (time.time() - start_time) / max(1, current_batch_size)
                    print('\t %d/%d Molecules generated at %.2f secs/sample' % (
                        min(generated_total, num_samples), num_samples, secs_per_sample))

                    for m in range(current_batch_size):
                        pos = pos_list[m]
                        atom_type = atom_list[m]

                        # charge / atom logits split
                        atom_logits = atom_type[:, :-1]
                        charge = atom_type[:, -1]

                        atom_type = torch.argmax(atom_logits, dim=1)
                        position_list.append(pos.cpu().detach())
                        atom_type_list.append(atom_type.cpu().detach())

                        mol = build_molecule(pos, atom_type, dataset_info)
                        smile = mol2smiles(mol)

                        if show_detail:
                            print("g pos:", pos)
                            print("generated element:", atom_type)
                            print("generated smile:", smile)

                        result = {'atom_type': atom_type, 'pos': pos, 'smile': smile}
                        results.append(result)

                        if smile is not None:
                            valid += 1
                            if "." not in smile:
                                stable += 1

                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                            largest_mol = max(mol_frags, default=mol, key=lambda mm: mm.GetNumAtoms())
                            mol = largest_mol
                            smile = mol2smiles(mol)
                            smile_list.append(smile)

                            if args.save_sdf and mol is not None and mol.GetNumAtoms() > 0:
                                sdf_dir = './results/ddpm_sdfs'
                                os.makedirs(sdf_dir, exist_ok=True)
                                try:
                                    writer = Chem.SDWriter(os.path.join(sdf_dir, f'mol_{n}_{m}.sdf'))
                                    writer.write(mol)
                                    writer.close()
                                except Exception:
                                    pass

                    batch_finished = True

                except FloatingPointError:
                    retry_count += 1

                    # tighten GLOBAL clipping because current architecture uses global-only position branch
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
                    print('----------------------------')
                    logger.info("The %dth validity:%.4f" % (
                        n + 1, valid / max(1, generated_total)))
                    logger.info("The %dth stable:%.4f" % (
                        n + 1, stable / max(1, generated_total)))
                    logger.info("The %dth Uniq:%.4f" % (
                        n + 1, len(set(smile_list)) / max(1, generated_total)))
                    print('----------------------------')
                    continue
                else:
                    raise RuntimeError(msg)

            print('----------------------------')
            logger.info("The %dth validity:%.4f" % (
                n + 1, valid / max(1, generated_total)))
            logger.info("The %dth stable:%.4f" % (
                n + 1, stable / max(1, generated_total)))
            logger.info("The %dth Uniq:%.4f" % (
                n + 1, len(set(smile_list)) / max(1, generated_total)))
            print('----------------------------')

    # --------------------------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------------------------
    validity_dict = analyze_stability_for_molecules(position_list, atom_type_list, dataset_info)
    print(validity_dict)

    final_denom = max(1, generated_total)
    final_validity = valid / final_denom
    final_stable = stable / final_denom
    final_unique = len(set(smile_list)) / final_denom

    print("Final validity:", final_validity)
    print("Final stable:", final_stable)
    print("Final unique:", final_unique)

    if valid > 0:
        print(len(set(smile_list)) / valid)
    else:
        print(0.0)

    logger.info("Final validity:%.4f" % final_validity)
    logger.info("Final stable:%.4f" % final_stable)
    logger.info("Final unique:%.4f" % final_unique)

    uniq = list(set(smile_list))

    if args.dataset == 'qm9':
        dataset_smile_list = retrieve_qm9_smiles()
    else:
        dataset_smile_list = retrieve_geom_smiles()

    novel = []
    for smile in uniq:
        if smile not in dataset_smile_list:
            novel.append(smile)

    print(len(novel))
    novelty = len(novel) / max(1, len(uniq))
    logger.info("Final novelty:%.4f" % novelty)

    save = False
    if num_samples == 50:
        save = True

    if save:
        save_path = os.path.join(output_dir, 'samples_all.pkl')
        logger.info('Saving samples to: %s' % save_path)
        save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        with open(save_smile_path, 'wb') as f:
            pickle.dump(smile_list, f)