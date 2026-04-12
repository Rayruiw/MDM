import argparse
import math
import os
import time
import pickle

import torch
from tqdm.auto import tqdm
from rdkit import Chem
from torch_geometric.data import Batch
from torch_scatter import scatter_mean

from eval import retrieve_qm9_smiles, analyze_stability_for_molecules, retrieve_geom_smiles
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

    parser.add_argument('--dataset', type=str, default='qm9')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--save_traj', action='store_true')
    parser.add_argument('--save_sdf', action='store_true')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument("--context", nargs='+', default=[])

    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)

    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_steps', type=int, default=1000)

    parser.add_argument('--global_start_sigma', type=float, default=float('inf'))
    parser.add_argument('--local_start_sigma', type=float, default=float('inf'))

    parser.add_argument('--w_global_pos', type=float, default=1.0)
    parser.add_argument('--w_local_pos', type=float, default=0.0)

    parser.add_argument('--w_global_node', type=float, default=1.0)
    parser.add_argument('--w_local_node', type=float, default=1.0)

    parser.add_argument('--sampling_type', type=str, default='generalized')
    parser.add_argument('--eta', type=float, default=1.0)

    parser.add_argument('--max_retry', type=int, default=3)
    parser.add_argument('--skip_failed_batch', action='store_true')

    args = parser.parse_args()

    # ---------------- checkpoint ----------------

    import easydict
    torch.serialization.add_safe_globals([easydict.EasyDict])

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    args.dataset = 'qm9' if 'qm9' in args.ckpt else 'geom'

    seed_all(args.seed)

    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    # ---------------- dataset ----------------

    dataset_info = get_dataset_info(args.dataset, False)

    num_samples = args.num_samples
    batch_size = args.batch_size

    data_list, _ = construct_dataset(num_samples, batch_size, dataset_info)

    # ---------------- logging ----------------

    tag = 'test' if num_samples < 10000 else 'result'

    output_dir = get_new_log_dir(
        log_dir,
        args.sampling_type + "_{}_sigma{}_aligned_".format(args.seed, args.global_start_sigma) + tag,
        tag=tag
    )

    logger = get_logger('test', output_dir)

    logger.info(args)

    # ---------------- results folder ----------------

    results_dir = os.path.join(output_dir, "results")
    sdf_dir = os.path.join(results_dir, "sdf")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(sdf_dir, exist_ok=True)

    # ---------------- model ----------------

    logger.info('Building model...')
    logger.info(ckpt['config'].model['network'])

    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])

    model.eval()

    # ---------------- metrics ----------------

    valid = 0
    stable = 0
    generated_total = 0

    smile_list = []
    results = []

    position_list = []
    atom_type_list = []

    # ---------------- sampling ----------------

    for n, datas in enumerate(tqdm(data_list)):

        with torch.no_grad():

            batch = Batch.from_data_list(datas).to(args.device)

            retry_count = 0
            batch_finished = False
            current_clip = float(args.clip)

            while (not batch_finished) and (retry_count < args.max_retry):

                try:

                    # -------- random noise initialization --------

                    pos_init = torch.randn_like(batch.pos)

                    pos_init = pos_init - scatter_mean(
                        pos_init,
                        batch.batch,
                        dim=0
                    )[batch.batch]

                    if hasattr(batch, "atom_feat_full"):
                        atom_template = batch.atom_feat_full.float()
                    else:
                        atom_template = batch.x.float()

                    atom_init = torch.randn_like(atom_template)

                    # -------- sampling --------

                    pos_gen, pos_gen_traj, atom_type_raw, atom_traj = model.langevin_dynamics_sample(

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
                        debug_nan=True
                    )

                    pos_list = unbatch(pos_gen, batch.batch)
                    atom_list = unbatch(atom_type_raw, batch.batch)

                    current_batch_size = len(pos_list)

                    generated_total += current_batch_size

                    for m in range(current_batch_size):

                        pos = pos_list[m]
                        atom_type = atom_list[m]

                        atom_logits = atom_type[:, :-1]
                        atom_type = torch.argmax(atom_logits, dim=1)

                        position_list.append(pos.cpu().detach())
                        atom_type_list.append(atom_type.cpu().detach())

                        mol = build_molecule(pos, atom_type, dataset_info)

                        smile = mol2smiles(mol)

                        results.append({
                            'atom_type': atom_type,
                            'pos': pos,
                            'smile': smile
                        })

                        if mol is not None and mol.GetNumAtoms() > 0:

                            if args.save_sdf:

                                sdf_path = os.path.join(
                                    sdf_dir,
                                    f"mol_{n}_{m}.sdf"
                                )

                                try:
                                    writer = Chem.SDWriter(sdf_path)
                                    writer.write(mol)
                                    writer.close()
                                except Exception:
                                    pass

                        if smile is not None:

                            valid += 1

                            if "." not in smile:
                                stable += 1

                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)

                            largest_mol = max(
                                mol_frags,
                                default=mol,
                                key=lambda x: x.GetNumAtoms()
                            )

                            smile = mol2smiles(largest_mol)

                            smile_list.append(smile)

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
                        f'Batch {n} NaN retry {retry_count}/{args.max_retry} clip={current_clip}'
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not batch_finished:

                if args.skip_failed_batch:
                    logger.warning(f'Batch {n} skipped')
                    continue
                else:
                    raise RuntimeError(f'Batch {n} failed')

        # ---------------- batch log ----------------

        logger.info(
            f"Batch {n+1} | "
            f"Valid {valid/max(1,generated_total):.4f} | "
            f"Stable {stable/max(1,generated_total):.4f} | "
            f"Unique {len(set(smile_list))/max(1,generated_total):.4f}"
        )

    # ---------------- evaluation ----------------

    validity_dict = analyze_stability_for_molecules(
        position_list,
        atom_type_list,
        dataset_info
    )

    print(validity_dict)

    final_valid = valid / max(1, generated_total)
    final_stable = stable / max(1, generated_total)
    final_unique = len(set(smile_list)) / max(1, generated_total)

    logger.info(f"Final validity {final_valid:.4f}")
    logger.info(f"Final stable {final_stable:.4f}")
    logger.info(f"Final unique {final_unique:.4f}")

    # ---------------- novelty ----------------

    uniq_smiles = list(set(smile_list))

    if args.dataset == 'qm9':
        dataset_smiles = retrieve_qm9_smiles()
    else:
        dataset_smiles = retrieve_geom_smiles()

    novel_smiles = [s for s in uniq_smiles if s not in dataset_smiles]

    novelty = len(novel_smiles) / max(1, len(uniq_smiles))

    logger.info(f"Final novelty {novelty:.4f}")

    # ---------------- save results ----------------

    samples_path = os.path.join(results_dir, "samples_all.pkl")
    smiles_path = os.path.join(results_dir, "smiles.txt")

    with open(samples_path, "wb") as f:
        pickle.dump(results, f)

    with open(smiles_path, "w") as f:
        for s in smile_list:
            if s is not None:
                f.write(s + "\n")

    logger.info(f"Results saved to {results_dir}")