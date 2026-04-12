import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from torch_scatter import scatter_mean

from eval import retrieve_qm9_smiles, retrieve_geom_smiles
from configs.datasets_config import get_dataset_info
from models.epsnet import *
from utils.datasets import *
from utils.misc import *
from utils.reconstruct import *
from utils.sample import *


def postprocess_molecule(mol, add_hs=True, optimize=True, max_iters=200):
    if mol is None:
        return None, "mol is None"

    try:
        mol = Chem.Mol(mol)
    except Exception as e:
        return None, f"copy failed: {e}"

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None, f"sanitize failed: {e}"

    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 0:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
    except Exception:
        pass

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None, f"resanitize failed: {e}"

    if add_hs:
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception as e:
            return None, f"AddHs failed: {e}"

    if optimize:
        try:
            if mol.GetNumConformers() == 0:
                params = AllChem.ETKDGv3()
                params.randomSeed = 2021
                ok = AllChem.EmbedMolecule(mol, params)
                if ok != 0:
                    return None, "EmbedMolecule failed"

            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
        except Exception as e:
            return None, f"optimize failed: {e}"

    return mol, "ok"


def generate_for_web(
    ckpt_path,
    num_samples=20,
    batch_size=20,
    device="cuda",
    sampling_type="generalized",
    clip=3.0,
    eta=1.0,
    global_start_sigma=float("inf"),
    local_start_sigma=float("inf"),
    w_global_pos=1.0,
    w_local_pos=1.5,
    w_global_node=0.4,
    w_local_node=0.8,
    add_hs=True,
    optimize=True,
    ff_max_iters=200,
    output_root="./web_results",
    run_name="sample_run",
    max_retry=3,
    seed=2021,
):
    if num_samples < 1 or num_samples > 100:
        raise ValueError("num_samples must be between 1 and 100")

    import easydict
    torch.serialization.add_safe_globals([easydict.EasyDict])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "qm9" in ckpt_path.lower():
        dataset = "qm9"
    elif "geom" in ckpt_path.lower():
        dataset = "geom"
    else:
        dataset = "qm9"

    n_steps = ckpt["config"].model.num_diffusion_timesteps
    seed_all(seed)

    output_dir = os.path.join(output_root, run_name)
    sdf_dir = os.path.join(output_dir, "sdfs")
    os.makedirs(sdf_dir, exist_ok=True)

    dataset_info = get_dataset_info(dataset, False)
    data_list, _ = construct_dataset(num_samples, batch_size, dataset_info)

    model = get_model(ckpt["config"].model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    position_list = []
    atom_type_list = []
    smile_list = []
    saved_entries = []

    valid = 0
    stable = 0
    generated_total = 0
    saved_total = 0

    start_all = time.time()

    for datas in data_list:
        with torch.no_grad():
            batch = Batch.from_data_list(datas).to(device)

            retry_count = 0
            batch_finished = False
            current_clip = float(clip)

            while (not batch_finished) and (retry_count < max_retry):
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
                        n_steps=n_steps,
                        step_lr=1e-6,
                        w_global_pos=w_global_pos,
                        w_global_node=w_global_node,
                        w_local_pos=w_local_pos,
                        w_local_node=w_local_node,
                        global_start_sigma=global_start_sigma,
                        local_start_sigma=local_start_sigma,
                        clip=current_clip,
                        clip_local=None,
                        clip_node=5.0,
                        sampling_type=sampling_type,
                        eta=eta,
                        context=None,
                        debug_nan=True,
                    )

                    pos_list = unbatch(pos_gen, batch.batch)
                    atom_list = unbatch(atom_type, batch.batch)

                    for m in range(len(pos_list)):
                        if saved_total >= num_samples:
                            break

                        pos = pos_list[m]
                        atom_pred = atom_list[m]
                        atom_logits = atom_pred[:, :-1]
                        atom_idx = torch.argmax(atom_logits, dim=1)

                        position_list.append(pos.cpu().detach())
                        atom_type_list.append(atom_idx.cpu().detach())
                        generated_total += 1

                        mol = build_molecule(pos, atom_idx, dataset_info)
                        smile = mol2smiles(mol)

                        if smile is not None:
                            valid += 1
                            if "." not in smile:
                                stable += 1

                            try:
                                mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                                if len(mol_frags) > 0:
                                    mol = max(mol_frags, key=lambda mm: mm.GetNumAtoms())
                            except Exception:
                                pass

                            smile = mol2smiles(mol)
                            if smile is not None:
                                smile_list.append(smile)

                            final_mol, status = postprocess_molecule(
                                mol,
                                add_hs=add_hs,
                                optimize=optimize,
                                max_iters=ff_max_iters,
                            )

                            if final_mol is not None and final_mol.GetNumAtoms() > 0:
                                file_name = f"mol_{saved_total:03d}.sdf"
                                sdf_path = os.path.join(sdf_dir, file_name)

                                writer = Chem.SDWriter(sdf_path)
                                writer.write(final_mol)
                                writer.close()

                                try:
                                    display_smiles = Chem.MolToSmiles(Chem.RemoveHs(final_mol))
                                except Exception:
                                    display_smiles = smile if smile is not None else ""

                                saved_entries.append({
                                    "id": saved_total,
                                    "file_name": file_name,
                                    "sdf_path": sdf_path,
                                    "smiles": display_smiles,
                                    "num_atoms": int(final_mol.GetNumAtoms()),
                                    "num_bonds": int(final_mol.GetNumBonds()),
                                    "status": status,
                                })
                                saved_total += 1

                    batch_finished = True

                except FloatingPointError:
                    retry_count += 1
                    if current_clip > 5.0:
                        current_clip = 5.0
                    elif current_clip > 2.5:
                        current_clip = 2.5
                    else:
                        current_clip = 2.0

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if saved_total >= num_samples:
                break

        if saved_total >= num_samples:
            break

    final_denom = max(1, generated_total)
    final_validity = valid / final_denom
    final_stability = stable / final_denom
    final_uniqueness = len(set(smile_list)) / final_denom if len(smile_list) > 0 else 0.0

    uniq = list(set(smile_list))
    if dataset == "qm9":
        dataset_smile_list = retrieve_qm9_smiles()
    else:
        dataset_smile_list = retrieve_geom_smiles()

    novel = [s for s in uniq if s not in dataset_smile_list]
    final_novelty = len(novel) / max(1, len(uniq))

    metrics = {
        "dataset": dataset,
        "checkpoint": ckpt_path,
        "num_requested": num_samples,
        "num_generated": generated_total,
        "num_saved": saved_total,
        "validity": final_validity,
        "stability": final_stability,
        "uniqueness": final_uniqueness,
        "novelty": final_novelty,
        "sampling_type": sampling_type,
        "clip": clip,
        "w_global_pos": w_global_pos,
        "w_local_pos": w_local_pos,
        "w_global_node": w_global_node,
        "w_local_node": w_local_node,
        "elapsed_seconds": time.time() - start_all,
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "molecules.json"), "w", encoding="utf-8") as f:
        json.dump(saved_entries, f, indent=2, ensure_ascii=False)

    return metrics, saved_entries, output_dir