import os
import json
import math
import random
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdchem import BondType

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn.pool import radius_graph


BOND_TYPE_TO_INT = {
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}

DEFAULT_ALLOWED_ATOMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

# Strict alignment with configs/datasets_config.py::macrolides_with_h
# atom_index = {1:0, 6:1, 7:2, 8:3, 9:4, 15:5, 16:6, 17:7, 35:8, 53:9}
# train.py then sets config.model.num_atom = len(atom_decoder) + 1 = 11.
# Therefore atom_feat_full must be [num_atoms, 11], with slot 0 reserved.
MODEL_ATOM_INDEX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9}
MODEL_NUM_ATOM_CHANNELS = len(MODEL_ATOM_INDEX) + 1


class MacrolidesDataset(InMemoryDataset):
    """
    Macrolides dataset loader for Macrocycle-DB CSV files.

    This dataset reads a CSV containing at least the columns:
        - MCD_ID
        - SMILES
        - Largest Ring Size

    It converts each SMILES to an RDKit molecule, generates a single 3D conformer,
    builds a molecular graph, and stores PyG Data objects with the following fields:

        Required structural fields
        --------------------------
        z               : [num_atoms] atomic numbers
        pos             : [num_atoms, 3] 3D coordinates
        edge_index      : [2, num_edges] directed bond edges
        edge_attr       : [num_edges, 4] [bond_type_id, is_conjugated, is_aromatic, is_ring_bond]

        Helpful optional fields
        -----------------------
        x               : [num_atoms, 5] node scalar features
                          [atomic_num, formal_charge, is_aromatic, is_in_ring, is_in_largest_ring]
        smiles          : original SMILES string
        mol_id          : Macrocycle-DB ID string
        largest_ring_size : scalar tensor
        mw, logp, hba, hbd, psa, rotb : selected descriptors if present in CSV

    Notes
    -----
    1) This class is intentionally model-agnostic. Most 3D molecular diffusion / score
       models can consume z, pos, edge_index, and edge_attr directly.
    2) If your project expects a custom field name (for example `atom_type` instead of `z`),
       adapt that in `mol_to_data` only.
    3) Hydrogens are kept by default because many 3D generation pipelines train on explicit-H graphs.
    """

    def __init__(
        self,
        root: str,
        csv_name: str = "Macrolides.csv",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        add_hs: bool = True,
        min_largest_ring_size: int = 12,
        max_atoms: int = 100,
        allowed_atomic_nums: Optional[Sequence[int]] = None,
        sanitize: bool = True,
        conformer_seed: int = 2026,
        use_mmff: bool = True,
        use_uff_fallback: bool = True,
        add_radius_edges: bool = False,
        radius: float = 5.0,
        max_num_neighbors: int = 32,
        split_seed: int = 2026,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        force_reprocess: bool = False,
    ):
        self.csv_name = csv_name
        self.add_hs = add_hs
        self.min_largest_ring_size = min_largest_ring_size
        self.max_atoms = max_atoms
        self.allowed_atomic_nums = set(allowed_atomic_nums or DEFAULT_ALLOWED_ATOMS)
        self.sanitize = sanitize
        self.conformer_seed = conformer_seed
        self.use_mmff = use_mmff
        self.use_uff_fallback = use_uff_fallback
        self.add_radius_edges = add_radius_edges
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.force_reprocess = force_reprocess

        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        if force_reprocess:
            processed_file = os.path.join(root, "processed", self.processed_file_names[0])
            split_file = os.path.join(root, "processed", self.processed_file_names[1])
            for path in [processed_file, split_file]:
                if os.path.exists(path):
                    os.remove(path)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.splits = torch.load(self.processed_paths[1], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.csv_name]

    @property
    def processed_file_names(self) -> List[str]:
        stem = os.path.splitext(self.csv_name)[0]
        return [f"{stem}_data.pt", f"{stem}_splits.pt"]

    def download(self):
        # User supplies the CSV manually.
        pass

    def process(self):
        csv_path = self.raw_paths[0]
        df = pd.read_csv(csv_path)

        required_cols = ["MCD_ID", "SMILES", "Largest Ring Size"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing}")

        data_list: List[Data] = []
        skipped = {
            "missing_smiles": 0,
            "parse_fail": 0,
            "multi_fragment": 0,
            "ring_too_small": 0,
            "atom_too_large": 0,
            "disallowed_atom": 0,
            "conformer_fail": 0,
            "empty_graph": 0,
            "pre_filter": 0,
        }

        for idx, row in df.iterrows():
            smiles = str(row.get("SMILES", "")).strip()
            mol_id = str(row.get("MCD_ID", f"mol_{idx}")).strip()
            if not smiles or smiles.lower() == "nan":
                skipped["missing_smiles"] += 1
                continue
            if "." in smiles:
                skipped["multi_fragment"] += 1
                continue

            try:
                mol = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
            except Exception:
                mol = None

            if mol is None:
                skipped["parse_fail"] += 1
                continue

            csv_ring = safe_int(row.get("Largest Ring Size", 0), default=0)
            rdkit_ring = largest_ring_size_from_mol(mol)
            largest_ring = max(csv_ring, rdkit_ring)
            if largest_ring < self.min_largest_ring_size:
                skipped["ring_too_small"] += 1
                continue

            if self.add_hs:
                mol = Chem.AddHs(mol)

            if mol.GetNumAtoms() > self.max_atoms:
                skipped["atom_too_large"] += 1
                continue

            atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            if any(z not in self.allowed_atomic_nums for z in atomic_nums):
                skipped["disallowed_atom"] += 1
                continue

            ok = embed_and_optimize_mol(
                mol,
                seed=self.conformer_seed + idx,
                use_mmff=self.use_mmff,
                use_uff_fallback=self.use_uff_fallback,
            )
            if not ok:
                skipped["conformer_fail"] += 1
                continue

            data = mol_to_data(
                mol=mol,
                mol_id=mol_id,
                smiles=smiles,
                csv_row=row,
                largest_ring_size=largest_ring,
                add_radius_edges=self.add_radius_edges,
                radius=self.radius,
                max_num_neighbors=self.max_num_neighbors,
            )

            if data is None or data.num_nodes == 0:
                skipped["empty_graph"] += 1
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                skipped["pre_filter"] += 1
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        if len(data_list) == 0:
            raise RuntimeError(
                "No valid molecules were processed. Check CSV format, RDKit installation, and filtering rules."
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(self._make_splits(len(data_list)), self.processed_paths[1])

        summary = {
            "num_total_rows": int(len(df)),
            "num_processed": int(len(data_list)),
            "num_skipped": int(sum(skipped.values())),
            "skip_breakdown": skipped,
            "config": {
                "csv_name": self.csv_name,
                "add_hs": self.add_hs,
                "min_largest_ring_size": self.min_largest_ring_size,
                "max_atoms": self.max_atoms,
                "allowed_atomic_nums": sorted(self.allowed_atomic_nums),
                "conformer_seed": self.conformer_seed,
                "use_mmff": self.use_mmff,
                "use_uff_fallback": self.use_uff_fallback,
                "add_radius_edges": self.add_radius_edges,
                "radius": self.radius,
                "max_num_neighbors": self.max_num_neighbors,
            },
        }
        with open(os.path.join(self.processed_dir, "processing_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _make_splits(self, n: int) -> Dict[str, Tensor]:
        g = torch.Generator().manual_seed(self.split_seed)
        perm = torch.randperm(n, generator=g)

        n_train = int(round(n * self.train_ratio))
        n_val = int(round(n * self.val_ratio))
        n_test = n - n_train - n_val

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:n_train + n_val + n_test]

        return {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

    def get_split(self, split: str):
        split = split.lower()
        if split not in self.splits:
            raise KeyError(f"Unknown split: {split}. Available: {list(self.splits.keys())}")
        return self.index_select(self.splits[split])


def safe_float(v, default: float = float("nan")) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def largest_ring_size_from_mol(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if not atom_rings:
        return 0
    return max(len(r) for r in atom_rings)

def embed_and_optimize_mol(
    mol: Chem.Mol,
    seed: int = 2026,
    use_mmff: bool = True,
    use_uff_fallback: bool = True,
    max_embed_attempts: int = 3,) -> bool:
    """
    Generate one 3D conformer for macrocycles.

    Strategy:
    1) ETKDGv3 embedding with macrocycle-aware settings
    2) MMFF optimization if parameters are available
    3) otherwise UFF fallback
    """

    def _make_params(cur_seed: int, use_random_coords: bool):
        p = AllChem.ETKDGv3()
        p.randomSeed = int(cur_seed)
        p.useRandomCoords = bool(use_random_coords)
        p.pruneRmsThresh = -1.0
        p.enforceChirality = True

        if hasattr(p, "useMacrocycleTorsions"):
            p.useMacrocycleTorsions = True
        if hasattr(p, "useMacrocycle14config"):
            p.useMacrocycle14config = True

        if hasattr(p, "maxIterations"):
            p.maxIterations = 1000
        if hasattr(p, "timeout"):
            p.timeout = 30

        return p

    embedded = False
    for k in range(max_embed_attempts):
        try:
            work_params = _make_params(
                cur_seed=seed + 17 * k,
                use_random_coords=(k > 0),
            )
            status = AllChem.EmbedMolecule(mol, work_params)
            if status == 0:
                embedded = True
                break
        except Exception:
            continue

    if not embedded:
        return False

    try:
        if use_mmff:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if mmff_props is not None:
                AllChem.MMFFOptimizeMolecule(
                    mol,
                    mmffVariant="MMFF94s",
                    maxIters=500
                )
                return True

        if use_uff_fallback:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            return True

        return True

    except Exception:
        return False


def build_atom_feat_full(z: Tensor) -> Tensor:
    """
    Build the node feature tensor consumed by PaiNN_mdm.py.

    The current PaiNN MDM projects `batch.atom_feat_full` with input dimension
    `config.model.num_atom`, and train.py sets that to
    `len(dataset_info["atom_decoder"]) + 1`. For macrolides_with_h this equals 11.
    We therefore construct a strict 11-channel one-hot tensor with slot 0 reserved.
    """
    atom_feat_full = torch.zeros((z.numel(), MODEL_NUM_ATOM_CHANNELS), dtype=torch.float)
    for i, atomic_num in enumerate(z.tolist()):
        if atomic_num not in MODEL_ATOM_INDEX:
            raise ValueError(f"Atomic number {atomic_num} is not registered in MODEL_ATOM_INDEX.")
        atom_feat_full[i, MODEL_ATOM_INDEX[atomic_num] + 1] = 1.0
    return atom_feat_full


def mol_to_data(
    mol: Chem.Mol,
    mol_id: str,
    smiles: str,
    csv_row: pd.Series,
    largest_ring_size: int,
    add_radius_edges: bool = False,
    radius: float = 5.0,
    max_num_neighbors: int = 32,
) -> Optional[Data]:
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    pos = []
    z = []
    x = []

    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    if atom_rings:
        largest_ring = max(atom_rings, key=len)
        largest_ring_atoms = set(largest_ring)
    else:
        largest_ring_atoms = set()

    for atom_idx, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(atom_idx)
        pos.append([p.x, p.y, p.z])
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())
        is_in_largest_ring = int(atom_idx in largest_ring_atoms)
        z.append(atomic_num)
        x.append([atomic_num, formal_charge, is_aromatic, is_in_ring, is_in_largest_ring])

    pos = torch.tensor(pos, dtype=torch.float)
    pos = pos - pos.mean(dim=0, keepdim=True)
    z = torch.tensor(z, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)
    atom_feat_full = build_atom_feat_full(z)

    edge_index_list = []
    edge_attr_list = []
    bond_pairs = set()

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = BOND_TYPE_TO_INT.get(bond.GetBondType(), 0)
        e = [bt, int(bond.GetIsConjugated()), int(bond.GetIsAromatic()), int(bond.IsInRing())]

        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_attr_list.append(e)
        edge_attr_list.append(e)
        bond_pairs.add((i, j))
        bond_pairs.add((j, i))

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    if edge_attr.size(0) > 0:
        edge_type = edge_attr[:, 0].long()
    else:
        edge_type = torch.empty((0,), dtype=torch.long)

    data = Data(
        z=z,
        atom_type=z.clone(),
        atom_feat_full=atom_feat_full,
        pos=pos,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        num_nodes=num_atoms,
        smiles=smiles,
        mol_id=mol_id,
        largest_ring_size=torch.tensor([largest_ring_size], dtype=torch.long),
        mw=torch.tensor([safe_float(csv_row.get("MW"))], dtype=torch.float),
        logp=torch.tensor([safe_float(csv_row.get("LogP"))], dtype=torch.float),
        hba=torch.tensor([safe_float(csv_row.get("HBA"))], dtype=torch.float),
        hbd=torch.tensor([safe_float(csv_row.get("HBD"))], dtype=torch.float),
        psa=torch.tensor([safe_float(csv_row.get("PSA"))], dtype=torch.float),
        rotb=torch.tensor([safe_float(csv_row.get("RotB"))], dtype=torch.float),
    )

    if add_radius_edges:
        rg_edge_index = radius_graph(
            pos,
            r=radius,
            loop=False,
            max_num_neighbors=max_num_neighbors,
        )
        if rg_edge_index.numel() == 0:
            data.radius_edge_index = torch.empty((2, 0), dtype=torch.long)
            data.radius_edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            rg_attr = []
            for src, dst in rg_edge_index.t().tolist():
                dist = float((pos[src] - pos[dst]).norm(p=2).item())
                is_bond = 1.0 if (src, dst) in bond_pairs else 0.0
                rg_attr.append([dist, is_bond])
            data.radius_edge_index = rg_edge_index
            data.radius_edge_attr = torch.tensor(rg_attr, dtype=torch.float)

    return data


if __name__ == "__main__":
    """
    Example usage
    -------------
    Put your CSV at:
        <root>/raw/Macrolides.csv

    Then run:
        python Macrolides_dataset.py
    """
    root = "./macrolides_dataset"
    os.makedirs(os.path.join(root, "raw"), exist_ok=True)

    dataset = MacrolidesDataset(
        root=root,
        csv_name="Macrolides.csv",
        add_hs=True,
        min_largest_ring_size=12,
        max_atoms=100,
        add_radius_edges=False,
        force_reprocess=False,
    )

    print(dataset)
    print(f"Total molecules: {len(dataset)}")
    print(f"Train split: {len(dataset.get_split('train'))}")
    print(f"Val split:   {len(dataset.get_split('val'))}")
    print(f"Test split:  {len(dataset.get_split('test'))}")
    first = dataset[0]
    print(first)
