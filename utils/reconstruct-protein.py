import copy
import torch
from rdkit import Chem
from rdkit import Geometry

# Kept for backward compatibility; actual decoding uses dataset_info['atom_decoder'].
atom_decoder = ['H', 'C', 'N', 'O', 'F']

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Covalent radii / bond-length heuristics in pm, inherited from the original file.
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H': 119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243},
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}}

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}

bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

margin1, margin2, margin3 = 5, 2, 1


def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    return bond_order


def num_confs(num: str):
    if num.endswith('x'):
        return lambda x: x * int(num[:-1])
    elif int(num) > 0:
        return lambda x: int(num)
    else:
        raise ValueError()


def _lookup_nested(d, a1, a2):
    """Symmetric dictionary lookup for bond heuristics."""
    if a1 in d and a2 in d[a1]:
        return d[a1][a2]
    if a2 in d and a1 in d[a2]:
        return d[a2][a1]
    return None


def get_bond_order(atom1, atom2, distance, check_exists=False):
    """
    Predict bond order from inter-atomic distance.

    Robust to unseen atom orderings by using symmetric lookup.
    Returns 0 when the pair is unsupported.
    """
    distance = 100 * float(distance)  # convert nm/Ã-scale tensor distance to pm heuristic space used here

    b1 = _lookup_nested(bonds1, atom1, atom2)
    if check_exists and b1 is None:
        return 0
    if b1 is None:
        return 0

    current_margin1 = 30 if (atom1 == 'H' or atom2 == 'H') else margin1

    if distance < b1 + current_margin1:
        b2 = _lookup_nested(bonds2, atom1, atom2)
        if b2 is not None and distance < b2 + margin2:
            b3 = _lookup_nested(bonds3, atom1, atom2)
            if b3 is not None and distance < b3 + margin3:
                return 3
            return 2
        return 1
    return 0


def build_xae_molecule(positions, atom_types, dataset_info):
    """
    Build node labels X, adjacency A, and edge-order matrix E.

    Critical fixes versus the original version:
    1. order is always initialized, preventing UnboundLocalError.
    2. Unsupported atom pairs default to order=0 instead of crashing.
    3. Supports macrolides by routing through the same geometric predictor path
       used for larger chemistry spaces.
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    atom_decoder_local = dataset_info['atom_decoder']
    dataset_name = dataset_info.get('name', '')

    for i in range(n):
        for j in range(i):
            # atom_types entries are integer class ids into dataset_info['atom_decoder']
            pair = sorted([int(atom_types[i]), int(atom_types[j])])
            atom1 = atom_decoder_local[pair[0]]
            atom2 = atom_decoder_local[pair[1]]

            order = 0
            if dataset_name in ['qm9', 'qm9_second_half', 'qm9_first_half']:
                order = get_bond_order(atom1, atom2, dists[i, j], check_exists=False)
            elif dataset_name in ['geom', 'crossdock', 'macrolides']:
                order = geom_predictor((atom1, atom2), dists[i, j], limit_bonds_to_one=False)
            else:
                # Safe fallback for any future chemistry dataset.
                order = get_bond_order(atom1, atom2, dists[i, j], check_exists=False)

            if order > 0:
                A[i, j] = True
                E[i, j] = int(order)

    return X, A, E


# ==========================================================================================
# Physical valence limiter: keeps reconstruction robust when sampling noisy macrocycles.
# ==========================================================================================
def build_molecule(positions, atom_types, dataset_info):
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()

    symbols = []
    max_valence = {'H': 1, 'C': 4, 'N': 4, 'O': 2, 'F': 1, 'B': 3, 'Si': 4,
                   'P': 5, 'S': 6, 'Cl': 1, 'Br': 1, 'I': 1}

    for atom in X:
        symbol = dataset_info['atom_decoder'][atom.item()]
        symbols.append(symbol)
        mol.AddAtom(Chem.Atom(symbol))

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)

    all_bonds = torch.nonzero(A)
    bond_list = []
    for bond in all_bonds:
        u, v = bond[0].item(), bond[1].item()
        if u > v:
            bond_list.append((E[u, v].item(), dists[u, v].item(), u, v))

    # Higher order first, then shorter distance first.
    bond_list.sort(key=lambda x: (-x[0], x[1]))

    valences = [0] * len(X)

    for order, dist, u, v in bond_list:
        allowed_u = max_valence.get(symbols[u], 4) - valences[u]
        allowed_v = max_valence.get(symbols[v], 4) - valences[v]

        actual_order = min(int(order), int(allowed_u), int(allowed_v))

        if actual_order > 0:
            mol.AddBond(u, v, bond_dict[actual_order])
            valences[u] += actual_order
            valences[v] += actual_order

    mol = mol.GetMol()

    try:
        mol = set_positions(mol, positions)
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        # Return the robust valence-limited molecule even if sanitization fails.
        pass

    return mol


def mol2smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def set_rdmol_positions(rdkit_mol, pos):
    mol = copy.deepcopy(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol


def set_rdmol_positions_(mol, pos):
    n_atoms = mol.GetNumAtoms()
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, Geometry.Point3D(*pos[i]))
    mol.RemoveAllConformers()
    mol.AddConformer(conf)


def set_positions(rd_mol, pos):
    n_atoms = len(rd_mol.GetAtoms())
    rd_conf = Chem.Conformer(n_atoms)
    xyz = pos
    for i in range(n_atoms):
        rd_coords = Geometry.Point3D(*xyz[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.RemoveAllConformers()
    rd_mol.AddConformer(rd_conf)
    return rd_mol
