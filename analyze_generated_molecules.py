from pathlib import Path
from rdkit import Chem
import numpy as np

SDF_DIR = Path("results/ddpm_sdfs-1200pt")

def load_molecules():

    mols = []

    for sdf in SDF_DIR.glob("*.sdf"):

        try:
            suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
            mol = suppl[0]

            if mol:
                mols.append(mol)

        except:
            continue

    return mols


def analyze():

    mols = load_molecules()

    total = len(mols)

    valid = 0
    smiles_set = set()
    multi_fragment = 0
    atom_counts = []
    ring_counts = []

    for mol in mols:

        try:

            Chem.SanitizeMol(mol)

            valid += 1

            smi = Chem.MolToSmiles(mol)
            smiles_set.add(smi)

            frags = Chem.GetMolFrags(mol)

            if len(frags) > 1:
                multi_fragment += 1

            atom_counts.append(mol.GetNumAtoms())
            ring_counts.append(mol.GetRingInfo().NumRings())

        except:
            continue

    uniqueness = len(smiles_set) / valid if valid else 0

    print("\n===== Generation Statistics =====\n")

    print("Total molecules:", total)

    print("Validity:",
          valid,
          f"({valid/total:.3f})")

    print("Uniqueness:",
          len(smiles_set),
          f"({uniqueness:.3f})")

    print("Multi-fragment:",
          multi_fragment,
          f"({multi_fragment/valid:.3f})")

    print("Average atoms:",
          np.mean(atom_counts))

    print("Average rings:",
          np.mean(ring_counts))


if __name__ == "__main__":
    analyze()