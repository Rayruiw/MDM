import os
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

# Silence RDKit warnings for cleaner batch evaluation logs.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# -----------------------------------------------------------------------------
# Supplementary diagnostics helper functions
# -----------------------------------------------------------------------------

def is_rdkit_valid_mol(mol: Optional[Chem.Mol]) -> bool:
    """
    Basic RDKit sanitization check used only as an entry filter for diagnostics.

    Note:
        This script is intended for supplementary diagnostics only. Primary
        paper metrics such as validity / uniqueness / novelty should be computed
        by the dedicated main evaluation pipeline.
    """
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def is_qm9_domain_compliant(mol: Optional[Chem.Mol], max_heavy_atoms: int = 9) -> bool:
    """
    Supplementary dataset-domain compliance check.

    This does NOT replace primary validity. It only checks whether a valid
    molecule falls into the usual QM9 domain assumptions.
    """
    if mol is None:
        return False

    heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if len(heavy_atoms) < 1 or len(heavy_atoms) > max_heavy_atoms:
        return False

    allowed_atomic_nums = {1, 6, 7, 8, 9}  # H, C, N, O, F
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in allowed_atomic_nums:
            return False

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 and atom.GetDegree() == 0:
            return False

    return True


def calculate_uff_sanity_metrics(
    mol: Chem.Mol,
    high_energy_threshold: float = 1000.0,
    max_iters: int = 200,
) -> Dict[str, Optional[float]]:
    """
    Supplementary geometry sanity analysis based on RDKit UFF.

    Returns separate fields instead of collapsing different failure modes into a
    single indicator.
    """
    result = {
        "uff_success": False,
        "high_energy": False,
        "raw_energy": None,
        "opt_energy": None,
        "energy_drop": None,
    }

    try:
        mol_with_h = Chem.AddHs(mol, addCoords=True)
        ff = AllChem.UFFGetMoleculeForceField(mol_with_h)
        if ff is None:
            return result

        raw_energy = ff.CalcEnergy()
        result["uff_success"] = True
        result["raw_energy"] = float(raw_energy)

        if raw_energy > high_energy_threshold:
            result["high_energy"] = True
            return result

        AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=max_iters)
        ff_opt = AllChem.UFFGetMoleculeForceField(mol_with_h)
        if ff_opt is None:
            return result

        opt_energy = ff_opt.CalcEnergy()
        result["opt_energy"] = float(opt_energy)
        result["energy_drop"] = float(raw_energy - opt_energy)
        return result
    except Exception:
        return result


def calculate_internal_diversity(mols: List[Chem.Mol]) -> float:
    """
    Fingerprint-based internal diversity for supplementary analysis.

    Uses Morgan fingerprints (radius=2, nBits=2048), corresponding to a common
    ECFP4-style setup in the literature.
    """
    fps = []
    for mol in mols:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(fp)
        except Exception:
            continue

    if len(fps) < 2:
        return 0.0

    sim_sum = 0.0
    pairs = 0
    for i in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        sim_sum += float(sum(sims))
        pairs += len(sims)

    if pairs == 0:
        return 0.0
    return 1.0 - sim_sum / pairs


@dataclass
class SupplementaryStats:
    total: int = 0
    valid_for_diagnostics_count: int = 0
    qm9_domain_count: int = 0
    uff_success_count: int = 0
    high_energy_count: int = 0
    read_failure_count: int = 0
    sanitize_failure_count: int = 0

    raw_energies: List[float] = None
    opt_energies: List[float] = None
    energy_drops: List[float] = None
    valid_mols: List[Chem.Mol] = None

    def __post_init__(self):
        if self.raw_energies is None:
            self.raw_energies = []
        if self.opt_energies is None:
            self.opt_energies = []
        if self.energy_drops is None:
            self.energy_drops = []
        if self.valid_mols is None:
            self.valid_mols = []


# -----------------------------------------------------------------------------
# Main supplementary diagnostics
# -----------------------------------------------------------------------------

def analyze_sdfs(
    sdf_dir: str,
    high_energy_threshold: float = 1000.0,
    uff_max_iters: int = 200,
) -> Dict[str, float]:
    sdf_files = sorted(glob.glob(os.path.join(sdf_dir, "*.sdf")))
    if not sdf_files:
        raise FileNotFoundError(f"No SDF files found in: {sdf_dir}")

    print(f"\n📂 Found {len(sdf_files)} SDF files")

    stats = SupplementaryStats()

    for sdf_path in tqdm(sdf_files, desc="🧪 Running supplementary diagnostics"):
        stats.total += 1

        mol = None
        try:
            supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
            mol = supplier[0] if supplier is not None and len(supplier) > 0 else None
        except Exception:
            mol = None

        if mol is None:
            stats.read_failure_count += 1
            continue

        if not is_rdkit_valid_mol(mol):
            stats.sanitize_failure_count += 1
            continue

        stats.valid_for_diagnostics_count += 1
        stats.valid_mols.append(mol)

        if is_qm9_domain_compliant(mol):
            stats.qm9_domain_count += 1

        uff_info = calculate_uff_sanity_metrics(
            mol,
            high_energy_threshold=high_energy_threshold,
            max_iters=uff_max_iters,
        )
        if uff_info["uff_success"]:
            stats.uff_success_count += 1
        if uff_info["high_energy"]:
            stats.high_energy_count += 1
        if uff_info["raw_energy"] is not None:
            stats.raw_energies.append(float(uff_info["raw_energy"]))
        if uff_info["opt_energy"] is not None:
            stats.opt_energies.append(float(uff_info["opt_energy"]))
        if uff_info["energy_drop"] is not None:
            stats.energy_drops.append(float(uff_info["energy_drop"]))

    qm9_domain_compliance_valid = (
        100.0 * stats.qm9_domain_count / stats.valid_for_diagnostics_count
        if stats.valid_for_diagnostics_count else 0.0
    )
    uff_success_valid = (
        100.0 * stats.uff_success_count / stats.valid_for_diagnostics_count
        if stats.valid_for_diagnostics_count else 0.0
    )
    high_energy_valid = (
        100.0 * stats.high_energy_count / stats.valid_for_diagnostics_count
        if stats.valid_for_diagnostics_count else 0.0
    )
    internal_diversity = calculate_internal_diversity(stats.valid_mols)

    raw_energy_mean = float(np.mean(stats.raw_energies)) if stats.raw_energies else 0.0
    raw_energy_median = float(np.median(stats.raw_energies)) if stats.raw_energies else 0.0
    opt_energy_mean = float(np.mean(stats.opt_energies)) if stats.opt_energies else 0.0
    opt_energy_median = float(np.median(stats.opt_energies)) if stats.opt_energies else 0.0
    energy_drop_mean = float(np.mean(stats.energy_drops)) if stats.energy_drops else 0.0
    energy_drop_median = float(np.median(stats.energy_drops)) if stats.energy_drops else 0.0

    results = {
        "total_sdf_files": stats.total,
        "read_failure_count": stats.read_failure_count,
        "sanitize_failure_count": stats.sanitize_failure_count,
        "diagnostic_pool_count": stats.valid_for_diagnostics_count,
        "qm9_domain_compliance_percent_over_diagnostic_pool": qm9_domain_compliance_valid,
        "uff_success_percent_over_diagnostic_pool": uff_success_valid,
        "high_energy_percent_over_diagnostic_pool": high_energy_valid,
        "internal_diversity": internal_diversity,
        "raw_energy_mean": raw_energy_mean,
        "raw_energy_median": raw_energy_median,
        "opt_energy_mean": opt_energy_mean,
        "opt_energy_median": opt_energy_median,
        "energy_drop_mean": energy_drop_mean,
        "energy_drop_median": energy_drop_median,
    }

    print_report(results, sdf_dir=sdf_dir)
    return results


def print_report(results: Dict[str, float], sdf_dir: str) -> None:
    print("\n" + "=" * 88)
    print("Supplementary Diagnostics Report")
    print("This script is intended only for appendix / supplementary analysis.")
    print("Primary metrics such as validity, stability, uniqueness, and novelty")
    print("should be reported by the main evaluation pipeline.")
    print("=" * 88)
    print(f"📁 Directory:                                {sdf_dir}")
    print(f"🧪 Total SDF files:                          {results['total_sdf_files']}")
    print(f"📦 Read failures:                            {results['read_failure_count']}")
    print(f"⚠️  RDKit sanitize failures:                 {results['sanitize_failure_count']}")
    print(f"🔬 Diagnostic pool size:                     {results['diagnostic_pool_count']}")
    print("-" * 88)
    print("Supplementary diagnostics")
    print(f"📚 QM9-domain compliance (% of pool):        {results['qm9_domain_compliance_percent_over_diagnostic_pool']:.2f}%")
    print(f"⚙️  UFF success (% of pool):                  {results['uff_success_percent_over_diagnostic_pool']:.2f}%")
    print(f"🔥 High-energy (% of pool):                  {results['high_energy_percent_over_diagnostic_pool']:.2f}%")
    print(f"🌌 Internal diversity:                       {results['internal_diversity']:.4f}")
    print(f"📉 Raw energy mean / median:                 {results['raw_energy_mean']:.4f} / {results['raw_energy_median']:.4f}")
    print(f"📉 Optimized energy mean / median:           {results['opt_energy_mean']:.4f} / {results['opt_energy_median']:.4f}")
    print(f"📉 Energy drop mean / median:                {results['energy_drop_mean']:.4f} / {results['energy_drop_median']:.4f}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supplementary diagnostics for generated QM9-style 3D molecules"
    )
    parser.add_argument("--dir", type=str, required=True, help="Directory containing SDF files")
    parser.add_argument(
        "--high-energy-threshold",
        type=float,
        default=1000.0,
        help="Supplementary UFF high-energy threshold (diagnostic only)",
    )
    parser.add_argument(
        "--uff-max-iters",
        type=int,
        default=200,
        help="Maximum number of UFF optimization iterations (diagnostic only)",
    )
    args = parser.parse_args()

    analyze_sdfs(
        sdf_dir=args.dir,
        high_energy_threshold=args.high_energy_threshold,
        uff_max_iters=args.uff_max_iters,
    )

# python evaluation.py --dir ./logs/qm9_full_ddpm_2losses_2025_11_01__20_23_28_resume/generalized_2021_sigmainf_aligned_result_2026_03_11__20_07_54_result/results/sdf
# python evaluation.py --dir  ./results/ddpm_sdfs-1200pt