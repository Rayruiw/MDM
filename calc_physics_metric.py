import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit import DataStructs

# 屏蔽 RDKit 的底层 C++ 报错信息，让控制台输出保持干净
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def calculate_uff_metrics(mol, clash_threshold=10000.0):
    """计算 UFF 能量，并判断是否发生严重的空间位阻碰撞"""
    # 必须添加 3D 氢原子，否则力场计算会极不准确
    mol_with_h = Chem.AddHs(mol, addCoords=True)
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol_with_h)
        if ff is None: 
            return True, None # 无法构建力场，直接判定为严重碰撞
        
        raw_energy = ff.CalcEnergy()
        
        # 如果初始能量大于阈值，说明原子重叠了 (Steric Clash)
        if raw_energy > clash_threshold:
            return True, None 
            
        # 运行力场松弛优化
        AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=1000)
        ff_opt = AllChem.UFFGetMoleculeForceField(mol_with_h)
        opt_energy = ff_opt.CalcEnergy()
        
        delta_e = raw_energy - opt_energy
        return False, delta_e
    except Exception:
        return True, None

def calculate_diversity(mols):
    """基于 Morgan Fingerprint (ECFP4) 计算 Tanimoto 内部多样性"""
    fps = []
    for mol in mols:
        try:
            # 提取摩根指纹 (半径2，相当于 ECFP4，2048位)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
        except Exception:
            continue
    
    if len(fps) < 2:
        return 0.0
        
    similarity_sum = 0.0
    pairs = 0
    # 计算两两之间的 Tanimoto 相似度
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarity_sum += sim
            pairs += 1
            
    mean_sim = similarity_sum / pairs
    diversity = 1.0 - mean_sim
    return diversity

def analyze_sdfs(sdf_dir):
    sdf_files = glob.glob(os.path.join(sdf_dir, '*.sdf'))
    if len(sdf_files) == 0:
        print(f"❌ 错误: 在目录 {sdf_dir} 中没有找到任何 .sdf 文件！请检查路径。")
        return

    print(f"\n📂 成功找到 {len(sdf_files)} 个 SDF 文件于: {sdf_dir}")
    
    valid_mols = []
    delta_e_list = []
    clash_count = 0
    total_evaluated = 0
    
    for sdf_file in tqdm(sdf_files, desc="🧪 正在进行物理与化学评估"):
        supplier = Chem.SDMolSupplier(sdf_file)
        for mol in supplier:
            if mol is None:
                # RDKit 读取失败（坐标错乱导致化合价完全崩溃），记为严重碰撞
                total_evaluated += 1
                clash_count += 1
                continue
            
            total_evaluated += 1
            valid_mols.append(mol)
            
            # 计算物理能量与碰撞
            is_clash, d_e = calculate_uff_metrics(mol)
            if is_clash:
                clash_count += 1
            else:
                if d_e is not None:
                    delta_e_list.append(d_e)

    # ==========================================
    # 📊 打印排版优美的论文级数据报告
    # ==========================================
    print("\n" + "="*60)
    print(f"🌟 论文级核心指标报告 (Metrics Report) 🌟")
    print("="*60)
    print(f"📁 测试路径: {sdf_dir}")
    print(f"🧪 总评估分子数: {total_evaluated}")
    
    # 1. 物理合理性 (Steric Clash & Delta E)
    clash_rate = (clash_count / total_evaluated) * 100 if total_evaluated > 0 else 100
    print(f"\n[1] 物理 3D 合理性 (Physical Rationality)")
    print(f"    -> 💥 Steric Clash Rate (碰撞率): {clash_rate:.2f}%  (越低越好)")
    
    if len(delta_e_list) > 0:
        delta_e_array = np.array(delta_e_list)
        print(f"    -> 📉 Mean Delta E (平均能量差):  {np.mean(delta_e_array):.2f} kcal/mol")
        print(f"    -> 🎯 Median Delta E (中位能量差):{np.median(delta_e_array):.2f} kcal/mol  (🔥 论文核心比较项，越低越好)")
    else:
        print("    -> ⚠️ Delta E: N/A (所有分子均发生碰撞)")

    # 2. 生成多样性 (Tanimoto Diversity)
    print(f"\n[2] 化学空间多样性 (Chemical Diversity)")
    if len(valid_mols) > 0:
        diversity = calculate_diversity(valid_mols)
        print(f"    -> 🧬 Tanimoto Diversity:        {diversity:.4f}  (🔥 越高越好)")
    else:
        print("    -> ⚠️ Tanimoto Diversity:        N/A")
    print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Physical and Chemical Metrics for 3D Molecules")
    parser.add_argument('--dir', type=str, required=True, help='Path to the directory containing SDF files')
    args = parser.parse_args()
    
    analyze_sdfs(args.dir)

#COMMAND
#python calc_physics_metric.py --dir ./results/ddpm_sdfs-1200pt
#python calc_physics_metric.py --dir ./logs/qm9_full_ddpm_2losses_2025_11_01__20_23_28_resume/generalized_2021_sigmainf_aligned_result_2026_03_11__20_07_54_result/results/sdf