import os, sys, pickle, random, time, traceback
from glob import glob

def log(*a): print(*a, flush=True)
def exists(p): return os.path.isfile(os.path.expanduser(p))

def main():
    t0 = time.time()
    log("=== make_geom_pickles_safe START ===")
    log("CWD:", os.getcwd())
    log("PYTHON:", sys.version)
    log("PYTHONPATH:", os.environ.get("PYTHONPATH",""))
    # 0) 检查原始 npy
    npy_dir = os.path.join(".", "data", "GEOM", "Drugs", "geom")
    f1 = os.path.join(npy_dir, "geom_drugs_1.npy")
    f2 = os.path.join(npy_dir, "geom_drugs_n_1.npy")
    log("Check npy:", f1); log("Check npy:", f2)
    if not (exists(f1) and exists(f2)):
        log("ERROR: 缺少 npy。请把两份 npy 放到 ./data/GEOM/Drugs/geom/ 下。")
        return

    # 1) 导入 Geom
    try:
        from utils.datasets import Geom
    except Exception as e:
        log("ERROR: 无法 import utils.datasets.Geom（路径/依赖问题）")
        traceback.print_exc()
        return

    # 2) 触发 Geom(raw->processed)
    try:
        root = "./data/GEOM/Drugs"
        log("Instantiating Geom(root=%s) ..." % root)
        ds = Geom(root=root)  # 若还没 processed，会自动执行 process()
        log("Geom processed file should be at:", ds.processed_paths[0])
        if not exists(ds.processed_paths[0]):
            log("WARNING: processed 文件看起来没生成：", ds.processed_paths[0])
            log("可能原因：process() 内部报错、保存路径权限问题、或 torch.save 路径被覆盖。")
    except Exception:
        log("ERROR: Geom 初始化/处理失败：")
        traceback.print_exc()
        return

    # 3) 读取数据
    try:
        n = len(ds)
        log("Geom dataset size:", n)
        if n == 0:
            log("ERROR: 数据集为空（.npy 内容不对或 process 逻辑异常）。")
            return
        from tqdm import tqdm
        data_list = [ds[i] for i in tqdm(range(n), desc="Collect Data")]
        # 简单 sanity check
        has_atom_type = hasattr(data_list[0], "atom_type")
        has_pos = hasattr(data_list[0], "pos")
        log(f"Sample[0] keys -> atom_type:{has_atom_type}, pos:{has_pos}")
    except Exception:
        log("ERROR: 读取 Geom 数据失败：")
        traceback.print_exc()
        return

    # 4) 切分与保存 pkl
    try:
        out_dir = os.path.join(".", "data", "GEOM", "QM9")
        os.makedirs(out_dir, exist_ok=True)
        target = (40000, 5000, 1000)
        random.seed(2021)
        idx = list(range(len(data_list)))
        random.shuffle(idx)

        if len(idx) < sum(target):
            log("样本不足，使用 80/10/10 切分。")
            n_total = len(idx)
            n_train = int(0.8 * n_total)
            n_val   = int(0.1 * n_total)
            n_test  = n_total - n_train - n_val
            tr, va, te = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
            tp = os.path.join(out_dir, f"train_data_{n_train//1000}k.pkl")
            vp = os.path.join(out_dir, f"val_data_{n_val//1000}k.pkl")
            ep = os.path.join(out_dir, f"test_data_{n_test//1000}k.pkl")
        else:
            tr, va, te = idx[:target[0]], idx[target[0]:target[0]+target[1]], idx[target[0]+target[1]:sum(target)]
            tp = os.path.join(out_dir, "train_data_40k.pkl")
            vp = os.path.join(out_dir, "val_data_5k.pkl")
            ep = os.path.join(out_dir, "test_data_1k.pkl")

        def dump(pth, ids):
            with open(pth, "wb") as f:
                pickle.dump([data_list[i] for i in ids], f)
            log(f"Saved: {pth}  count={len(ids)}  size={os.path.getsize(pth)/1024/1024:.2f} MB")

        log("Saving pickles ...")
        dump(tp, tr); dump(vp, va); dump(ep, te)

        log("\nOutput files:")
        for p in glob(os.path.join(out_dir, "*.pkl")):
            log(" -", p)
    except Exception:
        log("ERROR: 保存 pkl 失败：")
        traceback.print_exc()
        return

    log(f"=== DONE in {time.time()-t0:.1f}s ===")

if __name__ == "__main__":
    main()