"""学长可以在自己电脑上跑这个，看 dump 的 (A, b) 跟 x_xref 一不一致。

用法（任何机器，scipy 能装就行）：
    python3 verify_dump_consistency.py /path/to/melting_or_evaporation_root

预期输出（如果 dump 完整 + 边界修正过）：
    ‖A·x_xref - b‖₂ / ‖b‖₂  ≈ 1e-8   (= OF tol)

实际我们在 dev 上跑出来：
    ‖A·x_xref - b‖₂ / ‖b‖₂  ≈ 18     (差 9 个数量级，绝不可能是数值噪声)

如果学长这边一跑也是 18，那 dump 流程确实漏了点什么；
如果他这边跑出来是 1e-8，那 dev 这边是 CSV 解析 bug，需要查。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def load_value(p: Path):
    rows = []
    with p.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5: continue
            try: rows.append((int(parts[0]), float(parts[4])))
            except ValueError: continue
    a = np.array(rows, dtype=np.float64)
    return a[np.argsort(a[:, 0])][:, 1]


def load_matrix(p: Path, n: int):
    rs, cs, vs = [], [], []
    with p.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 9: continue
            try:
                rs.append(int(parts[0])); cs.append(int(parts[4]))
                vs.append(float(parts[8]))
            except ValueError: continue
    rs = np.asarray(rs, dtype=np.int32)
    cs = np.asarray(cs, dtype=np.int32)
    vs = np.asarray(vs, dtype=np.float64)
    off = rs != cs
    return csr_matrix(
        (np.concatenate([vs, vs[off]]),
         (np.concatenate([rs, cs[off]]), np.concatenate([cs, rs[off]]))),
        shape=(n, n)).tocsr()


def check_one(label: str, mp: Path, sp: Path, x0p: Path, pp: Path):
    print(f"\n=== {label} ===")
    print(f"  matrix : {mp.name}")
    print(f"  source : {sp.name}")
    print(f"  x_pre  : {x0p.name}")
    print(f"  x_post : {pp.name}")
    if not all(p.exists() for p in [mp, sp, x0p, pp]):
        print("  !! some files missing"); return

    b = load_value(sp)
    x_pre = load_value(x0p)        # Pre_Solving/solution
    x_post = load_value(pp)        # After_Solving/pd_PISO
    n = b.size
    A = load_matrix(mp, n)

    print(f"  N={n}, A.nnz={A.nnz}")
    print(f"  b stats: min={b.min():.3e}, max={b.max():.3e}, "
          f"sum/|sum|={b.sum()/max(np.abs(b).sum(),1e-300):.4f}")
    print(f"     (sum/|sum| ≈ 0 → b 在 col(A) 内可解;  ≈ ±1 → b 偏向单方向, 可能不在 col(A))")

    denom_b = max(np.linalg.norm(b), 1e-300)
    rel_pre  = float(np.linalg.norm(A @ x_pre  - b) / denom_b)
    rel_post = float(np.linalg.norm(A @ x_post - b) / denom_b)
    print(f"  ‖A·x_pre  - b‖₂/‖b‖₂ = {rel_pre:.3e}     (Pre_Solving/solution)")
    print(f"  ‖A·x_post - b‖₂/‖b‖₂ = {rel_post:.3e}     (After_Solving/pd_PISO)")
    if min(rel_pre, rel_post) < 1e-3:
        print("  ✅ 至少一个 x 跟 (A,b) 自洽到 < 1e-3, dump 应该 OK")
    else:
        print(f"  ❌ 两个 x 都跟 (A,b) 严重不自洽 (>{min(rel_pre, rel_post):.1e})")
        print("     dump 出来的 (A,b) 跟 x_xref 不在同一道方程上")


def main():
    if len(sys.argv) >= 2:
        root = Path(sys.argv[1])
    else:
        # 自动找
        repo = Path(__file__).resolve().parents[2]
        root = repo / "benchmark" / "Melting" / "Melting"
        if not root.exists():
            root = repo / "benchmark" / "Evaporation" / "Evaporation"
        if not root.exists():
            print("用法: python verify_dump_consistency.py "
                  "/path/to/{Melting/Melting | Evaporation/Evaporation}")
            sys.exit(1)
    print(f"Dataset root: {root}")

    pre = root / "Pre_Solving"
    after = root / "After_Solving"

    # Auto-pick a few representative steps
    matrix_files = sorted(pre.glob("matrix_pd_*.csv"))
    matrix_files = [f for f in matrix_files if not f.name.endswith("Zone.Identifier")]
    if not matrix_files:
        print("找不到 matrix_pd_*.csv"); sys.exit(2)

    # Test first, middle, last
    indices = [0, len(matrix_files) // 2, len(matrix_files) - 1]
    seen = set()
    for idx in indices:
        if idx in seen: continue
        seen.add(idx)
        mp = matrix_files[idx]
        # name: matrix_pd_<step>_<corr>.csv
        stem = mp.stem  # e.g. matrix_pd_70_1
        parts = stem.split("_")
        step, corr = parts[2], parts[3]
        sp = pre / f"source_pd_{step}_{corr}.csv"
        x0p = pre / f"solution_pd_{step}_{corr}.csv"
        pp_matches = list(after.glob(f"pd_PISO_{step}_{corr}_t*.csv"))
        pp_matches = [p for p in pp_matches if not p.name.endswith("Zone.Identifier")]
        if not pp_matches: continue
        pp = pp_matches[0]
        check_one(f"step {step} / corr {corr}", mp, sp, x0p, pp)


if __name__ == "__main__":
    main()
