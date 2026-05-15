"""Quick test: can CHOLMOD factor a T_corr0 matrix (despite tiny asymmetry)?

Tries 4.1e-07 (mid-melt) which T01 shows asymmetry rel ~7e-7. Reports:
  - factor wall (compare to ~5-7 min CHOLMOD on pd)
  - rel_resid of x_LU = factor.solve(b)
  - any CholmodWarning

If WORKS → switch T02 to CHOLMOD (5-10x faster than splu).
If FAILS → stick with slow splu.
"""
import scipy.io as sio
from sksparse.cholmod import cho_factor
import numpy as np
import time
import warnings

p = "/home/yzk/cases/single_track_dump/postProcessing/matrices/4.1e-07/T_corr0/"
print(f"Loading A and b from {p}")
A = sio.mmread(p + "A.mm").tocsr()
b = np.asarray(sio.mmread(p + "b.mm")).flatten()
x_OF = np.asarray(sio.mmread(p + "x_final.mm")).flatten()
print(f"  N={A.shape[0]}  nnz={A.nnz}  diag_mean={A.diagonal().mean():.3e}")

# Symmetric part for sanity
import scipy.sparse as sp
asym = sp.linalg.norm(A - A.T, "fro") / sp.linalg.norm(A, "fro")
print(f"  asymmetry rel: {asym:.3e}")

print("\n--- attempting cho_factor on T ---")
t0 = time.time()
try:
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        factor = cho_factor(A.tocsc())
    t_factor = time.time() - t0
    print(f"  factor: {t_factor:.1f}s  -- CHOLMOD WORKS ✓")
    for w in wlist:
        print(f"  WARNING: {w.category.__name__}: {w.message}")

    t0 = time.time()
    x = factor.solve(b)
    t_solve = time.time() - t0
    print(f"  solve: {t_solve:.2f}s")

    resid = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    print(f"  rel_resid (Ax-b): {resid:.3e}  "
          f"({'GOOD < 1e-10' if resid < 1e-10 else 'CHECK — likely (A+A^T)/2 solution'})")

    diff_vs_OF = np.abs(x - x_OF)
    print(f"  vs OF x: max abs diff = {diff_vs_OF.max():.3e} K  "
          f"(OF rel_resid ~1e-13 so OF is already truth-like)")
    print(f"  vs OF x: median diff = {np.median(diff_vs_OF):.3e} K")
except Exception as e:
    print(f"  CHOLMOD FAILS: {type(e).__name__}: {e}")
    print("  → Fall back to splu (slow but correct)")
