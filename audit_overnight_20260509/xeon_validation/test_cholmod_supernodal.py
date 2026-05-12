"""Quick test: does forcing supernodal_mode='supernodal' make CHOLMOD multi-thread?

5-min test. Reads one matrix from dense_track_dump_500K, factors it with two
modes, and reports wall time.

Run on Xeon:
  ~/jax-env/bin/python3 audit_overnight_20260509/xeon_validation/test_cholmod_supernodal.py
"""
import scipy.io as sio
from sksparse.cholmod import cho_factor
import time
import os

# Confirm thread config
print(f"OPENBLAS_NUM_THREADS = {os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}")
print(f"OMP_NUM_THREADS      = {os.environ.get('OMP_NUM_THREADS', 'unset')}")
print(f"MKL_NUM_THREADS      = {os.environ.get('MKL_NUM_THREADS', 'unset')}")
print()

path = '/home/yzk/cases/dense_track_dump_500K/postProcessing/matrices/9.44e-07/pd_corr0/A.mm'
print(f"Loading {path} ...")
t0 = time.time()
A = sio.mmread(path).tocsr()
print(f"  load: {time.time()-t0:.1f}s  N={A.shape[0]}  nnz={A.nnz}")

if A.diagonal().mean() < 0:
    A = -A
    print(f"  sign-flipped to SPD form")

A_csc = A.tocsc()

print("\n--- Mode: supernodal (forced) ---")
t0 = time.time()
f1 = cho_factor(A_csc, supernodal_mode="supernodal")
print(f"  factor: {time.time()-t0:.1f}s")

print("\n--- Mode: simplicial (forced) ---")
t0 = time.time()
f2 = cho_factor(A_csc, supernodal_mode="simplicial")
print(f"  factor: {time.time()-t0:.1f}s")

print("\n--- Mode: auto (default) ---")
t0 = time.time()
f3 = cho_factor(A_csc)
print(f"  factor: {time.time()-t0:.1f}s")

print("\nVerdict:")
print("  if supernodal << simplicial → CHOLMOD does use BLAS multi-thread,")
print("    but default 'auto' is picking simplicial → force supernodal in E11.")
print("  if all 3 ~similar → library is not multi-thread, fallback to overnight batch.")
