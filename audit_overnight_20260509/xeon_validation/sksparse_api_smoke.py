"""Minimal sksparse API smoke — run this BEFORE the full --smoke if you suspect
sksparse cholesky() API breakage.

scikit-sparse 0.5.0 changed cholesky():
  pre-0.5.0:  cholesky(A) -> Factor;  factor(b) -> x;  factor.cholesky_inplace(A2)
  0.5.0:      cholesky(A) -> (R, p) tuple (not callable!)
              Use: cho_factor(A) -> CholeskyFactor;  factor.solve(b);  factor.factorize(A2)

This script:
  1. probes which API is present
  2. runs a 5x5 trivial CHOLMOD solve via cho_factor path (preferred)
  3. exits 0 on success, prints DIAG on failure

Usage:  ~/jax-env/bin/python3 sksparse_api_smoke.py
"""
from __future__ import annotations
import sys
import traceback

import numpy as np
import scipy.sparse as sp


def main():
    print("=== sksparse API smoke ===")
    try:
        import sksparse
        print(f"sksparse version: {sksparse.__version__}")
    except Exception as e:
        print(f"FAIL: cannot import sksparse: {e}")
        sys.exit(2)

    # Build SPD 5x5: A = 2I (trivial SPD), b = ones; expect x = b/2 = 0.5 * ones
    A = (sp.eye(5) * 2.0).tocsc()
    b = np.ones(5)
    expected = np.full(5, 0.5)

    have_cho_factor = False
    have_cholesky = False
    try:
        from sksparse.cholmod import cho_factor
        have_cho_factor = True
    except ImportError:
        pass
    try:
        from sksparse.cholmod import cholesky
        have_cholesky = True
    except ImportError:
        pass
    print(f"cho_factor present: {have_cho_factor}")
    print(f"cholesky   present: {have_cholesky}")

    fail = False

    # --- Path A: cho_factor (sksparse 0.5.0+, preferred) ---
    if have_cho_factor:
        try:
            f = cho_factor(A)
            print(f"cho_factor(A) -> type={type(f).__name__}")
            x = f.solve(b)
            err = float(np.abs(x - expected).max())
            print(f"  factor.solve(b) max|x - 0.5| = {err:.2e}  {'OK' if err < 1e-12 else 'FAIL'}")
            if err >= 1e-12: fail = True
            try:
                f.factorize(A)  # symbolic-reuse refact API for E06
                x2 = f.solve(b)
                err2 = float(np.abs(x2 - expected).max())
                print(f"  factor.factorize(A) + solve = {err2:.2e}  {'OK' if err2 < 1e-12 else 'FAIL'}")
                if err2 >= 1e-12: fail = True
            except Exception as e:
                print(f"  factor.factorize FAIL: {e}")
                fail = True
        except Exception:
            print("cho_factor path raised:")
            traceback.print_exc()
            fail = True

    # --- Path B: cholesky() — what does it return in this version? ---
    if have_cholesky:
        try:
            r = cholesky(A)
            print(f"cholesky(A) -> type={type(r).__name__}")
            if isinstance(r, tuple):
                print(f"  cholesky(A) IS a tuple of length {len(r)} -> CONFIRMS sksparse 0.5.0 API change")
            elif callable(r):
                print(f"  cholesky(A) is callable -> pre-0.5.0 API")
                x_legacy = r(b)
                err_legacy = float(np.abs(x_legacy - expected).max())
                print(f"  legacy factor(b) max|err| = {err_legacy:.2e}  {'OK' if err_legacy < 1e-12 else 'FAIL'}")
                if err_legacy >= 1e-12: fail = True
            else:
                print(f"  cholesky(A) is neither tuple nor callable: {r!r}")
                fail = True
        except Exception:
            print("cholesky path raised:")
            traceback.print_exc()
            fail = True

    print()
    print("=== verdict ===")
    if fail:
        print("FAIL — at least one API path did not produce expected result")
        sys.exit(1)
    else:
        print("PASS — sksparse API works; runners should run after smoke fix")
        sys.exit(0)


if __name__ == "__main__":
    main()
