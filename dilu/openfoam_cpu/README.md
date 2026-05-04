# `dilu/openfoam_cpu/` — OpenFOAM single-process CPU DILU + PBiCG replication

## 1. Module purpose

Reverse-engineer OpenFOAM v2506's single-process CPU `DILUPreconditioner` +
`PBiCG` outer solver as a standalone reference implementation. Goal: **on a
dumped (A, b) from LaserbeamFoam's T equation, produce the same iter count and
the same x as OpenFOAM, to within fp rounding**.

This is a **replication / verification target**, not a performance target. The
existing GPU paths (`dilu/cusparse/`, `dilu/multicolor/`, `dilu/amgx/`) are the
performance work. This module exists so we can:

1. Have a **byte-faithful CPU reference** to compare every GPU optimization
   against, independent of OpenFOAM itself (so the regression diff doesn't
   include LaserbeamFoam build noise / stdout parsing fragility).
2. Document the OpenFOAM algorithm in code that we can step through with
   `pdb`, instead of through stripped C++ optimized binaries.
3. Provide a baseline to measure "what does single-thread NumPy DILU look
   like on this CPU" vs "what does single-thread OpenFOAM (`mpirun -np 1`)
   look like" — the gap is purely implementation overhead, not algorithm.

## 2. Relationship to sibling modules

| Sibling | Path | What it does | Relation |
|---------|------|--------------|----------|
| cuSPARSE DILU-PCG | `dilu/cusparse/` | GPU DILU + PCG (pd) / BiCGStab (T) using cuSPARSE level scheduling | Reference for "what CPU algorithm we're trying to replicate, then port to GPU level-scheduling" |
| Multicolor DILU | `dilu/multicolor/` | GPU DILU + multicolor (red-black) reordering for OpenMP-style parallelism | Same algorithm but reordered cells. **Will diverge from this module's results in iter count** because reordering changes the preconditioner. That's expected. |
| AMGx | `dilu/amgx/` | NVIDIA AMG library wrapper, not strictly DILU | Different algorithm class (multigrid). Just a competitor in the benchmark, no algorithmic overlap with this module. |
| **`openfoam_cpu` (this)** | `dilu/openfoam_cpu/` | Plain Python/NumPy single-thread re-impl of OpenFOAM DILU + PBiCG | The "ground truth" CPU baseline that all GPU paths get diff'd against |

The audit document this module is built on:
**`/home/yzk/DILU-Research/docs/reference/openfoam_dilu_source_audit.md`** —
read it before writing any code here.

The skill that loads this knowledge into Claude:
**`~/.claude/skills/openfoam-dilu-solver-replication/SKILL.md`**.

## 3. Planned directory structure (NOT implemented yet)

```
dilu/openfoam_cpu/
├── README.md                      # this file
├── python/                        # planned: pure-python ref impl
│   ├── ldu_addressing.py          # owner/neighbour/losort, lazy compute
│   ├── dilu.py                    # calcReciprocalD + precondition + preconditionT
│   ├── spmv.py                    # Amul + Tmul + sumA + residual
│   ├── pbicg.py                   # outer solver + normFactor + convergence
│   └── io_npz.py                  # load (diag, lower, upper, owner, neighbour, b) from dump
├── tests/                         # planned: regression vs OpenFOAM-dumped (A, b, x_OF, iter_OF)
│   ├── test_dilu_factor.py        # rD vs OpenFOAM rD on small cases
│   ├── test_spmv_amul.py          # Apsi == OpenFOAM Apsi
│   ├── test_spmv_tmul.py          # Tpsi == OpenFOAM Tpsi (uses losort)
│   ├── test_normfactor.py         # normFactor == OpenFOAM normFactor
│   └── test_pbicg_full.py         # iter count + |x_ours - x_OF| / |x_OF| < 1e-10
└── bench/                         # optional: time vs single-rank OpenFOAM
    └── compare_single_rank.py     # ours / NumPy vs `mpirun -np 1 laserMeltFoam`
```

C/C++ port (`cpp/` sibling) is **out of scope for the first milestone**. Pure
NumPy is enough to demonstrate algorithm fidelity. C++ port comes later only
if NumPy is too slow to be useful for regression sweeps.

## 4. First-milestone definition (Minimum Viable)

**Goal**: pick **one** dumped T matrix and reproduce OpenFOAM's iter count and
solution x to within `‖x_ours - x_OF‖∞ / ‖x_OF‖∞ < 1e-10`.

**Test data**:
```
/home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/
  <time>/T_corr0/data.npz
```
Each `data.npz` should contain (per dump-design in
`docs/design/openfoam_crosscheck_plan.md`):

- `diag` (nCells,)
- `lower` (nFaces,)
- `upper` (nFaces,)
- `owner` (nFaces,) — equivalently OpenFOAM's `lowerAddr`
- `neighbour` (nFaces,) — equivalently OpenFOAM's `upperAddr`
- `b` (nCells,) — RHS
- `x_OF` (nCells,) — OpenFOAM's solution
- `x0` (nCells,) — initial guess (usually previous step's solution)
- metadata (`iter`, `init_residual`, `final_residual`, `tolerance`, `relTol`,
  `minIter`, `eq_name`, `time`, `corrector`)

Confirm exact field names by `python -c "import numpy as np; d = np.load('.../data.npz'); print(list(d.keys()))"` before coding the loader.

**Success criteria for milestone 1**:

1. `iter_ours == iter_OF` (exact match)
2. `init_residual_ours == init_residual_OF` to within 1e-12 relative
3. `final_residual_ours == final_residual_OF` to within 1e-10 relative
4. `‖x_ours - x_OF‖∞ / ‖x_OF‖∞ < 1e-10`

If iter differs by ±1 after honest implementation, the most likely cause is
`normFactor` mismatch — see audit §4.4 and trap-checklist #5 in the skill.

## 5. After milestone 1 — sweep & validate

Once one matrix passes:

1. Sweep over all 100 T matrices in
   `collected/spot_melt_npz/*/T_corr*/data.npz`. Log per-matrix iter delta.
   Acceptable: 0 deltas. ±1 on a small fraction (say < 5%) is fp-tolerable.
2. Sweep pd matrices using **PCG + DIC** (separate but trivial — DIC is just
   the symmetric simplification of DILU; see audit §2.5). This validates that
   the LDU/Amul/normFactor infrastructure also works for SPD path.
3. Single-thread NumPy timing comparison vs `mpirun -np 1 laserMeltFoam` on
   the same case. Expect NumPy to be 5-50× slower per iter. Document the gap;
   it's the cost of "interpreted, no SIMD vectorization".

## 6. Hard constraints (do NOT violate)

These are the unbreakable contracts from the audit doc §8:

- LDU format only — no CSR conversion of the input.
- Face order from the dump must be preserved exactly (face index sorted by `(owner, neighbour)`).
- `calcReciprocalD` is single-thread sequential. Do not OpenMP it.
- Forward sweep: face ascending. Backward sweep: face descending.
- `preconditionT` must use `losort` (computed per `lduAddressing.C:34-91`).
- Tolerance is `Σ |r_i| / normFactor` where normFactor follows
  `lduMatrixSolver.C:235-274` (sumA-based, L1 scaled).
- Store `rD = 1/D̃`, not `D̃`. All sweeps multiply by `rD`.
- Honor `minIter` (LaserbeamFoam sets `minIter 1` for T).

## 7. What this module is NOT

- Not a performance optimization target. If you want speed, use AMGx
  (`dilu/amgx/`) — it already beats OpenFOAM 8.6× on pd and 2.8× on T per
  v3.2 benchmark.
- Not an MPI / multi-rank reproduction. Single process, single thread only.
  MPI halo exchange is out of scope (it has zero impact on the per-rank DILU
  algorithm anyway, since DILU itself is rank-local).
- Not a port of `lduMatrix` infrastructure. We only need the minimal bits to
  run DILU + PBiCG on a dumped (A, b). No fvMesh, no `fvSchemes`, no
  finiteVolume layer.

## 8. Quick verify (after first impl exists)

```bash
source /home/yzk/jax-env/bin/activate

# planned (not yet implemented):
python -m dilu.openfoam_cpu.tests.test_pbicg_full \
    --npz /home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/2.49322e-06/T_corr0/data.npz
```
