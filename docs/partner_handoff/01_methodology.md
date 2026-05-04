# Guide: Outputting the Linear Systems Solved by `laserMeltFoam`

**Purpose**: Extract the linear systems (matrix A, source vector b, initial guess x₀, final solution x_final) solved during each pressure and temperature `solve()` call in LaserbeamFoam's `laserMeltFoam` solver, suitable for downstream cross-solver benchmarking.

**Target**: LaserbeamFoam `laserMeltFoam`, OpenFOAM v2506, serial run.

This document is self-contained. Following its instructions on a fresh machine with OpenFOAM v2506 installed will reproduce the toolchain bit-for-bit.

---

## 1. Background

### 1.1 What gets solved per time step in `laserMeltFoam`

The PIMPLE loop in `laserMeltFoam.C` (lines 132–186) calls each equation's solver as follows:

| Equation | Solver | Preconditioner | Symmetry | Calls per Δt |
|----------|--------|----------------|----------|--------------|
| `pd` (dynamic pressure) | PCG | DIC | symmetric | `nCorrectors` (= 3 in default tutorial; last invocation uses `pdFinal`) |
| `T` (temperature) | PBiCG | DILU | non-symmetric | up to `nTCorrectors` (= 250 in default tutorial; typically converges in 5–30) |
| `U` (momentum) | PBiCGStab | DILU | non-symmetric | 0 if `momentumPredictor: no`, else 1 |
| `alpha.metal` | isoAdvector | — | — | 0 (geometric advection, no linear solve) |

So a typical time step in this case produces **3 pd matrices + 5–30 T matrices** that are candidates for extraction.

### 1.2 OpenFOAM's matrix storage

OpenFOAM splits the linear operator A into two pieces:

1. **`lduMatrix` part** (carried by `fvMatrix`):
   - `diag()` — N entries (one per cell)
   - `upper()` — nFaces entries (one per internal face)
   - `lower()` — nFaces entries (same as `upper` for symmetric matrices)
   - addressing: `lduAddr().lowerAddr()` (owner cell of each face), `lduAddr().upperAddr()` (neighbour cell)

2. **Boundary part** (per-patch arrays):
   - `internalCoeffs_[patchi]` — virtual contributions to diagonals of cells touching patch `i` (e.g. from `fixedValue` BCs, `setReference()`)
   - `boundaryCoeffs_[patchi]` — virtual contributions to source

When the solver does its sparse matrix-vector multiply (`Amul`), it adds the boundary contributions on the fly. The "true" matrix the solver iterates on is therefore:

```
A_effective.diag       = diag()  +  Σ_patches scatter(internalCoeffs[patchi], lduAddr.patchAddr(patchi))
A_effective.off-diag   = upper(),  lower()                                  (unchanged)
b_effective            = source() +  Σ_patches scatter(boundaryCoeffs[patchi], lduAddr.patchAddr(patchi))
```

To extract a matrix faithfully, **both halves must be folded together** before serialisation.

### 1.3 Sign convention

OpenFOAM's `fvm::laplacian` discretises Δφ such that the resulting matrix has **negative diagonals and positive off-diagonals** (symmetric, but negative-semidefinite by the standard SPD convention). Iterative solvers that assume positive-diagonal SPD must operate on the equivalent system `(−A) x = (−b)` instead. Sign normalisation is performed downstream at read time, not at dump time. The metadata records whether negation was applied.

---

## 2. Architecture

### 2.1 Hook point

The dumper wraps `fvMatrix.solve()` calls in `TEqn.H` and `pEqn.H`, capturing the full system before each solve and the result after:

```
fvMatrix → [dumpPreSolve(matrix, name, x0)] → matrix.solve() → [dumpPostSolve(name, x_final, perf, matrix)]
```

The wrapper is implemented in a single header file `matrixDumper.H` placed alongside the solver source. It is included once in `laserMeltFoam.C`, instantiated once after `setInitialDeltaT.H`, and called twice per equation per corrector inside the equation files.

### 2.2 Public OpenFOAM API used

The dumper uses **only** public methods of `fvMatrix` and `lduAddressing`. It does not inherit from or friend any internal class:

| Method | Returns | Purpose |
|--------|---------|---------|
| `fvMatrix::D()` | `tmp<scalarField>` | Diagonal **including** boundary `internalCoeffs` contribution |
| `fvMatrix::source()` | `const Field<Type>&` | Source **without** boundary contribution |
| `fvMatrix::upper()` / `lower()` | `const scalarField&` | Off-diagonal entries |
| `fvMatrix::psi()` | `const GeometricField&` | The field being solved |
| `fvMatrix::boundaryCoeffs()` | `const FieldField&` | Per-patch boundary coefficients (needed to fold source) |
| `fvMatrix::lduAddr()` | `const lduAddressing&` | Face-to-cell addressing |
| `fvMatrix::hasUpper()`, `hasLower()`, `symmetric()` | `bool` | Structural flags written to metadata |

### 2.3 Step-index handling

`runTime.timeIndex()` is the absolute step counter from `startTime = 0`. To allow dumping after a `startFrom latestTime` resume (where `timeIndex` may be at, say, 200), the dumper records the first `timeIndex` it observes (`firstTimeIdx_`) and computes a **relative index** `relIdx = timeIndex − firstTimeIdx + 1`. Configuration entries like `dumpTimeSteps (1 2 3 ... 50)` therefore mean "the first 50 time steps after the dumper became active".

---

## 3. Files Added or Modified

### 3.1 New file: `matrixDumper.H`

Path: `~/LaserbeamFoam/applications/solvers/laserMeltFoam/matrixDumper.H`
Size: 488 lines, header-only.

Defines `class Foam::matrixDumper` with public methods:

```cpp
matrixDumper(const Time& runTime);

bool enabled() const;
bool shouldDumpEq(const word& eqName) const;

void dumpPreSolve(const fvScalarMatrix& matrix,
                  const word& eqName,
                  const volScalarField& x0);

void dumpPostSolve(const word& eqName,
                   const volScalarField& xFinal,
                   const SolverPerformance<scalar>& perf,
                   const fvScalarMatrix& matrix);
```

The constructor reads `<case>/system/matrixDumperDict` if present. If the dict is absent, `enabled() == false` and all dump calls are no-ops.

### 3.2 Patch: `laserMeltFoam.C` (4 lines added)

Original location: `~/LaserbeamFoam/applications/solvers/laserMeltFoam/laserMeltFoam.C`

```diff
@ line 84  (inside the "// Added" includes block, after laserHeatSource.H)
+ // DILU-Research crosscheck matrix dumper (see matrixDumper.H)
+ #include "matrixDumper.H"

@ line 119  (after #include "setInitialDeltaT.H", before "// Starting time loop")
+ // DILU-Research matrixDumper: reads system/matrixDumperDict if present.
+ Foam::matrixDumper matrixDumper_(runTime);
```

### 3.3 Patch: `TEqn.H` (4 lines added)

```diff
@ line 49  (right after "TEqn.relax();")
+ // --- DILU-Research matrixDumper: pre-solve snapshot (A, b, x0) ---
+ matrixDumper_.dumpPreSolve(TEqn, "T", T);

  Tp = TEqn.solve();

+ // --- DILU-Research matrixDumper: post-solve (x_final, perf) ---
+ matrixDumper_.dumpPostSolve("T", T, Tp, TEqn);
```

### 3.4 Patch: `pEqn.H` (5 lines added/modified)

The `pdEqn.solve(...)` return value, originally discarded, must now be captured:

```diff
@ line 124–135  (around pdEqn.solve)
  pdEqn.relax();

+ // --- DILU-Research matrixDumper: pre-solve snapshot ---
+ matrixDumper_.dumpPreSolve(pdEqn, "pd", pd);

- pdEqn.solve(
+ Foam::SolverPerformance<Foam::scalar> pdPerf = pdEqn.solve(
      mesh.solutionDict(pd.select(pimple.finalInnerIter()))
  );

+ // --- DILU-Research matrixDumper: post-solve ---
+ matrixDumper_.dumpPostSolve("pd", pd, pdPerf, pdEqn);
```

### 3.5 New per-case file: `system/matrixDumperDict`

Generated automatically by `patch_case.sh` (see §5.2):

```cpp
FoamFile { version 2.0; format ascii; class dictionary;
           location "system"; object matrixDumperDict; }

enabled        true;
binaryMM       false;                              // ASCII only in v1.0
outputDir      "postProcessing/matrices";

// Time steps (1-based, RELATIVE to dumper construction).
// E.g. "(1 2 3)" means: dump the first 3 steps that occur after dumper init.
dumpTimeSteps  (1 2 3 ... N);

// Equation whitelist. Empty list = all equations the dumper sees.
equations      (pd T);

// Per-equation cap on corrector index. Solves beyond this index
// within the same time step are skipped, even if the equation is whitelisted.
maxCorrectorsPerEq
{
    pd  3;
    T   2;
}
```

---

## 4. Output Format

### 4.1 Directory layout

For each (time step, equation, corrector) tuple that the dict accepts:

```
<case>/postProcessing/matrices/
  <time>/                                e.g. 2.25322e-06/
    pd_corr0/                            corrector index, 0-based
      A.mm                               MatrixMarket: coordinate, real, general
      b.mm                               MatrixMarket: array, real, general (Nx1)
      x0.mm                              psi at solve entry
      x_final.mm                         psi after solve
      metadata.json                      see §4.4
    pd_corr1/  ...
    pd_corr2/  ...
    T_corr0/  ...
    T_corr1/  ...
```

### 4.2 Matrix construction (COO, 1-based indices)

```
diag entry           : (c+1, c+1, diag_full[c])               for c in 0..N-1
upper entry of face f: (lowerAddr[f]+1, upperAddr[f]+1, upper[f])
lower entry of face f: (upperAddr[f]+1, lowerAddr[f]+1, lower[f])
```

where:

```cpp
diag_full[c] = matrix.D()()[c]                               // includes boundary
upper        = matrix.upper()
lower        = matrix.lower()                                 // == upper for symmetric
```

For symmetric pd matrices (`matrix.symmetric() == true`), `lower` aliases `upper`. Both lower and upper triangles are still emitted as separate COO entries (`general` MatrixMarket format, not `symmetric`), so the file is always self-contained without exploiting symmetry.

### 4.3 Source vector construction

```cpp
b_full = matrix.source();                              // raw source field

for each non-coupled patch i (i.e. ptf.coupled() == false):
    pbc  = matrix.boundaryCoeffs()[i];                  // per-face boundary coeffs
    addr = matrix.lduAddr().patchAddr(i);               // face -> cell mapping
    for each face f in addr:
        b_full[addr[f]] += pbc[f];
```

This replicates the boundary-fold step that `fvMatrix::solveSegregated` performs internally, using only public API.

### 4.4 metadata.json schema

```json
{
  "schema_version": "1.0",
  "case": "<case directory name>",
  "openfoam_version": "v2506",
  "time": {
    "step_index": <int>,                 // OpenFOAM absolute timeIndex
    "time_value": <double>,              // simulation time in seconds
    "delta_t":    <double>
  },
  "equation": {
    "name": "pd" | "T",
    "corrector_index": <int>             // 0-based, per equation per step
  },
  "matrix": {
    "n_rows": <int>,
    "n_cols": <int>,
    "nnz":    <int>,
    "has_upper": bool,
    "has_lower": bool,
    "symmetric": bool
  },
  "solver_openfoam": {
    "initial_residual": <double>,
    "final_residual":   <double>,
    "iterations":       <int>,
    "converged":        bool
  }
}
```

---

## 5. Step-by-Step Procedure

All commands assume bash, OpenFOAM v2506 installed at `/usr/lib/openfoam/openfoam2506`, and a LaserbeamFoam source tree at `~/LaserbeamFoam`.

### 5.1 Install the dumper into the solver source

The deploy bundle (`dilu/benchmark/openfoam_crosscheck/deploy/`) contains:

- `matrixDumper.H` — the dumper header
- `laserMeltFoam.C.diff`, `TEqn.H.diff`, `pEqn.H.diff` — unified diffs for the three patches
- `install.sh` — automation: backup → patch → wmake

Install:

```bash
cd ~/deploy
source /usr/lib/openfoam/openfoam2506/etc/bashrc
./install.sh ~/LaserbeamFoam/applications/solvers/laserMeltFoam
```

The script:

1. Backs up `laserMeltFoam.C`, `TEqn.H`, `pEqn.H` to `*.orig`.
2. Applies the three unified diffs.
3. Copies `matrixDumper.H` into the solver source dir.
4. Runs `wmake`.

A successful run ends with:

```
[install] SUCCESS
  New binary: /home/<user>/OpenFOAM/<user>-v2506/platforms/linux64GccDPInt32Opt/bin/laserMeltFoam
```

### 5.2 Apply to a case

```bash
~/deploy/patch_case.sh <case_dir> <N_steps>
# Example:
~/deploy/patch_case.sh ~/cases/openfoam_laserMeltFoam_spot 50
```

The script:

1. Finds `latestTime` (the largest numeric subdirectory containing `pd`).
2. Reads `maxDeltaT` from the case's `controlDict`.
3. Computes a new `endTime = latestTime + N × maxDeltaT`.
4. Backs up `controlDict` → `controlDict.orig`.
5. Sets `startFrom latestTime;` and the new `endTime;`.
6. Writes `system/matrixDumperDict` with `dumpTimeSteps (1 2 … N)`.

### 5.3 Pre-conditions on the case

Before running the dump, the case must satisfy:

1. **Latest time folder is complete.** It must contain `pd`, `T`, `U`, `alpha.metal` field files. A "Co-only" folder (containing only a Courant number written by a `functionObject`) is not a valid restart point; OpenFOAM will fail with `cannot find file ".../<time>/pd"`.

2. **`writeControl` writes fields frequently.** Use:
   ```
   writeControl    timeStep;
   writeInterval   10;        // or whatever cadence is acceptable
   ```
   not `adjustableRunTime` with a coarse `writeInterval` (e.g. `5e-6`), which only writes fields at `5e-6 s` boundaries and leaves no valid restart point if the run is interrupted earlier.

3. **No `functionObject` writing per-step partial data.** A function object that emits, e.g., a Courant number every step creates time directories that lack full field data, causing problem (1) above. Comment out or relocate such function objects before dumping. Example:

   ```diff
   @ system/controlDict
   - functions { #include "monitoring" }
   + // functions { #include "monitoring" }   // disabled for matrix dump
   ```

### 5.4 Run

```bash
cd <case_dir>
laserMeltFoam 2>&1 | tee log.dump
```

Per-solve log entries appear interleaved with normal solver output:

```
[matrixDumper] ENABLED. outputDir="postProcessing/matrices"  dumpTimeSteps=50(1 2 3...)  equations=2(pd T)
...
DICPCG:  Solving for pd, Initial residual = 1.05e-08, Final residual = 9.91e-09, No Iterations 22
[matrixDumper] pre-solve pd_corr0  N=1024000  nnz=7104000
[matrixDumper] post-solve pd_corr0  finalRes=9.91e-09  iters=22
```

### 5.5 Verify

```bash
python3 ~/deploy/sanity.py <case_dir>/postProcessing/matrices
```

For each matrix directory, the script:

1. Reads `A.mm`, `b.mm`, `x_final.mm`, `metadata.json`.
2. Computes `‖A · x_final − b‖₂` and `‖A · x_final − b‖₂ / ‖b‖₂`.
3. Compares the absolute residual against `100 × final_residual_OF` from metadata.

PASS criterion: `‖A x − b‖₂ < 100 × final_residual_OF` (or `< 1e-6` if OF residual is missing).

Expected end-of-run line:

```
Total: <K> matrices, <K> pass, 0 fail
```

Any FAIL means the dump is mathematically inconsistent with OpenFOAM's solve and the data should not be used downstream.

### 5.6 Convert to compact binary (optional)

ASCII MatrixMarket files for million-cell matrices are large (≈330 MB per A.mm at 1M cells). For data transfer, convert to NumPy NPZ + zlib + drop x0:

```bash
python3 ~/deploy/shrink_dump.py \
    <case_dir>/postProcessing/matrices \
    <output_dir> \
    <n_workers>
# Example:
python3 ~/deploy/shrink_dump.py \
    ~/cases/openfoam_laserMeltFoam_spot/postProcessing/matrices \
    ~/spot_melt_npz/ 16
```

This script:

- Walks every `<time>/<eq>_corr<k>` subdirectory.
- Filters to a configurable subset of correctors (default: `pd_corr0`, `T_corr0`, `T_corr1`).
- Loads `A.mm`, `b.mm`, `x_final.mm`.
- Writes `data.npz` containing `A_data`, `A_indices`, `A_indptr` (CSR), `n_rows`, `b`, `x_final`.
- Copies `metadata.json` unchanged.
- Uses `multiprocessing.Pool` with `n_workers` parallel processes.

Typical compression: 9× smaller (330 MB ASCII → 37 MB NPZ per matrix).

---

## 6. Application: 316L Spot-Melt Validation Case

### 6.1 Case parameters

| Parameter | Value |
|-----------|-------|
| Geometry | 200 × 400 × 200 μm rectangular box |
| Mesh | 80 × 160 × 80 = 1,024,000 cells (uniform, dx = 2.5 μm) |
| Material | 316L stainless steel (ρ = 7900, c_p = 700, k = 20, μ = 5×10⁻³, T_s = 1650, T_l = 1700) |
| Laser | Stationary spot, P = 150 W, r₀ = 25 μm, on 0–50 μs |
| Physics | Two-phase VOF (metal + Ar) + surface tension + Marangoni (dσ/dT = +1×10⁻⁴ N/m/K) + Darcy mushy zone |
| pd solver | PCG + DIC, tol = 1×10⁻⁸ |
| T solver | PBiCG + DILU, tol = 1×10⁻¹² |
| `nCorrectors` (pd) | 3 |
| `nTCorrectors` (T) | up to 250 (typical 5–30 actual) |
| `nOuterCorrectors` (PIMPLE) | 1 |

### 6.2 Procedure used

1. **Warmup** (no dumper): integrated from t = 0 to latestTime ≈ 2.23 μs (≈ 200 steps; dt adaptively ramped from 1×10⁻¹² s up to `maxDeltaT = 2×10⁻⁸ s`).
2. **Dump** (with dumper): `patch_case.sh . 50`, then ran 50 additional steps covering t ∈ [2.25, 3.23] μs. Wallclock 6602 s on a 56-core Xeon (single-process).

### 6.3 Output

| Quantity | Value |
|----------|-------|
| Matrix directories produced | 250 (50 steps × {pd_corr0, pd_corr1, pd_corr2, T_corr0, T_corr1}) |
| Sanity-check pass rate | 250 / 250 |
| `‖A · x_final − b‖₂` (pd matrices) | 7–8 × 10⁻¹⁷ |
| `‖A · x_final − b‖₂` (T matrices) | 1 × 10⁻¹¹ – 9 × 10⁻¹¹ |
| `‖A · x_final − b‖₂ / ‖b‖₂` (pd) | ≈ 1 × 10⁻¹⁰ |
| `‖A · x_final − b‖₂ / ‖b‖₂` (T) | ≈ 5 × 10⁻¹⁴ |
| Disk (ASCII MM) | 81 GB |
| Disk after `shrink_dump.py` (NPZ; pd_corr0 + T_corr0 + T_corr1 only) | 5.5 GB (150 matrices × ≈37 MB) |

The residual figures confirm that the extracted (A, b, x_final) tuple is consistent with OpenFOAM's solve to machine precision for pd, and to the PBiCG-DILU final residual for T.

---

## 7. Verification Checklist

Before using extracted data downstream, verify:

```
[V1] Number of matrix directories matches expected count:
     find postProcessing/matrices -mindepth 2 -maxdepth 2 -type d | wc -l
     expected: N_dump_steps × Σ_eq min(maxCorrectorsPerEq[eq], actual_corr_count[eq])

[V2] Each matrix directory contains exactly 5 files (A.mm, b.mm, x0.mm, x_final.mm, metadata.json):
     for d in postProcessing/matrices/*/*_corr*; do
         test "$(ls "$d" | wc -l)" -eq 5 || echo "INCOMPLETE: $d"
     done

[V3] All matrices pass sanity:
     python3 ~/deploy/sanity.py postProcessing/matrices
     tail line must read: "Total: <N> matrices, <N> pass, 0 fail"

[V4] metadata.json reports the correct mesh size:
     python3 -c "import json; m=json.load(open(p)); assert m['matrix']['n_rows']==<expected>"

[V5] Each metadata.json reports converged == true:
     verifies dumps were taken at converged solves, not at failures.
```

---

## 8. Limitations

1. **Serial only.** The boundary-fold logic in `dumpPreSolve` skips coupled patches:
   ```cpp
   if (ptf.coupled()) continue;
   ```
   For parallel runs (`mpirun -np N`), processor patches are coupled and require additional cross-rank handling. Not implemented in v1.0.

2. **Scalar `fvMatrix` only.** `fvMatrix<vector>` and `fvMatrix<tensor>` (e.g. for U if `momentumPredictor: yes` is enabled) are not supported.

3. **OpenFOAM version.** Tested on v2506. The `getOrDefault<T>(...)` API used in dictionary reading was introduced in v2212. For earlier versions, replace with `lookupOrDefault<T>(...)` at the three call sites in `matrixDumper.H`'s constructor.

4. **ASCII MatrixMarket file size.** A 1-million-cell single matrix is ≈330 MB ASCII. The `shrink_dump.py` step (§5.6) is essentially mandatory for any case > 100K cells if results need to be moved off the simulation machine.

5. **`writeInterval` interaction.** The case's `controlDict` must use a write cadence that produces full field writes frequently enough to provide valid restart points. `writeControl adjustableRunTime; writeInterval 5e-6;` writes only at t = 5×10⁻⁶ s boundaries, leaving long stretches of the run without restart points (§5.3 condition 2).

6. **`functionObject`-induced empty time folders.** Per-step partial writes from function objects (e.g. Courant number recorders) create time directories that lack full field data, breaking `startFrom latestTime`. Such function objects must be disabled before dumping (§5.3 condition 3).

7. **Sign convention.** OpenFOAM's Laplacian discretisation produces matrices with negative diagonals. Downstream solvers that assume positive-diagonal SPD must operate on `(−A) x = (−b)`. Sign normalisation is handled by the downstream `reader.py` based on `np.all(diag ≤ 0)`; the `metadata.json` does not record this transform — it is applied at read time, not dump time.

---

## 9. File Inventory

| Artifact | Path |
|---|---|
| `matrixDumper.H` (dumper source) | `LaserbeamFoam/applications/solvers/laserMeltFoam/matrixDumper.H` |
| Patched `laserMeltFoam.C` | `LaserbeamFoam/applications/solvers/laserMeltFoam/laserMeltFoam.C` |
| Patched `TEqn.H` | `LaserbeamFoam/applications/solvers/laserMeltFoam/TEqn.H` |
| Patched `pEqn.H` | `LaserbeamFoam/applications/solvers/laserMeltFoam/pEqn.H` |
| Compiled binary | `$FOAM_USER_APPBIN/laserMeltFoam` |
| Deploy bundle (12 files, 88 KB) | `dilu/benchmark/openfoam_crosscheck/deploy/` |
| `install.sh`, `patch_case.sh`, `sanity.py`, `shrink_dump.py` | `dilu/benchmark/openfoam_crosscheck/deploy/` |
| Python reader (auto-detects MM vs NPZ) | `dilu/benchmark/openfoam_crosscheck/reader.py` |
| Methodology design document | `docs/design/openfoam_crosscheck_plan.md` |
| Empirical benchmark report | `docs/benchmark/OPENFOAM_CROSSCHECK_20260426.md` |

---

## 10. Reproducibility from this document

The procedure described in §5 is fully self-contained provided:

1. OpenFOAM v2506 is installed and `WM_PROJECT_DIR` is set.
2. A LaserbeamFoam source tree exists at a known path (the install script takes this as argument).
3. The deploy bundle is available — included in the DILU-Research repository at the path in §9.

A reproducibility check was performed on 2026-04-23 to 2026-04-26:

- Independent rebuild of the patched `laserMeltFoam` binary on a fresh OpenFOAM v2506 install: succeeded.
- Independent execution of the warmup + dump procedure on the spot-melt validation case: produced 250 matrices.
- Sanity check via `sanity.py`: 250 / 250 PASS.
- Conversion to NPZ via `shrink_dump.py`: 150 matrices retained, 5.5 GB total.
- Round-trip read via `reader.py` after off-machine transfer: continued to pass sanity.

No undocumented step exists in the procedure as of the date below.

---

**Document version**: 1.0
**Last verified**: 2026-04-26
**Validation case**: `openfoam_laserMeltFoam_spot` at `t ∈ [2.25, 3.23] μs`, 1,024,000 cells, 250 matrices written, 250 sanity-passed.
