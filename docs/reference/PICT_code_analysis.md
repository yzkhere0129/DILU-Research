# PICT — Code-Level Analysis of the Linear Solver and "Preconditioner"

**Source**: tum-pbs/PICT, ICLR/JCP-style preprint (Franz, Wei, Guastoni,
Thuerey, TU Munich). Local mirror: `/home/yzk/PICT-reference/PICT/`.

**Why this note exists**: a senior colleague described PICT as a
"PyTorch-based preconditioner method." That description is misleading.
PICT's contribution is a **differentiable PISO solver**, not a new
preconditioner. The preconditioner used inside PICT is stock NVIDIA
cuSPARSE ILU0 — strictly less sophisticated than what our stack already
runs. This note documents what PICT actually does, with file:line
pointers, and lists a handful of engineering tricks worth borrowing.

---

## TL;DR

| Question | Answer |
|----------|--------|
| Is PICT a preconditioner-research project? | **No.** It is a differentiable PISO solver. |
| What preconditioner does PICT use? | **cuSPARSE ILU0**, only inside BiCGStab. CG runs unpreconditioned. |
| Is the BiCGStab+ILU0 path novel? | **No.** It is essentially a port of NVIDIA's `pbicgstab.cpp` sample, modernized to the generic cuSPARSE API. The PICT source even cites the sample directly (`bicgstab_solver_kernel.cu:1-6`). |
| What is novel in PICT? | (a) End-to-end differentiability of the PISO loop (custom CUDA backwards kernels for autograd); (b) multi-block geometry with custom convolutions; (c) a pure-Neumann rank-deficient correction baked into CG. |
| Is anything worth borrowing into our stack? | A handful of engineering practices: rank-deficient correction `A + (trace(A)/n³)·eeᵀ`, periodic residual reset, best-iterate fallback, float32→float64 fallback chain, deterministic SpMV variant. None of these is a preconditioner. |
| Should we treat PICT as a competitor for "GPU preconditioner for LPBF matrices"? | **No.** PICT does not target ill-conditioned near-singular Neumann Laplacians. Their pressure CG is unpreconditioned; on our pd matrices (λ_min ≈ 1e-13, 14-orders diag span) it would either stall or rely on the rank-deficient correction. Our AMGx + DIAGONAL_SYMMETRIC path is mathematically stronger. |

---

## 1. What PICT is, at a glance

PISO algorithm in PyTorch + custom CUDA. The point is autograd through
the *entire* solver, including the linear solve (via the standard
adjoint trick `Aᵀ·∂b = ∂x`). Application targets are turbulence-model
learning, super-resolution correctors, and inverse problems — not
preconditioner research.

Layout of the source we read (`/home/yzk/PICT-reference/PICT/`):

```
extensions/
├── PISOtorch.cpp                   # pybind11 bindings, exposes SolveLinear
├── PISO_multiblock_cuda_kernel.cu  # SolveLinear dispatch (line 7022)
├── bicgstab_solver.h               # public C++ API of the linear solvers
├── bicgstab_solver_kernel.cu       # BiCGStab + ILU0
├── cg_solver_kernel.cu             # CG (unpreconditioned, with rank-def trick)
├── cublas_templates.h              # float/double templated wrappers
└── solver_helper.h                 # ComputeConvergenceCriterion shared helper
PISOtorch_diff.py                   # Python autograd.Function wrapping SolveLinear
```

The whole linear-solver footprint is **3 source files**: ~400 lines of
BiCGStab, ~450 lines of CG, ~270 lines of templated cuBLAS/cuSPARSE
wrappers. Nothing else.

---

## 2. The BiCGStab+ILU0 solver

File: `extensions/bicgstab_solver_kernel.cu` (404 lines).

### 2.1 The header tells you exactly what it is

```cpp
// bicgstab_solver_kernel.cu:1-6
/*
  https://github.com/tpn/cuda-samples/blob/master/v8.0/7_CUDALibraries/BiCGStab/pbicgstab.cpp
  https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
  https://docs.nvidia.com/cuda/archive/11.7.1/cublas/index.html
*/
```

This is the literal NVIDIA `pbicgstab.cpp` sample, ported to the modern
cuSPARSE generic API and templated for float/double.

### 2.2 The setup phase — factor once, solve N times

```cpp
// bicgstab_solver_kernel.cu:191-210
if(withPreconditioner){
    cusparseTcsrilu02_analysis<scalar_t>(... info_m ...);   // symbolic
    cusparseXcsrilu02_zeroPivot(handle, info_m, &structural_zero);
    cusparseSpSV_analysis(handle, trans_L, ..., descrSp_l, info_l, buffer);
    cusparseSpSV_analysis(handle, trans_U, ..., descrSp_u, info_u, buffer);
    cusparseTcsrilu02<scalar_t>(... info_m ...);             // numeric
}

// then:
for (index_t batchIdx=0; batchIdx<nBatches; ++batchIdx){    // line 221
    // BiCGStab inner loop, reusing info_l, info_u
}
```

This is **the same pattern we use with AMGx**: build the preconditioner
once, sweep over many right-hand sides. Difference: PICT freezes the
*matrix* across batches (same A, varying b — true multi-RHS). AMGx in
our `amortized_run` (`driver_amgx.py:252`) reuses the AMG hierarchy
across timesteps via `plan.update_coefficients(vv)`, which allows the
matrix values to change as long as the sparsity pattern is preserved.
Conceptually the same idea; AMGx's version is more flexible.

### 2.3 Optional preconditioner toggle

```cpp
// bicgstab_solver_kernel.cu:272-279
if(withPreconditioner){
    cusparseSpSV_solve(handle, trans_L, &one, descrSp_l, descr_p, descr_t, ...);
    cusparseSpSV_solve(handle, trans_U, &one, descrSp_u, descr_t, descr_pw, ...);
} else {
    cublasTcopy<scalar_t>(handle, n, p, 1, pw, 1);    // M = I
}
```

This matches the paper's claim that they default to no preconditioner
and fall back to ILU0 on failure (Python wrapper logic below).

### 2.4 Likely bug — copy-paste at descriptor attributes

```cpp
// bicgstab_solver_kernel.cu:155-163
cusparseCreateCsr(&descrSp_l, ...);
cusparseSpMatSetAttribute(descrSp_l, FILL_MODE, &fillmode_L, ...);   // L: fill=LOWER
cusparseSpMatSetAttribute(descrSp_l, DIAG_TYPE, &diagtype_L, ...);   // L: diag=UNIT

cusparseCreateCsr(&descrSp_u, ...);
cusparseSpMatSetAttribute(descrSp_l, FILL_MODE, &fillmode_U, ...);   // <-- should be descrSp_u
cusparseSpMatSetAttribute(descrSp_l, DIAG_TYPE, &diagtype_U, ...);   // <-- should be descrSp_u
```

The attributes for the upper factor are being set on the **lower**
descriptor. The visible end-state is: `descrSp_l` claims
`fill=UPPER, diag=NON_UNIT` (its LOWER attributes were overwritten);
`descrSp_u` has **no attributes set** at all (cuSPARSE default
behavior).

I have not run PICT to confirm whether this manifests as wrong
answers, NaN, or silently-correct behavior. cuSPARSE's
`cusparseSpSV_analysis` documentation (CUDA 12) states that
fill_mode and diag_type **must** be set before analysis, otherwise
the result is undefined. So the path either:
(a) errors at analysis (most likely) and the BiCGStab+ILU0 path is
broken in practice;
(b) silently produces wrong factor solutions and the BiCGStab outer
absorbs the error via more iterations.

Before using PICT's BiCGStab+ILU0 path for any of our matrices,
trace through what `cusparseSpSV_analysis` actually does on this
descriptor configuration, or just fix the typo and rebuild.

### 2.5 Tolerance change vs the NVIDIA reference

NVIDIA reference (`pbicgstab.cpp:167`):
```cpp
if (nrmr < tol*nrmr0){ ... break; }   // RELATIVE residual
```

PICT (`bicgstab_solver_kernel.cu:308`):
```cpp
if (nrmr < tol){ ... break; }          // ABSOLUTE residual
// the relative form is in the source as a commented-out line
```

PICT defaults to absolute residual with `ConvergenceCriterion =
NORM2_NORMALIZED` (i.e., L2 / √N). Sensible if the user controls the
RHS scale; can over-/under-converge on poorly scaled systems.

---

## 3. The CG solver — much more interesting than the BiCGStab

File: `extensions/cg_solver_kernel.cu` (911 lines, of which ~440 are
the active CG and ~400 are a commented-out preconditioned-CG variant).

### 3.1 The active path is **unpreconditioned**

`cgSolveGPU` (`cg_solver_kernel.cu:114-456`) calls `cusparseSpMV` for
A·p but never invokes `cusparseTcsrilu02*` or `cusparseSpSV_*`. There
is no M⁻¹ application. The function signature includes `const scalar_t
*aDiag` but it is used **only** inside the rank-deficient correction
(line 246), not as a Jacobi preconditioner.

The preconditioned variant (`cgSolvePreconGPU`, lines 466-867) is
written but commented out — including the dispatch in
`PISO_multiblock_cuda_kernel.cu:7046-7055`. The variant uses the legacy
`cusparseTcsrsv2_solve` API and supports an optional Polak-Ribière
β formula (`cgSolvePreconGPU:472`, `796-818`). That branch never runs.

### 3.2 The pure-Neumann / rank-deficient trick

This is the most interesting algorithmic detail in PICT.

```cpp
// cg_solver_kernel.cu:242-256
if(laplaceRankDeficient){  // compute (A + e^T·e) · x instead
    if(aDiag==nullptr){
        rankDeficientScaling = csrMatrixTrace<scalar_t>(...) / n / n / n;
    }else{
        cublasTasum<scalar_t>(handle, n, aDiag, 1, &rankDeficientScaling);
        rankDeficientScaling = rankDeficientScaling / n / n / n;
    }
    cublasTasum<scalar_t>(handle, n, x, 1, &temp);   // sum(x)
    temp *= rankDeficientScaling;
    cublasTaxpy<scalar_t>(handle, n, &temp, gpuOne, 0, r, 1);  // r += temp · 1
}
```

Mathematically: replace `A` with `A_ε = A + ε · e·eᵀ`, where
`e = (1,1,…,1)ᵀ` and `ε = trace(A)/n³`. The rank-1 update `e·eᵀ` is
applied implicitly: `(e·eᵀ)·x = (sum(x)) · e`, so they compute
`sum(x)` with `cublasTasum` and then broadcast-add a scalar via
`cublasTaxpy(... gpuOne, incx=0, ...)` (an axpy with `incx=0`
effectively reads a single scalar repeatedly — neat use of cuBLAS).

The intent is to anchor the constant-vector null space with a
perturbation small enough not to change the answer.

#### Spectral analysis

For a pure-Neumann Laplacian, `e = (1,…,1)ᵀ` is in the null space:
`A·e = 0`. Adding `ε·eeᵀ`:

> `A_ε · e = A·e + ε·n·e = ε·n·e`

so the zero eigenvalue is shifted to `ε·n = trace(A)/n²`. Other
eigenvectors are orthogonal to `e` (because A is symmetric and the
zero eigenspace is spanned by `e` for an irreducible Laplacian), so
they are unchanged.

#### Empirical test on our LPBF pd matrix

Sample case: `pd_corr0` at `t = 2.71322e-06 s`, N = 1,024,000.

```
diag:         |min| = 7.50e-27,  |max| = 2.30e-13   (span 3.06e+13)
trace(A) = -9.48e-08
ε = trace/n³ = -8.83e-26
λ_min      ≈ 9.27e-14   (lobpcg, 3 smallest)
ε · n      ≈ -9.04e-20  ← shift produced by PICT's correction
```

The shift `ε·n ≈ 10⁻¹⁹` is **six orders of magnitude smaller than the
existing `λ_min ≈ 10⁻¹³`**. The correction is therefore **invisible
to the spectrum** of our matrix. Two reasons:
1. `trace(A)` for our matrix is dominated by laser-deposition cells
   that have near-zero diag (`|min| = 7.5e-27`). Most of the trace is
   noise.
2. The `n³` denominator scales away anything except very large traces.

Worse, when applied to `r = b - A·x` the correction adds the constant
`ε · sum(x_OF) · e` to `r`. Empirically `sum(x_OF) = 1.07e+11`
(pressure not anchored to zero — physically valid since we use
`pRefCell 0` for one face only). The resulting per-entry residual
shift is `ε · sum(x_OF) ≈ 9.4e-15`, which is comparable to the
existing `‖A·x_OF − b‖₂ / ‖b‖₂ ≈ 1.5e-10` baseline:

```
||A·x_OF − b||/||b||  before correction = 1.53e-10
||A_ε·x_OF − b||/||b|| after correction  = 1.28e-05   ← 5 orders worse
```

So: **PICT's correction is wrong for our LPBF problem as-written.**

The reason it works in PICT's incompressible-flow setups is two-fold:
1. Their pressure system has a smaller dynamic range (uniform mesh,
   single fluid) → `trace(A)/n³` is closer to the actual `λ_min`.
2. Their pressure solution is mean-zero (or close to it) by construction
   in the incompressible PISO loop, so `eᵀx ≈ 0` and the residual
   shift is negligible.

For us, neither holds: the LPBF pd matrix has the 14-orders diag
span, and OpenFOAM's `pRefCell` only pins **one** cell so `sum(x)` is
free to drift to ~10¹¹.

#### Confirmation: PICT itself never enables the correction

`grep` over the entire PICT repo confirms that all real call sites
hard-code `matrix_rank_deficient = False`:

```
PISOtorch_diff.py:302   matrix_rank_deficient = False #not use_BiCG
PISOtorch_simulation.py:1024,1237,1280   # all comments-out: matrix_rank_deficient=False
```

The `laplaceRankDeficient` flag exists in the C++ API and the Python
wrapper but **is never set to True in any of the bundled PISO loops or
test cases**. PICT relies on its lid-driven-cavity / channel-flow setups
having a non-trivial pressure forcing (moving wall, periodic forcing
term) that breaks the pure-Neumann singularity by construction.

So the trick is essentially **dead code** in PICT itself. It is plausibly
correct for incompressible-flow setups *if you happened to need it*, but
neither PICT's authors nor the test suite exercise it. Treating it as a
"production-tested feature" of PICT would be misleading.

#### Implication for our stack

The PICT rank-deficient trick is **not a free upgrade** for our pd
solvers. To make it work we would need to either:
- (a) Project to mean-zero each iteration (`x ← x − mean(x)`),
  matching the PICT implicit assumption.
- (b) Replace `ε = trace/n³` with a problem-specific scale, e.g.
  `ε = (max|diag| − min|diag|) / n` or a multiple of the AMGx-detected
  diagonal scaling.
- (c) Skip the correction and rely on the AMGx `DIAGONAL_SYMMETRIC`
  scaling we already use. This is the path of least resistance.

We should **not** add `laplaceRankDeficient`-style logic to our drivers
without one of (a)–(c). It looks elegant in the PICT code but its
default scale is wrong for our matrix class.

### 3.3 Periodic residual reset

```cpp
// cg_solver_kernel.cu:266-287
if(resetResidual && (i+1)%residualResetSteps==0){
    cusparseSpMV(handle, transOp, &mone, descrSp_a, descr_x, &zero, descr_r, ...);
    // ... rebuild r = b - A·x from scratch, reset rho, p ...
}
```

Avoids floating-point drift in the implicit residual update
`r ← r − α·A·p` over many iterations. Standard CG hygiene. We don't
do this in any of our drivers and on our pd matrices (where iter ~50)
it is unlikely to matter; on T (iter can hit hundreds for tight tol)
it could.

### 3.4 Best-iterate fallback

```cpp
// cg_solver_kernel.cu:330-346, 399-412
if(returnBestResult){
    if(i==0 || criterion < bestCriterion){
        bestCriterion = criterion;
        bestCriterionIt = i;
        cublasTcopy(handle, n, x, 1, best_x, 1);
    }
    if(i>0 && criterion >= lastCriterion) ++criterionRisingSteps;
    else                                   criterionRisingSteps = 0;
    // ...
    if(criterionRisingSteps >= 100) {  // give up, return best
        cublasTcopy(handle, n, best_x, 1, x, 1);
        break;
    }
}
```

When CG diverges (residual rises 100 iters in a row), revert to the
best iterate and return with `converged=false`. Useful for autograd
training loops where a NaN propagates and corrupts the model — the
caller gets a finite (if not optimal) `x` and a flag.

### 3.5 ConvergenceCriterion enum

```cpp
// bicgstab_solver.h:11-18, cg_solver_kernel.cu:77-110
enum class ConvergenceCriterion : int8_t {
    NORM2 = 0,            // ‖r‖₂
    NORM2_NORMALIZED = 1, // ‖r‖₂ / √N    ← PICT default
    ABS_SUM = 2,          // Σ|rᵢ|
    ABS_MEAN = 3,         // (1/N) Σ|rᵢ|
    ABS_MAX = 4,          // max|rᵢ|
};
```

Five termination flavors selectable at the call site. We support only
the AMGx-defined ones (`L2` etc.) and `nrm2(r)` directly in our
cuSPARSE/Multicolor drivers.

---

## 4. Python-level fallback chain

File: `PISOtorch_diff.py` lines 245-293, the `_linear_solve_wrapper`.

```python
# PISOtorch_diff.py:251-272
solver_infos = PISOtorch.SolveLinear(csrMat, rhs, result, ..., useBiCG, ...)

if double_fallback and rhs.dtype == torch.float32 and not_solved(solver_infos):
    # retry in float64
    result_dp = torch.zeros_like(result, dtype=torch.float64)
    solver_infos = PISOtorch.SolveLinear(csrMat.toType(dp), rhs.to(dp), ...)

if BiCG_precondition_fallback and use_BiCG and not BiCG_with_preconditioner \
        and not_solved(solver_infos):
    # retry with ILU0 enabled
    result.zero_()
    solver_infos = PISOtorch.SolveLinear(..., BiCGwithPreconditioner=True)
```

This is the "case-by-case preconditioner" the paper mentions, made
concrete: try fast path (no preconditioner, float32), fall back to
preconditioner, then fall back to double precision. The default
tolerances are `1e-5` (float32) and `1e-8` (float64) — significantly
looser than our `1e-8` PCG tolerance.

For our work this fallback chain is interesting as a *driver*-level
pattern. We could add a fallback in `driver_amgx.py` so that an AMGx
solve that returns `converged=false` is retried with a stricter config
or in double precision before declaring failure.

---

## 5. The PISOtorch.SolveLinear API surface

```cpp
// PISOtorch.cpp:559-563
m.def("SolveLinear", &SolveLinear, ...,
    py::arg("A"), py::arg("RHS"), py::arg("x"),
    py::arg("maxIterations") = 1000,
    py::arg("tolerance") = 1e-8,
    py::arg("convergenceCriterion") = ConvergenceCriterion::NORM2_NORMALIZED,
    py::arg("useBiCG") = false,                  // CG by default (unpreconditioned)
    py::arg("matrixRankDeficient") = false,
    py::arg("residualResetSteps") = 0,
    py::arg("transposeA") = false,
    py::arg("printResidual") = false,
    py::arg("returnBestResult") = false,
    py::arg("BiCGwithPreconditioner") = true);   // ILU0 only when BiCG=true
```

Dispatch (`PISO_multiblock_cuda_kernel.cu:7037-7064`):
- `useBiCG = true` → `bicgstabSolveGPU` (with `withPreconditioner` toggle)
- `useBiCG = false` → `cgSolveGPU` (unpreconditioned, with `matrixRankDeficient` toggle)

The intermediate branch (`useBiCG=false && matrixRankDeficient`) that
*would* call `cgSolvePreconGPU` is commented out. The Python user has
no way to invoke preconditioned CG.

---

## 5b. How PICT folds boundary conditions into the pressure matrix

This section is for the senior who is investigating "BC matrix assembly"
issues. PICT and OpenFOAM disagree on **where** boundary information
goes — matrix vs RHS — even on the simplest Dirichlet/Neumann case.

### 5b.1 The PICT pattern

`PISO_build_pressure_matrix` (`PISO_multiblock_cuda_kernel.cu:4798-4963`)
loops over each cell and each of its 6 faces. For each face it asks:

```cpp
const bool atPrescribedBound = atBound && isEmptyBound(bound, s_block.boundaries);
```

where `isEmptyBound` returns `true` for `DIRICHLET`,
`DIRICHLET_VARYING`, `GRADIENT`, or `FIXED`
(`PISO_multiblock_cuda_kernel.cu:186-194`). Note: **all four are
treated the same** at matrix-assembly time.

```cpp
if(!atPrescribedBound){
    // ... add the face's Laplace contribution to diag (rowValues[0])
    //     and to the off-diagonal for the neighbor cell (rowValues[bound+1])
} else { // prescribedBound
    indices[bound+1] = -1;     // skip — no off-diag entry for the missing neighbor
    // diag is also untouched for this face
}
```

So at a prescribed boundary cell, **PICT contributes nothing to the
pressure matrix from that face**. The boundary's effect is added to
the RHS in `PISO_build_pressure_rhs`. This is symmetric, simple, and
makes the matrix structure *boundary-type-agnostic*: a Dirichlet
boundary cell and a Neumann boundary cell have the same matrix row.

### 5b.2 The OpenFOAM pattern (what our matrixDumper sees)

OpenFOAM's `fvMatrix<Type>` distinguishes BC types at assembly:

- `zeroGradient` (Neumann): face flux = 0 → no contribution to diag,
  no contribution to RHS. (Same as PICT.)
- `fixedValue` (Dirichlet): face flux = `−α/Δb · (φ_b − φ_P)` →
  - diag picks up `+α/Δb`
  - RHS picks up `+α/Δb · φ_b`
- `mixed` / `cyclic` / `processor`: more complex contributions to
  both diag and RHS (and, for `processor`, off-diag rows in another
  rank's local matrix).

Our `matrixDumper.H` reproduces this fold manually because OpenFOAM's
`addBoundarySource()` is `protected`. The validation
`‖A·x_OF − b‖/‖b‖ ≈ 1e-7` (`docs/specs/MATRIX_EXTRACTION_SPEC.md`)
confirms the fold matches OpenFOAM for `fixedValue` + `zeroGradient`.
We have **not** verified `mixed`/`cyclic`/`processor` — and that is
exactly the area where the senior's BC investigation matters.

### 5b.3 The mismatch — why PICT's pressure matrix would not solve our pd

If you handed our `pd_corr0` matrix (extracted from OpenFOAM) to
PICT's CG (without the rank-deficient correction), the matrix has
diag entries that already include the boundary fold from
`fixedValue` faces. If you then re-applied PICT's pressure-assembly
logic on top, you would double-count the boundary contribution. So
PICT's solver expects an **un-folded** Laplacian + a separate boundary
RHS, while we extract a **folded** Laplacian where the boundary
information already lives inside `A` and `b`.

This is the right framing of the senior's concern: "matrix assembly
with BC" is non-trivial because different solver backends assume
different conventions. Three you'll see in the wild:
1. **OpenFOAM convention** — Dirichlet face contributes to diag and
   RHS; Neumann face contributes nothing.
2. **PICT convention** — All boundary faces are skipped in the matrix;
   all boundary information lives in RHS.
3. **Pinning convention** — Insert a row `[1, 0, …, 0]` at one cell to
   pin the constant null space; leave the rest unchanged.

A solver that consumes a matrix from one convention and assumes
another will produce wrong answers, often with a clean-looking
self-consistency residual. Our `rel_vs_OF` metric catches this
because it compares against OpenFOAM's own `x_OF`, not against
`‖A·x − b‖`.

---

## 6. Cross-comparison: NVIDIA pbicgstab.cpp vs PICT vs us

| Aspect                         | NVIDIA sample (2016)            | PICT (2025)                              | Ours (DILU/AMGx, 2026)                                |
|--------------------------------|----------------------------------|------------------------------------------|--------------------------------------------------------|
| cuSPARSE API                   | Legacy `csrsv`/`csrmv`           | Modern `SpSV`/`SpMV` generic             | Legacy direct + AMGx C-API                            |
| BiCGStab preconditioner         | ILU0 (always on)                 | ILU0 (toggleable, default on)            | DILU (cuSPARSE), Multicolor DILU, AMG                 |
| CG preconditioner               | n/a                              | **None** (active path)                   | DILU, Multicolor DILU, AMG                            |
| Convergence test                | Relative `tol·‖r₀‖`              | Absolute `tol`, 5 criteria               | AMGx `RELATIVE_INI_CORE`; absolute in cuSPARSE driver |
| Multi-RHS amortization          | No                               | Yes — same A, N right-hand sides         | AMGx via `update_coefficients` (A values can change)  |
| Across-timestep amortization    | n/a                              | Not exposed                              | AMGx `update_coefficients` per timestep               |
| Singular-system handling        | None                             | `A + (trace(A)/n³)·eeᵀ` (CG only)       | `DIAGONAL_SYMMETRIC` AMGx scaling (different mechanism) |
| Float32 fallback to float64     | No                               | Yes (Python wrapper)                     | No (we run float32 throughout)                        |
| Best-iterate fallback           | No                               | Yes (CG only)                            | No                                                    |
| Periodic residual reset         | No                               | Yes (CG only)                            | No                                                    |
| Differentiability               | No                               | Yes (custom CUDA backwards kernels)      | No                                                    |
| Smoother                        | n/a                              | n/a                                      | AMGx `BLOCK_JACOBI`                                   |
| Coarsening                      | n/a                              | n/a                                      | AMGx `CLASSICAL` + interpolator `D2`                  |
| Ground-truth precision metric   | n/a                              | Self-consistency only                    | `‖x − x_OF‖_∞ / ‖x_OF‖_∞` against OpenFOAM PCG-DIC    |

Reading the table left to right is essentially a chronology of GPU
sparse-iterative engineering: NVIDIA gave the kernel building blocks,
PICT bundled them inside a differentiable PISO loop, and our work uses
strictly more sophisticated preconditioners (real AMG, multicolor
DILU) on a harder class of matrices (near-singular Neumann LPBF
Laplacians).

---

## 7. What is worth borrowing into our stack

Concrete, low-cost engineering pickups, in priority order. **Note:**
the rank-deficient correction (Section 3.2) was demoted from this list
after the empirical test on our pd matrix showed it makes the residual
worse. See §3.2 for the analysis.

1. **Driver-level fallback chain**
   In `driver_amgx.py`, when `solver.solve()` returns
   `convergence=false` or NaN: retry with stricter config (or double
   precision). Pattern from `PISOtorch_diff.py:251-286`. Cost: ~30
   lines wrapping the existing solve call.

2. **Best-iterate fallback for cuSPARSE/Multicolor PCG**
   Add `return_best_result` flag to `_make_pcg_jit` in
   `driver_cusparse.py:62`. The JAX `while_loop` body already carries
   the residual; tracking `argmin` adds one comparison and one
   `lax.cond`. Useful for sanity on rare divergent runs.

3. **Periodic residual reset**
   Add to the same JIT'd `pcg` body. Every K iterations, recompute
   `r = b - A·x` directly. Likely a no-op on our pd (CG converges in
   ~50 iters), but cheap insurance for the T-equation BiCGStab inner
   that runs longer.

4. **Document the SpMV algorithm choice**
   PICT explicitly chooses `CUSPARSE_SPMV_CSR_ALG2` over `ALG1`
   "(faster but not deterministic)" — `bicgstab_solver_kernel.cu:104`.
   We should document our SpMV backend choice (`SPMV_BACKEND=seg_static`)
   with the same determinism rationale.

5. **Float32-with-double-fallback as a default mode**
   PICT defaults to float32 (tol=1e-5) and only escalates to float64
   when the float32 solve fails — `PISOtorch_diff.py:261-272`. This is
   relevant for AMGx where float32 setup is roughly 2× faster than
   float64 but occasionally produces a non-finite residual on
   pathologically scaled rows. Cost: a `try/retry` wrapper around the
   AMGx solve.

Things explicitly **not** worth borrowing:

- The BiCGStab+ILU0 path itself — we already have stronger
  preconditioners (DILU and AMG) and our Multicolor DILU is faster
  per-iteration than ILU0 (because of the GPU parallel-friendly
  multicolor reordering).
- The unpreconditioned CG default — on our pd matrices (cond ≈ 10¹⁴)
  unpreconditioned CG would not converge in any reasonable iter budget.
- The likely descriptor copy-paste bug at lines 162-163.

---

## 7b. Practical experiment — could we run our pd/T matrices on PICT?

For the senior, who may be wondering whether to benchmark PICT against
our stack on the same matrix data: **yes, with a small wrapper**, but
the comparison would be lopsided. Here is the practical recipe and the
expected outcome.

### Wrapper sketch

PICT's `SolveLinear` takes a `CSRmatrix` (`A`), a flat RHS tensor, and
a flat result tensor. Our npz dataset stores `A_data`, `A_indices`,
`A_indptr`, `b`, `x_final` per matrix. A 30-line Python wrapper would:

```python
import torch, PISOtorch, numpy as np
data = np.load("collected/spot_melt_npz/<TS>/pd_corr0/data.npz")
n   = int(data["n_rows"])
csr = PISOtorch.CSRmatrix.from_arrays(
    torch.from_numpy(data["A_data"]).cuda(),
    torch.from_numpy(data["A_indices"]).cuda(),
    torch.from_numpy(data["A_indptr"]).cuda(),
    n=n)
b = torch.from_numpy(data["b"]).cuda()
x = torch.zeros_like(b)
infos = PISOtorch.SolveLinear(
    csr, b, x,
    maxIterations=torch.IntTensor([5000]),
    tolerance=torch.tensor([1e-8]),
    convergenceCriterion=PISOtorch.ConvergenceCriterion.NORM2_NORMALIZED,
    useBiCG=False,                # CG path — unpreconditioned
    matrixRankDeficient=True,     # turn on the eeᵀ correction
    residualResetSteps=50,        # avoid drift on long runs
    returnBestResult=True,
)
x_of = torch.from_numpy(data["x_final"]).cuda()
print("rel_vs_OF =",
      torch.linalg.vector_norm(x - x_of, ord=float("inf")) /
      torch.linalg.vector_norm(x_of, ord=float("inf")))
```

(`CSRmatrix.from_arrays` — or whatever the exact constructor is —
will need verification; PICT's normal flow constructs it via
`Domain.PrepareSolve` from a Block with boundaries.)

### Expected outcome

- **pd matrices** (CG, near-singular Neumann): PICT's unpreconditioned
  CG will likely stall or run thousands of iterations before reaching
  any meaningful tolerance, because (a) condition number of our pd is
  ≈10¹⁴ and (b) the rank-deficient correction's `ε = trace/n³` is too
  small to anchor the spectrum on this matrix class (§3.2). Expect
  `convergence=False` after `maxit` and a `rel_vs_OF` of order 1.
- **T matrices** (BiCGStab+ILU0, non-symmetric): assuming the BC
  fold convention mismatch (§5b) does not corrupt the input — and
  assuming the descriptor bug (§2.4) does not silently corrupt the
  ILU0 — the path *might* converge, but slower than our cuSPARSE
  BiCGStab+DILU because ILU0 requires inherently sequential
  triangular solves while DILU is essentially diagonal.

So the experiment is *interesting as a confirmation that PICT's
default solver is not built for our matrix class*, but it will not
produce a competitive number for PICT. If asked "did you compare
against PICT", the honest answer is: PICT is not a competitor — it
wraps a strictly weaker subset of cuSPARSE preconditioners.

---

## 8. Honest framing for the senior

If asked to summarize what PICT does for preconditioning:

> PICT is a differentiable PISO solver (PyTorch + custom CUDA) for
> ML-coupled fluid simulation. The linear solver inside PICT is a
> textbook cuSPARSE BiCGStab+ILU0 (for advection-diffusion) plus an
> unpreconditioned CG with a pure-Neumann correction `A + ε·eeᵀ` (for
> pressure). The BiCGStab path is essentially a port of NVIDIA's
> `pbicgstab.cpp` sample to the modern cuSPARSE generic API. There is
> no new preconditioner. The paper's contribution is differentiability
> through PISO, not solver acceleration.
>
> Useful pickups for our DILU/AMGx work: the best-iterate fallback,
> the float32→float64 retry pattern, periodic residual reset, and
> explicit deterministic SpMV. None of these is a preconditioner.
> The rank-deficient `A + (trace/n³)·eeᵀ` correction looks tempting
> but is empirically wrong-scaled for our LPBF pd matrix (§3.2): the
> shift it produces is six orders smaller than `λ_min`, and the
> residual perturbation `ε·sum(x_OF)` is large enough to kick
> `‖A·x − b‖` from 1e-10 to 1e-5. Don't port it without re-deriving
> `ε` for our matrix class.

If the senior is interested in PICT for *preconditioner* research, the
honest answer is: PICT is a downstream consumer of NVIDIA's
preconditioner libraries; it is not a contributor to that area. Our
benchmark (`docs/benchmark/OPENFOAM_CROSSCHECK_20260427_v3.2.md`)
already operates at a more sophisticated level (AMGx + DIAGONAL_SYMMETRIC
scaling on a harder matrix class).

---

## 9. References

Original cited by PICT:
- NVIDIA pbicgstab sample:
  `https://github.com/tpn/cuda-samples/blob/master/v8.0/7_CUDALibraries/BiCGStab/pbicgstab.cpp`
- NVIDIA Incomplete LU/Cholesky guide:
  `https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html`
- cuBLAS reference (CUDA 11.7):
  `https://docs.nvidia.com/cuda/archive/11.7.1/cublas/index.html`

Local clones for reading:
- `/home/yzk/PICT-reference/PICT/` (full repo, depth 1)
- `/home/yzk/PICT-reference/cuda-samples-tpn/v8.0/7_CUDALibraries/BiCGStab/`
  (sparse checkout of the BiCGStab sample only)

PICT paper: `docs/reference/PICT.md` (this repo).
