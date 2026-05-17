# Highlights — DILU-Research

Project talking points in single-sentence form. Each has a verifiable number; pointers to evidence
are in `EVIDENCE.md`. Pick 3–5 depending on the application.

---

## Variant A — solver / HPC framing (best for systems / numerical-methods labs)

- **Designed and shipped a JAX-FFI binding to NVIDIA AMGx 2.5** (1.6 k LoC
  C++ + 0.9 k LoC Python + 11 docs) that exposes setup-once / solve-many
  semantics under JAX JIT through a process-global opaque-token registry
  guarded by `std::mutex`, enabling warm-start reuse of the AMG hierarchy
  across CFD timesteps with **47 % median iteration savings** on 500 K-cell
  LPBF pressure systems.

- **Achieved relative error ≤ 2.1 × 10⁻¹⁵ versus scipy `spsolve` direct LU**
  on 50 real-world LPBF matrices (pd + T) by chaining AMGx Classical V-cycle
  with a 1-step iterative-refinement wrapper; verified across **400
  independent solves** on dev RTX 3050 (sm_86) and lab RTX 5060 (sm_120) with
  zero failures.

- **Re-implemented OpenFOAM v2506's CPU `DILUPreconditioner + PBiCG`** from
  source (`lduMatrix*.C`, ~3 k LoC C++) as a 1 k-LoC Python / 0.35 k-LoC
  C++ reference module, matching OpenFOAM's iteration count exactly and
  field-level diff `‖x − x_OF‖∞ / ‖x_OF‖∞ < 10⁻¹⁰` on dumped LPBF systems;
  used as ground-truth oracle for all GPU optimizations.

- **Built a cross-hardware benchmark suite** (50 matrices × 4 protocols × 2
  equations, portable binary npz format) and ran a forensic precision audit
  (100+ matrices, Xeon Gold 5120, 134 git commits) that settled 14
  quantitative claims with calibrated confidence — including disproving an
  earlier mis-attribution that AMGx was more accurate than OpenFOAM
  (the gap was condition-number-amplified tolerance noise, not solver error).

- **Wrote reproducibility documentation to engineering-spec standard**: a
  fresh-context LLM agent, given only `REPRODUCE.md` + `KNOWN_GOTCHAS.md`
  and no source access, reimplemented the iterative-refinement layer and
  reproduced iteration count, final residual, and solution-vector SHA-256
  byte-exactly.

---

## Variant B — additive-manufacturing / CFD-application framing

- Studied numerical-precision trade-offs of GPU algebraic-multigrid solvers
  in the laser powder-bed-fusion CFD hot loop (laserMeltFoam / OpenFOAM
  v2506), where pressure and temperature solves dominate 60–80 % of wall
  time and where condition-number amplification (κ ≈ 10⁶) makes
  tolerance-only convergence metrics misleading.

- Identified and quantified a **previously-undocumented 10-dimensional
  near-null singular subspace** in the LPBF pressure matrix that persists
  across all sampled phases (pre-melt, melt, evaporation early/late) —
  showed via `svds k=10` on 7 timesteps that κ ≈ 10⁶ is a *structural
  property of the discretization*, not a single-timestep artifact.

- Delivered a production-validated AMGx FFI module to a research partner
  (private PR, 58 files / 9 117 LoC) and an independently-reproducible
  benchmark suite that ships across CUDA hardware generations (sm_86 →
  sm_120) without code changes.

---

## Variant C — software-engineering framing (one-liners for the "engineering
maturity" question)

- 134 commits over 6 months, all on a single research repo, with full
  pre-merge code review docs and a 15-item gotcha registry.
- Shipped as installable Python package (`pyproject.toml`, MIT) with API /
  install / usage / reproduction docs separated for different reader
  audiences.
- Used a fresh-context AI agent as a blind-reimplementation oracle to grade
  the quality of my own reproducibility documentation — byte-exact passes
  meant the docs were complete.

---

## How I'd phrase the "what did you do" elevator pitch

> "I rebuilt the inner sparse-solver layer of OpenFOAM's laser-melting CFD
> solver as a JAX-compatible GPU module. The hard part wasn't the multigrid
> — NVIDIA ships that — it was making the iteration count and the
> per-cell pressure agree with a direct-LU reference to 11 digits across
> 50 real LPBF matrices, on two GPU generations, and proving it with a
> reproducibility test that a blank-slate AI agent passes."
