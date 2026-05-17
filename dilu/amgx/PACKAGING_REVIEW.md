# AMGx FFI Module — Packaging / Drop-in Submodule Review

**Reviewer:** Claude (code-review agent)
**Date:** 2026-05-17
**Scope:** `dilu/amgx/{python,cpp,CMakeLists.txt,build.sh,tests,configs,bench/suite}` as a drop-in submodule for an external JAX-based CFD repository.
**Out of scope:** the 60-ish research scripts under `dilu/amgx/bench/*.py` (not the suite). Sibling phases `dilu/cusparse/`, `dilu/openfoam_cpu/`, `dilu/benchmark/`.

This complements (and does NOT repeat) `dilu/amgx/CODE_REVIEW.md` (4 Critical + 5 High + 8 Medium + 10 Low).
Where I reference `CODE_REVIEW.md` I use its IDs (C1-C4, H1-H5, M1-M8, L1-L10).

---

## 1. Verdict

**Blocked-on-cleanup.** The C++ shim, the Python public API, the configs, and the binary-suite runner are functionally complete and architecturally clean enough to merge. But six categories of issue make the current tree unsafe to drop into a sibling's repository AS-IS:

1. **All four `CODE_REVIEW.md` Critical bugs are still present in the source** (C1-C4 verified live).
2. **Three test files have hard cross-phase imports** (`dilu.cusparse.*`, `dilu.benchmark.*`) that will fail to collect after the submodule is detached from this monorepo.
3. **The diagnostic message in `CMakeLists.txt:24` and `build.sh:5` literally name `/home/yzk/jax-env`** — surface-level cosmetic but explicit user-facing leak.
4. **No packaging metadata** (`pyproject.toml`, `requirements.txt`, `LICENSE`, `MANIFEST.in`) exists at all.
5. **The benchmark suite's git-sha discovery climbs five directory levels** — will return wrong SHA (or "unknown") once relocated.
6. **`test_precision_vs_truth.py` hardcodes four `/home/yzk/LaserbeamFoam/...` paths and silently skips on absence** (H5), so CI on the partner repo will green-light with zero precision coverage.

After resolving these (estimated ~half a day of mechanical work, no design rethink), the module is mergeable. The core lifecycle/correctness design has nothing wrong with it.

---

## 2. Checklist (PASS/FAIL/N/A + evidence)

### A. Module self-containment

| ID | Check | Verdict | Evidence |
| --- | --- | --- | --- |
| A1 | `dilu/amgx/` self-contained Python imports | **FAIL** | Core `python/` is closed (only relative imports inside the package — verified by `grep "from dilu\." python/`). But tests reach outside: `tests/test_t8_correctness.py:20` imports `dilu.cusparse.python`, `tests/test_t12_physical_64.py:38` and `tests/test_t_acid_32.py:26` import `dilu.cusparse.tests.physical_benchmark`, `tests/test_precision_vs_truth.py:25` imports `dilu.benchmark.openfoam_crosscheck.reader`. Four test files are NOT self-contained. |
| A2 | Public API documented | **PARTIAL** | `python/__init__.py` defines `__all__` cleanly (17 symbols). All four FFI primitives, the `Plan` class, 9 config strings, `with_tolerance`, and `amgx_solve_with_refinement` are exported. Docstrings exist on all of them, but type hints are absent (CODE_REVIEW.md L1) and `MINI_AMG_TEST` is exposed as production API even though it is a smoke-test config (CODE_REVIEW.md L3). |
| A3 | `bench/suite/` is self-contained | **PASS-with-caveat** | Only `run_benchmark.py` imports `dilu.amgx.python` (suite README documents `PYTHONPATH=.` requirement). `select_50_matrices.py`, `convert_mm_to_npz.py`, `compute_truth_err.py`, `plot_results.py` are stdlib + numpy/scipy/matplotlib only. However `run_benchmark.py:99` does `Path(__file__).parent.parent.parent.parent.parent` to find the git root — 5 levels up assumes the current monorepo layout exactly. Once moved, this returns the wrong directory and `git rev-parse HEAD` silently logs the wrong SHA. |
| A4 | CMake finds AMGx + CUDA robustly | **PASS** | `CMakeLists.txt:35-42` chains `-DAMGX_ROOT=` → `$ENV{AMGX_ROOT}` → `$HOME/local/amgx`, errors loudly if `libamgxsh.so` or `amgx_c.h` missing (lines 45-52). `find_package(CUDAToolkit)` is portable. |
| A5 | Hidden absolute paths in source/tests | **FAIL** | `CMakeLists.txt:24` (error message), `build.sh:5` (comment), `tests/test_precision_vs_truth.py:33,36,39,42,118,144-145,149` (8 sites hardcoded to `/home/yzk/LaserbeamFoam/...`). The first two are cosmetic; the test paths block CI coverage. Suite + core C++ + core Python (other than test data) are clean. |

### B. Upstream compatibility (toward sibling JAX repo)

| ID | Check | Verdict | Notes |
| --- | --- | --- | --- |
| B1 | Package name viable as-is | **NEEDS DECISION** | The current path is `dilu.amgx.python.*` — three levels deep with the throwaway `python` directory, named with this lab's project handle. *Recommendation:* rename to a single top-level Python package (e.g. `jaxcfd_amgx` or `<sibling_project>.solvers.amgx`) and flatten the `python/` directory into the package root. The `python/` directory exists to separate FFI C++ from Python; once packaged, the C++ build artifact (`libdilu_amgx.so`) lives under the wheel data dir, and the Python module is the package. *Speculative:* if their layout is `pkg/solvers/<name>/`, target `pkg/solvers/amgx/` containing `__init__.py`, `wrapper.py`, etc. directly. |
| B2 | Dependency declaration | **FAIL** | No `pyproject.toml`, no `setup.py`, no `setup.cfg`, no `requirements.txt`. Hard runtime deps inferred from imports: `jax`, `jax[cuda12]`, `numpy`, `scipy` (refinement.py uses scipy.sparse via the caller's A object; tests use scipy explicitly). Build deps: `cmake>=3.24`, AMGx 2.5.0 from source, `gcc>=11`. **Action:** add at minimum a `pyproject.toml` with `[project]` table (name, version, deps), a `MANIFEST.in` (include C++ sources, CMakeLists, config JSONs), and a `LICENSE` (the sibling repo likely has a chosen license — copy theirs or coordinate). |
| B3 | Target directory tree in sibling repo | **GUIDANCE** | See §3 below for the recommended packaging tree. The minimal-invasion target is: rename `dilu/amgx/python/` → `<pkg>/amgx/`, move `cpp/` + `CMakeLists.txt` + `build.sh` next to it as `<pkg>/amgx/_native/`, put tests at `<pkg>/amgx/tests/`, suite at `<pkg>/amgx/benchmarks/`. |

### C. CODE_REVIEW.md Critical fix status (live grep against current source)

| ID | What it claimed | Status now | Evidence |
| --- | --- | --- | --- |
| C1 | `cudaMemcpyAsync` of stack-local scalars in all 4 handlers (use-after-scope risk) | **NOT FIXED** | `cpp/amgx_setup.cc:208-211` still does `uint64_t token = ...; cudaMemcpyAsync(...)` with no post-sync. `cpp/amgx_solve.cc:138-145` still stack-allocates `iters_host`/`status_host`. `cpp/amgx_update_coefficients.cc:94-96, 106-109` still stack-allocates `status`. `cpp/amgx_release.cc:43-45` likewise. |
| C2 | `amgx_release` returns 1 on success / 0 on unknown (inverted vs Python docstring) | **NOT FIXED** | `cpp/amgx_release.cc:43`: `int32_t status = removed ? 1 : 0;` unchanged. `python/wrapper.py:85`: still claims "Idempotent: releasing an unknown token returns 0 rather than erroring." Contracts still disagree. |
| C3 | `refinement.py` flips `jax_enable_x64=True` at import | **NOT FIXED** | `python/refinement.py:17-18`: `from jax import config as _jc; _jc.update("jax_enable_x64", True)` runs on `import dilu.amgx.python` (re-exported via `__init__.py:30`). |
| C4 | `CMakeLists.txt` + `build.sh` hardcode `sm_86` | **NOT FIXED** | `CMakeLists.txt:9-11` still defaults to 86. `build.sh:21` still unconditionally sets `-DCMAKE_CUDA_ARCHITECTURES=86`, defeating the CMake `if(NOT DEFINED ...)` guard. |

#### Suggested patches (each ≤ 10 lines diff, mechanically appliable)

**C1 — add post-copy `cudaStreamSynchronize` (4 files, mechanical):**
```diff
--- a/dilu/amgx/cpp/amgx_setup.cc
+++ b/dilu/amgx/cpp/amgx_setup.cc
@@ -207,9 +207,11 @@
   uint64_t token = plan_cache_insert(entry);
   CHECK_CUDA(cudaMemcpyAsync(token_out->typed_data(), &token,
                              sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream));
+  // Stack-local `token` MUST out-live the H->D copy. Force sync.
+  CHECK_CUDA(cudaStreamSynchronize(stream));
   return ffi::Error::Success();
 }
```
Apply the same `cudaStreamSynchronize(stream)` right before each `return ffi::Error::Success();` in `amgx_solve.cc`, `amgx_update_coefficients.cc`, and `amgx_release.cc`.

**C2 — make `amgx_release` status `0 = success` (idempotent):**
```diff
--- a/dilu/amgx/cpp/amgx_release.cc
+++ b/dilu/amgx/cpp/amgx_release.cc
@@ -41,8 +41,10 @@
                           cudaGetErrorString(e));
   }
-  bool removed = plan_cache_remove(token_host);
-  int32_t status = removed ? 1 : 0;
+  // Idempotent: 0 = success (released OR was already absent). Reserve
+  // non-zero for "actual teardown error" once such cases are implemented.
+  (void)plan_cache_remove(token_host);
+  int32_t status = 0;
   e = cudaMemcpyAsync(status_out->typed_data(), &status, sizeof(int32_t),
                       cudaMemcpyHostToDevice, stream);
```
Also update `REPRODUCE.md` Step 4 (line 170-173) which currently documents the inverted C++ behavior as authoritative — that section must be retracted once the C++ matches Python.

**C3 — move `jax_enable_x64=True` out of import-time:**
```diff
--- a/dilu/amgx/python/refinement.py
+++ b/dilu/amgx/python/refinement.py
@@ -14,9 +14,11 @@
 import numpy as np

 from jax import config as _jc
-_jc.update("jax_enable_x64", True)   # AMGx FFI expects float64
 import jax.numpy as jnp

+if not _jc.read("jax_enable_x64"):
+    raise RuntimeError(
+        "dilu.amgx requires jax_enable_x64=True. "
+        "Call jax.config.update('jax_enable_x64', True) BEFORE importing.")
+
 from .plan import Plan
```
Add a paragraph to `INSTALL.md` (when created — see §E) telling users to set the flag before import.

**C4 — honor `DILU_AMGX_CUDA_ARCH` env var:**
```diff
--- a/dilu/amgx/build.sh
+++ b/dilu/amgx/build.sh
@@ -17,10 +17,13 @@ AMGX_ROOT="${AMGX_ROOT:-$HOME/local/amgx}"

 export AMGX_ROOT

+# Override e.g. DILU_AMGX_CUDA_ARCH=120 (Blackwell) or "86;90;120" (fat binary).
+# Ampere=86, Ada=89, Hopper=90, Blackwell=120.
+CUDA_ARCH="${DILU_AMGX_CUDA_ARCH:-86}"
 CMAKE_FLAGS=(
   -DCMAKE_BUILD_TYPE=Release
-  -DCMAKE_CUDA_ARCHITECTURES=86
+  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
   "-DAMGX_ROOT=${AMGX_ROOT}"
 )
```
No CMake change needed; CMake's existing `if(NOT DEFINED ...)` guard already supports an explicit `-D` from upstream.

### D. Test minimum-runnable set (post-detach)

Classification by external dependency, assuming the sibling repo will NOT have `dilu.cusparse`, `dilu.benchmark`, or LaserbeamFoam matrix dumps:

| Test file | Runs cleanly on fresh repo? | Why / why not |
| --- | --- | --- |
| `test_smoke.py` | **YES** | Uses only `dilu.amgx.python` + local `_harness.laplacian_3d_7point`. Should be the first-run smoke test. |
| `test_t9_iter_count_128.py` | **YES** | Only `dilu.amgx.python` + local `_harness.stiff_laplacian_3d`. Caveat: M3 nvidia-smi parse is fragile. |
| `test_t10_wall_time_128.py` | **YES** | Same as T9. Caveat: hardcoded `DILU_BASELINE_128 = 28.47` (line 25) is a single-machine baseline that probably won't hold on the sibling's hardware — fix to env-var or skip. |
| `test_t11_under_jit.py` | **YES** | Local-only deps. Caveat: M6 brittle HLO grep. |
| `test_t13_vram_budget.py` | **YES** | Local-only deps. Caveat: M3 nvidia-smi parsing. |
| `test_update_coefficients.py` | **YES** | Local-only deps. |
| `test_regression_oracle.py` | **YES** | Local-only deps (verified via grep). |
| `test_t8_correctness.py` | **NO** | `from dilu.cusparse.python import Plan as CusparsePlan, build_diag_offset` at line 20 + reaches into `plan._values` private state (CODE_REVIEW.md H4). Must swap to `scipy.sparse.linalg.spsolve` as reference per H4. |
| `test_t12_physical_64.py` | **NO** | Imports `dilu.cusparse.tests.physical_benchmark` (line 38) for the projection step helpers. Must vendor those helpers into `tests/_harness.py` or drop the test from the packaged module. |
| `test_t_acid_32.py` | **NO** | Same as T12 — imports `dilu.cusparse.tests.physical_benchmark` (line 26). Same fix. |
| `test_precision_vs_truth.py` | **PARTIAL/NO** | Imports `dilu.benchmark.openfoam_crosscheck.reader` (line 25). Also hardcodes 4 `/home/yzk/LaserbeamFoam/...` paths (line 33-44) and silently skips on absence. Either ship a small canonical `.npz` and rewrite to read it (per H5), OR drop the test from the core suite and re-host under `bench/suite/` as an opt-in regression. |

**First-wave CI on sibling repo (no external data, no cusparse):** `test_smoke.py`, `test_t9`, `test_t10`, `test_t11`, `test_t13`, `test_update_coefficients`, `test_regression_oracle`. Seven tests, ~5-10 minutes wall on a CUDA host, zero external data.

### E. Documentation matrix

Existing `dilu/amgx/*.md`:

| File | Purpose | Adequate? |
| --- | --- | --- |
| `CODE_REVIEW.md` | Pre-merge code review (this is the input to PACKAGING) | Yes; ship as `REVIEW_NOTES.md` or `HISTORY/` archive. Not user-facing. |
| `REPRODUCE.md` | Build + reproduction spec (779 lines, agent-authored) | Mostly yes (see "REPRODUCE.md ↔ code drift" below). |

Suite has `bench/suite/README.md` (98 lines) which doubles as USAGE for the benchmark workflow.

**Missing for a drop-in package:**

| Document | Recommendation | Why |
| --- | --- | --- |
| `README.md` | **CREATE** (~80-120 lines) at module root. Sections: what it is, requirements, 3-line quickstart, link to INSTALL.md / USAGE.md / REPRODUCE.md, link to CODE_REVIEW for known issues. | A submodule with no README looks abandoned. |
| `INSTALL.md` | **CREATE** (~60 lines) by extracting Steps 1-3 of `REPRODUCE.md`. Include AMGx-source-build + JAX-version pin + `build.sh` invocation + smoke-test command. | REPRODUCE.md is too long for a first-time installer to read. |
| `USAGE.md` | **CREATE** (~80 lines) showing: the `Plan` context-manager pattern, the IR helper, the 9 configs and when to use each, the sign-flip rule for pd matrices, the `update_coefficients` amortization pattern. | The user has to read `python/refinement.py` + `python/config.py` to learn this today. |
| `API.md` (or `docs/api.md`) | **CREATE** by extracting `REPRODUCE.md` Appendix A (symbol table) and Step 4 (signatures). Add type hints (CODE_REVIEW.md L1). | A standalone API reference is what a senior collaborator will scan first. |
| `CHANGELOG.md` | **CREATE** with a single "v1.0.0 — initial drop" entry. | Standard library hygiene; the partner's CI may key releases on it. |
| `LICENSE` | **COORDINATE WITH PARTNER** before adding. AMGx is Apache-2.0; JAX is Apache-2.0; safest default for the shim is Apache-2.0 too, but the partner may have a house preference (BSD-3, MIT, etc.). | Currently absent. A merged module without LICENSE is unredistributable. |
| `pyproject.toml` | **CREATE** declaring name, version, deps, and the C++ extension build (`scikit-build-core` or plain CMake-driven `setuptools`). | See §B2. |

**REPRODUCE.md ↔ code drift to fix before publishing:**

1. REPRODUCE.md Step 4 (lines 170-173) **deliberately documents the C2 bug** as authoritative. Once C2 is fixed, this section must be retracted and replaced with "0 = success, non-zero reserved".
2. REPRODUCE.md Step 3 (lines 121-126) mentions the `build.sh` arch hardcode as "gotcha #6" but never points to a `KNOWN_GOTCHAS.md` file — that file does not exist in the tree. Either create it or strike the reference.
3. REPRODUCE.md Step 4 IR helper claims (line 192-194) that `import dilu.amgx` mutates `jax_enable_x64`; once C3 is fixed this paragraph also needs rewriting.
4. REPRODUCE.md Appendix A symbol table is accurate (cross-checked vs `python/__init__.py:32-49`).
5. REPRODUCE.md Step 5 config strings (9 configs) match `python/config.py` verbatim by spec; not byte-checked here but `CODE_REVIEW.md` M1 already flagged the duplication-by-hand risk.
6. REPRODUCE.md Step 7 reproduction script references `MINI_AMG_TEST` import — works today (`__init__.py:27, 46`); if you act on `CODE_REVIEW.md` L3 (drop from `__all__`) then the script must import from `dilu.amgx.python.config` instead. Add that note.
7. REPRODUCE.md "single GPU, device 0" (line 20) — no source-code grep contradicts this, but `plan_registry.cc:87` says `AMGX_resources_create_simple` on the bootstrap config, which DOES target device 0 implicitly via the CUDA context. Document confirmed correct.

---

## 3. Recommended packaging tree

What `dilu/amgx/` should look like the moment before `git push` to the sibling repo (or before `tar`-ing for a submodule):

```
<pkg>/amgx/                              # rename from dilu/amgx/python/ + cpp/
├── README.md                            (CREATE — 80-120 lines)
├── INSTALL.md                           (CREATE — extracted from REPRODUCE.md Steps 1-3)
├── USAGE.md                             (CREATE — Plan, IR, configs, sign-flip)
├── API.md                               (CREATE — symbol table from REPRODUCE.md App A + type hints)
├── CHANGELOG.md                         (CREATE — "v1.0.0 initial drop")
├── LICENSE                              (CREATE after coordinating with partner)
├── REPRODUCE.md                         (KEEP — strip out gotcha references to bugs once C1-C4 fixed)
├── REVIEW_NOTES.md                      (RENAME from CODE_REVIEW.md — historical record)
├── pyproject.toml                       (CREATE — name, deps, scikit-build-core or cmake hook)
├── MANIFEST.in                          (CREATE — include _native/* sources for sdist)
│
├── __init__.py                          (was python/__init__.py; drop MINI_AMG_TEST from __all__)
├── wrapper.py                           (was python/wrapper.py + type hints)
├── plan.py                              (was python/plan.py + H1 fix: __del__ → warning only)
├── config.py                            (was python/config.py + M1 factor-out)
├── registration.py                      (was python/registration.py + H2 narrow exceptions)
├── refinement.py                        (was python/refinement.py + C3 fix)
│
├── _native/                             (was cpp/ + CMakeLists.txt + build.sh)
│   ├── CMakeLists.txt                   (drop /home/yzk/jax-env diagnostic; keep AMGX_ROOT chain)
│   ├── build.sh                         (C4 fix: honor DILU_AMGX_CUDA_ARCH)
│   ├── plan_registry.h
│   ├── plan_registry.cc
│   ├── amgx_setup.cc                    (C1 fix)
│   ├── amgx_solve.cc                    (C1 fix)
│   ├── amgx_update_coefficients.cc      (C1 fix)
│   └── amgx_release.cc                  (C1 + C2 fix)
│
├── configs/                             (KEEP — classical_rs.json, future presets)
│   └── classical_rs.json
│
├── tests/
│   ├── README.md                        (CREATE — "run test_smoke.py first, then ...")
│   ├── conftest.py                      (KEEP; add comment for M7 env defaults)
│   ├── _harness.py                      (KEEP — vendored helpers for self-containment)
│   ├── data/                            (CREATE; ship 1-2 canonical .npz, ~5 MB each)
│   │   ├── pd_small_32cubed.npz         (small SPD pd from synthetic Laplacian or trimmed LPBF)
│   │   └── T_small_32cubed.npz          (small non-symmetric T)
│   ├── test_smoke.py                    (KEEP — first-run)
│   ├── test_t9_iter_count_128.py        (KEEP; address M3 nvidia-smi fragility)
│   ├── test_t10_wall_time_128.py        (KEEP; parameterize DILU_BASELINE_128 via env)
│   ├── test_t11_under_jit.py            (KEEP; address M6 HLO grep)
│   ├── test_t13_vram_budget.py          (KEEP; address M3)
│   ├── test_update_coefficients.py      (KEEP)
│   ├── test_regression_oracle.py        (KEEP)
│   ├── test_precision_vs_truth.py       (REWRITE: env-var paths + tests/data/ fallback per H5)
│   ├── test_t8_correctness.py           (REWRITE: drop dilu.cusparse ref, use scipy.sparse.linalg.cg per H4)
│   ├── test_t12_physical_64.py          (DECIDE: vendor physical_benchmark helpers or drop)
│   └── test_t_acid_32.py                (DECIDE: same as t12)
│
└── benchmarks/                          (was bench/suite/)
    ├── README.md                        (KEEP — current 98 lines is already good)
    ├── manifest.json                    (KEEP — 50-matrix selection)
    ├── select_50_matrices.py            (KEEP)
    ├── convert_mm_to_npz.py             (KEEP)
    ├── run_benchmark.py                 (FIX A3: parameterize git-root resolution via env var or remove the 5-level parent traversal)
    ├── compute_truth_err.py             (KEEP)
    └── plot_results.py                  (KEEP)
```

**What to leave behind in the source repo** (do NOT package):

- `dilu/amgx/_scratch/` — empty / development scratch, no packaging value.
- `dilu/amgx/build/` — local CMake artifacts, must be `.gitignore`d.
- `dilu/amgx/bench/*.py` (excluding `bench/suite/`) — the ~60 research-stream scripts (`bench_amgx_5060.py`, `plot_*.py`, `analyze_E01.py`, etc.) reference absolute paths under `audit_overnight_20260509/` and `docs/benchmark/figures/`; they are paper/figure pipelines specific to this monorepo. Keep them upstream-only.
- All `*.npz` data files in `dilu/amgx/bench/` root — large (~100 MB total), should be re-acquired from the suite generator.

---

## 4. Pre-upload action list (atomic, ordered by dependency)

Phase ordering: fix code → fix tests → add metadata → restructure → publish.

**Phase 1 — apply the 4 Critical patches (½ day):**

- [ ] Apply C1 patch (4 files: amgx_setup.cc, amgx_solve.cc, amgx_update_coefficients.cc, amgx_release.cc — add `cudaStreamSynchronize(stream)` before each `return Success`).
- [ ] Apply C2 patch (amgx_release.cc: status=0 always; update `python/wrapper.py:82-92` docstring to match — already correct).
- [ ] Apply C3 patch (refinement.py: replace import-time `_jc.update` with import-time read + raise; update `__init__.py` docstring).
- [ ] Apply C4 patch (build.sh: honor `DILU_AMGX_CUDA_ARCH` env var; add header comment listing arch numbers).
- [ ] Rebuild `libdilu_amgx.so` and re-run `test_smoke.py` + `test_regression_oracle.py` to confirm no regression.

**Phase 2 — make tests self-contained (½ day):**

- [ ] Vendor `dilu.cusparse.tests.physical_benchmark` helpers into `tests/_harness.py` (or delete `test_t12_physical_64.py` + `test_t_acid_32.py` from the packaged module if they are not load-bearing).
- [ ] Rewrite `test_t8_correctness.py` to use `scipy.sparse.linalg.cg` as the reference solver instead of cuSPARSE `Plan` (per CODE_REVIEW.md H4).
- [ ] Generate `tests/data/pd_small_32cubed.npz` and `tests/data/T_small_32cubed.npz` (≤10 MB each).
- [ ] Rewrite `test_precision_vs_truth.py` to read these fixture matrices and ONLY skip if the env-var override is unset AND the fixture is missing (so default behavior is "always covered").
- [ ] Vendor `dilu.benchmark.openfoam_crosscheck.reader.{load_ofmm, normalize_sign}` into `_harness.py` OR replace with a `np.load(...).item()` of the .npz bundle.

**Phase 3 — add packaging metadata (¼ day):**

- [ ] Create `pyproject.toml` with name, version `0.1.0`, `requires-python>=3.10`, deps `[jax>=0.5, numpy, scipy]`, build-system entry for `scikit-build-core` (or a plain CMake hook).
- [ ] Create `MANIFEST.in` including `_native/*.cc`, `_native/*.h`, `_native/CMakeLists.txt`, `_native/build.sh`, `configs/*.json`, `tests/data/*.npz`.
- [ ] Coordinate with partner on `LICENSE` choice and add the file.
- [ ] Strike `/home/yzk/jax-env` from CMakeLists.txt:24 diagnostic (replace with "is the JAX virtualenv active?") and from build.sh:5 header comment.

**Phase 4 — write user-facing docs (½ day):**

- [ ] Write `README.md` (80-120 lines).
- [ ] Write `INSTALL.md` by extracting REPRODUCE.md Steps 1-3.
- [ ] Write `USAGE.md` covering Plan / IR / configs / sign-flip / amortization.
- [ ] Write `API.md` from REPRODUCE.md Appendix A + Step 4 signatures.
- [ ] Write `CHANGELOG.md` initial entry.
- [ ] Strike sections of REPRODUCE.md that describe the now-fixed C2/C3 behavior (lines 170-173, 192-194); delete the dangling "gotcha #6 in KNOWN_GOTCHAS.md" reference (line 126) since that file does not exist.
- [ ] Write `tests/README.md` (10-15 lines, ordered first-run instructions).

**Phase 5 — restructure and validate (¼ day):**

- [ ] Decide final package name with partner (`<sibling_project>.solvers.amgx` recommended).
- [ ] Move `python/*` files up one level, rename to package root.
- [ ] Move `cpp/` + `CMakeLists.txt` + `build.sh` into `_native/`.
- [ ] Move `bench/suite/` → `benchmarks/`.
- [ ] Fix `benchmarks/run_benchmark.py:99` 5-level parent traversal → env-var-driven git-root (or just drop the git-sha enrichment).
- [ ] Add a `.gitignore` that excludes `_native/build/`, `__pycache__/`, `*.so`, `tests/data/*.npz` IF the partner prefers Git LFS or download script for fixtures.
- [ ] Run the seven first-wave tests on a clean clone to confirm nothing assumes the monorepo path.

**Phase 6 — publish (small):**

- [ ] Bump version to `0.1.0`, tag, push.
- [ ] Open PR in sibling repo; include this PACKAGING_REVIEW.md as the "due diligence" attachment alongside CODE_REVIEW.md.

Total estimated effort: **2-3 person-days**, of which Phase 1 (~half a day, mechanical) is the only true blocker; Phases 2-5 can be parallelized.

---

## Closing notes

- The core C++ shim is **architecturally clean and correct**; the four Criticals are all surface-level boundary/lifetime bugs, not design flaws. Once patched, the FFI layer is genuinely production-quality.
- The IR helper is mathematically sound (Higham §12.1) and its 1e-15 precision claim is empirically supported by the suite v1 results — but the test that proves it (`test_precision_vs_truth.py`) silently skips on the partner's machine, which means the partner has no way to verify the headline number without running the benchmark suite. Fix the fixture path issue (Phase 2) so the headline claim is enforced in CI.
- The benchmark suite (`bench/suite/`) is the strongest asset for a sibling-repo merge: it gives them a single command to reproduce all reported numbers on their hardware. Keep it.
- *Speculative:* if the partner repo uses JAX-CFD-style pytrees and `jit`-friendly state, consider adding (Phase 7, post-merge) a `Plan`-equivalent that holds the AMGx token in a `flax.struct.dataclass` so it survives `jax.tree_util` traversals cleanly. Not required for the initial drop.

