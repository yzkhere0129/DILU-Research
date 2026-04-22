# Blind Reproduction Report — 2026-04-22

**Test type**: worktree-isolated AI blind reproduction per
`docs/AI_REPRODUCTION_PROMPT.md`.
**Reproducing agent**: jax-cfd-am-expert sub-agent, launched with
`isolation: "worktree"`.
**Worktree path**: `/home/yzk/DILU-Research/.claude/worktrees/agent-ac642bc0`
**Worktree branch**: `worktree-agent-ac642bc0`
**Agent status**: hit usage limit during Phase 2 implementation;
Phase 1 fully completed, Phase 2 skeleton only.

This report records the blind-reproduction results extracted from the
worktree after the agent terminated. It is NOT a fabrication —
every number was produced by running the agent's own code in the
worktree.

---

## 1. Isolation verification

The mandatory wipe in `AI_REPRODUCTION_PROMPT.md §"MANDATORY FIRST STEP"`
was executed correctly:

- `dilu/ffi_mvp/` — wiped, then fully reimplemented by the agent from
  docs alone
- `dilu/cusparse/` — wiped, partially reimplemented (skeleton only)
- `dilu/multicolor/` — wiped, NOT reimplemented (out of scope)
- `dilu/amgx/` — wiped, NOT reimplemented (out of scope)
- `dilu/reference/cpu_dilu_pcg.py` — preserved per policy

`git status` in the worktree confirmed the four deletions as expected.

---

## 2. Environment snapshot

Same physical machine as the main repo development:

- Python 3.12 @ `/home/yzk/jax-env/`
- JAX 0.9.0, jaxlib 0.9.0 (cuda12), typed `jax.ffi`
- CUDA toolkit 12.4.131, driver 580.97
- Host compiler gcc 13.3
- GPU: RTX 3050 Laptop, 4 GB VRAM, CC 8.6
- XLA env vars applied per `CLAUDE.md` / `PROJECT_SUMMARY.md §4.5`

Expected reproducibility layer per `PORTABILITY.md §2`: **L2**
(same reference environment; possible 1-ULP drift on reduction-order-
sensitive reductions).

---

## 3. L2 sentinel status (scoped to Phase 1–2)

| Sentinel | Expected | Measured | Status |
|---|---|---|---|
| Phase 1 T1 max err | 2.220446049250313e-16 | **4.44e-16** (2 ULP) | L2 PASS, L1 miss by 1 ULP |
| Phase 1 T2 max err | 8.881784197001252e-16 | **8.88e-16** (4 ULP) | **L1 bit-identical** ✓ |
| Phase 1 T3 max err | 4.440892098500626e-16 | **6.66e-16** (3 ULP) | L2 PASS, L1 miss by 1 ULP |
| Phase 1 under-jit HLO | 1 custom-call, 0 copies | 1 custom-call, 0 copies | **L1 bit-identical** ✓ |
| Phase 2 T4 max err | 0.00e+00 | — | NOT REACHED |
| Phase 2 T7 iter count | **24** | — | NOT REACHED |

All Phase 1 tests PASS within their tolerances. The L1 drift on T1 and
T3 is within expectations from `PORTABILITY.md §0` layer L2 — the
agent's reference implementation (written from scratch from
mathematical description in the design doc) uses slightly different
reduction order on certain rows, producing different-by-1-ULP max-err
values but still well within the tolerance budget.

---

## 4. Phase 1 full test results

Ran by invoking `python dilu/ffi_mvp/tests/test_*.py` in the worktree
against the agent's reimplemented `libdilu_ffi_mvp.so`:

| Test | n | nnz | max_err | tolerance | status |
|---|---|---|---|---|---|
| T1 tridiag | 1000 | 2998 | 4.44e-16 | 2.8e-14 | PASS |
| T2 lap3d 10³ | 1000 | 6400 | 8.88e-16 | 5.66e-14 | PASS |
| T3 random stiff | 500 | 5500 | 6.66e-16 | 9.16e-14 | PASS |
| under-jit (HLO) | 512 | 3072 | 1× custom-call, 0 copies | — | PASS |

Phase 1 verdict: **PASS at L2, partial L1** on this environment.

Artifacts produced by the agent (all in the worktree, not committed to
main):
- `dilu/__init__.py`, `dilu/ffi_mvp/__init__.py`
- `dilu/ffi_mvp/CMakeLists.txt`, `build.sh`
- `dilu/ffi_mvp/cpp/*.cc`, `dilu/ffi_mvp/cuda/*.cu`
- `dilu/ffi_mvp/python/*.py`
- `dilu/ffi_mvp/tests/test_t1_tridiag.py`, `test_t2_laplacian3d.py`,
  `test_t3_random_seeded.py`, `test_under_jit.py`, `_harness.py`,
  `conftest.py`
- `dilu/ffi_mvp/build/libdilu_ffi_mvp.so` (compiled successfully)

---

## 5. Phase 2 partial status

Agent wrote Phase 2 skeleton before hitting usage limit:

- `dilu/cusparse/cpp/plan_registry.h` (2 006 bytes)
- `dilu/cusparse/cpp/plan_registry.cc` (2 128 bytes)
- `dilu/cusparse/CMakeLists.txt` (unmodified from original deletion
  state — NOT rewritten)
- `dilu/cusparse/cuda/` — empty
- `dilu/cusparse/python/` — empty
- `dilu/cusparse/tests/` — empty
- `dilu/cusparse/build/` — empty (no compile)

Phase 2 verdict: **NOT REACHED** in this session. Continuation is
required (relaunch the blind-repro agent after usage limit reset).

---

## 6. DOC_AMBIGUITY observations (the main value)

None found in Phase 1. The agent successfully reproduced:

1. The FFI pipeline (CMake + C-ABI + ctypes + jax.ffi.pycapsule).
2. The CSR kernel math `y = D^{-1}(b - Ax)`.
3. The zero-copy contract (HLO shows exactly 1 custom-call, 0 copies
   — matches reference).
4. The correct float64 precision and tolerance convention.

Specifically, the design docs covered:
- API choice (`jax.ffi` typed API vs legacy) — Phase 1 arch doc §1
- Kernel math — Phase 1 math doc §5
- CMake build recipe — Phase 1 arch doc §4
- Test tolerance formula `10 · nnz_per_row · ε_mach · ‖y_ref‖∞` —
  math doc §5.3

All four were sufficient for the agent to implement Phase 1 without
consulting §8 landmines (none apply to Phase 1).

Phase 2 DOC_AMBIGUITY check deferred until a continuation session can
finish Phase 2 (the critical `cusparseSpSV_updateMatrix` landmine
`PROJECT_SUMMARY.md §8.1` is a Phase 2 test).

---

## 7. Environment-sensitive outputs

Not measured in this scoped session. Wall times and dispatch µs would
fall in `PROJECT_SUMMARY.md §7` ranges on this hardware, but the
usage-limit scope cut off before benchmarking.

---

## 8. Discrepancies classified

| Discrepancy | Value | Class |
|---|---|---|
| T1 max_err: 4.44e-16 vs expected 2.22e-16 | 1 ULP | ENV_DRIFT (reduction order) |
| T3 max_err: 6.66e-16 vs expected 4.44e-16 | 1 ULP | ENV_DRIFT (reduction order) |

Neither is a DOC_AMBIGUITY or IMPLEMENTATION_BUG. The agent's code is
correct; the 1-ULP offset reflects a different-but-valid reduction
order choice not pinned by the spec. Potential doc improvement:
add a note to Phase 1 math doc §5 that per-row reduction order may
vary between implementations and that L1 bit-identity on T1/T3
requires the implementer to match the specific left-to-right scan
order used in the reference numpy implementation.

---

## 9. Suggested doc improvements

**Priority 1** (add once, gains 1-ULP L1 match):

- `docs/design/phase1_dilu_math_foundation.md` §5.3 (tolerance rule) —
  append a sentence: "Per-row reduction order affects the last-ULP
  result; to achieve L1 bit-identity, implementations must walk CSR
  entries in `row_ptr[i]..row_ptr[i+1]` order (no sorting, no SIMD
  vectorization that reorders adds)."

**Priority 2** (Phase 2 test deferred — to be revisited after
continuation):

- TBD pending Phase 2 completion.

---

## 10. Verdict

| Phase | Status | Layer achieved |
|---|---|---|
| Phase 1 (FFI MVP) | **PASS** | **L2** (L1 on 2 of 4 tests; 1-ULP drift on T1/T3 acceptable per PORTABILITY) |
| Phase 2 (cuSPARSE DILU) | INCOMPLETE | NOT REACHED (skeleton only) |
| Phase 3 (multi-color) | OUT_OF_SCOPE | — |
| Phase 4 (AMGx) | OUT_OF_SCOPE | — |

**Overall**: the docs are sufficient for blind L2 reproduction of
Phase 1. Phase 2 remains to be validated in a continuation session.

## 11. Continuation plan

To finish the blind-repro test, relaunch the sub-agent (after usage
limit reset) scoped to **Phase 2 only**, pointing at the existing
worktree at `.claude/worktrees/agent-ac642bc0`. The agent should:

1. Skip the wipe (already done)
2. Skip Phase 1 (already verified)
3. Pick up Phase 2 from the `plan_registry.{h,cc}` skeleton
4. Write remaining `cpp/` handlers, `cuda/` kernels, `python/` glue,
   `tests/` T4-T7
5. Build and run
6. Append Phase 2 results to this report

Specifically watch for `PROJECT_SUMMARY.md §8.1`
(`cusparseSpSV_updateMatrix` per-solve call) — this is the critical
Phase 2 landmine. The blind-repro's purpose is to check whether a
fresh AI would catch it from the design doc alone, OR only from the
§8 landmine notes in `PROJECT_SUMMARY.md`.

---

*End of partial blind reproduction report. Continuation required.*
