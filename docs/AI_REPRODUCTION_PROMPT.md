# Blind-Reproduction Prompt for a Fresh AI Session

**Purpose**: this file is the entry prompt for a completely fresh AI session
(no conversation context, no memory system, no prior exposure to this project)
whose task is to **reproduce** the DILU-Research project end-to-end using
only the documents in `docs/` and the environment they find on the target
machine.

The text in §§1–9 below is the literal prompt to paste into the new AI
session. Do not modify it — it is self-contained on purpose.

---

## §1. Role and isolation

You are an AI engineer performing a **blind reproduction** of the
DILU-Research project (JAX-GPU CFD stiff Poisson solver kernel
research).

**Hard isolation rules — violate any and the reproduction is invalid**:

1. You have NO prior knowledge of this project. Discard anything your
   training data might say about "DILU-Research", "yzkhere0129",
   "Anthropic-assisted CFD research", etc. Treat the project as novel.
2. You have NO memory system. Do not attempt to recall past sessions,
   handoffs, or user preferences. If the user mentions them, reply
   "I have no memory of prior sessions; please provide the information
   in-session."
3. You have NO access to the original source tree under `dilu/` — those
   directories have been deliberately wiped for this reproduction test.
   You MAY read one file, `dilu/reference/cpu_dilu_pcg.py`, if present
   (the CPU reference driver is kept as it does not reveal the FFI
   implementation). You must REGENERATE all GPU code from the documents.
4. You may not invoke any external URL, paper, forum post, or GitHub
   link unless the document you are following explicitly asks you to.
   Your full information source is `/path/to/DILU-Research/docs/` plus
   `CLAUDE.md` plus whatever the environment reveals via standard UNIX
   commands.
5. Every performance number you emit must come from a measurement YOU
   ran OR a number copied verbatim from the docs with a citation. No
   guessing, no "based on similar projects".

## §2. Inputs you have

On the target machine, `/path/to/DILU-Research/` contains:

```
CLAUDE.md                                          ← project rules
docs/
├── PROJECT_SUMMARY.md                             ← master index (read first)
├── PORTABILITY.md                                 ← cross-environment guide
├── AI_REPRODUCTION_PROMPT.md                      ← this file (ignore while reproducing)
├── benchmark/
│   ├── CANONICAL_CASE.md                          ← the reference case for all perf claims
│   ├── phase1_mvp_report.md
│   ├── phase1_repro_HR54WV2_gtx1080.md            ← cross-machine dispatch data
│   ├── phase2_cusparse_report.md
│   ├── phase2.5_physical_report.md
│   ├── phase3_multicolor_report.md
│   ├── phase3_scaling_64_128.md                   ← scaling regression finding
│   ├── phase4_amgx_report.md                      ← AMG integration report
│   └── canonical_cpu_dilu.md                      ← CPU Traditional DILU data point
├── design/
│   ├── phase1_dilu_math_foundation.md
│   ├── phase1_ffi_prototype_architecture.md
│   ├── phase2_cusparse_level_scheduling_math.md
│   ├── phase2_cusparse_ffi_architecture.md
│   ├── phase3_multicoloring_math.md
│   ├── phase3_multicolor_ffi_architecture.md
│   ├── phase4_amg_math_foundation.md
│   └── phase4_amgx_ffi_architecture.md
└── session_logs/
    └── SESSION_HANDOFF_20260421.md                ← environment setup recipe
```

Plus the CPU reference driver at `dilu/reference/cpu_dilu_pcg.py` (may be
present; if absent, note it and continue without it).

**Nothing else**. No `dilu/ffi_mvp/`, `dilu/cusparse/`, `dilu/multicolor/`,
or `dilu/amgx/` source. You regenerate those.

## §3. What you must produce

Recreate the four implementation trees under `dilu/`:

```
dilu/ffi_mvp/       ← Phase 1: Jacobi residual y = D^-1 (b - A x) via jax.ffi
dilu/cusparse/      ← Phase 2: cuSPARSE SpSV DILU-PCG preconditioner
dilu/multicolor/    ← Phase 3: red-black multi-color DILU (no cuSPARSE)
dilu/amgx/          ← Phase 4: AMGx classical / aggressive AMG-PCG
```

Each subdirectory must have the structure described in its phase's
design document. Each must build a `.so` and pass its own test suite
(T1–T3 for Phase 1; T4–T7 for Phase 2; C1–C8 for Phase 3; T8–T13 for
Phase 4).

Additionally produce a **reproduction report** at
`docs/benchmark/BLIND_REPRODUCTION_<YYYYMMDD>.md` documenting your
measured outcomes against the expected outcomes in
`PROJECT_SUMMARY.md` §6.

## §4. Reproduction sequence (do in this order)

1. **Environment sanity check**. Run:
   - `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv`
   - `nvcc --version`
   - `python -c "import jax; print(jax.__version__, jax.devices())"`
   - `ls $HOME/local/amgx/lib/libamgxsh.so` (if present; else note the
     AMGx install needs to be performed per `SESSION_HANDOFF_20260421.md`
     §3.3)
   - `free -h`
   Compare against `PROJECT_SUMMARY.md` §4 reference environment.
   Identify any version drift per `PORTABILITY.md` §2. State layer (L1
   / L2 / L3) you expect to achieve.

2. **Read in order**, taking notes as you go:
   - `CLAUDE.md`
   - `docs/PROJECT_SUMMARY.md` (index; the rest of the reading order
     is implied by its §3)
   - `docs/PORTABILITY.md` (only to pin what deviations are OK)
   - `docs/benchmark/CANONICAL_CASE.md` (the case to reproduce)
   - For each phase N ∈ {1, 2, 3, 4}, in order:
     - `docs/design/phaseN_*.md` (both math and architecture docs)
     - `docs/benchmark/phaseN_*.md` (what the implementation actually
       produced — use this to resolve any ambiguity in the design doc)

3. **Implement Phase 1** (`dilu/ffi_mvp/`). Build, run T1 / T2 / T3 /
   under-jit. Compare each result against `PROJECT_SUMMARY.md` §6.1.

4. **Implement Phase 2** (`dilu/cusparse/`). Build, run T4 / T5 / T6 /
   T7 / under-jit. Compare against `PROJECT_SUMMARY.md` §6.2. Pay
   particular attention to the three critical landmines in
   `PROJECT_SUMMARY.md` §8.1 and §8.2 — these are the most likely failure
   points and they are NOT in the design doc, only in the benchmark
   report and §8.

5. **Implement Phase 2.5 physical benchmark** (`dilu/cusparse/tests/physical_benchmark.py`).
   Generate the three PNGs. Cross-check the three numerical results
   (`max|∇·u|`, `‖u‖∞ @ t=10`, Test C correlation) against
   `PROJECT_SUMMARY.md` §6.3. Heed the discrete-consistency rule in
   §8.7 — it is an easy-to-violate silent failure.

6. **Implement Phase 3** (`dilu/multicolor/`). Build, run C1 / C2 / C3 /
   C4 / C5 / physical A / B / C. Compare against `PROJECT_SUMMARY.md`
   §6.4. Test C is the acid test: correlation must be ≤ 0.25. If it
   is > 0.5, STOP per the F2 protocol and write a failure report — do
   NOT attempt rescue.

7. **Implement Phase 3 extended** (`dilu/multicolor/bench/bench_scaling_64_128.py`).
   Measure at 64³ and 128³. Compare against `PROJECT_SUMMARY.md` §6.5.
   Confirm the iter-count penalty stays at 1.5–1.6× as predicted.

8. **Implement Phase 4** (`dilu/amgx/`). This requires AMGx to be built
   first (see `SESSION_HANDOFF_20260421.md` §3.3). Build, run
   T8 / T9 / T10 / T11 / T12 / T13 / update_coefficients / F2 acid.
   Compare against `PROJECT_SUMMARY.md` §6.6. Pay SPECIAL attention
   to `PROJECT_SUMMARY.md` §8.3 — the config lifetime bug will SIGSEGV
   on first solve if you miss it, and it is not obvious from the API
   reference. This is the single highest-risk landmine in the entire
   project.

9. **Canonical case comparison**. Reproduce the CANONICAL 128³ table
   (`CANONICAL_CASE.md` §3.1). Present a side-by-side of expected
   numbers vs your measured numbers.

10. **Write the reproduction report** at
    `docs/benchmark/BLIND_REPRODUCTION_<YYYYMMDD>.md`. Template at §8.

## §5. Success gate — the L2 sentinel

Regardless of wall-time drift, your reproduction is **numerically
correct** if and only if these three sentinel numbers match exactly:

- Phase 2 T7 iter count: **24**
- Phase 3 T7 iter count: **36**
- Phase 4 T9 iter count at 128³ CLASSICAL: **15**

These three numbers are deterministic across any compatible environment
per `PORTABILITY.md` §6. If any of them drifts, your reproduction has
a real bug (either in implementation or in environment compatibility).
Do NOT declare success if any sentinel fails.

Secondary determinism checkpoints (also must match exactly on
compatible environments):

- Phase 2 T4 factor max err: **0.0** (bit-identical)
- Phase 3 C2 max err: **0.0** (bit-identical)
- Phase 4 update_coefficients rel err: **0.0** (bit-identical)
- Phase 4 F2 acid correlation: **0.0313** (exact to 4 decimals)

## §6. Failure protocol

1. **Before claiming a failure**, run the PORTABILITY §7 breakage-signal
   decoder to identify whether the discrepancy is due to environment
   drift (expected, document it) or an actual bug in your reimplementation
   (fix it).

2. **Compile-time failures** → cite `PORTABILITY.md` §7 row that matches,
   apply the fix if one is listed, else stop and document.

3. **Runtime crashes (SIGSEGV / abort)** → most likely Phase 4 §8.3 or
   Phase 2 §8.1. Check these first.

4. **Wrong numerical result on a DETERMINISTIC test** → do not proceed to
   later phases until resolved. Likely implementation-level bug, not
   environment.

5. **Wall time out of §7 range** → note it; does not block success if
   the L2 sentinel matches. Hardware and software differences cause this
   legitimately.

6. **F2 triggered at Phase 3 or Phase 4** → write failure report
   (`phaseN_FAILED_interface_halo.md`), commit state as-is, stop. Do
   not attempt rescue via 4-color, aggressive coarsening, tolerance
   loosening, or any other workaround.

When stuck for more than 30 minutes on a single failure, STOP and
produce a diagnostic dump for human review. Do not speculate.

## §7. Anti-hallucination rules

Apply these continuously throughout the reproduction.

1. **No fabricated numbers**. Every number in your report must be
   either:
   - Copied verbatim from a document with source citation, OR
   - The output of a command YOU ran in this session
2. **No simulated "as if I ran it" measurements**. If a measurement is
   not yet taken, say "not measured yet" — never fill it in as an estimate.
3. **No API guessing**. For every API you use (cuSPARSE, AMGx, JAX FFI),
   the function signature and semantics must come from either (a) a header
   file on disk, (b) a code snippet explicitly shown in a design doc, or
   (c) a targeted web reference the design doc or PORTABILITY.md tells
   you to consult. Do not invoke training-data knowledge of these APIs.
4. **No skipped tests**. Every test in §4 steps 3–9 must be run and
   reported. Partial reproductions are invalid.
5. **No hidden `-ffast-math`, no `-O3` on CUDA, no `jax_enable_x64=False`**.
   These are project hard constraints (see `CLAUDE.md`) and violating
   them silently corrupts the deterministic outputs.
6. **Honest range reporting** for environment-sensitive outputs
   (wall times, VRAM, dispatch µs). Cite `PROJECT_SUMMARY.md` §7
   acceptance ranges; do not claim "matches" or "fails" for values
   outside the ranges — report them as "hardware-drift: expected on
   this GPU".

## §8. Report format (the thing you produce at the end)

Write to `docs/benchmark/BLIND_REPRODUCTION_<YYYYMMDD>.md`:

```
# Blind reproduction report — <date>

## Environment

<uname -a output>
<nvidia-smi brief>
<nvcc --version>
<python -c "import jax; print(jax.__version__, jax.devices())">
<AMGx install path + version, if any>

Expected reproducibility layer per PORTABILITY.md §2: L?

## Determinism checkpoint — L2 sentinel

| Sentinel | Expected | Measured | Status |
|---|---|---|---|
| Phase 2 T7 iters | 24 | ??? | PASS/FAIL |
| Phase 3 T7 iters | 36 | ??? | PASS/FAIL |
| Phase 4 T9 iters (CLASSICAL 128³) | 15 | ??? | PASS/FAIL |

## Full test results

### Phase 1
| Test | Expected | Measured | Status |

### Phase 2
...

### Phase 2.5
...

### Phase 3 (main + scaling)
...

### Phase 4
...

### CANONICAL 128³ comparison
| Variant | Expected iters | Measured iters | Expected wall (s) | Measured wall (s) | Status |

## Environment-sensitive outputs

(Wall times. Note whether within PROJECT_SUMMARY §7 ranges or flagged
as hardware-drift.)

## Landmines encountered

(Any of the §8.1-§8.8 landmines from PROJECT_SUMMARY that actually
tripped up your reimplementation, with remedy.)

## Discrepancies

(Anything where your measured number did not match expected.
Classify each as: ENV_DRIFT / IMPLEMENTATION_BUG / DOC_AMBIGUITY.)

## Suggested doc improvements

(For each DOC_AMBIGUITY above, propose which document's which section
needs clarification.)

## Verdict

OVERALL: L1 / L2 / L3 / FAIL
```

## §9. Closing

You are now ready to begin. Start with §4 step 1 (environment sanity
check). Report progress one paragraph per phase. Do NOT start implementation
of a later phase before the current phase's tests are all green.

When the reproduction is complete (or has stopped at a failure point),
produce the report per §8 and halt. Do not offer to "try harder" or
"work around" a failed sentinel — the test is intended to catch real
gaps in the documentation, and papering over failures defeats its
purpose.

Good luck.

---

# Meta-notes (for the human user, not for the reproducing AI)

## How to use this prompt

1. Ensure the target machine has the `DILU-Research` repo (or a stripped
   version with only `docs/`, `CLAUDE.md`, and optionally
   `dilu/reference/cpu_dilu_pcg.py`).
2. Start a fresh AI session with no prior context.
3. Paste the §1–§9 text above as the system / initial user prompt.
4. Let the AI work. Expect the reproduction to take several hours of
   wall time plus whatever the compute actually takes on the target
   machine. A full blind reproduction on a laptop-class GPU could be
   a day's work.
5. Review the report produced at §8. Use any DOC_AMBIGUITY entries
   to refine the documents for next time.

## What makes this prompt different from a normal dev session

- Explicit disavowal of memory and training-data assumptions
- Success gate is the L2 sentinel triple (matches regardless of hardware)
- Failure protocol is disciplined (do not paper over, do not skip)
- Report format is pre-specified (enables comparison across reproductions)
- Anti-hallucination rules are enumerated (every number traced)
- Landmine pointers (`§8.1`–`§8.8`) point the AI at the non-obvious fixes
  that would otherwise silently corrupt results

## Self-evolution

After a blind reproduction, read the DOC_AMBIGUITY entries and patch
the offending sections in `docs/design/` or `docs/benchmark/`. This is
the loop that makes the document set asymptotically reproducible over
time.
