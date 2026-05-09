LAST_REVIEWED: 2026-05-10T03:00+08:00
ITERATION: v1

# Adversary Notes — Pass 4 (against CLAIM_LEDGER) + Pass 6 (against XEON_PLAN)

I am a hostile JCP reviewer. Here are my attacks. Each is numbered A0XX and either:
- (a) revises a CLAIM, or
- (b) modifies the XEON_PLAN, or
- (c) adds a known-unknown.

---

## Pass 4 — attacks on CLAIM_LEDGER

### A001 — Top-5 worst AMGx-vs-LU rel cases are all dumper_pipeline 8K at very-short timesteps (4.4e-12 to 1.98e-11). Why?
Reading `precision_results_ir1.json`: max rel 2.89e-15 came from 8K matrices at picosecond timesteps. Larger 500K matrices would presumably have WORSE accumulation due to size, but we see 1.13e-11 max which is 4 orders of magnitude WORSE than 2.89e-15. The 4-order gap is consistent with 500K being harder, but still: maybe the picosecond-timestep matrices have some structural feature making them harder. Should add to E08: stratify error by case + timestep.
ACTION: Add to E08 analysis a "matrices ranked by max rel_diff vs LU" stratification.

### A002 — `‖x_OF - x_LU‖∞ = 5914 Pa` reproducibility
Tonight I re-ran SuperLU on dev and got 5914 Pa. Lab Xeon CHOLMOD also got 5914 Pa per conversation paste. That's 2 independent confirmations. But: is the bit-exact same x_LU returned by both? They should — A·x = b has unique solution (assuming non-singular). But due to ill-conditioning (κ ~ 1e14), different LU implementations may return slightly different x_LU. Need to verify dev SuperLU vs lab CHOLMOD produce same x_LU bit-for-bit OR same diff vs x_OF.
ACTION: E07 should compare SuperLU and CHOLMOD diff to detect divergence.

### A003 — AMGx tol=1e-12 is set as PCG stopping criterion, but actual achievable depends on matrix
The AMGx `Plan.solve` may not reach 1e-12 if max_iter caps. Check npz meta: `amgx_truth.iters`. If iters maxed out at 2000 (the cap I set), the rel_resid may be much worse than 1e-12 → IR step compensates but only for 1 step. Look at iter values: 559-1123 < 2000, so they did NOT cap — they converged before max_iter. Good. But: what if for 5000-step replay this happens differently? The cap may not be high enough.
ACTION: For E03/E04, set max_iter=5000 (vs 2000 currently). Add warning if iter > 2000.

### A004 — The 8K AMGx-vs-LU verification doesn't establish the SAME AMGx config works on 500K
`precision_results_ir1.json` used same CLASSICAL_V_DIAGSCALED + 1 IR. But matrix structure on 8K and 500K can differ enough that AMG hierarchy quality varies. The "AMGx algorithm correct" generalization (R04→C001→C004) is WEAK as flagged in EVIDENCE_CHAIN. E07 settles this directly.
ACTION: No action — already covered by E07.

### A005 — wall_seconds_method=perf_counter_ns may include CUDA async issues
For AMGx GPU solves, `plan.solve(...)` returns before GPU finishes (async launch). We use `x.block_until_ready()` which forces sync, but order is: solve() launch → block_until_ready() → record solve_ns. This is correct ONLY if block_until_ready waits exactly for the solve, not other GPU work.
ACTION: Add cuda.synchronize() (or jax-equivalent) before AND after the timed section to ensure clean isolation.

### A006 — Cold-cache only does sync + sleep 2s, not actual page-cache flush
True cold cache requires `echo 3 > /proc/sys/vm/drop_caches` (sudo). Without it, the second rep onwards may be hot. This biases later reps to be faster.
ACTION: In E02/E05 specifically, add a configurable cold-cache mode. If sudo is available, drop caches; else accept hot-cache and report.

### A007 — Single rep of E07 may have CHOLMOD non-determinism
CHOLMOD/SuperLU should be deterministic given identical input but BLAS multi-threading can introduce slight variation. With 1 rep we cannot assess variance.
ACTION: Add reps=2 for E07 sanity, compare if x_LU bit-exact.

### A008 — All experiments single-threaded
We don't pin threads. Lab Xeon may decide to multi-thread BLAS. Affects wall comparability across runs.
ACTION: Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` at the start of each E0X. (Or use a deliberate setting and record.)

### A009 — AMGx GPU memory may interact with Xeon CPU experiments
If E02-E04 hold the GPU and E05/E06/E07 (CPU CHOLMOD) start, no interference. But if E01 (laserMeltFoam) doesn't use GPU at all (single-core OF, CPU-only), parallelism is OK. But what if AMGx wrapper accidentally allocates GPU memory and doesn't release between reps?
ACTION: Each E0X using AMGx should call `gc.collect()` and (if jax has it) `jax.clear_caches()` between reps.

### A010 — Lanczos eigsh with tol=1e-3 has 1-2 order of magnitude noise
Especially σ_min via shift-invert can converge to spurious value if matrix has clusters near zero. The 4.5e-29 we got might be artifact (literally numerical noise floor). Better: use SVD on a sub-Krylov projection.
ACTION: For E09, run Lanczos with multiple seeds (5 reps with different random initial vectors). If results vary by > 100×, σ_min is poorly defined and we should report that as data, not converge to one number.

---

## Pass 6 — attacks on XEON_VALIDATION_PLAN

### A011 — E01 is most expensive (7h) but easiest to fail
laserMeltFoam re-run depends on (a) OF v2412 working, (b) the case dir being intact, (c) no numerical divergence in 1.2 μs of physics, (d) writing correct files. ANY of these failing wastes 7h. Mitigation: smoke test the OF run first.
ACTION: Add to E01 a `--smoke` mode that runs only 5 ns (5 timesteps), confirms log.run + solverInfo files are produced, then exits. User runs smoke first; if PASS, then commits to full 1.2μs.

### A012 — E03/E04 reps may interfere with each other
5 reps of E03 each run a 6-step sequence. Between reps, GPU/CPU caches affect timing. A06 covers this in part. Also: Plan() context manager release — in E03 we call `plan.__exit__(None, None, None)` to release between reps. Verify this actually frees GPU memory.
ACTION: Add `nvidia-smi --query-gpu=memory.used --format=csv` print before/after each rep in E03/E04.

### A013 — E08 doesn't compute amortization speedup ratio explicitly
E08 aggregates rows but doesn't compute the headline metric "AMGx amortized total wall / AMGx fresh total wall". Need to add this.
ACTION: Extend E08 to compute and print: 
- total_amortized = E03 mean total_wall_s
- total_fresh = E02 mean wall × 6
- ratio = total_fresh / total_amortized
- speedup_pct = (1 - 1/ratio) × 100

### A014 — predicted ranges in expected_results_template.json are too generous
Some ranges span 2-3 orders of magnitude (e.g., similarity 0.001-0.5). That's not really a prediction, more "anywhere in range". Tighten where defensible.
ACTION: For E03 amortized speedup, predicted_range was [1.0, 5.0]. Realistic given setup ~3s, solve ~10s, 6 steps: if amortized saves only 5×3s setup = 15s out of 60s, ratio = 60/45 ≈ 1.3. Tighten to [1.05, 2.5].

### A015 — No experiment measures the case where both AMGx amortized AND warm-start beat OF DICPCG
The headline question "does AMGx production beat OF?" needs ratio:
  total_AMGx_amortized_warm (E04 total)  vs  6 × OF_per_pd_solve_wall (from E01 × 6 timesteps)

Both numerators are measured, denominator from E01. Need to compute and report explicitly.
ACTION: E08 must compute `AMGx_E04_total_s / (6 × OF_E01_pd_wall_per_step)` and put in summary.

### A016 — Repository state during Xeon run may drift
Tonight committed audit dir (TBD). Tomorrow user pulls and runs. Between commit and run, someone may push other commits to main. The git_commit field in env captures the version at run time, but reproducibility requires referenced commit to be permanent.
ACTION: RUNBOOK §1 should `git log -1` to record commit before launch. expected_results_template.json should NOT match against a specific commit (it's about absolute values).

### A017 — Some experiments output may overwrite each other
E08 reads from results/E0X/*/result.json — but E08 itself outputs to results/E08/result.json. The folder structure is: `results/E08/result.json` (single file, not per-rep). That's fine, but the `manifest.json` mentioned in §6.2 of overnight prompt is missing from my plan.
ACTION: Add a `manifest.json` writer that lists all experiments + status for each.

### A018 — E01 cannot literally re-run laserMeltFoam from inside the Python script
My E01_runner.py prints "Action required: launch laserMeltFoam manually". This is correct (we don't want to spawn 7h subprocess from runner), but RUNBOOK should make it explicit.
ACTION: RUNBOOK §5 must specify: "If running E01, after `bash run_xeon_validation.sh --only=E01`, manually launch the OF run as instructed by E01's stdout."

### A019 — No version freeze of AMGx config across experiments
Different E0X scripts each call `with_tolerance(CLASSICAL_V_DIAGSCALED, ...)`. If we ever change the wrapper's default config, results not comparable across experiments. Each script should record the full config dict in result.json.
ACTION: Add `config_full_str` field that is `repr(cfg)` (complete config object representation) to each E0X result.json.

### A020 — No backup/checkpoint between experiments
If E03 produces partial results then crashes during E04, all of E03 sits in xeon_validation/results/E03/ but no aggregation has happened. If user kills the master script, data is lost from in-flight reps.
ACTION: Each runner already writes per-rep result.json atomically (.tmp + rename). Aggregation in E08 picks up partial data. Sufficient.

---

## Summary of action items (numbered for v2 plan)

| Action | Modifies |
|---|---|
| A001 | E08 — stratify by case+timestep |
| A002 | E07 — compare SuperLU vs CHOLMOD x_LU |
| A003 | E03/E04 — bump max_iter to 5000 |
| A005 | E02/E03/E04 — explicit cuda.synchronize() before/after timing |
| A006 | E02/E05 — sudo-aware cold-cache flush |
| A007 | E07 — add 2-rep variance check |
| A008 | All — pin OPENBLAS/OMP threads to 1, record value |
| A009 | E02/E03/E04 — gc.collect() + jax.clear_caches() between reps |
| A010 | E09 — multi-seed Lanczos for σ_min |
| A011 | E01 — add --smoke mode for 5ns run first |
| A012 | E03/E04 — print nvidia-smi between reps |
| A013 | E08 — compute amortization speedup ratio |
| A014 | expected_results_template.json — tighten ranges |
| A015 | E08 — compute final AMGx_warm / OF ratio |
| A016 | RUNBOOK — record git rev before run |
| A017 | run_xeon_validation.sh — write manifest.json |
| A018 | RUNBOOK — explicit instruction for E01 manual launch |
| A019 | All — record config_full_str |

These will be incorporated into v2 of XEON_VALIDATION_PLAN and the runner scripts.
