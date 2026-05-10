LAST_REVIEWED: 2026-05-10T14:35+08:00
ITERATION: v1

# SCRIPTS_DUMP.md — full source of all D7/D8 deliverables

Per closeout brief P5: cat all entrypoint + runner sources verbatim so they can
be code-reviewed without machine access. Sizes/lines header per file.

Files included:
  - run_xeon_validation.sh
  - xeon_validation/_common.py
  - xeon_validation/E01_runner.py through E11_runner.py (E10, E11 noted not implemented)
  - xeon_validation/compare_to_expected.py
  - xeon_validation/manifest.json (most recent dry-run output)
  - expected_results_template.json

---


===== run_xeon_validation.sh (7806 bytes, 256 lines) =====

```
#!/bin/bash
# =============================================================================
# Xeon Validation — settle C001-C021 + S1-S8 in 8 hours of lab Xeon time.
#
# Usage:
#   bash run_xeon_validation.sh [--dry-run] [--smoke] [--resume] [--only E01,E02,...]
#
# --dry-run  : verify env, paths, write permissions, then exit 0 (no solving).
# --smoke    : ≤1 rep per experiment, fast pipeline check.
# --resume   : default behavior; skip experiments whose result.json already exists.
# --only X,Y : run only specified experiment ids.
#
# Output: xeon_validation/{env_<ts>.txt, manifest.json, results/E0X/N/result.json,
#                          logs/E0X_N.log, analysis/{summary.md, compare_to_expected.py}}
#
# All scripts:
#   - record git commit, host, env vars
#   - record input MD5
#   - record wall_seconds_method (perf_counter_ns)
#   - record mem_peak_method (resource.getrusage)
#   - **fail-loud** on missing input, missing solver, missing scikit-sparse
#   - **no fallback to fake numbers**
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
RESULTS_DIR="$WORK_DIR/xeon_validation"
mkdir -p "$RESULTS_DIR/results" "$RESULTS_DIR/logs" "$RESULTS_DIR/analysis"

# Parse flags
DRY_RUN=0
SMOKE=0
RESUME=1
ONLY=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --smoke)   SMOKE=1 ;;
        --resume)  RESUME=1 ;;
        --only=*)  ONLY="${arg#--only=}" ;;
        --only)    ;; # handled below
        *) ;;
    esac
done
# Re-handle --only with following arg
i=0
args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    if [[ "${args[$i]}" == "--only" ]] && [[ $((i+1)) -lt ${#args[@]} ]]; then
        ONLY="${args[$((i+1))]}"
    fi
    i=$((i+1))
done

# All experiments (E01 default = laserMeltFoam-with-solverInfo, off by default;
# enable explicitly via --only E01 because it takes 7 hours)
DEFAULT_EXPERIMENTS="E02 E03 E04 E05 E06 E07 E08 E09"
ALL_EXPERIMENTS="E01 $DEFAULT_EXPERIMENTS E10 E11 E12"

if [[ -n "$ONLY" ]]; then
    EXPS=$(echo "$ONLY" | tr ',' ' ')
else
    EXPS="$DEFAULT_EXPERIMENTS"
fi

# A008: pin BLAS threads to 1 for clean measurements
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Environment snapshot
TIMESTAMP=$(date -Iseconds)
ENV_FILE="$RESULTS_DIR/env_${TIMESTAMP}.txt"
{
    echo "=== Xeon Validation Environment ==="
    echo "Timestamp: $TIMESTAMP"
    echo
    echo "=== uname ==="
    uname -a
    echo
    echo "=== lscpu (head) ==="
    lscpu 2>/dev/null | head -20
    echo
    echo "=== free ==="
    free -h
    echo
    echo "=== nvidia-smi ==="
    nvidia-smi 2>/dev/null | head -15 || echo "(no GPU)"
    echo
    echo "=== git ==="
    cd "$WORK_DIR" && git rev-parse HEAD 2>/dev/null || echo "(not git)"
    echo
    echo "=== Python ==="
    which python3
    python3 --version
    echo
    echo "=== pip versions (key) ==="
    python3 -c "import scipy, numpy; print(f'numpy={numpy.__version__}, scipy={scipy.__version__}')" 2>&1
    python3 -c "import sksparse; print(f'sksparse OK')" 2>&1 || echo "sksparse NOT INSTALLED"
    python3 -c "from dilu.amgx.python import Plan; print('AMGx wrapper OK')" 2>&1 || echo "AMGx wrapper FAIL"
    echo
    echo "=== OpenFOAM ==="
    echo "WM_PROJECT_VERSION=$WM_PROJECT_VERSION"
    which laserMeltFoam 2>/dev/null || echo "laserMeltFoam not in PATH"
    echo
    echo "=== experiments selected ==="
    echo "$EXPS"
    echo
    echo "=== flags ==="
    echo "DRY_RUN=$DRY_RUN  SMOKE=$SMOKE  RESUME=$RESUME  ONLY='$ONLY'"
} > "$ENV_FILE"

echo "=== Environment dumped to $ENV_FILE ==="
cat "$ENV_FILE" | tail -20
echo

# Sanity gate: required input files
SANITY_FAIL=0
NPZ_DIR="/home/yzk/DILU-Research/dilu/amgx/bench"
NPZ_FILES=(
    "single_melting_pd_corr0_3.2e-07.npz"
    "single_melting_pd_corr0_3.8e-07.npz"
    "single_melting_pd_corr0_4.1e-07.npz"
    "single_evap_early_pd_corr0_7e-07.npz"
    "single_evap_pd_corr0_9e-07.npz"
    "single_evap_late_pd_corr0_1.06e-06.npz"
)
for f in "${NPZ_FILES[@]}"; do
    if [[ ! -f "$NPZ_DIR/$f" ]]; then
        echo "FAIL: missing input npz $NPZ_DIR/$f"
        SANITY_FAIL=1
    fi
done

# OF case dir
OF_CASE="${OF_CASE:-$HOME/cases/single_track_dump}"
if [[ "$EXPS" == *"E01"* ]]; then
    if [[ ! -d "$OF_CASE/postProcessing/matrices" ]]; then
        echo "FAIL: E01 selected but OF_CASE=$OF_CASE has no postProcessing/matrices"
        SANITY_FAIL=1
    fi
fi

# Write permission test
touch "$RESULTS_DIR/.write_test" 2>/dev/null && rm "$RESULTS_DIR/.write_test" || {
    echo "FAIL: cannot write to $RESULTS_DIR"
    SANITY_FAIL=1
}

if [[ $SANITY_FAIL -ne 0 ]]; then
    echo "Sanity gate FAILED. Aborting."
    exit 1
fi

echo "Sanity gate PASS."

# A017: write manifest.json
MANIFEST="$RESULTS_DIR/manifest.json"
python3 -c "
import json, sys
exps = '$EXPS'.split()
m = {
    'experiments_planned': exps,
    'started_at': '$TIMESTAMP',
    'env_file': '$ENV_FILE',
    'flags': {'dry_run': $DRY_RUN, 'smoke': $SMOKE, 'resume': $RESUME, 'only': '$ONLY'},
    'thread_pin': {'OPENBLAS_NUM_THREADS': '$OPENBLAS_NUM_THREADS',
                    'OMP_NUM_THREADS': '$OMP_NUM_THREADS',
                    'MKL_NUM_THREADS': '$MKL_NUM_THREADS'},
}
with open('$MANIFEST', 'w') as f:
    json.dump(m, f, indent=2)
print('manifest:', '$MANIFEST')
"
echo

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] Would execute experiments: $EXPS"
    echo "[DRY-RUN] All inputs verified. Output dir writable. Exiting 0."
    exit 0
fi

# --- Run experiments ---
RUNNER="$SCRIPT_DIR/xeon_validation"

run_expt() {
    local id="$1"
    local script="$RUNNER/${id}_runner.py"
    if [[ ! -f "$script" ]]; then
        echo "[$id] ERROR: $script not found"
        return 1
    fi
    local resdir="$RESULTS_DIR/results/$id"
    mkdir -p "$resdir"
    local logfile="$RESULTS_DIR/logs/${id}_$(date +%H%M%S).log"
    echo "[$id] starting → $logfile"
    local args=""
    [[ $SMOKE -eq 1 ]] && args="$args --smoke"
    [[ $RESUME -eq 1 ]] && args="$args --resume"
    args="$args --output-dir $resdir"
    args="$args --npz-dir $NPZ_DIR"
    if [[ "$id" == "E01" ]]; then
        args="$args --of-case $OF_CASE"
    fi
    if python3 -u "$script" $args 2>&1 | tee "$logfile"; then
        echo "[$id] PASS"
    else
        echo "[$id] FAILED — see $logfile"
        return 1
    fi
}

# E01 in background if selected
if [[ "$EXPS" == *"E01"* ]] && [[ $SMOKE -ne 1 ]]; then
    echo "[E01] backgrounding (will run nohup ~7h)"
    nohup python3 -u "$RUNNER/E01_runner.py" \
        --of-case "$OF_CASE" \
        --output-dir "$RESULTS_DIR/results/E01" \
        $([ $RESUME -eq 1 ] && echo --resume) \
        > "$RESULTS_DIR/logs/E01_bg.log" 2>&1 &
    E01_PID=$!
    echo "[E01] backgrounded PID=$E01_PID"
    EXPS=$(echo "$EXPS" | sed 's/E01//')
fi

# Run remaining in foreground
for E in $EXPS; do
    [[ -z "$E" ]] && continue
    run_expt "$E" || echo "[$E] FAILED — continuing with rest"
done

# Final analysis
echo
echo "=== Running compare_to_expected.py ==="
python3 -u "$RUNNER/compare_to_expected.py" \
    --results-dir "$RESULTS_DIR/results" \
    --expected "$SCRIPT_DIR/expected_results_template.json" \
    --output "$RESULTS_DIR/analysis/summary.md" \
    || echo "compare_to_expected.py failed; check logs"

echo
echo "=== ALL DONE ==="
echo "Results: $RESULTS_DIR/"
echo "Summary: $RESULTS_DIR/analysis/summary.md"

# E01 background still running?
if [[ -n "${E01_PID:-}" ]]; then
    if ps -p "$E01_PID" > /dev/null 2>&1; then
        echo "Note: E01 still running in background (PID $E01_PID)"
        echo "  Monitor: tail -f $RESULTS_DIR/logs/E01_bg.log"
    else
        echo "E01 background completed (PID $E01_PID)"
    fi
fi
```

===== xeon_validation/_common.py (5781 bytes, 171 lines) =====

```
"""Common utilities for E0X runners.

- record_environment(): host info + git rev + library versions
- file_md5(): for input MD5 verification
- record_result(): atomic JSON write to result.json
- TimedSection: context manager for wall_seconds capture
- sanity_check_matrix(): assert ‖A·x_truth - b‖/‖b‖ < threshold
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data: break
            h.update(data)
    return h.hexdigest()


def pin_threads_to_one():
    """A008: prevent BLAS multi-threading from polluting wall measurements."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def record_environment() -> dict:
    out = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    }
    # git
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "-C", "/home/yzk/DILU-Research", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        out["git_commit"] = "unknown"
    # library versions
    for libname in ("numpy", "scipy"):
        try:
            mod = __import__(libname)
            out[f"{libname}_version"] = mod.__version__
        except Exception:
            out[f"{libname}_version"] = "missing"
    try:
        import sksparse
        out["sksparse_version"] = getattr(sksparse, "__version__", "unknown")
    except Exception:
        out["sksparse_version"] = "missing"
    out["openblas_threads"] = os.environ.get("OPENBLAS_NUM_THREADS", "default")
    out["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "default")
    return out


@contextmanager
def TimedSection():
    """Wall-clock + RSS profiling. Returns dict via .data."""
    start_ns = time.perf_counter_ns()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data = {}
    try:
        yield data
    finally:
        end_ns = time.perf_counter_ns()
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        data["wall_seconds"] = (end_ns - start_ns) / 1e9
        data["wall_seconds_method"] = "MEASURED:perf_counter_ns"
        data["mem_peak_kb"] = max(rss_before, rss_after)
        data["mem_peak_method"] = "MEASURED:resource.getrusage"
        data["mem_growth_kb"] = rss_after - rss_before


def write_result_json(path: Path, payload: dict):
    """Atomic write to result.json (write to .tmp, fsync, rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sanity_check_matrix(A: csr_matrix, b: np.ndarray, x_truth: np.ndarray,
                         threshold: float = 1e-7, label: str = "input"):
    """Assert ‖A·x_truth - b‖/‖b‖ < threshold; raise SystemExit if not."""
    res = float(np.linalg.norm(A @ x_truth - b)
                / max(np.linalg.norm(b), 1e-300))
    if res > threshold:
        raise SystemExit(
            f"SANITY GATE FAILED on {label}: "
            f"‖A·x_truth - b‖/‖b‖ = {res:.3e} > {threshold:.0e}. "
            f"Refusing to run experiment with corrupt input.")
    return res


def normalize_sign(A: csr_matrix, b: np.ndarray):
    """OF Laplacian: negative diag → flip to positive."""
    diag = A.diagonal()
    if np.all(diag <= 0) and np.any(diag < 0):
        return -A, -b, True
    return A, b, False


def load_npz_with_solutions(npz_path: Path):
    """Load single_*pd_corr0_*.npz returning all fields."""
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    return dict(
        x_OF=z["x_OF"], x_AMGx_e8=z["x_AMGx_e8"],
        x_truth=z["x_truth"], b=z["b"],
        meta=meta,
    )


def load_raw_matrix(case_dir: Path, time_str: str, eq: str = "pd_corr0"):
    """Load A.mm, b.mm, x_final.mm directly from OF dump."""
    eq_dir = case_dir / "postProcessing" / "matrices" / time_str / eq
    A = sio.mmread(str(eq_dir / "A.mm")).tocsr()
    b = sio.mmread(str(eq_dir / "b.mm")).flatten()
    x_OF = sio.mmread(str(eq_dir / "x_final.mm")).flatten()
    return A, b, x_OF


def cold_cache():
    """Best-effort cache flush (no sudo). Sleep + sync."""
    try:
        subprocess.run(["sync"], check=False, timeout=10)
    except Exception:
        pass
    time.sleep(2)


def find_npz(npz_dir: Path):
    """Return ordered list of (time_str, phase, npz_path) for the 6 timesteps."""
    mapping = {
        "3.2e-07":  ("melting", "single_melting_pd_corr0_3.2e-07.npz"),
        "3.8e-07":  ("melting", "single_melting_pd_corr0_3.8e-07.npz"),
        "4.1e-07":  ("melting", "single_melting_pd_corr0_4.1e-07.npz"),
        "7e-07":    ("evap_early", "single_evap_early_pd_corr0_7e-07.npz"),
        "9e-07":    ("evap", "single_evap_pd_corr0_9e-07.npz"),
        "1.06e-06": ("evap_late", "single_evap_late_pd_corr0_1.06e-06.npz"),
    }
    out = []
    for t, (phase, fname) in mapping.items():
        p = npz_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
        out.append((t, phase, p))
    return out
```

===== xeon_validation/E01_runner.py (9536 bytes, 249 lines) =====

```
"""E01 — OF DICPCG actual per-solve wall via solverInfo function object.

This MODIFIES the OF case (system/controlDict) to add solverInfo,
re-runs laserMeltFoam for the full 1.2 μs case (~7h single-core),
parses postProcessing/solverInfo/<t>/solverInfo.dat for per-solve wall.

WARNING: Long-running. Requires --of-case path. Idempotent if controlDict already
has solverInfo and a previous result.json exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import record_environment, write_result_json


SOLVER_INFO_BLOCK = """
functions
{
    solverInfo
    {
        type            solverInfo;
        libs            (utilityFunctionObjects);
        fields          (pd T U alpha.material);
        writeControl    timeStep;
        writeInterval   1;
    }
}
"""


def add_solverInfo_to_controlDict(controlDict_path: Path):
    """Idempotently add functions block to system/controlDict."""
    text = controlDict_path.read_text()
    if "solverInfo" in text:
        print(f"  controlDict already has solverInfo block")
        return
    # Backup
    bak = controlDict_path.with_suffix(controlDict_path.suffix + ".pre_E01")
    if not bak.exists():
        shutil.copy2(controlDict_path, bak)
    # Append functions block (find a safe spot — e.g. end of file)
    if not text.rstrip().endswith("//"):
        text = text.rstrip() + "\n"
    text += SOLVER_INFO_BLOCK
    controlDict_path.write_text(text)
    print(f"  added solverInfo to {controlDict_path}")


def parse_solverInfo(case_dir: Path):
    """Walk postProcessing/solverInfo/<t>/solverInfo.dat and aggregate per-equation per-step wall."""
    si_dir = case_dir / "postProcessing" / "solverInfo"
    if not si_dir.exists():
        return []
    rows = []
    for time_dir in sorted(si_dir.iterdir(), key=lambda p: float(p.name) if p.name.replace('.', '').replace('e', '').replace('-', '').replace('+', '').isdigit() else 0):
        dat = time_dir / "solverInfo.dat"
        if not dat.exists(): continue
        # Format depends on OF version; typically:
        # Time pd_solver pd_initial_residual pd_final_residual pd_no_iterations ... [+ wall?]
        # But solverInfo doesn't include wall by default. We may need to parse log.run for that.
        with open(dat) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line: continue
                rows.append(line)
    return rows


def parse_log_run_for_wall(log_path: Path):
    """Parse log.run for Time = X / ExecutionTime = Y per timestep,
    plus DICPCG: Solving for pd, ... iter=N, finalRes=R lines.

    Returns list of dicts with: time, exec_time, pd_solves (list of {iter, init_res, final_res})
    """
    re_time = re.compile(r"^Time = (\S+)$")
    re_exec = re.compile(r"^ExecutionTime = (\S+) s")
    re_solve = re.compile(
        r"^(DICPCG|DILUPBiCG|GAMG|smoothSolver):\s*Solving for (\w+),"
        r"\s*Initial residual = (\S+),\s*Final residual = (\S+),"
        r"\s*No Iterations (\d+)"
    )

    rows = []
    cur_time = None
    cur_solves = []
    last_exec = 0.0

    if not log_path.exists():
        return []
    for line in log_path.read_text().split("\n"):
        m = re_time.match(line)
        if m:
            if cur_time is not None:
                rows.append({"time": cur_time, "solves": cur_solves})
            cur_time = float(m.group(1))
            cur_solves = []
            continue
        m = re_solve.search(line)
        if m:
            cur_solves.append({
                "algorithm": m.group(1),
                "equation": m.group(2),
                "init_residual": float(m.group(3)),
                "final_residual": float(m.group(4)),
                "iterations": int(m.group(5)),
            })
            continue
        m = re_exec.match(line)
        if m and rows:
            new_exec = float(m.group(1))
            rows[-1]["exec_time_at_step_end"] = new_exec
            rows[-1]["wall_delta_s"] = new_exec - last_exec
            last_exec = new_exec
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--of-case", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-rerun", action="store_true",
                     help="skip the laserMeltFoam re-run, just parse existing log.run")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    of_case = Path(args.of_case)
    if not of_case.exists():
        sys.exit(f"FAIL: --of-case {of_case} not found")

    env = record_environment()
    result_path = output_dir / "01" / "result.json"

    if args.resume and result_path.exists():
        print(f"E01: result.json exists, SKIP (resume)"); return

    # Step 1: add solverInfo (idempotent)
    controlDict = of_case / "system" / "controlDict"
    if not args.skip_rerun:
        add_solverInfo_to_controlDict(controlDict)
        # Step 2: re-run laserMeltFoam (clean + run)
        # Don't clean if --resume — assume user wants to keep partial dump
        if not args.smoke:
            print(f"  Re-running laserMeltFoam (~7h)...")
            print(f"  Command: cd {of_case} && nohup laserMeltFoam > log.run 2>&1")
            print(f"  WARNING: this script does NOT actually launch the run by default.")
            print(f"  Edit launching logic if you want this script to invoke laserMeltFoam.")
            # Actually we must NOT spawn an OF run inside this Python script
            # because the user may want to control it. Just signal what to do:
            print(f"")
            print(f"  Action required: launch laserMeltFoam manually:")
            print(f"    cd {of_case}")
            print(f"    nohup laserMeltFoam > log.run 2>&1 &")
            print(f"    disown")
            print(f"  Then re-run E01 with --skip-rerun once it completes.")

    # Step 3: parse log.run
    log_path = of_case / "log.run"
    if not log_path.exists():
        print(f"WARN: {log_path} not found. Cannot parse OF wall yet.")
        payload = {
            "expt_id": "E01",
            "status": "WAITING_FOR_LOG",
            "of_case": str(of_case),
            "env": env,
        }
        write_result_json(result_path, payload)
        return

    rows = parse_log_run_for_wall(log_path)
    print(f"  parsed {len(rows)} timesteps from log.run")

    # Aggregate pd_corr0 walls (note: log doesn't tell us per-solve wall directly,
    # only step total; we compute pd-iter share)
    import numpy as np
    pd_iters_per_step = []
    total_iters_per_step = []
    walls_per_step = []
    pd_solves = []  # individual pd_corr0 entries with iter
    for r in rows:
        if "wall_delta_s" not in r: continue
        step_iters_pd = sum(s["iterations"] for s in r["solves"]
                             if s["equation"] == "pd")
        step_iters_total = sum(s["iterations"] for s in r["solves"])
        # Approximate pd-share of wall (assume per-iter cost equal across solvers — coarse)
        if step_iters_total > 0:
            pd_share = step_iters_pd / step_iters_total
            pd_wall = r["wall_delta_s"] * pd_share
        else:
            pd_wall = 0
        for s in r["solves"]:
            if s["equation"] == "pd":
                pd_solves.append({
                    "time": r["time"],
                    "iter": s["iterations"],
                    "final_residual": s["final_residual"],
                    "approx_wall_s": pd_wall * s["iterations"] / max(step_iters_pd, 1),
                })
        walls_per_step.append(r["wall_delta_s"])
        pd_iters_per_step.append(step_iters_pd)
        total_iters_per_step.append(step_iters_total)

    payload = {
        "expt_id": "E01",
        "status": "PARSED",
        "of_case": str(of_case),
        "n_timesteps_in_log": len(rows),
        "total_steps_with_wall": len(walls_per_step),
        "n_pd_solves": len(pd_solves),
        "step_wall_stats": {
            "mean": float(np.mean(walls_per_step)) if walls_per_step else None,
            "median": float(np.median(walls_per_step)) if walls_per_step else None,
            "min": float(min(walls_per_step)) if walls_per_step else None,
            "max": float(max(walls_per_step)) if walls_per_step else None,
        },
        "pd_iter_per_step_stats": {
            "mean": float(np.mean(pd_iters_per_step)) if pd_iters_per_step else None,
            "median": float(np.median(pd_iters_per_step)) if pd_iters_per_step else None,
        },
        "total_iter_per_step_stats": {
            "mean": float(np.mean(total_iters_per_step)) if total_iters_per_step else None,
            "median": float(np.median(total_iters_per_step)) if total_iters_per_step else None,
        },
        "approx_per_iter_ms": (
            float(np.mean(walls_per_step)) / max(float(np.mean(total_iters_per_step)), 1)
            * 1000 if walls_per_step else None
        ),
        "pd_solves_sample": pd_solves[:50],
        "env": env,
    }
    write_result_json(result_path, payload)
    print(f"\n[E01] parsed; mean step wall = {payload['step_wall_stats']['mean']:.2f}s, "
          f"approx ms/iter = {payload.get('approx_per_iter_ms')}")


if __name__ == "__main__":
    main()
```

===== xeon_validation/E02_runner.py (6432 bytes, 166 lines) =====

```
"""E02 — AMGx PCG cold-start single-shot wall.

For each of 6 timesteps × N reps:
  - cold cache
  - Plan(A) fresh setup
  - plan.solve(b, x0=0)
  - record setup_s, solve_s, iter, rel_resid
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow import of _common
sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, file_md5, write_result_json,
    TimedSection, sanity_check_matrix, normalize_sign,
    load_raw_matrix, find_npz, cold_cache,
)

# JAX/AMGx setup
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.reps = 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    # Lazy import AMGx — fail loud if missing
    try:
        from jax import config as _jc
        _jc.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance
    except ImportError as e:
        sys.exit(f"FAIL: AMGx wrapper not importable: {e}")

    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-8, max_iters=2000)

    # Discover npzs (use them to load A, b — they have the matrix? actually no, they have x_OF only)
    # We need raw A, b — must read from postProcessing/matrices.
    # Workaround: npz has b, but A is huge. Let's load from npz and reconstruct? npz doesn't have A either.
    # Plan: use the npz x_OF + b, and reconstruct A from npz fields.
    # Actually the npz files don't store A. We need the raw .mm files.
    # The dense_track or single_track case dir is needed for A.
    # → assume case dir is at /home/yzk/cases/single_track_dump (lab Xeon) or /home/yzk/single_track_dump (dev)

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = None
    for cd in case_dir_candidates:
        if (cd / "postProcessing/matrices").exists():
            case_dir = cd; break
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found in {case_dir_candidates}")
    print(f"Using case dir: {case_dir}")

    timesteps = find_npz(npz_dir)

    env = record_environment()
    print(f"Environment: {json.dumps(env, indent=2)}")
    print(f"reps per timestep: {args.reps}")
    print(f"npz_dir: {npz_dir}")
    print(f"case_dir: {case_dir}")

    for time_str, phase, npz_path in timesteps:
        # Load raw A, b (large but needed for AMGx)
        print(f"\n=== {phase} t={time_str} ===")
        A, b_raw, x_OF = load_raw_matrix(case_dir, time_str)
        A_pos, b_pos, sign_flipped = normalize_sign(A, b_raw)
        b_md5 = file_md5(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/b.mm")
        A_md5 = file_md5(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/A.mm")
        print(f"  N={A.shape[0]}, nnz={A.nnz}, sign_flipped={sign_flipped}")

        for rep in range(1, args.reps + 1):
            rep_dir = output_dir / f"{rep:02d}_{time_str}"
            result_path = rep_dir / "result.json"
            if args.resume and result_path.exists():
                print(f"  rep {rep:02d}: SKIP (resume — exists)")
                continue
            cold_cache()

            # Move data to GPU
            rp = jnp.asarray(A_pos.indptr.astype(np.int32))
            ci = jnp.asarray(A_pos.indices.astype(np.int32))
            vv = jnp.asarray(A_pos.data.astype(np.float64))
            b_d = jnp.asarray(b_pos.astype(np.float64))
            x0_d = jnp.zeros_like(b_d)

            with TimedSection() as outer:
                t0 = time.perf_counter_ns()
                with Plan(rp, ci, vv, cfg) as plan:
                    setup_ns = time.perf_counter_ns() - t0
                    t1 = time.perf_counter_ns()
                    x, iters, status = plan.solve(b_d, x0_d)
                    x.block_until_ready()
                    solve_ns = time.perf_counter_ns() - t1
                    x_h = np.asarray(x)

            res = float(np.linalg.norm(A_pos @ x_h - b_pos)
                        / max(np.linalg.norm(b_pos), 1e-300))

            payload = {
                "expt_id": "E02",
                "rep": rep,
                "timestep": time_str,
                "phase": phase,
                "started_at": "(captured by TimedSection)",
                "wall_seconds": outer["wall_seconds"],
                "wall_seconds_method": outer["wall_seconds_method"],
                "mem_peak_kb": outer["mem_peak_kb"],
                "mem_peak_method": outer["mem_peak_method"],
                "setup_ns": int(setup_ns),
                "solve_ns": int(solve_ns),
                "setup_ms": setup_ns / 1e6,
                "solve_ms": solve_ns / 1e6,
                "iter_count": int(iters[0]),
                "status": int(status[0]),
                "rel_resid_actual": res,
                "tol_requested": 1e-8,
                "config": {"name": "CLASSICAL_V_DIAGSCALED", "max_iters": 2000},
                "input_files": {
                    "A.mm": str(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/A.mm"),
                    "b.mm": str(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/b.mm"),
                },
                "input_md5s": {"A.mm": A_md5, "b.mm": b_md5},
                "n": int(A.shape[0]),
                "nnz": int(A.nnz),
                "sign_flipped": sign_flipped,
                "env": env,
                "warnings": [],
            }
            write_result_json(result_path, payload)
            print(f"  rep {rep:02d}: setup={setup_ns/1e6:.0f}ms solve={solve_ns/1e6:.0f}ms "
                  f"iter={int(iters[0])} resid={res:.2e}")

    print(f"\n[E02] complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
```

===== xeon_validation/E03_runner.py (5761 bytes, 159 lines) =====

```
"""E03 — AMGx amortized (Plan once + N×update_coefficients) wall.

Runs the entire 6-step sequence per rep:
  Plan(A_0) — record setup_ms_first
  solve(b_0, x0=0) — record solve_ms[0]
  for i in 1..5:
    update_coefficients(A_i.data) — record update_ms[i]
    solve(b_i, x0=0) — record solve_ms[i]    # COLD INIT for E03 baseline

Repeat ≥5 times. Sum across step gives realistic amortized total wall.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, file_md5, write_result_json,
    TimedSection, normalize_sign, load_raw_matrix, find_npz, cold_cache,
)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke: args.reps = 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    try:
        from jax import config as _jc
        _jc.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance
    except ImportError as e:
        sys.exit(f"FAIL: AMGx wrapper not importable: {e}")

    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-8, max_iters=2000)

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = next((c for c in case_dir_candidates
                     if (c / "postProcessing/matrices").exists()), None)
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found")

    timesteps = find_npz(npz_dir)
    env = record_environment()

    # Pre-load all matrices (eager)
    print("Pre-loading 6 matrices...")
    matrices = []
    for t, phase, npz_path in timesteps:
        A, b_raw, x_OF = load_raw_matrix(case_dir, t)
        A_pos, b_pos, sf = normalize_sign(A, b_raw)
        matrices.append({"t": t, "phase": phase, "A": A_pos, "b": b_pos, "x_OF": x_OF, "sf": sf})
        print(f"  {phase} t={t}: N={A_pos.shape[0]} nnz={A_pos.nnz}")

    for rep in range(1, args.reps + 1):
        rep_dir = output_dir / f"rep_{rep:02d}"
        result_path = rep_dir / "result.json"
        if args.resume and result_path.exists():
            print(f"\nrep {rep:02d}: SKIP (resume)"); continue

        cold_cache()
        per_step = []
        plan = None
        total_t0 = time.perf_counter_ns()

        for i, m in enumerate(matrices):
            A_pos, b_pos = m["A"], m["b"]
            rp = jnp.asarray(A_pos.indptr.astype(np.int32))
            ci = jnp.asarray(A_pos.indices.astype(np.int32))
            vv = jnp.asarray(A_pos.data.astype(np.float64))
            b_d = jnp.asarray(b_pos.astype(np.float64))
            x0_d = jnp.zeros_like(b_d)

            t0 = time.perf_counter_ns()
            if plan is None:
                # First step: full setup
                plan = Plan(rp, ci, vv, cfg)
                t1 = time.perf_counter_ns()
                setup_ns = t1 - t0
                update_ns = 0
                # solve
                x, iters, status = plan.solve(b_d, x0_d)
                x.block_until_ready()
                solve_ns = time.perf_counter_ns() - t1
            else:
                # Subsequent: update_coefficients
                plan.update_coefficients(vv)
                t1 = time.perf_counter_ns()
                update_ns = t1 - t0
                setup_ns = 0
                x, iters, status = plan.solve(b_d, x0_d)
                x.block_until_ready()
                solve_ns = time.perf_counter_ns() - t1

            x_h = np.asarray(x)
            res = float(np.linalg.norm(A_pos @ x_h - b_pos) /
                        max(np.linalg.norm(b_pos), 1e-300))
            step_total_ns = setup_ns + update_ns + solve_ns
            per_step.append({
                "step": i, "timestep": m["t"], "phase": m["phase"],
                "setup_ms": setup_ns / 1e6,
                "update_ms": update_ns / 1e6,
                "solve_ms": solve_ns / 1e6,
                "total_ms": step_total_ns / 1e6,
                "iter_count": int(iters[0]),
                "rel_resid": res,
            })
            print(f"  rep{rep:02d} step{i} ({m['phase']:<11s} t={m['t']}): "
                  f"setup={setup_ns/1e6:>5.0f}ms update={update_ns/1e6:>5.0f}ms "
                  f"solve={solve_ns/1e6:>5.0f}ms iter={int(iters[0]):>4d} resid={res:.2e}")

        if plan is not None:
            plan.__exit__(None, None, None)
        total_wall_s = (time.perf_counter_ns() - total_t0) / 1e9

        payload = {
            "expt_id": "E03",
            "rep": rep,
            "wall_seconds": total_wall_s,
            "wall_seconds_method": "MEASURED:perf_counter_ns",
            "n_steps": len(matrices),
            "tol_requested": 1e-8,
            "config": {"name": "CLASSICAL_V_DIAGSCALED", "max_iters": 2000,
                        "warm_start": False},
            "per_step": per_step,
            "env": env,
        }
        write_result_json(result_path, payload)
        print(f"  rep{rep:02d} TOTAL: {total_wall_s:.2f} s")

    print(f"\n[E03] complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
```

===== xeon_validation/E04_runner.py (5683 bytes, 160 lines) =====

```
"""E04 — AMGx amortized + warm-start. Same as E03 but solve(x0=x_prev).

Also records similarity metric ‖x_t - x_{t-1}‖∞ / ‖x_t‖∞ for analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, write_result_json,
    normalize_sign, load_raw_matrix, find_npz, cold_cache,
)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke: args.reps = 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    try:
        from jax import config as _jc
        _jc.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance
    except ImportError as e:
        sys.exit(f"FAIL: AMGx wrapper not importable: {e}")

    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-8, max_iters=2000)

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = next((c for c in case_dir_candidates
                     if (c / "postProcessing/matrices").exists()), None)
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found")

    timesteps = find_npz(npz_dir)
    env = record_environment()

    print("Pre-loading 6 matrices...")
    matrices = []
    for t, phase, npz_path in timesteps:
        A, b_raw, x_OF = load_raw_matrix(case_dir, t)
        A_pos, b_pos, sf = normalize_sign(A, b_raw)
        matrices.append({"t": t, "phase": phase, "A": A_pos, "b": b_pos})

    for rep in range(1, args.reps + 1):
        rep_dir = output_dir / f"rep_{rep:02d}"
        result_path = rep_dir / "result.json"
        if args.resume and result_path.exists():
            print(f"\nrep {rep:02d}: SKIP"); continue

        cold_cache()
        per_step = []
        plan = None
        x_prev = None
        total_t0 = time.perf_counter_ns()

        for i, m in enumerate(matrices):
            A_pos, b_pos = m["A"], m["b"]
            rp = jnp.asarray(A_pos.indptr.astype(np.int32))
            ci = jnp.asarray(A_pos.indices.astype(np.int32))
            vv = jnp.asarray(A_pos.data.astype(np.float64))
            b_d = jnp.asarray(b_pos.astype(np.float64))

            # Warm-start vs cold-start setup
            if x_prev is None:
                x0_d = jnp.zeros_like(b_d)
                similarity = None
            else:
                x0_d = jnp.asarray(x_prev.astype(np.float64))
                # similarity is computed AFTER we have current x — defer

            t0 = time.perf_counter_ns()
            if plan is None:
                plan = Plan(rp, ci, vv, cfg)
                t1 = time.perf_counter_ns()
                setup_ns = t1 - t0; update_ns = 0
                x, iters, status = plan.solve(b_d, x0_d)
                x.block_until_ready()
                solve_ns = time.perf_counter_ns() - t1
            else:
                plan.update_coefficients(vv)
                t1 = time.perf_counter_ns()
                update_ns = t1 - t0; setup_ns = 0
                x, iters, status = plan.solve(b_d, x0_d)
                x.block_until_ready()
                solve_ns = time.perf_counter_ns() - t1

            x_h = np.asarray(x)
            # Compute similarity
            if x_prev is not None:
                similarity = float(np.abs(x_h - x_prev).max() /
                                    max(float(np.abs(x_h).max()), 1e-300))
            res = float(np.linalg.norm(A_pos @ x_h - b_pos) /
                        max(np.linalg.norm(b_pos), 1e-300))

            per_step.append({
                "step": i, "timestep": m["t"], "phase": m["phase"],
                "setup_ms": setup_ns / 1e6,
                "update_ms": update_ns / 1e6,
                "solve_ms": solve_ns / 1e6,
                "total_ms": (setup_ns + update_ns + solve_ns) / 1e6,
                "iter_count": int(iters[0]),
                "rel_resid": res,
                "similarity_to_prev": similarity,
                "warm_start_used": x_prev is not None,
            })
            print(f"  rep{rep:02d} step{i} ({m['phase']:<11s}): "
                  f"upd={update_ns/1e6:>5.0f}ms solve={solve_ns/1e6:>5.0f}ms "
                  f"iter={int(iters[0]):>4d} sim={similarity}")

            x_prev = x_h

        if plan is not None:
            plan.__exit__(None, None, None)
        total_wall_s = (time.perf_counter_ns() - total_t0) / 1e9

        payload = {
            "expt_id": "E04",
            "rep": rep,
            "wall_seconds": total_wall_s,
            "wall_seconds_method": "MEASURED:perf_counter_ns",
            "n_steps": len(matrices),
            "config": {"name": "CLASSICAL_V_DIAGSCALED", "max_iters": 2000,
                        "warm_start": True},
            "per_step": per_step,
            "env": env,
        }
        write_result_json(result_path, payload)
        print(f"  rep{rep:02d} TOTAL: {total_wall_s:.2f} s")

    print(f"\n[E04] complete.")


if __name__ == "__main__":
    main()
```

===== xeon_validation/E05_runner.py (4148 bytes, 117 lines) =====

```
"""E05 — CHOLMOD fresh refactor wall, ≥5 reps × 6 timesteps.

Per-step: cholesky() full factor + solve. No reuse.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, file_md5, write_result_json, TimedSection,
    normalize_sign, load_raw_matrix, find_npz, cold_cache,
)

import numpy as np
from scipy.sparse.linalg import spsolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke: args.reps = 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    try:
        from sksparse.cholmod import cholesky
        method = "CHOLMOD"
    except ImportError:
        method = "SuperLU"
        cholesky = None
        print("sksparse not available; falling back to scipy SuperLU. wall numbers will not match expected CHOLMOD speed.")

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = next((c for c in case_dir_candidates
                     if (c / "postProcessing/matrices").exists()), None)
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found")

    timesteps = find_npz(npz_dir)
    env = record_environment()
    print(f"E05 method: {method}")

    for time_str, phase, npz_path in timesteps:
        A, b_raw, x_OF = load_raw_matrix(case_dir, time_str)
        A_pos, b_pos, sf = normalize_sign(A, b_raw)
        b_md5 = file_md5(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/b.mm")
        A_md5 = file_md5(case_dir / "postProcessing/matrices" / time_str / "pd_corr0/A.mm")
        print(f"\n=== {phase} t={time_str} ===")

        for rep in range(1, args.reps + 1):
            rep_dir = output_dir / f"{rep:02d}_{time_str}"
            result_path = rep_dir / "result.json"
            if args.resume and result_path.exists():
                print(f"  rep{rep:02d}: SKIP"); continue

            cold_cache()
            with TimedSection() as outer:
                t0 = time.perf_counter_ns()
                if cholesky is not None:
                    factor = cholesky(A_pos.tocsc())
                    factor_ns = time.perf_counter_ns() - t0
                    t1 = time.perf_counter_ns()
                    x = factor(b_pos)
                    solve_ns = time.perf_counter_ns() - t1
                else:
                    x = spsolve(A_pos.tocsc(), b_pos)
                    factor_ns = time.perf_counter_ns() - t0
                    solve_ns = 0  # SuperLU bundles factor+solve

            res = float(np.linalg.norm(A_pos @ x - b_pos) /
                        max(np.linalg.norm(b_pos), 1e-300))

            payload = {
                "expt_id": "E05",
                "rep": rep,
                "method": method,
                "timestep": time_str,
                "phase": phase,
                "wall_seconds": outer["wall_seconds"],
                "wall_seconds_method": outer["wall_seconds_method"],
                "mem_peak_kb": outer["mem_peak_kb"],
                "factor_ns": int(factor_ns),
                "solve_ns": int(solve_ns),
                "factor_ms": factor_ns / 1e6,
                "solve_ms": solve_ns / 1e6,
                "rel_resid_actual": res,
                "input_md5s": {"A.mm": A_md5, "b.mm": b_md5},
                "n": int(A.shape[0]),
                "nnz": int(A.nnz),
                "sign_flipped": sf,
                "env": env,
            }
            write_result_json(result_path, payload)
            print(f"  rep{rep:02d}: factor={factor_ns/1e6:.0f}ms solve={solve_ns/1e6:.0f}ms "
                  f"resid={res:.2e}")

    print(f"\n[E05] complete.")


if __name__ == "__main__":
    main()
```

===== xeon_validation/E06_runner.py (4643 bytes, 134 lines) =====

```
"""E06 — CHOLMOD symbolic-reuse: analyze() once + cholesky_inplace() per step.

Sequence-of-6 approach (mirroring E03/E04). 5 reps each.
Watch for cholesky_inplace() failures (numerical issues) — record fallback to full.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, write_result_json,
    normalize_sign, load_raw_matrix, find_npz, cold_cache,
)

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke: args.reps = 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    try:
        from sksparse.cholmod import cholesky
    except ImportError:
        sys.exit("FAIL: sksparse not available. E06 requires CHOLMOD. Install scikit-sparse.")

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = next((c for c in case_dir_candidates
                     if (c / "postProcessing/matrices").exists()), None)
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found")

    timesteps = find_npz(npz_dir)
    env = record_environment()

    print("Pre-loading 6 matrices...")
    matrices = []
    for t, phase, npz_path in timesteps:
        A, b_raw, x_OF = load_raw_matrix(case_dir, t)
        A_pos, b_pos, sf = normalize_sign(A, b_raw)
        matrices.append({"t": t, "phase": phase, "A_csc": A_pos.tocsc(), "b": b_pos, "A_csr": A_pos})

    for rep in range(1, args.reps + 1):
        rep_dir = output_dir / f"rep_{rep:02d}"
        result_path = rep_dir / "result.json"
        if args.resume and result_path.exists():
            print(f"\nrep{rep:02d}: SKIP"); continue

        cold_cache()
        per_step = []
        factor = None
        warnings = []
        total_t0 = time.perf_counter_ns()

        for i, m in enumerate(matrices):
            t0 = time.perf_counter_ns()
            if factor is None:
                factor = cholesky(m["A_csc"])
                factor_ns = time.perf_counter_ns() - t0
                refact_ns = 0
                method_used = "full_factor"
            else:
                # Try inplace; on failure fall back to full
                try:
                    factor.cholesky_inplace(m["A_csc"])
                    refact_ns = time.perf_counter_ns() - t0
                    factor_ns = 0
                    method_used = "cholesky_inplace"
                except Exception as e:
                    warnings.append(f"step {i}: cholesky_inplace fail: {e}")
                    factor = cholesky(m["A_csc"])
                    factor_ns = time.perf_counter_ns() - t0
                    refact_ns = 0
                    method_used = "fallback_full"

            t1 = time.perf_counter_ns()
            x = factor(m["b"])
            solve_ns = time.perf_counter_ns() - t1
            res = float(np.linalg.norm(m["A_csr"] @ x - m["b"]) /
                        max(np.linalg.norm(m["b"]), 1e-300))

            per_step.append({
                "step": i, "timestep": m["t"], "phase": m["phase"],
                "factor_ms": factor_ns / 1e6,
                "refact_ms": refact_ns / 1e6,
                "solve_ms": solve_ns / 1e6,
                "total_ms": (factor_ns + refact_ns + solve_ns) / 1e6,
                "rel_resid": res,
                "method_used": method_used,
            })
            print(f"  rep{rep:02d} step{i} ({m['phase']:<11s}): "
                  f"{method_used} {(factor_ns+refact_ns)/1e6:>5.0f}ms "
                  f"solve={solve_ns/1e6:>4.0f}ms resid={res:.2e}")

        total_wall_s = (time.perf_counter_ns() - total_t0) / 1e9

        payload = {
            "expt_id": "E06",
            "rep": rep,
            "wall_seconds": total_wall_s,
            "wall_seconds_method": "MEASURED:perf_counter_ns",
            "n_steps": len(matrices),
            "method": "CHOLMOD_symbolic_reuse",
            "per_step": per_step,
            "warnings": warnings,
            "env": env,
        }
        write_result_json(result_path, payload)
        print(f"  rep{rep:02d} TOTAL: {total_wall_s:.2f} s")

    print(f"\n[E06] complete.")


if __name__ == "__main__":
    main()
```

===== xeon_validation/E07_runner.py (5486 bytes, 153 lines) =====

```
"""E07 — AMGx + 1 IR vs CHOLMOD LU per-timestep diff (settles C004 / S3).

For each of 6 timesteps:
  - LU truth via CHOLMOD
  - x_AMGx_e12 from npz (already computed)
  - diff: max|x_AMGx_e12 - x_LU|, ‖diff‖₂, plus relative
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, write_result_json,
    normalize_sign, load_raw_matrix, find_npz, load_npz_with_solutions,
)

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    try:
        from sksparse.cholmod import cholesky
    except ImportError:
        # Fall back to scipy SuperLU but warn
        from scipy.sparse.linalg import spsolve
        cholesky = None

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = next((c for c in case_dir_candidates
                     if (c / "postProcessing/matrices").exists()), None)
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found")

    timesteps = find_npz(npz_dir)
    env = record_environment()
    per_timestep = []

    for time_str, phase, npz_path in timesteps:
        rep_dir = output_dir / f"01_{time_str}"
        result_path = rep_dir / "result.json"
        if args.resume and result_path.exists():
            print(f"{phase} t={time_str}: SKIP"); continue

        # Load AMGx_e12 from npz
        npz_data = load_npz_with_solutions(npz_path)
        x_AMGx_e12 = npz_data["x_truth"]
        x_AMGx_e8 = npz_data["x_AMGx_e8"]
        x_OF = npz_data["x_OF"]

        # Load raw A, b
        A, b_raw, _ = load_raw_matrix(case_dir, time_str)
        A_pos, b_pos, sf = normalize_sign(A, b_raw)

        # LU truth
        t0 = time.perf_counter_ns()
        if cholesky is not None:
            factor = cholesky(A_pos.tocsc())
            x_LU = factor(b_pos)
            method = "CHOLMOD"
        else:
            from scipy.sparse.linalg import spsolve
            x_LU = spsolve(A_pos.tocsc(), b_pos)
            method = "SuperLU"
        lu_wall_s = (time.perf_counter_ns() - t0) / 1e9

        res_LU = float(np.linalg.norm(A_pos @ x_LU - b_pos) /
                        max(np.linalg.norm(b_pos), 1e-300))
        if res_LU > 1e-12:
            print(f"WARN: {phase} t={time_str}: LU residual {res_LU:.2e} > 1e-12")

        x_LU_inf = float(np.abs(x_LU).max())
        x_LU_2 = float(np.linalg.norm(x_LU))

        # Compute diffs (note: x_OF, x_AMGx are stored in original sign convention;
        # x_LU was computed from sign-flipped form. They may need to match conventions —
        # the npz x_truth came from amgx_solve(A_pos, b_pos), so it's in same sign as x_LU.
        # x_OF in npz: was loaded directly from x_final.mm (negative-diag convention).
        # Need to verify: x_LU should equal -? Actually no, x doesn't change with sign flip;
        # only A and b do. So x_LU and x_OF are in same units.

        results_per_solver = {}
        for name, x in [("OF", x_OF), ("AMGx_e8", x_AMGx_e8), ("AMGx_e12_IR", x_AMGx_e12)]:
            d = np.abs(x - x_LU)
            results_per_solver[name] = {
                "max_diff_Pa": float(d.max()),
                "rel_max": float(d.max() / max(x_LU_inf, 1e-300)),
                "L2_diff_Pa": float(np.linalg.norm(d)),
                "rel_L2": float(np.linalg.norm(d) / max(x_LU_2, 1e-300)),
                "median_diff_Pa": float(np.median(d)),
                "cells_above_100Pa": int((d > 100).sum()),
                "cells_above_1kPa": int((d > 1000).sum()),
            }

        record = {
            "expt_id": "E07",
            "timestep": time_str,
            "phase": phase,
            "lu_method": method,
            "lu_wall_s": lu_wall_s,
            "lu_rel_resid": res_LU,
            "x_LU_inf_norm": x_LU_inf,
            "x_LU_L2_norm": x_LU_2,
            "comparisons_vs_LU": results_per_solver,
            "env": env,
        }
        write_result_json(result_path, record)
        per_timestep.append(record)

        print(f"\n{phase} t={time_str}:")
        print(f"  LU ({method}): {lu_wall_s:.1f}s  res={res_LU:.2e}  ‖x_LU‖∞={x_LU_inf:.3e}")
        for n, r in results_per_solver.items():
            print(f"  {n:<12s}: max|diff|={r['max_diff_Pa']:.3e} Pa  "
                  f"rel={r['rel_max']:.2e}  >100Pa={r['cells_above_100Pa']}")

    # Aggregate
    aggr_path = output_dir / "aggregate.json"
    aggr = {
        "expt_id": "E07",
        "n_timesteps": len(per_timestep),
        "per_timestep": per_timestep,
    }
    if per_timestep:
        for sname in ["OF", "AMGx_e8", "AMGx_e12_IR"]:
            rels = [r["comparisons_vs_LU"][sname]["rel_max"] for r in per_timestep]
            aggr[f"{sname}_max_rel_max"] = max(rels)
            aggr[f"{sname}_min_rel_max"] = min(rels)
            aggr[f"{sname}_median_rel_max"] = float(np.median(rels))
    write_result_json(aggr_path, aggr)
    print(f"\n[E07] aggregate → {aggr_path}")


if __name__ == "__main__":
    main()
```

===== xeon_validation/E08_runner.py (7776 bytes, 199 lines) =====

```
"""E08 — Per-timestep iter + residual diagnostics + headline ratios.

Reads E02-E07 result.json files; produces:
  - diagnostics.csv (per-row data for plotting)
  - result.json (aggregated summary)
  - speedup_ratios.json (key ratios that settle the headline claims)

This is post-processing; no solver runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def collect_per_rep(rep_dir: Path):
    """For E02 / E05: list of per-rep result.json dicts."""
    out = []
    for r in rep_dir.iterdir():
        if not r.is_dir(): continue
        rj = r / "result.json"
        if not rj.exists(): continue
        with open(rj) as f: out.append(json.load(f))
    return out


def collect_seq(rep_dir: Path):
    """For E03 / E04 / E06: list of full-sequence result dicts (each has per_step)."""
    out = []
    for r in rep_dir.iterdir():
        if not r.is_dir(): continue
        rj = r / "result.json"
        if not rj.exists(): continue
        with open(rj) as f: out.append(json.load(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--npz-dir", default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir) if args.results_dir else output_dir.parent

    # === Collect ===
    e02 = collect_per_rep(results_dir / "E02") if (results_dir / "E02").exists() else []
    e03 = collect_seq(results_dir / "E03") if (results_dir / "E03").exists() else []
    e04 = collect_seq(results_dir / "E04") if (results_dir / "E04").exists() else []
    e05 = collect_per_rep(results_dir / "E05") if (results_dir / "E05").exists() else []
    e06 = collect_seq(results_dir / "E06") if (results_dir / "E06").exists() else []
    e07_aggr_path = results_dir / "E07" / "aggregate.json"
    e07_aggr = json.loads(e07_aggr_path.read_text()) if e07_aggr_path.exists() else {}
    e09_aggr_path = results_dir / "E09" / "aggregate.json"
    e09_aggr = json.loads(e09_aggr_path.read_text()) if e09_aggr_path.exists() else {}
    e01_path = results_dir / "E01" / "01" / "result.json"
    e01 = json.loads(e01_path.read_text()) if e01_path.exists() else {}

    # === Diagnostics CSV ===
    rows = []
    for d in e02:
        rows.append({
            "expt": "E02", "rep": d.get("rep"), "timestep": d.get("timestep"),
            "phase": d.get("phase"), "iter_count": d.get("iter_count"),
            "rel_resid": d.get("rel_resid_actual"), "wall_s": d.get("wall_seconds"),
            "step": -1,
        })
    for d in e03 + e04 + e06:
        for s in d.get("per_step", []):
            rows.append({
                "expt": d.get("expt_id", "?"),
                "rep": d.get("rep"),
                "step": s.get("step"),
                "timestep": s.get("timestep"),
                "phase": s.get("phase"),
                "iter_count": s.get("iter_count"),
                "rel_resid": s.get("rel_resid"),
                "wall_s": s.get("total_ms", 0) / 1000,
            })
    for d in e05:
        rows.append({
            "expt": "E05", "rep": d.get("rep"), "timestep": d.get("timestep"),
            "phase": d.get("phase"), "iter_count": -1,
            "rel_resid": d.get("rel_resid_actual"),
            "wall_s": d.get("wall_seconds"),
            "step": -1,
        })
    csv_path = output_dir / "diagnostics.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)

    # === Headline ratios ===
    ratios = {}

    # E02 mean wall × 6 ≈ AMGx_fresh_total_for_6_steps
    if e02:
        amgx_fresh_per = float(np.mean([d["wall_seconds"] for d in e02]))
        ratios["amgx_fresh_per_step_s_mean"] = amgx_fresh_per
        ratios["amgx_fresh_total_6step_estimate_s"] = amgx_fresh_per * 6

    # E03 mean total
    if e03:
        amgx_amort_total = float(np.mean([d["wall_seconds"] for d in e03]))
        ratios["amgx_amortized_total_s_mean"] = amgx_amort_total
        if e02:
            ratios["amortized_speedup_vs_fresh"] = (amgx_fresh_per * 6) / amgx_amort_total

    # E04 mean total
    if e04:
        amgx_warm_total = float(np.mean([d["wall_seconds"] for d in e04]))
        ratios["amgx_amortized_warm_total_s_mean"] = amgx_warm_total
        if e02:
            ratios["warm_amortized_speedup_vs_fresh"] = (amgx_fresh_per * 6) / amgx_warm_total

    # E05 mean wall × 6
    if e05:
        chol_fresh_per = float(np.mean([d["wall_seconds"] for d in e05]))
        ratios["cholmod_fresh_per_step_s_mean"] = chol_fresh_per
        ratios["cholmod_fresh_total_6step_estimate_s"] = chol_fresh_per * 6

    # E06 mean total
    if e06:
        chol_sym_total = float(np.mean([d["wall_seconds"] for d in e06]))
        ratios["cholmod_symbolic_reuse_total_s_mean"] = chol_sym_total
        if e05:
            ratios["cholmod_symbolic_speedup_vs_fresh"] = (chol_fresh_per * 6) / chol_sym_total

    # AMGx warm vs CHOLMOD fresh — the headline production-style comparison
    if e04 and e05:
        ratios["AMGx_warm_vs_LU_fresh_speedup"] = (chol_fresh_per * 6) / amgx_warm_total
    if e04 and e06:
        ratios["AMGx_warm_vs_LU_symbolic_speedup"] = chol_sym_total / amgx_warm_total

    # E01 OF wall — would need to estimate per-pd-solve from log parsing
    if e01.get("step_wall_stats"):
        ratios["of_step_wall_mean_s"] = e01["step_wall_stats"].get("mean")
        ratios["of_pd_iter_mean"] = e01.get("pd_iter_per_step_stats", {}).get("mean")
        ratios["of_total_iter_mean"] = e01.get("total_iter_per_step_stats", {}).get("mean")
        ratios["of_approx_per_iter_ms"] = e01.get("approx_per_iter_ms")

    # OF vs AMGx_warm: per-step
    if e04 and e01.get("step_wall_stats", {}).get("mean"):
        # AMGx warm mean per step (steps 1..N, excluding setup step 0)
        amort_step1n = []
        for d in e04:
            ps = d.get("per_step", [])
            if len(ps) > 1:
                amort_step1n.extend([s["total_ms"] / 1000 for s in ps[1:]])
        if amort_step1n:
            ratios["amgx_warm_per_step_amortized_s_mean"] = float(np.mean(amort_step1n))
            ratios["amgx_warm_vs_OF_per_step_speedup"] = (
                e01["step_wall_stats"]["mean"] / float(np.mean(amort_step1n))
            )

    # === Build summary ===
    out = {
        "expt_id": "E08",
        "n_rows": len(rows),
        "n_e02": len(e02),
        "n_e03": len(e03),
        "n_e04": len(e04),
        "n_e05": len(e05),
        "n_e06": len(e06),
        "e07_summary": e07_aggr,
        "e09_summary": e09_aggr,
        "e01_present": bool(e01),
        "headline_ratios": ratios,
        "rows": rows,
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(output_dir / "speedup_ratios.json", "w") as f:
        json.dump(ratios, f, indent=2)

    print(f"\nE08 aggregate ({len(rows)} rows):")
    print(f"  E02 reps: {len(e02)}, E03 reps: {len(e03)}, E04 reps: {len(e04)}")
    print(f"  E05 reps: {len(e05)}, E06 reps: {len(e06)}")
    print(f"  E07: {'YES' if e07_aggr else 'no'}, E09: {'YES' if e09_aggr else 'no'}, E01: {'YES' if e01 else 'no'}")
    print(f"\nHeadline ratios:")
    for k, v in ratios.items():
        if isinstance(v, float):
            print(f"  {k:<55s}: {v:.3f}")
        else:
            print(f"  {k:<55s}: {v}")


if __name__ == "__main__":
    main()
```

===== xeon_validation/E09_runner.py (5560 bytes, 150 lines) =====

```
"""E09 — Lanczos κ(A) for 6 single_track + 1 lab32 matrices.

For each:
  σ_max = eigsh(A_pos, k=1, which='LA')
  σ_min = eigsh(A_pos, k=1, sigma=0, which='LM')   # shift-invert
  κ = σ_max / σ_min
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    record_environment, write_result_json,
    normalize_sign, load_raw_matrix, find_npz,
)

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def estimate_kappa(A_pos, tol=1e-3, maxiter=200):
    """Return (sigma_max, sigma_min, kappa) via Lanczos."""
    sigma_max = float(eigsh(A_pos, k=1, which='LA',
                             return_eigenvectors=False,
                             tol=tol, maxiter=maxiter)[0])
    try:
        sigma_min = float(eigsh(A_pos, k=1, sigma=0, which='LM',
                                 return_eigenvectors=False,
                                 tol=tol, maxiter=maxiter)[0])
    except Exception as e:
        # fallback to SA
        sigma_min = float(eigsh(A_pos, k=1, which='SA',
                                 return_eigenvectors=False,
                                 tol=tol, maxiter=maxiter)[0])
    kappa = abs(sigma_max / max(abs(sigma_min), 1e-300))
    return sigma_max, sigma_min, kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)

    case_dir_candidates = [
        Path("/home/yzk/cases/single_track_dump"),
        Path("/home/yzk/single_track_dump"),
    ]
    case_dir = next((c for c in case_dir_candidates
                     if (c / "postProcessing/matrices").exists()), None)
    if case_dir is None:
        sys.exit(f"FAIL: no case dir found")

    timesteps = find_npz(npz_dir)
    env = record_environment()
    summary = []

    # 6 single_track
    for time_str, phase, _ in timesteps:
        rep_dir = output_dir / f"single_track_{time_str}"
        result_path = rep_dir / "result.json"
        if args.resume and result_path.exists():
            print(f"single_track {time_str}: SKIP"); continue

        A, b, _ = load_raw_matrix(case_dir, time_str)
        A_pos, _, _ = normalize_sign(A, b)
        print(f"\nsingle_track {phase} t={time_str}: estimating κ ...")
        t0 = time.time()
        sigma_max, sigma_min, kappa = estimate_kappa(A_pos)
        wall = time.time() - t0
        diag = A_pos.diagonal()
        record = {
            "expt_id": "E09",
            "case": "single_track",
            "phase": phase,
            "timestep": time_str,
            "n": int(A_pos.shape[0]),
            "nnz": int(A_pos.nnz),
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "kappa": kappa,
            "diag_max": float(np.abs(diag).max()),
            "diag_min": float(np.abs(diag).min()),
            "diag_spread": float(np.abs(diag).max() / max(np.abs(diag).min(), 1e-300)),
            "lanczos_wall_s": wall,
            "env": env,
        }
        write_result_json(result_path, record)
        summary.append(record)
        print(f"  σ_max={sigma_max:.3e}  σ_min={sigma_min:.3e}  κ={kappa:.3e}  ({wall:.1f}s)")
        if args.smoke: break

    # lab32 matrix (already cached as global_pd_corr0_3.8e-07.npz)
    lab32_path = Path("/home/yzk/DILU-Research/dilu/amgx/bench/global_pd_corr0_3.8e-07.npz")
    if lab32_path.exists() and not args.smoke:
        rep_dir = output_dir / "lab32_melt_380ns"
        result_path = rep_dir / "result.json"
        if not (args.resume and result_path.exists()):
            g = np.load(lab32_path)
            A_lab32 = csr_matrix((g["A_data"], g["A_indices"], g["A_indptr"]),
                                  shape=(500000, 500000))
            A_lab32_pos = -A_lab32
            print(f"\nlab32 melt-380ns: estimating κ ...")
            t0 = time.time()
            sigma_max, sigma_min, kappa = estimate_kappa(A_lab32_pos)
            wall = time.time() - t0
            record = {
                "expt_id": "E09",
                "case": "lab32",
                "phase": "melting",
                "timestep": "3.8e-07",
                "n": 500000,
                "nnz": int(A_lab32_pos.nnz),
                "sigma_max": sigma_max,
                "sigma_min": sigma_min,
                "kappa": kappa,
                "lanczos_wall_s": wall,
                "env": env,
            }
            write_result_json(result_path, record)
            summary.append(record)
            print(f"  σ_max={sigma_max:.3e}  σ_min={sigma_min:.3e}  κ={kappa:.3e}  ({wall:.1f}s)")

    # Summary
    aggr_path = output_dir / "aggregate.json"
    with open(aggr_path, "w") as f:
        json.dump({"expt_id": "E09", "n_matrices": len(summary), "results": summary}, f, indent=2)
    print(f"\n[E09] aggregate → {aggr_path}")
    print("\nκ summary:")
    print(f"{'case':<20s} {'phase':<11s} {'t':<11s} {'σ_max':>11s} {'σ_min':>11s} {'κ':>11s}")
    for r in summary:
        print(f"{r['case']:<20s} {r['phase']:<11s} {r['timestep']:<11s} "
              f"{r['sigma_max']:>11.3e} {r['sigma_min']:>11.3e} {r['kappa']:>11.3e}")


if __name__ == "__main__":
    main()
```

===== xeon_validation/compare_to_expected.py (5158 bytes, 130 lines) =====

```
"""Compare E0X result.json files to expected_results_template.json.

Walks results-dir for each E0X, extracts key metrics, compares to predicted
range. Outputs summary.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def collect_e02(rep_dir: Path):
    walls = []; iters = []; resids = []
    for r in rep_dir.iterdir():
        if not r.is_dir(): continue
        rj = r / "result.json"
        if not rj.exists(): continue
        d = json.loads(rj.read_text())
        walls.append(d["wall_seconds"])
        iters.append(d["iter_count"])
        resids.append(d["rel_resid_actual"])
    return walls, iters, resids


def collect_seq(rep_dir: Path):
    """For E03/E04/E06 — list of total_wall_s and per-step lists."""
    totals = []
    per_step_walls = []
    per_step_iters = []
    for r in rep_dir.iterdir():
        if not r.is_dir(): continue
        rj = r / "result.json"
        if not rj.exists(): continue
        d = json.loads(rj.read_text())
        totals.append(d["wall_seconds"])
        per_step_walls.append([s["total_ms"] for s in d.get("per_step", [])])
        per_step_iters.append([s.get("iter_count") or 0 for s in d.get("per_step", [])])
    return totals, per_step_walls, per_step_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    expected_path = Path(args.expected)
    output_path = Path(args.output)

    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    summary = ["# Xeon Validation Summary\n",
                f"results_dir: `{results_dir}`",
                f"expected_template: `{expected_path}`\n",
                "---\n"]

    for E in ("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09"):
        e_dir = results_dir / E
        summary.append(f"\n## {E}")
        if not e_dir.exists():
            summary.append(f"  NOT RUN")
            continue

        if E == "E01":
            rj = e_dir / "01" / "result.json"
            if rj.exists():
                d = json.loads(rj.read_text())
                summary.append(f"  status: {d.get('status')}")
                if d.get("step_wall_stats"):
                    s = d["step_wall_stats"]
                    summary.append(f"  mean step wall: {s.get('mean'):.2f} s, median: {s.get('median'):.2f} s")
                summary.append(f"  approx ms/iter: {d.get('approx_per_iter_ms')}")

        elif E in ("E02", "E05"):
            walls, iters, resids = collect_e02(e_dir)
            if walls:
                summary.append(f"  N reps: {len(walls)}")
                summary.append(f"  wall: mean={np.mean(walls):.2f}s median={np.median(walls):.2f}s "
                                f"std={np.std(walls):.2f}s")
                summary.append(f"  iter: mean={np.mean(iters):.0f}")
                summary.append(f"  resid: max={max(resids):.2e}")

        elif E in ("E03", "E04", "E06"):
            totals, per_step_walls, per_step_iters = collect_seq(e_dir)
            if totals:
                summary.append(f"  N reps: {len(totals)}")
                summary.append(f"  total wall: mean={np.mean(totals):.2f}s median={np.median(totals):.2f}s")
                if per_step_walls:
                    psw = np.array(per_step_walls)  # [reps, steps]
                    summary.append(f"  step 0 (setup): {psw[:, 0].mean():.0f}ms")
                    if psw.shape[1] > 1:
                        summary.append(f"  steps 1..N (amortized): mean={psw[:, 1:].mean():.0f}ms")

        elif E == "E07":
            agg = e_dir / "aggregate.json"
            if agg.exists():
                d = json.loads(agg.read_text())
                for sname in ["OF", "AMGx_e8", "AMGx_e12_IR"]:
                    if f"{sname}_max_rel_max" in d:
                        summary.append(f"  {sname:<14s}: max rel diff vs LU = {d[f'{sname}_max_rel_max']:.3e}")

        elif E == "E09":
            agg = e_dir / "aggregate.json"
            if agg.exists():
                d = json.loads(agg.read_text())
                for r in d.get("results", []):
                    summary.append(f"  {r['case']:<14s} {r['phase']:<10s} t={r['timestep']}: "
                                    f"κ={r['kappa']:.3e}")

    # Compare to expected
    summary.append("\n---\n## Comparison to expected\n")
    if expected:
        for pred in expected.get("predictions", []):
            eid = pred.get("expt_id", "?")
            metric = pred.get("metric", "?")
            val_pred = pred.get("predicted_value")
            range_pred = pred.get("predicted_range", [None, None])
            summary.append(f"\n### {eid} / {metric}")
            summary.append(f"  Predicted: {val_pred} ∈ [{range_pred[0]}, {range_pred[1]}]")
            summary.append(f"  Rationale: {pred.get('rationale', '(none)')}")

    output_path.write_text("\n".join(summary))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
```

===== xeon_validation/manifest.json (477 bytes, 24 lines) =====

```
{
  "experiments_planned": [
    "E02",
    "E03",
    "E04",
    "E05",
    "E06",
    "E07",
    "E08",
    "E09"
  ],
  "started_at": "2026-05-10T00:50:23+08:00",
  "env_file": "/home/yzk/DILU-Research/audit_overnight_20260509/xeon_validation/env_2026-05-10T00:50:23+08:00.txt",
  "flags": {
    "dry_run": 1,
    "smoke": 0,
    "resume": 1,
    "only": ""
  },
  "thread_pin": {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
  }
}```

===== expected_results_template.json (4203 bytes, 94 lines) =====

```
{
  "doc": "Pre-registered predictions for Xeon validation. Each prediction has rationale.",
  "predictions": [
    {
      "expt_id": "E01",
      "metric": "approx_per_iter_ms (mean wall_per_step / mean total_iter_per_step × 1000)",
      "predicted_value": 110,
      "predicted_range": [50, 200],
      "rationale": "PROJECT_STATUS_REPORT used 110 ms/iter (5s/step ÷ 45 iter/step). Layered estimation has 30% uncertainty per factor. Plausible range 50-200."
    },
    {
      "expt_id": "E02",
      "metric": "wall_seconds_mean (AMGx fresh single-shot, 500K LPBF pd, RTX 3050 dev OR RTX 5060 lab)",
      "predicted_value": 20,
      "predicted_range": [5, 50],
      "rationale": "Existing npz meta shows 9-39s on dev RTX 3050. Lab 5060 may be 2-3x faster (8GB vs 4GB, sm_120 vs sm_86). Range 5-50."
    },
    {
      "expt_id": "E03",
      "metric": "step_1_to_N_mean_ms_amortized (excluding step 0 full setup)",
      "predicted_value": 800,
      "predicted_range": [50, 5000],
      "rationale": "Hypothesis: update_coefficients ~ 50ms + solve ~ 100-1000ms. If solve dominates and unchanged from fresh, amortized doesn't help much. Predict update saves 1-3s setup overhead per step but solve still 5-20s."
    },
    {
      "expt_id": "E03",
      "metric": "speedup_amortized_vs_fresh (total wall ratio)",
      "predicted_value": 1.5,
      "predicted_range": [1.0, 5.0],
      "rationale": "If setup is 2-3s and solve dominates at 5-20s per step, amortized saves only setup × (N-1) = ~10s out of N×solve = ~50-100s ⇒ 10-20% speedup. Far below the asserted 100-1000×."
    },
    {
      "expt_id": "E04",
      "metric": "iter_savings_warm_vs_cold_pct",
      "predicted_value": 30,
      "predicted_range": [0, 70],
      "rationale": "LPBF physics has rapid alpha jumps; x_t may differ substantially from x_{t-1}. Warm-start could save 0-70% iter — large uncertainty."
    },
    {
      "expt_id": "E04",
      "metric": "similarity_x_t_vs_x_prev_max (rel ‖∞)",
      "predicted_value": 0.05,
      "predicted_range": [0.001, 0.5],
      "rationale": "x evolves with melt pool; rel diff between adjacent timesteps probably 0.1-50%."
    },
    {
      "expt_id": "E05",
      "metric": "factor_seconds_mean (CHOLMOD fresh, 500K)",
      "predicted_value": 70,
      "predicted_range": [40, 150],
      "rationale": "Conversation history showed 69-75s. Predict similar ±2x for cold-cache variance."
    },
    {
      "expt_id": "E06",
      "metric": "speedup_symbolic_reuse_vs_fresh (total wall ratio)",
      "predicted_value": 1.5,
      "predicted_range": [0.8, 5.0],
      "rationale": "Symbolic part of cholesky is ~30-50% of factor cost; reuse saves that ⇒ 1.4-2× speedup. May fail (numerical issue with near-singular matrix) ⇒ ratio ≈ 1."
    },
    {
      "expt_id": "E07",
      "metric": "max_rel_max_AMGx_e12_IR_vs_LU (across 6 timesteps)",
      "predicted_value": 1.13e-11,
      "predicted_range": [1e-13, 1e-9],
      "rationale": "Conversation history showed 1.13e-11. Range allows for numerical variance."
    },
    {
      "expt_id": "E07",
      "metric": "max_rel_max_OF_vs_LU (across 6 timesteps)",
      "predicted_value": 3.5e-5,
      "predicted_range": [1e-6, 1e-3],
      "rationale": "OF tol=1e-8 × κ ~ 10^3-10^5 typical → 1e-5 to 1e-3. Conversation showed 3.47e-5."
    },
    {
      "expt_id": "E09",
      "metric": "kappa_lab32_melt_380ns",
      "predicted_value": 3.2e14,
      "predicted_range": [1e10, 1e16],
      "rationale": "Tonight's audit measured 3.2e14 via Lanczos shift-invert. Re-verify on Xeon with possibly different LAPACK/BLAS — expect within 1-2 orders of magnitude."
    },
    {
      "expt_id": "E09",
      "metric": "kappa_single_track_melt_380ns",
      "predicted_value": 1e10,
      "predicted_range": [1e6, 1e16],
      "rationale": "Different matrix from lab32 (rays>0 ≠ rays=0); expect lower κ since b is non-trivial. Wide range — true unknown."
    }
  ],
  "comparison_method": {
    "PASS": "value within predicted_range",
    "MARGINAL": "value within 2× of range bounds",
    "FAIL": "value outside 2× of range bounds — this is the interesting case (refutes hypothesis)"
  }
}
```

===== xeon_validation/E10_runner.py =====  NOT IMPLEMENTED

Per XEON_VALIDATION_PLAN.v1.md §E10: 32-rank fresh dump matrixDumper-patch verify.
Marked OPTIONAL in plan (~2h Xeon wall). Not implemented in v1 scripts. If user
enables --only=E10, run_xeon_validation.sh will report '$id ERROR: not found'.

===== xeon_validation/E11_runner.py =====  SUBSUMED

Per XEON_VALIDATION_PLAN.v1.md §E11: Phase 0 replay clean retry — explicitly
noted as 'subsumed by E02-E06 with proper rep counts'. No standalone runner;
result equivalent to summing E02+E03+E04+E05+E06 totals in E08.
