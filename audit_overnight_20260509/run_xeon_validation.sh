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
# enable explicitly via --only E01 because it takes 7 hours).
# E08 must run AFTER E09 because E08 aggregates E09's results (smoke run revealed this bug).
DEFAULT_EXPERIMENTS="E02 E03 E04 E05 E06 E07 E09 E08"
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

# Allow `from dilu.amgx.python import Plan` from any CWD.
# Detect repo root by walking up from script dir.
REPO_ROOT="$(cd "$SCRIPT_DIR" && cd .. && pwd)"
if [[ -d "$REPO_ROOT/dilu" ]]; then
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    echo "PYTHONPATH set to include $REPO_ROOT"
else
    echo "WARN: dilu module not found at $REPO_ROOT/dilu — runners may fail"
fi

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

# AMGx .so existence check (Python import != .so present — dry-run missed this before)
# Only WARN, not FAIL: E05/E06/E07/E08/E09 don't need AMGx, only E02/E03/E04 do.
AMGX_SO="$REPO_ROOT/dilu/amgx/build/libdilu_amgx.so"
if [[ -n "${REPO_ROOT:-}" ]] && [[ ! -f "$AMGX_SO" ]]; then
    case "$EXPS" in
        *E02*|*E03*|*E04*)
            echo "WARN: $AMGX_SO not found — E02/E03/E04 will fail."
            echo "      Build it via:  cd $REPO_ROOT/dilu/amgx && bash build.sh"
            echo "      OR skip AMGx experiments: --only=E05,E06,E07,E09,E08" ;;
    esac
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
