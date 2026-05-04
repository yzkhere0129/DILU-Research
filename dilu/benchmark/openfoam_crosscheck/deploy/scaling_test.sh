#!/bin/bash
# scaling_test.sh — Multi-core scaling experiment for laserMeltFoam
#
# Tests different processor counts and measures per-solve wall time.
#
# Usage:
#   ./scaling_test.sh <case_dir> [options]
#
# Options:
#   --steps N               Steps per run (default: 10)
#   --cores "1 2 4 8 ..."  Core counts to test (default: auto-detect)
#   --deltaT VAL            Override deltaT (default: read from controlDict)
#   --endTime VAL           Override endTime (default: read from controlDict)
#
# Prerequisites:
#   - OpenFOAM v2506 with laserMeltFoam patched (timing patches in TEqn.H/pEqn.H)
#   - Case set up with system/controlDict, constant/, and at least one time folder
#   - MPI installed

set -euo pipefail

# =========================================================================
# Arguments
# =========================================================================
CASE_DIR="${1:?Usage: scaling_test.sh <case_dir> [--steps N] [--cores '1 2 4 ...']}"
shift

N_STEPS=10
CORES=""
OVERRIDE_DELTAT=""
OVERRIDE_ENDTIME=""
DIAGNOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) N_STEPS="$2"; shift 2 ;;
        --cores) CORES="$2"; shift 2 ;;
        --deltaT) OVERRIDE_DELTAT="$2"; shift 2 ;;
        --endTime) OVERRIDE_ENDTIME="$2"; shift 2 ;;
        --diagnose) DIAGNOSE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =========================================================================
# Validate case directory
# =========================================================================
if [[ ! -d "${CASE_DIR}" ]]; then
    echo "ERROR: Case directory not found: ${CASE_DIR}"
    exit 1
fi
CASE_DIR="$(cd "${CASE_DIR}" && pwd)"

# =========================================================================
# Diagnose mode: just show what we find
# =========================================================================
if ${DIAGNOSE}; then
    echo ""
    echo "========================================"
    echo "DIAGNOSE MODE — case structure"
    echo "========================================"
    echo "  CASE_DIR: ${CASE_DIR}"
    echo ""
    echo "  Top-level contents:"
    ls -la "${CASE_DIR}" | head -20
    echo ""
    echo "  Looking for controlDict:"
    find "${CASE_DIR}" -name "controlDict" -not -name "*.orig" 2>/dev/null | while read f; do
        echo "    found: $f"
    done
    echo ""
    echo "  Looking for system/ directory:"
    find "${CASE_DIR}" -name "system" -type d 2>/dev/null | while read d; do
        echo "    found: $d"
        ls "$d" | head -10
    done
    echo ""
    echo "  Time folders:"
    ls -d "${CASE_DIR}"/[0-9]* 2>/dev/null | head -10 || echo "    (none)"
    echo ""
    echo "  constant/ contents:"
    ls "${CASE_DIR}/constant/" 2>/dev/null | head -10 || echo "    (no constant/)"
    echo ""
    echo "  OpenFOAM env:"
    echo "    WM_PROJECT_VERSION=${WM_PROJECT_VERSION:-not set}"
    echo "    FOAM_USER_APPBIN=${FOAM_USER_APPBIN:-not set}"
    echo "    which laserMeltFoam: $(which laserMeltFoam 2>/dev/null || echo 'not found')"
    echo "    which decomposePar: $(which decomposePar 2>/dev/null || echo 'not found')"
    echo "    which mpirun: $(which mpirun 2>/dev/null || echo 'not found')"
    echo "    nproc: $(nproc 2>/dev/null || echo '?')"
    echo "========================================"
    exit 0
fi

if [[ ! -f "${CASE_DIR}/system/controlDict" ]]; then
    echo "WARNING: ${CASE_DIR}/system/controlDict not found"
    echo ""
    # Try to find controlDict somewhere in the tree
    FOUND_CD=$(find "${CASE_DIR}" -name "controlDict" -not -name "*.orig" 2>/dev/null | head -1)
    if [[ -n "${FOUND_CD}" ]]; then
        echo "  Found controlDict at: ${FOUND_CD}"
        # Derive the case root (parent of system/)
        CASE_DIR=$(dirname "$(dirname "${FOUND_CD}")")
        echo "  Adjusted CASE_DIR to: ${CASE_DIR}"
    else
        echo "  No controlDict found anywhere under ${CASE_DIR}"
        echo ""
        echo "  Your case directory contents:"
        ls -la "${CASE_DIR}" 2>/dev/null || echo "  (cannot list)"
        echo ""
        echo "  Usage: $0 <case_dir> [--steps N] [--cores '1 2 4 ...']"
        echo "  <case_dir> must contain system/controlDict"
        exit 1
    fi
fi

# =========================================================================
# Auto-detect core count if not specified
# =========================================================================
if [[ -z "${CORES}" ]]; then
    MAX_CORES=$(nproc 2>/dev/null || echo 4)
    # Generate powers of 2 up to MAX_CORES, plus MAX_CORES itself
    CORES=""
    c=1
    while [[ $c -le $MAX_CORES ]]; do
        CORES="${CORES} ${c}"
        c=$((c * 2))
    done
    # Add MAX_CORES if not already a power of 2
    if [[ $((MAX_CORES & (MAX_CORES - 1))) -ne 0 ]]; then
        CORES="${CORES} ${MAX_CORES}"
    fi
    CORES=$(echo "${CORES}" | xargs)  # trim whitespace
fi

RESULTS_DIR="${CASE_DIR}/scaling_results"
mkdir -p "${RESULTS_DIR}"

echo "========================================"
echo "Scaling test: ${CASE_DIR}"
echo "Machine logical CPUs (nproc): $(nproc 2>/dev/null || echo '?')"
PHYS_CORES=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/ {c=$2} /^Socket\(s\)/ {s=$2} END {gsub(/ /,"",c); gsub(/ /,"",s); if (c && s) print c*s}')
echo "Machine physical cores (lscpu): ${PHYS_CORES:-unknown}"
echo "Steps per run: ${N_STEPS}"
echo "Core counts: ${CORES}"
echo "MPI flags: ${MPIRUN_FLAGS:---oversubscribe --bind-to none}"
if [[ -n "${PHYS_CORES}" ]]; then
    for n in ${CORES}; do
        if [[ $n -gt $PHYS_CORES ]]; then
            echo "  NOTE: N=$n exceeds ${PHYS_CORES} physical cores — using SMT (slower than scaling-ideal)"
        fi
    done
fi
echo "========================================"

# =========================================================================
# Read deltaT and endTime from controlDict (or use overrides)
# =========================================================================
if [[ -n "${OVERRIDE_DELTAT}" ]]; then
    ORIG_DELTA_T="${OVERRIDE_DELTAT}"
else
    ORIG_DELTA_T=$(grep -E '^\s*deltaT\s' "${CASE_DIR}/system/controlDict" \
        | head -1 | awk '{print $2}' | tr -d ';' || true)
    # If deltaT looks like a placeholder (adaptive timestepping), detect from time folders
    if [[ "${ORIG_DELTA_T}" =~ ^1[.]?[0]*e-1[12]$ ]] 2>/dev/null; then
        echo "  deltaT=${ORIG_DELTA_T} looks like a placeholder (adaptive timestep)."
        DETECTED_DT=$(ls -d "${CASE_DIR}"/[0-9]* 2>/dev/null \
            | sed 's|.*/||' | sort -g | head -2 | python3 -c "
import sys
vals = [float(l.strip()) for l in sys.stdin if l.strip()]
if len(vals) >= 2:
    print(f'{vals[1]-vals[0]:.3g}')
")
        if [[ -n "${DETECTED_DT}" && "${DETECTED_DT}" != "0" ]]; then
            echo "  Detected real deltaT=${DETECTED_DT} from time folders"
            ORIG_DELTA_T="${DETECTED_DT}"
        else
            # Fallback: use system/controlDict's maxDeltaT if present (the steady
            # adaptive cap), else hard-fail with a clear message. NEVER silently
            # keep the 1e-12 placeholder — that gives endTime = N_STEPS * 1e-12
            # which only captures cold-start (dt < 1e-11) and produces
            # results NOT comparable to runs that reached stationary state.
            MAX_DT=$(grep -E '^\s*maxDeltaT\s' "${CASE_DIR}/system/controlDict" \
                | head -1 | awk '{print $2}' | tr -d ';' || true)
            if [[ -n "${MAX_DT}" ]]; then
                echo "  Time folders absent; falling back to maxDeltaT=${MAX_DT} (controlDict)"
                ORIG_DELTA_T="${MAX_DT}"
            else
                echo "ERROR: deltaT=${ORIG_DELTA_T} is a placeholder, no time folders to"
                echo "       detect from, and no maxDeltaT in controlDict."
                echo "       Pass --deltaT VAL explicitly (e.g. --deltaT 2e-8) so endTime"
                echo "       will be large enough to reach stationary state."
                exit 1
            fi
        fi
    fi
fi

if [[ -n "${OVERRIDE_ENDTIME}" ]]; then
    ORIG_END_TIME="${OVERRIDE_ENDTIME}"
else
    ORIG_END_TIME=$(grep -E '^\s*endTime\s' "${CASE_DIR}/system/controlDict" \
        | head -1 | awk '{print $2}' | tr -d ';' || true)
fi

if [[ -z "${ORIG_DELTA_T}" ]]; then
    echo "ERROR: Could not read deltaT from controlDict. Use --deltaT to specify."
    exit 1
fi
if [[ -z "${ORIG_END_TIME}" ]]; then
    echo "ERROR: Could not read endTime from controlDict. Use --endTime to specify."
    exit 1
fi

echo ""
echo "Original: deltaT=${ORIG_DELTA_T}, endTime=${ORIG_END_TIME}"
echo "Will run ${N_STEPS} steps per core count"
echo ""

# =========================================================================
# Helper: create decomposeParDict for N processors
# =========================================================================
write_decompose_dict() {
    local N="$1"
    cat > "${CASE_DIR}/system/decomposeParDict" <<DICT
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
method          scotch;
numberOfSubdomains  ${N};
DICT
}

# =========================================================================
# Helper: disable matrixDumper for scaling test
# =========================================================================
disable_dumper() {
    if [[ -f "${CASE_DIR}/system/matrixDumperDict" ]]; then
        sed -i 's/enable.*true/enable  false/' "${CASE_DIR}/system/matrixDumperDict"
    fi
}

# =========================================================================
# Helper: extract timing data from log
# =========================================================================
extract_timings() {
    local log_file="$1"
    local out_csv="$2"
    local N="$3"

    echo "step,time_s,eq,wall_ms" > "${out_csv}"

    grep -E "\[\s*TIMING_(pd|T)\s*\]" "${log_file}" 2>/dev/null | while read -r line; do
        eq=$(echo "${line}" | grep -oP 'TIMING_\K(pd|T)')
        ms=$(echo "${line}" | grep -oP '[0-9]+\.[0-9]+' | head -1)
        echo "${N},${eq},${ms}" >> "${out_csv}"
    done || true
}

# =========================================================================
# Main loop
# =========================================================================
# Save original files
cp "${CASE_DIR}/system/decomposeParDict" "${RESULTS_DIR}/decomposeParDict.bak" 2>/dev/null || true
if [[ -f "${CASE_DIR}/system/matrixDumperDict" ]]; then
    cp "${CASE_DIR}/system/matrixDumperDict" "${RESULTS_DIR}/matrixDumperDict.bak"
fi

disable_dumper

# Summary file — append-friendly. If file already has data, keep it; otherwise
# write header. This lets you re-run only failed N's without losing the rest.
SUMMARY="${RESULTS_DIR}/summary.csv"
if [[ ! -s "${SUMMARY}" ]]; then
    echo "cores,step,pd_ms,t_ms,wall_ms" > "${SUMMARY}"
fi

for N in ${CORES}; do
    echo "========================================"
    echo "Testing N=${N} processors"
    echo "========================================"

    # 1. Write decomposeParDict
    write_decompose_dict "${N}"

    # 2. Clean previous decomposition
    rm -rf "${CASE_DIR}/processor"* 2>/dev/null || true

    # 3. Decompose
    echo "  decomposing..."
    decompar_log="${RESULTS_DIR}/decompar_${N}.log"
    if ! (cd "${CASE_DIR}" && decomposePar -force > "${decompar_log}" 2>&1); then
        echo "  ERROR: decomposePar failed for N=${N}, see ${decompar_log}"
        continue
    fi

    # 4. Set endTime = currentTime + N_STEPS * deltaT
    # Find current max time from case time folders (works for both serial & parallel)
    CUR_TIME=$( (ls -d "${CASE_DIR}"/[0-9]*e-* "${CASE_DIR}"/[0-9]*.[0-9]* 2>/dev/null || true) \
        | sed 's|.*/||' | sort -g | tail -1)
    # Also check processor0 time folders (parallel residual)
    if [[ -z "${CUR_TIME}" ]]; then
        CUR_TIME=$( (ls -d "${CASE_DIR}/processor0"/[0-9]* 2>/dev/null || true) \
            | sed 's|.*/||' | sort -g | tail -1)
    fi
    if [[ -z "${CUR_TIME}" ]]; then
        CUR_TIME="0"
    fi
    echo "  Current time: ${CUR_TIME}"
    NEW_END_TIME=$(python3 -c "print(${CUR_TIME} + ${N_STEPS} * ${ORIG_DELTA_T})")
    sed -i "s/^\s*endTime\s.*/endTime     ${NEW_END_TIME};/" "${CASE_DIR}/system/controlDict"

    # 5. Run
    run_log="${RESULTS_DIR}/scaling_${N}cores.log"
    echo "  running laserMeltFoam with ${N} procs (steps=${N_STEPS})..."

    # MPIRUN_FLAGS: --oversubscribe lets us use SMT/hyperthreads when N exceeds
    # physical cores; safe no-op when there are enough slots. Override via env.
    MPIRUN_FLAGS="${MPIRUN_FLAGS:---oversubscribe --bind-to none}"

    START_WALL=$(date +%s%N)
    RUN_OK=1
    if [[ "${N}" -eq 1 ]]; then
        if ! (cd "${CASE_DIR}" && laserMeltFoam > "${run_log}" 2>&1); then
            RUN_OK=0
        fi
    else
        if ! (cd "${CASE_DIR}" && mpirun ${MPIRUN_FLAGS} -np "${N}" laserMeltFoam -parallel > "${run_log}" 2>&1); then
            RUN_OK=0
        fi
    fi
    END_WALL=$(date +%s%N)
    TOTAL_MS=$(( (END_WALL - START_WALL) / 1000000 ))

    # Distinguish "MPI never started" from "ran fully then crashed at finalize"
    # (LaserbeamFoam v2506 has a known dict double-free at shutdown).
    N_TIMING=$(grep -cE "\[\s*TIMING_(pd|T)\s*\]" "${run_log}" 2>/dev/null || echo 0)
    if [[ "${RUN_OK}" -eq 0 && "${N_TIMING}" -lt 4 ]]; then
        echo "  ERROR: laserMeltFoam never produced timing output (MPI start failure?)"
        echo "         see ${run_log}; common causes: not enough slots, scotch decomp"
        echo "${N},FAIL,0,0,0" >> "${SUMMARY}"
        rm -rf "${CASE_DIR}/processor"*
        continue
    fi
    if [[ "${RUN_OK}" -eq 0 ]]; then
        echo "  WARN: non-zero exit but ${N_TIMING} TIMING rows captured "\
             "(probably finalize-time crash, data is good)"
    fi

    # 6. Extract timings
    timing_csv="${RESULTS_DIR}/timing_${N}cores.csv"
    extract_timings "${run_log}" "${timing_csv}" "${N}"

    # Compute median pd and T times — split index FIXED (was [3], CSV has 3 cols)
    # Skip first ~5 steps (cold start, dt~1e-12, non-stationary)
    MED_PD=$(python3 -c "
import sys
vals = []
for line in open('${timing_csv}'):
    parts = line.strip().split(',')
    if len(parts) == 3 and parts[1] == 'pd':
        try: vals.append(float(parts[2]))
        except: pass
# drop first 15 pd entries (~5 steps × 3 correctors) to skip cold start
vals = vals[15:] if len(vals) > 30 else vals
vals.sort()
print(f'{vals[len(vals)//2]:.1f}' if vals else '0')
")
    MED_T=$(python3 -c "
import sys
vals = []
for line in open('${timing_csv}'):
    parts = line.strip().split(',')
    if len(parts) == 3 and parts[1] == 'T':
        try: vals.append(float(parts[2]))
        except: pass
vals = vals[5:] if len(vals) > 10 else vals
vals.sort()
print(f'{vals[len(vals)//2]:.1f}' if vals else '0')
")

    echo "  Results: N=${N}, wall=${TOTAL_MS}ms, median_pd=${MED_PD}ms, median_T=${MED_T}ms"
    echo "${N},total,${MED_PD},${MED_T},${TOTAL_MS}" >> "${SUMMARY}"

    # 7. Clean up processor dirs to save disk
    rm -rf "${CASE_DIR}/processor"*
done

# Restore original controlDict
sed -i "s/^\s*endTime\s.*/endTime     ${ORIG_END_TIME};/" "${CASE_DIR}/system/controlDict"

# Restore original decomposeParDict
if [[ -f "${RESULTS_DIR}/decomposeParDict.bak" ]]; then
    cp "${RESULTS_DIR}/decomposeParDict.bak" "${CASE_DIR}/system/decomposeParDict"
fi
# Restore dumper
if [[ -f "${RESULTS_DIR}/matrixDumperDict.bak" ]]; then
    cp "${RESULTS_DIR}/matrixDumperDict.bak" "${CASE_DIR}/system/matrixDumperDict"
fi

echo ""
echo "========================================"
echo "Scaling test complete!"
echo "Results: ${RESULTS_DIR}/summary.csv"
echo "========================================"

# Generate plot
python3 - "${RESULTS_DIR}" <<'PYEOF'
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir = sys.argv[1]
summary = os.path.join(results_dir, "summary.csv")

cores, pds, ts = [], [], []
with open(summary) as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 4 and parts[1] == "total":
            c = int(parts[0])
            pd_val = float(parts[2])
            t_val = float(parts[3])
            if pd_val > 0 and t_val > 0:  # skip zero entries
                cores.append(c)
                pds.append(pd_val)
                ts.append(t_val)

if not cores:
    print("No valid data to plot")
    sys.exit(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Speedup plot
base_pd, base_t = pds[0], ts[0]
speedup_pd = [base_pd / p for p in pds]
speedup_t = [base_t / t for t in ts]
ideal = [c / cores[0] for c in cores]

ax1.plot(cores, ideal, 'k--', label='ideal', alpha=0.5)
ax1.plot(cores, speedup_pd, 'bo-', label='pd speedup')
ax1.plot(cores, speedup_t, 'rs-', label='T speedup')
ax1.set_xlabel('Processors')
ax1.set_ylabel('Speedup vs 1-core')
ax1.set_title('Strong Scaling')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Wall time plot
ax2.plot(cores, pds, 'bo-', label='pd (median ms)')
ax2.plot(cores, ts, 'rs-', label='T (median ms)')
ax2.set_xlabel('Processors')
ax2.set_ylabel('Median solve time (ms)')
ax2.set_title('Per-solve wall time')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(results_dir, "scaling_plot.png")
fig.savefig(out, dpi=150)
print(f"Plot saved: {out}")

# Print optimal
best_pd_idx = pds.index(min(pds))
best_t_idx = ts.index(min(ts))
print(f"\nOptimal for pd: {cores[best_pd_idx]} cores ({pds[best_pd_idx]:.1f} ms)")
print(f"Optimal for T:  {cores[best_t_idx]} cores ({ts[best_t_idx]:.1f} ms)")
PYEOF
