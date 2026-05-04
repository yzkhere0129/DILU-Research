#!/bin/bash
# multi_timepoint_collect.sh — Collect matrices at specific physical phases
#
# Runs laserMeltFoam with matrixDumper configured to extract matrices at
# specific physical time points corresponding to different LPBF regimes:
#
#   Phase 1: Initial heating       (first few steps)
#   Phase 2: Melt pool formation   (pool starts to form)
#   Phase 3: Recoil pressure onset (vapor depression begins)
#   Phase 4: Keyhole formation     (deep narrow cavity)
#   Phase 5: Quasi-steady state    (stable keyhole + trailing pool)
#   Phase 6: Cooling / solidification (laser off or trailing)
#
# Usage:
#   ./multi_timepoint_collect.sh <case_dir> [options]
#
# Options:
#   --np N              Number of processors (default: 1, serial)
#   --every N           Dump every N steps within each phase (default: 1)
#   --phases "1 2 3 4 5 6"  Which phases to collect (default: all)
#   --outdir DIR        Output directory (default: <case_dir>/collected_matrices)
#
# Prerequisites:
#   - OpenFOAM v2506 with laserMeltFoam patched (matrixDumper.H v2 + timing)
#   - Case set up with initial conditions
#   - Python3 with numpy, scipy (for npz conversion)
#
# Output structure:
#   <outdir>/
#   ├── phase1_initial/       matrices from initial heating
#   ├── phase2_melt_pool/     matrices from melt pool formation
#   ├── phase3_recoil/        matrices from recoil pressure onset
#   ├── phase4_keyhole/       matrices from keyhole formation
#   ├── phase5_steady/        matrices from quasi-steady state
#   ├── phase6_cooling/       matrices from cooling
#   ├── timings.csv           all pd/T wall times
#   └── summary.json          phase-level statistics

set -euo pipefail

# =========================================================================
# Arguments
# =========================================================================
CASE_DIR="${1:?Usage: multi_timepoint_collect.sh <case_dir> [options]}"
shift

NP=1
EVERY=1
PHASES="1 2 3 4 5 6"
OUTDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --np) NP="$2"; shift 2 ;;
        --every) EVERY="$2"; shift 2 ;;
        --phases) PHASES="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${OUTDIR}" ]]; then
    OUTDIR="${CASE_DIR}/collected_matrices"
fi
mkdir -p "${OUTDIR}"

echo "========================================"
echo "Multi-timepoint matrix collection"
echo "Case: ${CASE_DIR}"
echo "Processors: ${NP}"
echo "Every N steps: ${EVERY}"
echo "Phases: ${PHASES}"
echo "Output: ${OUTDIR}"
echo "========================================"

# =========================================================================
# Phase time ranges
# =========================================================================
# These are default ranges for the spot melt case (150W, 0.3ms pulse).
# Adjust based on your specific case's physics.
#
# The ranges are in simulation time (seconds). They overlap slightly to
# ensure no gap at transitions.
#
# To determine the right ranges for your case:
#   1. Run the simulation once without dumping
#   2. Check when melt pool forms (alpha.metal > 0.5 in >100 cells)
#   3. Check when recoil pressure kicks in (p_rgh spike)
#   4. Check when keyhole forms (deep narrow depression in alpha.metal)

declare -A PHASE_START PHASE_END PHASE_NAME

# Phase 1: Initial heating — first 5% of simulation
PHASE_START[1]="2.25322e-06"
PHASE_END[1]="2.35e-06"
PHASE_NAME[1]="initial_heating"

# Phase 2: Melt pool forming — liquid fraction growing
PHASE_START[2]="2.40e-06"
PHASE_END[2]="2.60e-06"
PHASE_NAME[2]="melt_pool_formation"

# Phase 3: Recoil pressure onset — vapor depression starts
PHASE_START[3]="2.65e-06"
PHASE_END[3]="2.85e-06"
PHASE_NAME[3]="recoil_pressure"

# Phase 4: Keyhole formation — deep narrow cavity
PHASE_START[4]="2.85e-06"
PHASE_END[4]="3.05e-06"
PHASE_NAME[4]="keyhole_formation"

# Phase 5: Quasi-steady state — stable keyhole + trailing pool
PHASE_START[5]="3.05e-06"
PHASE_END[5]="3.30e-06"
PHASE_NAME[5]="quasi_steady"

# Phase 6: Late / cooling — trailing edge
PHASE_START[6]="3.30e-06"
PHASE_END[6]="3.50e-06"
PHASE_NAME[6]="cooling"

# =========================================================================
# Generate matrixDumperDict with time ranges
# =========================================================================
generate_dumper_dict() {
    local phases="$1"
    local every="$2"
    local dict_path="${CASE_DIR}/system/matrixDumperDict"

    cat > "${dict_path}" <<'HEADER'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      matrixDumperDict;
}

enabled         true;
outputDir       "postProcessing/matrices";
equations       (pd T);
maxCorrectorsPerEq
{
    pd  3;
    T   5;
}

HEADER

    # Build dumpTimeRanges from selected phases
    echo "dumpTimeRanges" >> "${dict_path}"
    echo "(" >> "${dict_path}"
    for p in ${phases}; do
        echo "    (${PHASE_START[$p]} ${PHASE_END[$p]})" >> "${dict_path}"
    done
    echo ");" >> "${dict_path}"
    echo "" >> "${dict_path}"
    echo "everyNSteps     ${every};" >> "${dict_path}"

    echo "  Generated matrixDumperDict with phases: ${phases}"
}

# =========================================================================
# Run simulation
# =========================================================================
run_simulation() {
    local np="$1"
    local log_file="${OUTDIR}/solver.log"

    echo "  Running laserMeltFoam with ${np} processors..."

    if [[ "${np}" -eq 1 ]]; then
        (cd "${CASE_DIR}" && laserMeltFoam > "${log_file}" 2>&1)
    else
        (cd "${CASE_DIR}" && mpirun -np "${np}" laserMeltFoam -parallel > "${log_file}" 2>&1)
    fi

    echo "  Solver complete. Log: ${log_file}"
}

# =========================================================================
# Collect and organize matrices by phase
# =========================================================================
organize_by_phase() {
    local mat_dir="${CASE_DIR}/postProcessing/matrices"

    if [[ ! -d "${mat_dir}" ]]; then
        echo "  WARNING: No matrices found in ${mat_dir}"
        return
    fi

    echo "  Organizing matrices by phase..."

    for p in ${PHASES}; do
        local phase_dir="${OUTDIR}/${PHASE_NAME[$p]}"
        mkdir -p "${phase_dir}"

        local t_start="${PHASE_START[$p]}"
        local t_end="${PHASE_END[$p]}"

        # Copy matrices whose timestamp falls in [t_start, t_end)
        for ts_dir in "${mat_dir}"/[0-9]*; do
            local ts=$(basename "${ts_dir}")
            # Use python for float comparison
            local in_range=$(python3 -c "
t = float('${ts}')
print('yes' if ${t_start} <= t < ${t_end} else 'no')
")
            if [[ "${in_range}" == "yes" ]]; then
                cp -r "${ts_dir}" "${phase_dir}/"
            fi
        done

        local count=$(find "${phase_dir}" -name "A.mm" | wc -l)
        echo "  Phase ${p} (${PHASE_NAME[$p]}): ${count} matrices"
    done
}

# =========================================================================
# Extract timings
# =========================================================================
extract_timings() {
    local log_file="${OUTDIR}/solver.log"
    local timing_csv="${OUTDIR}/timings.csv"

    echo "step,time_s,eq,wall_ms" > "${timing_csv}"

    local step=0
    local cur_time=""
    while IFS= read -r line; do
        # Track timestep
        if echo "${line}" | grep -q "^Time = "; then
            cur_time=$(echo "${line}" | grep -oP '[0-9e.+-]+')
            step=$((step + 1))
        fi
        # Extract timing
        if echo "${line}" | grep -qE "\[\s*TIMING_pd\s*\]"; then
            local ms=$(echo "${line}" | grep -oP '[0-9]+\.[0-9]+' | head -1)
            echo "${step},${cur_time},pd,${ms}" >> "${timing_csv}"
        fi
        if echo "${line}" | grep -qE "\[\s*TIMING_T\s*\]"; then
            local ms=$(echo "${line}" | grep -oP '[0-9]+\.[0-9]+' | head -1)
            echo "${step},${cur_time},T,${ms}" >> "${timing_csv}"
        fi
    done < "${log_file}"

    echo "  Timings extracted to ${timing_csv}"
}

# =========================================================================
# Generate summary
# =========================================================================
generate_summary() {
    python3 - "${OUTDIR}" <<'PYEOF'
import json, os, sys
from pathlib import Path
from collections import defaultdict

outdir = Path(sys.argv[1])
summary = {}

for phase_dir in sorted(outdir.iterdir()):
    if not phase_dir.is_dir() or not phase_dir.name.startswith("phase"):
        continue

    a_files = list(phase_dir.rglob("A.mm"))
    pd_count = sum(1 for f in a_files if "pd_corr" in str(f))
    t_count = sum(1 for f in a_files if "T_corr" in str(f))

    summary[phase_dir.name] = {
        "pd_matrices": pd_count,
        "T_matrices": t_count,
        "total": pd_count + t_count,
    }

# Total
total_pd = sum(v["pd_matrices"] for v in summary.values())
total_t = sum(v["T_matrices"] for v in summary.values())

summary["total"] = {
    "pd_matrices": total_pd,
    "T_matrices": total_t,
    "total": total_pd + total_t,
}

out_file = outdir / "summary.json"
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary:")
for k, v in summary.items():
    print(f"  {k}: pd={v['pd_matrices']}, T={v['T_matrices']}, total={v['total']}")
print(f"\nWrote {out_file}")
PYEOF
}

# =========================================================================
# Convert to npz (optional, saves disk space)
# =========================================================================
convert_to_npz() {
    echo "  Converting MatrixMarket to npz..."
    # Use the existing shrink_dump.py if available
    if [[ -f "${DEPLOY_DIR}/shrink_dump.py" ]]; then
        for phase_dir in "${OUTDIR}"/phase*; do
            if [[ -d "${phase_dir}" ]]; then
                python3 "${DEPLOY_DIR}/shrink_dump.py" "${phase_dir}" --delete-mm 2>/dev/null || true
            fi
        done
    fi
}

# =========================================================================
# Main
# =========================================================================

# Save original state
if [[ -f "${CASE_DIR}/system/matrixDumperDict" ]]; then
    cp "${CASE_DIR}/system/matrixDumperDict" "${OUTDIR}/matrixDumperDict.bak"
fi

# Step 1: Generate dumper config
echo ""
echo "[1/5] Generating matrixDumperDict..."
generate_dumper_dict "${PHASES}" "${EVERY}"

# Step 2: Run simulation
echo ""
echo "[2/5] Running simulation..."
run_simulation "${NP}"

# Step 3: Organize matrices
echo ""
echo "[3/5] Organizing matrices by phase..."
organize_by_phase

# Step 4: Extract timings
echo ""
echo "[4/5] Extracting timings..."
extract_timings

# Step 5: Summary
echo ""
echo "[5/5] Generating summary..."
generate_summary

# Optional: convert to npz
# convert_to_npz

echo ""
echo "========================================"
echo "Collection complete!"
echo "Output: ${OUTDIR}"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Check summary.json for matrix counts per phase"
echo "  2. Transfer to GPU machine for solving"
echo "  3. Run: python3 -m dilu.benchmark.openfoam_crosscheck.driver_amgx \\"
echo "           --cfg classical_v_diagscaled classical_v_diagscaled_tight \\"
echo "           --amortize ${OUTDIR}/<phase>/"
