#!/bin/bash
# run_everything.sh — Master script for the full experiment pipeline
#
# Run this on the LAB MACHINE after install.sh has been applied.
# It will:
#   1. Run scaling test (find optimal core count)
#   2. Collect 500+ matrices at 6 physical phases
#   3. Package results for transfer to GPU machine
#
# Usage:
#   ./run_everything.sh <case_dir> [options]
#
# Options:
#   --skip-scaling     Skip scaling test, use default NP
#   --np N             Processor count (default: auto from scaling test, or 1)
#   --every N          Dump every N steps (default: 2)
#   --skip-collect     Skip matrix collection (only run scaling)
#   --dry-run          Show what would be done, don't execute

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CASE_DIR="${1:?Usage: run_everything.sh <case_dir> [options]}"
shift

SKIP_SCALING=false
NP=""
EVERY=2
SKIP_COLLECT=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-scaling) SKIP_SCALING=true; shift ;;
        --np) NP="$2"; shift 2 ;;
        --every) EVERY="$2"; shift 2 ;;
        --skip-collect) SKIP_COLLECT=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  DILU-Research Full Experiment Pipeline"
echo "============================================"
echo ""
echo "  Case:       ${CASE_DIR}"
echo "  Skip scaling: ${SKIP_SCALING}"
echo "  NP:         ${NP:-auto}"
echo "  Every:      ${EVERY}"
echo "  Skip collect: ${SKIP_COLLECT}"
echo "  Dry run:    ${DRY_RUN}"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would execute:"
    echo "  1. scaling_test.sh ${CASE_DIR}"
    echo "  2. multi_timepoint_collect.sh ${CASE_DIR} --np <N> --every ${EVERY}"
    echo "  3. tar czf <case>_results.tar.gz ..."
    exit 0
fi

# ============================================================
# Step 1: Scaling test
# ============================================================
if ! $SKIP_SCALING; then
    echo ""
    echo "============================================"
    echo "  STEP 1: Scaling Test"
    echo "============================================"
    echo ""

    "${THIS_DIR}/scaling_test.sh" "${CASE_DIR}" \
        --steps 10 \
        --cores "1 2 4 8 16 24 32 40 48 56"

    # Extract optimal core count
    SUMMARY="${CASE_DIR}/scaling_results/summary.csv"
    if [[ -f "${SUMMARY}" ]]; then
        # Find the core count with lowest median pd time
        OPTIMAL=$(tail -n +2 "${SUMMARY}" | sort -t, -k3 -n | head -1 | cut -d, -f1)
        echo ""
        echo "  Optimal core count from scaling test: ${OPTIMAL}"
        if [[ -z "${NP}" ]]; then
            NP="${OPTIMAL}"
            echo "  Using ${NP} cores for data collection"
        fi
    fi
else
    echo "[SKIP] Scaling test"
fi

# Default NP if still empty
if [[ -z "${NP}" ]]; then
    NP=1
    echo "  WARNING: No NP specified, defaulting to serial (NP=1)"
fi

# ============================================================
# Step 2: Multi-timepoint collection
# ============================================================
if ! $SKIP_COLLECT; then
    echo ""
    echo "============================================"
    echo "  STEP 2: Multi-Timepoint Collection"
    echo "  NP=${NP}, Every=${EVERY}"
    echo "============================================"
    echo ""

    "${THIS_DIR}/multi_timepoint_collect.sh" "${CASE_DIR}" \
        --np "${NP}" \
        --every "${EVERY}"
else
    echo "[SKIP] Matrix collection"
fi

# ============================================================
# Step 3: Package results
# ============================================================
echo ""
echo "============================================"
echo "  STEP 3: Packaging Results"
echo "============================================"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CASE_NAME=$(basename "${CASE_DIR}")
PACKAGE="${CASE_DIR}/${CASE_NAME}_results_${TIMESTAMP}.tar.gz"

# Collect what to package
TAR_LIST=()
if [[ -d "${CASE_DIR}/scaling_results" ]]; then
    TAR_LIST+=("${CASE_NAME}/scaling_results")
fi
if [[ -d "${CASE_DIR}/collected_matrices" ]]; then
    TAR_LIST+=("${CASE_NAME}/collected_matrices")
fi

if [[ ${#TAR_LIST[@]} -gt 0 ]]; then
    (cd "$(dirname "${CASE_DIR}")" && tar czf "${PACKAGE}" "${TAR_LIST[@]}")
    echo "  Package: ${PACKAGE}"
    echo "  Size: $(du -h "${PACKAGE}" | cut -f1)"
else
    echo "  WARNING: No results to package"
fi

echo ""
echo "============================================"
echo "  COMPLETE!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Transfer ${PACKAGE} to GPU machine"
echo "  2. Extract: tar xzf $(basename "${PACKAGE}")"
echo "  3. Run precision experiment:"
echo "     python3 -m dilu.benchmark.openfoam_crosscheck.precision_experiment \\"
echo "         <path>/spot_melt_npz/"
echo "  4. Run global norm analysis:"
echo "     python3 dilu/benchmark/openfoam_crosscheck/global_norm_analysis.py \\"
echo "         <path>/spot_melt_npz/"
echo "  5. Generate distribution plots:"
echo "     python3 dilu/benchmark/openfoam_crosscheck/plot_distributions.py \\"
echo "         <path>/spot_melt_npz/"
