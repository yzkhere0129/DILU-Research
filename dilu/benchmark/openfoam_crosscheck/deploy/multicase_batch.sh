#!/bin/bash
# multicase_batch.sh — fully automated multi-case matrix dump.
#
# For each case in CASES, this script:
#   1. Calls patch_case.sh to attach matrixDumperDict
#   2. Runs laserMeltFoam (saves to log.dump_<case>)
#   3. Runs sanity.py and saves output
#   4. Optionally runs shrink_dump.py to convert MM→NPZ
#   5. Tars only the npz + sanity reports for transfer
#
# Designed to be run unattended overnight on the lab machine.
#
# Usage:
#   1. Edit CASES array below to point to your case directories
#   2. Edit N_STEPS to control dump count
#   3. ./multicase_batch.sh
#
# All output ends up in ~/multicase_dumps_<date>.tar
#
# Dependencies:
#   - install.sh already run (laserMeltFoam binary has matrixDumper)
#   - python3 with numpy + scipy (for sanity + shrink_dump)
#   - DILU-Research deploy/ in $DEPLOY_DIR

set -euo pipefail

# ============================================================
# CONFIG — EDIT BEFORE RUNNING
# ============================================================

# List of case directories. Each must have system/, constant/, initial/.
# Cases should already have run (have a numeric time folder with pd, T, U, alpha.metal).
CASES=(
    # "$HOME/cases/spot_melt_150W"
    # "$HOME/cases/spot_melt_250W"
    # "$HOME/cases/spot_melt_400W"
    # "$HOME/cases/scan_speed_0.5"
    # "$HOME/cases/scan_speed_1.0"
    # add more...
)

# Steps to dump per case (each step ~5 matrices × ~500 MB ASCII)
N_STEPS=50

# Path to the deploy/ dir (where install.sh / patch_case.sh / sanity.py / shrink_dump.py live)
DEPLOY_DIR="$HOME/deploy"

# Run shrink_dump.py after each case to convert MM → NPZ (saves 9× disk)?
DO_SHRINK=1

# Use N parallel workers in shrink_dump.py
SHRINK_WORKERS=16

# ============================================================
# END CONFIG
# ============================================================

if [ ${#CASES[@]} -eq 0 ]; then
    echo "ERROR: CASES array is empty. Edit this script first."
    exit 1
fi

if [ ! -x "$DEPLOY_DIR/patch_case.sh" ]; then
    echo "ERROR: $DEPLOY_DIR/patch_case.sh not found or not executable"
    exit 1
fi

# Verify OpenFOAM env
if [ -z "${WM_PROJECT_VERSION:-}" ]; then
    echo "ERROR: OpenFOAM env not sourced. Run:"
    echo "  source /usr/lib/openfoam/openfoam${WM_PROJECT_VERSION:-2506}/etc/bashrc"
    exit 1
fi

# Verify laserMeltFoam binary has matrixDumper (see if matrixDumper symbol exists)
if ! laserMeltFoam --help 2>&1 | head -3 > /dev/null; then
    echo "ERROR: laserMeltFoam not found in PATH"
    exit 1
fi

OUT_TAR="$HOME/multicase_dumps_$(date +%Y%m%d_%H%M).tar"
SUMMARY="$HOME/multicase_summary.txt"
{
    echo "==================================================="
    echo "Multi-case batch started: $(date)"
    echo "Cases: ${#CASES[@]}"
    echo "Steps per case: $N_STEPS"
    echo "==================================================="
} | tee "$SUMMARY"

ALL_NPZ_DIRS=()
ALL_SANITY=()

for CASE in "${CASES[@]}"; do
    CASE=$(realpath "$CASE")
    NAME=$(basename "$CASE")
    echo "" | tee -a "$SUMMARY"
    echo "=== [$(date +%H:%M:%S)] Processing $NAME ===" | tee -a "$SUMMARY"

    if [ ! -d "$CASE/system" ]; then
        echo "  SKIP: $CASE has no system/" | tee -a "$SUMMARY"
        continue
    fi

    # Step 1: patch case
    echo "  [patch] applying matrixDumperDict ($N_STEPS steps)" | tee -a "$SUMMARY"
    "$DEPLOY_DIR/patch_case.sh" "$CASE" "$N_STEPS" > "$CASE/log.patch_$NAME" 2>&1

    # Step 2: run laserMeltFoam
    echo "  [run]   laserMeltFoam (log: $CASE/log.dump_$NAME)" | tee -a "$SUMMARY"
    cd "$CASE"
    if ! laserMeltFoam > "log.dump_$NAME" 2>&1; then
        echo "  FAILED: see log.dump_$NAME" | tee -a "$SUMMARY"
        # Restore controlDict
        [ -f system/controlDict.orig ] && mv system/controlDict.orig system/controlDict
        continue
    fi

    # Step 3: sanity check
    echo "  [check] sanity.py" | tee -a "$SUMMARY"
    python3 "$DEPLOY_DIR/sanity.py" postProcessing/matrices > "sanity_$NAME.txt" 2>&1
    SANITY_TAIL=$(tail -1 "sanity_$NAME.txt")
    echo "  $SANITY_TAIL" | tee -a "$SUMMARY"
    ALL_SANITY+=("$CASE/sanity_$NAME.txt")

    # Step 4: optional shrink to NPZ
    if [ "$DO_SHRINK" = "1" ]; then
        echo "  [shrink] MM → NPZ ($SHRINK_WORKERS workers)" | tee -a "$SUMMARY"
        NPZ_DIR="$HOME/${NAME}_npz"
        mkdir -p "$NPZ_DIR"
        python3 "$DEPLOY_DIR/shrink_dump.py" \
            postProcessing/matrices "$NPZ_DIR" "$SHRINK_WORKERS" \
            > "$CASE/log.shrink_$NAME" 2>&1
        N_NPZ=$(find "$NPZ_DIR" -name "data.npz" | wc -l)
        SIZE=$(du -sh "$NPZ_DIR" 2>/dev/null | awk '{print $1}')
        echo "  npz: $N_NPZ matrices, $SIZE" | tee -a "$SUMMARY"
        ALL_NPZ_DIRS+=("$NPZ_DIR")
    fi

    # Step 5: restore controlDict
    [ -f system/controlDict.orig ] && mv system/controlDict.orig system/controlDict
    [ -f system/matrixDumperDict ] && rm -f system/matrixDumperDict

    cd - > /dev/null
done

# Tar everything for transport
echo "" | tee -a "$SUMMARY"
echo "=== Packing for transport: $OUT_TAR ===" | tee -a "$SUMMARY"
TAR_ITEMS=()
for d in "${ALL_NPZ_DIRS[@]}"; do
    TAR_ITEMS+=("${d#$HOME/}")
done
for s in "${ALL_SANITY[@]}"; do
    TAR_ITEMS+=("${s#$HOME/}")
done
TAR_ITEMS+=("multicase_summary.txt")

cd "$HOME"
if [ ${#TAR_ITEMS[@]} -gt 0 ]; then
    tar cf "$OUT_TAR" "${TAR_ITEMS[@]}" 2>/dev/null || \
        echo "WARN: some items not packed (paths may have changed)"
    echo "tar size: $(du -sh $OUT_TAR | awk '{print $1}')"
    echo "Done: $OUT_TAR"
else
    echo "ERROR: nothing to tar (all cases failed?)"
fi

echo "" | tee -a "$SUMMARY"
echo "=== Multi-case batch finished: $(date) ===" | tee -a "$SUMMARY"
