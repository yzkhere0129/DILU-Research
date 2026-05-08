#!/bin/bash
# One-shot LPBF matrix extraction:
#   1. Bump laser power 150 → 300W
#   2. Drop in our controlDict + matrixDumperDict
#   3. Smoke test at 6 ns (verify dump pipeline)
#   4. If smoke passes, kick off full 1.2μs run in nohup
#
# Run this from inside the case dir, e.g.:
#   cd ~/cases/single_track_dump
#   bash ~/DILU-Research/dilu/openfoam_patches/single_track_dump/run_extraction.sh
#
# Stages on failure: aborts with informative message. Idempotent on Stages 1-3.

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Patch dir: $PATCH_DIR"
echo "Case dir : $(pwd)"
echo

# ----- Sanity checks -----
[[ -f system/blockMeshDict ]] || { echo "FAIL: not in a case dir (no system/blockMeshDict)"; exit 1; }
[[ -f constant/timeVsLaserPower ]] || { echo "FAIL: missing constant/timeVsLaserPower"; exit 1; }
[[ -d initial ]] || { echo "FAIL: missing initial/ template (need this for 0/)"; exit 1; }
command -v laserMeltFoam >/dev/null || { echo "FAIL: laserMeltFoam not in PATH — source bashrc"; exit 1; }
command -v setSolidFraction >/dev/null || { echo "FAIL: setSolidFraction not in PATH"; exit 1; }
command -v blockMesh >/dev/null || { echo "FAIL: blockMesh not in PATH"; exit 1; }

# ===========================================================================
# Stage 1: Bump laser power 150 → 300W
# ===========================================================================
echo "=== Stage 1: bump laser power → 300W ==="
sed -i 's/(0            150)/(0            300)/' constant/timeVsLaserPower
sed -i 's/(2e-8         150)/(2e-8         300)/' constant/timeVsLaserPower
sed -i 's/(600e-6       150)/(600e-6       300)/' constant/timeVsLaserPower

if grep -q "300" constant/timeVsLaserPower; then
    echo "  Power lines after edit:"
    grep -E "^\s*\(.*[0-9].*\s+(150|300)\s*\)" constant/timeVsLaserPower | head -5
else
    echo "WARN: no 300 found; maybe already patched. Current file:"
    cat constant/timeVsLaserPower
fi

# ===========================================================================
# Stage 2: Drop in our controlDict + smoke matrixDumperDict
# ===========================================================================
echo
echo "=== Stage 2: install controlDict + smoke matrixDumperDict ==="

# Backup originals
[[ -f system/controlDict.orig ]]      || cp system/controlDict      system/controlDict.orig
[[ -f system/matrixDumperDict.full ]] || true   # may not exist

cp "$PATCH_DIR/controlDict"             system/controlDict
cp "$PATCH_DIR/matrixDumperDict.smoke"  system/matrixDumperDict
cp "$PATCH_DIR/matrixDumperDict.full"   system/matrixDumperDict.full

# For smoke, override endTime to 6 ns (covers step 5)
sed -i 's/endTime           1.2e-6/endTime           6e-9/' system/controlDict

echo "  controlDict endTime override:"
grep "endTime" system/controlDict
echo "  matrixDumperDict (smoke):"
grep "dumpTimeSteps" system/matrixDumperDict

# ===========================================================================
# Stage 3: Smoke test (6 ns)
# ===========================================================================
echo
echo "=== Stage 3: smoke test (6 ns) ==="

# Clean state
rm -rf 0 [0-9]*[eE]-* 0.* postProcessing log.run log.smoke VTKs 2>/dev/null
cp -r initial 0
setSolidFraction 2>&1 | tail -3

echo "  starting laserMeltFoam (smoke, ~30 s)..."
laserMeltFoam > log.smoke 2>&1
echo "  smoke run completed, last 5 lines:"
tail -5 log.smoke

echo
echo "  --- matrixDumper messages ---"
grep -E "matrixDumper" log.smoke | head -10 || echo "  (none — likely a problem)"

# ===========================================================================
# Stage 4: Verify smoke dump
# ===========================================================================
echo
echo "=== Stage 4: verify dump correctness ==="
python3 "$PATCH_DIR/verify_dump.py"
verify_rc=$?

if [[ $verify_rc -ne 0 ]]; then
    echo
    echo "✗ Smoke verification FAILED (rc=$verify_rc)."
    echo "  Inspect log.smoke / postProcessing/matrices/5e-09/ before retrying."
    exit $verify_rc
fi

# ===========================================================================
# Stage 5: Reset for full run + kick off in background
# ===========================================================================
echo
echo "=== Stage 5: configure full 1.2μs run ==="

cp "$PATCH_DIR/controlDict" system/controlDict
cp system/matrixDumperDict.full system/matrixDumperDict

echo "  endTime / deltaT / adjustTimeStep:"
grep -E "endTime|deltaT|adjustTimeStep" system/controlDict
echo "  matrixDumperDict (full):"
grep "dumpTimeSteps" system/matrixDumperDict

# Clean smoke run state
rm -rf 0 [0-9]*[eE]-* 0.* postProcessing log.run VTKs 2>/dev/null
cp -r initial 0
setSolidFraction 2>&1 | tail -3

echo
echo "=== Starting full 1.2μs run in nohup (~2 hours) ==="
nohup laserMeltFoam > log.run 2>&1 &
PID=$!
disown
echo "  PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $(pwd)/log.run | grep -E 'Time = |matrixDumper|min/max\(T\)'"
echo
echo "Expected dump files (after ~2 hours):"
echo "  postProcessing/matrices/{3.2,3.8,4.1,7,9,10.6}e-{07,07,07,07,07,07} / {pd_corr0,1,2 + T_corr0}"
echo
echo "After completion, package with:"
echo "  tar -czf /tmp/single_track_dump.tgz constant/polyMesh postProcessing/matrices"
