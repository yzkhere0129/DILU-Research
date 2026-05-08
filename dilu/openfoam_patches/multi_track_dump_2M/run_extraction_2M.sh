#!/bin/bash
# 2M single-core LPBF matrix extraction for overnight run.
#
# vs single_track_dump (500K):
#   - mesh: 80×320×80 = 2,048,000 cells (vs 50×200×50 = 500K)
#   - dx: 2.5μm (vs 4μm) — finer
#   - 24 dump points (vs 6)
#   - logs solve timings via solverInfo function object
#   - estimated wall: 30-40 hours single-core for full 1.2μs (won't finish in one
#     night — kill at morning, partial data still useful)
#
# Run from inside case dir (a fresh copy of LPBF_tutorial):
#   cp -r ~/src/laserMeltFoam/Tutorials/LPBF_tutorial ~/cases/multi_track_dump_2M
#   cd ~/cases/multi_track_dump_2M
#   bash ~/DILU-Research/dilu/openfoam_patches/multi_track_dump_2M/run_extraction_2M.sh

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Patch dir: $PATCH_DIR"
echo "Case dir : $(pwd)"
echo

# ----- sanity checks -----
[[ -f system/blockMeshDict ]] || { echo "FAIL: not in case dir"; exit 1; }
[[ -d initial ]] || { echo "FAIL: missing initial/"; exit 1; }
command -v laserMeltFoam >/dev/null || { echo "FAIL: source OF v2412 bashrc"; exit 1; }
command -v setSolidFraction >/dev/null || { echo "FAIL: setSolidFraction not built"; exit 1; }

# ===========================================================================
# Stage 1: Bump laser power 150 → 300W (idempotent; same as single_track)
# ===========================================================================
echo "=== Stage 1: laser power → 300W ==="
sed -i 's/(0            150)/(0            300)/' constant/timeVsLaserPower
sed -i 's/(2e-8         150)/(2e-8         300)/' constant/timeVsLaserPower
sed -i 's/(600e-6       150)/(600e-6       300)/' constant/timeVsLaserPower
grep -E "300|^$" constant/timeVsLaserPower | head -5

# ===========================================================================
# Stage 2: Mesh — keep tutorial default 80×320×80 = 2M (just verify)
# ===========================================================================
echo
echo "=== Stage 2: mesh (80×320×80 = 2M cells) ==="
grep "hex (0 1" system/blockMeshDict
# Should be (80 320 80). If user changed it elsewhere, reset:
sed -i 's/(50 200 50)/(80 320 80)/' system/blockMeshDict
sed -i 's/(40 160 40)/(80 320 80)/' system/blockMeshDict   # safety
grep "hex (0 1" system/blockMeshDict

# ===========================================================================
# Stage 3: Drop in our controlDict + matrixDumperDict
# ===========================================================================
echo
echo "=== Stage 3: install configs ==="
[[ -f system/controlDict.orig ]] || cp system/controlDict system/controlDict.orig

cp "$PATCH_DIR/controlDict"             system/controlDict
cp "$PATCH_DIR/matrixDumperDict.smoke"  system/matrixDumperDict
cp "$PATCH_DIR/matrixDumperDict.full"   system/matrixDumperDict.full

# Smoke override
sed -i 's/endTime           1.2e-6/endTime           6e-9/' system/controlDict

echo "  endTime override:"
grep endTime system/controlDict
echo "  smoke dumpTimeSteps:"
grep dumpTimeSteps system/matrixDumperDict
echo "  function object solverInfo:"
grep -A2 "type            solverInfo" system/controlDict

# ===========================================================================
# Stage 4: Smoke test (6 ns; ~5-15 minutes on 2M single-core)
# ===========================================================================
echo
echo "=== Stage 4: smoke test (6 ns, 2M single-core, ~5-15 min) ==="

rm -rf 0 [0-9]*[eE]-* 0.* postProcessing log.run log.smoke VTKs 2>/dev/null
cp -r initial 0
echo "  generating mesh ..."
blockMesh > log.blockMesh 2>&1
tail -3 log.blockMesh
echo "  initializing alpha.material ..."
setSolidFraction > log.setSolidFraction 2>&1
tail -3 log.setSolidFraction
echo "  smoke run ..."
laserMeltFoam > log.smoke 2>&1
tail -5 log.smoke

echo
grep "matrixDumper" log.smoke | head -10 || echo "  (no matrixDumper output)"

# ===========================================================================
# Stage 5: Verify smoke dump
# ===========================================================================
echo
echo "=== Stage 5: verify smoke dump ==="
python3 ~/DILU-Research/dilu/openfoam_patches/single_track_dump/verify_dump.py
verify_rc=$?

if [[ $verify_rc -ne 0 ]]; then
    echo
    echo "✗ smoke FAILED. Inspect log.smoke before retry."
    exit $verify_rc
fi

# ===========================================================================
# Stage 6: switch to full config + start overnight nohup
# ===========================================================================
echo
echo "=== Stage 6: configure full 1.2μs run ==="
cp "$PATCH_DIR/controlDict" system/controlDict
cp system/matrixDumperDict.full system/matrixDumperDict

grep -E "endTime|deltaT" system/controlDict
echo "  full dumpTimeSteps (24 points):"
grep dumpTimeSteps system/matrixDumperDict

# Clean smoke artefacts and re-init
rm -rf 0 [0-9]*[eE]-* 0.* postProcessing log.run VTKs 2>/dev/null
cp -r initial 0
setSolidFraction > log.setSolidFraction 2>&1
tail -2 log.setSolidFraction

echo
echo "=== Starting full run in nohup ==="
echo "  WARNING: 2M single-core estimated 30-40h for full 1.2μs."
echo "  Partial run is fine — OF writes dumps at each matched step,"
echo "  so killing in the morning still gives data up to that point."
nohup laserMeltFoam > log.run 2>&1 &
PID=$!
disown
echo "  PID: $PID"
echo
echo "Monitor:"
echo "  tail -f $(pwd)/log.run | grep -E 'Time = |matrixDumper|min/max\(T\)'"
echo
echo "Periodically check progress:"
echo "  ls $(pwd)/postProcessing/matrices/"
echo "  grep '^Time = ' $(pwd)/log.run | tail -3"
echo
echo "After (or before) full completion, parse timings:"
echo "  python3 ~/DILU-Research/dilu/openfoam_patches/multi_track_dump_2M/parse_solve_timings.py log.run"
echo
echo "Package partial/full data:"
echo "  tar -czf /tmp/multi_track_dump_2M.tgz constant/polyMesh postProcessing/matrices postProcessing/solverInfo postProcessing/solve_timings_*"
