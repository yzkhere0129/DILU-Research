#!/bin/bash
# 500K dense dump — 384 timesteps × pd_corr0 for amortized replay experiment.
#
# vs single_track_dump (which did 6 timesteps × 4 eq = 24 dumps):
#   - 384 dumps spanning 50→1199 ns at 3 ns intervals
#   - Only pd (corr0 only) — saves IO/storage (~4 GB raw, ~1.5 GB tar.gz)
#   - Otherwise identical: 50×200×50 mesh, dx=4μm, 300W laser, 1.2μs end
#   - Estimated total wall: ~7h (6.7h baseline + ~10 min dump IO overhead)
#
# Run on lab Xeon (after kill-ing 2M overnight if still running):
#   cp -r ~/src/laserMeltFoam/Tutorials/LPBF_tutorial ~/cases/dense_track_dump_500K
#   cd ~/cases/dense_track_dump_500K
#   bash ~/DILU-Research/dilu/openfoam_patches/dense_track_dump_500K/run_dense_extraction.sh

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_TRACK_PATCH="$(cd "$PATCH_DIR/.." && pwd)/single_track_dump"
echo "Patch dir       : $PATCH_DIR"
echo "Reusing from    : $SINGLE_TRACK_PATCH"
echo "Case dir        : $(pwd)"
echo

# ----- sanity -----
[[ -f system/blockMeshDict ]] || { echo "FAIL: not in case dir"; exit 1; }
[[ -d initial ]]               || { echo "FAIL: missing initial/"; exit 1; }
command -v laserMeltFoam >/dev/null || { echo "FAIL: source OF v2412 bashrc"; exit 1; }
command -v setSolidFraction >/dev/null || { echo "FAIL: setSolidFraction not built"; exit 1; }

# ===========================================================================
# Stage 1: bump laser 150→300W (idempotent)
# ===========================================================================
echo "=== Stage 1: laser power → 300W ==="
sed -i 's/(0            150)/(0            300)/'   constant/timeVsLaserPower
sed -i 's/(2e-8         150)/(2e-8         300)/'   constant/timeVsLaserPower
sed -i 's/(600e-6       150)/(600e-6       300)/'   constant/timeVsLaserPower
grep -E "300|^$" constant/timeVsLaserPower | head -5

# ===========================================================================
# Stage 2: mesh 50×200×50 (same as single_track)
# ===========================================================================
echo
echo "=== Stage 2: mesh 50×200×50 = 500K ==="
sed -i 's/(80 320 80)/(50 200 50)/' system/blockMeshDict
grep "hex (0 1" system/blockMeshDict

# ===========================================================================
# Stage 3: install controlDict (reuse single_track) + DENSE matrixDumperDict
# ===========================================================================
echo
echo "=== Stage 3: configs ==="
[[ -f system/controlDict.orig ]] || cp system/controlDict system/controlDict.orig
cp "$SINGLE_TRACK_PATCH/controlDict"            system/controlDict
cp "$PATCH_DIR/matrixDumperDict.dense"          system/matrixDumperDict

echo "  controlDict endTime/dt:"
grep -E "endTime|deltaT|adjustTimeStep" system/controlDict
echo "  matrixDumperDict (first 5 dump steps):"
grep -A1 dumpTimeSteps system/matrixDumperDict | head -2
echo "  total dump count: $(awk '/dumpTimeSteps/,/);/' system/matrixDumperDict | grep -oE '[0-9]+' | grep -v '^0$' | wc -l)"

# ===========================================================================
# Stage 4: smoke test (6 ns, ~30s on 500K)
# ===========================================================================
echo
echo "=== Stage 4: smoke test (6 ns) ==="
# Temporarily override endTime + dump only step 5 for quick verify
cp system/matrixDumperDict system/matrixDumperDict.dense_full
cp "$SINGLE_TRACK_PATCH/matrixDumperDict.smoke" system/matrixDumperDict
sed -i 's/endTime           1.2e-6/endTime           6e-9/' system/controlDict

rm -rf 0 [0-9]*[eE]-* 0.* postProcessing log.run log.smoke VTKs 2>/dev/null
cp -r initial 0
echo "  blockMesh ..."
blockMesh > log.blockMesh 2>&1; tail -2 log.blockMesh
echo "  setSolidFraction ..."
setSolidFraction > log.setSolidFraction 2>&1; tail -2 log.setSolidFraction
echo "  laserMeltFoam smoke ..."
laserMeltFoam > log.smoke 2>&1
tail -5 log.smoke

echo
grep "matrixDumper" log.smoke | head -10 || echo "(no matrixDumper output)"

python3 ~/DILU-Research/dilu/openfoam_patches/single_track_dump/verify_dump.py
verify_rc=$?
[[ $verify_rc -ne 0 ]] && { echo "✗ smoke FAILED. Inspect log.smoke."; exit $verify_rc; }

# ===========================================================================
# Stage 5: install full dense config + start nohup
# ===========================================================================
echo
echo "=== Stage 5: configure full 1.2μs dense run ==="
cp system/matrixDumperDict.dense_full system/matrixDumperDict
cp "$SINGLE_TRACK_PATCH/controlDict" system/controlDict
echo "  endTime/dt:"
grep -E "endTime|deltaT" system/controlDict
echo "  dumpTimeSteps count: $(awk '/dumpTimeSteps/,/);/' system/matrixDumperDict | grep -oE '[0-9]+' | grep -v '^0$' | wc -l)"

rm -rf 0 [0-9]*[eE]-* 0.* postProcessing log.run VTKs 2>/dev/null
cp -r initial 0
setSolidFraction > log.setSolidFraction 2>&1; tail -2 log.setSolidFraction

echo
echo "=== starting nohup (~7h estimated for 1200 steps + 384 dumps) ==="
nohup laserMeltFoam > log.run 2>&1 &
PID=$!
disown
echo "  PID: $PID"
echo
echo "Monitor:"
echo "  tail -f $(pwd)/log.run | grep -E 'Time = |matrixDumper|min/max\(T\)'"
echo
echo "Progress check:"
echo "  ls $(pwd)/postProcessing/matrices/ | wc -l   # should hit 384 when done"
echo
echo "After completion:"
echo "  tar -czf /tmp/dense_track_dump_500K.tgz \\"
echo "      constant/polyMesh \\"
echo "      postProcessing/matrices"
