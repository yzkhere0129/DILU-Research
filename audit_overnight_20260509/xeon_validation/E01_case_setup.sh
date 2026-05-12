#!/bin/bash
# Idempotent + sanity-checked patch for E01 — enables solverInfo per-step
# logging and extends endTime so laserMeltFoam continues from latestTime
# with timing instrumentation.
#
# Usage:    bash E01_case_setup.sh [CASE_DIR]      (default ~/cases/single_track_dump)
# Verify:   diff system/controlDict.preE01 system/controlDict
# Revert:   cp system/controlDict.preE01 system/controlDict

set -euo pipefail

CASE_DIR="${1:-$HOME/cases/single_track_dump}"
cd "$CASE_DIR" || { echo "FAIL: cannot cd to $CASE_DIR"; exit 1; }
echo "=== Patching $CASE_DIR/system/controlDict ==="

# 1. Idempotency: if already patched, exit success without re-editing
if grep -q "^functions" system/controlDict 2>/dev/null && \
   grep -q "solverInfo" system/controlDict 2>/dev/null; then
    echo "[OK] controlDict already has solverInfo function — nothing to do"
    grep -E "^startFrom|^endTime" system/controlDict
    exit 0
fi

# 2. Sanity: expected pre-state (endTime=1.2e-6 and NO functions block yet)
if ! grep -qE "^endTime\s+1\.2e-6" system/controlDict; then
    echo "FAIL: expected endTime=1.2e-6 in controlDict; aborting to avoid corrupting unknown state"
    grep "^endTime" system/controlDict
    exit 1
fi
if grep -qE "^functions" system/controlDict; then
    echo "FAIL: controlDict already has a functions block (not solverInfo); manual merge needed"
    exit 1
fi

# 3. Backup once
if [[ ! -f system/controlDict.preE01 ]]; then
    cp system/controlDict system/controlDict.preE01
    echo "[OK] backup saved as system/controlDict.preE01"
fi

# 4. Edit startFrom -> latestTime  (so laserMeltFoam continues from t=1.2e-6)
sed -i 's/^startFrom\s\+startTime/startFrom         latestTime/' system/controlDict

# 5. Extend endTime  1.2e-6 -> 1.5e-6
sed -i 's/^endTime\s\+1\.2e-6/endTime           1.5e-6/' system/controlDict

# 6. Append solverInfo functionObject — per-timestep pd and T iter+residual log
cat >> system/controlDict <<'EOF'

// === E01 timing instrumentation (audit_overnight_20260509, 2026-05-12) ===
functions
{
    solverInfo
    {
        type            solverInfo;
        libs            (utilityFunctionObjects);
        fields          (pd T);
        writeResidualFields off;
        executeControl  timeStep;
        executeInterval 1;
        writeControl    timeStep;
        writeInterval   1;
    }
}
EOF

echo
echo "=== Diff (preE01 → patched) ==="
diff system/controlDict.preE01 system/controlDict || true

echo
echo "=== Verification ==="
grep -E "^startFrom|^endTime" system/controlDict
grep -c "solverInfo" system/controlDict && echo "[OK] solverInfo references present"

echo
echo "=== Next step ==="
echo "  cd $CASE_DIR"
echo "  nohup laserMeltFoam > log.E01 2>&1 &"
echo "  echo \"E01 PID: \$!\""
echo
echo "Expected wall: 1-2h for ~300 timesteps (t=1.2e-6 → t=1.5e-6 at dt=1ns)."
echo "Outputs:"
echo "  log.E01"
echo "  postProcessing/solverInfo/<startTime>/solverInfo.dat  ← per-step iter+residual CSV"
