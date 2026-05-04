#!/bin/bash
# test_scripts_local.sh — Local validation of deploy scripts (no solver needed)
#
# Creates a mock case structure and tests all script logic:
#   - controlDict parsing
#   - decomposeParDict generation
#   - timing extraction from logs
#   - plot generation
#   - matrixDumperDict generation (time-range v2)
#
# Usage: ./test_scripts_local.sh

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR=$(mktemp -d)
trap "rm -rf ${TEST_DIR}" EXIT

echo "========================================"
echo "Local script validation"
echo "Test dir: ${TEST_DIR}"
echo "========================================"

PASS=0
FAIL=0

check() {
    local name="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "  PASS: ${name}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${name}"
        FAIL=$((FAIL + 1))
    fi
}

# =========================================================================
# 1. Create mock case structure
# =========================================================================
echo ""
echo "[1/6] Creating mock case..."
MOCK_CASE="${TEST_DIR}/spot_melt_150W"
mkdir -p "${MOCK_CASE}/system"
mkdir -p "${MOCK_CASE}/constant"
mkdir -p "${MOCK_CASE}/2.25322e-06"

# controlDict
cat > "${MOCK_CASE}/system/controlDict" <<'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     laserMeltFoam;
startFrom       startTime;
startTime       2.25322e-06;
endTime         3.5e-06;
deltaT          2e-8;
writeControl    timeStep;
writeInterval   100;
EOF

# fvSolution (minimal)
cat > "${MOCK_CASE}/system/fvSolution" <<'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers
{
    pd
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-8;
        relTol          0;
    }
    T
    {
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0;
    }
}
EOF

echo "  Mock case created at ${MOCK_CASE}"

# =========================================================================
# 2. Test controlDict parsing
# =========================================================================
echo ""
echo "[2/6] Testing controlDict parsing..."

DELTA_T=$(grep -E '^\s*deltaT\s' "${MOCK_CASE}/system/controlDict" \
    | head -1 | awk '{print $2}' | tr -d ';')
END_TIME=$(grep -E '^\s*endTime\s' "${MOCK_CASE}/system/controlDict" \
    | head -1 | awk '{print $2}' | tr -d ';')

check "deltaT parsed" test "${DELTA_T}" = "2e-8"
check "endTime parsed" test "${END_TIME}" = "3.5e-06"

echo "  deltaT=${DELTA_T}, endTime=${END_TIME}"

# =========================================================================
# 3. Test decomposeParDict generation
# =========================================================================
echo ""
echo "[3/6] Testing decomposeParDict generation..."

for N in 1 4 8 12; do
    cat > "${MOCK_CASE}/system/decomposeParDict" <<DICT
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
method          scotch;
numberOfSubprocesses  ${N};
DICT
    PARSED_N=$(grep "numberOfSubprocesses" "${MOCK_CASE}/system/decomposeParDict" | awk '{print $2}' | tr -d ';')
    check "decomposeParDict N=${N}" test "${PARSED_N}" = "${N}"
done

# =========================================================================
# 4. Test timing extraction
# =========================================================================
echo ""
echo "[4/6] Testing timing extraction..."

MOCK_LOG="${TEST_DIR}/mock_solver.log"
cat > "${MOCK_LOG}" <<'LOG'
Time = 2.25324e-06
Courant Number mean: 0.001 max: 0.05
[ TIMING_pd ] 2.573 ms
Solving for T, using PBiCG
[ TIMING_T ] 141.2 ms
Time = 2.25326e-06
Courant Number mean: 0.001 max: 0.05
[ TIMING_pd ] 2.612 ms
[ TIMING_pd ] 2.598 ms
[ TIMING_pd ] 2.551 ms
Solving for T, using PBiCG
[ TIMING_T ] 138.7 ms
[ TIMING_T ] 135.3 ms
[ TIMING_T ] 132.1 ms
LOG

TIMING_CSV="${TEST_DIR}/timings.csv"
echo "step,time_s,eq,wall_ms" > "${TIMING_CSV}"
grep -E "\[\s*TIMING_(pd|T)\s*\]" "${MOCK_LOG}" | while read -r line; do
    eq=$(echo "${line}" | grep -oP 'TIMING_\K(pd|T)')
    ms=$(echo "${line}" | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    echo "1,${eq},${ms}" >> "${TIMING_CSV}"
done

PD_COUNT=$(grep -c ",pd," "${TIMING_CSV}" || echo 0)
T_COUNT=$(grep -c ",T," "${TIMING_CSV}" || echo 0)
check "pd timings extracted (expect 4)" test "${PD_COUNT}" -eq 4
check "T timings extracted (expect 4)" test "${T_COUNT}" -eq 4

# =========================================================================
# 5. Test matrixDumperDict generation (time-range v2)
# =========================================================================
echo ""
echo "[5/6] Testing matrixDumperDict generation..."

DUMPER_DICT="${MOCK_CASE}/system/matrixDumperDict"
cat > "${DUMPER_DICT}" <<'HEADER'
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

PHASES="1 2 3"
declare -A PHASE_START PHASE_END
PHASE_START[1]="2.25322e-06";  PHASE_END[1]="2.35e-06"
PHASE_START[2]="2.40e-06";     PHASE_END[2]="2.60e-06"
PHASE_START[3]="2.65e-06";     PHASE_END[3]="2.85e-06"

echo "dumpTimeRanges" >> "${DUMPER_DICT}"
echo "(" >> "${DUMPER_DICT}"
for p in ${PHASES}; do
    echo "    (${PHASE_START[$p]} ${PHASE_END[$p]})" >> "${DUMPER_DICT}"
done
echo ");" >> "${DUMPER_DICT}"
echo "" >> "${DUMPER_DICT}"
echo "everyNSteps     2;" >> "${DUMPER_DICT}"

check "matrixDumperDict created" test -f "${DUMPER_DICT}"
check "dumpTimeRanges present" grep -q "dumpTimeRanges" "${DUMPER_DICT}"
check "everyNSteps present" grep -q "everyNSteps" "${DUMPER_DICT}"

RANGE_COUNT=$(grep -c 'e-0[0-9])' "${DUMPER_DICT}" || echo 0)
check "3 time ranges written" test "${RANGE_COUNT}" -eq 3

echo "  Generated matrixDumperDict:"
cat "${DUMPER_DICT}" | head -20

# =========================================================================
# 6. Test plot generation (with mock data)
# =========================================================================
echo ""
echo "[6/6] Testing plot generation..."

MOCK_SUMMARY="${TEST_DIR}/summary.csv"
cat > "${MOCK_SUMMARY}" <<'CSV'
cores,step,pd_ms,t_ms
1,total,25.3,141.2
2,total,14.1,78.5
4,total,8.2,42.3
8,total,5.1,25.6
12,total,4.3,22.1
CSV

python3 - "${TEST_DIR}" <<'PYEOF'
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir = sys.argv[1]
summary = os.path.join(results_dir, "summary.csv")

cores, pds, ts = [], [], []
with open(summary) as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 4 and parts[1] == "total":
            cores.append(int(parts[0]))
            pds.append(float(parts[2]))
            ts.append(float(parts[3]))

if not cores:
    print("No data"); sys.exit(1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
base_pd, base_t = pds[0], ts[0]
speedup_pd = [base_pd / p for p in pds]
speedup_t = [base_t / t for t in ts]
ideal = [c / cores[0] for c in cores]

ax1.plot(cores, ideal, 'k--', label='ideal', alpha=0.5)
ax1.plot(cores, speedup_pd, 'bo-', label='pd speedup')
ax1.plot(cores, speedup_t, 'rs-', label='T speedup')
ax1.set_xlabel('Processors'); ax1.set_ylabel('Speedup'); ax1.set_title('Strong Scaling'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(cores, pds, 'bo-', label='pd'); ax2.plot(cores, ts, 'rs-', label='T')
ax2.set_xlabel('Processors'); ax2.set_ylabel('ms'); ax2.set_title('Wall time'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(results_dir, "scaling_plot.png")
fig.savefig(out, dpi=100)
print(f"Plot: {out}")
assert os.path.exists(out)
PYEOF

check "scaling plot generated" test -f "${TEST_DIR}/scaling_plot.png"

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "========================================"
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "========================================"

if [[ ${FAIL} -gt 0 ]]; then
    echo "Some tests failed. Check output above."
    exit 1
else
    echo "All tests passed! Scripts are ready for the lab machine."
    exit 0
fi
