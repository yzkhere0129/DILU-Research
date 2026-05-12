#!/bin/bash
# Capture E01 (laserMeltFoam timing) + E10 (dense_track_dump_500K validation) outputs
# into the repo tree under audit_overnight_20260509/xeon_validation/results/.
# Idempotent + sanity-checked: prints what was found, what was copied, what's missing.
#
# Usage:
#   bash E01_collect.sh           (uses defaults below)
#   bash E01_collect.sh --no-validate   (skip E10 dense_validate copy)
#
# After it prints OK, run:
#   git add audit_overnight_20260509/xeon_validation/results/{E01,E10_validate_dense_pd}
#   git commit -m "<msg>"
#   git push

set -euo pipefail

CASE_DIR="${E01_CASE_DIR:-$HOME/cases/single_track_dump}"
VALIDATE_DIR="${E10_VALIDATE_DIR:-$HOME/DILU-Research/validate_dense}"
REPO="$HOME/DILU-Research"
OUT_E01="$REPO/audit_overnight_20260509/xeon_validation/results/E01"
OUT_E10="$REPO/audit_overnight_20260509/xeon_validation/results/E10_validate_dense_pd"

SKIP_VALIDATE=0
for arg in "$@"; do
    [[ "$arg" == "--no-validate" ]] && SKIP_VALIDATE=1
done

echo "========================================"
echo "E01_collect.sh — gather Xeon outputs into git tree"
echo "========================================"
echo "case dir:      $CASE_DIR"
echo "validate dir:  $VALIDATE_DIR"
echo "target E01:    $OUT_E01"
echo "target E10:    $OUT_E10"
echo

# ----------------------------------------------------------------
# E01 — laserMeltFoam run outputs
# ----------------------------------------------------------------
echo "=== [E01] copying laserMeltFoam outputs ==="
if [[ ! -d "$CASE_DIR" ]]; then
    echo "  FAIL: case dir not found: $CASE_DIR"
    exit 1
fi

mkdir -p "$OUT_E01"

for src in \
    "$CASE_DIR/log.E01" \
    "$CASE_DIR/system/controlDict" \
    "$CASE_DIR/system/controlDict.preE01"
do
    if [[ -f "$src" ]]; then
        sz=$(du -h "$src" | awk '{print $1}')
        # rename controlDict variants for clarity in repo
        case "$(basename "$src")" in
            controlDict)        dst="$OUT_E01/controlDict.patched" ;;
            controlDict.preE01) dst="$OUT_E01/controlDict.preE01" ;;
            *)                  dst="$OUT_E01/$(basename "$src")" ;;
        esac
        cp "$src" "$dst"
        echo "  [OK]   $sz  $src → $(basename "$dst")"
    else
        echo "  [MISS] $src"
    fi
done

# solverInfo functionObject output
si_src="$CASE_DIR/postProcessing/solverInfo"
if [[ -d "$si_src" ]]; then
    rm -rf "$OUT_E01/solverInfo"
    cp -r "$si_src" "$OUT_E01/solverInfo"
    sz=$(du -sh "$OUT_E01/solverInfo" | awk '{print $1}')
    echo "  [OK]   $sz  $si_src → solverInfo/"
    # List what's inside
    echo "          contents:"
    for d in "$OUT_E01/solverInfo"/*/; do
        [[ -d "$d" ]] || continue
        for f in "$d"/*.dat; do
            [[ -f "$f" ]] || continue
            lines=$(wc -l < "$f")
            fsz=$(du -h "$f" | awk '{print $1}')
            echo "          $fsz  $lines lines  $(basename "$d")/$(basename "$f")"
        done
    done
else
    echo "  [MISS] $si_src"
fi

echo
echo "  E01 size summary:"
du -sh "$OUT_E01"
ls -lh "$OUT_E01"
echo

# ----------------------------------------------------------------
# E10 — dense_track_dump_500K validation summary
# ----------------------------------------------------------------
if [[ $SKIP_VALIDATE -eq 0 ]]; then
    echo "=== [E10] copying dense_track_dump_500K validation results ==="
    if [[ ! -d "$VALIDATE_DIR" ]]; then
        echo "  [MISS] $VALIDATE_DIR — run validate_matrix_dumps.py first"
    else
        mkdir -p "$OUT_E10"
        cp -r "$VALIDATE_DIR"/* "$OUT_E10"/
        echo "  [OK]"
        du -sh "$OUT_E10"
        ls -lh "$OUT_E10"
        echo

        # Quick summary parse
        echo "  E10 validation summary:"
        if [[ -f "$OUT_E10/validation_summary.json" ]]; then
            "$REPO/dilu/amgx/bench/.venv_python" 2>/dev/null || true
            ~/jax-env/bin/python3 - "$OUT_E10/validation_summary.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
n_total = d["n_total"]; n_pass = d["n_pass"]
print(f"    Total matrices: {n_total}")
print(f"    PASS:           {n_pass}  ({n_pass/n_total*100:.1f}%)")
print(f"    FAIL:           {n_total-n_pass}")
print(f"    Total wall:     {d['wall_s_total']:.0f}s")
fails = {}
for r in d["results"]:
    if not r["pass"]:
        k = r["reason"].split(":")[0] if ":" in r["reason"] else r["reason"].split()[0]
        fails[k] = fails.get(k, 0) + 1
if fails:
    print(f"    Fail-mode breakdown:")
    for k, n in sorted(fails.items(), key=lambda x: -x[1]):
        print(f"      {n:4d}  {k}")
PYEOF
        fi
        if [[ -f "$OUT_E10/sane_pool.txt" ]]; then
            pool_n=$(grep -cv "^#" "$OUT_E10/sane_pool.txt" || echo 0)
            echo "    Sane pool entries: $pool_n"
        fi
    fi
fi

echo
echo "========================================"
echo "DONE. To commit:"
echo "========================================"
echo "  cd $REPO"
echo "  git add audit_overnight_20260509/xeon_validation/results/E01 \\"
[[ $SKIP_VALIDATE -eq 0 ]] && echo "          audit_overnight_20260509/xeon_validation/results/E10_validate_dense_pd \\"
echo "  git commit -m \"Xeon E01 laserMeltFoam timing + E10 dense validation pool\""
echo "  git push"
