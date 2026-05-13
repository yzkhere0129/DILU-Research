#!/usr/bin/env bash
# 5060 AMGx amortized replay launcher
#
# Pre-reqs:
#   - dense_track_dump_500K case present at $CASE
#   - validated sane_pool.txt present at $POOL (388 lines = 4 comments + 384 sane)
#   - ~/jax-env with jax / scipy / cusparse-amgx FFI built
#   - GPU: RTX 5060
#
# Output: audit_overnight_20260509/lab_5060_replay/results/
#   ├── config_select.json           (5-config sanity benchmark on K=5 matrices)
#   ├── run_meta.json                (env + GPU info)
#   ├── replay_fresh_e8.npz
#   ├── replay_amortized_e8.npz
#   ├── replay_fresh_e12_IR.npz
#   ├── replay_amortized_e12_IR.npz
#   ├── summary.json                 (aggregate stats per protocol)
#   └── plot.png                     (4-panel comparison)
#
# Wall budget: ~6-8 hours (384 × 4 protocols × ~5-15s/step on 5060)

set -euo pipefail

# ---- paths (edit if needed) ----
REPO=${REPO:-$HOME/DILU-Research}
CASE=${CASE:-$HOME/cases/dense_track_dump_500K}
POOL=${POOL:-$REPO/validate_dense/sane_pool.txt}
OUT=${OUT:-$REPO/audit_overnight_20260509/lab_5060_replay/results}
PYTHON=${PYTHON:-$HOME/jax-env/bin/python3}
MAX_MAT=${MAX_MAT:-384}

cd "$REPO"

echo "=========================================="
echo "  AMGx amortized replay — lab 5060"
echo "  REPO=$REPO"
echo "  CASE=$CASE"
echo "  POOL=$POOL"
echo "  OUT=$OUT"
echo "  MAX_MAT=$MAX_MAT"
echo "=========================================="

[ -d "$CASE/postProcessing/matrices" ] || { echo "ERR: case dir missing"; exit 1; }
[ -f "$POOL" ] || { echo "ERR: pool file missing"; exit 1; }

mkdir -p "$OUT"
nvidia-smi | tee "$OUT/nvidia-smi.txt"

# ---- Step 1: config sanity check (K=5 matrices, ~5 min) ----
echo
echo "### Step 1/3: config selection benchmark (K=5) ###"
PYTHONPATH="$REPO" "$PYTHON" -u dilu/amgx/bench/bench_5060_config_select.py \
    --case "$CASE" \
    --pool "$POOL" \
    --k 5 \
    --out "$OUT/config_select.json" \
    --max-iters 2000 2>&1 | tee "$OUT/step1_config_select.log"

# ---- Step 2: 384-step replay (~6-8 hours) ----
echo
echo "### Step 2/3: 384-step replay (4 protocols) ###"
PYTHONPATH="$REPO" "$PYTHON" -u dilu/amgx/bench/replay_5060_amgx_amortized.py \
    --case "$CASE" \
    --pool "$POOL" \
    --output-dir "$OUT" \
    --max "$MAX_MAT" \
    --protocols fresh_e8,amortized_e8,fresh_e12_IR,amortized_e12_IR \
    --max-iters 3000 2>&1 | tee "$OUT/step2_replay.log"

# ---- Step 3: plot ----
echo
echo "### Step 3/3: plot ###"
PYTHONPATH="$REPO" "$PYTHON" dilu/amgx/bench/plot_replay_5060.py \
    --input-dir "$OUT" \
    --output    "$OUT/plot.png" \
    --lu-wall-per-step 72.5 2>&1 | tee "$OUT/step3_plot.log"

echo
echo "=========================================="
echo "  DONE. See:"
echo "    $OUT/summary.json"
echo "    $OUT/plot.png"
echo "    $OUT/step2_replay.log"
echo "=========================================="
