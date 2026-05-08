#!/bin/bash
# One-shot dev pipeline:
#   - Untar single_track_dump.tgz
#   - Run prepare on 6 timesteps × 2 equations (pd, T) = 12 npz
#   - Generate physical fields + error + slice plots
#
# Usage:
#   bash dilu/amgx/bench/run_single_core_pipeline.sh /path/to/single_track_dump.tgz

set -euo pipefail

TGZ=${1:-/home/yzk/single_track_dump.tgz}
[[ -f $TGZ ]] || { echo "FAIL: tarball not found: $TGZ"; exit 1; }

CASE=$HOME/single_track_dump
mkdir -p $CASE
echo "=== Extracting $TGZ → $CASE ==="
tar -xzf "$TGZ" -C $CASE
ls $CASE/postProcessing/matrices/

echo
echo "=== Running prepare on 6 timesteps × 2 eqs (pd_corr0 + T_corr0) ==="
source /home/yzk/jax-env/bin/activate

# Phase tag: melting for early times, evap for late times
declare -A PHASE_TAG=(
    [3.2e-07]="melting"
    [3.8e-07]="melting"
    [4.1e-07]="melting"
    [7e-07]="evap_early"
    [9e-07]="evap"
    [1.06e-06]="evap_late"
)

for t in 3.2e-07 3.8e-07 4.1e-07 7e-07 9e-07 1.06e-06; do
    phase=${PHASE_TAG[$t]}
    for eq in pd_corr0 T_corr0; do
        echo
        echo "--- $phase / $t / $eq ---"
        python3 -u -m dilu.amgx.bench.prepare_single_core_plot_data \
            --case $CASE \
            --time $t --eq $eq --phase $phase \
            2>&1 | tail -20
    done
done

echo
echo "=== Generated npz files ==="
ls -la dilu/amgx/bench/single_*.npz
