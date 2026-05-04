#!/usr/bin/env bash
# Bootstrap script for lab 5060 GPU machine.
# Goal: from a fresh ssh login, get to "AMGx bench result on senior 21 pd matrices"
# Does NOT need OpenFOAM. Does NOT need root. Everything in $HOME.
#
# Run interactively step-by-step (don't blindly source — read each step's note):
#   bash dilu/amgx/bench/lab_deploy/bootstrap_lab_5060.sh
#
# Or step-by-step:
#   bash bootstrap_lab_5060.sh check       # only check what's installed
#   bash bootstrap_lab_5060.sh install     # install missing pieces
#   bash bootstrap_lab_5060.sh build       # build AMGx FFI shim
#   bash bootstrap_lab_5060.sh run         # run the bench (after install + build)

set -e  # stop on error

REPO=$HOME/DILU-Research
JAX_ENV=$HOME/jax-env
AMGX_PREFIX=$HOME/local/amgx
AMGX_SRC=$HOME/src/amgx
CUDA=${CUDA_HOME:-/usr/local/cuda}

CMD=${1:-help}

echo "=========================================="
echo "Lab 5060 bootstrap — DILU-Research / AMGx"
echo "Mode: $CMD"
echo "=========================================="

# -----------------------------------------------------------------------------
# CHECK: report current state
# -----------------------------------------------------------------------------
do_check() {
    echo "[1/7] hostname:"
    hostname
    echo
    echo "[2/7] GPU:"
    nvidia-smi -L 2>&1 | head -3 || echo "  ❌ nvidia-smi not found — no GPU?"
    echo
    echo "[3/7] CUDA toolkit:"
    nvcc --version 2>&1 | tail -1 || echo "  ❌ nvcc not found (need CUDA toolkit)"
    [ -d "$CUDA" ] && echo "  CUDA at $CUDA" || echo "  ❌ $CUDA not present"
    echo
    echo "[4/7] Python:"
    python3 --version 2>&1
    [ -d "$JAX_ENV" ] && echo "  jax-env at $JAX_ENV (exists)" || echo "  ⚠ jax-env not yet created"
    echo
    echo "[5/7] AMGx:"
    [ -f "$AMGX_PREFIX/lib/libamgxsh.so" ] && echo "  ✓ libamgxsh.so at $AMGX_PREFIX" \
        || echo "  ❌ AMGx not yet built — need install step"
    echo
    echo "[6/7] DILU-Research repo:"
    [ -d "$REPO/.git" ] && (cd $REPO && echo "  ✓ at $REPO, commit $(git log -1 --pretty=%h)") \
        || echo "  ⚠ repo not at $REPO — need to clone"
    echo
    echo "[7/7] DILU AMGx FFI shim:"
    [ -f "$REPO/dilu/amgx/build/libdilu_amgx.so" ] \
        && echo "  ✓ libdilu_amgx.so built" \
        || echo "  ⚠ FFI shim not built"
}

# -----------------------------------------------------------------------------
# INSTALL: AMGx + jax-env
# -----------------------------------------------------------------------------
do_install() {
    echo "[install/1] git clone DILU-Research (skipped if exists)"
    if [ ! -d "$REPO/.git" ]; then
        git clone git@github.com:yzkhere0129/DILU-Research.git "$REPO" \
            || git clone https://github.com/yzkhere0129/DILU-Research.git "$REPO"
    else
        (cd $REPO && git pull origin main)
    fi

    echo "[install/2] Python venv + jax + numba + scipy"
    if [ ! -d "$JAX_ENV" ]; then
        python3 -m venv "$JAX_ENV"
    fi
    "$JAX_ENV/bin/pip" install --upgrade pip
    # Detect CUDA major version from nvidia-smi
    CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
    [ -z "$CUDA_MAJOR" ] && CUDA_MAJOR=12
    echo "  using JAX for CUDA $CUDA_MAJOR"
    "$JAX_ENV/bin/pip" install "jax[cuda${CUDA_MAJOR}]" numba scipy numpy pytest matplotlib

    echo "[install/3] AMGx (clone + build, ~10 min)"
    if [ ! -f "$AMGX_PREFIX/lib/libamgxsh.so" ]; then
        mkdir -p "$AMGX_SRC" "$AMGX_PREFIX"
        if [ ! -d "$AMGX_SRC/.git" ]; then
            git clone --recurse-submodules https://github.com/NVIDIA/AMGX.git "$AMGX_SRC"
        fi
        cd "$AMGX_SRC"
        # Use the v2.5.0 tag we know works (2.5.0 verified on dev box)
        git checkout v2.5.0 || true
        mkdir -p build && cd build
        cmake .. \
            -DCMAKE_INSTALL_PREFIX="$AMGX_PREFIX" \
            -DCUDA_TOOLKIT_ROOT_DIR="$CUDA" \
            -DCUDA_ARCH="86" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_NO_MPI=ON
        # CUDA_ARCH="86" for RTX 30/40 series; 5060 (Blackwell?) might need "120"
        # If build fails, try DCUDA_ARCH="all" (slower compile)
        make -j8
        make install
    fi
    echo "  ✓ AMGx installed to $AMGX_PREFIX"
}

# -----------------------------------------------------------------------------
# BUILD: dilu/amgx FFI shim against just-installed AMGx
# -----------------------------------------------------------------------------
do_build() {
    cd "$REPO/dilu/amgx"
    rm -rf build
    mkdir build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMGX_INCLUDE_DIR="$AMGX_PREFIX/include" \
        -DAMGX_LIB="$AMGX_PREFIX/lib/libamgxsh.so"
    make -j8
    [ -f libdilu_amgx.so ] && echo "  ✓ libdilu_amgx.so built" || (echo "  ❌ build failed"; exit 1)
}

# -----------------------------------------------------------------------------
# RUN: the bench
# -----------------------------------------------------------------------------
do_run() {
    cd "$REPO"
    source "$JAX_ENV/bin/activate"

    # Verify env x64 + AMGx import
    echo "[run/0] sanity import"
    python -c "
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from jax import config; config.update('jax_enable_x64', True)
from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED
print('imports OK')
"

    OUT_LOG=/tmp/amgx_5060_bench.log
    OUT_JSON_BENCH=/tmp/amgx_5060_bench_results.json
    OUT_JSON_PREC=/tmp/amgx_5060_precision.json

    echo
    echo "[run/1] AMGx 3-mode bench vs OF baseline (~3 min)"
    python -u -m dilu.amgx.bench.bench_amgx_vs_of_n32 2>&1 \
        | grep -v 'memory leak\|ptr:\|AMGX version\|Built on\|Compiled with' \
        > "$OUT_LOG"
    tail -15 "$OUT_LOG"

    echo
    echo "[run/2] AMGx precision sweep (~2 min, no scipy truth — too RAM-heavy on 512K)"
    python -u -m dilu.amgx.bench.sweep_amgx_senior_data \
        --tol 1e-12 --n-refine 1 --no-truth \
        --out "$OUT_JSON_PREC" 2>&1 \
        | grep -v 'memory leak\|ptr:\|AMGX version\|Built on\|Compiled with' \
        | tail -25

    # Copy to repo for git
    mkdir -p "$REPO/dilu/amgx/bench/lab_deploy/results"
    cp "$OUT_LOG"       "$REPO/dilu/amgx/bench/lab_deploy/results/amgx_5060_bench.log"
    cp "$OUT_JSON_PREC" "$REPO/dilu/amgx/bench/lab_deploy/results/amgx_5060_precision.json"

    echo
    echo "✓ Done. Results in $REPO/dilu/amgx/bench/lab_deploy/results/"
    echo "  Run these to push back:"
    echo "    cd $REPO"
    echo "    git add dilu/amgx/bench/lab_deploy/results/"
    echo "    git commit -m 'lab 5060: AMGx bench + precision results'"
    echo "    git push origin main"
}

case "$CMD" in
    check)   do_check ;;
    install) do_install ;;
    build)   do_build ;;
    run)     do_run ;;
    all)     do_install && do_build && do_run ;;
    *)
        cat <<EOF
Usage: $0 [check|install|build|run|all]

  check    — report what's already installed (run first to see)
  install  — git clone repo + python venv + AMGx build (10-20 min)
  build    — build dilu/amgx FFI shim against installed AMGx (~1 min)
  run      — run AMGx bench scripts on senior matrices (~5 min)
  all      — install + build + run end-to-end

Recommended sequence on a fresh box:
  bash $0 check
  bash $0 install     # check first that GPU + nvcc visible
  bash $0 build
  bash $0 run

If anything fails, paste error to dev box for help.
EOF
        ;;
esac
