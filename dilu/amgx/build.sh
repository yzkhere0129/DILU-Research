#!/usr/bin/env bash
# Phase 4 build driver. Produces dilu/amgx/build/libdilu_amgx.so.
#
# Requires:
#   - /home/yzk/jax-env active (jax.ffi.include_dir())
#   - AMGx v2.5.0 installed at $AMGX_ROOT (default: $HOME/amgx-2.5.0)
#   - CUDA 12.x toolkit (nvcc)
#
# Flags:
#   DILU_AMGX_VERBOSE=1 ./build.sh  — enable cudaPointerGetAttributes logging.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
AMGX_ROOT="${AMGX_ROOT:-$HOME/local/amgx}"

export AMGX_ROOT

CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES=86
  "-DAMGX_ROOT=${AMGX_ROOT}"
)
if [[ "${DILU_AMGX_VERBOSE:-0}" == "1" ]]; then
  CMAKE_FLAGS+=(-DDILU_AMGX_VERBOSE=1)
fi

mkdir -p "${BUILD}"
cd "${BUILD}"
cmake "${CMAKE_FLAGS[@]}" "${HERE}"
cmake --build . --config Release -- -j4

echo ""
echo "Artifact: ${BUILD}/libdilu_amgx.so"
ls -l "${BUILD}/libdilu_amgx.so"
echo ""
echo "ldd:"
ldd "${BUILD}/libdilu_amgx.so" | sed 's/^/  /'
