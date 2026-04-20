#!/usr/bin/env bash
# Build libdilu_multicolor.so. Requires /home/yzk/jax-env active.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ARCH="${CUDA_ARCH:-86}"

if ! command -v nvcc >/dev/null; then
  echo "ERROR: nvcc not on PATH." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"
cmake --build "${BUILD_DIR}" --parallel

echo
echo "Built: ${BUILD_DIR}/libdilu_multicolor.so"
nm -D --defined-only "${BUILD_DIR}/libdilu_multicolor.so" \
  | grep -E "Multicolor" || true
echo
echo "Link dependencies:"
ldd "${BUILD_DIR}/libdilu_multicolor.so" | grep -E "libcu(da|sparse)|libcudart" || true
