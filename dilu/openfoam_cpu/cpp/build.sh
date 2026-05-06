#!/usr/bin/env bash
# Build the C++ kernels module.  Re-run after editing any cpp/ source.
#
# Usage:  ./build.sh        (Release build, default)
#         ./build.sh Debug  (Debug build, with -O0 -g)
#
# Output:  dilu/openfoam_cpu/python/_kernels_cpp.cpython-*.so

set -euo pipefail
cd "$(dirname "$0")"

BUILD_TYPE="${1:-Release}"
PYTHON_BIN="${PYTHON_BIN:-/home/yzk/jax-env/bin/python}"

PYBIND_DIR="$($PYTHON_BIN -c 'import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())')"

mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DPython_EXECUTABLE="$PYTHON_BIN" \
      -Dpybind11_DIR="$PYBIND_DIR" \
      ..

cmake --build . -j"$(nproc)"

echo
echo "Built _kernels_cpp.so → ../python/"
ls -la ../python/_kernels_cpp*.so 2>&1 || true
