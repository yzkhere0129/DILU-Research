#!/usr/bin/env bash
# Build the C++ kernels module.  Re-run after editing any cpp/ source.
#
# Usage:  ./build.sh        (Release build, default)
#         ./build.sh Debug  (Debug build, with -O0 -g)
#
# Override Python interpreter:
#   PYTHON_BIN=/path/to/python ./build.sh
#
# Output:  dilu/openfoam_cpu/python/_kernels_cpp.cpython-*.so

set -euo pipefail
cd "$(dirname "$0")"

BUILD_TYPE="${1:-Release}"

# Pick Python: explicit override > local jax-env > python3 on PATH
if [ -n "${PYTHON_BIN:-}" ]; then
    :
elif [ -x "/home/yzk/jax-env/bin/python" ]; then
    PYTHON_BIN="/home/yzk/jax-env/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
echo "Using Python: $PYTHON_BIN"

# Ensure pybind11 is available; pip-install if missing.
if ! "$PYTHON_BIN" -c 'import pybind11' 2>/dev/null; then
    echo "pybind11 not found; installing via pip..."
    "$PYTHON_BIN" -m pip install --user pybind11 \
        || "$PYTHON_BIN" -m pip install pybind11
fi

PYBIND_DIR="$($PYTHON_BIN -c 'import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())')"
echo "pybind11 cmake dir: $PYBIND_DIR"

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
