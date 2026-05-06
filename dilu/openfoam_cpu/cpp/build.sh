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

# Compiler selection.  If we're on a host where the Python headers live
# under /usr/include/python3.X (Ubuntu/Debian system Python), the
# multiarch shim there does `#include <x86_64-linux-gnu/python3.X/pyconfig.h>`
# which only resolves with gcc that has /usr/include/x86_64-linux-gnu/ in
# its default search path.  conda's gcc has its own sysroot and will fail
# with "pyconfig.h: No such file or directory".  Prefer system gcc when
# available unless the caller explicitly overrode CC/CXX.
if [ -z "${CC:-}" ] && [ -x "/usr/bin/gcc" ]; then
    export CC="/usr/bin/gcc"
fi
if [ -z "${CXX:-}" ] && [ -x "/usr/bin/g++" ]; then
    export CXX="/usr/bin/g++"
fi
echo "CC : ${CC:-cmake-default}"
echo "CXX: ${CXX:-cmake-default}"

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
