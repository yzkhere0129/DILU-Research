cmake_minimum_required(VERSION 3.18)
project(openfoam_cpu_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Byte-exact arithmetic flags. Keep IEEE-754, no FMA, no fast-math, no
# auto-vectorization that reorders reductions. We can lift these later
# kernel-by-kernel once a comparison test guards each kernel.
add_compile_options(
    -O2
    -fno-fast-math
    -fno-associative-math
    -ffp-contract=off       # disables a*b+c → fma fusion at the compiler level
    -fno-trapping-math
    -mno-fma                # belt+suspenders for x86 FMA fusion
    -Wall
    -Wextra
)

# Find Python (matches the venv invoking cmake)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_kernels_cpp
    src/kernels.cpp
    src/reductions.cpp
    bindings/pybind_module.cpp
)
target_include_directories(_kernels_cpp PRIVATE include)

# Place the .so inside dilu/openfoam_cpu/python so it imports as
#   from dilu.openfoam_cpu.python import _kernels_cpp
# CMAKE_SOURCE_DIR here is dilu/openfoam_cpu/cpp, so go one level up.
set_target_properties(_kernels_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../python"
)
