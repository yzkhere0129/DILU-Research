# Installing dilu-amgx

Three layers, in order:
1. NVIDIA AMGx 2.5.0 (C++ library; once per host)
2. The `libdilu_amgx.so` FFI shim (compiled against your CUDA + JAX);
3. The Python package `dilu_amgx`.

For the full byte-exact reproducibility recipe (different focus: lets a
fresh-context AI replicate the exact same iteration counts and residuals
this package produces), read `REPRODUCE.md` instead — that document is the
authoritative spec; this file is the human shortcut.

## Prerequisites

- Linux x86-64 (verified on Ubuntu 22.04 / 24.04).
- CUDA Toolkit 12.x or 13.x. The CUDA major version that AMGx was built
  against **must match** the CUDA that JAX is using. Mismatched majors
  produce `cudaErrorInvalidValue` at the first `cudaMemcpyAsync`.
- A Python venv (3.10–3.12) with `jax`, `numpy`, `scipy` installed.
  Activate it before invoking `./build.sh`.
- CMake ≥ 3.24, GCC 11 or 12 (C++17).
- A CUDA GPU with compute capability ≥ sm_75. Tested on sm_86 (RTX 3050)
  and sm_120 (RTX 5060 Blackwell).

## Step 1 — Build AMGx

```bash
cd ~/src
git clone --branch v2.5.0 --depth 1 https://github.com/NVIDIA/AMGX.git amgx-2.5.0
mkdir -p amgx-2.5.0/build && cd amgx-2.5.0/build
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$HOME/local/amgx" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_FOUND=FALSE \
  -DCMAKE_CUDA_ARCHITECTURES="86"     # match your GPU
make -j8 && make install
```

After install you should see `~/local/amgx/lib/libamgxsh.so` and
`~/local/amgx/include/amgx_c.h`.

## Step 2 — Build the FFI shim

From this package's root:

```bash
# Choose your CUDA arch (override the default 86):
#   86  = RTX 3050 Ampere
#   89  = RTX 4090 / L40 Ada
#   90  = H100 Hopper
#   120 = RTX 5060 Blackwell (CUDA 13+)
DILU_AMGX_CUDA_ARCH=86 ./build.sh
```

Expected output ends with `Artifact: build/libdilu_amgx.so` and an `ldd`
dump showing `libamgxsh.so => /home/USER/local/amgx/lib/libamgxsh.so`
resolved. If `ldd` shows `not found` you have a path or `LD_LIBRARY_PATH`
problem; the runtime loader uses `INSTALL_RPATH` set to `$AMGX_ROOT/lib`,
so a non-default `AMGX_ROOT=` at build time gets baked in correctly.

Environment overrides:

| Variable                 | Default              | Purpose                                 |
| ------------------------ | -------------------- | --------------------------------------- |
| `AMGX_ROOT`              | `$HOME/local/amgx`   | AMGx install prefix                     |
| `DILU_AMGX_CUDA_ARCH`    | `86`                 | Target SM (single or semicolon list)    |
| `DILU_AMGX_VERBOSE`      | unset                | If `1`, log device-pointer attrs        |

## Step 3 — Install the Python package

```bash
pip install -e .
```

The package itself is pure Python; it does NOT build the C++ shim. It
expects `libdilu_amgx.so` to already exist at `../build/libdilu_amgx.so`
relative to the installed `dilu_amgx/` package (matching the lookup in
`registration.py`). For an in-place dev install this is automatic — Step 2
produces it. For a site-install on a new host, re-run Step 2 first.

## Step 4 — Verify

```bash
pytest tests/test_smoke.py -v                       # ~5 s
pytest tests/test_precision_vs_truth.py -v -k fixture   # ~10 s
```

A passing run looks like:

```
test_smoke_8cubed PASSED
test_fixture_pd_amgx_ir_meets_user_spec PASSED      # rel_vs_truth ~ 1e-14
test_fixture_T_amgx_ir_meets_user_spec PASSED
test_fixture_pd_amgx_ir_meets_tight_floor PASSED    # ≤ 1e-13
test_fixture_T_amgx_ir_meets_tight_floor PASSED
```

If `test_smoke_8cubed` fails with `NOT_FOUND on platform CUDA (canonical
cuda)`, you are likely on an old conda JAX with a broken dispatch table
(see `KNOWN_GOTCHAS.md` G3). Switch to a pip-installed venv.

## Troubleshooting

`KNOWN_GOTCHAS.md` enumerates the 15 traps we hit during 6 months of
development. Read G1–G8 before opening a bug.
