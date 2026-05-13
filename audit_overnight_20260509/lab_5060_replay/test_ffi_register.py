"""Minimal AMGx FFI register + dispatch test.

Run on 5060:
    cd ~/DILU-Research
    PYTHONPATH=. python audit_overnight_20260509/lab_5060_replay/test_ffi_register.py

Outcomes interpreted:
  "call OK"                            → FFI works (won't see this; dummy config fails AMGx)
  "err: ... not a valid JSON"          → FFI works, AMGx setup just rejected dummy config (THIS IS GOOD)
  "err: NOT_FOUND ... on a platform"   → FFI register did NOT actually store (BAD; need different env)
  any AMGx- or CUDA-stack error        → FFI works, AMGx couldn't initialize (still good for our debug)
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import ctypes
import sys
import jax
import jax.ffi
import jax.numpy as jnp

from dilu.amgx.python.registration import _locate_library, SETUP_TARGET

print(f"python: {sys.version.split()[0]}")
print(f"jax:    {jax.__version__}")
print(f"jax dir: {jax.__file__}")
print(f"devices: {jax.devices()}")
print(f"platform: {[d.platform for d in jax.devices()]}")

lib_path = _locate_library()
print(f"so lib:  {lib_path}")
lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
lib.AmgxSetup.restype = ctypes.c_void_p
lib.AmgxSetup.argtypes = [ctypes.c_void_p]

# Try register with platform="CUDA". This is what we historically use; it works
# on dev jax 0.9.0 / py 3.12 (pip install). It is failing in the lab conda
# build with py 3.13/3.14 (this script's purpose is to confirm if THIS install
# fixes it).
print("\n--- register on 'CUDA' ---")
jax.ffi.register_ffi_target(
    SETUP_TARGET,
    jax.ffi.pycapsule(lib.AmgxSetup),
    platform="CUDA",
    api_version=1,
)
print("registered")

print("\n--- ffi_call test ---")
out = jax.ShapeDtypeStruct((1,), jnp.uint64)
call = jax.ffi.ffi_call(SETUP_TARGET, out, vmap_method="sequential")
rp = jnp.array([0, 1], dtype=jnp.int32)
ci = jnp.array([0], dtype=jnp.int32)
vv = jnp.array([1.0], dtype=jnp.float64)
try:
    r = call(rp, ci, vv, config_json="dummy")
    r.block_until_ready()
    print("call OK (unexpected — dummy config should have failed AMGx)")
except Exception as e:
    name = type(e).__name__
    msg = str(e)[:400]
    print(f"err: {name}: {msg}")
    if "NOT_FOUND" in msg:
        print("\n>>> VERDICT: FFI register did NOT store. Conda jax may be broken.")
        print(">>> Try: pip install in a venv (see lab_5060_replay/README later).")
    else:
        print("\n>>> VERDICT: FFI register OK — AMGx just rejected dummy config.")
        print(">>> 5060 environment is ready. Re-run the replay command.")
