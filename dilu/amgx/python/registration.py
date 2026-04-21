"""Load libdilu_amgx.so and register the four FFI handlers.

Mirrors Phase 2's registration.py: one-time registration per process, safe to
call register_once() repeatedly.

AMGx library resolution:
  - dilu/amgx/build/libdilu_amgx.so is the default search location.
  - DILU_AMGX_LIB env var overrides.
"""
from __future__ import annotations

import ctypes
import os
import threading

import jax
import jax.ffi

_LIB_NAME = "libdilu_amgx.so"

_TARGETS = (
    "AmgxSetup",
    "AmgxUpdateCoefficients",
    "AmgxSolve",
    "AmgxRelease",
)

# JAX-side custom-call target names (what gets passed to ffi_call).
SETUP_TARGET = "dilu_amgx_setup"
UPDATE_TARGET = "dilu_amgx_update_coefficients"
SOLVE_TARGET = "dilu_amgx_solve"
RELEASE_TARGET = "dilu_amgx_release"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.normpath(os.path.join(_PACKAGE_DIR, "..", "build"))

_lock = threading.Lock()
_registered = False
_lib = None


def _locate_library() -> str:
    override = os.environ.get("DILU_AMGX_LIB")
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"DILU_AMGX_LIB set but file missing: {override}"
            )
        return override
    candidate = os.path.join(_BUILD_DIR, _LIB_NAME)
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"{_LIB_NAME} not found at {candidate}. "
            f"Run dilu/amgx/build.sh first, or set DILU_AMGX_LIB."
        )
    return candidate


def register_once() -> None:
    global _registered, _lib
    if _registered:
        return
    with _lock:
        if _registered:
            return
        path = _locate_library()
        # RTLD_GLOBAL so AMGx library symbols remain available to our handlers
        # (AMGx symbols are resolved at FFI-library load time via DT_NEEDED).
        _lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        for sym in _TARGETS:
            h = getattr(_lib, sym)
            h.restype = ctypes.c_void_p
            h.argtypes = [ctypes.c_void_p]

        jax.ffi.register_ffi_target(
            SETUP_TARGET,
            jax.ffi.pycapsule(_lib.AmgxSetup),
            platform="CUDA", api_version=1,
        )
        jax.ffi.register_ffi_target(
            UPDATE_TARGET,
            jax.ffi.pycapsule(_lib.AmgxUpdateCoefficients),
            platform="CUDA", api_version=1,
        )
        jax.ffi.register_ffi_target(
            SOLVE_TARGET,
            jax.ffi.pycapsule(_lib.AmgxSolve),
            platform="CUDA", api_version=1,
        )
        jax.ffi.register_ffi_target(
            RELEASE_TARGET,
            jax.ffi.pycapsule(_lib.AmgxRelease),
            platform="CUDA", api_version=1,
        )
        _registered = True


def library_path() -> str:
    return _locate_library()
