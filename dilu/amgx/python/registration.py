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

        # JAX platform canonicalization varies across (jax, python) versions:
        #   jax 0.9.0 / py 3.12: accepts "CUDA"
        #   jax 0.9.0 / py 3.14: rejects "CUDA", needs "cuda" (canonical)
        # Register under BOTH names so we work everywhere. Skip silently if a
        # given name is already taken (e.g. one canonicalizes to the other).
        def _reg(name, fn):
            capsule = jax.ffi.pycapsule(fn)
            registered_any = False
            for plat in ("cuda", "CUDA"):
                try:
                    jax.ffi.register_ffi_target(name, capsule,
                                                  platform=plat, api_version=1)
                    registered_any = True
                except Exception:
                    pass
            if not registered_any:
                raise RuntimeError(
                    f"Failed to register FFI target {name!r} on "
                    f"either 'cuda' or 'CUDA' platform."
                )
        _reg(SETUP_TARGET,   _lib.AmgxSetup)
        _reg(UPDATE_TARGET,  _lib.AmgxUpdateCoefficients)
        _reg(SOLVE_TARGET,   _lib.AmgxSolve)
        _reg(RELEASE_TARGET, _lib.AmgxRelease)
        _registered = True


def library_path() -> str:
    return _locate_library()
