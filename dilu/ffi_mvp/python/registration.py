"""Load libdilu_ffi_mvp.so and register the JacobiResidual handler with XLA FFI.

Registration is once-per-process; calling `register_once()` repeatedly is safe.
"""
from __future__ import annotations

import ctypes
import os
import threading

import jax
import jax.ffi

_LIB_NAME = "libdilu_ffi_mvp.so"
_TARGET_NAME = "jacobi_residual"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.normpath(os.path.join(_PACKAGE_DIR, "..", "build"))

_lock = threading.Lock()
_registered = False
_lib = None


def _locate_library() -> str:
    override = os.environ.get("DILU_FFI_MVP_LIB")
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"DILU_FFI_MVP_LIB set but file missing: {override}"
            )
        return override
    candidate = os.path.join(_BUILD_DIR, _LIB_NAME)
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"{_LIB_NAME} not found at {candidate}. "
            f"Run dilu/ffi_mvp/build.sh first, or set DILU_FFI_MVP_LIB."
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
        _lib = ctypes.CDLL(path)
        # Expose the handler symbol as a ctypes function object so pycapsule
        # can extract its address. Returning a pointer keeps us ABI-agnostic.
        handler = _lib.JacobiResidual
        handler.restype = ctypes.c_void_p
        handler.argtypes = [ctypes.c_void_p]
        jax.ffi.register_ffi_target(
            _TARGET_NAME,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,  # typed FFI
        )
        _registered = True


def library_path() -> str:
    return _locate_library()


def target_name() -> str:
    return _TARGET_NAME
