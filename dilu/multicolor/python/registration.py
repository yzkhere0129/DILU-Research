"""Load libdilu_multicolor.so and register the four FFI handlers.

Mirror of Phase 2's registration logic. `register_once()` is idempotent and
thread-safe.
"""
from __future__ import annotations

import ctypes
import os
import threading

import jax
import jax.ffi

_LIB_NAME = "libdilu_multicolor.so"

_TARGETS = (
    "MulticolorAnalyze",
    "MulticolorApply",
    "MulticolorRefactor",
    "MulticolorRelease",
)

# JAX-side target names (what gets passed to ffi_call).
ANALYZE_TARGET = "multicolor_analyze"
APPLY_TARGET = "multicolor_apply"
REFACTOR_TARGET = "multicolor_refactor"
RELEASE_TARGET = "multicolor_release"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.normpath(os.path.join(_PACKAGE_DIR, "..", "build"))

_lock = threading.Lock()
_registered = False
_lib = None


def _locate_library() -> str:
    override = os.environ.get("DILU_MULTICOLOR_LIB")
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"DILU_MULTICOLOR_LIB set but file missing: {override}"
            )
        return override
    candidate = os.path.join(_BUILD_DIR, _LIB_NAME)
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"{_LIB_NAME} not found at {candidate}. "
            f"Run dilu/multicolor/build.sh first, or set DILU_MULTICOLOR_LIB."
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
        for sym in _TARGETS:
            h = getattr(_lib, sym)
            h.restype = ctypes.c_void_p
            h.argtypes = [ctypes.c_void_p]

        jax.ffi.register_ffi_target(
            ANALYZE_TARGET,
            jax.ffi.pycapsule(_lib.MulticolorAnalyze),
            platform="CUDA", api_version=1,
        )
        jax.ffi.register_ffi_target(
            APPLY_TARGET,
            jax.ffi.pycapsule(_lib.MulticolorApply),
            platform="CUDA", api_version=1,
        )
        jax.ffi.register_ffi_target(
            REFACTOR_TARGET,
            jax.ffi.pycapsule(_lib.MulticolorRefactor),
            platform="CUDA", api_version=1,
        )
        jax.ffi.register_ffi_target(
            RELEASE_TARGET,
            jax.ffi.pycapsule(_lib.MulticolorRelease),
            platform="CUDA", api_version=1,
        )
        _registered = True


def library_path() -> str:
    return _locate_library()
