"""API surface probe — verifies the JAX FFI contract matches design-doc §1.1.

Run before every session: `python3 -m dilu.ffi_mvp.python._probe` or directly.
Any drift here is a Phase 1 STOP signal (§6 of the architecture doc).
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    import jax
    import jax.ffi

    assert jax.__version__.startswith("0.9"), (
        f"Expected JAX 0.9.x, got {jax.__version__}. Re-pin environment before proceeding."
    )

    assert jax.default_backend() == "gpu", (
        f"Expected gpu backend, got {jax.default_backend()}. Source /home/yzk/jax-env first."
    )
    dev = jax.devices()[0]
    assert dev.platform == "gpu", f"Expected gpu device, got {dev.platform}"

    for name in ("ffi_call", "pycapsule", "register_ffi_target", "include_dir"):
        assert hasattr(jax.ffi, name), f"jax.ffi missing required symbol: {name}"

    inc = jax.ffi.include_dir()
    assert os.path.isdir(inc), f"include_dir does not exist: {inc}"
    ffi_h = os.path.join(inc, "xla", "ffi", "api", "ffi.h")
    c_api_h = os.path.join(inc, "xla", "ffi", "api", "c_api.h")
    for p in (ffi_h, c_api_h):
        assert os.path.isfile(p), f"Missing FFI header: {p}"

    # c_api.h declares the ABI version our .so compiles against.
    with open(c_api_h) as f:
        src = f.read()
    assert "#define XLA_FFI_API_MAJOR 0" in src, "Major version drift detected"
    assert "#define XLA_FFI_API_MINOR 2" in src, "Minor version drift detected"

    print(
        f"PROBE OK: jax {jax.__version__}, backend=gpu, device={dev}, "
        f"include_dir={inc}, FFI API 0.2"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
