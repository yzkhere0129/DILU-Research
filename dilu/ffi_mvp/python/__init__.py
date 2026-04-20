"""Phase 1 FFI MVP — Jacobi-preconditioned residual kernel.

Importing this package automatically loads the compiled .so and registers the
FFI target, so downstream code only needs `from dilu.ffi_mvp.python import
jacobi_residual`.
"""
from .registration import register_once
from .wrapper import jacobi_residual, jacobi_residual_reference, assemble_diag_inv

register_once()

__all__ = ["jacobi_residual", "jacobi_residual_reference", "assemble_diag_inv"]
