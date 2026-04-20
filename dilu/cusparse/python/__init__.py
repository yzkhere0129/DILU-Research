"""Phase 2 cuSPARSE-based DILU preconditioner for JAX.

Public API:
    dilu_factor(row_ptr, col_idx, values, diag_offset) -> d_star
    cusparse_dilu_analyze(row_ptr, col_idx, values) -> token (uint64 scalar)
    cusparse_dilu_apply(token, row_ptr, col_idx, values, d_star,
                        diag_offset, r) -> z
    cusparse_dilu_release(token) -> int32 scalar status
    Plan(row_ptr, col_idx, values)  # context manager around analyze/release
"""
from .wrapper import (
    dilu_factor,
    cusparse_dilu_analyze,
    cusparse_dilu_apply,
    cusparse_dilu_release,
    build_diag_offset,
)
from .plan import Plan

__all__ = [
    "dilu_factor",
    "cusparse_dilu_analyze",
    "cusparse_dilu_apply",
    "cusparse_dilu_release",
    "build_diag_offset",
    "Plan",
]
