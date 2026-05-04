"""Phase 4 AMGx-backed PCG preconditioner for JAX.

Public API:
    amgx_setup(row_ptr, col_idx, values, config_json) -> token (uint64[1])
    amgx_update_coefficients(token, values) -> status (int32[1])
    amgx_solve(token, b, x0) -> (x, iters[1], status[1])
    amgx_release(token) -> status (int32[1])
    Plan(row_ptr, col_idx, values, config_json)  # context manager
    CLASSICAL_V_CYCLE, AGGRESSIVE_COARSENING, MINI_AMG_TEST   # config strings
"""
from .wrapper import (
    amgx_setup,
    amgx_update_coefficients,
    amgx_solve,
    amgx_release,
)
from .plan import Plan
from .config import (
    CLASSICAL_V_CYCLE,
    AGGRESSIVE_COARSENING,
    CLASSICAL_V_DIAGSCALED,
    CLASSICAL_V_DIAGSCALED_TIGHT,
    CLASSICAL_V_DIAGSCALED_BICGSTAB,
    CLASSICAL_GS_PCG,
    CLASSICAL_GS_BICGSTAB,
    AGGREGATION_PCG,
    MINI_AMG_TEST,
    with_tolerance,
)
from .refinement import amgx_solve_with_refinement

__all__ = [
    "amgx_setup",
    "amgx_update_coefficients",
    "amgx_solve",
    "amgx_release",
    "Plan",
    "CLASSICAL_V_CYCLE",
    "AGGRESSIVE_COARSENING",
    "CLASSICAL_V_DIAGSCALED",
    "CLASSICAL_V_DIAGSCALED_TIGHT",
    "CLASSICAL_V_DIAGSCALED_BICGSTAB",
    "CLASSICAL_GS_PCG",
    "CLASSICAL_GS_BICGSTAB",
    "AGGREGATION_PCG",
    "MINI_AMG_TEST",
    "with_tolerance",
    "amgx_solve_with_refinement",
]
