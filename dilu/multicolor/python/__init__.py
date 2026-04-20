"""Phase 3 multi-color DILU preconditioner package.

Public API:
    MulticolorPlan(row_ptr, col_idx, values, grid_shape=None)
        Context manager bundling coloring + permute + FFI lifecycle.
    red_black_color(nx, ny, nz)                 -- host closed-form
    greedy_color_csr(row_ptr, col_idx)          -- host first-fit fallback
    validate_coloring(...)                      -- invariant check
    permute_csr(...)                            -- host CSR permutation
    build_diag_offset_permuted(...)             -- host diag locator on Ã
    multicolor_analyze / apply / refactor / release  -- raw FFI wrappers
"""
from .coloring import (
    red_black_color,
    greedy_color_csr,
    validate_coloring,
    color_csr_auto,
)
from .permute import (
    permute_csr,
    build_diag_offset_permuted,
    inverse_permute_csr_values,
)
from .wrapper import (
    multicolor_analyze,
    multicolor_apply,
    multicolor_release,
)
from .plan import MulticolorPlan

__all__ = [
    "red_black_color",
    "greedy_color_csr",
    "validate_coloring",
    "color_csr_auto",
    "permute_csr",
    "build_diag_offset_permuted",
    "inverse_permute_csr_values",
    "multicolor_analyze",
    "multicolor_apply",
    "multicolor_release",
    "MulticolorPlan",
]
