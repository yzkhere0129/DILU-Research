"""AMGx JSON config presets for Phase 4.

These configs are passed to AMGX_config_create via the `config_json`
attribute of `amgx_setup`. Structure mirrors AMGx's shipped
`lib/configs/PCG_CLASSICAL_V_JACOBI.json` (smoother is a nested object with
its own scope; preconditioner nests inside `solver.solver="PCG"`).

Choice rationale (see phase4_amg_math_foundation.md §2, §6):

- **CLASSICAL_V_CYCLE**: Ruge-Stüben classical AMG V-cycle as PCG
  preconditioner. Default for our stiff Poisson with coefficient jumps.
    * `algorithm=CLASSICAL` selects Ruge-Stüben coarsening (as opposed to
      aggregation).
    * `interpolator=D2` = distance-2 classical interpolation, more robust
      against mild coefficient jumps than D1.
    * `smoother=BLOCK_JACOBI` — fully parallel, GPU-friendly, robust.
    * tol=1e-10, max_iters=200 for the outer PCG (mirrors Phase 2 T7).

- **AGGRESSIVE_COARSENING**: memory-lean variant. `aggressive_levels=2` plus
  D1 interpolator drops hierarchy size and working memory ~40% at the cost
  of +10-20% iter count. Recommended only if 128^3 OOMs under the default.

- **MINI_AMG_TEST**: max_levels=5 — for 16^3 smoke tests where a full-depth
  hierarchy is overkill.

Pass `str(config)` to `amgx_setup` as the `config_json` static argument.
"""

# Starter Phase 4 default. Mirrors AMGx's own PCG_CLASSICAL_V_JACOBI.json
# with our tolerances and print-output silenced.
CLASSICAL_V_CYCLE = """{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}"""

# Memory-lean stretch variant for 128^3 on a 4 GB card.
AGGRESSIVE_COARSENING = """{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "aggressive_levels": 2,
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D1",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}"""

# Diagonal-scaled variant for matrices with diag spanning many orders of
# magnitude (e.g. LPBF pd matrices where mushy-zone cells have diag ~1e-27
# while interior cells have diag ~1e-13).
#
# AMGx applies symmetric diagonal scaling D^-1/2 A D^-1/2 internally before
# building the AMG hierarchy, then unscales the solution. This makes the
# matrix have unit (signed) diagonal, dramatically improving AMG convergence
# on heterogeneous problems.
CLASSICAL_V_DIAGSCALED = """{
    "config_version": 2,
    "solver": {
        "scaling": "DIAGONAL_SYMMETRIC",
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}"""

# Tighter variant of CLASSICAL_V_DIAGSCALED for precision experiments.
# Changes vs the base:
#   - tolerance: 1e-10 → 1e-14 (tighter convergence)
#   - presweeps/postsweeps: 1 → 2 (better AMG preconditioning quality)
#   - max_iters: 200 → 500 (allow more iterations for tight tolerance)
# Goal: test whether tighter solve reduces rel_vs_OF gap from ~2.5e-6 toward ~1e-7.
# The gap is caused by condition-number amplification of floating-point path
# differences between AMGx and OpenFOAM PCG-DIC (κ(A) ≈ 10^14).
CLASSICAL_V_DIAGSCALED_TIGHT = """{
    "config_version": 2,
    "solver": {
        "scaling": "DIAGONAL_SYMMETRIC",
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 2,
            "postsweeps": 2,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 500,
        "tolerance": 1e-14,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}"""

# BiCGStab outer + AMG preconditioner + DIAGONAL_SYMMETRIC scaling.
# For non-symmetric matrices (e.g. T equation in laserMeltFoam, where the
# matrix has fvm::div(rhoCpPhi, T) advection making upper != lower).
# Key differences from CLASSICAL_V_DIAGSCALED:
#   - "solver": "BICGSTAB" (instead of "PCG") — handles non-symmetric A
#   - all preconditioner / scaling settings preserved
CLASSICAL_V_DIAGSCALED_BICGSTAB = """{
    "config_version": 2,
    "solver": {
        "scaling": "DIAGONAL_SYMMETRIC",
        "solver": "BICGSTAB",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}"""

# Strong-smoother variant: multicolor Gauss-Seidel, more sweeps.
# Use when BLOCK_JACOBI is too weak (e.g. AMGx PCG diverges on real
# multiphysics matrices like LPBF Marangoni-modified Poisson).
CLASSICAL_GS_PCG = """{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "smoother",
                "solver": "MULTICOLOR_GS",
                "relaxation_factor": 1.0,
                "symmetric_GS": 1,
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 2,
            "postsweeps": 2,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "coarse_solver": "DENSE_LU_SOLVER",
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}"""

# Outer BiCGStab instead of PCG. Use when matrix may not be strictly SPD
# (e.g. boundary-handling artifacts make A slightly asymmetric numerically).
CLASSICAL_GS_BICGSTAB = """{
    "config_version": 2,
    "solver": {
        "solver": "BICGSTAB",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "smoother",
                "solver": "MULTICOLOR_GS",
                "relaxation_factor": 1.0,
                "symmetric_GS": 1,
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 2,
            "postsweeps": 2,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "coarse_solver": "DENSE_LU_SOLVER",
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}"""

# Aggregation-based AMG (instead of Ruge-Stüben classical).
# More robust on non-M-matrices and matrices with weak diagonal dominance.
AGGREGATION_PCG = """{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "selector": "SIZE_2",
            "smoother": {
                "scope": "smoother",
                "solver": "MULTICOLOR_GS",
                "relaxation_factor": 1.0,
                "symmetric_GS": 1,
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 2,
            "postsweeps": 2,
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}"""

# Small-problem smoke test. Fewer levels, cheaper setup.
MINI_AMG_TEST = """{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 5,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}"""


def with_tolerance(base_config: str, tol: float, max_iters: int = 200) -> str:
    """Return a copy of `base_config` with tolerance / max_iters substituted.

    Uses `json.loads/dumps` to preserve structure. For building custom
    configs, import `json` and construct the dict directly.
    """
    import json
    cfg = json.loads(base_config)
    cfg["solver"]["tolerance"] = float(tol)
    cfg["solver"]["max_iters"] = int(max_iters)
    return json.dumps(cfg)
