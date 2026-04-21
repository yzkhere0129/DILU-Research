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
