"""OpenFOAM ↔ our-solvers cross-check benchmark.

Pipeline:
    reader.py            — load A.mm + b.mm + x0.mm + x_final.mm + metadata.json
    driver_scipy.py      — scipy spsolve (small-N reference)
    driver_cusparse.py   — our cuSPARSE DILU-PCG (SPD only)
    driver_amgx.py       — our AMGx (classical + aggressive)
    compare.py           — L2/L3 metrics & summary table
    run_all.py           — orchestrator

See docs/design/openfoam_crosscheck_plan.md for the full design.
"""
