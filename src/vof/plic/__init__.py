"""Eulerian PLIC VOF — JAX GPU parallelism research module.

This package is a clean-slate research implementation of traditional
Piecewise Linear Interface Calculation (PLIC) Volume-of-Fluid, targeted
at efficient GPU parallelism on JAX. It is *independent* from the
Lagrangian VOF work under ``jax_laseram.vof.lagrangian_3d``, which
remains as a reference / performance counter-example.

Design goals
------------
1. Research knowledge artefact: map out which JAX idioms (``jnp.where``,
   ``jax.lax.fori_loop``, ``jax.vmap``, ``jax.lax.conv_general_dilated``,
   ``jax.pure_callback``) hit the memory-bandwidth ceiling on PLIC-like
   workloads and which are black holes.
2. Reusable code artefact: all modules here are meant to be ported to a
   future team-internal AM-CFD platform, so dependencies on
   ``jaxfluids.*`` are kept minimal and well-scoped.
3. Mathematical rigour: operator split follows Weymouth & Zaleski
   (2010, JCP 229) for strict volume conservation; plane intercept is
   computed via the Scardovelli & Zaleski (2000, JCP 164) formula.

Module layout (target — filled in incrementally by stages)
----------------------------------------------------------
- ``volume_formula``     : V(plane, cube) analytic forward model (Stage 1)
- ``normal_youngs``      : Youngs/Parker-Young 3x3x3 conv normal   (Stage 1)
- ``intercept_solver``   : d(F,n) root solver, Phase A + Phase B   (Stage 1/4)
- ``geometric_flux``     : donor-region sweep-box flux integration (Stage 2)
- ``strang_sweep``       : Weymouth-Zaleski conservative Strang split (Stage 2)
- ``diagnostics``        : volume drift / boundedness metrics      (Stage 2)
- ``plic_handler``       : top-level handler, ``LevelsetHandler``-like (Stage 3)
- ``plic_simulation_manager`` : ``SimulationManager`` subclass      (Stage 3)

See ``/home/yzk/.claude/plans/vast-napping-scott.md`` for the full
research plan and ``docs/PLIC_JAX_PARALLEL_LESSONS.md`` (produced in
Stage 5) for the final knowledge artefact.
"""
