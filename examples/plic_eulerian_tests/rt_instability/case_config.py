"""Case parameters for the Rayleigh-Taylor air/helium PLIC benchmark.

Pure Python config dict (no JSON). This case is the PLIC-research
counterpart of the OpenFOAM reference in
``examples/RT_air_helium/`` — all numerical values are copied VERBATIM
from the reference's ``initAlpha.py`` and ``postProcess.py``.

Physical setup
--------------
- Rectangular vertical channel ``[0, 1] x [0, 4]``, quasi-2D (1 cell z).
- Initial interface: ``y_interface(x) = 2.0 + 0.05 * cos(2*pi*x)``.
- ``y > y_interface``: heavy fluid (air),  ``rho_heavy = 1.225``
- ``y < y_interface``: light fluid (helium), ``rho_light = 0.169``
- Density ratio ~7.25 (air/helium at STP).
- Gravity ``g = -9.81`` in y (to be coupled with NS in Stage 4).

Important
---------
Stage 2 only implements **advection**, not the Navier-Stokes solver
required to actually drive RT instability. The ``run.py`` in this
folder therefore only exercises the Stage-1 PLIC primitives on the
initial sinusoidal interface. Full RT evolution is gated on Stage 4
of the plan.
"""

from __future__ import annotations

import math

Nx, Ny = 128, 512
Lx, Ly = 1.0, 4.0
dx = Lx / Nx  # 1/128

CASE = {
    # --- Identification ---
    "name": "plic_rt_instability",
    "description": (
        "Air/helium Rayleigh-Taylor instability - PLIC advection benchmark "
        "(needs Stage 4 NS coupling to run the full evolution)."
    ),
    "reference": (
        "examples/RT_air_helium/{initAlpha.py, postProcess.py}; "
        "ground-truth h1/h2 in of_h1_h2.npz."
    ),
    # --- Domain ---
    # Quasi-2D caveat: PLIC Youngs 3x3x3 stencil requires at least 3
    # cells along every axis. The OpenFOAM reference is 1-cell-thick
    # (``frontAndBack empty``); to stay compatible with the Stage 1
    # primitives we use ``nz=3`` z-replicated layers with physical
    # thickness ``Lz = 3*dx``. The solution remains z-homogeneous so
    # this does not change the physical problem.
    "domain": {
        "x": (0.0, Lx),
        "y": (0.0, Ly),
        "z": (0.0, 3 * dx),
    },
    "grid": {
        "nx": Nx,
        "ny": Ny,
        "nz": 3,
        "nh": 1,
    },
    # --- Initial interface (initAlpha.py:21-40) ---
    "rt": {
        "interface_y0": 2.0,         # initAlpha.py:22
        "amplitude": 0.05,           # initAlpha.py:22
        "wavenumber": 2 * math.pi,   # k = 2*pi / Lx (with Lx=1 => mode 1)
        "rho_heavy": 1.225,          # initAlpha.py:5
        "rho_light": 0.169,          # initAlpha.py:6
    },
    # --- Initialization sub-sampling (initAlpha.py:23) ---
    "subsample": 8,  # 8x8 per cell for a smooth F at the interface
    # --- Gravity (used by Stage 4 NS coupling, not by PLIC advection) ---
    "gravity": (0.0, -9.81, 0.0),
    # --- Time integration ---
    "time": {
        "end": 1.0,
        "save_dt": 0.1,
        "cfl": 0.5,
    },
    # --- Boundary conditions ---
    # Mapped 1:1 from the OpenFOAM boundary conditions (initAlpha.py:66-72).
    "bcs": {
        "top": {"type": "fixed_value", "value": 1.0},      # alpha=1 (heavy)
        "bottom": {"type": "fixed_value", "value": 0.0},   # alpha=0 (light)
        "x_min": "symmetry",                                # left
        "x_max": "symmetry",                                # right
        "z_min": "inactive",                                # frontAndBack empty
        "z_max": "inactive",
    },
    # --- Precision ---
    "precision": "float64",
    # --- Diagnostic probes (postProcess.py:146-157) ---
    "diagnostics": {
        "h1_probe_x_index": 0,            # bubble tip at x=0 (i=0 col)
        "h2_probe_x_index": Nx // 2,      # spike tip at x=Lx/2 (i=Nx//2)
        "h_times": [0.7, 0.8, 0.9, 1.0],
        "snapshot_times": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.8, 0.9, 1.0],
    },
    # --- Reference OpenFOAM data (copied into this folder) ---
    "of_reference": "of_h1_h2.npz",
    # --- Gate metrics (Stage 4) ---
    "gate": {
        "h1_rel_err_max": 0.10,  # < 10 % vs OpenFOAM
        "h2_rel_err_max": 0.10,
    },
}
