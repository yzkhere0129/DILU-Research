---
name: lagrangian-vof
description: Lagrangian VOF (Volume-of-Fluid) advection method for free-surface and two-phase interface tracking. Use when implementing or analyzing VOF advection algorithms, comparing interface reconstruction methods (PLIC, SLIC, Least Squares), debugging volume conservation or interface distortion issues, designing numerical schemes for sharp interface tracking, or evaluating advection accuracy at arbitrary flow angles. Covers Barkhudarov's 3D piecewise linear interface reconstruction, Lagrangian cell-face advection, and Eulerian overlay procedures. Use this whenever working on free-surface flow solvers, droplet/jet simulations, interface tracking code, or evaluating VOF scheme accuracy.
---

# Lagrangian VOF Advection Method

Technical reference on the Lagrangian VOF advection method (Barkhudarov, Flow Science, 2004).
This method reconstructs the fluid interface in 3D using piecewise linear surfaces and advects
fluid volumes in a single Lagrangian step, avoiding operator-splitting errors of standard methods.

## When to Use This Skill

- **Implementing** a VOF advection solver or interface tracking module
- **Comparing** VOF schemes: standard donor-acceptor vs. Lagrangian vs. PLIC
- **Debugging** volume conservation failures, interface distortion, or cell overfill errors
- **Analyzing** accuracy for flows at angles to coordinate axes (45-degree worst case)
- **Choosing** between advection methods based on physics and stability requirements

## Algorithm Overview

The Lagrangian VOF method has **three steps**:

### Step 1: Interface Reconstruction
Approximate the fluid interface in each cell with a plane:

$$n_x x + n_y y + n_z z = C$$

- Normal **n** computed via least-squares on a 27-point stencil (Eq. 4-5 in reference)
- Constant C solved iteratively to match actual fluid volume in cell

### Step 2: Lagrangian Advection
Move each cell face using the velocity component at that face:

$$dx = \frac{A_x}{V_f} \cdot \frac{U \Delta t}{1 - \frac{1}{2}\frac{\partial U}{\partial x}\Delta t}$$

- Second-order accurate in cell size (Eq. 6-7)
- Cell faces stay parallel to themselves; aspect ratio may change
- Normal **n** is adjusted for compression/stretching

### Step 3: Eulerian Overlay
Map the deformed cell volume back onto the Eulerian grid:

- Apportion fluid to acceptor cells based on geometric overlap
- Adjust using dV = V_old / V_new for volume conservation (Eq. 8)
- Each donor cell used only once (prevents over-emptying)
- Overfill in acceptors is discarded and tracked as volume error

## Standard vs. Lagrangian Comparison

| Property | Standard (donor-acceptor) | Lagrangian |
|----------|--------------------------|------------|
| Operator splitting | Yes (x, y, z sequential) | No (single step) |
| Accuracy at 45-degree flow | Poor (worst case distortion) | Good (shape preserved) |
| Volume conservation | Partial cancellation of +/- errors | Positive errors only (overfill) |
| Error location | Near interface | Away from interface (full cells) |
| Time step accuracy | First-order in dt | Second-order in dx |
| CPU cost | Baseline | Within +/- 3% |

## Key Equations Reference

| Equation | Description | Reference |
|----------|-------------|-----------|
| VOF kinematic eq. | $V_f \partial F/\partial t + \nabla \cdot (\mathbf{AU}F) = 0$ | Eq. 2 |
| Plane interface | $n_x x + n_y y + n_z z = C$ | Eq. 4 |
| Least-squares linearization | $F = F_0 + \nabla F \cdot (\mathbf{x} - \mathbf{x}_0)$ | Eq. 5 |
| Face displacement | Second-order integration of $dx/dt = (A_x/V_f)U$ | Eq. 6-7 |
| Volume ratio | $dV = V_{old} / V_{new}$ | Eq. 8 |

## Implementation Notes

- **Non-interface cells**: Skip reconstruction; use first-order donor advection
- **Overfill handling**: Discard excess volume, reduce dt via cumulative error tracking
- **Swirling flows**: Largest errors occur away from interface in high-vorticity regions
- **Coarse grids**: Surface normal accuracy degrades when fluid resolved by ~2 cells
- **Compatibility**: Works with turbulence, thermal energy, scalar transport models
- **Sharp/diffuse**: Compatible with both sharp-interface (ITB=1) and no-sharp-interface (ITB=0)

## Detailed Reference

For complete equations, derivations, figures, and test results:

- [Barkhudarov Method — Full Paper](references/barkhudarov_method.md)
