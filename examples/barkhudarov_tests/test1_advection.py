#!/usr/bin/env python3
"""Barkhudarov Test 1: 2D circular droplet advection at multiple angles.

Paper: Barkhudarov (2004) §4.1
Grid: cell_size = D/10 (10 cells/diameter)
Droplet travels 5D at angles 0°, 6°, 30°, 45°
Metrics: volume error < 0.0002%, perimeter change < 1%
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_laseram.grid import create_grid
from jax_laseram.vof.lagrangian import advect_vof_lagrangian

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Paper configuration ─────────────────────────────────────────
# D = 0.1 (R = 0.05), cell_size = 0.01 = D/10
# Grid: 100×100 on [0,1]×[0,1]
# Travel: 5D = 0.5 at unit speed → t_end = 0.5

D = 0.10
R = D / 2       # 0.05
NX = 100
DOMAIN = 1.0
TRAVEL = 5 * D  # 0.5

ANGLES = [
    (  0, "0°",  (0.15, 0.50)),
    (  6, "6°",  (0.15, 0.40)),
    ( 30, "30°", (0.15, 0.35)),
    ( 45, "45°", (0.20, 0.20)),
]


def create_droplet(grid, cx, cy, r):
    nh, nx, ny = grid.nh, grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    x = np.linspace(dx/2, DOMAIN-dx/2, nx)
    y = np.linspace(dy/2, DOMAIN-dy/2, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
    hd = np.sqrt(2)*dx/2
    F_int = np.where(dist+hd<=r, 1.0,
            np.where(dist-hd>=r, 0.0,
                     np.clip((r-dist+hd)/(2*hd), 0, 1)))
    shape = (nx+2*nh, ny+2*nh, 1)
    F = np.zeros(shape)
    F[nh:nh+nx, nh:nh+ny, 0] = F_int
    F[:nh,:,:]  = F[nh:nh+1,:,:]
    F[-nh:,:,:] = F[-nh-1:-nh,:,:]
    F[:,:nh,:]  = F[:,nh:nh+1,:]
    F[:,-nh:,:] = F[:,-nh-1:-nh,:]
    return jnp.array(F)


def make_velocity(grid, angle_deg):
    nh = grid.nh
    nx, ny = grid.nx, grid.ny
    Ntx, Nty = nx+2*nh, ny+2*nh
    theta = np.radians(angle_deg)
    ux = np.cos(theta)
    uy = np.sin(theta)
    u_face = jnp.full((Ntx-1, Nty, 1), ux)
    v_face = jnp.full((Ntx, Nty-1, 1), uy)
    return u_face, v_face


def perimeter(F_int, dx, dy):
    v = np.array(F_int)
    hc = ((v[:-1,:]-0.5)*(v[1:,:]-0.5)) < 0
    vc = ((v[:,:-1]-0.5)*(v[:,1:]-0.5)) < 0
    return float(np.sum(hc)*dy + np.sum(vc)*dx)


results = {}

for angle_deg, label, (cx0, cy0) in ANGLES:
    print(f"\n{'='*60}")
    print(f"  Angle={label}: droplet ({cx0},{cy0}) → traveling {TRAVEL:.2f}")

    grid = create_grid(NX, NX, nz=1, x_range=(0.0, DOMAIN), y_range=(0.0, DOMAIN), nh=1)
    dx, dy = grid.dx, grid.dy
    nh = grid.nh

    F = create_droplet(grid, cx0, cy0, R)
    u_face, v_face = make_velocity(grid, angle_deg)

    # CFL=0.45, speed = |velocity| = 1 for all angles
    dt = 0.45 * dx / 1.0
    n_steps = int(np.ceil(TRAVEL / dt))
    dt = TRAVEL / n_steps
    print(f"  dt={dt:.6f}, n_steps={n_steps}")

    ic = slice(nh, -nh)
    F_int0 = np.array(F[ic, ic, 0])
    V0 = float(np.sum(F_int0) * dx * dy)
    P0 = perimeter(F_int0, dx, dy)

    # JIT warmup
    _ = advect_vof_lagrangian(F, u_face, v_face, grid, dt)

    t0_wall = time.time()
    snapshots = [(0.0, F_int0.copy())]
    vol_history = [V0]
    per_history = [P0]

    for step in range(n_steps):
        F = advect_vof_lagrangian(F, u_face, v_face, grid, dt)
        if (step+1) % max(1, n_steps//4) == 0 or step == n_steps-1:
            Fi = np.array(F[ic, ic, 0])
            snapshots.append(((step+1)*dt, Fi.copy()))
            V = float(np.sum(Fi)*dx*dy)
            P = perimeter(Fi, dx, dy)
            vol_history.append(V)
            per_history.append(P)
            print(f"  step {step+1}/{n_steps}  dV/V={abs(V-V0)/V0:.2e}  dP/P={( P-P0)/P0*100:+.3f}%")

    wall = time.time() - t0_wall
    F_final_int = np.array(F[ic, ic, 0])
    V_final = float(np.sum(F_final_int)*dx*dy)
    P_final = perimeter(F_final_int, dx, dy)
    dV = abs(V_final-V0)/V0
    dP = (P_final-P0)/P0*100

    print(f"  Done {wall:.1f}s  |dV/V|={dV:.2e}  dP/P={dP:+.4f}%")
    results[label] = dict(
        angle=angle_deg, cx0=cx0, cy0=cy0,
        V0=V0, P0=P0, V_final=V_final, P_final=P_final,
        dV=dV, dP=dP,
        snapshots=snapshots,
        F_final=F_final_int,
        grid=grid,
        vol_history=vol_history, per_history=per_history,
    )

# ── Plot 1: Interface shapes for all angles ──────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
NX_plt = NX
x = np.linspace(0.01/2, DOMAIN-0.01/2, NX_plt)
y = x.copy()
X, Y = np.meshgrid(x, y, indexing="ij")

for ax, (label, r) in zip(axes, results.items()):
    g = r["grid"]
    theta_v = np.radians(r["angle"])
    ux, uy = np.cos(theta_v), np.sin(theta_v)

    ax.contourf(X, Y, r["F_final"], levels=np.linspace(0,1,21),
                cmap="Blues", alpha=0.7)
    ax.contour(X, Y, r["F_final"], levels=[0.5], colors="navy", lw=2)

    # Exact final position
    cx_f = r["cx0"] + ux * TRAVEL
    cy_f = r["cy0"] + uy * TRAVEL
    theta_c = np.linspace(0, 2*np.pi, 100)
    ax.plot(cx_f + R*np.cos(theta_c), cy_f + R*np.sin(theta_c),
            "g--", lw=1.5, alpha=0.8, label="Exact")
    # Initial
    ax.plot(r["cx0"] + R*np.cos(theta_c), r["cy0"] + R*np.sin(theta_c),
            "r:", lw=1, alpha=0.5, label="Initial")

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    ax.set_title(f"Angle={label}\n|dV/V|={r['dV']:.2e}  dP/P={r['dP']:+.3f}%",
                 fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Test 1: 2D Droplet Advection — Lagrangian VOF (Barkhudarov 2004)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test1_final_interfaces.png"), dpi=180, bbox_inches="tight")
plt.close()
print("\nSaved: test1_final_interfaces.png")

# ── Plot 2: Volume & perimeter evolution (all angles) ────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
colors = ["tab:blue","tab:orange","tab:green","tab:red"]
for (label, r), col in zip(results.items(), colors):
    n = len(r["vol_history"])
    t_pts = np.linspace(0, TRAVEL, n)
    dV_arr = [abs(V-r["V0"])/r["V0"]*100 for V in r["vol_history"]]
    dP_arr = [(P-r["P0"])/r["P0"]*100 for P in r["per_history"]]
    ax1.semilogy(t_pts, np.maximum(dV_arr, 1e-8), "-o", ms=4, lw=1.5,
                 color=col, label=label)
    ax2.plot(t_pts, dP_arr, "-s", ms=4, lw=1.5, color=col, label=label)

ax1.axhline(0.0002, color="gray", ls="--", lw=1, label="Target 0.0002%")
ax1.set_xlabel("Travel distance"); ax1.set_ylabel("|ΔV/V₀| (%)")
ax1.set_title("Volume Error"); ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

ax2.axhline(1.0, color="gray", ls="--", lw=1, label="Target ±1%")
ax2.axhline(-1.0, color="gray", ls="--", lw=1)
ax2.set_xlabel("Travel distance"); ax2.set_ylabel("ΔP/P₀ (%)")
ax2.set_title("Perimeter Change"); ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

fig.suptitle("Test 1: Volume & Perimeter Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test1_metrics.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test1_metrics.png")

# ── Summary table ────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  {'Angle':<8} {'|dV/V| (%)':>12} {'dP/P (%)':>12} {'Vol pass':>10} {'Peri pass':>10}")
print("  " + "-"*56)
for label, r in results.items():
    vp = "✓" if r["dV"]*100 < 0.0002 else "✗"
    pp = "✓" if abs(r["dP"]) < 1.0 else "✗"
    print(f"  {label:<8} {r['dV']*100:>12.6f} {r['dP']:>12.4f} {vp:>10} {pp:>10}")
print("  " + "-"*56)
print("  Target: |dV/V| < 0.0002%, |dP/P| < 1%")
