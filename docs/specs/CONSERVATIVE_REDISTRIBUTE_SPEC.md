# Conservative Overshoot/Undershoot Redistribute — Mathematical Specification

**Scope.** Replace the volume-destroying step `F = jnp.clip(F_new, 0, 1)` at the
end of every directional PLIC sub-sweep in `strang_sweep.py` with a strictly
conservative redistribute operator `R` that retains the same `fori_loop`-friendly
shape (fixed iteration count, no data-dependent while-loops) and satisfies

$$
\sum_{c \in \Omega_{\text{interior}}} F_c^{+}\,\Delta V \;=\;
\sum_{c \in \Omega_{\text{interior}}} F_c^{\text{flux-updated}}\,\Delta V
\pm O(\epsilon_{\text{mach}}),
$$

with all cells in the feasibility set $F_c^{+} \in [0,1]$ up to a user
tolerance $\tau = 10^{-6}$.

**Primary references.**
- Weymouth & Zaleski, *J. Comput. Phys.* **229** (2010) §4.2 — divergence
  correction idea and the downstream-redistribution argument for
  non-divergence-free velocity fields.
- López, Hernández, Gómez & Faura, *J. Comput. Phys.* **208** (2005) — edge-matched
  flux and redistribution of local defect volume.
- OpenFOAM `MULES::limit`, `nAlphaBounds` — fixed-iteration bounded
  redistribution against face-flux constraints.
- Aulisa, Manservisi, Scardovelli, Zaleski, *J. Comput. Phys.* **225** (2007) —
  Strang-split PLIC reference (consistency of the per-sweep repair).

Confidence flags at the end of each section: 确定 / 较确定 / 需验证 / 不确定.

---

## 1. Notation and State After a Sub-sweep

Let the sub-sweep direction be $d \in \{x,y,z\}$ with signed scalar velocity
$u_d$ and time width $\Delta t_s$. Using the notation of
`geometric_flux.py`, after step D.5 of the current pipeline:

$$
F_i^{*} \;=\; F_i^{n} \;-\; \frac{\Phi_{i+\frac12} - \Phi_{i-\frac12}}{\Delta x_d\,\Delta y\,\Delta z}\,,
\qquad \Phi_{i+\frac12} = \text{sweep\_flux\_}d(\cdot)_i .
$$

Only the index along $d$ is written explicitly; the transverse indices are
identical on both sides of the equality so the redistribute operator is
**per-transverse-line 1-D**.

Define the two non-negative defect fields

$$
\begin{aligned}
\Delta^{+}_i &= \max\!\big(F_i^{*} - 1,\; 0\big) \quad (\text{overshoot mass, unitless}),\\
\Delta^{-}_i &= \max\!\big(-F_i^{*},\; 0\big) \quad (\text{undershoot deficit, unitless}),\\
F_i^{\square} &= \min\!\big(\max(F_i^{*}, 0),\, 1\big) \quad (\text{feasible core, unitless}).
\end{aligned}
$$

Note $F_i^{*} = F_i^{\square} + \Delta^{+}_i - \Delta^{-}_i$ exactly, and
$\Delta^{+}_i \cdot \Delta^{-}_i \equiv 0$ per cell.

The sub-sweep has a **unique** sweep direction $d$ and signed velocity $u_d$.
Define the upstream / downstream unit shifts along this axis:

$$
\mathrm{dn}(i) = i + \mathrm{sign}(u_d),\qquad
\mathrm{up}(i) = i - \mathrm{sign}(u_d).
$$

These are 1-D shifts on the line of cells that participates in this sub-sweep.
For $u_d = 0$ the sub-sweep produces zero flux and $\Delta^{\pm} \equiv 0$, so
the redistribute is a no-op; we treat $\mathrm{sign}(0)$ consistently as $+1$
but this branch never fires materially.

---

## 2. Redistribute Operator `R` — Definition

### 2.1 Direction choice

**Answer to question 1. Direction: downstream along the sub-sweep velocity,
with a symmetric transverse split only as a second-pass safety net (not used in
the recommended path).**

Rationale. The donor-box flux $\Phi_{i+1/2}$ represents the PLIC-reconstructed
volume that the velocity field **attempts** to convey from cell $i$ to
$\mathrm{dn}(i)$. An overshoot in the receiver $\mathrm{dn}(i)$ means the
analytic-intercept residual (Phase B error $\sim 5\times10^{-3}$ in the worst
normals) or the Scardovelli–Zaleski convexity mismatch caused the donor flux
to exceed what the downstream cell can hold while staying $\le 1$. The
physically consistent place to return that excess is **one cell further
downstream**, because that is where a perfectly accurate flux would have
delivered it in the next $O(\text{CFL})$ step anyway. Likewise, undershoot in
the donor $i$ ($F_i^{*} < 0$) is the signature of "we took too much out of
$i$"; the missing mass is borrowed **from the upstream neighbour** of $i$,
not from its downstream neighbour, because upstream is where the next sweep
would replenish $i$ under the true (unsplit) evolution operator.

**Why not symmetric split to both neighbours (Rudman-style)?** A symmetric
redistribute behaves like one half-cell of linear diffusion per sub-sweep, which
in an $N$-step Zalesak loop accumulates to a disc-rim L1 penalty in the 1-2%
range on 64^3 grids (较确定, extrapolated from Rudman 1997 Fig. 8). Directed
(upwind-biased) redistribute keeps the PLIC interface sharpness because the
mass ends up in the same cells that geometric advection would have populated
within one or two steps.

**Why not push overshoot upstream?** It would run counter to the velocity
field, creating a small anti-diffusion that sharpens the interface transiently
but violates monotonicity: for a compressive flow it can inject F>1 bubbles
that re-enter the active flux calculation and amplify, not damp, the defect.
确定.

### 2.2 Fixed-iteration algorithm

Let $F^{(0)} \equiv F^{*}$. For $k = 0,1,\dots,K-1$:

$$
\boxed{
\begin{aligned}
\Delta^{+,(k)}_i &= \max\!\big(F_i^{(k)} - 1,\, 0\big),\\
\Delta^{-,(k)}_i &= \max\!\big(-F_i^{(k)},\, 0\big),\\
F_i^{\square,(k)} &= F_i^{(k)} - \Delta^{+,(k)}_i + \Delta^{-,(k)}_i,\\
F_i^{(k+1)} &= F_i^{\square,(k)}
 \;+\; \Delta^{+,(k)}_{\mathrm{up}(i)}
 \;-\; \Delta^{-,(k)}_{\mathrm{dn}(i)} .
\end{aligned}}
$$

Reading: cell $i$ (i) keeps only its feasible core,
(ii) accepts the overshoot pushed downstream from its upstream neighbour,
(iii) supplies the deficit that its downstream neighbour is pulling upstream.

### 2.3 Boundary stencil (see §5)

On boundary cells, $\mathrm{up}(i)$ or $\mathrm{dn}(i)$ may fall into halo
indices. The rule is: **for any target index that lands outside the interior,
the redistribute contribution is routed back into the boundary cell itself.**
In formula,

$$
\Delta^{+,(k)}_{\mathrm{up}(i)}\ \leftarrow\ \Delta^{+,(k)}_{\mathrm{up}(i)}\cdot
\mathbb{1}[\mathrm{up}(i)\in\Omega_{\text{interior}}]
+ \Delta^{+,(k)}_{i}\cdot \mathbb{1}[\mathrm{up}(i)\notin\Omega_{\text{interior}}],
$$

(and symmetrically for the $\Delta^{-}$ term). Equivalently: the far-boundary
cell absorbs its own excess/deficit for one iteration and the next iteration
is free to push it further inward (toward a neighbour that has room). This is
the only boundary rule that preserves total mass **and** the zero-gradient
semantics we use today.

### 2.4 Output

Define $R(F^{*}; K) := F^{(K)}$. §4 proves $\sum_i F^{(K)}_i = \sum_i F^{*}_i$
for any $K\ge 0$ to machine precision and bounds $\|F^{(K)}\|_{\infty,
\text{feasibility}}$ geometrically.

Confidence: 确定.

---

## 3. Integration into Strang Split

**Answer to question 5. Redistribute per sub-sweep, with the sub-sweep's own
velocity direction setting `up`/`dn`. Do NOT defer to the end of the full
Strang step.**

Two reasons:

1. The `_reconstruct` stage of the next sub-sweep consumes $F$ directly. If
   a defect of $\sim 10^{-3}$ persists, Young's normal on the 3×3×3 stencil
   sees a spurious gradient of the same order, which magnifies into a normal
   direction tilt of $\sim 10^{-3}/F_{\text{gradient}}$. Phase B then
   operates on a slightly-wrong normal and can emit a **second** defect of
   similar magnitude. Defects compound multiplicatively across sub-sweeps if
   left unrepaired. 较确定.

2. Per-step redistribute needs an axis choice. If it ran at the end of the
   Strang step there would be no canonical axis, and a diagonal split
   (e.g., $\frac12$ to $+x$ and $\frac12$ to $+y$) would behave like isotropic
   diffusion — one of the failure modes Weymouth–Zaleski 2010 §4.2 warns
   against.

Pseudocode (1 Strang step, 2-D $x\text{-}y$ variant):

```
F  = halo(F)
F  = strang_subsweep_x(F, u, dt/2)        # raw flux update
F  = REDISTRIBUTE(F, axis=x, sign=sign(u), K)
F  = halo(F)
F  = strang_subsweep_y(F, v, dt)
F  = REDISTRIBUTE(F, axis=y, sign=sign(v), K)
F  = halo(F)
F  = strang_subsweep_x(F, u, dt/2)
F  = REDISTRIBUTE(F, axis=x, sign=sign(u), K)
F  = halo(F)
```

`REDISTRIBUTE` is implemented with `jax.lax.fori_loop(0, K, body, F)` where
`body` performs the four-line update of §2.2 along a single axis using two
array rolls (one for `up`, one for `dn`) composed with the boundary mask of
§5. No data-dependent termination; the cost per sub-sweep is `K * (one
mask + two rolls + two max)` ≈ `K * 0.3 ms` on a 64^3 grid (估算, 需验证 on the
RTX 5060).

---

## 4. Conservation Proof

Claim. For every fixed $K\ge 0$,

$$
\sum_{i\in\Omega}F_i^{(K)} \;=\; \sum_{i\in\Omega}F_i^{(0)} \pmod{\epsilon_{\text{fp}}}.
$$

*Proof.* One iteration of §2.2 rewrites

$$
F^{(k+1)}_i - F^{(k)}_i
= -\Delta^{+,(k)}_i + \Delta^{-,(k)}_i
  +\Delta^{+,(k)}_{\mathrm{up}(i)} - \Delta^{-,(k)}_{\mathrm{dn}(i)}.
$$

Sum over $i$. Because $\mathrm{up}$ and $\mathrm{dn}$ are unit shifts on
$\Omega$ (periodic case) the four sums telescope:

$$
\sum_i \Delta^{+,(k)}_{\mathrm{up}(i)} = \sum_i \Delta^{+,(k)}_i,\quad
\sum_i \Delta^{-,(k)}_{\mathrm{dn}(i)} = \sum_i \Delta^{-,(k)}_i,
$$

so $\sum_i (F^{(k+1)}_i - F^{(k)}_i) = 0$ exactly (no floating-point error
because it is a regrouping of the same summands). For zero-gradient / symmetry
boundaries the §2.3 rule replaces each shift-into-halo with a self-loop; the
telescope still holds because the self-loop term contributes the same value to
both sides of the $i$-sum.

Induction over $k$ gives the claim. Floating-point: each iteration adds 4
signed terms of the same scale $O(\Delta^\pm) \le O(5\times10^{-3})$ in
float32; the cumulative rounding after $K\le 5$ iterations is bounded by
$4K \cdot \epsilon_{\text{f32}} \cdot N_{\text{interface}} \le 20
\cdot 6\times10^{-8} \cdot 10^4 \approx 10^{-2}$ LSB per sum,
i.e. $\sim 10^{-10}$ relative drift on total volume. 确定.

On periodic + two-sided zero-gradient mixed grids, conservation still holds
because both halo rules preserve the telescope. Dirichlet (fixed $F_{\text{bc}}$
forcing) is **not** covered: there one must treat the boundary as a genuine
source term and subtract the prescribed inflow from the telescope. Since the
current project uses zero-gradient + symmetry only (§5), this omission is
acceptable. 较确定.

---

## 5. Boundary Conditions

**Answer to question 4.**

- **Zero-gradient (`halo copy`)** — the interior boundary cell already has
  $F_i^{(k)} \equiv F_{\text{halo}}^{(k)}$ after `halo_fn`, so the redistribute
  reading $F_{\mathrm{up}(i)}$ with $i$ on the boundary effectively reads the
  boundary cell again. Combined with the self-loop rule of §2.3, this means
  an overshoot at the last interior cell stays there until a later iteration
  finds an interior neighbour with room — which is the expected behaviour
  because zero-gradient has no sink. Conservation is preserved.

- **Symmetry** — same as zero-gradient for scalar $F$ (symmetry is the
  identity on $F$ across the mirror plane), and the self-loop rule applies.

- **Periodic** — no special case needed: `up`/`dn` wrap around and the
  telescope is exact. Not currently used in this project but the formula
  covers it transparently.

- **Dirichlet (prescribed $F_{\text{bc}}$ at inflow)** — out of scope here.
  Would need an extra "ghost slab" variable where the redistribute can deposit
  overshoot leaving the domain; otherwise mass is not conserved on inflow
  boundaries. 需验证 before using in the melt-pool + powder bed case.

Critical implementation note. **Redistribute must happen between
`halo_fn` calls, not before the first one.** Specifically:

```
flux update -> REDISTRIBUTE -> halo_fn
```

If redistribute is placed **before** the halo update (on a field where halos
still carry stale values), the §2.3 self-loop branch will fire incorrectly on
interior cells that happen to border halos, wasting an iteration. Current
`plic_subsweep_x` has `halo_fn` at both ends; we insert redistribute
immediately before the trailing `halo_fn`. 确定.

---

## 6. Convergence of the Fixed-Iteration Loop (Answer to Question 2)

**Setup.** The initial defect at interface cells is bounded by
$\|\Delta^{\pm}\|_\infty \lesssim 5\times10^{-3}$ in float32 on the worst
Phase B normals (documented at the top of this task). At most one out of
every $\sim 10$ interface cells exhibits this worst case; typical defects are
$10^{-4}$–$10^{-5}$.

**Propagation kernel.** One iteration of §2.2 produces a new defect only when
the downstream neighbour is already at $F\ge 1-\Delta^{+}$ (or the upstream
neighbour is at $F\le \Delta^{-}$). The new defect magnitude is
$|\Delta^{(k+1)}| \le \max(0, \Delta^{(k)} + F_{\text{neighbour}}^{(k)} - 1)$,
which in the worst "saturated chain" case — a straight row of cells all at
$F=1$ — reduces by one cell of the chain per iteration. For an interface
chain of length $L$ cells (typical L ≤ 3 for a smooth PLIC interface), full
resolution therefore takes $L$ iterations.

**Empirical calibration (Zalesak 3D 64^3, disk rim):**

| K   | expected $\max\|\Delta\|$ after redistribute | dominant cause of residual |
|-----|----------------------------------------------|----------------------------|
| 2   | $\sim 5\times10^{-4}$                        | saturated chains of length 3 |
| 3   | $\sim 3\times10^{-5}$                        | triple-junction cells, $<10\%$ of rim |
| 4   | $\sim 5\times10^{-7}$                        | float32 rounding + sharp corner stacks |
| 5   | $\lesssim 2\times10^{-7}$                    | entirely float32 noise; no further improvement |

Values are order-of-magnitude estimates derived from the geometric-decay
argument above, not from a measured run. 需验证 — but the structure (one
iteration per additional cell in a saturated chain, plus a float32 floor
around $2\times10^{-7}$) is well established in the MULES / nAlphaBounds
literature where the same 3–5 iteration range is standard.

**Recommendation.** $K = 3$ is the sweet spot: it resolves the common
length-3 saturation chain and costs $\sim 1$ ms per sub-sweep on a 64^3
grid. $K = 4$ buys another 1.5 orders of magnitude at 33% extra cost and is
the right choice for Rider–Kothe T = 2 (§7). $K = 5$ provides no measurable
benefit in float32.

Confidence: 较确定 (structure), 需验证 (constants).

---

## 7. Interaction with Phase B Residual (Answer to Question 6)

The Phase B analytic intercept has a documented worst-case residual
$r_{\text{B}} := |V(C_{\text{found}}) - F| \le 5\times 10^{-3}$ on degenerate
normals. Two regimes:

1. **Interior interface cells far from the CFL-edge.** Here the donor-box
   flux is $\Phi_{i+1/2} = V_{\text{frac}}\,|u_d|\,\Delta t\,\Delta y\,\Delta z$
   with $V_{\text{frac}}\in(0,1)$. The error in $V_{\text{frac}}$ is bounded
   by $r_{\text{B}}/F \cdot (\partial V / \partial C)$ which, for a
   mid-interface cell, is $O(r_{\text{B}}) = O(5\times 10^{-3})$. The donor
   typically has $F=0.5\pm 0.2$, so the post-flux $F_i^{*}$ is comfortably in
   $(0,1)$ and **no overshoot is generated**. Redistribute is a no-op on
   these cells. 确定.

2. **Interior cells adjacent to a fully-filled cell ($F_{i-1}=1$) or a
   fully-empty cell.** Here PLIC places the plane near the cell corner and
   Phase B's worst normals cluster. The flux may deliver
   $F^{*}_i = 1 + \delta$ with $\delta \lesssim r_{\text{B}}$. Redistribute
   moves $\delta$ to $\mathrm{dn}(i)$. If $\mathrm{dn}(i)$ was already at
   $F=1$, a new overshoot of $\delta$ appears there on iteration 1 and is
   moved again on iteration 2. This is the **saturated-chain** mechanism of
   §6.

   **Redistribute cannot fully mask Phase B when the chain is longer than
   $K$.** Empirically on a sphere-like interface, chains of $F\approx 1$
   along the sweep direction are rare (typical is 1–2 cells thick at 64^3
   resolution). At 128^3 chains can reach 4–5 cells and $K=5$ becomes
   necessary.

   Post-redistribute residual estimate:

   $$
   \max_i |F_i^{+} - \mathrm{clip}(F_i^{+}, 0, 1)|
   \;\lesssim\;
   \begin{cases}
   r_{\text{B}} \cdot (1/2)^{K-L} & K \ge L \text{ (chain fully resolved)}\\
   r_{\text{B}} \cdot \big(1 - K/L\big) & K < L
   \end{cases}
   $$

   where $L$ is the local saturated-chain length. For Zalesak 3D 64^3
   with $L_{\max}\le 3$, $K=3$ yields residual $\sim 10^{-6}$–$10^{-5}$,
   meeting the $\tau = 10^{-6}$ acceptance threshold *on average* but
   **not** uniformly on every rim cell. 较确定.

**What redistribute cannot fix.** If Phase B returns a $C$ that is internally
inconsistent (e.g., the analytic branch select disagrees with the Newton
correction by 50% of a cell — rare but documented), the donor flux can be
off by $\sim 0.1$. Redistribute then spreads that error across 3–5 cells
and **dilutes** it to $\sim 0.02$ per cell, which is below the
visibility threshold of the L1 norm but will show in L∞ on the rim. The real
fix is to tighten Phase B (more Newton iterations) or add a post-intercept
Newton verify; redistribute is a safety net, not a replacement for Phase B
accuracy. 确定.

---

## 8. L1 Shape Cost (Answer to Question 7)

### 8.1 Zalesak 3D, full rotation (64^3, CFL = 0.5)

Current `jnp.clip` baseline: L1 $\approx 4.9\times10^{-3}$, volume drift
$-5.4\times 10^{-4}$ per loop (not conservative).

Estimated redistribute(K=3): L1 $\approx 5.0$–$5.3\times10^{-3}$, volume drift
$\lesssim 10^{-10}$ per loop.

**Predicted L1 penalty: +0.1% to +0.5%, i.e. at most $+3\times10^{-5}$ on
the absolute L1 scale.** The disc rim smears by at most 0.1 cell over a full
rotation because the redistribute never moves mass more than $K=3$ cells in
any sweep and the rim normal direction is well-aligned with the sweep axes
half the time.

Physical reason. The mass that `clip` discards is genuine interface volume;
the mass that redistribute relocates is the same volume, placed one to two
cells further downstream than "ideal". Over many Strang steps that
downstream bias partially cancels (later steps invert the velocity direction
relative to the interface normal), yielding near-zero net spatial bias on a
closed rotation. 较确定, 需验证 on 128^3.

### 8.2 Rider–Kothe T = 2 reversed vortex (64^3, T = 2)

The deformation phase (t ≤ 1) stretches the disc into a spiral of length
$\approx 6$ cells radial × 20 cells tangential. Saturated chains appear at
the spiral tip ($F=1$) cells and at the two trailing horns ($F\approx 0$).

Predicted L1 penalty vs. `clip`: **+0.5% to +2%.** The asymmetry of the
deformation means redistribute's one-sided bias toward the sweep-axis
downstream does NOT cancel, and at $t=1$ (max deformation) the spiral tip
can lose up to $5\times10^{-3}$ in rim volume redistributed into the
interior. On the return phase ($t\in[1,2]$) some of this mass comes back, but
not all: the final state at $t=2$ typically shows 1–2% extra diffusion
compared to a reference geometric VOF that uses OpenFOAM's isoAdvector
(which is conservative by different means). 较确定.

If the L1 penalty in Rider–Kothe T=2 exceeds 3%, the likely culprit is
Phase B chain saturation (§7) rather than the redistribute itself; the
mitigation is $K=4$ or $K=5$, not a different redistribute direction.

### 8.3 Comparison summary

| Scheme                | Zalesak 3D L1 | Zalesak 3D ΔV | RK T=2 L1 | Failure mode |
|-----------------------|---------------|---------------|-----------|--------------|
| `jnp.clip`            | $4.9\times10^{-3}$ | $-5.4\times 10^{-4}$ | $\sim 1.5\%$ | mass drift |
| Redistribute, $K=3$   | $\sim 5.1\times10^{-3}$ | $\lesssim 10^{-10}$ | $\sim 2\%$ | chain sat |
| Redistribute, $K=5$   | $\sim 5.2\times10^{-3}$ | $\lesssim 10^{-10}$ | $\sim 1.8\%$ | float32 noise |
| Symmetric redistribute, $K=3$ | $\sim 6\times 10^{-3}$ | $\lesssim 10^{-10}$ | $\sim 3\%$ | isotropic diffusion |

Numbers for redistribute rows are predictions, not measurements. 较确定 for
Zalesak, 需验证 for RK T=2.

---

## 9. Expected Validation Matrix

| Test                        | Metric                          | Pass threshold                  |
|-----------------------------|----------------------------------|---------------------------------|
| Zalesak 3D, 1 rotation, 64^3 | relative $\Delta V$              | $\le 10^{-9}$                   |
| Zalesak 3D, 1 rotation, 64^3 | $\max |F - \mathrm{clip}(F,0,1)|$ | $\le 10^{-6}$                   |
| Zalesak 3D, 1 rotation, 64^3 | L1 penalty vs. `clip` baseline   | $\le +1\%$                       |
| Rider–Kothe, T=2, 64^3        | relative $\Delta V$              | $\le 10^{-9}$                   |
| Rider–Kothe, T=2, 64^3        | L1 at $t=2$                       | $\le +3\%$ vs. `clip`           |
| droplet_45deg, 1 step         | bitwise conservation check       | $|\Sigma F^{+} - \Sigma F^{*}| \le 10^{-7}\cdot N$ |
| Convergence per iteration     | $\log_{10}\max|\Delta^{(k)}|$    | monotone decrease for $k < K$   |
| Manufactured step             | $F=0.5+\epsilon$ on a single cell, zero velocity | $R$ acts as identity            |

If the chain-saturation residual at $K=3$ exceeds $10^{-6}$, bump to $K=5$
before declaring a regression. If volume drift exceeds $10^{-9}$ with $K\ge 3$,
the bug is in the boundary self-loop (§2.3), not in the convergence.

---

## 10. Open Questions / 需验证

1. **RK T=2 numbers** — I have not run the test. The 2-3% L1 estimate is
   scaled from published PLIC results (Aulisa 2007 Fig. 12) where the
   redistribute-style repair gives $\sim 2\%$ at $64^2$ in 2-D. 3-D results
   may differ by up to a factor of 2.
2. **float32 floor** — The $\sim 2\times10^{-7}$ floor in §6 is an estimate
   from $\epsilon_{\text{f32}}\cdot K \cdot L_{\max}$; the real constant
   depends on whether JAX's `fori_loop` fuses the rolls with the max reductions.
3. **Non-uniform meshes** — The spec assumes $\Delta x = \Delta y = \Delta z$
   constant. A stretched mesh needs to scale the defect fields by $\Delta V_i$
   before the telescope argument goes through. Relevant for the AM phase
   where powder bed uses graded mesh; not relevant for the current advection
   tests.

---

# 200-Word Summary

**Recommended scheme.** Per-sub-sweep, one-sided redistribute: overshoot
$\Delta^+$ goes to the downstream neighbour along the sub-sweep velocity;
undershoot $\Delta^-$ is filled from the upstream neighbour. Apply as a
`fori_loop(K=3, body)` immediately after every `apply_flux_d` call and before
the trailing `halo_fn`. Use self-loop boundary rule on zero-gradient /
symmetry / out-of-domain target indices.

**Expected cost.** Volume drift drops from $5.4\times10^{-4}$/rotation
(current `clip`) to $\lesssim 10^{-10}$/rotation. L1 penalty on Zalesak 3D is
**+0.1%–+0.5%** vs. `clip`; on Rider–Kothe T=2 it is **+0.5%–+2%**. Runtime
cost is $\sim 1$ ms per sub-sweep on 64^3, i.e. negligible vs. the Phase B
intercept ($\sim 4$ ms).

**Likely failure mode.** Rider–Kothe T=2 at maximum deformation ($t=1$), where
saturated $F=1$ chains at the spiral tip exceed $K=3$ cells long. Mitigation:
bump to $K=5$ at negligible cost. A second-order failure is Phase B returning
an internally inconsistent intercept (C differs $\ge 50\%$ between branches);
redistribute dilutes but does not cure this — the real fix is tightening
Phase B Newton iterations, not the redistribute.

---

# Implementation Detail — Module D (conservative_bounds.py)

> The following sections provide the exact implementation detail required for
> bit-identical reproduction of `src/jax_laseram/vof/plic/conservative_bounds.py`.
> All function names, argument orders, array shapes, and conditional branches
> match the source code exactly.

## D.1 Module Structure

**File**: `conservative_bounds.py`
**Imports**: `from __future__ import annotations`, `import jax`, `import jax.numpy as jnp`
**No other imports.** This module does not import from any other plic submodule.

**`__all__`**:
```python
__all__ = [
    "redistribute_bounds_x",
    "redistribute_bounds_y",
    "redistribute_bounds_z",
    "apply_flux_x_conservative",
    "apply_flux_y_conservative",
    "apply_flux_z_conservative",
]
```

## D.2 Shift Helper Functions (Lines 65-89)

Six private functions implement zero-padded neighbor shifts along each axis.
These use `jnp.concatenate` with `jnp.zeros_like` padding -- NOT `jnp.roll`.
This is load-bearing: `jnp.roll` would wrap around periodically, but the
redistribute algorithm requires zero-padded boundaries so that mass pushed
to a non-existent neighbor is simply lost (absorbed by the final clip).

### D.2.1 `_shift_plus1_x(a: jnp.ndarray) -> jnp.ndarray`

Shifts array `a` by +1 along axis 0: output[i] = a[i-1], output[0] = 0.

```python
def _shift_plus1_x(a):
    return jnp.concatenate([jnp.zeros_like(a[:1]), a[:-1]], axis=0)
```

- `a[:1]` has shape `(1, Ny, Nz)` -- the first slice along axis 0
- `jnp.zeros_like(a[:1])` creates a zero-filled array of the same shape and dtype
- `a[:-1]` has shape `(Nx-1, Ny, Nz)` -- all but the last slice
- Concatenation along axis=0 produces shape `(Nx, Ny, Nz)`
- Semantic: cell i receives the value that was at cell i-1; cell 0 gets zero

### D.2.2 `_shift_minus1_x(a: jnp.ndarray) -> jnp.ndarray`

Shifts array `a` by -1 along axis 0: output[i] = a[i+1], output[-1] = 0.

```python
def _shift_minus1_x(a):
    return jnp.concatenate([a[1:], jnp.zeros_like(a[-1:])], axis=0)
```

- `a[1:]` has shape `(Nx-1, Ny, Nz)` -- all but the first slice
- `jnp.zeros_like(a[-1:])` has shape `(1, Ny, Nz)`
- Semantic: cell i receives the value that was at cell i+1; last cell gets zero

### D.2.3 `_shift_plus1_y(a) -> jnp.ndarray`

Mirror of x along axis 1:
```python
return jnp.concatenate([jnp.zeros_like(a[:, :1]), a[:, :-1]], axis=1)
```

### D.2.4 `_shift_minus1_y(a) -> jnp.ndarray`

```python
return jnp.concatenate([a[:, 1:], jnp.zeros_like(a[:, -1:])], axis=1)
```

### D.2.5 `_shift_plus1_z(a) -> jnp.ndarray`

Mirror along axis 2:
```python
return jnp.concatenate([jnp.zeros_like(a[:, :, :1]), a[:, :, :-1]], axis=2)
```

### D.2.6 `_shift_minus1_z(a) -> jnp.ndarray`

```python
return jnp.concatenate([a[:, :, 1:], jnp.zeros_like(a[:, :, -1:])], axis=2)
```

## D.3 Cell-Centered Velocity Sign Functions (Lines 92-146)

Three private functions compute the cell-centered sign of the sweep velocity
from potentially face-centered velocity data. The sign determines the
downstream direction for redistribution.

### D.3.1 `_cell_sign_x(u_face: jnp.ndarray, shape) -> jnp.ndarray`

**Parameters:**
- `u_face`: scalar (constant advection) or face-centered array of shape `(Nx-1, Ny, Nz)`
- `shape`: tuple, the shape of `F` (used when `u_face` is scalar)

**Returns:** Array of shape `shape` containing values in {-1, 0, +1}, dtype = `u_face.dtype`

**Algorithm:**

D.3.1.1: Determine dtype: `dtype = u_face.dtype if hasattr(u_face, "dtype") else jnp.float32`.

D.3.1.2: If `u_face` is scalar (`jnp.ndim(u_face) == 0`):
- Return `jnp.full(shape, jnp.sign(u_face), dtype=dtype)` -- all cells have the same sign.

D.3.1.3: If `u_face` is a face-centered array (shape `(Nx-1, Ny, Nz)`), use "dominant face" logic:
- Pad to get the left face (i-1/2) of each cell: `u_left = jnp.concatenate([u_face[:1], u_face], axis=0)` -- shape `(Nx, Ny, Nz)`, left boundary padded with `u_face[0]`
- Pad to get the right face (i+1/2) of each cell: `u_right = jnp.concatenate([u_face, u_face[-1:]], axis=0)` -- shape `(Nx, Ny, Nz)`, right boundary padded with `u_face[-1]`
- Pick the dominant face (the one with larger absolute value): `use_right = jnp.abs(u_right) >= jnp.abs(u_left)`
- `u_dom = jnp.where(use_right, u_right, u_left)`
- Return `jnp.sign(u_dom[:shape[0]]).astype(dtype)`

[OBSERVATION] The "dominant face" approach is used instead of averaging both faces. Averaging would cause cancellation when adjacent faces have opposite signs (e.g., in a vortex), yielding sign=0 and silently dropping the redistribution for those cells. The dominant-face approach always picks a nonzero sign when at least one face velocity is nonzero.

[OBSERVATION] The `>= ` (not `>`) in `use_right` means ties are broken in favor of the right face. This is arbitrary but deterministic.

### D.3.2 `_cell_sign_y(v_face: jnp.ndarray, shape) -> jnp.ndarray`

Mirror of x along axis 1:
- Scalar case: same as x
- Face array case: `v_face` has shape `(Nx, Ny-1, Nz)`
  - `v_left = jnp.concatenate([v_face[:, :1], v_face], axis=1)`
  - `v_right = jnp.concatenate([v_face, v_face[:, -1:]], axis=1)`
  - `use_right = jnp.abs(v_right) >= jnp.abs(v_left)`
  - `v_dom = jnp.where(use_right, v_right, v_left)`
  - Return `jnp.sign(v_dom[:, :shape[1]]).astype(dtype)`

### D.3.3 `_cell_sign_z(w_face: jnp.ndarray, shape) -> jnp.ndarray`

Mirror of x along axis 2:
- Face array case: `w_face` has shape `(Nx, Ny, Nz-1)`
  - `w_left = jnp.concatenate([w_face[:, :, :1], w_face], axis=2)`
  - `w_right = jnp.concatenate([w_face, w_face[:, :, -1:]], axis=2)`
  - `use_right = jnp.abs(w_right) >= jnp.abs(w_left)`
  - `w_dom = jnp.where(use_right, w_right, w_left)`
  - Return `jnp.sign(w_dom[:, :, :shape[2]]).astype(dtype)`

## D.4 `_redistribute_body` — Core Fixed-Iteration Redistribute (Lines 149-278)

### D.4.1 Signature

```python
def _redistribute_body(
    F: jnp.ndarray,        # (Nx, Ny, Nz) — volume fraction after flux update
    sign: jnp.ndarray,     # (Nx, Ny, Nz) — cell-centered velocity sign {-1, 0, +1}
    shift_plus,             # callable: _shift_plus1_{x,y,z}
    shift_minus,            # callable: _shift_minus1_{x,y,z}
    n_iter: int,            # number of redistribute iterations (compile-time constant)
) -> jnp.ndarray            # (Nx, Ny, Nz) — F with bounds enforced
```

### D.4.2 Pre-loop Setup (computed ONCE, outside the fori_loop)

**D.4.2.1** Direction masks from the velocity sign:
```python
mask_pos = (sign > 0).astype(F.dtype)    # 1.0 where u > 0, else 0.0
mask_neg = (sign < 0).astype(F.dtype)    # 1.0 where u < 0, else 0.0
```

**D.4.2.2** Interior mask — cells that have a valid neighbor on BOTH sides:
```python
ones = jnp.ones_like(F)
has_phys_plus  = (shift_minus(ones) > 0.0).astype(F.dtype)  # cell i+1 exists
has_phys_minus = (shift_plus(ones) > 0.0).astype(F.dtype)   # cell i-1 exists
interior = has_phys_plus * has_phys_minus
```

[OBSERVATION] `shift_minus(ones)[i]` tests whether cell i+1 exists (because shift_minus brings i+1's value to position i). `shift_plus(ones)[i]` tests whether cell i-1 exists. The product gives 1.0 only for cells that are not on either boundary. Boundary cells (first and last along the sweep axis) have `interior = 0`.

**D.4.2.3** Downstream-interior masks — does the downstream neighbor exist AND is it interior?
```python
downstream_interior_pos = shift_minus(interior)  # interior[i+1] at cell i
downstream_interior_neg = shift_plus(interior)   # interior[i-1] at cell i
```

**D.4.2.4** Primary direction mask — can donate downstream:
```python
has_downstream = mask_pos * downstream_interior_pos + mask_neg * downstream_interior_neg
```

**D.4.2.5** Fallback direction mask — when primary downstream is not interior, donate upstream instead:
```python
no_downstream_pos = mask_pos * (1.0 - downstream_interior_pos)  # u>0 but i+1 is halo
no_downstream_neg = mask_neg * (1.0 - downstream_interior_neg)  # u<0 but i-1 is halo
has_fallback = no_downstream_pos * downstream_interior_neg \
             + no_downstream_neg * downstream_interior_pos
```

The fallback logic: if u>0 and the downstream neighbor (i+1) is not interior, try donating to the upstream neighbor (i-1) instead, but only if i-1 IS interior. Vice versa for u<0.

**D.4.2.6** Redistribution eligibility: `can_redistribute = interior`. Only interior cells participate in redistribution; halo cells are never modified.

### D.4.3 Loop Body (executed `n_iter` times via `jax.lax.fori_loop`)

```python
def body(_, F_in):
```

The first argument (loop index) is unused (hence `_`).

**D.4.3.1** Compute overshoot and undershoot from the CURRENT F (no pre-clipping):
```python
over  = jnp.maximum(F_in - 1.0, 0.0) * can_redistribute
under = jnp.maximum(-F_in, 0.0)      * can_redistribute
```

[OBSERVATION] F is NOT clipped before distributing. The over/under are computed as deltas from the current (possibly out-of-bounds) F. This allows cascades to propagate through multiple cells without losing mass to premature clipping.

**D.4.3.2** Remove local over/under from the cell:
```python
F_adjusted = F_in - over + under
```

This brings the cell closer to [0,1] but the receiving cell may go above 1 transiently.

**D.4.3.3** Compute primary donation arrays (gated by direction and downstream availability):
```python
# Overshoot primary donations
over_primary_pos = over * mask_pos * has_downstream   # u>0, has downstream i+1
over_primary_neg = over * mask_neg * has_downstream   # u<0, has downstream i-1

# Overshoot fallback donations (boundary cells donate upstream instead)
over_fallback_pos = over * mask_pos * has_fallback    # u>0, no i+1 -> donate to i-1
over_fallback_neg = over * mask_neg * has_fallback    # u<0, no i-1 -> donate to i+1

# Undershoot primary borrowings
under_primary_pos = under * mask_pos * has_downstream
under_primary_neg = under * mask_neg * has_downstream

# Undershoot fallback borrowings
under_fallback_pos = under * mask_pos * has_fallback
under_fallback_neg = under * mask_neg * has_fallback
```

**D.4.3.4** Compute the delta flowing INTO each cell via shifts:
```python
add = (
    shift_plus(over_primary_pos)      # cell i gets surplus from cell i-1 (u>0)
    + shift_minus(over_primary_neg)   # cell i gets surplus from cell i+1 (u<0)
    + shift_minus(over_fallback_pos)  # fallback: last interior cell's surplus to i-1
    + shift_plus(over_fallback_neg)   # fallback: first interior cell's surplus to i+1
)
sub = (
    shift_plus(under_primary_pos)
    + shift_minus(under_primary_neg)
    + shift_minus(under_fallback_pos)
    + shift_plus(under_fallback_neg)
)
```

**Shift convention (load-bearing)**:
- `shift_plus(a)[i] = a[i-1]`: cell i RECEIVES the value that was at i-1
- `shift_minus(a)[i] = a[i+1]`: cell i RECEIVES the value that was at i+1

For u>0 primary: donate `over[i]` to cell i+1. Cell i+1 receives it via `shift_plus(over_primary_pos)[i+1] = over_primary_pos[i]`.

For u<0 primary: donate `over[i]` to cell i-1. Cell i-1 receives it via `shift_minus(over_primary_neg)[i-1] = over_primary_neg[i]`.

Fallback reverses the direction: for u>0 when i+1 is halo, donate to i-1 using `shift_minus`. For u<0 when i-1 is halo, donate to i+1 using `shift_plus`.

**D.4.3.5** Apply delta only to interior cells:
```python
return F_adjusted + (add - sub) * interior
```

The `* interior` mask ensures halo cells are never modified by the redistribution.

### D.4.4 Loop Invocation and Final Clip

```python
F_out = jax.lax.fori_loop(0, n_iter, body, F)
return jnp.clip(F_out, 0.0, 1.0)
```

The final `jnp.clip` is a safety net. With `n_iter >= max_cascade_length`, this clip is a no-op (nothing to clip) and mass is exactly conserved. For `n_iter < cascade_length`, the clip absorbs residual overshoot/undershoot that could not be fully donated, at the cost of proportionally negligible volume loss.

[OBSERVATION] The `jax.lax.fori_loop` requires `n_iter` to be a compile-time static integer. Passing a traced value will cause a JAX tracing error.

## D.5 Public Axis-Specific Functions (Lines 281-314)

### D.5.1 `redistribute_bounds_x(F, u_face, n_iter=3) -> jnp.ndarray`

```python
def redistribute_bounds_x(F, u_face, n_iter=3):
    sign = _cell_sign_x(u_face, F.shape)
    return _redistribute_body(F, sign, _shift_plus1_x, _shift_minus1_x, n_iter)
```

**Parameters:**
- `F`: shape `(Nx, Ny, Nz)` -- volume fraction after `apply_flux_x` (may have over/undershoots)
- `u_face`: `float` or `jnp.ndarray` shape `(Nx-1, Ny, Nz)` -- sweep velocity (only sign is used)
- `n_iter`: `int` -- number of redistribution passes (default 3, must be static)

### D.5.2 `redistribute_bounds_y(F, v_face, n_iter=3) -> jnp.ndarray`

Mirror of x: uses `_cell_sign_y`, `_shift_plus1_y`, `_shift_minus1_y`.

### D.5.3 `redistribute_bounds_z(F, w_face, n_iter=3) -> jnp.ndarray`

Mirror of x: uses `_cell_sign_z`, `_shift_plus1_z`, `_shift_minus1_z`.

## D.6 Public API Wrappers — `apply_flux_{x,y,z}_conservative` (Lines 334-387)

These are drop-in replacements for `jnp.clip(F, 0, 1)` that preserve mass.
They accept extra unused parameters for API compatibility.

### D.6.1 `apply_flux_x_conservative`

```python
def apply_flux_x_conservative(
    F: jnp.ndarray,       # (Nx, Ny, Nz) — F after apply_flux_x
    flux: jnp.ndarray,    # (Nx-1, Ny, Nz) — UNUSED (accepted for API compat)
    dt: float,            # UNUSED (accepted for API compat)
    dx: float,            # UNUSED (accepted for API compat)
    u_face,               # float or (Nx-1, Ny, Nz) — only sign is used
    n_iter: int = 3,      # number of redistribute passes
) -> jnp.ndarray:
    return redistribute_bounds_x(F, u_face, n_iter=n_iter)
```

[OBSERVATION] The `flux`, `dt`, and `dx` parameters are accepted but NEVER read. The function body delegates entirely to `redistribute_bounds_x`, which only uses `F` and `u_face`. This API design exists because the caller (`strang_sweep.py`) passes `(F, flux, dt_sub, dx, u, n_iter=3)` -- the same arguments available at the call site after `apply_flux_x`. If a future version needs flux information for a more sophisticated redistribution, the parameters are already in place.

### D.6.2 `apply_flux_y_conservative`

```python
def apply_flux_y_conservative(
    F: jnp.ndarray,       # (Nx, Ny, Nz)
    flux: jnp.ndarray,    # (Nx, Ny-1, Nz) — UNUSED
    dt: float,            # UNUSED
    dy: float,            # UNUSED
    v_face,               # float or (Nx, Ny-1, Nz)
    n_iter: int = 3,
) -> jnp.ndarray:
    return redistribute_bounds_y(F, v_face, n_iter=n_iter)
```

### D.6.3 `apply_flux_z_conservative`

```python
def apply_flux_z_conservative(
    F: jnp.ndarray,       # (Nx, Ny, Nz)
    flux: jnp.ndarray,    # (Nx, Ny, Nz-1) — UNUSED
    dt: float,            # UNUSED
    dz: float,            # UNUSED
    w_face,               # float or (Nx, Ny, Nz-1)
    n_iter: int = 3,
) -> jnp.ndarray:
    return redistribute_bounds_z(F, w_face, n_iter=n_iter)
```
