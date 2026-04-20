# Phase 1 — DILU 串行本质的数学剖析与 JAX-FFI 原型数学规范

**Project**: JAX-GPU AM-CFD Platform — Stiff Pressure Poisson Solver Kernel Research
**Phase**: 1 / 4 — 理论打底与 JAX-FFI 原型跑通 (Week 1)
**Author role**: CFD-math expert (this document)
**Counterpart**: jax-cfd-am-expert (engineering team, Part 2 implementation)
**Date**: 2026-04-19
**Scope**: Mathematical analysis only. No code. Sets the rigor baseline for the project.

---

## 0. Notation and Problem Setting

Let $A \in \mathbb{R}^{n \times n}$ be the sparse matrix arising from a cell-centered FVM discretization of the variable-coefficient pressure Poisson equation

$$
-\nabla \cdot \!\left( \frac{1}{\rho}\nabla p \right) = -\nabla \cdot \mathbf{u}^* / \Delta t
$$

on a structured (or block-structured) grid. Under the standard 7-point stencil in 3D with a cell-centered collocation, $A$ is:

- **Sparse**: nnz per row $\leq 7$ (interior), with sparsity pattern $\mathcal{S}(A) = \{(i,j) : A_{ij} \neq 0\}$.
- **Symmetric** ($A = A^\top$) when the face-averaged $1/\rho$ coefficient is the same for $(i,j)$ and $(j,i)$, which it is for a cell-centered FVM with harmonic (or arithmetic) face averaging.
- **Diagonally dominant in the weak sense** ($|A_{ii}| \geq \sum_{j \neq i}|A_{ij}|$), with strict dominance at Dirichlet-adjacent rows. Pure-Neumann (closed-box) problems leave $A$ singular with null space $\mathrm{span}(\mathbf{1})$.
- **Positive semi-definite** under the above assumptions; positive definite once any Dirichlet face is imposed or the constant mode is pinned.

We will write $A = D + L + U$, where $D = \mathrm{diag}(A)$, $L$ is the strict lower-triangular part, $U$ is the strict upper-triangular part. For a symmetric $A$ we have $U = L^\top$. The sparsity pattern of $L$ (resp. $U$) is $\mathcal{S}(L) = \{(i,j) \in \mathcal{S}(A) : i > j\}$.

**Confidence tag for this section**: 确定.

---

## 1. ILU(0) and DILU — Precise Definitions

### 1.1 ILU(0): the zero-fill incomplete $LU$

Classic incomplete LU (Saad 2003, *Iterative Methods for Sparse Linear Systems*, §10.3) computes factors $\tilde L$, $\tilde U$ with

$$
\mathcal{S}(\tilde L) \cup \mathcal{S}(\tilde U) \subseteq \mathcal{S}(A), \qquad (\tilde L \tilde U)_{ij} = A_{ij} \quad \forall (i,j) \in \mathcal{S}(A),
$$

i.e. the product equals $A$ exactly on the original sparsity pattern and is allowed to differ (the "fill-in error") off-pattern. The recursive construction (IKJ-form, row-wise) is:

$$
\begin{aligned}
&\text{for } i = 1, \ldots, n:\\
&\quad \text{for } k = 1, \ldots, i-1 \text{ with } (i,k) \in \mathcal{S}(A):\\
&\qquad a_{ik} \leftarrow a_{ik} / a_{kk} \quad\text{($\tilde L_{ik}$)}\\
&\qquad \text{for } j = k+1, \ldots, n \text{ with } (i,j) \in \mathcal{S}(A):\\
&\qquad\quad a_{ij} \leftarrow a_{ij} - a_{ik} \, a_{kj}
\end{aligned}
$$

The diagonal pivot $a_{kk}$ after this update is $\tilde U_{kk}$. ILU(0) requires $A$ to be an $H$-matrix (or M-matrix) for the factorization to be well-defined without breakdown (Meijerink & van der Vorst 1977).

**Cost**: $O(\mathrm{nnz}(A))$ flops for the factorization. No fill, so factors have the same memory as $A$.

### 1.2 DILU: OpenFOAM's diagonal-incomplete LU

DILU (Pollard & Siu 1981; widely known as the "diagonal-based ILU" used as the default asymmetric-matrix preconditioner in OpenFOAM's `DILU` and the symmetric variant `DIC`) is a cheaper cousin. It **only modifies the diagonal** — the off-diagonal factors are taken directly from $A$.

Formally, DILU seeks a factorization of the form

$$
M_{\text{DILU}} = (D_* + L)\, D_*^{-1}\, (D_* + U),
$$

where $L, U$ are the **strict** lower/upper triangular parts of $A$ (reused verbatim), and $D_* = \mathrm{diag}(d_1, \ldots, d_n)$ is the modified diagonal. $D_*$ is determined by the requirement that $M_{\text{DILU}}$ has the same diagonal as $A$:

$$
d_i = a_{ii} - \sum_{k < i,\; (i,k) \in \mathcal{S}(A)} \frac{a_{ik}\, a_{ki}}{d_k}.
$$

Equivalently, one can derive the recurrence by demanding that $M_{\text{DILU}}$ matches $A$ on the diagonal entries only (rather than on the whole sparsity pattern as ILU(0) does). For SPD $A$ (symmetric, $a_{ik} = a_{ki}$), this reduces to

$$
d_i = a_{ii} - \sum_{k < i,\; (i,k) \in \mathcal{S}(A)} \frac{a_{ik}^2}{d_k},
$$

which is exactly the Stone / Jacobs "modified incomplete Cholesky" diagonal, and OpenFOAM calls this the **DIC** (Diagonal Incomplete Cholesky) variant.

**Key structural properties**:

- DILU stores only $D_*$ in addition to $A$ (one vector of length $n$). ILU(0) must store $\tilde L$ and $\tilde U$ separately because their off-diagonal entries differ from $A$'s.
- Setup cost: $O(\mathrm{nnz}(A))$ but with a much smaller constant than ILU(0) (one multiply-divide per off-diagonal, no update of off-diagonals).
- DILU is **strictly weaker** than ILU(0) as a preconditioner in the sense of capturing fill: it leaves the off-diagonals of $L_{\text{DILU}}, U_{\text{DILU}}$ exactly as in $A$, so the "implicit fill-in" error is larger. In exchange, it is cheaper to form and cheaper to apply (the solve steps use $A$'s off-diagonals, which are already in memory).

**Confidence**: 确定 for the recurrence form. 较确定 for the historical attribution to Pollard & Siu; some sources credit Jacobs / Stone independently.

### 1.3 The forward/backward substitution and its dependency graph

Applying the preconditioner to a vector $r$ (i.e. computing $z = M^{-1} r$) requires two triangular solves. For DILU, with $M = (D_* + L) D_*^{-1} (D_* + U)$:

**Forward solve** $(D_* + L)\, y = r$, in scalar form:

$$
y_i = \frac{1}{d_i}\!\left( r_i - \sum_{k < i,\; (i,k) \in \mathcal{S}(A)} a_{ik}\, y_k \right), \quad i = 1, 2, \ldots, n.
$$

**Backward solve** $(D_* + U)\, z = D_*\, y$, equivalently $z = y - D_*^{-1} U z$ marched from $i = n$ down:

$$
z_i = y_i - \frac{1}{d_i}\sum_{k > i,\; (i,k) \in \mathcal{S}(A)} a_{ik}\, z_k, \quad i = n, n-1, \ldots, 1.
$$

**Dependency graph**. Define a directed acyclic graph (DAG) $G_L = (V, E_L)$ with $V = \{1, \ldots, n\}$ and $E_L = \{k \to i : k < i,\; (i,k) \in \mathcal{S}(A)\}$. Then:

- Row $i$ of the forward solve can fire **only after** all $y_k$ with $k \in \mathrm{pred}_{G_L}(i)$ are known.
- The backward solve has the mirror DAG $G_U$ on the strict upper pattern.

For the 7-point 3D stencil on an $N_x \times N_y \times N_z$ grid with natural (lexicographic) ordering, row $i \equiv (i_x, i_y, i_z)$ depends in the forward sweep on the $-x$, $-y$, and $-z$ neighbors:

$$
\mathrm{pred}(i_x, i_y, i_z) \subseteq \{(i_x{-}1, i_y, i_z),\; (i_x, i_y{-}1, i_z),\; (i_x, i_y, i_z{-}1)\}.
$$

This is the prototypical **wavefront** dependency: the cell $(i_x, i_y, i_z)$ becomes computable on the diagonal plane $\ell = i_x + i_y + i_z$ once all cells on plane $\ell - 1$ are done. The number of levels in the DAG is

$$
L_{\max} = N_x + N_y + N_z - 2 \quad (\text{natural ordering, 3D 7-point}).
$$

**Confidence**: 确定.

---

## 2. Why Naive JAX (`scan`, `fori_loop`, `while_loop`) Cannot Parallelize This

### 2.1 What `lax.scan` / `lax.fori_loop` actually compile to

`jax.lax.scan` and `jax.lax.fori_loop` lower to an XLA `While` op with a sequential carry. Semantically, iteration $i+1$ reads the output of iteration $i$. On GPU, XLA emits this as a sequence of kernel launches (or, with kernel fusion, a single kernel that loops serially on a single CUDA block or even a single thread of a block). There is **no automatic dependency analysis** that discovers "iteration $i+1$ depends only on iteration $i-2$, so we can run them 3-wide".

Concretely, if an engineer writes
```
# pseudo-code, illustrative only
y = lax.fori_loop(0, n, body_fn, y0)
```
where `body_fn(i, y)` performs row $i$'s forward substitution, the compiler sees a scalar carry and produces:

- **$n$ sequential steps**, each loading $A$'s row $i$, performing a tiny sparse dot product ($\leq 6$ multiply-adds), and writing $y_i$.
- A single active thread (or at best one warp of 32 doing trivial work) per iteration.
- The whole SM (Streaming Multiprocessor) array sits idle. With $n \sim 10^6$ and per-iteration latency $\sim 1\,\mu s$ (kernel-launch plus memory-round-trip), wall time is $\sim 1\,\mathrm{s}$ per preconditioner application. A PCG solve takes dozens of these. **This is 3-4 orders of magnitude worse than a CPU.**

### 2.2 Quantifying the GPU occupancy disaster

Take a modest AM case: $N_x = N_y = 256$, $N_z = 128$, so $n \approx 8.4 \times 10^6$ and $\mathrm{nnz}(A) \approx 5.9 \times 10^7$ (7-point, 3D).

- **Arithmetic intensity per row**: $\leq 12$ flops (6 multiplies + 6 adds + 1 divide).
- **Memory traffic per row**: 7 reads from $A$, up to 6 reads from $y$, 1 write to $y$ $\;\Rightarrow\; \sim 14$ doubles $= 112$ bytes.
- **Peak flop rate** if fully serial: $12 \;\mathrm{flops} / 1\,\mu s = 12\,\mathrm{MFLOP/s}$. A modern H100 does $\sim 60\,\mathrm{TFLOP/s}$ double-precision. **Utilization: $\approx 2 \times 10^{-7}$.**

Even under the most optimistic assumptions about kernel-launch fusion, the sequential carry forces a critical path of length $n$ with essentially zero arithmetic per step. This is why the naive `lax.fori_loop` path is not merely "slow" — it is structurally incompatible with SIMT.

### 2.3 Why `scan` doesn't help with sparsity

`lax.scan` is an associative-scan-friendly primitive *only* when the body is an associative operator (cf. Blelloch 1990). Triangular solve is expressible as a matrix product scan

$$
\begin{pmatrix} y_i \\ 1 \end{pmatrix} = \begin{pmatrix} 1/d_i & b_i/d_i \\ 0 & 1 \end{pmatrix}\! \cdots\! \begin{pmatrix} 1/d_1 & b_1/d_1 \\ 0 & 1 \end{pmatrix}\! \begin{pmatrix} 0 \\ 1 \end{pmatrix},
$$

but only for **dense bidiagonal** systems; for a sparse multi-diagonal system the "partial product" matrix densifies along the scan — you lose sparsity and pay $O(n \log n \cdot k^2)$ with $k$ the bandwidth. For a 3D 7-point stencil $k \sim N_x N_y$, which makes this cure worse than the disease.

XLA/JAX has **no built-in primitive for sparse-pattern-aware DAG scheduling**. That is precisely why we must go to FFI — to call cuSPARSE's `csrsv2` (which does level scheduling under the hood) or our own hand-rolled CUDA kernel.

**Confidence**: 确定 on the fundamental argument. 较确定 on the exact H100 utilization number (depends on driver/compiler behavior; the point is "orders of magnitude off", not the exact exponent).

---

## 3. Three Mathematical Escape Routes — Preview and Trade-offs

### 3.1 Route A — Level Scheduling (cuSPARSE `csrsv2` internally)

**Idea**. Partition the DAG $G_L$ into levels $\mathcal{L}_0, \mathcal{L}_1, \ldots, \mathcal{L}_{L_{\max}}$ where

$$
\mathcal{L}_0 = \{i : \mathrm{pred}(i) = \emptyset\}, \qquad \mathcal{L}_\ell = \{i : \mathrm{pred}(i) \subseteq \textstyle\bigcup_{m < \ell} \mathcal{L}_m,\; i \notin \bigcup_{m<\ell}\mathcal{L}_m\}.
$$

Within each level, rows are independent and can fire in parallel. Across levels, we have a barrier.

**Properties**:

- **Mathematically exact DILU**: the preconditioner is bit-identical to the serial version (up to FP associativity). PCG iteration count does not change.
- **Parallelism bounded by $L_{\max}$**. For natural lexicographic ordering on a 3D 7-point stencil, $L_{\max} = N_x + N_y + N_z - 2$. For $256^3$, $L_{\max} = 766$, and average level width $\bar{w} = n / L_{\max} \approx 2.2 \times 10^4$ — comfortable occupancy per level.
- **Setup cost**: BFS on $G_L$, $O(\mathrm{nnz}(A))$. cuSPARSE caches this in an "analysis" phase.
- **Weak point**: level-to-level barrier latency and load imbalance (the first and last few levels are very narrow — $|\mathcal{L}_0| = 1$ for lexicographic order). Naumov 2011 ("Parallel solution of sparse triangular linear systems in the preconditioned iterative methods on the GPU") showed 3-7× speedup over serial CPU DILU on structured problems, but the speedup degrades for unstructured meshes or longer critical paths.

**Expected performance on our AM problem**: $O(L_{\max})$ kernel launches per solve $\approx 10^3$ launches at $\sim 5\,\mu s$ each $\Rightarrow \sim 5\,\mathrm{ms}$ per application, which is in the ballpark of cuSPARSE `csrsv2_solve` benchmarks. 较确定.

### 3.2 Route B — Multi-coloring / Red-Black Reordering

**Idea**. Permute $A$ by a coloring $\pi$ of the adjacency graph such that same-colored rows have no edges in $\mathcal{S}(A)$. Then within each color class, the triangular solve has no intra-class dependencies — it is fully parallel.

For the 7-point 3D stencil, a 2-coloring (red-black, $i_x + i_y + i_z$ parity) suffices: reds depend only on blacks and vice versa. More general stencils may need 4, 8, or more colors.

**The catch**: DILU of the permuted matrix $P A P^\top$ is **not** the permutation of DILU of $A$. Explicitly,

$$
\mathrm{DILU}(P A P^\top) \neq P \cdot \mathrm{DILU}(A) \cdot P^\top
$$

in general, because the recurrence for $d_i$ depends on the ordering of $k < i$, and permuting reorders which off-diagonals enter the sum. The coloring changes the *numerical* preconditioner.

**Quantitative penalty**. Empirically, for M-matrices from structured Poisson problems, red-black ILU(0) typically increases PCG iteration count by **1.5×–3×** compared to natural-ordering ILU(0) (Bruaset 1995, *A Survey of Preconditioned Iterative Methods*, ch. 4; Doi & Lichnewsky 1990). The penalty grows with condition number. For **stiff** problems — large density jumps in our AM setting — the literature is less consistent; some studies (e.g. Duff & Meurant 1989 on reorderings) report penalties of 5× or more, especially when the reordering destroys the "Stieltjes" structure.

**我不确定** about the exact penalty for AM density-ratio-1000 problems — this is in the regime where published benchmarks thin out. Part of Phase 2-3 of this project will be to measure it directly.

**Upside**: within each color, trivial parallelism — no level barriers, no wavefronts. On GPU this can out-run level-scheduled exact DILU *per iteration*, and the question is whether the iteration-count inflation eats the gain. Classic trade.

### 3.3 Route C — Algebraic Multigrid (AMGx)

**Idea**. Not an ILU at all. AMG builds a hierarchy of coarse operators $A_0 = A, A_1, \ldots, A_\ell$ via algebraic coarsening (classical Ruge-Stüben or smoothed aggregation) and applies a V-/W-/F-cycle with a cheap smoother (Jacobi, Gauss-Seidel, or polynomial) at each level. The coarse-grid correction attacks the low-frequency modes that make the problem stiff.

**Why it's Plan B for our worst cases**:

- AMG's convergence rate is (under ideal assumptions) **$h$-independent** — condition number growth with mesh refinement is absorbed by the hierarchy. For the AM Poisson system, this is the *right* tool when $\kappa(A)$ is dominated by mesh-scale effects near the interface.
- AMGx (NVIDIA) gives a production GPU implementation with tuned smoothers and a classical coarsening path.
- But: AMG setup is expensive (often $\sim 5-10\times$ one PCG iteration). If the matrix changes every time step (it does, because $\rho$ changes with interface motion), you pay this repeatedly. Re-use strategies (freeze coarsening for $k$ steps) are possible but introduce approximation error.

**When to prefer AMG over DILU variants**: when the density ratio is so extreme ($\geq 10^3$, our AM regime) that DILU+PCG stagnates regardless of parallelization strategy. Evidence from the multiphase-flow literature (e.g. Sussman et al. on VOF-Poisson preconditioning) suggests the crossover is exactly in our regime. 较确定.

### 3.4 Summary table

| Route | Preconditioner fidelity | Parallelism | PCG iter count | GPU kernel complexity | Best for |
|---|---|---|---|---|---|
| A. Level scheduling | exact | $O(n / L_{\max})$ per level | unchanged | moderate (BFS + per-level kernel) | structured grids, moderate stiffness |
| B. Multi-coloring | modified | $O(n / n_{\text{colors}})$ | 1.5× – 5× worse | simple (one kernel per color) | when raw per-iter throughput dominates |
| C. AMG (AMGx) | different method | $h$-independent convergence | often 2-5× fewer PCG iters | complex (setup + V-cycle) | stiff / large density ratio (our AM worst case) |

---

## 4. Why the AM Melt-Pool Poisson System Is So Stiff

The stiffness of $A$ has three compounding sources:

### 4.1 Density ratio

With variable-density Poisson $-\nabla \cdot (\rho^{-1} \nabla p) = f$, the face coefficient between a liquid cell and a gas cell scales like the harmonic mean:

$$
\left( \frac{1}{\rho} \right)_{\text{face}} = \frac{2}{\rho_{\text{liq}} + \rho_{\text{gas}}} \approx \frac{2}{\rho_{\text{liq}}}
$$

while liquid-liquid faces see $1/\rho_{\text{liq}}$ and gas-gas faces see $1/\rho_{\text{gas}}$. The ratio of gas-gas to liquid-liquid coefficients is $\rho_{\text{liq}}/\rho_{\text{gas}} \sim 10^3$ for steel/argon. The matrix $A$ therefore has diagonal entries spanning **three orders of magnitude** across the interface.

The 2-norm condition number scales, for anisotropic Poisson with coefficient ratio $\gamma$, as (Axelsson 1994, *Iterative Solution Methods*, §7.2)

$$
\kappa_2(A) \sim \gamma \cdot \frac{1}{h^2}.
$$

For $\gamma = 10^3$ and $h = 1/256$, $\kappa_2(A) \sim 10^3 \cdot 6.5 \times 10^4 \approx 6.5 \times 10^7$. Unpreconditioned CG needs $O(\sqrt{\kappa}) \sim 8 \times 10^3$ iterations — unusable.

### 4.2 Viscosity ratio (indirect)

Viscosity does not enter the pressure Poisson equation directly in the projection method. But it enters the momentum-predictor step that produces $\mathbf{u}^*$, and it sets the time-scale: for resolving Marangoni convection and keyhole vapor dynamics, $\Delta t$ is bounded by the capillary CFL

$$
\Delta t \leq \sqrt{\frac{\rho_{\text{avg}} h^3}{2\pi \sigma}},
$$

which for molten-steel scales at $\mu$m mesh is $\sim 10^{-9}$ s. Small $\Delta t$ means $A$ is dominated by its diagonal (since the mass-matrix term $\rho / \Delta t^2$ goes large relative to the stiffness $1/h^2$), but this **alleviates** stiffness rather than worsening it — so viscosity ratio is not a direct stiffness driver for Poisson. (It is for the velocity update; different story.)

**较确定**: I'm confident viscosity doesn't enter Poisson stiffness, less sure of the exact capillary-CFL coefficient.

### 4.3 Near-interface cell size and aspect ratio

If the VOF/PLIC reconstruction produces cells with $\alpha \in (0, 1)$ and the projection operates on "apparent" cell volumes $\alpha h^3$, the effective face coefficient near the interface scales like $\alpha / \rho_{\text{liq}}$. Thin films ($\alpha \ll 1$) therefore introduce a **second** source of coefficient spread independent of $\rho_{\text{liq}}/\rho_{\text{gas}}$. This is the mechanism that makes ghost-fluid / cut-cell Poisson systems notoriously ill-conditioned (Gibou et al. 2002, 2005 on sharp-interface Poisson).

For our AM problem where we track a thin liquid film over a powder bed, $\alpha$-thin cells are the norm, not the exception.

### 4.4 Consequences for DILU

DILU's spectral effect can be characterized (for M-matrices) by the eigenvalue clustering of $M^{-1}A$: rather than the $\kappa(A)^{-1/2}$ of Jacobi, DILU delivers $\kappa(M^{-1}A) \sim \sqrt{\kappa(A)}$ for 2D Poisson on nice grids (Gustafsson 1978, "A class of first-order factorization methods"). For variable-coefficient, discontinuous-coefficient Poisson with $\gamma \gg 1$, this favorable scaling **breaks down**; DILU's eigenvalue clustering degrades to essentially Jacobi for the subset of modes localized at the interface. This is why, for AM-like problems, we anticipate the answer to be **DILU helps on the bulk, AMG helps on the interface-coupled modes**, motivating the hybrid solver strategy already implicit in the project roadmap.

"Jacobi is not enough" = we have empirical evidence from OpenFOAM practitioners (see the `lasermeltfoam-audit` skill's reference material on Newcastle's LaserbeamFoam) that on AM-scale Poisson systems, GaussSeidel/Jacobi preconditioning doubles or triples PCG iterations relative to DILU/DIC; AMG wins further.

**Confidence**: 确定 on the qualitative picture. 需验证 on the exact iteration-count ratios for our specific geometry and density ratio — that is itself a Phase 2-3 measurement.

---

## 5. Mathematical Specification for the Phase 1 FFI Prototype Kernel

The roadmap's Part 2 calls for "a simple op on the diagonal of a sparse matrix" to validate the FFI toolchain. As a CFD-math expert, I recommend a **slightly richer** target that has the same minimal complexity but exercises the handshakes we will need for real work.

### 5.1 Recommended target: one step of Jacobi-preconditioned residual

Compute, on device, given CSR-format $A$ and device-resident vectors $b$, $x$:

$$
y = D^{-1} (b - A x), \qquad D = \mathrm{diag}(A).
$$

This is one sweep of weighted Jacobi iteration (weight $\omega = 1$, producing the preconditioned residual that would be the first Krylov vector in a PCG). It is numerically trivial but exercises every handshake we need:

### 5.2 Data handshakes the kernel must validate

1. **CSR pointer unpacking**: the kernel receives four device pointers — `row_ptr` ($n{+}1$ int32), `col_idx` ($\mathrm{nnz}$ int32), `values` ($\mathrm{nnz}$ float32 or float64), `diag_idx` or computed on the fly — plus input vectors $b$ and $x$ and an output buffer $y$.
2. **Dtype plumbing**: the FFI descriptor must propagate dtype (float32 vs float64) from the JAX side; a stiff pressure Poisson will need float64 at the reduction step even if we compute in float32, so design the descriptor to carry both.
3. **Shape & layout**: the kernel must accept arbitrary $n$ (not hard-coded), and enforce C-contiguous input (no stride handling in v1).
4. **Error signaling**: validate that `row_ptr[n] == nnz`, that `col_idx[k] < n`, and that diagonal entries exist. Return an error code via the FFI `ffi::Error` channel (not a CUDA abort).
5. **Stream awareness**: the kernel must run on the XLA-provided CUDA stream, not the default stream, or we break JAX's asynchronous-execution invariants.

### 5.3 Numerical verification recipe

The kernel's output $y$ must satisfy (bitwise-comparable to a reference NumPy/JAX computation up to FP-associativity differences in the SpMV reduction):

$$
\| y_{\text{kernel}} - y_{\text{reference}} \|_\infty \leq C \cdot \epsilon_{\text{mach}} \cdot \max_i |y_{\text{ref},i}|
$$

with $C = O(\mathrm{nnz\_per\_row})$ — i.e. a few tens for our 7-point stencils. Any discrepancy larger than this points to a bug (dtype mixup, stride bug, atomic race in the SpMV accumulator).

**Suggested test matrices**:
- **T1**: 1D Laplacian tridiagonal, $n = 10^3$. Hand-verifiable.
- **T2**: 2D 5-point Poisson on $100 \times 100$. Dense enough to catch stride/indexing bugs, small enough to reference-check.
- **T3**: Synthetic "AM-like" matrix with 3-order-of-magnitude diagonal spread (random $D \in [1, 10^3]$, M-matrix off-diagonals). Catches dtype-precision issues before real geometry.

### 5.4 Why not just "read the diagonal"

A pure "copy the diagonal to a buffer" kernel would validate **FFI invocation but nothing about data layout**. The Jacobi residual kernel above:
- Requires reading $A$'s row structure (exercises CSR),
- Requires a reduction over nonzeros (exercises the SpMV memory-access pattern — which is the kernel we'll write *in anger* in Phase 2),
- Requires a small elementwise postprocess (exercises combining SpMV output with another device vector),
- Yet is small enough that the whole thing fits in ~40 lines of CUDA.

This is the minimal kernel that is *not wasted code* when we move to Phase 2's real SpMV-in-PCG.

### 5.5 What the engineering team does NOT need from me for this step

- Choice of threading strategy (warp-per-row vs block-per-row SpMV) — that is a tuning question for Phase 2. V1 can be one-thread-per-row and still correct.
- Choice of float32 vs float64 default — v1 should plumb both paths; the numerics-relevant decision is Phase 3.
- CUDA graph / stream capture — overkill for v1.

---

## 6. Phase 1 Mathematical Acceptance Checklist

Verify the following before approving transition to Phase 2. Each item is the user's responsibility (or the user + engineering team jointly).

1. **DILU recurrence is written down explicitly** (§1.2) and it is clear that DILU $\neq$ ILU(0) — DILU only modifies the diagonal; ILU(0) modifies off-diagonals as well. The user should be able to state which one OpenFOAM's `DILU` preconditioner is (answer: DILU, as named). **Confidence: 确定**.

2. **The DAG-dependency argument for why naive `lax.fori_loop` is fatal is understood** (§2): it's not just "slow", it's $\sim 10^{-7}$ utilization of a GPU because of serial carry + low arithmetic intensity. The user should be able to explain this to a skeptical GPU engineer without hand-waving.

3. **Three escape routes are understood as distinct mathematical objects**, not three implementations of the same thing (§3):
   - Route A preserves the preconditioner, parallelizes the application.
   - Route B changes the preconditioner to buy parallelism.
   - Route C discards DILU entirely for a hierarchical method.

4. **The AM-stiffness argument is quantified** (§4): the user should be able to point at a back-of-envelope estimate of $\kappa_2(A)$ for the target geometry (e.g., $\sim 10^7$–$10^8$ for $256^3$ grid with $\gamma = 10^3$) and explain why Jacobi alone cannot rescue it.

5. **FFI prototype target is agreed upon with the engineering team**: one-step Jacobi residual on CSR (§5.1), not "just copy a diagonal". Both sides sign off that this exercises all the handshakes we need later.

6. **Verification recipe for the FFI prototype is pre-agreed** (§5.3): three test matrices T1/T2/T3 with explicit error tolerance $\leq O(\mathrm{nnz\_per\_row}) \cdot \epsilon_{\text{mach}} \cdot \|y\|_\infty$. No prototype will be declared "working" unless it passes all three.

7. **Known unknowns are logged**, not forgotten:
   - What is the multi-coloring iteration-count penalty for our specific $\gamma = 10^3$ AM geometry? (Literature is thin.)
   - Where exactly is the DILU→AMG crossover in density ratio? (We will measure in Phase 3.)
   - What is the $\alpha$-thin-cell condition-number blowup for our PLIC reconstructions? (Couples to the upstream VOF work; must be measured on real interfaces.)

8. **Scope boundary is explicit**: Phase 1 delivers *understanding and a prototype*, not a working DILU solver. Nobody should approve Phase 2 transition on the expectation that "we know DILU works on GPU now" — we only know that the FFI toolchain works and we have a plan for the three routes.

---

## References (author + title; no fabricated DOIs)

- Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003 — §10.3 on ILU(0), §10.4 on modified ILU variants, §12 on parallel triangular solves.
- Meijerink, J. A. & van der Vorst, H. A. "An iterative solution method for linear systems of which the coefficient matrix is a symmetric M-matrix", *Math. Comp.*, 1977.
- Pollard, A. & Siu, A. L.-W. "The calculation of some laminar flows using various discretisation schemes", *Comp. Meth. Appl. Mech. Eng.*, 1982 (DILU origin).
- Gustafsson, I. "A class of first order factorization methods", *BIT*, 1978 — $\kappa(M^{-1}A) \sim \sqrt{\kappa(A)}$ for modified ILU.
- Naumov, M. "Parallel solution of sparse triangular linear systems in the preconditioned iterative methods on the GPU", NVIDIA Technical Report NVR-2011-001 — level scheduling for cuSPARSE.
- Bruaset, A. M. *A Survey of Preconditioned Iterative Methods*, Pitman Research Notes, 1995 — reordering penalties.
- Doi, S. & Lichnewsky, A. "A graph-theory approach for analyzing the effects of ordering on ILU preconditioning", INRIA RR-1452, 1991.
- Duff, I. S. & Meurant, G. A. "The effect of ordering on preconditioned conjugate gradients", *BIT*, 1989.
- Axelsson, O. *Iterative Solution Methods*, Cambridge, 1994 — condition-number scaling for anisotropic/discontinuous-coefficient Poisson.
- Gibou, F., Fedkiw, R., Cheng, L.-T., Kang, M. "A second-order-accurate symmetric discretization of the Poisson equation on irregular domains", *J. Comput. Phys.*, 2002.
- OpenFOAM source: `src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner` and `DICPreconditioner` — canonical reference for "what DILU means in practice".
- NVIDIA AMGx documentation & Naumov et al., "AmgX: A library for GPU accelerated algebraic multigrid and preconditioned iterative methods", 2015.

---

*End of Phase 1 report. Engineering team (jax-cfd-am-expert) takes over for Part 2 implementation against §5's specification.*
