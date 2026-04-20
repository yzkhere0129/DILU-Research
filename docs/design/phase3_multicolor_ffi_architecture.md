# Phase 3 — Multi-Color DILU / XLA FFI Integration Architecture

**Project**: DILU Solver Low-Level Optimization Research (JAX-GPU)
**Scope**: Phase 3 Part 2 — engineering architecture for a multi-color-reordered DILU preconditioner that bypasses cuSPARSE SpSV level-scheduling in favor of color-stride custom kernels.
**Companion doc**: `phase3_multicolor_math_foundation.md` (math expert, parallel track — not yet landed at time of writing)
**Predecessor docs**: `phase1_ffi_prototype_architecture.md`, `phase2_cusparse_ffi_architecture.md`, `phase2_cusparse_level_scheduling_math.md`
**Predecessor reports**: `phase2_cusparse_report.md`, `phase2.5_physical_report.md`
**Date**: 2026-04-20
**Author**: `jax-cfd-expert`
**Hardware target**: RTX 3050 Laptop (4 GB, compute 8.6, CUDA 12.4 toolkit) — unchanged from Phase 2
**Status**: DESIGN — awaiting user sign-off before any Phase 3 code is written

---

## 0. TL;DR

Phase 2 closed a numerically healthy cuSPARSE-based DILU-PCG stack: 24 iterations vs Jacobi's 71 on the 16³ stiff test, ULP-floor T4/T5/T6 correctness, and clean physical benchmarks (zero interface halo in Tests A/B/C). The PCG-apply hot path is measured at ~1.2 ms on a 10³ Laplacian and ~1.8 ms on 16³ — all time is inside two `cusparseSpSV_solve` calls whose internal level-scheduling the user does not own.

**Phase 3's question** (restated from the user's directive):

> "在通过这三套物理验证（绝不能出现几何误差花纹）的前提下, 看看为了极高的 GPU 并行度, 我们需要付出怎样的迭代数 Trade-off."

The engineering answer this doc specifies:

1. **Write our own color-stride triangular-solve kernels.** Path B beats Path A on first principles (§1). Permuting the matrix and still handing it to `cusparseSpSV` pays for cuSPARSE's internal level-scheduling machinery when the entire point of multi-coloring is to *replace* level scheduling with two dense-stride kernels per sweep. We keep Path A alive only as a measurement baseline for §7's Trade-off table.

2. **Coloring runs on host at `multicolor_analyze` time.** For structured 7-point Laplacians we hard-code red-black (parity of `i+j+k`). For general CSR we run a greedy first-fit coloring in a ~30-LoC Python routine, results shipped to device as four integer arrays: `perm[N]`, `iperm[N]`, `color[N]`, `color_offsets[n_colors+1]`. Parallel on-device coloring (Jones-Plassmann etc.) is Phase 4+ material — overkill for a single-device PCG on a static grid.

3. **Phase 2 and Phase 3 coexist as sibling directories.** `dilu/multicolor/` is the new home. Zero modifications to `dilu/cusparse/`. The PCG driver and the Phase 2.5 physical-benchmark plot pipeline are reused via a tiny shim that abstracts the preconditioner as a `apply(values, d_star, r) -> z` callable. Both stacks publish plots to their own `bench/plots/` subdirectories for side-by-side comparison.

4. **Four FFI primitives mirror Phase 2 exactly**: `multicolor_analyze`, `multicolor_apply`, `multicolor_refactor`, `multicolor_release`. Opaque `uint64` token + process-global plan registry. Same 8-byte D→H per apply tolerated exception; zero host-device copies otherwise. The `_refactor` primitive (new in Phase 3) is called when A's values change but the pattern doesn't — it recomputes $\tilde{D}_*$ on the already-permuted matrix without re-running coloring.

5. **Physical tests A/B/C are the acid test.** The user's directive is explicit: 绝不能出现几何误差花纹. Phase 3 must emit its own `divergence_map.png`, `spurious_currents.png`, and `residual_halo.png` (Phase 2's three plots, regenerated through the multi-color stack). A new interface-localized error pattern in any of those plots is a non-negotiable STOP signal (§9).

**What this doc does NOT commit to**: the exact number of colors for the reference Laplacian (math doc territory — 2 for red-black 7-point, 8 for red-black with corner neighbors, general CSR is greedy-determined at runtime), nor a claim that Phase 3 will be *faster* than Phase 2 on the 3050. The per-iteration wall-time win is plausible (the 8-byte D→H sync that dominates at small n would go away with a more direct token path; the color-stride kernel is a straight SpMV-shaped operation rather than cuSPARSE's multi-kernel level-scheduled DAG), but the convergence penalty from the reordering can swing iteration count up by 10%–40% in published studies. **The honest outcome Phase 3 must measure is the product `T_per_iter × iter_count`** — a per-iter speedup that loses to iteration-count growth is a loss the user explicitly wants to quantify.

---

## 1. Architectural Decision: Reuse cuSPARSE or Write Our Own Kernels?

### 1.1 The three candidate paths

Restated from the Phase 3 brief:

| Path | Description | Who does level scheduling? |
|---|---|---|
| A | Permute A by `perm` → feed $\tilde{A} = PAP^T$ to `cusparseSpSV_solve`. Reuse the entire Phase 2 pipeline with a permuted CSR. | cuSPARSE internally |
| B | Own kernel per color. `n_colors` kernel launches for forward sweep, `n_colors` for backward. Within a color all rows are independent — trivially parallel. | We do, externally, via kernel-launch ordering |
| C | Hybrid: Path B for ≤4-color structured cases (red-black 7-pt, checkerboard 2D), Path A fallback for general unstructured CSR. | Mix |

### 1.2 Why Path B wins on first principles

**Multi-coloring's *purpose* is to eliminate level scheduling.** A $k$-color ordering is valid precisely when each color class is an independent set in the matrix's adjacency graph — i.e., no two rows in the same color share an off-diagonal nonzero. Given that, the forward sweep `(D* + L̃) ỹ = b̃` decomposes into *exactly $k$ independent stages*:

```
for c = 0 .. k-1:
    for i in color_rows[c]:   # all INDEPENDENT in this stage
        ỹ[i] = (b̃[i] - sum_{j in color < c} L̃[i,j] * ỹ[j]) / D*[i]
```

Stage $c$ depends only on stages $0 \ldots c-1$, and within stage $c$ every row's arithmetic is independent of every other row's. This is exactly a masked SpMV followed by a diagonal scale — a pattern a custom CUDA kernel handles in ~20 lines with no shared memory, no atomics, no warp primitives. It is embarrassingly memory-bound in the same way SpMV is.

**Handing that same structure to `cusparseSpSV` is doubly wasteful**:

1. **Analysis cost is paid for structure cuSPARSE already has**. `cusparseSpSV_analysis` traverses the matrix, builds a DAG of row dependencies, and produces a level-schedule (or whatever CUSPARSE_SPSV_ALG_DEFAULT's opaque equivalent is). For a red-black-reordered matrix the DAG has exactly $k=2$ levels — but cuSPARSE has no API to tell it this. It re-derives the structure at a cost we measured in Phase 2 at 17 ms for n=1000 and 125 ms for n=10 000. This cost does amortize over ~3 apply calls (Phase 2 §4), but the amortization is flat overhead we're eating for nothing.
2. **Apply-time kernel launches are still opaque**. Whatever `CUSPARSE_SPSV_ALG_DEFAULT` does internally on the permuted matrix — level-scheduled multi-kernel dispatch, a cooperative groups super-kernel, a stream of tiny kernels — we don't get to control or measure it. On structured 3-D Laplacian at n=4k we measured 1.8 ms per apply (Phase 2 report §5), but we don't know how much of that is level-synchronization overhead cuSPARSE is forced to do because it doesn't know the matrix is already coloring-permuted.

**Path A's *only* win is code reuse** — we already have `cusparse_dilu_apply`, we'd just change how we build the CSR. That is a real engineering savings. But the *point* of Phase 3, per the master plan:

> 如果官方库的 Level Scheduling 并行度在我们的特定网格上表现不佳, 我们需要从数学层面进行重构.

We are not trying to outcompete cuSPARSE on *its own algorithm over our reordered matrix*. We are trying to outcompete it on **a different algorithm** that exploits the reordering in ways cuSPARSE's general-purpose SpSV cannot. Path A doesn't let us do that.

### 1.3 Why Path C (hybrid) is not worth the complexity

A hybrid dispatcher — detect red-black, use Path B; otherwise fall back to Path A — adds two things we don't want: (a) a detector (when is a matrix "structured enough" for Path B to win? not a first-principles question, needs benchmarking), and (b) two code paths to maintain. Our target workload (7-pt Laplacian on cartesian grids, with density jumps but fixed connectivity) is *always* red-black colorable. General CSR is Phase 4+ territory. YAGNI.

**We do, however, keep Path A as a measurement baseline** (§7, `bench_multicolor_vs_cusparse.py`): build a red-black-permuted CSR, feed it through Phase 2's unchanged cuSPARSE stack, measure. If Path A miraculously beats Path B on the 3050, that's important information the user should see before committing to Phase 4. But it does not change the architecture — Path B is what we build.

### 1.4 Recommendation

**Path B: write our own color-stride forward/backward solve kernels.** Phase 2's cuSPARSE plumbing (`cusparse_dilu_apply.cc`, `libdilu_cusparse.so`) is untouched. Phase 3's new library `libdilu_multicolor.so` builds with **no** cuSPARSE dependency (§6) — just `CUDA::cudart`. This is both a cleaner architecture and a useful independence proof: we can demonstrate a DILU preconditioner that works without leaning on any NVIDIA library beyond the CUDA runtime itself.

### 1.5 What Path B is NOT

- **Not an SpMV.** The loop is masked: only `j < i` entries in CSR (for forward) or `j > i` (for backward) contribute. Generic CSR doesn't give us "strict-lower triangle" cheaply without either (a) precomputing and storing the triangle offsets, or (b) per-element branching `if (j < row)`. For the 7-point stencil, precomputing triangle offsets is trivial (§4.4). For general CSR we pay the branch; memory-bound anyway.
- **Not a full level-scheduled SpSV.** We have exactly $k$ stages, not $L_\max$ levels. For red-black $k=2$; for a worst-case ordering it could be higher, but we control it.
- **Not a claim of iteration-count parity with Phase 2.** The multi-color ordering changes $D_*$ and therefore changes the PCG preconditioner's spectrum. The math expert's parallel track will quantify this. Engineering-side we commit to *measuring* iteration count and *reporting* it, not to a specific number.

---

## 2. Coloring Algorithm Placement

### 2.1 Where does graph coloring run?

Two choices, both known-viable:

| Location | Pros | Cons | Phase |
|---|---|---|---|
| Host Python at `multicolor_analyze` time | Simple (NetworkX or ~30-LoC greedy). Deterministic given a seed. No CUDA kernel needed. Runs once per matrix, amortized same as Phase 2's analysis. | Host↔device copy of 4 small integer arrays (negligible at analyze time). Doesn't work for truly dynamic grids — but Phase 3 assumes static grid per brief. | **Phase 3 (this doc)** |
| On-device parallel (Jones-Plassmann / Iterated Preprocessing / Luby) | No host↔device copy. Scales to adaptive/dynamic grids. | Implementation cost: another CUDA kernel + its correctness validation. Color count is usually 1–2 higher than greedy (more colors = more kernel launches = slower apply). Only makes sense when coloring is itself in a hot loop. | Phase 4+ |

**Recommendation: host-side.** Justifications:

- **Static grid assumption holds for Phase 3.** AM pressure solvers typically re-mesh at part-scale (macro time scales), not per-PCG-iteration. The matrix pattern is fixed for the whole PCG loop. Coloring cost is O(analyze-time), amortized over 100–500 PCG apply calls — the same calculus as Phase 2's cuSPARSE analysis.
- **Color count matters more than analyze-time speed.** Greedy coloring on host produces near-optimal color counts (2 for 7-pt stencil, 4 for 9-pt, matching chromatic-number lower bounds). Jones-Plassmann typically uses 1–2 more colors than greedy because its speculative-conflict resolution is conservative. Each extra color = one more kernel launch per sweep = more launch overhead = slower apply. On our 3050 with ~5 µs per launch, paying +5 µs per apply call forever to save a one-time 50 ms host coloring is a bad trade.
- **We reuse an existing battle-tested tool.** NetworkX's `greedy_color` has been stable since 2017 (still published as NetworkX 3.6.x with the same API). Alternative: ~25-LoC hand-rolled first-fit greedy for CSR — no networkx dependency. We pick the hand-rolled version (§2.3) to keep the dependency graph flat.

### 2.2 Library survey (web-verified, 2026-04-20)

- **NetworkX 3.6.x**: stable `networkx.algorithms.coloring.greedy_color(G, strategy="largest_first")`. Pure-Python, quadratic in worst case but fast enough for n=10⁶ CSR matrices we care about. We do **not** take a NetworkX dependency — it pulls in scipy/matplotlib transitively, and our one-shot greedy routine is 25 LoC.
- **SciPy csgraph**: **no graph-coloring primitive.** csgraph exposes BFS, connected components, shortest paths, MST, max-flow — but not vertex coloring. Confirmed against SciPy's current documentation.
- **RAPIDS cuGraph**: no published graph-coloring algorithm. cuGraph focuses on centrality, community detection, core numbers, traversals. A historical `nvgraph` (discontinued) had coloring but was absorbed into cuGraph without re-exposing the coloring API. We do not take a RAPIDS dependency.
- **Prior literature**: Jones-Plassmann (1993) + parallel variants (Berkeley/Cornell 2012–2019) are the canonical GPU references. NVIDIA's blog post "Graph Coloring: More Parallelism for Incomplete-LU Factorization" (developer.nvidia.com) describes exactly our problem but is from AMGx's internals, not a public library.
- **US Patent 9798698 (NVIDIA)**: "System and method for multi-color DILU preconditioner" — NVIDIA does have internal IP here, but no public API for it outside AMGx. Phase 4 is where AMGx enters the picture; Phase 3 is explicitly not AMGx.

**Conclusion**: we write a ~30-LoC host-side greedy first-fit coloring in `dilu/multicolor/python/coloring.py` plus a ~10-LoC hard-coded red-black routine for the 7-point stencil fast path. Zero new external dependencies.

### 2.3 Coloring output shape — the contract the kernel sees

The `multicolor_analyze` handler receives from Python **four integer arrays** (int32, contiguous, device-resident):

- `perm[N]`: the permutation. `perm[new_idx] = old_idx`. I.e., the row that now lives at position `new_idx` in the reordered matrix was at position `old_idx` in the original.
- `iperm[N]`: the inverse permutation. `iperm[old_idx] = new_idx`.
- `color[N]`: `color[new_idx]` = the color (0-based) of the row now at position `new_idx`. **Rows are already grouped contiguously by color** after permutation, so equivalently `color[new_idx] = c` iff `color_offsets[c] ≤ new_idx < color_offsets[c+1]`. We keep the explicit array for sanity checks, not for indexing.
- `color_offsets[n_colors+1]`: `color_offsets[0] = 0`, `color_offsets[c]` = starting row index of color `c`, `color_offsets[n_colors] = N`. This is the array the kernel actually reads at launch time to compute `tid + color_offsets[c]`.

**Invariant the Python side guarantees** (verified before the analyze FFI call):

1. `perm` is a permutation of `[0, N)`.
2. `iperm[perm[i]] == i` for all `i`.
3. For each `c ∈ [0, n_colors)`, every pair of rows `(i, j)` in color `c` has *no* CSR off-diagonal between them. I.e., there is no edge `(i, j)` in the adjacency graph of $A$ with `color[iperm[i]] == color[iperm[j]]`.

Invariant (3) is the *coloring validity* property and is checked on the host at `analyze` time by a 5-LoC post-validation pass (§9 STOP #1). Any violation is a catastrophic bug (produces wrong results silently in the sweep kernel); the validator fails loudly before the plan is registered.

### 2.4 Red-black fast path

For a 7-point Laplacian on a structured $n_x \times n_y \times n_z$ grid, we do not run the greedy coloring. The red-black assignment is closed-form:

```python
def red_black_color(nx, ny, nz):
    N = nx * ny * nz
    ijk = np.arange(N)
    i = ijk % nx
    j = (ijk // nx) % ny
    k = ijk // (nx * ny)
    parity = (i + j + k) % 2
    # perm: reds first (perm[0 : n_red] = red_old_indices), then blacks
    red_old = np.where(parity == 0)[0]
    black_old = np.where(parity == 1)[0]
    perm = np.concatenate([red_old, black_old]).astype(np.int32)
    iperm = np.empty(N, dtype=np.int32)
    iperm[perm] = np.arange(N, dtype=np.int32)
    n_red = len(red_old)
    color_offsets = np.array([0, n_red, N], dtype=np.int32)
    color = np.concatenate([np.zeros(n_red, dtype=np.int32),
                            np.ones(N - n_red, dtype=np.int32)])
    return perm, iperm, color, color_offsets
```

The exact same machinery then applies `perm` to the CSR (`dilu/multicolor/python/permute.py`, ~20 LoC: `row_ptr_new[i+1] - row_ptr_new[i]` rebuilt from `perm`, `col_idx_new[k] = iperm[col_idx_old[...]]` after reordering rows, then **sort within each row** to keep cuSPARSE-compatible ascending column order — important for the kernel's triangle-offset assumptions).

### 2.5 Generic greedy coloring (fallback)

When the caller doesn't pass `(nx, ny, nz)` or passes a non-7-point matrix, we fall through to:

```python
def greedy_color_csr(row_ptr, col_idx):
    N = len(row_ptr) - 1
    color = -np.ones(N, dtype=np.int32)
    for v in range(N):
        neighbor_colors = set()
        for k in range(row_ptr[v], row_ptr[v+1]):
            u = col_idx[k]
            if u != v and color[u] >= 0:
                neighbor_colors.add(int(color[u]))
        c = 0
        while c in neighbor_colors:
            c += 1
        color[v] = c
    n_colors = int(color.max()) + 1
    # Stable sort rows by color to produce perm.
    perm = np.argsort(color, kind="stable").astype(np.int32)
    iperm = np.empty(N, dtype=np.int32)
    iperm[perm] = np.arange(N, dtype=np.int32)
    color_permuted = color[perm]
    color_offsets = np.concatenate([[0], np.searchsorted(color_permuted, np.arange(n_colors+1))]).astype(np.int32)
    return perm, iperm, color_permuted.astype(np.int32), color_offsets
```

This is pseudocode of intent, not final. The n=10⁶ worst-case runtime is ~10 seconds in pure Python; for Phase 3's 32³=32k test grid it's milliseconds. If we ever need n > 10⁵ unstructured, we vectorize with numpy + a numba inner loop. Not a Phase 3 hot path.

### 2.6 Why not on-device coloring now?

The brief's prompt mentions Jones-Plassmann / IPP as "Phase 4+". Honest reasons:

1. **Amortization math.** On-device coloring gains over host-side coloring only when coloring is itself in a hot loop. It is not, in Phase 3. Moving it to device saves ~50 ms one-time at the cost of writing and validating a new CUDA kernel that can produce silently-wrong colorings.
2. **Correctness validation is harder on device.** The host-side greedy always produces a valid coloring by construction (the `while c in neighbor_colors` loop is deterministic). The parallel speculative variants (Jones-Plassmann, Luby) need a conflict-resolution pass that sometimes leaves monochromatic adjacent pairs until the next iteration converges. A bug in conflict resolution produces exactly the failure mode our physical tests detect — "interface halo from wrong coloring" — but pins the bug at a deeper level than we can afford to debug in Phase 3.
3. **It's not in the master plan.** Quoted from the user's directive: "多色排序算法实现...在 JAX 中实现一个图染色算法". "In JAX" is loose — the user accepts host-side in Python. On-device is a strictly bigger commitment.

Phase 4 can revisit if/when we need to re-color inside a time-marching loop (e.g., for AMR).

---

## 3. FFI Primitive Design for Multi-Color

We mirror Phase 2's 4-primitive structure exactly — same lifecycle, same token discipline, same plan-registry pattern. Only the contents of the plan change.

### 3.1 `multicolor_analyze`

**Purpose**: given original CSR (pattern + values + `diag_offset`) and the host-computed coloring arrays, produce a fully reordered plan stored under an opaque token.

**Signature**:
```
Inputs (device, C-contiguous):
  row_ptr     : int32[N+1]    -- original CSR row offsets
  col_idx     : int32[nnz]    -- original CSR column indices
  values      : float64[nnz]  -- original CSR values (seed for the plan's Ã values copy)
  diag_offset : int32[N]      -- original CSR diagonal-entry offsets in col_idx
  perm        : int32[N]      -- permutation, perm[new] = old
  iperm       : int32[N]      -- inverse, iperm[old] = new
  color_offsets : int32[n_colors+1]  -- row offsets by color in the permuted matrix
Output:
  token       : uint64[1]     -- opaque plan handle
```

**Work** (all on the XLA stream):
1. `cudaMallocAsync` the plan-owned permuted CSR: `row_ptr_tilde[N+1]`, `col_idx_tilde[nnz]`, `values_tilde[nnz]`, `diag_offset_tilde[N]`.
2. Launch a `permute_csr_kernel`: compute `row_ptr_tilde` via a segmented scan of row lengths by `perm`, compute `col_idx_tilde` via `iperm[col_idx_old[...]]` with per-row sorting to restore ascending order. This is the one non-trivial device-side operation at analyze time.
   - **Note**: for Phase 3 we do this on device with a simple kernel (one block per permuted-row, each block sorts its row with a small bitonic sort since 7-pt has ≤7 entries). For general CSR with longer rows, a host-side permute + upload is acceptable fallback (analyze is not a hot path).
3. Copy the host-computed `perm`, `iperm`, `color_offsets` arrays into plan-owned device buffers.
4. Compute the DILU factorization $\tilde{D}_*$ on the permuted matrix $\tilde{A}$. **This is the Phase 2 `dilu_factor_kernel` applied to the permuted CSR.** We reuse it verbatim — copy the .cu file into `dilu/multicolor/cuda/` unchanged. Math doc confirms DILU's recurrence is invariant to ordering modulo the ordering itself; $\tilde{D}_* = \text{DILU}(\tilde{A})$, not $P D_*(A) P^T$.
5. Store in `PlanEntry` (struct in §3.5): all the above device arrays, `n_colors`, `(N, nnz)` fingerprint.
6. Insert into process-global plan map; return `uint64` token via `cudaMemcpyAsync(H2D)` into the output buffer.

**Errors**:
- `InvalidArgument` on shape mismatch, `perm`/`iperm` round-trip violation (checked host-side before the call), `color_offsets[n_colors] != N`.
- `Internal` on CUDA allocation / kernel-launch failure.

**Idempotence**: each call produces a new token. Caller must call `release` to free.

### 3.2 `multicolor_apply`

**Purpose**: hot path. Given a token + the *original*-ordering CSR values + $D_*$ + $r$, produce $z = M_{\text{mcDILU}}^{-1} r$ where $M_{\text{mcDILU}} = (D_* + \tilde{L}) D_*^{-1} (D_* + \tilde{U})$ and $\tilde{L}/\tilde{U}$ are the strict lower/upper parts of the permuted matrix $\tilde{A}$.

**Signature**:
```
Inputs:
  token   : uint64[1]    -- from multicolor_analyze
  values  : float64[nnz] -- ORIGINAL ordering (not permuted) — we permute inside
  d_star  : float64[N]   -- ORIGINAL ordering — we permute inside
  r       : float64[N]   -- ORIGINAL ordering
Output:
  z       : float64[N]   -- ORIGINAL ordering
```

**Why original ordering for I/O**: keeping the Python caller's vectors in original ordering is a non-negotiable usability requirement. The PCG driver, the SpMV kernel, and every user-facing array comes from the natural grid ordering. Exposing permuted vectors at the boundary would force every caller to thread `perm/iperm` through their code. Phase 3's preconditioner is a black box with original-ordering I/O.

**Cost of the internal permutation**: 3× `N`-element gather kernels per apply (permute `values`, `d_star`, `r` in; permute `z` out). Each is ~5 µs at N=10⁴. Total ~15 µs overhead per apply vs Phase 2's 0 (Phase 2 doesn't permute). Measured against Phase 2's current 1.2 ms hot-path, this is ~1% overhead — acceptable. We budget it explicitly.

Actually we only need to permute `r` in and inverse-permute `z` out — `values` and `d_star` we can either permute in or fold into the plan's persistent buffers. Optimization: let the plan's `values_tilde` buffer be updated per-apply via a gather from the caller's `values`, and similarly for `d_star_tilde`. No reorder of the plan's index arrays per call.

**Handler flow** (all on XLA stream; 8-byte D→H token copy at start is the tolerated exception per Phase 2):

1. Read token (`cudaMemcpyAsync(D2H, 8B)` + `cudaStreamSynchronize`).
2. Plan cache lookup. `InvalidArgument` if missing. Verify `(N, nnz)` fingerprint.
3. Permute values: launch `gather_kernel(values_tilde, values, perm, nnz_map)` where `nnz_map` is a per-plan array computed once at analyze time mapping original nnz index → permuted nnz index. (§4.5 discusses alternatives.)
4. Permute d_star and r: `d_star_tilde[new] = d_star[perm[new]]`, `r_tilde[new] = r[perm[new]]`. Two N-element gathers.
5. **Forward sweep on $\tilde{L}$**: for `c = 0, 1, ..., n_colors-1`:
   - Launch `multicolor_forward_sweep_kernel<<<grid, block, 0, stream>>>` with `color_offset = color_offsets[c]`, `color_count = color_offsets[c+1] - color_offsets[c]`.
   - The kernel computes `y_tilde[i] = (r_tilde[i] - sum_{j: j<i, (i,j) ∈ L̃} L̃[i,j] * y_tilde[j]) / d_star_tilde[i]` for `i` in color `c`.
   - Because colors are laid out contiguously (rows 0..color_offsets[1]-1 are color 0, etc.), `tid → i = tid + color_offset`.
6. **Middle**: elementwise `y_scaled_tilde[i] = d_star_tilde[i] * y_tilde[i]`, reused from Phase 2's `elem_scale_kernel` verbatim.
7. **Backward sweep on $\tilde{U}$**: for `c = n_colors-1, ..., 1, 0`:
   - Launch `multicolor_backward_sweep_kernel` with same offset/count.
   - Computes `z_tilde[i] = (y_scaled_tilde[i] - sum_{j: j>i, (i,j) ∈ Ũ} Ũ[i,j] * z_tilde[j]) / d_star_tilde[i]` for `i` in color `c`.
8. Inverse-permute: `z[old] = z_tilde[iperm[old]]`. One N-element gather.

**Kernel-launch count per apply**: `2 * n_colors + 3 (permutes) + 1 (elem_scale)` = `2 k + 4`. For red-black 7-pt, $k=2$, so 8 launches per apply. Phase 2 by comparison has ~6 visible launches (2 SpSV_solve — each internally a few kernels — plus copy/scatter/elem_scale) but cuSPARSE's internal kernel count per SpSV_solve is opaque. Either way, we are in the same order of magnitude; kernel-launch overhead is not the discriminator.

**Errors**: same as Phase 2's apply (`InvalidArgument` on fingerprint mismatch; `Internal` on CUDA failure).

### 3.3 `multicolor_refactor`

**Purpose**: when A's values change but pattern (and therefore coloring) doesn't, recompute $\tilde{D}_*$ without re-running the analyze path.

**Signature**:
```
Inputs:
  token  : uint64[1]
  values : float64[nnz]  -- NEW values in ORIGINAL ordering
Output:
  d_star_new : float64[N]  -- NEW d_star in ORIGINAL ordering
```

**Why a separate primitive and not a Python workflow**: unlike Phase 2 (where the user's `values` array is fed directly to `dilu_factor` and produces `d_star` in the original ordering via a standalone kernel), Phase 3's DILU factorization must run on the *permuted* matrix $\tilde{A}$ to produce $\tilde{D}_*$ whose diagonal entries match the color-permuted row order. If we computed $D_*$ on the original matrix and then tried to permute it, we would get the wrong $\tilde{D}_*$ — because DILU's recurrence depends on the fill-in-consistent order, and the reordering changes which $(k,i)$ pairs contribute to row $i$'s recurrence. Specifically, the math expert will confirm: DILU is order-dependent; $\tilde{D}_*(\tilde{A}) \neq P D_*(A) P^T$.

So `refactor` must:
1. Permute the new values: `values_tilde[new_k] = values[old_k]` via plan's `nnz_map`.
2. Update the plan's `values_tilde` buffer in place.
3. Run the Phase 2 `dilu_factor_kernel` on `values_tilde` → produces $\tilde{D}_*$ in permuted ordering.
4. Inverse-permute to original ordering for output: `d_star[old] = d_star_tilde[iperm[old]]`.
5. Also update the plan's cached `d_star_tilde` for use in `multicolor_apply`.

**Cost**: one call to the Phase 2 `dilu_factor_kernel` (~1 ms at N=10⁴) + two N/nnz-element gathers (~20 µs). Amortized across PCG iterations just like the apply cost.

**Why return `d_star_new` in original ordering**: consistency with `dilu_factor`'s output contract (Phase 2). The caller may want to inspect `d_star` for debugging without knowing about permutation.

### 3.4 `multicolor_release`

**Purpose**: free a plan. Identical in structure to `cusparse_dilu_release`.

**Signature**: `(token: uint64[1]) -> status: int32[1]`. Idempotent — releasing a gone token is a no-op warning, not an error.

### 3.5 Plan entry layout (C++)

```cpp
struct MulticolorPlanEntry {
  // Fingerprint — same principle as Phase 2, just (N, nnz).
  int32_t N;
  int32_t nnz;
  int32_t n_colors;

  // Plan-owned permuted CSR. All device pointers.
  int32_t* row_ptr_tilde;      // [N+1]
  int32_t* col_idx_tilde;      // [nnz]
  double*  values_tilde;       // [nnz] — updated per refactor and per apply
  int32_t* diag_offset_tilde;  // [N]

  // Plan-owned coloring metadata.
  int32_t* perm;               // [N] perm[new] = old
  int32_t* iperm;              // [N] iperm[old] = new
  int32_t* color_offsets;      // [n_colors+1]

  // Plan-owned value-permutation map so apply can gather in one kernel.
  int32_t* nnz_map;            // [nnz]; values_tilde[new_k] = values[nnz_map[new_k]]

  // Plan-owned persistent buffers.
  double* d_star_tilde;        // [N]  — permuted d_star, set by refactor
  double* r_tilde;             // [N]  — scratch for permuted r
  double* y_tilde;             // [N]  — scratch for forward-sweep output
  double* y_scaled_tilde;      // [N]  — scratch for elem_scale output
  double* z_tilde;             // [N]  — scratch for backward-sweep output
};
```

Memory footprint at N=10⁴ 7-pt Laplacian: `CSR permuted ~840 KB + 3×int32×N ~120 KB + 6×f64×N ~480 KB + nnz_map ~280 KB` = ~1.7 MB/plan. Same as Phase 2's working_values + y_mid + y_scaled + pattern copies.

### 3.6 I/O contract summary (engineering table)

| Primitive | Input dtypes | Output dtype | Hot path? | Host↔device copy |
|---|---|---|---|---|
| `multicolor_analyze` | 4× int32 CSR + int32 coloring (N + 2×N + (n_c+1)) + f64 values | uint64[1] | No (one-shot) | 8-byte H→D for token |
| `multicolor_apply` | uint64[1] + f64[nnz] + f64[N] + f64[N] | f64[N] | YES | 8-byte D→H for token (tolerated) |
| `multicolor_refactor` | uint64[1] + f64[nnz] | f64[N] | Warm (per outer step) | 8-byte D→H for token |
| `multicolor_release` | uint64[1] | int32[1] | No | 8-byte D→H for token |

---

## 4. CUDA Kernel Design for Color-Stride Solve

This is the heart of Phase 3. We specify the **forward** sweep; the **backward** is structurally symmetric (reverse color iteration, test `j > i` instead of `j < i`, start from `y_scaled_tilde` and write into `z_tilde`).

### 4.1 Pseudocode

```c
// Operates on the permuted CSR. All arrays device-resident. One kernel launch
// per color. Within a color, rows are INDEPENDENT by construction of the
// multi-coloring (§2.3 invariant 3).
__global__ void multicolor_forward_sweep_kernel(
    const int32_t* __restrict__ row_ptr,        // row_ptr_tilde
    const int32_t* __restrict__ col_idx,        // col_idx_tilde
    const double*  __restrict__ values,         // values_tilde (with D* on diagonal)
    const double*  __restrict__ d_star,         // d_star_tilde
    const double*  __restrict__ r,              // r_tilde (RHS of forward solve)
    double*              __restrict__ y,        // y_tilde (output)
    int32_t color_offset,                       // color_offsets[c]
    int32_t color_count)                        // color_offsets[c+1] - color_offsets[c]
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= color_count) return;
    int32_t row = color_offset + tid;    // because rows are laid out by color

    double sum = r[row];
    int32_t rs = row_ptr[row];
    int32_t re = row_ptr[row + 1];
    for (int32_t p = rs; p < re; ++p) {
        int32_t j = col_idx[p];
        if (j < row) {
            sum -= values[p] * y[j];
        }
    }
    // Diagonal of values_tilde holds d_star_tilde (scattered at apply start),
    // but we use the separately-supplied d_star vector for clarity and so the
    // scatter-diag step could later be elided if we keep the diag separate.
    y[row] = sum / d_star[row];
}
```

Note: `j < row` is the lower-triangle mask; for the backward sweep the kernel is identical with `j > row` and reading `y_scaled` instead of `r`, writing `z` instead of `y`.

### 4.2 Memory access pattern

After permutation, rows within one color are *contiguous indices*. This is the key property Path B exploits:

- **Row-ptr / col-idx / values reads**: thread `tid` reads `row_ptr_tilde[color_offset + tid]` and `row_ptr_tilde[color_offset + tid + 1]`. Adjacent threads read adjacent int32 slots → coalesced. Same argument for `col_idx[p]` and `values[p]` within each thread's row slice — but rows have variable-length slices (6 or 7 for interior 7-pt, 3–5 on boundary), so alignment is slightly ragged. Not a big deal at 7 entries per row.
- **y reads** (`y[j]`): this is the irregular one. `j` can point to *any* row in colors `0, 1, ..., c-1`. Two rows in the current color may have neighbors in entirely different prior-color regions → scattered reads, not coalesced. This is SpMV's classic gather-bottleneck. **Unavoidable.** Memory bandwidth is the limit, not arithmetic — same bottleneck Phase 2's cuSPARSE SpSV has internally, just more visible to us.
- **y writes** (`y[row]`): `row = color_offset + tid`, adjacent threads write adjacent slots → fully coalesced.
- **d_star reads**: `d_star[row]`, coalesced same as y writes.

**Prediction**: kernel is memory-bound with bandwidth roughly equal to SpMV over the same rows. On RTX 3050 (~224 GB/s peak), for a 10³ 7-pt case with ~7000 nnz in each color, we should see ~30 µs per color-sweep kernel at 100% BW — and half that at ~50% achieved. Two colors × 2 sweeps = ~60–120 µs per apply, vs Phase 2's 1200 µs. **If this prediction holds within 2×, Phase 3 is a meaningful win even after the 3× color-permutation overhead and even with iteration-count penalty.** If it doesn't hold (e.g., y-gather stalls dominate and bandwidth achievability drops to 10%), we report that empirically in §7 and the user decides whether to proceed.

**Disclaimer**: this is an engineering prediction based on memory-bandwidth arithmetic, not a measurement. The 3050's L2 cache (2 MB) can fit an entire N=10⁴ `y_tilde` array and most `values_tilde` for small problems — actual achieved bandwidth may exceed naive peak calc. Measurement in §7 is authoritative.

### 4.3 Warp divergence

Threads in one warp belong to adjacent rows in the same color. Their row-length (`re - rs`) varies by at most the stencil-radius variation — for a 7-pt interior, all rows have 7; for boundary rows, 4–6. So within a warp, up to 3 threads exit the for-loop 1–3 iterations earlier than others. This is a very mild divergence penalty. Not worth optimizing around.

For the 7-point stencil we can take a stronger approach: hard-code the stencil offsets and unroll the inner loop. But this forks the kernel into "structured 7-pt" and "general CSR" variants — premature specialization for Phase 3. We keep the general CSR kernel and measure.

### 4.4 The `j < row` branch

Generic CSR stores all entries per row (lower + diagonal + upper). The `if (j < row)` branch is taken roughly 3 times out of 7 per row for a 7-pt interior — not predictable, but the predicated-addition is cheap (~1 instruction). For the backward sweep the branch flips.

**Alternative (measured optimization, possibly for §7 step 6)**: precompute per-row triangle offsets — store `lower_end[i] = index in col_idx of last entry with j<i` and `upper_start[i] = first entry with j>i`. Then `for p in rs..lower_end[i]` for forward, `for p in upper_start[i]..re` for backward. Eliminates the branch entirely. Cost: 2× int32[N] extra in the plan. Benefit: modest (~10% kernel speedup likely). We defer until measurement shows whether it's worth the plan-structure change.

### 4.5 Atomics and synchronization

**None needed within a color.** By construction of the coloring (validated at analyze time), no two rows in color `c` share an off-diagonal. Hence the `sum -= values[p] * y[j]` accumulates always to thread-local `sum`, never to shared memory or global-via-atomic.

**Between colors**, we use kernel-launch boundaries as the synchronization. Each kernel launch's implicit grid-level sync ensures all writes from color `c-1` are visible to reads in color `c`. This is the simplest model and has no explicit `__syncthreads` or CUDA events.

**Is a single cooperative-groups kernel faster than N launches?** Possibly — cooperative-groups avoids the ~5 µs launch overhead per color. For red-black ($k=2$), the savings are 1× launch overhead = ~5 µs out of ~60–120 µs = 5–8%. Modest. For $k > 5$ the savings grow. We defer; measurement at §7 will show if launch overhead is a bottleneck.

### 4.6 Shared memory

**Not used.** The 7-point stencil accesses at most 7 distinct `y[j]` values per row. Loading them into shared memory and reusing is impossible because adjacent rows in the same color have totally different `j` neighborhoods by the coloring invariant. Each thread loads its own neighbors independently from global memory (through L1/L2 cache).

For a dense row-group (e.g., a Jacobi block), shared memory would matter. For our stencil matrix, it doesn't.

### 4.7 FLOP/byte analysis

Per nnz entry processed in a sweep:
- **Arithmetic**: 2 FLOPs (multiply-accumulate of `values[p] * y[j]`), plus 1 FLOP per row (the final divide by `d_star[row]`). So FLOPs/nnz ≈ 2.

- **Memory**: read `col_idx[p]` (4 bytes), `values[p]` (8 bytes), `y[j]` (8 bytes, scattered) → 20 bytes/nnz. Plus per-row: `row_ptr` + endpoints (~8 bytes), `d_star[row]` (8), `y[row]` write (8), `r[row]` read (8) → 32 bytes/row ÷ 7 nnz/row ≈ 5 bytes/nnz.

Total: ~25 bytes of traffic per 2 FLOPs = arithmetic intensity 0.08 FLOPs/byte. For the 3050 (5 TFLOPS FP64 / 224 GB/s → balance point 22 FLOPs/byte), this is **deeply memory-bound**, as expected for SpMV-family operations. The only knob is achieved bandwidth.

This is the *same* memory-bound character as cuSPARSE SpSV. The win Phase 3 is going for is **not** a fundamentally faster arithmetic kernel — it's **removing the level-schedule overhead between kernel stages** that cuSPARSE pays for the worst case. Red-black has exactly 2 stages; cuSPARSE doesn't know that and does the general-case dance.

---

## 5. Pattern Fingerprint and Lifetime Management

Phase 2's rule: plan owns copies of the pattern arrays, fingerprint check on `(N, nnz)` at apply. We extend it for Phase 3.

### 5.1 What Phase 2's fingerprint already catches

- Accidentally feeding apply a different-sized matrix: `(N, nnz)` mismatch → `InvalidArgument`.

### 5.2 What Phase 3 must additionally guard

- **Coloring invalidity**: the coloring we stored was computed for pattern P₁. If the caller reuses the plan with pattern P₂ (same `(N, nnz)`, different connectivity) but same dimensions, the cached coloring is wrong for the new pattern. Our sweep kernel assumes rows in one color are independent; violating that produces silently-wrong z with an error pattern that looks like a race condition on the `y` array.
  - **Detection**: same as Phase 2 — we cannot feasibly hash the pattern every call (GPU reduction cost). We rely on *(a)* the Python `MulticolorPlan` wrapper class holding strong refs to the caller's pattern arrays so they cannot be mutated out from under the plan, and *(b)* a clear docstring on `Plan.apply` warning that pattern mutation is undefined behavior.
- **Values change between refactor and apply**: values are expected to drift (that's why `refactor` exists). The plan tracks a `values_stale` flag (`bool`) set `true` by `refactor` (after update) and `apply` tolerates any values — they're gathered fresh from the caller's array per call.
- **Coloring staleness**: if the caller re-computes the coloring (e.g., they changed strategies) and wants to use it, they must call `release(old_token)` and `analyze(new_pattern, new_coloring)`. There's no "update coloring" primitive; coloring is part of the plan's identity.

### 5.3 Lifecycle contract

```
create plan           analyze(A_csr, coloring)           => token
recompute D*          refactor(token, new_values)        => new_d_star
apply preconditioner  apply(token, values, d_star, r)    => z
destroy plan          release(token)                     => ok
```

Invariants the Python `MulticolorPlan` class enforces (similar to Phase 2's `Plan`):

1. `row_ptr`, `col_idx`, `diag_offset`, `perm`, `iperm`, `color_offsets` arrays passed to `__init__` are held by strong refs for the plan's lifetime.
2. `apply(values, d_star, r)` is the only public entry point for the hot path. `values` must come from the same pattern as `__init__`'s `row_ptr/col_idx`, though values may differ.
3. `refactor(new_values)` recomputes `d_star` and returns it; caller threads the returned `d_star` into subsequent `apply` calls.
4. `__exit__` / `__del__` call `release`.

### 5.4 Fork-safety

Identical to Phase 2. The plan map is process-local; child processes after `fork()` inherit the map but not the CUDA context. We install a `pthread_atfork` handler that clears the map in the child. Phase 3 is single-process per brief; this is a defensive guard.

---

## 6. Build System

Mirror Phase 2's `CMakeLists.txt` with three targeted changes:

### 6.1 New target, no cuSPARSE dependency

```cmake
add_library(dilu_multicolor SHARED
  cpp/plan_registry.cc            # NEW — multicolor variant (same structure as Phase 2)
  cpp/multicolor_analyze.cc       # NEW — FFI handler
  cpp/multicolor_apply.cc         # NEW — FFI handler
  cpp/multicolor_refactor.cc      # NEW — FFI handler (simpler than analyze)
  cpp/multicolor_release.cc       # NEW — FFI handler
  cuda/dilu_factor_kernel.cu      # COPIED from Phase 2 verbatim
  cuda/permute_csr_kernel.cu      # NEW — analyze-time CSR permutation
  cuda/gather_kernel.cu           # NEW — apply-time value/vector gathers
  cuda/multicolor_sweep_kernel.cu # NEW — the heart of Phase 3
  cuda/elem_scale_kernel.cu       # COPIED from Phase 2 (middle D* scale)
)

target_link_libraries(dilu_multicolor PRIVATE
  CUDA::cudart)      # NOTE: no CUDA::cusparse.
```

**Explicit non-dependency on cuSPARSE** is a feature: we demonstrate that a DILU preconditioner can run on plain CUDA without any NVIDIA-library-specific analysis call.

### 6.2 Compile flags — unchanged from Phase 2

```cmake
target_compile_options(dilu_multicolor PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra -fno-fast-math>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 -lineinfo>)
```

Same CLAUDE.md conventions: no `-ffast-math`, no `--use_fast_math`, `-O2` (not `-O3`) for CUDA, `-lineinfo` for Nsight attribution.

### 6.3 CUDA architecture parameterization — unchanged

`CMAKE_CUDA_ARCHITECTURES=86` default (RTX 3050 Laptop); user can override for A100 (`80`) / H100 (`90`) in Phase 4.

### 6.4 Python side — sibling of Phase 2

```
dilu/multicolor/python/
├── __init__.py           # re-exports MulticolorPlan, coloring helpers
├── registration.py       # same pattern as Phase 2, registers 4 FFI targets
├── plan.py               # MulticolorPlan context manager
├── wrapper.py            # Python wrappers for the 4 FFI primitives
├── coloring.py           # red_black_color() + greedy_color_csr()
└── permute.py            # host-side permute_csr() (consistency check + fallback)
```

No package-level imports from `dilu/cusparse/` — the two are independent. A shared `dilu/common/` directory could hold the PCG driver skeleton, but to avoid touching Phase 2 we copy the PCG driver's skeleton into Phase 3's tests directory (it's ~40 LoC and copying it once is cleaner than a refactor).

### 6.5 Directory layout

```
dilu/
├── ffi_mvp/             [Phase 1, unchanged]
├── cusparse/            [Phase 2, unchanged]
└── multicolor/          [Phase 3, new]
    ├── CMakeLists.txt
    ├── build.sh
    ├── cpp/
    │   ├── plan_registry.h / plan_registry.cc
    │   ├── multicolor_analyze.cc
    │   ├── multicolor_apply.cc
    │   ├── multicolor_refactor.cc
    │   └── multicolor_release.cc
    ├── cuda/
    │   ├── dilu_factor_kernel.cu            # copied from Phase 2
    │   ├── elem_scale_kernel.cu             # copied from Phase 2 (or split off scatter_diag)
    │   ├── permute_csr_kernel.cu
    │   ├── gather_kernel.cu
    │   └── multicolor_sweep_kernel.cu
    ├── python/
    │   ├── __init__.py
    │   ├── registration.py
    │   ├── plan.py
    │   ├── wrapper.py
    │   ├── coloring.py
    │   └── permute.py
    ├── tests/
    │   ├── _harness.py                      # small CSR + PCG helpers
    │   ├── conftest.py                      # subprocess-per-test pattern
    │   ├── test_t4_factor_correctness.py
    │   ├── test_t5_diagonal_equivalence.py
    │   ├── test_t6_laplacian_vs_dense.py
    │   ├── test_t7_pcg_iteration_count.py
    │   ├── test_t8_coloring_validity.py     # NEW (Phase 3 specific)
    │   ├── test_under_jit_phase3.py
    │   └── physical_benchmark_phase3.py     # reuses Phase 2.5's three tests
    └── bench/
        ├── bench_multicolor_apply.py
        ├── bench_multicolor_pcg.py
        ├── bench_multicolor_vs_cusparse.py  # THE table the user wants
        └── plots/                           # Phase 3's own divergence_map.png etc.
```

### 6.6 What Phase 3 copies from Phase 2 unchanged

- `dilu_factor_kernel.cu`: the serial-on-device DILU factorization. Same math, applied to the permuted CSR.
- `elem_scale_kernel.cu`: the $y_\text{scaled} = D_* \odot y$ middle step. Literally identical.
- FFI handler macros (`CHECK_CUDA_APPLY`, `CHECK_CUSPARSE_APPLY` → renamed to drop the cuSPARSE one): same defensive style.
- Plan-registry pattern (singleton map, thread-id tripwire, `destroy_plan_entry`): same structure, adapted to the multicolor struct.
- Python registration pattern (`ctypes.CDLL` → `jax.ffi.pycapsule` → `jax.ffi.register_ffi_target`): copy of `registration.py` with different target names.

Keeping these verbatim is a deliberate choice: it isolates the Phase 3 risk surface to **the new sweep kernel + the coloring correctness**. Every other moving part has been battle-tested in Phase 2.

---

## 7. Testing & Verification Plan

### 7.1 Hard requirement: Phase 3 passes every Phase 2 correctness test AND every Phase 2.5 physical test

The user's directive is non-negotiable: "绝不能出现几何误差花纹." Phase 3 may produce a slower preconditioner, may produce a higher iteration count, may produce a different $D_*$ — but it **must not produce a different pattern in the three physical benchmarks**. Any new interface halo, any new parasitic vortex beyond Phase 2's levels, any interface-correlated residual pattern, is a STOP signal.

Implementation: `dilu/multicolor/tests/physical_benchmark_phase3.py` imports the Phase 2.5 physical-benchmark driver (`dilu/cusparse/tests/physical_benchmark.py`) and monkeypatches the `apply(values, d_star, r) -> z` callable to use `MulticolorPlan.apply` instead of `Plan.apply`. All physical setup (32³ grid, density-ratio-1000 sphere, 8000× three-tier, etc.) is identical. Plots land in `dilu/multicolor/bench/plots/` for side-by-side comparison. The `physical_benchmark.py` module's functions are already written to accept any callable; we verify that `dilu/cusparse/tests/physical_benchmark.py` has this property and, if not, we do a minimal refactor **only of the phase 3 copy** — no modification to Phase 2 code.

### 7.2 Test matrix

```
dilu/multicolor/tests/
├── test_t4_factor_correctness.py       # DILU factor, on the PERMUTED matrix
├── test_t5_diagonal_equivalence.py     # DILU on a pure-diagonal matrix reduces to Jacobi
├── test_t6_laplacian_vs_dense.py       # DILU apply vs dense-M solve on small Laplacians
├── test_t7_pcg_iteration_count.py      # DILU-mcPCG vs Jacobi-PCG AND vs Phase 2's DILU-PCG
├── test_t8_coloring_validity.py        # Coloring invariant check
├── test_under_jit_phase3.py            # HLO inspection: 1 custom-call, minimal copies
└── physical_benchmark_phase3.py        # re-run Phase 2.5 tests with the multicolor stack
```

### 7.3 Per-test specifications

#### T4 — DILU factor correctness (on permuted matrix)

- **Setup**: build the Phase 2 test matrices T1 (tridiag n=500), T2 (3D 7-pt Laplacian 10³ = 1000), T3 (diag-dominant random n=200). Compute a red-black or greedy coloring. Permute the matrix. Run `multicolor_analyze`, then retrieve `d_star_tilde` from the plan (exposed via a debug helper — not in the public API, but accessible for testing).
- **Reference**: scipy-based serial DILU on the permuted matrix, evaluated in the same row order.
- **Acceptance**: `|d_star_tilde_kernel - d_star_tilde_ref|_∞ ≤ tol` with tol matching Phase 2's T4 (2.22e-11 for tridiag, 1.33e-10 for Laplacian 10³, 4.43e-11 for random). The expectation is that on the *permuted* matrix the factor reaches the same ULP floor as Phase 2 did on the original matrix — DILU's recurrence is equally numerically stable in either ordering.
- **Likely failure**: wrong `diag_offset_tilde` computation (we must rebuild it on the permuted matrix, not reuse the original). Catching this early.

#### T5 — Diagonal matrix → Jacobi equivalence

- **Setup**: $A = \text{diag}(d)$. Every coloring is valid ($N$ colors, one per row — degenerate case handled by our greedy code). Apply should give $z = r / d$.
- **Acceptance**: `|z - r/d|_∞ ≤ 10 * eps`. Same as Phase 2.
- **Importance**: this is the correctness tripwire between our sweep kernel and the plan machinery — a pure Jacobi scenario lets us rule out sweep bugs.

#### T6 — Laplacian vs dense-LU reference

- **Setup**: 6³ = 216 Laplacian. Dense on CPU via scipy. Red-black coloring, `multicolor_analyze`, `multicolor_apply` on a random RHS.
- **Reference**: assemble the dense matrix $M_\text{mcDILU} = (D_* + \tilde{L}) D_*^{-1} (D_* + \tilde{U})$ **in the permuted ordering**, solve with `scipy.linalg.solve`, then inverse-permute back to original ordering.
- **Acceptance**: `|z_kernel - z_ref|_∞ ≤ 100 * cond(M) * eps * |r|_∞`. Phase 2 passed at ULP floor; Phase 3 may not because the sweep kernel's summation order differs from a left-to-right serial. Budget ~1000× ULP at worst.

#### T7 — PCG iteration count comparison (THE trade-off measurement)

- **Setup**: exactly the Phase 2 T7 stiff 16³ Laplacian (contrast 100, tol 1e-8, harmonic-mean faces). This reuses Phase 2's `_stiff_laplacian_3d()`.
- **Runs**: three PCGs on the same problem:
  1. Jacobi-PCG (reference baseline, expected ~71 iters per Phase 2 report).
  2. Phase 2 DILU-PCG via cuSPARSE (reference baseline, expected 24 iters).
  3. Phase 3 DILU-mcPCG via multicolor sweep (THE measurement).
- **Acceptance criteria** (published as success/fail):
  - Phase 3 iter count **must be ≤ 1.5× Phase 2 iter count** (i.e., ≤ 36 iters on the 16³ test). The 1.5× is a generous bound based on published red-black DILU studies: iteration inflation 10%–40% is typical, >50% suggests a bug.
  - Phase 3 iter count **must be strictly < Jacobi's** (on this stiff problem, this is a low bar — DILU-mc is still a vastly better preconditioner than Jacobi).
- **If Phase 3 iter count > 1.5× Phase 2**: STOP (§9 signal). The math expert is consulted: is this the inherent cost of red-black reordering on this stiffness, or is there a bug in our $\tilde{D}_*$ computation?
- **Report**: the iteration-count ratio is the headline number in the Phase 3 bench report.

#### T8 — Coloring validity (Phase 3 specific)

- **Setup**: for each of T1/T2/T3 and a few random CSR patterns, run `red_black_color` (where applicable) and `greedy_color_csr`. Verify invariant (§2.3 invariant 3): for every pair of nodes `(i, j)` with an edge in the adjacency graph of $A$ (i.e., `A[i,j] != 0` or `A[j,i] != 0` with `i != j`), `color_permuted[iperm[i]] != color_permuted[iperm[j]]`.
- **Acceptance**: zero violations. Any violation → STOP; the bug is in coloring code.
- **Rationale**: this is the "foundation under everything" test. The whole apply kernel's correctness assumes invariant (3).

#### test_under_jit_phase3

- **Setup**: wrap `MulticolorPlan.apply` inside `jax.jit`. Inspect compiled HLO.
- **Acceptance**:
  - Exactly one `custom-call` op with target `multicolor_apply` per apply invocation.
  - Zero `copy-start`/`copy-done` pairs (the 8-byte D→H token copy is handler-internal, same as Phase 2).
  - No `cudaMalloc`/`cudaFree` inside the compiled region (profile with Nsight Systems to confirm at runtime).

#### physical_benchmark_phase3

- **Setup**: re-run Phase 2.5's three tests (A/B/C) against the multicolor stack.
- **Acceptance**:
  - Test A max|∇·u| ≤ O(1e-8) after convergence (Phase 2: 4.13e-9; Phase 3 may be somewhat worse due to iter-count penalty but must be well below 1e-6).
  - Test B ‖u‖∞ grows sub-exponentially over 10 steps. Phase 2 measured linear 1e-7 → 1e-6; Phase 3 should be in the same order of magnitude.
  - Test C: mean-detrended `E - <E>` must be a smooth low-frequency gradient with **no halo tracing the three-tier density geometry**. This is the directive's 几何误差花纹 test. Visual inspection + automated spatial-correlation analysis: compute the correlation coefficient between `|E|` and `|∇c|` (the geometry gradient). Phase 2's Test C has correlation ~0.02 (uncorrelated). Phase 3 must also be < 0.1 (generous bound).
- **Plots**: all three PNGs regenerated under `dilu/multicolor/bench/plots/`. Document in the Phase 3 bench report with captions noting which version produced which plot.

### 7.4 The headline bench: `bench_multicolor_vs_cusparse.py`

This is the number the user wants.

- **Setup**: three matrix sizes: n=1000 (1-D tridiag, reuses Phase 2 baseline), n=10 000 (2-D or structured 3-D), 16³ = 4096 (the T7 stiff case), 32³ = 32 768 (Phase 2.5 test grid).
- **Measurements** per matrix, per preconditioner (Phase 2 vs Phase 3), on the same CUDA stream / same hardware / same JAX version:
  - `T_analyze` — one-shot cost.
  - `T_apply_median` — median over 1000 calls after warmup.
  - `T_apply_p95`.
  - `iter_count` — to reach tol = 1e-8.
  - `T_total_pcg = T_apply_median * iter_count + T_analyze + T_factor`.
- **Output**: a markdown table (committed to `docs/benchmark/phase3_multicolor_report.md`) listing Phase 2 vs Phase 3 for each matrix size, plus the ratio. Green highlight where Phase 3 wins, red where it loses.
- **Additionally**: Path A measurement. Build the same red-black permuted CSR, feed it through the unchanged Phase 2 cuSPARSE apply. Report that number as a third column. This tells us whether the multi-coloring *ordering itself* is beneficial (even to cuSPARSE) vs whether our custom kernel is what delivers the win.

This is the table from which the user decides whether Phase 3 was worth the engineering cost and whether to proceed to Phase 4.

### 7.5 What we do not test in Phase 3

- float32 precision — Phase 3 stays float64 per master plan.
- Multi-device / multi-stream — single device, single stream.
- Adaptive-refinement grids — static grid per brief.
- AMGx competitive comparison — Phase 4.

---

## 8. Execution Checklist

Ordered, blocking. Each step: **(a)** what, **(b)** acceptance, **(c)** likely failure.

### Step 1 — Coloring correctness (pure Python)

- **(a)** Implement `dilu/multicolor/python/coloring.py` with `red_black_color(nx, ny, nz)` and `greedy_color_csr(row_ptr, col_idx)`. Also implement `validate_coloring(row_ptr, col_idx, color)` that walks every edge and asserts invariant (3).
- **(b)** T8 passes on T1/T2/T3 and a set of 100 random-CSR tests with n=50.
- **(c)** Off-by-one on sort within a color; non-ascending `color_offsets`. Fix with unit tests before touching any CUDA.

### Step 2 — Host-side permutation

- **(a)** Implement `dilu/multicolor/python/permute.py: permute_csr(row_ptr, col_idx, values, perm, iperm) -> (row_ptr_new, col_idx_new, values_new, nnz_map)`. Also `compute_diag_offset` on the permuted matrix. Include a round-trip test: permute then inverse-permute should recover the original.
- **(b)** Round-trip produces bit-identical CSR.
- **(c)** Forgetting to sort within each row after remapping columns → cuSPARSE / Phase 2 compatibility breaks. Greedy coloring's natural order may produce ascending columns by accident — we can't rely on that.

### Step 3 — CMakeLists + empty .so

- **(a)** Copy Phase 2's CMakeLists as `dilu/multicolor/CMakeLists.txt`. Remove `CUDA::cusparse` link. Define all 4 FFI handlers as empty stubs returning success. Build.
- **(b)** `./build.sh` produces `libdilu_multicolor.so`. `nm -D` shows 4 handler symbols. Links against `libcudart.so` and not `libcusparse.so`.
- **(c)** Transitive link on cuSPARSE from a copied `.cu` file. Phase 2's `dilu_factor_kernel.cu` does not depend on cuSPARSE — confirm.

### Step 4 — `multicolor_analyze` skeleton

- **(a)** Implement the analyze handler. Permute CSR on device (via `permute_csr_kernel.cu`; for Phase 3 scope, we may do host-side permute + upload if device permute is tricky — fallback). Allocate plan-owned buffers. Store metadata. Return token.
- **(b)** Calling `multicolor_analyze` + `multicolor_release` in a tight 100-iter loop leaks < 5 MB on `nvidia-smi`.
- **(c)** Forgetting to copy `nnz_map` into the plan — apply's value gather reads garbage. Catch in Step 6.

### Step 5 — `multicolor_sweep_kernel`: forward only, one color

- **(a)** Implement the forward-sweep kernel. Test against a known-answer case: diagonal matrix (color `0` contains all rows, single kernel launch should produce `y = r / d_star`).
- **(b)** T5 (diagonal → Jacobi) passes.
- **(c)** `j < row` branch inverted. Fix by walking T5 by hand.

### Step 6 — `multicolor_apply`: full forward+middle+backward

- **(a)** Add backward sweep kernel (same body, reversed branch, reversed color iteration). Add middle `elem_scale`. Wire up the permute-in / inverse-permute-out plumbing.
- **(b)** T6 (Laplacian vs dense-LU reference) passes at 100× ULP or better.
- **(c)** Color-iteration order flipped (forward from high-color to low-color) — symptom is divergence on multi-color tests but correctness on 1-color diagonal. Debug with a 2-color 4×4 hand-worked example.

### Step 7 — `multicolor_refactor`

- **(a)** Implement the refactor handler: gather `values` to `values_tilde`, run Phase 2's `dilu_factor_kernel` on it, gather `d_star_tilde` back to `d_star`.
- **(b)** After `refactor`, a subsequent `apply` produces the same result as a fresh `analyze` + `apply` at ULP floor.
- **(c)** Forgetting to update the plan's cached `d_star_tilde` — subsequent applies use stale d_star. Fix by updating explicitly.

### Step 8 — T4 / T7 end-to-end

- **(a)** T4: `multicolor_analyze` factor correctness on the permuted matrix. T7: full PCG with `MulticolorPlan.apply`.
- **(b)** T4 passes at ULP. T7 DILU-mcPCG iter count ≤ 1.5× Phase 2 DILU-PCG. And both ≤ Jacobi-PCG.
- **(c)** T7 iter count explodes → STOP (§9 #2).

### Step 9 — under-jit + physical benchmark

- **(a)** Run `test_under_jit_phase3.py` and `physical_benchmark_phase3.py`.
- **(b)** HLO clean. All three physical plots visually free of interface halos. Automated correlation check `|E|` vs `|∇c|` < 0.1.
- **(c)** Physical Test C shows a new halo → STOP (§9 #3). This is the directive's non-negotiable line.

### Step 10 — Headline bench

- **(a)** Run `bench_multicolor_vs_cusparse.py` across all four grid sizes.
- **(b)** Produce `docs/benchmark/phase3_multicolor_report.md` with the full table. Include a textual honest assessment of the trade-off (per-iter speedup × iter inflation = total PCG time comparison).
- **(c)** Phase 3 is slower than Phase 2 on every size → this is useful information, write it up clearly and present to user. NOT a STOP — a legitimate negative result.

### Step 11 — Sign-off

- **(a)** Append "Phase 3 implementation results" section to this architecture doc. User reviews the trade-off table.
- **(b)** User signs Phase 3 complete, decides whether Phase 4 (AMGx) is justified by the Phase 3 numbers.
- **(c)** N/A.

---

## 9. Fail-Fast Escape Hatches

Per master plan rule #1. The top 7 STOP signals, ordered by likelihood × severity:

### STOP #1 — Coloring invariant violation

- **Symptom**: T8 fails. Or T6 passes but T7 produces wrong PCG residuals mid-iteration (iteration count does converge but final residual is 10⁻² instead of 10⁻⁸).
- **Detection**: T8 runs before any kernel work. `validate_coloring` is a 10-LoC Python check.
- **Escalation**: STOP. Fix the coloring algorithm. Do NOT attempt to patch around it in the kernel by adding color-internal syncs (that defeats the entire Phase 3 premise).

### STOP #2 — Iteration-count penalty > 1.5× on T7 stiff

- **Symptom**: Phase 3 DILU-mcPCG needs > 36 iters where Phase 2 needed 24.
- **Detection**: T7 in Step 8.
- **Escalation**: STOP. Write a 1-page diagnostic noting the penalty. Consult math expert: is this fundamental (the red-black ordering destroys enough of DILU's weak-coupling structure that the eigenvalue clustering is significantly worse) or is it a bug (e.g., $\tilde{D}_*$ is being computed from the *original* order not the permuted order)? Do not proceed to physical benchmarks until this is settled.

### STOP #3 — New interface halo in Phase 2.5 physical tests

- **Symptom**: Test C produces an `|E|` pattern correlated with `|∇c|` (correlation > 0.1). Or Test A shows a bright ring tracing the sphere interface in the divergence map. Or Test B's parasitic currents grow quadratically instead of linearly.
- **Detection**: `physical_benchmark_phase3.py` in Step 9. Both visual inspection and the automated correlation check.
- **Escalation**: STOP. This is the user's explicit red line. Do not ship Phase 3 with any hint of geometry error patterns. Likely cause: coloring is "valid" but introduces systematic error that manifests at interfaces because the red-black partition cuts across the interface in ways the grid ordering does not. This is a known theoretical risk for red-black SOR on stratified problems; Phase 3 must not exhibit it.

### STOP #4 — T4 factor mismatch on permuted matrix

- **Symptom**: `d_star_tilde_kernel` diverges from the scipy serial DILU on the permuted matrix by more than ULP.
- **Detection**: T4 in Step 8.
- **Escalation**: STOP. The DILU factorization is order-sensitive; if our permuted CSR is wrong (e.g., columns not sorted within rows after permutation) the kernel walks the wrong neighbors. Do not proceed; fix the permutation pipeline.

### STOP #5 — HLO shows unexpected copies

- **Symptom**: `test_under_jit_phase3` HLO contains `copy-start`/`copy-done` on any data other than the 8-byte token. Or the compiled region shows runtime `cudaMalloc` (caught via Nsight).
- **Detection**: Step 9 HLO inspection + Nsight capture.
- **Escalation**: STOP. Most likely: accidentally using `np.asarray` instead of `jnp.asarray` somewhere in the wrapper, forcing a host-roundtrip. Fix and re-verify before any performance measurement.

### STOP #6 — Plan map leak

- **Symptom**: analyze/release in a 1000-iter loop leaks > 50 MB on `nvidia-smi`.
- **Detection**: explicit leak test in Step 4.
- **Escalation**: STOP. This is a Phase 2 pattern we inherited; any leak suggests we missed a `cudaFree` in the destructor. Walk the `destroy_plan_entry` body against the struct definition field-by-field.

### STOP #7 — Path B loses to Path A on every grid size

- **Symptom**: `bench_multicolor_vs_cusparse.py` shows Phase 3 (our kernel) slower than a Phase-2-on-permuted-matrix (Path A baseline) on n=1000, n=10 000, 16³, AND 32³.
- **Detection**: Step 10.
- **Escalation**: NOT a silent fix. This is a legitimate negative result. Report to user with an honest write-up: "Path B was worth trying but did not beat Path A on this hardware; the kernel-launch overhead of our N-color sweep approach, combined with our inability to beat cuSPARSE's internal bandwidth-achievability on a permuted matrix, is the likely root cause." The user may still proceed to Phase 4 (AMG) with this data in hand, or may decide to go back to Phase 2 and call Phase 3 a learning loss. Either decision is OK; the important thing is the user is not misled.

---

## 10. Hardware Budget

### 10.1 Per-plan footprint

Per-matrix Phase 3 plan, for a 7-pt 3-D Laplacian at N³:

| Item | Size | At N=32³ = 32768, nnz≈229k |
|---|---|---|
| Permuted CSR (`row_ptr_tilde`, `col_idx_tilde`, `values_tilde`) | 4×(N+1) + 4×nnz + 8×nnz | 131 KB + 917 KB + 1834 KB ≈ 2.9 MB |
| `diag_offset_tilde` | 4×N | 131 KB |
| Coloring metadata (`perm`, `iperm`, `color_offsets`) | 4×N + 4×N + 4×(n_colors+1) | 262 KB + tiny |
| `nnz_map` | 4×nnz | 917 KB |
| `d_star_tilde`, `r_tilde`, `y_tilde`, `y_scaled_tilde`, `z_tilde` | 5×8×N | 1310 KB |
| **Total per plan** | | **≈ 6 MB** |

Compare Phase 2's per-plan cost at N=32³: ~5 MB (similar CSR + cuSPARSE workspaces). Phase 3 is ≈ same order — not a regression.

### 10.2 Max grid size on 4 GB VRAM dev box

Unchanged from Phase 2: comfortable at N ≤ 10⁶. Phase 3 has no cuSPARSE workspace overhead (saves a few MB at large N), offset by the `nnz_map` (adds a few MB at large N). Net zero.

### 10.3 Phase 3 recommended test grid sizes

- **Correctness**: 5⁴ = 625 (T4), 6³ = 216 (T6). Phase 2 numbers.
- **Iteration count (T7)**: 16³ = 4096 to match Phase 2's report headline.
- **Physical benchmarks (A/B/C)**: 32³ = 32768, exactly matching Phase 2.5.
- **Bench headline**: n=1000 (tridiag), n=10 000 (3-D 20³-ish), n=4096 (T7), n=32768 (physical).

No grid exceeds Phase 2's cap. Phase 3 stays in the 3050's comfort zone.

### 10.4 Plan-cache cap

Same as Phase 2: default 64 entries, env-var override. At 6 MB/plan, 64 plans = 384 MB — well under the 2 GB headroom. Not a concern.

---

## 11. Open Questions for Math Expert (Parallel Track)

These are the specific mathematical questions whose answers shape implementation details but not architecture:

1. **Is $\tilde{D}_* = \text{DILU}(P A P^T)$ always well-defined for red-black 7-pt Laplacians with density jumps?** I.e., are there degenerate cases where a row's recurrence produces $\tilde{d}_i = 0$ (pivot breakdown)? If yes, we need a numerical-breakdown guard in `dilu_factor_kernel.cu` — which already exists (the kernel emits NaN on breakdown).

2. **Expected iteration-count penalty from red-black reordering on our T7 stiff test**: is 10%–40% the right ballpark, or should we expect a larger penalty due to the 100× density jump? If the answer is "we don't know empirically, measure", that's fine — we'll measure and report.

3. **Should the sweep kernel use $\tilde{D}_*$ from the plan's diagonal of `values_tilde` (scattered at apply time, as Phase 2 does) or from a separate `d_star_tilde` vector (as §4.1 pseudocode uses)?** Both produce the same math. The architectural question is whether we want a `scatter_diag` step on the permuted matrix (Phase 2 pattern) or whether we keep the diagonal logically separate (simpler, one less kernel launch per apply). Recommend: keep separate for Phase 3, so we save a kernel launch. Math expert confirms equivalence.

4. **Does the coloring choice (red-black vs greedy) affect the numerical quality of $\tilde{D}_*$ beyond the trivial reordering?** Intuition: no — $\tilde{D}_*$ depends on the ordering, but any valid coloring produces a valid $\tilde{D}_*$ at the same numerical quality (within ULP of the reordered-serial reference). Math expert may confirm or reveal a subtlety we missed.

---

## 12. Phase 3 Engineering Acceptance Checklist

User sign-off required on all 8 before implementation begins.

1. **Path B committed.** Custom color-stride forward/backward sweep kernels. No cuSPARSE dependency in `libdilu_multicolor.so`. Path A baseline preserved in `bench_multicolor_vs_cusparse.py` for measurement only, not in the production stack.

2. **Host-side coloring only.** `red_black_color()` for 7-pt stencil fast path, `greedy_color_csr()` first-fit as fallback, both in pure Python with `validate_coloring()` invariant check (§2.3 invariant 3). Zero new external dependencies (no NetworkX, no SciPy coloring, no cuGraph). Python coloring runs at `analyze` time; result marshalled as 4 int32 arrays (`perm`, `iperm`, `color_offsets` + explicit `color`) through the FFI.

3. **Four FFI primitives agreed**: `multicolor_analyze` (returns token), `multicolor_apply` (hot path, takes original-ordering I/O), `multicolor_refactor` (values-only update), `multicolor_release`. Same opaque uint64-token lifetime and same 8-byte D→H-per-apply tolerated exception as Phase 2.

4. **Plan-registry pattern mirrors Phase 2.** Process-global `unordered_map<uint64, MulticolorPlanEntry*>`. Plan owns copies of the permuted CSR, coloring metadata, `nnz_map`, and all persistent scratch. Fingerprint check on `(N, nnz)` at apply. Same fork-safety / lifetime rules as Phase 2.

5. **Physical-benchmark acid test committed.** Phase 3 must pass Phase 2.5's Tests A/B/C with **zero new interface halo** on the 32³ grid. Plots regenerated under `dilu/multicolor/bench/plots/` for side-by-side visual comparison. Automated correlation check `|E|` vs `|∇c|` < 0.1 on Test C. Any violation is STOP signal #3, non-negotiable per user directive.

6. **Iteration-count trade-off measurement committed.** `bench_multicolor_vs_cusparse.py` produces the headline Phase 2 vs Phase 3 vs Path A table. T7's iter-count ratio (Phase 3 / Phase 2) must be ≤ 1.5. Failure is STOP signal #2 requiring math-expert consultation before proceeding.

7. **Build & directory hygiene**: new library `libdilu_multicolor.so` under `dilu/multicolor/`, lexically separate from `dilu/cusparse/`. Links `CUDA::cudart` only. Phase 1 and Phase 2 artifacts are NOT modified. `dilu_factor_kernel.cu` and `elem_scale_kernel.cu` are COPIED verbatim from Phase 2 into Phase 3's `cuda/` directory — no cross-directory imports.

8. **Top 7 STOP signals committed** (§9): coloring invariant violation, T7 iter count > 1.5×, new interface halo, T4 factor mismatch, unexpected HLO copies, plan-map leak, Path B loses to Path A on all sizes (honest negative result reporting). When any fires, implementer writes a diagnostic and escalates.

When all 8 are approved, implementation proceeds step-by-step per §8, with the user free to halt between any two steps. Phase 3 does NOT gate to Phase 4 (AMGx) — the final sign-off (Step 11) produces the data on which the user decides whether Phase 4 is justified.

**End of Phase 3 design document.**
