# AMGx 精度对齐计划（→ ‖x_AMGx − x_OF‖∞ / ‖x_OF‖∞ ≤ 1e-10）

**日期**: 2026-05-04
**作者**: Claude + yzk
**前置文档**:
- `docs/benchmark/OPENFOAM_CROSSCHECK_20260427_v3.2.md`（v3.2 实测：pd 2.5e-6, T 1.04e-8）
- `docs/reference/openfoam_dilu_source_audit.md` §4 / §10（PBiCG + normFactor）

**目标**:
| 方程 | v3.2 实测 rel_vs_OF | 本次目标 | 缺口（数量级） |
|------|--------------------|----------|---------------|
| pd   | 2.5e-06             | **1e-10** | **4 OOM** ← 主战场 |
| T    | 1.04e-08            | **1e-10** | 2 OOM |

**Reference**: 沿用现有 `x_final.mm`（OpenFOAM PBiCG-DILU @ tol=1e-12 解）。不重跑 OpenFOAM。

**测试范围**: 48 个 dump（12 cases × 3 pd correctors + 1 T），全部 ≤ 8K cells，ms-级 solve。

---

## §1 Audit findings（Phase 1 已完成）

### §1.1 AMGx 模块结构

```
dilu/amgx/python/
├── config.py        9 个 preset configs
├── wrapper.py       JAX FFI 入口（amgx_setup/solve/update/release）
├── plan.py          Plan context manager + update_coefficients
└── registration.py  one-time FFI registration
```

driver 在 `dilu/benchmark/openfoam_crosscheck/driver_amgx.py`，模式为 `dDDI`（double everything），`fastmath=False`。

### §1.2 已有 9 个 config 一览（按"潜在精度"排序）

| Config | outer | 预条件 | scaling | smoother | sweeps | tol | max_iter | 用于 |
|--------|-------|--------|---------|----------|--------|-----|----------|------|
| classical_v | PCG | classical AMG V | none | BLOCK_JACOBI | 1/1 | 1e-10 | 200 | pd baseline 起点 |
| aggressive | PCG | classical + aggressive_levels=2 | none | BLOCK_JACOBI | 1/1 | 1e-10 | 200 | OOM 备选 |
| **classical_v_diagscaled** | PCG | classical AMG V | DIAG_SYM | BLOCK_JACOBI | 1/1 | 1e-10 | 200 | **v3.2 pd winner** |
| **classical_v_diagscaled_tight** | PCG | classical AMG V | DIAG_SYM | BLOCK_JACOBI | **2/2** | **1e-14** | **500** | 已存在但未系统验证 |
| classical_v_diagscaled_bicgstab | BiCGStab | classical AMG V | DIAG_SYM | BLOCK_JACOBI | 1/1 | 1e-10 | 200 | **v3.2 T winner** |
| classical_gs_pcg | PCG | classical AMG V | none | MULTICOLOR_GS sym | 2/2 | 1e-10 | 200 | 强 smoother |
| classical_gs_bicgstab | BiCGStab | 同上 | none | MULTICOLOR_GS sym | 2/2 | 1e-10 | 200 | T 强 smoother |
| aggregation_pcg | PCG | aggregation AMG | none | MULTICOLOR_GS | 2/2 | 1e-10 | 200 | classical 失败时 fallback |
| mini_amg_test | PCG | max_levels=5 | none | BLOCK_JACOBI | 1/1 | 1e-10 | 200 | smoke test，不用 |

收敛准则均为 `convergence: RELATIVE_INI_CORE` + `norm: L2`，即 `‖r_k‖₂ / ‖r₀‖₂ ≤ tol`。
跟 OpenFOAM 的 `‖r‖₁ / normFactor ≤ tol` 不同 → tol 数字不可直接互比，**实际收敛深度可能差 1-2 OOM**。

### §1.3 driver 已有的辅助操作

- **`normalize_sign`**：LPBF pd 用负 diag laplacian（OpenFOAM 约定），driver 自动 negate `(A,b)→(-A,-b)`，使 AMGx PCG 看到正定 diag。x 不变。✅ 不影响精度。
- **`regularize(eps_rel)`**：`A → A + eps_rel·max(|diag|)·I`。**默认 0**（v3.2 没用）。如果用了，`rel_vs_OF` 下界就是 O(eps_rel)，**1e-10 目标下不能用 eps_rel ≥ 1e-10**。
- **`amortized` 模式**：跨 case 复用 plan，新 A 走 `update_coefficients`。可能让 AMG hierarchy 跟当前矩阵 sub-optimal。**Phase 2 必须先在 fresh setup 下做基线**。

### §1.4 已有 prior work

`dilu/benchmark/openfoam_crosscheck/precision_experiment.py` —— 已经实现 baseline (`tol=1e-10, sweeps=1`) vs tight (`tol=1e-14, sweeps=2`) 在 spot_melt_npz 数据集上的对比。**但跑过的输出在 `collected/spot_melt_npz/` 那个数据集，不是我们刚 dump 的 12 cases**。

需要：把 precision_experiment.py 的逻辑迁移到指向当前 dumps（`/home/yzk/LaserbeamFoam/.../{LPBF_sanity, dumper_pipeline_test}/postProcessing/matrices/...`）。

### §1.5 数据可用性

| 数据集 | path | cases | mat 类型 | 总 dump | 含 sumA/normFactor.txt |
|--------|------|-------|----------|---------|----------------------|
| LPBF_sanity (2K) | `~/LaserbeamFoam/.../LPBF_sanity/postProcessing/matrices/` | 2 | pd_corr0/1/2 + T_corr0 | 8 | ✅ (新 dump) |
| dumper_pipeline_test (8K) | 同上 path 但 dumper_pipeline_test | 10 | 同上 | 40 | ✅ (新 dump) |
| **合计** | | **12** | **3 pd + 1 T 每 case** | **48** | ✅ |

PBiCG-DILU 复刻已在这 12 cases 上 byte-exact 通过（**意味着 (A, b, x_final) 确实是 OpenFOAM 自洽 reference**）。

---

## §2 精度差距的可能根源（Phase 2 待验证）

按"修起来从易到难"排：

### §2.1 [P0] AMGx tolerance 不够紧

baseline `tol=1e-10` 是 PCG 的 `‖r_k‖₂ / ‖r₀‖₂`。κ(A) 高时，x 误差 ~ κ × residual。
- LPBF pd 矩阵已知 κ ≈ 10^14（v3.2 §1.1 说"几乎奇异 + 14 数量级 diag 跨度"）
- 理论上：x 1e-10 误差 → residual 需要 ~ 1e-10 / κ_eff ≈ 1e-24 → **远超 float64 机器精度 1e-16**
- 但实际 κ_eff（DIAG_SYM 后）可能 ≈ 10^4-10^6，那 residual 只需 1e-14 ~ 1e-16

**Action**: tighten tol 到 `[1e-12, 1e-13, 1e-14, 1e-16]` sweep。

### §2.2 [P0] AMG hierarchy 质量

v3.2 的 BLOCK_JACOBI smoother + 1/1 sweeps 是 minimum quality。`tight` config 已经升到 2/2 sweeps。可能还可以：
- presweeps/postsweeps = 3/3
- 换 MULTICOLOR_GS smoother（sym=1）
- interpolator D2 → 试试 multipass / extended

**Action**: smoother + sweeps sweep。

### §2.3 [P1] DIAG_SYM scaling 的 unscale 误差

`scaling: DIAGONAL_SYMMETRIC` 把 A → D⁻¹ᐟ² A D⁻¹ᐟ²，b → D⁻¹ᐟ² b，解出 y = D¹ᐟ² x → x = D⁻¹ᐟ² y。
- diag 跨 14 个数量级时，D⁻¹ᐟ² 的某些元素 ~ 10^7
- 一次 unscale 引入 ~ 1 ULP × 10^7 = 1e-9 量级误差 → **可能正是 2.5e-6 → 1e-10 难以跨越的隐藏地板**

**Action**: 对照实验 — fresh setup, no scaling vs DIAG_SYM。如果 rel_vs_OF 上限被 scaling 卡住，要换策略（外层迭代 refinement，或不 scaling 但加更多 sweeps）。

### §2.4 [P1] amortized hierarchy 漂移

amortize 用 case_1 的 A 建层级，到 case_50 时 A 已变化。如果非线性太强 → AMG sub-optimal → 收敛到的 x 精度受限。

**Action**: Phase 2 默认 fresh setup（不 amortize）。amortize 留作性能阶段再调。

### §2.5 [P2] AMGx 自身的 PCG 实现细节

AMGx 的 PCG 内部用 `r ← r − α A·p` recursive update（跟 OpenFOAM 一样），可能在收敛末段 true residual ≠ recursive residual。AMGx tol=1e-14 时停的是 recursive，可能 true 差 1-2 OOM。

**Action**: 检查 AMGx 报的 `iters` + 我们计算的 `‖A x − b‖` 真残差，确认两者吻合。

### §2.6 [低风险，已确认无问题] 浮点精度

wrapper.py:34/50/70 全部 `jnp.float64`，C++ FFI 用 `dDDI` mode（double-double-double-int）。无 float32 路径。✅

### §2.7 [低风险，已确认无问题] 矩阵输入

PBiCG-DILU 复刻验证了 (A, b, x_final) 三元组在 12 cases 上 byte-exact reference。AMGx 拿同一份 (A, b)，输入端无误差。✅

---

## §3 Phase 2 — 单 case 深入诊断（轻负载）

### §3.1 选样

- **pd**: `dumper_pipeline_test/2.64e-12/pd_corr0`（中等 dt, 8K cells, 应有非平凡条件数）
- **T**:  `dumper_pipeline_test/2.64e-12/T_corr0`（同 timestep，便于对照）

每 case 单 solve < 1s。

### §3.2 sweep matrix（5 axis × ~3 levels = ~30-60 configs）

| Axis | 试 |
|------|---|
| **outer** | PCG（pd）/ BiCGStab（T） |
| **scaling** | none / DIAG_SYM |
| **smoother** | BLOCK_JACOBI / MULTICOLOR_GS_sym |
| **sweeps** | 1/1 / 2/2 / 3/3 |
| **tol** | 1e-10 / 1e-12 / 1e-14 / 1e-16 |
| **interpolator** | D2 (默认；保持) |
| **cycle** | V (默认；保持) |
| **amortize** | OFF（Phase 2 全部 fresh setup） |
| **regularize** | 0（绝不开） |

总 configs: pd 1×2×2×3×4 = **48 个**, T 同 = **48 个**。共 96 次 solve, 总用时 < 2 分钟。

### §3.3 输出

每个 (case, config) 记录：

```json
{
  "case": "dumper_pipeline/2.64e-12/pd_corr0",
  "cfg_axes": {"outer":"PCG", "scaling":"DIAG_SYM", "smoother":"BJ",
               "sweeps":[2,2], "tol":1e-14},
  "iters": 23,
  "amgx_status": 0,
  "rel_residual": 4.7e-15,    # ‖A x_amgx − b‖₂ / ‖b‖₂
  "rel_vs_OF": 8.3e-11,       # ‖x_amgx − x_OF‖∞ / ‖x_OF‖∞   ← TARGET ≤ 1e-10
  "wall_s": 0.067
}
```

### §3.4 分析（不跑额外 solve，纯数据分析）

- 出 `rel_vs_OF` heatmap by (sweeps × tol)，按 scaling/smoother 分面
- 找出 minimal config 让 `rel_vs_OF ≤ 1e-10`
- 如果 **没有任何 config 达到 1e-10** —— 说明走到 §2.3 / §2.5 的上限，需要换策略（mixed precision iter refinement）
- 输出表 + winning config 的精确 JSON

---

## §4 Phase 3 — 全集验证（**只在 Phase 2 找到 winning config 后**，需 user 同意才跑）

12 cases × winning config × 2 方程（pd + T）= 24 次 solve, < 1 分钟。
- 验证 winning config 在所有 12 cases 上 `rel_vs_OF ≤ 1e-10`
- 如果某些 case 不达标，分析特征（cond number, dt, scale）→ 给出 case-dependent strategy

---

## §5 Phase 4 — 产出

- `docs/benchmark/AMGX_PRECISION_20260504.md` — 完整结果 + 推荐 config
- `dilu/amgx/python/config.py` — 加 `CLASSICAL_V_DIAGSCALED_PRECISE` (winning config)
- `dilu/amgx/tests/test_amgx_vs_openfoam_precision.py` — 单测
- `docs/reference/openfoam_dilu_source_audit.md` — 加 §11 "AMGx 精度对齐"

---

## §6 风险与不确定性

| # | 风险 | 影响 | 缓解 |
|---|------|------|------|
| R1 | κ(A) ≈ 10^14 让 1e-10 在 float64 下不可达 | 主目标失败 | 接受 case-dependent ceiling；或上 mixed-precision iter refinement |
| R2 | DIAG_SYM unscale 引入 1e-9 级误差，是 2.5e-6 → 1e-10 的隐藏地板 | 所有 DIAG_SYM config 都 cap 在 1e-9 | Phase 2 fresh setup + no scaling 对照实验 |
| R3 | AMGx tol=1e-14 在某些 case 不收敛（recursive vs true 偏离） | 部分 case status≠0 | fallback 链：1e-14 → 1e-12 → 1e-10 + warn |
| R4 | x_OF reference 本身只到 OpenFOAM tol=1e-12 量级 → 1e-10 是其本征精度的下边缘 | 1e-10 接近 reference 的极限 | 接受这个 ceiling；或重跑 OpenFOAM tol=1e-14 拿更紧 reference |
| R5 | 8K case 可能太小，找出的 winning config 在 1M+ cell 上不 robust | 推广性问题 | 后续 用 LPBF_crosscheck 2M 重 dump 后再验证（出本计划范围）|

---

## §7 立即下一步（要 user 同意才执行）

Phase 2 是"轻负载"按用户允许：
- 单 case × 96 configs × 2 方程 ≈ 2 分钟
- 不写新模块（复用 driver_amgx.py 即可），写一个 sweep 脚本
- 全程在 LPBF_sanity 2K + dumper_pipeline_test 8K 数据上跑

要我开始 Phase 2 吗？
