# Solver wall + precision comparison: OF DICPCG vs AMGx

## 速度对比表（同等 mesh 规模 ~500K cells）

| Solver                  | Hardware              | tol     | wall (ms) | iter | rel_resid_actual | 数据来源 |
|-------------------------|-----------------------|---------|-----------|------|------------------|----------|
| **OF DICPCG**           | Lab Xeon, 32 MPI ranks (28 phys + 4 SMT) | **1e-8** | **49.57** | ~5  | 1e-8 (= tol)    | spot_melt 500K, historical baseline (`bench_amgx_vs_of_n32_results.json`) |
| **AMGx**                | Lab 5060 (RTX 5060, sm_120, CUDA 13.2)    | **1e-8** | **120.6** | 17  | 8.1e-9          | senior initial-period 512K, `tol_sweep_results.json` |
| **AMGx**                | Lab 5060 (RTX 5060)                       | **1e-12** | **190.2** | 26  | 2.6e-13         | 同上 |
| AMGx + 1 IR             | Lab 5060 (RTX 5060)                       | 1e-12   | 240.0     | 26  | 2.2e-15 (= ε)   | 同上, `tol_sweep_results.json` (`n_refine=1`) |

**关键观察**：

1. **OF MPI 32-rank 在 500K cells 上比 AMGx 单 GPU 快 2.4×** —— 因为问题不够大，GPU 带宽优势没显现。GPU 优势随 N 增大显现（AMG 层次化的复杂度对大问题更友好）。
2. **AMGx tol 从 1e-12 放到 1e-8 → 加速 1.6×**（190.2 → 120.6 ms）。线性回报，不是指数级。
3. **AMGx + 1 IR 在 OF 永远到不了的精度 (2e-15) 上完成** —— 用 OF tol=1e-8 的代价换 7 个数量级精度。

---

## 精度对比（图：`docs/benchmark/figures/amgx_3d_solver_error.png`）

数据：LPBF_crosscheck pd_corr0 t=3.5e-9（2M cells，200×800×200 μm，dev RTX 3050 跑出的 x_AMGx）。
Truth = AMGx tol=1e-12（`‖A·x - b‖/‖b‖ = 1.7e-12`，机器精度代理；scipy SuperLU OOM 跑不动 2M）。

| 解 | tol | max abs err vs truth | rel err vs truth | 空间分布特征 |
|---|---|---|---|---|
| OF DICPCG | 1e-8 | **20.9 Pa** | **9.0e-6** | 均匀分布在熔池区 z=80-160μm |
| AMGx | 1e-8 | **5.9 Pa** | **2.6e-6** | 同样均匀，但比 OF 小 4× |
| AMGx | 1e-12 | 0 (定义) | 0 | (= truth 本身) |

**OF 在 tol=1e-8 时残差比 AMGx tol=1e-8 大 4×**（虽然两个 tol 同名，但因为算法不同，实际收敛点不一样）。
**AMGx tol=1e-8 已经比 OF 准 4 倍**，AMGx tol=1e-12 又比那个准 1000 倍以上。

---

## 综合评价：**AMGx 单 GPU vs OF MPI 32 核**

| 维度 | OF DICPCG @ tol=1e-8 | AMGx @ tol=1e-8 | AMGx @ tol=1e-12 |
|---|---|---|---|
| **wall 时间** | **快** (49.57 ms) | 中 (120.6 ms) | 慢 (190.2 ms) |
| **绝对精度** | 中 (1e-5 rel err) | 中-高 (1e-6 rel err) | **极高** (1e-13 rel err) |
| **算法优势** | DILU 单核简单 | AMG 层次扩展性好 | + 1 IR 直达机器 ε |
| **硬件需求** | 32 核 CPU MPI | 1 GPU | 1 GPU |
| **大问题扩展** | MPI 通信瓶颈 | GPU 带宽线性 | 同 |

**结论**：
- 500K cells 这个尺度，OF 在 wall time 上还能领先（多年优化）；
- AMGx 优势在 **更大 mesh** + **要求更高精度**（OF 的 tol=1e-8 是天花板）；
- 实际工程中如果要做精确 cross-check 或者 mesh > 5M，AMGx + 1 IR 唯一能给出机器精度解。

---

## 数据来源链接

- `bench_amgx_vs_of_n32_results.json` — Lab Xeon 32-rank OF baseline
- `tol_sweep_results.json` — Lab 5060 AMGx tol sweep
- `figures/amgx_wall_vs_tol.png` — wall vs tol 曲线
- `figures/amgx_3d_solver_error.png` — 3D 空间误差对比
