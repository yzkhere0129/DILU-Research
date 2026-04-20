# 45° 液滴平流 — PLIC 研究 benchmark

## 物理 case 描述

Barkhudarov (2004) 经典 VOF 基准:一个半径 `R = 0.05` 的圆形液滴
以恒定速度场 `u = v = 1.0` 沿 45° 方向平流 `5` 个直径的距离。
纯运动学测试 — 没有 Navier-Stokes 求解、没有重力、没有表面张力 —
专门用于检验 PLIC 平流算法的**界面保形能力**和**体积守恒**。

**域**:`[0, 1] × [0, 1] × [0, 0.01]`,`100 × 100 × 1` 网格(dx = dy = 0.01)
**初始液滴**:center `(0.2, 0.2)`,smooth tanh 过渡,`ε = 1.5·dx`
**速度场**:字面量 `u = v = 1.0`(不是 `velocity_ref × sin 45°`)
**终态时间**:`t_end = 0.3536`(≈ `0.5/√2`,即 5 倍直径的 45° 位移)
**解析终点**:`center = (0.5536, 0.5536)`
**边界条件**:四个侧壁 symmetry(零梯度),z 方向 inactive
**精度**:float64

## 参数来源

所有数值 **严格照搬**
`/home/yzk/JAX-LaserAM-plic-research/examples/droplet_advection_45deg/run_vof_only.py`
的 `run_droplet_45deg` / `create_droplet` / `constant_velocity_faces` 函数:

| 参数 | 值 | 源行号 |
|---|---|---|
| 网格 100×100×1,nh=1 | `create_grid(nx, nx, nz=1, ..., nh=1)` | `run_vof_only.py:89` |
| 液滴 center 默认 | `center=(0.2, 0.2)` | `run_vof_only.py:27` |
| 液滴 radius 默认 | `radius=0.05` | `run_vof_only.py:27` |
| tanh 过渡宽度 | `eps = 1.5 * dx` | `run_vof_only.py:44` |
| F tanh 公式 | `F = 0.5 * (1 - tanh((r-R)/eps))` | `run_vof_only.py:45` |
| 速度 | `u_face = 1`, `v_face = 1` | `run_vof_only.py:69-70` |
| CFL | `cfl=0.5` 默认参数 | `run_vof_only.py:85` |
| t_end | `0.3536` | `run_vof_only.py:100` |

## 运行方式

```bash
# Stage 1 smoke test(现在可用):
cd /home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/droplet_45deg
/home/yzk/jax-env/bin/python run.py
```

**注意**:当前 `run.py` 只做 Stage 1 primitives 的 smoke test —
它计算初始 F 的 Youngs 法线 + intercept,然后打印统计量就退出。
真正的时间循环需要 **Stage 2** 完成以下两个还不存在的模块:

```
jax_laseram.vof.plic.geometric_flux    # sweep_flux_x, sweep_flux_y
jax_laseram.vof.plic.strang_sweep      # strang_split_step
```

完成后,`run.py` 的 `TODO(Stage 2)` 块会替换为完整时间循环,产出
`results/snapshot_*.npz` 文件,供 `extract_metrics.py` 和
`plot_results.py` 消费。

## Gate 指标(Stage 2 完成判据)

| 指标 | 阈值 | 解释 |
|---|---|---|
| 体积相对误差 `|V(t_end) - V0| / V0` | `< 0.1 %` | PLIC 通量算法必须**严格守恒** |
| 周长相对变化 `|P(t_end) - P0| / P0` | `< 5 %` | 数值扩散 / 界面分辨率是否足够 |

解析初值:
- `V0 = π·R²·dz = π·(0.05)²·0.01 ≈ 7.854e-5`
- `P0 = π·D = π·0.1 ≈ 0.3142`

## 目录结构

```
droplet_45deg/
├── case_config.py       # 参数字典(CASE)
├── init_droplet.py      # create_droplet_vof(grid) → F
├── run.py               # skeleton,含 Stage 1 smoke test
├── extract_metrics.py   # V / P 诊断(需 Stage 2 产出)
├── plot_results.py      # 可视化(带 mock fallback)
└── README.md
```
