# Rayleigh-Taylor air/helium — PLIC 研究 benchmark

## 物理 case 描述

经典 Rayleigh-Taylor 不稳定基准(低密度比,Atwood ≈ 0.76):上方的
重流体(空气 `ρ_heavy = 1.225`)压在下方的轻流体(氦气
`ρ_light = 0.169`)上,初始界面是一个振幅 `0.05` 的单模余弦扰动。
重力驱动下,轻流体会在 `x = 0` 处形成上升 bubble,重流体会在
`x = 0.5` 处形成下沉 spike。本 case 的作用是**独立验证 PLIC 模块
在尖锐界面 + 高剪切二相流下的守恒性与几何精度**,ground truth 由
OpenFOAM (`interFoam`) 的运行结果给出,已打包在 `of_h1_h2.npz`。

**域**:`[0, 1] × [0, 4] × [0, 1/128]`,`128 × 512 × 1` 网格
**初始界面**:`y_interface(x) = 2.0 + 0.05 · cos(2π·x)`
**初始化**:每个 cell 做 `8 × 8` sub-grid 采样 → 光滑界面过渡 cell
**重力**:`g = -9.81 m/s²`(Stage 4 NS 耦合时启用)
**边界条件**(翻译自 OpenFOAM):
- top (`y = 4`):`fixedValue alpha = 1`
- bottom (`y = 0`):`fixedValue alpha = 0`
- left / right(`x = 0`, `x = 1`):`symmetry`
- frontAndBack(z):`empty`(inactive,单元厚度 `dz = dx`)

**精度**:float64

## 参数来源

所有数值 **严格照搬**
`/home/yzk/JAX-LaserAM-plic-research/examples/RT_air_helium/initAlpha.py`
和 `postProcess.py`:

| 参数 | 值 | 源行号 |
|---|---|---|
| Nx × Ny | 128 × 512 | `initAlpha.py:12` |
| Lx × Ly | 1.0 × 4.0 | `initAlpha.py:13` |
| 界面 baseline `y0` | 2.0 | `initAlpha.py:22` |
| 振幅 | 0.05 | `initAlpha.py:22` |
| 波数 | `2π` | `initAlpha.py:36` |
| ρ_heavy | 1.225 | `initAlpha.py:5` |
| ρ_light | 0.169 | `initAlpha.py:6` |
| sub-sampling | `n_sub = 8`(8×8 per cell)| `initAlpha.py:23` |
| h1 探针 | `x = 0`(i=0)| `postProcess.py:144-149` |
| h2 探针 | `x = Lx/2`(i=Nx//2)| `postProcess.py:153-158` |
| `find_alpha05_y` 二分插值 | — | `postProcess.py:70-80` |

**参考数据**:`of_h1_h2.npz`(从
`examples/RT_air_helium/of_h1_h2.npz` 复制过来)。

## 运行方式

```bash
# Stage 1 smoke test(现在可用):
cd /home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/rt_instability
/home/yzk/jax-env/bin/python run.py
```

**重要**:PLIC **单独**不能独立跑 RT 演化。理由:Rayleigh-Taylor
是重力驱动的**密度不稳定问题**,需要完整的不可压 Navier-Stokes 求解器
(动量对流 + 粘性应力 + 重力体力 + **压力投影**)才能产生响应界面
运动的速度场。Stage 2 仅交付**平流** sweep(给定速度场把 F 运过去),
没有动量/压力求解。

**真正的 RT 演化需要 Stage 4 NS 耦合**,本目录下 `run.py` 只做:
1. 构造初始 α 场(8×8 sub-grid 光滑)
2. 计算 Youngs 法线,验证 `|n_y| >> |n_x|`(界面接近水平)
3. 求解每个界面 cell 的 intercept `C`
4. 打印诊断并退出

## Gate 指标(Stage 4 完成判据)

| 指标 | 阈值 | 解释 |
|---|---|---|
| `h1` 在 `t ∈ {0.7, 0.8, 0.9, 1.0}` 的相对误差 | `< 10 %` | bubble 上升速率匹配 OpenFOAM |
| `h2` 在 `t ∈ {0.7, 0.8, 0.9, 1.0}` 的相对误差 | `< 10 %` | spike 下沉速率匹配 OpenFOAM |

参考值来自 `of_h1_h2.npz`(由 `save_of_npz.py` 导出)。

## 目录结构

```
rt_instability/
├── case_config.py       # 参数字典(CASE)
├── init_alpha.py        # create_rt_vof(grid) → F,8×8 sub-grid 向量化
├── run.py               # skeleton,含 Stage 1 smoke test
├── post_process.py      # h1/h2 提取,OpenFOAM 对比
├── of_h1_h2.npz         # OpenFOAM ground truth(从参考 case 复制)
└── README.md
```
