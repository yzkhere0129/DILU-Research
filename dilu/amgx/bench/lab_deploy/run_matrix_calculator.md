# Matrix-calculator benchmark — both lab machines

直接把 senior 那 21 + 57 = 78 个 pd 矩阵当作输入数据，跑各自的求解器测 wall + residual。
**完全不动 OpenFOAM**。

数据已经在 git 里 (605MB npz)，每台机 `git pull` 后就能跑。

## 矩阵预处理

每个 case 上做两件事保证 PCG / AMGx 都能正常跑：
1. `normalize_sign(A, b)` — OF 的 pd 矩阵 diag 是负的，翻转一下使 diag 为正
2. `b ← b - mean(b)` — Neumann 纯 Laplace 的 ker(A^T) = span(1)，把 b 投到 col(A) 上

报告的精度指标统一用：
```
rel_resid_actual = ‖A·x - b‖₂ / ‖b‖₂
```

---

## Lab Xeon (HR54WV2) — C++ PCG (DIC) 单核

```bash
ssh manyxu@HR54WV2  # 或 yzk@
cd ~/DILU-Research
git pull origin main

# 一次性: 编译 C++ kernels
cd dilu/openfoam_cpu/cpp
./build.sh                         # 用 jax-env Python + pybind11 + cmake
ls ../python/_kernels_cpp*.so      # 应该看到 .so

# 跑 bench (~10-30 分钟，取决于多少 case 收敛)
cd ~/DILU-Research
python3 -u -m dilu.amgx.bench.bench_pcg_xeon \
    --tol 1e-10 --max-iter 500 \
    --datasets initial,evaporation \
    > /tmp/bench_pcg_xeon.log 2>&1 &
tail -f /tmp/bench_pcg_xeon.log
```

输出：`/tmp/bench_pcg_xeon.json`。每个 case 一行 (iter, wall, converged, rel_resid_actual)。

**已知问题**：evap 数据集 corr=2,3 的某些 case DIC 预条件器不收敛（结构性，AMGx 没这个问题）。bench 会把这些标 `converged=N`，继续跑下一个，不会卡死。

回传：
```bash
mkdir -p ~/DILU-Research/dilu/amgx/bench/lab_deploy/results
cp /tmp/bench_pcg_xeon.{json,log} ~/DILU-Research/dilu/amgx/bench/lab_deploy/results/
cd ~/DILU-Research
git add dilu/amgx/bench/lab_deploy/results/bench_pcg_xeon.*
git commit -m "lab Xeon HR54WV2: bench_pcg_xeon results (initial+evap, 78 matrices)"
git push origin main
```

---

## Lab 5060 (manyxu@5060) — AMGx + iterative refinement

```bash
ssh manyxu@5060
cd ~/DILU-Research
git pull origin main

# 跑 bench (3 模式: fresh / amortized / +1 IR; 每个数据集 ~1-2 分钟)
python3 -u -m dilu.amgx.bench.bench_amgx_5060 \
    --tol 1e-12 --max-iter 500 \
    --datasets initial,evaporation \
    > /tmp/bench_amgx_5060.log 2>&1 &
tail -f /tmp/bench_amgx_5060.log
```

输出：`/tmp/bench_amgx_5060.json`。3 个模式 × 78 cases，每行 (iter, setup/update/solve_ms, rel_resid_actual)。

回传：
```bash
mkdir -p ~/DILU-Research/dilu/amgx/bench/lab_deploy/results
cp /tmp/bench_amgx_5060.{json,log} ~/DILU-Research/dilu/amgx/bench/lab_deploy/results/
cd ~/DILU-Research
git add dilu/amgx/bench/lab_deploy/results/bench_amgx_5060.*
git commit -m "lab 5060: bench_amgx_5060 results (initial+evap, 78 matrices)"
git push origin main
```

---

## 两边数据回流后

我会读 `bench_pcg_xeon.json` + `bench_amgx_5060.json`，写综合对比图：
- per-case wall heatmap (Xeon C++ vs 5060 AMGx)
- per-iter cost 对比
- precision frontier (residual vs wall)
- 蒸发期 57 cases 的时间序列图

如果 Xeon 上有些 case 不收敛，AMGx 那边对应 case 我会标出来作为参考；converged 的 case 之间做 head-to-head 比较。
