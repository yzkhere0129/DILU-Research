# Matrix-calculator benchmark — both lab machines

把 senior 的 melting (33 个) + evaporation (57 个) = **90 个 pd 矩阵**直接当做求解器的输入，
测 wall time + residual。**不动 OpenFOAM、不要 npz**，每台机直接读 raw CSV。

---

## 0. 数据前提

两台 lab 机上已经手动解压好原始 CSV：

```
~/DILU-Research/dilu/benchmark/Melting/Melting/Pre_Solving/...
~/DILU-Research/dilu/benchmark/Melting/Melting/After_Solving/...
~/DILU-Research/dilu/benchmark/Evaporation/Evaporation/Pre_Solving/...
~/DILU-Research/dilu/benchmark/Evaporation/Evaporation/After_Solving/...
```

bench 脚本和 C++ 求解器源码走 git，每台机 `git pull` 拿。

---

## 1. Lab Xeon (HR54WV2) — C++ PCG (DIC) 单核

```bash
ssh manyxu@HR54WV2  # 或 yzk@
cd ~/DILU-Research && git pull origin main

# 一次性: 编 C++ kernels
cd dilu/openfoam_cpu/cpp
./build.sh                      # 自动 pip install pybind11，强制 /usr/bin/g++
ls ../python/_kernels_cpp*.so   # 应该看到 .so

# 跑 bench (~10-15 min: preload ~7min + solve ~5min)
cd ~/DILU-Research
python3 -u -m dilu.amgx.bench.bench_pcg_xeon \
    --tol 1e-10 --max-iter 500 \
    --datasets melting,evaporation \
    > /tmp/bench_pcg_xeon.log 2>&1 &
tail -f /tmp/bench_pcg_xeon.log
```

输出 `/tmp/bench_pcg_xeon.json`。每个 case 报告：iter、wall、`converged`、`rel_resid_actual = ‖A·x - b‖₂/‖b‖₂`。

**已知**：DIC 在某些蒸发期 corr=2,3 case 上不收敛（结构性，不是 bug），bench 把这些标 `converged=N` 继续跑。AMGx 那边对应 case 一般是 OK 的。

回传：
```bash
mkdir -p ~/DILU-Research/dilu/amgx/bench/lab_deploy/results
cp /tmp/bench_pcg_xeon.{json,log} ~/DILU-Research/dilu/amgx/bench/lab_deploy/results/
cd ~/DILU-Research
git add dilu/amgx/bench/lab_deploy/results/bench_pcg_xeon.*
git commit -m "lab Xeon HR54WV2: bench_pcg_xeon results (90 matrices)"
git push origin main
```

---

## 2. Lab 5060 (manyxu@5060) — AMGx GPU + iterative refinement

```bash
ssh manyxu@5060
cd ~/DILU-Research && git pull origin main

# 跑 bench (3 模式 × 90 case，~10 min)
python3 -u -m dilu.amgx.bench.bench_amgx_5060 \
    --tol 1e-12 --max-iter 500 \
    --datasets melting,evaporation \
    > /tmp/bench_amgx_5060.log 2>&1 &
tail -f /tmp/bench_amgx_5060.log
```

输出 `/tmp/bench_amgx_5060.json`：3 模式 × 90 case，每行 (iter、setup_ms、update_ms、solve_ms、`rel_resid_actual`)。
3 模式：
- **fresh** — 每个 case 重 setup（最慢，含完整 cold-start 成本）
- **amortized** — 1 次 setup + N-1 次 update_coefficients（warm，模拟 PISO 同结构系列）
- **+1 IR** — amortized 基础上加一次迭代精化（机器精度）

回传：
```bash
mkdir -p ~/DILU-Research/dilu/amgx/bench/lab_deploy/results
cp /tmp/bench_amgx_5060.{json,log} ~/DILU-Research/dilu/amgx/bench/lab_deploy/results/
cd ~/DILU-Research
git add dilu/amgx/bench/lab_deploy/results/bench_amgx_5060.*
git commit -m "lab 5060: bench_amgx_5060 results (90 matrices)"
git push origin main
```

---

## 3. 两边数据回流后

我读两份 JSON 做综合：
- per-case wall heatmap (Xeon C++ 单核 vs 5060 AMGx GPU)
- precision frontier (residual vs total wall)
- melting → evaporation 时序 (per-step pd_corr0 wall 演化)
- 95% case CPU 大概 N×慢于 GPU 的对比图
