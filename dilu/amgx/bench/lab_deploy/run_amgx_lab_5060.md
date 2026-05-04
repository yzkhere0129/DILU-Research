# Lab 部署 #2：在 RTX 5060 机器上跑 AMGx on 学长 21 个 pd 矩阵

## 目标

**实测**AMGx 在 lab 5060 GPU 上的 wall time（不是本机 RTX 3050）。

5060 算力理论 ~3× RTX 3050（CUDA cores + memory bandwidth），所以期待 AMGx wall 从本机 1324ms → ~400-500ms。

## 前置依赖（在 5060 机器上）

```bash
# 1. CUDA + AMGx (跟本机一样)
# 2. JAX + jax-env
# 3. dilu-research repo + AMGx FFI build
git clone <repo URL> ~/DILU-Research
cd ~/DILU-Research
# build AMGx FFI lib (CMake)
cd dilu/amgx
mkdir -p build && cd build
cmake .. -DAMGX_INCLUDE_DIR=/path/to/amgx/include -DAMGX_LIB=/path/to/libamgxsh.so
make -j8
# 验证 libdilu_amgx.so 在 dilu/amgx/build/

# 4. 学长矩阵数据（拷贝整个目录）
scp -r yzk@source:/home/yzk/DILU-Research/dilu/benchmark/DICPCG_Benchmark_Data/ \
    ~/DILU-Research/dilu/benchmark/

# 5. python env
cd ~/DILU-Research
source jax-env/bin/activate
pip install numba scipy numpy
```

## 跑 bench

```bash
cd ~/DILU-Research

# 方法 1: 直接复用现有 bench script
python -u -m dilu.amgx.bench.bench_amgx_vs_of_n32 \
    > /tmp/amgx_5060_results.log 2>&1

# 方法 2: 跑 3-way 但只取 AMGx 部分
python -u -m dilu.amgx.bench.bench_threeway_senior \
    > /tmp/threeway_5060.log 2>&1

# 把结果拷回
scp /tmp/amgx_5060_results.log yzk@yourdesktop:/home/yzk/DILU-Research/dilu/amgx/bench/lab_deploy/amgx_5060_run.log
```

## 预期

| Metric | 本机 RTX 3050 | 5060 期待 |
|--------|--------------|----------|
| AMGx amortized solve | 1324ms | ~400-500ms |
| AMGx + 1 IR | 1438ms | ~450-550ms |
| AMGx setup (one-shot) | ~350ms | ~120ms |

## 注意

- AMGx 2.5.0 内部 update_coefficients 在 PCG+AMG 配置下是 ~600ms 固有开销（任何 GPU 都有，前面已诊断）
- 所以 5060 加速主要在 setup 和 solve 两段，update 那段不会变
- 总加速预计 ~2.5-3×（不会到 5×）

## 拷回后

我会用 5060 的真实 AMGx 数字 + lab Xeon 56 核的真实 OF 数字，**重做 3-way 表**。
