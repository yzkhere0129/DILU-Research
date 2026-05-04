# Lab 一键启动（git pull 之后照着敲）

代码已 push 到 `git@github.com:yzkhere0129/DILU-Research.git` 的 `main` 分支
（commit `315df5e`）。两台 lab 机都按下面操作。

---

## 共用第 0 步：在两台 lab 机上 git pull / clone

```bash
# 如果是新机器
git clone git@github.com:yzkhere0129/DILU-Research.git ~/DILU-Research

# 如果已 clone 过
cd ~/DILU-Research && git pull
```

---

## Lab 机 #1 (HR54WV2, 56 核 Xeon Gold 5120) — 跑 OpenFOAM N=32

**目的**：在 80³=512K mesh (跟学长矩阵同尺寸) 上实测 OF PCG-DIC 在 N=32 的 wall。

```bash
cd ~/cases  # 或者你 OF case 习惯的目录
cp -r /home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/spot_melt_150W ./spot_melt_80cube_n32_bench
cd spot_melt_80cube_n32_bench

# === Mesh 改成 80x80x80 (匹配学长 mesh) ===
# blockMeshDict 里找到 "hex (...) (X Y Z)" 那行
# 把 X Y Z 都改成 80
nano system/blockMeshDict
# 或 sed 一把（如果格式固定）：
# sed -i 's/(\s*[0-9]\+\s\+[0-9]\+\s\+[0-9]\+\s*)\s*simpleGrading/(80 80 80) simpleGrading/' system/blockMeshDict

# === endTime 跑 30 步左右 ===
sed -i 's/^endTime.*/endTime  6e-7;/' system/controlDict

# === T solver 加 log 2 让它打 normFactor + 时间 ===
# pd 部分已经在 fvSolution 里了；如果没有 log 字段就加上
grep -A 5 "^    pd$" system/fvSolution

# === Build mesh ===
blockMesh
setSolidFraction

# === decompose 成 32 ===
cat > system/decomposeParDict <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
method scotch;
numberOfSubdomains 32;
EOF
decomposePar -force

# === RUN ===
mpirun --oversubscribe --bind-to none -np 32 laserMeltFoam -parallel > log.run 2>&1

# === 提取 pd 求解 wall time ===
# 假设 matrixDumper patched solver 打 [TIMING_pd] X.XXX ms 行
grep "TIMING_pd" log.run | awk '{print $3}' | sort -n | \
    awk 'BEGIN{c=0} {a[NR]=$1; c++} END {
        print "N=" c, "median=" a[int(c/2)] " ms",
              "min=" a[1] " ms", "max=" a[c] " ms"}'

# 如果 log 里没有 TIMING_pd（patched solver 未装），用：
grep "DICPCG.*Solving for pd" log.run
# 这只能给 iter 数，不给 wall time。需要 patched solver。

# === 拷回本机 ===
# 直接发本机时把 pd_timings.txt 一起发我
grep -E "TIMING_pd|DICPCG.*pd" log.run > pd_timings.txt
# 然后 scp 或贴给我
```

**预期产出**: `pd_timings.txt` 含 N≈90 个 pd solve wall time，median 应在 45-60ms 区间。

---

## Lab 机 #2 (RTX 5060) — 跑 AMGx on 学长 21 矩阵

**目的**：实测 AMGx 在 lab 5060 GPU (vs 本机 RTX 3050) 上的 wall。

### 步骤 1 — 一次性环境 setup（跳过如已 setup）

```bash
# CUDA + AMGx — 假设已装。如未装：
# https://github.com/NVIDIA/AMGX
# build 后 lib 在 ~/local/amgx/lib/libamgxsh.so

# JAX env
python3 -m venv ~/jax-env
source ~/jax-env/bin/activate
pip install --upgrade pip
pip install jax[cuda12] numba scipy numpy pytest matplotlib

# Build dilu/amgx FFI lib (links to AMGx)
cd ~/DILU-Research/dilu/amgx
mkdir -p build && cd build
cmake .. \
    -DAMGX_INCLUDE_DIR=$HOME/local/amgx/include \
    -DAMGX_LIB=$HOME/local/amgx/lib/libamgxsh.so
make -j8
# 生成 dilu/amgx/build/libdilu_amgx.so
```

### 步骤 2 — 学长数据（已通过 git 提交，无需 scp）

学长 21 个 pd 矩阵已转成 161 MB npz 提交到仓库
（`dilu/benchmark/DICPCG_Benchmark_Data_npz/bundle_pd_*.npz`）。
git pull 时一起拉下来。

验证：
```bash
ls ~/DILU-Research/dilu/benchmark/DICPCG_Benchmark_Data_npz/ | wc -l
# 应该 21 个 .npz 文件
```

### 步骤 3 — 跑 bench

```bash
cd ~/DILU-Research
source ~/jax-env/bin/activate

# 3-way wall time bench (~2 min)
python -u -m dilu.amgx.bench.bench_amgx_vs_of_n32 \
    > /tmp/amgx_5060_bench.log 2>&1
tail -15 /tmp/amgx_5060_bench.log

# 精度 sweep + IR (~3 min)
python -u -m dilu.amgx.bench.sweep_amgx_senior_data \
    --tol 1e-12 --n-refine 1 --no-truth \
    --out /tmp/amgx_5060_precision.json \
    > /tmp/amgx_5060_precision.log 2>&1
tail -10 /tmp/amgx_5060_precision.log

# === 把 log + json 拷回本机 ===
scp /tmp/amgx_5060_*.{log,json} \
    yourdev:/home/yzk/DILU-Research/dilu/amgx/bench/lab_deploy/
```

**预期产出**:
- `amgx_5060_bench.log` — 含 fresh / amortized / amortized+IR 三个 wall time
- `amgx_5060_precision.json` — 21 矩阵的 AMGx 精度（vs xref）

期待 5060 wall ≈ **400-500ms**（vs 本机 RTX 3050 1324ms，约 3× 加速）。

---

## 拷回本机后

把 `pd_timings.txt` (lab Xeon) 和 `amgx_5060_*.{log,json}` (lab 5060) 都贴给我或放到
`dilu/amgx/bench/lab_deploy/` 下。我会写**真正的** 3-way 报告（替换现在 DRAFT 那个用了
旧数据的版本）。

---

## 时间预算

| 步骤 | 预计 |
|------|------|
| Lab Xeon: clone + blockMesh + decompose + 跑 30 步 | 30-40 分钟 |
| Lab 5060: clone + 装环境 + scp 数据 + 跑 bench | 15-30 分钟（环境已装则 5 分钟）|
| 数据拷回 + 我重写报告 | 10 分钟 |

总共 1 小时左右，能拿到全 fair 的 3-way 数据。
