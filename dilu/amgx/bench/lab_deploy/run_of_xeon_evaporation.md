# Lab Xeon (HR54WV2) — OpenFOAM benchmark on Evaporation phase

**目的**: 实测 OpenFOAM PCG-DIC 在 lab Xeon N=32 跑到**蒸发阶段** (t = 700ns - 1.06μs)
的 pd_corr0 wall, 跟 AMGx (lab 5060 同时跑) 对比。

**前提**: 学长 Evaporation 数据集已经 copy 到 lab Xeon 同路径
(`~/DILU-Research/dilu/benchmark/DICPCG_Benchmark_Data-20260506.../`).
我们这次**不**跑同样矩阵 (那需要专门的 OF utility), 而是**让 OF 自己跑到蒸发阶段
然后测 wall time** —— 跟之前 spot_melt N=32 49.57ms baseline 同方法。

## 步骤

```bash
ssh manyxu@HR54WV2  # 或 yzk@HR54WV2
cd ~/DILU-Research && git pull origin main

# 1. spot_melt case 之前 endTime=6e-7 (600ns), 蒸发阶段在 700ns - 1.06μs,
#    需要把 endTime 拉到 1.2e-6 (留点 buffer)
cd ~/cases/spot_melt_150W
cp system/controlDict system/controlDict.bak.evap

# 2. endTime → 1.2e-6 (覆盖整个蒸发阶段; 之前是 6e-7)
sed -i 's/^endTime.*/endTime         1.2e-6;/' system/controlDict
grep endTime system/controlDict   # 验证

# 3. 清旧 processor / time dirs (避免接续跑)
rm -rf processor* postProcessing
ls [0-9]* 2>&1 | head   # 应该只剩 0/

# 4. decompose 32 (跟之前一样)
cat > system/decomposeParDict <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
method scotch;
numberOfSubdomains 32;
EOF
decomposePar -force 2>&1 | tail -3

# 5. RUN (这次比之前长 2x, 大约 40-50 分钟)
mpirun --oversubscribe --bind-to none -np 32 laserMeltFoam -parallel > log.run.evap 2>&1
echo "exit code: $?  (134 = signal 6 SIGABRT at finalize is OK, dict double-free quirk)"

# 6. 提取 timing, 按 phase 分段
grep "TIMING_pd" log.run.evap > pd_timings_evap.txt

# 7. 找出每个 step 的时间, 按 t=700ns 分早/晚
grep -E "^Time = |TIMING_pd" log.run.evap > log_with_time.txt
/home/yzk/jax-env/bin/python << 'PYEOF' > evap_summary.txt
import re, statistics as s
early, late = [], []
cur_t = None
with open('log_with_time.txt') as f:
    for line in f:
        m = re.match(r"Time = ([0-9.eE+-]+)", line)
        if m:
            cur_t = float(m.group(1))
            continue
        m = re.match(r"\[TIMING_pd\] ([0-9.]+) ms", line)
        if m and cur_t is not None:
            ms = float(m.group(1))
            (early if cur_t < 7e-7 else late).append(ms)

print("=== EARLY phase (t < 700ns, cold-start + melt) ===")
print(f"  N={len(early)}  median={s.median(early):.2f}ms  mean={s.mean(early):.2f}ms")
print(f"  min={min(early):.2f}  max={max(early):.2f}")
print()
print("=== EVAPORATION phase (t >= 700ns) ===")
print(f"  N={len(late)}  median={s.median(late):.2f}ms  mean={s.mean(late):.2f}ms")
print(f"  min={min(late):.2f}  max={max(late):.2f}")
PYEOF

cat evap_summary.txt

# 8. Push 回 git
mkdir -p ~/DILU-Research/dilu/amgx/bench/lab_deploy/results
cp ~/cases/spot_melt_150W/log.run.evap          ~/DILU-Research/dilu/amgx/bench/lab_deploy/results/of_xeon_evap.log
cp ~/cases/spot_melt_150W/pd_timings_evap.txt   ~/DILU-Research/dilu/amgx/bench/lab_deploy/results/of_xeon_evap_pd_timings.txt
cp ~/cases/spot_melt_150W/evap_summary.txt      ~/DILU-Research/dilu/amgx/bench/lab_deploy/results/of_xeon_evap_summary.txt

cd ~/DILU-Research
git add dilu/amgx/bench/lab_deploy/results/
git commit -m "lab Xeon: OF N=32 wall on evaporation phase (t=700ns-1.2us)"
git push origin main
```

## 期望

- **EARLY phase (t<700ns)**: pd_corr0 median 应该跟之前 49.57ms 基线一致 (同一 case 同一 phase, 重测)
- **EVAPORATION phase (t>=700ns)**: pd_corr0 median 可能略高 (矩阵更病态 -> 更多 iter), 估计 60-100ms

## 平行任务 — 5060 同时跑

```bash
ssh manyxu@5060
cd ~/DILU-Research && git pull origin main

# 直接跑新的 evaporation bench
python3 -u -m dilu.amgx.bench.bench_amgx_evaporation_5060 > /tmp/amgx_5060_evap.log 2>&1 &

# 几分钟后看结果
tail -30 /tmp/amgx_5060_evap.log
```

两边数据回流后, 我会写真正的"3-way 蒸发阶段对比报告" (代替之前用单时刻数据的 DRAFT)。
