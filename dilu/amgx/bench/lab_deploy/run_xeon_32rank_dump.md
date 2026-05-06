# Lab Xeon — 32 ranks OF run + matrix dump (melting + evaporation 阶段)

**目标**：在 lab Xeon (HR54WV2, 56 核) 上用 **32 MPI ranks** 跑我们自己的 LPBF OF case，
覆盖 **melting (320-380 ns) + evaporation (700-1100 ns)** 两个物理阶段，
用我们带 `addBoundaryDiag` 的 matrixDumper dump (A, b, x) 矩阵。

**为啥不用学长 dump**：他那批 dump 漏了 boundary patch 项 (`‖A·x_xref - b‖/‖b‖ = 18`)，
我们自己 dump 已验证自洽 (`= 1.66e-8`)。

---

## 步骤

### 0. 验证 lab Xeon 上 OF + matrixDumper 状态

```bash
ssh manyxu@HR54WV2
which laserMeltFoam
# 应该指向 ~/OpenFOAM/yzk-v2506/.../bin/laserMeltFoam

# matrixDumper 是 include 进 laserMeltFoam 源码的 (不是单独 .so),
# 所以确认 binary 里有它的 symbol:
strings $(which laserMeltFoam) | grep -i matrixDumper | head -3
# 期望看到 "matrixDumperDict" / "matrixDumper" 这种字串
# 如果什么都没看到, 说明这个 laserMeltFoam 不是从 LaserbeamFoam 仓库
# 编出来的, 需要重编:
#   cd ~/LaserbeamFoam && ./Allwmake -j32

ls -d ~/LaserbeamFoam/tutorials/laserMeltFoam/* 2>&1
# 看有没有 LPBF_crosscheck / LPBF_sanity / spot_melt
```

### 1. 准备 case：用 LPBF_crosscheck 改 endTime + writeInterval

```bash
cd ~/LaserbeamFoam/tutorials/laserMeltFoam
# 拷一份新的不污染原 case
cp -r LPBF_crosscheck LPBF_lab32_dump
cd LPBF_lab32_dump

# 清旧时间步
foamCleanCase
rm -rf processor* postProcessing
```

### 2. 改 controlDict + matrixDumperDict

**controlDict** — 端到端跑到 1.2 μs:

```bash
cat > system/controlDict <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application       laserMeltFoam;
startFrom         startTime;
startTime         0;
stopAt            endTime;
endTime           1.2e-6;        // 覆盖 melting + evaporation 两阶段
deltaT            5e-10;          // 0.5 ns step
writeControl      adjustableRunTime;
writeInterval     5e-8;          // 每 50 ns 写一次场量
purgeWrite        0;
writeFormat       binary;
writePrecision    8;
writeCompression  off;
timeFormat        general;
timePrecision     6;
runTimeModifiable yes;
EOF
```

**matrixDumperDict** — 6 个目标 timestep, 横跨 melting + evaporation:

```bash
# step index = time / deltaT = time / 5e-10
# 我们要的:
#   melting:    320 ns (step 640), 380 ns (step 760), 410 ns (step 820)
#   evaporation: 700 ns (step 1400), 900 ns (step 1800), 1060 ns (step 2120)
cat > system/matrixDumperDict <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary;
           location "system"; object matrixDumperDict; }

enabled        true;
binaryMM       false;
outputDir      "postProcessing/matrices";

// 6 个 timestep: 3 melting + 3 evaporation
dumpTimeSteps  (640 760 820 1400 1800 2120);

// dump 哪几个方程
equations      (pd T);

// pd 有 3 个 corrector 都 dump; T 只 dump 第一个 corrector (矩阵结构同)
maxCorrectorsPerEq
{
    pd  3;
    T   1;
}
EOF
```

预计磁盘: 6 timestep × 4 矩阵 (T + 3 pd) × ~670 MB = **~16 GB** 总量, 跨 32 个 processor 副本各占一份 (~50 MB / matrix / processor).

### 3. decomposeParDict — 32 ranks

```bash
cat > system/decomposeParDict <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains  32;
method              scotch;
EOF
```

### 4. 跑

```bash
# 域分解
decomposePar -force 2>&1 | tail -10

# 起跑 (32 ranks, 估计 8-30 分钟 wall, 取决于 timestep 复杂度)
nohup mpirun --oversubscribe --bind-to none -np 32 \
    laserMeltFoam -parallel \
    > log.run 2>&1 &
echo "PID: $!"
disown   # 防 ssh 断了 SIGHUP 杀进程
tail -f log.run | grep -E "Time = |TIMING|matrixDumper|Final|deltaT"
# 看到 "matrixDumper: dumping ..." 出现 6 次说明 6 个 dump 都触发了
```

预计 **8-30 分钟** wall。每 timestep 大约 200-400 ms (32 ranks 上), 2400 个 step 总共。

### 5. 跑完处理

```bash
# 重新组装 (matrices 已经在每个 processor*/postProcessing/matrices/ 里)
# matrixDumper 的输出本来就是 per-processor 的, reconstructPar 不动它
ls processor0/postProcessing/matrices/ | head

# 我们要的是 GLOBAL 矩阵 — 后处理脚本会把 32 份 processor*/matrices/<t>/<eq>/
# 重组成单个 (A, b, x) per timestep per equation
# 这步留给 dev 这边写, 不用 lab xeon 干
```

### 6. 把 dump 传回去（或者发回 git LFS）

```bash
# 估算: 32 timestep × 4 eq × 670 MB ≈ 86 GB
# 不上 git! 直接 scp 或者只传子集
du -sh ~/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_lab32_dump/processor*/postProcessing/

# 选 6 个代表 timestep (3 melting + 3 evap) 打包传回 dev:
mkdir -p /tmp/lab_dump_subset
for t in 3.20e-07 3.50e-07 3.80e-07 7.00e-07 9.00e-07 1.06e-06; do
    cp -r ~/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_lab32_dump/processor*/postProcessing/matrices/$t \
          /tmp/lab_dump_subset/processor_$t/  # 简化版,实际要循环每个 processor
done

# 总计 6 timestep × 4 eq × 670 MB ≈ 16 GB, scp 或者 rsync 到 dev
rsync -avz /tmp/lab_dump_subset/ yzk@<dev_ip>:/home/yzk/lab_lpbf_dump/
```

---

## 替代方案 (如果上面太复杂)

**用现有的 spot_melt_150W case** (lab xeon 上已经跑过 32 ranks)：

```bash
ssh manyxu@HR54WV2
ls -d ~/cases/spot_melt_150W 2>&1   # 之前 N=32 49.57 ms baseline 的那个
```

如果还在，把它改成跑到 1.2 μs + 加 matrixDumper functionObject，省去重新部署 case 的麻烦。

---

## 跟我目前 dev 工作的衔接

跑完之后我这边：
1. 写一个 reconstruct 工具：从 32 个 processor*/postProcessing/matrices/<t>/<eq>/ 拼回 GLOBAL (A, b, x)
2. 直接验证：`‖A·x_final - b‖/‖b‖` 应该是 OF tol 量级 (1e-8)，证明我们 dump 是干净的
3. 跑 LU (大概 1M cell 范围内 SuperLU 能搞定) 或者紧 PCG 当 truth
4. 出 melting + evaporation 两阶段的 3D 物理量图 + 误差图

预期对比图最终长这样：
- "我们自己 dump 的 melting 期 ‖A·x - b‖ = 1e-8 ✓"
- "我们自己 dump 的 evap 期 ‖A·x - b‖ = 1e-8 ✓"
- "学长那批同时间段的 dump ‖A·x - b‖ = 18 ✗"

板上钉钉的对比。

---

## 快速决策路径

要不要先**只跑一个简化版**：
- 用现有 spot_melt_150W (mesh 已知, decomposeParDict 已配)
- 32 ranks
- endTime 跑到 1.2 μs
- matrixDumper 每 25 ns dump 一次（粗一点）
- **大概 4-6 小时跑完**，磁盘 ~50 GB

跑出来直接给我们 melting + evap 两段的干净 dump，所有问题一锅端解决。

要这么做的话告诉我 `~/cases/spot_melt_150W` 是否还在，以及 `matrixDumper` 是否 build 进去了。
