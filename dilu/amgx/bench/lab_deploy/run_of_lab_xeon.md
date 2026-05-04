# Lab 部署 #1：在 56 核 Xeon (HR54WV2) 上跑 OpenFOAM PCG-DIC on senior 矩阵尺寸

## 目标

**实测**OpenFOAM PCG-DIC 在 80³=512K cells (学长 mesh 尺寸) 上 N=32 的 pd solve wall time。

不是用 spot_melt 旧数据外推，**而是真在学长矩阵尺寸上跑**。

## 为什么不直接喂学长 CSV 给 OF

OpenFOAM 没有"读外部矩阵 → 解"的标准接口。要写个 C++ utility 反向 load 矩阵，工作量 4-8h。**捷径**：用 LaserbeamFoam 的 spot_melt case，把 mesh 改成 80×80×80（跟学长一致），跑前 30 步，就拿到了"同 mesh、同算法、同机器"的 OF wall time —— 学长矩阵也是这么生成的，所以是 apples-to-apples。

## 步骤

```bash
# 在 lab 机
cd ~/cases
cp -r /home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/spot_melt_150W ./spot_melt_80cube
cd spot_melt_80cube

# 1. 改 blockMeshDict 让 mesh = 80x80x80 = 512K（跟学长一致）
nano system/blockMeshDict
# 找到 hex (...) (X Y Z) 那行，把 X Y Z 都改成 80
# 保存

# 2. 改 endTime 让仿真跑 30 步左右
sed -i 's/^endTime.*/endTime  6e-7;/' system/controlDict
# (5e-8 timestep × 30 steps = 1.5e-6, 留 buffer 6e-7 跑到稳态)

# 3. 改 fvSolution 加 log 2 让每步打印 normFactor + iter（含时间）
sed -i '/    pd$/,/^    }/ s/relTol.*0;/relTol  0; log 2;/' system/fvSolution

# 4. mesh
blockMesh
setSolidFraction

# 5. decompose 到 32 (per scaling test 最优)
cat > system/decomposeParDict <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
method scotch;
numberOfSubdomains 32;
EOF
decomposePar -force

# 6. 跑（带 oversubscribe 因为 32 > 28 物理核）
mpirun --oversubscribe --bind-to none -np 32 laserMeltFoam -parallel > log.run 2>&1

# 7. 提取 pd 求解时间
grep -E "DICPCG.*pd|TIMING_pd" log.run > pd_timings.txt

# 取 median wall time per pd_corr0 solve
# (TIMING_pd 行由 patched solver 打出)
awk '/TIMING_pd/ {print $3}' pd_timings.txt | sort -n | \
    awk '{a[NR]=$1} END {print "median pd wall (ms):", a[int(NR/2)]}'

# 把这个数字传给我
```

## 预期

- 30 步 × 4 corrector = 120 个 pd solve
- N=32 在 spot_melt 500K 之前测过 44ms，80³=512K 应该 **45-50ms** 左右

## 拷贝结果回本机

```bash
scp -P <port> ~/cases/spot_melt_80cube/log.run user@yourdesktop:/home/yzk/DILU-Research/dilu/amgx/bench/lab_deploy/of_lab_run_80cube.log
```

或者直接把 `pd_timings.txt` 传回。
