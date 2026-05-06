# OpenFOAM patches for DILU-Research

我们对 LaserbeamFoam 上游的改动都放这里，**不污染 LaserbeamFoam 公用仓库**。

## `laserMeltFoam/matrixDumper.H`

这个文件是 DILU-Research **新加**的（不是改上游），dump 求解器实际看到的
`(A, b, x_initial, x_final)` 用于 cross-check。

历史版本只支持 serial 模式 —— 第 404 行明确写：
```cpp
if (ptf.coupled())
{
    // Serial-only guard. For parallel runs extend this block.
    continue;
}
```

**MPI 模式下** (decomposePar + mpirun) 会跳过 processor patches (= rank 间的 halo
"边界")，导致 dump 的 b 漏掉 inter-rank 贡献，`‖A·x_final - b‖/‖b‖` 变成 0.5 量级
而不是预期的 1e-8。

**当前 patched 版本** 增加了对 coupled patches (= processor patches) 的处理：
对每个 coupled patch, 通过 `ptf.patchNeighbourField()` 取邻居 rank 的 ψ，
然后跟 boundaryCoeffs 相乘加进 source。这样 MPI dump 也能跟 OF 实际求解的方程
byte-exact 对应。

## 部署到 lab Xeon

```bash
ssh manyxu@HR54WV2
cd ~/DILU-Research && git pull origin main

# 把 patched 文件拷到 LaserbeamFoam 源码树
cp dilu/openfoam_patches/laserMeltFoam/matrixDumper.H \
   ~/LaserbeamFoam/applications/solvers/laserMeltFoam/matrixDumper.H

# 重编 (5 min)
cd ~/LaserbeamFoam/applications/solvers/laserMeltFoam
wmake

# 验证 binary 还有 matrixDumper symbols
strings $(which laserMeltFoam) | grep -i matrixDumper | head -3
```

之后回 LPBF_lab32_dump 重跑同一个 case 即可，dump 应该这次干净 (~1e-8)。
