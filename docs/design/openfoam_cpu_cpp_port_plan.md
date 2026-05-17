# 复刻版 C++ Port + MPI 并行计划（A+B 路径）

**目标**: 让 `dilu/openfoam_cpu/` 在 lab Xeon 32 核上跑出 ≤50ms / pd_solve（与 OpenFOAM PCG-DIC 同水平），同时**保持 byte-exact match OpenFOAM 解**。

**当前**: Python+numba 单线程 11.7s/pd_solve（235× 慢于 OF）。

**预算**: 1-2 周认真做。

---

## 目录架构（拟）

```
dilu/openfoam_cpu/
├── python/                    # 现有 (fidelity reference, 不动)
│   ├── ldu.py
│   ├── dilu.py
│   ├── pbicg.py
│   ├── pcg.py                 # 新：PCG outer (pd 用 PCG+DIC)
│   ├── dic.py                 # 新：DIC preconditioner (对称版 DILU)
│   └── ...
├── cpp/                       # 新：C++ 实现
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── ldu.hpp            # struct LDU, addressing
│   │   ├── kernels.hpp        # amul/tmul/dilu/dic/pcg face-loop
│   │   ├── reductions.hpp     # gSumProd / gSumMag (left-fold)
│   │   ├── normfactor.hpp     # OF normFactor formula
│   │   ├── pbicg.hpp / pcg.hpp
│   │   └── parallel.hpp       # MPI domain decompose + halo exchange
│   ├── src/
│   │   ├── kernels.cpp        # SIMD-vectorized face loops
│   │   ├── pbicg.cpp / pcg.cpp
│   │   ├── normfactor.cpp
│   │   └── mpi_layer.cpp      # halo exchange, processor patches
│   ├── bindings/              # pybind11 → Python interop (regression tests)
│   │   └── pybind_module.cpp
│   ├── tests/                 # GoogleTest
│   │   ├── test_kernels.cpp   # vs Python byte-exact
│   │   ├── test_pcg_pbicg.cpp
│   │   └── test_mpi_halo.cpp  # MPI parallel correctness (mpirun -np 4)
│   └── bench/
│       ├── bench_pcg.cpp      # standalone solver wall time
│       └── bench_pbicg.cpp
└── docs/                      # 现有
```

---

## Phase A: 单核 C++ port (3-5 天)

**目标**: 单核 C++ 跑 1 矩阵 ≤ 200ms（vs OF 单核 567ms = 2-3× 快）；byte-exact 一致 Python 版。

### A1. 数据类型 + LDU 结构 (0.5 天)

```cpp
struct LDU {
    int n_cells;
    int n_faces;
    std::vector<double> diag;       // [n_cells]
    std::vector<double> lower;      // [n_faces]
    std::vector<double> upper;      // [n_faces]
    std::vector<int>    owner;      // [n_faces]
    std::vector<int>    neighbour;  // [n_faces]
    std::vector<int>    losort;     // [n_faces], lazy
};
```

读 npz 接口（pybind11 互操作）→ 验证跟 Python `csr_to_ldu` 输出 byte-exact。

### A2. SpMV + sumA (0.5 天)

```cpp
// face-loop unrolled, AVX2 tight
inline void amul(const LDU& m, const double* psi, double* Apsi) {
    const int N = m.n_cells, F = m.n_faces;
    for (int i = 0; i < N; ++i) Apsi[i] = m.diag[i] * psi[i];
    for (int f = 0; f < F; ++f) {
        Apsi[m.neighbour[f]] += m.lower[f] * psi[m.owner[f]];
        Apsi[m.owner[f]]     += m.upper[f] * psi[m.neighbour[f]];
    }
}
```

测试：跟 Python `amul()` byte-exact, ULP=0 on 21 senior matrices.

### A3. DILU + DIC preconditioner (1 天)

DILU (asymmetric) + DIC (symmetric):
```cpp
// DIC for pd (symmetric, OF default for pressure)
void calc_reciprocal_d_dic(const LDU& m, double* rD) {
    std::copy(m.diag.begin(), m.diag.end(), rD);
    for (int f = 0; f < m.n_faces; ++f) {
        rD[m.neighbour[f]] -= m.upper[f] * m.upper[f] / rD[m.owner[f]];
        // DIC = DILU when matrix symmetric: lower==upper, so upper*lower == upper^2
    }
    for (int i = 0; i < m.n_cells; ++i) rD[i] = 1.0 / rD[i];
}
```

测试：DIC byte-exact match OpenFOAM `DICPreconditioner.C` （新读 OF 源码 `DICPreconditioner.C`，复刻 calcReciprocalD + precondition）。

### A4. PCG outer + normFactor (1 天)

复刻 `PCG.C` 主循环（已有 PBiCG 模板，PCG 是 PBiCG 在 SPD 上简化版，移除 transpose 操作）。

normFactor 复刻已经在 Python 里完整做过 (audit §10.3-§10.4)，C++ 直接搬。

测试：21 senior pd 矩阵跑 PCG-DIC，**iter 数 + final residual 跟 OpenFOAM byte-exact**（之前 Python PBiCG-DILU 在 T 上 byte-exact 验证过技术，PCG-DIC 同思路）。

### A5. 性能调优 (0.5-1 天)

- AVX2/AVX-512 intrinsics for face loops
- Cache-friendly cell ordering (Cuthill-McKee?) — **但破坏 byte-exact**, 跳过
- Reduce malloc churn (preallocate workspaces)

**目标**: lab Xeon 单核 ≤ 200ms / pd solve。

### A6. pybind11 bindings (0.5 天)

```python
from dilu.openfoam_cpu.cpp import LDU, PCGSolver
ldu = LDU.from_csr(A)
solver = PCGSolver(ldu, tol=1e-8, max_iter=200)
x, perf = solver.solve(b, x0)
```

跑现有 pytest 全套（13 tests），全部仍 byte-exact。

**Phase A 交付**: lab Xeon 单核 ≤ 200ms，比 OF 单核 (~567ms) **快 2-3×**（因为我们更紧凑实现，无 OF 的多余 boundary 处理）。

---

## Phase B: MPI 并行 (3-5 天)

**目标**: lab Xeon 32 核 ≤ 50ms（match OF）, byte-exact match OF 32-rank 输出。

### B1. Mesh decomposition 输入接口 (0.5 天)

OpenFOAM 用 `decomposePar` 切 mesh, 输出每个 rank 的 (local cells, processor patches, owner/neighbour 局部 indexing)。

我们的接口：直接读 OpenFOAM 切好的输出（`processorN/constant/polyMesh/`），**不 reimplement decomposition**（那是 metis/scotch 的事）。

每个 rank 加载：
- local LDU
- processor boundary patches: 哪些 face 是跨 rank, 对面的 rank id, face 列表
- 全局 cell-to-rank map (for sparse data exchange only)

### B2. Halo exchange 层 (1 天)

```cpp
class ProcInterfaceList {
    void initExchange(const double* psi);      // post non-blocking sends
    void updateAfterAmul(double* Apsi);         // wait + apply contributions
    // 对应 OF 的 lduMatrix::initMatrixInterfaces / updateMatrixInterfaces
};
```

测试：mpirun -np 4 跑学长 21 矩阵，每个 rank Apsi byte-exact match 单核版.

### B3. PCG outer 加 MPI reduce (0.5 天)

```cpp
double dot_global(const double* a, const double* b, int n_local) {
    double local = 0.0;
    for (int i = 0; i < n_local; ++i) local += a[i] * b[i];
    double global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;  // OF 朴素左折叠 → 用 MPI_SUM 默认（实现可能用树状 reduce, 但 OF 也是这样）
}
```

注意：MPI_SUM 在不同 rank 顺序下结果会变 (浮点非结合)。OF 也接受这一点 (rank 不变结果就一致)。

### B4. PCG normFactor 全局聚合 (0.5 天)

normFactor 涉及 gSum, sumA — 都要 MPI_Allreduce.

### B5. 端到端测试 (1 天)

mpirun -np 32 跑 spot_melt 80³ case → wall time + iter 跟 OpenFOAM N=32 比较.

**目标**: ≤ 50 ms / pd solve, byte-exact match OF 32-rank 输出 (修一个 mpirun 命令而已).

### B6. 集成到 OpenFOAM 编译链（可选, 0.5 天）

写一个 fvSolution 选项让 OF 用我们的 solver 而不是它自己的 PCG。这样**用户在 OF case 里选 solver 就能用**, 不用改 case structure.

---

## 风险 + Reality Check

| 风险 | 概率 | 缓解 |
|------|------|------|
| C++ port byte-exact 出 bug 难调 | 高 | 每 kernel 写 pybind 互操作测试, ULP=0 才 merge |
| MPI halo exchange 跟 OF 不一致 | 中 | 直接读 OF processorN 输出, 跟 OF 同 mesh decompose |
| 浮点 MPI_Allreduce 顺序破坏 byte-exact | 中 | 跟 OF 一致接受 rank-dependent 结果 |
| 单核优化达不到 OF 单核速度 | 低 | OF 算法本身是 face-loop, 我们做同样的，C++ -O3 应不弱于 OF |
| 整个项目超 2 周 | 中 | Phase A 单独可交付 (~5 天), 单核优化版本就比 Python 快 50× |

**最差情况**: Phase A 完成（5 天，单核 200ms）。Phase B 卡 MPI 集成。**仍然有可交付**: "C++ 版本单核 200ms vs Python 版 11.7s, 加速 50×, 跟 OF 单核 567ms 还快 2-3×"。

---

## 第一周里程碑（如果今晚开始）

| Day | 内容 | 交付 |
|-----|------|------|
| 1 | Phase A1 + A2 (LDU struct + SpMV) | C++ amul byte-exact match Python on senior 21 |
| 2 | Phase A3 (DIC preconditioner) | calcReciprocalD byte-exact OF |
| 3 | Phase A4 (PCG outer) | 21 senior pd 跑通 + iter match OF |
| 4 | Phase A5 (单核优化) + A6 (pybind) | 单核 ≤ 200ms, 13 pytest 全过 |
| 5 | Phase B1 + B2 (MPI halo exchange) | mpirun -np 4 byte-exact |
| 6-7 | Phase B3 + B4 + B5 (MPI PCG + 全集成) | mpirun -np 32 ≤ 50ms |

---

## 现在做的第一步

写 Phase A1 + A2 的代码：
- `dilu/openfoam_cpu/cpp/CMakeLists.txt`
- `dilu/openfoam_cpu/cpp/include/ldu.hpp`
- `dilu/openfoam_cpu/cpp/src/kernels.cpp`
- `dilu/openfoam_cpu/cpp/bindings/pybind_module.cpp`
- pytest 在现有 13 tests 基础上加 `test_cpp_kernels.py` (call C++ via pybind, compare to Python)

**等 tol sweep 跑完 → 决定 Phase A 立即开工 vs 先看 sweep 结果再决定**。

---

## 关键决策点（等 user 决定）

Phase A 立即开工**不可逆**（写代码会很多）。先等 tol sweep 数据看 AMGx 是否能压到合理速度（如 1.5-2× of OF）。如果 AMGx 那条路能拿到“比 OF 快或同速 + 高精度”，C++ port 就**只为 fidelity 标杆做**, 不为速度做（缩小 scope 到 Phase A 单核, ~5 天）。

如果 AMGx 即便 tol=1e-4 也不够快, **C++ port 是唯一速度路径**, 那就 Phase A+B 完整做。
