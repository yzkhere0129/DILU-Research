# Multi-core OpenFOAM Replica — Engineering Plan

**日期**: 2026-05-06
**目标**: 让我们的 `dilu/openfoam_cpu/` 复刻版从单线程 Python (12s/solve)
推到 **MPI N=32 跟 OpenFOAM 同时段**（49.57ms 实测 baseline）, **保持 byte-exact**。
**代号**: P0–P5（Phase 0 是这份计划本身，Phase 1-5 是实施）。

**不写代码的承诺**: 这份 plan 评审通过前不动 src/。

---

## 0. 决策点 — 哪种"多核"

OpenFOAM 多核 = **MPI domain decomposition**（每 rank 一份本地 lduMatrix
+ halo exchange）。这跟"在单进程内多线程并行 DILU sweep"是不同概念。

### 三条路线对比

| 路线 | 内容 | byte-exact match OF? | 性能上限 | 工作量 |
|------|------|---------------------|---------|--------|
| **A. MPI domain decomp** (= OF 自己用的) | 32 进程, 每进程独立 lduMatrix + halo | ✅ 可以做到 | OF 49ms 的 0.8-1.5× 范围 | **9-13 天** |
| B. OpenMP shared mem | 单进程, 多线程 DILU sweep（重排 cell + 着色）| ❌ 算法变了, 跟 OF 不再 byte-exact | 单进程上限 ~150ms | 5-7 天 |
| C. C++ 单核 + SIMD | 完全单线程, 极致优化 | ✅ | 单核 ~200ms | 5 天 |

**用户已确认: 走 A**（"完全对标 OF"）。但 **A 必须先做 C** —— C++ 单核是 MPI 的脚手架。
所以实际路径是 **C → A**。

### 为什么不能跳过 C 直接 MPI

1. MPI 进程内**仍然是 1 核 DILU sweep**, 每个 rank 跑的就是一份单核 solver。
2. Python+numba 单核太慢 (12s) → 32 ranks 也救不了（节点内通信开销吃掉所有加速）。
3. 必须先**把单核降到 ~200ms**, MPI 32 ranks 才能落到 ~50ms。
4. C++ 是唯一能给 MPI 用的形态 (numba 在 MPI 进程间不容易共享, pybind11 + MPI 是标准做法)。

### 一个反共识但要事先澄清的事实

**OpenFOAM 自己也不是"无条件 byte-exact"**。同一 case 用 N=4 跟 N=8
跑出的 x **在机器精度上不一样**。原因：MPI_Allreduce 的归约顺序随
rank topology 变, 浮点结合律不成立。

所以"byte-exact match OF" 必须**特指**:
- **同 mesh decomposition**（用 scotch/METIS 同套切法）
- **同 N**（我们目标 N=32, 跟 lab Xeon scaling test 一致）
- **同 MPI 实现**（OpenMPI / Intel MPI 不同版本可能给不同 reduce 顺序）

ULP 级 drift 不可避免, target 是 **rel(x, x_OF) ≤ 1e-12**, 不是 ULP=0
（除非愿意复刻 OF 完整 reduce tree, 工作量 +5 天）。

---

## 1. Phase 1 — C++ Single-Core Port (5 天)

### 1.1 目标
- **byte-exact**: T 矩阵 ULP=0 vs 现有 Python+numba 复刻
- **性能**: 单核 ≤ 200ms per 512K pd solve（OF 单核 567ms）
- 接口: pybind11 模块 `dilu.openfoam_cpu.cpp_kernels`, 跟现 Python 接口同形

### 1.2 文件清单
```
dilu/openfoam_cpu/cpp/
├── CMakeLists.txt
├── include/
│   ├── ldu.hpp           # LDU struct, plain old data
│   ├── kernels.hpp       # amul/sumA/calcReciprocalD/precondition signatures
│   ├── pbicg.hpp         # solver state machine
│   └── reductions.hpp    # left-fold sum_abs / dot
├── src/
│   ├── kernels.cpp       # face-loop kernels with AVX2 intrinsics
│   ├── pbicg.cpp         # outer iteration
│   └── reductions.cpp
└── bindings/
    └── pybind_module.cpp # exposes solve(A, b, x0, tol) → x, iter
```

### 1.3 性能预算（单核 512K pd 矩阵）

| Kernel | Python+numba | 预期 C++ | 加速 |
|--------|-------------|---------|------|
| amul (per call) | ~50ms | ~5ms | 10× |
| precondition (forward+backward) | ~80ms | ~8ms | 10× |
| calc_reciprocal_d (once) | ~50ms | ~5ms | 10× |
| reductions (sum_abs, dot) | ~10ms | ~2ms | 5× |
| **per-iter wall** | **~150ms** | **~15ms** | 10× |
| **iter count** | 195 | 195 (same algo) | — |
| **total per solve** | **12s** | **~200ms** | **60×** |

OF 单核 567ms ≈ 我们 200ms 的 2.8 倍 → **C++ 版应该能比 OF 单核还快一点**
（因为 OF C++ 不是极致优化, 我们能 push SIMD）。

### 1.4 验证门
- `tests/test_cpp_byte_exact.py`: 对 12 LPBF T cases ULP=0 vs Python+numba 复刻
- `tests/test_cpp_perf.py`: 21 senior pd 单核 wall time, median ≤ 250ms 才 pass
- 跑学长 21 矩阵, residual ≤ 现有 Python 残差

### 1.5 风险
- **R1.1**: AVX2 vs AVX-512 不同 round-off → 跨机器 ULP drift。**缓解**: 强制 AVX2 only, `-mavx2 -mno-fma`。
- **R1.2**: pybind11 与 numpy zero-copy 失败 → 隐式 memcpy. **缓解**: 用 `pybind11::array_t<double, py::array::c_style | py::array::forcecast>`。
- **R1.3**: scatter add (`np.add.at` 等价物) 在 C++ 没现成。**缓解**: 手写 sequential face loop（DILU 本来就 sequential, no harm）。

### 1.6 完成标志
- C++ 模块 build 成功
- 12 LPBF T cases ULP=0 byte-exact 测试 pass
- 21 senior pd cases 单核 ≤ 250ms median wall

---

## 2. Phase 2 — PCG+DIC for pd byte-exact (1-2 天)

### 2.1 为啥单独一个 phase
**OF pd 用 PCG+DIC, 我们现在用 PBiCG+DILU**。两个算法对**对称矩阵**收敛
到同一个 x（数学上一致）, 但 **iter 轨迹不同, 浮点 round-off 不同**, 所以
我们 pd 解跟 OF pd 解不是 byte-exact。

T 矩阵是 byte-exact 的（因为 OF T 也用 PBiCG+DILU）, pd 不行。

### 2.2 实施
- DIC = DILU 的对称版本。代码上 DILU 已经处理对称矩阵正确（lower==upper 时 calc_rD 退化为 DIC）。**主要工作是写 PCG outer**（PBiCG 去掉 transpose 一半即可, ~50 行）。

### 2.3 验证门
- `tests/test_pcg_pd_byte_exact.py`: 对 21 senior pd cases iter count + ULP=0 vs OF native xref
  - **注意**: 学长 xref 是 OF tol=1e-8, 直接比会有 2.5% 截断噪声
  - 缓解: 让学长在 lab 重跑 OF tol=1e-12, 拿新 xref. 然后 PCG+DIC 对 new_xref 应该 byte-close (~1e-12)

### 2.4 风险
- **R2.1**: setReference cell 选择 OF 跟我们不一致 → null space 不同。**缓解**: 复制 OF 的 setReference 逻辑（取第一个 patch 的某个 cell），写一个 dump util 验证一致。

---

## 3. Phase 3 — MPI Domain Decomposition + Halo Exchange (3 天)

### 3.1 目标
单进程 N=1 → MPI N=32 在 lab Xeon。byte-exact 等于：
- **mesh 切法跟 OF decomposePar 一样** (scotch method, same seed)
- **每 rank 本地 lduMatrix 跟 OF processorN/ 输出一致**
- **halo exchange 跟 OF processor patch 通信顺序一致**

### 3.2 实施

#### 3.2.1 直接读 OF decomposed output 而非自己 decompose
最省事。我们做：
```bash
# OF 已经 decompose 好的 case
ls case/processor0/{0/T, constant/polyMesh/owner, ...}
ls case/processor0/{0/T, constant/polyMesh/processorPatches/...}
```
我们写 `read_decomposed_case(case_dir, rank)` → 拿 local lduMatrix +
processorPatch list (每个 patch 知道 neighbour rank + face indices)。

这样我们**完全跳过自己 decompose**, 直接用 OF 切好的, byte-exact 保证。

#### 3.2.2 Halo exchange API
每次 amul 前后:
```cpp
void halo_exchange(MPI_Comm comm, ...) {
  for (each processor patch p with neighbour rank q):
    pack send buffer[p] = psi[patch_addr_p]
    MPI_Isend(send[p], ..., q, tag, comm, &req_send[p])
    MPI_Irecv(recv[p], ..., q, tag, comm, &req_recv[p])
  // 等所有 recv 完
  MPI_Waitall(req_recv)
  // 把 recv buffer 加到 Apsi 上 (跟 internalCoeffs 类似)
  for (each patch): Apsi[patch_cell] += interface_coeff[p] * recv[p]
}
```

跟 OF 的 `lduMatrix::initMatrixInterfaces` + `updateMatrixInterfaces` 一样。

### 3.3 验证门
- `tests/test_mpi_amul_match_serial.py`: 1 个 32-rank run vs 1 个 1-rank run, 同一矩阵, max diff ≤ ULP×N (浮点 reduce 漂)
- `tests/test_mpi_iter_match_OF.py`: 1 step OF run + 1 step 我们 run, iter 数完全一致, x diff ≤ 1e-13

### 3.4 风险
- **R3.1**: OF 的 processor patch 顺序对 reduce 结果有影响 → ULP drift > target. **缓解**: 接受 1e-12 不是 ULP=0。
- **R3.2**: MPI buffer pack/unpack 顺序错 → wrong result.  **缓解**: 写小 unit test 用 4 ranks 玩具矩阵 sanity。
- **R3.3**: `lduInterface boundaryCoeffs` 的 fold 我们之前已经 audit 清楚（PBiCG-DILU audit §10.6）, 但 MPI 版要再 verify processorPatch 有没有特殊行为。

---

## 4. Phase 4 — MPI Reductions (1 天)

### 4.1 内容
现在的 `reductions.py` (sum_abs, dot) 是单进程 left-fold。MPI 版要：
```cpp
double gSumMag(const double* x, int n_local, MPI_Comm comm) {
  double local = 0;
  for (i=0; i<n_local; i++) local += std::abs(x[i]);
  double global;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global;
}
```
对应 OF 的 `gSumMag` / `gSumProd`。

### 4.2 byte-exact?
**MPI_Allreduce 的归约顺序由 MPI 实现决定**, **跟 OF 不一定一致**。
如果 OF 用 OpenMPI 的 ring-reduce, 我们用 MPI 的 tree-reduce, **结果会差 ULP 量级**。

### 4.3 缓解 (3 种, 由轻到重)
1. **接受 1e-12 drift** (推荐) — 成本 0
2. 写自己的 reduction tree 强制跟 OF 一样的二叉树 — 成本 1 天, 收益 微小
3. 单进程 gather + 复算 reduction — 成本低但破坏 scalability

### 4.4 验证门
- 21 senior pd matrices 在 N=32 rank 跑, x_ours vs x_OF_native max ≤ 1e-12

---

## 5. Phase 5 — End-to-End on Lab Xeon (2 天)

### 5.1 验证矩阵
- **Mesh**: spot_melt 500K (500K cells, 跟之前 scaling test 一致)
- **Hardware**: HR54WV2 28 phys + 4 SMT, mpirun -np 32
- **Compare**: 我们 replica 32 ranks vs OF 32 ranks 跑同 case

### 5.2 接受标准
| 指标 | 目标 | 失败时 |
|------|------|--------|
| iter count per pd_corr0 | == OF | 算法 mismatch (PCG vs PBiCG?), 检查 P2 |
| ‖x_ours - x_OF‖∞ / ‖x_OF‖∞ | ≤ 1e-12 | reduce 顺序不同 (P4 R3.1), 退化到 1e-10 接受 |
| pd_corr0 wall time median | ≤ 70ms (1.5× of OF 49ms) | 通信开销过大, profile 看 halo% |
| 收敛行为 | converged 全部 | DILU 跨 rank 处理 bug |

### 5.3 失败回滚
如果 P5 不达标, 不 land main, 留 r25/multicore-replica 分支, 写诊断报告。

---

## 6. 总时间预算

| Phase | 内容 | 时间 |
|-------|------|------|
| P0 | 这份 plan | 0.5 天（已完成） |
| P1 | C++ 单核 port | 5 天 |
| P2 | PCG+DIC for pd | 1-2 天 |
| P3 | MPI decomposition + halo | 3 天 |
| P4 | MPI reductions | 1 天 |
| P5 | End-to-end on lab Xeon | 2 天 |
| **总计** | | **12-13 天** |

**关键路径**: P1 → P3 → P4 → P5 (10 天). P2 可以跟 P3 并行做.

---

## 7. 决策门

每 phase 做完, 写一份小报告 + 数据表, **回这里**确认是否进下一 phase:
- P1 完成: 单核 ≤ 200ms + T byte-exact → 进 P2 + P3
- P3 完成: 32 ranks vs 单核 byte-close → 进 P4
- P5 完成: vs OF native within target → land

任何 phase 失败：
- 不强行硬推. 写诊断, 回这里讨论是否减 scope (比如停在 P2 单核但 byte-exact, 不做 MPI)

---

## 8. 立即下一步 (今晚开工)

P1 第 1 步：建 C++ 骨架 + 写 amul / precondition 两个核心 kernel + pybind11 binding + 1 个 byte-exact unit test。

具体子任务：
1. **P1.A** (1.5h): `cpp/CMakeLists.txt` + `cpp/include/ldu.hpp` + `cpp/include/kernels.hpp`
2. **P1.B** (3h): `cpp/src/kernels.cpp` 含 amul, sumA, calc_reciprocal_d, precondition, precondition_t
3. **P1.C** (1h): `cpp/bindings/pybind_module.cpp` 暴露 `cpp_amul`, `cpp_calc_rd`, `cpp_precondition`
4. **P1.D** (0.5h): `tests/test_cpp_kernels_byte_exact.py` 单 kernel 比对

第一晚目标: P1.A + P1.B + P1.C, 跑 amul vs Python ULP=0。

---

## 9. 不做的事

- 不重写 LBM (这是另一个项目)
- 不复刻 OF 的 fvSchemes / fvOptions 调度（我们只复刻**线性求解器**, 不复刻整个 PISO loop）
- 不做 GPU offload（那是 cuSPARSE / AMGx 的活）
- 不写 OpenMP shared mem 版（路线 B, 已经在 §0 排除）
- 不实现 GAMG（OpenFOAM 的 algebraic multigrid, 复杂度等于半个 AMGx）

---

## 10. 风险登记表

| ID | 风险 | 概率 | 影响 | 缓解 |
|----|------|-----|------|------|
| R1.1 | AVX2 vs AVX-512 跨机器 round-off | 中 | 中 | 强制 AVX2 only |
| R1.2 | pybind11 zero-copy 失败 | 低 | 低 | force_cast |
| R2.1 | setReference cell 跟 OF 不一致 | 中 | 高 (pd byte-exact 不达标) | dump util 验证 |
| R3.1 | MPI reduce 顺序不同 | **高** | 中 (1e-12 drift) | 接受 |
| R3.2 | halo pack/unpack bug | 中 | 高 | 4-rank 玩具测试 |
| R4.x | (continue numbering as needed) | | | |
| R5.1 | 上 lab Xeon 后通讯开销过大 | 中 | 中 | profile + tune |
| R大 | 总工期 13 天 → 实际 25 天 | 中 | 中 | 每 phase 决策门 |
