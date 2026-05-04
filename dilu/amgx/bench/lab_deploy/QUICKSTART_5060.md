# Lab 5060 全自动 bootstrap（不需要 OpenFOAM, 不需要 root）

适用：clean lab GPU 机器，假设有 nvidia driver + CUDA 已装（最低要求）。

## 1. ssh 到 5060, clone 仓库

```bash
# (如果是新机器)
git clone git@github.com:yzkhere0129/DILU-Research.git ~/DILU-Research

# (如果已 clone)
cd ~/DILU-Research && git pull
```

## 2. 看现状

```bash
bash ~/DILU-Research/dilu/amgx/bench/lab_deploy/bootstrap_lab_5060.sh check
```

输出会告诉你：
- GPU 是否能 `nvidia-smi` 检测到
- CUDA toolkit 在哪 (期望 nvcc 可用)
- Python / jax-env 状态
- AMGx 是否已 build
- DILU FFI shim 是否已 build

把 check 输出贴给我（如果有 ❌），我看哪里需要 unblock。

## 3. 一键 install + build + run

如果 check 全过 (CUDA + nvcc 在), 直接：

```bash
bash ~/DILU-Research/dilu/amgx/bench/lab_deploy/bootstrap_lab_5060.sh all
```

总耗时 ~20-30 分钟（AMGx build 占大头 ~10-15 分钟）。

## 4. 分步（推荐第一次跑）

```bash
# install 阶段：python venv + AMGx git+build
bash ~/DILU-Research/dilu/amgx/bench/lab_deploy/bootstrap_lab_5060.sh install

# build 阶段：dilu/amgx FFI shim
bash ~/DILU-Research/dilu/amgx/bench/lab_deploy/bootstrap_lab_5060.sh build

# run 阶段：bench
bash ~/DILU-Research/dilu/amgx/bench/lab_deploy/bootstrap_lab_5060.sh run
```

每个阶段失败就把错误贴给我。

## 5. 把结果 push 回 git

`bootstrap.sh run` 末尾会自动提示这些命令，直接复制粘贴：

```bash
cd ~/DILU-Research
git add dilu/amgx/bench/lab_deploy/results/
git commit -m "lab 5060: AMGx bench + precision results"
git push origin main
```

## 6. 我（开发机）pull 后写最终 3-way 报告

```bash
cd ~/DILU-Research && git pull   # 我会做
```

然后：
- Lab Xeon OF wall = **49.57 ms** (median pd_corr0, N=32) ✓ 已有
- Lab 5060 AMGx wall = (你跑出来的)
- 我们 replica = 11655 ms (本机, 已有)

写真 3-way 报告。

---

## 已知坑

1. **CUDA_ARCH**: bootstrap script 默认 `86`（Ampere RTX 30/40 series）。RTX 5060 是 Blackwell, 可能需要 `120`。如果 AMGx build 报 "no kernel image" 错误：
   ```bash
   # 编辑 bootstrap script 这行: -DCUDA_ARCH="86"
   # 改成: -DCUDA_ARCH="all"   (兼容所有架构, 编译慢但稳)
   ```

2. **AMGx 编译慢**: 第一次 ~10-15 分钟，之后增量。

3. **CUDA toolkit 没装**: 需要联系 lab 管理员装 `cuda-toolkit-12-x`（不能 sudo 也告诉我，我给 conda 路径方案）。

4. **python3 太老 (< 3.10)**: bootstrap 用 `python3` 默认；如太老用 conda 装新的。
