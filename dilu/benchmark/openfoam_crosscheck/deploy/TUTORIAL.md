# 完整操作教程 — 从另一台电脑采集 OpenFOAM 矩阵

> **谁写的**：DILU-Research 项目（yzk + Claude）
> **目的**：把 OpenFOAM `laserMeltFoam` 跑过的 LPBF case 里的 **(A, b, x, metadata)** 矩阵挖出来，拿回来喂 cuSPARSE / AMGx 对标
> **预计时长**：你第一次走完大约 2 小时（含编译、1 个烟雾 case、1 个正式 case）
> **需要两台电脑**：
>   - **源机**（DILU-Research 代码所在，下称 "本机"）—— 发送包、接收结果
>   - **目标机**（跑 OpenFOAM 的那台，下称 "远机"）—— 编译 + 运行

---

## 0. 开始前确认清单（3 分钟）

**本机**：
- [x] `/home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/deploy.tar.gz` 存在（~20 KB）
- [x] 你能 ssh 到远机，或至少有 U 盘

**远机你需要知道**：
- [x] OpenFOAM 装在哪（典型 `/usr/lib/openfoam/openfoamXXXX/` 或 `/opt/openfoam/`）
- [x] `WM_PROJECT_VERSION` 是啥（`v2506` / `v2412` / ...）
- [x] `laserMeltFoam` 源码在哪（通常 `~/LaserbeamFoam/applications/solvers/laserMeltFoam/`）
- [x] 你要对标的 case 在哪（至少 1 个跑完的）
- [x] 磁盘空闲**至少 200 GB**（2M cells × 50 步 × 5 矩阵 × 700 MB ≈ 175 GB）

**不确定也没关系**，第 3 步有检测脚本。

---

## 1. 本机：打包 + 准备传输

### 1.1 确认 deploy 包就绪

```bash
cd /home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck
ls -la deploy.tar.gz
tar tzf deploy.tar.gz | head
```

**期望输出**：
```
deploy.tar.gz        (~20 KB)
deploy/
deploy/README.md
deploy/TUTORIAL.md
deploy/install.sh
deploy/patch_case.sh
deploy/sanity.py
deploy/matrixDumper.H
deploy/*.diff
...
```

### 1.2 传到远机

**方式 A — scp**（推荐，假设远机 `myremote.example`，用户 `alice`）：
```bash
scp deploy.tar.gz alice@myremote.example:~/
```

**方式 B — rsync**（能断点续传）：
```bash
rsync -avP deploy.tar.gz alice@myremote.example:~/
```

**方式 C — U 盘**：
```bash
cp deploy.tar.gz /media/yzk/USB/
# 物理拷贝到远机，放到 ~/
```

---

## 2. 远机：登录 + 解压

### 2.1 登录

```bash
ssh alice@myremote.example          # 或直接坐到远机前
```

### 2.2 解压

```bash
cd ~
tar xzf deploy.tar.gz
cd deploy
ls
```

**期望看到**：
```
README.md        TUTORIAL.md       install.sh      patch_case.sh
sanity.py        matrixDumper.H    *.diff  *.patched  *.pristine
```

---

## 3. 远机：环境检测（3 分钟）

### 3.1 确认 OpenFOAM 能用

```bash
# 找 OpenFOAM 安装位置
ls /usr/lib/openfoam/ 2>/dev/null || ls /opt/openfoam* 2>/dev/null || ls ~/OpenFOAM 2>/dev/null
```

### 3.2 source OpenFOAM 环境

按你实际路径调整：
```bash
source /usr/lib/openfoam/openfoam2506/etc/bashrc       # v2506
# 或:
source /opt/openfoam2412/etc/bashrc                     # v2412
# 或:
source ~/OpenFOAM/OpenFOAM-v2506/etc/bashrc            # 用户空间装的
```

### 3.3 验证

```bash
echo "OpenFOAM version: $WM_PROJECT_VERSION"
which wmake
which laserMeltFoam    # 当前二进制路径
```

**期望**：
```
OpenFOAM version: v2506
/usr/lib/openfoam/openfoam2506/bin/wmake
/home/alice/OpenFOAM/alice-v2506/platforms/linux64GccDPInt32Opt/bin/laserMeltFoam
```

**如果 `WM_PROJECT_VERSION` 为空** → OpenFOAM env 没 source 成功，回到 3.2 改路径。

**如果版本不是 v2506**：
- v2412/v2406：`install.sh` 会警告但大概率仍能编译。若报 `getOrDefault` 不存在的错，按 README.md 说明改一行
- v2206 及更早 / foam-extend：**大概率编译失败**，联系我

---

## 4. 远机：装补丁 + 重编译（5-10 分钟）

### 4.1 找到 laserMeltFoam 源码目录

```bash
# 常见位置：
ls ~/LaserbeamFoam/applications/solvers/laserMeltFoam/
# 或:
ls ~/LaserbeamFoam/LaserbeamFoam/applications/solvers/laserMeltFoam/
```

**应该看到** `laserMeltFoam.C`, `TEqn.H`, `pEqn.H`, `Make/` 等。

记下这个完整路径，比如 `SRC=~/LaserbeamFoam/applications/solvers/laserMeltFoam`。

### 4.2 运行 install.sh

```bash
cd ~/deploy
./install.sh ~/LaserbeamFoam/applications/solvers/laserMeltFoam
```

**期望末尾输出**：
```
[install] SUCCESS
  New binary: /home/alice/OpenFOAM/alice-v2506/platforms/linux64GccDPInt32Opt/bin/laserMeltFoam
  Original sources saved as *.orig in ~/LaserbeamFoam/.../laserMeltFoam

Next: apply case patch with patch_case.sh
```

### 4.3 如果 install.sh 报 patch 失败

意思是你的 `laserMeltFoam.C / TEqn.H / pEqn.H` 和我的基准版本不一致（可能你魔改过）。两条路：

**Option 1 — 手动小修改**：看 `~/deploy/*.diff`，每个文件就插入 2-4 行，手工对齐插入点

**Option 2 — 暴力覆盖**（仅当你没魔改过这 3 个文件）：
```bash
cp ~/deploy/laserMeltFoam.C.patched ~/LaserbeamFoam/applications/solvers/laserMeltFoam/laserMeltFoam.C
cp ~/deploy/TEqn.H.patched          ~/LaserbeamFoam/applications/solvers/laserMeltFoam/TEqn.H
cp ~/deploy/pEqn.H.patched          ~/LaserbeamFoam/applications/solvers/laserMeltFoam/pEqn.H
cp ~/deploy/matrixDumper.H          ~/LaserbeamFoam/applications/solvers/laserMeltFoam/
cd ~/LaserbeamFoam/applications/solvers/laserMeltFoam
wmake
```

### 4.4 验证编译成功

```bash
ls -la $(which laserMeltFoam)
# 时间戳应该是刚刚（几秒前）
```

---

## 5. 远机：选一个 case 做烟雾测试（5 分钟跑 + 3 分钟验证）

**重要**：第一次不要直接上大 case。先用最小的参数验证整条链路通。

### 5.1 挑一个最小的 case（或最早跑的）

```bash
ls ~/my_lpbf_cases/       # 你已有的 case 所在目录
# 假设有 case_150W case_250W 等。挑一个
CASE=~/my_lpbf_cases/case_150W
ls $CASE
```

**应该看到**：`constant/` `system/` `0/` + 若干时间文件夹（`1e-07/`, `5e-07/`, ...）。

### 5.2 检查 case 能接着跑

```bash
cd $CASE
ls -d [0-9]* | sort -g | tail    # 看最新时间
cat system/controlDict | grep -E "(startFrom|endTime|maxDeltaT)"
```

确认**最新时间文件夹**里有 `U T pd alpha.metal` 等字段。

### 5.3 贴 dumper + 改 controlDict（只跑 3 步做烟雾）

```bash
~/deploy/patch_case.sh $CASE 3
```

**期望输出**：
```
[patch_case] Latest time folder: 5e-07
[patch_case] maxDeltaT = 2e-08 (detected)
[patch_case] new endTime = 5.6e-07  (= 5e-07 + 3 × 2e-08)
[patch_case] wrote system/matrixDumperDict (dump steps 1..3)
[patch_case] backed up controlDict → controlDict.orig

[patch_case] DONE.
...
```

### 5.4 看一眼改对了没

```bash
cat $CASE/system/matrixDumperDict
cat $CASE/system/controlDict | grep -E "(startFrom|endTime)"
```

确认：
- `matrixDumperDict` 里 `dumpTimeSteps (1 2 3);`
- `controlDict` 里 `startFrom latestTime;` 且 `endTime` 比最新时间略大

### 5.5 跑 laserMeltFoam

```bash
cd $CASE
laserMeltFoam 2>&1 | tee log.laserMeltFoam
```

**期望看到**（从早到晚）：
```
Time = 5.0000000002e-07
...
[matrixDumper] ENABLED. outputDir="postProcessing/matrices" dumpTimeSteps=3(1 2 3) equations=2(pd T)
...
DICPCG: Solving for pd, Initial residual = ...
[matrixDumper] pre-solve pd_corr0  N=2048000 nnz=14220800
[matrixDumper] post-solve pd_corr0 finalRes=... iters=...
...
End
```

整个 3 步大约 3-8 分钟（视 mesh 大小）。

**如果卡在 "Time loop" 开始前** → OpenFOAM 初始化阶段失败（通常是 fvSchemes 或 fvSolution 的问题），贴 log 末尾 30 行给我

**如果看不到 `[matrixDumper] ENABLED`** → `matrixDumperDict` 没读进去，检查 step 5.4

### 5.6 验证 dump 正确

```bash
cd $CASE
python3 ~/deploy/sanity.py postProcessing/matrices
```

**期望末尾**：
```
Total: 15 matrices, 15 pass, 0 fail
```

（15 = 3 步 × 5 矩阵/步）

**如果全 pass** ✅ 你的 dumper + case 配置都没问题，下一步放大
**如果有 fail** → 贴给我完整 sanity.py 输出

---

## 6. 远机：查看磁盘，估算规模

```bash
du -sh $CASE/postProcessing/matrices/
df -h $HOME
```

**看一下**：15 矩阵用了多少 GB，按比例估算 50 步会占多少。

**举例**：
- 15 矩阵 = 10 GB → 50 步 × 5 矩阵 = 250 矩阵 = ~170 GB
- 如果磁盘只有 100 GB 空闲，**降到 N=25**（~85 GB）

### 6.1 决定每个 case 跑多少步

填下面这个公式：
```
safe_N = (你可用磁盘 GB × 0.8) / (单矩阵 GB × 5)
```

**例**：可用 300 GB，单矩阵 0.7 GB → `safe_N = 300 × 0.8 / (0.7 × 5) ≈ 68` → 取 **N=60** 保守

---

## 7. 远机：正式跑多个 case（时间最长的一步）

### 7.1 还原刚才烟雾 case 的 controlDict

```bash
cd $CASE
mv system/controlDict.orig system/controlDict    # 恢复原来的 endTime
rm system/matrixDumperDict                        # 清理 dumper 配置
rm -rf postProcessing/matrices                     # 清掉烟雾测试的矩阵（保留就一起跑）
# 删除烟雾测试跑出来的新时间文件夹（恢复到烟雾前状态）
# 注意：找 "latestTime" 之后新产生的那几个文件夹
ls -d [0-9]* | sort -g | tail
# 人工确认哪些是烟雾产物，rm -rf 它们
```

（如果觉得麻烦，也可以保留烟雾结果，反正它们也是有用的数据）

### 7.2 为每个 case 写一个 batch 脚本

在远机 `~/deploy/` 目录下建：

```bash
cat > ~/deploy/run_all_cases.sh <<'EOF'
#!/bin/bash
# 跑所有 case，每个 N 步，全部 dump
set -e
N=50                              # ← 改成你的 safe_N
CASES=(
    ~/my_lpbf_cases/case_150W
    ~/my_lpbf_cases/case_250W
    ~/my_lpbf_cases/case_400W
    # 继续加...
)

for CASE in "${CASES[@]}"; do
    echo "============================================"
    echo "=== Processing: $CASE"
    echo "============================================"

    # patch
    ~/deploy/patch_case.sh "$CASE" $N

    # run
    cd "$CASE"
    laserMeltFoam 2>&1 | tee log.laserMeltFoam_matdump

    # sanity
    python3 ~/deploy/sanity.py postProcessing/matrices \
        | tee sanity_report.txt

    # restore controlDict so we don't break your normal workflow
    mv system/controlDict.orig system/controlDict

    cd -
done

echo ""
echo "ALL CASES DONE"
df -h $HOME
EOF
chmod +x ~/deploy/run_all_cases.sh
```

### 7.3 编辑 `CASES=(...)` 填上你实际的 case 路径

```bash
nano ~/deploy/run_all_cases.sh     # 或 vim
```

### 7.4 启动（建议挂在 tmux/screen 里）

```bash
tmux new -s lpbfdump
# 在 tmux 里：
~/deploy/run_all_cases.sh 2>&1 | tee ~/run_all.log
# 按 Ctrl-B D 脱离 tmux；过几小时回来 `tmux attach -t lpbfdump`
```

**预计**：3-5 个 case × 每 case 30-60 min = 2-5 小时。

### 7.5 跑的过程中可以看进度

（另开一个 ssh 或 tmux 窗口）：
```bash
tail -f ~/run_all.log
watch -n 30 "df -h $HOME"              # 监控磁盘
watch -n 60 "du -sh ~/my_lpbf_cases/*/postProcessing/matrices/"
```

---

## 8. 远机：打包结果回传（5-15 分钟）

### 8.1 检查所有 case 都 pass

```bash
cd ~/my_lpbf_cases
for d in */; do
    echo "=== $d ==="
    tail -3 "$d/sanity_report.txt"
done
```

每个 case 末尾应该是 `Total: X matrices, X pass, 0 fail`。

### 8.2 打包（注意：不打包 case 里的其他内容，只打包 matrices + sanity 报告 + 重要 log）

```bash
cd ~/my_lpbf_cases
tar czf ~/matrices_collected.tar.gz \
    */postProcessing/matrices/ \
    */sanity_report.txt \
    */log.laserMeltFoam_matdump
ls -lh ~/matrices_collected.tar.gz
```

**如果总量 > 50 GB**，`tar czf` 会很慢。用未压缩的 tar：
```bash
tar cf ~/matrices_collected.tar \
    */postProcessing/matrices/ \
    */sanity_report.txt \
    */log.laserMeltFoam_matdump
# ASCII MM 文件压缩率一般（~2×），不压也能接受
```

### 8.3 scp 回本机

从**本机**端：
```bash
scp alice@myremote.example:~/matrices_collected.tar.gz \
    /home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/
```

或从**远机**：
```bash
scp ~/matrices_collected.tar.gz yzk@yourhome.example:/home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/
```

---

## 9. 本机：解压 + 跑对比

### 9.1 解压

```bash
cd /home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck
mkdir -p collected
tar xzf matrices_collected.tar.gz -C collected/
ls collected/
# 应看到: case_150W/  case_250W/  case_400W/
```

### 9.2 激活 Python 环境

```bash
source /home/yzk/jax-env/bin/activate
python -c "import jax; print(jax.__version__)"  # 应输出 0.9.0
```

### 9.3 先对一个 case 跑代表性子集（10 min）

```bash
cd /home/yzk/DILU-Research
ROOT=dilu/benchmark/openfoam_crosscheck/collected/case_150W/postProcessing/matrices

# 挑一个时间步的 pd_corr0 做快速验证
python -m dilu.benchmark.openfoam_crosscheck.driver_scipy "$ROOT" \
    --pattern "*/pd_corr0" --force-gmres
python -m dilu.benchmark.openfoam_crosscheck.driver_cusparse "$ROOT" \
    --pattern "*/pd_corr0"
python -m dilu.benchmark.openfoam_crosscheck.driver_amgx "$ROOT" \
    --pattern "*/pd_corr0" --cfg aggressive
python -m dilu.benchmark.openfoam_crosscheck.compare "$ROOT"
```

**查看结果**：
```bash
cat "$ROOT/summary.md" | head -30
cat "$ROOT/headline.json"
```

### 9.4 没问题的话跑全量（每 case 45 min）

```bash
for CASE_DIR in dilu/benchmark/openfoam_crosscheck/collected/*/postProcessing/matrices/; do
    echo "=== $CASE_DIR ==="
    python -m dilu.benchmark.openfoam_crosscheck.run_all "$CASE_DIR"
done
```

---

## 10. 本机：生成最终报告

```bash
# 合并所有 case 的 summary.csv（手动或脚本）
cd dilu/benchmark/openfoam_crosscheck/collected
cat case_*/postProcessing/matrices/summary.csv | head -1 > all_summary.csv
for f in case_*/postProcessing/matrices/summary.csv; do
    tail -n +2 "$f" >> all_summary.csv
done
wc -l all_summary.csv    # 应该 = 1 + 1000+ 行
```

然后让我帮你写 benchmark 报告（`docs/benchmark/OPENFOAM_CROSSCHECK_<date>.md`），或者用 Python/Excel 自己出图表。

---

## 11. 故障排查 — 常见问题

### P1. install.sh: "patch failed"

→ 看 §4.3 手动覆盖路径

### P2. `[matrixDumper] No system/matrixDumperDict — DISABLED`

→ `matrixDumperDict` 没放对位置。必须在 `<case>/system/matrixDumperDict`

### P3. `sanity.py`: 某些矩阵 "FAIL"

→ 贴完整 sanity.py 输出给我。常见原因：
- `has_lower=true` 但我们假设 pd 对称 → 你的 fvSolution 可能不是 DIC，改了成 DILU 导致 OpenFOAM 存了独立 lower
- `abs_residual` 极大 → 可能 boundary 类型特殊（如 processor / cyclic），我的 boundary source 折叠没覆盖

### P4. wmake 报 `addBoundaryDiag is protected`

→ OpenFOAM 版本差异。matrixDumper.H 第 ~340 行我用 `matrix.D()()` 代替了 `addBoundaryDiag` 来避开这个。如果你的版本连 `D()` 都没有，需要自己下补，联系我

### P5. 磁盘快满了

- `du -sh ~/my_lpbf_cases/*/postProcessing/matrices/` 看哪个占得多
- 可以只保留 pd_corr0 和 T_corr0（删掉 pd_corr1/2）：信息量差不多但省 40% 空间
  ```bash
  find ~/my_lpbf_cases -path "*/pd_corr1" -o -path "*/pd_corr2" | xargs rm -rf
  ```

### P6. 想提前中止

- tmux 里 Ctrl-C 杀掉当前 case，或 `kill` laserMeltFoam pid
- 已经 dump 的矩阵保留，没完成的 case 可以重跑（`patch_case.sh` 会自动用 `controlDict.orig`）

---

## 12. 回滚一切（跑完不想留痕迹）

```bash
# 1. 还原 laserMeltFoam 源码
cd ~/LaserbeamFoam/applications/solvers/laserMeltFoam
for f in laserMeltFoam.C TEqn.H pEqn.H; do mv $f.orig $f; done
rm matrixDumper.H
wmake

# 2. 还原每个 case
for CASE in ~/my_lpbf_cases/*/; do
    [ -f "$CASE/system/controlDict.orig" ] && \
        mv "$CASE/system/controlDict.orig" "$CASE/system/controlDict"
    rm -f "$CASE/system/matrixDumperDict"
done

# 3. 清理 matrices（如果还没传回本机，先别删！）
# rm -rf ~/my_lpbf_cases/*/postProcessing/matrices
```

---

## 13. 时间 / 磁盘速查表

| 单 case 步数 N | 单 case 矩阵数 | 单 case 磁盘 (2M cells ASCII) | 单 case 运行时间 |
|----------------|---------------|-------------------------------|------------------|
| 3 (烟雾)       | 15            | ~10 GB                        | 5-8 min          |
| 25             | 125           | ~85 GB                        | 30 min           |
| 50             | 250           | ~170 GB                       | 60 min           |
| 100            | 500           | ~340 GB                       | 2 h              |

**3-5 case × N=50 → 750-1250 矩阵**（够学长要求）占 500-850 GB，跑 3-5 小时。

---

## 14. 联系检查点

每一步完成后贴给我的信息模板：

**4 步后（install.sh）**:
```
[install] 结果：SUCCESS / FAILED
wmake 末尾 10 行：...
```

**5 步后（烟雾 case）**:
```
case 路径：...
sanity.py 输出：
Total: 15 matrices, X pass, Y fail
```

**7 步后（全量）**:
```
total 矩阵数：...
所有 case 的 sanity_report.txt：...
```

这样我能快速诊断 + 决定下一步。
