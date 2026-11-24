---
name: op-perf-bottleneck-analysis
description: 定位 NPU 算子性能瓶颈并给出优化方向：用 msprof application 模式跑 tests/perf 单算子用例，采集主要负载（单/多 batch、长短序列 prefill/decode）基线；先评估多核负载均衡（核利用率 + 核间均衡度 + 任务分配模型对照，区分任务独裁型/带宽饱和型，不均衡先修任务划分再谈流水）；MIX 算子通过注释 AIC/AIV 主代码段（保留核间同步与框架）判别关键路径；再按各 pipeline time(us) 与 Task Duration(us) 的 max/sum 关系判断流水并行性；最后对瓶颈流水线逐段消融定位代码段，输出瓶颈结论与优化手段建议。当用户要求分析算子性能瓶颈、定位 kernel 慢的原因、给算子优化方向时使用。
---

# 算子性能瓶颈分析

按六步推进，每一步都以 msprof 实测数据为依据，禁止凭代码阅读直接下结论：

```
负载基线采集 → 多核负载均衡分析 → MIX 判别 AIC/AIV 关键路径 → pipeline 级分析 → 流水并行性判断 → 代码段定位
```

## 输入

`$ARGUMENTS` 为算子名（如 `indexer_topk`）或 perf 脚本名。中间产物（msprof 输出、消融日志、报告）统一放**主仓库** `xlite/agent_log_scripts/<op>_perf/`——worktree 固定建在主仓库根下（见前置准备），从 `<worktree>/xlite` 出发即 `../../xlite/agent_log_scripts/<op>_perf/`。产物**不放 worktree 内**，否则清理 worktree 时 report 与 PROF 数据会随 `--force` 一并删除。

## 前置准备

1. **worktree**：步骤 2/5 需要临时注释 kernel 代码，属代码修改，必须先建 worktree（全局 CLAUDE.md 规则）：
   ```bash
   cd <主仓库>/xlite
   git worktree add ../xlite-wt-<op>-perf -b wt/<op>-perf
   ```
   必须在主仓库 `xlite/` 目录下执行（`../xlite-wt-<op>-perf` 才落在主仓库根下；后文 Step 1 的 `--output=../../xlite/...` 产物相对路径同样依赖此布局）。基线与所有消融版本都在同一 worktree 内构建、同窗口测试，排除环境差异。
2. **编译**：单算子测试用 clean CMake 构建（全局 CLAUDE.md 编译方式 1；勿用裸 `-j`，共享机无界并行会 EMFILE）：
   ```bash
   cd <worktree>/xlite
   rm -rf build && cmake -B build && cmake --build build -j "${MAX_JOBS:-128}" && cmake --install build
   ```
   `cmake --install` 把产物装回**本 worktree 源码树**（口径同 `doc/developer_guide/kernels/microbench.md` §2.1/§2.2）：Python binding 是 `xlite/xlite_sabi.abi3.so` + `xlite/_C.py` shim（无 `_C.so`），`libxlite.so` 与 `libxlite_kernels_*.so` 在 `xlite/xlite/lib/`，cmake 不安装任何 `.py`。运行 perf 脚本按源码树口径导入本次构建：
   ```bash
   export PYTHONPATH=<worktree>/xlite
   export LD_LIBRARY_PATH=$PYTHONPATH/xlite/lib:<torch>/lib:$LD_LIBRARY_PATH
   ```
   site-packages 中的旧 xlite wheel / editable 安装会遮蔽源码树构建（静默导入旧二进制），先 `pip uninstall xlite` 排除干扰。
3. **perf 脚本**：确认 `xlite/tests/perf/<op>.py` 存在；缺失时有两条路：
   - 参照 `tests/perf/indexer_topk.py` 模板补写：输入全部预构建 → sleep 过 msprof `--delay`（`-T` 是**发射前等待秒数**，脚本在 T+0.5s 处才开始发射，help 明言 should match msprof `--delay`，取 ≈delay；发射点必须落在采集窗口 [delay, delay+duration] 内）→ 窗口内只发射目标 kernel，无 sync；**每次发射各打一行发射摘要**（含 model 与序列参数；`-N` 重复发射时每次都打，日志行数 = 发射数——入库模板只有 `[INFO] Firing/Done` 两行、不足以对齐，汇总脚本靠这些行与 CSV 按序严格对齐）；
   - 以 `tests/kernels/<op>.py` 单测用例为底稿改造：其 work 配置与输入构建可直接复用（shape 分支覆盖比手写全），去掉精度校验/print/CPU 参考计算，加预构建 + sleep-delay 逻辑后即成 perf 用例。

## Step 1: 主要负载基线采集

msprof **application 模式**（msprof 前缀包裹测试程序，进程生命周期即采集生命周期）：

```bash
cd <worktree>/xlite
msprof --delay=8 --duration=5 --output=../../xlite/agent_log_scripts/<op>_perf \
    python tests/perf/<op>.py -T 10
```

- `--output` 指向主仓库产物目录（worktree 建在主仓库根下，`../..` 即主仓库根），勿放 worktree 内。
- `-T` 是脚本的**发射前等待**秒数（默认模板在 T+0.5s 才开始发射，`--help` 明言 should match msprof `--delay`）：发射点 T+0.5 须 ≥ delay 且 < delay+duration，取 `-T` ≈ delay~delay+2（示例 10 → 发射点 10.5s ∈ [8, 13]）。**加大 `-T` 会把发射点推出窗口外、采不到任何 kernel**。
- `--duration` 须装下发射点之后的全部负载发射：≥ (T+0.5 − delay) + 全部发射耗时；装不下（尾部 case 缺行）时**加大 `--duration`**（如 `--duration=15`），`-T` 不动。
- perf 脚本对同一负载重复发射多次（如 `-N 3`），供取中位数、看波动。

**负载矩阵**（"主要负载"必须覆盖）：

- 单 batch 与多 batch（如 batch=1 与 2/4/8/16/32）；
- 入参涉及 cached_lens / query_lens 等序列参数时，覆盖长短序列的 prefill 与 decode：
  - prefill：cached_lens=0，query_len 取短/长（如 3142 / 8192 / 16384）；
  - chunked prefill: query_len=8196, cached_lens 取短/长 (如 16384 / 32768 / 65536 / 131072)
  - decode：query_len=1，cached_lens 取短/长（如 3186 / 16384 / 32768 / 65536 / 131072）；
  - mixed：同 batch 内 decode + prefill 混合（有则覆盖）。

采集完成后（post-processing 约 10s）解析主仓库 `xlite/agent_log_scripts/<op>_perf/PROF_*/mindstudio_profiler_output/op_summary_*.csv`：每个 kernel 发射一行。先核对行数覆盖全部负载发射——缺行说明发射超出采集窗口尾部（`--duration` 不足，加大重测）或发射点整个落在窗口外（`-T` 过大，采不到任何行），缺行无报错且会让后续步骤整体错位。核对无误后再按发射顺序（Task Start Time）对齐回负载 case。关键列：

| 列 | 含义 |
|---|---|
| Task Duration(us) | 任务总时长，瓶颈分析锚点 |
| Task Type | AIC / AIV / MIX_AIC / MIX_AIV（MIX 判别依据，配合 Mix Block dim）|
| aicore_time / aiv_time | AIC / AIV 侧时长 |
| aic_mac_time (ratio) | AIC Cube 矩阵计算流水 |
| aic_scalar_time (ratio) | AIC 标量流水（地址/循环/分支）|
| aic_mte1_time (ratio) | AIC MTE1：L1→L0A/L0B 搬入（Cube 输入；写 UB 的是 Fixpipe） |
| aic_mte2_time (ratio) | AIC MTE2：GM→L1/L0 搬入 |
| aic_fixpipe_time (ratio) | AIC Fixpipe：L0C→UB/GM 搬出 |
| aiv_vec / aiv_scalar / aiv_mte2 / aiv_mte3 | AIV 向量计算 / 标量 / 搬入 / 搬出 |

汇总解析按下述参考实现（原 `analyze_op_summary.py` 为一次性脚本未入库，此处内嵌模板；CSV/日志路径与算子名作参数，日志正则须按目标算子 perf 脚本的打印格式调整；模板按组求和聚合，同负载多实例取中位数时把累加改 median）。保存为 `agent_log_scripts/<op>_perf/analyze_op_summary.py` 使用：

```python
#!/usr/bin/env python3
# 用法: python3 analyze_op_summary.py <op_summary.csv> <perf测试日志> [算子名前缀]
import csv, re, sys

summary, testlog = sys.argv[1], sys.argv[2]
op = sys.argv[3] if len(sys.argv) > 3 else "indexer_topk"

# case 序列取自 perf 测试日志（每次发射一行, -N 重复发射逐次都打, 与 CSV 内 kernel 实例按发射顺序一一对应）。
# 注意: 入库的 tests/perf/indexer_topk.py 只有 [INFO] Firing/Done 两行、无逐发射打印;
# 此正则对应**加了逐发射摘要**的改造版(见前置准备 3), 逐字段改到与 perf 脚本实际打印一致
cases = []
pat = re.compile(
    rf"{op} (\S+) \(\d+ heads, \d+ head_dim, \S+\) work "
    r"\(\d+ batch, cached_lens=\[([\d,]+)\], query_lens=\[([\d,]+)\], topK=(\d+), compress_ratio=(\d+)\)"
)
for line in open(testlog):
    m = pat.search(line)
    if m:
        cases.append(dict(model=m.group(1), cached=m.group(2), ql=m.group(3),
                          topk=int(m.group(4)), ratio=int(m.group(5))))
assert cases, "log matched 0 cases: regex does not fit perf script output (see above)"

rows = [r for r in csv.DictReader(open(summary)) if r["Op Name"].startswith(op)]
print(f"csv {op} rows: {len(rows)}, log cases: {len(cases)}", file=sys.stderr)
assert len(rows) == len(cases)  # 严格一一对应(每次发射各一行); < 为窗口截断(--duration 不足), > 为混入他行或日志漏打, 见上文覆盖率校验

def bucket(c):  # prefill/decode 分桶（按该算子的负载形态调整阈值）
    q = max(int(x) for x in c["ql"].split(","))
    return "decode" if q <= 4 else "prefill"

cols = ["Task Duration(us)", "aicore_time(us)", "aic_mac_ratio", "aic_scalar_ratio",
        "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
        "aiv_time(us)", "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"]

groups = {}
for c, r in zip(cases, rows):
    key = (c["model"], bucket(c), f"ratio{c['ratio']}")
    g = groups.setdefault(key, dict(n=0, dur=0.0, **{k: 0.0 for k in cols[1:]}))
    g["n"] += 1
    g["dur"] += float(r["Task Duration(us)"])
    for k in cols[1:]:
        v = float(r[k]) if r[k] not in ("N/A", "") else 0.0
        if k.endswith("(us)"):
            g[k] += v
        else:  # ratio: duration-weighted
            g[k] += v * float(r["Task Duration(us)"])

print(f"{'group':<42}{'n':>4}{'dur_ms':>10}{'mac':>7}{'scal':>7}{'mte1':>7}{'mte2':>7}{'fixp':>7}"
      f"{'|aiv_vec':>9}{'aiv_m2':>8}{'aiv_m3':>8}")
for key in sorted(groups):
    g = groups[key]
    d = g["dur"]
    print(f"{key[0]+'/'+key[1]+'/'+key[2]:<42}{g['n']:>4}{d/1000:>10.1f}"
          f"{g['aic_mac_ratio']/d:>7.2f}{g['aic_scalar_ratio']/d:>7.2f}"
          f"{g['aic_mte1_ratio']/d:>7.2f}{g['aic_mte2_ratio']/d:>7.2f}"
          f"{g['aic_fixpipe_ratio']/d:>7.2f}"
          f"{g['aiv_vec_ratio']/d:>9.2f}{g['aiv_mte2_ratio']/d:>8.2f}{g['aiv_mte3_ratio']/d:>8.2f}")
```

## Step 2: 多核负载均衡分析（先于 pipeline 分析）

pipeline 级结论以核间均衡为前提：不均衡产生的核间空泡会被后续步骤误读为等待/同步/标量开销。基线解析完成后，对每个主要负载 case 依次做：

1. **核利用率 = Block Num / 物理 AIC 核数**（op_summary 的 Block Num 列；910B3 AIC=24）。Block Num < 核数说明任务数少于核数、有核闲置（欠划分），小 shape 算子常见——先把任务切细，任何流水优化都无济于事。
2. **核间均衡度 = aicore_time / Task Duration**（纯 AIC kernel；AIV/MIX 侧用 `aiv_time` 同理）。**aicore_time 就是「全部核的活跃时长平均」，即平均核忙时**——matmul 实测逐实例精确验证 aicore_time = aic_total_cycles / (Block Num × f)（f ≈ 核主频，910B3 ≈1800 cycles/us，与芯片标称一致），msprof 已直接给出该值，直接取比值即可，**不要**再用 aic_total_cycles/主频重算。比值 ≤1、越接近 1 越均衡。首次使用该指标时，建议按上述公式反推做一次**语义交叉验证**（确认本机 msprof 的 aicore_time 确为核平均而非最长核/任务墙钟——不同语义下同样数字的结论完全不同）；验证过一次即可，后续直接用。
3. **任务分配模型对照**：从 host 侧 tiling/任务划分代码复刻每核任务分布（xlite kernel 均为 `for (idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum())` 的 stride 分配，每核 ceil/floor 任务数；尾 tile 的 mActual/nActual/kLoop 差异单独计入），得理论均衡度 = 平均任务量 / 最忙核任务量。注意 `aic_total_cycles` 只计核的活跃 cycles——核在 DMA 后台传输期间 wait_flag 空转不烧 cycles，所以**搬运主导型 kernel 的实测均衡度天然混有「流水等待空泡」，不能直接当任务不均读**：
   - 先测**整除对照形状**（tiles 恰整除 Block Num 且无尾 tile）：其均衡度 = 该负载的等待空泡基线（流水/同步问题，归 Step 5 范畴）；
   - 不整除形状相对整除对照的均衡度差 = **纯任务分配不均损失**，才是修任务划分能回收的收益（matmul decode 实测：63 tiles 的 3/2 分配均衡度 0.77，同最忙核流量的 72 tiles 整除对照 0.86 → 任务不均净损失 ~10%，与理论 3/2 分配的 12.5% 同量级）；
   - 计算主导型 kernel（cycles 全烧在 Cube/Mmad 上，如大 m matmul 实测 0.99–1.00）均衡度直接可信，无需对照。
4. **形状扫描因果验证**（理论对不上或需要钉死时）：构造 tiles 恰整除 Block Num 的均衡形状与不整除形状（总流量相近、最忙核任务量相同或不同），同窗口对比均衡度与 Task Duration；另用 `Duration ≈ 总搬运量 / DRAM 峰值带宽`（910B3 ≈1.6TB/s）判**带宽饱和型**（时长贴着带宽下限、均衡无收益，瓶颈转向搬运总量，如 lm_head 1.49TB/s）。扫描通过 perf 脚本的自定义负载参数发射（如 `tests/perf/matmul.py -W name:n:k:m1,m2`），不改 kernel；注意区分冷/热实例（首发射 L2 全冷更接近生产 decode 逐层换权重的真实态），逐实例看，勿只看中位数。
5. 处置：
   - 任务不均损失显著（≥10%）→ **先修均衡再进 Step 3**：m0/n0 微调使 tiles 整除 Block Num（例：n=16128、n0=256 时 63 tiles 不整除 24 核；n0=224 → 72 tiles 恰整除）、任务切细（每核 ≥3~4 个任务）、尾任务均摊；改后同窗口 A/B 重测均衡度应逼近整除对照、Task Duration 应相应下降。

## Step 3: MIX 算子 AIC/AIV 关键路径判别（消融法）

Task Type 为 MIX（MIX_AIC/MIX_AIV 或 Mix Block Dim > 0）时执行；纯 AIC/AIV 算子直接进 Step 4。

分别注释 AIC、AIV 的**主代码段**，重测 msprof：

1. **纯 AIC 版**：注释 AIV 侧主体计算/搬运代码；
2. **纯 AIV 版**：注释 AIC 侧主体计算/搬运代码；
3. **注释纪律**：只注释计算/搬运主体，**保留核间同步（set_flag/wait_flag 等 flag 收发）与任务框架（task 划分、循环骨架）**——set/wait 必须成对保留，否则死锁或时延结构失真；
4. 每个消融版本 clean 重建，用与 Step 1 完全相同的负载与命令采集。

判读（D = Task Duration 中位数，本机噪声 ±12% 内视为无差别）：

- D(纯AIC) ≈ D(完整) → AIV 被 AIC 完全掩盖，**AIC 是关键路径**，进 Step 4 分析 AIC 各流水；
- D(纯AIV) ≈ D(完整) → AIC 被掩盖，**AIV 是关键路径**，按 aiv_vec / aiv_scalar / aiv_mte2 / aiv_mte3 同法分析；
- 两者都明显小于 D(完整) → 两侧都在关键路径（串行接力或各占时段），分别分析。

辅助佐证：aicore_time / aiv_time 相对 Task Duration 的占比；往轻的一侧塞负载看 Task Duration 是否传导（不传导 = 该侧被掩盖）。

**所有消融版本与完整版本同窗口交替测试**（完整 → 消融 → 再完整），防机器状态漂移污染对比。

## Step 4: pipeline 级瓶颈分析

对关键路径侧（以 AIC 为例）从 op_summary 取各 pipeline time(us)：

1. 找出 time 最大的 pipeline（mac / scalar / mte1 / mte2 / fixpipe 之一）；
2. 看该最大值是否**接近 Task Duration(us)**（或该侧 aicore_time）：
   - 接近 → 该流水主导任务时长，是瓶颈流水；
   - 明显小于 → 时长耗在流水之外（等待/同步/尾空泡），进 Step 5 检查流水组织；若 Step 2 已发现任务不均/欠划分，先修均衡再回看本步。

注意：pipeline time 是该流水线的活跃时长（含 stall），多条流水的 time 可以重叠；单看某条 ratio 高不能直接断言串行，须结合 Step 5。

## Step 5: 流水并行性判断（先于代码段优化）

比较 Task Duration 与各 pipeline time 的关系：

| 观察 | 结论 | 动作 |
|---|---|---|
| Task Duration ≈ 各 pipeline time 之**和**（如 mte1+mac+fixpipe ≈ 总时长，sum/max 接近流水条数）| 流水**没有并行**，串行接力 | 先优化流水线间的并行方式与同步点（见下），优化后重测再进 Step 6 |
| Task Duration ≈ max(pipeline time)，sum >> Task Duration | 流水已充分并行 | 瓶颈=最长流水，直接进 Step 6 |
| max(pipeline) 与 sum 都对不上 Task Duration | 时长耗在等待/同步/尾空泡 | 检查核间同步开销、任务划分是否留空泡、尾块处理；Step 2 的整除对照形状所测得的等待空泡基线属于此类 |

流水未并行时的优化顺序：

1. **并行方式**：双/多缓冲让搬数与计算重叠、软件流水（上一块计算与下一块搬入重叠）、任务切细让各流水持续有活干、L1/L0 数据跨任务驻留复用；
2. **同步点**：消除不必要的 `pipe_barrier(PIPE_ALL)`，粗粒度同步改 set_flag/wait_flag 定向同步，收窄同步区间；
3. 每改一版同窗口 A/B 重测：Task Duration 应下降，且 sum(pipeline time)/Task Duration 比值应**上升**（串行接力时 ≈1，充分并行后趋近流水条数 n，见上表两种判型的定义）——重叠度上升，以此验证流水真的并行了。

## Step 6: 瓶颈流水线内的代码段定位（逐段消融）

流水并行性达标后，对最大 time 的流水线定位具体代码段。若该流水线内有多个代码步骤（多段 LoadData / copy_loop / matmul / fixpipe 步骤等）：

1. 每次只注释**一个**步骤（同样保留同步、循环骨架与数据流框架），clean 重建后用相同负载重测；
2. 记录 Δ(Task Duration) 与 Δ(该 pipeline time)，对步骤按贡献排序；
3. 逐个推进直到找出占比最大的代码段；对可疑段可再做"只保留该段"的反向验证。

注意：注释一段后流水会重排，Δ 不严格等于该段的独立贡献，但足以排序定位主导段。

## 产出：瓶颈分析报告

分析结论落盘主仓库 `xlite/agent_log_scripts/<op>_perf/report.md`（勿放 worktree 内），包含：

1. **瓶颈结论**：哪个负载下、AIC 还是 AIV、哪条 pipeline、哪段代码；附每步关键数据表（Task Duration / 各 pipeline time 与 ratio / 消融对照），并注明数据来源 PROF 目录与测试命令，保证可复现；
2. **多核负载均衡**：每负载的核利用率与核间均衡度表、理论任务分配对照、整除对照形状的等待空泡基线、任务独裁型/带宽饱和型判别结论，以及均衡修复建议（若适用）；
3. **优化手段建议**（按分析层级给）：
   - 欠划分/任务不均 → tiles 切细、tiling 参数微调整除 Block Num、尾任务均摊（见 Step 2 处置）；
   - 搬运类流水（mte1/mte2/fixpipe）主导 → 减少搬运量：合并 LoadData、Nd2Nz 等布局变换、L1/L0 跨任务驻留、预取掩盖；
   - 计算类流水（mac）主导 → 算法等价变换减计算量、tile 重排提高 Cube 利用率；
   - 标量流水（scalar）主导 → 地址/循环计算外提、常量折叠、整除改乘法/参数透传；
   - 流水未并行 → 按 Step 5 的并行方式与同步点改造；
   - 等待/空泡主导 → 核间同步减粒度、任务划分均衡化；
4. **下一步实验**：每条建议附验证方法（改哪里、怎么测、预期哪个指标变化、如何排除 ±12% 噪声）。

## 注意事项

- **噪声控制**：本机 perf 波动 ±12%。一切消融对比必须同窗口交替测试、同负载多实例取中位数；差异在噪声带内的不得下结论，需要时用 +N/-N 交替放大差异。
- **NPU 共享**：用完立即释放，不留后台进程；测试因内存不足失败不追究，直接重试。
- **worktree 清理**：先确认 report.md 与 PROF_* 均已落盘**主仓库** `xlite/agent_log_scripts/<op>_perf/`（本 skill 所有 `--output` 与报告路径均指向主仓库，worktree 内不留分析产物）；再 `git worktree remove --force ../xlite-wt-<op>-perf` 并删分支（worktree 内剩余的 build/ 等未跟踪构建产物随 `--force` 一并清除）；消融注释一律不留在代码里。
- **msprof 模式区分**：本 skill 全程用 application 模式（msprof 前缀 + `--delay`/`--duration`）；vllm 服务级采集才用动态模式（`--dynamic=on` + fifo start/stop），勿混用。
- 需要看各 stage 时间线重叠细节时导出 timeline：
  ```bash
  python3 /usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msprof/msprof.py export timeline -dir=../../xlite/agent_log_scripts/<op>_perf/PROF_XXX
  ```
