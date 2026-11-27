# fused_group_matmul_dequant

## 功能概述

融合两段式 W8A8 MoE grouped 管线（group_matmul → dequant）为单次 launch：AIC 侧做 per-expert INT8 grouped matmul（FixPipe 乘 deq_scale 反量化输出 FP16），AIV 侧按锁步镜像 walk 将各 tile 反量化为 BF16。量化段在核外完成（`quant_dynamic`，或 MoE dispatch 的 permutation 量化前移），输入即为 INT8，AIC 进入即开始 walk——相比旧三段融合核（quant 在核内）删去了 V2C 粗粒度 barrier。相比串行链路省去一次中间 launch 与整链路同步。数学语义与串行链路 `group_matmul → dequant` 一致。

模型侧派发入口为 `XliteOpGroupMatmulDeQuant`（`csrc/model.cpp` ForwardMoE INT8 路径，w13/w2 两处调用）：deqScales 非空且收益函数 `FusedGroupMatmulDequantBeneficial(m, n, k)`（`csrc/auto_tuner.h`，与 dense 侧 `FusedMatmulDeQuantBeneficial` 统一管理）判定有收益时走本算子，否则回退串行 `XliteOpGroupMatmul + XliteOpDeQuant` 两 launch 链（串行链应用动态 token 收缩 `num`，融合核消费全部计数行）。INT8 输入有两种互斥来源：量化前移（`hoistQuant`，INT8 且 hiddenSize%32==0 时默认开启，`XLITE_DISABLE_MOE_QUANT_HOIST=1` 关闭）时 `expertsSorted` 在 dispatch permutation 中已被量化为 INT8（per-token scale 在 `expertsScale`）；未前移时先 `XliteOpQuantDyn` 再进派发入口。w2 输入 `h2` 由 SiluAndMul 现算、恒为 BF16，不受前移影响，恒走 QuantDyn 路径。收益函数 `FusedGroupMatmulDequantBeneficial(m, n, k)` 现行判据为 **`m >= 4096 && k >= 1024 && n >= 2048`**。标定口径沿用 DevSpd+3us（SerDev/(FusDev+3us)，融合侧计一次 launch 气泡），依据 26.11.17 gmd_single 全网格——当前两段融合核实现下的全 5-g 重测：g∈{1,4,8,16,32} × m 10 档（1~16384）× (n,k)∈{1024,2048,4096,6144}² × native/vllm 共 1600 配置，correctness 全过，全网格 Dev+3us geomean 随 g 单调（g=1 0.926x → g=32 0.999x）。m≥4096∧k≥1024 放行区 geomean 1.058x（g=1）→ 1.134x（g=32；native 1.144x），封禁区（m<4096 或 k<1024）0.81~0.88x（融合亏损、保持串行）；放行区内残余回退点集中在 n=1024 小形状（最差 0.925x、4-9 个/g），`n >= 2048` 下限将其排除在放行区外；5-g（g=1..32）放行区审计均成立，g 不参与判据。生产命中面（GLM-5.2 w13 (4096,6144)/w2 (6144,2048)、DS-V4-Flash w13 (4096,4096)/w2 (4096,2048)）均满足三下限。e2e 佐证（GLM-5.2-w8a8 8 卡 A/B，13 配置）：串行锚点档（m<4096）ratio 0.994~1.000 零回退，融合档全胜（长 prefill +0.35~+1.15%，短序列大 batch bs16/32@512 +3.0~+3.1%）；e2e 命中面全部 n≥4096，n≥2048 下限不改变其派发。m 取 host 侧期望路由行数（均衡路由均值 tokens×top-k/EP，由模型侧 `expectedRows` 传入；精确值 sum(counts[start:end)) 仅设备侧可知，缓冲容量 x.shape[0] 为最坏情形不可用作 m）。

```text
AIC b:         per-expert matmul walk (输入已量化, 进入即开始; 与独立 group_matmul 相同) → out (FP16 视图)
               --C2V flag 8 per tile (挂 PIPE_FIX, 骑 CopyToGm 之后)-->
AIV 2b/2b+1:   dequant 镜像 walk (FP16 → BF16 原位, 乘 outScale)
               <--V2C flag 9 ack per 4 tiles (挂 PIPE_MTE2, 骑本批 GM 载入之后; AIC 每 4 tile 等一次, 限制 C2V set 流超前)--
```

## 输入输出参数

Python 入口 `fused_group_matmul_dequant(rt, in_, weights, scales, counts, start, end, out_dim, in_dim, output, out_scale = None, weight_nz = False, transpose = False)`（`csrc/_C.cpp`）。kernel 签名见 `FUSED_GROUP_MATMUL_DEQUANT_FUNC_DEFINE` 宏（[csrc/kernels/fused_group_matmul_dequant.h](../../csrc/kernels/fused_group_matmul_dequant.h)）:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| in_(x_quant) | 输入 | `[m, in_dim]` | int8 | 已量化的激活（`quant_dynamic` 或 dispatch 前移产出），行按 expert 分段（`counts` 前缀） |
| weights | 输入 | 每 expert `[out_dim, in_dim]`（或转置/NZ） | int8 | per-expert 权重指针数组 |
| scales | 输入 | 每 expert `[2*out_dim]` | float32 | FixPipe 反量化 scale（FP32 按 uint64 交错存储，同 group_matmul） |
| counts | 输入 | `[num_experts]` | int32 | per-expert 行数（kernel 读 uint32），`sum(counts[start:end))` 须等于 m |
| output | 输出 | `[m, out_dim]` | bfloat16 | 结果（AIC 写 FP16 视图、AIV 原位覆盖 BF16） |
| out_scale | 输入 | `[m]` | float32 | per-token scale（核外量化段产出、核内 dequant 读）；**可缺省**（`= torch::arg::none()`，与 `fused_quant_matmul_dequant` 同款）：缺省/None/空张量均走 dequant 无 scale 路径（同独立 dequant 的 hasScale=false） |
| weight_nz / transpose | 标量 | - | bool | 权重布局 |
| m0/n0/k0 | 标量 | - | uint64 | tiling 固定 m0=128 / n0=256 / k0=128×8/weightBits（单次 Mmad 的 K 量程，与 Matmul::Init 同步；与两段融合 group 算子一致，双侧必须一致） |

与串行链路的已知差异：动态 token 数收缩（`num`）不作用于本算子（与两段融合 group 算子一致，全部 m 行参与）。对齐约束：`in_dim % 32` 由量化侧入口保证（QuantDyn / 前移条件本身要求 hiddenSize%32==0）；`out_dim` 须满足 dequant 2D DMA 的 32B 列块约束（与独立 dequant 相同，GLM 形状天然满足）。

测试参考：[tests/kernels/fused_group_matmul_dequant.py](../../tests/kernels/fused_group_matmul_dequant.py)，golden 为串行两 kernel 链（同一份 INT8 输入 + per-token scale），覆盖 nz/transpose/有/无 out_scale 交叉、零计数 expert、极小路由批（m=1/2/5/8，m<g 且 NZ 部署布局）、matmul k 分块路径（k 超单 tile k 预算触发 kLoop≥2）与长序列。perf 参考：[tests/performance/perf_fused_group_matmul_dequant.py](../../tests/performance/perf_fused_group_matmul_dequant.py)。

## 支持的数据类型

- x `int8_t`、weight `int8_t`、out `bfloat16_t`；matmul 累加 `int32_t`、中间输出 `half`
- kernel 入口符号 `fused_group_matmul_dequant_int8_t`（host 经 `aclrtlaunch_...` 启动，`csrc/op.cpp` 的 `XliteOpFusedGroupMatmulDequant`）

## 实现原理

实现位于 [csrc/kernels/fused_group_matmul_dequant.h](../../csrc/kernels/fused_group_matmul_dequant.h)，类模板 `FusedGroupMatmulDequant<int8_t, int32_t, half, bfloat16_t>`，AIC 侧复用 `GroupMatmul`，AIV 侧复用 `Dequant` 子算子原语。expert/tile 遍历的唯一事实来源是共享骨架 `GroupTileWalker`（`csrc/kernels/group_tile_walker.h`），骨架为纯遍历、不含任何跨核调用；C2V/V2C 双向同步是融合壳内显式调用（`SetC2VFlag`/`WaitC2VFlag`/`SetV2CFlag`/`WaitV2CFlag`，定义在骨架头 flag id 旁），两侧 tile 循环都由融合壳自带、完全对称：AIC 侧经 `GroupMatmul::BeginExpert` 绑定专家后每 tile `RunTileByIdx` → `SetC2VFlag()`，并按背压周期 `WaitV2CFlag()`；AIV 侧以父核索引播种同一骨架，每 tile 先 `WaitC2VFlag()` 再调 `Dequant::RunGroupExpertTile`（半行块反量化，内部含空尾跳过），并按同一周期 `SetV2CFlag()` ack——两侧的 tile 认领序列（含 `coreOffset` 相位与零计数专家跳过）由同一份骨架构造保证一致，C2V flag 的 set/wait 计数因此逐对平衡。

- **无 V2C 输入交接**：输入 `xQuant` 由前序 kernel（QuantDyn 或 dispatch 前移的 permutation）在同流上产出，流序保证可见性，AIC 进入即开始 group walk。旧三段融合核的 V2C 粗粒度 barrier（AIV `SyncAll<true>` 聚合 + flag 9，`PIPE_MTE3` 骑量化 store）随核内量化段一并删除。
- **C2V per tile**（flag 8 挂 PIPE_FIX）：与两段融合 group 算子一致的已验证契约。AIC 侧**无退出回执**：MIX kernel 以全部核结束为准，AIC 可在配对 AIV 仍在 dequant 尾部 tile 时先行返回，与 dense 融合算子退出契约一致。
- **V2C backpressure ack**（flag 9 挂 PIPE_MTE2，每 `V2C_SYNC_INTERVAL=4` tile 一次，延迟 `FUSION_PIPELINE_DEPTH=1` 个同步周期，与 dense 侧 !654 修复同款标定）：AIC 的 C2V set 流若无背压可任意超前两个配对 AIV 的 wait 消费流，大 walk 下 ffts flag 置位-消费配对被打乱，尾部 wait 永不触发，kernel 以 AI Core "timeout or trap" 挂死（26.11 gmd 线上报、26.10 dense 侧 !609 同款故障形态）。AIV 每 4 个已消费 tile（含空尾跳过，其 wait 计数已发生）set 一次 ack，骑 MTE2 队列保证发出时本批 GM 载入已退休；AIC 每 4 tile 前标量 wait 一次，walk 结束后排空剩余 `DEPTH` 个 ack，两侧计数恒等式为 `ceil(T/4) = floor((T-1)/4) + 1`（T 为该 AIC-pair 认领的 tile 总数，计数跨 expert 连续累计）。
- **dequant 镜像 walk**：融合壳以 `walkBlockIdx = GetBlockIdx()/GetTaskRatio()`（父 AIC 索引）统一播种同一个 `GroupTileWalker` 骨架后，AIV 分支自带 tile 循环：每 tile 先显式 `WaitC2VFlag()`（排空 `CrossCoreWaitFlag(C2V_CROSS_CORE_FLAG)`）再调 `Dequant::RunGroupExpertTile`（`csrc/kernels/dequant.h`）按半行块（`halfM0 = m0/GetTaskRatio()`，`sub = GetBlockIdx()%2`）原位转换本 AIV 负责的行半块；方法内部的空行半块跳过发生在 wait 之后，保证等待计数与 AIC 的 set 计数恒等。每块 FP16 tile 落地即被配对 AIV 转换。行空间完全由 walker 给出（`Dequant::TaskTilesInit` 的 m 参数不被 group 镜像读取，仅绑定 GM 缓冲与行 stride n=kN）。
- **ffts mode id 路由**：MIX 组内路由官方 mode 为 `0x2`，C2V 方向统一用 `0x2`（CANN 自身实现只出现 `0x0` 与 `0x2`；早期版本 AIV 侧曾用非官方值 `0x3`，在 cann-9.0.1 上恰好同效，现已统一为 `0x2`）。flagId 8 与核内事件号空间独立。

## 性能特征（910B 实测，2026-10-15/16）

> **历史记录**：本节数据测自旧三段融合核（quant 在核内）对三 launch 串行链（quant_dynamic + group_matmul + dequant），仅作历史记录。26.11 起量化段拆出，本算子改为两段融合（对两 launch 串行链 group_matmul + dequant）；26.11.17 gmd_single 已按当前核实现完成全 5-g 重测（1600 配置，correctness 全过），收益函数据此定为 `m >= 4096 && k >= 1024 && n >= 2048`（标定与放行区/封禁区数据见「功能概述」）。

perf 矩阵：GLM MoE TP16 w13/w2 × hidden 7168/6144 × m 1~4096（g=16，transpose+NZ 部署布局），`perf_xlite_ops.py` DEVICE_TIME=False 口径（逐调用 wall，warmup 1 轮剔除、10 轮取中位）。

- **m=1~4096 全区间融合占优，无劣化点**：m=1/2/4/8 为 1.34~1.46x / 1.31~1.33x / 1.26~1.32x / 1.13~1.26x（3 次 launch 归一，小 m 收益最大），m=128/1024/4096 为 1.04~1.36x（最差点 w13 m=4096 1.04x，小 m 大 k 收益最大）。据此 10-16 撤掉了原 64 行派发门槛（`FUSED_GROUP_MIN_ROWS`）；
- 对归一前中间态（quant + 两段融合 matmul+dequant）为 1.05~1.11x，即融合 quant 再省一次独立 launch+sync（~40-90us）；
- 对串行链每 MoE 层 w13+w2（m=128/1024/4096）：h7168 省 122/108/146us、h6144 省 88/104/83us，按 78 层估算单 forward 省 ~6.5-11.4ms；对中间态每层省 h7168 58/74/97us（**7~9%**）、h6144 29/31/63us（**6~7%**）；
- 收益占比小于 dense 侧三段融合（decode 35%）：group kernel 单次调用覆盖全部专家、本身时延大，省一次 launch 的相对占比低。

> **26.10.29 设备时间口径修正**：上述 wall 口径（串行按 3 次 launch 归一）的"全区间占优"结论在 torch_npu.profiler 设备时间口径下不成立。384 配置全网格复测（`results/26.10.29/gmd_single`，串行 = 三核设备时间求和，融合侧另加 3µs 气泡修正）显示：m=1 融合仅 0.35~0.72x，m≤512 全带 ≤1.03x，且 n≥6144∧k≥4096 在 m≥4096 存在 0.78~1.02x 亏损区。派发已据此改为收益函数 `FusedGroupMatmulDequantBeneficial`（DevSpd+3us 口径标定，见「功能概述」），`XLITE_FUSED_GROUP_W8A8` 环境变量删除；本节 wall 数据仅作历史记录保留。

测量注意：共占设备时 kernel 可能被静默丢弃，perf 前确认卡空闲（HBM 亦须余量，曾因共占 <10MB 致 kernel 二进制加载失败）；`DEVICE_TIME=True` 给 wall 加 ~100us/调用，干净对比须置 False。空闲卡探测用 torch `mem_get_info`（容器内可见 16 设备，npu-smi 仅列 8 chip 且口径含 ~2.7GB 预留显存）；测试脚本经 `XLITE_TEST_DEVICE` 指定目标卡。

## 边界处理

- 零计数 expert 由 walk 自然跳过（group_matmul 既有行为）；空路由区间（区间内全部 expert 零计数）时两侧 walk 均零 tile，C2V 与 V2C 的 set/wait 计数由同一骨架构造保证恒等（walk 尾的 ack 排空对零 tile 直接跳过），不会挂起。
- dequant/matmul 子算子各自有 `mActual/nActual` 边界处理；走格由编译期宏 `FUSION_GROUP_MATMUL_DQ_LINEAR_SWIZZLE`（`kernels/fused_group_matmul_dequant.h`，现全局 0x601 = Nz count 6 分块蛇形）统一给定，两臂 walker 与 Matmul 解码同码（双射、bitwise 不变）；host 侧 fuse/skip 收益规则为 `m >= 4096 && k >= 1024 && n >= 2048`（标定依据见「功能概述」）。
- `out_scale` 传空时 dequant 镜像走 hasScale=false 路径（跳过 per-token 乘法），与独立 dequant 语义一致。
