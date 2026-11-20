# fused_quant_matmul_dequant

## 功能概述

融合三段式 W8A8 dense 管线（quant → matmul → dequant）为单次 launch：AIV 侧先把 BF16 激活量化为 INT8（静态 per-channel 或动态 per-token 二选一），AIC 侧做 INT8 matmul（FixPipe 乘 deq_scale 反量化输出 FP16），AIV 侧再反量化为 BF16。相比 [fusion_operator_matmul_dequant_pipeline](fusion_operator_matmul_dequant_pipeline.md)（输入已是 INT8），本算子把上游量化 kernel 也吸收进来，省去两次独立 launch。数学语义与串行链路 `quant/quant_dynamic → matmul_dequant → dequant` 一致。

```text
AIV t (全部):  quant(x 行段 [t*chunk, (t+1)*chunk)) → xQuant (+ per-token scale → outScale, dyn)
               --SyncAll<true> → V2C flag 9 (mode 0x2, 每 AIV set 一次, AIC wait 一次)-->
AIC b:         grid-stride matmul tiles → out (FP16 视图)
               --C2V flag 8 per tile (挂 PIPE_FIX, 骑 CopyToGm 之后)-->
AIV 2b/2b+1:   dequant(out FP16 → BF16 原位, dyn 乘 outScale)
```

## 输入输出参数

Python 入口 `fused_quant_matmul_dequant(rt, x, quant_scale_rec, quant_offset, weight, out, bias, deq_scale, x_quant, out_scale, num, dyn_quant, weight_nz, transpose)`（`csrc/_C.cpp`）。kernel 签名见 `FUSED_QUANT_MATMUL_DEQUANT_FUNC_DEFINE` 宏（[csrc/kernels/fused_quant_matmul_dequant.h](../../../csrc/kernels/fused_quant_matmul_dequant.h)）:`weight_nz` 已废弃并将被移除,传入值被忽略,`weight` 的 FRACTAL_NZ 存储格式由 `InitXTensor` 自动检测。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| x | 输入 | `[m, k]` | bfloat16 | 原始激活 |
| quant_scale_rec / quant_offset | 输入 | `[k]` | bfloat16 | 静态模式量化 scale 倒数/偏置；dyn 模式传空 |
| weight | 输入 | `[n, k]`（或转置/NZ） | int8 | 权重 |
| x_quant | workspace | `[m, k]` | int8 | 量化输出（AIV 写、AIC 读），host 分配 |
| out | 输出 | `[m, n]` | bfloat16 | 结果（AIC 写 FP16 视图、AIV 原位覆盖 BF16） |
| bias | 输入(可选) | `[n]` | int32 | matmul bias |
| deq_scale (weight_scale) | 输入(可选) | `[2n]` | float32 | FixPipe 反量化 scale（FP32 按 uint64 交错存储，同 matmul） |
| out_scale | workspace | `[m]` | float32 | dyn 模式 per-token scale（quant 写、dequant 读）；静态模式传空 |
| num | 输入(可选) | `[1]` | uint32 | 动态 token 数，dequant 侧行收缩（同独立 dequant） |
| dyn_quant | 标量 | - | bool | true=动态 per-token；false=静态 per-channel |
| weight_nz | 标量 | - | bool | 已废弃,传入值被忽略（NZ 由自动检测决定） |
| transpose | 标量 | - | bool | 权重是否转置存储 |
| m0/n0/k0 | 标量 | - | uint64 | tiling，默认走 `PickMatmulTiling`（与独立 matmul 一致） |

测试参考：[tests/kernels/fused_quant_matmul_dequant.py](../../../tests/kernels/fused_quant_matmul_dequant.py)，golden 为串行三 kernel 链，覆盖 nz/transpose/dyn/static 交叉、quant 尾路径与长序列。perf 参考：[tests/performance/perf_fused_quant_matmul_dequant.py](../../../tests/performance/perf_fused_quant_matmul_dequant.py)。

## 支持的数据类型

- x `bfloat16_t`、weight `int8_t`、out `bfloat16_t`；matmul 累加 `int32_t`、中间输出 `half`
- kernel 入口符号 `fused_quant_matmul_dequant_int8_t`（host 经 `aclrtlaunch_...` 启动，`csrc/op.cpp` 的 `XliteOpFusedQuantMatmulDequant`）

## 实现原理

实现位于 [csrc/kernels/fused_quant_matmul_dequant.h](../../../csrc/kernels/fused_quant_matmul_dequant.h)，类模板 `FusedQuantMatmulDequant<int8_t, int32_t, half, bfloat16_t>`，AIC 侧复用 `Matmul`，AIV 侧复用 `Quant`/`QuantDyn`/`Dequant` 子算子原语。

- **双侧均衡**：AIC 跑与独立 matmul 相同的 grid-stride（`idx ≡ b mod N`）；AIV 量化行按全 AIV 均分连续段（`chunk = ceil(m/总AIV数)`）——量化并行度与独立 kernel 相同，matmul 负载与独立 matmul 相同。
- **V2C 粗粒度 barrier**：AIV 全部完成量化后 `AscendC::SyncAll<true>`（AIV-only 软屏障）聚合，每个 AIV 向本组 AIC set flag 9（挂 PIPE_MTE3 骑量化 store 之后——ffts 屏障是本平台唯一带数据可见性保证的跨核释放），AIC 单次 `CrossCoreWaitFlag<0x2>` 即可：mode 2 在 AIC 侧的 wait 是聚合语义——本组**两个 AIV 都 set** 后才放行；每个 set 又被 SyncAll<true> 排在全体量化完成之后（量化行段跨全体 AIV 均分，非仅本组），故两个 set 落地即代表全部行段就绪（与 CANN 自身 `SyncAll` 的 cube↔vector 分层栅栏同范式：mode 0 管"全体完成"，mode 2 管"组内通知"）。细粒度 per-m-tile 交接在本平台不可行（无标量 atomic / MTE1 不过 L2 / ffts flag 组内路由）。
- **C2V per tile**（flag 8 挂 PIPE_FIX，骑 CopyToGm 之后）：AIC 每完成一个 matmul tile 通知配对 AIV 做 dequant；AIV 的 dequant 循环逐 tile `CrossCoreWaitFlag<0x2>(C2V_CROSS_CORE_FLAG)`，结束后 `dequantOp.WaitFlags()` 只清算核内 flag。早期版本的 **V2C_DONE（flag 10）AIC 退出回执已删除**：AIC 不再等待配对 AIV 完成最后的 dequant tiles 即可退出（其全部产出经挂 PIPE_FIX 的 C2V set 保证落地后才唤醒 AIV），跨核 flag 只剩 8/9 两个。
- **AIV 两阶段串行**：全部量化（行段跨 AIV 均分；行内 k 按 UB 容量分 chunk，每 AIV 仅分得单行时走无 ping-pong 的单行路径）→ SyncAll → dequant 锁步，规避 quant/dequant 子算子的 UB 基址与 EVENT_ID 复用冲突。
- **ffts 同步模式与 flagId**：`CrossCoreSetFlag/WaitFlag` 的模板参数是 ffts 同步模式 modeId，函数实参才是 flagId（CANN 接口签名 `CrossCoreSetFlag<uint8_t modeId, pipe_t pipe>(uint16_t flagId)`）。本平台 MIX 块内 AIC↔AIV 双向统一用 mode `0x2`（子块同步模式，CANN 自身 `SyncAll` 的 cube↔vector 通知同款），set/wait 两侧模板参数须一致——`CrossCoreWaitFlag` 不写模板参数会落到默认 mode 0。flagId 8/9 为 0~15 编号空间，与核内事件号空间独立。

### 片上缓冲分配与占用汇总（AIC L1/L0A/L0B/L0C、AIV UB）

`Init`（`csrc/kernels/fused_quant_matmul_dequant.h` 的 `FusedQuantMatmulDequant::Init`）按核型分派：AIC 侧（`__DAV_C220_CUBE__`）复用 `Matmul::Init`（`csrc/kernels/matmul.h` 的 `Matmul::Init`）的 L1/L0 静态排布（L1 上 `l1aBuf`/`l1bBuf` 两对乒乓自 0 累进，`l1BiasBuf`/`l1DeqScaleBuf` 共享 `l1bBuf[0]` 槽位，L0A/L0B/L0C 各自从 0 起排）；AIV 侧（`__DAV_C220_VEC__`）两阶段串行复用同一 UB——量化阶段 `QuantDyn`/`Quant` 的布局在 `TaskTilesInit` 中按 k/行数动态推导（`csrc/kernels/quant_dyn.h` 的 `QuantDyn::TaskTilesInitMultiRow`、`csrc/kernels/quant.h` 的 `Quant::TaskTilesInitMultiRow`），反量化阶段 `Dequant` 的布局在 `Init` 中固定（`csrc/kernels/dequant.h` 的 `Dequant::Init`），两阶段基址都从 UB 0 起、SyncAll 锁步保证不共存。MIX 块内 AIC 与 AIV 分属不同物理核，L1/L0 与 UB 互不挤占。硬件容量（910B/dav C220，NpuArch 2201，CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`）：L0A 64KB、L0B 64KB、L1 512KB，L0C 总容量 256KB 为 CO1/CO2 两区之和，mmad 累加（`TPosition::CO1`）可用的 CO1 区为 **128KB**，下文 L0C 占用均按 CO1 128KB 计；AIV UB 预算取 `UB_SIZE=192KB`（`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`，4KB×48 bank），每个 AIV subblock 各持一份独立 UB、布局相同。

dtype 记法（宏 `FUSED_QUANT_MATMUL_DEQUANT_FUNC_DEFINE` 固定 `FusedQuantMatmulDequant<int8_t, int32_t, half, bfloat16_t>`）：AIC 侧 `Matmul<int8_t, int32_t, half>`；AIV 侧量化 `QuantDyn<bfloat16_t>`/`Quant<bfloat16_t>`（XDtype=bfloat16_t）、反量化 `Dequant<half>`。典型配置取 GLM-5.1 TP16 per-rank dense/shared-expert 动态 W8A8（`tests/performance/perf_fused_quant_matmul_dequant.py` 的 `MODEL_SIZES` 中 `dense_gate_up`：m=4096、n=1536、k=6144，dyn outScale、无 bias）：host `PickMatmulTiling` 产出 m0=128、n0=256、k0=512；AIV 量化阶段 k=6144、总 AIV 数=2×AIC_NUM（本机 AIC_NUM=25）→ tileRows=82>1 走多行路径，`QuantDyn` 行数 rows=2；反量化阶段 `nTile` 固定 12288、AIV 行块 m0/GetTaskRatio()=64。仅 int8 一个实例化，占用列取单列。

**AIC L1（A1/B1/C1，容量 512KB）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `l1aBuf` | `(m0, kDtileSize)`=`(128, 2×k0)`=`(128, 1024)` | int8_t | 2（事件0/1） | `m0*kDtileSize*sizeof(Dtype)` | 262144 |
| `l1bBuf` | `(n0, k0)`=`(256, 512)` | int8_t | 2（事件2/3） | `n0*k0*sizeof(Dtype)` | 262144 |
| `l1BiasBuf` | `(n0,)`=`(256,)`，仅 bias 路径 | int32_t | 1（事件4） | `n0*sizeof(MatDtype)` | 0（与 `l1bBuf[0]` 共享槽位） |
| `l1DeqScaleBuf` | `(n0,)`=`(256,)` | uint64_t | 1（事件5） | `n0*sizeof(uint64_t)` | 0（与 `l1bBuf[0]` 共享槽位） |
| **合计** | | | | | **524288** |

**AIC L0A（A2，容量 64KB）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `l0aBuf` | `(m0, kQtileSize)`=`(128, 128)` | int8_t | 2（事件0/1，kIdx2） | `m0*kQtileSize*sizeof(Dtype)` | 32768 |
| **合计** | | | | | **32768** |

**AIC L0B（B2，容量 64KB）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `l0bBuf` | `(n0, kQtileSize)`=`(256, 128)` | int8_t | 2（事件0/1，kIdx2） | `n0*kQtileSize*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C（CO1 区，容量 128KB；总容量 256KB 含 CO2）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `l0cBuf` | `(m0, n0)`=`(128, 256)` | int32_t | 1（事件0） | `m0*n0*sizeof(MatDtype)`（Mmad 分形累加，按满 tile 计） | 131072 |
| **合计** | | | | | **131072** |

**AIV UB——阶段一（动态量化 `QuantDyn`，k=6144、rows=2；每 AIV 一份）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `x1`/`x2` | `(rows, kPad)`=`(2, 6144)` | bfloat16_t | 2（事件0/1，xId） | `rows*kPad*sizeof(XDtype)` | 49152 |
| `xf32` | `(rows, kPad)` | float | 1 | `rows*kPad*sizeof(float)` | 49152 |
| `z1`/`z2` | `(rows, kPad)` | int8_t | 2（事件0/1，eId） | `rows*kPad*sizeof(int8_t)` | 24576 |
| `xAbs` | `(kPad,)` | float | 1 | `kPad*sizeof(float)` | 24576 |
| `xf16` | `(rows, kPad)` | half | 1 | `rows*kPad*sizeof(half)` | 24576 |
| `rowAbs` | `(rows*8,)` | float | 1（32B 对齐） | `rows*8*sizeof(float)` | 64 |
| `scaleStage` | `(2*rows,)` | float | 1 | `2*rows*sizeof(float)` | 16 |
| **合计** | | | | | **172112** |

静态变体 `Quant`（同 k=6144，`csrc/kernels/quant.h` 的 `Quant::TaskTilesInitMultiRow`）布局共 `12*(rows+1)*kPad` 字节（rMax=1 → rows=1，即 147456 B）；单行路径（tileRows==1）为 `21*kPad`。两变体的行数上限公式 `(UB_SIZE-4*kPad)/(12*kPad+40)`（QuantDyn）与 `(UB_SIZE-12*kPad)/(12*kPad)`（Quant）即 UB 反推，kernel 内 `assert(... <= UB_SIZE)` 兜底。

**AIV UB——阶段二（反量化 `Dequant<half>`，固定布局；每 AIV 一份）**

`nTile = DEQUANT_N_TILE = 12288`：in（half）+tmp（f32）+out（bf16）三段共 8B/元素，两套乒乓恰好填满 192KB（`csrc/kernels/dequant.h` 的 `DEQUANT_N_TILE` 注释）。

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `inUbBuf` | `(nTile,)`=12288 | half | 2（事件0/1） | `nTile*sizeof(half)` | 49152 |
| `tmpUbBuf` | `(nTile,)` | float32_t | 2（事件0/1） | `nTile*sizeof(float32_t)` | 98304 |
| `outUbBuf` | `(nTile,)` | bfloat16_t | 2（事件0/1） | `nTile*sizeof(bfloat16_t)` | 49152 |
| **合计** | | | | | **196608** |

**占用汇总**（各表占用列之和 vs 硬件容量；UB 取两阶段串行复用的峰值；字节数）：

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| L1（AIC） | 512 KB | 524288 B | 100% |
| L0A（AIC） | 64 KB | 32768 B | 50% |
| L0B（AIC） | 64 KB | 65536 B | 100% |
| L0C（CO1，AIC） | 128 KB | 131072 B | 100% |
| UB（AIV，两阶段取大） | 192 KB | 196608 B | 100% |

要点：

- **AIV UB 的峰值恒在反量化阶段**：`Dequant<half>` 固定布局恰好顶满 192KB（100%），量化阶段（QuantDyn 172112 B=87.5%、静态 Quant 147456 B=75%）低于它——UB 预算由 `DEQUANT_N_TILE=12288` 常量封顶，与 m/n/k 无关；
- AIC 侧默认分块恰好同时顶满 L1、L0B 与 L0C CO1（同 [matmul.md](matmul.md) 的容量分析），L0A 半满；
- 三段融合不改任何子算子的静态布局：两侧 UB 与 L1/L0 排布与独立 quant/matmul/dequant 完全一致，融合层只改调度与跨核同步；AIV 两阶段 UB 基址都从 0 起、串行复用，「量化全量完成（SyncAll）后才进入 dequant 锁步」正是其容量前提；
- bias/weightScale 走 CO1 之外的 C2 Bias Table（1KB）与 C2PIPE2GM fixpipe buffer（2KB）两个固定专用区，`n0 ≤ 256` 钳制的容量来源同 [matmul.md](matmul.md)。

## 性能特征（910C 实测，2026-10-13）

对 `quant + 现有两段融合`（XLITE_FUSED_DENSE_W8A8 开启路径）：

- GLM 全部 8 个真实 size（含 attn 静态 ×4）每 forward 节省：**decode(m=128) ~13ms（35%）**、**prefill(m=2048) ~11ms（18%）**；
- 逐 shape 全区间 0.61~0.93x 无劣化（收益 ~90% 来自省 launch/sync，device 纯 kernel 时间基本持平——粗粒度 barrier 无重叠，kernel 计算量不变）；
- 测量注意：共占设备时 kernel 可能被静默丢弃，perf 前 `npu-smi` 确认空闲；`perf_xlite_ops.py` 的 `DEVICE_TIME=True` 给 wall 加 ~100us/调用，干净对比须置 False。

## 边界处理

- AIV 行段 `rowStart >= m` 跳过、`rowEnd` 向 m 截断；dequant/matmul 子算子各自有 `mActual/nActual` 边界处理。
- k 非 32 的倍数时 host 直接抛错，无逐行回退路径：静态/dyn 量化段均为 batched engine（2D DMA 以 32B 为单元），两条路径一致（`csrc/op.cpp` 的 `XliteOpFusedQuantMatmulDequant`，关键字 `must be a multiple of 32`）。
- V2C 的 set/wait 无条件执行（AIC tiles 为零时同样等待），保证 flag 计数平衡。
- quant 按全量 m 量化（与串行链路一致）；`num` 仅作用于 dequant 侧行收缩。
