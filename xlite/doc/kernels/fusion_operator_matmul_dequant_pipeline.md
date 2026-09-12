# fusion_operator_matmul_dequant_pipeline

## 功能概述

W8A8 量化矩阵乘与反量化的**跨核流水融合算子**：单个 MIX kernel 内 AIC（Cube）核计算 `matmul + fixpipe 反量化`（int8×int8 → int32 累加 × deqScale → fp16 中间结果写 GM），AIV（Vector）核逐 tile 等待 AIC 通知后将 fp16 中间结果反量化为 bf16（`out[i,:] = fp16中间[i,:] * outScale[i]`，outScale 可选）并原地写回同一 GM buffer。数学上等价于串行路径 `matmul_dequant → dequant`（`csrc/op.cpp:1497-1513` 的 fallback 分支），目的是让不同 AIC block 的 matmul 与 dequant 在时间上重叠（block b 的 AIV 做 dequant 时，block c 的 AIC 还在算 matmul），省去第二个 kernel 的 launch 与全量 GM 往返等待。

## 输入输出参数

kernel 入口（`csrc/kernels/fusion_operator_matmul_dequant_pipeline.h:99-102`）：

```cpp
fusion_operator_matmul_dequant_pipeline_##dtype(x, weight, out, bias, weightScale,
                                                outScale, num, m, n, k, weightNz,
                                                transposeWeight, m0, n0, k0)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x | 输入 | [M, K]（ND，行主序） | int8 | 已量化的激活矩阵 A |
| weight | 输入 | [N, K]（ND）或 NZ；transpose 时 [K, N] | int8 | 权重矩阵 B，语义同 matmul 的 y |
| out | 输入输出（原位） | [M, N] | bfloat16 | AIC 先写 fp16 中间结果（低 16bit 视图），AIV 原地转 bf16 覆盖；fp16/bf16 同为 2 字节，原地安全。host 侧**不做** `View(FP16)`/`View(BF16)` 切换（`csrc/op.cpp:1536-1539`） |
| bias | 输入（可选） | [N] | int32 | Mmad bias 表，传空 tensor 时 `hasBias=false` |
| weightScale | 输入（可选） | [2N] | float32（偶数位为 scale、奇数位补 0，即 uint64 视角的 fp32 流） | fixpipe 反量化 scale，约定同 matmul 的 deqScale |
| outScale | 输入（可选） | [M] | float32 | per-token（行）缩放，AIV 侧 vmuls 使用；传空 tensor 时跳过乘法 |
| num | 输入（可选） | [1] | int32 | 动态 token 数指针（pnumTokens），取 min(*num, M)；传空 tensor 时用 M |
| m, n, k | 标量 | - | uint64_t | GEMM 维度，host 侧 `n = transpose ? weight.shape[1] : weight.shape[0]`（`csrc/op.cpp:1528`） |
| weightNz / transposeWeight | 标量 | - | uint64_t | 同 matmul 的 nz / transpose |
| m0, n0, k0 | 标量 | - | uint64_t | host 侧 `PickMatmulTiling` 自动选择的分块（`csrc/op.cpp:1534`），与独立 matmul 完全同一策略 |

Python 调用方式（`tests/kernels/fusion_operator_matmul_dequant_pipeline.py:52`）：`fusion_operator_matmul_dequant_pipeline(rt, x, weight, out, bias, scale, weight_nz, transpose, out_scale, num)`。host 侧 `XliteOpFusionOperatorMatmulDequantPipeline`（`csrc/op.cpp:1515`）做 dtype/shape/参数完整性校验后直接 launch，不经过任何 dtype 视图转换。

模型路径开关：`XliteOpMatmulDeQuant` 中 `enableFused = rt.enableFusedDenseW8A8 && shape 为 2D && weightScale 非空`（`csrc/op.cpp:1498-1501`），运行时由环境变量 `XLITE_FUSED_DENSE_W8A8` 开启（`csrc/runtime.cpp:95-99`）；关闭时走串行 `matmul_dequant → dequant` fallback。

## 支持的数据类型

| Dtype（x/weight） | bias | out | 实例化文件 | kernel 符号 |
|------|------|------|------|------|
| int8_t | int32_t | bfloat16（kernel 内先 half 后 bfloat16） | `csrc/kernels/fusion_operator_matmul_dequant_pipeline_int8_t.cpp` | `fusion_operator_matmul_dequant_pipeline_int8_t` |

模板实例化为 `FusionMatmulDequantPipeline<int8_t, int32_t, half>`（`csrc/kernels/fusion_operator_matmul_dequant_pipeline.h:105`）。kernel 任务类型 `KERNEL_TYPE_MIX_AIC_1_2`（`fusion_operator_matmul_dequant_pipeline.h:104`）：每个物理 block = 1 个 AIC + 2 个 AIV，AIC 与 AIV 同时执行同一 kernel 代码，靠 `__DAV_C220_CUBE__` / `__DAV_C220_VEC__` 宏分成两条路径。

## 实现原理

### 整体结构

`FusionMatmulDequantPipeline` 是一个薄编排层（`csrc/kernels/fusion_operator_matmul_dequant_pipeline.h:16-96`），内部按核型持有不同 subOp：

- **AIC 路径**：`Matmul<int8_t, int32_t, half>` 子算子（复用独立 matmul 全部实现，含 bias/deqScale fixpipe 路径），外加一行跨核通知；
- **AIV 路径**：`Dequant<half>` 子算子（复用独立 dequant 的 fp16→bf16 行处理），外加 tile 映射改造与跨核等待。

两侧 subOp 的其余细节（L1/L0 布局、UB 布局、行内流水）见 [matmul.md](matmul.md) 与 [dequant.md](dequant.md)，本文只描述融合编排层。

### tiling：AIC 与 AIV 的行块配对

host 传入的 `m0/n0/k0` 与独立 matmul 完全一致（`PickMatmulTiling`，`csrc/op.cpp:1534`）。AIC 侧直接以 `(m0, n0)` 划分 tile：`tileCount_AIC = ceil(m/m0) * ceil(n/n0)`（`csrc/kernels/matmul.h:156-158`）。

AIV 侧行块减半：`Init` 中 `m0_aiv = m0 / GetTaskRatio()`（taskRatio=2，`fusion_operator_matmul_dequant_pipeline.h:26-28`），即一个 AIC m-tile 的 `m0` 行对半分给同组的 2 个 AIV。AIV 的 tile 总数为 AIC 的 2 倍：`mTiles = ceil(m / (m0_aiv*2)) * 2`、`nTiles = ceil(n/n0)`（`csrc/kernels/dequant.h:79-85`），`tileCount_AIV = mTiles * nTiles = 2 * tileCount_AIC`。

### 跨核同步与锁步 tile 映射（核心机制）

物理配对由硬件 flag 语义决定：`KERNEL_TYPE_MIX_AIC_1_2` 下 AIC block b 的 `CrossCoreSetFlag<0x2, PIPE_FIX>` 只唤醒同组的 AIV `2b` 与 `2b+1`（AIV 的 `GetBlockIdx() = get_block_idx()*2 + get_subblockid()`）。两侧调度循环（`fusion_operator_matmul_dequant_pipeline.h:79-80`）：

```cpp
AIC: for (a = GetBlockIdx(); a < tileCount_AIC; a += GetBlockNum())            // 步长 N
AIV: for (t = GetBlockIdx(); t < tileCount_AIV; t += GetBlockNum()*GetTaskRatio())  // 步长 2N
```

任意一轮迭代中 AIV t 与 AIC `a = t/2` 严格同物理 block、同轮次（**锁步**）。因此 AIV 的 tile 分解（`csrc/kernels/dequant.h:94-98`）必须先把线性索引映射到 AIC tile 再选行半块：

```cpp
aicIdx = idx >> 1;                          // 对应的 AIC tile 线性序
sub    = idx & 1;                           // 组内第几个 AIV → 负责哪个行半块
mIdx   = (aicIdx / nTiles) * 2 + sub;       // AIV 行块编号（行方向粒度是 m0/2）
nIdx   = aicIdx % nTiles;                   // 列块与 AIC tile 相同
```

`rowOffset = mIdx * m0_aiv` 恰为该 AIC m-tile 的第 sub 个半块。这样保证 **AIV 等待 flag 的核（block t/2）与写它所读数据的核是同一个**。若直接用独立模式的线性化 `mIdx = idx/nTiles`，会产生"等 A 核信号、读 B 核数据"的错位（历史上 15/24 case 失败、稀疏漂移坏点的 root cause，修复见 `dequant.h:94-98`）。

每个 tile 的同步时序（`fusion_operator_matmul_dequant_pipeline.h:58-70`）：

- **AIC**：`subOp.RunTileByIdx(tileIdx)` 完成（含 `CopyToGmWithDequant` 把 fp16 结果写 GM）后执行 `CrossCoreSetFlag<0x2, PIPE_FIX>(C2V_CROSS_CORE_FLAG)`（flagId=8，`fusion_operator_matmul_dequant_pipeline.h:13`）。通知指令挂在 FIX 管线队列尾，天然排在 CopyToGm 之后发出——被唤醒的 AIV 看到的 GM 数据必然已落地，无需额外的标量流等待；
- **AIV**：`CrossCoreWaitFlag(C2V_CROSS_CORE_FLAG)` 阻塞至同组 AIC 的通知，再执行 `subOp.RunTileByIdx(tileIdx)`（GM fp16 → UB → vconv/vmuls/vconv → GM bf16，见 [dequant.md](dequant.md)）。

flagId 8 的选择避开了 matmul（EVENT_ID0-5）与 dequant（EVENT_ID0-1）内部已占用的事件号；跨核 flag 是独立的 16 个编号空间，与核内 `set_flag/wait_flag` 不冲突。

### 数据流总览

```
AIC block b:  x,y GM --MTE2--> L1 --MTE1--> L0A/L0B --Mmad--> L0C(int32)
              --FIX: ×deqScale(VDEQF16)--> GM[out 的 fp16 视图, tile (mIdx_aic, nIdx)]
              --CrossCoreSetFlag(8)-->
AIV 2b/2b+1:  CrossCoreWaitFlag(8)
              GM fp16 半块 --MTE2--> UB --V: vconv_f162f32--> --V: vmuls(outScale)-->
              --V: vconv_f322bf16r--> UB bf16 --MTE3--> GM[out 同一地址, 原位覆盖]
```

流水收益：block b 的 AIV 做 dequant 时，其它 block 的 AIC 仍在算 matmul，两类工作跨核重叠；相比串行两 kernel，省去第二次 launch 与"全量 matmul 完成后才能开始 dequant"的全局串行点。

### 边界处理

- **m0 为奇数或 m0 < 2**：AIV 的 `m0/GetTaskRatio()` 下限保护为 1（`fusion_operator_matmul_dequant_pipeline.h:28`），此时 AIC/AIV 行块粒度不再严格 1:2，配对假设退化（当前 host 侧 `PickMatmulTiling` 产生的 m0 恒为 ≥32 的偶数，该分支仅防御）；
- **tileCount 为奇数**：`mTiles` 公式 `ceil(m/(m0_aiv*2))*2` 强制 AIV 行块成对（needPad 语义，`dequant.h:80`），多余的空 tile 在 `RunTileByIdx` 的越界检查（`dequant.h:106-108`）中直接返回，不会死等没有任务的 AIC flag；
- **尾部行列**：AIC 侧 `mActual/nActual` + 分形 pad（见 matmul.md）；AIV 侧 `actualRows = min(m-rowOffset, m0_aiv)`、`localCols = min(n-colOffset, n0)`（`dequant.h:109-110`），行内列尾块按精确字节数走 `*_align_b16` 原语（见 dequant.md）；
- **num 动态 token**：`pnumTokens` 非空时 AIV 的 m 收缩为 min(*num, m)（`dequant.h:73-76`），跳过 padding 行。

## 关键代码位置

| 内容 | 位置 |
|------|------|
| 编排类与跨核同步 | `csrc/kernels/fusion_operator_matmul_dequant_pipeline.h:16-96` |
| AIC→AIV 通知 / AIV 等待 | `csrc/kernels/fusion_operator_matmul_dequant_pipeline.h:58-70` |
| AIV m0 减半 | `csrc/kernels/fusion_operator_matmul_dequant_pipeline.h:26-28` |
| AIV 锁步 tile 映射 | `csrc/kernels/dequant.h:94-98` |
| AIV 成对 mTiles（needPad） | `csrc/kernels/dequant.h:79-83` |
| kernel 入口（MIX_AIC_1_2） | `csrc/kernels/fusion_operator_matmul_dequant_pipeline.h:98-109` |
| host 侧封装与校验 | `csrc/op.cpp:1515-1544` |
| 模型路由开关（enableFused） | `csrc/op.cpp:1497-1513`、`csrc/runtime.cpp:95-99` |
| Python 绑定 | `csrc/_C.cpp:2831-2834`、`xlite/_C.pyi:2411` |
| 正确性测试 | `tests/kernels/fusion_operator_matmul_dequant_pipeline.py` |
