# conv1d_and_silu

## 功能概述

融合的因果（causal）depthwise conv1d + SiLU 激活 + 可选 conv state 更新，用于线性注意力模型（如 Qwen3-Next / GLM-4.5 类 hybrid 模型）的线性注意力投影层。数学语义（`csrc/kernels/conv1d_and_silu.h:5-9`）：

```
concat = cat(state[K], input[S])          # 沿序列维拼接
out[i] = SiLU(dot(concat[i+1 : i+1+K], weight))
if updateState: state = concat[..., -K:]
```

即对每个通道独立做 K 抽头因果卷积（groups=C），再逐元素 SiLU（`x * sigmoid(x)`）。不使用 GM workspace 做拼接，state/input 直接搬入对齐的 UB 浮点缓冲。相比"Transpose → ConcatCol → Conv1d → SiLU → Transpose"的多 kernel 方案省去中间往返。

## 输入输出参数

kernel 入口（`csrc/kernels/conv1d_and_silu.h:594-597`）：

```cpp
conv1d_and_silu_##dtype(state, input, weight, output, batch, channels,
                        seqLen, kernelDim, updateState, queryStartLoc, queryLens)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| state | 输入/输出 | [B, C, K] | float16 / bfloat16 / float | 卷积左侧上下文（前 K-1 个虚拟位置）。`updateState=1` 时被原地更新为拼接序列的最后 K 列 |
| input | 输入 | 均匀模式：[B, S, C]；packed 模式：[T, C] | 同 state | 输入 token。通道 C 是最内连续维 |
| weight | 输入 | [C, 1, K] | 同 state | depthwise 卷积核，每通道 K 个抽头 |
| output | 输出 | 均匀模式：[B, S, C]；packed 模式：[T, C] | 同 state | SiLU 后的卷积输出 |
| batch | 标量 | - | uint32 | B |
| channels | 标量 | - | uint32 | C |
| seqLen | 标量 | - | uint32 | 均匀模式为 S；**0 表示 packed 混合长度模式**（host 约定，`conv1d_and_silu.h:504-509`） |
| kernelDim | 标量 | - | uint32 | K，限制 K <= 16 |
| updateState | 标量 | - | uint32 | 是否更新 state |
| queryStartLoc | 输入（packed） | [B]（>=8，int32） | int32 | packed 模式下各请求在 [T, C] 中的起始行 |
| queryLens | 输入（packed） | [B]（>=8，int32） | int32 | packed 模式下各请求的序列长度 |

限制（host 侧强制，`csrc/op.cpp:1867-1910`）：`kernelDim <= 16`；均匀模式 `seqLen <= 4096`；packed batch <= 256。均匀模式要求 state/input/output 均为 3D 且 B/C/S 匹配；packed 模式 input/output 为 2D，以 `queryStartLoc/queryLens`（int32）描述变长请求，二者 numel 需 >= batch（测试中 pad 到 8 个 int32 保证 32B 块拷贝不越界，`tests/kernels/conv1d_and_silu.py:68-76`）。

Python 调用方式：

- 均匀模式：`linear_att_conv_and_silu(rt, input, conv_state, weight, output)`（`tests/kernels/conv1d_and_silu.py:20`，input [B,S,C]）；
- packed 模式：`linear_att_conv_and_silu(rt, input, conv_state, weight, output, query_start_loc, query_lens)`（`tests/kernels/conv1d_and_silu.py:90-92`，input [T,C]）。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|-------|-----------|-------------|
| float16_t | `csrc/kernels/conv1d_and_silu_float16_t.cpp` | `conv1d_and_silu_float16_t` |
| bfloat16_t | `csrc/kernels/conv1d_and_silu_bfloat16_t.cpp` | `conv1d_and_silu_bfloat16_t` |
| float | `csrc/kernels/conv1d_and_silu_float.cpp` | `conv1d_and_silu_float` |

仅支持 `__DAV_C220_VEC__`（向量核），其他架构导出空实现（`csrc/kernels/conv1d_and_silu.h:604-622`）。四组 tensor dtype 必须一致（`EachXDtype`，`csrc/op.cpp:1914-1919`）。

## 实现原理

### 并行切分：通道为主维度

核心设计（`conv1d_and_silu.h:18-24` 注释）：**64 lane 的向量化维度是通道 C 而不是序列**。通道按 64（`VECTOR_MAX_NUM_OF_FP32`，`kBlock`）分块，`nBlocks = ceil(C/64)`，块间由多核按取模轮转分配：`if (cb % GetBlockNum() != GetBlockIdx()) continue;`（`conv1d_and_silu.h:531-534`）。这样每个卷积抽头是一次连续、32B 对齐的 64 通道向量乘加（vmul+vadd），替代了 [B,C,S] 布局所需的逐抽头 gather；直接消费/产出 [B,S,C] 也省掉了 kernel 前后两次 [B,S,C]<->[B,C,S] Transpose。

每个核内：`LoadWeights(c0, w)` 一次，然后串行处理所有 batch 的该通道块（`conv1d_and_silu.h:539-545`）。

### UB 缓冲布局

`Init`（`conv1d_and_silu.h:50-134`）在 UB 上静态规划（均 32B 对齐）：

| 缓冲区 | 大小 | 用途 |
|--------|------|------|
| stage_buf | `kMaxInputF * sizeof(Dtype)`（4096 元素） | GM<->UB dtype 转换的中转对齐缓冲 |
| w_f / w_reorg | `kBlock * kMaxKernel * 4B` | 权重 fp32 原始 [w, K] / 重排为 [K, kBlock]（抽头 j 对应一条连续 64-lane 向量） |
| state_f / state_reorg | 同上 | 卷积 state fp32 原始 / 重排 [K, kBlock] |
| win_raw（非 float） | `kWin * kBlock * sizeof(Dtype)`，`kWin = K-1+128` | 原始 dtype 的输入滑窗 tile |
| window_f | `kWin * kBlock * 4B` | fp32 输入滑窗 |
| state_win | `kMaxKernel * kBlock * 4B` | 更新 state 用的尾部输入窗 |
| new_state_f | `kMaxKernel * kBlock * 4B` | 新 state 拼装缓冲 [w, K] |
| acc_buf / calc_buf | 8*32B | 累加器 / SiLU 中间量 |
| out_tile | `kTile * kBlock * sizeof(Dtype)`，`kTile=128` | 输出 tile 暂存 |
| meta_start / meta_lens | `kMaxBatchMeta * 4B`（256） | packed 模式的 queryStartLoc/queryLens 副本 |

fp16/bf16 输入在 UB 内统一升精度为 fp32 计算（`ConvertTile` 用 `vconv_f162f32/vconv_bf162f32`，`conv1d_and_silu.h:321-334`）；输出经 `vconv_f322f16/vconv_f322bf16r` 舍回原 dtype。

### 数据搬运

- **分块滑窗**：`ProcessSequence`（`conv1d_and_silu.h:416-445`）把序列按 `kTile=128` 个输出位置分块。每块加载 `[loPos, hiPos]` 的输入窗：首块从位置 0 起且 UB 目的行偏移 `K-1`（前面留给 state），后续块从 `s0-K+1` 起以携带跨块上下文。
- **LoadTileInto**（`conv1d_and_silu.h:299-319`）：当通道块满 64 且行/通道均 32B 对齐时，用 `__set_dmi_config` 构造多 burst DMA 一次搬 `nPos` 行；否则退化为逐位置 `CopyGmToUbufAligned`。
- **LoadWeights/LoadState**（`conv1d_and_silu.h:338-372`）：经 `LoadGmToFloat` 搬入 fp32 后，用标量循环把 [w, K] 重排成 [K, kBlock]，使每个抽头 j 落在 `w_reorg + j*kBlock` 的连续 64-lane 向量上。
- **StoreTile**（`conv1d_and_silu.h:396-414`）：满块时多 burst DMA 写回，否则逐行 `CopyUbufToGmAligned`，步长为 C。

### 计算：逐位置 K 抽头乘加 + SiLU

`ComputePosition`（`conv1d_and_silu.h:376-392`）对输出位置 s：

```cpp
acc = 0
for j in 0..K-1:
    p = s + 1 + j                          # 拼接序列中的绝对位置
    tap = (p < K) ? state_reorg[p] : window_f[p - s0 - 1]   # 前段来自 state，后段来自输入窗
    acc += tap * w_reorg[j]                # 64-lane vmul + vadd
SiLU(out_tile + (s - s0))
```

`SiLU`（`conv1d_and_silu.h:191-212`）以向量指令实现 `x / (1 + exp(-x))`：`vmuls(-x)` → `vexp` → `vadds(1)` → `vdiv`，非 float dtype 再 `vconv` 舍回。

### state 更新

`updateState=1` 时 `WriteBackState`（`conv1d_and_silu.h:472-502`）：先 `LoadStateWin` 取输入尾部 K-1 个位置，再在标量域把新旧上下文拼成 `concat[..., -K:]`（前缀来自旧 `state_reorg`、后缀来自 `state_win`），转回 [w, K] 布局经 `StoreFloatToGm` 写回 GM state。

### 流水线同步

注释明确（`conv1d_and_silu.h:146-148`）：单独的 `pipe_barrier(PIPE_X)` **不能**排序跨 pipe（MTE2/V/MTE3/S）产生的 UB 数据，必须用显式事件旗标：

- EVENT_ID0：窗口/out_tile 流水上的 MTE2→V（`FlagMTE2V`）、V→MTE2（`FlagVMTE2`）、V→MTE3（`FlagVMTE3`）、MTE3→V（`FlagMTE3V`），在 `ProcessSequence` 的 load → convert → compute → store 各级间接力（`conv1d_and_silu.h:429-443`）；
- EVENT_ID1：标量（S）pipe 的 fence——权重/state 重排的标量写与 V/S 相互可见（`FlagMTE2S/FlagVS/FlagSV/FlagSMTE3`，`conv1d_and_silu.h:170-189`）；
- EVENT_ID2：`LoadGmToFloat/StoreFloatToGm` 内部 stage_buf 的 MTE2↔V 互斥（`conv1d_and_silu.h:216-292`）。

### packed（混合长度 decode）模式

`Packed()` 以 `seqLen == 0` 判定（GM 指针非空不可靠，CANN 可能传非空哑指针，`conv1d_and_silu.h:504-509`）。进入时把最多 256 项 `queryStartLoc/queryLens` 拷入 UB meta 缓冲，随后每 batch 的 `S = meta_lens[b]`、`tokenBase = meta_start[b]`（`conv1d_and_silu.h:543-546`）；`S <= 0` 或 `S > 4096` 的请求跳过。计算路径与均匀模式完全复用。
