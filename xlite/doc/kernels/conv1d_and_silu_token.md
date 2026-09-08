# conv1d_and_silu_token

## 功能概述

按 token 行并行的因果 depthwise conv1d + SiLU，面向**均匀长度 prefill**（seqLen >= K）场景；与通道并行的 `conv1d_and_silu` 语义相同（`out[i] = SiLU(dot(concat[i+1 : i+1+K], weight))`，`concat = cat(state[K], input[S])`），但以 token-major `[T, C]` 布局直接计算，专为高吞吐 prefill 优化（P3-A v2，设计文档见 `.xlite-opt/plans/p3a-token-parallel-rewrite-20260818.md`）。本 kernel 只算卷积+SiLU 输出；conv state 更新由 host 随后另行启动的独立 kernel `conv1d_state_update` 完成（跨核 GM 竞争无法在单次 launch 内安全合并，launch 边界是唯一安全的核间屏障，`csrc/kernels/conv1d_and_silu_token.h:22-26`、`79-81`）。

## 输入输出参数

kernel 入口（`csrc/kernels/conv1d_and_silu_token.h:593-609`），同一实例化文件定义两个 kernel：

```cpp
conv1d_and_silu_token_##dtype(state, input, weight, output, batch, channels, seqLen, kernelDim)
conv1d_state_update_##dtype(state, input, batch, channels, seqLen, kernelDim)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| state | 输入（conv kernel）/ 输入输出（state kernel） | [B, C, K] | float16 / bfloat16 / float | 卷积上下文。conv kernel 只读；state kernel 原地写入 `state[b][c][:] = input 的最后 K 行` |
| input | 输入 | [T, C]，T = B*S，token-major | 同 state | 输入（通常为 mix_qkv 投影的 reshape 视图） |
| weight | 输入 | [C, 1, K]（或 [C, K]） | 同 state | depthwise 卷积核；kernel 按 `weight + cb*1024*K` 连续读取 |
| output | 输出 | [T, C] | 同 state | SiLU 后输出 |
| batch | 标量 | - | uint32 | B |
| channels | 标量 | - | uint32 | C，**必须为 1024 的倍数**（host 强制，`csrc/op.cpp:1959-1963`） |
| seqLen | 标量 | - | uint32 | 均匀 S，要求 `S >= K` 且 `B*S == T`（`csrc/op.cpp:1954-1958`） |
| kernelDim | 标量 | - | uint32 | K，仅允许 {1, 2, 4}（host 强制，`csrc/op.cpp:1949-1953`） |

窗口语义（`conv1d_and_silu_token.h:40-46`）：对 batch 内相对行 i，tap k 的来源为 `i+1+k < K → state[i+1+k]`，否则 `input` 行 `i-K+1+k`（同 batch）。

Python 调用方式（`tests/kernels/conv1d_and_silu_token.py:44`）：

```python
linear_att_conv_and_silu_token(rt, mix_qkv, conv_state, weight, output, seq_len)
```

host 侧（`csrc/op.cpp:1931-2003`）按 `coreNum = min(aivNum, B*S)` 启动 conv kernel；若 `updateState` 再按 `stateCores = min(aivNum, B * (C/1024))` 启动 state kernel。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|-------|-----------|-------------|
| float16_t | `csrc/kernels/conv1d_and_silu_token_float16_t.cpp` | `conv1d_and_silu_token_float16_t` / `conv1d_state_update_float16_t` |
| bfloat16_t | `csrc/kernels/conv1d_and_silu_token_bfloat16_t.cpp` | `conv1d_and_silu_token_bfloat16_t` / `conv1d_state_update_bfloat16_t` |
| float | `csrc/kernels/conv1d_and_silu_token_float.cpp` | `conv1d_and_silu_token_float` / `conv1d_state_update_float` |

仅支持 `__DAV_C220_VEC__`，其他架构导出空实现（`conv1d_and_silu_token.h:610-636`）。dtype 约束见 `csrc/op.cpp:1976-1989`。

## 实现原理

### 设计动机（文件头注释，`conv1d_and_silu_token.h:4-33`）

- 通道并行的 packed 路径按 stride-C 跨 token gather 通道（每元素 2B DMA 进 32B UB 槽，DMA 效率约 6%），在 S=512 时比它替换的两次 Transpose 往返代价更高（实测 TTFT +4.4%）；
- v1 token kernel 失败（9.08ms vs 1.07ms）：每个 (row, cb) 重复加载并标量转置权重块（约 41 万次标量操作/核/层）、标量收集 64-lane 输出（约 10 万次）、行跨核切分导致 4 倍输入 DMA 放大；
- v2 要点：cb 外层/行内层循环（权重块每 (core, cb) 只加载并 vgather 转置一次，零标量循环）、每核连续行段 + 滑窗、state 更新独立 kernel、每次 vconv/vgather 生产者后接硬 V-pipe fence（裸 `pipe_barrier(PIPE_V)` 在该芯片上不能排序 vconv UB 写回与后续向量读，实测出现 chunk 尾部离散的陈旧元素）。

### 并行切分：连续行段 per core

`Process`（`conv1d_and_silu_token.h:353-412`）：`rowsPerCore = ceil(totalRows/blockNum)`，每核认领连续行段 `[rowStart, rowEnd)`，段内再按 batch 边界拆子段（行段不跨 batch，保证窗口上下文正确）。每子段内 `for cb in 0..numChanBlocks`：`LoadWeightBlock(cb)` → `PrefillCtx(b, s0, cb)` → 按 4 行一 chunk 循环 `LoadChunk → ComputeChunk → StoreChunk`（→`SlideCtx`）。

### UB 布局与常量

常量（`conv1d_and_silu_token.h:68-73`）：`kMaxKernel=4`、`kChanBlock=1024`（CB）、`kBlock=64` lane、`kRows=4`（R，每 chunk 行数）、`kCombRows=K-1+R=7`、`kGroups=16`。UB 总预算约 110KB < 192KB（`conv1d_and_silu_token.h:99-102`）：

| 缓冲区 | 布局 | 用途 |
|--------|------|------|
| combined | `[K-1+R][CB]` fp32 | 滑动窗口：前 K-1 行上下文 + R 行 chunk |
| raw_chunk | `[R][CB]` Dtype | chunk DMA 暂存 |
| w_raw / w_f | `[CB][K]` | 权重原始 / fp32（channel-major） |
| w_t | `[K][CB]` fp32 | 权重 tap-major 转置（tap k 为连续 1024 通道向量） |
| out_stage | `[R][CB]` Dtype | 输出暂存 |
| s_raw / s_f | `[CB][K]` | state 暂存 / fp32 |
| acc_buf / calc_buf / out_f | 64-lane fp32 | 乘加 / SiLU 中间量 |
| off_ramp_k | 64 x uint32 | vgather 偏移斜坡：lane p → `p*K*4` 字节 |

### 权重加载与 vgather 转置

`LoadWeightBlock`（`conv1d_and_silu_token.h:211-232`）：DMA 取 `[CB][K]` 块（非 float 先 vconv 升 fp32），随后对每个 tap k、每 64 通道组 g 用一条 `vgather`（偏移斜坡 `off_ramp_k[p] = p*K*4`，即 lane p 读 `base + p*K*4` 字节）把 `w_f[c*K+k]`（c = g*64..g*64+63）聚到 `w_t[k*CB + g*64]`——一次 vgather 完成 64 个 strided 元素的转置搬运，替代 v1 的标量双循环。前后各接 `XLITE_V_FENCE()`。

### 滑窗流水（每 chunk）

1. **PrefillCtx**（`conv1d_and_silu_token.h:235-275`）：为子段起始行 s0 填 `combined[0..K-2]`：batch 内相对虚拟行 `v = s0-(K-1)+j >= 0` 时从 GM input 行取，否则从 state 取（`combined[j][c] = state[b][c][K+v]`）——state 同样经 vgather（列 `K+v`）按 tap 抽出。
2. **LoadChunk**（`conv1d_and_silu_token.h:277-289`）：`DmaInRows` 用 `copy_gm_to_ubuf` 的 strided 多 burst 配置（`lenBurst = CB*sizeof/32`，`srcStride = (C-CB)*sizeof/32`）一次取 rTake 行（<=4），非 float 经 `ConvUbToFloat` 升精度到 `combined[K-1]` 起。
3. **ComputeChunk**（`conv1d_and_silu_token.h:292-313`）：对每行 j、每 64 通道组 g：`vmul(acc, win[j], w_t[0])` 后 K-1 次 `vmul+vadd` 累加，随后 `SiLU64` 写 `out_stage`。窗口行取 `combined[j .. j+K-1]`（滑窗保证 tap k 的行 `winBase + k*CB`）。
4. **StoreChunk**（`conv1d_and_silu_token.h:316-328`）：逐行 `CopyUbufToGmAligned` 写 GM（整行 CB 通道连续，天然对齐）。
5. **SlideCtx**（`conv1d_and_silu_token.h:331-351`）：下一 chunk 存在时，把 `combined` 后 K-1 行前移为首 K-1 行（`vadds` 加 0 的向量拷贝），维持窗口滑动。

`SiLU64`（`conv1d_and_silu_token.h:161-182`）与通道并行版相同：`vmuls(-x) → vexp → vadds(1) → vdiv`，非 float 再 vconv 舍回。

### 流水线同步

- `XLITE_V_FENCE()`（`conv1d_and_silu_token.h:56-62`）：`set/wait(PIPE_V, PIPE_S)` 往返的硬 V-pipe 栅栏——本芯片上裸 `pipe_barrier(PIPE_V)` 不保证 vconv/vgather 的 UB 写回对后续向量读可见（`conv1d_and_silu_token.h:25-31` 注释），所有 vgather/vconv 生产者之后必须加。
- `DmaIn/DmaInRows`（`conv1d_and_silu_token.h:186-208`）：先 `pipe_barrier(PIPE_V)` 等先前 V 读释放暂存缓冲，DMA 后 `MTE2→V` EVENT_ID2 旗标通知数据就绪。
- out_stage 复用由 `MTE3→V` EVENT_ID0 管理：`ComputeChunk` 开头 wait（上次 DMA-out 排空），`StoreChunk` 尾部 set；`Process` 末尾再补一次 wait 保持事件配平（`conv1d_and_silu_token.h:410-411`）。
- off_ramp_k 由标量写一次后经 `S→V` EVENT_ID1 旗标交给 V pipe（`conv1d_and_silu_token.h:366-370`）。

### state 更新独立 kernel（conv1d_state_update）

`XliteConvStateUpdate`（`conv1d_and_silu_token.h:446-591`）：单元 `(b, cb)` 轮转分配到核（`for (u = blockIdx; u < B*numChanBlocks; u += blockNum)`）。对每个单元：

1. 取 batch b 的最后 K 行输入（`input[(b*S + S-K+j) * C + cb*CB]`），逐行 DMA+vconv 到 `rows_f[j]`（[K][CB] fp32）；
2. 反向 vgather 打包：偏移 `off_tile[p] = (p%K)*CB*4 + (p/K)*4`，lane p（组 g）读 `rows_f[p%K][g*(64/K) + p/K]`，把 `[K][CB]` 行数据聚成 GM 所需的 `[CB][K]` tile（`conv1d_and_silu_token.h:532-539`）；
3. 非 float 先 vconv 舍回 dtype（tile_raw），再整块 `CopyUbufToGmAligned` 写 `state + (b*C + cb*CB)*K`。

独立 launch 的原因：conv 主 kernel 各核读 state GM 作窗口上下文，若同 launch 内末行核改写 state 会与其他核竞争，两次 launch 的边界即天然核间屏障（`conv1d_and_silu_token.h:440-445` 注释、`csrc/op.cpp:1995-2002`）。
