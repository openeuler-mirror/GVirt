# compressor

## 功能概述

DeepSeek-V4 KV Compressor 融合算子(核内模板 `compressor`,见 `csrc/kernels/compressor.h` 的 `compressor` 模板函数)。镜像 `deepseek_v4.py` 的 `Compressor.forward`(在 wkv/wgate GEMM 之后):overlap_transform + softmax-gated 加权求和 + RMSNorm + rotary + 压缩 KV cache 写入,多 batch 融合于单 kernel。

核心数学:对每个压缩块(mergeSize = `(1+overlap)*ratio` 行)计算

```
weightedSum[block] = sum_k softmax(score[k] + ape[k]) * kv[k]      # k ∈ [0, mergeSize)
```

随后对 `weightedSum` 做 RMSNorm 与 RoPE,按 `compressSlotMapping` 写入 paged 压缩 KV cache。prefill 尾段与 decode token 额外"溢写"(spill)到流式 `state`,供下一 chunk 的窗口读取。

**统一路径**:同一 launch 覆盖 prefill / chunked-prefill / decode 三种场景——块数 `((cached+qLen)/ratio - cached/ratio)` 在 decode 压缩步为 1、非压缩步为 0,故 decode 无需独立分支。核间按全局压缩块索引 `totalBlockIdx % block_num == block_idx` 切分,norm/rope 尾部遍历同一区间,每个核只读自己写的 `weightedSum` 行,**无跨核 GM 同步**。

退化/分支:

- `nTotalBlocks == 0`(decode 非压缩步、或短 prefill 段无完整 ratio 窗口):compress pass 仍写 decode token + spill 到 state,但 norm/rope 尾部区间为空,直接 return(`csrc/kernels/compressor.h` 的 `compressor`(关键字 `if (nTotalBlocks == 0)`))。
- `ratio=128, overlap=0` 走**分块列切片**(tiled)路径;`ratio=2/4` 走非分块路径(`csrc/kernels/compressor.h` 的 `compressor_blocks_weighted_sum`(关键字 `useTiled`))。

## 输入输出参数

Python 侧调用(`xlite/_C.pyi`,测试 `tests/kernels/compressor.py` 的 `compressor(...)` 调用(关键字 `compressor(rt, kv, score`)):

```python
compressor(rt, kv, score, ape=None, norm=None, freqs=None, weighted_sum, query_start_loc,
           query_lens, cached_lens=None, batch, n_total_blocks, ratio, overlap, head_dim,
           rope_head_dim, norm_eps, compress_kv, compress_positions, compress_slot_mapping,
           compress_block_size, state=None, state_slot_mapping=None, state_block_table=None,
           state_block_size, state_cache_stride_dim0, do_rotate=False)
```

host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpCompressor`(以 `rt.aivNum` 个 block 启动)。kernel 签名(`csrc/kernels/compressor.h` 的 `COMPRESSOR_FUNC_DEFINE` 宏,关键字 `compressor_##dtype`):

```cpp
compressor_<dtype>(GM_ADDR kv, GM_ADDR score, GM_ADDR ape, GM_ADDR normWeight, GM_ADDR freqs,
                   GM_ADDR weightedSum, GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                   uint32_t batch, uint32_t nTotalBlocks, uint32_t ratio, uint32_t overlap,
                   uint32_t head_dim, uint32_t rope_head_dim, float norm_eps, GM_ADDR compressKv,
                   GM_ADDR compressPositions, GM_ADDR compressSlotMapping, uint32_t compressBlockSize,
                   GM_ADDR state, GM_ADDR stateSlotMapping, GM_ADDR stateBlockTable,
                   uint32_t stateBlockSize, uint32_t maxStateBlocks, uint64_t stateCacheStrideDim0,
                   uint32_t doRotate, float rotateScale)
```

`coff = 1 + overlap`,`full_dim = coff * head_dim`。`normWeight`/`freqs`/`ape`/`state` 等可为空(nullptr),kernel 据此跳过对应步骤。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| kv | 输入 | `[batchedTokens, coff*head_dim]` | fp32 | 压缩前 KV(每行前半 KV、后半 overlap KV) |
| score | 输入 | `[batchedTokens, coff*head_dim]` | fp32 | 压缩前 score;与 kv 同 shape |
| ape | 输入 | `[ratio, coff*head_dim]` | fp32 / null | 绝对位置编码;null 表示用原始 score(不加 ape) |
| norm(normWeight) | 输入 | `[head_dim]` | fp32 / null | RMSNorm weight;null 跳过 norm |
| freqs | 输入 | `[max_seq_len, rope_head_dim/2]` | complex64 / null | 预计算复数 RoPE 频率;null 跳过 rope+cache 写 |
| weightedSum | 输出(scratch GM) | `[nTotalBlocks, head_dim]` | bf16 / fp16 | 加权求和结果(norm/rope 原地更新);最终为压缩 KV 写入前的中间量 |
| queryStartLoc | 输入 | `[batch]` | int32 | 每 batch 请求在 batchedTokens 中的起始行 |
| queryLens | 输入 | `[batch]` | int32 | 每 batch 请求长度(decode 为 1) |
| cachedLens | 输入 | `[batch]` | int32 / null | 每 batch 已缓存长度;null 时按 0 处理 |
| compressKv | 输出 | `[blocks, compressBlockSize, 1, head_dim]` | bf16 / fp16 / fp32 | paged 压缩 KV cache;RoPE 后按 `compressSlotMapping` 写入 |
| compressPositions | 输入 | `[nTotalBlocks]` | int64 | 压缩块的绝对位置(压缩 token 空间) |
| compressSlotMapping | 输入 | `[nTotalBlocks]` | int32 | 压缩块 → cache 平坦槽位 |
| state | 输入/输出 | `[blocks, stateBlockSize, 1, 2*coff*head_dim]` | fp32 / null | 流式 state(prefill 溢写、decode 读);null 禁用 spill |
| stateSlotMapping | 输入 | `[batchedTokens]` | int32 / null | prefill token → state 平坦槽位(FLAT id) |
| stateBlockTable | 输入 | `[batch, maxStateBlocks]` | int32 / null | paged state-cache 块表;条目 0 = 未分配哨兵 |
| batch / nTotalBlocks / ratio / overlap / head_dim / rope_head_dim / compressBlockSize / stateBlockSize | — | 标量 | uint32 | `ratio ∈ {2,4,128}`;`overlap ∈ {0,1}` |
| maxStateBlocks | — | 标量 | uint32 | host 取 `stateBlockTable.stride[0]` |
| stateCacheStrideDim0 | — | 标量 | uint64 | state 的 dim0 步长;0(pybind 默认)→ 解析为 `stateBlockSize*2*coff*head_dim` |
| norm_eps | — | 标量 | float | RMSNorm epsilon |
| doRotate / rotateScale | — | 标量 | bool / float | `doRotate` 时 `rotateScale = 1/√head_dim`,传给 `rope_complex_and_cache` |

host 校验见 `csrc/op.cpp` 的 `XliteOpCompressor` 的 `compCheck` 校验 lambda(`kv/score` 必须 fp32 且同 shape、`ape`/`norm` fp32、`freqs` complex64、`weightedSum` 为 bf16/fp16 且 `numel == nTotalBlocks*head_dim`、`compressKv` 末维 == head_dim、`ratio ∈ {2,4,128}` 等)。

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `compressor_float16_t` | `csrc/kernels/compressor_float16_t.cpp` | compressKv/weightedSum 为 fp16(kv/score 仍 fp32) |
| `compressor_bfloat16_t` | `csrc/kernels/compressor_bfloat16_t.cpp` | compressKv/weightedSum 为 bf16 |
| `compressor_float` | `csrc/kernels/compressor_float.cpp` | compressKv 为 fp32(kv/score 仍 fp32);weightedSum 仍为 bf16/fp16(host `compCheck` 强制 `weightedSum.dtype == BF16 || FP16`,不接受 fp32) |

dtype 分派见 `csrc/op.cpp` 的 `XliteOpCompressor`(关键字 `compressKv.dtype`,按 `compressKv.dtype` 选)。kernel 为 `__DAV_C220_VEC__`、任务类型 `KERNEL_TYPE_MIX_AIV_1_0`(`csrc/kernels/compressor.h` 的 `compressor`(关键字 `KERNEL_TYPE_MIX_AIV_1_0`),纯 AIV)。

## 实现原理

### 总体流程

顶层 `compressor`(`csrc/kernels/compressor.h` 的 `compressor` 模板函数)按序:

1. **compress pass** `compressor_blocks_weighted_sum`(`csrc/kernels/compressor.h` 的 `compressor_blocks_weighted_sum`):统一处理 prefill/chunked/decode——写 `weightedSum[0..nTotalBlocks)`,并把 prefill 尾段与 decode token 溢写到 `state`;
2. `nTotalBlocks == 0` 则 return(`csrc/kernels/compressor.h` 的 `compressor`(关键字 `if (nTotalBlocks == 0)`));
3. **RMSNorm** `norm`(复用 `norm.h`,`NormKind::Rms`)原地作用于 `[0, nTotalBlocks)`(`csrc/kernels/compressor.h` 的 `compressor`(关键字 `if (hasNorm)`));
4. **RoPE + cache 写** `rope_complex_and_cache`(复用 `rope_complex_and_cache.h`)按 `compressPositions`/`compressSlotMapping` 写 `compressKv`(`csrc/kernels/compressor.h` 的 `compressor`(关键字 `if (hasRope)`))。

### paged state 寻址

state 按 paged 块表寻址(`csrc/kernels/compressor.h` 的 `state_block_id` / `state_row_addr` / `state_rows_in_block`):`state_slot = blockId*stateBlockSize + rowInBlock`,其中 `blockId = stateBlockTable[b*maxStateBlocks + slot/stateBlockSize]`;`state_row_addr` 给出 GM 行地址;`state_rows_in_block` 计算从某 slot 起在同一块内的最大连续行数(2D copy 需连续性)。**块表条目 0 = 未分配哨兵**:写跳过、读不保护(stale)。

### spill:prefill 尾段溢写

`compressor_spill_prefill_to_state`(`csrc/kernels/compressor.h` 的 `compressor_spill_prefill_to_state`)把每请求尾部 `[spillBeginAbs, absEnd)`(overlap 且 `cutoff>=ratio+cached` 时含上一个完整窗口)溢写到 state——`state[slot] = [kv | score+ape]`(kv 原值、score 含 ape)。decode(`qLen==1`)跳过(其 state 写由 `compressor_decode_token_state` 完成)。运行长度受 `SPILL_ROWS=4` 上限,并在 state 块边界拆分;**乒乓**双缓冲(`curr = 1-curr`):MTE2 载入 k+1 行时 V 计算 k 行。

### decode token 写 state

`compressor_decode_token_state`(`csrc/kernels/compressor.h` 的 `compressor_decode_token_state`)把每个 decode 请求(`qLen==1`)的单 token 按**绝对位置** `cachedLens[b]` 写入 state(kv 原值、score+ape),与 host 的 per-token 约定一致;prefill 请求跳过。每步(无论是否压缩)都执行,保证下一步窗口读取能命中。

### 加权求和 softmax_weighted_sum

`softmax_weighted_sum`(`csrc/kernels/compressor.h` 的 `softmax_weighted_sum`)计算 `out = sum_k softmax(score[k]+ape[k]) * kv[k]`,8 步:`score += ape` → `max` → `sub` → `exp` → `sum` → `div` → `mul(kv)` → `sum`。`dCur > 0`(分块路径)时用 `SetMask(dCur)` 仅计算低 `dCur` lane,UB 步长不变,完成后恢复全 mask。`kv` 与 `out` 可别名(vmul 先消费 kv 再写 out)。

### 窗口加载 load_window

`load_window`(`csrc/kernels/compressor.h` 的 `load_window`)统一加载 `ratio` 行的 `rowWidth`-float 列切片到 UB。逐行按绝对位置选源:`abs < cached` → paged **state** 行(score 已含 ape);`abs >= cached` → **gmKv** 行(原始 score)。同源连续行合并为一次 2D DMA,在 state/gmKv 边界拆分。两种调用形态:

- 非分块半窗口:`rowWidth=head_dim`,加载前半 `[:head_dim]` 或后半 `[head_dim:]`;
- 分块列切片(`ratio=128, overlap=0`):`rowWidth=dCur`(32 float 列),加载 `[:head_dim]` 的 `[c0, c0+dCur)` 切片。

### 非分块路径(ratio=2/4)

`csrc/kernels/compressor.h` 的 `compressor_blocks_weighted_sum`(关键字 `if (!useTiled)` 非分块分支)。overlap 时 mergeSize=2*ratio,前半 `[0,ratio)` = 上一窗口 `[:head_dim]`(从 state 读),后半 `[ratio,2*ratio)` = 当前窗口 `[head_dim:]`。`noPredecessor`(overlap 且 `bStart<ratio`,即首个窗口)跳过前半加载、用 `vector_dup` 填 0(kv)/-inf(score)。ape 处理:state 行已含 ape(spill 时烘焙),gmKv 行按**连续后缀** `vadd` 预加 ape,随后 softmax 以 `ape=null` 运行(避免重复加)。乒乓:加载(MTE2)↔ 计算(V)↔ 写回(MTE3)用 `EVENT_ID0/1`(MTE2↔V)与 `EVENT_ID2/3`(V↔MTE3)交叉同步,`curr = 1-curr`。

### 分块路径(ratio=128, overlap=0)

`csrc/kernels/compressor.h` 的 `compressor_blocks_weighted_sum`(关键字 `} else {` 分块列切片分支,`const uint32_t slice = calcPad / 2`)。head_dim 按列切成 `slice = 32` float 的 `nSlice` 片,逐片完成全部块的 compress(降低 UB 占用:head_dim=512/ratio=128 时约 128KB ≤ 192KB)。每片内 `SetMask(dCur)` 屏蔽到当前列宽,ape 预加同样按连续后缀 `vadd`。片间复用同一套乒乓缓冲。该路径无 overlap、无 noPredecessor 填充。

### 流水线同步

compress pass 与两个 spill 子函数均用 MTE2(载入)↔ V(计算)↔ MTE3(写回)的 `set_flag/wait_flag` 计数事件 ID 交叉同步(无 S 管线标量归约——`softmax_weighted_sum` 全向量),乒乓双缓冲隐藏搬运延迟。spill 子函数进入前预设全部依赖 flag(`csrc/kernels/compressor.h` 的 `compressor_spill_prefill_to_state`(关键字 `set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2)` 预设段)、`compressor_decode_token_state`(关键字 `set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0)` 预设段))、结束时 `wait_flag` 清空(`csrc/kernels/compressor.h` 的 `compressor_spill_prefill_to_state`(关键字 `wait_flag` 清空段)、`compressor_decode_token_state`(关键字 `wait_flag` 清空段))。compress pass 末尾 `pipe_barrier(PIPE_ALL)`(`csrc/kernels/compressor.h` 的 `compressor_blocks_weighted_sum`(关键字 `pipe_barrier(PIPE_ALL)`))保证 state 写完成后再进 norm/rope。norm/rope 尾部因核间切分同一 `totalBlockIdx` 区间,每核只读自写行,无需跨核 GM 屏障。

### 关键内存布局要点

- 非分块 UB:8×(mergeSize×paddedHeadDim) + 4×paddedHeadDim 个 float(kv/score 乒乓 + calc + tmp + ape + mx/sm + kvOut 乒乓),`assert(off <= UB_SIZE)`(`csrc/kernels/compressor.h` 的 `compressor_blocks_weighted_sum`(关键字 `assert(off <= UB_SIZE)` 非分块));
- 分块 UB:列切片 32 float,`assert` 验证 head_dim=512/ratio=128 约 128KB ≤ UB(`csrc/kernels/compressor.h` 的 `compressor_blocks_weighted_sum`(关键字 `assert(off <= UB_SIZE)` 分块));
- state 行布局 `[kv | score]`:前 `coff*head_dim` float 为 kv、后 `coff*head_dim` float 为 score(含 ape),`stateRowFloats = 2*coff*head_dim`。
