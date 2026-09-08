# flash_attention

## 功能概述

MHA/GQA 的 flash(在线 softmax)注意力算子,是 attention 的长序列版本:当 KV 总长 `maxNumBlocks * blockSize` 超过 host 侧 tile 阈值(`_tileSizeOfCachedKV`,默认 `MAX_KV_TILE_SIZE=8192`,`csrc/auto_tuner.h:13`)时,模型层路由到本算子(`csrc/model.cpp:636-655`)。KV 维按 `tileSizeOfCachedKV` 分块,AIC/AIV 流水地在每个 KV tile 上做 QK^T → 局部 softmax → SV,AIV 再用 online softmax 增量合并各 tile 的部分和,最终 `out = softmax(mask(QK^T)) · V` 与 attention 完全等价,但 qk workspace 只需一个 tile 的宽度。

## 输入输出参数

Python 侧调用(测试经 `attention(..., enable_flash=True)`,`tests/kernels/mla.py` 风格同;`tests/kernels/attention.py:23` 的 `enable_flash` 开关即选择本算子):

```python
attention(rt, qkv, k_cache, v_cache, output, query_start_loc, query_lens,
          cached_lens, block_tables, n_heads, n_kv_heads, head_dim,
          BLOCK_SIZE, batch, enable_flash=True, tile_size)
```

host 封装 `Attention(enableFlashAttention=true)` 分支 → `XliteOpFlashAttention`(`csrc/_C.cpp:1485-1506`、`csrc/op.cpp:933`)。kernel 签名(`csrc/kernels/flash_attention.h:428`):

```cpp
flash_attention_<dtype>(input, kCache, vCache, qk, sv, max, sum, lastMax, lastSum,
                        sync, output, queryStartLoc, queryLens, cachedLens, blockTables,
                        nHeads, nKVHeads, headSize, blockSize, batch, maxNumBlocks,
                        tileSizeOfCachedKV)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input (qkv) | 输入 | `[total_tokens, (nHeads+2*nKVHeads)*headSize]` | FP16/BF16 | 融合 QKV,只读 Q 段 |
| kCache / vCache | 输入 | `[numBlocks, blockSize, nKVHeads, headSize]` | FP16/BF16 | paged KV cache |
| qk | workspace | `[aicNum * XLITE_MAX_M0 * 2, tileSizeOfCachedKV]` | FP16/BF16 | 当前 KV tile 的 QK^T 分数,按核双缓冲 |
| sv | workspace | `[aicNum * XLITE_MAX_M0 * 2, headSize]` | FP16/BF16 | 当前 tile 的 softmax·V 部分和(未归一化),按核双缓冲 |
| max / sum | workspace | `[aivNum * XLITE_MAX_M0 * 2]` | FP32 | 当前 tile 每行的局部 max / Σexp,按核×subblock 双缓冲 |
| lastMax / lastSum | workspace | `[total_tokens, nHeads]` | FP32 | 跨 tile 的全局 max / Σexp(online softmax 状态) |
| sync | workspace | `[1, aivNum]` | INT32 | RingSync 跨核同步计数器(host Memset(0),`csrc/_C.cpp:1493-1494`) |
| output | 输出 | `[total_tokens, nHeads * headSize]` | FP16/BF16 | 最终注意力输出 |
| queryStartLoc / queryLens / cachedLens | 输入 | `[batch]` | INT32 | 同 attention |
| blockTables | 输入 | `[batch, maxNumBlocks]` | INT32 | 逻辑块→物理块映射 |
| nHeads / nKVHeads / headSize / blockSize / batch / maxNumBlocks | 标量 | - | uint32 | 同 attention |
| tileSizeOfCachedKV | 标量 | - | uint32 | KV tile 长度,host 侧经 `csrc/op.h:14` 的 static_assert 保证 `MAX_KV_TILE_SIZE(8192) ≤ MAX_SOFTMAX_PINGPONG_LEN(11776)`,由 auto-tuner 在 8192 内选取 |

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| float16_t | `csrc/kernels/flash_attention_float16_t.cpp` | `flash_attention_float16_t` |
| bfloat16_t | `csrc/kernels/flash_attention_bfloat16_t.cpp` | `flash_attention_bfloat16_t` |

host 校验 qkv/qk/kCache/vCache/output dtype 一致(`csrc/op.cpp:945-952`)。

## 实现原理

混合 AIC/AIV kernel,`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`(`csrc/kernels/flash_attention.h:30`)。相对 attention 的核心差异:任务空间多了 KV tile 维(`taskNum = queryNum * nKVHeads * kvNum`,`kvNum = ceil((cachedLen+queryLen)/tileSizeOfCachedKV)`,`csrc/kernels/flash_attention.h:108-110`),并且多出 AIV 的 **online softmax update** 阶段。

### 五级软件流水

每个 (query tile, kvHead, kv tile) 任务的执行顺序在时间上错开为:

1. AIC `RunAicQK`(复用 `AicHelper`,见 attention 文档;`kvOffset/kvLen` 表达 KV tile 窗口)→ `ffts_cross_core_sync(PIPE_FIX, flag0)` 通知 AIV;
2. AIV 对 qk tile 做 `RunAivSoftmaxPingPong`(单趟版 softmax,行宽 tileSizeOfCachedKV ≤ 11776,UB 常量校验见 `csrc/kernels/softmax_attn_aiv.h:17-39`),同时把每行局部 max/sum 存到 `max/sum` workspace → `ffts_cross_core_sync(PIPE_MTE3, flag2)`;
3. AIC `wait_flag_dev(2)` 等 softmax 完成,`RunAicSV`(gqaScatter=false,紧凑写 sv,行距 headSize)算出 `Σ softmax_local(qk)·V` → `ffts_cross_core_sync(PIPE_FIX, flag1)`;
4. AIV `wait_flag_dev(1)` 等 SV 完成,`RunAivSoftmaxUpdate` 把 sv/max/sum 并入 output/lastMax/lastSum。

相邻任务错峰:AIC 发起任务 i+1 的 QK 后回头做任务 i 的 SV;AIV 做完任务 i+1 的 softmax 后回头做任务 i 的 update(`csrc/kernels/flash_attention.h:149-165`、`:291-329`),QK/SV 在 Cube、softmax/update 在 Vector 各自成流水。

### online softmax 合并(RunAivSoftmaxUpdate,softmax_attn_aiv.h 并入)

`RunAivSoftmaxUpdate`(`csrc/kernels/softmax_attn_aiv.h:563`)实现标准 flash 在线归并,对每行:

```
new_max = max(max_prev, max_curr)
scale_prev = exp(max_prev - new_max);  scale_curr = exp(max_curr - new_max)
new_sum = sum_prev*scale_prev + sum_curr*scale_curr
sv_out  = sv_prev*scale_prev + sv_curr*scale_curr
output  = sv_out / new_sum
```

要点:

- **首 tile 快路径**:`isFirstKvTile` 时直接把 sv/max/sum 拷到 output/lastMax/lastSum,无归并(`csrc/kernels/softmax_attn_aiv.h:666-687`)。注意 AIV 用的 `actualCalcSoftmaxLen = cachedLen + queryTaskStart + 1 - kvOffset`(clip 到 kvLen),行内 causal 位置在此长度内由 softmax 的 maskOff 机制处理。
- **精度**:sv 以 fp32 参与归并(`vconv_*2f32`),scale/sum 全程 fp32,最后 `vconv_f322*` 回写;由于每次 update 都除以 new_sum,数值上等价于全局 softmax。
- **归一化时机**:非首 tile 的 update 每步都除 new_sum(而非最后一步才除,`csrc/kernels/softmax_attn_aiv.h:754-760`),这样 output 始终保持归一化形态,lastSum/lastMax 记录未归一化的统计量。

### 跨核 RingSync

一个 query 块的连续 KV tile 会被 `totalIdx % block_num` 轮转分到不同核,而 update 需要读前一 tile 写入的 output/lastMax/lastSum。`RingSync`(`csrc/kernels/ring_sync.h`)用 GM 计数器解决:每核在 `sync[blockIdx*2+subBlockIdx]` 写递增 generation,做 update 前 `WaitPrevCore()` 自旋等待前一个核的 generation 到位;最后一个 tile 不再 `SetNextCore()`,收尾 `ResetPrevCore()` 清零供下次使用(`csrc/kernels/flash_attention.h:301-304`、`:324-328`、`:380-383`)。

### 任务分配与 GQA

- `queryTileSize = XLITE_MAX_M0 / headNumInGroup`(固定 m0=128,不分档),GQA 组内 query 头拼满 Cube 的 M 维(`csrc/kernels/flash_attention.h:93`)。
- AIV 按 `get_subblockid()` 把 `queryTaskLen * headNumInGroup` 行对半分给 2 个 subblock;max/sum workspace 因此按 `block_idx * M0 * 2 + subblock * M0` 布局(`csrc/kernels/flash_attention.h:61-70`)。
- causal:每个 query tile 的 `calcLen = cachedLen + queryTaskStart + queryTaskLen`,整 tile 落在 calcLen 之外的 KV tile 直接跳过(`calcLen <= kvOffset` continue,`csrc/kernels/flash_attention.h:122-124`)。

### 关键代码位置

- 主类与流水调度:`csrc/kernels/flash_attention.h:77`(RunAic)、`:196`(RunAiv)
- QK/SV:`csrc/kernels/attention_aic_helper.h`(与 attention 共用;SV 的 flash 紧凑写分支 `:270-272`)
- tile 内 softmax(ping-pong 单趟):`csrc/kernels/softmax_attn_aiv.h:65`
- online softmax update:`csrc/kernels/softmax_attn_aiv.h:563`
- 跨核同步:`csrc/kernels/ring_sync.h`
- host launch:`csrc/op.cpp:933`、workspace 分配 `csrc/_C.cpp:1485-1506`、路由 `csrc/model.cpp:636`
