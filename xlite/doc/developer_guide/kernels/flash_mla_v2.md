# flash_mla_v2

## 功能概述

MLA 吸收式注意力的 flash(在线 softmax)版本,是 mla_v2 的长序列路径:当 paged KV 总长 `maxNumBlocks * blockSize > tileSizeOfCachedKV` 时由模型层路由(`csrc/model.cpp` 的 `ForwardAttnMLAV2`,flash 路由分支关键字 `XliteOpFlashMLAV2`)。KV 维按 `tileSizeOfCachedKV` 分块,每块上 AIC 算 `QK = (qAbsorb·kCache^T + qr·peCache^T)*scale` 与 `sv = softmax_local(QK)·kCache`(注意 MLA 的 V 也是 latent kCache),AIV 用 online softmax 跨 tile 增量合并,最终 `oAbsorb = softmax_causal(QK) · kCache`。支持 DSA top-k token 选择(topkIndices),始终走 paged KV cache。

## 输入输出参数

Python 侧经 `mla_v2(..., enable_flash_attention=True)` 进入 host 分支(规范签名见 `xlite/_C.pyi` 的 `mla_v2`,关键字参数 `nz: bool = False`、`enable_flash_attention: bool = False`、`tile_size_of_cached_kv: int = 8192`、`dense: bool = False`;`tests/kernels/mla.py` 以局部变量 `enable_flash`/`tile_size`/`weight_nz` 按位置传入,关键字 `enable_flash = True`/`tile_size = 8192`):

```python
mla_v2(rt, q_with_qr, qr, k_cache, pe_cache, wuk_t, wuv, output, ..., topk_indices,
       top_k, nz, enable_flash_attention=True, tile_size_of_cached_kv=8192)
```

host 封装 `MLAV2` 的 flash 分支(`csrc/_C.cpp` 的 `MLAV2`,`enableFlashAttention=true` 分支)→ `XliteOpFlashMLAV2`(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`)。kernel 签名(`csrc/kernels/flash_mla_v2.h` 的 `flash_mla_v2_<dtype>` 入口,宏 `FLASH_MLA_V2_FUNC_DEFINE`):

```cpp
flash_mla_v2_<dtype>(qAbsorb, qr, kCache, peCache, topkIndices, qk, sv, max, sum,
                     lastMax, lastSum, sync, oAbsorb, queryStartLoc, queryLens,
                     cachedLens, blockTables, nHeads, ropeHeadDim, kvLoraRank,
                     blockSize, batch, maxNumBlocks, scale, tileSizeOfCachedKV, topK,
                     kCacheStride, peCacheStride)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| qAbsorb | 输入 | `[total_tokens, nHeads, kvLoraRank]` | BF16 | host 已做 WUK 吸收的 Q latent |
| qr | 输入 | `[total_tokens, nHeads, ropeHeadDim]` | BF16 | Q 的 RoPE 部分 |
| kCache / peCache | 输入 | `[numBlocks, blockSize, kvLoraRank]` / `[..., ropeHeadDim]` | BF16 | paged latent K / RoPE K cache |
| topkIndices | 输入(可选) | `[total_tokens, topK]` | INT32 | DSA top-k token 下标(升序);host 限制 `topK ≤ MAX_TOPK_NUM=2048`(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`,关键字 `MAX_TOPK_NUM`) |
| qk | workspace | `[aicNum * XLITE_MAX_M0 * 2, tileSizeOfCachedKV]` | BF16 | 当前 KV tile 的 QK 分数,按核双缓冲 |
| sv | workspace | `[aicNum * XLITE_MAX_M0 * 2, kvLoraRank]` | BF16 | 当前 tile 的 softmax·kCache 部分和 |
| max / sum | workspace | `[aivNum * XLITE_MAX_M0 * 2]` | FP32 | 当前 tile 局部 max / Σexp |
| lastMax / lastSum | workspace | `[total_tokens, nHeads]` | FP32 | 跨 tile 全局 max / Σexp |
| sync | workspace | `[1, aivNum]` | INT32 | RingSync 计数器(host Memset(0)) |
| oAbsorb | 输出 | `[total_tokens, nHeads, kvLoraRank]` | BF16 | 最终吸收式输出(WUV 投影在 host) |
| queryStartLoc / queryLens / cachedLens / blockTables | 输入 | `[batch]` / `[batch, maxNumBlocks]` | INT32 | 同 mla_v2 |
| nHeads / ropeHeadDim / kvLoraRank / blockSize / batch / maxNumBlocks | 标量 | - | uint32 | 同 mla_v2 |
| scale | 标量 | - | float | `(nopeHeadDim + ropeHeadDim)^-0.5` |
| tileSizeOfCachedKV | 标量 | - | uint32 | KV tile 长度,host 要求 ≤ `MAX_SOFTMAX_PINGPONG_LEN=11776`(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`,关键字 `MAX_SOFTMAX_PINGPONG_LEN`) |
| kCacheStride / peCacheStride | 标量 | - | uint32 | paged kCache / peCache 的 dim-0 跨度(元素数),块寻址 `block * stride`;连续时等于 blockSize×kvLoraRank / blockSize×ropeHeadDim,非连续时由宿主传入真实跨距 |

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| bfloat16_t | `csrc/kernels/flash_mla_v2_bfloat16_t.cpp` | `flash_mla_v2_bfloat16_t` |

host 侧仅 BF16(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`,关键字 `EachXDtype(BF16`)。

## 实现原理

与 flash_attention 同构的五级流水(混合 AIC/AIV,`KERNEL_TYPE_MIX_AIC_1_2`),矩阵环节换成 `MlaAicHelper` 的吸收式 GEMM(见 mla_v2 文档)。任务空间 `taskNum = queryNum * kvNum`(`kvNum = ceil((cachedLen+queryLen)/tileSizeOfCachedKV)`,见 `csrc/kernels/flash_mla_v2.h` 的 `RunAic`,关键字 `taskNum = queryNum * kvNum`),按 `totalIdx % block_num` 轮转分核。

### 流水线

每个 (query tile, kv tile) 任务:

1. AIC `RunAicQK(qAbsorb, qr, kCache, peCache, ..., kvOffset, kvLen, qk[curr])`——KV tile 窗口由 kvOffset/kvLen 表达,分页寻址 blockTable,块基址 = 物理块号 × dim-0 跨度(kCacheStride/peCacheStride);完成 `ffts_cross_core_sync(PIPE_FIX, flag0)`(`csrc/kernels/flash_mla_v2.h` 的 `RunAic`,关键字 `ffts_cross_core_sync(PIPE_FIX, softmaxConfig`);
2. AIV `RunAivSoftmaxPingPong`:对 tile 内行做带 scale 的 softmax,行有效长度 `actualCalcSoftmaxLen = cachedLen + queryTaskStart + 1 - kvOffset`(clip 到 kvLen);输出行宽 `ROUND_UP(kvLen, 4*svk0)`(SV 每次 256 列预取,防残留脏数据,见 `csrc/kernels/flash_mla_v2.h` 的 `RunAiv`,关键字 `4 * svk0`);若 `calcLen > topK` 且 topK>0,走 vgather/scatter 的 top-k softmax 路径(见 `RunAiv`,关键字 `calcLen > topK`);局部 max/sum 写 workspace → `ffts_cross_core_sync(PIPE_MTE3, flag2)`;
3. AIC `wait_flag_dev(2)` 后 `RunAicSV(qk[last], kCache, ..., kvOffset, kvLen, sv[last])`:`sv = softmax_local(QK)·kCache` → `ffts_cross_core_sync(PIPE_FIX, flag1)`;
4. AIV `wait_flag_dev(1)` 后 `RunAivSoftmaxUpdate`:把 sv/max/sum 与 output/lastMax/lastSum 在线合并(算法见 flash_attention 文档;此处 headSize 参数为 kvLoraRank,maskStride=nHeads)。

相邻任务错峰(QK 先行、SV/update 滞后一个任务),末尾补做,与 flash_attention 完全一致(`csrc/kernels/flash_mla_v2.h` 的 `RunAic`(关键字 `RunAicQK`/`RunAicSV`)与 `RunAiv`(关键字 `RunAivSoftmaxPingPong`/`RunAivSoftmaxUpdate`))。

### top-k(DSA)语义

topkIndices 形状 `[total_tokens, topK]`,按 `topkIndices + topK * queryTaskOffset` 定位当前 query tile 的行;softmax 内用 vgather 只对 top-k token 的分数归一,scatter 回写后非 top-k 位置为 0,从而 SV 只在 top-k token 上有非零贡献(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`,关键字 `vcmpvs_ge`/`vgather` 与 scatter 段)。kvOffset 参数同时用于 topk 下标窗过滤(`vcmpvs_ge/lt` 与 [kvOffset, kvOffset+calcLen) 比较)。

### 跨核 RingSync

同一 query 块的连续 KV tile 轮转落在不同核,update 读前一 tile 的 output/lastMax/lastSum 前必须 `ringSync.WaitPrevCore()`,非最后 tile 完成后 `SetNextCore()`,kernel 尾部 `PipeBarrier<PIPE_ALL>()` + `ResetPrevCore()` 复位(`csrc/kernels/flash_mla_v2.h` 的 `RunAiv`,RingSync 调用点关键字 `ringSync.WaitPrevCore`/`SetNextCore`/`ResetPrevCore`;实现 `csrc/kernels/ring_sync.h`)。

### 关键代码位置

- 主类与流水:`csrc/kernels/flash_mla_v2.h` 的 `FLASHMLAV2::RunAic` / `FLASHMLAV2::RunAiv`
- 吸收式 QK/SV:`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK` / `MlaAicHelper::RunAicSV`(与 mla_v2 共用)
- tile softmax(ping-pong,含 top-k):`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`
- online softmax update:`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxUpdate`
- RingSync:`csrc/kernels/ring_sync.h`
- host launch:`csrc/op.cpp` 的 `XliteOpFlashMLAV2`、workspace `csrc/_C.cpp` 的 `MLAV2`(`enableFlashAttention=true` 分支)、路由 `csrc/model.cpp` 的 `ForwardAttnMLAV2`(flash 路由分支)
