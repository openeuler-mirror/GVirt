# topk

## 功能概述

对一批变长序列做 topK 选取的通用向量算子(DeepSeek V3.2 DSA indexer 的 topk 后处理路径):每个 query token 在自身 `cachedLen + queryLen` 长度的得分行上选出最大的 `topK`(仅支持 2048)个元素的**索引**。等价于 `torch.topk(scores, k=K, dim=-1)[1]`,但按 2048 元素的 chunk 流式处理,支持远大于 UB 容量的序列长度(测试覆盖到 131072)。注意与 `indexer_topk` 的区别:本算子输入是**已经算好的 scores 矩阵**,不包含得分计算与核间接力。

## 输入输出参数

Python 侧调用(`tests/kernels/topk.py:73`):

```python
topk(rt, scores, indices, out_indices, query_lens, cached_lens, K)
```

host 侧 launch 见 `csrc/op.cpp:1290`(`XliteOpTopK`):`maxSeqLen = scores.shape[1]`;`maxSeqLen <= k` 时直接跳过;`k != 2048` 抛错(注释明确 "Only topK equals 2048 is supported",`csrc/kernels/topk.h:15`)。kernel 签名(`csrc/kernels/topk.h:363`):

```cpp
topk_<dtype>(GM_ADDR scores, GM_ADDR indices, GM_ADDR outIndices, GM_ADDR queryLens,
             GM_ADDR cachedLens, uint32_t maxSeqLen, uint32_t numBatches, uint32_t topK)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| scores | 输入 | `[numBatches, maxSeqLen]`(每 batch 一行,padded) | bf16 / fp32 | 得分矩阵;行内前 `queryLens[i]+cachedLens[i]` 个有效,其余为 padding(测试用 0 填充,`tests/kernels/topk.py:35-39`) |
| indices | 输入 | `[maxSeqLen]` | int32 | `arange(maxSeqLen)` 恒等索引表 |
| outIndices | 输出 | `[numBatches, topK]`(扁平 `sum*topK`?) | int32 | 注意:kernel 按 `mIdx`(全局 query 计数)× topK 写出,host 侧分配 `sum(query_lens) × K`(`tests/kernels/topk.py:67`) |
| queryLens | 输入 | `[numBatches]` | int32 | 各 batch query 长度(行数) |
| cachedLens | 输入 | `[numBatches]` | int32 | 各 batch 已缓存长度;有效序列长 = queryLen + cachedLen |
| maxSeqLen / numBatches / topK | — | 标量 | uint32 | maxSeqLen 来自 scores 第二维;topK 固定 2048 |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `topk_bfloat16_t` | `csrc/kernels/topk_bfloat16_t.cpp` | scores 为 bf16(核内 `vconv_bf162f32` 转 fp32 再排序) |
| `topk_float` | `csrc/kernels/topk_float.cpp` | scores 为 fp32 |

dtype 分派见 `csrc/op.cpp:1307-1315`(要求 indices 为 INT32)。纯向量核(`__DAV_C220_VEC__`,以 `rt.aivNum` launch)。

## 实现原理

### 任务划分

`Run`(`csrc/kernels/topk.h:136-201`):外层 batch,内层遍历该 batch 的每个 query token,全局计数 `mIdx` 作为工作项 id,按 `mIdx % block_num == block_idx` 静态分配到向量核(`topk.h:158-161`)。每个工作项独立完成一行的完整 topk,核间无同步。测试中 batch≤8、query 总数不大时多数核空闲,长 query(如 8231)时并行度来自行数。

### 分块流式 topk 算法(CHUNK_SIZE=2048)

每行的处理(`topk.h:164-197`):

1. **`initTopk()`**(`topk.h:309-326`):把当前全局候选缓冲 `topkBuf[0]` 初始化为 topK 个 `(-inf, 0)`(值+索引)对 —— 用偶/奇 mask 的两次 `vector_dup` 分别填充 2048 个 float 的 `-inf` 与 0;
2. **chunk 循环**:`for (processed = 0; processed < len;)` 每次取 `length = min(len - processed, CHUNK_SIZE=2048)`(`topk.h:168-169`,常量定义 `topk.h:35`);
   - `CopyInIndices/CopyInScores`:该 chunk 的索引与得分 GM→UB(乒乓交替);
   - `PadInputs`(`topk.h:224-277`):不足 2048 时尾块用 `vector_dup` 填 `-inf`(float)或 `-3.4e38`(bf16,避免 bf16 的 inf 表示问题),mask 精确控制尾部对齐;
   - `ConvertInput`(`topk.h:279-288`):bf16 时 `vconv_bf162f32` 统一转 fp32;
   - `Sort(CHUNK_SIZE, current)`(`topk.h:290-307`):
     - `vbitsort(scratch0, scoresIn, indicesIn, 2048/32)`:64 组、每组 32 元素的位排序(值+索引成对,每组输出 64 元素);
     - `MrgSort(scratch0, scratch1, repeat, &dstBufIdx)`(`csrc/kernels/kernel_macro.h:748-788`):`vmrgsort4` 四路归并循环把 64 组归并为整块降序序列 `localSort`;
     - `Merge2(topk1, topk0, localSort, 1, 2048)`(`topk.h:121-134`):用 `vmrgsort4`(只开 2 个有效队列,`validBits=0b11`,两队列长度均为 2048)把"上一轮的全局候选 topk0"与"本 chunk 的 localSort"二路归并成新的全局候选 `topk1` —— 归并结果天然截断在前 2048 对(队列长度编码 `blockSize | blockSize<<16`);
3. **`FillOutScores(mIdx, current)`**(`topk.h:328-339`):最终候选在 `topkBuf[current]`,`vreducev2`(mode=2,隔 2 取 1)抽出索引载荷,`copy_ubuf_to_gm` 写到 `outIndicesGm + mIdx*topK`。

因此本算法本质是 **k 路归并式流式 topk**:维护一个始终有序的 topK 候选集,每个 2048 chunk 完整排序后与候选集做一次二路归并,复杂度 O(len/2048 × 归并代价),无需堆或位图,完全由 `vbitsort`/`vmrgsort4` 硬件排序指令承担。

当 `len < topK` 时,参考实现用最后一个索引补齐(`tests/kernels/topk.py:53-56`),kernel 输出中 padding 部分为候选集残余(值 `-inf`,索引 0)。

### UB 内存布局

`Init`(`topk.h:64-102`)按 `topK` 上界静态分配(全部按 fp32 字节数对齐 `pad = ROUND_UP(topK, 64)*4`):

| 缓冲 | 大小 | 用途 |
|---|---|---|
| `scoresIn` | pad | fp32 得分(或 float 直接搬入) |
| `indicesIn` | pad | 恒等索引 chunk |
| `indices` | pad | (预留) |
| `topkBuf[2]` | 各 4*pad | 全局 topK 候选乒乓(值+索引交错,4 倍) |
| `scratch[2]` | 各 2*pad | vbitsort/MrgSort 工作区乒乓 |
| `scoresInTmp` | pad/2 | 仅 bf16:原始 dtype 搬入中转 |
| `queryLensIn`/`cachedLensIn` | numBatches×u32 | 长度表 |

乒乓语义:`current` 在 chunk 间翻转,`topkBuf[current]` 为上一轮候选、`topkBuf[1-current]` 为归并输出;`scratch` 同理,使相邻 chunk 的排序写回不冲突。

### 流水线同步

MTE2 搬入与 V 计算用计数事件 ID0(数据)/ID1(得分搬入的上一轮释放)配对;`PipeBarrier<PIPE_ALL>` 在工作项切换时确保 UB 状态干净(`topk.h:162`)。S 管线在循环前读 `queryLensIn`/`cachedLensIn`(`topk.h:150-151` 的 MTE2→S 配对)。
