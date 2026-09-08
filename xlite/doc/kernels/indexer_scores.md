# indexer_scores

## 功能概述

DeepSeek V3.2 DSA indexer 的得分计算算子(纯 Cube 核)。对每个 query token 计算 indexer 头的注意力得分并与头权重融合,数学上等价于两步 einsum(见 `csrc/kernels/indexer_scores.h:39-46` 注释与测试 `tests/kernels/indexer_scores.py:102-107`):

```
scores[s,h,t]      = sum_d q[s,h,d] * kCache[t,d]        # Q·K^T
index_score[s,t]   = sum_h scores[s,h,t] * weight[s,h]    # 加权头融合
```

输出 `[total_query_len, max_num_blocks*block_size]` 的稠密得分矩阵,供后续 topk 选取稀疏注意力 token。与 `indexer_topk` 的区别:本算子只算分数,不融合 topk。

## 输入输出参数

Python 侧调用(`tests/kernels/indexer_scores.py:133`):

```python
indexer_scores(rt, q, k_cache, weight, scores, query_start_loc, query_lens,
               cached_lens, block_tables, n_heads, head_dim, block_size, batch)
```

host 侧 launch 见 `csrc/op.cpp:1679`(`XliteOpIndexerScores`,`KERNEL_TASK_TYPE_AIC_ONLY`,用 `rt.aicNum` 个 Cube 核)。kernel 签名(`csrc/kernels/indexer_scores.h:293`):

```cpp
indexer_scores_<dtype>(GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR scores,
                       GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                       GM_ADDR blockTables, uint32_t nHeads, uint32_t headDim,
                       uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q | 输入 | `[total_query_len, nHeads*headDim]` | fp16 / bf16 | 所有 batch 拼接的 flatten query |
| kCache | 输入 | `[kvcache_block_num, blockSize, headDim]` | fp16 / bf16 | paged indexer K cache,物理块号由 blockTables 索引 |
| weight | 输入 | `[total_query_len, headDim + nHeads]` | fp16 / bf16 | 每 token 的融合权重取自第 `headDim` 列起的 `nHeads` 列(与 Q 同投影输出拼接存放) |
| scores | 输出 | `[total_query_len, maxNumBlock*blockSize]` | fp16 / bf16 | 每行一个 query token 对全部 KV 位置的 index_score(行内按全局 token 位置布局,valid 长度 = cachedLens[i]+queryLens[i]) |
| queryStartLoc | 输入 | `[batch]` | int32 | 各 batch query 在拼接 q 中的起始偏移(前缀和) |
| queryLens | 输入 | `[batch]` | int32 | 各 batch 的 query 长度 |
| cachedLens | 输入 | `[batch]` | int32 | 各 batch 已缓存 KV 长度(总 KV 长度 = cachedLen + queryLen) |
| blockTables | 输入 | `[batch, maxNumBlock]` | int32 | 逻辑块 → 物理块映射表;`maxNumBlock` 由 host 侧 `DeriveMaxNumBlocks` 推导(`csrc/op.cpp:1697`) |
| nHeads / headDim / blockSize / batch | — | 标量 | uint32 | 测试配置:nHeads=64, headDim=128, blockSize=128;约束 `blockSize <= MAX_M0=128`、`headDim <= k0`(`indexer_scores.h:49-51`) |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `indexer_scores_bfloat16_t` | `csrc/kernels/indexer_scores_bfloat16_t.cpp` | q/kCache/weight/scores 全为 bf16 |
| `indexer_scores_float16_t` | `csrc/kernels/indexer_scores_float16_t.cpp` | 全为 fp16 |

dtype 分派要求四张 tensor 同 dtype(`csrc/op.cpp:1688-1691`)。无 fp32 变体。

## 实现原理

### 两级 MMAD 分块策略

每次 mmad 的两级矩阵乘语义(`csrc/kernels/indexer_scores.h:39-46`):

- 第一级 `scores = k * q`:m 维为 KV token 块(`blockSize`),n 维为 `queryTileSize*nHeads`(一次装下若干 query token 的全部头),k 维为 `headDim`;
- 第二级 `index_scores = weight * scores`:m 维为 query(单 token),n 维为 KV token 块,k 维为 `nHeads` —— 即对每个 query token,用其 `nHeads` 个权重把各头得分加权求和。

query tile 划分:`queryTileSize = MAX_M0 / nHeads`(`indexer_scores.h:109-112`,MAX_M0=128;nHeads=64 时 queryTileSize=2)。任务枚举(`indexer_scores.h:131-147`):外层 batch,内层 `taskNum = queryNum * kvNum`,其中 `kvNum = DIV_ROUND_UP(totalLen, blockSize)`(以 cache 物理块为粒度切 KV)。所有任务按 `totalIdx % block_num == block_idx` 静态分配到 Cube 核(`indexer_scores.h:145-147`),核间无依赖、无需同步。

### L1/L0 内存布局与乒乓

`Init` 中手工排布 L1 与 L0(`indexer_scores.h:47-99`):

- L1(A1):`kl1Buf[2]`(K 块,`blockSize*headDim`,乒乓)、`ql1Buf[2]`(Q tile,`MAX_N0*headDim`,乒乓)、`wl1Buf[2]`(weight tile,`MAX_N0*nHeads`,乒乓)、`kql1Buf[2]`(第一级 mmad 结果中转,`blockSize*MAX_N0`,乒乓);
- L0:A2 `l0aBuf[2]`(`MAX_M0*k0`)、B2 `l0bBuf[2]`(`MAX_N0*k0`)、CO1 `l0cBuf`(fp32 累加)。`k0 = 256/sizeof(Dtype)`(fp16/bf16 为 128)。

数据搬运均用 `CopyGmToL1Nd2Nz` 完成 ND→NZ 布局转换(`indexer_scores.h:174-179`),即 GM 的行主序在 L1 中转置成分块列主,供 mmad 按 `CopyToL0ACol/CopyToL0BCol` 装载。

### 单任务流水(事件驱动)

单个任务(query tile × KV 块)内的流水(`indexer_scores.h:171-249`),事件 ID 按缓冲轨道复用(k 用 0/1、q 用 2/3、weight 用 4/5):

1. 等 `kl1Buf[curr]` 空闲 → K 块 GM→L1(通过 `blockTable[kvIdx]` 找物理块,`block * blockSize * headDim` 定位);
2. 等 `ql1Buf[curr]` 空闲 → Q tile GM→L1;
3. K→L0A、Q→L0B,`CalMmad(l0cBuf, l0a, l0b, mBlockPad, nBlockPad, headDim)` 得到 `[kvLen, queryTileSize*nHeads]` 的头得分,`CopyL0CToL1` 暂存到 `kql1Buf`;
4. weight tile GM→L1;
5. 对 tile 内每个 query token q:weight 行(1,nHeads)→L0A、对应头得分列(kvLen,nHeads)→L0B,第二次 `CalMmad`(`m=MBLOCKSIZE`)得到 `(1, kvLen)` 的 index_score,`CopyToGm` 直接从 L0C 写 GM,目的地址 `scores[(qOffset+q) * maxNumBlock*blockSize + kvStart]`,以 `maxNumBlock*blockSize` 为 dstStride 保持行内全局位置布局(`indexer_scores.h:246-247`);
6. `curr = 1 - curr` 切换乒乓,下一任务的搬运与本任务的 mmad/写回重叠。

块尺寸对齐:kvLen 尾块按 `MBLOCKSIZE`/`NBLOCKSIZE` 向上取整(`indexer_scores.h:155-158`),mmad 的 m/n/k 均为 `kBlockSize = 32/sizeof(Dtype)` 的倍数(`indexer_scores.h:103,114`)。

### 流水线同步

全部用 `HardEvent` 计数事件配对(MTE1↔MTE2、M↔MTE1、FIX↔M、MTE1↔FIX),`Run` 开头预置 11 个 flag 使首个任务无需等待(`indexer_scores.h:120-130`),结尾逐一 `WaitFlag` 清算(`indexer_scores.h:255-265`)。mmad 结果写 GM 前经 `M_FIX`→`FIX_M` 配对保证 FIX 管线完成。
