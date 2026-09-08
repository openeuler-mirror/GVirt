# gather_sparse_kv_cache

## 功能概述

DSA(Dual Sparse Attention)decode 长序列路径的 KV 收集算子:按 indexer 给出的每 batch top-k token 下标(`topkIndices`),把 paged(分页)latent KV cache 中的 `kCache`(kvLoraRank)与 `peCache`(ropeHeadDim)逐 token 收集到每 batch 连续的 dense cache 中,供 mla_v2 的 dense 模式连续读取。本质上是一次 `dense[b, i] = paged[blockTable[b][tok/bs]*bs + tok%bs]` 的双通道 gather,host 侧要求 `kvHeads == 1`(`csrc/op.cpp:1044-1047`)。

## 输入输出参数

Python 侧调用(`tests/kernels/gather_sparse_kv_cache.py:119`):

```python
gather_sparse_kv_cache(rt, k_cache, pe_cache, block_tables, topk_indices,
                       query_lens, cached_lens, k_dense, pe_dense, batch, topK,
                       BLOCK_SIZE, kv_lora_rank, rope_head_dim)
```

host 封装 `GatherSparseKVCache`(`csrc/_C.cpp:1596`)→ `XliteOpGatherSparseKVCache`(`csrc/op.cpp:1035`)。kernel 签名(`csrc/kernels/gather_sparse_kv_cache.h:137`):

```cpp
gather_sparse_kv_cache_<dtype>(kCache, peCache, blockTables, topkIndices, queryLens,
                               cachedLens, kDenseCache, peDenseCache, batch, indexTopK,
                               blockSize, maxNumBlocks, kvLoraRank, ropeHeadDim)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| kCache | 输入 | `[numBlocks, blockSize, 1, kvLoraRank]` | BF16 | paged latent K cache |
| peCache | 输入 | `[numBlocks, blockSize, 1, ropeHeadDim]` | BF16 | paged RoPE K cache |
| blockTables | 输入 | `[batch, maxNumBlocks]` | INT32 | 逻辑块 → 物理块映射 |
| topkIndices | 输入 | `[batch, indexTopK]` | INT32 | 每 batch 的 top-k token 下标(token 级,decode 每 batch 1 个 query);无需有序 |
| queryLens / cachedLens | 输入 | `[batch]` | INT32 | decode 时 queryLens 全 1;`totalLen = queryLen + cachedLen` 决定有效槽位数 |
| kDenseCache | 输出 | `[batch, indexTopK, 1, kvLoraRank]` | BF16 | 收集后的连续 latent K(零初始化,槽位 ≥ min(totalLen, topK) 不写) |
| peDenseCache | 输出 | `[batch, indexTopK, 1, ropeHeadDim]` | BF16 | 收集后的连续 RoPE K |
| batch | 标量 | - | uint32 | batch 数 |
| indexTopK | 标量 | - | uint32 | 每 batch 收集的 token 数上限(host 语义上 ≤ `MAX_TOPK_NUM=2048`,测试覆盖 512/2048) |
| blockSize | 标量 | - | uint32 | KV cache 块大小(128) |
| maxNumBlocks | 标量 | - | uint32 | host 由 blockTables 推得 |
| kvLoraRank / ropeHeadDim | 标量 | - | uint32 | 两个通道的向量宽(测试 512/64) |

任务粒度:`totalTasks = batch * ceil(indexTopK / 16)`,每任务处理一个 batch 的 16 个连续 top-k 槽位(`CNT_ONE_LOOP=16`,`csrc/kernels/gather_sparse_kv_cache.h:11`)。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| bfloat16_t | `csrc/kernels/gather_sparse_kv_cache_bfloat16_t.cpp` | `gather_sparse_kv_cache_bfloat16_t` |

host 仅 BF16(`csrc/op.cpp:1049`);kernel 在 `#ifdef __DAV_C220_VEC__` 下,非向量核平台为空实现(`csrc/kernels/gather_sparse_kv_cache.h:147-156`)。仅在 `rt.aivNum` 个向量核上 launch(`csrc/op.cpp:1050`)。

## 实现原理

纯 AIV kernel,grid-stride 分配(`task = block_idx; task < totalTasks; task += block_num`,`csrc/kernels/gather_sparse_kv_cache.h:72`)。每任务流水四步,ping-pong 双缓冲(`kBuf/peBuf/topks` 各 2 份)重叠 16-token 组之间的 GM 访问:

1. **载入下标**:MTE2 把 16 个 `topkIndices` 拷入 UB(`copy_gm_to_ubuf_align_b16`),事件 EVENT_ID0/1 通知标量核;
2. **地址计算(标量核)**:对每个 `tok = topks[i]` 求 `physBlock[i] = blockTable[tok/blockSize]`、`rem[i] = tok % blockSize`(`csrc/kernels/gather_sparse_kv_cache.h:97-104`)。这里 blockTable 是 GM 指针,逐元素标量读——16 个下标一组的批处理让标量开销摊薄;
3. **GM→UB gather**:对 16 个 token 逐个 `CopyGmToUbufAligned` 把 `kCache[physBlock*blockSize + rem]`(kvLoraRank 个元素)与 `peCache[...]`(ropeHeadDim 个元素)拷入当前 ping-pong 槽(`csrc/kernels/gather_sparse_kv_cache.h:110-116`);源地址完全随机(top-k 本身离散、跨物理块),因此无法向量化为整段拷贝;
4. **UB→GM 散写**:16 个 token 的数据已成段拼接,一次 `CopyUbufToGmAligned` 连续写出 `kHeadBytes*cnt` 到 `kDenseCache + offset*kvLoraRank`(pe 同理,`csrc/kernels/gather_sparse_kv_cache.h:122-125`)——这是把随机访存收敛为两端连续访存的关键:GM 侧随机读不可避免,但写侧通过 UB 中转变成 burst 写。

**有效长度裁剪**:`cnt = min(16, totalLen - offsetInBatch)`,越界槽位直接跳过(`csrc/kernels/gather_sparse_kv_cache.h:85-91`),dense cache 尾部保持零初始化(测试即按"仅有效槽位比对"验证,`tests/kernels/gather_sparse_kv_cache.py:130-137`);批切换时从 GM 重新读该 batch 的 queryLen/cachedLen(EVENT_ID4 同步,`:77-84`)。

**UB 布局**:每槽 `ROUND_UP(kHeadBytes, BLOCK_SIZE=32) * 16` 字节(对齐到 32B×16 组),加 topks 两份 16×uint32,总占用由 `assert(off <= UB_SIZE)` 保证(`csrc/kernels/gather_sparse_kv_cache.h:29-45`)。

流水同步:MTE3→MTE2(缓冲写完可复用,EVENT_ID0/1)、S(标量核)→MTE2(下标可用/地址算好,EVENT_ID2/3)两类事件贯穿循环,行间 `curr = 1 - curr` 交替双缓冲(`csrc/kernels/gather_sparse_kv_cache.h:68-128`)。

**在模型中的位置**:decode + DSA 且序列较长时(`maxNumBlocks*blockSize > tileSizeOfCachedKV` 或 `> 280*batch`,阈值 `XLITE_MLA_DENSE_THRESHOLD=280`,`csrc/model.cpp:17`、`:483-508`),先本算子收集 dense cache,再以 `dense=True` 调 mla_v2(原独立 mla_v3 算子已并入 mla_v2 的 dense 路径);短序列或 prefill 时直接走 paged mla_v2,不需要本算子。

## 关键代码位置

- 主体实现:`csrc/kernels/gather_sparse_kv_cache.h:15`(函数)、`:72`(任务循环)
- 常量 `CNT_ONE_LOOP=16`:`csrc/kernels/gather_sparse_kv_cache.h:11`
- host launch:`csrc/op.cpp:1061`;Python 绑定 `csrc/_C.cpp:1599`
- 模型路由(与 mla_v2 dense 路径配合):`csrc/model.cpp:483-508`
