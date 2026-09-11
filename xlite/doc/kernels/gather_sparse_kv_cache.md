# gather_sparse_kv_cache

## 功能概述

DSA(Dual Sparse Attention)decode 长序列路径的 KV 收集算子:按 indexer 给出的每 batch top-k token 下标(`topkIndices`),把 paged(分页)cache 中的 `kCache`(kvLoraRank)与 `peCache`(ropeHeadDim)逐 token 收集到每 batch 连续的 dense cache 中,供 mla_v2 / cxa 的 dense 模式连续读取。本质上是一次 `dense[b, i] = paged[blockTable[b][tok/bs]*bs + tok%bs]` 的双通道 gather,host 侧要求 `kvHeads == 1`(`csrc/op.cpp:1098-1101`)。

`compressRatio` 参数控制 `totalLen` 的统计粒度:`compressRatio == 0` 时 `totalLen = queryLen + cachedLen`(按原始 token 计);否则 `totalLen = (queryLen + cachedLen) / compressRatio`(按压缩 token 计,压缩 KV 场景)。`compressRatio` 默认为 1,即按原始 token 收集(MLA 场景)。cxa 的 dense 模式内部以 `compressRatio` 调用本算子收集压缩 KV(`csrc/_C.cpp:1650-1653`)。

## 输入输出参数

Python 侧调用(`tests/kernels/gather_sparse_kv_cache.py:119`):

```python
gather_sparse_kv_cache(rt, k_cache, pe_cache, block_tables, topk_indices,
                       query_lens, cached_lens, k_dense, pe_dense, batch, topK,
                       BLOCK_SIZE, kv_lora_rank, rope_head_dim)
```

Python 绑定 `gather_sparse_kv_cache`(`csrc/_C.cpp:2818`,`kv_heads` 默认 1,无 `compress_ratio` 形参)→ `GatherSparseKVCache`(`csrc/_C.cpp:1599`)→ `XliteOpGatherSparseKVCache`(`csrc/op.cpp:1088`)。kernel 签名(`csrc/kernels/gather_sparse_kv_cache.h:15`):

```cpp
gather_sparse_kv_cache_<dtype>(kCache, peCache, blockTables, topkIndices, queryLens,
                               cachedLens, kDenseCache, peDenseCache, batch, indexTopK,
                               blockSize, maxNumBlocks, kvLoraRank, ropeHeadDim,
                               compressRatio = 1)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| kCache | 输入 | `[numBlocks, blockSize, 1, kvLoraRank]` | BF16 | paged K cache(MLA 为 latent K,压缩 KV 为压缩 K) |
| peCache | 输入 | `[numBlocks, blockSize, 1, ropeHeadDim]` | BF16 | paged PE / RoPE K cache;cxa dense 模式内部调用时传空 |
| blockTables | 输入 | `[batch, maxNumBlocks]` | INT32 | 逻辑块 → 物理块映射 |
| topkIndices | 输入 | `[batch, indexTopK]` | INT32 | 每 batch 的 top-k token 下标(token 级,decode 每 batch 1 个 query);无需有序 |
| queryLens / cachedLens | 输入 | `[batch]` | INT32 | decode 时 queryLens 全 1;`totalLen = compressRatio==0 ? queryLen+cachedLen : (queryLen+cachedLen)/compressRatio` 决定有效槽位数 |
| kDenseCache | 输出 | `[batch, indexTopK, 1, kvLoraRank]` | BF16 | 收集后的连续 K(零初始化,槽位 ≥ min(totalLen, topK) 不写) |
| peDenseCache | 输出 | `[batch, indexTopK, 1, ropeHeadDim]` | BF16 | 收集后的连续 PE / RoPE K;cxa dense 模式内部调用时传空 |
| batch | 标量 | - | uint32 | batch 数 |
| indexTopK | 标量 | - | uint32 | 每 batch 收集的 token 数上限(host 语义上 ≤ `MAX_TOPK_NUM=2048`(`csrc/kernels/kernel_param.h:42`),测试覆盖 512/2048) |
| blockSize | 标量 | - | uint32 | KV cache 块大小(128) |
| maxNumBlocks | 标量 | - | uint32 | host 由 blockTables 推得(`DeriveMaxNumBlocks`,`csrc/op.cpp:1102`) |
| kvLoraRank / ropeHeadDim | 标量 | - | uint32 | 两个通道的向量宽(测试 512/64);cxa dense 模式内部调用时 kvLoraRank=headDim、ropeHeadDim=0(仅 K 通道) |
| compressRatio | 标量 | - | uint32 | 压缩比;`0` 按 `(qLen+cLen)` 统计 totalLen,否则按 `(qLen+cLen)/compressRatio` 统计;默认 1。cxa dense 模式内部传入实际压缩比 |

`kCacheGm`/`peCacheGm`/`kDenseGm`/`peDenseGm` 均做非空判断(`csrc/kernels/gather_sparse_kv_cache.h:114`、`:118`、`:130`、`:133`),为空时该通道跳过,故 cxa dense 模式可只收集 K 通道(pe 全空)。

任务粒度:`totalTasks = batch * ceil(indexTopK / 16)`,每任务处理一个 batch 的 16 个连续 top-k 槽位(`CNT_ONE_LOOP=16`,`csrc/kernels/gather_sparse_kv_cache.h:11`)。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| bfloat16_t | `csrc/kernels/gather_sparse_kv_cache_bfloat16_t.cpp` | `gather_sparse_kv_cache_bfloat16_t` |

host 仅 BF16(`csrc/op.cpp:1103`);kernel 在 `#ifdef __DAV_C220_VEC__` 下,非向量核平台为空实现(`csrc/kernels/gather_sparse_kv_cache.h:158-167`)。仅在 `rt.aivNum` 个向量核上 launch(`csrc/op.cpp:1104`)。

## 实现原理

纯 AIV kernel,grid-stride 分配(`task = block_idx; task < totalTasks; task += block_num`,`csrc/kernels/gather_sparse_kv_cache.h:70`)。每任务流水四步,ping-pong 双缓冲(`kBuf/peBuf/topks` 各 2 份)重叠 16-token 组之间的 GM 访问:

1. **载入下标**:MTE2 把 16 个 `topkIndices` 拷入 UB(`copy_gm_to_ubuf_align_b16`),事件 EVENT_ID0/1 通知标量核;
2. **地址计算(标量核)**:对每个 `tok = topks[i]` 求 `physBlock[i] = blockTable[tok/blockSize]`、`rem[i] = tok % blockSize`(`csrc/kernels/gather_sparse_kv_cache.h:99-104`)。这里 blockTable 是 GM 指针,逐元素标量读——16 个下标一组的批处理让标量开销摊薄;
3. **GM→UB gather**:对 16 个 token 逐个 `CopyGmToUbufAligned` 把 `kCache[physBlock*blockSize + rem]`(kvLoraRank 个元素)与 `peCache[...]`(ropeHeadDim 个元素)拷入当前 ping-pong 槽(`csrc/kernels/gather_sparse_kv_cache.h:112-122`);源地址完全随机(top-k 本身离散、跨物理块),因此无法向量化为整段拷贝;
4. **UB→GM 散写**:16 个 token 的数据已成段拼接,一次 `CopyUbufToGmAligned` 连续写出 `kHeadBytes*cnt` 到 `kDenseCache + offset*kvLoraRank`(pe 同理,`csrc/kernels/gather_sparse_kv_cache.h:128-135`)——这是把随机访存收敛为两端连续访存的关键:GM 侧随机读不可避免,但写侧通过 UB 中转变成 burst 写。

**有效长度裁剪**:`cnt = min(16, totalLen - offsetInBatch)`,越界槽位直接跳过(`csrc/kernels/gather_sparse_kv_cache.h:87-93`),dense cache 尾部保持零初始化(测试即按"仅有效槽位比对"验证,`tests/kernels/gather_sparse_kv_cache.py:136-140`);批切换时从 GM 重新读该 batch 的 queryLen/cachedLen 并按 `compressRatio` 重算 totalLen(EVENT_ID4 同步,`csrc/kernels/gather_sparse_kv_cache.h:75-86`)。

**UB 布局**:每槽 `ROUND_UP(kHeadBytes, BLOCK_SIZE=32) * 16` 字节(对齐到 32B×16 组),加 topks 两份 16×uint32,总占用由 `assert(off <= UB_SIZE)` 保证(`csrc/kernels/gather_sparse_kv_cache.h:30-46`)。

流水同步:MTE3→MTE2(缓冲写完可复用,EVENT_ID0/1)、S(标量核)→MTE2(下标可用/地址算好,EVENT_ID2/3)两类事件贯穿循环,行间 `curr = 1 - curr` 交替双缓冲(`csrc/kernels/gather_sparse_kv_cache.h:66-139`)。

**在模型中的位置**:decode + DSA 且序列较长时(`maxNumBlocks*blockSize > tileSizeOfCachedKV` 或 `> 280*batch`,阈值 `XLITE_MLA_DENSE_THRESHOLD=280`,`csrc/model.cpp:17`、`:486-509`),先本算子收集 dense cache,再以 `dense=True` 调 mla_v2(原独立 mla_v3 算子已并入 mla_v2 的 dense 路径);短序列或 prefill 时直接走 paged mla_v2,不需要本算子。cxa 的 dense 模式亦在 binding 层(`csrc/_C.cpp:1650`)内部调用本算子,把压缩 KV 的 top-k token 收集成连续 dense cache 后再进 cxa kernel。
