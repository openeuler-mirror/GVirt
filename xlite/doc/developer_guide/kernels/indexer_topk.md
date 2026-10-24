# indexer_topk

## 功能概述

DeepSeek V3.2 DSA indexer 的核心融合算子:在单个 kernel 内完成"indexer 得分计算 + 跨核全局 topk 选取"(AIC 算分数、AIV 做 topk,通过核间同步拼接)。数学语义(见 `csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init` 注释块 `scores = k * q` 与测试 `tests/kernels/indexer_topk.py` 的 `run_test` 参考实现,关键字 `torch.einsum("chd,td->cht"` / `scores.clamp(min=0)`):

```plaintext
scores[s,h,t]    = ReLU(sum_d q[s,h,d] * kCache[t,d])    # AIC 第一级 mmad + fixpipe ReLU
index_score[s,t] = sum_h scores[s,h,t] * weight[s,h]     # AIC 第二级 mmad
topkIndices[s,:] = TopK(index_score[s, :p0+1], topK)     # AIV,因果限定候选 [0, p0]
```

其中 `p0` 为该 query token 的**因果边界**(KV 空间下的含自身末位索引)。两个关键语义:

- **因果掩码**:每个 query token 只在先验及自身位置 `[0, p0]` 中选 topK(不含未来),通过按位置截断 `validKvLen` + 尾部 `FLOAT_MIN` 补齐实现(见下文)。`p0` 由 `MapPosToKv(pos)` 给出(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::MapPosToKv`):`compressRatio==1`(token 级 KV)时 `p0 = pos`(含自身);`compressRatio>1`(压缩块级 KV)时 `p0 = (pos+1)/ratio - 1`,即最后一个**已完成**压缩块的末 KV 索引,**排除 token 自身所在未完成块**(首块、`clen==0` 时为 `-1`,表示无有效 KV)。
- **稀疏/稠密分流**:`p0 < topK` 的 token 走稠密注意力(全部先验位置都参加注意力,无需选路),kernel 不为其排序,直接输出**默认模板** `0...topK-1`(升序,覆盖 `[0, topK)`);下游消费时必须过滤掉 `> p0` 的索引——模板头部的前 `min(p0+1, topK)` 项恒为 `[0, p0]` 内的有效位置(`gather_sparse_kv_cache` 依赖此约定)。

与 `indexer_scores` + `topk` 两算子组合相比,本算子把两步融合进同一 launch,并通过 `lastTopk` 环形接力避免中间大矩阵落盘。

## 输入输出参数

Python 侧调用(`tests/kernels/indexer_topk.py`):

```python
indexer_topk(rt, q, k_cache, weight, topk_indices, query_start_loc,
             query_lens, cached_lens, block_tables, n_heads, head_dim,
             block_size, batch, topK)
```

host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpIndexerTopK`:`topK >= maxNumBlocks*blockSize` 且非 `skipDenseTopk` 时全部行都是稠密,kernel 不必启动,host 侧直接把 `seqPositions` 模板前 `topK` 项 D2D 广播进 `topkIndices` 后返回(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 `topK >= maxNumBlocks * blockSize` dense-skip D2D 广播段);否则校验 `topK <= MAX_TOPK_NUM=2048`(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 `if (topK > MAX_TOPK_NUM)` throw)。`maxSeqLen` 不再是 launch 参数:host 经 `DeriveMaxNumBlocks`(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 `maxNumBlocks = DeriveMaxNumBlocks`)仅推导 `maxNumBlock` 传入;`maxSeqLen = MAX(blockSize*maxNumBlock, INIT_MIN_SEQ_POS=102400)` 由 kernel 侧(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`,关键字 `maxSeqLen = MAX((uint64_t)blockSize * maxNumBlock, INIT_MIN_SEQ_POS)`)推导。host 不校验 `seqPositions` 长度,覆盖范围由 `XModel::Init` 预生成保证(见下)。kernel 签名(`csrc/kernels/indexer_topk.h` 的 `INDEXER_TOPK_FUNC_DEFINE`):

```cpp
indexer_topk_<dtype>(GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc,
                     GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,
                     GM_ADDR scores, GM_ADDR lastTopk, GM_ADDR seqPositions,
                     GM_ADDR topkIndices, GM_ADDR sync, uint32_t kCacheStride,
                     uint32_t nHeads, uint32_t headDim, uint32_t blockSize, uint32_t batch,
                     uint32_t maxNumBlock, uint32_t topK, uint8_t skipDenseTopk,
                     uint32_t compressRatio)
```

C8(`int8_t`)入口 `INDEXER_TOPK_C8_FUNC_DEFINE`(`csrc/kernels/indexer_topk.h` 的 `INDEXER_TOPK_C8_FUNC_DEFINE`)签名不同:同样在 `sync` 后接收 `uint32_t k_cache_stride`,末参为 `GM_ADDR k_scale_cache`(无 `compressRatio`,C8 固定按 token 级 KV)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q | 输入 | `[total_query_len, nHeads*headDim]` | fp16 / bf16 / int8 | 拼接的 flatten query;C8 为 `[total_query_len,4096]` |
| kCache | 输入 | `[kvcache_block_num, blockSize, headDim]` | fp16 / bf16 / int8 | paged indexer K cache;C8 为 `[kvcache_block_num,blockSize,1,128]` |
| weight | 输入 | `[total_query_len, headDim + nHeads]` | fp16 / bf16 | 头融合权重,位于每行第 headDim 列起;C8 为独立的 fp16 `[total_query_len,32]`,已乘 Q scale |
| queryStartLoc | 输入 | `[batch]` | int32 | 各 batch query 起始偏移 |
| queryLens / cachedLens | 输入 | `[batch]` | int32 | 各 batch query / 已缓存长度;token 绝对位置 `pos = cachedLens[i] + 段内偏移`,经 `MapPosToKv` 映射为 KV 空间因果边界 `p0` |
| blockTables | 输入 | `[batch, maxNumBlock]`(2-D)或 `[batch * maxNumBlock]`(1-D 扁平) | int32 | 逻辑块 → 物理块映射;`maxNumBlock` 由 host 侧 `DeriveMaxNumBlocks` 推导(2-D 取 `shape[1]`,1-D 取 `len // batch`,`csrc/op.h` 的 `DeriveMaxNumBlocks`) |
| scores | workspace | `[2 * block_num * XLITE_MAX_M0, MAX_INDEXER_KV_TILE_LEN]` 视作 `[block_num][2][XLITE_MAX_M0 * 4096]` | fp16 / bf16 / fp32 | C8 使用 fp32;AIC→AIV 的得分中转缓冲,乒乓两份,每 block 一段(见下文布局) |
| lastTopk | workspace | `[total_query_len, 2*topK]`(uint32 对) | uint32 | 跨核 topk 接力缓冲:处理非首 KV tile 的核把当前全局 topK 候选(值+索引对)写给下一个核 |
| seqPositions | 输入 | `[max(max(blockSize*maxNumBlock, INIT_MIN_SEQ_POS), topK)]` | int32(uint32 语义) | `arange(...)` 恒等索引表,由 host 侧(`csrc/_C.cpp` `IndexerTopK` / `csrc/model.cpp` `XModel::Init`)预先生成;长度必须覆盖 kernel 侧推导的 `maxSeqLen = MAX(blockSize*maxNumBlock, INIT_MIN_SEQ_POS=102400)`(host 侧 `XliteOpIndexerTopK` 不校验长度,由 `XModel::Init` 预生成保证覆盖)。`XModel::Init` 按 `MAX(INIT_MIN_SEQ_POS, ROUND_UP(max_seq_len, blockSize))` 一次性预生成。AIV 每个 subblock 搬入属于自己的一段,之后按 `kvOffset` 增量推进;同时它也是稠密行模板的来源——首个 `topK` 项即 `0...topK-1`(模板 writer 核整体搬入 UB `defaultTopkIndices`,见下文) |
| topkIndices | 输出 | `[total_query_len, topK]` | int32(uint32 语义) | 每 query token 的 topK 索引,**按索引升序**(最小 id 在前)排列,所有行有效。`p0 >= topK` 的稀疏行是真实 topK(候选满 `topK` 个,全部落在 `[0, p0]`);`p0 < topK` 的稠密行是默认模板 `0...topK-1`,其中 `> p0` 的表项无效,下游必须过滤——模板头部的前 `min(p0+1, topK)` 项恒为有效位置(`gather_sparse_kv_cache` 依赖此约定做头部有效收集) |
| kScaleCache | 输入(C8) | `[kvcache_block_num,blockSize,1,1]` | fp16 | 与 K cache 同分页布局;Python 参数为 `k_scale_cache` |
| sync | workspace | `[AIV_TO_AIC*block_num]` | int32 | 核间环形同步 flag 数组(每核 `AIV_TO_AIC=2` 个,对应 2 个 subblock,`csrc/kernels/indexer_topk.h` 的 `AIV_TO_AIC` 宏,关键字 `#define AIV_TO_AIC 2u`) |
| kCacheStride | — | 标量 | uint32 | kCache 的 dim-0 跨度(元素数);连续时等于 blockSize×headDim(C8 为 blockSize×128),非连续时由宿主传入真实跨距 |
| nHeads / headDim / blockSize / batch / topK / skipDenseTopk | — | 标量 | uint32 / uint8 | 测试配置:nHeads=64, headDim=128, blockSize=128, topK ∈ {512, 2048};约束 `topK <= MAX_TOPK_NUM=2048`、`kvLen <= MAX_INDEXER_KV_TILE_LEN=4096`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `assert(kvLen <= MAX_INDEXER_KV_TILE_LEN && topK <= MAX_TOPK_NUM)`)。`skipDenseTopk=1` 时不写稠密模板(旧行为,稠密行内容不定),默认 0 |
| compressRatio | — | 标量 | uint32 | KV 压缩比:`1`=token 级 KV(因果含自身);`>1`=压缩块级 KV,`MapPosToKv` 把 token 位置映射到已完成压缩块边界(因果排除自身未完成块)。host 默认 `compressRatio` 由模型层传入(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 浮点路径 `launchKernel(rt.aicNum, ...)` 末尾 `compressRatio` 实参);C8 路径不传(固定 token 级) |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `indexer_topk_bfloat16_t` | `csrc/kernels/indexer_topk_bfloat16_t.cpp` | q/kCache/weight/scores 全 bf16 |
| `indexer_topk_float16_t` | `csrc/kernels/indexer_topk_float16_t.cpp` | 全 fp16 |
| `indexer_topk_int8_t` | `csrc/kernels/indexer_topk_int8_t.cpp` | Q/K 为 int8,weight/K scale 为 fp16,scores 为 fp32 |

dtype 分派见 `csrc/op.cpp` 的 `XliteOpIndexerTopK`(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 浮点 dtype 分派段 `EachXDtype(FP16, q, kCache, weight, scores)` / `EachXDtype(BF16, ...)`;C8 路径段关键字 `if (kCache.dtype == INT8)`)。浮点路径要求 q/kCache/weight/scores 同 dtype。

C8 使用 32 个 head、headDim=128,并要求 `block_size ∈ {16,32,64,128}`(须整除 `MAX_INDEXER_KV_TILE_LEN=4096`)、`topK` 为 32 的倍数且 ∈ [32, `MAX_TOPK_NUM`=2048](`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 C8 校验段 `if (kCache.dtype == INT8)` 内 `blockSize < 16 || blockSize > 128 || blockSize % 16 != 0 || MAX_INDEXER_KV_TILE_LEN % blockSize != 0 || topK < 32 || topK % 32 != 0`);内核入口额外接收 `k_scale_cache`。QK 用 int32 累加,经 `FP16(relu(dot)/1024)` 后与 weight 相乘,得到 fp32 scores;AIV 乘 K scale 后使用相同的因果遮罩和 Top-K。

## 实现原理

本算子是混合核(`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`,关键字 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`):每个 block 配 1 个 AIC(Cube,算得分)+ 2 个 AIV(Vector,topk,`AIV_TO_AIC=2`,`csrc/kernels/indexer_topk.h` 的 `AIV_TO_AIC` 宏,关键字 `#define AIV_TO_AIC 2u`)。AIC 与 AIV 通过 `ffts_cross_core_sync` 组内同步(`mode=2`,flag 定义 `csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `a2vSyncFlag` / `v2aSyncFlag` flag 定义段;置位点关键字 `ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[0/1])` 与 `ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr])` 与 `ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[curr])`):flag 0/1 由 AIC 置位表示某乒乓 scores 缓冲就绪,flag 2/3 由 AIV 置位表示缓冲已消费完毕。

### 任务划分、稠密/未来 chunk 跳过与 scores workspace 布局

`Run`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`)按 batch 枚举任务:

- `queryTileSize = XLITE_MAX_M0 / nHeads`(nHeads=64 时为 2;若 nHeads > XLITE_MAX_M0 则退化为单 token per tile,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `int queryTileSize = XLITE_MAX_M0 / nHeads` / fallback `queryTileSize = 1`);
- KV 块数以**因果边界**计:`nKv = MapPosToKv(cachedLen + queryLen - 1) + 1`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `int nKv = MapPosToKv(cachedLen + queryLen - 1) + 1`),`kvNumMax = DIV_ROUND_UP(nKv, tileSizeOfCachedKV)`,其中 `tileSizeOfCachedKV = ROUND_UP(INDEXER_KV_TILE_LEN=4096, SORT_BLOCK_SIZE=32)`(`csrc/kernels/indexer_topk.h` 的 `INDEXER_KV_TILE_LEN` 宏 与 `IndexerTopK::Init` 中 `tileSizeOfCachedKV = ROUND_UP(INDEXER_KV_TILE_LEN, SORT_BLOCK_SIZE)`,带编译期检查 `INDEXER_KV_TILE_LEN ≤ MAX_INDEXER_KV_TILE_LEN`,`csrc/kernels/indexer_topk.h` 的 `INDEXER_KV_TILE_LEN` 编译期断言,关键字 `#if INDEXER_KV_TILE_LEN > MAX_INDEXER_KV_TILE_LEN` / `#error`)。即 KV 维按 4096 切大 tile;`compressRatio>1` 时 kCache 每行对应一个压缩块,`MapPosToKv` 把 token 末位映射到压缩块空间;
- 任务总数上界 `taskNum = queryNum * kvNumMax`,但**跳过的 chunk 不消耗全局任务号**:`totalIdx` 只在任务有效时递增,再按 `totalIdx % block_num == block_idx` 静态分配(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `totalIdx++; if (totalIdx % block_num != block_idx) continue;`),稠密/未来 chunk 不会在核间留下"空任务"空洞。

每个 (query chunk, KV chunk) 任务的**稠密模板写出与位置感知跳过**(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `assignedDefaultTopk` 模板写出 + `queryPosEndKvBound` 跳过段),`queryPosBase = cachedLen + queryOffset` 为该 chunk 首 token 的绝对位置,`queryPosEnd`(含)为末 token 位置,`queryPosEndKvBound = MapPosToKv(queryPosEnd)` 为其 KV 空间因果边界:

1. **稠密行模板写出**:每个 AIV subblock 都会枚举到所有任务,但只有指定的模板 writer(`assignedDefaultTopk = !skipDenseTopk && blockIdx == block_num-1 && subBlockIdx == AIV_TO_AIC-1`,即最后一个核的 subblock 1,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `bool assignedDefaultTopk = !skipDenseTopk && blockIdx == block_num - 1 && subBlockIdx == AIV_TO_AIC - 1`)在 `kvOffset == 0 && MapPosToKv(queryPosBase) < topK` 时,把默认模板 `0...topK-1` 逐行复制到该 query chunk **稠密前缀**的各行(行内 `MapPosToKv(queryPosBase + i) >= topK` 即 `break`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `if (MapPosToKv(queryPosBase + i) >= (int32_t)topK) { break; }`)——只写 `p0 < topK` 的行,混合 chunk 的稀疏行(归属其他核做真实 topk)不被覆盖。此检查在跳过判断**之前**进行,保证整块稠密的 chunk 也被覆盖;
2. **稠密/未来 chunk 跳过**:`queryPosEndKvBound < topK`(整块稠密)或 `queryPosEndKvBound < kvOffset`(整块全未来)—— 直接 `idx = NEXT_MULTIPLE(idx, kvNumMax)` 跳过(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `if (queryPosEndKvBound < (int32_t)topK || queryPosEndKvBound < kvOffset) { idx = NEXT_MULTIPLE(idx, kvNumMax); continue; }`);
3. 否则 `kvLen = MIN(tileSizeOfCachedKV, queryPosEndKvBound - kvOffset + 1)`:KV chunk 只需覆盖到 chunk 末 token 的因果边界为止(最后一个 KV tile 通常不满)。混合 chunk(稠密行与稀疏行并存)不跳过,其稠密行在 `RunAivTopk` 内直接跳过(模板已由 writer 写出,见下文)。

`scores` workspace 布局(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`,关键字 `scores[0]`/`scores[1]` SetGlobalBuffer 段 `block_idx * XLITE_MAX_M0 * tileSizeOfCachedKV`):大小为 `[block_num][2][XLITE_MAX_M0 * 4096]` 的 dtype 数组,第 `b` 个核的第 `p` 份乒乓缓冲位于偏移 `(b + p*block_num) * XLITE_MAX_M0 * 4096`。每份缓冲按 `[queryTileSize 行, 4096 列]` 行主序存放(行距 `tileSizeOfCachedKV`),即一个 query tile × KV tile 的 index_score 矩阵。**同一 (batch, query tile) 的不同 KV tile 任务被顺序分配到不同核,由 AIV 通过 `lastTopk` 接力融合**,这是本算子区别于 `indexer_scores` 的关键。

### AIC 侧:RunAicIndexerScores(两级 mmad + fixpipe ReLU)

`RunAicIndexerScores`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAicIndexerScores`)与 `indexer_scores` 的两级 mmad 相同,差异在于:

- KV 粒度是 blockSize(128)的物理 cache 块,一个 tile 内按 `mLoop = DIV_ROUND_UP(kvLen, blockSize)` 循环,通过 `blockTable[mIdx + mIdxStart]`(`mIdxStart = kvOffset/blockSize`)逐块取 K(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAicIndexerScores`,关键字 `int mIdxStart = kvOffset / blockSize` / `uint32_t block = blockTable[mIdx + mIdxStart]`);
- weight 只在任务开始搬一次(`queryTaskLen, nHeads` → L1),因为一个 tile 内所有 KV 块共享同一 query tile;
- 第一级 mmad 的 m 为 `blockSize`(结果暂存 L1 的 `kql1Buf`),第二级对 tile 内每个 query token 输出 `(1, mSize)` 行,写到 `scores[q * tileSizeOfCachedKV + mOffset]`;
- **ReLU 融合进 L0C→L1 搬运**:第一级 per-head QK 得分在 `CopyL0CToL1` 下 L0C 时以 `reluEn=1` 过滤(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAicIndexerScores`,关键字 `CopyL0CToL1(..., 1.0f / 1024.0f, true)` C8 路径 / `CopyL0CToL1(..., /*reluEn=*/1)` 浮点路径;C8 路径以 `1/1024` 缩放取整),即每头得分先过 `max(x, 0)` 再进第二级加权求和——DSA indexer 定义中的负得分截断,不占额外向量指令;
- L1/L0 布局与 `indexer_scores` 一致(kl1/ql1/wl1/kql1 乒乓,L0A/L0B 乒乓,L0C 单份),AIC 完成后 `ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr])` 通知 AIV(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr])`)。

### AIV 侧:RunAivTopk —— 逐位置因果 topk + 分块排序 + 归并 + 跨核接力

`RunAivTopk`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`)参数 `queryPosBase` 为该 subblock 分到的首 query token 的绝对位置。对 tile 内每个 query token(`nWorkPerCore = DIV_ROUND_UP(nWork, 2)`,2 个 subblock 均分 query 行,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `int nWorkPerCore = DIV_ROUND_UP(nWork, 2)`):

**逐位置因果截断**(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `int p0 = MapPosToKv(queryPosBase + idx)` / `int validKvLen = MIN(p0 - kvOffset + 1, kvLen)` 因果截断段):

- `p0 = MapPosToKv(queryPosBase + idx)`,候选只含 `[kvOffset, kvOffset + validKvLen)`,其中 `validKvLen = MIN(p0 - kvOffset + 1, kvLen)`——同一 KV tile 内不同 query token 的有效长度随位置递增,"未来"位置(`> p0`)不进入排序,等价于因果掩码;`compressRatio>1` 时 `p0` 取已完成压缩块边界,故一个压缩块内所有 token 共享同一 `p0`(块内因果等价);
- **稠密行跳过**:`p0 < topK` 的稠密行(混合 chunk 内)不进入排序,直接 `continue`——其默认模板已由模板 writer 在任务枚举阶段写出(见上文);`skipDenseTopk=1` 时旧有稠密行同样被跳过,内容不定(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `if (p0 < (int32_t)topK || validKvLen <= 0) { continue; }` 稠密行跳过);
- `validKvLen <= 0`(该 token 的全部先验位置已在更早的 KV chunk 覆盖)同样直接 `continue`;
- `isFirst = kvOffset <= 0`(首 KV tile,无需融合上一核结果);`isFinal = kvOffset + validKvLen > p0`(该 tile 覆盖到 token 自身位置,是末 tile,负责写出最终 topk)。

**UB 布局**(Init 中一次性排布为成员缓冲,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`(`__DAV_C220_VEC__` UB 布局段,关键字 `totalSort`/`mrgSortBuf1`/`sortIndices`/`defaultTopkIndices`/`in[2]`/`lastSort[2]`/`out[2]`/`mrgSortBuf0`),总大小受 `assert(off <= invoff)` 约束):

| 缓冲 | 大小 | 用途 |
|---|---|---|
| `totalSort` | `MAX_TOPK_NUM*4*float` | 跨核融合后的全局 topK 候选(值+索引交错对,故 4 倍) |
| `mrgSortBuf1` | `MAX_INDEXER_KV_TILE_LEN*2*float` | vbitsort 归并工作区(与 `mrgSortBuf0` 分置两端避免 bank conflict) |
| `sortIndices` | `MAX_INDEXER_KV_TILE_LEN` 个 u32 | 全局索引表(单份,非乒乓;见下) |
| `defaultTopkIndices` | `topK` 个 u32 | 稠密行默认模板 `0...topK-1`(仅模板 writer 从 `seqPositions` 前 `topK` 项整体搬入一次) |
| `in[2]` | 各 `MAX_INDEXER_KV_TILE_LEN` 个 WeightDtype | 乒乓:浮点路径搬入得分,C8 搬入 K scale |
| `lastSort[2]` | 各 `MAX_TOPK_NUM*2*float` | 乒乓:上一核传来的 topK 候选 |
| `out[2]` | 各 `MAX_TOPK_NUM` 个 u32 | 乒乓:最终结果搬出 |
| `mrgSortBuf0` | `MAX_INDEXER_KV_TILE_LEN*2*float` | 计算区,置于 `UB_SIZE` 向下(高地址) |

C8 的 fp32 scores 直接搬入 `mrgSortBuf0`,`in[0]/in[1]` 交替暂存 fp16 K scale。先搬入当前 query 可见的 K scale,再等待排序工作区并搬入 scores,通过 MTE2→V `EVENT_ID0 + curr` 等待搬运完成。缩放后重新填充排序尾部。

`MAX_TOPK_NUM=2048` 与 `MAX_INDEXER_KV_TILE_LEN=4096`(均定义于 `csrc/kernels/kernel_param.h` 的 `MAX_INDEXER_KV_TILE_LEN` / `MAX_TOPK_NUM`)正是这套 UB 布局的容量上限来源,因此 host 侧强制 `topK <= 2048`(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 `if (topK > MAX_TOPK_NUM)` throw)。

**索引表分段搬入 + 增量推进**:`Run` 启动时,每个 subblock 依据 `(blockIdx*2 + subBlockIdx) % max(1, maxSeqLen/4096)` 从 `seqPositions` 搬入**属于自己的** `MIN(4096, maxSeqLen)` 项(`sortIndicesStart`/`sortIndicesBytes`;`maxSeqLen < 4096` 时所有核共享首段),其中 `maxSeqLen = MAX(blockSize*maxNumBlock, INIT_MIN_SEQ_POS)` 由核内从 block table 容量推导(不再是 launch 参数),再通过 EVENT_ID4 搬入同步;之后每个任务若 `kvOffset` 相对 `sortIndicesStart` 推进了 `idxDiff`,用一条 `vadds` 给整个表加上 `idxDiff`(起点窗口落在恒等表内任意位置都正确,`vadds` 会把它重定位到 `kvOffset`),不再逐任务从 GM 重复拷贝索引表。同时,模板 writer 把 `seqPositions` 的首个 `topK` 项(`0...topK-1`)搬入 `defaultTopkIndices` 一次,用 MTE2→MTE3 EVENT_ID0 自等待确保 MTE3 读模板时已就绪。

**topk 算法**(基于硬件排序指令,非堆、非位图):

1. **搬入 + 转 fp32**:tile 得分(`scores + idx*4096`,前 `validKvLen` 个)搬入并 `convert_input`(`csrc/kernels/kernel_macro.h` 的 `convert_input`)成 float;尾部 `[validKvLen, sortRepeat*SORT_BLOCK_SIZE)` 用带 mask 的 `vector_dup(FLOAT_MIN)` 补齐(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `vector_dup(mrgSortBuf0 + fullRepeat * calcPad, float(min)` 尾部补齐段)——无效位置以 `FLOAT_MIN` 参与排序,天然不会胜出;
2. **块内排序**:`vbitsort(mrgSortBuf1, mrgSortBuf0, sortIndices, sortRepeat)` 按 `SORT_BLOCK_SIZE=32` 一组做位排序(值+原始索引成对输出,`sortRepeat = DIV_ROUND_UP(validKvLen, 32)`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `int sortRepeat = DIV_ROUND_UP(validKvLen, SORT_BLOCK_SIZE)`);
3. **归并截断**:`MrgSort`(`csrc/kernels/kernel_macro.h` 的 `MrgSort`)用 `vmrgsort4` 四路归并循环把所有 32 元素组归并成单个降序序列,`topK` 参数让归并尽早截断到 topK 宽度;首 tile 时以 `preferredDst = totalSort` 直接把结果写进 `totalSort`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `MrgSort(mrgSortBuf1, mrgSortBuf0, sortRepeat, &localSort, topK, totalSort)` 首 tile preferredDst=totalSort),省去一次中间缓冲拷贝;
4. **与上一核候选融合**:
   - 非 `isFirst`:先 `WaitPrevCore()` 等前一个核把它的全局候选写入本核的 `lastTopk` 段,搬入 `lastSort[curr]`(`lastSortLen = MIN(topK, kvOffset)` 对),`vmrgsort4(totalSort, {localSort, lastSort}, ...)` 二路归并(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `WaitPrevCore()` + `vmrgsort4(totalSort, addrs, totalSortLen | (lastSortLen << 16)` 二路归并段);
   - `isFirst`:`totalSort` 已由步骤 3 直接得到;
5. **中间结果传递**(`!isFinal`):把 `totalSort` 的前 `MIN(topK, totalSortLen)` 对(值,索引)写入 `lastTopk + idx*topK*2` 供下一核消费;最后一个 query 处理完后 `SetNextCore()` 唤醒下一核(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `CopyUbufToGmAligned(lastTopk + idx * topK * 2, totalSort` + `SetNextCore()`)。MTE3 异步读 `totalSort` 的释放等待用 `totalSortOnHold` 标记**推迟到下次真正要写它之前**(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `totalSortOnHold` 标记多处 `wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2)` / `set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2)`),避免空等;
6. **末 tile 收尾**(`isFinal`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `isFinal` 末 tile 收尾段 `vreducev2(index0, ...)` / `vmuls(totalSort, ..., -1.0f` / `vbitsort + MrgSort` 按索引升序):`vreducev2` 从 `totalSort` 抽出索引载荷;由于 `vbitsort` 按 **float** 键比较(位 cast 的 int 仅在符号位为 0 时保序),而 topK id 可为任意 `[0, p0]` 值,这里将非负索引按位视作 float,乘 `-1` 作为排序键(其降序即 id 升序),对 topK 个候选走 `vbitsort + MrgSort` **按索引从小到大**排列,再 `vreducev2` 抽出原始索引,经 `out[curr]` 的 `CopyUbufToGmAligned` 写到 `topkIndices + idx*topK`。最终输出为**按索引升序**的 topK 位置列表(测试按集合比较稀疏行、稠密行按模板精确比较,`tests/kernels/indexer_topk.py`;`gather_sparse_kv_cache` 顺序无关地消费稀疏行——升序/降序/乱序均可,运行时检测连续段组 burst——稠密行则依赖模板头部 `[0, p0]` 有效这一约定)。

### 核间环形同步(software pipe)

KV tile 顺序分布在 `block_num` 个核上,本算子用 **GM flag 数组 + generation 计数**实现单向环形链(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::SetNextCore` / `IndexerTopK::WaitPrevCore` / `IndexerTopK::ResetPrevCore`):

- `sync` 数组每核 `AIV_TO_AIC=2` 个 u32(`sync[blockIdx*AIV_TO_AIC + subBlockIdx]`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`,关键字 `setNextSync = (__gm__ int32_t *)sync + blockIdx * AIV_TO_AIC + subBlockIdx` / `waitPrevSync = ...`),核 `i` 的 `setNextSync` 即核 `(i+1)%block_num` 的 `waitPrevSync`;
- `SetNextCore()`:把 `setNextGeneration` 写入下一核的 flag(generation 每次递增,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::SetNextCore`);
- `WaitPrevCore()`:自旋 `copy_gm_to_ubuf` 轮询上一核 flag 直到 `*val >= waitPrevGeneration`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::WaitPrevCore`);
- `ResetPrevCore()`:核处理完属于自己的所有非首 tile 后,把上一核的 flag 清零,供下一次 launch 复用(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::ResetPrevCore`,在 `Run` 末尾按 `resetPrevCore` 标志执行,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `if (resetPrevCore) { ResetPrevCore(); }`)。

由此形成软件流水:核 `i` 在做第 `k` 个 KV tile 的 topk 时,核 `i+1` 可以并行做第 `k+1` 个 tile 的得分计算(AIC)与排序,只在融合点同步。

### AIC/AIV 乒乓同步

`Run` 主循环内(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 batch/task 主循环段 `for (int batchIdx = 0; batchIdx < batch; batchIdx++)`):AIC 消费 `v2aSyncFlag[curr]`(AIV 置位的"缓冲空闲"),写完 scores 后置 `a2vSyncFlag[curr]`;AIV 反之。两份 scores 乒乓使 AIC 写第 `curr+1` 个任务与 AIV 排序第 `curr` 个任务重叠。AIV 侧事件:MTE2 搬入用 EVENT_ID0/1(in)、2/3(lastSort),MTE3 搬出用 EVENT_ID0/1(out)、EVENT_ID2(lastTopk 写出/`totalSort` 释放);`sortIndices` 搬入用 EVENT_ID4、模板 writer 的 `defaultTopkIndices` 搬入用 EVENT_ID0(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `CopyGmToUbufAligned(this->defaultTopkIndices` / `CopyGmToUbufAligned(this->sortIndices` 模板/索引表搬入段)。循环末尾清空所有挂起 flag(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `RunAivTopk` 末尾 `wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3)` 起的清算段)并 `PipeBarrier<PIPE_ALL>()`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `Run` 末尾 `PipeBarrier<PIPE_ALL>()` / `if (resetPrevCore) ResetPrevCore()`)。
