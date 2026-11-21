# indexer_topk

## 功能概述

DSA indexer 的核心融合算子:在单个 kernel 内完成"indexer 得分计算 + topk 选取"(AIC 算分数、AIV 做 topk)。分核采用**混合策略,由 host 在算子下发前决策**(`csrc/auto_tuner.cpp` 的 `PickIndexerKvSplit`,模型路径在 `XRuntime::PrepareAttn` 每个 forward 执行一次,结果存 `rt._indexerKvSplit`,经 `uint8_t kvSplit` launch 参数传入 kernel):稀疏 query tile 数达到 block 数一半时(`kvSplit=0`),一个 (batch, query tile) 任务整体归属一个 block,该 block 串行走完该 tile 的所有 KV tile,AIC 每算完一个 KV tile 的 scores 通知本 block 的 AIV,各 block 之间完全解耦;tile 数不足 block 数一半时(`kvSplit=1`,典型如少 batch 长 cache 的 decode),退回 (tile, chunk) 粒度平铺分核 + 跨核接力,让空闲 block 也参与并把 chunk 间的 AIC/AIV 流水摊到多核。数学语义(见 `csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init` 注释块 `scores = k * q` 与测试 `tests/kernels/indexer_topk.py` 的 `run_test` 参考实现,关键字 `torch.einsum("chd,td->cht"` / `scores.clamp(min=0)`):

```plaintext
scores[s,h,t]    = ReLU(sum_d q[s,h,d] * kCache[t,d])    # AIC 第一级 mmad + fixpipe ReLU
index_score[s,t] = sum_h scores[s,h,t] * weight[s,h]     # AIC 第二级 mmad
topkIndices[s,:] = TopK(index_score[s, :p0+1], topK)     # AIV,因果限定候选 [0, p0]
```

其中 `p0` 为该 query token 的**因果边界**(KV 空间下的含自身末位索引)。两个关键语义:

- **因果掩码**:每个 query token 只在先验及自身位置 `[0, p0]` 中选 topK(不含未来),通过按位置截断 `validKvLen` + 尾部 `FLOAT_MIN` 补齐实现(见下文)。`p0` 由 `MapPosToKv(pos)` 给出(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::MapPosToKv`):`compressRatio==1`(token 级 KV)时 `p0 = pos`(含自身);`compressRatio>1`(压缩块级 KV)时 `p0 = (pos+1)/ratio - 1`,即最后一个**已完成**压缩块的末 KV 索引,**排除 token 自身所在未完成块**(首块、`clen==0` 时为 `-1`,表示无有效 KV)。
- **稀疏/稠密分流**:`p0 < topK` 的 token 走稠密注意力(全部先验位置都参加注意力,无需选路),kernel 不为其排序,直接输出**默认模板** `0...topK-1`(升序,覆盖 `[0, topK)`);下游消费时必须过滤掉 `> p0` 的索引——模板头部的前 `min(p0+1, topK)` 项恒为 `[0, p0]` 内的有效位置(`gather_sparse_kv_cache` 依赖此约定)。

与 `indexer_scores` + `topk` 两算子组合相比,本算子把两步融合进同一 launch,并通过 `lastTopk`(KV tile 间的逐位置中转)避免中间大矩阵落盘。

## 输入输出参数

Python 侧调用(`tests/kernels/indexer_topk.py`,两种 `kv_split` 全用例覆盖;模型路径由 PrepareAttn 自动决策,standalone 由调用方显式选择):

```python
indexer_topk(
    rt,
    q,
    k_cache,
    weight,
    topk_indices,
    query_start_loc,
    query_lens,
    cached_lens,
    block_tables,
    n_heads,
    head_dim,
    block_size,
    batch,
    topK,
    kv_split=0,  # 0 = serial query-tile mode, 1 = (tile, chunk) relay mode
)
```

host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpIndexerTopK`:`topK >= maxNumBlocks*blockSize` 且非 `skipDenseTopk` 时全部行都是稠密,kernel 不必启动,host 侧直接把 `seqPositions` 模板前 `topK` 项 D2D 广播进 `topkIndices` 后返回(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 `topK >= maxNumBlocks * blockSize` dense-skip D2D 广播段);否则校验 `topK <= MAX_TOPK_NUM=2048`(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 `if (topK > MAX_TOPK_NUM)` throw)。`maxSeqLen` 不再是 launch 参数:`maxNumBlock`(运行期块数)由 host 取 `blockTables.shape[1]` 传入(V2 表已由 PrepareAttn 收窄到运行期宽度,模型层 `ForwardAttnIndexer` 关键字 `rt._attnBlockTables[0].shape[1]`;standalone/测试入口由 `SliceBlockTableToLens` 从 lens 推导收窄),`blockTableStride = blockTables.stride[0]` 一并传入;`maxSeqLen = MAX(blockSize*maxNumBlock, INIT_MIN_SEQ_POS=102400)` 由 kernel 侧(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`,关键字 `maxSeqLen = MAX((uint64_t)blockSize * maxNumBlock, INIT_MIN_SEQ_POS)`)推导。host 不校验 `seqPositions` 长度,覆盖范围由 `XModel::Init` 预生成保证(见下)。kernel 签名(`csrc/kernels/indexer_topk.h` 的 `INDEXER_TOPK_FUNC_DEFINE`):

```cpp
indexer_topk_<dtype>(GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc,
                     GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,
                     GM_ADDR scores, GM_ADDR lastTopk, GM_ADDR seqPositions,
                     GM_ADDR topkIndices, GM_ADDR sync, uint32_t kCacheStride,
                     uint32_t nHeads, uint32_t headDim, uint32_t blockSize, uint32_t batch,
                     uint32_t blockTableStride, uint32_t maxNumBlock, uint32_t topK, uint8_t skipDenseTopk,
                     uint32_t compressRatio, uint32_t weightCols, uint8_t kvSplit)
```

C8(`int8_t`)入口 `INDEXER_TOPK_C8_FUNC_DEFINE`(`csrc/kernels/indexer_topk.h` 的 `INDEXER_TOPK_C8_FUNC_DEFINE`)签名不同:同样在 `sync` 后接收 `uint32_t k_cache_stride`,末五参为 `GM_ADDR k_scale_cache`、`uint32_t k_scale_cache_stride`、`uint32_t compress_ratio`、`uint32_t weight_cols` 与 `uint8_t kvSplit`(因果边界语义与浮点路径相同,C8 同样支持压缩块级 KV)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q | 输入 | `[total_query_len, nHeads*headDim]` | fp16 / bf16 / int8 | 拼接的 flatten query;C8 为 `[total_query_len, nHeads*128]` |
| kCache | 输入 | `[kvcache_block_num, blockSize, headDim]` | fp16 / bf16 / int8 | paged indexer K cache;C8 为 `[kvcache_block_num,blockSize,1,128]` |
| weight | 输入 | `[total_query_len, nHeads]` 或 `[total_query_len, headDim + nHeads]` | fp16 / bf16 | 头权重,行距由 `weightCols = shape[1]` 决定(op.cpp 派生并校验):稠密 `[T, nHeads]` 从每行第 0 列起(C8 的 scaledWeights 已乘 Q scale;CXA bf16 的独立 head weights);DSA 融合 kw `[T, headDim+nHeads]` 尾对齐于每行第 headDim 列起。内核按 `wOffset = qto*weightCols + (weightCols-nHeads)` 寻址 |
| queryStartLoc | 输入 | `[batch]` | int32 | 各 batch query 起始偏移 |
| queryLens / cachedLens | 输入 | `[batch]` | int32 | 各 batch query / 已缓存长度;token 绝对位置 `pos = cachedLens[i] + 段内偏移`,经 `MapPosToKv` 映射为 KV 空间因果边界 `p0` |
| blockTables | 输入 | `[rows >= batch, blockTableStride]` | int32 | 逻辑块 → 物理块映射;持久化缓冲,行跨度为分配跨度,仅每行 lens 推导的有效前缀被读取;行跨度 `blockTableStride` 仅用于行寻址 |
| scores | workspace | `[2 * block_num * MAX_INDEXER_Q_TILE_LEN, MAX_INDEXER_KV_TILE_LEN]` 视作 `[block_num][2][MAX_INDEXER_Q_TILE_LEN * 4096]` | fp16 / bf16 / fp32 | C8 使用 fp32;AIC→AIV 的得分中转缓冲,乒乓两份,每 block 一段(见下文布局) |
| lastTopk | workspace | `[total_query_len, 2*topK]`(uint32 对) | uint32 | KV tile 间的 topk 中转缓冲:处理非末 KV tile 时,把该 query token 当前累积的全局 topK 候选(值+索引对)写出,下一个 KV tile 读回融合——`kvSplit=0` 时下一个 tile 属同一核(行独占,无竞争),`kvSplit=1` 时属于接办该 chunk 的下一个 block(经 `sync` 接力同步) |
| sync | workspace | `[AIV_TO_AIC*block_num]` | int32 | 核间接力同步 flag 数组(每核 `AIV_TO_AIC=2` 个,对应 2 个 subblock),仅 `kvSplit=1` 模式使用;`kvSplit=0` 模式下分配与清零照常但 kernel 不触碰 |
| kvSplit | — | 标量 | uint8 | 分核模式,host 下发前由 `PickIndexerKvSplit` 决策(模型路径:PrepareAttn 存 `rt._indexerKvSplit`;standalone 入口:外部显式传入 `kv_split`):`0` = 串行 tile 模式(稀疏 query tile 数 ≥ block_num/2),`1` = chunk 平铺 + 跨核接力模式(稀疏 query tile 数 < block_num/2) |
| seqPositions | 输入 | `[max(max(blockSize*maxNumBlock, INIT_MIN_SEQ_POS), topK)]` | int32(uint32 语义) | `arange(...)` 恒等索引表,由 host 侧(`csrc/_C.cpp` `IndexerTopK` / `csrc/model.cpp` `XModel::Init`)预先生成;长度必须覆盖 kernel 侧推导的 `maxSeqLen = MAX(blockSize*maxNumBlock, INIT_MIN_SEQ_POS=102400)`(host 侧 `XliteOpIndexerTopK` 不校验长度,由 `XModel::Init` 预生成保证覆盖)。`XModel::Init` 按 `MAX(INIT_MIN_SEQ_POS, ROUND_UP(max_seq_len, blockSize))` 一次性预生成。AIV 每个 subblock 搬入属于自己的一段,之后按 `kvOffset` 增量推进;同时它也是稠密行模板的来源——首个 `topK` 项即 `0...topK-1`(模板 writer 核整体搬入 UB `defaultTopkIndices`,见下文) |
| topkIndices | 输出 | `[total_query_len, topK]` | int32(uint32 语义) | 每 query token 的 topK 索引,**按索引升序**(最小 id 在前)排列,所有行有效。`p0 >= topK` 的稀疏行是真实 topK(候选满 `topK` 个,全部落在 `[0, p0]`);`p0 < topK` 的稠密行是默认模板 `0...topK-1`,其中 `> p0` 的表项无效,下游必须过滤——模板头部的前 `min(p0+1, topK)` 项恒为有效位置(`gather_sparse_kv_cache` 依赖此约定做头部有效收集) |
| kScaleCache | 输入(C8) | `[kvcache_block_num,blockSize,1,1]` | fp16 | 与 K cache 同分页布局;Python 参数为 `k_scale_cache`。DSV4 的 vllm 原生池把 K 与 scale 按页打包进同一分配(scale 视图带 storage_offset、stride[0] = 每 block 字节数/2),允许 pitch |
| kCacheStride | — | 标量 | uint32 | kCache 的 dim-0 跨度(元素数);连续时等于 blockSize×headDim(C8 为 blockSize×128),非连续时由宿主传入真实跨距 |
| kScaleCacheStride | — | 标量(C8) | uint32 | kScaleCache 的 dim-0 跨度(fp16 元素数);dense 池传 0,内核以 `MAX(kScaleCacheStride, blockSize)` 守护后退化为 block 内连续寻址。核内 scale 读取为 `kScaleCache + block*kScaleCacheStride`(块内连续),vllm 打包页的 storage_offset 常量偏移由基指针吸收 |
| nHeads / headDim / blockSize / batch / topK / skipDenseTopk | — | 标量 | uint32 / uint8 | 测试配置:nHeads=64, headDim=128, blockSize=128, topK ∈ {512, 2048};host 校验 `nHeads` 为 16 的倍数且 ≤64、`headDim` 为 16 的倍数且 ≤128(C8 额外 `%32`,见下节)、`blockSize` 为 16 的倍数且 ≤128 并整除 `MAX_INDEXER_KV_TILE_LEN=4096`(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 `nHeads == 0 || nHeads % 16 != 0 || nHeads > 64` 共通校验段);约束 `topK <= MAX_TOPK_NUM=2048`、`kvLen <= MAX_INDEXER_KV_TILE_LEN=4096`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `assert(kvLen <= MAX_INDEXER_KV_TILE_LEN && topK <= MAX_TOPK_NUM)`)。`skipDenseTopk=1` 时不写稠密模板(旧行为,稠密行内容不定),默认 0 |
| compressRatio | — | 标量 | uint32 | KV 压缩比:`1`=token 级 KV(因果含自身);`>1`=压缩块级 KV,`MapPosToKv` 把 token 位置映射到已完成压缩块边界(因果排除自身未完成块)。host 由模型层传入(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,浮点路径 `launchKernel(rt.aicNum, ...)` 与 C8 路径 `aclrtlaunch_indexer_topk_int8_t` 末尾均传 `compressRatio` 实参) |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `indexer_topk_bfloat16_t` | `csrc/kernels/indexer_topk_bfloat16_t.cpp` | q/kCache/weight/scores 全 bf16 |
| `indexer_topk_float16_t` | `csrc/kernels/indexer_topk_float16_t.cpp` | 全 fp16 |
| `indexer_topk_int8_t` | `csrc/kernels/indexer_topk_int8_t.cpp` | Q/K 为 int8,weight/K scale 为 fp16,scores 为 fp32 |

dtype 分派见 `csrc/op.cpp` 的 `XliteOpIndexerTopK`(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 浮点 dtype 分派段 `EachXDtype(FP16, q, kCache, weight, scores)` / `EachXDtype(BF16, ...)`;C8 路径段关键字 `if (kCache.dtype == INT8)`)。浮点路径要求 q/kCache/weight/scores 同 dtype。

host 对**所有 dtype** 共通校验 `nHeads` 为 16 的倍数且 ≤64、`blockSize` 为 16 的倍数、≤128 且整除 `MAX_INDEXER_KV_TILE_LEN=4096`、`headDim` 为 16 的倍数且 ≤128(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 `nHeads == 0 || nHeads % 16 != 0 || nHeads > 64` 共通校验段;kernel 侧另有 L0C 容量断言 `assert(nHeads <= 64 && blockSize <= 128)`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`,关键字 `Consider L0C capacity, limit heads and blockSize`);C8 路径(`if (kCache.dtype == INT8)`)额外要求 `headDim % 32 == 0`(关键字 `INT8 K cache requires headDim % 32 == 0`)及 `kScaleCache` 非空、`q` 为 INT8、`weight`/`kScaleCache` 为 FP16、`scores` 为 FP32;`topK` 仅约束 `≤ MAX_TOPK_NUM=2048`(不再要求 32 倍数)。C8 典型配置为 32 个 head、headDim=128;内核入口额外接收 `k_scale_cache`。QK 用 int32 累加,经 `FP16(relu(dot)/1024)` 后与 weight 相乘,得到 fp32 scores;AIV 乘 K scale 后使用相同的因果遮罩和 Top-K。

## 实现原理

本算子是混合核(`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`,关键字 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`):每个 block 配 1 个 AIC(Cube,算得分)+ 2 个 AIV(Vector,topk,`AIV_TO_AIC=2`,`csrc/kernels/indexer_topk.h` 的 `AIV_TO_AIC` 宏,关键字 `#define AIV_TO_AIC 2u`)。AIC 与 AIV 通过 `ffts_cross_core_sync` 组内同步(`mode=2`,flag 定义 `csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `a2vSyncFlag` / `v2aSyncFlag` flag 定义段;置位点关键字 `ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[0/1])` 与 `ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr])` 与 `ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[curr])`):flag 0/1 由 AIC 置位表示某乒乓 scores 缓冲就绪,flag 2/3 由 AIV 置位表示缓冲已消费完毕。

### 混合分核策略、稠密/未来 chunk 跳过与 scores workspace 布局

**模式自判**(host 下发前,每个 forward 一次):`PrepareAttn` 先用 `PickIndexerQTile`(`csrc/auto_tuner.cpp`)选定本 forward 的 query tile 长度(见下),再由 `PickIndexerKvSplit`(`csrc/auto_tuner.cpp` 的 `PickIndexerKvSplit`)以同一 tile 统计**稀疏 query tile 数** `sparseTiles`——对每个 query tile,若其末 token 因果边界 `p0End >= topK`(稀疏,`p0End` 复刻 kernel 的 `MapPosToKv`:ratio≤1 为 token 位置、否则为已完成压缩块边界)则计数加一;`kvSplit = sparseTiles < blockNum / 2`。**不能按 (tile, chunk) 任务数判**——少 tile 多 chunk 的长 cache decode(如 batch=1 cachedLen≥10 万)任务数足以过关,但串行模式只会用 1 个核;阈值取**半满**(blockNum/2)而非满 block 是 910B 同窗交替 A/B 的实测结论:sparseTiles 超过 block 数一半后,各 tile 的 chunk 接力链会在同一批 block 上排队(每 chunk q/w L1 重载 + 跨核 topK 接力),而串行模式此时已满宽度且对 batch 平坦(decode 并发翻转点 ≈ block_num/2:8 并发 64k-KV 时 split 快 24%,12/16 并发反慢 16%/34%,16 并发 64k 下 split 492.4us vs 串行 366.1us),故 [blockNum/2, blockNum) 区间应留在串行模式。调用点:模型路径在 `XRuntime::PrepareAttn`(host 镜像 `rt._hostLens`/`rt._cachedLensHost` 上计算,无 D2H,结果存 `rt._indexerKvSplit`,DSA 取 ratio=1、CXA 取 indexer 层的 ratio=4),`csrc/op.cpp` 的 `XliteOpIndexerTopK` 只透传 launch(launch 用的 block 数即决策比较的 `rt.aicNum`);standalone 入口(`csrc/_C.cpp` 的 `IndexerTopK`)没有 PrepareAttn,由外部通过 `kv_split`/`q_tile` 参数显式传入(测试 `tests/kernels/indexer_topk.py` 双模式×tile 全用例覆盖;perf 脚本 `tests/perf/indexer_topk.py` 用 `-S 0/1`/`-Q` 选择)。CP 路径(`ForwardAttnMLAV2DsaCp`)同样消费 `rt._indexerKvSplit`(按全局 lens 决策)。**有意不按 rank 重判**:CP 跑的是 prefill 宽 tile,窗口内(全局 ≥blockNum/2 串行而本 rank 切片后 <blockNum/2)的实测(910B, tile=22, S0/S1 交替中位数)表明宽 tile 下 S0/S1 优劣由 KV chunk 链长主导、与本 rank tile 数无关——cl=65536(8 chunk)split 劣 16~43%,cl=16384(2 chunk)split 优 8~12%,cl=4096(1 chunk)持平;`blockNum/2` 翻转点是 decode 薄 tile(ql=1)的实测结论,不能外推到宽 tile(曾按 rank 重判的 99fdcf83 已实测证伪并回退)。**CXA(ratio=4)同样消费该半满阈值,但实测网格未覆盖 ratio=4**(A/B 网格全部按 ratio=1/DSA 标定——41 例可测负载,10 例同噪声 tie、31 例分出胜负,其中旧阈值判错 3、新阈值 0 错;standalone 绑定的 `compress_ratio` 参数 perf 脚本从未透传):ratio=4 下 kCache 行为压缩块,同 token 位置的 KV 链长约为 DSA 的 1/4,`MapPosToKv` 使稀疏判定移动到 `cl ≳ 4·topK` token——半满翻转点在 CXA 的链长坐标下未标定,已知链长会移动翻转点(见上 CP 段实测),故窗口 [blockNum/2, blockNum) 内的 CXA 负载(长 cache CXA decode、≥12 稀疏 tile 的 CXA prefill)的翻转点可能不在半满;调 CXA 前须先按 ratio=4 补标定(`tests/perf/indexer_topk.py` 加 `compress_ratio` 透传即可,绑定已支持)。两种模式:

- **`kvSplit=0`(串行 tile 模式,稀疏 query tile 数 ≥ block_num/2,prefill/大 batch)**:`Run`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`)按 batch 枚举任务,一个任务 = 一个 (batch, query tile),`totalIdx` 只对非稠密 tile 递增,再按 `totalIdx % block_num == block_idx` 静态分配(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `totalIdx++; if (totalIdx % block_num != block_idx) { continue; }`);分到任务的 block 在内层循环串行走完该 tile 的所有 KV tile(`for (int kvOffset = 0; kvOffset <= queryPosEndKvBound; kvOffset += tileSizeOfCachedKV)`),核间无任何同步与数据依赖;
- **`kvSplit=1`(chunk 平铺模式,稀疏 query tile 数 < block_num/2,典型为少 batch 长 cache 的 decode)**:退回 `(query tile, KV tile)` 平铺任务,`kvNumMax = DIV_ROUND_UP(nKv, tileSizeOfCachedKV)`(`nKv = MapPosToKv(cachedLen + queryLen - 1) + 1`),任务按 `totalIdx % block_num` 轮转分核让空闲 block 参与,相邻 chunk 靠跨核接力传递 topK(见下文),chunk 间的 AIC/AIV 流水摊到多个 block 上;**跳过的 chunk 不消耗全局任务号**(`totalIdx` 只在任务有效时递增),核间不留"空任务"空洞;
- `queryTileSize`(编译期默认 `MAX_INDEXER_Q_TILE_LEN = 24`,与 nHeads 无关;`csrc/kernels/kernel_param.h` 的 `MAX_INDEXER_Q_TILE_LEN`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init` 由 `qTileOverride` 覆写、`IndexerTopK::Run` 读 `this->queryTileSize`)——query 维按 tile 长度切 tile,`queryNum = DIV_ROUND_UP(queryLen, queryTileSize)`,AIC 的 q/weight L1 缓冲亦按 `MAX_INDEXER_Q_TILE_LEN` token 上限分配。tile 长度由 host 每 forward 一次的 `PickIndexerQTile`(`csrc/auto_tuner.cpp`)按批内最大稀疏 prefill qlen 分档:≤2048 → 22、≤4096 → 17、decode-only 与 >4096 → 编译期默认 24(910B 实测 chunked prefill/混合批快 3.2%/6.9%;tile 25 较 24 回退 1.6-2.8%,故上限档维持 24);模型路径决策结果存 `rt._indexerQTile` 透传 launch,standalone 入口用 `q_tile` 参数显式指定(0 = 默认,> `MAX_INDEXER_Q_TILE_LEN` 抛错拒绝);KV tile 大小 `tileSizeOfCachedKV = ROUND_UP(INDEXER_KV_TILE_LEN=4096, SORT_BLOCK_SIZE=32)`(`csrc/kernels/indexer_topk.h` 的 `INDEXER_KV_TILE_LEN` 宏 与 `IndexerTopK::Init` 中 `tileSizeOfCachedKV = ROUND_UP(INDEXER_KV_TILE_LEN, SORT_BLOCK_SIZE)`),受两条编译期 `static_assert` 约束:`INDEXER_KV_TILE_LEN ≤ MAX_INDEXER_KV_TILE_LEN` 与 `INDEXER_KV_TILE_LEN > MAX_TOPK_NUM`(`csrc/kernels/indexer_topk.h` 的 `INDEXER_KV_TILE_LEN` 编译期断言,关键字 `static_assert(INDEXER_KV_TILE_LEN`——后者保证非 final 位置恒为稀疏行,是下文 EVENT_ID5 fence 1:1 配对证明的前提)。`compressRatio>1` 时 kCache 每行对应一个压缩块,`MapPosToKv` 把 token 末位映射到压缩块空间。

每个 (batch, query tile) 任务的**稠密模板写出与跳过**(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `assignedDefaultTopk` 模板写出 + `queryPosEndKvBound` 跳过段),`queryPosBase = cachedLen + queryOffset` 为该 tile 首 token 的绝对位置,`queryPosEnd`(含)为末 token 位置,`queryPosEndKvBound = MapPosToKv(queryPosEnd)` 为其 KV 空间因果边界:

1. **稠密行模板写出**:每个 AIV subblock 都会枚举到所有任务,但只有指定的模板 writer(`assignedDefaultTopk = !skipDenseTopk && blockIdx == block_num-1 && subBlockIdx == AIV_TO_AIC-1`,即最后一个核的 subblock 1,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `bool assignedDefaultTopk = !skipDenseTopk && blockIdx == block_num - 1 && subBlockIdx == AIV_TO_AIC - 1`)在 `MapPosToKv(queryPosBase) < topK` 时,把默认模板 `0...topK-1` 逐行复制到该 tile **稠密前缀**的各行(行内 `MapPosToKv(queryPosBase + i) >= topK` 即 `break`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `if (MapPosToKv(queryPosBase + i) >= (int32_t)topK) { break; }`)——只写 `p0 < topK` 的行,混合 tile 的稀疏行(后续做真实 topk)不被覆盖。此检查在跳过判断**之前**进行,保证整块稠密的 tile 也被覆盖;`kvSplit=1` 的平铺展开下额外带 `kvOffset == 0` 守卫(tile 的 chunk0 任务时写,tile 级一次),`kvSplit=0` 的 tile 循环天然每 tile 一次;
2. **稠密/未来 chunk 跳过**:`kvSplit=0` 时 `queryPosEndKvBound < topK`(整 tile 稠密,每个 token 的因果边界都 `< topK`)直接 `continue`,不消耗全局任务号(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `if (queryPosEndKvBound < (int32_t)topK) { continue; }`);`kvSplit=1` 时沿用平铺跳过 `queryPosEndKvBound < topK || queryPosEndKvBound < kvOffset` → `idx = NEXT_MULTIPLE(idx, kvNumMax)` 跳到下一个 query tile(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `idx = NEXT_MULTIPLE(idx, kvNumMax)`),稠密/未来 chunk 不消耗任务号,核间不留空任务空洞;
3. `kvLen = MIN(tileSizeOfCachedKV, queryPosEndKvBound - kvOffset + 1)`:KV tile 只需覆盖到 tile 末 token 的因果边界为止(最后一个 KV tile 通常不满)。混合 tile(稠密行与稀疏行并存)的稠密行在 `RunAivTopk` 内直接跳过(模板已由 writer 写出,见下文)。

两种模式共用同一个 `(query tile, KV tile)` 执行体 `RunTaskChunk`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunTaskChunk`):AIC 侧 `kvSplit=1` 时每个工作项都重载 q/w 进 L1(chunk 散在不同 block,无法复用),`kvSplit=0` 时每 tile 只在其首个 chunk 加载一次(守卫 `if (kvSplit || kvOffset == 0)`,q/w 的 `CopyGmToL1Nd2Nz` 内联于此,见下节);AIV 侧两 subblock 均分 query 行。

`scores` workspace 布局(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`,关键字 `scores[0]`/`scores[1]` SetGlobalBuffer 段 `block_idx * MAX_INDEXER_Q_TILE_LEN * tileSizeOfCachedKV`):大小为 `[block_num][2][MAX_INDEXER_Q_TILE_LEN * 4096]` 的 dtype 数组,第 `b` 个核的第 `p` 份乒乓缓冲位于偏移 `(b + p*block_num) * MAX_INDEXER_Q_TILE_LEN * 4096`。每份缓冲按 `[queryTileSize 行, 4096 列]` 行主序存放(行距 `tileSizeOfCachedKV`),即一个 query tile × KV tile 的 index_score 矩阵。AIV 持续经 `lastTopk` 在相邻 KV tile 间逐 token 传递累积 topK(见下文),这是本算子区别于 `indexer_scores` 的关键。

### AIC 侧:RunAicIndexerScores(逐 token 两级 mmad + MTE2/MTE1/FIX 乒乓流水)

`RunAicIndexerScores`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAicIndexerScores`)仍是 `indexer_scores` 的两级 mmad 结构(QK → 加权求和),但第一级改为**逐 query token** 发起(mmad 分形 `m0=blockSize, n0=nHeads, k0=headDim`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init` 注释块),并以 MTE2/MTE1/FIX 事件链的块级、token 级双层乒乓把拷贝与计算重叠起来:

- **q 与 weight 的 GM→L1 加载由 `RunTaskChunk` 统一发起**(至多 24 token:`queryTaskLen*nHeads*headDim` 与 `queryTaskLen*nHeads` 一次性 `CopyGmToL1Nd2Nz` 进单份 `ql1Buf`/`wl1Buf`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunTaskChunk`,关键字 `copy q (queryTaskLen, nHeads, headDim) to L1`,守卫 `if (kvSplit || kvOffset == 0)`):`kvSplit=0` 时同一 tile 的所有 KV tile 串行处理且共享同一 query tile,仅首个 chunk 加载一次;`kvSplit=1` 时 tile 的 chunk 散在不同 block,每个工作项都需重载;
- **块级乒乓(MTE2)**:KV 粒度是 blockSize 的物理 cache 块,一个 tile 内按 `mLoop = DIV_ROUND_UP(kvLen, blockSize)` 循环,通过 `blockTable[mIdx + mIdxStart]`(`mIdxStart = kvOffset/blockSize`)逐块取 K 进乒乓 `kl1Buf[kPingpong]`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAicIndexerScores`,关键字 `int mIdxStart = kvOffset / blockSize` / `uint32_t block = blockTable[mIdx + mIdxStart]`)——下一块的 GM→L1 与本块的逐 token 计算重叠;
- **token 级乒乓(MTE1/M/FIX 双事件族)**:块内逐 token 处理,两组事件族交替——族 0/1(q 行 L0B `l0bBuf`、QK 结果 L0C `qkl0cBuf`)与族 2/3(weight 行 L0A `wl0aBuf`、QK 结果 L1 `kql1Buf`、scores L0B `kql0bBuf`、scores L0C `l0cBuf`)。先行段(`if (queryLen != 0)`)发出 token 0 的整条 QK 链(L0B/L0A 拷贝 → mmad → FIX→`kql1Buf`);此后循环内迭代 q 先经 `if (q + 1 < queryLen)` 把 token q+1 的 QK 链预取进 `next = 1 - curr` 组,再消费 `curr` 组做第二级 scores mmad(`m0=MBLOCKSIZE=16, n0=blockSize, k0=nHeads`)并 `CopyToGm` 输出 `(1, mSize)` 行到 `scores[q * tileSizeOfCachedKV + mOffset]`——token q 的 scores 相与 token q+1 的 QK 相重叠(关键字 `int next = 1 - curr` 预取段 / `mmad scores (1, blockSize)` 消费段);
- K 进 L0A 的 `l0aBuf` 为单份缓冲(事件 4):QK mmad 以它为 A 操作数,按块串行,M 与 MTE1 通过 EVENT_ID4 互斥(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAicIndexerScores`,关键字 `CopyToL0ACol(l0aBuf, kl1Buf[kPingpong]`);
- **从 Nd2Nz 布局的 L1 读 q/weight 须带列块步距**:`CopyToL0ACol`/`CopyToL0BCol`(`csrc/kernels/kernel_macro.h`)新增可选参 `srcNzC0Stride`,q 搬运传 `nTotalBlockPad`、weight 搬运传 `wmBlockPad` 按 NZ 列块步距寻址(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAicIndexerScores`,关键字 `CopyToL0BCol(l0bBuf[curr], ql1Buf, nBlockNum, 0, kBlockNum, nTotalBlockPad)`);省缺 0 保持旧的连续布局语义,`indexer_scores` 等既有调用不受影响;
- **ReLU 融合进 L0C→L1 搬运**:per-head QK 得分在 `CopyL0CToL1` 下 L0C 时以 `reluEn=1` 过滤(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAicIndexerScores`,关键字 `CopyL0CToL1(..., 1.0f / 1024.0f, true)` C8 路径 / `CopyL0CToL1(..., /*reluEn=*/1)` 浮点路径;C8 路径以 `1/1024` 缩放取整),即每头得分先过 `max(x, 0)` 再进第二级加权求和——DSA indexer 定义中的负得分截断,不占额外向量指令;
- L1/L0 布局:ql1/wl1 任务级单份(事件 4,24 token 上限);kl1 乒乓(事件 0/1,与 `indexer_scores` 一致),kql1 乒乓(事件 2/3,indexer_scores 为 0/1);l0aBuf 单份(事件 4),l0bBuf 乒乓(事件 0/1),wl0aBuf/kql0bBuf/l0cBuf 乒乓(事件 2/3),qkl0cBuf 乒乓(事件 0/1)——L0C 不再是 QK 与 scores 共享的单份 CO1,两相各有独立乒乓。AIC 完成后 `ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr])` 通知 AIV(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunTaskChunk`,关键字 `ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr])`)。

### AIV 侧:RunAivTopk —— 逐位置因果 topk + 分块排序 + 归并 + 跨 KV tile 融合

`RunAivTopk`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`)参数 `queryPosBase` 为该 subblock 分到的首 query token 的绝对位置。对 tile 内每个 query token(`nWorkPerCore = DIV_ROUND_UP(nWork, 2)`,2 个 subblock 均分 query 行,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `int nWorkPerCore = DIV_ROUND_UP(nWork, 2)`):

**逐位置因果截断**(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `int p0 = MapPosToKv(queryPosBase + idx)` / `int validKvLen = MIN(p0 - kvOffset + 1, kvLen)` 因果截断段):

- `p0 = MapPosToKv(queryPosBase + idx)`,候选只含 `[kvOffset, kvOffset + validKvLen)`,其中 `validKvLen = MIN(p0 - kvOffset + 1, kvLen)`——同一 KV tile 内不同 query token 的有效长度随位置递增,"未来"位置(`> p0`)不进入排序,等价于因果掩码;`compressRatio>1` 时 `p0` 取已完成压缩块边界,故一个压缩块内所有 token 共享同一 `p0`(块内因果等价);
- **稠密行跳过**:`p0 < topK` 的稠密行(混合 chunk 内)不进入排序,直接 `continue`——其默认模板已由模板 writer 在任务枚举阶段写出(见上文);`skipDenseTopk=1` 时旧有稠密行同样被跳过,内容不定(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `if (p0 < (int32_t)topK || validKvLen <= 0) { continue; }` 稠密行跳过);
- `validKvLen <= 0`(该 token 的全部先验位置已在更早的 KV chunk 覆盖)同样直接 `continue`;
- `isFirst = kvOffset <= 0`(tile 的首 KV tile,无需融合上一 tile 结果);`isFinal = kvOffset + validKvLen > p0`(该 tile 覆盖到 token 自身位置,是末 tile,负责写出最终 topk)。

**UB 布局**(Init 中一次性排布为成员缓冲,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`(`__DAV_C220_VEC__` UB 布局段,关键字 `totalSort`/`mrgSortBuf1`/`sortIndices`/`defaultTopkIndices`/`in[2]`/`lastSort[2]`/`out[2]`/`mrgSortBuf0`),总大小受 `assert(off <= invoff)` 约束):

| 缓冲 | 大小 | 用途 |
|---|---|---|
| `totalSort` | `MAX_TOPK_NUM*4*float` | 融合后的全局 topK 候选(值+索引交错对,故 4 倍) |
| `mrgSortBuf1` | `MAX_INDEXER_KV_TILE_LEN*2*float` | vbitsort 归并工作区(与 `mrgSortBuf0` 分置两端避免 bank conflict) |
| `sortIndices` | `MAX_INDEXER_KV_TILE_LEN` 个 u32 | 全局索引表(单份,非乒乓;见下) |
| `defaultTopkIndices` | `topK` 个 u32 | 稠密行默认模板 `0...topK-1`(仅模板 writer 从 `seqPositions` 前 `topK` 项整体搬入一次) |
| `in[2]` | 各 `MAX_INDEXER_KV_TILE_LEN` 个 WeightDtype | 乒乓:浮点路径搬入得分,C8 搬入 K scale |
| `lastSort[2]` | 各 `MAX_TOPK_NUM*2*float` | 乒乓:上一 KV tile 传来的 topK 候选 |
| `out[2]` | 各 `MAX_TOPK_NUM` 个 u32 | 乒乓:最终结果搬出 |
| `mrgSortBuf0` | `MAX_INDEXER_KV_TILE_LEN*2*float` | 计算区,置于 `UB_SIZE` 向下(高地址) |

C8 的 fp32 scores 直接搬入 `mrgSortBuf0`,`in[0]/in[1]` 交替暂存 fp16 K scale。先搬入当前 query 可见的 K scale,再等待排序工作区并搬入 scores,通过 MTE2→V `EVENT_ID0 + curr` 等待搬运完成。缩放后重新填充排序尾部。

`MAX_TOPK_NUM=2048` 与 `MAX_INDEXER_KV_TILE_LEN=4096`(均定义于 `csrc/kernels/kernel_param.h` 的 `MAX_INDEXER_KV_TILE_LEN` / `MAX_TOPK_NUM`)正是这套 UB 布局的容量上限来源,因此 host 侧强制 `topK <= 2048`(`csrc/op.cpp` 的 `XliteOpIndexerTopK`,关键字 `if (topK > MAX_TOPK_NUM)` throw)。

**索引表分段搬入 + 增量推进**:`Run` 启动时,每个 subblock 依据 `(blockIdx*2 + subBlockIdx) % max(1, maxSeqLen/4096)` 从 `seqPositions` 搬入**属于自己的** `MIN(4096, maxSeqLen)` 项(`sortIndicesStart`/`sortIndicesBytes`;`maxSeqLen < 4096` 时所有核共享首段),其中 `maxSeqLen = MAX(blockSize*maxNumBlock, INIT_MIN_SEQ_POS)` 由核内从 launch 参数 `maxNumBlock`(运行期块数,由 lens 推导)计算,再通过 EVENT_ID4 搬入同步;之后每个任务若 `kvOffset` 相对 `sortIndicesStart` 推进了 `idxDiff`,用一条 `vadds` 给整个表加上 `idxDiff`(起点窗口落在恒等表内任意位置都正确,`vadds` 会把它重定位到 `kvOffset`),不再逐任务从 GM 重复拷贝索引表。同时,模板 writer 把 `seqPositions` 的首个 `topK` 项(`0...topK-1`)搬入 `defaultTopkIndices` 一次,用 MTE2→MTE3 EVENT_ID0 自等待确保 MTE3 读模板时已就绪。

**topk 算法**(基于硬件排序指令,非堆、非位图):

1. **搬入 + 转 fp32**:tile 得分(`scores + idx*4096`,前 `validKvLen` 个)搬入并 `convert_input`(`csrc/kernels/kernel_macro.h` 的 `convert_input`)成 float;尾部 `[validKvLen, sortRepeat*SORT_BLOCK_SIZE)` 用带 mask 的 `vector_dup(FLOAT_MIN)` 补齐(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `vector_dup(mrgSortBuf0 + fullRepeat * calcPad, float(min)` 尾部补齐段)——无效位置以 `FLOAT_MIN` 参与排序,天然不会胜出;
2. **块内排序**:`vbitsort(mrgSortBuf1, mrgSortBuf0, sortIndices, sortRepeat)` 按 `SORT_BLOCK_SIZE=32` 一组做位排序(值+原始索引成对输出,`sortRepeat = DIV_ROUND_UP(validKvLen, 32)`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `int sortRepeat = DIV_ROUND_UP(validKvLen, SORT_BLOCK_SIZE)`);
3. **归并截断**:`MrgSort`(`csrc/kernels/kernel_macro.h` 的 `MrgSort`)用 `vmrgsort4` 四路归并循环把所有 32 元素组归并成单个降序序列,`topK` 参数让归并尽早截断到 topK 宽度;首 tile 时以 `preferredDst = totalSort` 直接把结果写进 `totalSort`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `MrgSort(mrgSortBuf1, mrgSortBuf0, sortRepeat, &localSort, topK, totalSort)` 首 tile preferredDst=totalSort),省去一次中间缓冲拷贝;
4. **与上一 KV tile 候选融合**(读序按模式二选一,每 chunk 调用一次,`lastChunkFenceWaited` 守护):
   - 非 `isFirst` 且 `kvSplit=0`:首个非首位置读前 `wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5)` 等待上一 chunk 调用(同核)里 `lastTopk` 的 MTE3 写完成;非 `isFirst` 且 `kvSplit=1`:首个非首位置前 `WaitPrevCore()` 自旋等待接办上一 chunk 的 block 完成并发布候选(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `if (kvSplit) { WaitPrevCore(); ... } else { wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5); }`);随后搬入 `lastSort[curr]`(`lastSortLen = MIN(topK, kvOffset)` 对),`vmrgsort4(totalSort, {localSort, lastSort}, ...)` 二路归并(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `vmrgsort4(totalSort, addrs, totalSortLen | (lastSortLen << 16)` 二路归并段);
   - `isFirst`:`totalSort` 已由步骤 3 直接得到;
5. **中间结果传递**(`!isFinal`):把 `totalSort` 的前 `MIN(topK, totalSortLen)` 对(值,索引)写入 `lastTopk + idx*topK*2`,供下一个 KV tile 读回融合(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `CopyUbufToGmAligned(lastTopk + idx * topK * 2, totalSort`)。`kvSplit=0` 时本次调用处理到 tile 内最后一个 query 位置(`idx == queryLen - 1`)且该位置非 final 时,同分支内 `set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5)` 完成与下一调用 wait 的配对(见下文 fence 小节);`kvSplit=1` 时最后一个 query(`idx == queryLen - 1`)处理完后 `SetNextCore()` 唤醒接办下一 chunk 的 block(内置 `PipeBarrier<PIPE_ALL>()` 保证写完成),kernel 尾部按 `resetPrevCore` 标志 `ResetPrevCore()` 清零上一 block 的 flag 供下次 launch 复用。MTE3 异步读 `totalSort` 的释放等待用 `totalSortOnHold` 标记**推迟到下次真正要写它之前**(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `totalSortOnHold` 标记多处 `wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2)` / `set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2)`),避免空等;
6. **末 tile 收尾**(`isFinal`,`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `isFinal` 末 tile 收尾段 `vreducev2(index0, ...)` / `vmuls(totalSort, ..., -1.0f` / `vbitsort + MrgSort` 按索引升序):`vreducev2` 从 `totalSort` 抽出索引载荷;由于 `vbitsort` 按 **float** 键比较(位 cast 的 int 仅在符号位为 0 时保序),而 topK id 可为任意 `[0, p0]` 值,这里将非负索引按位视作 float,乘 `-1` 作为排序键(其降序即 id 升序),对 topK 个候选走 `vbitsort + MrgSort` **按索引从小到大**排列,再 `vreducev2` 抽出原始索引,经 `out[curr]` 的 `CopyUbufToGmAligned` 写到 `topkIndices + idx*topK`。最终输出为**按索引升序**的 topK 位置列表(测试按集合比较稀疏行、稠密行按模板精确比较,`tests/kernels/indexer_topk.py`;`gather_sparse_kv_cache` 顺序无关地消费稀疏行——升序/降序/乱序均可,运行时检测连续段组 burst——稠密行则依赖模板头部 `[0, p0]` 有效这一约定)。

### KV tile 间候选传递与定序(按模式)

`lastTopk` 的每一行(tile 内第 `idx` 个 query token)在 KV tile 间逐级融合,读写的定序按模式各走一条路径(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `kvSplit` 分支段):

**`kvSplit=0`(同核 + MTE3→MTE2 fence,EVENT_ID5)**:任务的 KV tile 串行处理全部发生在同一个 block 内,行只被所属核读写——核间完全解耦,无环形链/GM flag/generation。唯一需要显式保证的是**同核跨调用**的存储序:chunk `i` 调用经 MTE3 写 `lastTopk`,chunk `i+1` 调用经 MTE2 读回——MTE2 与 MTE3 是独立队列,必须由事件 fence 定序:

- **set**(写方,每 chunk 调用至多一次):处理到 tile 内最后一个 query 位置(`idx == queryLen - 1`)且该位置非 final 时,在该 `!isFinal` 分支内 `set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5)`,断言 MTE3 队列中此前所有写(含本次 `lastTopk` 拷贝)完成——非末 chunk 调用中 tile 末 token 恒非 final(其因果边界 `p0 = queryPosEndKvBound ≥ kvOffset + tileSizeOfCachedKV`),故 set 恰好每调用一次;
- **wait**(读方,每 chunk 调用至多一次):下一 chunk 调用在首个非 `isFirst` 位置读 `lastTopk` 前 `wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5)`,插入 MTE2 队列,阻塞后续读直到上一调用的 set;
- **1:1 配对证明**:chunk `i` 存在非 final 位置 ⟺ 存在 `p0 >= kvOffset_i + tileSizeOfCachedKV` 的位置 ⟺ chunk `i+1` 存在活跃且非 isFirst 的位置,谓词严格对偶;set/wait 逐 chunk 消费;tile 首 chunk 无 wait、末 chunk 无 set,跨任务也不残留。又因 `tileSizeOfCachedKV=4096 > MAX_TOPK_NUM=2048`(编译期 `static_assert`),非 final 位置恒为稀疏行(`p0 ≥ 4096 > topK`),不会出现"set 了但下一 chunk 走稠密跳过不 wait"的失配。两个 AIV subblock 各自独立走这套 fence(各自枚举自己的位置半区,谓词在半区内同样对偶)。

**`kvSplit=1`(跨核接力,software ring)**:KV tile 顺序分布在多个 block 上,用 **GM flag 数组 + generation 计数**实现单向环形链(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::SetNextCore` / `IndexerTopK::WaitPrevCore` / `IndexerTopK::ResetPrevCore`):`sync` 数组每核 `AIV_TO_AIC=2` 个 u32(`sync[blockIdx*AIV_TO_AIC + subBlockIdx]`),核 `i` 的 set 槽即核 `(i+1)%block_num` 的 wait 槽;`SetNextCore()` 把递增的 generation 写入下一核的 flag(内置 `PipeBarrier<PIPE_ALL>` 保证此前 `lastTopk` 的 MTE3 写全部完成);**调用点必须先 `set_flag/wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0)` 等 MTE3 队列排空**——`SetNextCore` 的标量 `*val` 写在 UB 偏移 0,与 `totalSort` 起始地址别名,不等排空会污染尚未完成的 `lastTopk` 拷贝源;`WaitPrevCore()` 自旋 `copy_gm_to_ubuf` 轮询上一核 flag 直到 `*val >= waitPrevGeneration`——轮询成功即链上全部前驱写完成,天然涵盖"同核前序 chunk"的可见性,因此不需要 EVENT_ID5 fence;`ResetPrevCore()` 在 kernel 尾部把上一核的 flag 清零供下次 launch 复用。由此 chunk 间的 AIC/AIV 流水摊到多个 block:block `i` 的 AIC 算 chunk `j+1` 时,block `i+1` 的 AIV 融合 chunk `j`,只在融合点同步。

`kvSplit=0` 下各 block 完全并行、互不等待:AIC 算 chunk `i+1` 的 scores 时,本 block 的 AIV 经 scores 乒乓并行融合 chunk `i` 的 topk。

### AIC/AIV 乒乓同步

`Run` 主循环内(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 batch/task 主循环段 `for (int batchIdx = 0; batchIdx < batch; batchIdx++)`):AIC 消费 `v2aSyncFlag[curr]`(AIV 置位的"缓冲空闲"),写完 scores 后置 `a2vSyncFlag[curr]`;AIV 反之。两份 scores 乒乓使 AIC 算下一个 KV tile 的 scores 与 AIV 融合当前 KV tile 的 topk 重叠。AIV 侧事件:MTE2 搬入用 EVENT_ID0/1(in)、2/3(lastSort),MTE3 搬出用 EVENT_ID0/1(out)、EVENT_ID2(lastTopk 写出/`totalSort` 释放);跨 chunk `lastTopk` 读写序按模式走 MTE3→MTE2 的 EVENT_ID5 fence(`kvSplit=0`)或 `sync` 接力链(`kvSplit=1`,见上节);`sortIndices` 搬入用 EVENT_ID4、模板 writer 的 `defaultTopkIndices` 搬入用 EVENT_ID0(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `CopyGmToUbufAligned(this->defaultTopkIndices` / `CopyGmToUbufAligned(this->sortIndices` 模板/索引表搬入段)。循环末尾清空所有挂起 flag(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::RunAivTopk`,关键字 `RunAivTopk` 末尾 `wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3)` 起的清算段)并 `PipeBarrier<PIPE_ALL>()`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Run`,关键字 `Run` 末尾 `PipeBarrier<PIPE_ALL>()`)。

### 片上缓冲分配与占用汇总(AIC L1/L0A/L0B/L0C、AIV UB)

`Init`(`csrc/kernels/indexer_topk.h` 的 `IndexerTopK::Init`)手动排布全部片上缓冲:AIC 侧(`__DAV_C220_CUBE__`)依次在 L1(`TPosition::A1`)/L0A(`A2`)/L0B(`B2`)/L0C(`CO1`)上累进 `bufferAddr` 偏移,L1 四个缓冲连续排布,L0A/L0B/L0C 三级物理独立、各自从偏移 0 重新起排;AIV 侧(`__DAV_C220_VEC__`)在 UB 上排布,低地址区自 0 顺序累进、`mrgSortBuf0` 钉在顶端,`assert(off <= invoff)` 保证低区不越界。硬件容量(910B/dav C220,NpuArch 2201,CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`):L0A 64KB、L0B 64KB、L1 512KB,L0C 总容量 128KB（`l0_c_size=131072`，16 bank × 8KB），Mmad 累加缓冲即 `TPosition::CO1` 且占满整个 L0C;AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),每个 AIV subblock 各持一份独立 UB、布局相同。

dtype 记法(模板实参,见 `INDEXER_TOPK_FUNC_DEFINE` / `INDEXER_TOPK_C8_FUNC_DEFINE`):`Dtype` 为 q/KCache 元素类型,`WeightDtype` 为 weight 与 per-head QK 得分类型,`MatDtype` 为 mmad 累加器类型——浮点路径为 `<fp16|bf16, float, fp16|bf16, fp16|bf16>`(2B/2B/4B),C8 路径为 `<int8_t, int32_t, half, float>`(1B/2B/4B);`WeightDtype` 在所有路径恒为 2B,故 UB 布局与 dtype 无关。下表 shape 为逻辑形状(L1/L0 内实际按 Nd2Nz/NZ 分形排布),单份大小列为与代码一致的字节数公式;占用列为典型模型配置下该变量的实际字节数(含乒乓份数)——两条路径同取 nHeads=64、headDim=128、blockSize=128,占用差异仅来自 dtype。

**AIC L1(A1,容量 512KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 浮点占用(B) | C8 占用(B) |
|---|---|---|---|---|---|---|
| `kl1Buf` | `(blockSize, headDim)` | Dtype | 2(事件0/1) | `blockSize*headDim*sizeof(Dtype)` | 65536 | 32768 |
| `kql1Buf` | `(blockSize, nHeads)` | WeightDtype | 2(事件2/3) | `blockSize*nHeads*sizeof(WeightDtype)` | 32768 | 32768 |
| `ql1Buf` | `(24, nHeads, headDim)` | Dtype | 1(事件4) | `MAX_INDEXER_Q_TILE_LEN*nHeads*headDim*sizeof(Dtype)` | 393216 | 196608 |
| `wl1Buf` | `(32, nHeads)` | WeightDtype | 1(事件4) | `ROUND_UP(MAX_INDEXER_Q_TILE_LEN,MBLOCKSIZE)*nHeads*sizeof(WeightDtype)` | 4096 | 4096 |
| **合计** | | | | | **495616** | **266240** |

**AIC L0A(A2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 浮点占用(B) | C8 占用(B) |
|---|---|---|---|---|---|---|
| `l0aBuf` | `(blockSize, headDim)` | Dtype | 1(事件4) | `blockSize*headDim*sizeof(Dtype)` | 32768 | 16384 |
| `wl0aBuf` | `(16, nHeads)` | WeightDtype | 2(事件2/3) | `MBLOCKSIZE*nHeads*sizeof(WeightDtype)` | 4096 | 4096 |
| **合计** | | | | | **36864** | **20480** |

**AIC L0B(B2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 浮点占用(B) | C8 占用(B) |
|---|---|---|---|---|---|---|
| `l0bBuf` | `(nHeads, headDim)` | Dtype | 2(事件0/1) | `nHeads*headDim*sizeof(Dtype)` | 32768 | 16384 |
| `kql0bBuf` | `(blockSize, nHeads)` | WeightDtype | 2(事件2/3) | `blockSize*nHeads*sizeof(WeightDtype)` | 32768 | 32768 |
| **合计** | | | | | **65536** | **49152** |

**AIC L0C(CO1 区,容量 128KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 浮点占用(B) | C8 占用(B) |
|---|---|---|---|---|---|---|
| `qkl0cBuf` | `(blockSize, nHeads)` | MatDtype | 2(事件0/1) | `blockSize*nHeads*sizeof(MatDtype)` | 65536 | 65536 |
| `l0cBuf` | `(16, blockSize)` | float | 2(事件2/3) | `MBLOCKSIZE*blockSize*sizeof(float)` | 16384 | 16384 |
| **合计** | | | | | **81920** | **81920** |

**AIV UB(预算 `UB_SIZE=192KB`,每 subblock 一份)**

低地址区各槽位按 `VECTOR_MAX_BYTESIZE=256B` 向上对齐推进;`totalSort` 槽位额外加 `UB_BANK_CONFLICT_OFFSET=256B` 垫片,使其后缓冲相对 512B bank group 行错开半行,与钉在 UB 顶端的 `mrgSortBuf0`(512B 对齐)错开以避免 bank conflict;`mrgSortBuf0` 起点 `invoff = UB_SIZE - ROUND_UP(MAX_INDEXER_KV_TILE_LEN*2*sizeof(float), UB_BANKGROUP_ROW_SIZE=512)` = 163840。占用与 dtype 无关(`WeightDtype` 恒 2B),下表占用按 topK=2048 计。

| 变量 | shape(每份) | dtype | 乒乓个数 | 大小(每份) | 占用(B) |
|---|---|---|---|---|---|
| `totalSort` | 8192 float(`MAX_TOPK_NUM*4`,值+索引交错对) | float | 1 | 32768 B | 33024(槽位,含垫片) |
| `mrgSortBuf1` | 8192 float(`MAX_INDEXER_KV_TILE_LEN*2`) | float | 1 | 32768 B | 32768 |
| `sortIndices` | 4096 uint32(`MAX_INDEXER_KV_TILE_LEN`) | uint32 | 1 | 16384 B | 16384 |
| `defaultTopkIndices` | `topK` uint32 | uint32 | 1 | `topK*4` B(topK=2048 时 8192 B) | 8192 |
| `in` | 4096 WeightDtype(`MAX_INDEXER_KV_TILE_LEN`) | WeightDtype | 2(事件0/1) | 8192 B | 16384 |
| `lastSort` | 4096 float(`MAX_TOPK_NUM*2`,值+索引对) | float | 2(事件2/3) | 16384 B | 32768 |
| `out` | 2048 uint32(`MAX_TOPK_NUM`) | uint32 | 2(事件0/1) | 8192 B | 16384 |
| `mrgSortBuf0` | 8192 float(`MAX_INDEXER_KV_TILE_LEN*2`) | float | 1 | 32768 B | 32768 |
| **合计** | | | | | **188672** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 浮点路径占用 | 占比 | C8 路径占用 | 占比 |
|---|---|---|---|---|---|
| L1 | 512 KB | 495616 B | 94.5% | 266240 B | 50.8% |
| L0A | 64 KB | 36864 B | 56.3% | 20480 B | 31.3% |
| L0B | 64 KB | 65536 B | 100% | 49152 B | 75% |
| L0C(CO1) | 128 KB | 81920 B | 62.5% | 81920 B | 62.5% |
| UB(每 AIV subblock) | 192 KB | 188672 B | 96.0% | 188672 B | 96.0% |

要点:

- **L0B 与 L1 是上限配置的实际卡点**:浮点路径 nHeads=64、blockSize=128 时 L0B 恰好占满 64KB(100%)、L1 达 94.5%、CO1 达 62.5%——断言注释 `Consider L0C capacity` 按 CO1=128KB 成立(blockSize 翻倍到 256 时 `qkl0cBuf+l0cBuf` 共 163840 B 溢出 CO1),但最先被卡住的仍是 L0B;
- `ql1Buf`(query tile 全量 q)按 `MAX_INDEXER_Q_TILE_LEN=24` 编译期上限分配(浮点 393216 B),独占 L1 的 79.3%,是绝对大头;运行期 tile 由 host 按负载经 `qTileOverride` launch 参数传入(kernel 钳到 ≤24),L1 分配不随运行期 tile 缩小;C8 路径 q/K 为 1B,L1/L0A/L0B 占用更低,L0C 两路径相同(`MatDtype` 均 4B);
- UB 布局与 dtype 无关(`WeightDtype` 恒 2B),低地址区合计 155904 B(topK=2048),加顶端 `mrgSortBuf0` 32768 B 共 188672 B,`[off, invoff)` 间余 7936 B 空隙;`defaultTopkIndices` 是唯一随运行期 `topK` 缩放的槽位,`topK` 超过 `MAX_TOPK_NUM=2048` 低区即越过 `invoff` 触发断言——这是 host 侧 `topK <= 2048` 校验的容量来源;
- 跨核接力(`kvSplit=1`)的标量 `val`(`SetNextCore`/`WaitPrevCore`/`ResetPrevCore`)固定借用 UB 偏移 0 的 4 字节,与 `totalSort` 首元素别名(见上文 `kvSplit=1` 段),不占独立槽位。
