# indexer_topk

## 功能概述

DeepSeek V3.2 DSA indexer 的核心融合算子:在单个 kernel 内完成"indexer 得分计算 + 跨核全局 topk 选取"(AIC 算分数、AIV 做 topk,通过核间同步拼接)。数学语义(见 `csrc/kernels/indexer_topk.h:63-70` 注释与测试 `tests/kernels/indexer_topk.py:110-115`):

```
scores[s,h,t]    = ReLU(sum_d q[s,h,d] * kCache[t,d])   # AIC 第一级 mmad + fixpipe ReLU
index_score[s,t] = sum_h scores[s,h,t] * weight[s,h]     # AIC 第二级 mmad
topkIndices[s,:] = TopK(index_score[s, :p0+1], topK)       # AIV,因果限定候选 [0, p0]
```

其中 `p0` 为该 query token 的绝对序列位置。两个关键语义:

- **因果掩码**:每个 query token 只在**先验及自身**位置 `[0, p0]` 中选 topK(含自身,不含未来),通过按位置截断 `validKvLen` + 尾部 `FLOAT_MIN` 补齐实现(见下文);
- **稀疏/稠密分流**:`p0 < topK` 的 token 走稠密注意力(全部先验位置都参加注意力,无需选路),topk 只对 `p0 >= topK` 的稀疏行计算——整块稠密的 query chunk 直接跳过,其 `topkIndices` 行是 dummy 数据,下游不得消费(`gather_sparse_kv_cache` 会跳过这些行)。

与 `indexer_scores` + `topk` 两算子组合相比,本算子把两步融合进同一 launch,并通过 `lastTopk` 环形接力避免中间大矩阵落盘。

## 输入输出参数

Python 侧调用(`tests/kernels/indexer_topk.py:193`):

```python
indexer_topk(rt, q, k_cache, weight, indices, topk_indices, query_start_loc,
             query_lens, cached_lens, block_tables, n_heads, head_dim,
             block_size, batch, topK)
```

host 侧 launch 见 `csrc/op.cpp:1755`(`XliteOpIndexerTopK`,要求 `topK <= MAX_TOPK_NUM=2048`)。kernel 签名(`csrc/kernels/indexer_topk.h:666`):

```cpp
indexer_topk_<dtype>(GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc,
                     GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,
                     GM_ADDR scores, GM_ADDR lastTopk, GM_ADDR indices, GM_ADDR topkIndices,
                     GM_ADDR sync, uint32_t nHeads, uint32_t headDim, uint32_t blockSize,
                     uint32_t batch, uint32_t maxNumBlock, uint32_t topK)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q | 输入 | `[total_query_len, nHeads*headDim]` | fp16 / bf16 | 拼接的 flatten query |
| kCache | 输入 | `[kvcache_block_num, blockSize, headDim]` | fp16 / bf16 | paged indexer K cache |
| weight | 输入 | `[total_query_len, headDim + nHeads]` | fp16 / bf16 | 头融合权重,位于每行第 headDim 列起 |
| queryStartLoc | 输入 | `[batch]` | int32 | 各 batch query 起始偏移 |
| queryLens / cachedLens | 输入 | `[batch]` | int32 | 各 batch query / 已缓存长度;token 绝对位置 `p0 = cachedLens[i] + 段内偏移` |
| blockTables | 输入 | `[batch, maxNumBlock]` | int32 | 逻辑块 → 物理块映射 |
| scores | workspace | `[2 * block_num * XLITE_MAX_M0, MAX_INDEXER_KV_TILE_LEN]` 视作 `[block_num][2][XLITE_MAX_M0 * 4096]` | fp16 / bf16 | AIC→AIV 的得分中转缓冲,乒乓两份,每 block 一段(见下文布局) |
| lastTopk | workspace | `[total_query_len, 2*topK]`(uint32 对) | uint32 | 跨核 topk 接力缓冲:处理非首 KV tile 的核把当前全局 topK 候选(值+索引对)写给下一个核 |
| indices | 输入 | `[max_seq_len]`(= `maxNumBlock*blockSize`) | int32(uint32 语义) | `arange(max_seq_len)` 恒等索引表;AIV 侧只整体搬入一次,之后按 `kvOffset` 增量推进(见下文) |
| topkIndices | 输出 | `[total_query_len, topK]` | int32(uint32 语义) | 每 query token 的 topK 索引,**按索引降序**(最大 id 在前)排列。仅 `p0 >= topK` 的稀疏行保证有效(候选满 `topK` 个,全部落在 `[0, p0]`);`p0 < topK` 的稠密行内容是 dummy,不得消费(测试只比较稀疏行,`tests/kernels/indexer_topk.py:226-228`;`gather_sparse_kv_cache` 依赖该降序约定做 burst 拷入) |
| sync | workspace | `[2*block_num]` | int32 | 核间环形同步 flag 数组(每核 2 个,对应 2 个 subblock) |
| nHeads / headDim / blockSize / batch / topK | — | 标量 | uint32 | 测试配置:nHeads=64, headDim=128, blockSize=128, topK ∈ {512, 2048};约束 `topK <= MAX_TOPK_NUM=2048`、`kvLen <= MAX_INDEXER_KV_TILE_LEN=4096`、`topK <= queryPosBase + queryLen`(`indexer_topk.h:337`) |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `indexer_topk_bfloat16_t` | `csrc/kernels/indexer_topk_bfloat16_t.cpp` | q/kCache/weight/scores 全 bf16 |
| `indexer_topk_float16_t` | `csrc/kernels/indexer_topk_float16_t.cpp` | 全 fp16 |

dtype 分派见 `csrc/op.cpp:1769-1777`,四张计算 tensor 必须同 dtype。

## 实现原理

本算子是混合核(`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`,`csrc/kernels/indexer_topk.h:32`):每个 block 配 1 个 AIC(Cube,算得分)+ 2 个 AIV(Vector,topk)。AIC 与 AIV 通过 `ffts_cross_core_sync` 组内同步(`mode=2`,`indexer_topk.h:484-503`):flag 0/1 由 AIC 置位表示某乒乓 scores 缓冲就绪,flag 2/3 由 AIV 置位表示缓冲已消费完毕。

### 任务划分、稠密/未来 chunk 跳过与 scores workspace 布局

`Run`(`csrc/kernels/indexer_topk.h:482-613`)按 batch 枚举任务:

- `queryTileSize = XLITE_MAX_M0 / nHeads`(nHeads=64 时为 2;若 nHeads > XLITE_MAX_M0 则退化为整块,`indexer_topk.h:517-520`);
- `kvNumMax = DIV_ROUND_UP(cachedLen + queryLen, tileSizeOfCachedKV)`,其中 `tileSizeOfCachedKV = ROUND_UP(INDEXER_KV_TILE_LEN=4096, SORT_BLOCK_SIZE=32)`(`csrc/kernels/indexer_topk.h:11,49`,带编译期检查 `INDEXER_KV_TILE_LEN ≤ MAX_INDEXER_KV_TILE_LEN` 与运行期 `tileSizeOfCachedKV >= topK` 断言,`indexer_topk.h:13-14,65`)。即 KV 维按 4096 切大 tile;
- 任务总数上界 `taskNum = queryNum * kvNumMax`,但**跳过的 chunk 不消耗全局任务号**:`totalIdx` 只在任务有效时递增,再按 `totalIdx % block_num == block_idx` 静态分配(`indexer_topk.h:565-568`),稠密/未来 chunk 不会在核间留下"空任务"空洞。

每个 (query chunk, KV chunk) 任务的**位置感知跳过**(`indexer_topk.h:548-563`),`queryPosBase = cachedLen + queryOffset` 为该 chunk 首 token 的绝对位置,`queryPosEnd`(含)为末 token 位置:

1. **稠密 chunk 跳过**:`queryPosEnd < topK` —— 该 chunk 的所有 token 都走稠密注意力,无需 topk,直接 `idx = NEXT_MULTIPLE(idx, kvNumMax)` 跳到下一个 query chunk(`indexer_topk.h:551-553`);
2. **全未来 KV chunk 跳过**:`queryPosEnd < kvOffset` —— 该 KV chunk 的所有位置都大于 chunk 内任何 token 的位置,因果意义上完全无效,同样整段跳过(`indexer_topk.h:558-561`);
3. 否则 `kvLen = MIN(tileSizeOfCachedKV, queryPosEnd - kvOffset)`:KV chunk 只需覆盖到 chunk 末 token 的位置为止(最后一个 KV tile 通常不满)。

`scores` workspace 布局(`indexer_topk.h:54-65`):大小为 `[block_num][2][XLITE_MAX_M0 * 4096]` 的 dtype 数组,第 `b` 个核的第 `p` 份乒乓缓冲位于偏移 `(b + p*block_num) * XLITE_MAX_M0 * 4096`。每份缓冲按 `[queryTileSize 行, 4096 列]` 行主序存放(行距 `tileSizeOfCachedKV`),即一个 query tile × KV tile 的 index_score 矩阵。**同一 (batch, query tile) 的不同 KV tile 任务被顺序分配到不同核,由 AIV 通过 `lastTopk` 接力融合**,这是本算子区别于 `indexer_scores` 的关键。

### AIC 侧:RunAicIndexerScores(两级 mmad + fixpipe ReLU)

`RunAicIndexerScores`(`csrc/kernels/indexer_topk.h:198-291`)与 `indexer_scores` 的两级 mmad 相同,差异在于:

- KV 粒度是 blockSize(128)的物理 cache 块,一个 tile 内按 `mLoop = DIV_ROUND_UP(kvLen, blockSize)` 循环,通过 `blockTable[mIdx + kvOffset/blockSize]` 逐块取 K(`indexer_topk.h:252-256`);
- weight 只在任务开始搬一次(`queryTaskLen, nHeads` → L1),因为一个 tile 内所有 KV 块共享同一 query tile;
- 第一级 mmad 的 m 为 `blockSize`(结果暂存 L1 的 `kql1Buf`),第二级对 tile 内每个 query token 输出 `(1, mSize)` 行,写到 `scores[q * tileSizeOfCachedKV + mOffset]`;
- **ReLU 融合进 L0C→L1 搬运**:第一级 per-head QK 得分在 `CopyL0CToL1` 下 L0C 时以 `reluEn=1` 过滤(`indexer_topk.h:283-285`),即每头得分先过 `max(x, 0)` 再进第二级加权求和——DSA indexer 定义中的负得分截断,不占额外向量指令;
- L1/L0 布局与 `indexer_scores` 一致(kl1/ql1/wl1/kql1 乒乓,L0A/L0B 乒乓,L0C 单份),AIC 完成后 `ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr])` 通知 AIV(`indexer_topk.h:579`)。

### AIV 侧:RunAivTopk —— 逐位置因果 topk + 分块排序 + 归并 + 跨核接力

`RunAivTopk`(`csrc/kernels/indexer_topk.h:332-479`)新增 `queryPosBase` 参数(该 subblock 分到的首 query token 的绝对位置)。对 tile 内每个 query token(`nWorkPerCore = DIV_ROUND_UP(nWork, 2)`,2 个 subblock 均分 query 行):

**逐位置因果截断**(`indexer_topk.h:353-360`):

- `p0 = queryPosBase + idx`,候选只含 `[kvOffset, kvOffset + validKvLen)`,其中 `validKvLen = MIN(p0 - kvOffset + 1, kvLen)`——同一 KV tile 内不同 query token 的有效长度随位置递增,"未来"位置(`> p0`)不进入排序,等价于因果掩码;
- `validKvLen <= 0`(该 token 的全部先验位置已在更早的 KV chunk 覆盖)直接 `continue`;
- `isFirst = kvOffset <= 0`(首 KV tile,无需融合上一核结果);`isFinal = kvOffset + validKvLen > p0`(该 tile 覆盖到 token 自身位置,是末 tile,负责写出最终 topk)。

**UB 布局**(Init 中一次性排布为成员缓冲,`indexer_topk.h:127-155`,总大小受 `assert(off <= invoff)` 约束):

| 缓冲 | 大小 | 用途 |
|---|---|---|
| `totalSort` | `MAX_TOPK_NUM*4*float` | 跨核融合后的全局 topK 候选(值+索引交错对,故 4 倍) |
| `mrgSortBuf1` | `MAX_INDEXER_KV_TILE_LEN*2*float` | vbitsort 归并工作区(与 `mrgSortBuf0` 分置两端避免 bank conflict) |
| `sortIndices` | `MAX_INDEXER_KV_TILE_LEN` 个 u32 | 全局索引表(单份,非乒乓;见下) |
| `in[2]` | 各 `MAX_INDEXER_KV_TILE_LEN` 个 Dtype | 乒乓:当前 tile 得分搬入 |
| `lastSort[2]` | 各 `MAX_TOPK_NUM*2*float` | 乒乓:上一核传来的 topK 候选 |
| `out[2]` | 各 `MAX_TOPK_NUM` 个 u32 | 乒乓:最终结果搬出 |
| `mrgSortBuf0` | `MAX_INDEXER_KV_TILE_LEN*2*float` | 计算区,置于 `UB_SIZE` 向下(高地址) |

`MAX_TOPK_NUM=2048` 与 `MAX_INDEXER_KV_TILE_LEN=4096`(均定义于 `csrc/kernels/kernel_param.h:41-42`)正是这套 UB 布局的容量上限来源,因此 host 侧强制 `topK <= 2048`(`csrc/op.cpp:1764-1767`)。

**索引表一次性搬入 + 增量推进**:`Run` 启动时把 `indices` 前 4096 项整体搬入 `sortIndices` 一次(`indexer_topk.h:506-510`);之后每个任务若 `kvOffset` 相对上次推进了 `idxDiff`,用一条 `vadds` 给整个表加上 `idxDiff`(`indexer_topk.h:367-371`),不再逐任务从 GM 重复拷贝索引表。

**topk 算法**(基于硬件排序指令,非堆、非位图):

1. **搬入 + 转 fp32**:tile 得分(`scores + idx*4096`,前 `validKvLen` 个)搬入并 `convert_input`(`csrc/kernels/kernel_macro.h:899`)成 float;尾部 `[validKvLen, sortRepeat*SORT_BLOCK_SIZE)` 用带 mask 的 `vector_dup(FLOAT_MIN)` 补齐(`indexer_topk.h:373-378`)——无效位置以 `FLOAT_MIN` 参与排序,天然不会胜出;
2. **块内排序**:`vbitsort(mrgSortBuf1, mrgSortBuf0, sortIndices, sortRepeat)` 按 `SORT_BLOCK_SIZE=32` 一组做位排序(值+原始索引成对输出,`sortRepeat = DIV_ROUND_UP(validKvLen, 32)`,`indexer_topk.h:380-382`);
3. **归并截断**:`MrgSort`(`csrc/kernels/kernel_macro.h:772`)用 `vmrgsort4` 四路归并循环把所有 32 元素组归并成单个降序序列,`topK` 参数让归并尽早截断到 topK 宽度;首 tile 时以 `preferredDst = totalSort` 直接把结果写进 `totalSort`(`indexer_topk.h:396-405`),省去一次中间缓冲拷贝;
4. **与上一核候选融合**:
   - 非 `isFirst`:先 `WaitPrevCore()` 等前一个核把它的全局候选写入本核的 `lastTopk` 段,搬入 `lastSort[curr]`(`lastSortLen = MIN(topK, kvOffset)` 对),`vmrgsort4(totalSort, {localSort, lastSort}, ...)` 二路归并(`indexer_topk.h:407-429`);
   - `isFirst`:`totalSort` 已由步骤 3 直接得到;
5. **中间结果传递**(`!isFinal`):把 `totalSort` 的前 `MIN(topK, totalSortLen)` 对(值,索引)写入 `lastTopk + idx*topK*2` 供下一核消费;最后一个 query 处理完后 `SetNextCore()` 唤醒下一核(`indexer_topk.h:433-444`)。MTE3 异步读 `totalSort` 的释放等待用 `totalSortOnHold` 标记**推迟到下次真正要写它之前**(`indexer_topk.h:345,400-424,470-472`),避免空等;
6. **末 tile 收尾**(`isFinal`,`indexer_topk.h:445-467`):`vreducev2` 从 `totalSort` 抽出索引载荷(id 最高位为 0,直接按位 cast 成 float 即保序),对 topK 个候选走 `vbitsort + MrgSort` **按索引从大到小**排列,再 `vreducev2` 抽出索引直接进 `out[curr]` 后 `CopyUbufToGmAligned` 写到 `topkIndices + idx*topK`。最终输出为**按索引降序**的 topK 位置列表(测试按集合比较,容忍近并列顺序差异,`tests/kernels/indexer_topk.py:241-246`;`gather_sparse_kv_cache` 的 burst 拷入窗口依赖该降序约定提升跨核 L2 命中率)。

### 核间环形同步(software pipe)

KV tile 顺序分布在 `block_num` 个核上,本算子用 **GM flag 数组 + generation 计数**实现单向环形链(`indexer_topk.h:160-196`):

- `sync` 数组每核 2 个 u32(`sync[blockIdx*2 + subBlockIdx]`),核 `i` 的 `setNextSync` 即核 `(i+1)%block_num` 的 `waitPrevSync`;
- `SetNextCore()`:把 `setNextGeneration` 写入下一核的 flag(generation 每次递增,`indexer_topk.h:160-171`);
- `WaitPrevCore()`:自旋 `copy_gm_to_ubuf` 轮询上一核 flag 直到 `*val >= waitPrevGeneration`(`indexer_topk.h:173-184`);
- `ResetPrevCore()`:核处理完属于自己的所有非首 tile 后,把上一核的 flag 清零,供下一次 launch 复用(`indexer_topk.h:186-196`,在 `Run` 末尾按 `resetPrevCore` 标志执行,`indexer_topk.h:609-611`)。

由此形成软件流水:核 `i` 在做第 `k` 个 KV tile 的 topk 时,核 `i+1` 可以并行做第 `k+1` 个 tile 的得分计算(AIC)与排序,只在融合点同步。

### AIC/AIV 乒乓同步

`Run` 主循环内(`indexer_topk.h:567-605`):AIC 消费 `v2aSyncFlag[curr]`(AIV 置位的"缓冲空闲"),写完 scores 后置 `a2vSyncFlag[curr]`;AIV 反之。两份 scores 乒乓使 AIC 写第 `curr+1` 个任务与 AIV 排序第 `curr` 个任务重叠。AIV 侧事件:MTE2 搬入用 EVENT_ID0/1(in)、2/3(lastSort),MTE3 搬出用 EVENT_ID0/1(out)、EVENT_ID2(lastTopk 写出/`totalSort` 释放);`sortIndices` 一次性搬入用 EVENT_ID4(`indexer_topk.h:506-510`)。循环末尾清空所有挂起 flag(`indexer_topk.h:470-479`)并 `PipeBarrier<PIPE_ALL>()`(`indexer_topk.h:608,613`)。
