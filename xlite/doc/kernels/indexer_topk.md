# indexer_topk

## 功能概述

DeepSeek V3.2 DSA indexer 的核心融合算子:在单个 kernel 内完成"indexer 得分计算 + 跨核全局 topk 选取"(AIC 算分数、AIV 做 topk,通过核间同步拼接)。数学语义(见 `csrc/kernels/indexer_topk.h:63-70` 注释与测试 `tests/kernels/indexer_topk.py:110-115`):

```
scores[s,h,t]    = sum_d q[s,h,d] * kCache[t,d]         # AIC 第一级 mmad
index_score[s,t] = sum_h scores[s,h,t] * weight[s,h]     # AIC 第二级 mmad
topkIndices[s,:] = TopK(index_score[s, :cachedLen+queryLen], topK)  # AIV
```

即对每个 query token,在全部 KV 位置(已缓存 + 当前 query)中选出得分最高的 `topK` 个位置索引,作为稀疏注意力的选路结果。与 `indexer_scores` + `topk` 两算子组合相比,本算子把两步融合进同一 launch,并通过 `lastTopk` 环形接力避免中间大矩阵落盘。

## 输入输出参数

Python 侧调用(`tests/kernels/indexer_topk.py:157`):

```python
indexer_topk(rt, q, k_cache, weight, indices, topk_indices, query_start_loc,
             query_lens, cached_lens, block_tables, n_heads, head_dim,
             block_size, batch, topK)
```

host 侧 launch 见 `csrc/op.cpp:1729`(`XliteOpIndexerTopK`,要求 `topK <= MAX_TOPK_NUM=2048`)。kernel 签名(`csrc/kernels/indexer_topk.h:660`):

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
| queryLens / cachedLens | 输入 | `[batch]` | int32 | 各 batch query / 已缓存长度;总长 = 两者之和 |
| blockTables | 输入 | `[batch, maxNumBlock]` | int32 | 逻辑块 → 物理块映射 |
| scores | workspace | `[2 * block_num * XLITE_MAX_M0, MAX_INDEXER_KV_TILE_LEN]` 视作 `[block_num][2][XLITE_MAX_M0 * 4096]` | fp16 / bf16 | AIC→AIV 的得分中转缓冲,乒乓两份,每 block 一段(见下文布局) |
| lastTopk | workspace | `[total_query_len, 2*topK]`(uint32 对) | uint32 | 跨核 topk 接力缓冲:处理非首 KV tile 的核把当前全局 topK 候选(值+索引对)写给下一个核 |
| indices | 输入 | `[max_seq_len]`(= `maxNumBlock*blockSize`) | int32(uint32 语义) | `arange(max_seq_len)` 恒等索引表 |
| topkIndices | 输出 | `[total_query_len, topK]` | int32(uint32 语义) | 每 query token 的 topK 索引;当 `totalLen < topK` 时仅前 `totalLen` 列有效(测试 `tests/kernels/indexer_topk.py:169-171`) |
| sync | workspace | `[2*block_num]` | int32 | 核间环形同步 flag 数组(每核 2 个,对应 2 个 subblock) |
| nHeads / headDim / blockSize / batch / topK | — | 标量 | uint32 | 测试配置:nHeads=64, headDim=128, blockSize=128, topK ∈ {512, 2048};约束 `topK <= MAX_TOPK_NUM=2048`、`kvLen <= MAX_INDEXER_KV_TILE_LEN=4096`(`indexer_topk.h:300-301`) |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `indexer_topk_bfloat16_t` | `csrc/kernels/indexer_topk_bfloat16_t.cpp` | q/kCache/weight/scores 全 bf16 |
| `indexer_topk_float16_t` | `csrc/kernels/indexer_topk_float16_t.cpp` | 全 fp16 |

dtype 分派见 `csrc/op.cpp:1742-1751`,四张计算 tensor 必须同 dtype。

## 实现原理

本算子是混合核(`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`,`csrc/kernels/indexer_topk.h:28`):每个 block 配 1 个 AIC(Cube,算得分)+ 2 个 AIV(Vector,topk)。AIC 与 AIV 通过 `ffts_cross_core_sync` 组内同步(`mode=2`,`indexer_topk.h:510-527`):flag 0/1 由 AIC 置位表示某乒乓 scores 缓冲就绪,flag 2/3 由 AIV 置位表示缓冲已消费完毕。

### 任务划分与 scores workspace 布局

`Run`(`csrc/kernels/indexer_topk.h:504-619`)按 batch 枚举任务:

- `queryTileSize = XLITE_MAX_M0 / nHeads`(nHeads=64 时为 2;若 nHeads > 128 则退化为整块 128,`indexer_topk.h:529-532`);
- `kvNum = DIV_ROUND_UP(cachedLen + queryLen, tileSizeOfCachedKV)`,其中 `tileSizeOfCachedKV = MAX_INDEXER_KV_TILE_LEN = 4096`(`csrc/kernels/indexer_topk.h:45`,常量定义于 `csrc/kernels/kernel_param.h:41`)。即 KV 维按 4096 切大 tile,每个 AIC 任务一次算满一个 query tile 在一个 4096 KV tile 上的得分;
- 任务总数 `taskNum = queryNum * kvNum`,按 `totalIdx % block_num == block_idx` 静态分配。

`scores` workspace 布局(`indexer_topk.h:54-58`):大小为 `[block_num][2][XLITE_MAX_M0 * 4096]` 的 dtype 数组,第 `b` 个核的第 `p` 份乒乓缓冲位于偏移 `(b + p*block_num) * XLITE_MAX_M0 * 4096`。每份缓冲按 `[queryTileSize 行, 4096 列]` 行主序存放(行距 `tileSizeOfCachedKV`,`indexer_topk.h:274`),即一个 query tile × KV tile 的 index_score 矩阵。**同一 (batch, query tile) 的不同 KV tile 任务被顺序分配到不同核,由 AIV 通过 `lastTopk` 接力融合**,这是本算子区别于 `indexer_scores` 的关键。

### AIC 侧:RunAicIndexerScores

`RunAicIndexerScores`(`csrc/kernels/indexer_topk.h:160-290`)与 `indexer_scores` 的两级 mmad 相同,差异在于:

- KV 粒度是 blockSize(128)的物理 cache 块,一个 4096 tile 内按 `mLoop = DIV_ROUND_UP(kvLen, blockSize)` 循环,通过 `blockTable[mIdx + kvOffset/blockSize]` 逐块取 K(`indexer_topk.h:214-216`);
- weight 只在任务开始搬一次(`queryTaskLen, nHeads` → L1,`indexer_topk.h:185-189`),因为一个 tile 内所有 KV 块共享同一 query tile;
- 第一级 mmad 的 m 为 `blockSize`(结果暂存 L1 的 `kql1Buf`),第二级对 tile 内每个 query token 输出 `(1, mSize)` 行,写到 `scores[q * tileSizeOfCachedKV + mOffset]`(`indexer_topk.h:274`);
- L1/L0 布局与 `indexer_scores` 一致(kl1/ql1/wl1/kql1 乒乓,L0A/L0B 乒乓,L0C 单份,`indexer_topk.h:71-119`),AIC 完成后 `ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr])` 通知 AIV(`indexer_topk.h:584`)。

### AIV 侧:RunAivTopk —— 分块排序 + 归并 + 跨核接力

`RunAivTopk`(`csrc/kernels/indexer_topk.h:293-501`)对 tile 内每个 query token(`nWorkPerCore = DIV_ROUND_UP(nWork, 2)`,2 个 subblock 均分 query 行,`indexer_topk.h:586-592`):

**UB 布局**(`indexer_topk.h:304-337`,总大小受 `assert(off <= UB_SIZE)` 约束):

| 缓冲 | 大小 | 用途 |
|---|---|---|
| `totalSort` | `MAX_TOPK_NUM*4*float` | 跨核融合后的全局 topK 候选(值+索引交错对,故 4 倍) |
| `in[2]` | 各 `MAX_INDEXER_KV_TILE_LEN` 个 Dtype | 乒乓:当前 tile 得分搬入 |
| `sortIndices[2]` | 各 `MAX_INDEXER_KV_TILE_LEN` 个 u32 | 乒乓:全局索引表 `indices[kvOffset..]` 搬入 |
| `lastSort[2]` | 各 `MAX_TOPK_NUM*2*float` | 乒乓:上一核传来的 topK 候选 |
| `out[2]` | 各 `MAX_TOPK_NUM` 个 u32 | 乒乓:最终结果搬出 |
| `mrgSortBuf[2]` | 各 `MAX_INDEXER_KV_TILE_LEN*2*float` | vbitsort 归并工作区 |

`MAX_TOPK_NUM=2048` 与 `MAX_INDEXER_KV_TILE_LEN=4096`(均定义于 `csrc/kernels/kernel_param.h:41-42`)正是这套 UB 布局的容量上限来源,因此 host 侧强制 `topK <= 2048`(`csrc/op.cpp:1738-1741`)。

**topk 算法**(基于硬件排序指令,非堆、非位图):

1. **搬入 + 转 fp32**:tile 得分(`scores + idx*4096`,kvLen 个)与对应索引段搬入并 `vconv` 成 float(`indexer_topk.h:359-374`);尾部用 `SetMaskFromHighBit` + `vector_dup(FLOAT_MIN)` 补齐到 `calcPad` 对齐(`indexer_topk.h:379-385`);
2. **块内排序**:`vbitsort(mrgSortBuf1, mrgSortBuf0, sortIndices[curr], sortRepeat)` 按 `SORT_BLOCK_SIZE=32` 一组做位排序(值+原始索引成对输出,每组占 64 元素;`sortRepeat = DIV_ROUND_UP(kvLen, 32)`,`indexer_topk.h:389`);
3. **归并**:`MrgSort`(`csrc/kernels/kernel_macro.h:748-788`)用 `vmrgsort4` 四路归并循环(4→3→2 路自适应,不足 4 组时用 `FLOAT_MIN`/0 按 mask 补齐偶奇位)把所有 32 元素组归并成单个降序序列 `localSort`;
4. **与上一核候选融合**:
   - 若 `kvOffset != 0`(非首 tile):先 `WaitPrevCore()` 等前一个核把它的全局候选写入本核的 `lastTopk` 段,然后 `vmrgsort4(totalSort, {localSort, lastSort}, lens, config)` 二路归并,`lens = min(kvLen,topK) | topK<<16` 表示两个输入队列的有效长度(`indexer_topk.h:400-416`);
   - 若 `kvOffset == 0`(首 tile):直接把 `localSort` 拷入 `totalSort`,`outNum = min(kvLen, topK)`(`indexer_topk.h:417-423`);
5. **中间结果传递**:若本 tile 不是该 (batch, query) 的最后一个 tile(`kvOffset + kvLen != totalLen`),把 `totalSort` 的前 `outNum` 对(值,索引)写入 `lastTopk + idx*topK*2` 供下一核消费;最后一个 query 处理完后 `SetNextCore()` 唤醒下一核(`indexer_topk.h:477-487`);
6. **末 tile 收尾**(即全局 topk 已在 `totalSort` 中):重新搬入恒等索引表前 `outNum` 个,`vreducev2` 从 `totalSort` 抽出值载荷与索引载荷(值转 int 再转 float 以便排序携带),对 `outNum` 个候选再走一次 `vbitsort + MrgSort` **按索引升序**排列,`vreducev2` 抽出索引,`copy_ubuf_to_ubuf` 拷到 `out[curr]` 后 `CopyUbufToGmAligned` 写到 `topkIndices + idx*topK`(`indexer_topk.h:425-476`)。最终输出为**按索引升序**的 topK 位置列表(测试按集合比较,容忍近并列顺序差异,`tests/kernels/indexer_topk.py:183-189`)。

### 核间环形同步(software pipe)

KV tile 顺序分布在 `block_num` 个核上,本算子用 **GM flag 数组 + generation 计数**实现单向环形链(`indexer_topk.h:123-158`):

- `sync` 数组每核 2 个 u32(`sync[blockIdx*2 + subBlockIdx]`),核 `i` 的 `setNextSync` 即核 `(i+1)%block_num` 的 `waitPrevSync`;
- `SetNextCore()`:把 `setNextGeneration` 写入下一核的 flag(generation 每次递增,`indexer_topk.h:123-134`);
- `WaitPrevCore()`:自旋 `copy_gm_to_ubuf` 轮询上一核 flag 直到 `*val >= waitPrevGeneration`(`indexer_topk.h:136-147`);
- `ResetPrevCore()`:核处理完属于自己的所有非首 tile 后,把上一核的 flag 清零,供下一次 launch 复用(`indexer_topk.h:149-158`,在 `Run` 末尾按 `resetPrevCore` 标志执行,`indexer_topk.h:613-618`)。

由此形成软件流水:核 `i` 在做第 `k` 个 KV tile 的 topk 时,核 `i+1` 可以并行做第 `k+1` 个 tile 的得分计算(AIC)与排序,只在融合点同步。

### AIC/AIV 乒乓同步

`Run` 主循环内(`indexer_topk.h:574-605`):AIC 消费 `v2aSyncFlag[curr]`(AIV 置位的"缓冲空闲"),写完 scores 后置 `a2vSyncFlag[curr]`;AIV 反之。两份 scores 乒乓使 AIC 写第 `curr+1` 个任务与 AIV 排序第 `curr` 个任务重叠。AIV 侧事件:MTE2 搬入用 EVENT_ID0/1(in)、2/3(sortIndices)、4/5(lastSort),MTE3 搬出用 EVENT_ID0/1(out)、EVENT_ID2(lastTopk 写出),`indexer_topk.h:348-356` 预置、`indexer_topk.h:492-500` 清算。
