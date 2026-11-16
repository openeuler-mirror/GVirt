# DSA-CP 算子族(dsa_cp)

## 功能概述

DSA Context Parallelism(DSA-CP)面向 DSA 稀疏注意力的 prefill:在 TP 组内按 query token 分片,每卡计算本卡 token 的全部 heads,同时每卡保留完整 KV cache。本算子族是该流程在 kernel 层的组件,共 3 个 Python 入口 + 2 个内部 kernel:

- `dsa_cp_metadata`:把全局请求元数据投影到本卡 token shard 的局部元数据;
- `dsa_cp_write_kv_cache_by_slot`:各卡生成的新 KV 经 TP AllGather 复制后,按全局 slot 写入持久 paged cache(正/负 slot 惟一决定落点);
- `dsa_cp_gather_mla`:对本卡每个 query 收集其 Top-K KV 并运行 dense-per-query 的 MLA 注意力(host 组装 `dsa_cp_gather_metadata` + `gather_sparse_kv_cache_per_query` + `mla_v2`(dense/densePerQuery)三段流水);
- 内部 kernel `dsa_cp_reorder_wo`:由 `csrc/op.cpp` 的 `XliteOpDsaCpGatherWeight`(模型内部使用)调用,把 AllGather 得到的 NZ 权重从 `[rank, N1, localK1, 16, C0]` 重排为 matmul 需要的 `[N1, rank, localK1, 16, C0]`;另有测试绑定 `dsa_cp_reorder_wo`(`csrc/_C.cpp`)供 kernel 级单测直接调用。

模型层的整体设计(token 分片、全量 cache 复制、Top-K 组织、本卡输出投影)见 [mla_v2](mla_v2.md) 的「DSA-CP MLA」章节;路由条件为 `csrc/model.cpp` 的 `XModel::RunDsaCp`(`enableDsaCp` 且 `maxQueryLen > numSpecTokens+1` 且 `batchedTokens > XLITE_DSA_CP_TP_TOKEN_THRESHOLD=1024`,小 batch 回退常规 TP MLA)。

## 输入输出参数

三个 Python 入口(`csrc/_C.cpp` 的 `DsaCpMetadata` / `DsaCpWriteKVCacheBySlot` / `DsaCpGatherMLA`;pybind 定义关键字 `m.def("dsa_cp_metadata"` / `m.def("dsa_cp_write_kv_cache_by_slot"` / `m.def("dsa_cp_gather_mla"`)。所有入口绑定侧同步(`rt.Synchronize()`),C++ 模型层直接 launch 不同步。

### dsa_cp_metadata

`dsa_cp_metadata(rt, query_start_loc, query_lens, cached_lens, local_query_start_loc, local_query_lens, local_cached_lens, local_slot_mapping, shard_begin, shard_end)`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| query_start_loc / query_lens / cached_lens | 输入 | `[batch]` | int32 | 全局请求元数据;batch 由 `query_lens.numel` 推导 |
| local_query_start_loc / local_query_lens / local_cached_lens | 输出(原位) | `[batch]` | int32 | 本卡 shard 视角的局部元数据 |
| local_slot_mapping | 输出(原位) | `[localTokenCapacity]` | int32 | 临时本地 KV buffer 的恒等映射 `[0, localTokenCapacity)` |
| shard_begin / shard_end | 标量 | - | uint32 | 本卡 token shard 的全局区间 `[shard_begin, shard_end)` |

语义:请求 `i` 的全局 token 区间 `[startLoc[i], startLoc[i]+lens[i])` 与 shard 求交。非空交集时 `local_query_start_loc[i] = 交集起点 - shard_begin`、`local_cached_lens[i] = cached_lens[i] + 交集起点 - startLoc[i]`(本卡 query 的因果前缀含请求先验与 shard 内更早的 token);空交集时局部 lens 为 0、start_loc 为 0(保持全局请求行序,含零长度行,block table 行号不重排)。

### dsa_cp_write_kv_cache_by_slot

`dsa_cp_write_kv_cache_by_slot(rt, input, cache, slot_mapping, num_tokens, block_size)`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input | 输入 | `[>=num_tokens, width]` | 任意 | AllGather 后的 packed 新 KV 行;dim-0 可有 pitch(`input.stride[0]`) |
| cache | 输出(原位) | `[num_blocks, block_size, width]` | 同 input | 持久 paged cache;块间距取 `MAX(cache.stride[0], block_size*width)` |
| slot_mapping | 输入 | `[>=num_tokens]` | int32 | 全局 slot(= 逻辑块 × block_size + 块内偏移);负值跳过该行 |
| num_tokens / block_size | 标量 | - | uint32 | 实际 token 数 / cache 块大小 |

host 校验(`csrc/op.cpp` 的 `XliteOpDsaCpWriteKVCacheBySlot`):input 与 cache dtype 一致、`num_tokens <= input.shape[0]`、`num_tokens <= slot_mapping.numel`。dtype 无关,按元素字节宽整段拷贝(测试覆盖 BF16/FP16/INT8)。

### dsa_cp_gather_mla

`dsa_cp_gather_mla(rt, q_absorb, qr, k_cache, pe_cache, output, query_start_loc, query_lens, cached_lens, block_tables, topk_indices, scale)`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q_absorb | 输入 | `[Q, nHeads, kvLoraRank]` | bfloat16 | 已吸收 W_UK 的本卡 query(latent) |
| qr | 输入 | `[Q, nHeads, ropeHeadDim]` | bfloat16 | RoPE 之后的 query rope 分量 |
| k_cache / pe_cache | 输入 | `[numBlocks, blockSize, kvLoraRank]` / `[..., ropeHeadDim]` | bfloat16 | paged latent K / RoPE K;块可有 pitch(`stride[0]`) |
| output | 输出(原位) | `[Q, nHeads, kvLoraRank]` | bfloat16 | `softmax(QK)·k_cache` 结果(W_UV 投影在调用方完成) |
| query_start_loc / query_lens / cached_lens | 输入 | `[batch]` | int32 | **本卡局部**请求元数据(`dsa_cp_metadata` 的输出);cached_lens 已含 shard 前缀 |
| block_tables | 输入 | `[rows >= batch, maxBlocks]` | int32 | 逻辑块 → 物理块;持久化缓冲,保留原始行距(`stride[0]`),列宽由运行期收窄 |
| topk_indices | 输入 | `[Q, topK]` | int32 | 每 query 的 Top-K token 下标(indexer 产出,任意顺序;候选数 <= topK 的稠密行忽略此输入) |
| scale | 标量 | - | float | softmax 缩放因子 |

标量参数由张量 shape 推导(`csrc/_C.cpp` 的 `DsaCpGatherMLA`):`nHeads = q_absorb.shape[1]`、`ropeHeadDim = qr.shape[2]`、`kvLoraRank = q_absorb.shape[2]`、`blockSize = k_cache.shape[1]`、`batch = query_lens.numel`、`topK = topk_indices.shape[1]`。

host 约束(`csrc/op.cpp` 的 `XliteOpDsaCpGatherMLA`):全 BF16 张量、INT32 元数据;`0 < topK <= MAX_TOPK_NUM=2048` 且 **topK 为 128 的倍数**;`blockSize` 为 `GATHER_SPARSE_KV_CACHE_CNT_ONE_LOOP=32` 的倍数;`nHeads <= XLITE_MAX_M0=128`(MLA tile 须容纳一个 query token 的全部 heads);`block_tables` 至少 batch 行。

## 支持的数据类型

- `dsa_cp_metadata` / `dsa_cp_write_kv_cache_by_slot`:kernel 按字节搬运,dtype 无关;实例 `csrc/kernels/dsa_cp_metadata.cpp` / `csrc/kernels/dsa_cp_write_kv_cache_by_slot.cpp`(均单文件,`__DAV_C220_VEC__` 向量核)。
- `dsa_cp_gather_mla` 链路:BF16(`csrc/kernels/gather_sparse_kv_cache_per_query_bfloat16_t.cpp` 实例化 `csrc/kernels/gather_sparse_kv_cache_per_query.h` 的 `gather_sparse_kv_cache_per_query<bfloat16_t>`;MLA 部分复用 `csrc/kernels/mla_v2_bfloat16_t.cpp`)。
- `dsa_cp_reorder_wo`:字节搬运,dtype 无关(`csrc/kernels/dsa_cp_reorder_wo.cpp`)。

## 实现原理

### dsa_cp_metadata(`csrc/kernels/dsa_cp_metadata.cpp` 的 `dsa_cp_metadata`)

单核标量 kernel(launch grid=1)。标量 GM 访问经数据 cache,元数据可能刚被上一层的 DMA 更新,故读前与写后都对涉及 cache line 做 `DataCacheCleanAndInvalid`(`csrc/kernels/dsa_cp_metadata.cpp` 的 `dsa_cp_metadata`,关键字 `DataCacheCleanAndInvalid`)。请求按 `i = block_idx; i < batch; i += block_num` 轮转(实际 grid=1 单核串行),`local_slot_mapping` 由全部核轮转写恒等值。

### dsa_cp_write_kv_cache_by_slot(`csrc/kernels/dsa_cp_write_kv_cache_by_slot.cpp` 的 `dsa_cp_write_kv_cache_by_slot`)

AIV 多核按 token 轮转(`t = block_idx; t < num_tokens; t += block_num`),每核独立读自己的 `slot_mapping` 元素(读前 `DataCacheCleanAndInvalid`)。每行经一个 UB 中转缓冲按 `UB_SIZE` 分段 GM→UB→GM;输入地址按 `t * inputStrideBytes` 行距(支持 packed AllGather 缓冲的 pitch),输出地址按 `(slot/blockSize)*blockStrideBytes + (slot%blockSize)*tokenBytes`。MTE2/MTE3 用 `EVENT_ID0` 单事件握手:初始 `set_flag(PIPE_MTE3, PIPE_MTE2)` 置缓冲空闲,搬入前 `wait_flag(MTE3→MTE2)`、搬入后 `set_flag(MTE2→MTE3)`,写出前 `wait_flag(MTE2→MTE3)`、写出后 `set_flag(MTE3→MTE2)` 回收。负 slot 直接跳过。

### dsa_cp_gather_mla(`csrc/op.cpp` 的 `XliteOpDsaCpGatherMLA`)

host 侧按组流水,workspace 一次性按容量分配、逐组 `View` 复用:

1. **组容量** `groupCapacity = min(shardLen, 2 * aicNum)`:每 cube core 两个 query,配合 `mla_v2` dense-per-query 模式的流水(一个 query 的 softmax 与下一个 query 的 QK 重叠),同时限制展开 KV workspace 峰值(`kDense/peDense [groupCapacity, topK, dim]`)。
2. **`dsa_cp_gather_metadata`**(`csrc/kernels/dsa_cp_gather_metadata.cpp` 的 `dsa_cp_gather_metadata`,单核标量):把组内每个 query token 展开为独立"单 token 请求"——`groupQueryStartLoc[row] = row`、`groupQueryLens[row] = 1`、`groupCachedLens[row] = cachedLens[request] + query - requestBegin`(因果前缀含本 prefill chunk 内更早的 token)、`groupRequestIds[row] = request`。同样以 `DataCacheCleanAndInvalid` 维护标量一致性,末尾 `pipe_barrier(PIPE_ALL)`。
3. **`gather_sparse_kv_cache_per_query`**(`csrc/kernels/gather_sparse_kv_cache_per_query.h` 的 `gather_sparse_kv_cache_per_query`,AIV 多核):任务二维展平为 `query × topK chunk`(`CNT_ONE_LOOP=32` 个 KV 行一批,含 padding chunk),按 `task % block_num` 分核。要点:
   - `groupCachedLens` 分窗口搬入 UB(剩余 UB 全部用作 `queryWindow`),`groupRequestIds` 直接 GM 读(DCache 去重);
   - **稠密行短路**:候选数 `cachedLen + 1 <= topK` 的行不读 `topk_indices`,slot `i` 直接映射 token `offsetInTopk + i`(与 [indexer_topk](indexer_topk.md) 的稠密模板、[gather_sparse_kv_cache](gather_sparse_kv_cache.md) 的头部有效约定一致);`blockSize % 32` 保证整 chunk 落在同一物理块;
   - **稀疏行**:topk 索引按 `copy_gm_to_ubuf_align_b16` 搬入(源仅 4B 对齐),token 号经 `blockTables` 行(`blockTableStride * request`)映射为物理块 + 块内偏移;
   - **连续段合并 burst**:同一物理块内相邻 token(升序或降序游程)合并为一次 DMA,升序落位;降序段反转落位——消费侧按集合语义处理(burst 上限 `GM_UB_BURST_SIZE_TILING=8192` 字节,保持与另一乒乓缓冲的写出重叠);
   - **尾部清零**:chunk 不满 32 时仅对当前 chunk 缓冲 `vector_dup` 清零(SV 可能读对齐 padding,需有限值),不全量清整个展开缓冲;
   - 两套 chunk UB 缓冲乒乓(`GATHER_SPARSE_KV_CACHE_BUF_NUM=2`),S/MTE2/MTE3 间按 `EVENT_ID0 + curr` 握手(topk 索引缓冲的获取/释放、chunk 缓冲的获取/加载完成/写回释放),搬入下一 chunk 与写回上一 chunk 重叠;尾块清零(`vector_dup`)的 MTE2↔V 握手与查询窗口重载的 S↔MTE2 握手用独立的 `EVENT_ID4`(两处均为紧邻 set/wait 的瞬时握手,复用同一事件号);整 chunk 结果作为一次连续 burst UB→GM 写入 dense cache;
   - `kvLoraRank < ropeHeadDim` 时 host 在传参时交换两组 cache 的角色(实现要求宽通道在前)。
4. **MLA**:`XliteOpMLAV2(..., dense=true, densePerQuery=true)` 复用 [mla_v2](mla_v2.md)——每 query 一行 `[topK, kvLoraRank]` 的 dense cache,`densePerQuery` 选择 m0=nHeads 的 per-query tile 布局。

### dsa_cp_reorder_wo(`csrc/kernels/dsa_cp_reorder_wo.cpp` 的 `dsa_cp_reorder_wo`,模型内部)

`csrc/op.cpp` 的 `XliteOpDsaCpGatherWeight` 把本卡 W^O 权重分片经 `XliteOpAllGather` 汇集为 `[tp, storageLen]` 后:

- NZ 且转置(`weightNZ`)时 `dsa_cp_reorder_wo` 把 AllGather 产物 `[rank, N1, localK1, 16, C0]` 重排为 matmul 需要的 `[N1, rank, localK1, 16, C0]`:任务 = rank×columnBlocks×tilesPerBlock(`bufBytes = UB_SIZE/2` 下取整 32B 对齐后的半区大小),子块字节搬运(不转型)。双缓冲 ping-pong 协议(与 `split.cpp`/`reorder_moe.cpp` 同款):UB 对半分为两个暂存半区,任务 k 读入 `buffer[k % 2]`,上一任务的子块延迟一拍从另一半区写回,使 MTE2(GM→UB)与 MTE3(UB→GM)重叠;延迟写回跨任务边界成立(input 只读、output 只写,事件 flag 仅保护 UB 半区复用),MTE2/MTE3 按 `EVENT_ID0 + 半区号` 握手,收尾冲刷最后一个待写子块并排空两个半区;
- NZ 分片须含完整 NZ 块(`csrc/dsa_cp_weight.h` 的 `DsaCpWeightStorageShape`:FRACTAL_NZ 按 `[列块, 行块, 16, C0]` 存储,`DSA_CP_NZ_ROWS=16`、列块 `C0` 为 32 字节对应元素数);
- ND 非转置时直接 `XliteOpConcatCol` 按列拼接;非转置 NZ 与转置 ND 返回原 AllGather 结果的 View。

### 片上缓冲分配与占用汇总(AIV UB)

本算子族 5 个内部 kernel 各自独立 launch,UB 布局互不共存(launch 之间硬件自动回收),故按 kernel 分表。全部 kernel 为向量核(`KERNEL_TYPE_MIX_AIV_1_0`),无 L1/L0 分配;`dsa_cp_metadata` 与 `dsa_cp_gather_metadata` 为单核标量 kernel(纯标量 GM 读写,仅以 `DataCacheCleanAndInvalid` 维护 cache line 一致性),**无任何片上缓冲排布**;`dsa_cp_gather_mla` 链路的 MLA 段复用 `mla_v2` 算子(其 AIC L1/L0 与 AIV UB 布局见 [mla_v2](mla_v2.md),不在本文展开)。硬件容量基准(910B/dav C220,NpuArch 2201):AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`),每个 AIV subblock 各持一份独立 UB。典型配置取 `tests/kernels/mla.py` 的 `check_dsa_cp_gather_mla` 主配置:kvLoraRank=512、ropeHeadDim=64、blockSize=128、topK=2048、BF16(DeepSeek-V3.2 形状的 MLA,nHeads=64)。dtype 无关的搬运 kernel 按 dtype 字节宽整行搬运,占用与 dtype 无关。

**UB——`dsa_cp_write_kv_cache_by_slot`(GM→UB→GM 单缓冲中转)**

`tokenBytes` 为单行字节数(host launch 由 `input.shape[1] * 元素字节` 算出,`csrc/op.cpp` 的 `XliteOpDsaCpWriteKVCacheBySlot`);行宽超过 `UB_SIZE` 时按 192KB 分段循环。模型路径四个通道(latent K `kvLoraRank=512`、rope K Pe `ropeHeadDim=64`、indexer K、indexer scale)同核串行复用同一缓冲,下表代入最宽的 latent K 行。

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `buffer` | `tokenBytes`(单行字节,latent K 行 = 512×2B) | uint8 | 1(事件0 单事件握手) | `MIN(UB_SIZE, 剩余行字节)` | 1024 |
| **合计** | | | | | **1024** |

**UB——`dsa_cp_reorder_wo`(GM→UB→GM 双半乒乓中转)**

`blockBytes` 为一个 NZ 列块的行块字节(`csrc/op.cpp` 的 `XliteOpDsaCpGatherWeight`:`storageLen/columnBlocks*元素字节`,即 `rowBlocks*DSA_CP_NZ_ROWS*C0*sizeof(Dtype)`,bf16 下 = rowBlocks×512B);超过单侧暂存 `bufBytes = ROUND_DOWN(UB_SIZE/2, 32B)`(96KB)时按 96KB 分段循环。`buffer[0]`/`buffer[1]` 两半乒乓(split.cpp 协议):MTE2 装入任务 k+1 与 MTE3 写回任务 k 重叠,写回延迟跨任务边界(input 只读、output 只写、GM 不相交,事件标志 EVENT_ID0/1 独自守护半区复用,见上文数据流)。

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `buffer[2]` | `MIN(bufBytes, blockBytes)` 的分段(示例代入:W^O 每 rank K 分片 1024 行、hiddenSize=7168 → rowBlocks=64 → blockBytes=32768,单段即足) | uint8 | 2(事件0/1) | `MIN(bufBytes, 剩余段字节)` | 65536(两半交替各 32768) |
| **合计** | | | | | **65536** |

**UB——`gather_sparse_kv_cache_per_query`(双份 chunk 乒乓 + 查询窗口)**

`off` 从 0 起按 chunk 乒乓缓冲排布(每槽 `ROUND_UP` 到 `BLOCK_SIZE=32B`),剩余 UB 全部用作 `groupCachedLens` 的查询窗口(`queryWindowCapacity = (UB_SIZE-off)/sizeof(int32_t)`);窗口非空的下限由 kernel 断言保证(见要点)。

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `kBufs` | `(CNT_ONE_LOOP=32, kvLoraRank=512)` | bf16 | 2(事件0/1) | `ROUND_UP(kvLoraRank*sizeof(Dtype)*CNT_ONE_LOOP, BLOCK_SIZE)` | 65536 |
| `peBufs` | `(32, ropeHeadDim=64)` | bf16 | 2(事件0/1) | `ROUND_UP(ropeHeadDim*sizeof(Dtype)*CNT_ONE_LOOP, BLOCK_SIZE)` | 8192 |
| `topkIndicesUb` | `CNT_ONE_LOOP=32` 个 int32 | int32 | 2(事件0/1) | `ROUND_UP(CNT_ONE_LOOP*sizeof(uint32_t), BLOCK_SIZE)` | 256 |
| `groupCachedLensUb` | `(UB_SIZE-off)/4 = 30656` 个 int32(查询窗口) | int32 | 1(事件4) | `(UB_SIZE-off)/sizeof(int32_t)*sizeof(int32_t)` | 122624 |
| **合计** | | | | | **196608** |

**占用汇总**(各表占用 vs 硬件容量;三个 kernel 独立 launch、UB 互不共存,不跨 kernel 叠加;字节数):

| kernel | 硬件容量 | 典型占用 | 占比 |
|---|---|---|---|
| `gather_sparse_kv_cache_per_query` UB | 192 KB | 196608 B(查询窗口吃满剩余 UB) | 100% |
| `dsa_cp_write_kv_cache_by_slot` UB | 192 KB | 1024 B(latent K 行) | 0.5% |
| `dsa_cp_reorder_wo` UB | 192 KB | MIN(196608, 2*blockBytes) B | ≤100% |
| `dsa_cp_metadata` / `dsa_cp_gather_metadata` | — | 0(纯标量,无片上缓冲) | — |

要点:

- **查询窗口设计把 UB 恰好吃满**:`gather_sparse_kv_cache_per_query` 把双份 chunk 缓冲之外的剩余 UB 全部用作 `groupCachedLens` 查询窗口(典型 30656 个 int32),窗口越大、跨查询重载越少;窗口非空的下限由 kernel 断言 `off + BLOCK_SIZE <= UB_SIZE` 保证(至少 8 个 int32),host 侧还按任务数收缩 launch grid(`PickMinBlockNum`,任务 = groupLen×topkChunks);
- **host 校验与 kernel 断言的容量来源**:host(`csrc/op.cpp` 的 `XliteOpDsaCpGatherMLA`)校验 `0 < topK <= MAX_TOPK_NUM=2048` 且 128 倍数、`blockSize` 为 `GATHER_SPARSE_KV_CACHE_CNT_ONE_LOOP=32` 的倍数、`nHeads <= XLITE_MAX_M0=128`(MLA tile 须容纳单 token 全部头);kernel(`csrc/kernels/gather_sparse_kv_cache_per_query.h`)断言 `kvLoraRank >= ropeHeadDim` 且 `perKBytes/perPeBytes % BLOCK_SIZE == 0`——保证 chunk 缓冲的 `ROUND_UP` 垫片恰为零、无对齐损耗;
- **两个 metadata kernel 无片上缓冲的原因**:元数据只做标量 GM 读写且可能刚被上一层 DMA 更新,读前写后均以 `DataCacheCleanAndInvalid` 维护标量一致性,launch grid=1 单核串行,不排布任何 UB/L1/L0;
- 两个纯搬运 kernel 中,`dsa_cp_write_kv_cache_by_slot` 仍为 UB 单缓冲:按 `UB_SIZE` 分段,行字节超过 192KB 的通道自动多段循环,同核多通道串行复用同一缓冲,MTE2 搬入与 MTE3 写出以单 `EVENT_ID0` 轮转握手;`dsa_cp_reorder_wo` 则如上表所述按 96KB 半区乒乓,双事件握手、写回跨任务边界延迟。

## 边界处理

- `dsa_cp_metadata`:空交集行写零但不改变行序;`local_token_capacity` 可大于实际 shard 长度(AllGather padding 区),slot mapping 覆盖整个容量;
- `dsa_cp_write_kv_cache_by_slot`:负 slot 与 gap(物理块间距)不被触碰;`num_tokens=0` 时 host 直接跳过 launch;
- `dsa_cp_gather_mla`:shard 为空(`shardLen=0`)直接返回;稠密行忽略 indices、稀疏行要求 topK 个互异有效因果下标(indexer 保证);`topK` 非满 chunk 的 padding 槽清零。

## 验证

- [tests/kernels/dsa_cp.py](../../../tests/kernels/dsa_cp.py):metadata 逐 rank 重构全局序列、cache 写入 gap/pitch/负 slot 精确比对、`dsa_cp_reorder_wo` 块置换(INT8 字节精确比对,覆盖单/多子块、尾子块与毒化输出缓冲)、局部全头 attention 对比 FP32 参考实现;
- [tests/kernels/dsa_cp_comm.py](../../../tests/kernels/dsa_cp_comm.py):torchrun 双卡验证集体通信边界(Norm/AllGather 分片恒等);
- [tests/kernels/mla.py](../../../tests/kernels/mla.py) 的 `check_dsa_cp_gather_mla`(以 `--dsa-cp` 运行):含短/长历史、多请求、稠密行、乱序/降序索引、pitched block tables、多组溢出等 case,与 FP32 参考实现对比。
