# gather_sparse_kv_cache

## 功能概述

DSA(Dual Sparse Attention)decode 长序列路径的 KV 收集算子,**仅限 decode 批次**(每 batch query 长度恒为 1,不做 spec decode):按 indexer 给出的每 batch top-k token 下标(`topkIndices`),把 paged(分页)cache 中的 `kCache`(kvLoraRank)与 `peCache`(ropeHeadDim)收集到每 batch 连续的 dense cache 中,供 mla_v2 / cxa 的 dense 模式连续读取。本质上是一次 `dense[b, i] = paged[blockTable[b][tok/bs]*bs + tok%bs]` 的双通道 gather,host 侧要求 `kvHeads == 1`(`XliteOpGatherSparseKVCache`,`csrc/op.cpp`)。

每 batch 行按候选 token 数 `numCandidates = (cachedLens[b] + 1) / compressRatio` 分两类(+1 即 decode 自身 token):

- **dense 行**(`numCandidates <= topK`):top-k 结果必然是 `[0, pos]` 全前缀接无效槽位,因此 slot `i` 直接映射 token `i`(恒等模板),**不读 `topkIndices`**(该行的 topk_indices 为 dummy 值);有效前缀之外的输出槽位不写。
- **sparse 行**(`numCandidates > topK`):读 `topkIndices`,**下标顺序任意**(升序 = `indexer_topk` 契约、降序 = 旧契约、乱序均可),kernel 运行时检测连续段。

`compressRatio` 控制候选 token 的统计粒度(`>= 1`,kernel 前置条件,见下):`numCandidates = (cachedLens[b] + 1) / compressRatio`。默认 1,即按原始 token 收集(MLA 场景);cxa 的 dense 模式以实际压缩比调用本算子收集压缩 KV(`CXA`,`csrc/_C.cpp`)。旧实现中 `compressRatio == 0` 表示按原始 token 统计的特殊取值已移除(cxa 传 0 表示不压缩,走 sparse 路径,此时不会调用本算子;若强行传 0 到本算子,设备侧除零,release 构建的断言被 `-DNDEBUG` 编译掉、无保护)。

## 输入输出参数

Python 侧调用(`tests/kernels/gather_sparse_kv_cache.py`):

```python
gather_sparse_kv_cache(rt, k_cache, pe_cache, block_tables, topk_indices,
                       cached_lens, k_dense, pe_dense, batch, topK,
                       BLOCK_SIZE, kv_lora_rank, rope_head_dim,
                       kv_heads=1, compress_ratio=1, burst_copy_size=16*1024)
```

Python 绑定 `gather_sparse_kv_cache`(`csrc/_C.cpp`)→ `GatherSparseKVCache`(`csrc/_C.cpp`)→ `XliteOpGatherSparseKVCache`(`csrc/op.cpp`)。kernel 签名(`csrc/kernels/gather_sparse_kv_cache.h`):

```cpp
gather_sparse_kv_cache_<dtype>(kCache, peCache, blockTables, topkIndices, cachedLens,
                               kDenseCache, peDenseCache, seqNum, indexTopK, blockSize,
                               maxNumBlocks, kvLoraRank, ropeHeadDim,
                               compressRatio = 1, burstCopySize = GM_UB_BURST_SIZE)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| kCache | 输入 | `[numBlocks, blockSize, 1, kvLoraRank]` | BF16 | paged K cache(MLA 为 latent K,压缩 KV 为压缩 K);单 PE 通道时可传空 |
| peCache | 输入 | `[numBlocks, blockSize, 1, ropeHeadDim]` | BF16 | paged PE / RoPE K cache;cxa dense 模式内部调用时传空 |
| blockTables | 输入 | `[batch, maxNumBlocks]` | INT32 | 逻辑块 → 物理块映射 |
| topkIndices | 输入 | `[batch, indexTopK]` | INT32 | 每 batch 的 top-k token 下标(token 级);**顺序任意**(升序/降序/乱序均可),dense 行不被读取 |
| cachedLens | 输入 | `[batch]` | INT32 | 每 batch 已缓存长度;decode-only,query 长度恒 1,`numCandidates = (cachedLens[b]+1)/compressRatio` 决定有效槽位数 |
| kDenseCache | 输出 | `[batch, indexTopK, 1, kvLoraRank]` | BF16 | 收集后的连续 K(零初始化,槽位 ≥ min(numCandidates, topK) 不写) |
| peDenseCache | 输出 | `[batch, indexTopK, 1, ropeHeadDim]` | BF16 | 收集后的连续 PE / RoPE K;cxa dense 模式内部调用时传空 |
| seqNum | 标量 | - | uint32 | batch 数 |
| indexTopK | 标量 | - | uint32 | 每 batch 收集的 token 数上限,`> 0`(kernel 前置条件);kernel 按 32 槽位分 chunk,任意正值均可。`2048`(`MAX_TOPK_NUM`,`csrc/kernels/kernel_param.h`)是上游 `indexer_topk` 的约定上限,本算子不作限制;测试覆盖 1/16/100/512/2048 |
| blockSize | 标量 | - | uint32 | KV cache 块大小(128);须为 `CNT_ONE_LOOP=32` 的倍数(kernel 前置条件) |
| maxNumBlocks | 标量 | - | uint32 | host 由 blockTables 推得(`DeriveMaxNumBlocks`,`csrc/op.cpp`) |
| kvLoraRank / ropeHeadDim | 标量 | - | uint32 | 两通道向量宽,均须为 16 的倍数(每 token 字节 32B 对齐)且不超出 kernel 的 UB 容量(kernel 前置条件);kernel 要求第一通道不小于第二,host 保证:两通道都在且 `kvLoraRank < ropeHeadDim` 时换位传入,单通道时活动通道作第一通道 |
| compressRatio | 标量 | - | uint32 | 压缩比,`>= 1`(kernel 前置条件);默认 1。cxa dense 模式内部传入实际压缩比 |
| burstCopySize | 标量 | - | uint32 | burst 拷入大小上限(字节),`GM_UB_BURST_SIZE` 的运行时替代,只能收紧、不能放大;详见下文 |

任务粒度:`totalTasks = seqNum * ceil(indexTopK / CNT_ONE_LOOP=32)`,每任务处理一个 batch 的 32 个连续 top-k 槽位;launch grid 为 `min(aivNum, seqNum * ceil(indexTopK/16))`,kernel 内 grid-stride 分配。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| bfloat16_t | `csrc/kernels/gather_sparse_kv_cache_bfloat16_t.cpp` | `gather_sparse_kv_cache_bfloat16_t`(双通道) |
| bfloat16_t | `csrc/kernels/gather_sparse_kv_cache_nope_bfloat16_t.cpp` | `gather_sparse_kv_cache_nope_bfloat16_t`(单通道,nope 变体) |

host 仅 BF16;kernel 在 `#ifdef __DAV_C220_VEC__` 下,非向量核平台为空实现。仅在向量核上 launch。通道选择(host,`XliteOpGatherSparseKVCache`):两通道指针均为空时直接返回;任一通道为空时 launch nope 变体(`XLITE_GATHER_SPARSE_KV_CACHE_NOPE` 编译期裁掉 PE 通道,活动通道作第一通道);两通道都在时若 `kvLoraRank < ropeHeadDim` 则换位调用双通道 kernel。

**前置条件**:kernel 的设备端断言在 debug 构建中覆盖 `compressRatio >= 1`、`indexTopK > 0`、`blockSize % 32 == 0`、通道宽为 16 的倍数、4 份轮转缓冲加 cachedLens 窗口不超 192KB UB(C220;3510 的 UB 更宽但 kernel 为空实现,该界保守);release 构建中断言被 `-DNDEBUG` 编译掉,前置条件由调用方保证。host 侧仅检查 `kvHeads == 1`、BF16,以及通道输入 cache 与输出 dense cache 的空置状态一致(否则会在设备上写空指针或留下从未写入却被消费的输出)。`CNT_ONE_LOOP` 与 `GATHER_SPARSE_KV_CACHE_BUF_NUM` 定义于 `csrc/kernels/kernel_param.h`。

## 实现原理

纯 AIV kernel,grid-stride 分配。每任务(32 个槽位)流水四步,`GATHER_SPARSE_KV_CACHE_BUF_NUM=4` 份轮转缓冲(`kBufs/peBufs/topks` 各 4 份)重叠相邻 chunk 之间的 GM 访问:

1. **载入下标**:sparse 行把 32 个 `topkIndices` 拷入 UB 当前轮转槽,MTE2 完成后通知标量核(EVENT_ID0..3 按缓冲编号);dense 行跳过(恒等模板无需下标)。
2. **地址计算(标量核)**:对每个 token 求 `physBlock[i] = blockTable[tok/blockSize]`、`physInBlockOffset[i] = tok % blockSize`(blockTable 为 GM 指针,逐元素标量读,DCache 摊薄重复块);dense 行直接 `blk = blockTable[offsetInBatch/blockSize]`、`base = offsetInBatch % blockSize`、`physInBlockOffset[i] = base + i`(`blockSize % CNT_ONE_LOOP == 0` 保证整个 chunk 落在同一物理块)。
3. **burst 拷入(GM→UB,组段)**:在 slot `i` 处比较 slot `i+1` 判定方向(`+1` 升序段、`-1` 降序段、否则单 token),向后延伸:同物理块、同方向等差、且段长 < `burstCopyCnt`。一次 burst 只取该段的 token 区间(升序内存序),降序段取 `[blkOffset+1-iBurst, blkOffset]`,落位后自然为升序——**消费方(mla_v2/cxa dense)对收集结果按集合语义做注意力,槽位成员正确即可,次序无关**。段内约束:不跨物理块(构造保证)、不跨 chunk(`i+iBurst < cnt`)、不超 `burstCopyCnt`(超长逻辑段拆成多次 burst);无 UB→UB 搬运,未选中的 cache 不读 GM。burst 计数按第一通道(K,per-token 更大)推导,K 与 PE 的 burst 覆盖同一 token 段。
4. **整段拷出(UB→GM)**:整个 chunk 一次 `CopyUbufToGmAligned` 连续写出 `cnt * perKBytes`(PE 同理)——随机访存收敛为"读侧多段 burst、写侧整段连续"。

**burst 计数推导**:`burstSize = MIN(MIN(burstCopySize, GM_UB_BURST_SIZE=16KB), GM_UB_BURST_SIZE_TILING=8KB)`,`burstCopyCnt = CLAMP(burstSize / perKBytes, 1, MIN(CNT_ONE_LOOP, blockSize))`。`GM_UB_BURST_SIZE_TILING`(定义于 `csrc/kernels/gather_sparse_kv_cache.h`)是拷入的编译期最优值:每次 burst 只取被选中的 token,burst 也要足够小,以便与其它轮转缓冲上正在进行的拷出交错(单次过大拷入会独占 GM 带宽、阻塞拷出)。运行时 `burstCopySize` 只能收紧:glm5(kvLoraRank=512,BF16,perKBytes=1024)默认 16KB 被 TILING 压到 8KB,即每 burst 8 token;`burstCopySize <= perKBytes`(如 512)时 `burstCopyCnt == 1`,退化为逐 token 拷入——dense 行亦然,与移除 burst 窗口前的实现完全一致。

**有效长度裁剪**:`cnt = min(32, min(numCandidates, topK) - offsetInBatch)`,`cnt <= 0` 的整 chunk(纯 dummy 槽位)直接跳过,输出尾部保持零初始化(测试按"槽位 ≥ 有效数不写"整张断言);批切换时从 GM 重新读该 batch 的 `cachedLens` 并重算 `numCandidates`。cachedLens 分窗 staging:窗口装满当前处理批次的长度(`maxBatchSize = (UB_SIZE - off) / 4`,按 UB 余量自适应),窗口耗尽才回 GM 续载(EVENT_ID4 同步)。

**UB 布局**:4 份 `kBufs`(每份 `ROUND_UP(32*perKBytes, 32B)`)+ 4 份 `peBufs` + 4 份 `topks`(32×uint32)+ cachedLens 窗口,总占用须在 192KB UB 内(kernel 断言仅 debug 构建生效);glm5 形状下约 148KB。

**流水同步**:事件按缓冲编号配对——`(S, MTE2)`/`(MTE2, S)` EVENT_ID0..3 管理 `topks[b]` 的可用/释放,`(MTE2, MTE3)`/`(MTE3, MTE2)` EVENT_ID0..3 管理数据缓冲 `kBufs[b]/peBufs[b]` 的写完/复用,cachedLens staging 独占 EVENT_ID4;行间 `curr = (curr + 1) % 4` 轮转,chunk `i` 的拷入与 chunk `i-3..i-1` 的拷出重叠。kernel 结束前 drain 全部 4 个缓冲的 MTE2 事件。

**在模型中的位置**:decode + DSA 且序列较长时(阈值 `XLITE_MLA_DENSE_THRESHOLD`,`csrc/model.cpp`),先本算子收集 dense cache,再以 `dense=True` 调 mla_v2;短序列或 prefill 时直接走 paged mla_v2,不需要本算子。cxa 的 dense 模式亦在 binding 层(`CXA`,`csrc/_C.cpp`)内部调用本算子(仅 K 通道,peCache 传空,走 nope 变体),把压缩 KV 的 top-k token 收集成连续 dense cache 后再进 cxa kernel。
