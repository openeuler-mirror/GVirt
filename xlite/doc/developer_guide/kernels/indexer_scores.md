# indexer_scores

## 功能概述

DSA indexer 的得分计算算子(纯 Cube 核)。对每个 query token 计算 indexer 头的注意力得分并与头权重融合,数学上等价于两步 einsum(见 `csrc/kernels/indexer_scores.h` 的 `IndexerScores::Init`(注释块 `scores = k * q`)与测试 `tests/kernels/indexer_scores.py` 的参考实现,关键字 `torch.einsum("bshd,btd->bsht"` / `index_score = torch.einsum("sht,sh->st"`):

```text
scores[s,h,t]      = sum_d q[s,h,d] * kCache[t,d]        # Q·K^T
index_score[s,t]   = sum_h scores[s,h,t] * weight[s,h]    # 加权头融合
```

输出 `[total_query_len, max_num_blocks*block_size]` 的稠密得分矩阵,供后续 topk 选取稀疏注意力 token。与 `indexer_topk` 的区别:本算子只算分数,不融合 topk。

## 输入输出参数

Python 侧调用(`tests/kernels/indexer_scores.py` 的参考实现段,关键字 `indexer_scores(rt, q_xlite, k_cache_xlite`):

```python
indexer_scores(
    rt,
    q,
    k_cache,
    weight,
    scores,
    query_start_loc,
    query_lens,
    cached_lens,
    block_tables,
    n_heads,
    head_dim,
    block_size,
    batch,
)
```

host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpIndexerScores`(`KERNEL_TASK_TYPE_AIC_ONLY`,用 `rt.aicNum` 个 Cube 核)。kernel 签名(`csrc/kernels/indexer_scores.h` 的 `INDEXER_SCORES_FUNC_DEFINE`):

```cpp
indexer_scores_<dtype>(GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR scores,
                       GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                       GM_ADDR blockTables, uint32_t kCacheStride, uint32_t nHeads,
                       uint32_t headDim, uint32_t blockSize, uint32_t batch,
                       uint32_t maxNumBlock)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q | 输入 | `[total_query_len, nHeads*headDim]` | fp16 / bf16 | 所有 batch 拼接的 flatten query |
| kCache | 输入 | `[num_blocks, blockSize, 1, headDim]` | fp16 / bf16 | paged indexer K cache(末维前的 `1` 为单 head 维),物理块号由 blockTables 索引 |
| weight | 输入 | `[total_query_len, headDim + nHeads]` | fp16 / bf16 | 每 token 的融合权重取自第 `headDim` 列起的 `nHeads` 列(与 Q 同投影输出拼接存放) |
| scores | 输出 | `[total_query_len, maxNumBlock*blockSize]` | fp16 / bf16 | 每行一个 query token 对全部 KV 位置的 index_score(行内按全局 token 位置布局,valid 长度 = cachedLens[i]+queryLens[i]) |
| queryStartLoc | 输入 | `[batch]` | int32 | 各 batch query 在拼接 q 中的起始偏移(前缀和) |
| queryLens | 输入 | `[batch]` | int32 | 各 batch 的 query 长度 |
| cachedLens | 输入 | `[batch]` | int32 | 各 batch 已缓存 KV 长度(总 KV 长度 = cachedLen + queryLen) |
| blockTables | 输入 | `[rows >= batch, blockTableStride]` | int32 | 逻辑块 → 物理块映射表;持久化缓冲,行跨度为分配跨度,仅每行 lens 推导的有效前缀被读取;行跨度由 host 取 `blockTables.stride[0]` |
| maxNumBlock | 标量 | - | uint32 | scores 输出行宽对应的块数,host 取 `blockTables.shape[1]`(scores 由调用方按运行期宽度分配) |
| kCacheStride | — | 标量 | uint32 | kCache 的 dim-0 跨度(元素数);连续时等于 blockSize×headDim,非连续时由宿主传入真实跨距 |
| nHeads / headDim / blockSize / batch | — | 标量 | uint32 | 测试配置:nHeads=64, headDim=128, blockSize=128;约束 `blockSize <= MAX_M0=128`、`headDim <= k0`(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Init`,关键字 `assert(blockSize <= MAX_M0)` / `assert(headDim <= k0)`) |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `indexer_scores_bfloat16_t` | `csrc/kernels/indexer_scores_bfloat16_t.cpp` | q/kCache/weight/scores 全为 bf16 |
| `indexer_scores_float16_t` | `csrc/kernels/indexer_scores_float16_t.cpp` | 全为 fp16 |

dtype 分派要求四张 tensor 同 dtype(`csrc/op.cpp` 的 `XliteOpIndexerScores`,关键字 `EachXDtype(FP16, q, kCache, weight, scores)` 分派段)。无 fp32 变体。

## 实现原理

### 两级 MMAD 分块策略

每次 mmad 的两级矩阵乘语义(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Init`,关键字注释块 `scores = k * q` / `index_scores = weights * scores`):

- 第一级 `scores = k * q`:m 维为 KV token 块(`blockSize`),n 维为 `queryTileSize*nHeads`(一次装下若干 query token 的全部头),k 维为 `headDim`;
- 第二级 `index_scores = weight * scores`:m 维为 query(单 token),n 维为 KV token 块,k 维为 `nHeads` —— 即对每个 query token,用其 `nHeads` 个权重把各头得分加权求和。

query tile 划分:`queryTileSize = MAX_M0 / nHeads`(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `int queryTileSize = MAX_M0 / nHeads`,MAX_M0=128;nHeads=64 时 queryTileSize=2)。任务枚举(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `for (int batchIdx = 0; batchIdx < batch; batchIdx++)` 任务枚举段):外层 batch,内层 `taskNum = queryNum * kvNum`,其中 `kvNum = DIV_ROUND_UP(totalLen, blockSize)`(以 cache 物理块为粒度切 KV)。所有任务按 `totalIdx % block_num == block_idx` 静态分配到 Cube 核(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `if (totalIdx % block_num != block_idx) continue;`),核间无依赖、无需同步。

### L1/L0 内存布局与乒乓

`Init` 中手工排布 L1 与 L0(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Init`,关键字 `kl1Buf`/`ql1Buf`/`wl1Buf`/`kql1Buf`/`l0aBuf`/`l0bBuf`/`l0cBuf` 布局段):

- L1(A1):`kl1Buf[2]`(K 块,`blockSize*headDim`,乒乓)、`ql1Buf[2]`(Q tile,`MAX_N0*headDim`,乒乓)、`wl1Buf[2]`(weight tile,`MAX_N0*nHeads`,乒乓)、`kql1Buf[2]`(第一级 mmad 结果中转,`blockSize*MAX_N0`,乒乓);
- L0:A2 `l0aBuf[2]`(`MAX_M0*k0`)、B2 `l0bBuf[2]`(`MAX_N0*k0`)、CO1 `l0cBuf`(fp32 累加)。`k0 = 256/sizeof(Dtype)`(fp16/bf16 为 128)。

数据搬运均用 `CopyGmToL1Nd2Nz` 完成 ND→NZ 布局转换(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `CopyGmToL1Nd2Nz(kl1Buf[curr], kCache[block * kCacheStride]`),即 GM 的行主序在 L1 中转置成分块列主,供 mmad 按 `CopyToL0ACol/CopyToL0BCol` 装载。

### 单任务流水(事件驱动)

单个任务(query tile × KV 块)内的流水(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `// copy k (kvLen, headDim) to L1` 起的单任务段),事件 ID 按缓冲轨道复用(k 用 0/1、q 用 2/3、weight 用 4/5):

1. 等 `kl1Buf[curr]` 空闲 → K 块 GM→L1(通过 `blockTable[kvIdx]` 找物理块,`block * kCacheStride` 定位);
2. 等 `ql1Buf[curr]` 空闲 → Q tile GM→L1;
3. K→L0A、Q→L0B,`CalMmad(l0cBuf, l0a, l0b, mBlockPad, nBlockPad, headDim)` 得到 `[kvLen, queryTileSize*nHeads]` 的头得分,`CopyL0CToL1`(`reluEn=1`)暂存到 `kql1Buf`——L0C→L1 的 fixpipe 搬运同时完成 ReLU 过滤(负得分置 0,DSA indexer 要求得分非负),省去单独一次 ReLU pass(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `CopyL0CToL1(kql1Buf[curr], l0cBuf, mBlockPad, nBlockPad, mBlockPad, mBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE, /*reluEn=*/1)`);
4. weight tile GM→L1;
5. 对 tile 内每个 query token q:weight 行(1,nHeads)→L0A、对应头得分列(kvLen,nHeads)→L0B,第二次 `CalMmad`(`m=MBLOCKSIZE`)得到 `(1, kvLen)` 的 index_score,`CopyToGm` 直接从 L0C 写 GM,目的地址 `scores[(qOffset+q) * maxNumBlock*blockSize + kvStart]`,以 `maxNumBlock*blockSize`(来自 scores 自身行宽)为 dstStride 保持行内全局位置布局(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `CopyToGm(scores[(qOffset + q) * maxNumBlock * blockSize + kvStart]`);
6. `curr = 1 - curr` 切换乒乓,下一任务的搬运与本任务的 mmad/写回重叠。

块尺寸对齐:kvLen 尾块按 `MBLOCKSIZE`/`NBLOCKSIZE` 向上取整(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `mBlockPad = ROUND_UP(kvLen, MBLOCKSIZE)` / `wnBlockPad = ROUND_UP(kvLen, NBLOCKSIZE)`),mmad 的 m/n/k 均为 `kBlockSize = 32/sizeof(Dtype)` 的倍数(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `constexpr int kBlockSize = 32 / sizeof(Dtype)` / `int kBlockNum = headDim / kBlockSize`)。

### 流水线同步

全部用 `HardEvent` 计数事件配对(MTE1↔MTE2、M↔MTE1、FIX↔M、MTE1↔FIX),`Run` 开头预置 11 个 flag 使首个任务无需等待(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `Run` 起始处连续 11 个 `SetFlag<HardEvent::...>` 预置段),结尾逐一 `WaitFlag` 清算(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Run`,关键字 `Run` 末尾连续 `WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID1)` 起的清算段)。mmad 结果写 GM 前经 `M_FIX`→`FIX_M` 配对保证 FIX 管线完成。

### 片上缓冲分配与占用汇总(AIC L1/L0A/L0B/L0C)

`Init`(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Init`)手动排布全部片上缓冲:L1(`TPosition::A1`)上四个缓冲连续累进 `bufferAddr`,L0A(`A2`)/L0B(`B2`)/L0C(`CO1`)三级物理独立、各自从偏移 0 重新起排。本算子为纯 Cube 核(`KERNEL_TASK_TYPE_AIC_ONLY`),无 AIV/UB 分配,实际使用的存储级即 L1/L0A/L0B/L0C 四级。硬件容量(910B/dav C220,NpuArch 2201,CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`):L0A 64KB、L0B 64KB、L1 512KB,L0C 总容量 256KB(`l0_c_size=262144`)为 CO1/CO2 两区之和,mmad 累加缓冲(`TPosition::CO1`)可用的 CO1 区为 **128KB**,下文 L0C 占用均按 CO1 128KB 计。dtype 记法:`Dtype` 为 q/kCache/weight/scores 的元素类型(fp16/bf16 实例化,`l0cBuf` 固定 `LocalTensor<float>`);两条路径均为 2B、`k0=256/sizeof(Dtype)=128`,占用完全相同,下表只给单占用列。缓冲大小只依赖编译期上限(`MAX_M0=MAX_N0=128`、`k0`)与 nHeads/headDim/blockSize,与 batch、序列长度无关;典型配置取测试配置(`tests/kernels/indexer_scores.py`)nHeads=64、headDim=128、blockSize=128——此时 `queryTileSize=MAX_M0/nHeads=2`,两级 mmad 的分块上界 `mBlockPad`(`ROUND_UP(kvLen,MBLOCKSIZE)`≤blockSize)、`nBlockPad`(`ROUND_UP(queryTileSize*nHeads,NBLOCKSIZE)`)与 `wnBlockPad`(`ROUND_UP(kvLen,NBLOCKSIZE)`)均取满 128。

**AIC L1(A1,容量 512KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `kl1Buf` | `(blockSize, headDim)` | Dtype | 2(事件0/1) | `blockSize*headDim*sizeof(Dtype)` | 65536 |
| `ql1Buf` | `(MAX_N0, headDim)` | Dtype | 2(事件2/3) | `MAX_N0*headDim*sizeof(Dtype)` | 65536 |
| `wl1Buf` | `(MAX_N0, nHeads)` | Dtype | 2(事件4/5) | `MAX_N0*nHeads*sizeof(Dtype)` | 32768 |
| `kql1Buf` | `(blockSize, MAX_N0)` | Dtype | 2(事件0/1) | `blockSize*MAX_N0*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **229376** |

**AIC L0A(A2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0aBuf` | `(MAX_M0, k0)` | Dtype | 2(事件0/1) | `MAX_M0*k0*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0B(B2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0bBuf` | `(MAX_N0, k0)` | Dtype | 2(事件0/1) | `MAX_N0*k0*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C(CO1 区,容量 128KB;总容量 256KB 含 CO2)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0cBuf` | 第一级 `(mBlockPad, nBlockPad)` / 第二级 `(MBLOCKSIZE, wnBlockPad)` | float | 1(无乒乓,两级 mmad 串行共享) | 峰值 `mBlockPad*nBlockPad*sizeof(float)` | 65536 |
| **合计** | | | | | **65536** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| L1 | 512 KB | 229376 B | 43.8% |
| L0A | 64 KB | 65536 B | 100% |
| L0B | 64 KB | 65536 B | 100% |
| L0C(CO1) | 128 KB | 65536 B | 50% |

要点:

- **L0A/L0B 恰好占满 64KB**:`l0aBuf`/`l0bBuf` 按 `MAX_M0×k0 = MAX_N0×k0 = 128×128` 的编译期上限分配、与运行期形状无关,任何 2B dtype 下都是 100%——除 `assert(headDim <= k0)`、`assert(nHeads <= k0)`(`csrc/kernels/indexer_scores.h` 的 `IndexerScores::Init`)外,这两级是最先被卡住的容量约束;
- `l0cBuf` 单份被两级 mmad 串行共享,峰值取第一级 `(mBlockPad, nBlockPad)=(128,128)` 的 fp32 累加(64KB,占 CO1 一半),第二级 `(MBLOCKSIZE=16, wnBlockPad=128)` 仅 8KB,远低于峰值,无需乒乓;
- `kl1Buf`/`ql1Buf`/`kql1Buf` 三个 128×128 规格缓冲共 196608 B、占 L1 的 86%,是 L1 占用大头;`wl1Buf` 因 nHeads=64 只有 MAX_N0×64,单份减半为 16384 B;
- 全部槽位尺寸由编译期上限(`MAX_M0`/`MAX_N0`/`k0`)与 nHeads/headDim/blockSize 决定,占用与 batch、KV 长度无关;fp16 与 bf16 同为 2B,占用一致。
