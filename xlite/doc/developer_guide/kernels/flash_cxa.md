# flash_cxa

## 功能概述

CXA 的 flash(online-softmax)变体,是 cxa 的长序列路径:当某行压缩 KV 总宽 `compressTotalLen = (cachedLen + queryLen) / compressRatio` 超过 PingPong UB 预算(`MAX_SOFTMAX_PINGPONG_LEN = 10496`,`csrc/kernels/kernel_param.h` 的 `MAX_SOFTMAX_PINGPONG_LEN` 宏)时,由 Python 侧 `enable_flash_attention=True` 路由进入(`csrc/_C.cpp` 的 `CXA` 的 `else if (enableFlashAttention && compressRatio != 0)` 分支)。它把 KV-len(压缩 token)维度切成 `tileSizeOfCachedKV` 宽的 tile,每 tile 上 AIC 算 `QK = (q · swaK^T) | (q · compressK^T)` 与 `sv = softmax_local(QK) · K^T`,AIV 用 online softmax 跨 tile 增量合并,最终 `output = softmax_causal(QK) · K^T`。支持 DSA top-k token 选择(`topkIndices`)。

与 cxa(单遍)的关系:cxa 把整行 `[swaSegWidth, kvSize]` 分数一次性算进 softmax,行宽受 UB 预算限制;flash_cxa 把压缩段切成 `tileSizeOfCachedKV` 个 tile,SWA 段**只挂在第一个 KV tile**(`kvIdx == 0`)上,其余 tile 是纯压缩段。每个 tile 的 QK/SV/softmax 五级流水重叠,跨 tile 用 online softmax 合并 max/sum 与 output。数学上两者等价(容差 `atol=5e-5, rtol=5e-2`,见 `tests/kernels/cxa.py` 的 `assert_close`(关键字 `atol=5e-5, rtol=5e-02`))。

SWA 段语义、压缩段分页寻址、attn_sink 并入分母、topk gather/scatter 等均与 cxa 完全一致,详见 [cxa.md](cxa.md);本文仅描述 flash 变体的差异部分。

## 输入输出参数

Python 侧经 `cxa(..., enable_flash_attention=True, tile_size_of_cached_kv=8192)` 进入(`xlite/_C.pyi` 的 `def cxa(...)`,flash 参数说明见其 docstring(关键字 `enable_flash_attention` / `tile_size_of_cached_kv`)):

```python
cxa(
    rt,
    q,
    swa_k_cache,
    compress_k_cache,
    swa_block_tables,
    compress_block_tables,
    swa_block_size,
    compress_block_size,
    attn_sink,
    output,
    batch,
    query_start_loc,
    query_lens,
    cached_lens,
    n_heads,
    head_dim,
    scale,
    window_size,
    compress_ratio,
    index_topk,
    topk_indices,
    enable_flash_attention=True,
    tile_size_of_cached_kv=8192,
)
```

host 封装 `XliteOpFlashCXA`(`csrc/op.cpp` 的 `XliteOpFlashCXA`)→ kernel `flash_cxa_bfloat16_t`(`csrc/kernels/flash_cxa.h` 的 `FLASH_CXA_FUNC_DEFINE`(关键字 `flash_cxa_##dtype`))。kernel 签名(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::Init`):

```cpp
flash_cxa_<dtype>(q, swaKCache, compressKCache, swaBlockTables, compressBlockTables,
                  swaBlockSize, compressBlockSize, swaBlockTableStride, compressBlockTableStride,
                  attnSink, qk, sv, max, sum, lastMax, lastSum, sync, output, batch,
                  queryStartLoc, queryLens, cachedLens, nHeads, headDim, scale, windowSize,
                  compressRatio, indexTopK, topkIndices, swaKCacheStride, compressKCacheStride,
                  tileSizeOfCachedKV, zeroGm)
```

末尾的 `zeroGm` 是 host 自动提供的零缓冲(`rt._attnZeroBuffer`,host launch 见 `csrc/op.cpp` 的 `XliteOpFlashCXA`),`CxaAicHelper::Init` 用它把两份 L1B ping-pong 缓冲预填零,避免上一 tile 残留 K 数据导致 mmad NaN(与 attention 同源 nan bug 修复)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q | 输入 | `[totalQ, nHeads, headDim]` | BF16 | query |
| swa_k_cache | 输入 | `[swaBlockNum, swaBlockSize, headDim]` | BF16 | 分页滑动窗口 KV cache(始终分页);支持 dim-0 跨度(内核按 `swaKCacheStride = MAX(stride[0], swaBlockSize*headDim)` 寻址) |
| compress_k_cache | 输入 | `[compressBlockNum, compressBlockSize, headDim]` | BF16 | 分页压缩 KV cache(flash 仅走 sparse 分页路径,不分页由非 flash 的 dense 路径处理);支持 dim-0 跨度(`compressKCacheStride = MAX(stride[0], compressBlockSize*headDim)`) |
| swa_block_tables / compress_block_tables | 输入 | `[rows >= batch, 对应表行跨度]` | INT32 | 分页 block table;持久化缓冲,行跨度为对应表 `stride[0]`(分配跨度),持久化 V2 表的 `shape[1]` 已由 PrepareAttn 收窄到运行期宽度,仅每行 lens 推导的有效前缀被读取 |
| swa_block_size / compress_block_size | 标量 | - | uint32 | cache block 大小 |
| attn_sink | 输入 | `[nHeads]` | FP32 | 每头注意力 sink 偏置;kernel 仅在第一个 KV tile(`kvIdx == 0`)读入 softmax 分母 |
| qk | workspace | `[aicNum * XLITE_MAX_M0 * 2, qkStride]`,`qkStride = swaSegWidth + tileSizeOfCachedKV` | BF16 | 当前 KV tile 的 QK 分数,按核双缓冲 |
| sv | workspace | `[aicNum * XLITE_MAX_M0 * 2, headDim]` | BF16 | 当前 tile 的 softmax·K^T 部分和 |
| max / sum | workspace | `[aivNum * XLITE_MAX_M0 * 2]` | FP32 | 当前 tile 局部 max / Σexp |
| lastMax / lastSum | workspace | `[totalQ, nHeads]` | FP32 | 跨 tile 全局 max / Σexp |
| sync | workspace | `[1, aivNum]` | INT32 | RingSync 计数器(host `Memset(0)`) |
| output | 输出 | `[totalQ, nHeads, headDim]` | BF16 | 最终注意力输出,原位写入(online 合并) |
| query_start_loc / query_lens / cached_lens | 输入 | `[batch]` | INT32 | query 长度前缀和 / 当前 query 长度 / 已缓存长度 |
| n_heads / head_dim / scale / window_size / compress_ratio / index_topk | 标量 | - | uint32/float | 同 cxa;flash 要求 `compressRatio != 0` |
| topk_indices | 输入(可选) | `[totalQ, indexTopK]` | INT32 | DSA top-k 压缩索引;`indexTopK == 0` 时禁用 |
| tile_size_of_cached_kv | 标量 | - | uint32 | KV-len(压缩 token)tile 宽度;host 要求 ≤ `MAX_SOFTMAX_PINGPONG_LEN = 10496`(`csrc/op.cpp` 的 `XliteOpFlashCXA`(关键字 `tileSizeOfCachedKV > MAX_SOFTMAX_PINGPONG_LEN`));`window_size > 0` 时还要求 `swaSegWidth + tileSizeOfCachedKV ≤ 10496`(tile 0 的 PingPong softmax 行宽 = SWA 段 + 压缩段,关键字 `swaSegWidth + tileSizeOfCachedKV > MAX_SOFTMAX_PINGPONG_LEN`)与 `swaSegWidth ≤ MAX_SWA_SEG_WIDTH(4096)`(topK 路径 `inTopK` 预算,关键字 `swaSegWidth > MAX_SWA_SEG_WIDTH`),默认 `MAX_KV_TILE_SIZE = 8192`(`csrc/auto_tuner.h` 的 `MAX_KV_TILE_SIZE` 宏) |
| zeroGm | 标量 | - | GM | host 自动提供的零缓冲(`rt._attnZeroBuffer`),非 Python API 参数;`CxaAicHelper::Init` 用它预填零 L1B(见上文) |

workspace 由 `csrc/_C.cpp` 的 `CXA`(关键字 `qkWidth = swaSegWidth + tileSizeOfCachedKV`,workspace 分配)分配:`qk`/`sv` 宽度分别为 `qkWidth = swaSegWidth + tileSizeOfCachedKV` 与 `headDim`,`max`/`sum` 各 `aivNum * XLITE_MAX_M0 * 2`,`lastMax`/`lastSum` 为 `[totalQ, nHeads]`,`sync` 为 `[1, aivNum]` 并 `Memset(0)`。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| bfloat16_t | `csrc/kernels/flash_cxa_bfloat16_t.cpp` | `flash_cxa_bfloat16_t` |

host 侧仅 BF16(`csrc/op.cpp` 的 `XliteOpFlashCXA`(关键字 `EachXDtype(BF16`),即 `EachXDtype(BF16, ...)` 分支,否则抛 unsupported)。

调度类型为 `KERNEL_TYPE_MIX_AIC_1_2`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::Init`(关键字 `KERNEL_TYPE_MIX_AIC_1_2`)):1 AIC + 2 AIV 混合调度,与 cxa 一致。

host 侧约束(`csrc/op.cpp` 的 `XliteOpFlashCXA`(约束校验段,关键字 `MAX_SOFTMAX_PINGPONG_LEN` / `MAX_TOPK_NUM` / `compressRatio`)):`tileSizeOfCachedKV ≤ MAX_SOFTMAX_PINGPONG_LEN`、`indexTopK ≤ MAX_TOPK_NUM = 2048`、`compressRatio != 0`(flash 必须有压缩段)。

## 实现原理

与 flash_mla_v2、flash_attention 同构的五级流水(混合 AIC/AIV),cube 环节换成 `CxaAicHelper` 的 CXA GEMM(见 [cxa.md](cxa.md) 的 CxaAicHelper 段)。任务空间 `taskNum = queryNum * kvNum`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `taskNum = queryNum * kvNum`)),其中:

- `queryTileSize = XLITE_MAX_M0 / nHeads`(一个 query tile 内所有头一起算,`mSize = queryTaskLen * nHeads` 行,`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `queryTileSize = XLITE_MAX_M0 / nHeads`));
- `queryNum = ceil(queryLen / queryTileSize)`;
- `compressTotalLen = (cachedLen + queryLen) / compressRatio`;
- `kvNum = ceil(compressTotalLen / tileSizeOfCachedKV)`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `kvNum = DIV_ROUND_UP(compressTotalLen`));当 `windowSize != 0` 且 `kvNum == 0` 时强制 `kvNum = 1`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `kv tile 0 must always run`)),保证 SWA 段至少有一个 tile 承载;
- 每个 tile:`kvIdx = idx % kvNum`,`queryIdx = idx / kvNum`,`kvOffset = kvIdx * tileSizeOfCachedKV * compressRatio`,`kvLen = min(tileSizeOfCachedKV * compressRatio, calcLen - kvOffset)`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `kvOffset = kvIdx * tileSizeOfCachedKV * compressRatio`))。

任务按 `totalIdx % block_num` 轮转分核(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `totalIdx % block_num`)),AIC 与 AIV 用同一任务编号一一对应。

### SWA 段只挂第一个 KV tile

`hasSwa = (windowSize != 0) && (kvIdx == 0)`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `hasSwa = (windowSize != 0) && (kvIdx == 0)`)):SWA 段(`[0, swaSegWidth)`)仅出现在第一个 KV tile 的分数行前缀,后续 tile 是纯压缩段(`swaSegWidthEff = hasSwa ? swaSegWidth : 0`,`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `swaSegWidthEff = hasSwa ? swaSegWidth : 0`))。因为 `swaSegWidth = windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B` 远小于 `tileSizeOfCachedKV`(典型 8192),SWA 段不会跨 tile。`CxaAicHelper::RunAicQK`/`RunAicSV` 的 `hasSwa` 参数据此控制是否读写 SWA 段列(详见 cxa.md)。

### 流水线(flag 编号)

每个 (query tile, kv tile) 任务:

1. **AIC `RunAicQK`**(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `RunAicQK`)):算本 tile 的 `QK = q · K`(SWA 段仅 hasSwa 时算,压缩段按 kvOffset/kvLen 窗口分页寻址),写 `qk[curr]` → `ffts_cross_core_sync(PIPE_FIX, softmaxConfig)`(flag0,mode 2 组内 AIC/AIV 同步,`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `softmaxConfig` / `ffts_cross_core_sync(PIPE_FIX, softmaxConfig)`))。
2. **AIV `RunAivSoftmaxPingPong`**(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `RunAivSoftmaxPingPong`)):`wait_flag_dev(0)`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `wait_flag_dev(0)`))等 QK 完成后,对 tile 内行做带 scale 的 softmax。行有效长度 `actualCalcSoftmaxLen = calcSoftmaxLen - kvOffset`(clip 到 kvLen,`calcSoftmaxLen = cachedLen + queryTaskStart + 1`,`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `actualCalcSoftmaxLen`));输出行宽 `outN = swaSegWidthEff + ROUND_UP(kvLen / compressRatio, 4 * svk0)`(clip 到 `qkStride`,`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `outN = swaSegWidthEff + ROUND_UP`));若 `calcCompressLen > indexTopK` 且 `indexTopK > 0`,走 vgather/scatter 的 top-k softmax 路径(`topkIndices + indexTopK * queryTaskOffset`,`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `topkIndices + indexTopK * queryTaskOffset`));`attnSink` 仅 `kvIdx == 0` 时传入(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `kvIdx == 0 ? attnSink : nullptr`)),后续 tile 传 `nullptr`(sink 只并入第一个 tile 的分母,等价于并入全局分母)。局部 max/sum 写 `max[curr]`/`sum[curr]` → `ffts_cross_core_sync(PIPE_MTE3, config)`(flag2,AIV config flagIdx=2,`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `flagIdx = 2` / `ffts_cross_core_sync(PIPE_MTE3, config)`))。
3. **AIC `RunAicSV`**(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `RunAicSV`)):`wait_flag_dev(2)`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `wait_flag_dev(2)`))等 softmax 完成后,算 `sv = softmax_local(QK) · K^T`,写 `sv[last]` → `ffts_cross_core_sync(PIPE_FIX, updateConfig)`(flag1,`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `ffts_cross_core_sync(PIPE_FIX, updateConfig)`))。
4. **AIV `RunAivSoftmaxUpdate`**(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `RunAivSoftmaxUpdate`)):`wait_flag_dev(1)`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `wait_flag_dev(1)`))等 SV 完成后,把 sv/max/sum 与 output/lastMax/lastSum 在线合并(算法见 flash_attention 文档;headSize 参数为 headDim,maskStride=nHeads,`compressRatio` 传给 update 用于压缩坐标换算,`lastKvOffset == 0` 标志首 tile)。

相邻任务错峰:本 tile 的 QK 与上一 tile 的 SV/update 重叠(`curr`/`last` 双缓冲翻转,`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `curr = 1 - curr`)/ `FlashCXA::RunAiv`(关键字 `curr = 1 - curr`)),末尾补做最后一个 tile 的 SV(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`(关键字 `do last SV`))与 update(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `do last update`)),与 flash_mla_v2 完全一致。

### 跨核 RingSync

同一 query 块的连续 KV tile 轮转落在不同核,update 读前一 tile 的 output/lastMax/lastSum 前必须 `ringSync.WaitPrevCore()`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `ringSync.WaitPrevCore`)),非最后 tile 完成后 `ringSync.SetNextCore()`(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `ringSync.SetNextCore`),经 `set_flag/wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0)` 与 MTE3 流水握手,关键字 `set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0)`),kernel 尾部 `PipeBarrier<PIPE_ALL>()` + `ResetPrevCore()` 复位(`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`(关键字 `PipeBarrier<PIPE_ALL>()` / `ResetPrevCore()`);实现 `csrc/kernels/ring_sync.h`)。仅 `lastKvOffset != 0`(非首 tile)时才需要等待前一核,首 tile 无依赖。

### 关键代码位置

- 主类与 Init:`csrc/kernels/flash_cxa.h` 的 `FlashCXA` 类、`FlashCXA::Init`
- AIC 流水(`RunAic`):`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAic`
- AIV 流水(`RunAiv`,softmax + online update):`csrc/kernels/flash_cxa.h` 的 `FlashCXA::RunAiv`
- QK/SV cube 计算:`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicQK` / `CxaAicHelper::RunAicSV`(与 cxa 共用,`hasSwa` 参数由 flash_cxa 传入)
- tile softmax(ping-pong,含 top-k):`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`
- online softmax update:`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxUpdate`
- RingSync:`csrc/kernels/ring_sync.h`
- host launch:`csrc/op.cpp` 的 `XliteOpFlashCXA`、workspace 分配 `csrc/_C.cpp` 的 `CXA`(关键字 `qkWidth = swaSegWidth + tileSizeOfCachedKV`)、Python 路由 `csrc/_C.cpp` 的 `CXA`(关键字 `else if (enableFlashAttention && compressRatio != 0)`)

### 片上缓冲分配与占用汇总(AIC L1/L0A/L0B/L0C、AIV UB)

AIC 侧与 cxa 完全共用 `csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::Init` 布局(flash 恒 sparse,tiling 常量与典型配置下的数值均与 cxa 一致,排布机制与逐表解读见 [cxa.md](cxa.md) 的同名小节):QK 段与 SV 段在同级的 `bufferAddr` 各自从 0 重新起排、串行互斥复用,该级实际占用取两段较大者。AIV 侧为 `RunAivSoftmaxPingPong` + `RunAivSoftmaxUpdate`(`csrc/kernels/softmax_attn_aiv.h`)函数内联排布,`off` 从 0 顺序累进、每槽按 `VECTOR_MAX_BYTESIZE=256B` 向上对齐。硬件容量(910B/dav C220,NpuArch 2201,CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`):L1 512KB、L0A 64KB、L0B 64KB,L0C 总容量 128KB（`l0_c_size=131072`，16 bank × 8KB），Mmad 累加缓冲即 `TPosition::CO1` 且占满整个 L0C;AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`),每个 AIV subblock 各持一份独立 UB。

dtype 记法:`FlashCXA<Dtype>` 仅以 bfloat16_t 实例化(`csrc/kernels/flash_cxa_bfloat16_t.cpp`),`Dtype`=bf16(2B),L0C mmad 累加器为 float(4B)。典型配置取 `tests/kernels/cxa.py` 的 c4a-topk 模型(nHeads=16、headDim=512、windowSize=128 → `swaSegWidth=272`、compressRatio=4、indexTopK=512、swaBlockSize=64、compressBlockSize=128)与长序列工作负载 `(1, [131071], [1])`(`tileSizeOfCachedKV=8192`:`compressTotalLen = 131072/4 = 32768` → `kvNum=4`,SWA 段只挂首 tile,`qkStride = 272+8192 = 8464`)。AIC 表与 cxa.md 相同(swaBlockSize=64、compressBlockSize=128、headDim=512),占用 = 单份大小 × 乒乓个数。

**AIC L1(A1,容量 512KB)——QK 段**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `ql1aBuf` | `(XLITE_MAX_M0, headDim)`=(128, 512) | bf16 | 1(事件0) | `XLITE_MAX_M0*headDim*sizeof(Dtype)` | 131072 |
| `kl1bBuf` | `(max(qkwn0,qkcn0), 4*qkk0)`=(128, 512) | bf16 | 2(事件2/3) | `max(qkwn0,qkcn0)*4*qkk0*sizeof(Dtype)` | 262144 |
| **合计** | | | | | **393216** |

**AIC L1(A1)——SV 段(与 QK 段互斥复用)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `scoresl1aBuf` | `(XLITE_MAX_M0, 4*svwkMax)`=(128, 256) | bf16 | 2(事件0/1) | `XLITE_MAX_M0*4*svwkMax*sizeof(Dtype)` | 131072 |
| `ktl1bBuf` | `(svn0, max(svwk0,2*svck0))`=(256, 128) | bf16 | 2(事件2/3) | `max(svwk0,2*svck0)*svn0*sizeof(Dtype)` | 131072 |
| **合计** | | | | | **262144** |

**AIC L0A(A2,容量 64KB)——QK 段**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `qkl0aBuf` | `(XLITE_MAX_M0, qkk0)`=(128, 128) | bf16 | 2(事件0/1) | `XLITE_MAX_M0*qkk0*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0A(A2)——SV 段(与 QK 段互斥复用)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `svl0aBuf` | `(XLITE_MAX_M0, svwkMax)`=(128, 64) | bf16 | 2(事件0/1) | `XLITE_MAX_M0*svwkMax*sizeof(Dtype)` | 32768 |
| **合计** | | | | | **32768** |

**AIC L0B(B2,容量 64KB)——QK 段**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `qkl0bBuf` | `(MAX_N0, qkk0)`=(128, 128) | bf16 | 2(事件0/1) | `MAX_N0*qkk0*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0B(B2)——SV 段(与 QK 段互斥复用)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `svl0bBuf` | `(svn0, svwkMax)`=(256, 64) | bf16 | 2(事件0/1) | `svn0*svwkMax*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C(CO1 区,容量 128KB)——QK 段**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `qkl0cBuf` | `(XLITE_MAX_M0, MAX_N0)`=(128, 128) | float | 1(事件0) | `XLITE_MAX_M0*MAX_N0*sizeof(float)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C(CO1 区)——SV 段(与 QK 段互斥复用)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `svl0cBuf` | `(XLITE_MAX_M0, svn0)`=(128, 256) | float | 1(事件0) | `XLITE_MAX_M0*svn0*sizeof(float)` | 131072 |
| **合计** | | | | | **131072** |

**AIV UB(预算 `UB_SIZE=192KB`,每 subblock 一份)——`RunAivSoftmaxPingPong`(tile softmax,含 top-k)**

flash 路径 `saveMaxSum` 恒为 true(`max[curr]`/`sum[curr]` 工作区非空,`maxOut`/`sumOut` 槽占位);`len = MAX(outN, topK+swaSegWidth)`,首 KV tile(`hasSwa`)行最宽——`outN = swaSegWidth + ROUND_UP(kvLen/compressRatio, 4*svk0) = 272+8192 = 8464`(clip 到 `qkStride`),非首 tile 纯压缩段 `outN=8192`;下表按最宽的首 tile 计,`sink` 槽仅 `kvIdx==0`(attnSink 只并入首 tile 分母)时占用。

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `in` | `len=8464` 个 Dtype | bf16 | 2(事件2/3) | `ROUND_UP(len*sizeof(Dtype),256)` | 34304 |
| `out` | `len=8464` 个 Dtype | bf16 | 2(事件2/3) | `ROUND_UP(len*sizeof(Dtype),256)` | 34304 |
| `cal` | `len=8464` float | float | 1 | `ROUND_UP(len*sizeof(float),256)` | 34048 |
| `temp` | `(len/2)` float 量级的归约暂存 | float | 1 | `(DIV_ROUND_UP(len*4,256)/2)*256` | 16896 |
| `maxOut` | 1 个 fp32 标量广播槽 | float | 2 | `VECTOR_MAX_BYTESIZE` | 512 |
| `sumOut` | 1 个 fp32 标量广播槽 | float | 2 | `VECTOR_MAX_BYTESIZE` | 512 |
| `indices` | `topK=512` 个 int32 | int32 | 2(事件0) | `ROUND_UP(topK*4,256)` | 4096 |
| `inTopK` | `(topK+swaSegWidth)=784` 个 Dtype | bf16 | 1 | `ROUND_UP((topK+swaSegWidth)*sizeof(Dtype),256)` | 1792 |
| `indicesFp32` | `topK=512` float | float | 1 | `ROUND_UP(topK*4,256)` | 2048 |
| `cmp0` | 边界标量广播槽(kvOffset 基) | int32 | 1 | `VECTOR_MAX_BYTESIZE*2` | 512 |
| `cmp1` | 边界标量广播槽(因果上界) | int32 | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `cmpFp320` | 边界 fp32 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `cmpFp321` | 边界 fp32 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `topkMask0` | `topK/8` 位 hit-mask | uint8 | 1 | `ROUND_UP(topK/8,256)` | 256 |
| `topkMask1` | `topK/8` 位 hit-mask | uint8 | 1 | `ROUND_UP(topK/8,256)` | 256 |
| `currIndices` | `topK=512` 个 int32 | int32 | 1 | `ROUND_UP(topK*4,256)` | 2048 |
| `sink` | sink 标量广播槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| **合计** | | | | | **132608** |

**AIV UB——`RunAivSoftmaxUpdate`(跨 tile online 合并)**

`nBytes = ROUND_UP(headSize*sizeof(Dtype),256)`、`nFloatBytes = ROUND_UP(headSize*sizeof(float),256)`,headSize 即 headDim;`svPrevUb`/`svCurrUb`(事件0/1,GM 搬入乒乓)与 `svOutUb`(事件2/3,写回乒乓)各两份,统计/缩放类槽位为单标量 256B 广播槽。

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `svPrevUb` / `svCurrUb` | `headSize=512` 个 Dtype | bf16 | 2(事件0/1) | `ROUND_UP(headSize*sizeof(Dtype),256)` | 4096 |
| `svOutUb` | `headSize=512` 个 Dtype | bf16 | 2(事件2/3) | `ROUND_UP(headSize*sizeof(Dtype),256)` | 2048 |
| `maxPrevUb` / `sumPrevUb` / `maxCurrUb` / `sumCurrUb` | 1 个 fp32 标量广播槽 | float | 2(事件0/1) | `VECTOR_MAX_BYTESIZE` | 2048 |
| `newSumOutUb` / `newMaxOutUb` | 1 个 fp32 标量广播槽 | float | 2(事件2/3) | `VECTOR_MAX_BYTESIZE` | 1024 |
| `scalePrevUb` / `scaleCurrUb` / `tempUb` / `newSumUb` / `newMaxUb` / `sumPrevCalUb` / `sumCurrCalUb` | 1 个 fp32 标量广播槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 1792 |
| `svPrevFloatUb` / `svCurrFloatUb` / `svOutFloatUb` | `headSize=512` float | float | 1 | `ROUND_UP(headSize*sizeof(float),256)` | 6144 |
| **合计** | | | | | **17152** |

**占用汇总**(各表占用 vs 硬件容量;L1/L0A/L0B/L0C 按 QK/SV 两段互斥取 max;字节数):

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| L1(取 QK/SV 较大者) | 512 KB | 393216 B(QK 段) | 75.0% |
| L0A(取 QK/SV 较大者) | 64 KB | 65536 B(QK 段) | 100% |
| L0B(取 QK/SV 较大者) | 64 KB | 65536 B(两段相等) | 100% |
| L0C CO1(取 QK/SV 较大者) | 128 KB | 131072 B(SV 段) | 100% |
| UB(PingPong 版,每 AIV subblock) | 192 KB | 132608 B | 67.4% |
| UB(Update 版,每 AIV subblock) | 192 KB | 17152 B | 8.7% |

要点:

- AIC 侧各级占用与 cxa 完全一致:`svl0cBuf` = XLITE_MAX_M0(128)×svn0(256)×4B = 128KB 恰好顶满 CO1 区(CO1 ≥ 128KB 的下界旁证,非 L0C 总量证明),L0A/L0B 顶满 64KB,排布机制(QK/SV 互斥复用、svn0 自适应)详见 [cxa.md](cxa.md) 的同名小节;
- **flash 独有的 UB 大头是 `RunAivSoftmaxPingPong`**(典型 132608B、67.4%):行宽 `len` 的上界是 `qkStride = swaSegWidth + tileSizeOfCachedKV`,host 校验 `swaSegWidth + tileSizeOfCachedKV <= MAX_SOFTMAX_PINGPONG_LEN=10496`(`csrc/op.cpp` 的 `XliteOpFlashCXA`,另有 `swaSegWidth <= MAX_SWA_SEG_WIDTH=4096` 与 `tileSizeOfCachedKV <= 10496` 两道前置校验)正是该布局的容量来源;SWA 段只挂首 tile 使首 tile 成为最宽行(8464),后续纯压缩 tile 行宽 8192、且无 sink 槽;`maxOut`/`sumOut` 因 flash 恒传 max/sum 工作区(`saveMaxSum=true`)而占位,与 cxa(恒传 nullptr)不同;
- **`indexTopK <= MAX_TOPK_NUM=2048` 的 host 校验**约束 topk 槽位(`indices`/`inTopK`/`currIndices` 等),与 `tileSizeOfCachedKV` 校验共同构成 PingPong UB 上界的双重来源;
- 跨核 RingSync 的标量 `val`(`SetNextCore`/`WaitPrevCore`/`ResetPrevCore`)固定借用 UB 偏移 0 的 4 字节,与 `in[0]` 首元素别名——`SetNextCore` 前先 `set_flag/wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0)` 等 MTE3 队列排空,且下一 tile 的 `in` 会被重新搬入覆写,不占独立槽位(与 [indexer_topk](indexer_topk.md) 的跨核接力同模式)。
