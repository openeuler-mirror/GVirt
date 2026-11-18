# flash_mla_v2

## 功能概述

MLA 吸收式注意力的 flash(在线 softmax)版本,是 mla_v2 的长序列路径:当 paged KV 运行期总长 `ceil(maxTotalLens/blockSize) * blockSize > tileSizeOfCachedKV`(即 `blockTables.shape[1] * blockSize`,运行期行宽,不取 block table 分配跨度)时由模型层路由(`csrc/model.cpp` 的 `ForwardAttnMLAV2`,flash 路由分支关键字 `XliteOpFlashMLAV2`)。KV 维按 `tileSizeOfCachedKV` 分块,每块上 AIC 算 `QK = (qAbsorb·kCache^T + qr·peCache^T)*scale` 与 `sv = softmax_local(QK)·kCache`(注意 MLA 的 V 也是 latent kCache),AIV 用 online softmax 跨 tile 增量合并,最终 `oAbsorb = softmax_causal(QK) · kCache`。支持 DSA top-k token 选择(topkIndices)，支持 paged 和 per-batch 连续 KV cache。

## 输入输出参数

Python 侧经 `mla_v2(..., enable_flash_attention=True)` 进入 host 分支(规范签名见 `xlite/_C.pyi` 的 `mla_v2`,关键字参数 `nz: bool = False`——已废弃,传入值被忽略,NZ 由 `InitXTensor` 自动检测——、`enable_flash_attention: bool = False`、`tile_size_of_cached_kv: int = 8192`、`dense: bool = False`;`tests/kernels/mla.py` 以局部变量按关键字传入 `enable_flash_attention=enable_flash`/`tile_size_of_cached_kv=tile_size`/`nz=weight_nz`):

```python
mla_v2(
    rt,
    q_with_qr,
    qr,
    k_cache,
    pe_cache,
    wuk_t,
    wuv,
    output,
    ...,
    topk_indices,
    top_k,
    nz,
    enable_flash_attention=True,
    tile_size_of_cached_kv=8192,
)
```

host 封装 `MLAV2` 的 flash 分支(`csrc/_C.cpp` 的 `MLAV2`,`enableFlashAttention=true` 分支)→ `XliteOpFlashMLAV2`(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`)。kernel 签名(`csrc/kernels/flash_mla_v2.h` 的 `flash_mla_v2_<dtype>` 入口,宏 `FLASH_MLA_V2_FUNC_DEFINE`):

```cpp
flash_mla_v2_<dtype>(qAbsorb, qr, kCache, peCache, topkIndices, qk, sv, max, sum,
                     lastMax, lastSum, sync, oAbsorb, queryStartLoc, queryLens,
                     cachedLens, blockTables, nHeads, ropeHeadDim, kvLoraRank,
                     blockSize, batch, blockTableStride, scale, tileSizeOfCachedKV, topK,
                     kCacheStride, peCacheStride, dense, zeroGm)
```

末尾的 `zeroGm` 是 host 自动提供的零缓冲(`rt._attnZeroBuffer`,host launch 见 `csrc/op.cpp` 的 `XliteOpFlashMLAV2`),`MlaAicHelper::Init` 用它把两份 L1B ping-pong 缓冲预填零,避免上一 tile 残留 K 数据导致 mmad NaN(与 attention 同源 nan bug 修复)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| qAbsorb | 输入 | `[total_tokens, nHeads, kvLoraRank]` | BF16 | host 已做 WUK 吸收的 Q latent |
| qr | 输入 | `[total_tokens, nHeads, ropeHeadDim]` | BF16 | Q 的 RoPE 部分 |
| kCache / peCache | 输入 | paged: `[numBlocks, blockSize, kvLoraRank]`;dense: `[batch, capacity, kvLoraRank]` / `[..., ropeHeadDim]` | BF16 | latent K / RoPE K cache |
| topkIndices | 输入(可选) | `[total_tokens, topK]` | INT32 | DSA top-k token 下标(升序);host 限制 `topK ≤ MAX_TOPK_RUN_NUM=2056`(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`,关键字 `MAX_TOPK_RUN_NUM`;语义见下文「all-miss tile 的 zeroSum 防御」末段) |
| qk | workspace | `[aicNum * XLITE_MAX_M0 * 2, tileSizeOfCachedKV]` | BF16 | 当前 KV tile 的 QK 分数,按核双缓冲 |
| sv | workspace | `[aicNum * XLITE_MAX_M0 * 2, kvLoraRank]` | BF16 | 当前 tile 的 softmax·kCache 部分和 |
| max / sum | workspace | `[aivNum * XLITE_MAX_M0 * 2]` | FP32 | 当前 tile 局部 max / Σexp |
| lastMax / lastSum | workspace | `[total_tokens, nHeads]` | FP32 | 跨 tile 全局 max / Σexp |
| sync | workspace | `[1, aivNum]` | INT32 | RingSync 计数器(host Memset(0)) |
| oAbsorb | 输出 | `[total_tokens, nHeads, kvLoraRank]` | BF16 | 最终吸收式输出(WUV 投影在 host) |
| queryStartLoc / queryLens / cachedLens / blockTables | 输入 | `[batch]` / `[rows >= batch, blockTableStride]` | INT32 | 同 mla_v2(持久化表,仅每行有效前缀被读取)；dense 时不使用 blockTables |
| nHeads / ropeHeadDim / kvLoraRank / blockSize / batch / blockTableStride | 标量 | - | uint32 | 同 mla_v2(blockTableStride 为表行跨度,仅用于行寻址) |
| scale | 标量 | - | float | `(nopeHeadDim + ropeHeadDim)^-0.5` |
| tileSizeOfCachedKV | 标量 | - | uint32 | KV tile 长度,host 要求 ≤ `MAX_SOFTMAX_PINGPONG_LEN=10496`(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`,关键字 `MAX_SOFTMAX_PINGPONG_LEN`) |
| kCacheStride / peCacheStride | 标量 | - | uint32 | kCache / peCache 的 dim-0 跨度(元素数)：paged 为每物理块跨度，dense 为每请求跨度 |
| dense | 标量 | - | uint32 | 0=paged(默认)，1=per-batch 连续缓存；两种布局都保留 Top-K 选择 |
| zeroGm | 标量 | - | GM | host 自动提供的零缓冲(`rt._attnZeroBuffer`),非 Python API 参数;`MlaAicHelper::Init` 用它预填零 L1B(见上文) |

## 支持的数据类型

SFA-C8 在 host 接受独立的 INT8 `kCache[numBlocks, blockSize, 1, kvLoraRank]`、BF16 `peCache[numBlocks, blockSize, 1, ropeHeadDim]` 和 FP32 `kScaleCache[numBlocks, blockSize, 1, kvLoraRank/128]`，先通过 [gather_sparse_kv_cache](gather_sparse_kv_cache.md) 恢复临时连续 BF16 latent、PE，不生成新页表，再运行 Flash MLA 并保留 Top-K 选择。临时缓存按请求的页表容量分配，使用后归还 Runtime 内存池；当前不是逐 tile 反量化。

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| bfloat16_t | `csrc/kernels/flash_mla_v2_bfloat16_t.cpp` | `flash_mla_v2_bfloat16_t` |

底层 Flash MLA kernel 仅使用 BF16(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`,关键字 `EachXDtype(BF16`)；SFA-C8 的 cache 在进入该 kernel 前恢复为 BF16。

## 实现原理

与 flash_attention 同构的五级流水(混合 AIC/AIV,`KERNEL_TYPE_MIX_AIC_1_2`),矩阵环节换成 `MlaAicHelper` 的吸收式 GEMM(见 mla_v2 文档)。任务空间 `taskNum = queryNum * kvNum`(`kvNum = ceil((cachedLen+queryLen)/tileSizeOfCachedKV)`,见 `csrc/kernels/flash_mla_v2.h` 的 `RunAic`,关键字 `taskNum = queryNum * kvNum`),按 `totalIdx % block_num` 轮转分核。

### 流水线

每个 (query tile, kv tile) 任务:

1. AIC `RunAicQK(qAbsorb, qr, batchKCache, batchPeCache, ..., kvOffset, kvLen, qk[curr])`——KV tile 窗口由 kvOffset/kvLen 表达；paged 查 blockTable，块基址 = 物理块号 × dim-0 跨度(kCacheStride/peCacheStride)；dense 先按请求跨度取子视图，再按 token 偏移读取。完成 `ffts_cross_core_sync(PIPE_FIX, flag0)`(`csrc/kernels/flash_mla_v2.h` 的 `RunAic`,关键字 `ffts_cross_core_sync(PIPE_FIX, softmaxConfig`);
2. AIV `RunAivSoftmaxPingPong`:对 tile 内行做带 scale 的 softmax,行有效长度 `actualCalcSoftmaxLen = cachedLen + queryTaskStart + 1 - kvOffset`(clip 到 kvLen);输出行宽 `ROUND_UP(kvLen, 4*svk0)`(SV 每次 256 列预取,防残留脏数据,见 `csrc/kernels/flash_mla_v2.h` 的 `RunAiv`,关键字 `4 * svk0`);若 `calcLen > topK` 且 topK>0,走 vgather/scatter 的 top-k softmax 路径(见 `RunAiv`,关键字 `calcLen > topK`);局部 max/sum 写 workspace → `ffts_cross_core_sync(PIPE_MTE3, flag2)`;
3. AIC `wait_flag_dev(2)` 后 `RunAicSV(qk[last], lastKCache, ..., lastKvOffset, lastKvLen, sv[last])`:`sv = softmax_local(QK)·kCache` → `ffts_cross_core_sync(PIPE_FIX, flag1)`；lastKCache 保留上一任务的请求视图；
4. AIV `wait_flag_dev(1)` 后 `RunAivSoftmaxUpdate`:把 sv/max/sum 与 output/lastMax/lastSum 在线合并(算法见 flash_attention 文档;此处 headSize 参数为 kvLoraRank,maskStride=nHeads;all-miss tile 防御见下文「all-miss tile 的 zeroSum 防御」)。

相邻任务错峰(QK 先行、SV/update 滞后一个任务),末尾补做,与 flash_attention 完全一致(`csrc/kernels/flash_mla_v2.h` 的 `RunAic`(关键字 `RunAicQK`/`RunAicSV`)与 `RunAiv`(关键字 `RunAivSoftmaxPingPong`/`RunAivSoftmaxUpdate`))。

### top-k(DSA)语义

topkIndices 形状 `[total_tokens, topK]`,按 `topkIndices + topK * queryTaskOffset` 定位当前 query tile 的行;softmax 内用 vgather 只对 top-k token 的分数归一,scatter 回写后非 top-k 位置为 0,从而 SV 只在 top-k token 上有非零贡献(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`,关键字 `vcmpvs_ge`/`vgather` 与 scatter 段)。kvOffset 参数同时用于 topk 下标窗过滤(`vcmpvs_ge/lt` 与 [kvOffset, kvOffset+calcLen) 比较)。

### all-miss tile 的 zeroSum 防御

**问题**：长序列 decode 下,某个 KV tile 内的所有 token 都未入选 topk(该 tile 的 QK score 经 top-k mask 后恒为 -inf),`RunAivSoftmaxPingPong` 对该 tile 输出的局部 `currSum=0`。`RunAivSoftmaxUpdate` 在线合并时 `newSum = sumPrev*scalePrev + sumCurr*scaleCurr` 也可能为 0,Step 10 归一化 `vdiv(svOut, svOut, newSum)` 变成 0/0=NaN,NaN 随 output/lastSum 写回并污染同一 query 后续所有 tile 的合并链,最终表现为长序列输出乱码(P0)。

**判定条件**：`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxUpdate`(调用点:`csrc/kernels/flash_mla_v2.h` 的 `RunAiv`,关键字 `RunAivSoftmaxUpdate`)。Step 10 中 `newSum` 经 `vbrcb` 广播到 `newSumOutUb` 后,以标量读出首元素 `newSumScalar`,判定 `zeroSum = (newSumScalar == 0.0f) || (newSumScalar != newSumScalar)`——newSum 为 0 或 NaN(自反比较不成立)均视为 all-miss。

**修复**：`zeroSum` 命中时跳过 `vdiv`,`vector_dup` 将 `svOutFloatUb` 与 `newSumOutUb` 直接置 0 再走正常写回——output 清零、`lastSum=0`、`lastMax` 保留,下一个命中 topk 的 tile 从干净的 `lastSum=0` 重新起步,不再继承 NaN(任何后续 tile 一旦有 token 入选 topk,newSum 即非 0,归一化恢复正常)。

**同步链**:`newSumOutUb` 在 PIPE_V 上写入,标量读 `newSumScalar` 在 PIPE_S 上执行,跨 pipe 读必须显式同步——写读两侧用 `set_flag/wait_flag(PIPE_V, PIPE_S, EVENT_ID6)` 与 `set_flag/wait_flag(PIPE_S, PIPE_V, EVENT_ID6)` 夹住(见 `csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxUpdate`,函数头注释 `EVENT_ID6: V<->S sync around the newSum scalar read`);漏掉该链时标量读可能取到旧值,zeroSum 判定失真。

**topk 缓冲上限**:`csrc/kernels/softmax_attn_aiv.h` 顶部的 UB 预算 static_assert 全部改按 `MAX_TOPK_RUN_NUM` 校验(原为 `MAX_TOPK_NUM`)。两者语义不同(`csrc/kernels/kernel_param.h`):`MAX_TOPK_NUM=2048` 是 DSA 常驻 topk 缓冲的编译期上限;运行期 topk 行宽可超出——GLM5 kpool 路径把 PKI causal tail 打包进 topk 缓冲的 `[topK, topK+kpool-1)` 列(如 2048+3=2051),故运行期上限放宽为 `MAX_TOPK_RUN_NUM = MAX_TOPK_NUM + 8 = 2056`,aiv softmax 的 UB workspace 按运行期 topK 缩放,host 校验同步改为 `topK > MAX_TOPK_RUN_NUM` 抛错(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`)。

### 跨核 RingSync

同一 query 块的连续 KV tile 轮转落在不同核,update 读前一 tile 的 output/lastMax/lastSum 前必须 `ringSync.WaitPrevCore()`,非最后 tile 完成后 `SetNextCore()`,kernel 尾部 `PipeBarrier<PIPE_ALL>()` + `ResetPrevCore()` 复位(`csrc/kernels/flash_mla_v2.h` 的 `RunAiv`,RingSync 调用点关键字 `ringSync.WaitPrevCore`/`SetNextCore`/`ResetPrevCore`;实现 `csrc/kernels/ring_sync.h`)。

### 关键代码位置

- 主类与流水:`csrc/kernels/flash_mla_v2.h` 的 `FLASHMLAV2::RunAic` / `FLASHMLAV2::RunAiv`
- 吸收式 QK/SV:`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK` / `MlaAicHelper::RunAicSV`(与 mla_v2 共用)
- tile softmax(ping-pong,含 top-k):`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`
- online softmax update:`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxUpdate`
- RingSync:`csrc/kernels/ring_sync.h`
- host launch:`csrc/op.cpp` 的 `XliteOpFlashMLAV2`、workspace `csrc/_C.cpp` 的 `MLAV2`(`enableFlashAttention=true` 分支)、路由 `csrc/model.cpp` 的 `ForwardAttnMLAV2`(flash 路由分支)

### 片上缓冲分配与占用汇总(AIC L1/L0A/L0B/L0C、AIV UB)

AIC 侧与 mla_v2 完全共用 `MlaAicHelper` 的 12 个 L1/L0 缓冲(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::Init`,flash 的 `Init` 传入同一组几何参数,仅 qkStride 换成 tileSizeOfCachedKV,不影响布局):QK 组与 SV 组各自在 L1(`TPosition::A1`)/L0A(`A2`)/L0B(`B2`)/L0C(`CO1`)上从偏移 0 累进 `bufferAddr`,L0 三级物理独立;**QK 组与 SV 组复用同一物理空间**,两相位在核内经 set/wait_flag 严格串行(KV tile 上先 `RunAicQK` 后 `RunAicSV`),各存储级物理占用按两相位取 max;`Init` 用 `zeroGm` 把两份 `aktl1bBuf` 预填零。AIV 侧 UB 由 softmax 函数在每次调用入口自偏移 0 现场排布,tile softmax(`RunAivSoftmaxPingPong`,saveMaxSum=true)与 online update(`RunAivSoftmaxUpdate`)两个布局串行复用同一 UB,占用取 max。硬件容量(910B/dav C220,NpuArch 2201,CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`):L1 512KB、L0A 64KB、L0B 64KB,L0C 总容量 128KB（`l0_c_size=131072`，16 bank × 8KB），Mmad 累加缓冲即 `TPosition::CO1` 且占满整个 L0C;AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),每个 AIV subblock 各持一份独立 UB、布局相同。

dtype 记法:flash_mla_v2 仅 BF16 单路径(`csrc/kernels/flash_mla_v2_bfloat16_t.cpp`),`Dtype=bfloat16_t`(2B);几何常量(`MlaAicHelper::Init`)为 `qkk0=256/sizeof(Dtype)=128`、`qkn0=dense?MAX_N0:blockSize`、`svn0=256`、`svk0=64`。典型配置:kvLoraRank=512、ropeHeadDim=64、blockSize=128(AIC 缓冲大小只依赖 `XLITE_MAX_M0`/kvLoraRank/ropeHeadDim,与 nHeads 无关);AIV 侧按 tileSizeOfCachedKV=8192(Python 接口默认值,亦为 host 上限内常用档)与 DSA topK=2048 代入,update 的 headSize=kvLoraRank=512;下表 shape 为逻辑形状(L1/L0 内实际按 Nd2Nz/NZ 分形排布)。

**AIC L1(A1,容量 512KB;QK/SV 两相位自偏移 0 复用)**

| 变量 | 相位 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|---|
| `aqnl1aBuf` | QK | `(XLITE_MAX_M0, kvLoraRank)`=(128,512) | Dtype | 1(事件0) | `XLITE_MAX_M0*kvLoraRank*sizeof(Dtype)` | 131072 |
| `aqrl1aBuf` | QK | `(XLITE_MAX_M0, ropeHeadDim)`=(128,64) | Dtype | 1(事件0) | `XLITE_MAX_M0*ropeHeadDim*sizeof(Dtype)` | 16384 |
| `akl1bBuf` | QK | `(qkn0, 4*qkk0)`=(128,512) | Dtype | 2(事件4/5) | `qkn0*4*qkk0*sizeof(Dtype)` | 262144 |
| `akrl1bBuf` | QK | `(qkn0, ropeHeadDim)`=(128,64) | Dtype | 2(事件2/3) | `qkn0*ropeHeadDim*sizeof(Dtype)` | 32768 |
| **QK 相位小计** | | | | | | **442368** |
| `aqkl1aBuf` | SV | `(XLITE_MAX_M0, 4*svk0)`=(128,256) | Dtype | 2(事件0/1) | `XLITE_MAX_M0*4*svk0*sizeof(Dtype)` | 131072 |
| `aktl1bBuf` | SV | `(svn0, 2*svk0)`=(256,128) | Dtype | 2(事件4/5) | `svn0*2*svk0*sizeof(Dtype)` | 131072 |
| **SV 相位小计** | | | | | | **262144** |
| **合计(峰值 = max(两相位))** | | | | | | **442368** |

**AIC L0A(A2,容量 64KB;两相位复用)**

| 变量 | 相位 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|---|
| `qkl0aBuf` | QK | `(XLITE_MAX_M0, qkk0)`=(128,128) | Dtype | 2(事件0/1) | `XLITE_MAX_M0*qkk0*sizeof(Dtype)` | 65536 |
| **QK 相位小计** | | | | | | **65536** |
| `svl0aBuf` | SV | `(XLITE_MAX_M0, svk0)`=(128,64) | Dtype | 2(事件0/1) | `XLITE_MAX_M0*svk0*sizeof(Dtype)` | 32768 |
| **SV 相位小计** | | | | | | **32768** |
| **合计(峰值 = max(两相位))** | | | | | | **65536** |

**AIC L0B(B2,容量 64KB;两相位复用)**

| 变量 | 相位 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|---|
| `qkl0bBuf` | QK | `(MAX_N0, qkk0)`=(128,128) | Dtype | 2(事件0/1) | `MAX_N0*qkk0*sizeof(Dtype)` | 65536 |
| **QK 相位小计** | | | | | | **65536** |
| `svl0bBuf` | SV | `(svn0, svk0)`=(256,64) | Dtype | 2(事件0/1) | `svn0*svk0*sizeof(Dtype)` | 32768 |
| **SV 相位小计** | | | | | | **32768** |
| **合计(峰值 = max(两相位))** | | | | | | **65536** |

**AIC L0C(CO1 区,容量 128KB;两相位复用)**

| 变量 | 相位 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|---|
| `qkl0cBuf` | QK | `(XLITE_MAX_M0, MAX_N0)`=(128,128) | float | 2(事件0/1) | `XLITE_MAX_M0*MAX_N0*sizeof(float)` | 131072 |
| **QK 相位小计** | | | | | | **131072** |
| `svl0cBuf` | SV | `(XLITE_MAX_M0, svn0)`=(128,256) | float | 1(事件0) | `XLITE_MAX_M0*svn0*sizeof(float)` | 131072 |
| **SV 相位小计** | | | | | | **131072** |
| **合计(峰值 = max(两相位))** | | | | | | **131072** |

**AIV UB——tile softmax(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`;按 len=MAX(outN, topK)=8192、topK=2048 代入,预算 192KB)**

flash 的 max/sum workspace 非空(saveMaxSum=true),`maxOut`/`sumOut` 各 2 份乒乓槽;swaSegWidth=0、attnSink 为空,相应槽位为 0;topK=0(未启用 DSA)时下表 topK 相关 9 行不分配,布局降为 115712 B。

| 变量 | shape(每份) | dtype | 份数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `in1`/`in2` | `(len,)`=(8192,) | Dtype | 2(事件2/3) | `ROUND_UP(len*sizeof(Dtype),256)` | 32768 |
| `out1`/`out2` | `(len,)` | Dtype | 2(事件2/3) | `ROUND_UP(len*sizeof(Dtype),256)` | 32768 |
| `cal` | `(len,)` | float | 1 | `ROUND_UP(len*sizeof(float),256)` | 32768 |
| `temp` | len/2 归约暂存 | float | 1 | `(DIV_ROUND_UP(len*sizeof(float),256)/2)*256` | 16384 |
| `maxOut0`/`maxOut1` | 1 个 256B 槽 | float | 2(事件2/3) | `VECTOR_MAX_BYTESIZE` | 512 |
| `sumOut0`/`sumOut1` | 1 个 256B 槽 | float | 2(事件2/3) | `VECTOR_MAX_BYTESIZE` | 512 |
| `indices0`/`indices1` | `(topK,)`=(2048,) | int32 | 2(事件0/1) | `ROUND_UP(topK*sizeof(int32_t),256)` | 16384 |
| `inTopK` | `(topK,)` | Dtype | 1 | `ROUND_UP(topK*sizeof(Dtype),256)` | 4096 |
| `indicesFp32` | `(topK,)` | float | 1 | `ROUND_UP(topK*sizeof(float),256)` | 8192 |
| `cmp0` | 2 个 256B repeat | int32 | 1 | `VECTOR_MAX_BYTESIZE*2` | 512 |
| `cmp1` | 1 个 256B repeat | int32 | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `cmpFp320` | 1 个 256B repeat | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `cmpFp321` | 1 个 256B repeat | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `topkMask0`/`topkMask1` | `(topK/8,)`=(256,) | uint8 | 2 | `ROUND_UP(topK/8,256)` | 512 |
| `currIndices` | `(topK,)` | int32 | 1 | `ROUND_UP(topK*sizeof(int32_t),256)` | 8192 |
| **合计** | | | | | **154368** |

**AIV UB——online softmax update(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxUpdate`;headSize=kvLoraRank=512)**

| 变量 | shape(每份) | dtype | 份数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `svPrevUb` | `(headSize,)`=(512,) | Dtype | 2(事件0/1) | `ROUND_UP(headSize*sizeof(Dtype),256)` | 2048 |
| `svCurrUb` | `(headSize,)` | Dtype | 2(事件0/1) | `ROUND_UP(headSize*sizeof(Dtype),256)` | 2048 |
| `svOutUb` | `(headSize,)` | Dtype | 2(事件2/3) | `ROUND_UP(headSize*sizeof(Dtype),256)` | 2048 |
| `maxPrevUb` | 1 个 256B 槽 | float | 2(事件0/1) | `VECTOR_MAX_BYTESIZE` | 512 |
| `sumPrevUb` | 1 个 256B 槽 | float | 2(事件0/1) | `VECTOR_MAX_BYTESIZE` | 512 |
| `maxCurrUb` | 1 个 256B 槽 | float | 2(事件0/1) | `VECTOR_MAX_BYTESIZE` | 512 |
| `sumCurrUb` | 1 个 256B 槽 | float | 2(事件0/1) | `VECTOR_MAX_BYTESIZE` | 512 |
| `newSumOutUb` | 1 个 256B 槽 | float | 2(事件2/3) | `VECTOR_MAX_BYTESIZE` | 512 |
| `newMaxOutUb` | 1 个 256B 槽 | float | 2(事件2/3) | `VECTOR_MAX_BYTESIZE` | 512 |
| `scalePrevUb` | 1 个 256B 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `scaleCurrUb` | 1 个 256B 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `tempUb` | 1 个 256B 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `newSumUb` | 1 个 256B 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `newMaxUb` | 1 个 256B 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `sumPrevCalUb` | 1 个 256B 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `sumCurrCalUb` | 1 个 256B 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `svPrevFloatUb` | `(headSize,)` | float | 1 | `ROUND_UP(headSize*sizeof(float),256)` | 2048 |
| `svCurrFloatUb` | `(headSize,)` | float | 1 | `ROUND_UP(headSize*sizeof(float),256)` | 2048 |
| `svOutFloatUb` | `(headSize,)` | float | 1 | `ROUND_UP(headSize*sizeof(float),256)` | 2048 |
| **合计** | | | | | **17152** |

**占用汇总**(各存储级按两相位/两布局取峰值 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用(B) | 占比 |
|---|---|---|---|
| L1 | 512 KB | 442368 | 84.4% |
| L0A | 64 KB | 65536 | 100% |
| L0B | 64 KB | 65536 | 100% |
| L0C(CO1) | 128 KB | 131072 | 100% |
| UB(每 AIV subblock,峰值取 tile softmax 布局) | 192 KB | 154368 | 78.5% |

要点:

- **L0 三级峰值全部顶满(100%)**:BF16 下 `qkk0=128`、`XLITE_MAX_M0=MAX_N0=128` 使 QK 相位 L0A/L0B 各恰 64KB、两相位 L0C(CO1)各恰 128KB——M/N/K 任一维度增大都会先撞 L0;host 侧 `nHeads <= XLITE_MAX_M0` 校验(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`,关键字 `nHeads > XLITE_MAX_M0`)保证单 token 全部头落进一个 M0 tile;
- AIV UB 峰值在 tile softmax 布局(154368 B,78.5%),大头是随 len 缩放的 in/out/cal/temp 四类(合计 114688 B);online update 布局仅 17152 B,与 tile softmax 串行复用同一 UB,不叠加;
- 容量来源是 host 校验 `tileSizeOfCachedKV <= MAX_SOFTMAX_PINGPONG_LEN=10496` 与 `topK <= MAX_TOPK_NUM=2048`(`csrc/op.cpp` 的 `XliteOpFlashMLAV2`)。按该上限组合(len=10496、topK=2048)代入 tile softmax 布局为 186624 B ≤ `UB_SIZE`(余 9984 B;旧上限 11776 的同组合实占 204544 B 越界,旧 static_assert 公式漏计 `indicesFp32` 槽位 8192 B 而误判通过——已随上限收紧到 10496 并补全断言公式一并修复);典型 tileSizeOfCachedKV=8192 时余量 42240 B。
