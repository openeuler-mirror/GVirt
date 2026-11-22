# mla_v2

## 功能概述

MLA(Multi-head Latent Attention)吸收式(weight-absorbed)decode/prefill 注意力算子:Q 侧拆为 `qAbsorb = q_nope · WUK`(kvLoraRank 维,吸收了上投影)与 `qr`(RoPE 部分),K 侧拆为 latent `kCache`(kvLoraRank 维)与 `peCache`(RoPE 后的 key)。kernel 计算

```text
QK = (qAbsorb · kCache^T + qr · peCache^T) * scale
oAbsorb = softmax_causal(QK) · kCache
```

即先算两个吸收式矩阵乘之和再做 softmax,然后用同一份 latent K 左乘还原输出(oAbsorb 随后由 host/WUV 投影到 v_head_dim)。单 kernel 内统一两种 KV 布局(`dense` 标志,见 `csrc/kernels/mla_v2.h` 的 `MLAV2` 类注释,关键字 `dense`/`sparse`;原 mla_v3 的 dense 路径已合并进来,独立 mla_v3 算子已删除):

- **sparse(dense=0,分页)**:经典 mla_v2 路径,KV 走 blockTables 分页寻址,可选 DSA top-k token 选择(topkIndices);
- **dense(dense=1,连续)**:每 batch 起始于 dense cache 的 `b × dim-0 跨度` 处(连续布局时即占据 `[b*maxSeqLen, (b+1)*maxSeqLen)`,由 gather_sparse_kv_cache 收集或展开),blockTable 不使用。Top-K 选择独立于缓存布局；已收集所选 token 的 Decode 路径传 `topK=0`。

该算子是"短序列"路径:总 KV 长 `ceil(maxTotalLens/blockSize) * blockSize ≤ tileSizeOfCachedKV`(即 `blockTables.shape[1] * blockSize`,运行期行宽,不取 block table 的分配跨度)时模型层选用(`csrc/model.cpp` 的 `ForwardAttnMLAV2`,短序列分支关键字 `<= rt._tileSizeOfCachedKV`),更长则走 flash_mla_v2;decode+DSA 长序列另行路由到 dense 路径(见"关键代码位置")。

## 输入输出参数

Python 侧调用(`tests/kernels/mla.py` 的 `mla_v2(rt, qWithQr_xlite, ...)` 调用点,关键字 `mla_v2(rt, qWithQr_xlite`;dense 用例在同文件的 `mla_v2 dense` 分支):

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
    query_start_loc,
    query_lens,
    cached_lens,
    block_tables,
    n_heads,
    rope_head_dim,
    nope_head_dim,
    v_head_dim,
    kv_lora_rank,
    BLOCK_SIZE,
    batch,
    scale,
    topk_indices,
    top_k,
    weight_nz,
    enable_flash_attention=False,
    tile_size_of_cached_kv=8192,
    dense=False,
)
```

调用块中的 `weight_nz` 即 Python 绑定的 `nz` 关键字参数(测试以局部变量 `nz=weight_nz` 传入):该参数已废弃并将被移除,传入值被忽略(`csrc/_C.cpp` 的 `MLAV2`,关键字 `(void)weightNz` 的 deprecated no-op 注释),`wuk_t`/`wuv` 的 FRACTAL_NZ 存储格式由 `InitXTensor` 自动检测。

host 封装 `MLAV2`(`csrc/_C.cpp` 的 `MLAV2`)的 `qAbsorb = einsum(q_nope, wukT)` 与 `output = einsum(oAbsorb, wuv)` 在 paged / dense 两条路径**均无条件执行**(对应 `XliteOpEinsumMhtHtdMhd` 分别位于 `if (dense)` 之前与所有分支之后);两条路径唯一区别是 dense(`dense=True`)在吸收前多一步 `gather_sparse_kv_cache` 收集 top-k token 到连续 `kDense`/`peDense`(见 `csrc/_C.cpp` 的 `MLAV2` `dense` 分支,关键字 `XliteOpGatherSparseKVCache`),随后以 `topK` 为 `maxSeqLen`、内核 `topK=0` 调 `XliteOpMLAV2`,输入/输出契约与 paged 相同。接口完整 docstring 见 `xlite/_C.pyi` 的 `mla_v2`(注意 Python API 不含 `max_seq_len`,host 取 `maxSeqLen = block_tables.shape[1] * blockSize`(`shape[1]` 为运行期宽度:V2 持久表由 PrepareAttn 收窄,standalone 由 lens 推导收窄),再传给 kernel)。kernel 签名(`csrc/kernels/mla_v2.h` 的 `mla_v2_<dtype>` 入口,宏 `MLA_V2_FUNC_DEFINE`):

```cpp
mla_v2_<dtype>(qAbsorb, qr, kCache, peCache, topkIndices, qk, oAbsorb,
               queryStartLoc, queryLens, cachedLens, blockTables, nHeads, ropeHeadDim,
               kvLoraRank, blockSize, batch, blockTableStride, maxSeqLen, scale, topK, kCacheStride,
               peCacheStride, dense, densePerQuery, zeroGm)
```

末尾的 `zeroGm` 是 host 自动提供的零缓冲(`rt._attnZeroBuffer`,host launch 见 `csrc/op.cpp` 的 `XliteOpMLAV2`),`MlaAicHelper::Init` 用它把两份 L1B ping-pong 缓冲预填零,避免上一 tile 残留 K 数据导致 mmad NaN(与 attention 同源 nan bug 修复)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| qAbsorb | 输入 | `[total_tokens, nHeads, kvLoraRank]` | BF16 | 吸收 WUK 后的 Q latent 部分 |
| qr | 输入 | `[total_tokens, nHeads, ropeHeadDim]` | BF16 | Q 的 RoPE 部分 |
| kCache | 输入 | sparse: `[numBlocks, blockSize, kvLoraRank]`;dense: `[batch, maxSeqLen, kvLoraRank]` | BF16 | latent K cache |
| peCache | 输入 | sparse: `[numBlocks, blockSize, ropeHeadDim]`;dense: `[batch, maxSeqLen, ropeHeadDim]` | BF16 | RoPE 后的 K cache |
| topkIndices | 输入(可选) | `[total_tokens, topK]` | INT32 | DSA 每行 query 的 top-k token 下标(升序),topK>0 时使用，与缓存布局无关 |
| qk | workspace | `[aicNum * XLITE_MAX_M0 * 2, maxSeqLen]` | BF16 | QK 分数缓冲;sparse 或 C8 完整展开时 `maxSeqLen = blockTables.shape[1] * blockSize`(运行期行宽,`shape[1]` 已由 PrepareAttn 收窄),Gather Decode 时 `maxSeqLen = indexTopK`(见 `csrc/_C.cpp` 的 `MLAV2` dense 分支,关键字 `rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, topK}`) |
| oAbsorb | 输出 | `[total_tokens, nHeads, kvLoraRank]` | BF16 | softmax(QK)·kCache |
| queryStartLoc / queryLens / cachedLens | 输入 | `[batch]` | INT32 | 同 attention |
| blockTables | 输入 | `[rows >= batch, blockTableStride]` | INT32 | sparse 用;dense 不使用，可传空。持久化缓冲,行跨度为分配跨度 `blockTables.stride[0]`,行宽为每步运行期宽度 `shape[1]`,仅每行 lens 推导的有效前缀被读取 |
| nHeads | 标量 | - | uint32 | 本卡头数(测试 1/8/16) |
| ropeHeadDim | 标量 | - | uint32 | RoPE 维(64) |
| kvLoraRank | 标量 | - | uint32 | latent 秩(16/512) |
| blockSize | 标量 | - | uint32 | KV 块大小(128);dense 模式下传入但仅经 MlaAicHelper 的 dense 标志忽略 |
| batch | 标量 | - | uint32 | batch |
| blockTableStride | 标量 | - | uint32 | blockTables 的行跨度(元素数),host 取 `blockTables.stride[0]`,仅用于 sparse 行寻址(dense 置 0) |
| maxSeqLen | 标量 | - | uint32 | qk workspace 行宽与 dense 长度:sparse 时为 `blockTables.shape[1] * blockSize`(qk workspace 行宽;V2 的 `shape[1]` 已由 PrepareAttn 收窄到运行期宽度),dense 时为连续 cache 每批容量(Gather Decode 为 indexTopK，C8 完整展开为调用方分配的容量) |
| scale | 标量 | - | float | `(nopeHeadDim + ropeHeadDim)^-0.5` |
| topK | 标量 | - | uint32 | DSA top-k;启用 Top-K 选择时，host 要求 `maxSeqLen ≤ MAX_SOFTMAX_PINGPONG_LEN=10496` 且 `topK ≤ MAX_TOPK_NUM=2048`(见 `csrc/op.cpp` 的 `XliteOpMLAV2`,关键字 `MAX_SOFTMAX_PINGPONG_LEN`/`MAX_TOPK_NUM`)。Python `dense=True` 仍要求 topK > 0，收集后以该值为 maxSeqLen、以 `topK=0` 调用内核；C8 完整展开保留原 topK 和索引 |
| kCacheStride / peCacheStride | 标量 | - | uint32 | kCache / peCache 的 dim-0 跨度(元素数),块或 batch 基址 = 序号 × 跨度:sparse 为每物理块跨度(连续时等于 blockSize×kvLoraRank / blockSize×ropeHeadDim),dense 为每 batch 跨度(连续时等于 maxSeqLen×kvLoraRank / maxSeqLen×ropeHeadDim);非连续时由宿主传入真实跨距 |
| dense | 标量 | - | uint32 | 1=dense cache(`[batch, maxSeqLen, ...]`,batch 起点 = b×kCacheStride,由 gather_sparse_kv_cache 收集,blockTable 不使用),0=paged(见 `csrc/op.cpp` 的 `XliteOpMLAV2`,关键字 `dense`) |
| densePerQuery | 标量 | - | uint32 | dense 时选择缓存行布局：1=每 query 一行，0=每请求一行；SFA-C8 完整展开使用 0 |
| zeroGm | 标量 | - | GM | host 自动提供的零缓冲(`rt._attnZeroBuffer`),非 Python API 参数;`MlaAicHelper::Init` 用它预填零 L1B(见上文) |

## 支持的数据类型

SFA-C8 在 host 接受独立的 INT8 `kCache[numBlocks, blockSize, 1, kvLoraRank]`、BF16 `peCache[numBlocks, blockSize, 1, ropeHeadDim]` 和 FP32 `kScaleCache[numBlocks, blockSize, 1, kvLoraRank/128]`。Decode dense 路径先 gather/dequantize 所选 token；其余路径先恢复各请求的临时连续 BF16 cache，不生成新页表，并保留 Top-K 选择。之后使用同一 BF16 MLA kernel，QK 和 PV 都读取恢复后的 latent。Q、输出及 `scale` 保持原约定，GLM-5.1 的 `scale=1/16`。

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| bfloat16_t | `csrc/kernels/mla_v2_bfloat16_t.cpp` | `mla_v2_bfloat16_t` |

底层 MLA kernel 仅接受 BF16(`csrc/op.cpp` 的 `XliteOpMLAV2`,关键字 `EachXDtype(BF16`);qAbsorb/qr/kCache/peCache/oAbsorb 需一致。SFA-C8 的 cache 在进入该 kernel 前恢复为 BF16。

## 实现原理

混合 AIC/AIV kernel(`KERNEL_TYPE_MIX_AIC_1_2`,`csrc/kernels/mla_v2.h` 的 `MLAV2::Init`,关键字 `KERNEL_TYPE_MIX_AIC_1_2`),三级流水 AIC-QK → AIV-softmax → AIC-SV,与 attention 同构但矩阵乘换成 MLA 的吸收式形态。

### 任务划分

- `m0 = densePerQuery ? nHeads : XliteFitM0ToHeads(GetOptimalM0(queryLen, cachedLen), nHeads)` 自适应(同 attention,`csrc/kernels/kernel_macro.h` 的 `GetOptimalM0` + `XliteFitM0ToHeads`;每 query 一行的 dense 模式直接取 nHeads,一个 tile 一 token),`queryTileSize = m0 / nHeads`——MLA 每 batch 一次算**全部 nHeads 个头**(无 KV 头分组),`mSize = queryTaskLen * nHeads` 行拼成 Cube 的 M 维。
- 任务空间为 `queryNum`(每 batch),核间用 `firstCore = (GetBlockIdx()+GetBlockNum()-coreOffset) % GetBlockNum()` 的循环 stride 分配,`coreOffset` 跨 batch 累计保证相邻 batch 起始核轮转、负载均衡(`csrc/kernels/mla_v2.h` 的 `RunAic`,关键字 `firstCore`/`coreOffset`)。
- dense 模式 `calcLen` clip 到 maxSeqLen(连续 cache 的每请求容量,见 `csrc/kernels/mla_v2.h` 的 `RunAic`,关键字 `dense && calcLen > maxSeqLen`);per-batch KV 视图 dense 时按 dim-0 跨度取 `kCache[batchIdx * kCacheStride]` 子视图,sparse 时走 blockTable(`csrc/kernels/mla_v2.h` 的 `RunAic`,关键字 `batchKCache = kCache[batchIdx * kCacheStride]`)。

### AIC 侧:MlaAicHelper 的吸收式双 GEMM

`MlaAicHelper`(`csrc/kernels/mla_aic_helper.h`)为 mla_v2/flash_mla_v2 共用,几何常量:`qkn0 = dense ? 128 : blockSize`(N 维 tile,dense 时块号即逻辑索引、N 维不受块大小约束)、`qkk0 = 256/sizeof(Dtype)`(K 维 tile)、`svn0 = 256`、`svk0 = 64`,L1/L0 共 12 个 ping-pong 缓冲在 `Init()`(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::Init`)静态布局并返回 svk0。dense 标志与 kCacheStride/peCacheStride 由 mla_v2 的 Init 传入(`csrc/kernels/mla_v2.h` 的 `MLAV2::Init`,关键字 `aicHelper.Init`),qkStride 即 maxSeqLen(`csrc/kernels/mla_v2.h` 的 `MLAV2::Init`,关键字 `this->qkStride`)。

**RunAicQK**(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK`):`QK = (qAbsorb·kCache^T + qr·peCache^T)`,m=queryTokens*nHeads, n=kvLen, 分两段:

1. 沿 kvLoraRank 按 `qkk0`(bf16 为 128)分 k 块,K 每次预取 `4*qkk0` 进 L1B(4 次复用),L0A/L0B ping-pong,`CalMmad(..., kIdx==0)` 控制首块清零累加出 C = qAbsorb·kCache;
2. 同一 n tile 上再算 R = qr·peCache(ropeHeadDim 一趟),`CalMmad(..., false)` 直接累加到同一 L0C,天然完成 `C+R`(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK`,关键字 `CalMmad(qkl0cBuf` 第二段 `false`);
3. 每个 n tile 结束 `CopyToGm` 写 qk(行距 qkStride=maxSeqLen),L0C 双缓冲与下一 tile 的 mmad 重叠。

分页寻址:`block = blockTable[nIdx + nIdxStart]`(sparse)或 token 偏移 `kvOffset + nIdx * qkn0`(dense,见 `csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK`,关键字 `!dense ? blockTable`);块基址 K 侧取 `block * kPitch`、PE 侧取 `block * pePitch`——sparse 时即传入的 dim-0 跨度 kCacheStride/peCacheStride,dense 时为 `kvLoraRank` / `ropeHeadDim`(batch 段内连续),拷贝形式相同、仅寻址不同。

**RunAicSV**(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicSV`):`oAbsorb = QK · kCache^T`,m=queryTokens*nHeads, n=kvLoraRank(按 svn0=256 分块), k=kvLen(按 svk0=64 分块)。QK 矩阵每 4*svk0 列预取进 L1A;K^T 侧每 2*svk0 预取进 L1B——dense 时单条 `CopyGmToL1Nd2Nz` 连续拷贝,sparse 时按物理块循环逐块拷(块基址 `block * kCacheStride`,见 `MlaAicHelper::RunAicSV`,关键字 `paged cache: per-block lookup`);`CopyToL0BTCol` 转置进 L0B。输出 `CopyToGm(out[nOffset], ..., kvLoraRank)` 按 kvLoraRank 行距写 oAbsorb。

### AIV 侧:softmax 与 top-k 路径

- **topK == 0**:`RunAivSoftmax`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmax`,转发至 `RunAivSoftmaxLong`)(见 attention 文档),带 `hasScale=true, scale`(QK 乘 scale 在 softmax 内完成)、causal 由 `calcSoftmaxLen = cachedLen + queryTaskStart + 1` + maskOff/headStride=nHeads 实现(`csrc/kernels/mla_v2.h` 的 `RunAiv`,关键字 `calcSoftmaxLen`/`RunAivSoftmax`)。
- **topK > 0**(DSA，与缓存布局无关):`RunAivSoftmaxPingPong`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`)。该路径先用 `vcmpvs_ge/vcmpvs_lt + vand` 生成命中位图,`vgather` 按 topkIndices 从 QK 行中收集 topK 个分数(未命中的槽位填 -3.4e38),对收集后的 topK 长度做 softmax,再用标量 scatter 按 64 槽粒度把概率散回原 token 位置、其余位置清零(均见 `RunAivSoftmaxPingPong`,关键字 `vcmpvs_ge`/`vgather` 与 scatter 段)。行有效长度 `calcLen > topK` 时才启用,否则退化为全量(`csrc/kernels/mla_v2.h` 的 `RunAiv`,关键字 `calcLen > topK`)。
- **outN 对齐**:softmax 输出行宽取 `ROUND_UP(calcLen, 4*svk0)`(clip 到 maxSeqLen),因为 RunAicSV 每次读 4*svk0=256 列 QK,残留脏数据会沿 KV 维混进 SV 累加(`csrc/kernels/mla_v2.h` 的 `RunAiv`,关键字 `4 * svk0`)。

### 跨核同步

与 attention 相同:QK 完成 `ffts_cross_core_sync(PIPE_FIX, flag0)` → AIV softmax(flag1)→ AIC `wait_flag_dev(1)` 后 SV;相邻任务错峰,末尾补做(`csrc/kernels/mla_v2.h` 的 `RunAic`(关键字 `ffts_cross_core_sync`)与 `RunAiv`(关键字 `ffts_cross_core_sync`))。

### 关键代码位置

- 主类/双模式说明:`csrc/kernels/mla_v2.h` 的 `MLAV2` 类注释(关键字 `dense`/`sparse`)、`MLAV2::RunAic`、`MLAV2::RunAiv`
- QK/SV 吸收式 GEMM:`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK` / `MlaAicHelper::RunAicSV` / `MlaAicHelper::Init`(缓冲布局)
- softmax 全量/top-k:`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmax` / `RunAivSoftmaxPingPong`(其内部实现 `RunAivSoftmaxLong` 新增 SWA 段/压缩比/attnSink 参数为 cxa 算子服务,mla_v2 调用时均取默认值、行为不变)
- host launch 与限制:`csrc/op.cpp` 的 `XliteOpMLAV2`(dense 校验关键字 `MAX_TOPK_NUM`);路由 `csrc/model.cpp` 的 `ForwardAttnMLAV2`(短序列分支)与 dense gather 路径(decode+DSA 长序列时启用,关键字 `XliteOpGatherSparseKVCache`,阈值 `XLITE_MLA_DENSE_THRESHOLD=280`,见 `csrc/model.cpp` 的 `XLITE_MLA_DENSE_THRESHOLD`)
- 常量:`csrc/kernels/kernel_param.h`(`XLITE_MAX_M0`/`MAX_TOPK_NUM`/`MAX_SOFTMAX_PINGPONG_LEN`)

### 片上缓冲分配与占用汇总(AIC L1/L0A/L0B/L0C、AIV UB)

AIC 侧 12 个 L1/L0 缓冲由共用的 `MlaAicHelper::Init`(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::Init`)手动排布:QK 组与 SV 组各自在 L1(`TPosition::A1`)/L0A(`A2`)/L0B(`B2`)/L0C(`CO1`)上从偏移 0 累进 `bufferAddr`,L0 三级物理独立、各自从偏移 0 重新起排;**QK 组与 SV 组复用同一物理空间**——`RunAicQK` 收尾等待 `FIX_M`/`M_MTE1`/`MTE1_MTE2` 全部排空后才切换 `RunAicSV`,两相位在核内严格串行,各存储级物理占用按两相位取 max;`Init` 还用 host 提供的 `zeroGm` 把两份 `aktl1bBuf` 预填零(见上文 nan bug 修复)。AIV 侧 UB 由 softmax 函数在每次调用入口自偏移 0 现场排布,topK=0(全量 `RunAivSoftmaxLong`)与 topK>0(`RunAivSoftmaxPingPong`)两条路径互斥,占用亦取 max。硬件容量(910B/dav C220,NpuArch 2201,CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`):L1 512KB、L0A 64KB、L0B 64KB,L0C 总容量 128KB（`l0_c_size=131072`，16 bank × 8KB），Mmad 累加缓冲即 `TPosition::CO1` 且占满整个 L0C;AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),每个 AIV subblock 各持一份独立 UB、布局相同。

dtype 记法:mla_v2 仅 BF16 单路径(`csrc/kernels/mla_v2_bfloat16_t.cpp`),`Dtype=bfloat16_t`(2B);几何常量(`MlaAicHelper::Init`)为 `qkk0=256/sizeof(Dtype)=128`、`qkn0=dense?MAX_N0:blockSize`、`svn0=256`、`svk0=64`。典型配置取 DeepSeek-V3 档(`tests/kernels/mla.py` 的 `models` 表):nHeads=16、kvLoraRank=512、ropeHeadDim=64、blockSize=128——AIC 缓冲大小只依赖 `XLITE_MAX_M0`/kvLoraRank/ropeHeadDim,与 nHeads 无关(dense 时 qkn0=MAX_N0=128,与 blockSize=128 同值);AIV 侧按短序列路径上限 maxSeqLen=8192(`tile_size_of_cached_kv` 默认值,超过即路由 flash_mla_v2)与 DSA topK=2048 代入,下表 shape 为逻辑形状(L1/L0 内实际按 Nd2Nz/NZ 分形排布)。

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
| `svl0bBuf` | SV | `(svn0, svk0)`=(256,64) | Dtype | 2(事件0/1) | `svn0*svk0*sizeof(Dtype)` | 65536 |
| **SV 相位小计** | | | | | | **65536** |
| **合计(峰值 = max(两相位))** | | | | | | **65536** |

**AIC L0C(CO1 区,容量 128KB;两相位复用)**

| 变量 | 相位 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|---|
| `qkl0cBuf` | QK | `(XLITE_MAX_M0, MAX_N0)`=(128,128) | float | 2(事件0/1) | `XLITE_MAX_M0*MAX_N0*sizeof(float)` | 131072 |
| **QK 相位小计** | | | | | | **131072** |
| `svl0cBuf` | SV | `(XLITE_MAX_M0, svn0)`=(128,256) | float | 1(事件0) | `XLITE_MAX_M0*svn0*sizeof(float)` | 131072 |
| **SV 相位小计** | | | | | | **131072** |
| **合计(峰值 = max(两相位))** | | | | | | **131072** |

**AIV UB——全量 softmax 路径(topK=0,`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`;每 subblock 一份,预算 192KB)**

in/out/calc/calcDst 按 `MAX_SUB_CONTEXT_SIZE=(VECTOR_MAX_REPEAT-1)*64=16256` 定长(与 outN 无关),仅 max/sum 随 `totalSubBlockNum=DIV_ROUND_UP(outN, 16256)` 缩放;mla_v2 的 outN ≤ maxSeqLen ≤ 8192 时 totalSubBlockNum 恒为 1;`sink` 槽为 `attnSink != nullptr` 条件分配,mla_v2 恒取默认 nullptr、占用为 0,不入表。

| 变量 | shape(每份) | dtype | 份数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `in` | 16256(`MAX_SUB_CONTEXT_SIZE`) | Dtype | 1 | `MAX_SUB_CONTEXT_SIZE*sizeof(Dtype)` | 32512 |
| `out` | 16256 | Dtype | 1 | `MAX_SUB_CONTEXT_SIZE*sizeof(Dtype)` | 32512 |
| `calc` | 16256 | float | 1 | `MAX_SUB_CONTEXT_SIZE*sizeof(float)` | 65024 |
| `calcDst` | 127 个 256B repeat 槽 | float | 1 | `(VECTOR_MAX_REPEAT-1)/2*VECTOR_MAX_BYTESIZE` | 32512 |
| `max` | totalSubBlockNum+1=2 个 256B 槽 | float | 1 | `(totalSubBlockNum+1)*VECTOR_MAX_BYTESIZE` | 512 |
| `sum` | 同上 | float | 1 | `(totalSubBlockNum+1)*VECTOR_MAX_BYTESIZE` | 512 |
| **合计** | | | | | **163584** |

**AIV UB——top-k softmax 路径(topK>0,`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`;按 len=MAX(outN, topK)=8192、topK=2048 代入)**

mla_v2 的 maxBuf/sumBuf 传 nullptr(saveMaxSum=false),`maxOut`/`sumOut` 槽位不分配;swaSegWidth=0、attnSink 为空,相应槽位亦为 0。

| 变量 | shape(每份) | dtype | 份数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `in1`/`in2` | `(len,)`=(8192,) | Dtype | 2(事件2/3) | `ROUND_UP(len*sizeof(Dtype),256)` | 32768 |
| `out1`/`out2` | `(len,)` | Dtype | 2(事件2/3) | `ROUND_UP(len*sizeof(Dtype),256)` | 32768 |
| `cal` | `(len,)` | float | 1 | `ROUND_UP(len*sizeof(float),256)` | 32768 |
| `temp` | len/2 归约暂存 | float | 1 | `(DIV_ROUND_UP(len*sizeof(float),256)/2)*256` | 16384 |
| `indices0`/`indices1` | `(topK,)`=(2048,) | int32 | 2(事件0) | `ROUND_UP(topK*sizeof(int32_t),256)` | 16384 |
| `inTopK` | `(topK,)` | Dtype | 1 | `ROUND_UP(topK*sizeof(Dtype),256)` | 4096 |
| `indicesFp32` | `(topK,)` | float | 1 | `ROUND_UP(topK*sizeof(float),256)` | 8192 |
| `cmp0` | 2 个 256B repeat | int32 | 1 | `VECTOR_MAX_BYTESIZE*2` | 512 |
| `cmp1` | 1 个 256B repeat | int32 | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `cmpFp320` | 1 个 256B repeat | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `cmpFp321` | 1 个 256B repeat | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `topkMask0`/`topkMask1` | `(topK/8,)`=(256,) | uint8 | 2 | `ROUND_UP(topK/8,256)` | 512 |
| `currIndices` | `(topK,)` | int32 | 1 | `ROUND_UP(topK*sizeof(int32_t),256)` | 8192 |
| **合计** | | | | | **153344** |

**占用汇总**(各存储级按两相位/两路径取峰值 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用(B) | 占比 |
|---|---|---|---|
| L1 | 512 KB | 442368 | 84.4% |
| L0A | 64 KB | 65536 | 100% |
| L0B | 64 KB | 65536 | 100% |
| L0C(CO1) | 128 KB | 131072 | 100% |
| UB(每 AIV subblock,峰值取全量 softmax 布局) | 192 KB | 163584 | 83.2% |

要点:

- **L0 三级峰值全部顶满(100%)**:BF16 下 `qkk0=128`、`XLITE_MAX_M0=MAX_N0=128` 使 QK 相位 L0A/L0B 各恰 64KB、SV 相位 L0B 亦恰满 64KB(`svn0=256` 单份即 32768 B、双份顶满),两相位 L0C(CO1)各恰 128KB——M/N/K 任一维度增大都会先撞 L0;host 侧 `nHeads <= XLITE_MAX_M0` 校验(`csrc/op.cpp` 的 `XliteOpMLAV2`,关键字 `nHeads > XLITE_MAX_M0`)正是「单 token 全部头必须落进一个 M0 tile」的容量前提;
- L1 峰值在 QK 相位 442368 B(84.4%),大头是双份 K 预取 `akl1bBuf`(262144 B,占 QK 相位 59%);SV 相位仅 262144 B(50%),与 QK 相位自偏移 0 复用同一物理 L1,无需按两组求和;
- AIV UB 峰值在全量 softmax 路径:in/out/calc/calcDst 按 `MAX_SUB_CONTEXT_SIZE=16256` 定长合计 162560 B,加 max/sum 两槽共 163584 B(159.75KB,83.2%);top-k 路径 153344 B(78.0%)随运行期 len/topK 缩放;
- top-k 路径的容量来源是 host 校验 `maxSeqLen <= MAX_SOFTMAX_PINGPONG_LEN=10496` 与 `topK <= MAX_TOPK_NUM=2048`(`csrc/op.cpp` 的 `XliteOpMLAV2`)。按该上限组合(len=10496、topK=2048)代入布局为 185600 B ≤ `UB_SIZE`(余 11008 B;旧上限 11776 的同组合实占 203520 B 越界,旧 static_assert 公式漏计 `indicesFp32` 槽位 8192 B 而误判通过——已随上限收紧到 10496 并补全断言公式一并修复);mla_v2 实际服务的短序列(maxSeqLen ≤ 8192,占 153344 B)尚有 43264 B 余量。

## DSA-CP MLA

DSACP 面向 prefill，在 TP 组内按 query token 分担 Indexer 和主 Attention。
其关键设计如下：

1. **按 token 划分计算。** Indexer 和主 Attention 使用相同的 token 分片，
   每卡计算本卡 query 的全部 heads，保留各 query 所属请求的位置和因果关系。
2. **各卡保留完整 KV cache。** 各卡生成本卡 token 的 main KV 和 index K，
   通过 AllGather 同步增量并写入完整缓存，使本卡 query 能访问所属请求的有效历史。
3. **按 Top-K 组织稀疏计算。** Indexer 为每个 query 选择 KV 位置，主 Attention 读取对应的
   latent KV 和 RoPE K，完成 QK、Softmax、SV。同一 query 的全部 heads 共享选中的 KV，
   不同 query 独立选择；有效历史不超过 Top-K 时使用全部有效历史。
   Query 分组处理，复用临时空间以控制峰值显存。
4. **本卡完成输出投影。** 汇集完整 Wᴼ，对本卡 token 的全部 heads 执行投影，
   得到本卡 token 的完整 Attention 输出，衔接按 token 分片的 residual/Norm。

MLA 的计算原理见前面的[实现原理](#实现原理)。
