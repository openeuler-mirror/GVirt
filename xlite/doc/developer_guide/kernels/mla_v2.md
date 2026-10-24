# mla_v2

## 功能概述

MLA(Multi-head Latent Attention)吸收式(weight-absorbed)decode/prefill 注意力算子:Q 侧拆为 `qAbsorb = q_nope · WUK`(kvLoraRank 维,吸收了上投影)与 `qr`(RoPE 部分),K 侧拆为 latent `kCache`(kvLoraRank 维)与 `peCache`(RoPE 后的 key)。kernel 计算

```
QK = (qAbsorb · kCache^T + qr · peCache^T) * scale
oAbsorb = softmax_causal(QK) · kCache
```

即先算两个吸收式矩阵乘之和再做 softmax,然后用同一份 latent K 左乘还原输出(oAbsorb 随后由 host/WUV 投影到 v_head_dim)。单 kernel 内统一两种 KV 布局(`dense` 标志,见 `csrc/kernels/mla_v2.h` 的 `MLAV2` 类注释,关键字 `dense`/`sparse`;原 mla_v3 的 dense 路径已合并进来,独立 mla_v3 算子已删除):

- **sparse(dense=0,分页)**:经典 mla_v2 路径,KV 走 blockTables 分页寻址,可选 DSA top-k token 选择(topkIndices);
- **dense(dense=1,连续)**:每 batch 起始于 dense cache 的 `b × dim-0 跨度` 处(连续布局时即占据 `[b*maxSeqLen, (b+1)*maxSeqLen)`,由 gather_sparse_kv_cache 收集,`maxSeqLen == indexTopK`),blockTable 不使用(块号即逻辑索引)。

该算子是"短序列"路径:总 KV 长 `ceil(maxTotalLens/blockSize) * blockSize ≤ tileSizeOfCachedKV`(即 `blockTables.shape[1] * blockSize`,运行期行宽,不取 block table 的分配跨度)时模型层选用(`csrc/model.cpp` 的 `ForwardAttnMLAV2`,短序列分支关键字 `<= rt._tileSizeOfCachedKV`),更长则走 flash_mla_v2;decode+DSA 长序列另行路由到 dense 路径(见"关键代码位置")。

## 输入输出参数

Python 侧调用(`tests/kernels/mla.py` 的 `mla_v2(rt, qWithQr_xlite, ...)` 调用点,关键字 `mla_v2(rt, qWithQr_xlite`;dense 用例在同文件的 `mla_v2 dense` 分支):

```python
mla_v2(rt, q_with_qr, qr, k_cache, pe_cache, wuk_t, wuv, output, query_start_loc,
       query_lens, cached_lens, block_tables, n_heads, rope_head_dim, nope_head_dim,
       v_head_dim, kv_lora_rank, BLOCK_SIZE, batch, scale, topk_indices, top_k,
       weight_nz, enable_flash_attention=False, tile_size_of_cached_kv=8192, dense=False)
```

host 封装 `MLAV2`(`csrc/_C.cpp` 的 `MLAV2`)的 `qAbsorb = einsum(q_nope, wukT)` 与 `output = einsum(oAbsorb, wuv)` 在 paged / dense 两条路径**均无条件执行**(对应 `XliteOpEinsumMhtHtdMhd` 分别位于 `if (dense)` 之前与所有分支之后);两条路径唯一区别是 dense(`dense=True`)在吸收前多一步 `gather_sparse_kv_cache` 收集 top-k token 到连续 `kDense`/`peDense`(见 `csrc/_C.cpp` 的 `MLAV2` `dense` 分支,关键字 `XliteOpGatherSparseKVCache`),随后以 `topK` 为 `maxSeqLen`、内核 `topK=0` 调 `XliteOpMLAV2`,输入/输出契约与 paged 相同。接口完整 docstring 见 `xlite/_C.pyi` 的 `mla_v2`(注意 Python API 不含 `max_seq_len`,host 取 `maxSeqLen = block_tables.shape[1] * blockSize`(`shape[1]` 为运行期宽度:V2 持久表由 PrepareAttn 收窄,standalone 由 lens 推导收窄),再传给 kernel)。kernel 签名(`csrc/kernels/mla_v2.h` 的 `mla_v2_<dtype>` 入口,宏 `MLA_V2_FUNC_DEFINE`):

```cpp
mla_v2_<dtype>(qAbsorb, qr, kCache, peCache, topkIndices, qk, oAbsorb,
               queryStartLoc, queryLens, cachedLens, blockTables, nHeads, ropeHeadDim,
               kvLoraRank, blockSize, batch, blockTableStride, maxSeqLen, scale, topK, kCacheStride,
               peCacheStride, dense, zeroGm)
```

末尾的 `zeroGm` 是 host 自动提供的零缓冲(`rt._attnZeroBuffer`,host launch 见 `csrc/op.cpp` 的 `XliteOpMLAV2`),`MlaAicHelper::Init` 用它把两份 L1B ping-pong 缓冲预填零,避免上一 tile 残留 K 数据导致 mmad NaN(与 attention 同源 nan bug 修复)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| qAbsorb | 输入 | `[total_tokens, nHeads, kvLoraRank]` | BF16 | 吸收 WUK 后的 Q latent 部分 |
| qr | 输入 | `[total_tokens, nHeads, ropeHeadDim]` | BF16 | Q 的 RoPE 部分 |
| kCache | 输入 | sparse: `[numBlocks, blockSize, kvLoraRank]`;dense: `[batch, maxSeqLen, kvLoraRank]`(`maxSeqLen == indexTopK`) | BF16 | latent K cache |
| peCache | 输入 | sparse: `[numBlocks, blockSize, ropeHeadDim]`;dense: `[batch, maxSeqLen, ropeHeadDim]`(`maxSeqLen == indexTopK`) | BF16 | RoPE 后的 K cache |
| topkIndices | 输入(可选) | `[total_tokens, topK]` | INT32 | DSA 每行 query 的 top-k token 下标(升序),仅 sparse + topK>0 时使用 |
| qk | workspace | `[aicNum * XLITE_MAX_M0 * 2, maxSeqLen]` | BF16 | QK 分数缓冲;sparse 时 `maxSeqLen = blockTables.shape[1] * blockSize`(运行期行宽,`shape[1]` 已由 PrepareAttn 收窄),dense 时 `maxSeqLen = indexTopK`(见 `csrc/_C.cpp` 的 `MLAV2` dense 分支,关键字 `rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, topK}`) |
| oAbsorb | 输出 | `[total_tokens, nHeads, kvLoraRank]` | BF16 | softmax(QK)·kCache |
| queryStartLoc / queryLens / cachedLens | 输入 | `[batch]` | INT32 | 同 attention |
| blockTables | 输入 | `[rows >= batch, blockTableStride]` | INT32 | sparse 用;dense 传原表但不使用。持久化缓冲,行跨度为分配跨度 `blockTables.stride[0]`,行宽为每步运行期宽度 `shape[1]`,仅每行 lens 推导的有效前缀被读取 |
| nHeads | 标量 | - | uint32 | 本卡头数(测试 1/8/16) |
| ropeHeadDim | 标量 | - | uint32 | RoPE 维(64) |
| kvLoraRank | 标量 | - | uint32 | latent 秩(16/512) |
| blockSize | 标量 | - | uint32 | KV 块大小(128);dense 模式下传入但仅经 MlaAicHelper 的 dense 标志忽略 |
| batch | 标量 | - | uint32 | batch |
| blockTableStride | 标量 | - | uint32 | blockTables 的行跨度(元素数),host 取 `blockTables.stride[0]`,仅用于 sparse 行寻址(dense 置 0) |
| maxSeqLen | 标量 | - | uint32 | qk workspace 行宽与 dense 长度:sparse 时为 `blockTables.shape[1] * blockSize`(qk workspace 行宽;V2 的 `shape[1]` 已由 PrepareAttn 收窄到运行期宽度),dense 时为 indexTopK(dense cache 每批长度) |
| scale | 标量 | - | float | `(nopeHeadDim + ropeHeadDim)^-0.5` |
| topK | 标量 | - | uint32 | DSA top-k;host 限制仅 sparse 模式生效:`maxSeqLen ≤ MAX_SOFTMAX_PINGPONG_LEN=11776` 且 `topK ≤ MAX_TOPK_NUM=2048`(见 `csrc/op.cpp` 的 `XliteOpMLAV2`,关键字 `MAX_SOFTMAX_PINGPONG_LEN`/`MAX_TOPK_NUM`);dense 模式 host 要求 topK > 0 且其值即 dense cache 长度 indexTopK(见 `csrc/_C.cpp` 的 `MLAV2`,关键字 `dense=True requires topK > 0`),内核内部将其用作 maxSeqLen 并把 softmax topK 置 0(`csrc/kernels/mla_v2.h` 的 `MLAV2::Init`,关键字 `this->topK`) |
| kCacheStride / peCacheStride | 标量 | - | uint32 | kCache / peCache 的 dim-0 跨度(元素数),块或 batch 基址 = 序号 × 跨度:sparse 为每物理块跨度(连续时等于 blockSize×kvLoraRank / blockSize×ropeHeadDim),dense 为每 batch 跨度(连续时等于 maxSeqLen×kvLoraRank / maxSeqLen×ropeHeadDim);非连续时由宿主传入真实跨距 |
| dense | 标量 | - | uint32 | 1=dense cache(`[batch, maxSeqLen, ...]`,batch 起点 = b×kCacheStride,由 gather_sparse_kv_cache 收集,blockTable 不使用),0=paged(见 `csrc/op.cpp` 的 `XliteOpMLAV2`,关键字 `dense`) |
| zeroGm | 标量 | - | GM | host 自动提供的零缓冲(`rt._attnZeroBuffer`),非 Python API 参数;`MlaAicHelper::Init` 用它预填零 L1B(见上文) |

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| bfloat16_t | `csrc/kernels/mla_v2_bfloat16_t.cpp` | `mla_v2_bfloat16_t` |

host 侧仅接受 BF16(`csrc/op.cpp` 的 `XliteOpMLAV2`,关键字 `EachXDtype(BF16`);qAbsorb/qr/kCache/peCache/oAbsorb 需一致。

## 实现原理

混合 AIC/AIV kernel(`KERNEL_TYPE_MIX_AIC_1_2`,`csrc/kernels/mla_v2.h` 的 `MLAV2::Init`,关键字 `KERNEL_TYPE_MIX_AIC_1_2`),三级流水 AIC-QK → AIV-softmax → AIC-SV,与 attention 同构但矩阵乘换成 MLA 的吸收式形态。

### 任务划分

- `m0 = dense ? nHeads : XliteFitM0ToHeads(GetOptimalM0(queryLen, cachedLen), nHeads)` 自适应(同 attention,`csrc/kernels/kernel_macro.h` 的 `GetOptimalM0` + `XliteFitM0ToHeads`;dense 模式直接取 nHeads,一个 tile 一 token),`queryTileSize = m0 / nHeads`——MLA 每 batch 一次算**全部 nHeads 个头**(无 KV 头分组),`mSize = queryTaskLen * nHeads` 行拼成 Cube 的 M 维。
- 任务空间为 `queryNum`(每 batch),核间用 `firstCore = (GetBlockIdx()+GetBlockNum()-coreOffset) % GetBlockNum()` 的循环 stride 分配,`coreOffset` 跨 batch 累计保证相邻 batch 起始核轮转、负载均衡(`csrc/kernels/mla_v2.h` 的 `RunAic`,关键字 `firstCore`/`coreOffset`)。
- dense 模式 `calcLen` clip 到 maxSeqLen(dense cache 只存 indexTopK 个 token,见 `csrc/kernels/mla_v2.h` 的 `RunAic`,关键字 `dense cache only holds maxSeqLen`);per-batch KV 视图 dense 时按 dim-0 跨度取 `kCache[batchIdx * kCacheStride]` 子视图,sparse 时走 blockTable(`csrc/kernels/mla_v2.h` 的 `RunAic`,关键字 `batchKCache = kCache[batchIdx * kCacheStride]`)。

### AIC 侧:MlaAicHelper 的吸收式双 GEMM

`MlaAicHelper`(`csrc/kernels/mla_aic_helper.h`)为 mla_v2/flash_mla_v2 共用,几何常量:`qkn0 = dense ? 128 : blockSize`(N 维 tile,dense 时块号即逻辑索引、N 维不受块大小约束)、`qkk0 = 256/sizeof(Dtype)`(K 维 tile)、`svn0 = 256`、`svk0 = 64`,L1/L0 共 12 个 ping-pong 缓冲在 `Init()`(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::Init`)静态布局并返回 svk0。dense 标志与 kCacheStride/peCacheStride 由 mla_v2 的 Init 传入(`csrc/kernels/mla_v2.h` 的 `MLAV2::Init`,关键字 `aicHelper.Init`),qkStride 即 maxSeqLen(`csrc/kernels/mla_v2.h` 的 `MLAV2::Init`,关键字 `this->qkStride`)。

**RunAicQK**(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK`):`QK = (qAbsorb·kCache^T + qr·peCache^T)`,m=queryTokens*nHeads, n=kvLen, 分两段:

1. 沿 kvLoraRank 按 `qkk0`(bf16 为 128)分 k 块,K 每次预取 `4*qkk0` 进 L1B(4 次复用),L0A/L0B ping-pong,`CalMmad(..., kIdx==0)` 控制首块清零累加出 C = qAbsorb·kCache;
2. 同一 n tile 上再算 R = qr·peCache(ropeHeadDim 一趟),`CalMmad(..., false)` 直接累加到同一 L0C,天然完成 `C+R`(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK`,关键字 `CalMmad(qkl0cBuf` 第二段 `false`);
3. 每个 n tile 结束 `CopyToGm` 写 qk(行距 qkStride=maxSeqLen),L0C 双缓冲与下一 tile 的 mmad 重叠。

分页寻址:`block = blockTable[nIdx + nIdxStart]`(sparse)或逻辑块号即物理号(dense,见 `csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK`,关键字 `!dense ? blockTable`);块基址 K 侧取 `block * kPitch`、PE 侧取 `block * pePitch`——sparse 时即传入的 dim-0 跨度 kCacheStride/peCacheStride,dense 时为 `qkn0 * kvLoraRank` / `qkn0 * ropeHeadDim`(batch 段内连续),拷贝形式相同、仅寻址不同。

**RunAicSV**(`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicSV`):`oAbsorb = QK · kCache^T`,m=queryTokens*nHeads, n=kvLoraRank(按 svn0=256 分块), k=kvLen(按 svk0=64 分块)。QK 矩阵每 4*svk0 列预取进 L1A;K^T 侧每 2*svk0 预取进 L1B——dense 时单条 `CopyGmToL1Nd2Nz` 连续拷贝,sparse 时按物理块循环逐块拷(块基址 `block * kCacheStride`,见 `MlaAicHelper::RunAicSV`,关键字 `paged cache: per-block lookup`);`CopyToL0BTCol` 转置进 L0B。输出 `CopyToGm(out[nOffset], ..., kvLoraRank)` 按 kvLoraRank 行距写 oAbsorb。

### AIV 侧:softmax 与 top-k 路径

- **topK == 0**:`RunAivSoftmax`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmax`,转发至 `RunAivSoftmaxLong`)(见 attention 文档),带 `hasScale=true, scale`(QK 乘 scale 在 softmax 内完成)、causal 由 `calcSoftmaxLen = cachedLen + queryTaskStart + 1` + maskOff/headStride=nHeads 实现(`csrc/kernels/mla_v2.h` 的 `RunAiv`,关键字 `calcSoftmaxLen`/`RunAivSoftmax`)。
- **topK > 0**(sparse + DSA):`RunAivSoftmaxPingPong`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`)。该路径先用 `vcmpvs_ge/vcmpvs_lt + vand` 生成命中位图,`vgather` 按 topkIndices 从 QK 行中收集 topK 个分数(未命中的槽位填 -3.4e38),对收集后的 topK 长度做 softmax,再用标量 scatter 按 64 槽粒度把概率散回原 token 位置、其余位置清零(均见 `RunAivSoftmaxPingPong`,关键字 `vcmpvs_ge`/`vgather` 与 scatter 段)。行有效长度 `calcLen > topK` 时才启用,否则退化为全量(`csrc/kernels/mla_v2.h` 的 `RunAiv`,关键字 `calcLen > topK`)。
- **outN 对齐**:softmax 输出行宽取 `ROUND_UP(calcLen, 4*svk0)`(clip 到 maxSeqLen),因为 RunAicSV 每次读 4*svk0=256 列 QK,残留脏数据会沿 KV 维混进 SV 累加(`csrc/kernels/mla_v2.h` 的 `RunAiv`,关键字 `4 * svk0`)。

### 跨核同步

与 attention 相同:QK 完成 `ffts_cross_core_sync(PIPE_FIX, flag0)` → AIV softmax(flag1)→ AIC `wait_flag_dev(1)` 后 SV;相邻任务错峰,末尾补做(`csrc/kernels/mla_v2.h` 的 `RunAic`(关键字 `ffts_cross_core_sync`)与 `RunAiv`(关键字 `ffts_cross_core_sync`))。

### 关键代码位置

- 主类/双模式说明:`csrc/kernels/mla_v2.h` 的 `MLAV2` 类注释(关键字 `dense`/`sparse`)、`MLAV2::RunAic`、`MLAV2::RunAiv`
- QK/SV 吸收式 GEMM:`csrc/kernels/mla_aic_helper.h` 的 `MlaAicHelper::RunAicQK` / `MlaAicHelper::RunAicSV` / `MlaAicHelper::Init`(缓冲布局)
- softmax 全量/top-k:`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmax` / `RunAivSoftmaxPingPong`(其内部实现 `RunAivSoftmaxLong` 新增 SWA 段/压缩比/attnSink 参数为 cxa 算子服务,mla_v2 调用时均取默认值、行为不变)
- host launch 与限制:`csrc/op.cpp` 的 `XliteOpMLAV2`(dense 校验关键字 `MAX_TOPK_NUM`);路由 `csrc/model.cpp` 的 `ForwardAttnMLAV2`(短序列分支)与 dense gather 路径(decode+DSA 长序列时启用,关键字 `XliteOpGatherSparseKVCache`,阈值 `XLITE_MLA_DENSE_THRESHOLD=280`,见 `csrc/model.cpp` 的 `XLITE_MLA_DENSE_THRESHOLD`)
- 常量:`csrc/kernels/kernel_param.h`(`XLITE_MAX_M0`/`MAX_TOPK_NUM`/`MAX_SOFTMAX_PINGPONG_LEN`)
