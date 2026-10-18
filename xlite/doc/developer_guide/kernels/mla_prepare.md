# mla_prepare

## 功能概述

MLA 注意力的前置准备算子(单 AIV 融合 kernel):对 `mlaQKVA` 线性层输出的 `attnQkvc = [q_lora | kv_lora | pe]` 逐 token 做三件事——(1) q_lora 段 RMSNorm(+bias) 输出 `attnNormQc`(后续 mlaQB 上投影的输入);(2) kv_lora 段 RMSNorm(+bias) 输出 `attnNormKvc` 并按 slotMapping 写入 paged `kCache`;(3) pe 段做 complex RoPE 后写入 paged `peCache`(`attnQkvc` 本身只读,pe 段不被写回——子函数 `rope_complex_and_cache` 的 `output_ptr` 实参传 `nullptr`,跳过写回分支)。即一次 launch 融合两次分段 norm + RoPE + 双路 cache 写入。模型侧调用在 `csrc/model.cpp` 的 `XModel::ForwardAttnMLACommonV2` 内(关键字 `XliteOpMlaPrepare`),紧跟 mlaQKVA 线性层的 `ForwardLinear`(同一函数内,关键字 `mlaQKVA, attnQkvc`)。

## 输入输出参数

Python 侧调用(见 `tests/kernels/mla_prepare.py`,关键字 `mla_prepare(rt,`):

```python
mla_prepare(rt, attn_qkvc, q_norm, q_norm_bias, attn_norm_qc, kv_norm, kv_norm_bias,
            attn_norm_kvc, freqs_cis, position, q_lora_rank, kv_lora_rank, rope_head_dim,
            BLOCK_SIZE, k_cache, pe_cache, slot_mapping, NORM_EPS)
```

host 封装 `MlaPrepare`(见 `csrc/_C.cpp` 的 `MlaPrepare`)→ `XliteOpMlaPrepare`(见 `csrc/op.cpp` 的 `XliteOpMlaPrepare`)。kernel 签名(见 `csrc/kernels/mla_prepare.h` 的 `mla_prepare` 模板函数):

```cpp
mla_prepare_<dtype>(attnQkvc, qNorm, qNormBias, attnNormQc, kvNorm, kvNormBias, attnNormKvc,
                    freqs, position, kCache, peCache, slotMapping, kCacheStride, peCacheStride,
                    token_num, qLoraRank, kvLoraRank, ropeHeadDim, blockSize, normEps, tpSize)
```

记 `totalDim = qLoraRank + kvLoraRank + ropeHeadDim`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| attnQkvc | 输入 | `[token_num, totalDim]` | FP16/BF16 | mlaQKVA 输出;q_lora/kv_lora 段供分段 norm 读取,pe 段供 RoPE 读取;整张量只读不被写回(pe 段 RoPE 结果只写入 peCache,见功能概述) |
| qNorm / qNormBias | 输入 | `[qLoraRank]` | FP16/BF16 | q_lora 段 RMSNorm 权重/偏置 |
| attnNormQc | 输出 | `[token_num, qLoraRank]` | FP16/BF16 | q_lora 段 norm 结果 |
| kvNorm / kvNormBias | 输入 | `[kvLoraRank]` | FP16/BF16 | kv_lora 段 RMSNorm 权重/偏置 |
| attnNormKvc | 输出 | `[token_num, kvLoraRank]` | FP16/BF16 | kv_lora 段 norm 结果 |
| freqs | 输入 | `[maxSeqLen, ropeHeadDim/2]` | FP32(complex) | 预计算 freqs_cis |
| position | 输入 | `[token_num]` | INT64 | 每 token 的旋转位置 |
| kCache | 输出 | `[numBlocks, blockSize, kvLoraRank]` | FP16/BF16 | latent K cache,按 slotMapping 写 |
| peCache | 输出 | `[numBlocks, blockSize, ropeHeadDim]` | FP16/BF16 | RoPE 后的 K cache,按 slotMapping 写 |
| slotMapping | 输入 | `[token_num]` | INT32 | 每 token 的目标 cache slot(物理块×blockSize+块内偏移) |
| kCacheStride / peCacheStride | 标量 | - | uint32 | kCache / peCache 的 dim-0 跨度(元素数);连续时等于 blockSize×kvLoraRank / blockSize×ropeHeadDim,非连续时由宿主传入真实跨距 |
| token_num | 标量 | - | uint32 | token 数(测试 80) |
| qLoraRank / kvLoraRank / ropeHeadDim | 标量 | - | uint32 | 三段宽度(测试 (2048,512,64)) |
| blockSize | 标量 | - | uint32 | cache 块大小(128) |
| normEps | 标量 | - | float | RMSNorm epsilon(1e-6) |
| tpSize | 标量 | - | uint32 | 张量并行度(透传给 norm 子函数做输出切分) |

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| float16_t | `csrc/kernels/mla_prepare_float16_t.cpp` | `mla_prepare_float16_t` |
| bfloat16_t | `csrc/kernels/mla_prepare_bfloat16_t.cpp` | `mla_prepare_bfloat16_t` |

host 按 attnQkvc.dtype 选择(见 `csrc/op.cpp` 的 `XliteOpMlaPrepare`,关键字 `attnQkvc.dtype == FP16`);kernel 整体在 `#ifdef __DAV_C220_VEC__` 下,非向量核平台为空实现(见 `csrc/kernels/mla_prepare.h` 的 `MLA_PREPARE_FUNC_DEFINE` 宏 `#else` 分支)。

## 实现原理

纯 AIV 向量 kernel,本身不做计算,而是把三个已有 kernel 函数按"核偏移接力"的方式串成一个 launch(见 `csrc/kernels/mla_prepare.h` 的 `mla_prepare` 模板函数,融合主体):

1. **q_lora 段 norm**:`norm<Dtype>(attnQkvc, nullptr, qNorm, qNormBias, attnNormQc, token_num, qLoraRank, normEps, NormKind::Rms, cntPerToken=1, inStep=totalDim, outStep=qLoraRank, inOffset=0, ..., tpSize, ...)`——对每个 token 的前 qLoraRank 列做 RMSNorm(带 weight/bias),从 totalDim 宽的行中按 inStep 步长取样;
2. **kv_lora 段 norm + 写 cache**:`norm<Dtype>(attnQkvc, nullptr, kvNorm, kvNormBias, attnNormKvc, ..., kvLoraRank, inOffset=qLoraRank, ..., kCache, slotMapping, blockSize, kCacheStride)`——同样的分段 RMSNorm,但子函数带 kcache/slot_mapping 参数时会把 norm 结果按 slot 直接写入 paged cache,块基址 = 物理块号 × MAX(kCacheStride, blockSize×kvLoraRank)(mla_prepare 传入 norm 的有效跨距,经 MAX 钳制 ≥ 自然步长 blockSize×kvLoraRank;norm 内层再做一次 `MAX(kcacheStride, block_size*total_dim)` 钳位);
3. **pe 段 RoPE + 写 cache**:`rope_complex_and_cache<Dtype>(token_num, nLocalHeads=1, stepDim=totalDim, ropeDim=ropeHeadDim, offset=qLoraRank+kvLoraRank, ..., freqs, position, blockSize, peCache, slotMapping, peCacheStride)`——对每 token 的 pe 段做 complex 形式 RoPE(fp32 精度,偶/奇分量分别乘 cos/sin 再重组),结果写入 peCache(`attnQkvc` 不写回——`output_ptr` 传 `nullptr`,跳过 `rope_complex_and_cache` 的写回分支)。

**核偏移接力(coreOffset)**:三个子函数都是 grid-stride 并行(`tok = block_idx + coreOffset; tok < token_num; tok += block_num`),返回各自消耗后的 `nextCoreOffset`。mla_prepare 把上一个函数的 `nextCoreOffset` 作为下一个函数的 `coreOffset` 传入,使三个阶段在核间的任务边界连续错开,避免每个阶段都从 block 0 起算造成负载倾斜;三个阶段串行于同一 stream,天然满足写读依赖(attnQkvc 各段互不重叠,attnNormQc/Kvc 独立输出)。

RoPE 细节(子函数见 `csrc/kernels/rope_complex_and_cache.h` 的 `rope_complex_and_cache` 模板函数):UB 中 double-buffer 载入输入与 freqs,`vconv` 升 fp32 后按复数旋转(需要 `nLocalHeads == 1` 的 cache 写断言,同函数内关键字 `assert(nLocalHeads == 1)`),写回原 dtype;cache 写入按 slot 分解定位:`(slot/blockSize)*MAX(peCacheStride, blockSize×ropeHeadDim) + (slot%blockSize)*ropeHeadDim`(物理块号 × 经 MAX 钳制的 dim-0 有效跨距 + 块内偏移 × 通道宽;`peCacheStride` 由 mla_prepare 传入前已 MAX 钳制 ≥ 自然步长)。

## 关键代码位置

- 融合主体:`csrc/kernels/mla_prepare.h` 的 `mla_prepare` 模板函数
- 分段 RMSNorm 与 cache 直写:`csrc/kernels/norm.h` 的 `norm` 函数(模板函数,带 `kcache`/`slot_mapping`/`coreOffset` 参数)
- complex RoPE + cache:`csrc/kernels/rope_complex_and_cache.h` 的 `rope_complex_and_cache` 模板函数
- host launch:`csrc/op.cpp` 的 `XliteOpMlaPrepare`;模型调用 `csrc/model.cpp` 的 `XModel::ForwardAttnMLACommonV2`
