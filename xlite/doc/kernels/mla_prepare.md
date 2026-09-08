# mla_prepare

## 功能概述

MLA 注意力的前置准备算子(单 AIV 融合 kernel):对 `mlaQKVA` 线性层输出的 `attnQkvc = [q_lora | kv_lora | pe]` 逐 token 做三件事——(1) q_lora 段 RMSNorm(+bias) 输出 `attnNormQc`(后续 mlaQB 上投影的输入);(2) kv_lora 段 RMSNorm(+bias) 输出 `attnNormKvc` 并按 slotMapping 写入 paged `kCache`;(3) pe 段做 complex RoPE 后原地写回 attnQkvc 并写入 paged `peCache`。即一次 launch 融合两次分段 norm + RoPE + 双路 cache 写入。模型侧调用在 `csrc/model.cpp:430`(ForwardAttnMLACommonV2),紧跟 mlaQKVA 线性层。

## 输入输出参数

Python 侧调用(`tests/kernels/mla_prepare.py:113`):

```python
mla_prepare(rt, attn_qkvc, q_norm, q_norm_bias, attn_norm_qc, kv_norm, kv_norm_bias,
            attn_norm_kvc, freqs_cis, position, q_lora_rank, kv_lora_rank, rope_head_dim,
            BLOCK_SIZE, k_cache, pe_cache, slot_mapping, NORM_EPS)
```

host 封装 `MlaPrepare`(`csrc/_C.cpp:1830`)→ `XliteOpMlaPrepare`(`csrc/op.cpp:1108`)。kernel 签名(`csrc/kernels/mla_prepare.h:42`):

```cpp
mla_prepare_<dtype>(attnQkvc, qNorm, qNormBias, attnNormQc, kvNorm, kvNormBias, attnNormKvc,
                    freqs, position, kCache, peCache, slotMapping, token_num, qLoraRank,
                    kvLoraRank, ropeHeadDim, blockSize, normEps, tpSize)
```

记 `totalDim = qLoraRank + kvLoraRank + ropeHeadDim`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| attnQkvc | 输入/输出 | `[token_num, totalDim]` | FP16/BF16 | mlaQKVA 输出;pe 段 RoPE 后原地更新 |
| qNorm / qNormBias | 输入 | `[qLoraRank]` | FP16/BF16 | q_lora 段 RMSNorm 权重/偏置 |
| attnNormQc | 输出 | `[token_num, qLoraRank]` | FP16/BF16 | q_lora 段 norm 结果 |
| kvNorm / kvNormBias | 输入 | `[kvLoraRank]` | FP16/BF16 | kv_lora 段 RMSNorm 权重/偏置 |
| attnNormKvc | 输出 | `[token_num, kvLoraRank]` | FP16/BF16 | kv_lora 段 norm 结果 |
| freqs | 输入 | `[maxSeqLen, ropeHeadDim/2]` | FP32(complex) | 预计算 freqs_cis |
| position | 输入 | `[token_num]` | INT64 | 每 token 的旋转位置 |
| kCache | 输出 | `[numBlocks, blockSize, kvLoraRank]` | FP16/BF16 | latent K cache,按 slotMapping 写 |
| peCache | 输出 | `[numBlocks, blockSize, ropeHeadDim]` | FP16/BF16 | RoPE 后的 K cache,按 slotMapping 写 |
| slotMapping | 输入 | `[token_num]` | INT32 | 每 token 的目标 cache slot(物理块×blockSize+块内偏移) |
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

host 按 attnQkvc.dtype 选择(`csrc/op.cpp:1119-1123`);kernel 整体在 `#ifdef __DAV_C220_VEC__` 下,非向量核平台为空实现(`csrc/kernels/mla_prepare.h:53-62`)。

## 实现原理

纯 AIV 向量 kernel,本身不做计算,而是把三个已有 kernel 函数按"核偏移接力"的方式串成一个 launch(`csrc/kernels/mla_prepare.h:24-39`):

1. **q_lora 段 norm**:`norm<Dtype>(attnQkvc, nullptr, qNorm, qNormBias, attnNormQc, token_num, qLoraRank, normEps, NormKind::Rms, cntPerToken=1, inStep=totalDim, outStep=qLoraRank, inOffset=0, ..., tpSize, ...)`——对每个 token 的前 qLoraRank 列做 RMSNorm(带 weight/bias),从 totalDim 宽的行中按 inStep 步长取样;
2. **kv_lora 段 norm + 写 cache**:`norm<Dtype>(attnQkvc, nullptr, kvNorm, kvNormBias, attnNormKvc, ..., kvLoraRank, inOffset=qLoraRank, ..., kCache, slotMapping, blockSize)`——同样的分段 RMSNorm,但子函数带 kcache/slot_mapping 参数时会把 norm 结果按 slot 直接写入 paged cache(norm 的 cache 直写分支);
3. **pe 段 RoPE + 写 cache**:`rope_complex_and_cache<Dtype>(token_num, nLocalHeads=1, stepDim=totalDim, ropeDim=ropeHeadDim, offset=qLoraRank+kvLoraRank, ..., freqs, position, blockSize, peCache, slotMapping)`——对每 token 的 pe 段做 complex 形式 RoPE(fp32 精度,偶/奇分量分别乘 cos/sin 再重组),结果原地写回 attnQkvc 并写入 peCache。

**核偏移接力(coreOffset)**:三个子函数都是 grid-stride 并行(`tok = block_idx + coreOffset; tok < token_num; tok += block_num`),返回各自消耗后的 `nextCoreOffset`。mla_prepare 把上一个函数的 `nextCoreOffset` 作为下一个函数的 `coreOffset` 传入,使三个阶段在核间的任务边界连续错开,避免每个阶段都从 block 0 起算造成负载倾斜;三个阶段串行于同一 stream,天然满足写读依赖(attnQkvc 各段互不重叠,attnNormQc/Kvc 独立输出)。

RoPE 细节(子函数 `csrc/kernels/rope_complex_and_cache.h:12`):UB 中 double-buffer 载入输入与 freqs,`vconv` 升 fp32 后按复数旋转(需要 `nLocalHeads == 1` 的 cache 写断言,`csrc/kernels/rope_complex_and_cache.h:26-28`),写回原 dtype;cache 写入按 slotMapping 的物理块号 + 块内偏移定位。

## 关键代码位置

- 融合主体:`csrc/kernels/mla_prepare.h:24-39`
- 分段 RMSNorm 与 cache 直写:`csrc/kernels/norm.h:223`(norm 函数签名)
- complex RoPE + cache:`csrc/kernels/rope_complex_and_cache.h:12`
- host launch:`csrc/op.cpp:1108`;模型调用 `csrc/model.cpp:430`
