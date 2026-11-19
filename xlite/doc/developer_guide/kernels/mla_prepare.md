# mla_prepare

## 功能概述

MLA 注意力的前置准备算子(单 AIV 融合 kernel):浮点路径对 `mlaQKVA` 线性层输出的 `attnQkvc = [q_lora | kv_lora | pe]` 逐 token 做三件事——(1) q_lora 段 RMSNorm(+bias) 输出 `attnNormQc`(后续 mlaQB 上投影的输入);(2) kv_lora 段 RMSNorm(+bias) 输出 `attnNormKvc` 并按 slotMapping 写入 paged `kCache`;(3) pe 段做 complex RoPE 后写入 paged `peCache`(`attnQkvc` 本身只读,pe 段不被写回——子函数 `rope_complex_and_cache` 的 `output_ptr` 实参传 `nullptr`,跳过写回分支)。即一次 launch 融合两次分段 norm + RoPE + 双路 cache 写入。模型侧调用在 `csrc/model.cpp` 的 `XModel::ForwardAttnMLACommonV2` 内(关键字 `XliteOpMlaPrepare`),紧跟 mlaQKVA 线性层的 `ForwardLinear`(同一函数内,关键字 `mlaQKVA, attnQkvc`)。SFA-C8 路径见下文。

## 输入输出参数

Python 侧调用(见 `tests/kernels/mla_prepare.py`,关键字 `mla_prepare(rt,`):

```python
mla_prepare(
    rt,
    attn_qkvc,
    q_norm,
    q_norm_bias,
    attn_norm_qc,
    kv_norm,
    kv_norm_bias,
    attn_norm_kvc,
    freqs_cis,
    position,
    q_lora_rank,
    kv_lora_rank,
    rope_head_dim,
    BLOCK_SIZE,
    k_cache,
    pe_cache,
    slot_mapping,
    NORM_EPS,
)
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
| position | 输入 | `[token_num]` | INT64 | 每 token 的旋转位置;二维(mrope)布局时仅消费第一行,host 打印一次性警告(`csrc/op.cpp` 的 `WarnPositions2D`) |
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

SFA-C8 由 `kCache.dtype == INT8` 选择：INT8 `kCache`、BF16 `peCache`、FP32 `kScaleCache` 分别为 `[numBlocks, blockSize, 1, kvLoraRank]`、`[numBlocks, blockSize, 1, ropeHeadDim]`、`[numBlocks, blockSize, 1, kvLoraRank/128]`。此时 host 启动 `mla_prepare_c8_bfloat16_t`，融合 Q/KV RMSNorm、RoPE 和 [mla_cache_c8](mla_cache_c8.md) 的分组量化，输入、norm 参数和量化前结果使用 BF16。RMSNorm 使用传入的 `normEps`。C8 Prepare 的 `ropeHeadDim` 须为正的 16 倍数，并受 UB 容量限制；非 C8 路径不变。

可选的 `attnNormKvc` 非空时写出 KV Norm 结果；为空时仅保留 UB 中的 BF16 结果供量化使用，不申请临时 latent/PE。浮点缓存继续走原融合 kernel。

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| float16_t | `csrc/kernels/mla_prepare_float16_t.cpp` | `mla_prepare_float16_t` |
| bfloat16_t | `csrc/kernels/mla_prepare_bfloat16_t.cpp` | `mla_prepare_bfloat16_t` |
| bfloat16_t → INT8 cache | `csrc/kernels/mla_prepare_c8_bfloat16_t.cpp` | `mla_prepare_c8_bfloat16_t` |

host 按 attnQkvc.dtype 选择(见 `csrc/op.cpp` 的 `XliteOpMlaPrepare`,关键字 `attnQkvc.dtype == FP16`);kernel 整体在 `#ifdef __DAV_C220_VEC__` 下,非向量核平台为空实现(见 `csrc/kernels/mla_prepare.h` 的 `MLA_PREPARE_FUNC_DEFINE` 宏 `#else` 分支)。

## 实现原理

浮点路径为纯 AIV 向量 kernel,本身不做计算,而是把三个已有 kernel 函数按"核偏移接力"的方式串成一个 launch(见 `csrc/kernels/mla_prepare.h` 的 `mla_prepare` 模板函数,融合主体):

1. **q_lora 段 norm**:`norm<Dtype>(attnQkvc, nullptr, qNorm, qNormBias, attnNormQc, token_num, qLoraRank, normEps, NormKind::Rms, cntPerToken=1, inStep=totalDim, outStep=qLoraRank, inOffset=0, ..., tpSize, ...)`——对每个 token 的前 qLoraRank 列做 RMSNorm(带 weight/bias),从 totalDim 宽的行中按 inStep 步长取样;
2. **kv_lora 段 norm + 写 cache**:`norm<Dtype>(attnQkvc, nullptr, kvNorm, kvNormBias, attnNormKvc, ..., kvLoraRank, inOffset=qLoraRank, ..., kCache, slotMapping, blockSize, kCacheStride)`——同样的分段 RMSNorm,但子函数带 kcache/slot_mapping 参数时会把 norm 结果按 slot 直接写入 paged cache,块基址 = 物理块号 × MAX(kCacheStride, blockSize×kvLoraRank)(mla_prepare 传入 norm 的有效跨距,经 MAX 钳制 ≥ 自然步长 blockSize×kvLoraRank;norm 内层再做一次 `MAX(kcacheStride, block_size*total_dim)` 钳位);
3. **pe 段 RoPE + 写 cache**:`rope_complex_and_cache<Dtype>(token_num, nLocalHeads=1, stepDim=totalDim, ropeDim=ropeHeadDim, offset=qLoraRank+kvLoraRank, ..., freqs, position, blockSize, peCache, slotMapping, peCacheStride)`——对每 token 的 pe 段做 complex 形式 RoPE(fp32 精度,偶/奇分量分别乘 cos/sin 再重组),结果写入 peCache(`attnQkvc` 不写回——`output_ptr` 传 `nullptr`,跳过 `rope_complex_and_cache` 的写回分支)。

**核偏移接力(coreOffset)**:三个子函数都是 grid-stride 并行(`tok = block_idx + coreOffset; tok < token_num; tok += block_num`),返回各自消耗后的 `nextCoreOffset`。mla_prepare 把上一个函数的 `nextCoreOffset` 作为下一个函数的 `coreOffset` 传入,使三个阶段在核间的任务边界连续错开,避免每个阶段都从 block 0 起算造成负载倾斜;三个阶段串行于同一 stream,天然满足写读依赖(attnQkvc 各段互不重叠,attnNormQc/Kvc 独立输出)。

RoPE 细节(子函数见 `csrc/kernels/rope_complex_and_cache.h` 的 `rope_complex_and_cache` 模板函数):UB 中 double-buffer 载入输入与 freqs,`vconv` 升 fp32 后按复数旋转(需要 `nLocalHeads == 1` 的 cache 写断言,同函数内关键字 `assert(nLocalHeads == 1)`),写回原 dtype;cache 写入按 slot 分解定位:`(slot/blockSize)*MAX(peCacheStride, blockSize×ropeHeadDim) + (slot%blockSize)*ropeHeadDim`(物理块号 × 经 MAX 钳制的 dim-0 有效跨距 + 块内偏移 × 通道宽;`peCacheStride` 由 mla_prepare 传入前已 MAX 钳制 ≥ 自然步长)。

SFA-C8 的 Q 阶段复用 `norm`，KV 阶段在同一核内完成 RMSNorm、RoPE 和量化。每个任务处理一个 token，输入及输出暂存区使用双缓冲，计算当前 token 时预取下一条输入。KV Norm 先舍入为 BF16，再调用与独立写缓存入口共用的 `mla_cache_c8_quant`；PE 使用 FP32 旋转后转换为 BF16。scale 经 S→V 同步后，由 V→MTE3 保证按相同 slot 写出 INT8 K、BF16 PE 和 FP32 scale，全部 MTE3 完成后才复用相应缓冲；`slot=-1` 不写缓存，但仍保留请求的 Norm 输出。C8 通过末尾可选参数 `k_scale_cache` 传入独立 scale 缓存。

### 片上缓冲分配与占用汇总(AIV UB)

mla_prepare 为纯 AIV 融合 kernel,自身不做缓冲排布,片上存储全部来自子函数:每个子函数在入口自 UB 偏移 0 顺序累进 `__ubuf__` 指针并在尾部 `assert(off <= UB_SIZE)` 收口;浮点路径三个阶段(两次 `norm`、一次 `rope_complex_and_cache`)在同一核上**串行复用同一 UB**,任一时刻占用即当前阶段布局大小,峰值取最大阶段;C8 路径(`csrc/kernels/mla_prepare_c8.h` 的 `mla_prepare_c8`)的 Q 阶段复用 `norm`,KV/RoPE/量化阶段为自有布局,与 Q 阶段串行。硬件容量:AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),每核一份。

dtype 记法:`Dtype∈{float16_t, bfloat16_t}`(两条实例化路径 `csrc/kernels/mla_prepare_float16_t.cpp` / `mla_prepare_bfloat16_t.cpp`),均为 2B,**UB 布局与 dtype 无关**,占用单列。典型配置取测试 `tests/kernels/mla_prepare.py` 的 `test_cases`:qLoraRank=2048、kvLoraRank=512、ropeHeadDim=64、token_num=80、blockSize=128;C8 路径同值(latentDim=kvLoraRank=512、ropeDim=64,`groupSize=128、groupNum=4`)。

**阶段 1:q_lora 段 RMSNorm(`csrc/kernels/norm.h` 的 `norm`,norm_dim=qLoraRank=2048;浮点/C8 两路径共用)**

| 变量 | shape(每份) | dtype | 份数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `in0`/`in1` | `(norm_dim,)`=(2048,) | Dtype | 2(事件0/1) | `ROUND_UP(norm_dim*sizeof(Dtype),32)` | 8192 |
| `out0`/`out1` | `(norm_dim,)` | Dtype | 2(事件0/1) | `ROUND_UP(norm_dim*sizeof(Dtype),32)` | 8192 |
| `calc0`/`calc1` | `(norm_dim,)` | float | 各 1 | `ROUND_UP(norm_dim*sizeof(float),32)` | 16384 |
| `weight_calc`/`bias_calc` | `(norm_dim,)` | float | 各 1 | `ROUND_UP(norm_dim*sizeof(float),32)` | 16384 |
| **合计** | | | | | **49152** |

**阶段 2:kv_lora 段 RMSNorm + 写 kCache(同 `norm`,norm_dim=kvLoraRank=512)**

| 变量 | shape(每份) | dtype | 份数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `in0`/`in1` | `(512,)` | Dtype | 2(事件0/1) | `ROUND_UP(norm_dim*sizeof(Dtype),32)` | 2048 |
| `out0`/`out1` | `(512,)` | Dtype | 2(事件0/1) | `ROUND_UP(norm_dim*sizeof(Dtype),32)` | 2048 |
| `calc0`/`calc1` | `(512,)` | float | 各 1 | `ROUND_UP(norm_dim*sizeof(float),32)` | 4096 |
| `weight_calc`/`bias_calc` | `(512,)` | float | 各 1 | `ROUND_UP(norm_dim*sizeof(float),32)` | 4096 |
| **合计** | | | | | **12288** |

**阶段 3:pe 段 RoPE + 写 peCache(`csrc/kernels/rope_complex_and_cache.h` 的 `rope_complex_and_cache`,ropeDim=64、nLocalHeads=1)**

`vdim==ropeDim` 时 `vRemain` 槽为 0 不占空间;`outInterleaved`/`doRotate` 槽位仅 CXA 等调用方启用,mla_prepare 均传 false,不分配。

| 变量 | shape(每份) | dtype | 份数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `input0`/`input1` | `(vdim,)`=(64,) | Dtype | 2(事件0/1) | `ROUND_UP(vdim*sizeof(Dtype),256)` | 512 |
| `freqs0`/`freqs1` | `(ropeDim,)`=(64,) | float | 2(事件2/3) | `ROUND_UP(ropeDim*sizeof(float),256)` | 512 |
| `out0`/`out1` | `(vdim,)` | Dtype | 2(事件0/1) | `ROUND_UP(vdim*sizeof(Dtype),256)` | 512 |
| `inOutFP32` | `(ropeDim,)` | float | 1 | `ROUND_UP(ropeDim*sizeof(float),256)` | 256 |
| `x_even`/`x_odd`/`cos`/`sin`/`x_even_cos`/`x_odd_sin`/`x_even_sin`/`x_odd_cos` | `(ropeDim,)` | float | 各 1 | `ROUND_UP(ropeDim*sizeof(float),256)` | 2048 |
| `positionUB` | 256 个 uint64(`maxCnt`) | uint64 | 1 | `ROUND_UP(maxCnt*sizeof(uint64_t),256)` | 2048 |
| `slotMappingUB` | 256 个 uint32 | uint32 | 1 | `ROUND_UP(maxCnt*sizeof(uint32_t),256)` | 1024 |
| **合计** | | | | | **6912** |

**C8 路径阶段 2:KV RMSNorm + RoPE + 分组量化(`csrc/kernels/mla_prepare_c8.h` 的 `mla_prepare_c8`,latentDim=512、ropeDim=64;Q 阶段布局同阶段 1)**

| 变量 | shape(每份) | dtype | 份数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `inputs0`/`inputs1` | `(latentDim+ropeDim,)`=(576,) | Dtype | 2(事件2/3) | `ROUND_UP((latentDim+ropeDim)*sizeof(Dtype),256)` | 2560 |
| `normalized0`/`normalized1` | `(latentDim,)`=(512,) | Dtype | 2 | `ROUND_UP(latentDim*sizeof(Dtype),256)` | 2048 |
| `packed0`/`packed1` | 656 B 混合行(INT8 latent|BF16 PE|FP32 scale) | int8 视图 | 2(事件4/5) | `ROUND_UP(latentDim+ropeDim*sizeof(Dtype)+groupNum*sizeof(float),256)` | 1536 |
| `trig0`/`trig1` | `(ropeDim,)` | float | 2(事件2/3) | `ROUND_UP(ropeDim*sizeof(float),256)` | 512 |
| `x` | `(latentDim,)`(兼放 RoPE fp32 行) | float | 1 | `MAX(latentDim*sizeof(float), ROUND_UP(ropeDim*sizeof(float),256))` | 2048 |
| `tmp` | `(latentDim,)` | float | 1 | `latentDim*sizeof(float)` | 2048 |
| `weight` | `(latentDim,)` | float | 1 | `latentDim*sizeof(float)` | 2048 |
| `bias` | `(latentDim,)` | float | 1 | `latentDim*sizeof(float)` | 2048 |
| `rope[8]` | `(ropeDim,)` | float | 8 | `ROUND_UP(ropeDim*sizeof(float),256)` | 2048 |
| `amaxBuf` | 1 个 256B repeat | float | 1 | `VECTOR_REPEAT_BYTESIZE` | 256 |
| `factorBuf` | 1 个 256B repeat | float | 1 | `VECTOR_REPEAT_BYTESIZE` | 256 |
| `halfValues` | `(groupSize,)`=(128,) | half | 1 | `groupSize*sizeof(half)` | 256 |
| **合计** | | | | | **17664** |

**占用汇总**(各阶段布局串行复用同一 UB,峰值取最大阶段;字节数):

| 阶段布局 | 硬件容量 | 占用(B) | 占比 |
|---|---|---|---|
| 阶段 1:q_lora norm(浮点/C8 共用,峰值) | UB 192 KB | 49152 | 25.0% |
| 阶段 2:kv_lora norm(浮点) | | 12288 | 6.25% |
| 阶段 3:RoPE(浮点) | | 6912 | 3.5% |
| C8 阶段 2:KV+RoPE+量化 | | 17664 | 9.0% |

要点:

- 峰值在阶段 1:q_lora 段 2048 维的 norm 每 norm_dim 元素占 24 B(4 条 Dtype 缓冲 8 B + 4 条 fp32 缓冲 16 B),共 49152 B,仅占 UB 25%——大头是 `calc0`/`calc1`/`weight_calc`/`bias_calc` 四条 fp32 行(32768 B,占该布局 67%);
- 其余阶段(kv norm 12288 B、RoPE 6912 B、C8 量化 17664 B)与阶段 1 串行复用同一 UB,均远未顶满,远小于单算子 norm/mla_cache_c8 独立 launch 时的预算压力;
- 容量来源:`csrc/kernels/norm.h` 的 `norm` 与 `csrc/kernels/mla_prepare_c8.h` 的 `mla_prepare_c8` 尾部的 `assert(off <= UB_SIZE)`;host 侧 SFA-C8 校验(`csrc/op.cpp` 的 `XliteOpMlaPrepare`,关键字 `unsupported SFA-C8 configuration`)要求 `kvLoraRank%128==0` 且 `ropeHeadDim*2` 按 32B 对齐,C8 布局中 `x`/`tmp`/`weight`/`bias` 四条 fp32 行(每条 latentDim×4 B)随 latentDim 线性增长,是「C8 Prepare 受 UB 容量限制」的具体来源。

## 关键代码位置

- 融合主体:`csrc/kernels/mla_prepare.h` 的 `mla_prepare` 模板函数
- 分段 RMSNorm 与 cache 直写:`csrc/kernels/norm.h` 的 `norm` 函数(模板函数,带 `kcache`/`slot_mapping`/`coreOffset` 参数)
- complex RoPE + cache:`csrc/kernels/rope_complex_and_cache.h` 的 `rope_complex_and_cache` 模板函数
- host launch:`csrc/op.cpp` 的 `XliteOpMlaPrepare`;模型调用 `csrc/model.cpp` 的 `XModel::ForwardAttnMLACommonV2`
