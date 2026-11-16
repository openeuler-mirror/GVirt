# mla_cache_c8

## 功能概述

将归一化后的 latent KV 和 RoPE 后的 PE 写入独立的 SFA-C8 分页缓存。latent 每 128 维动态量化一次，PE 保留 BF16，scale 单独保存为 FP32。

latent 和 PE 维度分别为 `D`、`R`，每个 token 的有效数据总量为 `D + 2R + 4(D/128)` 字节，分开存放：

| 缓存 | 每 token 内容 |
|---|---|
| cache | `D` 个 INT8 latent 元素 |
| pe_cache | `R` 个 BF16 PE 元素 |
| k_scale_cache | `D/128` 个 FP32 scale |

## 输入输出参数

```python
mla_cache_c8(rt, latent, pe, slot_mapping, cache, pe_cache, k_scale_cache)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| latent | 输入 | `[T, D]` | BF16 | RMSNorm 后的 latent KV |
| pe | 输入 | `[T, R]` | BF16 | RoPE 后的 PE，实部和虚部分半存放 |
| slot_mapping | 输入 | `[T]` | INT32 | 物理块号 × block size + 块内位置；`-1` 跳过 |
| cache | 输出 | `[numBlocks, blockSize, 1, D]` | INT8 | latent，未选中的 slot 不变 |
| pe_cache | 输出 | `[numBlocks, blockSize, 1, R]` | BF16 | PE，与 latent 使用相同 slot |
| k_scale_cache | 输出 | `[numBlocks, blockSize, 1, D/128]` | FP32 | 分组 scale，与 latent 使用相同 slot |
| tokens | kernel 标量 | — | uint32_t | 本次 token 数，由 `latent.shape[0]` 得到 |
| latentDim / ropeDim | kernel 标量 | — | uint32_t | 由 `latent.shape[1]`、`pe.shape[1]` 得到 |

`D` 为正且是 128 的整数倍，`R` 为正且 BF16 字节按 32 字节对齐；缓冲区须能放入 UB。输入和缓存连续存放；调用方保证有效 slot 不重复且在缓存范围内。输入不会被修改。host 侧校验 latent/pe 均为 2D、行数一致且 `slotMapping` 条目数等于 T，T=0 时直接返回（`csrc/op.cpp` 的 `XliteOpMlaCacheC8`）。

## 支持的数据类型

输入 BF16，latent 缓存 INT8，PE 缓存 BF16，scale 缓存 FP32。实例化入口为 `csrc/kernels/mla_cache_c8_bfloat16_t.cpp`，host 入口为 `XliteOpMlaCacheC8`。

## 实现原理

每个 AIV 任务处理一个完整 token，MTE2 分别搬入 `D` 维 latent 和 `R` 维 PE。latent 整行转换到 FP32 后，分为 `D/128` 个组，逐组计算绝对值和组内最大值，再缩放：

```text
amax = max(abs(x))
scale = amax * FP32(1/127)
factor = FP32(1/amax) * 127
q = round(x * factor)
```

除法、乘法和舍入保持分步执行；全零组写 `q=0, scale=0`。各组缩放完成后，整行采用 FP32→FP16(ODD)→INT8(RINT) 两段转换，避免 FP16 中间舍入改变量化值。量化值饱和到 INT8 范围。

UB 使用两套缓冲，`512/64` 配置下在 Ascend A3 上每核共 12 KiB，每套保存一个 token 的输入、连续排列的 K/PE/scale 输出暂存区、整行 FP32 数据和 FP16 转换缓冲，以及单组计算临时区。每个有效 token 切换一次 `eid`，在同一套缓冲内完成搬入、量化和写回；无效 slot 直接跳过。MTE3→MTE2→V 保证旧输出写回后才搬入并计算新输入，V→S 读取归约结果；scale 经 S→V 同步后，与量化结果一起由 V→MTE3 保证写回就绪。

K 和 PE 使用连续 DMA；scale 行可能不足 32 字节，使用按字节长度的 DMA。组内计算使用向量指令，外层循环分配 token，内层循环逐组独立量化。

`XliteOpMlaPrepare` 在 SFA-C8 模式使用融合内核，复用本文件中的 `mla_cache_c8_quant` 分组量化；RMSNorm 仍使用模型的 `normEps`。读取端见 [gather_sparse_kv_cache](gather_sparse_kv_cache.md)。

### 片上缓冲分配与占用汇总(AIV UB)

`mla_cache_c8`(`csrc/kernels/mla_cache_c8.h` 的 `mla_cache_c8`)为纯 AIV kernel,在函数入口自 UB 偏移 0 为两套 token 级乒乓缓冲顺序累进 `__ubuf__` 指针:每套依次排布搬入行(`inputs`)、K/PE/scale 连续打包行(`packed`)、整行 fp32 数据(`xBuf`)、单组计算临时(`tmpBuf`/`amaxBufs`/`factorBufs`)与 fp16 中转行(`halfBuf`),每套末尾按 `UB_BANKGROUP_ROW_SIZE=512B` 行对齐垫片推进,尾部 `assert(off <= UB_SIZE)` 收口。硬件容量:AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),每核一份。

dtype 记法:仅 BF16 单路径(`csrc/kernels/mla_cache_c8_bfloat16_t.cpp`),`Dtype=bfloat16_t`(2B),latent INT8 / PE BF16 / scale FP32 由 `packed` 行内混排。典型配置取 `tests/kernels/mla_cache_c8.py` 的默认 `latentDim=512、ropeDim=64`(即模型 latent 档 `D=512、R=64`),`groupSize=128、groupNum=D/128=4`,每 token 打包行 `cacheBytes=D+2R+4×groupNum=656` B。

| 变量 | shape(每套) | dtype | 乒乓套数 | 单套大小 | 占用(B) |
|---|---|---|---|---|---|
| `inputs` | `(latentDim,)`=(512,) | Dtype | 2(事件0/1) | `ROUND_UP(latentDim*sizeof(Dtype),256)` | 2048 |
| `packed` | 656 B 混合行(INT8 latent|BF16 PE|FP32 scale) | int8 视图 | 2(事件0/1) | `ROUND_UP(cacheBytes,256)` | 1536 |
| `xBuf` | `(latentDim,)` | float | 2 | `ROUND_UP(latentDim*sizeof(float),32)` | 4096 |
| `tmpBuf` | `(groupSize,)`=(128,) | float | 2 | `ROUND_UP(groupSize*sizeof(float),32)` | 1024 |
| `amaxBufs` | 1 个 256B repeat | float | 2 | `VECTOR_REPEAT_BYTESIZE` | 512 |
| `factorBufs` | 1 个 256B repeat | float | 2 | `VECTOR_REPEAT_BYTESIZE` | 512 |
| `halfBuf` | `(latentDim,)` | half | 2 | `ROUND_UP(latentDim*sizeof(half),32)` | 2048 |
| 套间 512B 行对齐垫片 | — | — | 2 | 各 256 B | 512 |
| **合计** | | | | | **12288** |

**占用汇总**(vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用(B) | 占比 |
|---|---|---|---|
| UB(每核) | 192 KB | 12288 | 6.25% |

要点:

- 两套乒乓各 5888 B(未对齐),加 2×256 B 行对齐垫片共 12288 B(恰 12 KiB,与上文「每核共 12 KiB」一致),仅占 UB 6.3%,非常宽裕;
- 大头是整行 fp32 暂存 `xBuf`(每套 2048 B)与搬入/打包行 `inputs`/`packed`(每套 1024/768 B);latentDim 增大时线性部分约 18 B/维(`inputs`/`packed`/`xBuf`/`halfBuf` 两套共 2×(2+1+4+2) B);
- 容量来源:kernel 内 `assert(off <= UB_SIZE)`(`csrc/kernels/mla_cache_c8.h` 的 `mla_cache_c8`);host 侧 `XliteOpMlaCacheC8`(`csrc/op.cpp` 的 `XliteOpMlaCacheC8`)校验 `latentDim%128==0`、`ropeDim*sizeof(Dtype)` 按 32B 对齐,但不设 D 的显式上界——更大的 latentDim 由该 assert 兜底。
