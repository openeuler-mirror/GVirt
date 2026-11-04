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

`D` 为正且是 128 的整数倍，`R` 为正且 BF16 字节数按 32 字节对齐；缓冲区须能放入 UB。输入和缓存连续存放；调用方保证有效 slot 不重复且在缓存范围内。输入不会被修改。

## 支持的数据类型

输入 BF16，latent 缓存 INT8，PE 缓存 BF16，scale 缓存 FP32。实例化入口为 `csrc/kernels/mla_cache_c8_bfloat16_t.cpp`，host 入口为 `XliteOpMlaCacheC8`。

## 实现原理

每个 AIV 任务处理一个完整 token，MTE2 分别搬入 `D` 维 latent 和 `R` 维 PE。latent 分为 `D/128` 个组，逐组转换到 FP32，计算绝对值和组内最大值，再量化：

```text
amax = max(abs(x))
scale = amax * FP32(1/127)
factor = FP32(1/amax) * 127
q = round(clip(x * factor, -128, 127))
```

除法、乘法和舍入保持分步执行；全零组写 `q=0, scale=0`。采用 FP32→FP16(ODD)→INT8(RINT) 两段转换，避免 FP16 中间舍入改变量化值。量化值饱和到 INT8 范围。

UB 使用两套缓冲，`512/64` 配置下在 Ascend A3 上每核共 7 KiB，每套保存一个 token 的输入、连续排列的 K/PE/scale 输出暂存区、FP32 计算区和转换缓冲。计算当前 token 时，MTE2 预取下一个 token 到另一套缓冲。MTE2→V 保证输入可用，V→S 读取归约结果，V/S→MTE3 后分别写回三份缓存；全部写回完成后，MTE2、V 和 S 才能复用该缓冲。

K 和 PE 使用连续 DMA；scale 行可能不足 32 字节，使用按字节长度的 DMA。组内计算使用向量指令，外层循环分配 token，内层循环逐组独立量化。
