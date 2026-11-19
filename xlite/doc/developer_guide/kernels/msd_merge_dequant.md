# msd_merge_dequant

## 功能概述

MSD W4A8 MoE 管线的收尾算子(Step 3):对 INT4×INT4 group_matmul 产出的中间结果做合并、低 nibble 偏置补偿和 per-token 反量化,一步算出最终的 BF16 输出:

```text
Y[r] = (Y_high[r] × 16 + Y_low[r] + scale_bias[expert(r)][c]) × perTokenScale[r]
```

- **Merge**:中间结果是按 token 交错的双行 `[2m, n]`——token r 的 Y_low 在行 2r、Y_high 在行 2r+1,合并即 `Y_high×16 + Y_low`(对应 unpack_activation 的分解恒等式);
- **Bias**:`scale_bias` 是每专家、每列的补偿项 `8 × Σ_k (W_int4[k] × w_scale[k])`,补回低 nibble 拆分时引入的 -8 偏移;
- **Dequant**:`perTokenScale[r]` 是 quant_dyn 输出的每 token 激活量化 scale,乘回还原 BF16 幅值。

多专家(MoE grouped)场景下,每行的专家归属由 `counts`(每专家 token 数,行轴按专家分组)前缀和确定,从而索引到该专家的 scale_bias 行指针。

## 输入输出参数

Python 入口 `msd_merge_dequant(rt, y_merged, scale_biases, counts, per_token_scale, out)`(`csrc/_C.cpp` 的 `MSDMergeDequant`,关键字 `m.def("msd_merge_dequant"`);scale_biases 是每专家一个 `[n]` fp32 张量的列表,host 侧拼成 INT64 指针数组 H2D 拷贝后传入(见 `csrc/_C.cpp` 的 `MSDMergeDequant`,关键字 `MemcpyH2D(_scaleBiasPtrs`)。kernel 签名见 `csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`(入口符号 `msd_merge_dequant_int8_t` 由 `MSD_MERGE_DEQUANT_FUNC_DEFINE` 宏展开):

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| y_merged | 输入 | `[2*m, n]` | float16 | 交错中间结果:token r 的 low 在行 2r、high 在行 2r+1 |
| scaleBiasPtrs | 输入 | `[numExperts]` | int64(GM 指针数组) | 每专家 scale_bias 矩阵 `[n]`(fp32)的 GM 地址;按全局专家 id 索引 |
| perTokenScale | 输入 | `[m]` | float32 | 每 token 激活反量化 scale(quant_dyn 输出) |
| y | 输出 | `[m, n]` | bfloat16 | 最终结果 |
| pnum_tokens | 输入(可选) | `[1]` | uint32 | 动态真实 token 数;host 当前固定传 null(`csrc/op.cpp` 的 `XliteOpMSDMergeDequant`,关键字 `nullptr`) |
| m | 标量 | - | uint32_t | 输出行数,host 由 `yMerged.shape[0] / 2` 得出 |
| n | 标量 | - | uint32_t | 列数,`yMerged.shape[1]` |
| counts | 输入 | `[numExperts]` | uint32 | 每专家 token 数(未翻倍,与 out 行轴一致);按全局专家 id 索引,仅 `[startIdx, endIdx)` 有 token |
| startIdx / endIdx | 标量 | - | uint32_t | 本次 pass 处理的专家范围(EP 分片);numExperts = endIdx - startIdx ≤ 8 |

测试参考:[tests/kernels/msd_dequant.py](../../../tests/kernels/msd_dequant.py),单专家覆盖 `[8, 96]`、`[40, 512]`、`[200, 2048]`、`[1, 2048]`、`[8192, 6144]`、`[8, 8064]`(n=8064 钉住 host 上界 `MSD_MERGE_DEQUANT_MAX_N`:上限回退或 UB 布局回归时该用例失败);多专家覆盖 `(200, 4096, E=4 均分)`、`(200, 4096, E=8 倾斜 [80,40,20,18,15,12,10,5])`、`(8192, 4096, E=3 [4096,3072,1024])`(倾斜分片迫使每个 block 的跨步行序列内频繁切换专家);单/多专家用例均与纯 torch 参考对齐,另有 10 次重复 launch 的 bit 级确定性验证(运行间自比对,不与 torch 参考比较;压测 bias_buf 每行重载时 MTE2 覆写与 V 读的交错)。

## 支持的数据类型

- `int8_t` 实例(算子语义为 fp16 输入 → bf16 输出;[msd_merge_dequant_int8_t.cpp](../../../csrc/kernels/msd_merge_dequant_int8_t.cpp))

host 校验 `yMerged.dtype == FP16 && perTokenScale.dtype == FP32 && out.dtype == BF16`,且 yMerged 为 2D、行数为偶数(`csrc/op.cpp` 的 `XliteOpMSDMergeDequant`,关键字 `yMerged.shape[0] % 2`);n 超出单行 UB 预算 `MSD_MERGE_DEQUANT_MAX_N = 8064`(`csrc/kernels/kernel_param.h`)时抛错(kernel 侧另有 `assert(end_addr <= UB_SIZE)` 布局断言)。

## 实现原理

实现位于 [csrc/kernels/msd_merge_dequant.h](../../../csrc/kernels/msd_merge_dequant.h),函数 `msd_merge_dequant`。

### 分块与多 Block 并行

- **无 n 切块**:整行一次搬入,`n_pad = ROUND_UP(n, 128)`;单行 UB 布局为 24B/元素(low/high fp16 ×2 + low_fp32/merged/bias fp32 ×3 + out bf16 ×2)+ 32B counts 切片,安全上界 `n ≤ MSD_MERGE_DEQUANT_MAX_N = 8064`(`csrc/kernels/kernel_param.h`;24×8064+32=193568 ≤ UB_SIZE 196608,再上一档 n_pad=8192 即越界),host 超界抛错、kernel 以布局末偏移断言兜底;
- **多 Block 并行**:`for (row = block_idx; row < m; row += block_num)`(`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`,关键字 `for (uint32_t row = block_idx`),行间轮转,low/high/out 三组缓冲 ping-pong。

### UB 内存布局(`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`)

| 缓冲 | 大小 | 用途 |
|---|---|---|
| low0/1, high0/1 | n_pad (half) | 交错输入双行的 ping-pong(low/high 在 UB 相邻,n 为 128 元素整数倍时可一次 DMA 搬两行) |
| low_fp32_buf | n_pad (float) | Y_low → fp32 |
| merged_fp32 | n_pad (float) | 合并 + bias + scale 的主计算缓冲(与 low_fp32 相邻,一次 vconv 同时填好两段) |
| bias_buf | n_pad (float) | 专家 scale_bias 行(单缓冲,行循环内每行重载) |
| out0/1 | n_pad (bfloat16) | 输出 ping-pong |
| counts_buf | ROUND_UP(numExperts×4, 32) | 本地专家 counts 切片(≤8 项) |

per-token scale 不占 UB:S 管线直接从 GM 标量地址读(`*(scale_gm + row)`,见 `csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`,关键字 `tokenScale`)。

### 流水线同步(MTE2 / V / MTE3 / S)

- 低/高行数据走标准 MTE2→V→MTE3 链,EVENT_ID0/1 ping-pong(`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`,关键字 `CopyGmToUbufAligned(low_bufs`);
- counts 在循环前一次性搬入 UB,**栅栏到 S 管线**(MTE2→S EVENT_ID0):S 管线的标量专家扫描每行都读 counts_buf,跨管线 UB 读必须显式 event flag,只栅栏到 V 不够(本次修复的 bug 之一,关键字 `PIPE_MTE2, PIPE_S, EVENT_ID0`);
- bias_buf 单缓冲,独立 EVENT_ID2 做 V↔MTE2 握手:行循环内每行无条件重载该行专家的 bias(同一专家的相邻行也重载),重载前等上一行的读者(消费 bias_buf 的 vadd)经 V→MTE2 释放缓冲,搬运完成后 MTE2→V flag 交给本行的 vadd(关键字 `PIPE_V, PIPE_MTE2, EVENT_ID2`);
- scale 标量不再走 `MTE2→S→读回` 往返,S 管线直接读 GM(见上文 UB 布局说明);
- 循环结束按轨道 drain(EVENT_ID0/1/2 的 V→MTE2 与 MTE3→V 各自 wait 清算)。

### 关键计算步骤(`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`)

1. 搬入 `merged_gm[2r]`(low)与 `merged_gm[2r+1]`(high):n 为 128 元素(VECTOR_MAX_NUM_OF_FP16)整数倍时 GM 两行连续、UB low/high 相邻,一次 `CopyGmToUbufAligned` 搬 2n 元素;否则分两次(`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`,关键字 `n % VECTOR_MAX_NUM_OF_FP16`);
2. **一次 vconv 填两段**:`vconv_f162f32(low_fp32_buf, low_bufs[event_id], fp32_rep * 2)`——low/high 两行 fp16 连续、`low_fp32_buf` 与 `merged_fp32` 两段 fp32 也相邻,repeat 翻倍(`fp32_rep * 2`)一次转换同时得到 Y_low 的 fp32 与 Y_high 的 fp32(后者直接落在 `merged_fp32`,关键字 `fp32_rep * 2`);
3. `vmuls(merged_fp32, merged_fp32, 16.0)` 原位 ×16 → `vadd(merged_fp32, merged_fp32, low_fp32_buf)` 得 `Y_high×16 + Y_low`;
4. **滚动式专家定位**:`expert_id`/`curr_expert_row` 跨行携带(循环外初始化),行号单调递增且专家按 counts 前缀和划分行轴,扫描只前进:每行从上次的 `expert_id` 起推进 `while (expert_id + 1 < numExperts && row >= curr_expert_row + counts_buf[expert_id])`,均摊 O(1)/行(替代旧版每行从头线性扫描,关键字 `expert_id`);本地 id `+ startIdx` 得全局 id,据此取 `bias_ptrs_gm[startIdx + expert_id]`;
5. 搬入该专家 `[n]` 的 scale_bias 到单缓冲 bias_buf(EVENT_ID2 握手,见流水线同步),`vadd` 加到 merged(低 nibble -8 偏移补偿);
6. S 管线直接从 GM 读 `perTokenScale[row]` 标量(`tokenScale`),`vmuls` 逐元素乘(关键字 `vmuls(merged_fp32`);
7. `vconv_f322bf16r` 转 BF16,写出 GM 第 r 行。

### 边界处理

- `pnum_tokens` 非空时 clamp m(`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`,关键字 `pnum_tokens_val`),host 当前传 null;
- n 非 128 整数倍时 `n_pad` 补齐计算,搬运按真实 `n * sizeof(half/float/bfloat16)` 字节;
- host 侧行数为奇数直接抛错(`csrc/op.cpp` 的 `XliteOpMSDMergeDequant`,关键字 `yMerged.shape[0] % 2`);Python 绑定固定 `start=0, end=numExperts`(`csrc/_C.cpp` 的 `MSDMergeDequant`,关键字 `0, numExperts`),EP 分片范围由模型侧 `XliteOpMSDMergeDequant` 直接调用时传入([csrc/model.cpp](../../../csrc/model.cpp) 的 `XModel::ForwardMoEMSD`,关键字 `XliteOpMSDMergeDequant`)。

### 片上缓冲分配与占用汇总（AIV UB）

`msd_merge_dequant`（`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`）为纯向量算子（`__DAV_C220_VEC__`，AIV 独立 kernel），全部片上缓冲在 UB 上以 `__ubuf__` 指针从偏移 0 顺序累进排布，`counts_buf` 起点额外向 32B 对齐；不使用 L1/L0（AIC 级不涉及）。UB 容量基准（`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`）：192KB（4KB×48 bank），每 AIV block 一份。dtype 固定：输入 half、计算 float、输出 bfloat16_t（实例 `msd_merge_dequant_int8_t`）。

典型配置取 `tests/kernels/msd_dequant.py` 的多专家用例 m=8192、n=4096（kN，MoE 中间维）、本地专家数 numExperts=E=3（`n_pad = ROUND_UP(n, VECTOR_MAX_NUM_OF_FP16=128) = 4096`）；单专家最大用例 `[8192, 6144]` 同一布局。占用随 n 线性、与 m/E 几乎无关（E 只影响 32B 级 counts 槽位）。

**AIV UB（每 block 一份，容量 192KB）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `low0`/`low1` | `(n_pad,)`=4096 | half | 2（事件0/1） | `n_pad*sizeof(half)` | 16384 |
| `high0`/`high1` | `(n_pad,)`=4096 | half | 2（事件0/1） | `n_pad*sizeof(half)` | 16384 |
| `low_fp32_buf` | `(n_pad,)` | float | 1 | `n_pad*sizeof(float)` | 16384 |
| `merged_fp32` | `(n_pad,)` | float | 1 | `n_pad*sizeof(float)` | 16384 |
| `bias_buf` | `(n_pad,)` | float | 1（事件2） | `n_pad*sizeof(float)` | 16384 |
| `out0`/`out1` | `(n_pad,)` | bfloat16_t | 2（事件0/1） | `n_pad*sizeof(bfloat16_t)` | 16384 |
| `counts_buf` | `(numExperts,)`，32B 对齐槽位 | uint32 | 1 | `ROUND_UP(numExperts*4, 32)` | 32 |
| **合计** | | | | | **98336** |

per-token scale 不占 UB：S 管线直接从 GM 标量地址读（`tokenScale`）；`low_fp32_buf` 与 `merged_fp32` 相邻是一次 vconv 填两段的前提，布局上成对、不额外占位。

**占用汇总**（各表占用列之和 vs 硬件容量；字节数）：

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| UB（每 AIV block） | 192 KB | 98336 B | 50.0% |

要点：

- **行布局 24B/元素是 n 上限的容量来源**：4×fp16（low/high 乒乓）+3×fp32（low_fp32/merged/bias）+2×bf16（out 乒乓）共 24B/元素，加 counts 槽位须 ≤192KB——n_pad ≤ 约 8.1K，128 对齐后实际 n ≤ 8064 才安全（n_pad 进位到 8192 时 24×8192=192KB 已容不下 counts 区）；测试注释的「n < ~12288」未随当前布局更新，单专家最大用例 n=6144 对应 147488 B（75.1%）仍在界内；
- fp32 三缓冲（`low_fp32_buf`/`merged_fp32`/`bias_buf`，共 12B/元素、占 50%）是占用大头，fp16/bf16 输入输出各占 4B/元素；
- **无显式容量断言、host 侧不校验 n 上限**（`csrc/op.cpp` 的 `XliteOpMSDMergeDequant` 只查 dtype/shape 偶数）：超限直接越界写，当前靠模型侧 kN 尺寸（n ≤ 6144 量级）保证；
- `bias_buf` 单缓冲每行重载（EVENT_ID2 独立握手）与 `counts_buf`（≤32B，本地专家数 ≤8）均不随 m/E 增长，不影响量级。
