# msd_merge_dequant

## 功能概述

MSD W4A8 MoE 管线的收尾算子(Step 3):对 INT4×INT4 group_matmul 产出的中间结果做合并、低 nibble 偏置补偿和 per-token 反量化,一步算出最终的 BF16 输出:

```
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

测试参考:[tests/kernels/msd_dequant.py](../../../tests/kernels/msd_dequant.py),单专家覆盖 `[8, 96]`、`[40, 512]`、`[200, 2048]`、`[1, 2048]`、`[8192, 6144]`;多专家覆盖 `(200, 4096, E=4 均分)`、`(200, 4096, E=8 倾斜 [80,40,20,18,15,12,10,5])`、`(8192, 4096, E=3 [4096,3072,1024])`(倾斜分片迫使每个 block 的跨步行序列内频繁切换专家),另有 10 次重复 launch 的 bit 级确定性验证(压测 bias_buf 跨专家行复用/重载时 MTE2 覆写与 V 读的交错);全部与纯 torch 参考对齐。

## 支持的数据类型

- `int8_t` 实例(算子语义为 fp16 输入 → bf16 输出;[msd_merge_dequant_int8_t.cpp](../../../csrc/kernels/msd_merge_dequant_int8_t.cpp))

host 校验 `yMerged.dtype == FP16 && perTokenScale.dtype == FP32 && out.dtype == BF16`,且 yMerged 为 2D、行数为偶数(`csrc/op.cpp` 的 `XliteOpMSDMergeDequant`,关键字 `yMerged.shape[0] % 2`)。

## 实现原理

实现位于 [csrc/kernels/msd_merge_dequant.h](../../../csrc/kernels/msd_merge_dequant.h),函数 `msd_merge_dequant`。

### 分块与多 Block 并行

- **无 n 切块**:整行一次搬入,`n_pad = ROUND_UP(n, 128)`;测试注释明确 n < ~12288 以内受 UB 单行布局约束(`tests/kernels/msd_dequant.py` 的 `test_cases` 注释,关键字 `n < ~12288`);
- **多 Block 并行**:`for (row = block_idx; row < m; row += block_num)`(`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`,关键字 `for (uint32_t row = block_idx`),行间轮转,low/high/out 三组缓冲 ping-pong。

### UB 内存布局(`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`)

| 缓冲 | 大小 | 用途 |
|---|---|---|
| low0/1, high0/1 | n_pad (half) | 交错输入双行的 ping-pong(low/high 在 UB 相邻,n 为 128 元素整数倍时可一次 DMA 搬两行) |
| low_fp32_buf | n_pad (float) | Y_low → fp32 |
| merged_fp32 | n_pad (float) | 合并 + bias + scale 的主计算缓冲(与 low_fp32 相邻,一次 vconv 同时填好两段) |
| bias_buf | n_pad (float) | 专家 scale_bias 行(单缓冲,按需重载) |
| out0/1 | n_pad (bfloat16) | 输出 ping-pong |
| counts_buf | ROUND_UP(numExperts×4, 32) | 本地专家 counts 切片(≤8 项) |

per-token scale 不占 UB:S 管线直接从 GM 标量地址读(`*(scale_gm + row)`,见 `csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`,关键字 `tokenScale`)。

### 流水线同步(MTE2 / V / MTE3 / S)

- 低/高行数据走标准 MTE2→V→MTE3 链,EVENT_ID0/1 ping-pong(`csrc/kernels/msd_merge_dequant.h` 的 `msd_merge_dequant`,关键字 `CopyGmToUbufAligned(low_bufs`);
- counts 在循环前一次性搬入 UB,**栅栏到 S 管线**(MTE2→S EVENT_ID0):S 管线的标量专家扫描每行都读 counts_buf,跨管线 UB 读必须显式 event flag,只栅栏到 V 不够(本次修复的 bug 之一,关键字 `PIPE_MTE2, PIPE_S, EVENT_ID0`);
- bias_buf 单缓冲,独立 EVENT_ID2 做 V↔MTE2 握手:重载前等上一专家周期最后一个读者(vadd)经 V→MTE2 释放缓冲,搬运完成后 MTE2→V flag 交给本行的 vadd(关键字 `PIPE_V, PIPE_MTE2, EVENT_ID2`);
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
