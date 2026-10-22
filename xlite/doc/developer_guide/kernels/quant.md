# quant

## 功能概述

静态 per-channel 量化:对 `[m, k]` 的 BF16 激活矩阵按列(channel)施加预先标定好的 scale/offset,转成 INT8。计算式:

```
out = int8(x * scale_reciprocal + offset)
```

其中 `scale_reciprocal` 是量化 scale 的倒数(`1/scale`,由离线校准给出),`offset` 为量化偏置;host 语义上等价 `clamp(round(x/scale + offset), -128, 127)`(饱和由 `vconv_f162s8a` 的饱和舍入指令完成)。scale/offset 与激活同一 k 维按列对齐,属于 W8A8 推理管线的激活量化步骤([csrc/model.cpp](../../../csrc/model.cpp) 的 `XModel::ForwardLinear`)。

## 输入输出参数

Python 入口 `quant(rt, x, scale_reciprocal, offset, out)`(`csrc/_C.cpp:2887`)。kernel 签名见 `QUANT_FUNC_DEFINE` 宏([csrc/kernels/quant.h](../../csrc/kernels/quant.h)):

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| in (x) | 输入 | `[m, k]` | bfloat16 | 激活矩阵 |
| scale_reciprocal | 输入 | `[k]` | bfloat16 | 每列量化 scale 的倒数(1/scale) |
| offset | 输入 | `[k]` | bfloat16 | 每列量化偏置 |
| out | 输出 | `[m, k]` | int8 | 量化结果 |
| m | 标量 | - | uint32_t | 行数,host 传 `x.shape[0]` |
| k | 标量 | - | uint32_t | 列数,host 传 `x.shape[1]` |

测试参考:[tests/kernels/quant.py](../../../tests/kernels/quant.py),覆盖 `[8192, 2048]`、`[20000, 96]`、`[20000, 12288]`,与 `clamp(round(x*scale+offset))` 的 CPU 参考对齐(atol=1, rtol=1/128)。

## 支持的数据类型

- 输入 `bfloat16_t` → 输出 `int8_t`([quant_bfloat16_t.cpp](../../../csrc/kernels/quant_bfloat16_t.cpp))

host 侧仅接受 `x.dtype == BF16`,kernel 入口符号为 `quant_static_bfloat16_t`(host 经 `aclrtlaunch_quant_static_bfloat16_t` 启动,见 `csrc/op.cpp` 的 `XliteOpQuant`)。

## 实现原理

实现位于 [csrc/kernels/quant.h](../../../csrc/kernels/quant.h),函数 `quant_to_i8<dtype>`(模板;BF16 实例化见 [quant_bfloat16_t.cpp](../../../csrc/kernels/quant_bfloat16_t.cpp),由 `QUANT_FUNC_DEFINE` 宏展开为入口符号 `quant_static_bfloat16_t`)。

### 分块策略

- **对齐 k 的批量多行快速路径**（`quant_to_i8_rows`）：当 `k % 32 == 0` 且 `k ≤ QUANT_K_TILE(4096)` 且可用行数 ≥ 2 时优先走此路径——一组连续 token 共享一次 2D MTE2/MTE3 突发，rows 受 UB 预算与单向量指令上限（`r*kPad ≤ 255×64 fp32`）钳制（`csrc/kernels/quant.h` 的 `quant_to_i8`，关键字 `quant_to_i8_rows`）。
- **k 维切块（fallback）**:固定 `k_tile = 4096`(`csrc/kernels/quant.h` 的 `quant_to_i8`,关键字 `k_tile = 4096`),`k_loop = ceil(k / k_tile)`,防止 UB 溢出。scale/offset 也按同样 tile 切。
- **m 维多 Block 并行**:每个 k_tile 内,`for (row = block_idx; row < m; row += block_num)`(`csrc/kernels/quant.h` 的 `quant_to_i8`,关键字 `for (uint32_t row = block_idx`),各 AIV 轮转处理不同行。

### UB 内存布局(`csrc/kernels/quant.h` 的 `quant_to_i8`)

按 ping-pong 成对分配(k_tile 为单位):

| 缓冲 | dtype | 用途 |
|---|---|---|
| x1 / x2 | bfloat16 | 行/tile 输入 ping-pong |
| xf32 | float | BF16→FP32 转换结果(单份,phase1/phase2 串行复用) |
| xAbs | float | abs 中间,ReduceMax 结果落点;两路径均复用为 `scaleUb` 标量中转 |
| zf16 | half | FP32→FP16 中转(单份) |
| z1 / z2 | int8 | INT8 输出 ping-pong |
| rowAbsUb | float(1 个) | tiled path 跨 tile absmax 归约累点(`vmax` 合并;fast path 不使用) |
| sum_addr | - | UB 末尾指针对齐断言 |

UB 容量与 ReduceMax 维度上限由两条 `static_assert` 守护(quant.h:28-36)。

### 流水线同步

每行进入前预置 4 个 ping-pong flag(`V→MTE2` ID0/ID1、`MTE3→V` ID0/ID1,quant.h:73-76),函数末尾统一 drain(quant.h:226-229)。

- 行数据链:`V→MTE2(EVENT_ID0/1)` 释放 x 缓冲 → `copy_gm_to_ubuf_align_b16` → `MTE2→V` 就绪 → `vconv_bf162f32` 后 `V→MTE2` 再释放;量化完成 `V→MTE3` → z 缓冲写出 GM 后 `MTE3→V` 回收(`csrc/kernels/quant.h` 的 `quant_to_i8`)。
- scale/offset 链:同样以 EVENT_ID2/ID3 ping-pong,跨 tile 预取(`csrc/kernels/quant.h` 的 `quant_to_i8`,关键字 `scale_offset_event_id`)。

行间 `event_id = 1 - event_id` 翻转,搬入下一行与写出上一行重叠。收尾统一 wait + `pipe_barrier(PIPE_ALL)`(`csrc/kernels/quant.h` 的 `quant_to_i8`,收尾 drain)。

### 关键计算步骤(`csrc/kernels/quant.h` 的 `quant_to_i8`)

1. `copy_gm_to_ubuf_align_b16` 搬入 `k_size` 个 BF16(按字节长度,b16 对齐变体);
2. `vconv_bf162f32`:BF16 → FP32(注意 repeat 以较宽 dtype 计,`k_size_pad / VECTOR_MAX_NUM_OF_FP32`,见 `csrc/kernels/quant.h` 的 `quant_to_i8`,关键字 `k_repeats`);
3. `vmul` 乘 scale_reciprocal(fp32),`vadd` 加 offset(fp32);
4. `vconv_f322f16a`:FP32 → FP16;
5. `vconv_f162s8a`:FP16 → INT8(`a` 后缀为饱和舍入变体,负溢出/正溢出截到 [-128, 127],替代显式 clamp+round);
6. `copy_ubuf_to_gm_align_b8` 写回 GM INT8。

### 边界处理

- 尾部 tile(`k_size < k_tile`):搬运按实际 `k_size` 字节;计算侧 `k_size_pad = ROUND_UP(k_size, VECTOR_MAX_NUM_OF_BF16)` 补齐到向量指令粒度(`csrc/kernels/quant.h` 的 `quant_to_i8`,关键字 `k_size_pad = ROUND_UP`)。
- m 不足 block_num 时循环自然跳过空闲核。
- host 侧 `x.numel == 0` 直接返回(op.cpp:1450)。
- host 侧拒绝 `k % 32 != 0`(ragged k),kernel 内 `assert(this->k % 32 == 0)` 守护。
