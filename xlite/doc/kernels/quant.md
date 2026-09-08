# quant

## 功能概述

静态 per-channel 量化:对 `[m, k]` 的 BF16 激活矩阵按列(channel)施加预先标定好的 scale/offset,转成 INT8。计算式:

```
out = int8(x * scale_reciprocal + offset)
```

其中 `scale_reciprocal` 是量化 scale 的倒数(`1/scale`,由离线校准给出),`offset` 为量化偏置;host 语义上等价 `clamp(round(x/scale + offset), -128, 127)`(饱和由 `vconv_f162s8a` 的饱和舍入指令完成)。scale/offset 与激活同一 k 维按列对齐,属于 W8A8 推理管线的激活量化步骤([csrc/model.cpp:352](../../csrc/model.cpp))。

## 输入输出参数

Python 入口 `quant(rt, x, scale_reciprocal, offset, out)`(`csrc/_C.cpp:2738`)。kernel 签名见 `csrc/kernels/quant.h:153-158`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| in (x) | 输入 | `[m, k]` | bfloat16 | 激活矩阵 |
| scale_reciprocal | 输入 | `[k]` | bfloat16 | 每列量化 scale 的倒数(1/scale) |
| offset | 输入 | `[k]` | bfloat16 | 每列量化偏置 |
| out | 输出 | `[m, k]` | int8 | 量化结果 |
| m | 标量 | - | uint32_t | 行数,host 传 `x.shape[0]` |
| k | 标量 | - | uint32_t | 列数,host 传 `x.shape[1]` |

测试参考:[tests/kernels/quant.py](../../tests/kernels/quant.py),覆盖 `[8192, 2048]`、`[20000, 96]`、`[20000, 12288]`,与 `clamp(round(x*scale+offset))` 的 CPU 参考对齐(atol=1, rtol=1/128)。

## 支持的数据类型

- 输入 `bfloat16_t` → 输出 `int8_t`([quant_bfloat16_t.cpp](../../csrc/kernels/quant_bfloat16_t.cpp))

host 侧仅接受 `x.dtype == BF16`(`csrc/op.cpp:1369`),算子名即 `quant_bf16_to_i8_static`。

## 实现原理

实现位于 [csrc/kernels/quant.h](../../csrc/kernels/quant.h),函数 `quant_bf16_to_i8`(quant.h:13-151)。

### 分块策略

- **k 维切块**:固定 `k_tile = 4096`(quant.h:27),`k_loop = ceil(k / k_tile)`,防止 UB 溢出。scale/offset 也按同样 tile 切。
- **m 维多 Block 并行**:每个 k_tile 内,`for (row = block_idx; row < m; row += block_num)`(quant.h:96),各 AIV 轮转处理不同行。

### UB 内存布局(quant.h:30-51)

按 ping-pong 成对分配(k_tile 为单位):

| 缓冲 | dtype | 用途 |
|---|---|---|
| x_ping / x_pong | bfloat16 | 激活行数据 |
| xf32_ping / xf32_pong | float | BF16→FP32 中间 |
| xf16_ping / xf16_pong | half | FP32→FP16 中间(量化前中转) |
| z_ping / z_pong | int8 | INT8 结果 |
| scale/offset(_fp32)_ping/pong | bfloat16 / float | 每 tile 的 scale/offset 原值及 fp32 转换 |

scale/offset 使用独立的 ping-pong(事件 EVENT_ID2/ID3),与行数据的事件(ID0/ID1)解耦:进入新 k_tile 时,当前 tile 的 scale/offset 在所有行间复用,同时预取下一 tile 的 scale/offset。

### 流水线同步(MTE2 / V / MTE3)

三管线事件协议,两组事件ID:

- 行数据链:`V→MTE2(EVENT_ID0/1)` 释放 x 缓冲 → `copy_gm_to_ubuf_align_b16` → `MTE2→V` 就绪 → `vconv_bf162f32` 后 `V→MTE2` 再释放;量化完成 `V→MTE3` → z 缓冲写出 GM 后 `MTE3→V` 回收(quant.h:99-135)。
- scale/offset 链:同样以 EVENT_ID2/ID3 ping-pong,跨 tile 预取(quant.h:80-93、140-141)。

行间 `event_id = 1 - event_id` 翻转,搬入下一行与写出上一行重叠。收尾统一 wait + `pipe_barrier(PIPE_ALL)`(quant.h:144-150)。

### 关键计算步骤(quant.h:104-135)

1. `copy_gm_to_ubuf_align_b16` 搬入 `k_size` 个 BF16(按字节长度,b16 对齐变体);
2. `vconv_bf162f32`:BF16 → FP32(注意 repeat 以较宽 dtype 计,`k_size_pad / VECTOR_MAX_NUM_OF_FP32`,quant.h:106-107);
3. `vmul` 乘 scale_reciprocal(fp32),`vadd` 加 offset(fp32);
4. `vconv_f322f16a`:FP32 → FP16;
5. `vconv_f162s8a`:FP16 → INT8(`a` 后缀为饱和舍入变体,负溢出/正溢出截到 [-128, 127],替代显式 clamp+round);
6. `copy_ubuf_to_gm_align_b8` 写回 GM INT8。

### 边界处理

- 尾部 tile(`k_size < k_tile`):搬运按实际 `k_size` 字节;计算侧 `k_size_pad = ROUND_UP(k_size, VECTOR_MAX_NUM_OF_BF16)` 补齐到向量指令粒度(quant.h:77-78)。
- m 不足 block_num 时循环自然跳过空闲核。
