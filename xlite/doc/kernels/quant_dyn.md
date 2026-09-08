# quant_dyn

## 功能概述

动态 per-token 量化:对 `[m, k]` 的 BF16 激活**逐行**统计绝对值最大值,在线计算量化 scale,输出 INT8 量化矩阵和每行一个 fp32 scale。计算式:

```
scale_row  = abs_max(x_row) / 127        (fp32, 输出)
x_scaled   = x_row * (127 / abs_max)
out        = int8(x_scaled)              (vconv_f162s8 饱和舍入)
```

与 `torch_npu.npu_dynamic_quant` 语义一致(测试直接与其对齐)。它是 MSD W4A8 MoE 管线第一步 "QuantDyn"([csrc/model.cpp:1461-1463](../../csrc/model.cpp)),输出供后续 unpack_activation 与 group_matmul 使用;也可独立用于 W8A8 动态量化(`csrc/model.cpp:345`)。

## 输入输出参数

Python 入口 `quant_dynamic(rt, x, scale, out)`(`csrc/_C.cpp:2740`)。kernel 签名见 `csrc/kernels/quant_dyn.h:99-104`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| x | 输入 | `[m, k]` | bfloat16 | 激活矩阵 |
| scale | 输出 | `[m]` | float32 | 每行量化 scale = absmax/127 |
| out (z) | 输出 | `[m, k]` | int8 | 量化结果 |
| pnum_tokens | 输入(可选) | `[1]` | uint32 | 动态 token 数(DP padding 的真实行数);为 null 时用 m |
| m | 标量 | - | uint32_t | 行数上限,host 传 `x.shape[0]` |
| k | 标量 | - | uint32_t | 列数,host 传 `x.shape[1]` |

测试参考:[tests/kernels/quant_dyn.py](../../tests/kernels/quant_dyn.py),覆盖 `[8192, 2048]`、`[40, 96]`、`[200000, 96]`,z 与 scale 同时与 `torch_npu.npu_dynamic_quant` 对比。

## 支持的数据类型

- 输入 `bfloat16_t` → 输出 `int8_t` + `float` scale([quant_dyn_bfloat16_t.cpp](../../csrc/kernels/quant_dyn_bfloat16_t.cpp))

host 仅接受 `x.dtype == BF16`(`csrc/op.cpp:1389`)。

## 实现原理

实现位于 [csrc/kernels/quant_dyn.h](../../csrc/kernels/quant_dyn.h),函数 `quant_bf16_to_i8`(quant_dyn.h:10-97)。

### 分块策略

- **无 k 切块**:整行一次搬入 UB(要求 k 不超 UB 容量,MoE 场景 k = hiddenSize 满足;`k_pad = ROUND_UP(k, 128)` 即按 BF16 16 字节粒度对齐,quant_dyn.h:22)。
- **多 Block 并行**:`for (row = block_idx; row < m; row += block_num)`(quant_dyn.h:45),行间轮转。行内为纯串行归约 + 量化,行间用 x/z 双缓冲 ping-pong 流水。

### UB 内存布局(quant_dyn.h:24-33)

| 缓冲 | dtype | 用途 |
|---|---|---|
| x1 / x2 | bfloat16 | 行输入 ping-pong |
| xf32 | float | BF16→FP32 转换结果(单份,行内串行使用) |
| xAbs | float | abs 中间,兼 ReduceMax 结果落点 |
| zf16 | half | FP32→FP16 中转(单份) |
| z1 / z2 | int8 | INT8 输出 ping-pong |
| scaleUb | float(1 个) | scale 标量写 GM 的中转(32B 对齐写出) |
| sum_addr | - | UB 末尾指针对齐断言 |

### 流水线同步

- 行数据链:MTE2/V 用 EVENT_ID0/ID1 ping-pong(`V→MTE2` 释放、`MTE2→V` 就绪,quant_dyn.h:48-56);INT8 结果 V→MTE3→GM→`MTE3→V` 回收 z 缓冲(quant_dyn.h:81-87)。
- scale 写出链走 **S 管线**:`ReduceMax` 后 `V→S` 同步读回 absmax 标量(quant_dyn.h:63-67),再 `S→MTE3` 把 scaleUb 单元素 `copy_ubuf_to_gm_align_b32` 写到 `scales[row]`(quant_dyn.h:68-75)。S 管线介入是因为标量读写只能走 Scalar pipe,`PIPE_MTE3→PIPE_S` 事件保证上一行 scale 写出完成后才复用 scaleUb。

### 关键计算步骤(quant_dyn.h:54-86)

1. 搬入整行 BF16,`vconv_bf162f32` 转 FP32;
2. `vabs` 取绝对值;
3. `ReduceMax(xAbs, xAbs, k)` 归约出整行 absmax(通用归约实现见 [kernel_macro.h:644](../../csrc/kernels/kernel_macro.h),二分折叠 + `vcmax`);
4. S 管线读回 `*xAbs`,标量算 `scale = absmax / 127`、`scaleRec = 127 / absmax`;
5. scale 写 GM;`vmuls` 把整行乘 scaleRec;
6. `vconv_f322f16` 转 FP16,`vconv_f162s8`(饱和舍入)转 INT8(注意此处用的是非 `a` 后缀的 `vconv_f162s8`,量化值已按 scale 压缩到 [-127, 127] 范围内,quant_dyn.h:82);
7. `copy_ubuf_to_gm_align_b8` 写回。

### 边界处理

- `pnum_tokens` 非空时 `m = min(*pnum_tokens, m)`(quant_dyn.h:17-20),跳过 DP padding 出来的无效行;
- k 非 128 整数倍时 `k_pad` 补齐做向量计算,搬运/写出仍按真实 k 字节;
- host 侧 `x.numel == 0` 直接返回(op.cpp:1380)。
