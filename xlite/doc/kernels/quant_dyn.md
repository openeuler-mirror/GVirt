# quant_dyn

## 功能概述

动态 per-token 量化:对 `[m, k]` 的 BF16 激活**逐行**统计绝对值最大值,在线计算量化 scale,输出 INT8 量化矩阵和每行一个 fp32 scale。计算式:

```
scale_row  = abs_max(x_row) / 127        (fp32, 输出)
x_scaled   = x_row * (127 / abs_max)
out        = int8(x_scaled)              (vconv_f162s8 舍入)
```

与 `torch_npu.npu_dynamic_quant` 语义一致(测试直接与其对齐)。它是 MSD W4A8 MoE 管线第一步 "QuantDyn"([csrc/model.cpp:1465-1467](../../csrc/model.cpp)),输出供后续 unpack_activation 与 group_matmul 使用;也可独立用于 W8A8 动态量化(`csrc/model.cpp:347`)。

## 输入输出参数

Python 入口 `quant_dynamic(rt, x, scale, out)`(`csrc/_C.cpp:2889`)。kernel 签名见 `csrc/kernels/quant_dyn.h:22-23`:

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

host 仅接受 `x.dtype == BF16`(`csrc/op.cpp:1457`),kernel 入口符号为 `quant_dynamic_bfloat16_t`(host 经 `aclrtlaunch_quant_dynamic_bfloat16_t` 启动,`csrc/op.cpp:1460`)。

## 实现原理

实现位于 [csrc/kernels/quant_dyn.h](../../csrc/kernels/quant_dyn.h),函数 `quant_dyn_to_i8<dtype>`(quant_dyn.h:22-231,模板;BF16 实例化见 [quant_dyn_bfloat16_t.cpp](../../csrc/kernels/quant_dyn_bfloat16_t.cpp),由 `QUANT_DYN_FUNC_DEFINE` 宏展开为入口符号 `quant_dynamic_bfloat16_t`)。

### 分块策略

- **k 维切块**:`QUANT_DYN_K_TILE = 8192`(quant_dyn.h:17),`k_loop = ceil(k / k_tile)`(quant_dyn.h:48)。整行 absmax 必须先于缩放完成,故每行分两阶段执行,按 `k_loop` 取两条路径(头部注释 quant_dyn.h:11-16):
  - **fast path**(`k_loop == 1`,即 k ≤ K_TILE):整行一次搬入 UB,x 只读一次;phase1(absmax)与 phase2(缩放/转 INT8)串行复用同一 `xf32_buf`,无需第二次 GM 读(quant_dyn.h:85-134)。
  - **tiled path**(`k_loop > 1`,即 k > K_TILE):phase1 逐 tile 做 `vabs`+`ReduceMax`,再用 `vmax` 跨 tile 合并到 `rowAbsUb`;phase2 重新从 GM 读每个 x tile 做缩放与 INT8 转换(x 不再常驻 UB,quant_dyn.h:136-223)。
- **m 维多 Block 并行**:`for (row = block_idx; row < m; row += block_num)`(quant_dyn.h:79),行间轮转。行内两阶段串行,行间用 x/z 双缓冲 ping-pong 流水。
- **对齐**:`k_pad = ROUND_UP(k_tile, 256/sizeof(dtype))`(quant_dyn.h:50,BF16 下为 128 元素 = 256B 粒度);fast path 另有 `k_pad_row = ROUND_UP(k, 256/sizeof(dtype))`(quant_dyn.h:51)用于整行向量计算。

### UB 内存布局(quant_dyn.h:54-63)

| 缓冲 | dtype | 用途 |
|---|---|---|
| x1 / x2 | bfloat16 | 行/tile 输入 ping-pong |
| xf32_buf | float | BF16→FP32 转换结果(单份,phase1/phase2 串行复用) |
| xAbs_buf | float | abs 中间,ReduceMax 结果落点;两路径均复用为 `scaleUb` 标量中转(quant_dyn.h:110、182) |
| zf16_buf | half | FP32→FP16 中转(单份) |
| z1 / z2 | int8 | INT8 输出 ping-pong |
| rowAbsUb | float(1 个) | tiled path 跨 tile absmax 归约累点(`vmax` 合并;fast path 不使用) |
| sum_addr | - | UB 末尾指针对齐断言 |

UB 容量与 ReduceMax 维度上限由两条 `static_assert` 守护(quant_dyn.h:28-36)。

### 流水线同步

每行进入前预置 4 个 ping-pong flag(`V→MTE2` ID0/ID1、`MTE3→V` ID0/ID1,quant_dyn.h:73-76),函数末尾统一 drain(quant_dyn.h:226-229)。

- **行/tile 数据链**:MTE2/V 用 `eId`(EVENT_ID0/ID1)ping-pong——搬入前 `V→MTE2` 释放 xBuf、`MTE2→V` 就绪后 `vconv`(fast path quant_dyn.h:87-96;tiled path 每阶段同此,quant_dyn.h:152-155、198-205);INT8 结果 `V→MTE3`→GM→`MTE3→V` 回收 zBuf(fast path quant_dyn.h:119-130;tiled path quant_dyn.h:212-221)。
- **scale 写出链走 S 管线**:`ReduceMax` 后 `V→S` 同步读回 absmax 标量(fast path quant_dyn.h:102-103;tiled path 跨 tile `vmax` 合并完成后 quant_dyn.h:173-174),S 管线算 `scale = absmax/127`、`scaleRec = 127/absmax`,把 `xAbs_buf` 复用为 `scaleUb` 单元素缓冲,`S→MTE3` 后 `copy_ubuf_to_gm_align_b32` 写到 `scales[row]`,再 `MTE3→S` 回收(quant_dyn.h:112-116、184-188)。S 管线介入是因为标量读写只能走 Scalar pipe,`PIPE_MTE3→PIPE_S` 事件保证 scale 写出完成后才复用 scaleUb。

### 关键计算步骤

两路径共享 phase1(absmax)/phase2(缩放)框架,差异在 absmax 的归约范围与 x 是否常驻 UB。

**fast path**(quant_dyn.h:85-134,`k_loop == 1`):

1. 整行 BF16 搬入,`convert_input`(`vconv_bf162f32`)转 FP32(quant_dyn.h:88-94);
2. `vabs` 取绝对值,`ReduceMax(xAbs_buf, xAbs_buf, k)` 归约整行 absmax(quant_dyn.h:99-101);
3. S 管线读回 `*xAbs_buf`,标量算 `scale = absmax/127`、`scaleRec = 127/absmax`(quant_dyn.h:105-107);
4. `xAbs_buf` 复用为 `scaleUb`,经 S→MTE3→`copy_ubuf_to_gm_align_b32` 写到 `scales[row]`(quant_dyn.h:110-116);
5. `vmuls` 把 `xf32_buf` 乘 scaleRec(复用同一缓冲),`vconv_f322f16` 转 FP16,`vconv_f162s8` 转 INT8(quant_dyn.h:120-124);
6. `copy_ubuf_to_gm_align_b8` 写回 GM(quant_dyn.h:128-130)。

**tiled path**(quant_dyn.h:136-223,`k_loop > 1`):

1. phase1:逐 tile 搬入 → `vabs` → `ReduceMax(xAbs_buf, xAbs_buf, k_size)` → `SetMask(1)` 后 `vmax(rowAbsUb, rowAbsUb, xAbs_buf)` 跨 tile 合并到 `rowAbsUb`(quant_dyn.h:144-172);
2. `V→S` 读回 `*rowAbsUb` 算 scale/scaleRec,写 scale 到 GM(quant_dyn.h:173-188,同 fast path);
3. phase2:逐 tile 重新读 x → `vmuls` → `vconv_f322f16` → `vconv_f162s8` 转 INT8 → 写回 GM(quant_dyn.h:190-222)。

通用归约实现见 [kernel_macro.h:658](../../csrc/kernels/kernel_macro.h)(二分折叠 + `vcmax`)。两路径均用非 `a` 后缀的 `vconv_f322f16`/`vconv_f162s8`:量化值已按 scale 压缩到 [-127, 127] 范围内,无需显式饱和(quant_dyn.h:122-124、210-213)。

### 边界处理

- `pnum_tokens` 非空时 `m = min(*pnum_tokens, m)`(quant_dyn.h:42-45),跳过 DP padding 出来的无效行;
- k 非 `256/sizeof(dtype)` 整数倍时由 `k_pad`/`k_pad_row`/`k_size_pad`(`ROUND_UP(..., 256/sizeof(dtype))`)补齐做向量计算,搬运/写出仍按真实 k(或尾部 `k_size`)字节;
- tiled path 尾部 tile(`k_size < k_tile`)按实际 `k_size` 处理(quant_dyn.h:146-149、192-195);
- host 侧 `x.numel == 0` 直接返回(op.cpp:1450)。
