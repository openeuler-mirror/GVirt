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

实现位于 [csrc/kernels/quant.h](../../csrc/kernels/quant.h),类模板 `Quant<dtype>`(BF16 实例化见 [quant_bfloat16_t.cpp](../../csrc/kernels/quant_bfloat16_t.cpp),由 `QUANT_FUNC_DEFINE` 宏展开为入口符号 `quant_static_bfloat16_t`,入口依次调用 `Init()` + `Run()`)。类接口与 `Dequant` 一致,可独立启动也可作为融合算子的子算子:

- `Init()`:原子/掩码设置与 UB ping-pong 布局;
- `SetFlags()`/`WaitFlags()`:预置与排空事件 flag;
- `TaskTilesInit(in, scaleReciprocal, offset, out, m, k)`:绑定 GM 地址,返回行 tile 数;
- `RunTileByIdx(idx)`:独立启动的 tile 分发;
- `RunRows(rowStart, rowCount)`:量化 `[rowStart, rowStart+rowCount)` 连续行段,核心原语——独立分发与融合算子共用;
- `CanBatchRows(rowCount)`/`RunRowsBatch(rowStart, rowCount)`:批量多行段入口(gate + 自包含执行)。**自带 flag 预挂/排空,不得与 SetFlags/WaitFlags 混用**;
- `Run(...)`:独立入口,优先走批量多行 fast path,否则 SetFlags → tile 分发 → WaitFlags。

### 分块策略

- **k 维切块**:`K_TILE_MAX = ROUND_DOWN(UB_SIZE/16, VECTOR_REPEAT_BYTESIZE / sizeof(dtype))`(quant.h:20),`k_loop = ceil(k / K_TILE_MAX)`。scale/offset 也按同样 tile 切。
- **m 维多 Block 并行**:`for (row = block_idx; row < m; row += block_num)`(quant.h:79),行间轮转。行内两阶段串行,行间用 x/z 双缓冲 ping-pong 流水。
- **rows 适配**:`rows = MIN(rMax, (uint32_t)this->tileRows)`,其中 `rMax = (UB_SIZE - 12*kPad) / (12*kPad)`;UB 公式为 `12*R*kPad + 4*kPad + 40*R`,rMax 保证 ≤ UB_SIZE。

### UB 内存布局(quant.h:54-63)

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

- **行/tile 数据链**:MTE2/V 用 `eId`(EVENT_ID0/ID1)ping-pong。2D block DMA 一次 burst 搬运整组 `[r, kSize]`,x 只读一次。phase1(absmax)与 phase2(缩放/转 INT8)串行复用同一 `xf32_buf`,无需第二次 GM 读。
- **scale 写出链走 S 管线**:`ReduceMax` 后 `V→S` 同步读回 absmax 标量,S 管线算 `scale = absmax/127`、`scaleRec = 127/absmax`,把 `xAbs_buf` 复用为 `scaleUb` 单元素缓冲,`S→MTE3` 后 `copy_ubuf_to_gm_align_b32` 写到 `scales[row]`,再 `MTE3→S` 回收。

### 关键计算步骤

两路径共享 phase1(absmax)/phase2(缩放)框架,差异在 absmax 的归约范围与 x 是否常驻 UB。

**fast path**(`k_loop == 1`):

1. 整行 BF16 搬入,`copy_gm_to_ubuf`(2D block DMA,`nB=r, lenBurst=kSize*sizeof(dtype)/32, srcGap=..., dstGap=...`)转 FP32;
2. `vabs` 取绝对值,`ReduceMax(xAbs_buf, xAbs_buf, k)` 归约整行 absmax;
3. S 管线读回 `*xAbs_buf`,标量算 `scale = absmax/127`、`scaleRec = 127/absmax`;
4. `xAbs_buf` 复用为 `scaleUb`,经 S→MTE3→`copy_ubuf_to_gm_align_b32` 写到 `scales[row]`;
5. `vmuls` 把 `xf32_buf` 乘 scaleRec(复用同一缓冲),`vconv_f322f16` 转 FP16,`vconv_f162s8` 转 INT8;
6. `copy_ubuf_to_gm`(2D block DMA)写回 GM。

### 边界处理

- 尾部 tile(`k_size < k_pad`)按实际 `k_size` 处理,计算侧仍按 `kPad` 宽度做向量运算(超出 `k_size` 的 stale lane 不参与 `ReduceMax` 和 store)。
- m 不足 block_num 时循环自然跳过空闲核。
- host 侧 `x.numel == 0` 直接返回(op.cpp:1450)。
- host 侧拒绝 `k % 32 != 0`(ragged k),kernel 内 `assert(this->k % 32 == 0)` 守护。
