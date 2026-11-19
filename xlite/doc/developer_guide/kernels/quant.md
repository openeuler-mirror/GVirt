# quant

## 功能概述

静态 per-channel 量化:对 `[m, k]` 的 BF16 激活矩阵按列(channel)施加预先标定好的 scale/offset,转成 INT8。计算式:

```text
out = int8(x * scale_reciprocal + offset)
```

其中 `scale_reciprocal` 是量化 scale 的倒数(`1/scale`,由离线校准给出),`offset` 为量化偏置;host 语义上等价 `clamp(round(x/scale + offset), -128, 127)`(饱和由 `vconv_f162s8r` 的饱和舍入指令完成)。scale/offset 与激活同一 k 维按列对齐,属于 W8A8 推理管线的激活量化步骤([csrc/model.cpp](../../../csrc/model.cpp) 的 `XModel::ForwardLinear`)。与 [`quant_dyn`](quant_dyn.md) 的区别:scale/offset 是离线标定的输入而非逐行在线统计,故无 absmax 归约、不输出 per-row scale。

## 输入输出参数

Python 入口 `quant(rt, x, scale_reciprocal, offset, out)`(`csrc/_C.cpp` 的 `Quant`)。kernel 签名见 `QUANT_FUNC_DEFINE` 宏([csrc/kernels/quant.h](../../../csrc/kernels/quant.h)):

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

实现位于 [csrc/kernels/quant.h](../../../csrc/kernels/quant.h),类模板 `Quant<dtype>`(BF16 实例化见 [quant_bfloat16_t.cpp](../../../csrc/kernels/quant_bfloat16_t.cpp),由 `QUANT_FUNC_DEFINE` 宏展开为入口符号 `quant_static_bfloat16_t`,入口依次调用 `Init()` + `Run()`)。类接口与 `Dequant` 一致,可独立启动也可作为融合算子的子算子:

- `Init()`:原子/掩码设置;
- `TaskTilesInit(in, scaleReciprocal, offset, out, m, k, taskRatio=1)`:绑定 GM 地址,按 `tileRows = ceil(m / GetBlockNum()*taskRatio)` 推导 tile 几何,据 `tileRows == 1` 选择单行/多行路径,返回 tile 数;
- `SetFlags()`/`WaitFlags()`:预置与排空事件 flag(单行/多行各一套);
- `RunTileByIdx(idx)`:按 tile 索引分发,内部据 `isSingleRow` 走 `RunTileByIdxSingleRow` 或 `RunTileByIdxMultiRow`;
- `Run(in, scaleReciprocal, offset, out, m, k)`:独立入口,`TaskTilesInit` → `SetFlags` → grid-stride `RunTileByIdx` → `WaitFlags`。

### 分块策略

- **k 维切块**:`maxOneRowKTile = ROUND_DOWN(UB_SIZE/24, VECTOR_REPEAT_BYTESIZE/sizeof(dtype))`([csrc/kernels/quant.h](../../../csrc/kernels/quant.h) 的 `Quant::TaskTilesInit`,关键字 `maxOneRowKTile`),`kLoop = ceil(k / maxOneRowKTile)`;`kPad = ROUND_UP(ceil(k/kLoop), align)`(`align = VECTOR_REPEAT_BYTESIZE/sizeof(dtype)`),整 tile 按 `kPad` 宽度做向量计算,尾部按实际 `kSize` 搬运/写出。
- **m 维多 Block 并行**:`Run` 中 `for (idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum())`([csrc/kernels/quant.h](../../../csrc/kernels/quant.h) 的 `Quant::Run`);每 tile 处理 `tileRows` 行,多行路径 tile 内再按 `rows`(UB 容量上限)分组。
- **rows 适配(多行路径)**:`rMax = MAX(1, (UB_SIZE - 12*kPad) / (12*kPad))`,`rows = MIN(rMax, tileRows)`([csrc/kernels/quant.h](../../../csrc/kernels/quant.h) 的 `Quant::TaskTilesInitMultiRow`,关键字 `rMax`)。

### UB 内存布局

**多行路径**(`TaskTilesInitMultiRow`),按 `rows × kPad` ping-pong([csrc/kernels/quant.h](../../../csrc/kernels/quant.h) 的 `Quant::TaskTilesInitMultiRow`):

| 缓冲 | dtype | 用途 |
|---|---|---|
| x1 / x2 | bfloat16 | 行输入 ping-pong(每份 `rows*kPad`) |
| xf32 | float | BF16→FP32 转换结果(单份,`rows*kPad`) |
| xf16 | half | FP32→FP16(ODD)中转(单份) |
| z1 / z2 | int8 | INT8 输出 ping-pong |
| scale / offset | bfloat16 | 每段 scale/offset(`kPad`,所有行共享,不需 ping-pong) |
| scale_f32 / offset_f32 | float | scale/offset 的 FP32(`kPad`) |

`xf16` 排在 `z1` 之前,使 `vconv` 硬件 overread 落在 `z1`/固定缓冲的 padding 上。UB 总占用 `12*(rows+1)*kPad`,`rMax` 保证 ≤ `UB_SIZE`(`assert`)。

**单行路径**(`tileRows == 1`,`TaskTilesInitSingleRow`):无 ping-pong、无多行数组,单份缓冲 `s_x`/`s_xf32`/`s_z`/`s_xf16`/`s_scale`/`s_offset`/`s_scale_f32`/`s_offset_f32`,总占用 `21*kPad`(< 多行下限 `24*kPad`)。

### 数据流与流水线同步

**多行路径**(`RunTileByIdxMultiRow`,[csrc/kernels/quant.h](../../../csrc/kernels/quant.h) 的 `Quant::RunTileByIdxMultiRow`):外层按 `kLoop` 切 k,内层按 `rows` 分组处理 `[base, base+r)`:

1. **scale/offset 载入**:`wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2)` → `copy_gm_to_ubuf_align_b16` 搬 scale/offset → `set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2)` → V 侧 `convert_input` 升 FP32 → `pipe_barrier(PIPE_V)` → `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2)`(scale/offset 缓冲可被下一 k 段复用)。
2. **x 载入(乒乓)**:`wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0+xId)` → 2D block DMA 搬 `r` 行 → `set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0+xId)` → `convert_input` 升 FP32 → `pipe_barrier` → `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0+xId)`(xBuf 可被下一组复用),`xId = 1-xId`。
3. **量化计算**:逐行 `vmul(row, row, scale_f32)` → `pipe_barrier` → `vadd(row, row, offset_f32)` → `pipe_barrier`。
4. **cast 链 + 写回**:`vconv_f322f16o(xf16, xf32)`(ODD 舍入)→ `pipe_barrier` → `wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0+eventId)` → `vconv_f162s8r(zBufs[eventId], xf16)`(RINT 饱和舍入)→ `set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0+eventId)` → `wait_flag(PIPE_V, PIPE_MTE3, ...)` → `copy_ubuf_to_gm` 2D burst 写回 → `set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0+eventId)`,`eventId = 1-eventId`。

**单行路径**(`RunTileByIdxSingleRow`):结构同上但单缓冲、无乒乓、无逐行循环——每 k 段一次性搬 1 行 x、`vmul`+`vadd` 单次、cast 链单次、写回单行;事件号 `EVENT_ID0`(x/z)与 `EVENT_ID2`(scale/offset)各一套。

### 关键计算步骤

静态量化无 absmax 归约,核心是 `vmul` + `vadd` + cast 链([csrc/kernels/quant.h](../../../csrc/kernels/quant.h) 的 `Quant::RunTileByIdxMultiRow`/`RunTileByIdxSingleRow`):

1. `convert_input`(BF16→FP32);
2. `vmul(xf32, xf32, scale_f32)`(乘 scale 倒数);
3. `vadd(xf32, xf32, offset_f32)`(加偏置);
4. `vconv_f322f16o`(FP32→FP16,ODD 舍入);
5. `vconv_f162s8r`(FP16→INT8,RINT 舍入 + 饱和到 [-128,127])。

cast 链用 `f322f16o`(ODD)→`f162s8r`(RINT)两步:ODD 舍入使 FP16 值停在 k.5 的同侧,`RINT(ODD(x)) == RINT(x)`,数值等价于旧的三步链 `f322int16(RINT)→int162fp16(ROUND)→fp162int8(RINT)` 但少一次 vconv([csrc/kernels/quant.h](../../../csrc/kernels/quant.h) 的 `Quant::RunTileByIdxMultiRow`,关键字 `ODD rounding` 注释)。

### 边界处理

- 尾部 k 段(`kSize < kPad`)按实际 `kSize` 搬运与写出,计算侧仍按 `kPad` 宽度做向量运算(超出 `kSize` 的 stale lane 不参与 store)。
- tile 越界(`rowStart >= m`)直接 return([csrc/kernels/quant.h](../../../csrc/kernels/quant.h) 的 `Quant::RunTileByIdxMultiRow`,关键字 `skip when tile out of range`)。
- m 不足 block_num 时循环自然跳过空闲核。
- host 侧拒绝 `k % 32 != 0`(ragged k,2D DMA 步长要求 32B 对齐),kernel 内 `assert(this->k % 32 == 0)` 守护([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpQuant`,关键字 `must be a multiple of 32`)。

### 片上缓冲分配与占用汇总(AIV UB)

本算子为纯向量(AIV)算子,只使用 UB 一级片上缓冲,无 L1/L0A/L0B/L0C 占用。`TaskTilesInit` 按运行期 `kPad`(`ROUND_UP(ceil(k/kLoop), 128)`)推导布局:多行路径(`csrc/kernels/quant.h` 的 `Quant::TaskTilesInitMultiRow`)在 UB 偏移 0 处按 `x1 → x2 → xf32 → xf16 → z1 → z2 → scale → offset → scale_f32 → offset_f32` 顺序累进指针排布,`xf16` 特意排在 `z1` 之前吸收 `vconv` 硬件 overread;单行路径(`Quant::TaskTilesInitSingleRow`)按 `s_x → s_xf32 → s_z → s_xf16 → s_scale → s_offset → s_scale_f32 → s_offset_f32` 单份排布。硬件容量基准(910B/dav C220,NpuArch 2201):UB 预算取 `kernel_macro.h` 的 `UB_SIZE=192KB`(4KB×48 bank),每个 AIV subblock 各持一份独立 UB。dtype 记法:模板实参 `dtype` 唯一实例化为 `bfloat16_t`(2B,见 [quant_bfloat16_t.cpp](../../../csrc/kernels/quant_bfloat16_t.cpp)),cast 链中转 half 2B、float 4B、int8 1B,单条 dtype 路径无需分列。

典型配置取 k=4096(W8A8 线性层 hiddenSize,即测试注释的 `kPad=4096, rows=3` 引擎边界档):`kLoop=1`、`kPad=4096`;多行路径 `rMax = (UB_SIZE - 12*kPad)/(12*kPad) = 3`,行数足够(tileRows ≥ 3)时 `rows=3`;单行路径(tileRows == 1,小 batch decode)同取 `kPad=4096`。两条路径互斥,占用差异仅来自乒乓份数。

**AIV UB(预算 `UB_SIZE=192KB`)——多行路径**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B,kPad=4096/rows=3) |
|---|---|---|---|---|---|
| `x1`/`x2` | `(rows, kPad)` | bfloat16 | 2(事件0/1,xId) | `rows*kPad*sizeof(dtype)` | 49152 |
| `xf32` | `(rows, kPad)` | float | 1 | `rows*kPad*sizeof(float)` | 49152 |
| `xf16` | `(rows, kPad)` | half | 1 | `rows*kPad*sizeof(half)` | 24576 |
| `z1`/`z2` | `(rows, kPad)` | int8 | 2(事件0/1,eventId) | `rows*kPad*sizeof(int8_t)` | 24576 |
| `scale` | `(kPad,)` | bfloat16 | 1(事件2) | `kPad*sizeof(dtype)` | 8192 |
| `offset` | `(kPad,)` | bfloat16 | 1(事件2) | `kPad*sizeof(dtype)` | 8192 |
| `scale_f32` | `(kPad,)` | float | 1 | `kPad*sizeof(float)` | 16384 |
| `offset_f32` | `(kPad,)` | float | 1 | `kPad*sizeof(float)` | 16384 |
| **合计** | | | | `12*(rows+1)*kPad` | **196608** |

**AIV UB(预算 `UB_SIZE=192KB`)——单行路径(tileRows == 1)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B,kPad=4096) |
|---|---|---|---|---|---|
| `s_x` | `(kPad,)` | bfloat16 | 1(事件0) | `kPad*sizeof(dtype)` | 8192 |
| `s_xf32` | `(kPad,)` | float | 1 | `kPad*sizeof(float)` | 16384 |
| `s_z` | `(kPad,)` | int8 | 1(事件0) | `kPad*sizeof(int8_t)` | 4096 |
| `s_xf16` | `(kPad,)` | half | 1 | `kPad*sizeof(half)` | 8192 |
| `s_scale` | `(kPad,)` | bfloat16 | 1(事件2) | `kPad*sizeof(dtype)` | 8192 |
| `s_offset` | `(kPad,)` | bfloat16 | 1(事件2) | `kPad*sizeof(dtype)` | 8192 |
| `s_scale_f32` | `(kPad,)` | float | 1 | `kPad*sizeof(float)` | 16384 |
| `s_offset_f32` | `(kPad,)` | float | 1 | `kPad*sizeof(float)` | 16384 |
| **合计** | | | | `21*kPad` | **86016** |

**占用汇总**(两条路径互斥,各对 UB 容量;字节数):

| 存储级 | 硬件容量 | 多行路径占用 | 占比 | 单行路径占用 | 占比 |
|---|---|---|---|---|---|
| UB(每 AIV subblock) | 192 KB | 196608 B | 100% | 86016 B | 43.8% |

要点:

- **UB 是唯一也是顶满的一级**:多行路径 k=4096 时 `12*(rows+1)*kPad` = 196608 B 恰好占满 192KB——`rMax = (UB_SIZE - 12*kPad)/(12*kPad)` 的推导保证 12*(rows+1)*kPad ≤ `UB_SIZE`,这是 `rows` 行组上限的容量来源,`Quant::TaskTilesInitMultiRow` 末尾的 `assert(offset_f32 + kPad <= UB_SIZE)` 兜底;
- `x1`/`x2` 与 `xf32` 各 49152 B(各占 25%)是占用大头,行输入乒乓与 FP32 升精度缓冲合计占 50%;
- `kPad` 的切块上限 `maxOneRowKTile = ROUND_DOWN(UB_SIZE/24, 128) = 8192` 保证 k > 8192 时裂成 kLoop 段后 `kPad ≤ 8192`、12*(rows+1)*kPad 仍不越界,即任意合法 k 的 UB 占用均由运行期公式自适应,host 侧无需容量校验(`k % 32 == 0` 是 2D DMA 步长约束而非容量约束);
- 单行路径合计 `21*kPad`(86016 B,43.8%)恒小于多行路径下限 `24*kPad`(rows=1 时),路径选择天然安全。
