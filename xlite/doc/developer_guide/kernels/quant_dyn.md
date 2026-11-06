# quant_dyn

## 功能概述

动态 per-token 量化:对 `[m, k]` 的 BF16 激活**逐行**统计绝对值最大值,在线计算量化 scale,输出 INT8 量化矩阵和每行一个 fp32 scale。计算式:

```text
scale_row  = abs_max(x_row) / 127        (fp32, 输出)
x_scaled   = x_row * (127 / abs_max)
out        = int8(x_scaled)              (vconv_f162s8r 舍入)
```

与 `torch_npu.npu_dynamic_quant` 语义一致(测试直接与其对齐)。它是 MSD W4A8 MoE 管线第一步 "QuantDyn"([csrc/model.cpp](../../../csrc/model.cpp) 的 `XModel::ForwardMoEMSD`,关键字 `XliteOpQuantDyn`),输出供后续 unpack_activation 与 group_matmul 使用;也可独立用于 W8A8 动态量化(`csrc/model.cpp` 的 `XModel::ForwardLinear`,DYNAMIC_QUANT 分支)。与 [`quant`](quant.md) 的区别:scale 在线逐行统计而非离线标定,故需 absmax 归约并输出 per-row scale。

## 输入输出参数

Python 入口 `quant_dynamic(rt, x, scale, out)`(`csrc/_C.cpp` 的 `QuantDyn`)。kernel 签名见 `QUANT_DYN_FUNC_DEFINE` 宏([csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h)):

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| x | 输入 | `[m, k]` | bfloat16 | 激活矩阵 |
| scale | 输出 | `[m]` | float32 | 每行量化 scale = absmax/127 |
| out (z) | 输出 | `[m, k]` | int8 | 量化结果 |
| pnum_tokens | 输入(可选) | `[1]` | uint32 | 动态 token 数(DP padding 的真实行数);为 null 时用 m |
| m | 标量 | - | uint32_t | 行数上限,host 传 `x.shape[0]` |
| k | 标量 | - | uint32_t | 列数,host 传 `x.shape[1]` |

测试参考:[tests/kernels/quant_dyn.py](../../../tests/kernels/quant_dyn.py),覆盖 `[8192, 2048]`、`[40, 96]`、`[200000, 96]`,z 与 scale 同时与 `torch_npu.npu_dynamic_quant` 对比。

## 支持的数据类型

- 输入 `bfloat16_t` → 输出 `int8_t` + `float` scale([quant_dyn_bfloat16_t.cpp](../../../csrc/kernels/quant_dyn_bfloat16_t.cpp))

host 仅接受 `x.dtype == BF16`,kernel 入口符号为 `quant_dynamic_bfloat16_t`(host 经 `aclrtlaunch_quant_dynamic_bfloat16_t` 启动,见 `csrc/op.cpp` 的 `XliteOpQuantDyn`)。

## 实现原理

实现位于 [csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h),类模板 `QuantDyn<dtype>`(BF16 实例化见 [quant_dyn_bfloat16_t.cpp](../../../csrc/kernels/quant_dyn_bfloat16_t.cpp),由 `QUANT_DYN_FUNC_DEFINE` 宏展开为入口符号 `quant_dynamic_bfloat16_t`,入口依次调用 `Init()` + `Run()`)。继承 `Quant` 的接口风格,可独立启动也可作为融合算子的子算子:

- `Init()`:原子/掩码设置;
- `TaskTilesInit(x, scales, z, pnumTokens, m, k, taskRatio=1)`:绑定 GM 地址,`pnumTokens` 非空时 `m = min(*pnumTokens, m)` 收缩,推导 `tileRows` 并据 `tileRows == 1` 选单行/多行路径,返回 tile 数;
- `SetFlags()`/`WaitFlags()`:单行/多行各一套 flag;
- `RunTileByIdx(idx)`:据 `isSingleRow` 走 `RunTileByIdxSingleRow` 或 `RunTileByIdxMultiRow`;
- `Run(x, scale, out, pnumTokens, m, k)`:独立入口,`TaskTilesInit` → `SetFlags` → grid-stride → `WaitFlags`。

### 分块策略

- **k 维切块**:`maxOneRowKTile = ROUND_DOWN((UB_SIZE-32)/16, VECTOR_REPEAT_BYTESIZE/sizeof(dtype))`([csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h) 的 `QuantDyn::TaskTilesInit`,关键字 `maxOneRowKTile`),`kLoop = ceil(k / maxOneRowKTile)`,`kPad = ROUND_UP(ceil(k/kLoop), align)`。整行 absmax 必须先于缩放完成,故每行分 phase1(absmax)/phase2(缩放)两阶段执行;`kLoop == 1` 时 phase1 的 `xf32` 常驻 UB,phase2 直接复用、不重读 x,`kLoop > 1` 时 phase2 逐段重载。
- **m 维多 Block 并行**:`Run` 中 `for (idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum())`([csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h) 的 `QuantDyn::Run`);多行路径 tile 内再按 `rows` 分组。
- **rows 适配(多行路径)**:`rowMax = MAX(1, (UB_SIZE - 4*kPad) / (12*kPad + 40))`(40*R 为 rowAbs + scaleStage),`repeatMax = VECTOR_MAX_REPEAT * VECTOR_MAX_NUM_OF_FP32 / kPad`,`rows = MIN(MIN(rowMax, repeatMax), tileRows)`([csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h) 的 `QuantDyn::TaskTilesInitMultiRow`,关键字 `rowMax`/`repeatMax`)。

### UB 内存布局

**多行路径**(`TaskTilesInitMultiRow`,[csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h) 的 `QuantDyn::TaskTilesInitMultiRow`):

| 缓冲 | dtype | 用途 |
|---|---|---|
| x1 / x2 | bfloat16 | 行输入 ping-pong(`rows*kPad`) |
| xf32 | float | BF16→FP32 转换结果(`rows*kPad`,phase1/phase2 串行复用) |
| z1 / z2 | int8 | INT8 输出 ping-pong |
| xAbs | float | `vabs` 中间 + `ReduceMax` 结果落点(`kPad`,逐行复用) |
| xf16 | half | FP32→FP16(ODD)中转(`rows*kPad`) |
| rowAbs | float | 每 `rows` 行 absmax 标量槽(32B 对齐,供 `vmax` 合并;`rows*8`) |
| scaleStage | float | scale 暂存:`[0,r)` = absmax/127,`[rows, rows+r)` = 127/absmax(`2*rows`) |

UB 总占用 `12*rows*kPad + 4*kPad + 40*rows`,`rowMax` 保证 ≤ `UB_SIZE`(`assert`)。

**单行路径**(`tileRows == 1`,`TaskTilesInitSingleRow`):单份缓冲 `s_x`/`s_xf32`/`s_z`/`s_xf16`/`s_xAbs` + 32B 对齐的 `s_scaleUbuf`(1 个 float,供 MTE3 DMA 回填 GM);scaleRec 作为 S 管线标量 `s_scaleRec` 直接传给 `vmuls`,不落 UB 数组。

### 数据流与流水线同步

每行两阶段串行(phase1 absmax → phase2 缩放),`x/z` 用 `EVENT_ID0+id` ping-pong,absmax 经 `V→S` 事件交标量核算 scale,再 `S→MTE3` 写 scale 到 GM。

**多行路径**(`RunTileByIdxMultiRow`,[csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h) 的 `QuantDyn::RunTileByIdxMultiRow`)按 `[base, base+r)` 分组:

1. **phase1(逐 k 段搬入 → absmax 合并)**:2D block DMA 搬 `r` 行 x → `convert_input` 升 FP32 → 逐行 `vabs` → `ReduceMax(xAbs, kSize)` → `SetMask(1)` 后 `vmax(rowAbs+i*8, ..., xAbs)` 把标量结果合并到 `rowAbs[i]` 的 32B 槽(首段 copy、后续段 merge);跨 k 段累加得整行 absmax。
2. **V→S**:`set_flag(PIPE_V, PIPE_S, EVENT_ID0)` 让 `rowAbs` 表对 S 可见;S 逐行读 `absmax = *(rowAbs+i*8)`,算 `scale = absmax/127`、`scaleRec = 127/absmax`,写入 `scaleStage[0..r)` 与 `scaleStage[rows..rows+r)`。
3. **写 scale**:`S→MTE3` → `copy_ubuf_to_gm_align_b32(scalesGm+base, scaleStage, r)` → `MTE3→S`(同时 gate V:scaleStage 已消费)。
4. **phase2(缩放 + 写回)**:`kLoop==1` 复用 phase1 的 `xf32`;`kLoop>1` 重载 x 段升 FP32;逐行 `vmuls(xf32+i*kPad, ..., scaleRec)` → `vconv_f322f16o(xf16, xf32)`(ODD)→ `wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0+eId)` → `vconv_f162s8r(zBufs[eId], xf16)`(RINT)→ `set_flag(PIPE_V, PIPE_MTE3, ...)` → `copy_ubuf_to_gm` 2D burst 写回,`eId = 1-eId`。

**单行路径**(`RunTileByIdxSingleRow`):结构同上但单缓冲、无乒乓、无逐行循环——phase1 把整行 absmax 合并到 `s_scaleUbuf[0]`,V→S 后 S 算 `scale`/`s_scaleRec`,`copy_ubuf_to_gm_align_b32` 写单 scale;phase2 `vmuls`(用标量 `s_scaleRec`,经 `S→V` 事件可见)→ cast 链 → 写回。

### 关键计算步骤

两路径共享 phase1(absmax)/phase2(缩放)框架,差异在归约范围与 x 是否常驻 UB([csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h) 的 `QuantDyn::RunTileByIdxMultiRow`/`RunTileByIdxSingleRow`):

1. **absmax 归约**:`vabs` → `ReduceMax`(单段)或 `vmax` 跨段合并到 `rowAbs`/`s_scaleUbuf`;
2. **scale 计算(S 管线)**:`scale = absmax/127`(写 GM)、`scaleRec = 127/absmax`(留作 `vmuls` 标量);
3. **rescale**:`vmuls(xf32, xf32, scaleRec)`;
4. **cast 链**:`vconv_f322f16o`(FP32→FP16,ODD)→ `vconv_f162s8r`(FP16→INT8,RINT 舍入 + 饱和)。

cast 链用 `f322f16o`(ODD)→`f162s8r`(RINT):量化值已按 scale 压缩到 [-127,127] 范围内,`RINT(ODD(x)) == RINT(x)`,数值等价于旧的三步链但少一次 vconv([csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h) 的 `QuantDyn::RunTileByIdxMultiRow`,关键字 `ODD rounding` 注释)。通用归约实现见 [kernel_macro.h](../../../csrc/kernels/kernel_macro.h) 的 `ReduceMax`(二分折叠 + `vcmax`)。

### 边界处理

- `pnum_tokens` 非空时 `m = min(*pnum_tokens, m)`([csrc/kernels/quant_dyn.h](../../../csrc/kernels/quant_dyn.h) 的 `QuantDyn::TaskTilesInit`,关键字 `pnumTokensValue`),跳过 DP padding 出来的无效行。
- k 非 `align` 整数倍时由 `kPad`/`kSize` 补齐做向量计算,搬运/写出仍按真实 `kSize` 字节;尾部 k 段 `kSize < kPad`。
- tile 越界(`rowStart >= m`)直接 return。
- host 侧 `x.numel == 0` 直接返回([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpQuantDyn`,关键字 `x.numel == 0`)。
- host 侧拒绝 `k % 32 != 0`(ragged k,2D DMA 步长要求 32B 对齐;[csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpQuantDyn`,关键字 `must be a multiple of 32`)。
