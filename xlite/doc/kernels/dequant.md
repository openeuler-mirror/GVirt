# dequant

## 功能概述

将 int8 量化矩阵乘的 fp16 中间输出反量化为 bf16：`out[i, :] = (float)in[i, :] * scale[i]`（scale 可选）。每个 token 一行 scale（per-token scale），无 scale 时等价于纯 dtype 转换 fp16 → bf16。常与 `matmul_dequant` 组合使用（`csrc/op.cpp:1436-1446`：int8×int8 matmul 输出 FP16 后接本算子）。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| in | 输入 | [m, n] | float16 | 量化矩阵乘的中间输出（fp16 视图的 int32 累加结果已缩放后的数据） |
| scale | 输入(可选) | [m] | float32 | 每 token（行）一个缩放系数；`hasScale == false` 时传空 tensor |
| out | 输出 | [m, n] | bfloat16 | 反量化结果；测试中与 in 为同一 buffer（fp16/bf16 均为 2 字节，原地覆盖安全） |
| pnum_tokens | 输入(可选) | [1] | uint32 | 实际 token 数指针（动态 batch），取 min(*pnum_tokens, m)；可为空 |
| m | 标量 | - | uint32_t | 行数，host 侧传 `in.shape[0]` |
| n | 标量 | - | uint32_t | 列数，host 侧传 `in.numel / in.shape[0]`（`csrc/op.cpp:1425-1431`） |

Python 调用方式（`tests/kernels/dequant.py:36`）：`dequant(rt, inout, scales, inout, has_scale)`。

注意：kernel 实例化中 dtype 模板参数是 fp16 的 `in` 类型；输出固定为 bfloat16_t（`csrc/kernels/dequant.h:42-44, 127`）。host 侧 `XliteOpDeQuant` 仅接受 `in.dtype == FP16`（`csrc/op.cpp:1428-1434`），out 的 dtype 由调用方保证（通常通过 `View(BF16)` / `View(FP16)` 切换，见 `csrc/op.cpp:1440-1444`）。

## 支持的数据类型

| dtype（in） | 实例化文件 | kernel 符号 | 输出 dtype |
|-------|-----------|-------------|-----------|
| float16_t | `csrc/kernels/dequant_float16_t.cpp` | `dequant_float16_t` | bfloat16_t（kernel 内固定） |

仅支持 `__DAV_C220_VEC__`，其余架构导出空实现（`csrc/kernels/dequant.h:186-192`）。

## 实现原理

实现为 `Dequant<dtype>` 类（`csrc/kernels/dequant.h:13`），流程 `Run → SetFlags → TaskTilesInit → RunTileByIdx（跨 block）→ WaitFlags`（`csrc/kernels/dequant.h:149-158`）。

### 任务切分与数据流

- `TaskTilesInit`（`csrc/kernels/dequant.h:61-75`）：记录 in/scale/out 的 GM 指针；若有 pnum_tokens 用其修正实际行数 m；返回 m 作为 tile 总数；
- 主循环 `for (idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum())`：以"行"为 tile，跨 block 并行（`csrc/kernels/dequant.h:154-156`）；
- `RunTileByIdx` 把第 idx 行交给 `RunTile(in + idx*n, scale + idx, out + idx*n, 1, n)`（`csrc/kernels/dequant.h:77-82`）。

每行内部再按列分块：`nTile = 7168`，`nLoop = DIV_ROUND_UP(n, 7168)`，逐块处理（`csrc/kernels/dequant.h:27, 90-94`）。单个行内分块的数据流：

GM(fp16 行块) --MTE2--> UB(inUbBuf) --V: vconv--> UB(tmpUbBuf) --V: vmuls(scale)--> UB(mulUbBuf) --V: vconv 舍入--> UB(outUbBuf, bf16) --MTE3--> GM(out 行块)

### UB 内存布局

`Init`（`csrc/kernels/dequant.h:21-50`）以 `nPad = ROUND_UP(7168, 128)`（fp16 按 256B 对齐粒度）为单位手工排布，ping-pong 双缓冲（`eventId` 在 0/1 之间翻转，`csrc/kernels/dequant.h:135`）：

| 缓冲区 | dtype | 数量 | 用途 |
|--------|-------|------|------|
| inUbBuf[2] | fp16 | nPad×2 | 原始 fp16 输入（双缓冲） |
| tmpUbBuf[2] | fp32 | nPad×2 | vconv 升精度结果（双缓冲） |
| mulUbBuf[2] | fp32 | nPad×2 | 乘 scale 后的 fp32（双缓冲） |
| outUbBuf[2] | bf16 | nPad×2 | 舍入后的 bf16 输出（双缓冲） |
| scaleUbBuf | fp32 | 1 | 当前行的 scale 标量 |

注意布局按各缓冲区自身 dtype 的元素计数排布（inUbBuf/outUbBuf 步进 nPad 个 2 字节元素，fp32 缓冲各占 nPad*4 字节），总占用约 `24 * nPad + 4` 字节（约 168KB），需不超过 UB_SIZE（192KB）。

### 计算步骤（`csrc/kernels/dequant.h:111-128`）

`nRepeats = DIV_ROUND_UP(nSizePad, VECTOR_MAX_NUM_OF_FP32)`（64 fp32/repeat）：

1. `vconv_f162f32(tmpUbBuf, inUbBuf, nRepeats, ...)`：fp16 → fp32；
2. （有 scale 时）`vmuls(mulUbBuf, tmpUbBuf, float(*scaleUbBuf), nRepeats, ...)`：整行块乘以该行的 fp32 scale 标量；scale 是先由 MTE2 搬入 UB、经 `PIPE_MTE2 → PIPE_S` flag 同步后由 Scalar 流水读出并内联到 vmuls 立即数（`csrc/kernels/dequant.h:105-109, 120-124`）；
3. `vconv_f322bf16r(outUbBuf, tmpPtr, nRepeats, ...)`：fp32 舍入（round-to-nearest-even）转 bf16。无 scale 时直接从 tmpUbBuf 转换。

### 流水线同步

`SetFlags` 初始化 4 个 flag（V→MTE2 的 ID0/ID1、MTE3→V 的 ID0/ID1，`csrc/kernels/dequant.h:52-59`）。每个行内块迭代（`csrc/kernels/dequant.h:99-134`）：

- `wait_flag(PIPE_V, PIPE_MTE2, ID0+eventId)` 等输入缓冲空闲 → `copy_gm_to_ubuf_align_b16` 搬入 fp16 行块（按字节数，16B 对齐原语）→ `set_flag(PIPE_MTE2, PIPE_V, ID0+eventId)`；
- 有 scale 时再搬 4 字节 scale 到 scaleUbBuf，`set_flag(PIPE_MTE2, PIPE_S, ID0)` 通知 Scalar 流水；
- V 侧 `wait_flag(PIPE_MTE2, PIPE_V, ...)` 等数据、`wait_flag(PIPE_MTE3, PIPE_V, ...)` 等输出缓冲空闲 → vconv → `set_flag(PIPE_V, PIPE_MTE2, ...)` 归还输入缓冲；
- vmuls（需先 `wait_flag(PIPE_MTE2, PIPE_S, ID0)` 等 scale 到位）→ vconv 转 bf16 → `set_flag(PIPE_V, PIPE_MTE3, ...)` 通知 MTE3；
- MTE3 `wait_flag` 后 `copy_ubuf_to_gm_align_b16` 写回 bf16 行块 → `set_flag(PIPE_MTE3, PIPE_V, ...)` 归还输出缓冲；
- `eventId = 1 - eventId` 切换 ping-pong，当前块的写回与下一块的搬入重叠。

`WaitFlags` 收尾 4 个 wait_flag + `pipe_barrier(PIPE_ALL)`（`csrc/kernels/dequant.h:140-147`）。

### 边界处理

- 列尾块：`nSize = nActual - nOffset`（最后一块可能小于 nTile），`nSizePad = ROUND_UP(nSize, 128)` 用于 repeat 计算，搬运与写回均按精确字节数 `nSize * sizeof(dtype)` 走 `*_align_b16` 原语，支持非 32B 对齐的 n；
- UB 内 padding 部分为无效数据，不会被写出；
- pnum_tokens 动态 token 数：只处理前 min(*pnum_tokens, m) 行。
