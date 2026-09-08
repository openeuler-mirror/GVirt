# cast

## 功能概述

dtype 转换算子。当前实现两个方向：bf16 → float32（cast_up，升精度）和 float32 → bf16（cast_down 内部使用，舍入到偶数）。基于向量转换指令 `vconv_bf162f32` / `vconv_f322bf16r` 完成逐元素转换。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x | 输入 | 任意（按长度 length 的一维流处理） | bfloat16（cast_up）/ float（fp32→bf16 方向） | 输入张量，shape 不参与计算，只取元素总数 |
| y | 输出 | 同 x | float（cast_up）/ bfloat16（fp32→bf16 方向） | 输出张量，长度与 x 相同 |
| length | 标量 | - | uint32_t | 元素总数，host 侧传 `in.numel`（`csrc/op.cpp:819`） |

Python 绑定：

- `cast_up(rt, x, y)`（`csrc/_C.cpp:1699-1709`，`tests/kernels/cast.py:21`）：bf16 → float32，host 侧 `XliteOpCastUp` 只接受 `in.dtype == BF16 && out.dtype == FP32`，其余抛异常；inScale 参数当前实现中仅作占位；
- fp32→bf16 方向（`aclrtlaunch_cast_float_bfloat16_t`）目前作为 matmul 等 host 侧组合路径的内部 kernel 使用（例如 `csrc/op.cpp:758-760` matmul 输出 FP32 转 BF16 时调用），未单独导出 Python 接口；`XliteOpCastDown` 为 TODO。

## 支持的数据类型

| 转换方向 | 实例化文件 | kernel 符号 |
|----------|-----------|-------------|
| bfloat16_t → float | `csrc/kernels/cast_bfloat16_t_float.cpp` | `cast_bfloat16_t_float` |
| float → bfloat16_t | `csrc/kernels/cast_float_bfloat16_t.cpp` | `cast_float_bfloat16_t` |

仅支持 `__DAV_C220_VEC__`，其余架构导出空实现（`csrc/kernels/cast.h:87-92`）。

## 实现原理

### 整体数据流

把张量当作长度为 `length` 的一维数据流，切成 `process_num` 个 tile，跨 block 并行：`for (process = block_idx; process < process_num; process += block_num)`（`csrc/kernels/cast.h:29`）。每个 tile 的数据流：

GM --MTE2--> UB(t1, 源 dtype) --V: vconv--> UB(t2, 目标 dtype) --MTE3--> GM

### tiling 策略

单次 vconv 的 repeat 数上限为 `VECTOR_MAX_REPEAT`(255)，每个 repeat 最多处理 256B，因此单 tile 上限 `max_num = MAX_REPEAT_TIMES * 256 / 4`（MAX_REPEAT_TIMES = 255，即 fp32 计 16320 个元素，`csrc/kernels/cast.h:18`）：

- 若 `length <= max_num * block_num`：`tile_size = DIV_ROUND_UP(length, block_num)`，让所有 block 均分（小块均衡负载）；
- 否则 `tile_size = max_num`，每个 block 分到多个整 tile；
- `process_num = DIV_ROUND_UP(length, tile_size)`。

每个 tile 的实际长度 `actual_len = min(tile_size, length - process * tile_size)`（`csrc/kernels/cast.h:30-31`）。

### UB 内存布局

两个缓冲区（`csrc/kernels/cast.h:19-21`）：

| 缓冲区 | 偏移 | 大小 | 用途 |
|--------|------|------|------|
| t1 | 0 | max_num * sizeof(源dtype) | 源数据（bf16 2B 或 fp32 4B） |
| t2 | max_num * sizeof(源dtype) | max_num * sizeof(目标dtype)（实际按 fp32 布局） | 转换结果 |

t2 的偏移表达式统一写为 `max_num * (bf16 ? 2 : 4)`，即按源 dtype 大小留出 t1 空间；bf16→fp32 时 t2 需要的 4B*max_num 由后续 UB 空间承接（fp32→bf16 时 t2 只需 2B*max_num，空间富余）。

### 向量计算指令

- bf16 → fp32：`vconv_bf162f32(t2, t1, DIV_ROUND_UP(actual_len * 4, 256), 1, 1, 8, 4)`（`csrc/kernels/cast.h:41-43`），repeat 数按 fp32 字节数折算；
- fp32 → bf16：`vconv_f322bf16r(t2, t1, DIV_ROUND_UP(actual_len * 4, 256), 1, 1, 4, 8)`（`csrc/kernels/cast.h:63-65`），带 r 后缀为 round-to-nearest-even 舍入。

### 流水线同步

两个方向使用同一套 flag 协议（以 bf16→fp32 为例，`csrc/kernels/cast.h:27-28, 34-52`）：

- 初始：`set_flag(PIPE_V, PIPE_MTE2, ID0)`（t1 空闲）、`set_flag(PIPE_MTE3, PIPE_V, ID0)`（t2 空闲）；
- 每迭代：`wait_flag(PIPE_V, PIPE_MTE2, ID0)` 等 t1 空闲 → `copy_gm_to_ubuf` 搬入 → `set_flag(PIPE_MTE2, PIPE_V, ID0)` → V `wait_flag` 等数据就绪 → `wait_flag(PIPE_MTE3, PIPE_V, ID0)` 等 t2 空闲 → vconv → `set_flag(PIPE_V, PIPE_MTE2, ID0)` 归还 t1 → `set_flag(PIPE_V, PIPE_MTE3, ID0)` 通知 MTE3 → MTE3 `wait_flag` 后 `copy_ubuf_to_gm` 搬出 → `set_flag(PIPE_MTE3, PIPE_V, ID0)` 归还 t2。

该 kernel 为单缓冲（t1/t2 各一份），重叠来自不同迭代间 MTE2 搬入与 MTE3 搬出的跨流水并发。退出前 `wait_flag` 收尾 + `pipe_barrier(PIPE_ALL)`（`csrc/kernels/cast.h:76-78`）。

### 边界处理

- 尾 tile 的 actual_len 可能小于 tile_size 且非 32B 对齐：搬运 burst 数按 `DIV_ROUND_UP(actual_len * sizeof(dtype), BLOCK_SIZE)` 向上取整（DMA 粒度 32B），UB 尾部 padding 数据无效；
- vconv 的 repeat 数同样按 actual_len 字节数折算取整，转换与写回都基于同一长度，GM 不会多写（最后一个 tile 的写回长度按取整后的字节数计，需保证输出缓冲区长度按 32B 块预留，实际使用中 length 为张量元素数，末尾多余字节落在本 tile 与下一 tile 之间时由 block 间不重叠保证正确性）。
