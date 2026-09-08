# add

## 功能概述

二维矩阵逐元素加法 `z = x + y`，用于推理过程中的残差累加等场景。fp16/bf16 输入会在片内转换为 float32 完成加法后再舍入回原 dtype，以保证计算精度。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x | 输入 | [x_numel, y_numel] | float16 / bfloat16 | 加数，行数为 `x_numel`（block 切分维度），每行 `y_numel` 个元素 |
| y | 输入 | [x_numel, y_numel] | float16 / bfloat16 | 加数，与 x 同形状 |
| z | 输出 | [x_numel, y_numel] | float16 / bfloat16 | 结果，可与 x/y 为同一 tensor（原地更新） |
| x_numel | 标量 | - | uint32_t | 矩阵行数（host 侧传入 `in1.shape[0]`，见 `csrc/op.cpp:594` XliteOpAdd） |
| y_numel | 标量 | - | uint32_t | 每行元素个数（host 侧传入 `in1.shape[1]`） |

Python 调用方式（`tests/kernels/add.py:27`）：`add(rt, x, y, z)`，要求 x/y/z dtype 一致（`EachXDtype` 检查，`csrc/op.cpp:588-593`）。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|-------|-----------|-------------|
| float16_t | `csrc/kernels/add_float16_t.cpp` | `add_float16_t` |
| bfloat16_t | `csrc/kernels/add_bfloat16_t.cpp` | `add_bfloat16_t` |

仅支持 `__DAV_C220_VEC__`（向量核），非该架构时导出空实现（`csrc/kernels/add.h:103-108`）。

## 实现原理

### 整体数据流

每个 block（AI Vector Core）以"行"为单位处理：`for (process = block_idx; process < x_numel; process += block_num)`（`csrc/kernels/add.h:33`），即行间由多 block 并行（block 间步长为 `block_num`），行内 `y_numel` 个元素一次性整体搬入 UB 处理。单行数据流为：

GM(x 行) --MTE2--> UB(t1_dtype) --V: vconv--> UB(t1_float) \
GM(y 行) --MTE2--> UB(t2_dtype) --V: vconv--> UB(t2_float) \
--V: vadd--> UB(t1_float) --V: vconv 舍入--> UB(t2_float) --MTE3--> GM(z 行)

### UB 内存布局

通过 `get_imm` 在编译期分配 4 个连续 UB 缓冲区（`csrc/kernels/add.h:20-23`），偏移按 dtype 尺寸（fp16/bf16 均为 2 字节）计算：

| 缓冲区 | 偏移 | 大小 | 用途 |
|--------|------|------|------|
| t1_dtype | 0 | y_numel*2B | x 行的原始 dtype 数据 |
| t2_dtype | y_numel*2 | y_numel*2B | y 行的原始 dtype 数据 |
| t1_float | y_numel*4 | y_numel*4B | x 转换后的 float32（同时作为 vadd 的目的缓冲） |
| t2_float | y_numel*8 | y_numel*4B | y 转换后的 float32；vconv 回 dtype 后作为 MTE3 搬出缓冲 |

四段流水（t1_dtype / t2_dtype / t1_float / t2_float）使下一次迭代的 MTE2 搬入可以与当前迭代的 V 计算、MTE3 搬出重叠。注意 `y_numel * sizeof(Dtype)` 会向上取整到 32B 的 BLOCK_SIZE 粒度做搬运，因此 y_numel 较小时 UB 实际占用略大于标称大小。

### 向量计算步骤

fp16/bf16 输入均升精度到 fp32 计算（`csrc/kernels/add.h:49-79`）：

1. `vconv_f162f32` / `vconv_bf162f32`：x、y 分别转 float32（repeat 数 `vec_repeat_float = DIV_ROUND_UP(y_numel * 4, 256)`）；
2. `vadd`：`t1_float = t1_float + t2_float`，fp32 加法；
3. `vconv_f322f16r` / `vconv_f322bf16r`：结果舍入（round-to-nearest-even）转回原 dtype，写入 t2_float。

每条向量指令后都有 `pipe_barrier(PIPE_V)` 保证指令间依赖。

### 流水线同步（set_flag / wait_flag）

使用 3 个 event id 协调 MTE2（搬入）/ V（计算）/ MTE3（搬出）三条流水（`csrc/kernels/add.h:29-31, 35-44, 48-55, 58-67, 83-87`）：

- 初始：`set_flag(PIPE_MTE3, PIPE_V, ID0)`（t2_float 空闲）、`set_flag(PIPE_V, PIPE_MTE2, ID0/ID1)`（t1_dtype/t2_dtype 空闲），使首次迭代无需等待；
- MTE2 搬 x：`wait_flag(PIPE_V, PIPE_MTE2, ID0)` 等 V 释放 t1_dtype → 搬运 → `set_flag(PIPE_MTE2, PIPE_V, ID0)` 通知 V；
- MTE2 搬 y：同上使用 EVENT_ID1；
- V 转 x：`wait_flag(PIPE_MTE2, PIPE_V, ID0)` 等数据就绪，转换完成后 `set_flag(PIPE_V, PIPE_MTE2, ID0)` 归还 t1_dtype 给下一次迭代的 MTE2；
- V 转 y：需先 `wait_flag(PIPE_MTE3, PIPE_V, ID0)`（上一轮 MTE3 已搬完 t2_float），完成后归还 t2_dtype；
- MTE3 搬出：V `set_flag(PIPE_V, PIPE_MTE3, ID0)` → MTE3 `wait_flag` 后 `copy_ubuf_to_gm`，完成后 `set_flag(PIPE_MTE3, PIPE_V, ID0)` 归还 t2_float。

内核退出前 `wait_flag` 收尾并 `pipe_barrier(PIPE_ALL)` 确保全部流水完成（`csrc/kernels/add.h:91-94`）。

### 边界处理

- GM↔UB 搬运的 burst 数按 `DIV_ROUND_UP(y_numel * sizeof(Dtype), BLOCK_SIZE)` 向上取整（32B 对齐），末尾可能的越界部分只影响 UB 内无效数据，最终按同样取整字节数写出，不会越界写 GM（z 每行长度与 x/y 相同）；
- 要求 `y_numel` 每行数据能放入 UB（单行整体处理，无行内分块）；
- 尾部元素若不满足 256B repeat 对齐，vconv/vadd 按取整后的 repeat 数执行，UB 中的 padding 数据不参与正确性（写出长度同样按 32B 取整）。
