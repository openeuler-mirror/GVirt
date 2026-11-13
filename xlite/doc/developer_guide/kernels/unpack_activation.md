# unpack_activation

## 功能概述

MSD(Mixed-precision Split-activation Decomposition,W4A8)管线的第二步:把 INT8 激活按字节拆成低 4bit / 高 4bit 两部分,打包成 INT4-packed 的交错双行布局。数学基础是恒等式 `INT8 = high_nibble × 16 + low_nibble`:

- 高 nibble:`floor(x / 16)`,值域 [-8, 7],算术右移天然带符号;
- 低 nibble:`x & 0x0F` 得 [0, 15] 的无符号值,再减 8 映射到 [-8, 7];这个 "-8" 偏移由下游 `msd_merge_dequant` 的 scale_bias(每列 `8 × Σ W_int4×w_scale`)补偿。

输入 `[m, k]` INT8(每字节含相邻两个 INT4 值),输出 `[2m, k/2]` INT8(每字节含两个 INT4):对输入第 r 行,输出第 `2r` 行放低 nibble 序列、第 `2r+1` 行放高 nibble 序列(相邻行交错)。这样激活被分解为两组 INT4,可走 INT4×INT4 的 cube 矩阵乘,再在 `msd_merge_dequant` 中合并还原。

## 输入输出参数

Python 入口 `unpack_activation(rt, input, output)`(`csrc/_C.cpp` 的 `UnpackActivation`,关键字 `m.def("unpack_activation"`)。kernel 签名见 `csrc/kernels/unpack_activation.h` 的 `unpack_activation`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| act_packed | 输入 | `[m, k]` | int8 | 量化后的激活;每字节 = 低4bit(偶数列 INT4,无符号 0-15)+ 高4bit(奇数列 INT4,带符号) |
| act_out | 输出 | `[2m, k/2]` | int8(INT4-packed) | 交错布局:token r 的低 nibble → 行 2r,高 nibble → 行 2r+1;每字节两个 4bit |
| m | 标量 | - | uint32_t | 输入行数,host 传 `input.shape[0]` |
| k | 标量 | - | uint32_t | 输入列数(字节数),host 传 `input.shape[1]`,要求偶数(host 校验) |

测试参考:[tests/kernels/unpack_activation.py](../../../tests/kernels/unpack_activation.py),覆盖 `[20, 64]`、`[100, 6144]`、`[20000, 2048]`;参考实现强调高 nibble 必须用算术右移(floor)而非截断除法,否则负数会破坏 `x = high×16 + (low&0xF)` 恒等式。

## 支持的数据类型

- `int8_t`(INT8 → INT4-packed INT8;[unpack_activation_int8_t.cpp](../../../csrc/kernels/unpack_activation_int8_t.cpp))

host 校验 `input.dtype == INT8 && shape 为 2D && shape[1] % 2 == 0`(`csrc/op.cpp` 的 `XliteOpUnpackActivation`,关键字 `input.shape[1] % 2 == 0`)。

## 实现原理

实现位于 [csrc/kernels/unpack_activation.h](../../../csrc/kernels/unpack_activation.h),函数 `unpack_activation`(入口符号 `unpack_activation_int8_t` 由 `UNPACK_ACTIVATION_FUNC_DEFINE` 宏展开)。

### 分块与多 Block 并行

- 无 k 切块,整行一次搬入(`k_pad = ROUND_UP(k, 256)` 按 INT8 向量粒度对齐);
- 行间按 block 跨步认领:每个 block 处理 `n = ceil((m - block_idx) / block_num)` 行,行号自 `block_idx` 起按 `block_num` 跨步推进(`csrc/kernels/unpack_activation.h` 的 `unpack_activation`,关键字 `uint32_t n = (m - block_idx + block_num - 1) / block_num`),输入/输出各 ping-pong。

### tile overlap 三级解耦流水

每个 block 对自己认领的行做三级重叠:迭代 `i` 同时执行 `store(i-1)` ∥ `load(i+1)` ∥ `compute(i)`(`csrc/kernels/unpack_activation.h` 的 `unpack_activation`,关键字 `store row (i-1)`/`Load row (i+1)`/`Compute row (i)` 注释块)。关键在于 **MTE2/MTE3 的 DMA 指令先于 V 的 `wait_flag` 下发**:V 即便阻塞在等待上,已派发的 load/store 仍在推进,使访存与计算真正重叠。结构为 prologue(载入第 0、1 行并算第 0 行)→ 稳态循环(store 上一行 + 预取下一行 + 计算当前行)→ epilogue(store 末行)→ drain 清算残留 flag(`csrc/kernels/unpack_activation.h` 的 `unpack_activation`,关键字 `Prologue`/`Steady state`/`Epilogue`/`Drain` 注释块)。

### UB 内存布局(`csrc/kernels/unpack_activation.h` 的 `unpack_activation`)

| 缓冲 | 大小 | 用途 |
|---|---|---|
| in_buf_0 / in_buf_1 | k_pad (int8) | 输入行 ping-pong |
| low_and_buf | k_pad (int8) | `x & 0x0F` 结果 |
| work_fp16 | k_pad × half | 低 nibble 的 fp16 工作区 |
| vand_mask_buf | k_pad × half | 0x0F0F 掩码(int16 视图) |
| fp16_buf | k_pad × half | 原值 fp16,后用作高 nibble fp16 |
| out_low_0/1, out_high_0/1 | kh_pad (int8) | 低/高 nibble 的 INT4-packed 输出 ping-pong(kh_pad = ROUND_UP(k/2, 256)) |

### 流水线同步(MTE2 / V / MTE3)

三管线事件协议,`eid = EVENT_ID0/1` ping-pong,循环前预置 4 条 `set_flag`(每 eid 两个方向):`V→MTE2` 各一条把两个 in 缓冲标记为已释放(首次 `unpack_load` 不阻塞),`MTE3→V` 各一条把两个 out 缓冲标记为空闲(首次 `unpack_compute` 不阻塞);结尾对称地以 4 条 `wait_flag` 排空残余事件:

- `unpack_load`:`wait_flag(V→MTE2, eid)` 等 V 释放 in 缓冲 → GM→UB → `set_flag(MTE2→V, eid)` 就绪;
- `unpack_compute`:`wait_flag(MTE2→V, eid)` 等输入就绪;`vand`/`vconv` 读完输入后 `set_flag(V→MTE2, eid)` 释放 in 缓冲;`wait_flag(MTE3→V, eid)` 等上一轮 store 回收 out 缓冲,INT4 打包(`vconv_f162s4`/`vconv_f162s4f`)后 `set_flag(V→MTE3, eid)` 交给写出;
- `unpack_store`:`wait_flag(V→MTE3, eid)` 等输出就绪 → 两行 UB→GM → `set_flag(MTE3→V, eid)` 回收 out 缓冲。

`vand_mask` 常量在循环外一次性 `vector_dup` 生成:以 int16 视图写 0x0F0F,使 int8 数据按 16bit 字与掩码对齐做按位与(`csrc/kernels/unpack_activation.h` 的 `unpack_activation`,关键字 `0x0f0f`)。

### 关键计算步骤(`csrc/kernels/unpack_activation.h` 的 `unpack_activation`)

对输入行 x(int8,k 字节,每字节两个 INT4):

1. **低 nibble 路径**:`vand(low_and, x, 0x0F0F)`(int16 视图)得每字节两个 [0,15] 的 nibble → `vconv_s82f16` 转 fp16([0.0, 15.0])→ `vadds(-8.0)` 平移到 [-8, 7];
2. **高 nibble 路径**:`vconv_s82f16(x)` 转 fp16 → `vmuls(0.0625)` 即除以 16,fp16 除法配合 vconv 的舍入实现 floor 语义,得 [-8, 7];
3. **INT4 打包**:`vconv_f162s4` 把低 nibble fp16 打包为 INT4-packed(`out_low`),`vconv_f162s4f` 把高 nibble fp16 打包(`out_high`;`f` 变体带 floor 舍入);
4. **写出**:`unpack_store` 经 `CopyUbufToGmAligned` 把低/高两行分别写到 `act_out + (2r)·(k/2)` 与 `act_out + (2r+1)·(k/2)`(`csrc/kernels/unpack_activation.h` 的 `unpack_store`)。

注意 k_pad 与 fp16 缓冲按 k 字节分配:一个 int8 字节展开成一个 fp16 值(两个 nibble 各一个),fp16 视图下 `fp16_rep = k_pad / VECTOR_MAX_NUM_OF_FP16`。

### 边界处理

- k 非 256 整数倍时计算侧 pad 到 k_pad(向量指令粒度),搬运/写出按真实 `k` / `k/2` 字节;
- 每行产出 2 行输出,GM 写地址按 `k_half = k/2` 字节行距计算,天然对齐到字节边界。
