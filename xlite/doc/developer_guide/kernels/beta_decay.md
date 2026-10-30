# beta_decay

## 功能概述

线性注意力/RWKV 类模型的门控预处理算子,一次 kernel 同时计算两个逐元素公式(按 v_head 维度并行,`num_v_heads` 很小,典型 16):

```
beta = sigmoid(b)                          # 门控系数
g    = -exp(A_log) * softplus(a + dt_bias) # 衰减系数
```

即 `Beta()` 与 `Decay()` 两个阶段(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Beta` 与 `BetaDecay::Decay`)。模型前向中用于 `beta = sigmoid(Bx)`, `g = -exp(A_log) * softplus(Ax + dt_bias)` 的 fused 计算([csrc/model.cpp](../../../csrc/model.cpp) 的 `XModel::ForwardAttnLinear`,关键字 `XliteOpBetaDecay`)。

## 输入输出参数

Python 入口 `beta_decay(rt, b, a, A_log, dt_bias, beta, g, bsz, seqlen, num_v_heads)`(`csrc/_C.cpp` 的 `BetaDecay`,关键字 `m.def("beta_decay"`)。kernel 签名见 `csrc/kernels/beta_decay.h` 的 `BetaDecay` 类(入口符号 `beta_decay_<dtype>` 由 `BETA_DECAY_FUNC_DEFINE` 宏展开):

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| b | 输入 | `[bsz, seqlen, num_v_heads]` | float / float16 / bfloat16 | beta 分支输入 |
| a | 输入 | `[bsz, seqlen, num_v_heads]` | 同上 | decay 分支输入 |
| A_log | 输入 | `[num_v_heads]` | 同上 | 每头 A_log(= log(-A)) |
| dt_bias | 输入 | `[num_v_heads]` | 同上 | 每头 dt 偏置 |
| beta | 输出 | `[bsz, seqlen, num_v_heads]` | 同上 | `sigmoid(b)` |
| g | 输出 | `[bsz, seqlen, num_v_heads]` | 同上 | `-exp(A_log) * softplus(a + dt_bias)` |
| bsz | 标量 | - | uint32_t | batch;模型侧展平 token 后传 m、seqlen=1 |
| seqlen | 标量 | - | uint32_t | 序列长度;总行数 = bsz × seqlen |
| num_v_heads | 标量 | - | uint32_t | v head 数(每行元素数) |

测试参考:[tests/kernels/beta_decay.py](../../../tests/kernels/beta_decay.py):batch 1~8 × seq_len 2^0~2^12 × fp32/fp16/bf16 全组合,与 `torch.sigmoid` / `-A_log.exp() * F.softplus(a + dt_bias)` 对比。

## 支持的数据类型

- `float`([beta_decay_float.cpp](../../../csrc/kernels/beta_decay_float.cpp))
- `float16_t`([beta_decay_float16_t.cpp](../../../csrc/kernels/beta_decay_float16_t.cpp))
- `bfloat16_t`([beta_decay_bfloat16_t.cpp](../../../csrc/kernels/beta_decay_bfloat16_t.cpp))

host 要求 b、a 同 dtype 且为 FP32/FP16/BF16(`csrc/op.cpp` 的 `XliteOpBetaDecay`,关键字 `EachXDtype(FP32, b, a)`)。

## 实现原理

实现位于 [csrc/kernels/beta_decay.h](../../../csrc/kernels/beta_decay.h),采用类封装(`BetaDecay<Dtype>`,Init / Beta / Decay)。

### 数据流与 UB 布局(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Init`)

数据流:GM(一行 `num_v_heads` 个元素)→ UB → 转 fp32 向量计算 → 转回 Dtype → GM。每行数据量极小(16 个头),一次一行,不做 k 切块。

UB 缓冲(全部按 `DIV_ROUND_UP(num_v_heads * sizeof, 256) * 256` 对齐分配):

| 缓冲 | dtype | 用途 |
|---|---|---|
| conv_b / conv_a / conv_A_log / conv_dt_bias | Dtype(非 fp32 时) | GM 搬入的原始 dtype 中转 |
| b_buf / a_buf / A_log_buf / dt_bias_buf / beta_buf / g_buf | float | fp32 计算缓冲 |
| ones_buf | float | 常量 1,供 `vdiv` 求倒数(sigmoid 用 1/x) |

`num = ceil(num_v_heads * sizeof(Dtype) / 32)`(32B 对齐搬运块数),`calcRepeat = ceil(num / 8)`(向量 repeat 数)。

### 多 Block 并行

按行轮转:`for (i = 0; i < bsz * seqlen; i++) { if (i % GetBlockNum() != GetBlockIdx()) continue; ... }`(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Beta` 与 `BetaDecay::Decay`,关键字 `i % GetBlockNum()`),每个 AIV 处理自己份额的 token 行。A_log / dt_bias / ones 只加载一次(Decay 中 A_log、dt_bias 在行循环外,见 `BetaDecay::Decay`)。

### 流水线同步

由于每行数据量小、搬运与计算交替密集,采用逐行 MTE2/MTE3 事件握手 + `pipe_barrier`:

- 搬入前 `set_flag(PIPE_MTE3, PIPE_MTE2)` / `wait_flag` 确保上一次 GM 写出完成、UB 缓冲可复用(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Beta`,关键字 `set_flag(PIPE_MTE3, PIPE_MTE2)`);
- `copy_gm_to_ubuf` + `pipe_barrier(PIPE_MTE2)` + `MTE2→V` 事件等待数据就绪(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Beta`,关键字 `copy_gm_to_ubuf(b_buf`);
- 计算完 `V→MTE3` 事件后写出,`pipe_barrier(PIPE_MTE3)` 收尾(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Beta`,关键字 `copy_ubuf_to_gm(beta`)。

fp32 直接在计算缓冲上搬运;fp16/bf16 先搬入 conv 缓冲再 `vconv_f162f32` / `vconv_bf162f32` 转 fp32,输出前反向转换(`vconv_f322f16` / `vconv_f322bf16r`)。

### 关键计算步骤

**Beta()**(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Beta`),对每行 b:

1. `vmuls(-1.0)` → `vexp` → `vadds(1.0)`:得 `1 + e^-x`;
2. `vdiv(b_buf, ones_buf, b_buf)`:sigmoid(x) = 1 / (1 + e^-x)(用 ones_buf 做被除数实现求倒);
3. 写回 `beta`。

**Decay()**(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Decay`):

1. 循环外:A_log 搬入转 fp32 → `vexp` → `vmuls(-1.0)` 得每头 `-exp(A_log)`(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Decay`,关键字 `vexp(A_log_buf`);dt_bias 搬入转 fp32 备用;
2. 对每行 a:`vadd(a + dt_bias)` → `vexp` → `vadds(1.0)` → `vln`:softplus(x) = ln(1 + e^x)(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Decay`,关键字 `vln(a_buf`);
3. `vmul` 乘上 `-exp(A_log)`:得 g,写回(`csrc/kernels/beta_decay.h` 的 `BetaDecay::Decay`,关键字 `vmul(a_buf, a_buf, A_log_buf`)。

### 边界处理

- `num_v_heads` 非 32B 块整数倍时,`num` 向上取整搬运(尾部会多搬补齐字节,但向量计算 `calcRepeat` 覆盖有效元素;写出同样按对齐块数);
- bsz × seqlen 不是 block_num 整数倍时轮转条件自然跳过空闲核;
- 每行元素数(num_v_heads)须 ≤ 向量一次 repeat 能覆盖的范围(calcRepeat 由 num 派生,16 头场景远在限内)。

## FP32 native gate 变体

当 activation 为 BF16/FP16 且 g 输出为 FP32 时，host 选择 `beta_decay_native_*`。
beta 保留 activation dtype；A_log 和 dt_bias 可分别为 activation dtype 或 FP32，
按真实 dtype 读取。原同精度入口保留，FP32 activation 使用原 FP32 实例。
`tests/kernels/beta_decay_native.py` 覆盖这些组合。
