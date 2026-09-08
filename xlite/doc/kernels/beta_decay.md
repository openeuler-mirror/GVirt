# beta_decay

## 功能概述

线性注意力/RWKV 类模型的门控预处理算子,一次 kernel 同时计算两个逐元素公式(按 v_head 维度并行,`num_v_heads` 很小,典型 16):

```
beta = sigmoid(b)                          # 门控系数
g    = -exp(A_log) * softplus(a + dt_bias) # 衰减系数
```

即 `Beta()` 与 `Decay()` 两个阶段(`csrc/kernels/beta_decay.h:79-252`)。模型前向中用于 `beta = sigmoid(Bx)`, `g = -exp(A_log) * softplus(Ax + dt_bias)` 的 fused 计算([csrc/model.cpp:786-787](../../csrc/model.cpp))。

## 输入输出参数

Python 入口 `beta_decay(rt, b, a, A_log, dt_bias, beta, g, bsz, seqlen, num_v_heads)`(`csrc/_C.cpp:2796`)。kernel 签名见 `csrc/kernels/beta_decay.h:282-291`:

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

测试参考:[tests/kernels/beta_decay.py](../../tests/kernels/beta_decay.py):batch 1~8 × seq_len 2^0~2^12 × fp32/fp16/bf16 全组合,与 `torch.sigmoid` / `-A_log.exp() * F.softplus(a + dt_bias)` 对比。

## 支持的数据类型

- `float`([beta_decay_float.cpp](../../csrc/kernels/beta_decay_float.cpp))
- `float16_t`([beta_decay_float16_t.cpp](../../csrc/kernels/beta_decay_float16_t.cpp))
- `bfloat16_t`([beta_decay_bfloat16_t.cpp](../../csrc/kernels/beta_decay_bfloat16_t.cpp))

host 要求 b、a 同 dtype 且为 FP32/FP16/BF16(`csrc/op.cpp:2014-2020`)。

## 实现原理

实现位于 [csrc/kernels/beta_decay.h](../../csrc/kernels/beta_decay.h),采用类封装(`BetaDecay<Dtype>`,Init / Beta / Decay)。

### 数据流与 UB 布局(Init,beta_decay.h:31-77)

数据流:GM(一行 `num_v_heads` 个元素)→ UB → 转 fp32 向量计算 → 转回 Dtype → GM。每行数据量极小(16 个头),一次一行,不做 k 切块。

UB 缓冲(全部按 `DIV_ROUND_UP(num_v_heads * sizeof, 256) * 256` 对齐分配):

| 缓冲 | dtype | 用途 |
|---|---|---|
| conv_b / conv_a / conv_A_log / conv_dt_bias | Dtype(非 fp32 时) | GM 搬入的原始 dtype 中转 |
| b_buf / a_buf / A_log_buf / dt_bias_buf / beta_buf / g_buf | float | fp32 计算缓冲 |
| ones_buf | float | 常量 1,供 `vdiv` 求倒数(sigmoid 用 1/x) |

`num = ceil(num_v_heads * sizeof(Dtype) / 32)`(32B 对齐搬运块数),`calcRepeat = ceil(num / 8)`(向量 repeat 数)。

### 多 Block 并行

按行轮转:`for (i = 0; i < bsz * seqlen; i++) { if (i % GetBlockNum() != GetBlockIdx()) continue; ... }`(beta_decay.h:83-85、190-192),每个 AIV 处理自己份额的 token 行。A_log / dt_bias / ones 只加载一次(Decay 中 A_log、dt_bias 在行循环外,beta_decay.h:148-188)。

### 流水线同步

由于每行数据量小、搬运与计算交替密集,采用逐行 MTE2/MTE3 事件握手 + `pipe_barrier`:

- 搬入前 `set_flag(PIPE_MTE3, PIPE_MTE2)` / `wait_flag` 确保上一次 GM 写出完成、UB 缓冲可复用(beta_decay.h:87-88);
- `copy_gm_to_ubuf` + `pipe_barrier(PIPE_MTE2)` + `MTE2→V` 事件等待数据就绪(beta_decay.h:90-100);
- 计算完 `V→MTE3` 事件后写出,`pipe_barrier(PIPE_MTE3)` 收尾(beta_decay.h:125-141)。

fp32 直接在计算缓冲上搬运;fp16/bf16 先搬入 conv 缓冲再 `vconv_f162f32` / `vconv_bf162f32` 转 fp32,输出前反向转换(`vconv_f322f16` / `vconv_f322bf16r`)。

### 关键计算步骤

**Beta()(beta_decay.h:79-143)**,对每行 b:

1. `vmuls(-1.0)` → `vexp` → `vadds(1.0)`:得 `1 + e^-x`;
2. `vdiv(b_buf, ones_buf, b_buf)`:sigmoid(x) = 1 / (1 + e^-x)(用 ones_buf 做被除数实现求倒);
3. 写回 `beta`。

**Decay()(beta_decay.h:145-252)**:

1. 循环外:A_log 搬入转 fp32 → `vexp` → `vmuls(-1.0)` 得每头 `-exp(A_log)`(beta_decay.h:166-171);dt_bias 搬入转 fp32 备用;
2. 对每行 a:`vadd(a + dt_bias)` → `vexp` → `vadds(1.0)` → `vln`:softplus(x) = ln(1 + e^x)(beta_decay.h:215-228);
3. `vmul` 乘上 `-exp(A_log)`:得 g,写回(beta_decay.h:231-232)。

### 边界处理

- `num_v_heads` 非 32B 块整数倍时,`num` 向上取整搬运(尾部会多搬补齐字节,但向量计算 `calcRepeat` 覆盖有效元素;写出同样按对齐块数);
- bsz × seqlen 不是 block_num 整数倍时轮转条件自然跳过空闲核;
- 每行元素数(num_v_heads)须 ≤ 向量一次 repeat 能覆盖的范围(calcRepeat 由 num 派生,16 头场景远在限内)。
