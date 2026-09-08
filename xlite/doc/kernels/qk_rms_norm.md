# qk_rms_norm

## 功能概述

对同一 qkv 缓冲中不相交的两段列区间(Q 段与 K 段)分别做 RMSNorm,并融合为**单次 kernel 启动**。它把两次独立的 `norm` 调用(plain qkNorm 时 Q 按 headDim 逐 head 归一化、K 按 headDim 逐 head 归一化;qkNormFull 时 Q/K 各按整段归一化)通过 coreOffset 接力机制串联,避免两次 launch 的开销。支持两种模式:

- `useNorm = true`:执行完整 RMSNorm(variance 为空时内联计算方差;非空时消费预先归一好的方差,即 qkNormFull 的 apply 阶段);
- `useNorm = false`:variance-only 阶段,只写每 token 的局部方差到 `qVariance` / `kVariance`,供 TP 场景先 AllReduce 再归一化。

数学语义(与 norm 的 Rms 变体一致):`y = x / sqrt(mean(x^2) + eps) * weight (+ bias)`。

## 输入输出参数

Python/模型侧无单独绑定,由 `XliteOpQkRmsNorm`(`csrc/op.cpp:1134-1161`)在模型前向中调用([csrc/model.cpp:594-616](../../csrc/model.cpp))。kernel 签名见 `csrc/kernels/qk_rms_norm.h:54-66`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input | 输入 | `[token_num, in_step]` | float16 / bfloat16 | qkv 拼接缓冲;Q 段从列 0 起,K 段从列 `k_start_offset` 起 |
| qNorm | 输入 | `[q_norm_dim]` | 同 input | Q 段仿射 γ(可为空) |
| qNormBias | 输入 | `[q_norm_dim]` | 同 input | Q 段仿射 β(可为空) |
| qOut | 输出 | `[token_num, out_step]`(useNorm)或 `[token_num, 1]`(variance-only) | 同 input / float32 | Q 段归一化结果,或 Q 每 token 方差 |
| kNorm / kNormBias | 输入 | `[k_norm_dim]` | 同 input | K 段仿射参数(可为空) |
| kOut | 输出 | 同 qOut | 同上 | K 段归一化结果或方差;K 段写回列 `k_start_offset` 起(useNorm)或列 0(variance-only) |
| token_num | 标量 | - | uint32_t | token 数,host 传 `in.shape[0]` |
| q_norm_dim / k_norm_dim | 标量 | - | uint32_t | Q/K 段单次归一化维度(plain qkNorm 为 headDim,qkNormFull 为 headDim×heads) |
| q_cnt_per_token / k_cnt_per_token | 标量 | - | uint32_t | Q/K 段内独立归一化的段数(逐 head 时为 head 数,整段时为 1) |
| in_step / out_step | 标量 | - | uint32_t | 行步长;variance-only 时 outStep=1 |
| norm_eps | 标量 | - | float | ε |
| k_start_offset | 标量 | - | uint32_t | K 段在行内的起始列(= qHeads × headDim) |
| useNorm | 标量 | - | bool | true=apply,false=variance-only |
| qVariance / kVariance | 输入/输出 | `[token_num, 1]` | float32 | apply 阶段作输入(预归约方差),variance-only 阶段作输出 |
| tpSize | 标量 | - | uint32_t | 张量并行规模 |

测试:无独立测试文件;行为由 [tests/kernels/rmsnorm_full.py](../../tests/kernels/rmsnorm_full.py) 覆盖的同一 norm 语义及 [tests/kernels/attention.py](../../tests/kernels/attention.py) 端到端覆盖。

## 支持的数据类型

- `float16_t`([qk_rms_norm_float16_t.cpp](../../csrc/kernels/qk_rms_norm_float16_t.cpp))
- `bfloat16_t`([qk_rms_norm_bfloat16_t.cpp](../../csrc/kernels/qk_rms_norm_bfloat16_t.cpp))

host 要求 `in.dtype` 为 FP16/BF16 且 `out.dtype` 为同 dtype 或 FP32(`csrc/op.cpp:1144-1147`)。

## 实现原理

实现位于 [csrc/kernels/qk_rms_norm.h](../../csrc/kernels/qk_rms_norm.h),函数 `qk_rms_norm<Dtype>`(qk_rms_norm.h:30-52)。**本身不做新计算,是 `norm<Dtype>` 模板(norm.h:222)的双段编排器**:

### coreOffset 接力

两段 Q/K 的读写区间互不相交、无数据依赖,所以可以背靠背发到同一批核上,而不是串成依赖链(qk_rms_norm.h:13-17 注释):

1. `coreOffset = 0`,调用 `norm<Dtype>` 处理 Q 段:`in_start_offset = 0`,传入 `&nextCoreOffset`;norm 内部按 `coreOffset + token_num` 轮转分配行,结束时把终点写回 `nextCoreOffset`(norm.h:359-361)。
2. `coreOffset = nextCoreOffset`,再调用 `norm<Dtype>` 处理 K 段:`in_start_offset = k_start_offset`;输出侧 `k_out_start_offset` 在 useNorm 时为 `k_start_offset`(写回 qkv 原位置),variance-only 时为 0(方差写到独立的 `[token_num,1]` 张量,host 侧 `outStep = 1`,见 op.cpp:1152-1156)。

接力让 Q 段从 block 0 起跳、K 段从 Q 段结束时的 block 起跳,两次遍历整体在所有 AIV 上均衡;相比两次独立 kernel launch 省去一次下发与同步。

### 两段参数的物理含义(qk_rms_norm.h:19-23 注释)

| 场景 | Q (normDim, cnt) | K (normDim, cnt) |
|---|---|---|
| plain qkNorm(逐 head) | (headDim, qHeads) | (headDim, kHeads) |
| qkNormFull variance 阶段(useNorm=false) | (headDim×qHeads, 1) | (headDim×kHeads, 1) |
| qkNormFull apply 阶段(useNorm=true + variance) | 同上 | 同上 |

每段实际的归一化、UB 布局、MTE2/V/MTE3 流水与边界处理全部复用 `norm` 的实现,详见 [norm.md](norm.md)。

### host 侧要点

`XliteOpQkRmsNorm`(op.cpp:1134-1161):`useNorm` 时 qOut/kOut 都指向 `out.ptr`、variance 张量作为参数传入;`!useNorm` 时 qOut/kOut 改指向各自 variance 张量、variance 参数传 null,从而用同一个 kernel 签名表达两个阶段。qkNormFull 的完整三步编排(两次 variance-only + 一次 AllReduce + 一次 apply)见 [csrc/model.cpp:598-617](../../csrc/model.cpp)。
