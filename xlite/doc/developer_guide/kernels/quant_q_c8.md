# quant_q_c8

## 功能概述

C8 查询量化算子(V4 indexer 路径):把连续 BF16 行——rope 之后但**未**做 rotate_activation 的 indexer Q——量化为稠密 INT8 `q8`,并把逐行去量化因子与逐 (token, head) 投影权重融合成 fp16 `scaledWeights`,二者正是 [indexer_topk](indexer_topk.md) C8(int8)分支消费的 `q` 与 `weight` 输入。行号 `r` 对应 `(token, head) = (r / heads, r % heads)`,`q`、`headWeight`、`q8`、`scaledWeights` 全部使用同一扁平行序,无需 slot mapping。

```
absmax      = abs_max(q_row)                     (不施加 rotate)
q8_row      = round_nearest(q_row * 127/absmax)  (与 indexer_prepare_c8 / quant_cache_c8 逐字节一致;零行 → 全零)
scaledWeights[r] = fp16(headWeight[r] * absmax * rotateScale / 127)
```

`rotateScale` 是 rotate_activation 因子(V4 为 1/√head_dim)。浮点路径把它整行施加(rope 区在 `rope_complex_and_cache` 的 cache/doRotate 路径内、nope 区由调用方 Muls——两条路径合起来是**整行均匀**缩放);由于均匀行缩放不改变任何 int8 值,本算子把它折叠进去量化因子:调用方以 doRotate=false rope 并跳过 nope Muls,topk 分数与浮点路径只差每查询常数(排序与 ReLU 不受影响)。

它是 DSV4 Path B 第三阶段("V4 Q 量化")的落点:与 [quant_cache_c8](quant_cache_c8.md)(第二阶段,K 侧)配对,凑齐 topk int8 分支的全部输入。

## 输入输出参数

Python 入口 `quant_q_c8(rt, q, head_weight, q8, scaled_weights, rotate_scale)`(`csrc/_C.cpp` 的 `QuantQC8`,关键字 `m.def("quant_q_c8"`)。kernel 签名见 `csrc/kernels/quant_q_c8.h` 的 `quant_q_c8`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q | 输入 | `[rows, dim]` | bfloat16 | rope 后未缩放的 Q;dim 为 64 的倍数且 ≤ `QUANT_Q_C8_MAX_DIM`(4096) |
| headWeight | 输入 | `[rows]` | bfloat16 | 逐 (token, head) 投影权重(weights_proj 输出,已乘 indexer score scale) |
| q8 | 输出 | `[rows, dim]`(任意同 numel 形状) | int8 | 量化 Q,行写在 `row * dim` |
| scaledWeights | 输出 | `[rows]` | float16 | 融合权重 = headWeight × absmax × rotateScale/127,直接作 topk 的 weight 输入 |
| rotateScale | 标量 | - | float | rotate_activation 因子(V4:1/√128;1.0 = 纯量化) |
| rows | 标量 | - | uint32_t | host 传 `q.shape[0]`(= tokens × heads) |
| dim | 标量 | - | uint32_t | host 传 `q.shape[1]` |

host 侧检查(`csrc/op.cpp` 的 `XliteOpQuantQC8`):q 必须 BF16 二维、dim 为非零 64 倍数且 ≤ `QUANT_Q_C8_MAX_DIM`、headWeight 与行数一致、`q8.numel == rows * dim`、scaledWeights 与行数一致;`rows == 0` 直接返回。launch 用 `PickMinBlockNum(rt.aivNum, rows)`。

测试参考:[tests/kernels/quant_q_c8.py](../../../tests/kernels/quant_q_c8.py)(dim 64/128/192 × 行数 0/1/17/129/257 × rotateScale {1, 1/√128},零/常量/outlier 行,零/负/大 head_weight,输出 view 双端 guard)。

## 支持的数据类型

- 输入 `bfloat16_t` → 输出 `int8_t` + fp16 融合权重([quant_q_c8_bfloat16_t.cpp](../../../csrc/kernels/quant_q_c8_bfloat16_t.cpp))

host 仅接受 `q.dtype == BF16`,kernel 入口符号为 `quant_q_c8_bfloat16_t`(host 经 `aclrtlaunch_quant_q_c8_bfloat16_t` 启动)。

## 实现原理

实现位于 [csrc/kernels/quant_q_c8.h](../../../csrc/kernels/quant_q_c8.h),函数 `quant_q_c8<dtype>`(模板;BF16 实例化由 `QUANT_Q_C8_FUNC_DEFINE` 宏展开为入口符号 `quant_q_c8_bfloat16_t`)。

### 分块策略

- **行间多 Block 轮转 + ping-pong 流水**:`for (row = block_idx; row < rows; row += block_num)`,槽号 `curr` 每行翻转(`curr = 1 - curr`),GM→UB 载入位于循环顶部且仅此一处(见下文"流水线同步"):当前行的 MTE3 写出与下一行的 MTE2 载入重叠;行稠密(无 slot),无越界跳过分支。
- **headWeight 标量**:每行随 q 行一起经 2B b16 路径 MTE2 载入(`CopyGmToUbufAligned` 的 b16 分支,与 2B scale 写出同族),**紧跟一次 `convert_input` 转 fp32**——放在 `V→MTE2` 释放 flag 之前,使该 flag 同时覆盖 qRow 与 headWeightHalf 的生命周期。

### UB 内存布局(`csrc/kernels/quant_q_c8.h` 的 `quant_q_c8`)

跨管线缓冲双缓冲(MTE2 写的 qRow/headWeightHalf、MTE3 读的 quant/scaledWeightsF16 各两份),V 内部缓冲单份(V 管线自身串行化行),总容量以 `QUANT_Q_C8_MAX_DIM` 为上限(20B/元素 + 7 个 256B 暂存,运行时 `assert(off <= UB_SIZE)` 守护):

| 缓冲 | dtype | 用途 |
|---|---|---|
| qRow[2] | bfloat16 | 行输入(ping-pong) |
| headWeightHalf[2] | bfloat16 | headWeight 标量载入(2B,lane 0 有效;ping-pong) |
| qRowF32 | float | BF16→FP32;复用为 round-trip 的 fp32 中转 |
| absValues | float | `vabs` 中间,`ReduceMax` 结果落点(absmax 在 absValues[0]) |
| rounded | int32 | `vconv_f322s32r` 最近整数中转 |
| halfValues | half | FP32→FP16 中转 |
| quant[2] | int8 | `vconv_f162s8` 结果(ping-pong) |
| headWeightF32 | float | headWeight 的 fp32 形式(lane 0 有效) |
| scaleF32 | float | `vmuls(absValues, rotateScale/127)` 的折叠去量化因子(lane 0 有效) |
| scaledWeightsF32 | float | `vmul(scaleF32, headWeightF32)` 中转(lane 0 有效) |
| scaledWeightsF16[2] | half | fp16 融合权重,256B 对齐供 2B MTE3 写出(ping-pong) |

### 流水线同步

[muls](muls.md) 式 ping-pong,**per-slot 事件 ID**(`EVENT_ID0 + curr`)守护跨管线缓冲的归还:循环前预置四个归还 flag(`V→MTE2` × 2、`MTE3→V` × 2),`curr` 每行翻转,末尾 drain 四路 + `pipe_barrier(PIPE_ALL)`——空 launch 时循环不执行,四个预置 flag 恰好被 drain 配平,无需特判:

- **载入链**(循环顶部,GM→UB 仅此一处):等 `V→MTE2[curr]`(该槽上一位使用者两行前归还;首两行由预置 flag 满足)后把 q 行 + headWeight 标量载入槽 curr,`MTE2→V` 同 ID set/wait 对确认载入完成后两次 `convert_input`(qRowF32、headWeightF32),`V→MTE2[curr]` 归还槽 curr(供两行后的载入复用);
- **scale/factor 链**:`ReduceMax` 后 `V→S`,S 管线读 `*absValues` 得 absmax,标量算 `factor = 127/absmax`(零行为 0)作为量化 `vmuls` 的立即数;折叠去量化因子在 V 管线(`vmuls(scaleF32, absValues, rotateScale/127)`);
- **写出链**:写 scaledWeightsF16[curr]/quant[curr] 前等 `MTE3→V[curr]`(两行前的写出已归还);量化链完成后 `V→MTE3` 同 ID set/wait 对,两笔写出(int8 行 + 2B 融合权重)后 `MTE3→V[curr]` 归还——当前行的 MTE3 写出与下一行的 MTE2 载入重叠。

### 关键计算步骤(`csrc/kernels/quant_q_c8.h` 的 `quant_q_c8`)

1. 行 BF16 搬入并转 FP32;headWeight 2B 搬入并转 FP32(关键字 `convert_input<dtype>(headWeightF32`);
2. `vabs` + `ReduceMax` 得**未缩放**行 absmax(关键字 `vabs(absValues`);
3. S 管线读 absmax 算 factor(零行 → 0);V 管线 `vmuls(scaleF32, absValues, rotateScale/127)` → `vmul(scaledWeightsF32, scaleF32, headWeightF32)` → `vconv_f322f16(scaledWeightsF16)` 产出融合权重(关键字 `float descale`);
4. `vmuls(qRowF32, qRowF32, factor)` 缩放,`vconv_f322s32r → vconv_s322f32 → vconv_f322f16 → vconv_f162s8` 四连转 INT8(与 indexer_prepare_c8 的 Q/K 量化链逐条一致);
5. `CopyUbufToGmAligned` 写 int8 行到 `q8 + row*dim`(dim 为 64 倍数 → 块路径 32B 对齐恒成立,与 prepare_c8 的 q8 写出同款),2B 融合权重写 `scaledWeights + row`(b16 路径,关键字 `CopyUbufToGmAligned(q8Gm`)。

### 边界处理

- 零行:factor=0 → int8 全零,融合权重 = headWeight × 0 = 0;
- 负 head_weight:融合权重为负(fp16 原生支持),量化不受影响;
- dim 必须 64 的倍数(op 检查;`convert_input` 无 tail)且 ≤ `QUANT_Q_C8_MAX_DIM`(op 检查);
- host 侧 `q.shape[0] == 0` 直接返回(`csrc/op.cpp` 的 `XliteOpQuantQC8`)。

## 与浮点路径的等价性

浮点路径(model.cpp 的 `ForwardAttnCXAIndexer`)对 Q 施加 rope(doRotate=true,rope 区 × 1/√dim)+ Muls(nope 区 × 1/√dim)——**整行均匀** × 1/√dim。本算子接收 doRotate=false 的 rope 结果(未缩放),`q8` 的 int8 值与"先整行缩放再量化"完全相同(逐行 scale 不变性),`rotateScale` 折叠进 `scaledWeights` 后,topk 的 int8 分数 = 浮点分数 × 每查询常数(√dim 相关)——排序与 ReLU 均不变。K 侧([quant_cache_c8](quant_cache_c8.md))同理(cache 路径 rotate 为整行均匀缩放,被 k_scale 吸收)。
