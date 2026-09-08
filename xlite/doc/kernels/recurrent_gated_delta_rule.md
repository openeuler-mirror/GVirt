# recurrent_gated_delta_rule

## 功能概述

门控 Delta Rule 线性注意力(Qwen3.5 等混合架构中 GDN block)的逐步递推 kernel(对应模型侧 "GDN Step 7"):对每个 (batch, head) 沿序列逐 token 更新循环状态 `S ∈ R^{kDim×vDim}` 并输出注意力结果。g 以 log-space 传入,kernel 内取 `exp(g)` 作为衰减;Q/K 的 L2Norm 已在上游完成(对应 `use_qk_l2norm=False`,[csrc/kernels/recurrent_gated_delta_rule.h:4-7](../../csrc/kernels/recurrent_gated_delta_rule.h#L4-L7))。

每个 token 的数学语义(与 `tests/models/qwen3_5.py::_torch_recurrent_gated_delta_rule` 一致):

```
q_t ← q_t * scale                       scale = 1/sqrt(kDim)
S   ← S * exp(g_t)                      状态衰减
kv  = Sᵀ k_t                            (kv[v] = Σ_k S[k,v]·k[k])
δ   = (v_t − kv) * β_t                  delta rule 误差
S   ← S + k_t δ_tᵀ                      (S[k,v] += k[k]·δ[v])
o_t = Sᵀ q_t                            (o[v] = Σ_k S[k,v]·q[k])
```

## 输入输出参数

Python 接口:`recurrent_gated_delta_rule(rt, query, key, value, beta, g, state, out, batch, seqlen, num_heads, k_dim, v_dim, query_start_loc=None, query_lens=None)`。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| query / key | 输入 | `[T, num_heads * k_dim]`(T = batch·seqlen 或 packed 总长) | float32 / float16 / bfloat16 | 行主序、token 打平;q/k 须已 L2 归一化 |
| value | 输入 | `[T, num_heads * v_dim]` | 同上 | 值向量 |
| beta | 输入 | `[T, num_heads]` | 同上 | delta 门控(模型侧通常为 sigmoid 输出) |
| g | 输入 | `[T, num_heads]` | 同上 | log-space 衰减(负数),kernel 内 `exp` |
| state | 输入/输出 | `[batch, num_heads, k_dim, v_dim]` | 同上 | 循环状态,原地更新 |
| out | 输出 | `[T, num_heads * v_dim]` | 同上 | 每 token 的注意力输出 |
| batch / seqlen | 标量 | - | uint32 | 均匀 batch 模式;`seqlen=0` 表示 packed 模式([csrc/kernels/recurrent_gated_delta_rule.h:13-16](../../csrc/kernels/recurrent_gated_delta_rule.h#L13-L16)) |
| num_heads / k_dim / v_dim | 标量 | - | uint32 | kDim、vDim ≤ 128(`GDR_MAX_K_DIM/GDR_MAX_V_DIM`,host 侧强制检查,[csrc/op.cpp:2074-2077](../../csrc/op.cpp#L2074-L2077)) |
| query_start_loc / query_lens | 输入(可选) | `[batch]` | int32 | packed 混合长度模式:第 b 条序列 token 区间为 `[queryStartLoc[b], +queryLens[b])`;均匀模式传 None |
| scale | 标量(host 生成) | - | float | 固定 `1/sqrt(kDim)`([csrc/op.cpp:2115-2118](../../csrc/op.cpp#L2115-L2118)) |

## 支持的数据类型

- `float`([recurrent_gated_delta_rule_float.cpp](../../csrc/kernels/recurrent_gated_delta_rule_float.cpp)):fp32 输入直接搬运,无转换;仅走标量慢路径
- `float16_t`([recurrent_gated_delta_rule_float16_t.cpp](../../csrc/kernels/recurrent_gated_delta_rule_float16_t.cpp))
- `bfloat16_t`([recurrent_gated_delta_rule_bfloat16_t.cpp](../../csrc/kernels/recurrent_gated_delta_rule_bfloat16_t.cpp))

fp16/bf16 输入在 UB 内升到 fp32 计算;16-bit dtype 且 `kDim==vDim==128` 时启用向量化快路径(见下)。要求 8 个张量 dtype 一致([csrc/op.cpp:2105-2110](../../csrc/op.cpp#L2105-L2110))。

## 实现原理

实现位于 [csrc/kernels/recurrent_gated_delta_rule.h](../../csrc/kernels/recurrent_gated_delta_rule.h),`RecurrentGatedDeltaRule` 类(Init / Process 两段式),C220 向量核。

### 并行划分

`Process` 以 (batch × num_heads) 为总任务数条带化分配:`idx = GetBlockIdx(); idx += GetBlockNum()`,每个 (b, h) 对的整个序列在同一个 block 内串行递推——这是递推依赖决定的(状态沿 token 链式依赖,无法按 token 并行)([csrc/kernels/recurrent_gated_delta_rule.h:507-512](../../csrc/kernels/recurrent_gated_delta_rule.h#L507-L512))。

序列长度获取:均匀模式 `t = b * seqlen + s`;packed 模式由 host 置 `seqlen=0`,kernel 从 GM 读 `queryStartLoc[b]`/`queryLens[b]`(不判 GM 指针空,CANN 可能传非空哑指针,[csrc/kernels/recurrent_gated_delta_rule.h:393-401](../../csrc/kernels/recurrent_gated_delta_rule.h#L393-L401))。

### UB 内存布局

Init 时从地址 0 顺序分配,全部 256B 对齐([csrc/kernels/recurrent_gated_delta_rule.h:71-117](../../csrc/kernels/recurrent_gated_delta_rule.h#L71-L117)):

| 区域 | Dtype | 大小 | 用途 |
|---|---|---|---|
| qIn / kIn | Dtype | kDim | 16-bit 输入暂存(转 fp32 前的落点) |
| vIn | Dtype | vDim | 同上 |
| stateIn | Dtype | kDim×vDim | 状态搬运暂存 |
| outTmp | Dtype | max(vDim,32) | 输出暂存 / 标量读取中继 |
| qF / kF | float | kDim | 升精度后的 q/k |
| vF | float | vDim | 升精度后的 v |
| stateF | float | kDim×vDim | fp32 状态(整个递推驻留 UB) |
| kvMem / delta / outF / tmpRow | float | vDim | 中间向量 |
| scalarF | float | 1 | 标量↔向量交接(g/beta/exp) |
| tile / prod / offZero(仅 16-bit) | float/u32 | 128×64 fp32 ×2 + 128 | kDim=vDim=128 快路径的广播 tile、乘积 tile、全零 lane 偏移表 |

状态 128×128 fp32 = 64KB,加上 16-bit 暂存与两个 32KB tile,总占用控制在 192KB UB 内;fp32 时 16-bit 暂存区翻倍,tile+prod 会超出 UB 约 6KB,故 fp32 固定走慢路径([csrc/kernels/recurrent_gated_delta_rule.h:101-116](../../csrc/kernels/recurrent_gated_delta_rule.h#L101-L116) 注释)。

### 数据流与流水线同步

每 token 的搬运与计算以事件标志衔接:

- `LoadVec`(GM→UB):16-bit 路径先 `copy_gm_to_ubuf` 到暂存区,`PIPE_MTE2→PIPE_V` 事件后 `vconv` 升 fp32;fp32 直接落 fp32 缓冲([csrc/kernels/recurrent_gated_delta_rule.h:155-171](../../csrc/kernels/recurrent_gated_delta_rule.h#L155-L171));
- `StoreVec`(UB→GM):先 `vconv` 降精度(`PIPE_V→PIPE_MTE3` 事件),再 `copy_ubuf_to_gm`([csrc/kernels/recurrent_gated_delta_rule.h:173-188](../../csrc/kernels/recurrent_gated_delta_rule.h#L173-L188));
- 标量载荷(g、beta 为逐 token 标量):`LoadScalar` 把单个元素搬入 UB、转 fp32 后经 `PIPE_V→PIPE_S`/`PIPE_S→PIPE_V` 事件交还标量单元;`ExpScalar` 同理在向量单元执行 `vexp` 后回读([csrc/kernels/recurrent_gated_delta_rule.h:212-245](../../csrc/kernels/recurrent_gated_delta_rule.h#L212-L245));
- 每 token 结束时 `PIPE_MTE3→PIPE_MTE2` 事件确保上一写回完成后才开始下一 token 的读入([csrc/kernels/recurrent_gated_delta_rule.h:484-487](../../csrc/kernels/recurrent_gated_delta_rule.h#L484-L487))。

长向量转换与缩放(`ConvHalfToFloat`/`ConvFloatToHalf`/`VMulsLarge`)均按 `VECTOR_MAX_REPEAT=255` 分段循环,处理 128×128 状态需要 256 个 repeat 的情况([csrc/kernels/recurrent_gated_delta_rule.h:120-153](../../csrc/kernels/recurrent_gated_delta_rule.h#L120-L153))。

### 慢路径:逐 k 行标量展开

`ProcessOneHead` 慢路径([csrc/kernels/recurrent_gated_delta_rule.h:417-488](../../csrc/kernels/recurrent_gated_delta_rule.h#L417-L488))把两个 matvec/秩一更新按 k 维逐行展开:

1. q/k/v 装载并升精度,`vmuls(qF, qF, scale)`;
2. 一次性把 kF/qF 的全部 kDim 个标量在 `PIPE_V→PIPE_S` 单次往返中拷入标量数组 `kArr/qArr`(替代逐元素两次往返,[csrc/kernels/recurrent_gated_delta_rule.h:427-437](../../csrc/kernels/recurrent_gated_delta_rule.h#L427-L437));
3. `VMulsLarge(stateF, stateF, gExp, kDim*vDim)` 先衰减;
4. `kv[v] = Σ_k S[k,v]·k[k]`:循环 kDim 次 `vmuls(tmpRow, stateF + ki*vDim, kArr[ki])` + `vadd(kvMem, ...)`([csrc/kernels/recurrent_gated_delta_rule.h:448-454](../../csrc/kernels/recurrent_gated_delta_rule.h#L448-L454));
5. `δ = (vF − kvMem) * beta`(`vsub` + `vmuls`);
6. `S[k,:] += kArr[ki] * δ`:再次 kDim 次循环(`vmuls`+`vadd`);
7. `o[v] = Σ_k S[k,v]·qArr[ki]`:同 4 的循环;
8. `StoreVec` 写出 o_t;序列结束后把最终 stateF 写回 GM([csrc/kernels/recurrent_gated_delta_rule.h:490-492](../../csrc/kernels/recurrent_gated_delta_rule.h#L490-L492))。

### 快路径(TokenStepFast,kDim=vDim=128 且 16-bit)

把状态矩阵视作两个 `[128][64]` 的 V 半区 tile,全用向量指令完成([csrc/kernels/recurrent_gated_delta_rule.h:301-378](../../csrc/kernels/recurrent_gated_delta_rule.h#L301-L378)):

1. **广播 tile**:`BuildBroadcastTile` 用 `vgather` + 全零 lane 偏移表 `offZero`,把 kF/qF 的每个元素广播成 `tile[i][0..63]` 一行(128 行)([csrc/kernels/recurrent_gated_delta_rule.h:282-289](../../csrc/kernels/recurrent_gated_delta_rule.h#L282-L289));vgather 的 UB 写回需要一次 `PIPE_V→PIPE_S` 往返围栏(`VFenceVToS`)保证可见性;
2. **matvec**:`prod = stateF[:, half] ⊙ tile`(`vmul`),`ReduceRows` 对 128 行做步长 64→32→…→1 的树状原地 `vadd` 归约,得到 `kv[half]`(归约实现见 [csrc/kernels/recurrent_gated_delta_rule.h:291-299](../../csrc/kernels/recurrent_gated_delta_rule.h#L291-L299),调用处 [csrc/kernels/recurrent_gated_delta_rule.h:332-341](../../csrc/kernels/recurrent_gated_delta_rule.h#L332-L341));
3. **delta**:`vsub` + `vmuls` 同慢路径;
4. **秩一更新**:`vmadd` 的 `src1RepeatStride=0` 在该芯片上有缺陷,改用 `vmul`(src1RepeatStride=0,广播 δ 行)+ `vadd` 实现 `stateF[:, half] += tile ⊙ δ`([csrc/kernels/recurrent_gated_delta_rule.h:350-359](../../csrc/kernels/recurrent_gated_delta_rule.h#L350-L359) 注释);
5. **输出 matvec**:以 qF 重建广播 tile 后重复步骤 2,得到 outF([csrc/kernels/recurrent_gated_delta_rule.h:361-372](../../csrc/kernels/recurrent_gated_delta_rule.h#L361-L372))。

快路径整个 token 无逐元素标量操作(避免该部件的寄存器复用冒险),先后顺序保证 `kv` 读到的是已衰减状态(先 `state *= exp(g)` 再算 kv,[csrc/kernels/recurrent_gated_delta_rule.h:325-326](../../csrc/kernels/recurrent_gated_delta_rule.h#L325-L326) 注释)。

### 测试参考

[tests/kernels/recurrent_gated_delta_rule.py](../../tests/kernels/recurrent_gated_delta_rule.py):三种 dtype × batch {1,2} × seqlen {1,4,8}(H=4,K=V=64)、Qwen3.5-0.8B 形状(H=16,K=V=128,触发快路径)以及 packed 混合长度([1,4,2]、[3,1]);q/k 先做 L2Norm 再喂入,与 fp32 torch 递推参考对比 out 与最终 state。
