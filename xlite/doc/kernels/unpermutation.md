# unpermutation

## 功能概述

MoE token 逆重排（unpermutation）：把按专家分组排序的计算结果 `input` 按路由位图和门控权重还原为按 token 排列的输出，即对每个 token 做 `output[t, :] = Σ_e w[t, e] * input[start[e] + j(e,t), :]`（e 取 token t 命中的专家）。与 `permutation` 配对，实现 MoE experts 计算后的 combine。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| input (in) | 输入 | [max_expert_sorted, dim] | bfloat16 | 按专家分组排序的 experts 输出（permutation 的 output）。host 侧传 `in.shape[1]` 作 dim（`csrc/op.cpp:833-834`）；元素大小硬编码为 `sizeof(bfloat16_t)`（`csrc/kernels/unpermutation.h:27`） |
| routing_map (routing) | 输入 | [num_tokens, n_routed_experts] 位图（BIT1，每行 n_routed_experts/8 字节） | uint8 位图 | 与 permutation 相同的路由位图 |
| weights_map (weights) | 输入 | [num_tokens, n_routed_experts] | bfloat16 或 float | 门控权重 w[t, e]；host 侧传 `weights.shape[1]` 作 n_routed_experts。bfloat16 变体要求 weights 为 BF16（`csrc/op.cpp:824`）；float 变体用于 in/out 为 BF16 而 weights 为 FP32 的组合（`csrc/op.cpp:826`），此时权重直接从 GM 按标量读取 |
| output (out) | 输出 | [num_tokens, dim] | bfloat16 | 还原并加权求和后的 token；host 侧传 `out.shape[0]` 作 n_tokens |
| unp_idx | 输入 | [n_routed_experts, num_tokens+1] | int32 | permutation 写出的索引：`[e, t]` 为 token t 在专家 e 段内的行号 j，最后一列 starts[e] 为专家 e 段起始行号（`csrc/kernels/unpermutation.h:19`） |
| experts_start_idx / experts_end_idx | 标量 | - | uint32_t | 本卡负责的专家区间 [start, end) |

Python 调用方式（`tests/kernels/permutation.py:88`）：`unpermutation(rt, experts_sorted, routing_xlite, weights, START, END, token_sorted, unp_idx)`。

## 支持的数据类型

| dtype 变体 | 实例化文件 | kernel 符号 | host 侧 dtype 条件（`csrc/op.cpp:824-827`） |
|------|-----------|-------------|------|
| bfloat16_t | `csrc/kernels/unpermutation_bfloat16_t.cpp` | `unpermutation_bfloat16_t` | in/out/weights 均为 BF16 |
| float | `csrc/kernels/unpermutation_float.cpp` | `unpermutation_float` | in/out 为 BF16，weights 为 FP32（权重从 GM 按 float 标量读取） |

模板定义在 `csrc/kernels/unpermutation.h`；两个变体中 input/output 数据通路均按 bfloat16 搬运（`sizeof(bfloat16_t)` 硬编码，`csrc/kernels/unpermutation.h:27`），差异仅在权重的读取与换精度路径。仅支持 `__DAV_C220_VEC__`（否则宏导出空实现，`csrc/kernels/unpermutation.h:134-141`）。

## 实现原理

### 数据流（`csrc/kernels/unpermutation.h:62-120`）

token 维多 block 并行：`for (curr_token = block_idx; curr_token < n_tokens; curr_token += block_num)`（`csrc/kernels/unpermutation.h:63`）。每个 token：

1. 从 GM 搬入该 token 的 routing 位图行（`routing`，n_routed_experts/8 字节）；bfloat16 变体同时搬入 `weights_map[curr_token, :]` 整行（n_routed_experts 个 bf16）；
2. bfloat16 变体用 `vconv_bf162f32` 把权重整行转成 FP32 存 UB（`w_float_addr`，`csrc/kernels/unpermutation.h:76-85`），之后按专家标量读取；float 变体直接从 GM 按 `weights_map[curr_token*n_routed_experts + i]` 读 float 标量（`csrc/kernels/unpermutation.h:98-100`）；
3. 累加器初始化：`vector_dup` 清零 FP32 累加缓冲 `row2` 和 BF16 结果缓冲 `row3`（`csrc/kernels/unpermutation.h:88-89`）；
4. 对 [start, end) 内每个位图命中的专家 i：
   - 读 `starts[i]`（专家段起始，GM 标量读）与 `unp_idx[i, curr_token]`（组内行号 j，`off_T == 0` 特判为 0，规避被 permutation2 覆盖的 `[0,0]`，`csrc/kernels/unpermutation.h:92-94`）；
   - 搬入 `input[start[i]+j, :]`（`row`，dim 个 bf16）→ `vconv_bf162f32` 升到 FP32（`rowf32`）→ `vmuls` 乘权重 w → `vadd` 累加进 `row2` → `vconv_f322bf16r` 舍入转回 BF16 存 `row3`（`csrc/kernels/unpermutation.h:104-113`）。每个专家迭代都刷新一次 BF16 结果，保证数值路径与测试参考一致（参考实现用 FP32 累加 + 单次最终转换，`tests/kernels/permutation.py:48-54`）；
5. 把 `row3`（dim 个 bf16）写回 `output[curr_token, :]`（`csrc/kernels/unpermutation.h:118`）。

### UB 内存布局（`csrc/kernels/unpermutation.h:31-59`）

手工偏移分配：`row`（一行 bf16）、`rowf32`（FP32 转换缓冲）、`row1`（乘权重结果）、`row2`（FP32 累加器）、`row3`（BF16 输出）、`routing`（位图行）、bfloat16 变体额外的 `w_float_addr` / `w_bf16_addr`（权重换精度缓冲）。每段按 VECTOR_MAX_BYTESIZE（256B）对齐。

### 流水线同步

- 行内专家循环中，`PIPE_V → PIPE_MTE2`、`PIPE_MTE2 → PIPE_V` 的 EVENT_ID0 交替握手，使下一专家行的 MTE2 搬入与上一专家行的向量计算重叠（`csrc/kernels/unpermutation.h:102-106`）；
- token 级：行计算（V）写回（MTE3）之间用 `PIPE_V → PIPE_MTE3` EVENT_ID0 同步；写回完成后再归还给下一轮的 V 阶段（`csrc/kernels/unpermutation.h:116-119`），配合初始 `set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0)`（`csrc/kernels/unpermutation.h:62`）形成跨 token 的读写重叠；
- 权重换精度路径额外用 MTE2→V、V→S 的 EVENT_ID1 同步（`csrc/kernels/unpermutation.h:79-84`）；
- kernel 末尾 `wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0)` + `pipe_barrier(PIPE_ALL)` 收尾（`csrc/kernels/unpermutation.h:121-122`）。
