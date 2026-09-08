# permutation

## 功能概述

MoE token 重排（permutation）：按路由位图 `routing_map` 将每个 token 的 hidden 向量散射（scatter）到按专家分组排序的输出缓冲区中，同时产出每个专家的 token 计数 `counts`、每个 token 在其专家组内的序号 `unp_idx`（供 `unpermutation` 逆操作使用）。它是 MoE dispatch 侧的核心算子，与 `unpermutation` 配对完成"按专家分组 → 计算 → 还原 token 顺序"的流程。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| input (in) | 输入 | [num_tokens, dim] | bfloat16 | 原始 token hidden（permutation3 中元素大小硬编码为 `sizeof(bfloat16_t)`，`csrc/kernels/permutation.cpp:145`） |
| routing_map (routing) | 输入 | [num_tokens, n_routed_experts] 位图（BIT1，每行 n_routed_experts/8 字节） | uint8 位图 | token→专家路由结果，bit(t*n_routed_experts+e)=1 表示 token t 路由到专家 e（测试中由 int32 张量按位拼出，`tests/kernels/permutation.py:60-67`） |
| output (out) | 输出 | [max_expert_sorted, dim] | bfloat16 | 按专家分组排序后的 token；第 e 个专家的段起始偏移由 unp_idx 的 starts 列给出 |
| unp_idx | 输出 | [n_routed_experts, num_tokens+1] | int32 | 布局见下文：`[e, t]` = token t 在专家 e 段内的行号；最后一列为各专家段起始偏移；`[0,0]` 被复用存放总 token 数 |
| counts | 输出 | [n_routed_experts, 1] | int32 | 每个专家收到的 token 数（host 侧取 `counts.shape[0]` 作 n_routed_experts，`csrc/op.cpp:812-814`） |
| n_tokens / dim | 标量 | - | uint32_t | `in.shape[0]` / `in.shape[1]` |
| max_expert_sorted | 标量 | - | uint32_t | `out.shape[0]`，即排序缓冲区容量，kernel 内 assert 总数不超过它（`csrc/kernels/permutation.cpp:119`） |
| experts_start_idx / experts_end_idx | 标量 | - | uint32_t | 本卡（EP rank）负责的专家区间 [start, end)，区间外专家不计数、不搬数 |

Python 调用方式（`tests/kernels/permutation.py:70`）：`permutation(rt, x, routing_xlite, START, END, experts_sorted, unp_idx, experts_counts)`。

## 支持的数据类型

| 变体 | 源文件 | kernel 符号 |
|------|--------|-------------|
| 纯 cpp（无模板实例化） | `csrc/kernels/permutation.cpp` | `permutation` |

kernel 内部按 bfloat16 处理数据搬运（`sizeof(bfloat16_t)` 硬编码于 `csrc/kernels/permutation.cpp:145,181,190`），元数据（counts/unp_idx）固定 int32；仅支持 `__DAV_C220_VEC__`。

## 实现原理

kernel 分三个阶段（`csrc/kernels/permutation.cpp:201-225`），阶段间用 `ffts_cross_core_sync` + `wait_flag_dev` 做全 block 同步：

### 阶段 1：permutation1 —— 计数与组内序号（`csrc/kernels/permutation.cpp:10-94`）

- 专家区间按 block 切分：`experts_local / experts_remain` 均分 [start, end)，边界 block（block 0 与最后一个 block）被强制扩展到 [0, n_routed_experts)，但循环内跳过区间外的专家（`csrc/kernels/permutation.cpp:26-44,62-65`）。
- token 按 `TOKEN_TILE = 4096` 分块（`csrc/kernels/permutation.cpp:8`），每块把 routing 位图 tile 从 GM 搬到 UB（`routing`，约 `TOKEN_TILE * n_routed_experts / 8` 字节）。
- 对本 block 负责的每个专家 e：标量扫描 tile 内所有 token，若位图命中则 `routing_count[i] = c[e]；c[e]++`（c 为该专家的 UB 计数器），随后把整段 `routing_count` 写到 `unp_idx[e, token_start : token_start+token_tile]`（`csrc/kernels/permutation.cpp:68-79`）。即 `[e, t]` 记录 token t 在专家 e 段内的相对行号。
- 结束后把每个本地专家的计数 c 写到 GM 的 `counts[e]`（`csrc/kernels/permutation.cpp:85-91`）。

### 阶段 2：permutation2 —— 前缀和（仅 block 0，`csrc/kernels/permutation.cpp:96-129`）

- 从 GM 读回 counts，标量累加得到每个专家的排他前缀和 `s[e]`（专家 e 段在 output 中的起始行号）与总和 `sum`（assert `sum <= max_expert_sorted`）。
- `s[0..n_routed_experts)` 写入 `unp_idx` 的最后一列（即 `unp_idx + n_routed_experts*n_tokens*4` 处，`csrc/kernels/permutation.cpp:124-125`）；总和 `psum` 写入 `unp_idx[0]`（`csrc/kernels/permutation.cpp:126`），因此 `[0,0]` 位置的序号被覆盖 —— 阶段 3/`unpermutation` 读取时对 `off_T == 0` 特判为 0（`csrc/kernels/permutation.cpp:189`、`csrc/kernels/unpermutation.h:94`）。

### 阶段 3：permutation3 —— 数据散射（`csrc/kernels/permutation.cpp:131-199`）

- token 维多 block 并行：`for (curr_token = block_idx; curr_token < n_tokens; curr_token += block_num)`（`csrc/kernels/permutation.cpp:167`）。
- 每个 token：把该 token 的 routing 位图行搬入 UB（双缓冲 `routing[2]`），对 [start, end) 内每个命中专家 e：
  - 首个命中专家时把 `input[curr_token, :]` 整行搬入 UB（双缓冲 `row[2]`），后续命中专家复用该行（一个 token 被多个专家接收时只读一次 GM，`csrc/kernels/permutation.cpp:180-186`）；
  - 读 `unp_idx[e, curr_token]` 得组内行号 j，把该行写到 `output[s[e] + j, :]`（`csrc/kernels/permutation.cpp:187-191`，s 为阶段 2 的前缀和，从 `starts` 区域重新搬入 UB，`csrc/kernels/permutation.cpp:161`）。

### 流水线同步

- 阶段 1/2 内部用 MTE2/MTE3 ↔ S 的 `set_flag`/`wait_flag` 保证搬运与标量读写顺序；
- 阶段 3 用 `routing`/`row` 双缓冲 + `PIPE_MTE3 → PIPE_MTE2` 的 EVENT_ID0/1 乒乓，使上一 token 的 MTE3 写回与下一 token 的位图读入重叠（`csrc/kernels/permutation.cpp:165-197`）；
- 三个阶段之间 `ffts_cross_core_sync(PIPE_MTE3, config)` + `wait_flag_dev(flag_id)` 全核同步（`csrc/kernels/permutation.cpp:213-221`），保证 counts/starts 全部就绪后再进入下一阶段。
