# experts_counts_sum

## 功能概述

MoE AllToAll 通信前的计数归约：对"每个 DP rank × 每个专家"的 token 计数矩阵做两个方向的求和——(1) 按专家所属 EP 组聚合，得到每个 DP rank 发往各 EP 组的 token 总数 `tokens_per_epgroup`（用于 AllToAll 通信量规划）；(2) 按专家列求和，得到每个专家的全局 token 总数 `experts_counts_output`（用于分组 GEMM 的 counts 输入）。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| experts_counts_input | 输入 | [ep_size, n_routed_experts] | int32 | 每行对应一个 DP rank，每列一个专家的 token 计数 |
| tokens_per_epgroup | 输出 | [ep_size, ep_size] | int32 | `tokens_per_epgroup[dp_idx, ep_id]` = DP rank dp_idx 发往 EP 组 ep_id 的 token 总和；ep_id = expert // (n_routed_experts/ep_size)。host 侧取 `tokensPerEpgroup.shape[0]` 作 ep_size（`csrc/op.cpp:1801`） |
| experts_counts_output | 输出 | [n_routed_experts] | int32 | `output[e] = Σ_dp counts[dp, e]`，每个专家跨全部 DP rank 的总计数 |
| n_routed_experts | 标量 | - | uint32_t | 专家总数，须被 ep_size 整除（测试约束，`tests/kernels/experts_counts_sum.py:37-38`） |
| ep_size | 标量 | - | uint32_t | DP/EP rank 数（隐含等于输入行数） |

Python 调用方式（`tests/kernels/experts_counts_sum.py:63-69`）：`experts_counts_sum(rt, experts_counts_input, tokens_per_epgroup, experts_counts_output, n_routed_experts)`。

## 支持的数据类型

| 变体 | 源文件 | kernel 符号 |
|------|--------|-------------|
| 纯 cpp，元数据固定 int32 | `csrc/kernels/experts_counts_sum.cpp` | `experts_counts_sum` |

仅支持 `__DAV_C220_VEC__`。

## 实现原理

单 kernel 两阶段，launch 时使用全部 AIV block（`rt.aivNum`，`csrc/op.cpp:1799`）。

### 任务切分（`csrc/kernels/experts_counts_sum.cpp:24-39`）

- Phase 1 按输入行（DP rank）切分：`ep_size` 行按 block 均分（带余数补偿），每 block 负责 `[row_start, row_end)` 行；
- Phase 2 固定由一个 block 执行：取第一个空闲 block（`ep_size < block_num` 时为 block `ep_size`，否则 block 0，`csrc/kernels/experts_counts_sum.cpp:39`）。Phase 1 与 Phase 2 无依赖（都是对输入的独立归约），可以并行执行。

### Phase 1：tokens_per_epgroup（`csrc/kernels/experts_counts_sum.cpp:42-75`）

对每个负责的 DP rank 行：

1. GM→UB 搬入整行 `c_row`（n_routed_experts 个 int32，`csrc/kernels/experts_counts_sum.cpp:54`）；
2. `vector_dup` 清零 `ep_counts_row`（ep_size 个 int32）；
3. 标量循环：`ep_counts_row[curr_expert / experts_ep_group] += c_row[curr_expert]`（experts_ep_group = n_routed_experts / ep_size，`csrc/kernels/experts_counts_sum.cpp:64-67`）；
4. UB→GM 写到 `tokens_per_epgroup[ep_idx, :]`（`csrc/kernels/experts_counts_sum.cpp:71-73`）。

### Phase 2：experts_counts_output（`csrc/kernels/experts_counts_sum.cpp:78-109`）

单个 block 把整张输入（`ep_size * n_routed_experts` 个 int32）一次性搬入 UB `c`，清零 `experts_counts` 后按行列双重标量循环累加：`experts_counts[curr_expert] += c_row[curr_expert]`（`csrc/kernels/experts_counts_sum.cpp:97-102`），最后写回 GM 的 experts_counts_output。

### 流水线同步

各阶段内部用 MTE2→S、S→MTE3 的 EVENT_ID0 握手保证"搬入完成→标量读写→写出完成"的顺序（如 `csrc/kernels/experts_counts_sum.cpp:55-56,69-70`）；`vector_dup` 后用 `pipe_barrier(PIPE_V)` + V→S flag 等向量写生效（`csrc/kernels/experts_counts_sum.cpp:60-62`）。Phase 2 结束处 `pipe_barrier(PIPE_ALL)` 收尾。两阶段间不需要跨 block 同步（输出互不相交）。

### 约束

整张计数矩阵须能放进单个 block 的 UB（Phase 2 一次性搬入 `ep_size * n_routed_experts * 4` 字节）；测试覆盖 ep_size ∈ {8,16}、n_routed_experts ∈ {160,256}（`tests/kernels/experts_counts_sum.py:35-36`），即最大 1KB×16 行量级。
