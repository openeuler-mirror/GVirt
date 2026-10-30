# recurrent_gated_delta_rule_paged

## 功能与输入输出

直接在 native VK 状态页上计算 grouped-head gated delta rule。
`recurrent_gated_delta_rule_paged` 处理每请求一个 token；
`recurrent_gated_delta_rule_prefill_paged` 处理 packed 变长请求。
旧 `recurrent_gated_delta_rule` 及其 dense/packed kernel 已删除。

| 参数 | 方向 | Shape / dtype |
| --- | --- | --- |
| query / key | 输入 | `[T,Hk*K]`，FP32/FP16/BF16 |
| value / out | 输入 / 输出 | `[T,Hv*V]`，与 query 同 dtype |
| beta | 输入 | `[T,Hv]`，与 query 同 dtype |
| g | 输入 | `[T,Hv]`，activation dtype 或 FP32，log-space decay |
| state | 输入输出 | `[P,Hv,V,K]`，activation dtype 或 FP32 |
| indices | 输入 | 连续 `int32[B]`，decode 可选 raw table 形式 |
| starts / initial | Prefill 输入 | 连续 `int32[B+1]` / `int32[B]` |

所有 activation 和输出连续；state 内层连续，dim-0 可有页间 padding，支持 storage offset。
Hk/Hv 必须为正且 Hk 整除 Hv；K 是 16 的倍数且不超过 128；V 为 64 或 128。
Q/K 由调用者预先 L2 归一化，kernel 对 Q 施加 `1/sqrt(K)`。

## 数学与实现

对 value head h，读取 key head `h // (Hv/Hk)`。每个 token 执行：

```text
S = exp(g) * S
Delta = beta * (v - S @ k)
S = S + Delta[:, None] * k[None, :]
out = S @ (q / sqrt(K))
```

实现位于 `csrc/kernels/recurrent_gated_delta_rule_paged.h`。
按 sequence/head/64 个 V 行划分 Vector core 任务，将 state tile 留在 UB 中作 FP32 运算。
Prefill 对 tile 内的 tokens 顺序递推，最后提交状态；它不是 chunk WY/矩阵乘算法。
Q/K 无需 repeat-interleave，SSM 无需 VK/KV 转置或全局临时状态。
Decode 与 Prefill 各有 FP32/FP16/BF16 activation 实例，文件名分别为
`recurrent_gated_delta_rule_paged_*` 和 `recurrent_gated_delta_rule_prefill_paged_*`。

initial 为零时忽略原页内容，从零状态开始；非零时从当前页继续。
空段不改变页面。零或负页号输出零且不更新 state；有效正页号必须在页范围内，
各请求的有效写入页不能重叠。调用者负责 pages、starts 和 flags 的内容正确性。

Decode 的 raw-table 模式与 [paged Conv](conv1d_and_silu_paged.md) 使用同一页选择约定。
所有 tensors 位于 Runtime 的 NPU；Python binding 同步，C++ launch 使用 Runtime stream。

## 验证

`recurrent_gated_delta_rule_paged.py` 验证 dtype、分组头、维度、raw-table 选择和非法 descriptor。
`gdn_prefill_paged.py` 对独立 FP32 recurrence 校验输出、最终 state、页保护和跨阶段交接。
`tests/models/gdn_paged.py` 验证小型混合模型的 V1/V2 metadata 和分段执行一致性。
