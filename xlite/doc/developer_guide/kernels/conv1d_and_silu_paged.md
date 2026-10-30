# conv1d_and_silu_paged

## 功能与接口

直接在调用者管理的原生 Conv 页面上执行 causal Conv1d + SiLU，并原位更新历史。
单 token 使用 `linear_att_conv_and_silu_paged`，packed 多 token 使用
`conv1d_and_silu_prefill_paged`。旧 dense 和 token Conv 入口已删除。

| 参数 | 方向 | Shape / dtype | 约束 |
| --- | --- | --- | --- |
| input / output | 输入 / 输出 | `[T,C]`，FP32/FP16/BF16 | 连续且 dtype 一致；decode 的 T=B |
| state | 输入输出 | `[P,W-1,C]`，与 input 同 dtype | 内层连续，允许 dim-0 页间 padding 和 storage offset |
| weight | 输入 | `[C,W]` 或 `[C,1,W]` | 连续；W=2..4，C 是 64 的正倍数 |
| indices | 输入 | `int32[B]` | 连续；Conv 与 SSM 可使用不同页号 |
| starts | Prefill 输入 | `int32[B+1]` | 从 0 到 T 的单调累计 token 边界 |
| initial | Prefill 输入 | `int32[B]` | 非零保留历史，零表示从零历史开始 |

Decode 还支持二维 raw block table：列连续、行可有 padding，另传连续
`cached_lens`、正的 `state_block_size` 和 `aligned`。非 aligned 模式选择第零列；
aligned 模式选择 `cached_lens // state_block_size`。该 raw-table decode 约定要求
cached_lens 为正；fresh 请求应调用带 initial flags 的 Prefill 入口。

## 状态、舍入和并行

每个 sequence/channel tile 由一个 Vector core 负责，64 通道为一组。
历史、当前输入和权重搬入 UB，升到 FP32 后按 tap 累加，再执行 SiLU。
Decode 保留 fused 的 FP32 SiLU 后舍入；Prefill 先将卷积结果舍入到 activation dtype，
再做 SiLU，与 native conv1d 后接 SiLU 的顺序一致。

Prefill 将滑动窗口留在 UB 中连续处理该请求的 tokens，最后只写回一次历史。
空段不改变页面。零或负页号输出零且不访问页面；正页号必须在物理页范围内。
同一次 launch 的有效请求必须拥有互不重叠的写入页。页的分配、清理和请求映射由调用者负责。

实现位于 `csrc/kernels/conv1d_and_silu_paged.h`。
Decode 实例为 `conv1d_and_silu_paged_{float,float16_t,bfloat16_t}.cpp`，
Prefill 实例为 `conv1d_and_silu_prefill_paged_{float,float16_t,bfloat16_t}.cpp`。
所有 Tensor 必须位于 Runtime 的 NPU。binding 显式允许 state 的页间 pitch，
通过主分支的 XTensor.stride 传入；不复制、不转置状态。

## 验证

`tests/kernels/gdn_native_pages.py` 覆盖 decode 的页重排、raw tables、padding、offset
和 guard 保护；`gdn_prefill_paged.py` 覆盖 fresh/continuation、空段、混合长度和
Prefill → Decode 的同页交接。旧 W>4、W=1 等配置不在新接口支持范围内。
