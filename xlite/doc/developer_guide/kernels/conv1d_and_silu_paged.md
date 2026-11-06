# conv1d_and_silu_paged

## 功能概述

直接在调用者管理的原生 Conv 页面上执行 causal Conv1d + SiLU，并原位更新历史。单 token 使用 `linear_att_conv_and_silu_paged`，packed 多 token 使用 `conv1d_and_silu_prefill_paged`。旧 dense 和 token Conv 入口已删除。

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

## 支持的数据类型

实现位于 `csrc/kernels/conv1d_and_silu_paged.h`。Decode 实例为
`conv1d_and_silu_paged_{float,float16_t,bfloat16_t}.cpp`，Prefill 实例为
`conv1d_and_silu_prefill_paged_{float,float16_t,bfloat16_t}.cpp`。
所有 Tensor 必须位于 Runtime 的 NPU。binding 显式允许 state 的页间 pitch，
通过主分支的 XTensor.stride 传入；不复制、不转置状态。

## 实现原理

**任务分配**。64 通道为一个 channel tile。Decode 以 tile 为任务粒度，各 Vector core
按 `GetBlockIdx()` 起始、`GetBlockNum()` 跨步认领，tile 内串行处理整个 batch；
Prefill 的任务是 (请求, tile) 二维展平后同样跨步分配，每个任务独占其页面行直到该段
token 全部算完，避免终态写回与 initial 历史读竞争。

**权重列聚集（外提复用）**。权重 tile `[64][K]`（`w_f[c*K+j]`）每个任务装载并升 FP32
一次，经 `vgather`（`off_k` 偏移表）重排为 `[K][64]`（`win_f[j*64+c]`），tap 累加由此
变为每 tap 一次连续乘加；一次聚集在全 batch（decode）或整个 token 段（prefill）内复用。

**Decode 双缓冲流水**。两套 staging（`s_raw/s_f` 与 `s_raw2/s_f2`，FP32 直接 DMA 或
低精度先搬后转）以 EVENT_ID3/ID4 分别握手：`PrefetchWindow` 预取 b+1 的历史+输入
（MTE2）与 `ComputeWindow` 计算 b（V）并写回输出/历史（MTE3）重叠，b+2 的预取等待
MTE3→MTE2 释放标志后才复用缓冲。窗口 = `[row0..row(K-2), input]`，新状态 =
`[row1..row(K-2), input]`，updateState 时移位写回页面。

**Prefill 时间轴分块**。每 64 个 token（`kPrefillT`）为一个 block：输入 T 行 DMA 进
扩展窗口 `y_f = [history(K-1), T 行输入]`。输出只依赖输入（无移位依赖），按时间轴批量
计算——`win_f` 权重列跨 T 个 token 广播（repeat=T、源步长 0），K 个 tap 逐次乘加。
尾部 `K-1` 行经 `vadds` 平移为下一 block 的历史；全部 block 处理完后一次性写回页面历史。

**数值路径**。历史与输入统一升 FP32（`vconv_f162f32`/`vconv_bf162f32`，单次 vconv 上限
4096 lane，超出分块）。Decode 保留 fused 的 FP32 SiLU 后舍入；Prefill 先将卷积结果
round-trip 到 activation dtype 再做 SiLU，与 native conv1d 后接 SiLU 的顺序一致。

**同步**。`XLITE_V_FENCE`（V↔S 双向 set_flag/wait_flag）保证 vconv 的 UB 写回对后续
V 指令可见（`pipe_barrier` 不覆盖该写回的 errata）。MTE2→V 标志数据就绪，MTE3→MTE2/V
标志缓冲释放，V→MTE3 保证输出暂存写回就绪；prefill 的 y_f 由 V→MTE2（EVENT_ID5）
授权重用。

**边界**。零或负页号、超出 numPages 的页号输出零且不访问页面存储；prefill 空段
（start==end）不改页面。同一次 launch 的有效请求必须拥有互不重叠的写入页。页的分配、
清理和请求映射由调用者负责。

## 验证

`tests/kernels/gdn_native_pages.py` 覆盖 decode 的页重排、raw tables、padding、offset
和 guard 保护；`gdn_prefill_paged.py` 覆盖 fresh/continuation、空段、混合长度和
Prefill → Decode 的同页交接。旧 W>4、W=1 等配置不在新接口支持范围内。
