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

### 片上缓冲分配与占用汇总

`XliteCausalConv1dSiLUDecode::Init`(`csrc/kernels/conv1d_and_silu_paged.h` 的 `XliteCausalConv1dSiLUDecode::Init`)为 decode 与 prefill 两个入口一次性静态排布全部 UB 槽位:偏移按段间 32B 对齐累进(本布局各段尺寸天然 32B 对齐、无垫片),段大小全部由编译期常量决定(`kMaxWin=256`、`kRawLen=320`、`kBlock=64`、`kPrefillWin=4288`、`kPrefillTile=4096`),与 channels/kernelDim 的运行期取值无关。硬件容量基准(910B/dav C220):UB 192KB(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE=196608`);本算子为纯 AIV 向量算子,不使用 L1/L0A/L0B/L0C。dtype 记法:模板实参 `Dtype` 为 state/input/weight/output 元素类型,fp16/bf16 实例均 2B 同列、float 实例 4B 单列。典型配置同取 GLM-5.2 短卷积(channels=8192、W=4,即 `tests/models/glm5_next.py` 的 `la_num_heads×la_head_dim=64×128`;Qwen3.5-MoE GDN 同为 2×16×128+32×128=8192),占用差异仅来自 sizeof(Dtype)。

**UB(静态布局,decode/prefill 共用,预算 `UB_SIZE=192KB`)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | fp16/bf16 占用(B) | float 占用(B) |
|---|---|---|---|---|---|---|
| `w_raw`(权重 tile 原始 dtype) | `(64,K)`,`kMaxWin=256` | Dtype | 1(事件2,每 tile 重装) | `256*sizeof(Dtype)` | 512 | 1024 |
| `s_raw`/`s_raw2`(decode 窗口暂存) | `(320,)`,`kRawLen` | Dtype | 2(事件3/4) | `320*sizeof(Dtype)` | 1280 | 2560 |
| `w_f`(权重 tile 升 fp32) | `(64,K)` | float | 1(事件2) | `kMaxWin*4` | 1024 | 1024 |
| `s_f`/`s_f2`(窗口升 fp32) | `(320,)` | float | 2(事件3/4) | `kRawLen*4` | 2560 | 2560 |
| `win_f`(权重列聚集 `[K][64]`) | `(256,)` | float | 1 | `kMaxWin*4` | 1024 | 1024 |
| `acc_f`(遗留槽位,未被引用) | `(256,)` | float | 1 | `kMaxWin*4` | 1024 | 1024 |
| `r_f`(遗留槽位,未被引用) | `(64,)` | float | 1 | `kBlock*4` | 256 | 256 |
| `calc_buf`(decode SiLU 中间量) | `(64,)` | float | 1 | `kBlock*4` | 256 | 256 |
| `out_f`(decode 卷积累加+SiLU 结果) | `(64,)` | float | 1 | `kBlock*4` | 256 | 256 |
| `out_buf`(decode 输出) | `(64,)` | Dtype | 1(事件0) | `kChanTile*sizeof(Dtype)` | 128 | 256 |
| `st_stage`(状态回写暂存) | `(256,)` | Dtype | 1(事件0) | `kMaxWin*sizeof(Dtype)` | 512 | 1024 |
| `off_win`(vgather 偏移表,遗留) | `(256,)` | uint32 | 1(BuildRamps 预留,未被调用) | `kMaxWin*4` | 1024 | 1024 |
| `off_k`(权重列聚集偏移表) | `(64,)` | uint32 | 1(事件1) | `kBlock*4` | 256 | 256 |
| `y_f`(prefill 扩展窗口) | `(4288,)`,`kPrefillWin` | float | 1(事件2/5) | `kPrefillWin*4` | 17152 | 17152 |
| `inp_raw`(prefill 输入/round-trip 暂存) | `(64,64)`,`kPrefillTile` | Dtype | 1(事件2) | `kPrefillTile*sizeof(Dtype)` | 8192 | 16384 |
| `pre_acc`(prefill 卷积累加) | `(64,64)` | float | 1(事件0) | `kPrefillTile*4` | 16384 | 16384 |
| `pre_calc`(prefill SiLU 中间量) | `(64,64)` | float | 1 | `kPrefillTile*4` | 16384 | 16384 |
| **合计** | | | | | **68224** | **78848** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | fp16/bf16 占用 | 占比 | float 占用 | 占比 |
|---|---|---|---|---|---|
| UB | 192 KB | 68224 B | 34.7% | 78848 B | 40.1% |

要点:

- **布局完全静态、与规模无关**:channels 只决定 tile 数(8192/64=128 个),kernelDim ≤ kMaxK=4 时槽位一律按上限预留——最小配置(C=64、W=2)与 8192 通道占用完全相同;`channels%64==0`、`K∈[2,4]` 由 kernel 入口守卫(`csrc/kernels/conv1d_and_silu_paged.h` 的 `XliteCausalConv1dSiLUDecode::ProcessPaged` 起始校验段)保证,容量本身无 host 校验、无 kernel 断言,布局常量即容量上限。
- **prefill 时间轴分块缓冲是大头**:`y_f`+`inp_raw`+`pre_acc`+`pre_calc` 在 fp16/bf16 路径合计 58112 B(占布局 85%);decode 专属的窗口双缓冲(`s_raw`/`s_raw2`+`s_f`/`s_f2`,prefill 借用 s_raw 暂存 initial 历史)仅 3840 B。
- **遗留槽位仍占 UB**:`off_win`(仅未被调用的 `BuildRamps` 写入)、`acc_f`/`r_f`(全文无引用)合计 2304 B,当前 decode/prefill 路径均不使用,回收可省同额 UB。

## 验证

`tests/kernels/gdn_native_pages.py` 覆盖 decode 的页重排、raw tables、padding、offset
和 guard 保护；`gdn_prefill_paged.py` 覆盖 fresh/continuation、空段、混合长度和
Prefill → Decode 的同页交接。旧 W>4、W=1 等配置不在新接口支持范围内。
