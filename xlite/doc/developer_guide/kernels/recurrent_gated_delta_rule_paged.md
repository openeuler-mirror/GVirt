# recurrent_gated_delta_rule_paged

## 功能概述

grouped-head gated delta rule(线性注意力 / GDN)直接在 vLLM 原生 `[P, Hv, V, K]` 分页状态页上递推,无状态转置、私有 cache、workspace 分配或 host 同步:

- `recurrent_gated_delta_rule_paged`:每请求单 token(decode);
- `recurrent_gated_delta_rule_prefill_paged`:packed 变长请求(prefill),对 tile 内 token 顺序递推——**不是** FLA 的 chunk WY/矩阵乘算法,移除状态转换不意味着长 prompt prefill 更快。

旧 `recurrent_gated_delta_rule` 及其 dense/packed kernel 已删除。每个 token 对 value head `h`(读取 key head `h // (Hv/Hk)`)执行:

```text
S = exp(g) * S                  # log-space decay
Delta = beta * (v - S @ k)
S = S + Delta[:, None] * k[None, :]
out = S @ (q / sqrt(K))
```

Q/K 由调用者预先 L2 归一化,kernel 对 Q 施加 `1/sqrt(K)`。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| query / key | 输入 | `[T, Hk*K]` | FP32/FP16/BF16 | 连续;T=decode 的 batch 或 prefill 的总 token 数 |
| value / out | 输入 / 输出 | `[T, Hv*V]` | 同 query | 连续;out 原位写出 |
| beta | 输入 | `[T, Hv]` | 同 query | 连续 |
| g | 输入 | `[T, Hv]` | 同 query 或 **FP32** | log-space decay;FP32 时置 `gateFp32` |
| state | 输入输出 | `[P, Hv, V, K]` | 同 query 或 **FP32** | 内层连续,dim-0 可有页间 padding,支持 storage offset;FP32 时置 `stateFp32` |
| indices | 输入 | `int32[B]`;decode 另支持 2D raw 表 | int32 | 每请求的 state 页号(prefill 只接受解析页号);decode 的 raw Mamba 块表须配 `cached_lens` + `state_block_size`,页选择与 [conv1d_and_silu_paged](conv1d_and_silu_paged.md) 同一约定(`csrc/gdn_metadata.h` 的 `GdnStatePage`):`cached_lens <= 0` 的行是 padding 取 0 页;`aligned=false` 取第 0 列,否则取第 `cached_lens / state_block_size` 列,列界外取 0 页 |
| cached_lens(decode raw 表) | 输入 | `[B]` | int32 | 各请求已缓存长度 |
| starts / initial(prefill) | 输入 | 连续 `int32[B+1]` / `int32[B]` | int32 | 段边界 / 初态标志(initial=0 忽略原页从零状态开始,非零从当前页继续) |

约束(host `csrc/op.cpp` 的 `XliteOpRecurrentGatedDeltaRulePaged` 与 `RecurrentPrefillImpl`):Hk/Hv 为正且 Hk 整除 Hv;K 是 16 的倍数且不超过 128;V 是 64 的倍数且不超过 128(即 64 或 128);**prefill 另要求 `Hv <= XLITE_GDN_GATE_CAP = 4096`**(`csrc/gdn_metadata.h` 的 `XLITE_GDN_GATE_CAP`,门控行 UB 缓冲 `decayRow`/`betaRow` 各容纳 4096 个 FP32 元素)。state 页描述符要求 `shape[1..3] == (Hv, V, K)` 且 dim-0 步长不小于 `Hv*V*K`。

## 支持的数据类型

decode 与 prefill 各有 FP32/FP16/BF16 activation 实例:

| 入口 | 实例文件 |
|---|---|
| decode | `csrc/kernels/recurrent_gated_delta_rule_paged_{float,float16_t,bfloat16_t}.cpp` |
| prefill | `csrc/kernels/recurrent_gated_delta_rule_prefill_paged_{float,float16_t,bfloat16_t}.cpp` |

state 与 g 可独立为 FP32(由 `stateFp32` / `gateFp32` launch 标志选择);host 校验 `g.dtype ∈ {query.dtype, FP32}`、`state.dtype ∈ {query.dtype, FP32}`。

## 实现原理

实现在 `csrc/kernels/recurrent_gated_delta_rule_paged.h` 的 `RecurrentGatedDeltaRulePaged`(模板类,decode 入口 `Process` → `ProcessHead`,prefill 入口 `ProcessPrefill`)。全程 FP32 计算:FP16/BF16 输入经 `LoadVector`/`LoadState`(MTE2 + `convert_input`)升精度进 UB,写出经 `convert_output` 降回原 dtype(`ioTmp`/`stateTmp` 中转)。

### 任务划分与 UB 状态驻留

- **decode**(`ProcessHead`):任务 = `batch × Hv`,AIV 按 `idx = GetBlockIdx(); idx < batch*vHeads; idx += GetBlockNum()` 轮转。每个任务顺序处理该 head 的 `vDim/64` 个 64 行 V tile。
- **prefill**(`ProcessPrefill`):任务 = `batch × Hv × (V/64)`,`task = GetBlockIdx(); task < batch*vHeads*tiles; task += GetBlockNum()` 三维展平(请求 / head / V tile)。state 的一个 64 行 tile 全程驻留 UB,token 递推完成后一次 `StoreState` 提交——Q/K 无需 repeat-interleave,SSM 无需 VK/KV 转置或全局临时状态。

每 64 行 tile 的 FP32 state 缓冲(`stateF`/`stateF2` 各 `64×128×4B`)按 K=128 上限分配;`K=128` 与 UB 容量校验由 `assert(off <= UB_SIZE)` 保证。

### DMA / 计算重叠

- **decode**:state tile 双缓冲乒乓(`EVENT_ID3/4`):计算第 `t` 个 tile 前 `LoadStateIssue` 预取第 `t+1` 块(`LoadStateFinish` 等待并按需转 FP32),当前块计算完立即 `StoreStateIssue` 异步写回 GM——写回入队后以同号 `set_flag(PIPE_MTE3, PIPE_MTE2)` 报告缓冲空闲,下一次 `LoadStateIssue` 以 `wait_flag(PIPE_MTE3, PIPE_MTE2)` 等 DMA 写完(循环前对两个 eid 预置该方向 `set_flag`,循环后两条 `wait_flag` 等末次写回;`FinishStore` 的 `EVENT_ID0` 排空只用于 `StoreVector`/`StoreState` 串行路径,不参与此乒乓)。
- **prefill**:
    - **门控批量预取**(`LoadGates`):每 `gateChunk = MIN(256, GATE_CAP/Hv)` 个 token 的 `exp(g)` 与 beta 一次成批搬入 UB 行 `decayRow`/`betaRow`(FP32,元素数 `n*Hv <= GATE_CAP`),避免逐 token 标量读 GM;
    - **q/k/v 乒乓预取**(`PrefetchIssue`/`PrefetchFinish`):FP16/BF16 经 `qStage/kStage/vStage` 中转,计算 token `i` 时预取 `i+1` 的三组输入,GM 加载与 V 计算重叠;每 token 的 64 行输出经 `StoreVector` 逐 token 写回;
    - state 单份驻留(`stateF`),仅段末一次提交。

### 核心向量步骤(单 token,`ProcessHead` / `ProcessPrefill` 内联)

1. `vmuls` state 衰减 `S *= exp(g)`;
2. `MatVec`(`vmul` + `vcadd` 行归约,K=128 时折半相加再归约)得 `S@k`;
3. `vsub`/`vmuls` 计算 `Delta = beta*(v - S@k)`;
4. `vbrcb` 把每个 Delta 标量广播为 32B 块,`vmul`+`vadd` 外积更新 `S += Delta ⊗ k`(vmul/vadd 分离规避 A2 vmadd 广播冒险);
5. `MatVec(state, q/sqrt(K))` 得 64 行输出。

### 页语义与边界处理

initial 为零时忽略原页内容,从零状态开始;非零时从当前页继续。空段不改变页面。零或负页号输出零且不更新 state(vLLM 的 0 页为 null block,不解引用);有效正页号必须在页范围内,各请求的有效写入页不能重叠。decode raw-table 行的 `cached_lens <= 0` 视为 padding,不访问 state。所有 tensors 位于 Runtime 的 NPU;Python binding 同步,模型层 C++ launch 使用 Runtime stream 不额外同步。

### 片上缓冲分配与占用汇总(AIV UB)

`RecurrentGatedDeltaRulePaged::Init`(`csrc/kernels/recurrent_gated_delta_rule_paged.h`)从 UB 偏移 0 顺序排布全部缓冲,decode 与 prefill 两个入口共用同一布局,`assert(off <= UB_SIZE)` 是容量断言。布局对维度定长:计算/中转缓冲按 host 上限 K=128、64 行 V tile(V≤128)分配,门控行按 `GATE_CAP = XLITE_GDN_GATE_CAP = 4096`(`csrc/gdn_metadata.h` 的 `XLITE_GDN_GATE_CAP`)个 FP32 元素分配——占用与头数/K/V 的具体取值无关,任意合法配置(如 Hk=16、Hv=32、K=128、V=128 的 GDN 模型)占用相同。硬件容量基准(910B/dav C220):UB 192KB(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`)。dtype 记法:模板 `Dtype` 为激活类型,FP32 路径 `<float>`、FP16/BF16 路径 `<half|bfloat16_t>`(2B,另增 fp16/bf16 转换暂存区);两路径同取上述定长布局,占用差异仅来自 dtype——FP32 路径下五个转换暂存区(`stateTmp`/`stateTmp2`/`qStage`/`kStage`/`vStage`)分配量为 0(`Init` 的 `std::is_same<Dtype, float>` 条件分支,指针与后继缓冲别名但从不使用),下表以 0 计入合计。

**AIV UB(容量 192KB,每 block 一份)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | FP32 占用(B) | FP16/BF16 占用(B) |
|---|---|---|---|---|---|---|
| `ioTmp` | 定长 512B 中转区(≤128 元素) | Dtype | 1(串行,事件 EVENT_ID0) | `512` B | 512 | 512 |
| `stateTmp`/`stateTmp2` | (64, 128) | Dtype | 2(事件 EVENT_ID3/4,decode 状态乒乓;prefill 仅用 `stateTmp` 兼作门控 gather 暂存) | `64*128*sizeof(Dtype)` B | 0 | 32768 |
| `qF`/`qF2` | kDim(≤128) | float | 2(事件 EVENT_ID0,prefill token 乒乓;decode 仅用 `qF`) | `512` B | 1024 | 1024 |
| `kF`/`kF2` | kDim(≤128) | float | 2(同上) | `512` B | 1024 | 1024 |
| `vF`/`vF2` | vDim(≤128) | float | 2(同上) | `512` B | 1024 | 1024 |
| `stateF`/`stateF2` | (64, 128) | float | 2(事件 EVENT_ID3/4,decode 状态乒乓;prefill 单份驻留 `stateF`) | `64*128*4` B | 65536 | 65536 |
| `product` | (64, 128) | float | 1 | `64*128*4` B | 32768 | 32768 |
| `delta` | 64(每 V 行一个标量) | float | 1 | `256` B | 256 | 256 |
| `outF` | vDim(≤128) | float | 1 | `512` B | 512 | 512 |
| `reduceTail` | 64 | float | 1 | `256` B | 256 | 256 |
| `deltaBlocks` | (64, 8)(每标量广播为 32B 块) | float | 1 | `64*32` B | 2048 | 2048 |
| `qStage` | kDim(≤128) | Dtype | 1(事件 EVENT_ID0,prefill 预取中转) | `128*sizeof(Dtype)` B | 0 | 256 |
| `kStage` | kDim(≤128) | Dtype | 1(同上) | `128*sizeof(Dtype)` B | 0 | 256 |
| `vStage` | 64 | Dtype | 1(同上) | `64*sizeof(Dtype)` B | 0 | 128 |
| `decayRow` | GATE_CAP | float | 1(prefill 批量门控行;decode 仅用首元素) | `GATE_CAP*4` B | 16384 | 16384 |
| `betaRow` | GATE_CAP | float | 1(同上) | `GATE_CAP*4` B | 16384 | 16384 |
| **合计** | | | | | **137728** | **171136** |

**占用汇总**(字节数):

| 存储级 | 硬件容量 | FP32 占用 | 占比 | FP16/BF16 占用 | 占比 |
|---|---|---|---|---|---|
| UB(每 block) | 192 KB | 137728 B | 70.1% | 171136 B | 87.0% |

要点:

- **UB 是唯一使用的存储级**(纯向量递推,无 cube),FP16/BF16 路径 87.0% 最紧:双份 64×128 FP32 state(`stateF`/`stateF2`,65536 B)+ `product`(32768 B)+ 双份 fp16 状态转换暂存(`stateTmp`/`stateTmp2`,32768 B)合计约 77%,是占用大头——这是 64 行 V tile + K=128 上限定长分配的直接结果,`assert(off <= UB_SIZE)` 即容量来源(源码注释"64 V rows keep even K=128 well below the A2 192 KiB UB limit"的落点);
- `decayRow`/`betaRow` 各 16384 B(GATE_CAP=4096 个 FP32 元素)是第二大头,支撑 prefill 的批量门控预取;host 侧 prefill 约束 `Hv <= XLITE_GDN_GATE_CAP`(见上文约束段)正来自这对行缓冲;
- FP32 路径省去全部转换暂存(比 FP16/BF16 少 33408 B,占 70.1%);FP16/BF16 增加 `stateTmp`×2 与 `qStage`/`kStage`/`vStage` 三段;
- decode 的状态乒乓(EVENT_ID3/4)与 prefill 的 token 乒乓(EVENT_ID0)共用这一静态布局,占用不随执行阶段与请求规模变化。

## KDA safe-gate 变体

KDA safe-gate 用同一骨架的变体入口（当前 GLM-5.3-Flash 使用）,复用页寻址、MatVec 归约与 64 行 V tile(`csrc/kernels/recurrent_gated_delta_rule_kda_paged.h` 的 `RecurrentGatedDeltaRuleKdaPaged`),差异对应 native `run_recurrent_kda` 契约:

- **beta 原始输入**:kernel 内 fp32 `sigmoid(b)`(native `beta_is_preprocessed=False`),不再要求调用者预处理;
- **safe_gate 按 K 行衰减**:gate 输入为原始 `[T, Hk*K]`(逐 K head 逐维),kernel 内 fp32 计算 `gate = lowerBound * sigmoid(exp(A_log[h]) * (g1 + dt_bias))`,按 K 行施加 log-decay(state 每 K 行一个 `exp(gate[k])` 广播 `vmul`),而非基础变体的每 head 标量 `g`;
- **A_log / dt_bias 参数**:`a_log` 为 FP32 `[Hk]`、`dt_bias` 为 FP32 `[Hk*K]`(host 校验,`csrc/op.cpp` 的 `XliteOpRecurrentGatedDeltaRuleKdaPaged`);
- **L2 归一化核内完成**:q/k 在 kernel 内做 `rstd = rsqrt(sum(x^2) + norm_eps)` 归一化(基础变体要求调用者预归一化),q 额外乘 `scale` 标量。

入口与基础变体一一对应:`recurrent_gated_delta_rule_kda_paged`(decode,raw 块表同样支持 `cached_lens`/`state_block_size`/`aligned`)与 `recurrent_gated_delta_rule_kda_prefill_paged`(packed prefill,`starts`/`initial` 契约相同)。维度约束与基础变体一致(K 16 的倍数 ≤128,V 64 的倍数 ≤128,Hk 整除 Hv);dtype 实例同为 FP32/FP16/BF16 三套(`csrc/kernels/recurrent_gated_delta_rule_kda_paged_{float,float16_t,bfloat16_t}.cpp`、`recurrent_gated_delta_rule_kda_prefill_paged_{float,float16_t,bfloat16_t}.cpp`)。测试见 `tests/kernels/recurrent_gated_delta_rule_kda_paged.py`。

## 验证

`tests/kernels/recurrent_gated_delta_rule_paged.py` 验证 dtype、分组头、维度、raw-table 页选择和非法 descriptor;`tests/kernels/gdn_prefill_paged.py` 对独立 FP32 recurrence 校验 fresh/续算/ragged/空段、混合单/多 token、null 页、分组头、pitch 与跨阶段交接;`tests/models/gdn_paged.py` 验证小型混合模型的 V1/V2 metadata 和分段执行一致性。
