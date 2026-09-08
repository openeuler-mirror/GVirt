# norm

## 功能概述

归一化算子族,一个 kernel 通过 `NormKind` 枚举(`csrc/kernels/kernel_param.h:26-30`)统一实现三种归一化:

- **Rms(RMSNorm)**:`y = x / sqrt(mean(x^2) + eps) * weight (+ bias)`,对每个 norm 段(每 head 或每 token)按均方根归一化;
- **Layer(LayerNorm)**:`y = (x - mean) / sqrt(variance + eps) * weight + bias`,先减均值再除以标准差;
- **L2(L2Norm)**:`y = x / sqrt(sum(x^2) + eps)`,不除以 n,直接按 L2 范数归一化,无仿射。

同一 kernel 还支持多种模式:纯归一化、带 weight/bias 仿射、variance-only(只算方差,供 TP 先 AllReduce 再归一化的两阶段 rmsnorm_full 流程)、AddAndRmsNorm 融合(残差加 + 归一化)、以及可选的 fp32 输出。

## 输入输出参数

Python 侧入口:`rmsnorm / rmsnorm_with_bias / rmsnorm_variance_only / layernorm / l2norm`(`csrc/_C.cpp:2661-2675`)。以 kernel 签名(`csrc/kernels/norm.h:223-229`)为准:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input | 输入 | `[token_num, in_step]` | float16 / bfloat16 | 输入矩阵,行主序;实际参与计算的是从 `in_start_offset` 起、每行 `norm_dim * cnt_per_token` 个元素 |
| addInOut | 输入/输出 | 同 input | 同 input | 残差缓冲,仅 AddAndRmsNorm 使用;先读入与 x 相加并写回,再对相加结果归一化 |
| weight | 输入 | `[norm_dim]` | 同 input | 仿射缩放(γ),Rms/Layer 可选,L2 恒为空 |
| bias | 输入 | `[norm_dim]` | 同 input | 仿射偏置(β),可选 |
| output | 输出 | `[token_num, out_step]` | 同输入 或 float32 | 归一化结果;`outFp32=true` 时为 fp32 |
| variance | 输入/输出 | `[token_num, 1]` | float32 | 双向:为空时内部算方差;非空时读入已 AllReduce 的方差(按 `1/tpSize` 缩放后使用);useNorm=false 时作为输出写出每 token 方差 |
| token_num | 标量 | - | uint32_t | token(行)数,host 传 `in.shape[0]` |
| norm_dim | 标量 | - | uint32_t | 单次归一化的维度(每 head 维度或整行维度) |
| norm_eps | 标量 | - | float | ε,加在方差上防除零 |
| kind | 标量 | - | int | NormKind 枚举值:0=Rms, 1=Layer, 2=L2 |
| cnt_per_token | 标量 | - | uint32_t | 每行内独立归一化的段数(如 MHA qkNorm 时每 head 一段) |
| in_step / out_step | 标量 | - | uint32_t | 输入/输出行步长(元素数),host 传 `in.shape[1]` / `out.shape[1]` |
| in_start_offset / out_start_offset | 标量 | - | uint32_t | 行内起始列偏移,支持在同一行内对不同列段做归一化 |
| useNorm | 标量 | - | bool | true=完整归一化;false=variance-only 模式 |
| tpSize | 标量 | - | uint32_t | 张量并行规模,用于把 AllReduce 后的方差和除回均值 |
| outFp32 | 标量 | - | bool | 输出是否为 fp32 |

测试参考:[tests/kernels/rmsnorm.py](../../tests/kernels/rmsnorm.py)(`[64, 8192]`、带 stride 的 `DIM=128, CNT=8`)、[tests/kernels/layernorm.py](../../tests/kernels/layernorm.py)、[tests/kernels/l2norm.py](../../tests/kernels/l2norm.py)、[tests/kernels/rmsnorm_full.py](../../tests/kernels/rmsnorm_full.py)(variance-only + all_reduce + apply 三步)。

## 支持的数据类型

- `float16_t`([norm_float16_t.cpp](../../csrc/kernels/norm_float16_t.cpp))
- `bfloat16_t`([norm_bfloat16_t.cpp](../../csrc/kernels/bfloat16_t.cpp))

host 侧要求输入为 FP16/BF16,输出为同 dtype 或 FP32(`csrc/op.cpp:526-529`);计算全程在 fp32 进行。

## 实现原理

实现位于 [csrc/kernels/norm.h](../../csrc/kernels/norm.h),模板函数 `norm<Dtype>`(norm.h:222)。另有一个大维度分块变体 `rmsnorm_noaffine_tiled`(norm.h:43-220)。

### 总体数据流

GM → (MTE2) → UB 输入缓冲 → (V pipe) `convert_input` 转 fp32 → 向量计算 → `convert_output` 转回 Dtype → (MTE3) → GM。UB 内所有中间计算都在 fp32 上进行,保证精度。

### 分块与多 Block 并行

- **主路径**:一个 token(一行中 `norm_dim * cnt_per_token` 的整段)一次性搬入 UB,不按 norm_dim 切块。因此 UB 需容纳 `total_dim = norm_dim * cnt_per_token` 个元素的双缓冲,大 hidden_size(如 8192)时 fp32 计算缓冲约占大头。
- **多 Block 并行**:按 token 循环跨 step 分配,`for (loop = first; loop < token_num; loop += block_num)`(norm.h:365),`block_idx` 不同的 AIV 处理不同 token。`coreOffset` 机制(norm.h:364,`first = (block_idx + block_num - coreOffset) % block_num`)让多次 norm 调用(如 qk_rms_norm 的 Q/K 两段、mla_prepare 的三段)在同一批核上接力,行号从上次结束位置继续轮转,避免每次都从 block 0 起跳造成负载不均;kernel 结束时通过 `nextCoreOffset = (coreOffset + token_num) % block_num` 把终点交回调用者(norm.h:359-361)。
- **大维度分块路径** `rmsnorm_noaffine_tiled`:当 `useNorm && !weight && norm_dim > 6144 && cnt_per_token == 1 && kind == Rms`(norm.h:237)时启用,专用于超大 hidden_size 的无仿射 RMSNorm(如 MHA qkNormFull 的 variance 阶段)。按 `tileDim = 8192` 分两遍扫描:第一遍累加 `sum(x^2)` 得标量,第二遍重读 GM 做 `y = x * rsqrt(sum/n + eps)`(行约 32KB,重读代价低)。该路径 UB 只需约 96KB,避开单次整行搬入的 UB 压力。

### UB 内存布局(主路径,norm.h:262-306)

| 缓冲 | 大小 | 用途 |
|---|---|---|
| in[0] / in[1] | `ROUND_UP(total_dim*sizeof(Dtype), 32)` ×2 | 输入 ping-pong |
| out[0] / out[1](或 out_float[0..1]) | 同上 ×2 | 输出 ping-pong;outFp32 时为 fp32 尺寸 |
| in_variance_float[0..1] | fp32 尺寸 ×2 | 仅有 variance 输入时分配 |
| calc0 / calc1 | `ROUND_UP(total_dim*4, 32)` ×2 | fp32 计算缓冲(calc0=x,calc1=平方和/均值) |
| weight_calc / bias_calc | fp32 尺寸 | 仿射参数,加载一次后按 cnt_per_token 复制扩展 |

weight/bias 先加载到 UB 并转 fp32(norm.h:313-327),再对 `cnt_per_token > 1` 的情况用 `copy_ubuf_to_ubuf` 在段间复制(norm.h:333-348),避免每行重复从 GM 搬运。

### 流水线同步

MTE2(搬入)/ V(计算)/ MTE3(搬出)三管线,事件标志协议:

- `V→MTE2`(EVENT_ID0/1):输入缓冲空闲,可发起下一次 GM→UB 搬运;
- `MTE2→V`:数据就绪,可以转换/计算;
- `V→MTE3`(EVENT_ID0/1):输出缓冲就绪;`MTE3→V`:输出缓冲已被搬出、可复用。

输入、输出各自 ping-pong(`inCurr`/`outCurr` 翻转,norm.h:381-390、539),搬入下一行与搬出上一行同当前行计算重叠。kernel 末尾 `wait_flag` 收干净所有事件并 `pipe_barrier(PIPE_ALL)`(norm.h:541-545)。标量回读(如 sum 后取 `*calc`)通过 `set_flag/wait_flag(PIPE_V, PIPE_S, EVENT_ID0)` 同步 S 管线(norm.h:429-430)。

### 关键计算步骤

1. **(可选)残差加**:`addInOut` 非空时搬入第二路数据,`vadd(calc0, calc1, calc0)` 后先 `convert_output` 写回 addInOut,再继续归一化(norm.h:392-417)。
2. **LayerNorm 减均值**:`vmuls` 乘 `1/n` → `reduce_sum` 逐段求和得 mean → `duplicate_item` 把段内标量广播 → `vsub` 减去均值(norm.h:420-439)。
3. **方差**:`vmul` 平方;`kind != L2` 时再 `vmuls` 乘 `1/n`(RMSNorm 除以维度,L2Norm 不除);`reduce_sum` 逐段归约到段首元素(norm.h:451-465)。若传入 variance,则直接读入 GM 方差并 `vmuls` 乘 `1/tpSize`(norm.h:441-450)——对应 rmsnorm_full:本 rank 方差 AllReduce 后求和,除以 tpSize 还原全局均值。
4. **归一化**:`vadds` 加 eps → `vsqrt` → `duplicate_item` 广播 → `vdiv` 除(norm.h:468-487);`SetMask(1)` + 特殊 stride 控制只对每段首元素做 sqrt 类运算(norm.h:468-475)。
5. **仿射**:`vmul(weight)`、`vadd(bias)`(norm.h:490-499)。
6. **输出**:useNorm 时 `convert_output` 转 Dtype 写 GM(outFp32 时 fp32 直写);variance-only(useNorm=false)时只把每段方差(fp32)写出(norm.h:509-537)。
7. **(可选)写 KV cache**:kcache/slot_mapping/block_size 非空时,归一化结果按 `slot_idx` 定位额外写一份到 paged cache(norm.h:522-526),该路径目前由 mla_prepare 等调用方使用,`norm_*` 导出符号固定传 nullptr(norm.h:557)。

`reduce_sum`(norm.h:11-26)对 `norm_dim == 128` 走快速路径:先一条 `vadd` 折半,再逐段 `vcadd`(`Order_t::ONLY_VALUE` 语义,结果落在段首);其他维度调用通用 `ReduceSum`(kernel_macro.h:701,二分折叠 + `vcadd` 收尾)。`duplicate_item`(norm.h:28-38)经 S 管线读段首标量后 `vector_dup` 广播整段。

### 边界处理

- 尾部 token 数不是 block_num 整数倍时,循环条件自然让部分 AIV 空转;`token_num < block_num` 时通过 coreOffset 相关的进入条件(norm.h:311-312)保证只有持有有效行的核参与。
- `rmsnorm_noaffine_tiled` 中尾 tile(`dCur < tileDim`)按实际长度搬算(norm.h:130-133);`outFp32` 时输出指针按 fp32 行距重解释,避免 dtype 指针运算按 sizeof(Dtype) 缩放导致错位(norm.h:120-123)。
- L2Norm 无 weight/bias,断言保证不会误传(norm.h:274);addInOut 与 outFp32 互斥(norm.h:278)。

### Host 侧调用

`csrc/op.cpp:518-623`:`XliteOpRmsNorm` / `XliteOpLayerNorm` / `XliteOpL2Norm` / `XliteOpAddAndRmsNorm` 分别以不同 `kind`、`useNorm`、仿射参数组合 launch 同一个 `norm_<dtype>` kernel,grid 为 `rt.aivNum`,`outFp32 = (out.dtype == FP32)`。
