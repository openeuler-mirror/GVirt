# rope_and_cache

## 功能概述

对融合 QKV 张量中的 Q 和 K 做 NeoX 风格的旋转位置编码(RoPE),对 Q 追加 `1/sqrt(headDim)` 缩放,并将旋转后的 K 以及未旋转的 V 按 `slot_mapping` 写入 paged KV cache;同时把旋转后的 K、缩放旋转后的 Q 原地写回 QKV 张量。支持多模态 MroPE(文本/高/宽三路位置)与标准单路 RoPE 两种模式。该算子由小艺团队贡献,参考论文《XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads》[ASPLOS 2026]([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 文件头注释,关键字 `XY-Serve`)。

数学语义(NeoX 半旋转,`half = rotDim / 2`):

```
q_out[:half]   = (q[:half] * cos - q[half:] * sin) * scale
q_out[half:]   = (q[half:] * cos + q[:half] * sin) * scale
k_out[:half]   = k[:half] * cos - k[half:] * sin
k_out[half:]   = k[half:] * cos + k[:half] * sin
```

MroPE 模式下 `cos/sin` 为文本、h、w 三路表按 lane 掩码逐元素相加的结果。

默认情况下 cos/sin 表为 FP32:旋转全程在 fp32 域完成,Q 缩放后再做**一次**最终舍入回 inout dtype,以减少中间舍入损失、提升精度。该路径由 host 侧根据 `cossin.dtype == FP32` 自动判定并经 `cossinInFp32` 标志传入 kernel(详见下文及 [实现原理](#实现原理))。

## 输入输出参数

Python 接口:`rope_and_cache(rt, inout, k_cache, v_cache, position, cosin, slot_mapping, n_heads, n_kv_heads, head_dim, rot_dim, block_size, is_neox, mrope_mask_h=0, mrope_mask_w=0)`。`cossinInFp32` 不是 Python 参数,由 host 从 `cossin.dtype` 推导;K/V cache 的 dim-0 跨度 `key_cache_stride` / `value_cache_stride` 以及 2D position 的 dim-0 跨度 `positions_stride` 亦非 Python 参数,由 host 从 `k_cache.stride[0]` / `v_cache.stride[0]` / `position.stride[0]` 读取传入。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| inout | 输入/输出 | `[num_tokens, (num_heads + 2 * num_kv_heads) * head_dim]` | float16 / bfloat16 | 融合 QKV。host 侧按 TP 切分后的头数拆出 q/k/v 三个指针([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeCache` 中 q/k/v 指针拆分段),q 在行首,k、v 依次随其后;Q/K 旋转后原地写回 |
| k_cache | 输出 | `[block_num, block_size, num_kv_heads, head_dim]` | 同 inout | KV cache 的 K 平面,按 slot 展开寻址 |
| v_cache | 输出 | 同 k_cache | 同 inout | KV cache 的 V 平面,V 不做旋转,直接搬运写入 |
| position | 输入 | `[num_tokens]`(MroPE 为 `[3, num_tokens]`) | int64 | 每 token 的位置 id;MroPE 时第 2/3 行为 h/w 位置。2D 时各行按 dim-0 跨度寻址,支持非连续(列切片产生的 pitch)且列数可超过 num_tokens(预分配缓冲);文本 1D 路按平坦连续读取,不消费跨度([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 MroPE 三路位置搬入段,关键字 `gm_pos + positions_stride`) |
| positions_stride | 标量(host 推导) | - | uint32 | 2D position 的 dim-0 跨度(元素数),host 传 `position.stride[0]`;仅 MroPE 的 h/w 行寻址消费,1D 平坦读取路径不使用 |
| cossin | 输入 | `[max_pos, rot_dim]` | **float32(推荐)** 或同 inout | 预计算表,前 `rot_dim/2` 列为 cos、后 `rot_dim/2` 列为 sin(见测试 `precompute_freqs_cis` 的 `cat((cos, sin), dim=-1)`)。FP32 时走精度优先的 fp32 旋转路径;与 inout 同 dtype 时走已废弃的 model-dtype 路径(bf16 舍入回退仿真 / fp16 直接计算) |
| slot_mapping | 输入 | `[num_tokens]` | int32 | 每 token 在 cache 中的平坦 slot 索引 |
| key_cache_stride / value_cache_stride | 标量(host 推导) | - | uint32 | K/V cache 的 dim-0 跨度(元素数),host 传 `kCache.stride[0]` / `vCache.stride[0]`;连续 cache 时等于 block_size × num_kv_heads × head_dim 的乘积,非连续时为宿主传入的真实跨距 |
| n_heads / n_kv_heads | 标量 | - | uint32 | **全局** Q 头数与 KV 头数(host 内部除以 tpSize 得本 rank 头数,见 [csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeCache`,关键字 `localHeads = nHeads / rt.tpSize()`) |
| head_dim | 标量 | - | uint32 | 每头维度,支持 64 / 128 |
| rot_dim | 标量 | - | uint32 | 旋转维度,支持 64 / 128(可小于 head_dim,如 head_dim=128 + rot_dim=64) |
| block_size | 标量 | - | uint32 | cache 块大小;kernel 将 slot_mapping 的平坦 slot 按 block_size 分解为块号与块内偏移,结合 cache 跨度寻址 |
| is_neox | 标量 | - | bool | 仅支持 NeoX 风格,gptj 会抛错([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeCache`,关键字 `unsupported rope type gptj`) |
| mrope_mask_h / mrope_mask_w | 标量 | - | uint64 | MroPE 的向量 lane 掩码,非 0 时启用三路位置模式 |
| cossinInFp32 | 标量(host 推导) | - | bool / uint32 | cos/sin 表是否按 FP32 读取以提升精度。host 侧 `cossinInFp32 = cossin.dtype == FP32`([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeCache`,关键字 `cossinInFp32 = cossin.dtype == FP32`),kernel 据此在 `rope_and_cache<Dtype, COSSIN_FP32>` 的两条模板实例间分派([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `ROPEANDCACHE_FUNC_DEFINE` 宏)。true 时旋转全程 fp32、仅末尾一次舍入;false 时走已废弃的 model-dtype 路径 |

`scale = 1.0 / sqrt(headDim)` 由 host 计算传入([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeCache`,关键字 `scale = 1.0f / sqrtf`);q/k/v 的行 stride 均为 `inout.shape[1]`(整行 QKV 宽度)。

## 支持的数据类型

inout、kCache、vCache 三者必须同为 `float16_t` 或 `bfloat16_t`;cossin 可为 **FP32(推荐)** 或与 inout 同 dtype。host 分派逻辑见 [csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeCache` dtype 分派段:

- **FP32 cossin(默认/推荐路径,`cossinInFp32=true`)**:fp16 与 bf16 均适用。cos/sin 直接以 fp32 搬入 UB,旋转在 fp32 域完成,Q 缩放后由 `convert_output<Dtype>` 做唯一一次 fp32→Dtype 舍入。精度最优,见 `calc_cossin_cast<Dtype, true>`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin_cast`)。
- **model-dtype cossin(已废弃,`cossinInFp32=false`)**:cossin 与 inout 同 dtype,保留至旧路径移除。
  - `bfloat16_t`([rope_and_cache_bfloat16_t.cpp](../../../csrc/kernels/rope_and_cache_bfloat16_t.cpp)):转 fp32 计算,两个乘积 `x*cos`、`x*sin` 先 `vconv_f322bf16r` 舍入回 bf16 再升回 fp32,然后才做 `vsub`/`vadd`,以仿真 torch_npu 的逐算子舍入行为(`calc_cossin_cast<Dtype, false>`,[csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin_cast` 中 bf16 往返舍入段,注释"为了保证与torch_npu计算结果一致")。
  - `float16_t`([rope_and_cache_float16_t.cpp](../../../csrc/kernels/rope_and_cache_float16_t.cpp)):fp16 域内直接计算(`calc_cossin`,[csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin`)。

## 实现原理

实现位于 [csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h),为 C220 向量核(`__DAV_C220_VEC__`)上的标量 C 风格 kernel,由 `XliteOpRopeCache`([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeCache`)以 `rt.aivNum` 个 AIV block 启动。kernel 主体 `rope_and_cache<Dtype, COSSIN_FP32>`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache`)以 `COSSIN_FP32` 模板参数在两条路径间静态分派,封装宏 `ROPEANDCACHE_FUNC_DEFINE` 根据 host 传入的 `cossinInFp32` 运行时选择实例([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `ROPEANDCACHE_FUNC_DEFINE` 宏)。

### 两级循环与多 Block 并行

- 外层按批搬运 `positions`/`slot_mapping`(Mrope 还包括 pos_h/pos_w)到 UB 尾部暂存区,批量大小 `iter_posslot_num` 由剩余 UB 空间动态推出([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 `iter_posslot_num` 推导段);Mrope 的 h/w 行按 `gm_pos + positions_stride` / `gm_pos + 2 * positions_stride` 定位(文本行与 1D 路为 `gm_pos` 平坦读取),`positions_stride` 为 2D position 的 dim-0 跨度,非连续与列数超出 num_tokens 的缓冲均正确;
- 内层将 token 条带化分配到各 block:`loop1 = block_idx + processed_num_tokens`,步长 `block_num`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 `loop1 = block_idx + processed_num_tokens` 内层循环),即 token 级多核并行。

每个 token 的 GM 地址由标量单元从 UB 暂存区读出 position/slot 计算得到:cos/sin 表偏移 `pos * rot_dim`(sin 再加 `rot_dim/2`);cache 侧把平坦 slot 按 `block_size` 分解为块号与块内偏移,K/V cache 各自按 `slot / block_size * key_cache_stride(value_cache_stride) + slot % block_size * num_kv_heads * head_dim` 寻址,连续 cache 时等价于 `slot * num_kv_heads * head_dim`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 slot/kcache 寻址段)。

### UB 内存布局

以 `PINGPONG_BUF_NUM = 2` 组乒乓缓冲组织,从地址 0 起依次排布([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 UB 布局分配段):

| 区域 | 份数 | 大小 | 用途 |
|---|---|---|---|
| query | 2 | `num_heads * head_dim` | Q 暂存 |
| key / value | 各 2 | `num_kv_heads * head_dim` | K/V 暂存 |
| cos / sin(Dtype 暂存) | 各 2 | `rot_dim`(对齐 32B) | model-dtype 路径的 cos/sin 暂存;FP32 路径下空闲 |
| cos/sin h、w(MroPE,Dtype) | 各 2 | 同上 | 三路位置表的 Dtype 暂存;FP32 路径下空闲 |
| fp32 工作区 | query/key/cos/sin 各 2 + mrope 各 2 + 计算区 | fp32 字节数 | `COSSIN_FP32 || is_bf16` 时启用:FP32 路径下 cos/sin 行为 GM→UB 直接落点;bf16 旧路径下为 `vconv_bf162f32` 的目标 |
| fp16 计算区 | 2 | `q_bytesize` | 仅 legacy fp16 直接计算路径使用(`calc_cossin` 的 `x*sin` 中间结果) |
| 尾部 params 区 | 1 | 剩余空间 | positions(pos/pos_h/pos_w)与 slot_mapping 分批暂存 |

搬运统一用预配置的 DMI 参数(`__set_dmi_config`,[csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 `__set_dmi_config` burst 描述段),q/kv/cossin/pos/slot 各一份 burst 描述;其中 `lenBurst_cossin` 随 `COSSIN_FP32` 在 fp32 与 Dtype 字节数间切换([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 `lenBurst_cossin` 段)。

### 数据流与流水线同步

每 token 按顺序执行,全程通过事件标志驱动 MTE2(搬入)→ V(计算)→ MTE3(搬出)三级流水,乒乓两组事件(`EVENT_ID0`/`EVENT_ID1`,[csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 `set_flag(PIPE_MTE3, PIPE_MTE2` 预置段):

1. **V 写 cache**:value 经 GM→UB→GM 中继写回 `gm_vcache`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 value GM→UB→GM 写 cache 段);
2. **加载 cos/sin**:`COSSIN_FP32` 时按 fp32 直接搬入 `cos_fp32_*`/`sin_fp32_*` 缓冲([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 `COSSIN_FP32` cos/sin 搬入段);否则按 Dtype 搬入 `cos_dtype_*`/`sin_dtype_*`,bf16 路径再 `vconv_bf162f32` 升精度([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 legacy Dtype 搬入 + `vconv_bf162f32` 段)。每份 `d/2` 数据加载两次分别落在缓冲前/后半,拼成全宽表;MroPE 模式额外加载 h/w 两路表(fp32 路径 [csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 MroPE fp32 h/w 加载段,legacy 路径 [csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 MroPE legacy 加载段);
3. **K 旋转**:`COSSIN_FP32 || is_bf16` 调 `calc_cossin_cast<Dtype, COSSIN_FP32>`,legacy fp16 调 `calc_cossin`;`COSSIN_FP32` 时旋转后 `convert_output<Dtype>` 做唯一一次舍入写回 `key_dtype_ubuf`,然后 `copy_ubuf_to_gm` 同时写回 `gm_key`(原地)与 `gm_kcache`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 K 旋转 + 写回段);
4. **Q 旋转 + 缩放**:再次调用旋转函数;`COSSIN_FP32` 时先 `vmuls ... scale_fp32` 在 fp32 域缩放,再 `convert_output<Dtype>` 一次舍入写回 `gm_query`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `rope_and_cache` 中 Q 旋转 + `vmuls ... scale_fp32` 缩放段);legacy bf16 路径则把已舍入的 Q 再升回 fp32 缩放、再次舍入,以匹配 torch_npu 的逐算子舍入;legacy fp16 路径直接 `vmuls ... scale` 在 fp16 域缩放。

同步链:`PIPE_S → PIPE_MTE2`(标量读 pos/slot 对 MTE2 可见)→ `PIPE_MTE3 → PIPE_MTE2`(上一 token 搬出完成、缓冲可复用)→ `PIPE_MTE2 → PIPE_V`(数据就绪)→ `PIPE_V → PIPE_MTE3`(结果就绪)。外层批间用 `wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0/1)` 等全部搬出完成。

### 关键计算:旋转的核心向量指令

旋转核心函数有两个,均按 `COSSIN_FP32` 静态分派:

**`calc_cossin_cast<Dtype, COSSIN_FP32>`**([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin_cast`,fp32 路径与 bf16 旧路径共用):

- 输入先经 `convert_input<Dtype>`([csrc/kernels/kernel_macro.h](../../../csrc/kernels/kernel_macro.h) 的 `convert_input`)升精度到 fp32;`COSSIN_FP32` 时 cos/sin 已是 fp32,无需此步;
- MroPE 合成:先按掩码 `vector_dup` 清零,再 `vadd` 累加 h/w 两路表,最后 `copy_ubuf_to_ubuf` 复制到后半形成全宽表([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin_cast` 中 MroPE 合成段);
- `vmul` 计算 `x * sin` 与 `x * cos`(`n_head` 次 repeat,块步长 `head_stride`,表侧步长 0 即按行广播);head_dim=128 时 fp32 数据达 512B,超出单次 repeat 的 256B(`VECTOR_MAX_BYTESIZE`),乘法分前/后两半两次 `vmul`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin_cast` 中 `vmul` x*sin/x*cos 段);
- `vsub`/`vadd` 完成半旋转:前半取 `x[:half]*cos - x[half:]*sin`,后半取 `x[half:]*cos + x[:half]*sin`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin_cast` 中 `vsub`/`vadd` 半旋转段);
- **舍入分叉**:`COSSIN_FP32=true` 时**不在函数内做最终舍入**,把 fp32 结果留给调用方(其 fp32 缩放步骤之后由 `convert_output` 统一舍入);`COSSIN_FP32=false && is_bf16` 时,在乘积之后做一次 bf16 往返舍入仿真 torch_npu 的逐算子舍入([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin_cast` 中 bf16 往返舍入段,注释"为了保证与torch_npu计算结果一致"),并在函数末尾把旋转结果舍入回 bf16 供调用方再次升精度缩放([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin_cast` 末尾 bf16 舍入段)。

**`calc_cossin<Dtype>`**([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin`,仅 legacy fp16 直接计算路径)在 fp16 域执行同类计算:

- MroPE 合成:先按掩码 `vector_dup` 清零,再 `vadd` 累加 h/w 两路表,最后 `copy_ubuf_to_ubuf` 复制到后半形成全宽表([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin` 中 MroPE 合成段);
- `vmul` 计算 `x * sin` 与 `x * cos`(`n_head` 次 repeat,块步长 `head_stride`,表侧步长 0 即按行广播,[csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin` 中 `vmul` x*sin/x*cos 段);
- `vsub`/`vadd` 完成半旋转:前半取 `x[:half]*cos - x[half:]*sin`(mask 限制只写前 `half` 条 lane),后半取 `x[half:]*cos + x[:half]*sin`([csrc/kernels/rope_and_cache.h](../../../csrc/kernels/rope_and_cache.h) 的 `calc_cossin` 中 `vsub`/`vadd` 半旋转段)。head_dim=128 时全宽 mask 一次 repeat 覆盖 128 个 fp16;head_dim=64 时 mask 取低 64 lane。

`convert_input` / `convert_output` 为 `kernel_macro.h` 提供的模板助手,内部按 Dtype 选择 `vconv_f162f32`/`vconv_bf162f32` 与 `vconv_f322f16`/`vconv_f322bf16r`,并处理不足一个 repeat 的尾部 lane(`SetMask`),使 fp32 路径下 fp16 与 bf16 的舍入行为统一。

### 测试参考

[tests/kernels/rope_and_cache.py](../../../tests/kernels/rope_and_cache.py):以 `use_fp32` 开关分别覆盖 FP32 cossin(默认)与 model-dtype cossin(已废弃)两条路径,fp16/bf16 × (head_dim, rot_dim) ∈ {(64,64), (128,128), (128,64)},batch 8 × seq 10,与 torch 的 NeoX 风格 `apply_rotary_emb` + Q 缩放 + cache 写入对比;另含 MroPE 三路位置交织用例、dim-0 pitched(非连续)cache 回归用例(K/V cache 为 `buf[::2]` 视图,dim-0 stride 为连续 pitch 的两倍,验证按 `slot/block_size*key_cache_stride + slot%block_size*kv_size` 的 stride 寻址),以及 2D MroPE position 非连续/超宽回归用例(pitched 列切片 `[3, 2N][:, :N]` 与列数超出 num_tokens 的 `[3, N+64]` 连续缓冲,验证按 dim-0 跨度定位 h/w 行)。
