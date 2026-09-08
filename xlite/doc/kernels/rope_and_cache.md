# rope_and_cache

## 功能概述

对融合 QKV 张量中的 Q 和 K 做 NeoX 风格的旋转位置编码(RoPE),对 Q 追加 `1/sqrt(headDim)` 缩放,并将旋转后的 K 以及未旋转的 V 按 `slot_mapping` 写入 paged KV cache;同时把旋转后的 K、缩放旋转后的 Q 原地写回 QKV 张量。支持多模态 MroPE(文本/高/宽三路位置)与标准单路 RoPE 两种模式。该算子由小艺团队贡献,参考论文《XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads》[ASPLOS 2026]([csrc/kernels/rope_and_cache.h:13-14](../../csrc/kernels/rope_and_cache.h#L13-L14))。

数学语义(NeoX 半旋转,`half = rotDim / 2`):

```
q_out[:half]   = (q[:half] * cos - q[half:] * sin) * scale
q_out[half:]   = (q[half:] * cos + q[:half] * sin) * scale
k_out[:half]   = k[:half] * cos - k[half:] * sin
k_out[half:]   = k[half:] * cos + k[:half] * sin
```

MroPE 模式下 `cos/sin` 为文本、h、w 三路表按 lane 掩码逐元素相加的结果。

## 输入输出参数

Python 接口:`rope_and_cache(rt, inout, k_cache, v_cache, position, cossin, slot_mapping, n_heads, n_kv_heads, head_dim, rot_dim, block_size, is_neox, mrope_mask_h=0, mrope_mask_w=0)`。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| inout | 输入/输出 | `[num_tokens, (num_heads + 2 * num_kv_heads) * head_dim]` | float16 / bfloat16 | 融合 QKV。host 侧按 TP 切分后的头数拆出 q/k/v 三个指针([csrc/op.cpp:880-886](../../csrc/op.cpp#L880-L886)),q 在行首,k、v 依次随其后;Q/K 旋转后原地写回 |
| k_cache | 输出 | `[block_num, block_size, num_kv_heads, head_dim]` | 同 inout | KV cache 的 K 平面,按 slot 展开寻址 |
| v_cache | 输出 | 同 k_cache | 同 inout | KV cache 的 V 平面,V 不做旋转,直接搬运写入 |
| position | 输入 | `[num_tokens]`(MroPE 为 `[3, num_tokens]`) | int64 | 每 token 的位置 id;MroPE 时第 2/3 行为 h/w 位置([csrc/kernels/rope_and_cache.h:458-460](../../csrc/kernels/rope_and_cache.h#L458-L460)) |
| cossin | 输入 | `[max_pos, rot_dim]` | 同 inout | 预计算表,前 `rot_dim/2` 列为 cos、后 `rot_dim/2` 列为 sin(见测试 `precompute_freqs_cis` 的 `cat((cos, sin), dim=-1)`) |
| slot_mapping | 输入 | `[num_tokens]` | int32 | 每 token 在 cache 中的平坦 slot 索引 |
| n_heads / n_kv_heads | 标量 | - | uint32 | **全局** Q 头数与 KV 头数(host 内部除以 tpSize 得本 rank 头数,见 [csrc/op.cpp:879-881](../../csrc/op.cpp#L879-L881)) |
| head_dim | 标量 | - | uint32 | 每头维度,支持 64 / 128 |
| rot_dim | 标量 | - | uint32 | 旋转维度,支持 64 / 128(可小于 head_dim,如 head_dim=128 + rot_dim=64) |
| block_size | 标量 | - | uint32 | cache 块大小(kernel 内不直接使用,slot_mapping 已是平坦索引) |
| is_neox | 标量 | - | bool | 仅支持 NeoX 风格,gptj 会抛错([csrc/op.cpp:889-891](../../csrc/op.cpp#L889-L891)) |
| mrope_mask_h / mrope_mask_w | 标量 | - | uint64 | MroPE 的向量 lane 掩码,非 0 时启用三路位置模式 |

`scale = 1.0 / sqrt(headDim)` 由 host 计算传入([csrc/op.cpp:887](../../csrc/op.cpp#L887));q/k/v 的行 stride 均为 `inout.shape[1]`(整行 QKV 宽度)。

## 支持的数据类型

- `float16_t`([rope_and_cache_float16_t.cpp](../../csrc/kernels/rope_and_cache_float16_t.cpp)):fp16 域内直接计算(`calc_cossin`)
- `bfloat16_t`([rope_and_cache_bfloat16_t.cpp](../../csrc/kernels/rope_and_cache_bfloat16_t.cpp)):转 fp32 计算,乘积经一次 bf16 舍入回退以保证与 torch_npu 位级一致(`calc_cossin_cast`)

要求 inout、kCache、vCache、cossin 四者 dtype 一致([csrc/op.cpp:894-897](../../csrc/op.cpp#L894-L897))。

## 实现原理

实现位于 [csrc/kernels/rope_and_cache.h](../../csrc/kernels/rope_and_cache.h),为 C220 向量核(`__DAV_C220_VEC__`)上的标量 C 风格 kernel,由 `XliteOpRopeCache`([csrc/op.cpp:869-907](../../csrc/op.cpp#L869-L907))以 `rt.aivNum` 个 AIV block 启动。

### 两级循环与多 Block 并行

- 外层按批搬运 `positions`/`slot_mapping`(Mrope 还包括 pos_h/pos_w)到 UB 尾部暂存区,批量大小 `iter_posslot_num` 由剩余 UB 空间动态推出([csrc/kernels/rope_and_cache.h:396-403](../../csrc/kernels/rope_and_cache.h#L396-L403));
- 内层将 token 条带化分配到各 block:`loop1 = block_idx + processed_num_tokens`,步长 `block_num`([csrc/kernels/rope_and_cache.h:467-468](../../csrc/kernels/rope_and_cache.h#L467-L468)),即 token 级多核并行。

每个 token 的 GM 地址由标量单元从 UB 暂存区读出 position/slot 计算得到:cos/sin 表偏移 `pos * rot_dim`(sin 再加 `rot_dim/2`),cache 偏移 `slot * num_kv_heads * head_dim`([csrc/kernels/rope_and_cache.h:474-485](../../csrc/kernels/rope_and_cache.h#L474-L485))。

### UB 内存布局

以 `PINGPONG_BUF_NUM = 2` 组乒乓缓冲组织,从地址 0 起依次排布([csrc/kernels/rope_and_cache.h:226-394](../../csrc/kernels/rope_and_cache.h#L226-L394)):

| 区域 | 份数 | 大小 | 用途 |
|---|---|---|---|
| query | 2 | `num_heads * head_dim` | Q 暂存 |
| key / value | 各 2 | `num_kv_heads * head_dim` | K/V 暂存 |
| cos / sin | 各 2 | `rot_dim`(对齐 32B) | 旋转表,加载 `d/2` 后复制到整个 `rot_dim` |
| cos/sin h、w(MroPE) | 各 2 | 同上 | 三路位置表 |
| bf16 专用 fp32 区 | query/key/cos/sin 各 2 + mrope 各 2 + 计算区 | fp32 字节数 | `calc_cossin_cast` 的 fp32 工作区 |
| fp16 计算区 | 2 | `q_bytesize` | `calc_cossin` 的 q*sin 中间结果 |
| 尾部 params 区 | 1 | 剩余空间 | positions(pos/pos_h/pos_w)与 slot_mapping 分批暂存 |

搬运统一用预配置的 DMI 参数(`__set_dmi_config`,[csrc/kernels/rope_and_cache.h:425-439](../../csrc/kernels/rope_and_cache.h#L425-L439)),q/kv/cossin/pos/slot 各一份 burst 描述。

### 数据流与流水线同步

每 token 按顺序执行,全程通过事件标志驱动 MTE2(搬入)→ V(计算)→ MTE3(搬出)三级流水,乒乓两组事件(`EVENT_ID0`/`EVENT_ID1`,[csrc/kernels/rope_and_cache.h:467-470](../../csrc/kernels/rope_and_cache.h#L467-L470)):

1. **V 写 cache**:value 经 GM→UB→GM 中继写回 `gm_vcache`([csrc/kernels/rope_and_cache.h:490-495](../../csrc/kernels/rope_and_cache.h#L490-L495));
2. **加载 cos/sin**:每份 `d/2` 数据加载两次分别落在缓冲前/后半,拼成全宽表([csrc/kernels/rope_and_cache.h:497-503](../../csrc/kernels/rope_and_cache.h#L497-L503));MroPE 模式额外加载 h/w 两路表并转 fp32([csrc/kernels/rope_and_cache.h:504-534](../../csrc/kernels/rope_and_cache.h#L504-L534));
3. **K 旋转**:调用 `calc_cossin` / `calc_cossin_cast`,结果 `copy_ubuf_to_gm` 同时写回 `gm_key`(原地)与 `gm_kcache`([csrc/kernels/rope_and_cache.h:536-558](../../csrc/kernels/rope_and_cache.h#L536-L558));
4. **Q 旋转 + 缩放**:再次调用旋转函数,随后 `vmuls` 乘 scale(fp16 直接乘,bf16 先转 fp32 再舍入回),写回 `gm_query`([csrc/kernels/rope_and_cache.h:560-594](../../csrc/kernels/rope_and_cache.h#L560-L594))。

同步链:`PIPE_S → PIPE_MTE2`(标量读 pos/slot 对 MTE2 可见)→ `PIPE_MTE3 → PIPE_MTE2`(上一 token 搬出完成、缓冲可复用)→ `PIPE_MTE2 → PIPE_V`(数据就绪)→ `PIPE_V → PIPE_MTE3`(结果就绪)。外层批间用 `wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0/1)` 等全部搬出完成。

### 关键计算:旋转的核心向量指令

fp16 路径 `calc_cossin`([csrc/kernels/rope_and_cache.h:132-204](../../csrc/kernels/rope_and_cache.h#L132-L204)):

- MroPE 合成:先按掩码 `vector_dup` 清零,再 `vadd` 累加 h/w 两路表,最后 `copy_ubuf_to_ubuf` 复制到后半形成全宽表([csrc/kernels/rope_and_cache.h:147-169](../../csrc/kernels/rope_and_cache.h#L147-L169));
- `vmul` 计算 `x * sin` 与 `x * cos`(`n_head` 次 repeat,块步长 `head_stride`,表侧步长 0 即按行广播,[csrc/kernels/rope_and_cache.h:178-183](../../csrc/kernels/rope_and_cache.h#L178-L183));
- `vsub`/`vadd` 完成半旋转:前半取 `x[:half]*cos - x[half:]*sin`(mask 限制只写前 `half` 条 lane),后半取 `x[half:]*cos + x[:half]*sin`([csrc/kernels/rope_and_cache.h:192-200](../../csrc/kernels/rope_and_cache.h#L192-L200))。head_dim=128 时全宽 mask 一次 repeat 覆盖 128 个 fp16;head_dim=64 时 mask 取低 64 lane。

bf16 路径 `calc_cossin_cast`([csrc/kernels/rope_and_cache.h:15-130](../../csrc/kernels/rope_and_cache.h#L15-L130))在 fp32 域执行同样的计算,差异在于:

- 输入/表先 `vconv_bf162f32` 升精度;head_dim=128 时 fp32 数据达 512B,超出单次 repeat 的 256B(`VECTOR_MAX_BYTESIZE`),乘法分前/后两半两次 `vmul`([csrc/kernels/rope_and_cache.h:65-80](../../csrc/kernels/rope_and_cache.h#L65-L80) 及注释);
- 两个乘积 `x*cos`、`x*sin` 先 `vconv_f322bf16r` 舍入到 bf16 再升回 fp32,然后才做 `vsub`/`vadd`——这是为了与 torch_npu 的舍入行为保持一致([csrc/kernels/rope_and_cache.h:93-105](../../csrc/kernels/rope_and_cache.h#L93-L105) 注释"为了保证与torch_npu计算结果一致")。

### 测试参考

[tests/kernels/rope_and_cache.py](../../tests/kernels/rope_and_cache.py):fp16/bf16 × (head_dim, rot_dim) ∈ {(64,64), (128,128), (128,64)},batch 8 × seq 10,与 torch 的 NeoX 风格 `apply_rotary_emb` + Q 缩放 + cache 写入对比。
