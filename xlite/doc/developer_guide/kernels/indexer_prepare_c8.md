# indexer_prepare_c8

## 功能概述

GLM-5.1 Lightning Indexer(LI)的 **C8(INT8 量化)** K/Q 预处理融合算子(核内函数 `indexer_prepare_c8`,见 `csrc/kernels/indexer_prepare_c8.h` 的 `indexer_prepare_c8`)。在一次向量 kernel 启动中同时处理 K 与(长序列时)Q:

- **K**:LayerNorm(使用传入 `norm_eps`)→ 前 64 维交错 RoPE → 7 级 Hadamard → 动态量化,按 `slot_mapping` 写入 paged **INT8** `index_k_cache` 与 **FP16** `k_scale_cache`;`slot = -1`(padding)跳过。
- **Q**(`is_long`):每 head 做 RoPE → Hadamard → 量化(**不做 LayerNorm**),稠密写入 INT8 `q8`、FP16 `q_scale`,并把 Q scale 折叠进头权重 `scaled_weights`。
- **独立 K 测试入口**(`indexer_k_cache_c8`):同一 kernel、Q 阶段禁用,用于单测 K 路径。

固定配置:`dim = 128`、`rope_dim = 64`、`hadamard_stages = 7`、`quant_max = 127`、Hadamard 归一化系数 `hadamard_coefficient = BF16(1/√128) = 0.08837890625`(`csrc/kernels/indexer_prepare_c8.h` 的 `IndexerC8Buffers`,关键字 `static constexpr uint32_t dim = 128` / `hadamard_coefficient` 配置段)。Indexer 用 32 个头、head dim 128、RoPE dim 64,局部 `tpSize = 1`。

数学语义:

1. K:`ln = LayerNorm(kw[:, :128], kNorm, kNormBias, eps)`;`ln[:, :64] = InterleavedRoPE(ln[:, :64], freqs[position])`;`h = Hadamard_7(ln) * (1/√128)`;动态量化 `q8, scale = dyn_quant(h, 127)`,按 slot 写 cache。
2. Q(每 head):`h = Hadamard_7(InterleavedRoPE(q_row[:, :64])) * (1/√128)`;量化得 `q8`、`q_scale`;`scaled_weights = FP16(原始头权重 × FP16(q_scale))`(不使用传入的 `scale`)。

动态量化(以变换后每行 128 元素 `x` 为单位,见 `csrc/kernels/indexer_prepare_c8.h` 的 `rope_hadamard_quant_c8`,关键字 `vabs(ub.tmp, ub.x` / `ReduceMax(ub.other, ub.tmp, ub.dim)` / `vconv_f322s32r` 动态量化段):`amax = max(|x|)`;`amax != 0` 时 `factor = vdiv(127, amax)`,对 `x*factor` 取整转 INT8,scale 存 `FP16(amax × 1/127)`;`amax == 0` 时量化值与 scale 均写零。

## 输入输出参数

Python 侧调用(`xlite/_C.pyi`):

```python
# 独立 K 测试入口
indexer_k_cache_c8(rt, k, k_norm, k_norm_bias, freqs, position, slot_mapping,
                   k_cache, scale_cache, norm_eps=1e-6)

# 完整 K+Q 路径(经 indexer_prepare,当 index_k_cache 为 INT8 时走 C8)
indexer_prepare(rt, kw, k_norm, k_norm_bias, freqs, position, index_head_dim,
               index_n_heads, rope_head_dim, block_size, index_k_cache, slot_mapping,
               norm_eps, q, scale, top_k, is_long, k_scale_cache=, q8=, q_scale=,
               scaled_weights=)   # k_scale_cache/q8/q_scale/scaled_weights 仅 C8 必填
```

host 侧有两条 launch 路径:

- `XliteOpIndexerPrepare`(`csrc/op.cpp` 的 `XliteOpIndexerPrepare`,关键字 `if (indexKCache.dtype == INT8)` LI-C8 分支)在 `indexKCache.dtype == INT8` 时走 LI-C8:约束 `indexHeadDim==128 && ropeHeadDim==64 && indexNHeads==32 && kw 为 BF16 && kNorm/kNormBias 为 FP32 && q 为 BF16 && tpSize==1`,长序列时 `q/q8/qScale/scaledWeights` 必须非空;
- `XliteOpIndexerKCacheC8`(`csrc/op.cpp` 的 `XliteOpIndexerKCacheC8`)为独立 K 测试接口,禁用 Q 阶段。kernel 签名(`csrc/kernels/indexer_prepare_c8.h` 的 `INDEXER_PREPARE_C8_FUNC_DEFINE` 宏):

```cpp
indexer_prepare_c8_<dtype>(GM_ADDR kw, GM_ADDR weight, GM_ADDR bias, GM_ADDR freqs,
                           GM_ADDR positions, GM_ADDR slots, GM_ADDR k_cache,
                           GM_ADDR k_scale_cache, GM_ADDR q, GM_ADDR q8, GM_ADDR q_scale,
                           GM_ADDR scaled_weights, uint32_t tokens, uint32_t row_stride,
                           uint32_t heads, uint32_t max_position, uint32_t capacity,
                           bool freqs_fp32, bool is_long, float norm_eps)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| kw / k | 输入(只读) | `[tokens, 160]` | bf16 | Indexer K 投影;`160 = 128(head_dim) + 32(heads)` |
| weight(kNorm) | 输入 | `[128]` | fp32 | LayerNorm weight |
| bias(kNormBias) | 输入 | `[128]` | fp32 | LayerNorm bias |
| freqs | 输入 | `[max_position, 64]` 交错 cos/sin | fp32 或 bf16 | 预计算 RoPE 频率表;`freqs_fp32` 由 host 按 dtype 推断 |
| positions | 输入 | `[tokens]` | int64 | 每个 token 的绝对序列位置 |
| slots(slot_mapping) | 输入 | `[tokens]` | int32 | token → cache 平坦槽位;`-1` 跳过(padding) |
| k_cache(index_k_cache) | 输出 | `[num_blocks, block_size, 1, 128]` | int8 | paged INT8 K cache;`capacity = num_blocks*block_size` |
| k_scale_cache | 输出 | `[num_blocks, block_size, 1, 1]` | fp16 | 每 token 一个 FP16 scale;短序列也写 |
| q | 输入(只读) | `[tokens, 4096]` | bf16 | Indexer Q(`4096 = 32*128`);仅 `is_long` 有效,否则传空 |
| q8 | 输出 | `[tokens, 4096]` | int8 | 量化 Q;仅 `is_long` |
| q_scale | 输出 | `[tokens, 32]` | fp16 | 每 Q head 一个 scale;仅 `is_long` |
| scaled_weights | 输出 | `[tokens, 32]` | fp16 | `FP16(原始头权重 × FP16(q_scale))`;仅 `is_long` |
| tokens / row_stride / heads / max_position / capacity | — | 标量 | uint32 | `heads=32`;`max_position = freqs.shape[0]`;`capacity = k_scale_cache.numel` |
| freqs_fp32 | — | 标量 | bool | freqs 是否为 fp32(host 推断) |
| is_long | — | 标量 | bool | 长序列开关,启用 Q 阶段 |
| norm_eps | — | 标量 | float | LayerNorm epsilon |

`dest < 0 || dest >= capacity || pos < 0 || pos >= max_position` 的行直接 `continue`(`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_prepare_k_c8`,关键字 `if (dest < 0 || static_cast<uint32_t>(dest) >= capacity` guard),不读 RoPE、不写 cache。

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `indexer_prepare_c8_bfloat16_t` | `csrc/kernels/indexer_prepare_c8_bfloat16_t.cpp` | kw/q 为 bf16;kNorm/kNormBias 为 fp32;k_cache 为 int8、scale 为 fp16 |

仅 BF16 一种实例化(配置固定为 32 头/128 维/64 RoPE)。dtype 分派见 `csrc/op.cpp` 的 `XliteOpIndexerPrepare` / `XliteOpIndexerKCacheC8`。kernel 为纯向量核(`__DAV_C220_VEC__`,以 `rt.aivNum` 个 block launch)。

## 实现原理

### 总体结构与并行

顶层 `indexer_prepare_c8`(`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_prepare_c8`):按 `is_long ? tokens*heads : tokens` 决定每核行数(`block_idx >= 该值` 的核直接返回),初始化 `IndexerC8Buffers` 后先调 K 阶段 `indexer_prepare_k_c8`,长序列再调 Q 阶段 `indexer_prepare_q_c8`。K、Q 共用同一份 UB workspace,因此 K 结束后 `pipe_barrier(PIPE_ALL)` 等本核流水排空再进 Q(`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_prepare_c8`,关键字 `IndexerC8Buffers ub; ub.Init(); indexer_prepare_k_c8(...)` K/Q 分派段)。行按 `for (row = block_idx; row < N; row += block_num)` 轮询分配(无双缓冲)。

### IndexerC8Buffers 与 Init

`IndexerC8Buffers`(`csrc/kernels/indexer_prepare_c8.h` 的 `IndexerC8Buffers`)持有全部 UB 缓冲(weight/bias/x/tmp/other/trig/cos/sin/bf/trig_bf/rounded/half_values/quant/scale/scale_half/position/slot 及 RoPE 与 Hadamard 的 gather 索引表),K、Q 阶段复用。每核约 13,952 字节。

`Init`(`csrc/kernels/indexer_prepare_c8.h` 的 `IndexerC8Buffers::Init`)在进入循环前**预计算全部 gather 索引与符号表**,使逐行变换用 `vgather` 替代标量寻址:

- `cos_indices`/`sin_indices`/`swap_indices`:RoPE 相邻对(cos 取偶位、sin 取奇位、swap 取配对元素)的 UB 字节偏移,用 `vadds` + `sign_masks`(`csrc/kernels/indexer_prepare_c8.h` 的 `IndexerC8Buffers::Init`,关键字 `sign_masks` / `vadds(rounded, rounded, int32_t((1u << bit) * sizeof(float))`)在 `rounded` 上生成;
- `h_indices`/`h_sign`:7 级 Hadamard 蝶形的配对偏移与 ±1 符号(前 6 级 `vadds` 偏移 + `vector_dup` 符号,第 7 级交换 64 元素两半);
- `rope_sign`/`h_sign` 初始化为全 1,再用 `sign_masks` 在对应位写 -1。

### 共享变换 rope_hadamard_quant_c8

`rope_hadamard_quant_c8`(`csrc/kernels/indexer_prepare_c8.h` 的 `rope_hadamard_quant_c8`)不区分 Q/K,完成 RoPE → Hadamard → 动态量化三步:

1. **RoPE**(前 64 维相邻对,尾部不变):`vgather` 按 `cos_indices`/`sin_indices`/`swap_indices` 取出 cos/sin/配对元素,`vmul`×2 + `vmul`(符号) + `vadd` 得旋转结果(`csrc/kernels/indexer_prepare_c8.h` 的 `rope_hadamard_quant_c8`,关键字 `vgather((__ubuf__ uint32_t *)ub.cos, ub.cos_indices, trig_base` RoPE 段);
2. **Hadamard**(7 级蝶形,FP32):每级 `vgather` 取配对 + `vmul`(符号) + `vadd`,随后 `vmuls` 乘 `hadamard_coefficient`(1/√128)归一化,`convert_output/input` 经 BF16 舍入(`csrc/kernels/indexer_prepare_c8.h` 的 `rope_hadamard_quant_c8`,关键字 `vmuls(ub.x, ub.x, float(ub.hadamard_coefficient)` Hadamard 段);
3. **动态量化**:`vabs` → `ReduceMax`(amax) →(非零时)`vdiv(127, amax)` 得 `scaleRec` → `vmuls`(x*factor) → `vconv_f322s32r` 直接 FP32→INT32 取整(避免 FP16→INT8 二次舍入)→ `vconv_s322f32` → `vconv_f322f16` → `vconv_f162s8` 得 INT8;scale 存为 FP16(`vconv_f322f16`),`amax==0` 时 `scaleRec=0`(`csrc/kernels/indexer_prepare_c8.h` 的 `rope_hadamard_quant_c8`,关键字 `vabs(ub.tmp, ub.x` / `vconv_f322s32r` 动态量化段)。

行加载 `indexer_c8_load_row`(`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_c8_load_row`):GM→UB 取 BF16 行与 freqs(trig),`set_flag/wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)` 同步后 `convert_input` 转 FP32;freqs 为 fp32 时多一步 BF16 转换。行写出 `indexer_c8_store_row`(`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_c8_store_row`):`set_flag/wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)` 后 MTE3 散列写 INT8 `quant` 到 cache slot,再精确 2 字节写 FP16 scale(不覆盖相邻 token 的 scale)。

### K 阶段 indexer_prepare_k_c8

`indexer_prepare_k_c8`(`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_prepare_k_c8`):每核先将 weight/bias 搬入 UB,然后轮询行。每行读 `position`/`slot`(S 管线标量,`MTE2→S` flag 同步),校验后 `indexer_c8_load_row`;接着 FP32 **LayerNorm**(`vmuls`(x/n) → `reduce_sum`(均值) → `vsub` → `vmul` → `reduce_sum`(方差) → `vadds`(eps) → `vsqrt` → `vdiv`(inv_std) → weight/bias 仿射,`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_prepare_k_c8`,关键字 `vmuls(ub.tmp, ub.x, inv_norm_dim` / `reduce_sum(ub.tmp, 1, ub.dim)` LayerNorm 段),`convert_output/input` 经 BF16 舍入;再调 `rope_hadamard_quant_c8` 与 `indexer_c8_store_row` 写 `k_cache`/`k_scale_cache`。

### Q 阶段 indexer_prepare_q_c8

`indexer_prepare_q_c8`(`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_prepare_q_c8`):`n_rows = tokens*heads`,轮询行(`position = positions[row/heads]`)。**无 LayerNorm**,直接 `indexer_c8_load_row` + `rope_hadamard_quant_c8` + `indexer_c8_store_row` 稠密写 `q8`/`q_scale`。随后把 FP16 Q scale 折叠进头权重(`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_prepare_q_c8`,关键字 `offset = (uint64_t)(row / heads) * (ub.dim + heads) + ub.dim + row % heads` / `vconv_f162f32(ub.other, ub.scale_half` Q scale 折叠段):按 `offset = (row/heads)*(dim+heads) + dim + row%heads` 从 `kw` 读头权重,`convert_input` 转 FP32,`vconv_f162f32` 取 scale,`vmul` 后 `vconv_f322f16` 写 `scaled_weights[row]`。

### 流水线同步

行内 MTE2(搬入)↔ V(计算)↔ MTE3(写回)用 `set_flag/wait_flag` 计数事件 ID 交叉同步:`EVENT_ID0` 用于 MTE2↔V 的行加载,LN/量化归约结果被 S 管线标量消费前有 `set_flag(PIPE_V, PIPE_S)`/`wait_flag`(`csrc/kernels/indexer_prepare_c8.h` 的 `rope_hadamard_quant_c8`,关键字 `ReduceMax(ub.other, ub.tmp, ub.dim)` 后的 `set_flag(PIPE_V, PIPE_S, EVENT_ID0)` / `wait_flag(PIPE_V, PIPE_S, EVENT_ID0)`),写回前后用 `PIPE_V↔PIPE_MTE3` 配对。K 阶段结束 `pipe_barrier(PIPE_ALL)` 排空后才进 Q(`csrc/kernels/indexer_prepare_c8.h` 的 `indexer_prepare_k_c8`,关键字 K 阶段末尾 `pipe_barrier(PIPE_ALL)`)。
