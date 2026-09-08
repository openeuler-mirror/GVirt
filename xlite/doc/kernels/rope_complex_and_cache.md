# rope_complex_and_cache

## 功能概述

复数风格的旋转位置编码(complex RoPE,DeepSeek MLA 系约定):把每个 head 中长度为 `ropeDim` 的 rope 区视为 `ropeDim/2` 个复数,与 `e^{i·pos·θ_k}` 相乘完成旋转,并把整个 head(含 rope 区前的 remain 部分)按 `slot_mapping` 写入 vCache。支持正向/逆向(`inverse`,共轭旋转)与输出交错/半分两种布局(`outInterleaved`),兼容 rope 区位于 head 头部(MLA pe cache)或尾部(CXA swa kv `[remain | rope]` 布局)两种排布。非 cache 的纯计算入口 `rope_complex` 复用同一 kernel(host 传 `output_ptr` 非 null、`vcache=nullptr`,见 [csrc/op.cpp:1061-1082](../../csrc/op.cpp#L1061-L1082))。

数学语义(输入按 `(r0, i0, r1, i1, ...)` 交错解释,`half = ropeDim/2`):

```
real[k] = x[2k]  * cos_k ∓ x[2k+1] * sin_k     (正向 -, 逆向 +)
imag[k] = x[2k+1] * cos_k ± x[2k]   * sin_k     (正向 +, 逆向 -)
```

输出布局:`outInterleaved=false` 时为半分 `[r0..r(half-1) | i0..i(half-1)]`(MLA/DSA kv-cache 约定),`true` 时为交错 `[r0,i0,r1,i1,...]`(torch `view_as_real().flatten` 约定)。

## 输入输出参数

Python 接口(cache 形式):`rope_complex_and_cache(rt, n_local_heads, step_dim, rope_dim, offset, vdim, input_with_r, freqs, position, block_size, v_cache, slot_mapping, out_interleaved)`。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input_with_r | 输入 | `[num_tokens, n_local_heads, step_dim]` | float16 / bfloat16 | 每 token 的 head 数据;`offset` 指向其中 rope 区起始(CXA:`[remain | rope]`,rope 在尾部;MLA:offset=0,rope 在头部) |
| freqs | 输入 | `[max_pos, rope_dim]` | float32 | 复数频率表,交错存 `cos(pos·θ_k), sin(pos·θ_k)`(`torch.polar` 实部/虚部 flatten,见测试 [tests/kernels/rope_complex.py:19-34](../../tests/kernels/rope_complex.py#L19-L34)) |
| position | 输入 | `[num_tokens]` | int64 | 每 token 的位置 id,用于索引 freqs 行 |
| v_cache | 输出 | `[num_blocks, block_size, n_local_heads, vdim]` | 同 input | 每 slot 存一个完整 head(vdim);CXA 场景 vdim=step_dim 包含 remain,MLA pe cache 场景 vdim=rope_dim 只存 rope 区 |
| slot_mapping | 输入 | `[num_tokens]` | int32 | token 的平坦 slot 索引;`v_cache + slot * n_local_heads * vdim` 寻址 |
| output | 输出(纯计算模式) | `[num_tokens, n_local_heads, rope_dim]` | 同 input | 仅 `rope_complex` 入口使用;cache 入口 host 传 `nullptr`([csrc/op.cpp:1103-1105](../../csrc/op.cpp#L1103-L1105)) |
| n_local_heads | 标量 | - | uint32 | 头数;cache 模式断言为 1 |
| step_dim | 标量 | - | uint32 | input 第 2 维(完整 head 维度) |
| rope_dim | 标量 | - | uint32 | rope 区长度(偶数),测试覆盖 64/128/192/576 |
| offset | 标量 | - | uint32 | rope 区在 head 内的起始偏移,`step_dim - rope_dim` |
| vdim | 标量 | - | uint32 | vCache 每 slot 的 head 宽度 |
| block_size | 标量 | - | uint32 | cache 块大小;非 0 且 vcache/slot_mapping 非空时启用 cache 写入([csrc/kernels/rope_complex_and_cache.h:22-25](../../csrc/kernels/rope_complex_and_cache.h#L22-L25)) |
| inverse | 标量 | - | bool | 逆向旋转(共轭),kernel 内按 `±` 选择 vadd/vsub 方向 |
| out_interleaved | 标量 | - | bool | 输出交错布局开关 |

## 支持的数据类型

- `float16_t`([rope_complex_and_cache_float16_t.cpp](../../csrc/kernels/rope_complex_and_cache_float16_t.cpp))
- `bfloat16_t`([rope_complex_and_cache_bfloat16_t.cpp](../../csrc/kernels/rope_complex_and_cache_bfloat16_t.cpp))

按 `inputWithR.dtype` 单独判别([csrc/op.cpp:1094-1099](../../csrc/op.cpp#L1094-L1099)),内部统一升到 fp32 计算。

## 实现原理

实现位于 [csrc/kernels/rope_complex_and_cache.h](../../csrc/kernels/rope_complex_and_cache.h),C220 向量核,token 级多 Block 并行:每个 block 处理 `index = (block_idx + block_num - coreOffset) % block_num` 起、步长 `block_num` 的 token([csrc/kernels/rope_complex_and_cache.h:138-139](../../csrc/kernels/rope_complex_and_cache.h#L138-L139))。

### UB 内存布局

从地址 0 顺序排布([csrc/kernels/rope_complex_and_cache.h:59-104](../../csrc/kernels/rope_complex_and_cache.h#L59-L104)),全部按 `VECTOR_MAX_BYTESIZE` 对齐:

- `input0/input1`、`out0/out1`:输入/输出乒乓对,大小 `nLocalHeads * headBytes`(cache 模式为整 head 宽度,非 cache 模式仅 rope 宽度);
- `freqs0/freqs1`:频率表乒乓对;
- `vRemain`:head 中 rope 区以外部分的暂存(cache 模式整 head 加载时用);
- fp32 工作区:`inOutFP32`、`x_even`/`x_odd`、`cos`/`sin`、`x_even_cos`/`x_odd_sin`/`x_even_sin`/`x_odd_cos`(各 `ropeDim * nLocalHeads` 个 float);
- `positionUB`(maxCnt=256 个 uint64)、`slotMappingUB`(cache 模式):索引批量预取区;
- `interleavedFP32Head` + `vgatherIndicesUB`:仅 `outInterleaved` 路径的单头暂存与 vgather 索引表。

`assert(off <= UB_SIZE)` 保证布局可容纳。

### 数据流与流水线同步

每 token 一次乒乓切换(`curr = 1 - curr`),用 4 组事件(`EVENT_ID0+curr` 数据链、`EVENT_ID2+curr` freqs 链)驱动 MTE2→V→MTE3 三级流水([csrc/kernels/rope_complex_and_cache.h:128-135](../../csrc/kernels/rope_complex_and_cache.h#L128-L135) 预置初始 flags):

1. **索引批量预取**:每 256 个 token 从 GM 搬一批 `position`(及 cache 模式的 `slot_mapping`)到 UB,标量单元直接 `positionUB[idx]` 取用([csrc/kernels/rope_complex_and_cache.h:141-157](../../csrc/kernels/rope_complex_and_cache.h#L141-L157)),用 `PIPE_S↔PIPE_MTE2` 事件同步;
2. **输入搬入**:`copy_gm_to_ubuf` 按 `nLocalHeads` 个 head、块步长 `srcStride` 装载([csrc/kernels/rope_complex_and_cache.h:170-171](../../csrc/kernels/rope_complex_and_cache.h#L170-L171));cache 模式且 remain>0 时整 head 装载(`fullHeadLoad`),否则只装 rope 区;
3. **freqs 搬入**:`freqs_gm = freqs + pos * ropeDim`([csrc/kernels/rope_complex_and_cache.h:165-176](../../csrc/kernels/rope_complex_and_cache.h#L165-L176));
4. **升精度**:`vconv_f162f32` / `vconv_bf162f32` 把 rope 区升到 fp32([csrc/kernels/rope_complex_and_cache.h:178-183](../../csrc/kernels/rope_complex_and_cache.h#L178-L183));`fullHeadLoad` 时同时把 remain 区 `copy_ubuf_to_ubuf` 暂存到 `vRemain`([csrc/kernels/rope_complex_and_cache.h:184-193](../../csrc/kernels/rope_complex_and_cache.h#L184-L193));
5. **计算**(纯向量指令,见下节);
6. **降精度写回**:非交错路径一次 `vconv_f322f16`/`vconv_f322bf16r` 转换全部头;交错路径逐头 `vgather` 重排后转换([csrc/kernels/rope_complex_and_cache.h:244-272](../../csrc/kernels/rope_complex_and_cache.h#L244-L272));`fullHeadLoad` 时把 remain 区拷回输出缓冲;
7. **写出**([csrc/kernels/rope_complex_and_cache.h:284-297](../../csrc/kernels/rope_complex_and_cache.h#L284-L297)):cache 模式把整 head 写到 `v_cache + slot * nLocalHeads * vdim`,若 `output_ptr` 非空再把 rope 区写到 output;非 cache 模式直接写 output。

### 关键计算步骤

全部在 fp32 域完成([csrc/kernels/rope_complex_and_cache.h:197-241](../../csrc/kernels/rope_complex_and_cache.h#L197-L241)):

1. **奇偶分离**:`vreducev2` 以 step=1/2 分别抽取偶数 lane(`x0,x2,...` → real)与奇数 lane(`x1,x3,...` → imag)——同时用于 freqs 表(得到交错的 cos/sin 序列)与输入([csrc/kernels/rope_complex_and_cache.h:200-212](../../csrc/kernels/rope_complex_and_cache.h#L200-L212));
2. **四组乘积**:`vmul` 计算 `x_even*cos`、`x_odd*sin`、`x_even*sin`、`x_odd*cos`(head 间块步长 `half_rope_blocks`,[csrc/kernels/rope_complex_and_cache.h:215-223](../../csrc/kernels/rope_complex_and_cache.h#L215-L223));
3. **复数乘法合成**(`inverse` 二选一,[csrc/kernels/rope_complex_and_cache.h:226-240](../../csrc/kernels/rope_complex_and_cache.h#L226-L240)):
   - 正向:`real = x_even*cos - x_odd*sin`(`vsub`),`imag = x_even*sin + x_odd*cos`(`vadd`),其中 imag 写到 `inOutFP32 + ropeDim/2`;
   - 逆向:符号取反,即旋转 `-θ`。

交错输出路径在 kernel 启动时用标量单元填一张 vgather 索引表:`indices[2k] = k`(real[k])、`indices[2k+1] = half+k`(imag[k]),之后每头一次 `vgather` 把半分布局 `[r0..r31 | i0..i31]` 重排为 `[r0,i0,r1,i1,...]`([csrc/kernels/rope_complex_and_cache.h:115-126](../../csrc/kernels/rope_complex_and_cache.h#L115-L126)、[csrc/kernels/rope_complex_and_cache.h:247-263](../../csrc/kernels/rope_complex_and_cache.h#L247-L263))。

### 测试参考

- [tests/kernels/rope_complex.py](../../tests/kernels/rope_complex.py):非 cache 路径(n_local_heads=64,inverse × interleaved × (rope_dim, q_dim) 组合),与 torch `view_as_complex` 复数乘对比;
- [tests/kernels/rope_complex_and_cache.py](../../tests/kernels/rope_complex_and_cache.py):cache 路径(CXA `[remain|rope]` head_dim 512/128 与 MLA rope-only 64,n_local_heads=1),校验 vCache 逐 slot 与参考一致。
