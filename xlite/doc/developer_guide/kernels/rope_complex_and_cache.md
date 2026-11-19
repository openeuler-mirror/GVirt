# rope_complex_and_cache

## 功能概述

复数风格的旋转位置编码(complex RoPE,DeepSeek MLA 系约定):把每个 head 中长度为 `ropeDim` 的 rope 区视为 `ropeDim/2` 个复数,与 `e^{i·pos·θ_k}` 相乘完成旋转,并把整个 head(含 rope 区前的 remain 部分)按 `slot_mapping` 写入 vCache。支持正向/逆向(`inverse`,共轭旋转)与输出交错/半分两种布局(`outInterleaved`),兼容 rope 区位于 head 头部(MLA pe cache)或尾部(CXA swa kv `[remain \| rope]` 布局)两种排布。非 cache 的纯计算入口 `rope_complex` 复用同一 kernel(host 传 `output_ptr` 非 null、`vcache=nullptr`,见 [csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeComplex`)。另支持可选的整 head 缩放(`do_rotate` 开关):启用时按 host 推导的 `rotateScale`(cache 路径 `1/sqrt(vdim)`、非 cache 路径 `1/sqrt(stepDim)`)对 rope 区做 `vmuls`,用于 `rotate_activation` 类融合(详见参数表 `do_rotate`/`rotateScale`)。

数学语义(输入按 `(r0, i0, r1, i1, ...)` 交错解释,`half = ropeDim/2`):

```text
real[k] = x[2k]  * cos_k ∓ x[2k+1] * sin_k     (正向 -, 逆向 +)
imag[k] = x[2k+1] * cos_k ± x[2k]   * sin_k     (正向 +, 逆向 -)
```

输出布局:`outInterleaved=false` 时为半分 `[r0..r(half-1) | i0..i(half-1)]`(MLA/DSA kv-cache 约定),`true` 时为交错 `[r0,i0,r1,i1,...]`(torch `view_as_real().flatten` 约定)。

## 输入输出参数

Python 接口(cache 形式):`rope_complex_and_cache(rt, n_local_heads, step_dim, rope_dim, offset, vdim, input_with_r, freqs, position, block_size, v_cache, slot_mapping, out_interleaved, do_rotate)`(完整签名见 `xlite/_C.pyi` 的 `rope_complex_and_cache`,末参 `out_interleaved: bool = False`、`do_rotate: bool = False`)。

| 参数 | 方向 | Shape | Dtype | 说明 |
| --- | --- | --- | --- | --- |
| input_with_r | 输入 | `[num_tokens, n_local_heads, step_dim]` | float16 / bfloat16 | 每 token 的 head 数据;`offset` 指向其中 rope 区起始(CXA:`[remain \| rope]`,rope 在尾部;MLA:offset=0,rope 在头部) |
| freqs | 输入 | `[max_pos, rope_dim]` | float32 | 复数频率表,交错存 `cos(pos·θ_k), sin(pos·θ_k)`(`torch.polar` 实部/虚部 flatten,见测试 [tests/kernels/rope_complex.py](../../../tests/kernels/rope_complex.py) 的 `precompute_freqs_cis`) |
| position | 输入 | `[num_tokens]` | int64 | 每 token 的位置 id,用于索引 freqs 行;二维(mrope)布局时仅消费第一行,host 打印一次性警告(`csrc/op.cpp` 的 `WarnPositions2D`) |
| v_cache | 输出 | `[num_blocks, block_size, n_local_heads, vdim]` | 同 input | 每 slot 存一个完整 head(vdim);CXA 场景 vdim=step_dim 包含 remain,MLA pe cache 场景 vdim=rope_dim 只存 rope 区 |
| slot_mapping | 输入 | `[num_tokens]` | int32 | token 的平坦 slot 索引;按 `v_cache + (slot / block_size) * MAX(vcacheStride, block_size*n_local_heads*vdim) + (slot % block_size) * n_local_heads * vdim` 分解寻址(块号方向步长经 MAX 钳制 ≥ 自然步长) |
| output | 输出(纯计算模式) | `[num_tokens, n_local_heads, rope_dim]` | 同 input | 仅 `rope_complex` 入口使用;cache 入口 host 传 `nullptr`([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeComplexAndCache`,cache 入口传 `output_ptr=nullptr`) |
| n_local_heads | 标量 | - | uint32 | 头数;cache 模式断言为 1 |
| step_dim | 标量 | - | uint32 | input 第 2 维(完整 head 维度) |
| rope_dim | 标量 | - | uint32 | rope 区长度(偶数),测试覆盖 64/128/192/576 |
| offset | 标量 | - | uint32 | rope 区在 head 内的起始偏移,`step_dim - rope_dim` |
| vdim | 标量 | - | uint32 | vCache 每 slot 的 head 宽度 |
| block_size | 标量 | - | uint32 | cache 块大小,用于把平坦 slot 分解为块号与块内偏移;非 0 且 vcache/slot_mapping 非空时启用 cache 写入([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `need_v_cache` 判定段) |
| vcacheStride | 标量(host 推导) | - | uint32 | vCache 的 dim-0 跨度(元素数),host 传 `vCache.stride[0]`;连续 cache 时等于 block_size × n_local_heads × vdim 的乘积,非连续时为宿主传入的真实跨距 |
| inverse | 标量 | - | bool | 逆向旋转(共轭),kernel 内按 `±` 选择 vadd/vsub 方向 |
| out_interleaved | 标量 | - | bool | 输出交错布局开关 |
| do_rotate | 标量 | - | bool | 可选整 head 缩放开关(Python 末参 `do_rotate`,host 透传为内核 `doRotate`);启用时按 `rotateScale` 对 rope 区做 `vmuls` 缩放(cache 模式 `doRotate && fullHeadLoad` 缩放整 head 含 remain,非 cache 模式 `doRotate && !need_v_cache` 只缩 rope 区)。非 cache 入口 `rope_complex` 的 nope 区缩放由调用方另做 `muls` 补齐([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeComplex` 注释,关键字 `rotate_activation`) |
| rotateScale | 标量(host 推导) | - | float | `doRotate` 缩放系数,host 内部推导(不从 Python 透传):cache 路径 `XliteOpRopeComplexAndCache` 取 `1/sqrt(vdim)`、非 cache 路径 `XliteOpRopeComplex` 取 `1/sqrt(stepDim)`([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeComplexAndCache`/`XliteOpRopeComplex`,关键字 `rotateScale = doRotate ? (1.0f / sqrtf(...))`) |
| minPosition | 标量(内核参数,host 不透传) | - | uint64 | 非 cache 模式下跳过 `position < minPosition` 的行(不搬入、不旋转、不写回;[csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `minPosition` 跳过段,关键字 `pos_per_row < minPosition`)。cache 模式强制归零(写 cache 的行不可跳)。调用方 `indexer_prepare` 传 `top_k` 跳过稠密注意力 token 的 Q-RoPE |

## 支持的数据类型

- `float16_t`([rope_complex_and_cache_float16_t.cpp](../../../csrc/kernels/rope_complex_and_cache_float16_t.cpp))
- `bfloat16_t`([rope_complex_and_cache_bfloat16_t.cpp](../../../csrc/kernels/rope_complex_and_cache_bfloat16_t.cpp))

按 `inputWithR.dtype` 单独判别([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpRopeComplex`/`XliteOpRopeComplexAndCache` dtype 校验段),内部统一升到 fp32 计算。

## 实现原理

实现位于 [csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h),C220 向量核,token 级多 Block 并行:每个 block 处理 `index = (block_idx + block_num - coreOffset) % block_num` 起、步长 `block_num` 的 token([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `first = (block_idx + block_num - coreOffset)` token 循环段)。

### UB 内存布局

从地址 0 顺序排布([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 UB 布局分配段),全部按 `VECTOR_MAX_BYTESIZE` 对齐:

- `input0/input1`、`out0/out1`:输入/输出乒乓对,大小 `nLocalHeads * headBytes`(cache 模式为整 head 宽度,非 cache 模式仅 rope 宽度);
- `freqs0/freqs1`:频率表乒乓对;
- `vRemain`:head 中 rope 区以外部分的暂存(cache 模式整 head 加载时用);
- fp32 工作区:`inOutFP32`、`x_even`/`x_odd`、`cos`/`sin`、`x_even_cos`/`x_odd_sin`/`x_even_sin`/`x_odd_cos`(各 `ropeDim * nLocalHeads` 个 float);
- `positionUB`(maxCnt=256 个 uint64)、`slotMappingUB`(cache 模式):索引批量预取区;
- `rotateFP32`:`doRotate && fullHeadLoad` 条件分配(`ROUND_UP(totalHeadFPBytes, VECTOR_MAX_BYTESIZE)`,整 head 的 fp32 暂存),供 cache 路径 `vconv→vmuls(rotateScale)→vconv` 整 head 缩放使用;`doRotate` 关闭或 `!fullHeadLoad` 时分配 0 字节([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `rotateFP32` 条件分配段,关键字 `doRotate && fullHeadLoad`);
- `vgatherIndicesUB` + `interleavedFP32`:仅 `outInterleaved` 路径的 vgather 索引表与交错暂存——`flatGather`(`ropeDim == 64` 且表装得进剩余 UB)时为扁平表(每个向量 repeat 一行 64 项、含各 head 绝对偏移),否则为单头相对偏移表(每头 vgather 基址复用);交错暂存 `interleavedFP32` 别名到 rope 数学死区 `x_odd_cos`(`ropeDim * nLocalHeads` 个 float),不额外占 UB。

`assert(off <= UB_SIZE)` 保证布局可容纳。

### 数据流与流水线同步

每 token 一次乒乓切换(`curr = 1 - curr`),用 4 组事件(`EVENT_ID0+curr` 数据链、`EVENT_ID2+curr` freqs 链)驱动 MTE2→V→MTE3 三级流水([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `set_flag(PIPE_V, PIPE_MTE2`/`PIPE_MTE3` 预置初始 flags 段):

1. **索引批量预取**:每 256 个 token 从 GM 搬一批 `position`(及 cache 模式的 `slot_mapping`)到 UB,标量单元直接 `positionUB[idx]` 取用([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `positionUB`/`slotMappingUB` 批量预取段),用 `PIPE_S↔PIPE_MTE2` 事件同步;
2. **输入搬入**:`copy_gm_to_ubuf` 按 `nLocalHeads` 个 head、块步长 `srcStride` 装载([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `copy_gm_to_ubuf(inputs[curr]` 搬入段);cache 模式且 remain>0 时整 head 装载(`fullHeadLoad`),否则只装 rope 区;
3. **freqs 搬入**:`freqs_gm = freqs + pos * ropeDim`([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `freqs_gm` 搬入段);
4. **升精度**:`vconv_f162f32` / `vconv_bf162f32` 把 rope 区升到 fp32([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `vconv_f162f32`/`vconv_bf162f32` 升精度段);`fullHeadLoad` 时同时把 remain 区 `copy_ubuf_to_ubuf` 暂存到 `vRemain`([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `copy_ubuf_to_ubuf(vRemain` 暂存段);
5. **计算**(纯向量指令,见下节);
6. **降精度写回**:非交错路径一次 `vconv_f322f16`/`vconv_f322bf16r` 转换全部头;交错路径在 `ropeDim == 64`(行与 head 对齐)且扁平表装得进剩余 UB 时(`flatGather`),一次多 repeat `vgather` 把整 token 重排为 `[r0,i0,r1,i1,...]` 后再一次 `vconv` 整体转换(替代每头 `vgather`+`vconv`+双 `pipe_barrier` 串行链,该链是 prefill 交错路径的主导开销);其余形态(ropeDim 非 64、表超 UB)退回逐头循环([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `vconv_f322f16`/`vgather` 降精度写回段);`fullHeadLoad` 时把 remain 区拷回输出缓冲;
7. **(可选)doRotate 缩放 + 写出**:`doRotate` 启用时先对旋转结果做 `vmuls(rotateScale)` 缩放——非 cache 路径(`doRotate && !need_v_cache`)直接对 fp32 rope 区缩放,cache 路径(`doRotate && fullHeadLoad`)对整 head(含 remain)做 `vconv→vmuls(rotateScale)→vconv` 缩放回原 dtype([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `doRotate` 缩放两段);随后写出([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 cache/output 写出段):cache 模式把整 head 写到 `v_cache + (slot / block_size) * MAX(vcacheStride, block_size*nLocalHeads*vdim) + (slot % block_size) * nLocalHeads * vdim`,若 `output_ptr` 非空再把 rope 区写到 output;非 cache 模式直接写 output。

### 关键计算步骤

全部在 fp32 域完成([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 fp32 计算段):

1. **奇偶分离**:`vreducev2` 以 step=1/2 分别抽取偶数 lane(`x0,x2,...` → real)与奇数 lane(`x1,x3,...` → imag)——同时用于 freqs 表(得到交错的 cos/sin 序列)与输入([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `vreducev2` 奇偶分离段);
2. **四组乘积**:`vmul` 计算 `x_even*cos`、`x_odd*sin`、`x_even*sin`、`x_odd*cos`(head 间块步长 `half_rope_blocks`,[csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中四组 `vmul` 乘积段);
3. **复数乘法合成**(`inverse` 二选一,[csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `inverse` 复数乘法合成段):
   - 正向:`real = x_even*cos - x_odd*sin`(`vsub`),`imag = x_even*sin + x_odd*cos`(`vadd`),其中 imag 写到 `inOutFP32 + ropeDim/2`;
   - 逆向:符号取反,即旋转 `-θ`。

交错输出路径在 kernel 启动时填一张 vgather 索引表,按 `flatGather` 是否成立分两种:`ropeDim == 64`(行与 head 对齐)且表装得进剩余 UB 时用扁平表——表项为每个输出槽位的绝对源字节偏移(含各 head 基址),标量单元只填第 0 行模板 `indices[2k] = k*4`(real[k])、`indices[2k+1] = (half+k)*4`(imag[k]),其余行各用一条 `vadds` 加上 head 基址(把每次 launch 的标量开销限制在 O(nLocalHeads),decode 不为填表买单),之后每 token 一次多 repeat `vgather` 把整 token 半分布局 `[r0..r31 | i0..i31]` 重排为 `[r0,i0,r1,i1,...]`;其余形态退回相对偏移表 + 每头一次 `vgather` 的逐头链([csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 中 `vgatherIndicesUB` 索引表填写与 `vgather` 重排两段)。

### 测试参考

- [tests/kernels/rope_complex.py](../../../tests/kernels/rope_complex.py):非 cache 路径(n_local_heads=64,inverse × interleaved × (rope_dim, q_dim) 组合),与 torch `view_as_complex` 复数乘对比;
- [tests/kernels/rope_complex_and_cache.py](../../../tests/kernels/rope_complex_and_cache.py):cache 路径(CXA `[remain|rope]` head_dim 512/128 与 MLA rope-only 64,n_local_heads=1),校验 vCache 逐 slot 与参考一致。

### 片上缓冲分配与占用汇总(AIV UB)

[csrc/kernels/rope_complex_and_cache.h](../../../csrc/kernels/rope_complex_and_cache.h) 的 `rope_complex_and_cache` 在 kernel 主体内以 `off` 偏移累进方式运行期排布全部 UB 缓冲,每个槽位按 `VECTOR_MAX_BYTESIZE=256B` 向上对齐,容量由 `assert(off <= UB_SIZE)` 守护。本算子为纯向量核,仅使用 UB,硬件容量基准(910B/dav C220,NpuArch 2201):UB 192KB([csrc/kernels/kernel_macro.h](../../../csrc/kernels/kernel_macro.h) 的 `UB_SIZE`)。dtype 记法:`Dtype` 为 input 模板实参(fp16/bf16,均 2B),UB 布局与 dtype 无关,单占用列。典型配置取 MLA Q-RoPE 的容量上界形状:ropeDim=64、nLocalHeads=64(每 token rope 元素数 64×64=4096)、非 cache 纯计算路径、半分输出(outInterleaved=false)、doRotate=false;cache 路径(nLocalHeads 断言为 1)占用远低,见要点。

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
| --- | --- | --- | --- | --- | --- |
| `input0`/`input1` | (nLocalHeads, ropeDim) = (64, 64)(非 cache 仅装 rope 区) | Dtype | 2(事件0/1) | `ROUND_UP(ropeDim*nLocalHeads*sizeof(Dtype),256B)` = 8192 B | 16384 |
| `freqs0`/`freqs1` | ropeDim 个 float | float | 2(事件2/3) | `ROUND_UP(ropeDim*4,256B)` = 256 B | 512 |
| `out0`/`out1` | (64, 64) | Dtype | 2(事件0/1) | 同 input | 16384 |
| `vRemain` | (vdim−ropeDim)×nLocalHeads 个 Dtype(cache 模式) | Dtype | 1 | `ROUND_UP(totalVRemainBytes,256B)` | 0(非 cache 不分配) |
| `inOutFP32` | ropeDim×nLocalHeads 个 float = 4096 | float | 1 | `ROUND_UP(ropeDim*nLocalHeads*4,256B)` = 16384 B | 16384 |
| `x_even` | 同上 | float | 1 | 同上 | 16384 |
| `x_odd` | 同上 | float | 1 | 同上 | 16384 |
| `cos` | 同上 | float | 1 | 同上 | 16384 |
| `sin` | 同上 | float | 1 | 同上 | 16384 |
| `x_even_cos` | 同上 | float | 1 | 同上 | 16384 |
| `x_odd_sin` | 同上 | float | 1 | 同上 | 16384 |
| `x_even_sin` | 同上 | float | 1 | 同上 | 16384 |
| `x_odd_cos` | 同上 | float | 1 | 同上 | 16384 |
| `positionUB` | 256 个 uint64 | uint64 | 1 | `ROUND_UP(256*8,256B)` = 2048 B | 2048 |
| `slotMappingUB` | 256 个 uint32(cache 模式) | uint32 | 1 | `ROUND_UP(256*4,256B)` = 1024 B | 0(非 cache 不分配) |
| 交错暂存 `interleavedFP32`(仅 outInterleaved) | ropeDim×nLocalHeads 个 float | float | - | 别名 `x_odd_cos`(rope 数学死区,转换点前已死) | 0(不额外占 UB) |
| `vgatherIndicesUB`(仅 outInterleaved) | flatGather 时 ropeDim×nLocalHeads、否则 ropeDim 个 uint32 | uint32 | 1 | `ROUND_UP(ropeDim*nLocalHeads*4,256B)`(flatGather)/`ROUND_UP(ropeDim*4,256B)`(逐头) | 0 |
| `rotateFP32`(仅 doRotate && fullHeadLoad) | vdim×nLocalHeads 个 float | float | 1 | `ROUND_UP(totalHeadFPBytes,256B)` | 0 |
| **合计** | | | | | **182784** |

**占用汇总**(vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用 | 占比 |
| --- | --- | --- | --- |
| UB(每 AIV) | 192 KB | 182784 B | 93.0% |

要点:

- **容量上界由非 cache 的 Q-RoPE 路径决定**:9 个 fp32 工作槽(`inOutFP32`、`x_even`/`x_odd`、`cos`/`sin`、四组乘积)合计 147456 B,占总占用的 80.7%,是绝对大头——复数旋转全程在 fp32 域展开的代价;
- **上界来源**:ropeDim=64、nLocalHeads=64(每 token 4096 个 rope 元素)时总占用 182784 B(178.5 KiB),正是 [csrc/kernels/kernel_param.h](../../../csrc/kernels/kernel_param.h) 的 `ROPE_COMPLEX_MAX_ELEMENTS_PER_TOKEN=4096`(注释「Conservative BF16 Q RoPE budget for 192 KiB UB (178.5 KiB at ropeDim=64)」)的出处;host 侧把多头 Q 按 `headsPerTile = MAX(1, ROPE_COMPLEX_MAX_ELEMENTS_PER_TOKEN/ropeHeadDim)` 切成不超过上界的 tile([csrc/model.cpp](../../../csrc/model.cpp) 的 `XModel::ForwardAttnMLAV2DsaCp`,关键字 `ROPE_COMPLEX_MAX_ELEMENTS_PER_TOKEN / _c.ropeHeadDim`),保证每 token rope 元素数 ≤ 4096;
- cache 路径占用远低于上界(nLocalHeads 断言为 1):CXA kv 写(vdim=512、ropeDim=128、outInterleaved=true)14080 B,MLA rope-only pe cache(vdim=ropeDim=64)6912 B;`outInterleaved` 输出的交错暂存别名到 rope 数学死区 `x_odd_cos`、不额外占 UB,额外仅 `vgatherIndicesUB` 一槽——`flatGather`(`ropeDim` 与 256B 行对齐且表装得进剩余 UB)时为 `ropeDim×nLocalHeads` 扁平表(上界配置 64×64 达 16KB、装不下即回退),否则为 `ropeDim` 项相对偏移表(上界配置 256 B);`doRotate && fullHeadLoad` 再加整 head 的 fp32 暂存;
- 容量守护仅 kernel 断言 `assert(off <= UB_SIZE)`(debug 构建生效),host 无等价校验,上界依赖上述切头约束。
