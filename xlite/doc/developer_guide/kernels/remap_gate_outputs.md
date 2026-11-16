# remap_gate_outputs

## 功能概述

EPLB(Expert-Level Load Balancing)场景下,将 `ForwardMoEGate` 产出的门控输出从**逻辑专家列**散列(scatter)到**物理专家列**的融合算子。一次 AIV kernel 启动同时完成两件事:

- **权重散列**:`wOut[t, log2phy[L]] = wIn[t, L]`(BF16/FP32,按字节拷贝,dtype 无关)
- **路由位散列**:若 `rIn` 中位 `(t, L)` 置位,则置位 `rOut` 中 `(t, log2phy[L])`(BIT1 位图,test-set)

`log2phy` 是本 rank 上逻辑→物理槽位的双射,因此散列无冲突。`rOut` 每行在 UB 内先清零再 bit-set(`bitmapSet` 是 OR 语义);`rIn` 的填充行(`[m, M)`)已被 `ForwardMoEGate` 置零,故 `rOut` 填充行自然为零。模型层在 `log2phy` 缺失(非 EPLB)时跳过本算子,零开销。

伪代码:

```text
for t in [0, M):              # 多 block 轮询
    for L in [0, nRoutedExperts):
        wOut[t, log2phy[L]] = wIn[t, L]            # 字节拷贝
        if bit_test(rIn[t], L): bit_set(rOut[t], log2phy[L])
```

## 输入输出参数

Python 侧调用(`xlite/_C.pyi`,`remap_gate_outputs`):

```python
remap_gate_outputs(rt, w_in, r_in, w_out, r_out, log2phy)
```

host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpRemapGateOutputs`(以 `PickMinBlockNum(rt.aivNum, M)` 个 block 启动,`elemBytes = XDtypeBit(wIn.dtype) / 8`)。kernel 签名(`csrc/kernels/remap_gate_outputs.cpp` 的 `remap_gate_outputs` 入口):

```cpp
remap_gate_outputs(GM_ADDR wIn, GM_ADDR rIn, GM_ADDR wOut, GM_ADDR rOut,
                   GM_ADDR log2phy, uint32_t M, uint32_t nRoutedExperts,
                   uint32_t totalPhysicalExperts, uint32_t elemBytes)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| wIn | 输入 | `[M, nRoutedExperts]` | bf16 / fp32 | 逻辑门控权重;按 `elemBytes` 字节拷贝,dtype 无关 |
| rIn | 输入 | `[M, ceil(nRoutedExperts/8)]` | BIT1 | 逻辑路由位图(逐 expert 一 bit),padding 行 `[m, M)` 已置零 |
| wOut | 输出 | `[M, totalPhysicalExperts]` | 与 wIn 同 | 物理门控权重;每行在 UB 内先清零再散列 |
| rOut | 输出 | `[M, ceil(totalPhysicalExperts/8)]` | BIT1 | 物理路由位图;每行在 UB 内先清零再 bit-set |
| log2phy | 输入 | `[nRoutedExperts]` | uint32 | 逻辑→物理槽位双射(本 rank) |
| M | — | 标量 | uint32 | token 数(`wIn.shape[0]`) |
| nRoutedExperts | — | 标量 | uint32 | 逻辑专家数(`wIn.shape[1]`) |
| totalPhysicalExperts | — | 标量 | uint32 | 物理专家数(`wOut.shape[1]`) |
| elemBytes | — | 标量 | uint32 | 单个权重元素字节数(host 由 dtype 推断) |

`M == 0 || nRoutedExperts == 0 || totalPhysicalExperts == 0 || elemBytes == 0` 时 kernel 直接返回(`csrc/kernels/remap_gate_outputs.cpp` 的 `remap_gate_outputs_kernel`,关键字 `if (M == 0 || nRoutedExperts == 0`)。

## 支持的数据类型

本算子为单文件 `csrc/kernels/remap_gate_outputs.cpp`(非模板),通过 `elemBytes` 标量对任意元素宽度的 `wIn/wOut` 做字节级拷贝,因此**无 dtype 实例化文件**。host 侧约束(`csrc/op.cpp` 的 `XliteOpRemapGateOutputs`):`wIn`/`wOut` 为 BF16 或 FP32(`XDtypeBit` 推断 `elemBytes`),`rIn`/`rOut` 为 BIT1,`log2phy` 为 uint32。kernel 仅在 `__DAV_C220_VEC__` 下编译,任务类型 `KERNEL_TYPE_MIX_AIV_1_0`(`csrc/kernels/remap_gate_outputs.cpp` 的 `remap_gate_outputs` 入口,关键字 `KERNEL_TYPE_MIX_AIV_1_0`,纯 AIV)。

## 实现原理

### 数据流与并行

核心函数 `remap_gate_outputs_kernel`(`csrc/kernels/remap_gate_outputs.cpp` 的 `remap_gate_outputs_kernel`)。token 维度按 block 轮询切分:`for (t = block_idx; t < M; t += block_num)`,每个 block 独立处理若干 token 行,无块间依赖。每行内串行执行"权重散列 → 路由位散列"两个阶段。

### UB 内存布局

UB 分五段,每段向上取整到 `VECTOR_MAX_BYTESIZE`(256B),使 `vector_dup` 清零(单次 256B repeat)不会越界写相邻段(与 `permutation.cpp` 同模式):

```text
log2phy | wInRow | wOutRow | rInRow | rOutRow
```

`log2phy` 在循环前由 MTE2 一次性搬入(`nRoutedExperts * 4B`,小),随后 `pipe_barrier(PIPE_ALL)` 保证落 UB,供 S 管线标量在循环内读取(`csrc/kernels/remap_gate_outputs.cpp` 的 `remap_gate_outputs_kernel`,关键字 `CopyGmToUbufAligned((__ubuf__ uint8_t *)log2phyUbuf`、`pipe_barrier(PIPE_ALL)` 之后的首次)。

### 权重散列

每行:MTE2 搬入 `wInRow`(`wInRowBytes = nRoutedExperts*elemBytes`),`vector_dup` 清零 `wOutRow`(`wOutRowBytes = totalPhysicalExperts*elemBytes`),`pipe_barrier(PIPE_ALL)` 后由 **S 管线标量**逐 expert 拷贝 `elemBytes` 字节:`dst = wOutRow + phy*elemBytes`、`src = wInRow + L*elemBytes`(`csrc/kernels/remap_gate_outputs.cpp` 的 `remap_gate_outputs_kernel`,关键字 `dst = wOutRow + (uint64_t)phy * elemBytes`、`src = wInRow + (uint64_t)L * elemBytes`)。完成后 `pipe_barrier(PIPE_ALL)` 再 MTE3 写回 GM。

### 路由位散列

每行:MTE2 搬入 `rInRow`(`ceil(nRoutedExperts/8)` 字节),`vector_dup` 清零 `rOutRow`,`pipe_barrier(PIPE_ALL)` 后用 `bitmapTest`/`bitmapSet`(按 `uint64_t` 行指针)做 test-set:`if (bitmapTest(rInRow64, L)) bitmapSet(rOutRow64, log2phy[L])`(`csrc/kernels/remap_gate_outputs.cpp` 的 `remap_gate_outputs_kernel`,关键字 `bitmapTest(rInRow64, L)`、`bitmapSet(rOutRow64, log2phyUbuf[L])`)。完成后写回 GM。

### 流水线同步

各 token 阶段以 `pipe_barrier(PIPE_ALL)` 串行化:MTE2 搬入、V 清零、S 标量散列、MTE3 写回四类管线两两依赖(散列读 wInRow/rInRow、读写 wOutRow/rOutRow),用全屏障保证正确性(比 `set_flag`/`wait_flag` 配对更重但实现简单,token 数有限,开销可忽略)。

### 模型层位置

`ForwardMoEGate` 之后、`ForwardMoEDispatch` 之前(`csrc/model.cpp` 的 EPLB 路径):仅当 `log2phy[layer]` 非空时执行,产出 `wPhy`/`rPhy` 供 Dispatch 按物理槽位索引权重;非 EPLB 时直接透传,零开销。

### 片上缓冲分配与占用汇总(AIV UB)

`remap_gate_outputs_kernel`(`csrc/kernels/remap_gate_outputs.cpp`)无 Init,UB 五段就地排布:`log2phy | wInRow | wOutRow | rInRow | rOutRow`,每段按 `VECTOR_MAX_BYTESIZE`(256B)向上取整(垫片按槽位字节计),使 `vector_dup` 清零的单次 256B repeat 不越界写相邻段(与 `permutation.cpp` 同模式)。dtype 记法:kernel 按 `elemBytes`(host 侧由 dtype 推断,BF16=2、FP32=4)做字节级拷贝,五段布局公式相同、仅 `wInRow`/`wOutRow` 两段随 elemBytes 缩放,给 BF16/FP32 两列占用。硬件容量基准(910B/dav C220):UB 192KB(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`)。典型模型配置:GLM-5.x 类 EPLB,nRoutedExperts=160、numRedundantExperts=32(totalPhysicalExperts=192)、BF16 权重;FP32 列同取 160/192 配置,占用差异仅来自 elemBytes。

**AIV UB(容量 192KB,每 block 一份)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | BF16 占用(B) | FP32 占用(B) |
|---|---|---|---|---|---|---|
| `log2phyUbuf` | nRoutedExperts | uint32 | 1(串行,事件 EVENT_ID0) | `ROUND_UP(nRoutedExperts*4, VECTOR_MAX_BYTESIZE)` B | 768 | 768 |
| `wInRow` | nRoutedExperts×elemBytes 字节 | uint8(字节视角) | 1(串行,事件 EVENT_ID0) | `ROUND_UP(nRoutedExperts*elemBytes, VECTOR_MAX_BYTESIZE)` B | 512 | 768 |
| `wOutRow` | totalPhysicalExperts×elemBytes 字节 | uint8(字节视角) | 1(串行,事件 EVENT_ID0) | `ROUND_UP(totalPhysicalExperts*elemBytes, VECTOR_MAX_BYTESIZE)` B | 512 | 768 |
| `rInRow` | DIV_ROUND_UP(nRoutedExperts, 8) 字节 | uint8(BIT1 行) | 1(串行,事件 EVENT_ID0) | `ROUND_UP(DIV_ROUND_UP(nRoutedExperts, 8), VECTOR_MAX_BYTESIZE)` B | 256 | 256 |
| `rOutRow` | DIV_ROUND_UP(totalPhysicalExperts, 8) 字节 | uint8(BIT1 行) | 1(串行,事件 EVENT_ID0) | `ROUND_UP(DIV_ROUND_UP(totalPhysicalExperts, 8), VECTOR_MAX_BYTESIZE)` B | 256 | 256 |
| **合计** | | | | | **2304** | **2816** |

**占用汇总**(字节数):

| 存储级 | 硬件容量 | BF16 占用 | 占比 | FP32 占用 | 占比 |
|---|---|---|---|---|---|
| UB(每 block) | 192 KB | 2304 B | 1.2% | 2816 B | 1.4% |

要点:

- 行级缓冲,占用仅约 1%——token 维按 block 轮询逐行处理,UB 只需容下一行权重、一行位图与 `log2phy` 映射表,容量完全不是瓶颈;
- `log2phyUbuf`+`wInRow`+`wOutRow` 三段合计 1792 B(BF16),约占 78%,是占用大头,随专家数线性增长,典型 160/192 规模远低于上限;
- 无 assert;唯一的"额外"开销是每段 256B 槽位垫片(`rInRow` 实际 20 字节按 256 字节槽位计),这是 `vector_dup` 清零不越界的实现代价(见上文 UB 内存布局段)。
