# hc_mean_collapse

## 功能概述

GLM-5.3-Flash MHC(Multi-Hyper-Connection)多流均值坍缩算子(无权重版),是 [`hc_pre`](hc_pre.md) 的无权重对应物:GLM-5.3 的 head 没有 `hc_head` 参数,因此对 `hcMult` 路残差流做普通的逐元素均值(而非 `pre` 加权和)。MHC 层输出的残差 `[m, hcMult*hidden]` 由 ConcatCol/ForwardHcPre 沿列方向拼接了 `hcMult` 份 `hidden` 长的流,本算子将其坍缩回一份:

```text
y[m, j] = (1 / hcMult) * Σ_{h=0}^{hcMult-1} x[m, h*hidden + j]
```

累加在 fp32 中完成(与 PyTorch bf16 `.mean()` 内部上转型为 fp32 的行为一致,见 `tests/kernels/hc_mean_collapse.py` 的参考实现 `x.view(m, hc_mult, hidden).float().mean(dim=1)`),最后转回输入 dtype。每流缩放因子 `1/hcMult` 被折叠进 vaxpy 累加器,无需单独的 vmuls 遍。

## 输入输出参数

Python 侧调用:`hc_mean_collapse(rt, input, output, m, hc_mult, hidden)`(`xlite/_C.pyi`),host 侧封装 `XliteOpHcMeanCollapse`(`csrc/op.cpp` 的 `XliteOpHcMeanCollapse`)。`invHcMult` 不出现在 Python 签名中,由 host 计算为 `1.0f / hcMult` 后传入(CCE 不允许 aicore 内做 uint→float 转换)。

kernel 签名(模板 `csrc/kernels/hc_mean_collapse.h` 的 `hc_mean_collapse`,入口宏 `HC_MEAN_COLLAPSE_FUNC_DEFINE` 导出 `hc_mean_collapse_bfloat16_t` / `hc_mean_collapse_float16_t`):

```cpp
hc_mean_collapse_<dtype>(GM_ADDR xResid, GM_ADDR yOut, uint32_t m, uint32_t hcMult, uint32_t hidden, float invHcMult)
```

记 `K = hcMult`、`D = hidden`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| xResid(input) | 输入 | `[m, K*D]` | BF16/FP16 | K 路残差流按列拼接;流 h 位于偏移 `h*hidden` |
| yOut(output) | 输出 | `[m, D]` | BF16/FP16 | 均值坍缩结果(非原位,独立输出张量) |
| m | 标量 | - | uint32 | token 数 |
| hcMult | 标量 | - | uint32 | MHC 流数 K |
| hidden | 标量 | - | uint32 | 每流特征维 D,**必须为 64 的倍数**(见下) |
| invHcMult | 标量 | - | float | host 预计算的 `1/hcMult` |

host 侧约束(`csrc/op.cpp` 的 `XliteOpHcMeanCollapse`,fail-closed):

- 维度合法性:`m`/`hcMult`/`hidden` 均不得为 0;`input.shape == [m, hcMult*hidden]`、`output.shape == [m, hidden]`;
- `hidden % 64 != 0` 直接抛错:kernel 内 Dtype→fp32 转换按 64 元素为一个 repeat 单元推进,不处理 hidden 余数,非 64 倍数的 hidden 会写越界 UB 缓冲(支持任意 hidden 需 kernel 侧尾部修复,留作后续);
- **UB 192KB 预算校验**:host 按 kernel 的 UB 分配(inDtype + outDtype + xFp32 + yCalc,各段按 `UB_BUF_ALIGN_SIZE=32B` 向上取整)镜像计算 `ubNeed`,超过 `kUbBytes = 192*1024`(kernel_macro.h NpuArch 2201 的 `UB_SIZE`)即抛错——kernel 侧 `assert` 在 release 编译下不可靠,故 host 侧 fail-closed 兜底;
- dtype 校验:input/output dtype 必须一致,且仅接受 BF16/FP16(fp32 hidden state 不是受支持的运行配置,`convert_input`/`convert_output` 也没有 float 特化);
- dummy runtime 或空张量直接返回;block 数取 `PickMinBlockNum(rt.aivNum, m)`;binding 同步。

## 支持的数据类型

- 模板 `csrc/kernels/hc_mean_collapse.h`(模板参数 Dtype 同时作用于输入与输出)。
- 实例化文件:
    - `csrc/kernels/hc_mean_collapse_bfloat16_t.cpp` → `hc_mean_collapse_bfloat16_t`(BF16);
    - `csrc/kernels/hc_mean_collapse_float16_t.cpp` → `hc_mean_collapse_float16_t`(FP16)。
- 累加类型固定 FP32;无 fp32 输入输出变体。
- kernel 整体用 `#ifdef __DAV_C220_VEC__` 保护,非 C220 向量核平台编译为空实现(`csrc/kernels/hc_mean_collapse.h` 的 `#else` 分支 `HC_MEAN_COLLAPSE_FUNC_DEFINE`)。

## 实现原理

单 AIV 向量核 kernel,`m` 个 token 按 `process = GetBlockIdx(); process < m; process += GetBlockNum()` 网格切分(`csrc/kernels/hc_mean_collapse.h` 的 `hc_mean_collapse`,关键字 `process = GetBlockIdx`),每个 token 的整条 `hcMult*hidden` 行一次性驻留 UB,单遍完成载入—转换—累加—写回。

### 片上缓冲分配与占用汇总(AIV UB)

`csrc/kernels/hc_mean_collapse.h` 的 `hc_mean_collapse`(关键字 `inDtype`/`outDtype`/`xFp32`/`yCalc` 的 UB 分配段)从 UB 偏移 0 依次分配 4 段,每段按 `UB_BUF_ALIGN_SIZE=32B` 向上取整,`assert(off <= UB_SIZE)` 收尾。典型配置取 GLM-5.3-Flash 的 `hcMult=4, hidden=4096`;BF16 与 FP16 的 `sizeof(Dtype)` 均为 2B,占用相同。

**AIV UB(容量 192KB,每 block 一份)**

| 缓冲 | 内容 | shape(每份) | dtype | 乒乓个数 | 单份大小(字节) | 典型配置占用(B) |
|---|---|---|---|---|---|---|
| `inDtype` | 输入残差 | `[hcMult, hidden]` | Dtype(2B) | 1 | `ROUND_UP(hcMult*hidden*2, 32)` | 32768 |
| `outDtype` | 输出行 | `[hidden]` | Dtype(2B) | 1 | `ROUND_UP(hidden*2, 32)` | 8192 |
| `xFp32` | 残差 fp32 影子 | `[hcMult, hidden]` | float | 1 | `ROUND_UP(hcMult*hidden*4, 32)` | 65536 |
| `yCalc` | fp32 累加器 | `[hidden]` | float | 1 | `ROUND_UP(hidden*4, 32)` | 16384 |
| **合计** | | | | | | **122880** |

**占用汇总**(字节数):

| 存储级 | 硬件容量 | BF16/FP16 占用 | 占比 |
|---|---|---|---|
| UB(每 block) | 192 KB | 122880 B | 62.5% |

要点:

- 布局对 `hcMult`/`hidden` 是编译期公式、运行期定长,单缓冲无乒乓,占用不随 `m` 变化;
- `xFp32`(65536 B)与 `inDtype`(32768 B)是占用大头,二者合计 98304 B,占 UB 的 50.0%;
- `hidden % 64 == 0` 由 host 保证;UB 预算由 host `HcMeanUbNeed`(同样按 32B 对齐)在 launch 前 fail-closed 校验;
- kernel 只注册 BF16/FP16,没有 FP32 输入输出变体,故不列 FP32 占用列。

### 计算步骤

每 token 循环体(`csrc/kernels/hc_mean_collapse.h` 的 `hc_mean_collapse`):

1. **MTE2 载入**:逐 h 把 `hcMult` 路残差行从 GM 拷入 `inDtype`(`CopyGmToUbufAligned`,每段 `hidden*sizeof(Dtype)` 字节),`pipe_barrier(PIPE_MTE2)` 后 `set/wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)` 交接给 V;
2. **Dtype→fp32 转换**:`convert_input` 分块执行——以 `calcPad = VECTOR_MAX_BYTESIZE / sizeof(float)`(每满向量 fp32 元素数)为单元,`totalRep = hcMult * vecRep`(`vecRep = DIV_ROUND_UP(hidden, calcPad)`)个 repeat 按 `VECTOR_MAX_REPEAT` 上限切 chunk,写入 `xFp32`;
3. **fp32 均值累加**:`vector_dup(yCalc, 0.0f, vecRep, ...)` 清零累加器,随后对每路 h 执行 `vaxpy(yCalc, xFp32 + h*hidden, invHcMult, vecRep, 1, 1, 8, 8)`——即 `yCalc += invHcMult * xFp32[h]`,`1/hcMult` 的缩放被折叠进 vaxpy 系数,K 次 vaxpy 后 `yCalc` 即均值,无需独立 vmuls 遍;
4. **fp32→Dtype 转换与写回**:`convert_output(outDtype, yCalc, vecRep)` 转回 Dtype,`set/wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)` 后 `CopyUbufToGmAligned` 把 `outDtype` 存入 GM 输出行 `yRow`。

### 流水线同步(自排空,免尾排)

采用"每轮 set 紧跟同轮 wait"的自包含同步对(镜像 `indexer_kcache_store_glm5`),无任何标志跨迭代残留,因此对任意 `m >= 1`(包括 ping-pong 尾排会死锁的 warmup 小 batch)都免挂死,不需要尾排 drain:

- **迭代内两对**:`MTE2→V`(保护 V 读 `inDtype`,在步骤 1 末)与 `V→MTE3`(保护 MTE3 存 `outDtype`,在步骤 4 中);
- **跨迭代两对**(每轮末尾):`MTE3→MTE2` 保护下一轮的 `inDtype` 载入、`MTE3→V` 保护下一轮的 `outDtype` 写入——均等价于"下一轮使用缓冲前等上一轮 MTE3 写出完成"。

## 关键代码位置

- 模板入口:`csrc/kernels/hc_mean_collapse.h` 的 `hc_mean_collapse` 模板函数
- 导出宏(BF16/FP16 与非向量核空实现):`csrc/kernels/hc_mean_collapse.h` 的 `HC_MEAN_COLLAPSE_FUNC_DEFINE`
- UB 分配与预算:`csrc/kernels/hc_mean_collapse.h` 的 `hc_mean_collapse`(关键字 `inDtype`/`outDtype`/`xFp32`/`yCalc`)
- vaxpy 均值累加:`csrc/kernels/hc_mean_collapse.h` 的 `hc_mean_collapse`(关键字 `vaxpy(yCalc, xFp32 + h * hidden, invHcMult`)
- host 封装与约束:`csrc/op.cpp` 的 `XliteOpHcMeanCollapse`
