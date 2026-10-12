# hc_pre

## 功能概述

DeepSeek-V4 Hyper-Connection 的 pre-activation **pre sum** 算子(非融合版),即 [`hc_act`](hc_act.md) 融合路径中 Step-5 pre-sum 的拆分对应物。它消费由 [`hc_split_sinkhorn`](hc_act.md)(=`hc_act` `preSum=0`)写到 GM 的 `pre` 门,把 K 份残差按 `pre` 加权求和成一份:

```
y[m, hidden] = Σ_h pre[m, h] * xResid[m, h, hidden]      # fp32 累加, bf16 输出
```

与 `tests/kernels/hc_pre_sum.py` 中参考实现 `torch.sum(pre.unsqueeze(-1) * x.float(), dim=2)` 一致(亦见 `tests/models/deepseek_v4.py` 的 `hc_pre`)。在拆分模型路径里,门控阶段由 `hc_split_sinkhorn` 完成、pre-sum 阶段由本算子完成;融合路径则由 `hc_act(preSum=1)` 一步完成,二者数学等价(`tests/kernels/hc_pre.py` 覆盖该融合端到端路径)。

## 输入输出参数

Python 侧调用:`hc_pre(rt, x_resid, pre, output, m, hc_mult, hidden)`(`xlite/_C.pyi`),host 侧封装 `XliteOpHcPre`(`csrc/op.cpp:2366`),以 `hc_pre_bfloat16_t` 启动(`csrc/op.cpp:2377-2378`)。

kernel 签名(模板 `csrc/kernels/hc_pre.h:13-15`,入口宏 `csrc/kernels/hc_pre.h:113-119`):

```cpp
hc_pre_bfloat16_t(GM_ADDR pre, GM_ADDR xResid, GM_ADDR yOut, uint32_t m, uint32_t hcMult, uint32_t hidden)
```

记 `K = hcMult`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| pre | 输入 | `[m, K]` | FP32 | pre 门(`hc_split_sinkhorn` 写到 GM 的输出) |
| xResid | 输入 | `[m, K, hidden]` | BF16 | 未归一化残差(pre-sum 输入) |
| yOut | 输出 | `[m, hidden]` | BF16 | pre-sum 结果 `Σ_h pre[h]*xResid[:,h,:]`(in-place 写出) |
| m | 标量 | - | uint32 | token 数 n |
| hcMult | 标量 | - | uint32 | Hyper-Connection 流数 K |
| hidden | 标量 | - | uint32 | 每流特征维 D |

host 校验 `xResid`/`yOut` 为 BF16、`pre` 为 FP32(`csrc/op.cpp:2373-2376`)。

## 支持的数据类型

- 模板 `csrc/kernels/hc_pre.h`(模板参数 Dtype 只影响 `xResid`/`yOut`)。
- 实例化文件 `csrc/kernels/hc_pre_bfloat16_t.cpp`,导出 `hc_pre_bfloat16_t`,固定 `Dtype = bfloat16_t`,即 **pre FP32、残差/输出 BF16、累加 FP32** 的单一组合。
- kernel 整体用 `#ifdef __DAV_C220_VEC__` 保护,非 C220 向量核平台编译为空实现(`csrc/kernels/hc_pre.h:120-126`)。

## 实现原理

单 AIV 向量核 kernel,`m` 个 token 按 `process = block_idx; process < m; process += block_num` 网格切分(`csrc/kernels/hc_pre.h:66`),每 token 全流程在 UB 内完成,双缓冲(`curr = 0/1`)流水重叠 MTE2 载入与 V 计算。

### UB 布局

`csrc/kernels/hc_pre.h:31-53` 依次分配:`inDtype0/1`(残差 bf16 双缓冲,`[K, hidden]`)、`outDtype0/1`(y bf16 双缓冲,`[hidden]`)、`preStage0/1`(pre fp32 双缓冲)、`preCalc`(ub2ub 中转目标,供标量核读取)、`xFp32`([K, hidden] 残差 fp32)、`yCalc`(累加器)。pre 经 MTE2 载入到 `preStage`,再由 `copy_ubuf_to_ubuf`(PIPE_V)中转到 `preCalc`,最后由标量核逐元素读取——这条 ub2ub 中转是让标量核安全访问 vector 写入值的关键(`copy_ubuf_to_ubuf` 是 PIPE_V 指令)。

### 计算步骤

每 token(`csrc/kernels/hc_pre.h:70-105`):

1. MTE2 载入 `preStage[curr]`(K 个 fp32)与 `inDtypeArr[curr]`(K×hidden bf16 残差,逐 h 载入,`csrc/kernels/hc_pre.h:71-74`);
2. `copy_ubuf_to_ubuf(preCalc, preStage[curr])` 把 pre 中转到标量可读区(`csrc/kernels/hc_pre.h:78`);
3. bf16 残差整块转 fp32(`convert_input`,按 `VECTOR_MAX_REPEAT`(255)次 repeat 分块,`csrc/kernels/hc_pre.h:79-83`),`vector_dup(yCalc, 0)` 清零累加器;
4. `pipe_barrier(PIPE_V)` 排空上述 V 操作,随后 `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0+curr)` 释放 mixesUb 给下一轮 MTE2(`csrc/kernels/hc_pre.h:85-86`);
5. 经 `set_flag/wait_flag(PIPE_V, PIPE_S, EVENT_ID4)` 让 V→S 可见(`csrc/kernels/hc_pre.h:88-89`),标量核从 `preCalc[h]` 逐个读出 `preH`,对 fp32 残差做 `vaxpy(yCalc, xFp32 + h*hidden, preH, vecRep)` 累加 K 次(`csrc/kernels/hc_pre.h:90-94`);
6. `convert_output` 把 `yCalc` 转 bf16,MTE3 写 `yRow`(`csrc/kernels/hc_pre.h:97-103`)。

### 流水线同步

`EVENT_ID0/1+curr` 管理 pre+残差载入与 V 计算的交接(`csrc/kernels/hc_pre.h:70-76`、`csrc/kernels/hc_pre.h:86`),`EVENT_ID2/3+curr` 管理 y 的 bf16 转换与写出(`csrc/kernels/hc_pre.h:97-103`),`EVENT_ID4` 做 V→S 事件往返让标量核安全读取 `preCalc`(`csrc/kernels/hc_pre.h:88-89`)。收尾 wait 与启动 set 一一对应(`csrc/kernels/hc_pre.h:107-110`)。每轮 `curr = 1 - curr` 翻转缓冲。

## 关键代码位置

- 模板入口:`csrc/kernels/hc_pre.h:13`
- `hc_pre_bfloat16_t` 导出宏:`csrc/kernels/hc_pre.h:113`
- token 循环:`csrc/kernels/hc_pre.h:66`
- ub2ub pre 中转:`csrc/kernels/hc_pre.h:78`
- vaxpy pre-sum:`csrc/kernels/hc_pre.h:90`
- host 封装 `XliteOpHcPre`:`csrc/op.cpp:2366`
