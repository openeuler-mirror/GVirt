# microbench（empty + scalar 依赖链 + 算子内计时 bench_timed + MTE 内存 bench_mte + 向量 bench_vec）

## 功能概述

微基准 kernel 家族，四个实验（`tests/performance/micro_bench.py` 按序执行），
覆盖 AIV / AIC / MIX 三种核型：

1. **实验 1 空算子**（`empty_{aiv,aic,mix}`）：无任何计算/搬运/同步的 kernel，
   测量一次 launch 的 device 侧固有开销（下发-调度-执行-返回），即该核型下任何
   真实算子都无法低于的单次执行底线（launch floor）；
2. **实验 2 标量依赖链**（`scalar_{add,xor,mul,div,fadd,add_unroll4}_{aiv,aic}`）：
   kernel 内执行一条依赖链上的基础标量运算（S pipe），循环次数 `iters` 为运行时
   kernel 参数、Python 侧可任意指定而无需重编，测量单次标量运算的 device 侧耗时
   （含循环控制开销），用于分离"循环控制"与"运算本身"、对比 AIV/AIC 标量单元
   差异。其中 2c 为**算子内计时孪生**（`scalar_<op>_<aiv|aic>_timed`，统一入口
   `bench_timed`）：每个标量变体的 timed 孪生，在 kernel 内用 `GetSystemCycle()`
   读 SYS_CNT 打 3 个点，不依赖 profiler 直接得到 device 侧耗时（机制见"算子内
   计时"一节）。**仅在 `XLITE_BENCH_TIMED=ON` 构建中存在**——默认构建中 timed
   kernel、host 入口与 Python 绑定完全不编译（见"文件组织与实现原理"）；
3. **实验 3 MTE 内存微基准**（`mte_{read,write,copy,chase}_aiv`，统一入口
   `bench_mte`）：lmbench 风格的带宽/时延 vs size 测试，`bytes`（工作集大小）与
   `iters` 均为运行时参数、命令行 `--mte-sizes` 外部传入，计时同样走 SYS_CNT
   三槽打点（每 block 3 槽）。read/write/copy 测 MTE 搬运带宽，chase 测依赖链
   时延（lat_mem_rd 的忠实映射）；**仅在 `XLITE_BENCH_TIMED=ON` 构建中存在**
   （见"MTE 内存微基准"一节）。
4. **实验 4 向量指令微基准**（`vec_{add_f32,mul_f32,add_i32,add_f16,sqrt_f32,
   exp_f32}_aiv`，统一入口 `bench_vec`）：V pipe 纯向量吞吐 vs 向量长度的 UB
   驻留测试，`elems`（每条链的元素数）与 `iters` 为运行时参数、命令行
   `--vec-sizes` 外部传入，SYS_CNT 三槽打点。小向量看每条指令的固定开销摊销，
   大向量看向量单元饱和吞吐（ns/元素 vs elems 曲线）；sqrt/exp 走"载体差分"测
   超越函数边际成本（见"向量指令微基准"一节）；**仅在 `XLITE_BENCH_TIMED=ON`
   构建中存在**。

空算子的核型与 blockNum 语义：

| 算子 | kernel 类型标记 | 调度方式 | blockNum 语义 |
|------|----------------|----------|---------------|
| `empty_aiv` | `KERNEL_TYPE_AIV_ONLY` | 仅矢量核 | 矢量核 block 数 |
| `empty_aic` | `KERNEL_TYPE_AIC_ONLY` | 仅 Cube 核 | Cube 核 block 数 |
| `empty_mix` | `KERNEL_TYPE_MIX_AIC_1_2` | Cube+矢量单一二进制，硬件按 1 AIC : 2 AIV 配比调度（同 indexer_topk/cxa/flash_attention 的 MIX 形态） | Cube 核 block 数，每个 Cube block 由硬件配 2 个矢量核 block |

标量变体（`iters` 均按**总运算次数**计，`add_unroll4` 内部除以 4，各变体 ns/op
可直接对比；纯标量运算无 Cube/矢量数据通路成分，无 MIX 变体）：

| 算子 | 循环体（每迭代） | Dtype | 测量目标 |
|------|-----------------|-------|----------|
| `scalar_add` | `acc = acc + (x ^ i)` | uint32 | add（操作数与计数器纠缠，见下） |
| `scalar_xor` | `acc = acc ^ (x ^ i)` | uint32 | xor |
| `scalar_add_unroll4` | 4 × `acc = acc + (x ^ (4t+k))` | uint32 | add 展开 4 次/trip |
| `scalar_mul` | `acc = acc * x` | uint32 | 乘 |
| `scalar_div` | `acc = acc / x` | uint32 | 整除 |
| `scalar_fadd` | `acc = acc + x` | float | fp32 加 |

## 文件组织与实现原理

"一处实现 + 每核型一个 .cpp 测试项清单"，timed/untimed 只有一套代码；两种构建
共用**同样的三个 kernel 源文件**（timed 构建不追加 TU，只在同一批 TU 内多实例
化 timed 家族）：

| 文件 | 职责 |
|------|------|
| `csrc/kernels/microbench.h` | **全部实现**：empty/scalar/MTE/向量四族生成宏与设计注释、每变体循环体（`BENCH_BODY_*`）、核型后缀→任务类型别名（`BENCH_TASK_*`）、测试项实例化宏（`BENCH_*(ct)`）与三个 timed 聚合宏（`BENCH_TIMED_ITEMS` / `BENCH_MTE_ITEMS` / `BENCH_VEC_ITEMS`：仅 `MICROBENCH_TIMED` 下展开为项、默认展开为空）；MTE/向量生成宏恒定义，`__DAV_C220_VEC__` 双 flavor 守卫 |
| `csrc/kernels/kernel_macro.h` | 通用 kernel 内 SYS_CNT 采样机制（`BENCH_TICK*` 宏，任意指令类可复用） |
| `csrc/kernels/bench_aiv.cpp` | AIV 测试项清单：7 条 plain 项 + 三行 timed 聚合宏（12 个 `*_timed` 孪生 + 4 条 MTE + 6 条向量；后两族仅 AIV——MTE 管线与向量 pipe 都在矢量核上，无 `_aic` 形态） |
| `csrc/kernels/bench_aic.cpp` | AIC 测试项清单：7 条 plain 项 + 一行 `BENCH_TIMED_ITEMS(aic)`（12 个 `*_timed` 孪生） |
| `csrc/kernels/bench_mix.cpp` | 单条 empty 测试项（纯标量无 MIX 变体，无 timed） |

`MICROBENCH_TIMED` 是**构建级**编译宏，由 CMake 选项 `XLITE_BENCH_TIMED` 控制
（默认 OFF）：

- **OFF（默认）**：三个 `*_ITEMS` 聚合宏全部展开为空，`*_timed` kernel、MTE /
  向量两家族 kernel、它们的 host stub、`XliteOpBenchTimed/Mte/Vec` 分发与
  Python `bench_timed/bench_mte/bench_vec` 入口**全部不存在**；三个 TU 与定标
  布局**逐 token 一致**（`g++ -E` 预处理对照验证），既有基准数据直接延续。
- **ON**（`cmake -B build -DXLITE_BENCH_TIMED=ON`）：同一批 .cpp 追加实例化 12 个
  `*_timed` 孪生（同函数体 + SYS_CNT 打点 + 额外 `GM_ADDR ticks` 参数，kernel
  名加 `_timed` 后缀），bench_aiv TU 同时追加 4 个 MTE 与 6 个向量 kernel
  （共 23 kernel）、bench_aic 变为 13 kernel；host/Python 侧出现 `bench_timed`、
  `bench_mte` 与 `bench_vec`。

宏的传递路径（CANN 机制）：`ascendc_compile_definitions(xlite_kernels_common
PRIVATE MICROBENCH_TIMED=1)` 把定义送进 ascendc 子工程的**全部四个阶段**——
precompile（stub 提取，从预处理产物里认出 timed kernel）、preprocess、aic/aiv
两个 device 通道；host 侧则由 `target_compile_definitions` 直接给 `xlite` 与 `_C`
两个目标。这确保 timed kernel 的 stub 提取与设备编译看到的是同一份宏语义。

timed kernel 名必须在**一次** `BENCH_CAT3(base, ct, _timed)` 粘贴中拼出、由生成宏
直接使用 `name`（microbench.h 内有注释）。原因：CANN 的 auto_gen 包装在
`#include` 测试项清单之前先 `#define scalar_add_aic scalar_add_aic_origin` 改名
基础名，若在生成宏内再补一次 `BENCH_CAT(name, _timed)` 粘贴，参数预扫描会先把
基础名替换成 `*_origin`、再把 `_timed` 粘上去，得到 `scalar_add_aic_origin_timed`
——与包装器调用的 `scalar_add_aic_timed_origin` 对不上，device 通道编译报
"use of undeclared identifier"（默认构建不受影响：timed 段整体被
`#if MICROBENCH_TIMED` 排除，预处理输出逐 token 不变）。

host 侧：`XliteOpEmptyAiv/Aic/Mix`（block 数收敛后 `aclrtlaunch_empty_*`）与 12
个 `XliteOpScalar*`；timed 家族由 `XliteOpBenchTimed` 按变体字符串
`"<op>_<aiv|aic>"` 统一分发（一个入口管全家族）；MTE 家族因 buffer 型签名独立为
`XliteOpBenchMte`，按 `"<op>_aiv"` 分发（MTE 在矢量核上，无 `_aic` 形态）；向量
家族同理由 `XliteOpBenchVec` 按 `"<variant>_aiv"` 分发。pybind
层对应 `empty_*/scalar_*`、`bench_timed(rt, variant, out, ticks, seed, iters)`、
`bench_mte(rt, variant, src, dst, ticks, bytes, iters, block_num)` 与
`bench_vec(rt, variant, src, out, ticks, elems, iters, block_num)`（后三者仅
timed 构建存在，`micro_bench.py` 实验 2c/3/4 自动检测并跳过）。

### TU（翻译单元）构成规则——两条测量有效性约束

1. **不同核型不能共用 TU**：ascendc precompile 对 kernel 源文件做**文件级**扫描，
   取文件内第一个 `KERNEL_TASK_TYPE_DEFAULT` 展开标记应用于该文件全部 kernel
   （CANN `extract_host_stub.py` 的 `find_kernel_type_by_source`）。同核型多
   kernel 共用一个 .cpp 是安全的（仓库惯例：all_reduce.cpp / all_gather.cpp）。
2. **默认构建的 TU 构成一经定标即冻结**：1 槽级整数运算暴露 TU 级代码生成上下文
   ——把 timed 孪生加进 bench_aiv.cpp 后，**未改动**的 `scalar_mul_aiv` 从 1.667
   漂移到 1.112 ns/op（3→2 发射槽）；延迟 bound 的 fadd/div 在全部布局变化下逐位
   不变。因此默认构建里 timed 项展开为空、TU 与定标布局逐 token 一致；timed 构建
   的 bench_aiv TU（23 kernel）是**另一套代码生成上下文，其数值自成基线**——
   timed 构建的 untimed/timed 读数都不可与默认构建对比（尤其 mul），切换构建后须
   重新定标；**timed 聚合宏内容的增删会改变 timed 构建的 TU 构成，须重新定标**
   （默认构建不受影响：聚合宏默认展开为空，冻结不变）。

### 测量有效性设计（缺一不可）

1. **anti-DCE**：累加结果在循环结束后直写 GM（`*(__gm__ Dtype*)out = acc` +
   `pipe_barrier(PIPE_ALL)`），循环不可被删除。
2. **anti-folding**：`seed` 为运行时参数只是第一步——**不够**。bisheng 会把不变
   操作数加法链 `acc += x` 重结合为闭式解 `acc + N*x`（实测：任意 iters 下
   device 恒为 ~1.7us，即循环根本没跑）。因此 add/xor/unroll4 的操作数与循环
   计数器纠缠（`x ^ i`）：IV 的异或无闭式解，链不可折叠。曾尝试空 inline-asm
   编译屏障并弃用：aicore 后端仅支持 `"l"` 约束（64 位操作数），强制的 64 位
   往返自带 ~6.7ns/iter 开销（mul 从 1.11 涨到 7.78 ns/op），会淹没被测量。
   mul/div/fp32 链无廉价闭式解，无需纠缠（实测线性）。
3. **链式语义**：每迭代消费上一迭代结果，测得的是**运算延迟 + 循环控制**，而
   非并行吞吐。
4. **双通道可编译**：common kernel 文件会被同时编入 aiv/aic 两个 device 通道，
   UB→GM 搬运类 intrinsic（`copy_ubuf_to_gm_align_b16`）仅 vec 通道支持（cube
   通道直接报错，CANN 自家 DataCopy 在 dav-c220 的 AIC 上也提前返回）。故结果
   回写用标量直存 GM：两通道均可编译，且其每 launch 一次的成本只进常数项
   （截距），不进每迭代斜率。

### 三方程分解

每核型三个未知数（b=循环控制、a=add、o=xor），三个变体联立：

```
s1 = scalar_add         = b + a + o
sx = scalar_xor         = b + 2*o
s4 = scalar_add_unroll4 = b/4 + a + o
=> b = 4*(s1-s4)/3,  o = (sx-b)/2,  a = s1-b-o
```

（实测 s1≈sx≈s4，即 b≈0：展开节省的循环控制恰好被 `4t+k` 的计数运算吃掉。）
`mul/div/fadd` 直接测得"运算+循环控制"之和。

## 输入输出参数

空算子 kernel 侧仅有一个占位参数（host 侧传 `nullptr`）：host stub 提取器要求
kernel 参数表非空，故保留一个不使用的 `GM_ADDR dummy`；可选参数以 nullptr 下发
是既有惯例（如 matmul 的 bias），不改变真实下发路径。

标量 kernel 签名（定义宏见 `csrc/kernels/microbench.h`）：

```cpp
name(GM_ADDR out, Dtype seed, uint32_t iters)          // untimed
name_timed(GM_ADDR out, GM_ADDR ticks, Dtype seed, uint32_t iters)  // timed
```

- `out`：1 元素输出张量（i32 或 f32），承载累加结果；
- `seed`：初值兼操作数，运行时传入，编译期不可知（防止常量折叠）；
- `iters`：循环次数，Python 侧控制——"外部控制循环次数"的入口；
- `ticks`（仅 timed）：≥3 元素 INT64 张量，**调用方预先分配**，kernel 批量回写。

Python 调用方式（`tests/performance/micro_bench.py`，四个实验共用一个入口；
`bench_timed` 仅 timed 构建可导入，脚本内用 `getattr` 探测）：

```python
from xlite._C import Runtime, empty_aiv, scalar_add_aiv
import xlite._C as _C
bench_timed = getattr(_C, "bench_timed", None)  # 默认构建为 None

rt = Runtime(0, 500)
out = torch.zeros(1, dtype=torch.int32, device="npu:0")

empty_aiv(rt)                                   # block_num=0 → 满核，返回实际 block 数
n = scalar_add_aiv(rt, out, seed=3, iters=1_000_000)   # 依赖链, block_num 默认 1
if bench_timed is not None:                     # XLITE_BENCH_TIMED=ON 构建
    ticks = torch.zeros(3, dtype=torch.int64, device="npu:0")
    bench_timed(rt, "add_aiv", out, ticks, seed=3, iters=1_000_000)
    t0, t1, t2 = ticks.cpu().tolist()           # SYS_CNT cycles, /50 = µs
```

host 侧校验：`out` 须为对应 dtype 的 1 元素张量；`div` 拒绝 seed=0；
`add_unroll4` 拒绝 iters 非 4 的倍数；timed 的 `ticks` 须为 ≥3 元素 INT64 张量、
variant 未知时报错。blockNum 语义：empty 族 0 → 满核（超上限收敛为满核）；标量
族 0 → 1（依赖链无法跨核并行，单核口径最清晰）。

## 算子内计时（bench_timed / GetSystemCycle）

`kernel_macro.h` 的 `BENCH_TICK*` 宏提供**通用的 kernel 内 SYS_CNT 采样机制**，
不限于标量：任何 MTE / vector / cube 微测试都可直接复用。

`GetSystemCycle()`（`kernel_operator_sys_var_intf.h`）读 SYS_CNT 系统计数器。
该计数器以**固定 50 MHz** 计数（**非核频**），`cycles / 50 = µs`，分辨率 20ns——
对聚合区域足够（1M 次标量运算 ≈ 83K cycles），单条指令须用 iters 斜率法。

三个采样点（`BENCH_TICK` 宏，值存寄存器数组）：

| 槽位 | 时刻 | 含义 |
|------|------|------|
| t0 | 测量区起点 | 循环开始前 |
| t1 | 循环下发完 | 最后一圈指令已 issue（S pipe 顺序） |
| t2 | **真实完成** | `pipe_barrier(PIPE_ALL)` 排空全部流水线后 |

两个关键设计：

1. **打点不进被测窗口**：ticks 张量由调用方**预先分配**，kernel 内采样值先存
   寄存器，测量区关闭（t2）之后才用标量直存循环**批量回写**——若每采一个点就
   写 GM，MTE3 store 会落进被测区间，污染打点。
2. **PIPE_ALL barrier 取完成时刻**：昇腾核上标量/矢量/Cube/MTE 流水线**并行**，
   指令"下发"≠"完成"。t1 只是标量 pipe 的下发完时刻；`pipe_barrier(PIPE_ALL)`
   等所有流水线排空后取 t2，才是测量区的真实完成时刻。`drain = t2 - t1` 即在飞
   指令的排空耗时——纯标量链 ≈0（S pipe 顺序执行），引入 MTE/vector/cube 工作
   后将变为非零，届时 drain 本身就是一个测量对象。

timed 孪生与 untimed 同源：同一 `microbench.h` 生成宏的两套定义（plain 恒编 +
timed 仅 `MICROBENCH_TIMED` 构建编译），函数体逐 token 一致，只是包了打点与
ticks 回写（`g++ -E` 预处理对照验证：默认构建 3 个 TU 与定标布局逐字节一致，
timed 构建的 12 个孪生函数体与历史 timed kernel 逐 token 一致）。

910B3 实测（@1M，5 次中位）：12 变体 in-kernel complete 与 profiler device 时间
逐变体差 0.9–1.1 µs（≈0.03%，恒定差 = kernel 前后缀）；**drain 全部为 0**——纯
标量链在 S pipe 顺序执行，t1 时无在飞指令，这是该机制的零点校准。timed
mul_aiv 曾测得 1.111（2 槽），为 TU 代码生成上下文差异（见 TU 规则），非硬件
差异。

## MTE 内存微基准（bench_mte）

lmbench 的访存测试映射到昇腾 MTE（Memory Transfer Engine）数据通路，4 个 kernel
共用一个签名（`variant` 挑选，`mte_<op>_aiv`）：

```cpp
name(GM_ADDR src, GM_ADDR dst, GM_ADDR ticks, uint64_t bytes, uint32_t iters)
```

| kernel | lmbench 对应 | 机制 | bytes / iters 语义 | blockNum |
|--------|--------------|------|-------------------|----------|
| `mte_read_aiv` | `bw_mem rd` | MTE2 GM→UB，96KB×2 双缓冲 tile 环，逐段**无同步**（MTE2 流水） | 每 rep 流过量 / rep 数 | 0→满核 AIV |
| `mte_write_aiv` | `bw_mem wr` | UB tile 一次性填充（**计时段外**完成握手）后 MTE3 流式写 GM | 同上 | 同上 |
| `mte_copy_aiv` | `bw_mem bcopy` | split.cpp 同款 MTE2↔MTE3 ping-pong（flag 对） | 同上（**字节只计一遍**，引擎侧实际读写各一遍） | 同上 |
| `mte_chase_aiv` | `lat_mem_rd` | 64B 块 GM→UB + MTE2→S 握手 + 读出下一偏移（依赖链，无法预取） | 工作集 / 追逐步数 | **恒钳 1** |

参数与校验（host 侧 `XliteOpBenchMte`）：

- `src`：INT8 工作集（read/copy/chase 需 `bytes` 元素；write 只读前
  `min(bytes, 96KB)` 的 pattern 字节）；`dst`：read/chase 为 INT32（≥16×blockNum
  元素，acc 字在元素 `16*block_idx`，即 64B 步距），write/copy 为 INT8（`bytes`
  元素，承载流写结果供核对）；
- `ticks`：INT64、≥8×blockNum 元素，**每 block 独占一条 64B GM line**（布局
  `ticks[byte 64*block_idx + {0,8,16}]`，即 int64 下标 `8*b + {0,1,2}`），kernel
  结束时批量回写——聚合时间取各 block complete 的**最大值**（并行执行，kernel
  完成于最慢 block）。64B 步距是硬件约束：**跨核标量 GM store 打进同一 64B line
  会整线互踩**（实测 910B3：紧排布局下 blockNum≥2 时约半数 per-block 字
  非确定丢失、按 line 全有/全无、永久丢失不因等待恢复；MTE3 store 不受影响，
  生产 kernel 的 GM store 全走 `copy_ubuf_to_gm` 即此因）；
- `bytes`：非零 32B 倍数（chase 额外 64B 倍数）；`iters` ≥1（chase 为追逐步数）；
- `blockNum`：chase 恒 1（依赖链无并行性）；其余 0→满核 AIV，各 block 以
  grid-stride 平分工作集（步长 = blockNum×96KB，尾段收缩）。

测量口径要点：

1. **drain 首次成为真实测量对象**：read/write/copy 的 issue（t1）只等搬运指令
   下发完，`PIPE_ALL` 排空后 t2 才是真实完成——drain = t2−t1 即 MTE 异步搬运的
   在飞排空时间（纯标量链 ≈0 作对照）。带宽分母必须用 complete 而非 issue。
2. **L2 驻留不刻意 flush**：带宽 sweep 中 size < L2 的档走 L2、更大的档落 HBM，
   聚合带宽在 L2 容量处出现拐点——这正是 lmbench 曲线的价值所在。
3. **chase 时延 = 经 MTE 的 load-to-use**：每步 = MTE2 拷贝下发 + GM 访问时延 +
   MTE2→S 握手 + 标量读，**含义与裸 HBM 时延不同**（这是该硬件经 MTE 取数的
   真实成本，文档与报告引用时须注明口径）。每步消费上一步读出的偏移，硬件无法
   预取/流水，是 lat_mem_rd 指针追逐的忠实映射；工作集为 Python 预构建的随机
   单一环置换（块首 u32 = 后继块偏移）。
4. **反 DCE/反折叠**：MTE 拷贝为不透明 DMA intrinsic 不可被消解（split.cpp 的
   load 从未被消费亦安全）；`micro_bench.py` 3b 仍做 iters 翻倍线性自检兜底。

双 flavor 编译（关键约束）：common kernel 文件会被**同时**编入 vec/cube 两个
preprocess flavor，MTE 搬运 intrinsic 仅 vec flavor 支持——MTE 生成宏按 dequant.h
式守卫：`#ifdef __DAV_C220_VEC__` 真体 / `#else` 同签名空体，保证两个 channel 的
符号集一致（cube 通道见到的是空 kernel）。

运行（timed 构建）：

```bash
python <xlite>/tests/performance/micro_bench.py           # 实验 3 自动跑默认 size 档
python <xlite>/tests/performance/micro_bench.py --mte-sizes 64k,256k,1m,4m,16m,64m,256m
python <xlite>/tests/performance/micro_bench.py --mte-only  # 只跑实验 3(MTE)
```

`--mte-sizes` 支持 `k/m/g` 后缀与十进制/0x（向下对齐 64B）；3a 数值核对 → 3b 反
消解 → 3c 带宽 sweep（size × {read,write,copy} × {单 block, 满核}，聚合/每核
GB/s）→ 3d 时延 sweep（chase ns/步 vs size）。`--analyze` 报告的 [8] 段给出
profiler 外部视角（与核内 complete 对照）。实测口径差（910B3，同跑逐调用配对）：
**单 block 安静态恒 +2µs**（kernel 前后缀，与 bench_timed <1µs 先例同量级，核内
GetSystemCycle 口径成立）；大额分配后的首次调用与满核配置会散发 +70~205µs 的
单向膨胀（device ≥ in-kernel 恒成立）——前者是 t0 前设备侧停顿计入整核寿命，后者
是跨 block 派发倾斜 + 收尾退休，属**口径差异**：3c/3d 的核内窗口测稳态每 block
搬运能力，整核端到端时间须以 profiler device_us 为准。

### 910B3 参考表（timed 构建，5 次中位）

> 2026-10-15 复测更新（bench_timed_aiv.cpp 并入 bench_aiv.cpp 后重取）：标量
> （实验 2c）与向量（实验 4）两族逐格复现（≤0.3%），MTE 族单核 read
> 210→230 GB/s、多 block 聚合档有 ±25% 量级环境漂移（见实验 1 一节）。

带宽（GB/s；B=1 单 block / 满核聚合，actB=活跃 block 数，copy 字节只计一遍）：

| size | read B=1 | read 满核 | write B=1 | write 满核 | copy B=1 | copy 满核 |
|---|---|---|---|---|---|---|
| 64k | 230.3 | 230.3 (actB=1) | 205.7 | 199.9 | 79.3 | 79.4 |
| 256k | 230.3 | 462.3 (actB=3) | 199.5 | 320.5 | 129.2 | 194.4 |
| 1m | 230.3 | 1551.3 (actB=11) | 196.7 | 1023.3 | 153.7 | 771.5 |
| 4m | 230.2 | 5866.2 (actB=43) | 204.3 | 3736.6 | 159.3 | 2791.6 |
| 16m | 230.0 | 5377.3 | 197.7 | 3687.3 | 165.3 | 3031.1 |
| 64m | 228.4 | 5500.7 | 199.8 | 3966.2 | 161.9 | 3375.7 |
| 256m | 139.2 | 1865.2 | 186.7 | 1490.1 | 100.1 | 623.7 |

chase 时延（ns/步，2048 步，单 block）：64k 119.2 / 256k 114.8 / 1m 118.4 /
4m 115.6 / 16m 117.6 / 64m 120.2 / 256m 119.8；drain 全 0。

要点：满核聚合 read 拐点在 64–256MiB 之间（5501→1865 GB/s，与 ~192MB L2 吻合）；
HBM 区（256m）单核 write 快于 read（187 vs 139），满核聚合方向两轮测量相反
（本轮 read 1865 > write 1490，上轮 1351 < 1928）——多 block 聚合档噪声大，
方向性结论以 B=1 列为准；read@256m 满核 drain 占 complete 的 53%（144µs 中
76µs）。详见 workspace 报告 `mte_mem_bw_lat_perf.md`。

**harness 侧两个易踩的坑**（micro_bench.py 已内置处理，自写调用方须知）：

1. `rt.stream` 是 `Runtime` 私有的 `aclrtCreateStream` 流，与 torch 当前流**互不
   排序**——launch 前必须 `torch.npu.synchronize()`，否则 kernel 与 torch 流上的
   H2D 上传/`zeros` 填充双向竞态（实测形态：copy 读到半落地 src 出现零尾、
   torch 填充晚于 kernel 落地覆写 dst、chase 链读到 0 原地打转，均间歇性）；
   micro_bench.py 的 `bench_mte()` 包装统一前置同步，调用点不得直连
   `_C.bench_mte`。
2. 输出张量按 64B/block 步距分配与解读（见上），紧排布局会被跨核同线 store
   互踩。

## 向量指令微基准（bench_vec）

V pipe 向量单元的纯吞吐 vs 向量长度测试。全部数据 UB 驻留（排除访存干扰），
6 个 kernel 共用一个签名（`variant` 挑选，`vec_<variant>_aiv`，仅 AIV——向量
intrinsic 不存在于 cube 通道）：

```cpp
name(GM_ADDR src, GM_ADDR out, GM_ADDR ticks, uint64_t elems, uint32_t iters)
```

| kernel | 循环体（每 chunk 256B） | 测量目标 |
|--------|------------------------|----------|
| `vec_add_f32_aiv` | `dst = dst + c`（vadd raw） | fp32 加向量吞吐 |
| `vec_mul_f32_aiv` | `dst = dst * c`（vmul raw） | fp32 乘（sqrt/exp 差分的基线） |
| `vec_add_i32_aiv` | `dst = dst + c`（int32 vadd raw） | 整数 vs 浮点同单元对比 |
| `vec_add_f16_aiv` | `dst = dst + c`（half vadd raw） | 16 位通道翻倍效应 |
| `vec_sqrt_f32_aiv` | `dst = dst * sqrt(c)`（载体差分） | vsqrt 边际成本 = 本变体 − mul_f32 |
| `vec_exp_f32_aiv` | `dst = dst * exp(c)`（载体差分） | vexp 边际成本 = 本变体 − mul_f32 |

**超越函数为什么用载体差分**：`dst = op(dst)` 直链不可行——exp 链几步内发散到
inf、sqrt 链收敛到 1.0，两者数值形态还不同。统一改为 `dst = dst * op(c)`：超越
函数跑在常量向量上（每迭代都是正常路径输入），依赖链由 mul 承载；
**cost(op) = compl(本变体) − compl(mul_f32 同 elems/iters)**（与标量基准的三方程
分解同思路）。常量选 op(c)=1（sqrt: c=1.0，exp: c=0.0）——载体乘后 dst 逐位不变，
数值核对退化为 out==x0 逐位相等，最强形式。差分口径注意：sqrt 不在链上、可与
mul 重叠，该差分是"在 mul 依赖链存在的前提下加一条超越指令"的边际成本，不是
孤立的 vsqrt 吞吐。`xor_u32` 原计划变体不可实现：本工具链无 raw `__ubuf__` vxor
（逻辑类 raw intrinsic 仅 16 位），host XDtype 又无 INT16，故以 add_i32 替代
（与 add_f32 等宽同形，整数/浮点同单元对比反而更干净）。

**⭐ 实测硬件发现——V pipe 相邻指令读写重叠（RAW 竞态）**：不插屏障时，
`dst = dst + c` 链 iters=2 的结果与 iters=1 **完全相同**（逐位等于 x0+c，即两次
加法只有一次生效）。向量 pipe 会把下一条指令的操作数读取与上一条在飞写入重叠
执行——相邻向量指令间存在真实 RAW 竞态，quant.h 里 vmul→`pipe_barrier(PIPE_V)`
→vadd 的习惯写法是**载荷性**的。修复：两个生成宏每 chunk 末尾各一条
`pipe_barrier(PIPE_V)`（二元：`[op; barrier]`；载体：`[op; barrier; mul]`，
每 chunk 恰一条屏障，与二元变体屏障数相同，差分不被屏障污染；vmul(k)→vmul(k+1)
跨迭代的 RAW 由下一条 vmul 前的屏障覆盖，vmul 与下一 chunk 的 op 间仅剩 tmp 的
WAR——c 全元素同值时值中性，7a 折叠核对兜底）。后果：小向量档测的是"依赖链上
一条向量指令 + 一条屏障"的节拍，不是无屏障背靠背发射；这本身是正确性所需，
写生产 kernel 时相邻向量指令同样绕不开。

参数与布局（host 侧 `XliteOpBenchVec`，与 micro_bench.py 三侧同步）：

- `src = [x0; c]` 拼接（各 `elems` 元素，dtype 按变体）：x0 为链初值（f32/f16
  取 [1,2)、i32 取 ±2^30 内），c 为常量向量（全元素同值；add_f32 2⁻¹³ /
  mul_f32 1+2⁻¹² / add_f16 2⁻¹⁰ / add_i32 3 / sqrt 1.0 / exp 0.0）；
- `out`：≥ blockNum×elems 元素，每 block 独立跑同一条链、结果写 `out +
  block_idx*elems`；`ticks` 与 MTE 家族共用 64B/block 独占 line 布局；
- `elems`：非零 256B/dtype_size 的整数倍（整 repeat、mask 恒满），且
  `2*elems*sz + 65280(载体 scratch) ≤ UB_SIZE(196608)`——4B dtype 上限 16384、
  2B 上限 32768（Python 侧 `_vec_max_elems` 同式对齐到 repeat 量子）；
- `iters`：≥1；`blockNum`：0→满核 AIV（各 block 独立同链，聚合 = B×单核窗口）。

运行（timed 构建）：

```bash
python <xlite>/tests/performance/micro_bench.py           # 实验 4 自动跑默认档位
python <xlite>/tests/performance/micro_bench.py --vec-sizes 64,256,2k,16k  # 自定义 elems 档
python <xlite>/tests/performance/micro_bench.py --vec-only  # 只跑实验 4(向量)
```

`--vec-sizes` 支持 `k/m` 后缀与十进制/0x（向下对齐 repeat 量子、钳 UB 上限）；
默认档位 = repeats 阶梯 {1,4,16,64,256} × 每 repeat 元素数。4a 数值核对（最小/
最大 elems × B=1/满核，每 block 全数组逐位比对，exp 用容差）→ 4b 反消解（iters
翻倍 compl 比值应 2.0）→ 4c 吞吐 sweep（6 变体 × elems 档 × B∈{1,满核}，
ns/元素 + Gelem/s + 超越函数差分表）。`--analyze` [9] 段给出 profiler 外部视角
（实测单 block 安静态差 +0.7~1.6µs，与 MTE 家族 ~+2µs 前后缀先例同量级）。

### 910B3 参考表（timed 构建，5 次中位，B=1）

> 2026-10-15 复测更新（同 MTE 一节所述的 TU 合并）：与首测逐格一致，差 ≤0.3%。

ns/元素（分子 = complete 含 drain；iters 按 8M 总元素量自适应）：

| elems(reps) | add_f32 | mul_f32 | add_i32 | add_f16 | sqrt_f32 | exp_f32 |
|---|---|---|---|---|---|---|
| 64(1) | 0.1997 | 0.2083 | 0.1823 | — | 0.2865 | 0.2778 |
| 128(1) | — | — | — | 0.0998 | — | — |
| 256(4) | 0.0651 | 0.0673 | 0.0608 | — | 0.0977 | 0.0955 |
| 512(4) | — | — | — | 0.0326 | — | — |
| 1024(16) | 0.0299 | 0.0309 | 0.0282 | — | 0.0526 | 0.0532 |
| 2048(16) | — | — | — | 0.0149 | — | — |
| 4096(64) | 0.0213 | 0.0224 | 0.0201 | — | 0.0408 | 0.0424 |
| 8192(64) | — | — | — | 0.0107 | — | — |
| 16384(256) | 0.0200 | 0.0190 | 0.0188 | — | 0.0367 | 0.0367 |
| 32768(256) | — | — | — | 0.0100 | — | — |

单核饱和吞吐与满核聚合（最大 elems 档）：

| 变体 | 单核 Gelem/s | 满核 Gelem/s(48 block) | 扩展性 |
|---|---|---|---|
| add_f32 | 50.0 | 2380.3 | 0.99×（纯 UB 计算完全线性） |
| mul_f32 | 52.6 | 2503.1 | 0.99× |
| add_i32 | 53.2 | 2530.2 | 0.99× |
| add_f16 | 99.7 | 4698.4 | 0.99× |
| sqrt_f32 | 27.2 | 1300.4 | 0.99× |
| exp_f32 | 27.3 | 1302.7 | 0.99× |

要点：

1. **饱和吞吐 50 Gelem/s（f32）/ 100 Gelem/s（f16）**：add_f16/add_f32 饱和
   ns/元素比 = 0.50，16 位通道精确翻倍；add_i32/add_f32 = 0.94，整数与浮点走
   同一向量单元同速。
2. **超越函数边际成本 ≈ 一条乘法**：sqrt−mul 与 exp−mul 差分在 16384 元素处
   均收敛到 +0.0177 ns/元素（≈ vmul 自身的 0.0190）——910B3 的 vsqrt/vexp 与
   加乘法吞吐同量级，**不是**慢路径；小向量档差分更大（64 元素处 +0.07）是
   每条指令固定开销未摊平，同 add_f32 曲线形态。
3. **每条指令固定开销**：64 元素（1 repeat）处 ~0.2 ns/元素 × 64 = ~12.8 ns/
   （vadd+barrier）对——即 RAW 屏障节拍；随 elems 增大单调摊平（issue 端每
   chunk 都等屏障，无投机空间），16384 处 0.020 ns/元素 = 1.28 ns/repeat。
4. **drain 很小且与变体无关**（0.5~10µs，< compl 的 6%，随 elems 弱增长）：
   与 MTE 家族（异步引擎，drain 可占 compl 四成）形成对照——V pipe 的完成信息
   被逐 chunk 屏障拉回 issue 端，t1 后仅剩末批写回排空。

详见 workspace 报告 `vec_op_perf.md`。

## 性能测试方法（复现）

```bash
# 构建（源码树 CMake 全量；ascendc 子工程不支持增量，必须 rm -rf build）
cd <xlite 源码树>
rm -rf build && cmake -B build && cmake --build build -j && cmake --install build

# timed 变体构建（bench_aiv.cpp 追加 12 个 *_timed 孪生 + 4 个 MTE + 6 个向量
# kernel，bench_aic.cpp 追加 12 个 *_timed；两种构建的 TU 构成不同，
# 数值不可互比，见 TU 规则；用完须重装默认构建）
rm -rf build && cmake -B build -DXLITE_BENCH_TIMED=ON \
    && cmake --build build -j && cmake --install build

# 运行（注意：本机 site-packages 若另有 xlite wheel，必须用 PYTHONPATH 指向
# 源码树构建，否则 import 到旧包）
cd <workspace>
export PYTHONPATH=<xlite 源码树>
export LD_LIBRARY_PATH=$PYTHONPATH/xlite/lib:<torch>/lib:$LD_LIBRARY_PATH

python <xlite>/tests/performance/micro_bench.py          # 冒烟：核对+计时(实验 2c/3/4
                                                         # timed 构建才有, 默认跳过)
python <xlite>/tests/performance/micro_bench.py --mte-sizes 64k,1m,64m  # MTE 自定义 size
python <xlite>/tests/performance/micro_bench.py --mte-only  # 只跑实验 3(MTE 内存微基准)
python <xlite>/tests/performance/micro_bench.py --vec-sizes 64,256,2k,16k  # 向量自定义 elems
python <xlite>/tests/performance/micro_bench.py --vec-only  # 只跑实验 4(向量指令微基准)
python <xlite>/tests/performance/perf_xlite_ops.py \
    <xlite>/tests/performance/micro_bench.py             # device 口径 → CSV
python <xlite>/tests/performance/micro_bench.py \
    --analyze xlite_prof_results/<csv>                   # 可读分析报告 + 自动结论
# 干净 wall 口径：perf_xlite_ops.py 配置区 DEVICE_TIME 置 False 再跑
```

> profiler 归因提示：`bench_timed` / `bench_mte` / `bench_vec` 这类统一入口的
> kernel 名按变体派生（`scalar_add_aiv_timed_15`、`mte_read_aiv_0`、
> `vec_add_f32_aiv_65` 等），与 op 名无共享前缀，perf 工具自动归因会拒绝；
> `perf_xlite_ops.py` 的 `KERNEL_OP_OVERRIDES` 已按前缀指回（`--analyze` 的
> [6]/[8]/[9] 段分别给出三者的外部视角；只含单一家族的 CSV 相应跳过缺失段）。

910B3 参考结果（24 AIC / 48 AIV）：

**空算子 launch floor（µs，满核 5 次中位）：**

| 核型 | device（稳态） | 干净 wall（稳态） | launch+sync+pybind |
|------|---------------|-------------------|-------------------|
| AIV（48 block） | 4.2 | ≈ 30 | ≈ 26 |
| AIC（24 block） | 2.0 | ≈ 26 | ≈ 24 |
| MIX（24+48 block） | 6.2 | ≈ 33 | ≈ 27 |

block_num=1 时三者趋同（≈2–3.5µs）：满核差异主要来自被调度的 block 数量。空
kernel 稳态 wall ≈26–33µs 中 device 仅 2–6µs——小 kernel 端到端成本由 host 下发
与同步主导，核型选择的 device 侧差异（≤4µs）通常不是瓶颈，减少下发次数（融合）
收益更大。

**标量成本（ns/op，@iters=1M，两轮中位，AIV 与 AIC 完全一致）：**

| 变体 | ns/op | 折合 0.556ns/槽 |
|------|-------|----------------|
| add / xor / mul / add_unroll4 | 1.667 | 3 槽 |
| fadd | 2.222 | 4 槽 |
| div | 4.445 | 8 槽 |

全部为 **0.556 ns/槽的整数倍**（≈1.8G 标量发射/s，时序反推非手册值）。分解结果
a ≈ o ≈ 0.83 ns，b ≈ 0（循环控制被流水完全隐藏；unroll4 展开零收益）。线性拟
合截距（launch floor）0.8–2.8µs，与空算子实验一致。
