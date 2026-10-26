# microbench：算子微基准套件（launch floor / 标量链 / MTE 访存 / 向量指令 / Cube matmul）

微基准 kernel 家族，统一入口 `tests/performance/micro_bench.py`。五个实验按序执行，
覆盖 AIV / AIC / MIX 三种核型，主要回答：一次 kernel 下发本身多久（实验 1）；单条
标量 / 向量指令或一次 matmul 的 device 侧成本（实验 2、4、5）；经 MTE 数据通路的
带宽与时延如何随工作集变化（实验 3）。

kernel 源文件只有一套：默认构建只含 untimed kernel；`XLITE_BENCH_TIMED=ON` 构建在
同一批 TU 内追加 kernel 内计时家族（`GetSystemCycle()` 读 SYS_CNT，不依赖 profiler）。

## 1. 实验总览

| 实验 | kernel / Python 入口 | 核型 | 测量对象 | 子步骤 | 构建 |
|------|----------------------|------|----------|--------|------|
| 1 空算子 launch floor | `empty_{aiv,aic,mix}` / `bench_empty` | AIV / AIC / MIX | 一次下发的 device 固有开销 | — | 默认 |
| 2 标量依赖链 | `scalar_<op>_{aiv,aic}` / `bench_scalar`；timed 孪生 / `bench_timed` | AIV / AIC | 单次标量运算 + 循环控制 | 2a 核对 / 2b wall / 2c 核内计时 | 2a、2b 默认；2c timed |
| 3 MTE 访存 | `mte_{read,write,copy,chase}_aiv` / `bench_mte` | AIV | 带宽 / 指针追逐时延 vs 工作集 | 3a 核对 / 3b 反消解 / 3c 带宽 / 3d 时延 | timed |
| 4 向量指令 | `vec_<variant>_aiv` / `bench_vec` | AIV | 向量吞吐 vs 向量长度 | 4a 核对 / 4b 反消解 / 4c 吞吐 | timed |
| 5 Cube matmul | `cube_mmad_aic` / `bench_cube` | AIC | 单次 matmul 完成时间 / TFLOPS vs tile | 5a 核对 / 5b 反消解 / 5c tile sweep | timed |

- **实验 1**：无计算、无搬运、无同步，测该核型下单次 launch 的执行底线。
- **实验 2**：kernel 内一条标量依赖链，`iters` 为运行时参数；2c 用 timed 孪生做
  核内计时，分离循环控制与运算本身。
- **实验 3**：lmbench 风格，read/write/copy 测 MTE 带宽，chase 测依赖链时延。
- **实验 4**：数据 UB 驻留，测向量吞吐 vs 向量长度；sqrt/exp 用载体差分。
- **实验 5**：每次下发一次完整 `A @ Bᵀ`（FP16 输入 / FP32 累加 / FP16 输出）；
  依赖链由 host 多次下发实现。

实验 2c/3/4/5 仅 timed 构建存在；默认构建下脚本打印提示并跳过。

## 2. 使用方法

### 2.1 构建

```bash
# 默认构建（实验 1、2a、2b）
rm -rf build && cmake -B build && cmake --build build -j && cmake --install build

# timed 构建（追加实验 2c、3、4、5）
rm -rf build && cmake -B build -DXLITE_BENCH_TIMED=ON \
    && cmake --build build -j && cmake --install build
```

- ascendc 子工程不支持增量编译，必须 `rm -rf build`。
- 两种构建的 TU 构成不同，数值不可互比；切回默认口径须重装默认构建。
- editable 安装等价：设置环境变量 `XLITE_BENCH_TIMED=ON` 后执行
  `pip install --force-reinstall --no-deps --no-build-isolation -v -e .`。

### 2.2 运行环境

```bash
export PYTHONPATH=<xlite 源码树>
export LD_LIBRARY_PATH=$PYTHONPATH/xlite/lib:<torch>/lib:$LD_LIBRARY_PATH
```

site-packages 中的旧 xlite wheel 会遮蔽源码树构建，必须用 `PYTHONPATH` 指回。

### 2.3 命令行

`python <xlite>/tests/performance/micro_bench.py [选项]`

| 选项 | 作用 | 约束 |
|------|------|------|
| （无参数） | 按 1→5 顺序执行；缺失 timed 的实验跳过 | — |
| `--quick` | 缩减重复次数与规模档位，约 1/4 时间 | 冒烟用，精度降低 |
| `--mte-sizes 64k,1m,...` | 实验 3 工作集档位；支持 `k/m/g` 与十进制/0x，对齐 64B | timed |
| `--mte-only` | 只跑实验 3 | timed；与另两个 only 互斥 |
| `--vec-sizes 64,256,2k,...` | 实验 4 elems 档位；对齐 repeat 量子并钳 UB 上限 | timed |
| `--vec-only` | 只跑实验 4 | 同上 |
| `--cube-sizes '128,128,128;...'` | 实验 5 tile 列表，分号分隔，每项 `m0,n0,k0` | timed；须满足 L0/L1 预算 |
| `--cube-only` | 只跑实验 5 | 同上 |
| `--analyze <csv> ...` | 读取 profiler CSV 输出分析报告 | 必须为第一个参数 |

`--*-only` 在默认构建下直接报错。实验 3/4 的 `iters` 由脚本按“总搬运量 / 总元素量
恒定”自适应；实验 5 缺省为五个固定 tile。

### 2.4 profiler 外部口径

```bash
python tests/performance/perf_xlite_ops.py tests/performance/micro_bench.py  # → CSV
python tests/performance/micro_bench.py --analyze xlite_prof_results/<csv>
```

- 报告分段：`[1]` launch floor、`[2]` 标量每操作成本、`[3]` 槽模型、`[4]` AIV vs
  AIC、`[5]` 循环开销分解、`[6]` `bench_timed` 外部视角、`[7]` 自动结论、`[8]`
  `bench_mte` 外部视角、`[9]` `bench_vec` 外部视角。
- 干净 wall 口径：`perf_xlite_ops.py` 配置区 `DEVICE_TIME` 置 False。
- `bench_*` 的 kernel 名按变体派生，自动归因会拒绝；`KERNEL_OP_OVERRIDES` 已按
  前缀指回。实验 5 目前没有外部视角分析段。

### 2.5 调用约束

- `bench_mte/bench_vec/bench_cube` 走 `Runtime` 私有流，与 torch 当前流不排序；
  调用前必须 `torch.npu.synchronize()`（`micro_bench.py` 的包装已内置）。
- ticks / acc 按 64B/block 步距分配，紧排会被跨核同线 store 互踩。
- host 校验：`div` 拒绝 seed=0；`add_unroll4` 要求 iters 为 4 的倍数；MTE 的
  `bytes` 为 32B 倍数（chase 64B）；向量 `elems` 为 repeat 量子整数倍且不超 UB；
  Cube 的 `m0/n0/k0` 为 16 的倍数且满足 L0/L1 预算。
- blockNum：empty / MTE / VEC 族 0 → 满核（chase 恒 1）；标量 / Cube 族 0 → 1。

## 3. 实验原理与数据说明

### 3.1 公共机制：kernel 内 SYS_CNT 计时

- `GetSystemCycle()` 读 SYS_CNT，固定 50 MHz（非核频），`cycles / 50 = µs`，
  分辨率 20 ns；单条指令成本须用 iters 斜率法。
- 三个采样点：t0 测量区起点；t1 循环/搬运全部下发完；t2
  `pipe_barrier(PIPE_ALL)` 排空后（真实完成）。
- 三个口径：`issue = t1-t0`、`complete = t2-t0`（主口径）、`drain = t2-t1`
  （在飞指令排空）。
- ticks 由调用方预分配，采样值先存寄存器，测量区关闭后批量回写；边采边写会让
  MTE3 store 落进被测区间。
- 多 block 聚合取各 block complete 的最大值（最慢 block 决定 kernel 完成）。
- 每 block 输出独占一条 64B GM line（`BENCH_MTE_STRIDE`），避免同线互踩。
- timed 与 untimed 同源：同一生成宏的两套定义，函数体一致，只多打点与 ticks 回写。

### 3.2 实验 1：空算子 launch floor

kernel 无计算/搬运/同步，测一次 launch 的 device 侧固有开销（下发-调度-执行-返回）。
`empty_mix` 为 MIX_AIC_1_2（1 AIC : 2 AIV）；blockNum 对 AIV/AIC 分别指对应核的
block 数，MIX 指 Cube block 数（硬件配 2 个 AIV）。kernel 保留一个不使用的
`GM_ADDR dummy`，因为 host stub 提取器要求参数表非空。

**数据说明**：运行输出 `wall_med_us`（含 host 下发 + 同步）；device 精确值看
`--analyze [1]`。空 kernel 的 wall 远大于 device，说明小 kernel 端到端成本由 host
下发与同步主导，减少下发次数（融合）比选核型收益更大。

### 3.3 实验 2：标量依赖链

变体为 `add / xor / mul / div / fadd / add_unroll4` × `{aiv, aic}`，`iters` 均按
总运算次数计（`add_unroll4` 内部除以 4），ns/op 可直接对比。add/xor/unroll4 的
操作数与循环计数器纠缠（`x ^ i`），mul/div/fadd 用普通操作数。

**测量有效性**：

1. anti-DCE：累加结果在循环后直写 GM + `pipe_barrier(PIPE_ALL)`。
2. anti-folding：仅靠运行时 `seed` 不够，bisheng 会把 `acc += x` 重结合为闭式解；
   故操作数与计数器纠缠。mul/div/fp32 链无廉价闭式解，无需纠缠。
3. 链式语义：每迭代消费上一结果，测得“运算延迟 + 循环控制”，不是吞吐。
4. 双通道可编译：结果回写用标量直存 GM（UB→GM 搬运 intrinsic 在 cube 通道不可
   用），每 launch 一次的成本只进截距。

**三方程分解**（b=循环控制、a=add、o=xor）：`s1 = b+a+o`、`sx = b+2o`、
`s4 = b/4+a+o`，解得 `b = 4(s1-s4)/3`、`o = (sx-b)/2`、`a = s1-b-o`。mul/div/fadd
直接测得“运算 + 循环控制”之和。

**数据说明**：2a 与 Python 参考模型逐位比对；2b 为含 host 的 wall 粗测；2c 输出
`issue / compl / ns/op / drain`，`ns/op = compl×1000/iters`。不同 iters 下 ns/op
应基本相等。`--analyze [2]` 拟合斜率与截距，`[3]` 槽模型，`[4]` AIV/AIC 对比，
`[5]` b/a/o 分解，`[6]` timed 与 untimed 之差（应为打点机制开销）。纯标量链
`drain ≈ 0`，是计时机制的零点校准。

### 3.4 实验 3：MTE 内存微基准

四个 kernel 共用 `bench_mte` 入口：

| kernel | lmbench 对应 | 机制 | bytes / iters |
|--------|--------------|------|---------------|
| `mte_read_aiv` | `bw_mem rd` | MTE2 GM→UB，96KB×2 双缓冲，逐段无同步 | 每 rep 流过量 / rep 数 |
| `mte_write_aiv` | `bw_mem wr` | tile 在计时段外填充后 MTE3 流式写 GM | 同上 |
| `mte_copy_aiv` | `bw_mem bcopy` | MTE2↔MTE3 ping-pong（flag 对） | 同上，字节只计一遍 |
| `mte_chase_aiv` | `lat_mem_rd` | 64B 块 GM→UB + 握手 + 读出下一偏移 | 工作集 / 追逐步数 |

- `src` 为 INT8 工作集；read/chase 的 `dst` 为 INT32（acc 字按 64B 步距），
  write/copy 的 `dst` 为 INT8；`ticks` 每 block 独占一条 64B line。
- 非 chase 时 block 以 grid-stride 平分工作集：block b 从 `b*96KB` 起，步长
  `blockNum*96KB`，尾段收缩。chase 恒 1 block。
- chase 工作集为随机单一环置换：`size/64` 个 64B 块，块首 u32 存后继块偏移。
- MTE intrinsic 仅 vec flavor 支持，生成宏用 `__DAV_C220_VEC__` 守卫，cube 通道
  得到同签名空体。

**数据说明**：3a 核对 copy/write/read/chase；3b 用 iters 翻倍检查搬运未被消解；
3c 输出 `issue / compl / drain / 聚合 GB/s / 每核 GB/s`，聚合
`GB/s = iters*bytes/compl`，每核 = 聚合 / actB（`actB` 为实际有工作的 block 数）；
3d 输出 `ns/步 = compl/iters`。带宽分母必须用 complete：MTE 是异步引擎，drain
可能很大。带宽 sweep 不刻意 flush L2，曲线在 L2 容量处出现拐点。chase 是“经 MTE
的 load-to-use”（含下发 + GM 时延 + 握手 + 标量读），不能当裸 HBM 时延引用。

### 3.5 实验 4：向量指令微基准

变体：`add_f32 / mul_f32 / add_i32 / add_f16 / sqrt_f32 / exp_f32`，数据 UB 驻留，
每 chunk 256B，只跑 AIV。

**载体差分**：`dst = op(dst)` 直链会发散或收敛，改为 `dst = dst * op(c)`，取
`op(c)=1` 使 dst 逐位不变；`cost(op) = compl(本变体) − compl(mul_f32 同参数)`。
sqrt 不在依赖链上，差分量是“mul 链上再加一条超越指令”的边际成本，不是孤立吞吐。
`xor_u32` 不可实现（无 raw 32 位 vxor、host 无 INT16），以 add_i32 替代。

**RAW 竞态**：相邻向量指令的读与在飞写会重叠，不插屏障时连续加法只有一次生效；
每 chunk 末尾必须 `pipe_barrier(PIPE_V)`（二元与载体变体各恰一条，差分不受污染）。
因此小向量档测的是“一条指令 + 一条屏障”的节拍。

**参数**：`src = [x0; c]`；`out` 每 block 独立一段；`elems` 为 repeat 量子整数倍，
且 `2*elems*sz + scratch ≤ UB_SIZE(196608)`（4B dtype 上限 16384，2B 上限 32768）。

**数据说明**：4a 用最小/最大 elems × 单 block/满核做全数组逐位比对；4b 用 iters
翻倍检查指令未被消解；4c 输出 `ns/elem = compl×1000/(iters×elems)`、
`Gelem/s` 与超越函数差分表。小向量档由固定开销主导，大向量档趋近饱和吞吐。V pipe
的 drain 很小（逐 chunk 屏障把完成信息拉回 issue 端）。

### 3.6 实验 5：Cube matmul 微基准

每次下发执行一次完整 `Matmul::Run()`，复用 `matmul.h`。kernel 内不循环：flag
生命周期在 `kLoop <= 4` 的 tile 上不能干净重置，几轮后 L0A/L0B 读写冲突；依赖链
由 host 多次下发实现。`blockNum` 恒 1，未分到 tile 的核早退。tile 约束（m0/n0/k0
为 16 的倍数）：`L0A=m0*k0≤64KB`、`L0B=n0*k0≤64KB`、`L0C=4*m0*n0≤64KB`、
`L1=4*k0*(2*m0+n0)≤512KB`。

**数据说明**：5a 与 numpy `A@Bᵀ` 比对；5b 用 host 下发次数翻倍检查未消解；5c 输出
`wall us/launch`（含 host）、`issue / compl / drain` 与
`TFLOPS = 2*m0*n0*k0/compl/1e6`。Cube 吞吐看核内 complete，端到端下发成本看 wall。

### 3.7 实现与构建机制

| 文件 | 职责 |
|------|------|
| `csrc/kernels/microbench.h` | 全部实现：五族生成宏、循环体、核型路由、测试项与四个 timed 聚合宏 |
| `csrc/kernels/kernel_macro.h` | 通用 kernel 内 SYS_CNT 采样（`BENCH_TICK*`） |
| `csrc/kernels/bench_aiv.cpp` | 7 条 plain + 12 个 timed 孪生 + 4 条 MTE + 6 条向量 |
| `csrc/kernels/bench_aic.cpp` | 7 条 plain + 12 个 timed 孪生 + 1 条 Cube |
| `csrc/kernels/bench_mix.cpp` | 1 条 empty |

- `MICROBENCH_TIMED` 由 CMake 选项 `XLITE_BENCH_TIMED` 控制（默认 OFF）。OFF 时
  四个 `*_ITEMS` 展开为空，timed / MTE / 向量 / Cube kernel、host 入口与 Python
  绑定均不存在，三个 TU 与定标布局逐 token 一致。
- ON 时在同一批 .cpp 内追加实例化，不新增 TU；宏经
  `ascendc_compile_definitions` 送进 precompile / preprocess / aic / aiv 四个阶段，
  host 侧由 `target_compile_definitions` 给 `xlite` 与 `_C`。
- timed kernel 名必须在一次 `BENCH_CAT3` 粘贴中拼出；否则 CANN auto_gen 的改名
  机制会拼出与包装器不一致的符号。
- 不同核型不能共用 TU：precompile 按文件取第一个 `KERNEL_TASK_TYPE_DEFAULT`。
  默认构建的 TU 构成必须冻结；timed 构建是另一套代码生成上下文，数值自成基线，
  切换构建或增删 timed 项后须重新定标。

## 4. 实测经验

**测量有效性**

- 运行时 `seed` 不足以阻止编译器把 `acc += x` 重结合为闭式解，必须让操作数与
  循环计数器纠缠。
- 空 inline-asm 屏障不可用：aicore 后端只支持 64 位 `"l"` 约束，往返开销会淹没
  被测运算。
- 1 槽级整数运算对 TU 代码生成上下文敏感：同 TU 增删其他 kernel 会让未改动的
  kernel 漂移发射槽数；默认 TU 必须冻结。
- profiler 与核内口径之差 = kernel 前后缀；大额分配后的首调、满核配置会出现单向
  膨胀（派发倾斜 + 收尾），属口径差异。稳态看核内窗口，端到端看 `device_us`。

**硬件与工具链**

- MTE 是异步引擎，带宽分母必须用 complete；drain 可能占很大比例。
- 跨核标量 GM store 打进同一 64B line 会整线互踩且不可恢复；ticks/acc 必须
  64B/block 独占（MTE3 store 不受影响）。
- `rt.stream` 与 torch 流不排序，不同步会与 H2D 上传 / `zeros` 填充竞态，且为
  间歇性；调用前必须同步。
- V pipe 相邻指令存在真实 RAW 竞态，`pipe_barrier(PIPE_V)` 是载荷性的，生产
  kernel 同样绕不开。
- Cube 不能在 kernel 内循环 `Matmul::Run()`；CANN 会在所有 cube core 上 launch，
  无 tile 的核必须早退。
