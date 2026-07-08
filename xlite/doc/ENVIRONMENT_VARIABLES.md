# xlite 环境变量说明文档

本文档介绍 xlite 使用的环境变量及其配置方法。

## 核心环境变量

| 环境变量 | 类型 | 默认值 | 说明 |
| --------- | ------ | -------- | ------ |
| `XLITE_DEVS_PER_NODE` | 整数 | 自动检测 | 每个节点的设备数量。未设置时自动通过 `aclrtGetDeviceCount` 获取。 |
| `XLITE_NODE_IPS` | 字符串 | `127.0.0.1` | 多节点推理时的节点 IP 地址列表，以逗号分隔。当 `rankSize > nDevPerNode` 时必须设置。 |
| `XLITE_PORT` | 整数 | `10266` | 通信基础端口号。TP 通信使用 `port + rankId/tpSize`，DP 通信使用 `port + 200 + rankId%tpSize`，XCCL 通信使用 `port + 400`。 |
| `XLITE_COMM_OPTIMIZE_LEN` | 整数 | `6144` | 通信优化阈值长度。用于优化prefill阶段长序列的通信性能。 |
| `XLITE_DISABLE_XCCL` | 布尔 | `false` | 是否禁用 XCCL（XLite 自定义通信算子）。设置为 `true` 时禁用，回退到 HCCL。 |
| `XLITE_MOE_ALLTOALL` | 布尔 | `false` | 是否启用 MoE AlltoAll 通信模式。启用后，MoE 的 dispatch/combine 阶段使用 AlltoAllV 集合通信替代默认的 AllGather + ReduceScatter 方式，跨 EP（Expert Parallel）rank 分发/收集 token，并消除 MoE 后的 TP AllReduce。 |
| `XLITE_ACTIVE_TOKENS_RATIO_PER_EP` | double | `1.0f` | 影响开启EP后单卡按专家排序后激活Tensor的内存占用。该环境变量有效配置范围为[1/ep_size ... 1.0f]。默认1.0f含义为单卡极端情况需要处理所有激活的Token，因此专家排序后激活Tensor的shape为[token个数 * num_experts_per_tok, hidden_size]; 配置为1/ep_size含义为单卡只需要处理所有激活的Token的1/ep_size，因此专家排序后激活Tensor的shape为[token个数 * num_experts_per_tok /ep_size, hidden_size]。该值越小代表EP负载越均衡，排序后激活Tensor预占内存越小，配置过小可能导致算子报错，建议按照实际负载均衡度评估合理值。注意：该配置仅当输入token个数（所有DP的batched token总和）>= 1024个时才生效，小于1024的情况使用默认1.0f进行计算。 |

## 布尔值解析规则

以下值被视为 `true`（不区分大小写）：

- `true`
- `1`
- `yes`
- `on`

其他任何值（包括空或未设置）均被视为 `false`。

## 端口分配说明

XLite 使用以下端口分配策略（基于 `XLITE_PORT`）：

| 通信类型 | 端口计算公式 | 说明 |
| --------- | ------------- | ------ |
| TP HCCL | `port + rankId/tpSize` | Tensor Parallel HCCL 通信 |
| DP HCCL | `port + 200 + rankId%tpSize` | Data Parallel HCCL 通信 |
| XCCL | `port + 400` | XLite 自定义通信层基础端口 |

**注意**：每次初始化会额外增加 500 的端口偏移，以支持多次运行实例。

## 调试环境变量

### `XLITE_DEBUG_ON`

控制 xlite 调试代码的**编译期**开关。这是一个**编译期**而非运行期变量：仅在 `cmake` 配置阶段被读取，改变取值后必须重新配置/编译才会生效（见 [开发指南](contributing.md) 的"编译"小节）。

**取值**：逗号分隔的类别 token 列表（不区分大小写）。

| 取值 | 说明 |
| ------ | ------ |
| 未设置 / 空 / `0` / `false` / `no` / `off` | 不定义任何调试宏，为发布构建（默认）。 |
| `base` | 仅启用基础调试基础设施（颜色化、原子打印、`Runtime.debug` 等），不开启任何具体类别。 |
| `forward` | 编译前向传播中间张量打印（`model.cpp` 中每层前后的张量检查，含 NaN/Inf/大值检测）。 |
| `tuner` | 编译 `auto_tuner` 的 tile size 选择调试日志。 |
| `gettensor` | 编译动态张量池 `GetTensor` 的分配日志。 |
| `misc` | 编译其他零散调试日志。 |
| `all` | 等价于 `forward,tuner,gettensor`，编译全部调试类别。 |
| 任意组合 | 如 `forward,tuner`，按所选类别分别编译。 |

**示例**：

```bash
# 仅前向中间张量打印
XLITE_DEBUG_ON=forward cmake -B build && cmake --build build -j

# 前向 + tuner 调试
XLITE_DEBUG_ON=forward,tuner cmake -B build && cmake --build build -j

# 全部调试类别
XLITE_DEBUG_ON=all cmake -B build && cmake --build build -j

# pip 可编辑安装时携带调试
XLITE_DEBUG_ON=forward,tuner pip install -v -e .[dev] --no-build-isolation
```

**机制说明**：CMake 将该变量解析为编译宏——任意非假值会定义基础宏 `XLITE_DEBUG_ON`，并为每个 token `TOK` 额外定义 `XLITE_DEBUG_ON_<TOK>`（大写）。具体哪个宏控制哪段代码由调试模块 `csrc/debug.h` 决定：

- `XLITE_DEBUG_ON`：基础调试基础设施。编译共享的 `XDebugStream`/`XDebugLog`（rank 感知、配色、单次 flush 的原子打印，避免多 rank/多线程输出交错）、`XDebug*` 实现（`csrc/debug.cpp`）、`Runtime.debug` 运行期门、`base.*` 中依赖 torch 的 `ToScalarType`/`Save` 等。所有类别都需要它。
- `XLITE_DEBUG_ON_FORWARD`：使前向传播中的 `XDEBUG_PRINT*` / `XDEBUG_SET_STATE` 宏生效（否则展开为空）。前向打印在运行期仍受 `Runtime.debug` 门控，可用 `XDEBUG_SET_STATE` 按 rank/层进一步收窄。
- `XLITE_DEBUG_ON_TUNER`：`auto_tuner.cpp` 中的 `#ifdef` 调试块。
- `XLITE_DEBUG_ON_GETTENSOR`：`base.cpp` 中动态张量池分配的 `#ifdef` 调试块。

> **关于 wheel/发布构建**：`setup.py` 在非可编辑（wheel）构建时会从 cmake 子进程环境中剥离 `XLITE_DEBUG_ON`，确保即便开发 shell 导出了该变量或 `cmake_build/` 存在残留缓存，发布包也不会带上调试代码。因此该变量仅在源码/可编辑安装的开发场景下有意义。
