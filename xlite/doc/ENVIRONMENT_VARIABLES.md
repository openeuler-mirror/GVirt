# xlite 环境变量说明文档

本文档介绍 xlite 使用的环境变量及其配置方法。

## 核心环境变量

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `XLITE_DEVS_PER_NODE` | 整数 | 自动检测 | 每个节点的设备数量。未设置时自动通过 `aclrtGetDeviceCount` 获取。 |
| `XLITE_NODE_IPS` | 字符串 | `127.0.0.1` | 多节点推理时的节点 IP 地址列表，以逗号分隔。当 `rankSize > nDevPerNode` 时必须设置。 |
| `XLITE_PORT` | 整数 | `10266` | 通信基础端口号。TP 通信使用 `port + rankId/tpSize`，DP 通信使用 `port + 200 + rankId%tpSize`，XCCL 通信使用 `port + 400`。 |
| `XLITE_COMM_OPTIMIZE_LEN` | 整数 | `6144` | 通信优化阈值长度。用于优化prefill阶段长序列的通信性能。 |
| `XLITE_DISABLE_XCCL` | 布尔 | `false` | 是否禁用 XCCL（XLite 自定义通信算子）。设置为 `true` 时禁用，回退到 HCCL。 |
| `XLITE_FLASH_ATTENTION_ENABLE` | 布尔 | `false` | 是否启用 Flash Attention 优化。长序列场景建议开启。设置为 `true`、`1`、`yes` 或 `on` 时启用。 |

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
|---------|-------------|------|
| TP HCCL | `port + rankId/tpSize` | Tensor Parallel HCCL 通信 |
| DP HCCL | `port + 200 + rankId%tpSize` | Data Parallel HCCL 通信 |
| XCCL | `port + 400` | XLite 自定义通信层基础端口 |

**注意**：每次初始化会额外增加 500 的端口偏移，以支持多次运行实例。