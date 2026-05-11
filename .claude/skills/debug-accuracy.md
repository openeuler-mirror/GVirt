---
name: debug-accuracy
description: Debug xlite accuracy issues by comparing outputs between xlite and torch_npu backends
---

# xlite Accuracy Debug Skill / xlite精度定位Skill

This skill helps locate accuracy issues in xlite by systematically comparing outputs between xlite debug mode and torch_npu backend.

本skill帮助定位xlite精度问题，通过系统对比xlite debug模式和torch_npu后端的输出差异。

## Prerequisites / 前置条件

**重要：所有测试必须在项目根目录下的 `xlite` 目录中运行。**

首先确定项目根目录路径：
```bash
# 获取项目根目录（GVirt仓库的根目录）
PROJECT_ROOT=$(git rev-parse --show-toplevel)
XLITE_DIR="${PROJECT_ROOT}/xlite"

# 确保在xlite目录下运行测试
cd "${XLITE_DIR}"
```

**Important: All tests must be run from the `<project_root>/xlite` directory.**

## Usage / 使用方法

```
/debug-accuracy <model_name> [model_path]
```

- `model_name`: Model name (e.g., qwen3, deepseek_v3, glm5, llama, qwen2.5, etc.)
- `model_path`: Optional model checkpoint path (defaults to `/mnt/nvme0n1/models/<model_name>`)

## Workflow Steps / 工作流程

### Step 1: Create Minimal Test Case / 创建最小测试用例

Create a single-card test with minimal configuration:
创建单卡最小配置测试：

1. Set `n_layers=1` (or minimal layers needed if model structure varies per layer)
2. Output only 1 token (`--max-new-tokens 1`)
3. Enable deterministic mode:
   ```bash
   export HCCL_DETERMINISTIC=true
   export LCCL_DETERMINISTIC=true
   ```

**Actions:**
1. Read the corresponding model test function from `xlite/tests/run.sh`
2. Create a minimal config JSON with `n_layers=1` and minimal parameters
3. Generate a test script with single card execution (no torchrun for small models)

### Step 2: Compare Outputs / 对比输出差异

Run both backends and compare their outputs:
运行两个后端并对比输出：

#### 2.1 xlite Debug Mode Output / xlite Debug模式输出

1. Rebuild xlite with debug mode:
   ```bash
   cd ${XLITE_DIR}
   rm -rf build && mkdir -p build
   cmake -B build -DXLITE_DEBUG=ON && cmake --build build -j
   cmake --install build
   ```

2. Run test with `FORWARD_BACKEND=xlite`:
   ```bash
   cd ${XLITE_DIR}
   export FORWARD_BACKEND=xlite
   export HCCL_DETERMINISTIC=true
   python tests/generate.py --model <model> --ckpt-path <path> --config <config> --input-file <input> --max-new-tokens 1
   ```

3. Capture output log with XLITE_DEBUG prints

#### 2.2 torch_npu Backend Output / torch_npu后端输出

1. Open the corresponding model Python file in `tests/models/`
2. Set `debug = True` at the top of the file (usually a global variable)
3. Run test with `FORWARD_BACKEND=torch_npu`:
   ```bash
   cd ${XLITE_DIR}
   export FORWARD_BACKEND=torch_npu
   export HCCL_DETERMINISTIC=true
   python tests/generate.py --model <model> --ckpt-path <path> --config <config> --input-file <input> --max-new-tokens 1
   ```

4. Capture output log with model debug prints

#### 2.3 Analyze Differences / 分析差异

Compare the two outputs:
对比两次输出：

**Important: Skip warmup phase / 重要：跳过warmup阶段**

Test outputs contain a warmup phase before actual computation. Ensure you **skip the warmup phase debug outputs** and only compare the actual inference outputs:
测试输出在实际计算前包含warmup阶段，确保**跳过warmup阶段的debug输出**，只对比实际推理阶段的输出：

The warmup phase may produce different intermediate values that are not representative of the actual accuracy issue.
warmup阶段可能产生不同的中间值，不能代表实际的精度问题。

**Analysis steps / 分析步骤:**

1. Identify where outputs start diverging (layer, operation, tensor)
2. Record the specific tensors/operations with differences
3. Note the magnitude of differences (relative error, absolute error)

#### 2.4 Generate Difference Table / 生成差异表格

Compare the **overall difference of the printed tensor portions**, not individual position values:
对比**整个tensor被打印部分的整体差距**，而非单个位置的值：

**Overall Tensor Statistics / 整体Tensor统计指标:**

| Stage | MSE | Max Abs Diff | Mean Abs Diff | Mean Rel Diff | Max Rel Diff | xlite Range | torch_npu Range |
|-------|-----|--------------|---------------|---------------|--------------|-------------|-----------------|
| layer0_in | ... | ... | ... | ... | ... | [min, max] | [min, max] |
| layer0_attn | ... | ... | ... | ... | ... | [min, max] | [min, max] |
| layer0_ffn | ... | ... | ... | ... | ... | [min, max] | [min, max] |

**Metrics Explanation / 指标说明:**

- **MSE (Mean Squared Error)**: 均方误差，衡量整体偏差程度
- **Max Abs Diff**: 最大绝对差异，找出差异最大的位置
- **Mean Abs Diff**: 平均绝对差异，整体差异水平
- **Mean Rel Diff**: 平均相对差异百分比，更直观的差异指标
- **Max Rel Diff**: 最大相对差异百分比，注意小数值位置可能有较大相对误差
- **Range**: 数值范围对比，确保两者在相同范围内

**Analysis Guidelines / 分析指南:**

1. BF16 normal precision range: Mean Rel Diff < 1% is acceptable
   BF16正常精度范围：平均相对误差 < 1% 可接受
2. Large Max Rel Diff on small values is expected (e.g., 1e-5 vs 5e-5 = 80% rel diff)
   小数值的大相对误差是预期的（如 1e-5 vs 5e-5 = 80% 相对误差）
3. Focus on Mean Rel Diff and Range consistency for accuracy assessment
   重点关注平均相对误差和数值范围一致性来判断精度问题

### Step 3: Refine Debug Points / 细化调试点

Based on Step 2 results, add more specific debug prints:
根据Step 2结果，增加更细粒度的调试打印：

#### 3.1 Add XLITE_DEBUG_POINT in xlite / xlite增加XLITE_DEBUG_POINT

In xlite C++ source files (`xlite/csrc/`):
在xlite C++源文件中：

1. Add `XLITE_DEBUG_POINT` macros at suspected locations:
   ```cpp
   XLITE_DEBUG_POINT("layer_{layer_id}_attention_q");
   ```

2. Rebuild and rerun xlite debug mode

#### 3.2 Add debug prints in model Python / 模型Python增加debug打印

In model Python files (`tests/models/<model>.py`):
在模型Python文件中：

1. Add print statements or debug logs at suspected operations:
   ```python
   if debug:
       print(f"layer_{layer_id}_attention_q: {tensor}")
   ```

2. Rerun torch_npu backend

#### 3.3 Iterate / 重复迭代

Repeat Step 2 with refined debug outputs until:
重复Step 2直到：

- Specific computation causing the issue is identified
- The root cause (numerical precision, algorithm difference, etc.) is found

#### 3.4 Memory Considerations / 内存注意事项

When increasing debug layers or adding more debug points, you may encounter NPU memory issues:
增加debug层数或添加更多调试点时，可能会遇到NPU内存不足的问题：

**Solution / 解决方案:**
- Increase the number of NPU cards used for testing
  适当增加使用的NPU卡数
- For example, if running with 1 card causes OOM, try 2 or more cards
  例如，单卡运行导致OOM时，可尝试使用2卡或更多卡

**Example / 示例:**
```bash
# Original single card test / 原单卡测试
python tests/generate.py --model <model> ...

# If OOM occurs, use multiple cards / 如遇OOM，使用多卡
torchrun --nproc_per_node=2 tests/generate.py --model <model> ...
```

**Important: Update moe_ep_size / 重要：同步修改moe_ep_size**

When changing the number of NPU cards, you must also update `moe_ep_size` in the config JSON file to match:
修改NPU卡数时，必须同步修改配置JSON文件中的`moe_ep_size`以匹配卡数：

```json
// For 1 card / 单卡
"moe_ep_size": 1

// For 2 cards / 双卡
"moe_ep_size": 2

// For 4 cards / 4卡
"moe_ep_size": 4
```

Failure to update `moe_ep_size` will cause incorrect expert parallelism and may lead to errors or inaccurate results.
不更新`moe_ep_size`会导致专家并行配置错误，可能引发错误或结果不准确。

### Step 4: Final Report / 最终报告

Generate final difference table with:
生成最终差异表格：

1. All identified differences across layers/stages
2. Root cause analysis
3. Recommendations for fixing the accuracy issue

## Model Configuration Templates / 模型配置模板

### qwen3 Minimal Config
```json
{
    "vocab_size": 151936,
    "dim": 5120,
    "head_dim": 128,
    "inter_dim": 25600,
    "n_layers": 1,
    "n_heads": 64,
    "n_kv_heads": 8,
    "norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "dtype": "bfloat16",
    "max_batch_size": 1,
    "max_seq_len": 1024
}
```

### deepseek_v3 Minimal Config
```json
{
    "vocab_size": 129280,
    "dim": 7168,
    "inter_dim": 18432,
    "moe_inter_dim": 2048,
    "n_layers": 1,
    "n_dense_layers": 1,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "route_scale": 2.5,
    "score_func": "sigmoid",
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "dtype": "bf16",
    "quantization": "none",
    "moe_ep_size": 1,
    "moe_tp_size": 1
}
```

### llama Minimal Config
```json
{
    "vocab_size": 32000,
    "dim": 4096,
    "inter_dim": 11008,
    "n_layers": 1,
    "n_heads": 32,
    "n_kv_heads": 32,
    "norm_eps": 1e-05,
    "dtype": "float16",
    "max_batch_size": 1,
    "max_seq_len": 1024
}
```

## Supported Models / 支持的模型

| Model Name | Model Type | Test File | Notes |
|------------|------------|-----------|-------|
| qwen3 | qwen3 | qwen3.py | - |
| qwen3_moe | qwen3_moe | qwen3_moe.py | MoE model |
| qwen3_5 | qwen3_5 | qwen3_5.py | - |
| qwen3_5_moe | qwen3_5_moe | qwen3_5_moe.py | MoE model |
| llama | llama | llama.py | - |
| deepseek_v3 | deepseek_v3 | deepseek_v3.py | MoE model |
| deepseek_v32 | deepseek_v32 | deepseek_v3.py | MoE model |
| glm4_moe | glm4_moe | glm4_moe.py | MoE model |
| glm5 | glm5 | deepseek_v3.py (model_type=glm5) | MoE model |
| minimax_m2 | minimax_m2 | minimax_m2.py | MoE model |

## Debug Variables / 调试变量

| Variable | Location | Purpose |
|----------|----------|---------|
| `XLITE_DEBUG` | CMake build flag | Enable xlite debug prints |
| `XLITE_DEBUG_POINT` | C++ macro | Print specific tensor at location |
| `debug = True` | Python model file | Enable torch_npu backend debug prints |
| `FORWARD_BACKEND` | Environment variable | Select backend: `xlite` or `torch_npu` |
| `HCCL_DETERMINISTIC` | Environment variable | Enable deterministic HCCL operations |
| `LCCL_DETERMINISTIC` | Environment variable | Enable deterministic LCCL operations |

## Common Accuracy Issues / 常见精度问题

1. **Numerical Precision Differences**
   - BF16 vs FP16 accumulation differences
   - Different rounding modes between backends

2. **Algorithm Implementation Differences**
   - Different attention kernel implementations
   - Different normalization algorithms

3. **Memory Layout Differences**
   - Tensor layout transformations
   - Buffer alignment issues

4. **Operator Fusion Differences**
   - Fused vs unfused operations
   - Different computation ordering

## Output Format / 输出格式

The skill will generate:
本skill会生成：

1. Minimal test configuration JSON
2. Test execution scripts for both backends
3. Difference analysis table
4. Root cause summary
5. Fix recommendations

## Notes / 注意事项

- Always enable `HCCL_DETERMINISTIC=true` for reproducible results
- Use minimal layers to reduce debug complexity
- Compare intermediate outputs, not just final token
- Document all differences found during the process
- For MoE models, check both dense and sparse layers