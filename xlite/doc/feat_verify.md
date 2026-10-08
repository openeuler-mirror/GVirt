# xlite × vllm-ascend 特性交互验证工具

验证 xlite（decode-only / full 两种模式）与 vllm-ascend 各特性的交互，汇总为特性支持表格；不支持的记录具体问题表现。

## 一、前置条件

- 昇腾 NPU 环境（Ascend910，已安装 CANN + vllm-ascend + xlite）
- 主模型：GLM-5.2-w8a8（MoE + W8A8 量化）
- 旁路模型：Qwen3-VL-30B-A3B-Instruct（多模态）
- 确保无残留 vllm 进程，否则显存被占会 OOM：

```bash
pkill -9 -f 'vllm.entrypoints'; pkill -9 -f 'VLLM::'; pkill -9 -f vllm_ascend; pkill -9 -f multiprocessing; sleep 5
```

## 二、快速开始（单用例）

每个用例 = 一个特性 × 一个 xlite 模式，一条命令完成全部流程
（启动服务 → 等待就绪 → 跑 3 个场景测试 → 精度校验 → 停服 → 写结果表）：

```bash
cd ./tests/e2e/
# 基本格式
python3 feat_verify.py --feature <特性名> --xlite-mode <模式> [--port 8055]
```

**两个必填参数**：
- `--feature`：特性名，共 31 个（见第三节列表，`--help` 可查）
- `--xlite-mode`：xlite 模式，`xlite_decode_only` / `xlite_full_mode`（另可选 `none` 跑无 xlite 的纯 vllm-ascend 基线对比）

**示例**：

```bash
# 测 APC × decode-only
python3 feat_verify.py --feature apc --xlite-mode xlite_decode_only

# 测 Weight NZ mode=2 × decode-only，指定端口
python3 feat_verify.py --feature weight_nz2 --xlite-mode xlite_decode_only --port 8056

# 测 CPU Binding × full_mode
python3 feat_verify.py --feature cpu_binding --xlite-mode xlite_full_mode
```

## 三、支持的特性列表

共 31 个特性（× 2 模式 = 62 用例）。特性的特有配置（additional_config、环境变量、额外 CLI、模型/tp 覆盖等）已在脚本 `FEATURES` 字典里预设好，无需手动传。

**配置原则**：baseline 仅含 `xlite_graph_config`（模式相关），不显式配置任何特性开关；各特性用例单独测非默认路径。
- **代码默认 True**（baseline 走默认即开）→ 各用例测**关闭**场景：`fuse_muls_add`、`enable_npugraph_ex`、`enable_mlapo`、`enable_cpu_binding`、`weight_nz_mode`(默认 1)、`multistream_dsv4_dsa_overlap`、`async_scheduling`、`enable_chunked_prefill`
- **代码默认 False**（baseline 不开）→ 各用例测**开启**场景：`enable_balance_scheduling`、`enable_flashcomm1`、`multistream_overlap_shared_expert`
- **APC 特例**：脚本 baseline 强制 `--no-enable-prefix-caching`（关闭），`apc` 用例改测 `--enable-prefix-caching` 开启路径

> **注**：ACLGraph Full_Decode_Only / Tensor Parallel / Expert Parallel 这 3 项配置等同 baseline（xlite decode-only 即走 FULL_DECODE_ONLY，TP/EP 是脚本默认参数），已合并到 baseline，不再单独列。Chunked Prefill（vLLM 默认开）有独立关闭用例 `chunked_prefill_off`。

| 特性名(feature) | 显示名 | 预设配置 |
|---|---|---|
| baseline | 基线(最小默认) | 仅 xlite_graph_config(模式相关)；gpu_memory_utilization=0.93(统一默认)；cudagraph_capture_sizes=[1,4,8,12,16,24,32,48,64,96] + 显式 --async-scheduling 对齐 `online_api_server.sh` |
| apc | Automatic Prefix Caching | `--enable-prefix-caching`（脚本默认关 APC，本用例测开启路径） |
| async_scheduling | Async Scheduling | `--no-async-scheduling`（vLLM 默认开，本用例测关闭场景） |
| chunked_prefill_off | Chunked Prefill(关闭) | `--no-enable-chunked-prefill`（vLLM 默认开，本用例测关闭场景；long_single 降至 4096 适配非分块 prefill） |
| short_request_first | Short Request First | scheduler_config.short_request_first_config: enabled+threshold=256+long_max_wait_ms=2000 |
| balance_scheduling | Balance Scheduling | additional_config: `enable_balance_scheduling=True`（代码默认 False，本用例测开启） |
| kv_cache_pool | KV Cache Pool | ⏭️跳过: 需 `--kv-transfer-config` + 外部 mooncake 服务，本脚本无法自动起 |
| kv_cache_cpu_offload | KV Cache CPU Offload | ⏭️跳过: OffloadingConnector + NPUOffloadingSpec, kv_role=kv_both；vllm-ascend npu.py import 路径与当前 vllm 不兼容（vllm 上游重构 v1/kv_offload），worker 报 `No module named 'vllm.v1.kv_offload.abstract'`，待 vllm-ascend 适配上游新结构 |
| piecewise | ACLGraph Piecewise | cudagraph_mode=PIECEWISE（以启动日志 effective cudagraph_mode 为准） |
| dp | Data Parallel(单机2×8) | `--data-parallel-size 2`, tp=8, max_model_len=28672(decode)/16384(full), env: XLITE_DEVS_PER_NODE=16 |
| context_parallel | Context Parallel | `--prefill-context-parallel-size 1 --decode-context-parallel-size 8`, tp=8, enable_dsa_cp+enable_flashcomm1 |
| lmhead_tp | Lmhead TP(Fine-grained TP) | finegrained_tp_config: lmhead_tp=4, dp=4/tp=4, max_model_len=3584(decode)/2048(full), max_num_batched_tokens=2048；注: full_mode max_model_len=2048 < long_single=2816，long_single 因输入超长失败（非启动问题） |
| shared_expert_dp | Shared Expert DP | additional_config: enable_shared_expert_dp（full max_model_len=16384） |
| eplb | Eplb(动态专家负载均衡) | eplb_config.dynamic_eplb + env: DYNAMIC_EPLB=true |
| eplb_recording | Eplb(Recording 录制 expert map) | eplb_config.expert_map_record_path + env: EXPERT_MAP_RECORD=true, num_redundant_experts=16 |
| eplb_static | Eplb(Static 加载预录 map) | eplb_config.expert_map_path（需先跑 eplb_recording 生成 map） |
| multistream_moe | Multistream Moe | additional_config: multistream_overlap_shared_expert |
| dsv4_dsa_overlap_off | Dsv4 Dsa Overlap(关闭) | additional_config: `multistream_dsv4_dsa_overlap=False`（代码默认 True，GLM-5.2 走 dsa_v1 注意力路径，本用例测关闭场景） |
| flashcomm1 | Flashcomm1 | additional_config: enable_flashcomm1（代码默认 False，本用例测开启） |
| mlapo | Mlapo | additional_config: enable_mlapo=False（代码默认 True，本用例测关闭场景） |
| cpu_binding | Cpu Binding | additional_config: enable_cpu_binding=False（代码默认 True，本用例测关闭场景） |
| fuse_muls_add_off | Fuse Muls Add(关闭) | additional_config: `fuse_muls_add=False`（代码默认 True，本用例测关闭场景） |
| npugraph_ex_off | Npugraph Ex(关闭) | additional_config: `ascend_compilation_config.enable_npugraph_ex=False`（代码默认 True，仅 310P 强制 False，本用例测关闭场景） |
| quant_w8a8 | Quantization W8A8 | 模型 GLM-5.2-w8a8 + `--quantization ascend` |
| quant_w4a8 | Quantization W4A8 | ⏭️跳过: 可用模型为 mixed_precision_w4 格式，非 vllm-ascend 合法值 |
| weight_nz0 | Weight NZ(关闭 mode=0) | additional_config: `weight_nz_mode=0`（代码默认 1=量化开 NZ，本用例测关闭场景；additional_config 优先级高于 env 故不用 env） |
| weight_nz2 | Weight NZ(mode=2) | additional_config: `weight_nz_mode=2`（代码默认 1，mode=2=BF16/FP16 也开 NZ，即 `online_api_server.sh` 生产配置，保留验证生产路径） |
| speculative_mtp | Speculative Decoding(MTP) | `--speculative-config '{"num_speculative_tokens":1,"method":"mtp"}'` |
| speculative_eagle3 | Eagle3 | ⏭️跳过: 需专用 eagle3 draft 模型，环境无 |
| disaggregated_prefill | Disaggregated Prefill | ⏭️跳过: 需 producer+consumer 多实例 |
| multimodal | Multimodal Inputs | 模型 Qwen3-VL-30B-A3B-Instruct, tp=8, 图片输入, `--allowed-local-media-path` |

## 四、可选参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--port` | 8055 | 服务端口 |
| `--model` | GLM-5.2-w8a8 | 主模型路径（quant_w4a8/multimodal 会自动覆盖） |
| `--tp` | 16 | tensor 并行数（dp/context_parallel/multimodal 等会自动覆盖） |
| `--max-model-len` | 20480 | 显存紧张时可调小（如 dp×full_mode 用 16384） |
| `--max-num-batched-tokens` | 8192 | |
| `--max-num-seqs` | 32 | |
| `--startup-timeout` | 1500 | 服务启动超时秒数（大模型加载慢，默认 25 分钟） |

## 五、测试场景

每个用例在同一服务实例上依次跑 3 个场景（由 feat_verify.py 内部直接发起请求）：

| 场景 | 并发 | 输出上限 | 输入填充词数 | 说明 |
|---|---|---|---|---|
| short_single | 1 | 128 | 200 | 单请求短输入短输出 |
| long_single | 1 | 2048 | 16548 | 单请求长输入长输出，触发 chunked prefill 分块 |
| multi_batch | 7 | 1024 | 5120 | 7 并发长输入，随机化避免前缀缓存，temperature=0.3 |

多模态用例用单独的图片场景（`mm_single` / `mm_multi_batch`）。

**精度校验规则**：输出须为与主题相关的有意义中文，非乱码/非退化重复；multi_batch 须 7 个响应全正确，取最差场景为综合结论。
校验等级：✅PASS > ⚠️SUSPECT（偏题/疑似异常）> ❌FAIL（乱码/退化重复/报错）。

**修改测试场景**：编辑 [feat_verify.py](../tests/e2e/feat_verify.py) 的 `TEST_SCENARIOS` 或 `MM_TEST_SCENARIOS`。

## 六、批量运行

用现成 batch 脚本，顺序跑、自动跳过已完成项、可随时中断恢复：

```bash
cd ./tests/e2e/
bash feat_verify_batch.sh
```

进度写入 `feat_verify_batch_progress.txt`。也可改脚本里的 `RUN_LIST` 选择要跑的用例子集，或自己写循环：

```bash
for feat in apc cpu_binding weight_nz2; do
  for mode in xlite_decode_only xlite_full_mode; do
    python3 feat_verify.py --feature $feat --xlite-mode $mode --port 8055
  done
done
```

## 七、结果与日志位置

- **结果表**（运行时生成，每跑一个用例自动追加一行，统一写入 feat_verify_logs/）：
  - `xlite/tests/e2e/feat_verify_logs/feat_verify.md`
  - `xlite/tests/e2e/feat_verify_logs/feat_verify_results.csv`（同内容的 CSV 版）
- **详细日志**：`xlite/tests/e2e/feat_verify_logs/<特性>_<模式>_<端口>/`

每个用例子目录包含：

| 文件 | 说明 |
|---|---|
| `server.log` | 服务完整日志 |
| `launch_cmd.txt` | 启动命令 + 环境变量 |
| `reply_<场景>.log` | 各场景的模型实际回复 |
| `scenario_summary.txt` | 三场景结论速览 |

## 八、注意事项

1. **跑前确保无残留进程**（否则显存被占，启动会 OOM）：
   ```bash
   pkill -9 -f 'vllm.entrypoints'; pkill -9 -f 'VLLM::'; pkill -9 -f vllm_ascend; pkill -9 -f multiprocessing; sleep 5
   ```
   > 注：DP 场景会残留 `VLLM::DPCoordinator` / `VLLM::EngineCore_DP*` 等进程（进程名不含 vllm.entrypoints），必须单独杀，否则占用显存致下个用例 OOM。
2. **每个用例约 10-20 分钟**（加载 5-8 分钟 + 测试 3-5 分钟 + 停服收尾 2 分钟；DP/CP 等并行类启动更慢）；单个启动失败约 5-8 分钟。
3. **dp 特性**：单机拆 DP=2×TP=8，预设 max_model_len 28672(decode)/16384(full) 适配单卡显存；DP 启动偶发 HCCL 通信错误（507018/ERR02005），重跑可成功。
4. **结果表是追加写入的**——若想全量重跑，先删掉旧结果表：
   ```bash
   rm xlite/tests/e2e/feat_verify_logs/feat_verify.md xlite/tests/e2e/feat_verify_logs/feat_verify_results.csv
   ```
5. **XLiteGraph 机制**：
   - decode-only：xlite 接管 decode，ACLGraph 走 FULL_DECODE_ONLY
   - full_mode：xlite 接管 prefill+decode，ACLGraph 被禁用走 eager（enforce_eager）
   - full_mode + speculative：仍走 FULL_DECODE_ONLY

## 九、相关文件

| 文件 | 说明 |
|---|---|
| [feat_verify.py](../tests/e2e/feat_verify.py) | 主验证脚本（参数化特性开关 + xlite 模式 + 多场景测试 + 精度校验） |
| [feat_verify_batch.sh](../tests/e2e/feat_verify_batch.sh) | 批量运行脚本 |
