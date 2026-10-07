#!/usr/bin/env python3
"""xlite × vllm-ascend 特性交互验证自动化脚本.

按用例(特性 × xlite 模式)启动 vllm 服务, 轮询就绪, 内部直接发起请求,
校验输出精度, 停服务, 把结果追加写入结果表格(Markdown + CSV). 每个用例
独立 port + log, 用完即停, 显式等待 NPU 显存释放.

用法:
  python3 feat_verify.py --feature baseline --xlite-mode xlite_decode_only --port 8055
"""

import argparse
import collections
import concurrent.futures
import csv
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field, replace

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(HERE, "feat_verify_logs")  # 每个用例一个子目录
RESULT_MD = os.path.join(LOG_ROOT, "feat_verify.md")       # 结果表统一放入 feat_verify_logs/
RESULT_CSV = os.path.join(LOG_ROOT, "feat_verify_results.csv")

MODEL_DEFAULT = "/mnt/nvme0n1/models/GLM-5.2-w8a8"
SERVED_NAME = "qwen"

# 一个有意义的、答案相对确定的问题, 便于人工判断输出是否"有意义且相关"
TEST_PROMPT = "请用中文简要介绍一下如何保持良好的睡眠质量, 列出三点建议。"

@dataclass
class FeatureSpec:
    """单个用例(特性)的配置: 启动参数/环境变量覆盖 + 备注."""
    name: str               # --feature 取值
    label: str              # 结果表显示名
    extra_config: dict = field(default_factory=dict)   # 合并入 additional_config
    extra_cli: list = field(default_factory=list)      # 追加的 CLI 参数
    extra_env: dict = field(default_factory=dict)      # 追加的环境变量
    prefix_caching: str = "off"          # "on"=--enable-prefix-caching, "off"=--no-enable-prefix-caching
    disable_chunked_prefill: bool = False  # True=--no-enable-chunked-prefill
    tp_override: int = 0                 # 0=用命令行 --tp
    model_override: str = ""             # ""=用命令行 --model
    max_model_len_override: int = 0      # 0=用命令行值
    max_model_len_full_override: int = 0  # 仅 full_mode 覆盖; 0=不覆盖
    max_num_batched_tokens_override: int = 0  # 0=用命令行值; full_mode 下缩 xlite pool
    gpu_memory_utilization_override: float = 0  # 0=用默认 0.93
    cudagraph_mode_override: str = ""   # ""=用默认 FULL_DECODE_ONLY
    disable_expert_parallel: bool = False  # True=不加 --enable-expert-parallel
    disable_quantization: bool = False   # True=不加 --quantization ascend
    multimodal: bool = False             # True=用图片场景测试
    needs_multi_instance: bool = False   # 需多实例/多节点, 脚本无法自动起
    note: str = ""                       # 备注(写入结果表)
    skip_reason: str = ""                # 非空则跳过实测(原因)
    scenario_input_override: dict = field(default_factory=dict)  # {场景: 输入词数}


FEATURES = {
    # --- 基线/调度/缓存类 ---
    "baseline": FeatureSpec(
        "baseline", "基线(最小默认)",
        note="最小默认: 仅 xlite_graph_config(模式相关)",
    ),
    "apc": FeatureSpec(
        "apc", "Automatic Prefix Caching",
        prefix_caching="on",
        note="脚本默认关闭 APC, 本用例显式开启, 验证与 xlite block 管理交互",
    ),
    "async_scheduling": FeatureSpec(
        "async_scheduling", "Async Scheduling",
        # vLLM 默认开 async_scheduling(生成式模型默认 True); 显式 --async-scheduling 与默认无差,
        # 故改测 --no-async-scheduling 关闭路径才有独立测试意义。
        extra_cli=["--no-async-scheduling"],
        note="vLLM 默认开 async_scheduling; 本用例 --no-async-scheduling 关闭, 测非异步调度与 xlite 交互"
             "(该特性交互测试测试的是关闭的场景)",
    ),
    "chunked_prefill_off": FeatureSpec(
        "chunked_prefill_off", "Chunked Prefill(关闭)",
        # enable_chunked_prefill vLLM 默认 True(scheduler.py 数据类默认); 显式开与默认无差,
        # per "默认就开启→测关闭" 原则, 改测 --no-enable-chunked-prefill 关闭路径才有独立测试意义。
        # 关闭后长输入须整段进单次 prefill(≤max_num_batched_tokens=8192), 不能分块;
        # SchedulerConfig 校验要求 max_num_batched_tokens >= max_model_len, 故 max_model_len
        # 须同步降至 <= 8192(默认 20480 会触发 validation error); long_single 降至 4096,
        # 4096+2048(output)=6144 < 8192, multi_batch 5120+1024=6144 < 8192, 均可整段进 prefill。
        disable_chunked_prefill=True,
        max_model_len_override=8192,
        scenario_input_override={"long_single": 4096},
        note="enable_chunked_prefill vLLM 默认 True; 本用例 --no-enable-chunked-prefill 关闭, "
             "测非分块 prefill 与 xlite 交互(该特性交互测试测试的是关闭的场景); "
             "关闭后长输入须整段进单次 prefill, long_single 由 16548 降至 4096 适配非分块路径",
    ),
    "short_request_first": FeatureSpec(
        "short_request_first", "Short Request First",
        # 短请求优先: prefill 阶段让 num_prompt_tokens<=threshold 的请求优先, 减少队头阻塞;
        # 配置嵌套于 scheduler_config, 不需 recompute_scheduler_enable。
        extra_config={"scheduler_config": {"short_request_first_config": {
            "enabled": True, "threshold": 256, "long_max_wait_ms": 2000}}},
        note="scheduler_config.short_request_first_config: enabled+threshold=256+long_max_wait_ms=2000; "
             "prefill 阶段短请求优先, 验证与 xlite prefill 交互",
    ),
    "balance_scheduling": FeatureSpec(
        "balance_scheduling", "Balance Scheduling",
        # enable_balance_scheduling 代码默认 False; 显式开启才是独立测试意义(baseline 不开)。
        # 顶层 additional_config 即可(ascend_config 先查 scheduler_config 再查 additional_config 顶层)。
        extra_config={"enable_balance_scheduling": True},
        note="enable_balance_scheduling=True; 代码默认 False, baseline 不开, 本用例显式开启 "
             "测负载均衡调度与 xlite 交互",
    ),
    "kv_cache_pool": FeatureSpec(
        "kv_cache_pool", "KV Cache Pool",
        skip_reason="需 --kv-transfer-config 指定 AscendStoreConnector(backend mooncake/memcache/yuanrong)"
                    " 或 MooncakeConnectorV1(MultiConnector 内) + 起外部 mooncake_master 服务; "
                    "PD-Mixed(kv_both)可单实例, PD-Disaggregation 需 producer+consumer+proxy; "
                    "本脚本无法自动起外部服务, 待手动实测",
        note="当前有效名: AscendStoreConnector(池化)+MooncakeConnectorV1(P2P, MultiConnector 内); "
             "VLLM_USE_KV_CACHE_POOL 环境变量不存在(非废弃, 系从未存在)",
    ),
    "kv_cache_cpu_offload": FeatureSpec(
        "kv_cache_cpu_offload", "KV Cache CPU Offload",
        # OffloadingConnector + NPUOffloadingSpec: 不活跃 KV 块异步卸载到 CPU, miss 时加载回 NPU;
        # kv_role=kv_both 单实例可起, 无需多节点/外部服务。
        # gpu_memory_utilization: 用统一默认 0.93(初版设 0.5 意在压小 NPU KV 池激活卸载, 但 GLM-5.2 单卡权重 47 GiB,
        # 0.5×61.27=30.6 GiB 连权重都装不下 → KV=-19.27 GiB 启动失败; @0.93 实测 KV 转正成功, 权重 47.02 GiB 可装下)。
        # 但 vllm-ascend npu.py import 路径与当前 vllm 版本不兼容(vllm 上游重构 v1/kv_offload:
        # abstract→base, mediums/spec/worker 移除), worker 报 No module named 'vllm.v1.kv_offload.abstract',
        # 启动仍失败。需 vllm-ascend 适配上游新结构, 非配置问题, 暂 skip。
        skip_reason="vllm-ascend/vllm 版本兼容: vllm_ascend/kv_offload/npu.py:7 import "
                    "'from vllm.v1.kv_offload.abstract import ...' 但 vllm 上游已重构该模块(abstract→base, "
                    "mediums/spec/worker 移除), worker 报 No module named 'vllm.v1.kv_offload.abstract'; "
                    "需 vllm-ascend 适配上游新结构后重测(非 KV cache 配置问题, @0.93 KV 已转正成功)",
        extra_cli=["--kv-transfer-config", json.dumps({
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "num_cpu_blocks": 1000,
                "block_size": 128,
                "spec_name": "NPUOffloadingSpec",
                "spec_module_path": "vllm_ascend.kv_offload.npu",
            },
        })],
        note="OffloadingConnector + NPUOffloadingSpec; kv_role=kv_both 单实例无外部服务",
    ),

    # --- 图模式类 ---
    "piecewise": FeatureSpec(
        "piecewise", "ACLGraph Piecewise",
        cudagraph_mode_override="PIECEWISE",
        note="cudagraph_mode=PIECEWISE(合法值); effective mode 可能由 platform/backend 按注意力后端兼容性调整, "
             "以启动日志 effective cudagraph_mode 为准验证",
    ),

    # --- 并行类 ---
    "dp": FeatureSpec(
        "dp", "Data Parallel(单机2×8)",
        extra_cli=["--data-parallel-size", "2"],
        tp_override=8,
        max_model_len_override=28672,
        max_model_len_full_override=16384,
        scenario_input_override={"long_single": 14016},
        extra_env={"XLITE_DEVS_PER_NODE": "16"},
        note="单机16卡拆 DP=2×TP=8, set XLITE_DEVS_PER_NODE=16; "
             "在线 CLI DP 路径 vLLM 自动分配 master port, 无需 VLLM_DP_MASTER_IP/PORT"
             "(仅 large_scale_ep 分布式 DP server/offline SPMD 才需); "
             "full_mode max_model_len=16384, 默认 long_single input=16548 > 16384 → HTTP 400 input-too-long; "
             "long_single 降至 14016(+2048out+chat template≈256=16320<=16384)",
    ),
    "context_parallel": FeatureSpec(
        "context_parallel", "Context Parallel",
        extra_cli=["--prefill-context-parallel-size", "1",
                   "--decode-context-parallel-size", "8"],
        tp_override=8,
        extra_config={"enable_dsa_cp": True, "enable_flashcomm1": True},
        note="GLM-5.2(SFA) 要求 decode_context_parallel_size==tensor_parallel_size, prefill CP 不支持(必须为1); "
             "DCP=8==TP=8; enable_dsa_cp 依赖 FlashComm1, 需同时开; "
             "full_mode 下被 xlite 接管 prefill 覆盖, 主要 decode-only 生效",
    ),
    "lmhead_tp": FeatureSpec(
        "lmhead_tp", "Lmhead TP(Fine-grained TP)",
        extra_config={"finegrained_tp_config": {
            "lmhead_tensor_parallel_size": 4}},
        extra_cli=["--data-parallel-size", "4"],
        tp_override=4,
        max_model_len_override=3584,
        max_model_len_full_override=5120,
        max_num_batched_tokens_override=2048,
        extra_env={"XLITE_DEVS_PER_NODE": "16"},
        scenario_input_override={
            "short_single": 128,
            "long_single": 1472,
            "multi_batch": 2496,
        },
        note="finegrained_tp_config: lmhead TP=4; 需DP场景(dp=4,tp=4,dp_size%lmhead_tp==0); "
             "dp=4,tp=4 显存极紧张, max_model_len=3584(decode)/5120(full); "
             "full_mode 缩 max_num_batched_tokens=2048 + 默认 util 0.93(KV 1.94GiB 启动); "
             "input_length 须满足 input+max_tokens<=max_model_len, 否则 HTTP 400 input-too-long; "
             "decode: long_single=1472(+2048out=3520<=3584)/multi_batch=2496(+1024=3520<=3584); "
             "full: max_model_len 升至 5120( KV 1.94GiB 容 long_single 4096 token; "
             "multi_batch 7并发峰值超 KV pool, vLLM 自动排队不 OOM); "
             "input 两模式共用: long_single=1472/multi_batch=2496(均 <= 各模式 max_model_len)",
    ),
    "shared_expert_dp": FeatureSpec(
        "shared_expert_dp", "Shared Expert DP",
        extra_config={"enable_shared_expert_dp": True},
        # full_mode tp=16 下 KV cache 仅 2.0GiB, 需降 max_model_len
        max_model_len_full_override=16384,
        scenario_input_override={
            "long_single": 14016,
        },
        note="enable_shared_expert_dp=true; 适用于带共享专家的 MoE 模型(GLM-5.2 属此类); "
             "full_mode KV cache 仅2.0GiB, max_model_len 降至 16384; "
             "long_single input=14016(+2048out+chat template≈256=16320<=16384, 避免 input-too-long 400)",
    ),

    # --- MoE/专家类 ---
    "eplb": FeatureSpec(
        "eplb", "Eplb(动态专家负载均衡)",
        # num_redundant_experts 须显式配(=16, 与 recording/map 一致): 上游 factory 据此扩容 allocated,
        # 不配则 allocated=16 ≠ map placement=17 触发 "EPLB local expert capacity mismatch" assert.
        # expert_map_path 复用 recording 产出的初始分布(生产场景: dynamic 在预录分布上 rebalance).
        extra_config={"eplb_config": {
            "dynamic_eplb": True,
            "num_redundant_experts": 16,
            "expert_map_path": os.path.join(HERE, "feat_verify_logs", "eplb_map.json"),
            "expert_heat_collection_interval": 400,
            "algorithm_execution_interval": 30}},
        extra_env={"DYNAMIC_EPLB": "true"},
        note="dynamic_eplb=true + num_redundant_experts=16 + expert_map_path(预录初始分布) + env DYNAMIC_EPLB=true; "
             "验证动态 EPLB(运行时 SwiftBalance rebalance)与 xlite MoE 交互; 需先跑 eplb_recording 生成 map",
    ),
    "eplb_recording": FeatureSpec(
        "eplb_recording", "Eplb(Recording 录制 expert map)",
        # Recording: 生成初始专家分布 map 到 expert_map_record_path, 供 eplb_static 加载;
        # 需 env EXPERT_MAP_RECORD=true 作 safety guard(config 单独不够)。
        extra_config={"eplb_config": {
            "expert_map_record_path": os.path.join(HERE, "feat_verify_logs", "eplb_map.json"),
            "num_redundant_experts": 16,
            "expert_heat_collection_interval": 400,
            "algorithm_execution_interval": 30}},
        extra_env={"EXPERT_MAP_RECORD": "true"},
        note="expert_map_record_path + env EXPERT_MAP_RECORD=true; 录制 expert map 到 feat_verify_logs/eplb_map.json, "
             "供 eplb_static 加载; num_redundant_experts=16",
    ),
    "eplb_static": FeatureSpec(
        "eplb_static", "Eplb(Static 加载预录 expert map)",
        # Static: 加载预录 map, 无需 env; 依赖 eplb_recording 产出的 map 文件。
        # num_redundant_experts 须与 recording 生成 map 时一致(=16): 上游 factory 据此扩容 allocated,
        # 不配则触发 "EPLB local expert capacity mismatch" assert (与 xlite 无关).
        extra_config={"eplb_config": {
            "expert_map_path": os.path.join(HERE, "feat_verify_logs", "eplb_map.json"),
            "num_redundant_experts": 16}},
        note="expert_map_path + num_redundant_experts(须与录制时一致); 加载预录 expert map, 需先跑 eplb_recording 生成 feat_verify_logs/eplb_map.json",
    ),
    "multistream_moe": FeatureSpec(
        "multistream_moe", "Multistream Moe",
        # multistream_overlap_shared_expert 代码默认 False; 显式开启才是独立测试意义(baseline 不开)。
        extra_config={"multistream_overlap_shared_expert": True},
        note="multistream_overlap_shared_expert=True; 代码默认 False, baseline 不开, 本用例显式开启 "
             "测 MoE 多流共享专家重叠与 xlite 交互",
    ),
    "dsv4_dsa_overlap_off": FeatureSpec(
        "dsv4_dsa_overlap_off", "Dsv4 Dsa Overlap(关闭)",
        # multistream_dsv4_dsa_overlap 代码默认 True(ascend_config.py, GLM-5.2 走 dsa_v1 注意力路径,
        # 该开关在 decode/prefill 多处直接判定非仅 compress_ratio=4 分支); per "默认就开启→测关闭" 原则,
        # 显式关闭才是独立测试意义(baseline 走默认 True 不显式配置)。
        extra_config={"multistream_dsv4_dsa_overlap": False},
        note="multistream_dsv4_dsa_overlap=False; 代码默认 True(ascend_config.py, GLM-5.2 DSA 注意力路径), "
             "baseline 不显式配置(走默认 True), 本用例显式关闭测 DSA 非多流重叠路径与 xlite 交互"
             "(该特性交互测试测试的是关闭的场景)",
    ),

    # --- 通信/优化类 ---
    "flashcomm1": FeatureSpec(
        "flashcomm1", "Flashcomm1",
        # enable_flashcomm1 代码默认 False; 显式开启才是独立测试意义(baseline 不开)。
        extra_config={"enable_flashcomm1": True},
        note="enable_flashcomm1=True; 代码默认 False, baseline 不开, 本用例显式开启",
    ),
    "mlapo": FeatureSpec(
        "mlapo", "Mlapo",
        # enable_mlapo 代码默认 True; 显式设 True 与默认无差, 故改测关闭路径。
        extra_config={"enable_mlapo": False},
        note="enable_mlapo 代码默认 True; 本用例改测关闭路径 enable_mlapo=False"
             "(该特性交互测试测试的是关闭的场景)",
    ),
    "cpu_binding": FeatureSpec(
        "cpu_binding", "Cpu Binding",
        # enable_cpu_binding 代码默认 True; 显式设 True 与默认无差, 故改测关闭路径。
        extra_config={"enable_cpu_binding": False},
        note="enable_cpu_binding 代码默认 True; 本用例改测关闭路径 enable_cpu_binding=False"
             "(该特性交互测试测试的是关闭的场景)",
    ),
    "fuse_muls_add_off": FeatureSpec(
        "fuse_muls_add_off", "Fuse Muls Add(关闭)",
        # fuse_muls_add 代码默认 True(AscendFusionConfig.fuse_muls_add kwargs 默认 True); per
        # "默认就开启→测关闭" 原则, 显式关闭才是独立测试意义(baseline 走默认 True 不显式配置)。
        extra_config={"fuse_muls_add": False},
        note="fuse_muls_add=False; 代码默认 True, baseline 不显式配置(走默认 True), "
             "本用例显式关闭测非融合 muls+add 路径与 xlite 交互(该特性交互测试测试的是关闭的场景)",
    ),
    "npugraph_ex_off": FeatureSpec(
        "npugraph_ex_off", "Npugraph Ex(关闭)",
        # enable_npugraph_ex 代码默认 True(AscendCompilationConfig 构造参数默认 True; 仅 310P 强制 False);
        # per "默认就开启→测关闭" 原则, 显式关闭才是独立测试意义.
        # 嵌套于 ascend_compilation_config, 经 extra_config 提供整个子 dict 覆盖(同 quant_w8a8 等结构)。
        extra_config={"ascend_compilation_config": {"enable_npugraph_ex": False}},
        note="ascend_compilation_config.enable_npugraph_ex=False; 代码默认 True(仅 310P 强制 False), "
             "baseline 不显式配置(走默认 True), 本用例显式关闭测无 npugraph_ex 路径"
             "(该特性交互测试测试的是关闭的场景); "
             "注: enable_static_kernel 依赖 npugraph_ex, 本用例未开 static_kernel 故无 assert 冲突",
    ),

    # --- 量化类 ---
    "quant_w8a8": FeatureSpec(
        "quant_w8a8", "Quantization W8A8",
        model_override="/mnt/nvme0n1/models/GLM-5.2-w8a8",
        note="GLM-5.2-w8a8 即 W8A8(含 quant_model_description.json, 自动 --quantization ascend)",
    ),
    "quant_w4a8": FeatureSpec(
        "quant_w4a8", "Quantization W4A8",
        model_override="/mnt/nvme0n1/models/GLM-5.2-W4A8",
        skip_reason="可用 GLM-5.2-W4A8 为 mixed_precision_w4 格式(config.json, 非 vllm-ascend 合法值), "
                    "无 quant_model_description.json → 既非 ModelSlim(ascend) 也非 compressed-tensors; "
                    "强制 --quantization ascend 会走期望读取描述文件的 ModelSlim 路径而失败; "
                    "官方 W4A8 为 GLM-5.2-w4a8c8(ModelSlim), 待获取正确格式模型后实测",
        note="config.json quant_method=mixed_precision_w4 + 无 quant_model_description.json, "
             "非 vllm-ascend 支持的量化格式",
    ),
    "weight_nz0": FeatureSpec(
        "weight_nz0", "Weight NZ(关闭 mode=0)",
        # weight_nz_mode 代码默认 1(量化模型开 NZ, 即默认开启); per "默认就开启→测关闭" 原则, 本用例测关闭路径。
        # additional_config 优先级高于 env(见 ascend_config._get_config_value), 故经 extra_config 下发。
        extra_config={"weight_nz_mode": 0},
        note="weight_nz_mode=0; 代码默认 1(量化开 NZ, 默认开启), baseline 不显式配置(走默认 1), "
             "本用例显式关闭测无 NZ 路径与 xlite 量化交互(该特性交互测试测试的是关闭的场景)",
    ),
    "weight_nz2": FeatureSpec(
        "weight_nz2", "Weight NZ(mode=2)",
        # mode=2 非"关闭路径", 而是 online_api_server.sh 生产配置(BF16/FP16 也开 NZ); 保留以验证生产路径。
        extra_config={"weight_nz_mode": 2},
        note="weight_nz_mode=2; 代码默认 1(仅量化), mode=2 = BF16/FP16 也开 NZ, "
             "即 online_api_server.sh 生产配置(非默认启用更多路径); 保留验证生产 NZ 路径",
    ),

    # --- 投机解码类 ---
    "speculative_mtp": FeatureSpec(
        "speculative_mtp", "Speculative Decoding(MTP)",
        extra_cli=["--speculative-config",
                   '{"num_speculative_tokens": 1, "method": "mtp"}'],
        note="GLM-5-w8a8 自带 MTP(num_nextn_predict_layers=1); method=mtp"
             "(deepseek_mtp 为 vLLM 上游已 deprecated 旧写法, 运行时归一化为 mtp); "
             "full_mode+speculative 走 FULL_DECODE_ONLY",
    ),
    "speculative_eagle3": FeatureSpec(
        "speculative_eagle3", "Eagle3",
        skip_reason="缺少 eagle3 draft 模型(Eagle3 为单实例特性, 文档无多实例要求; "
                    "待补充 draft 模型后用 --speculative-config "
                    "{method:eagle3, model:<draft>, draft_tensor_parallel_size:1} 实测)",
        note="Eagle3 需专用 draft 模型(--speculative-config 的 model 字段 required); "
             "单实例 target+draft 同进程, 非多实例场景",
    ),

    # --- Disaggregated 类 ---
    "disaggregated_prefill": FeatureSpec(
        "disaggregated_prefill", "Disaggregated Prefill",
        needs_multi_instance=True,
        note="PD 分离需 producer+consumer 两实例 + kv_transfer_config; 单机可起2实例但复杂, 暂标记",
    ),

    # --- 多模态类 ---
    "multimodal": FeatureSpec(
        "multimodal", "Multimodal Inputs",
        # Qwen3-VL-32B-Instruct(dense) 不存在, 改用 Qwen3-VL-30B-A3B-Instruct(MoE);
        # 替代模型为 MoE, 不再禁用 expert_parallel(dense 才须禁用)
        model_override="/mnt/nvme0n1/models/Qwen3-VL-30B-A3B-Instruct",
        tp_override=8,
        multimodal=True,
        max_model_len_override=16384,
        disable_quantization=True,
        extra_cli=["--allowed-local-media-path",
                   "/mnt/nvme1n1/models/DeepSeek-V3.2/assets",
                   "--limit-mm-per-prompt.image", "1",
                   "--limit-mm-per-prompt.video", "0",
                   "--mm-processor-cache-gb", "0"],
        extra_env={"OMP_NUM_THREADS": "1"},
        note="Qwen3-VL-30B-A3B-Instruct(MoE, 替代不存在的 32B); 图片输入测试多模态与 xlite 交互; "
             "tp=8 + max_model_len=16384 显存紧张, 设 OMP_NUM_THREADS=1 节省显存",
    ),
}


def case_dir(tag: str) -> str:
    """获取/创建某个用例的日志子目录 feat_verify_logs/<tag>/."""
    d = os.path.join(LOG_ROOT, tag)
    os.makedirs(d, exist_ok=True)
    return d


def base_env_init() -> dict:
    """环境变量初始化(vllm-ascend 特性开关走 additional_config, 不在此)."""
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["TASK_QUEUE_ENABLE"] = "1"
    env["HCCL_BUFFSIZE"] = "512"
    env["HCCL_OP_EXPANSION_MODE"] = "AIV"
    env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    env["OMP_PROC_BIND"] = "false"
    # XCCL: NPU-SMI >=25.3 不禁用
    env.setdefault("XLITE_DISABLE_XCCL", "False")
    return env


def build_additional_config(mode: str, spec: FeatureSpec) -> dict:
    """构建 additional_config.
    最小默认(baseline): 仅 xlite_graph_config(模式相关). 不显式配置额外参数
    """
    base = {}
    if mode != "none":
        base["xlite_graph_config"] = {"enabled": True}
        if mode == "xlite_full_mode":
            base["xlite_graph_config"]["full_mode"] = True
    if spec.extra_config:
        base.update(spec.extra_config)
    return base


def build_launch_cmd(spec: FeatureSpec, mode: str, model_path: str, port: int,
                    tp: int, max_num_batched_tokens: int,
                    max_num_seqs: int, max_model_len: int) -> list:
    additional_config = build_additional_config(mode, spec)
    if spec.model_override:
        model_path = spec.model_override
    if spec.tp_override:
        tp = spec.tp_override
    if spec.max_num_batched_tokens_override:
        max_num_batched_tokens = spec.max_num_batched_tokens_override
    if spec.max_model_len_override:
        max_model_len = spec.max_model_len_override
    gpu_mem_util = spec.gpu_memory_utilization_override or 0.93
    # full_mode 专用覆盖优先(显存紧张时仅 full_mode 需降)
    if mode == "xlite_full_mode" and spec.max_model_len_full_override:
        max_model_len = spec.max_model_len_full_override
    # cudagraph_capture_sizes 对齐 online_api_server.sh
    cudagraph_sizes = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96]
    cli = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        f"--max-num-seqs={max_num_seqs}",
        "--block-size", "128",
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        f"--served-model-name={SERVED_NAME}",
        "--additional-config", json.dumps(additional_config),
        "--compilation-config",
        json.dumps({"cudagraph_capture_sizes": cudagraph_sizes,
                    "cudagraph_mode": spec.cudagraph_mode_override or "FULL_DECODE_ONLY"}),
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    # 显式 --async-scheduling(同 online_api_server.sh; vLLM 默认即开, 显式仅为对齐).
    # async_scheduling 用例经 extra_cli 加 --no-async-scheduling 测关闭路径, 跳过此默认避免 flag 冲突.
    if "--no-async-scheduling" not in spec.extra_cli:
        cli.append("--async-scheduling")
    if spec.prefix_caching == "on":
        cli.append("--enable-prefix-caching")
    else:
        cli.append("--no-enable-prefix-caching")
    if spec.disable_chunked_prefill:
        cli.append("--no-enable-chunked-prefill")
    if not spec.disable_expert_parallel:
        cli.append("--enable-expert-parallel")
    # quant_model_description.json 存在则加 --quantization ascend
    if not spec.disable_quantization and os.path.exists(
            os.path.join(model_path, "quant_model_description.json")):
        cli += ["--quantization", "ascend"]
    cli += spec.extra_cli
    return cli


def wait_for_server(port: int, timeout: int, log_path: str,
                    proc: subprocess.Popen = None) -> tuple:
    """轮询 /v1/models; 返回 (ok, detail). 若 proc 已退出则提前返回失败."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    last_err = ""
    dead_count = 0
    while time.time() < deadline:
        # 若服务进程已退出, 不必等到超时; 但启动初期会 fork 子进程, 给点宽限
        if proc is not None and proc.poll() is not None:
            dead_count += 1
            if dead_count >= 3:  # 连续3次检测到主进程退出(约45s)才认定
                return False, "服务进程已退出(启动失败)"
        else:
            dead_count = 0
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    return True, "ready"
        except Exception as e:
            last_err = str(e)[:200]
        time.sleep(15)
    return False, last_err or "timeout"


@dataclass
class Scenario:
    name: str
    num_requests: int = 1          # 并发请求数
    max_tokens: int = 32           # 最大输出 token
    input_length: int = 0          # 输入填充到指定词数(0=不填充)
    randomize: bool = False        # 加随机 seed system 消息, 避免前缀缓存
    same_length: bool = False      # randomize 时保持 seed 长度一致
    system_prompt: bool = False    # 是否加 system prompt
    image: str = ""                # 多模态图片路径; 空=纯文本


# 测试场景: 短/长输入 × 单/多 batch, 覆盖 chunked prefill / 并发 / 前缀缓存规避
TEST_SCENARIOS = [
    Scenario("short_single", num_requests=1, max_tokens=128, input_length=200),
    Scenario("long_single", num_requests=1, max_tokens=2048, input_length=16548),
    Scenario("multi_batch", num_requests=7, max_tokens=1024, input_length=5120,
             randomize=True, same_length=True),
]

# 多模态场景: 图片+文字, 单请求与多并发
MM_TEST_IMAGE = "/mnt/nvme1n1/models/DeepSeek-V3.2/assets/benchmark.png"
MM_TEST_SCENARIOS = [
    Scenario("mm_single", num_requests=1, max_tokens=256, system_prompt=True,
             image=MM_TEST_IMAGE),
    Scenario("mm_multi_batch", num_requests=4, max_tokens=128, randomize=True,
             same_length=True, system_prompt=True, image=MM_TEST_IMAGE),
]

VERDICT_RANK = {"PASS": 0, "SUSPECT": 1, "FAIL": 2}


def send_requests(port: int, model: str, prompt: str, sc: Scenario,
                  temperature: float = 0.3, timeout: float = 900.0) -> tuple:
    """向服务并发发送 sc.num_requests 个 chat 请求(纯标准库 urllib + 线程池).

    返回 (blocks, error): blocks 为 [(idx, total, prompt, in_tok, out_tok, result), ...];
    error 非空表示整体超时(已完成的块仍返回). 单个请求异常记为 out_tok=0 的块.
    """
    total = sc.num_requests
    padded = prompt  # 填充到目标词数
    if sc.input_length > 0:
        p_len = len(prompt.strip().split())
        if p_len < sc.input_length:
            filler = " This is filler text to increase the input length for testing purposes."
            filler_len = len(filler.strip().split()) + 1
            padded = prompt + filler * ((sc.input_length - p_len) // filler_len + 1)

    base_url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def do_one(idx: int) -> tuple:
        messages = []
        if sc.system_prompt:
            messages.append({"role": "system", "content":
                "You are a helpful assistant that answers the question based on the provided text and "
                "possibly, image content. If the text is not related to the image, just focus on the text. "
                "Always think first before answering, and provide a detailed response."})
        if sc.randomize:
            if sc.same_length:
                s1, s2 = random.randint(1_000_000, 9_999_999), f"{idx + 1:0{len(str(total))}d}"
            else:
                s1, s2 = random.randint(0, 2**63 - 1), "-".join([str(idx)] * (idx + 1))
            messages.insert(0, {"role": "system", "content":
                f"{s1} and {s2} are your random seeds. "
                f"You must factor them into your answer in a creative way, but they are not the main focus."})
        content = ([{"type": "text", "text": padded},
                    {"type": "image_url", "image_url": {"url": f"file://{sc.image}"}}]
                   if sc.image else padded)
        messages.append({"role": "user", "content": content})
        body = json.dumps({"model": model, "messages": messages,
                           "max_tokens": sc.max_tokens,
                           "temperature": temperature}).encode("utf-8")
        req = urllib.request.Request(base_url, data=body, headers={
            "Content-Type": "application/json", "Authorization": "Bearer master-key"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            result = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage") or {}
            return (idx, total, padded,
                    int(usage.get("prompt_tokens", 0) or 0),
                    int(usage.get("completion_tokens", 0) or 0),
                    str(result).strip())
        except Exception as e:
            return idx, total, padded, 0, 0, f"<request error: {e}>"

    blocks = []
    error = ""
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(total, 1))
    futs = {ex.submit(do_one, i): i for i in range(total)}
    try:
        for fut in concurrent.futures.as_completed(futs, timeout=timeout):
            blocks.append(fut.result())
    except concurrent.futures.TimeoutError:
        error = "TIMEOUT"
        for fut, idx in futs.items():
            if not fut.done():
                blocks.append((idx, total, padded, 0, 0, "TIMEOUT"))
    finally:
        ex.shutdown(wait=False)
    blocks.sort(key=lambda b: b[0])
    return blocks, error


def run_one_scenario(port: int, sc: Scenario, tag: str,
                     prompt: str = TEST_PROMPT) -> tuple:
    """跑单个场景: 并发发请求, 回复写入 reply_<name>.log; 返回 (blocks, error)."""
    cdir = case_dir(tag)
    out_path = os.path.join(cdir, f"reply_{sc.name}.log")
    blocks, error = send_requests(port, SERVED_NAME, prompt, sc,
                                  temperature=0.3, timeout=900.0)
    lines = []
    for idx, total, p, in_tok, out_tok, result in blocks:
        model_repr = f" {SERVED_NAME} "
        if total > 1:
            model_repr = f"{model_repr}({idx + 1}/{total}) "
        shown = result if len(result) <= 10000 else \
            f"{result[:5000]}\n... [truncated] ...\n{result[-5000:]}"
        center = f" {in_tok} tks in → {out_tok} tks out "
        lines += [f"{model_repr:=^50}",
                  f"Prompt: {p}",
                  f"Tokens: {in_tok} tokens in → {out_tok} tokens out",
                  "-" * 50,
                  shown,
                  f"{center:=^50}"]
    text = "\n".join(lines) if blocks else (error or "(无响应)")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[场景 {sc.name}] 回复已保存 {out_path}")
    return blocks, error


def run_test_scenarios(port: int, tag: str, spec: FeatureSpec = None) -> tuple:
    """跑全部测试场景; 返回 (all_ok, list_of (scenario, verdict, evidence))."""
    results = []
    all_ok = True
    is_mm = bool(spec and spec.multimodal)
    scenarios = MM_TEST_SCENARIOS if is_mm else TEST_SCENARIOS
    prompt = "请描述这张图片的内容" if is_mm else TEST_PROMPT
    for sc in scenarios:
        # 显存受限时缩短长输入, 避免 input-too-long
        if spec and spec.scenario_input_override and sc.name in spec.scenario_input_override:
            sc = replace(sc, input_length=spec.scenario_input_override[sc.name])
            print(f"[场景 {sc.name}] 输入长度覆盖为 -L {sc.input_length}")
        is_multi = sc.num_requests > 1
        blocks, error = run_one_scenario(port, sc, tag, prompt=prompt)
        if error and not blocks:
            all_ok = False
            results.append((sc.name, "FAIL", f"无响应: {error}"))
            continue
        verdict, evidence = check_output_quality(
            blocks, multi=is_multi, multimodal=is_mm)
        if error:
            all_ok = False
            results.append((sc.name, "FAIL", f"部分请求超时({error}); {evidence}"))
            continue
        results.append((sc.name, verdict, evidence))
    return all_ok, results


def aggregate_verdicts(results: list) -> tuple:
    """综合多场景结果取最差 verdict; 返回 (总verdict, 汇总evidence)."""
    if not results:
        return "FAIL", "无场景结果"
    worst = "PASS"
    for _name, v, _ev in results:
        if VERDICT_RANK.get(v, 2) > VERDICT_RANK.get(worst, 0):
            worst = v
    parts = [f"{n}:{v}({ev[:40]})" for n, v, ev in results]
    # 用 / 分隔场景, 避免 | 与 markdown 表格列符冲突
    return worst, " / ".join(parts)



def _judge_block(block: tuple, multimodal: bool = False) -> tuple:
    """校验单个响应块; block=(idx, total, prompt, in_tok, out_tok, result); 返回 (verdict, evidence)."""
    _, _, _, _, out_tok, result_text = block
    preview = result_text[:60].replace("\n", " ").replace("\r", " ")  # 折叠换行避免拆断表格行
    if out_tok == 0:
        return "FAIL", "输出0 token"
    if not result_text:
        return "FAIL", "未提取到结果文本"
    # 乱码: 非可打印字符占比高
    non_printable = sum(1 for c in result_text
                        if not c.isprintable() and c not in "\n\t ")
    if non_printable / max(len(result_text), 1) > 0.1:
        return "FAIL", "疑似乱码, 非可打印字符占比高"
    # 单行退化重复: 字符种类过少
    for ln in result_text.splitlines():
        if len(ln) > 12 and len(set(ln)) < max(len(ln) // 10, 3):
            return "FAIL", f"疑似单行退化重复: {ln[:50]}"
    # 跨行退化重复: 句号/换行切分片段重复>=5次
    segs = []
    for line in result_text.splitlines():
        for s in line.replace("。", "\n").split("\n"):
            s = s.strip()
            if len(s) >= 8:
                segs.append(s)
    if segs:
        most_seg, most_cnt = collections.Counter(segs).most_common(1)[0]
        if most_cnt >= 5:
            return "FAIL", f"疑似退化重复: 片段重复{most_cnt}次: {most_seg[:40]}"
    # 主题相关性关键词
    if multimodal:
        kw = ["图", "图示", "图表", "柱", "坐标", "轴", "曲线", "数据",
              "对比", "性能", "显示", "内容", "可以看到", "图中"]
    else:
        kw = ["睡", "作息", "规律", "放松", "环境", "床", "避免", "建议",
              "保持", "良好", "习惯", "午", "咖啡", "屏幕", "安静", "黑暗",
              "运动", "入眠"]
    if not any(k in result_text for k in kw):
        return "SUSPECT", f"未检测到主题关键词; 前60字: {preview}"
    return "PASS", f"tokens={out_tok}; 前60字: {preview}"


def check_output_quality(blocks: list, multi: bool = False,
                         multimodal: bool = False) -> tuple:
    """校验输出是否有意义(非乱码/非空/与主题相关).

    multi=True 时多块逐个校验, 取最差 verdict.
    blocks 为 [(idx, total, prompt, in_tok, out_tok, result), ...].
    返回 (verdict, evidence). verdict: PASS/SUSPECT/FAIL
    """
    if not blocks:
        return "FAIL", "无响应块"
    verdicts = []
    for idx, block in enumerate(blocks):
        v, ev = _judge_block(block, multimodal=multimodal)
        verdicts.append((idx, v, ev))
    worst = "PASS"
    for _i, v, _ev in verdicts:
        if VERDICT_RANK.get(v, 2) > VERDICT_RANK.get(worst, 0):
            worst = v
    if multi:
        n = len(verdicts)
        bad = [i for i, v, _ in verdicts if v != "PASS"]
        summary = f"{n}个响应, 最差={worst}, 非PASS={bad}"
    else:
        summary = verdicts[0][2]
    return worst, summary


def kill_server(proc: subprocess.Popen, port: int):
    """停服务: 先 SIGTERM 主进程组, 再杀占用端口的残留进程."""
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(10):
            if proc.poll() is not None:
                break
            time.sleep(2)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    # 残留 vllm worker 进程清理
    time.sleep(5)
    subprocess.run("pkill -9 -f 'vllm.entrypoints.openai.api_server' || true",
                   shell=True)
    subprocess.run("pkill -9 -f 'vllm_ascend' || true", shell=True)
    # DP 场景会残留 VLLM::DPCoordinator / VLLM::EngineCore_DP* 等进程,
    # 进程名不含 vllm.entrypoints, 需单独清理, 否则占用显存致下个用例 OOM
    subprocess.run("pkill -9 -f 'VLLM::' || true", shell=True)
    subprocess.run("pkill -9 -f 'multiprocessing.resource_tracker' || true",
                   shell=True)
    subprocess.run("pkill -9 -f 'multiprocessing.spawn' || true", shell=True)
    # 等待显存释放
    for _ in range(12):
        time.sleep(10)


def append_result(spec: FeatureSpec, mode: str, port: int,
                  startup_ok: bool, quality_verdict: str, evidence: str,
                  elapsed: float, note_extra: str, log_subdir: str = ""):
    # Markdown
    md_header = "# xlite × vllm-ascend 特性交互验证结果\n\n"
    md_header += ("| 特性 | xlite模式 | 启动是否成功 | 输出精度校验 | "
                  "问题表现/备注 | 日志目录 | 耗时(分钟) |\n")
    md_header += "|---|---|---|---|---|---|---|\n"
    mode_cn = {"xlite_decode_only": "decode-only",
               "xlite_full_mode": "full_mode",
               "none": "none(无xlite)"}[mode]
    log_ref = os.path.relpath(log_subdir, HERE) if log_subdir else ""
    if quality_verdict == "SKIP":
        # 环境不支持/需多实例, 标为跳过
        qual = "N/A(跳过)"
        prob = evidence[:300]
        start_col = "⏭️跳过"
    elif not startup_ok:
        qual = "N/A"
        prob = f"服务启动失败: {evidence[:300]}"
        start_col = "❌失败"
    else:
        qual = quality_verdict
        prob = evidence if quality_verdict != "PASS" else "无"
        if note_extra:
            prob = (prob + "; " + note_extra) if prob != "无" else note_extra
        start_col = "✅成功"
    # markdown 表格 cell 不能含裸换行/|: 转义避免拆断表格行/列
    prob = prob.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    row = (f"| {spec.label} | {mode_cn} | "
           f"{start_col} | {qual} | "
           f"{prob} | {log_ref} | {elapsed/60:.1f} |\n")
    os.makedirs(os.path.dirname(RESULT_MD), exist_ok=True)
    if not os.path.exists(RESULT_MD):
        with open(RESULT_MD, "w", encoding="utf-8") as f:
            f.write(md_header + row)
    else:
        with open(RESULT_MD, "a", encoding="utf-8") as f:
            f.write(row)
    # CSV
    new = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["feature", "xlite_mode", "startup_ok", "quality",
                        "evidence", "log_dir", "elapsed_min"])
        w.writerow([spec.name, mode_cn, startup_ok, quality_verdict,
                    evidence[:500], log_ref, f"{elapsed/60:.1f}"])
    print(f"[结果已记录] {spec.label} × {mode_cn}: "
          f"{qual if startup_ok else '启动失败'} (日志: {log_ref})")


def run_case(spec: FeatureSpec, mode: str, port: int, model_path: str,
             tp: int, max_num_batched_tokens: int, max_num_seqs: int,
             max_model_len: int, startup_timeout: int):
    print(f"\n{'='*70}\n[用例] {spec.label} × {mode} (port={port})\n{'='*70}")
    tag = f"{spec.name}_{mode}_{port}"
    cdir = case_dir(tag)
    # 跳过: 环境不支持/模型缺失/需多实例等(skip_reason 给原因; needs_multi_instance 标多实例)
    skip_msg = spec.skip_reason
    if not skip_msg and spec.needs_multi_instance:
        skip_msg = ("该特性需多实例/多节点部署(producer+consumer 或多 DP rank), "
                    "本脚本无法自动起, 待后续手动验证")
    if skip_msg:
        with open(os.path.join(cdir, "notes.txt"), "w", encoding="utf-8") as f:
            f.write(skip_msg + "\n" + spec.note)
        append_result(spec, mode, port, False, "SKIP",
                      skip_msg + "; " + spec.note, 0.0, spec.note, cdir)
        print(f"[跳过] {spec.label}: {skip_msg}")
        return False
    log_path = os.path.join(cdir, "server.log")
    env = base_env_init()
    env.update(spec.extra_env)
    cmd = build_launch_cmd(spec, mode, model_path, port, tp,
                           max_num_batched_tokens, max_num_seqs, max_model_len)
    with open(os.path.join(cdir, "launch_cmd.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n[env diff]\n" + "\n".join(
            f"{k}={v}" for k, v in sorted(env.items())
            if k not in os.environ or str(env[k]) != str(os.environ.get(k))))
    print("[启动命令]", " ".join(cmd[:6]), "...", " ".join(cmd[-4:]))
    print(f"[日志目录] {cdir}")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    startup_ok = False
    evidence = ""
    quality = "N/A"
    try:
        ok, detail = wait_for_server(port, startup_timeout, log_path, proc)
        if ok:
            startup_ok = True
            evidence = "服务就绪"
        else:
            # 启动失败/超时, 读取日志关键错误
            evidence = extract_fatal(log_path) or detail or "超时/未就绪"
            startup_ok = False
        if startup_ok:
            all_ok, results = run_test_scenarios(port, tag, spec=spec)
            if all_ok:
                quality, evidence = aggregate_verdicts(results)
            else:
                quality = "FAIL"
                evidence = aggregate_verdicts(results)[1]
            # 把各场景明细写入用例子目录
            with open(os.path.join(cdir, "scenario_summary.txt"), "w",
                      encoding="utf-8") as f:
                for name, v, ev in results:
                    f.write(f"[{name}] {v}: {ev}\n")
    finally:
        elapsed = time.time() - start
        kill_server(proc, port)
    append_result(spec, mode, port, startup_ok, quality, evidence,
                  elapsed, spec.note, cdir)
    return startup_ok


def extract_fatal(log_path: str) -> str:
    """从日志提取致命错误信息."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return ""
    # 找 Error/Exception/Traceback 片段
    patterns = [r"Error[^\n]*", r"AssertionError[^\n]*",
                r"ValueError[^\n]*", r"RuntimeError[^\n]*",
                r"raise [^\n]*", r"NotImplementedError[^\n]*"]
    hits = []
    for p in patterns:
        for m in re.finditer(p, content):
            hits.append(m.group(0).strip()[:200])
    if hits:
        return " | ".join(hits[:4])
    return ""


def main():
    ap = argparse.ArgumentParser(description="xlite × vllm-ascend 特性交互验证")
    ap.add_argument("--feature", required=True, choices=list(FEATURES.keys()))
    ap.add_argument("--xlite-mode", required=True,
                    choices=["xlite_decode_only", "xlite_full_mode", "none"])
    ap.add_argument("--port", type=int, default=8055)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--tp", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=20480)
    ap.add_argument("--startup-timeout", type=int, default=1500,
                    help="服务启动超时(秒), GLM-5-w8a8 16卡加载较慢, 默认25分钟")
    args = ap.parse_args()
    spec = FEATURES[args.feature]
    ok = run_case(spec, args.xlite_mode, args.port, args.model, args.tp,
                  args.max_num_batched_tokens, args.max_num_seqs,
                  args.max_model_len, args.startup_timeout)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
