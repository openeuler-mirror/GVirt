#!/bin/bash
# xlite × vllm-ascend 全量特性交互验证 batch runner
# 顺序: 先 decode-only, 后 full_mode; 特性按分组顺序
# 每个用例: 启动服务→多场景测试→校验→停服, 由 feat_verify.py 完成
# 进度写入 feat_verify_batch_progress.txt, 便于中断后跳过已完成项

set -u
cd "$(dirname "$0")"
PROGRESS="feat_verify_batch_progress.txt"
: > "$PROGRESS"  # 清空进度(新一轮全量)

PORT=8055
TP=16
MNT=8192
SEQS=32
MLEN=20480
STARTUP_TO=1200

# 待执行用例列表: feature xlite_mode  (顺序即执行顺序)
# decode-only 组(先)
RUN_LIST=(
  "baseline xlite_decode_only"
  "apc xlite_decode_only"
  "async_scheduling xlite_decode_only"
  "chunked_prefill_off xlite_decode_only"
  "short_request_first xlite_decode_only"
  "balance_scheduling xlite_decode_only"
  "kv_cache_pool xlite_decode_only"
  "kv_cache_cpu_offload xlite_decode_only"
  "piecewise xlite_decode_only"
  "dp xlite_decode_only"
  "context_parallel xlite_decode_only"
  "lmhead_tp xlite_decode_only"
  "shared_expert_dp xlite_decode_only"
  "eplb xlite_decode_only"
  "eplb_recording xlite_decode_only"
  "eplb_static xlite_decode_only"
  "multistream_moe xlite_decode_only"
  "dsv4_dsa_overlap_off xlite_decode_only"
  "flashcomm1 xlite_decode_only"
  "mlapo xlite_decode_only"
  "cpu_binding xlite_decode_only"
  "fuse_muls_add_off xlite_decode_only"
  "npugraph_ex_off xlite_decode_only"
  "quant_w8a8 xlite_decode_only"
  "quant_w4a8 xlite_decode_only"
  "weight_nz0 xlite_decode_only"
  "weight_nz2 xlite_decode_only"
  "speculative_mtp xlite_decode_only"
  "speculative_eagle3 xlite_decode_only"
  "disaggregated_prefill xlite_decode_only"
  "multimodal xlite_decode_only"
  # full_mode 组(后)
  "baseline xlite_full_mode"
  "apc xlite_full_mode"
  "async_scheduling xlite_full_mode"
  "chunked_prefill_off xlite_full_mode"
  "short_request_first xlite_full_mode"
  "balance_scheduling xlite_full_mode"
  "kv_cache_pool xlite_full_mode"
  "kv_cache_cpu_offload xlite_full_mode"
  "piecewise xlite_full_mode"
  "dp xlite_full_mode"
  "context_parallel xlite_full_mode"
  "lmhead_tp xlite_full_mode"
  "shared_expert_dp xlite_full_mode"
  "eplb xlite_full_mode"
  "eplb_recording xlite_full_mode"
  "eplb_static xlite_full_mode"
  "multistream_moe xlite_full_mode"
  "dsv4_dsa_overlap_off xlite_full_mode"
  "flashcomm1 xlite_full_mode"
  "mlapo xlite_full_mode"
  "cpu_binding xlite_full_mode"
  "fuse_muls_add_off xlite_full_mode"
  "npugraph_ex_off xlite_full_mode"
  "quant_w8a8 xlite_full_mode"
  "quant_w4a8 xlite_full_mode"
  "weight_nz0 xlite_full_mode"
  "weight_nz2 xlite_full_mode"
  "speculative_mtp xlite_full_mode"
  "speculative_eagle3 xlite_full_mode"
  "disaggregated_prefill xlite_full_mode"
  "multimodal xlite_full_mode"
)

total=${#RUN_LIST[@]}
i=0
for entry in "${RUN_LIST[@]}"; do
  i=$((i+1))
  feat=$(echo "$entry" | awk '{print $1}')
  mode=$(echo "$entry" | awk '{print $2}')
  echo "================================================================" | tee -a "$PROGRESS"
  echo "[$(date '+%m-%d %H:%M:%S')] 用例 $i/$total: $feat × $mode" | tee -a "$PROGRESS"
  echo "================================================================" | tee -a "$PROGRESS"
  # 跳过已完成用例(结果表中已存在该 特性×模式 行则跳过)
  mode_cn="decode-only"
  [[ "$mode" == "xlite_full_mode" ]] && mode_cn="full_mode"
  if ls -d feat_verify_logs/${feat}_${mode}_${PORT} >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M:%S')] 用例 $i/$total 跳过(已完成): $feat × $mode" | tee -a "$PROGRESS"
    continue
  fi
  # 确保无残留进程
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'vllm_ascend' 2>/dev/null || true
  sleep 5
  # dp×full_mode 显存紧张, 降 max_model_len
  cur_mlen=$MLEN
  if [[ "$feat" == "dp" && "$mode" == "xlite_full_mode" ]]; then
    cur_mlen=16384
    echo "[$(date '+%m-%d %H:%M:%S')] 注意: dp×full_mode 降 max_model_len=$cur_mlen 避免显存不足" | tee -a "$PROGRESS"
  fi
  python3 feat_verify.py --feature "$feat" --xlite-mode "$mode" --port $PORT \
    --tp $TP --max-num-batched-tokens $MNT --max-num-seqs $SEQS \
    --max-model-len $cur_mlen --startup-timeout $STARTUP_TO \
    >> "feat_verify_batch.log" 2>&1
  rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] 用例 $i/$total 完成: $feat × $mode, rc=$rc" | tee -a "$PROGRESS"
done
echo "[$(date '+%m-%d %H:%M:%S')] 全量验证完成, 共 $total 个用例" | tee -a "$PROGRESS"
