#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

cd xlite/tools/quantization/

MODEL_PATH=${1:-"/mnt/nvme2n1/models/deepseek-R1/"}
QMODEL_PATH=${2:-"/mnt/nvme1n1/models/deepseek-R1-w8a8-rotquant/"}
NUM_PROCESSES=${3:-8}
NUM_PARTS_PER_PROCESS=$((163 / NUM_PROCESSES))

echo "MODEL_PATH: $MODEL_PATH"
echo "QMODEL_PATH: $QMODEL_PATH"
echo "NUM_PROCESSES: $NUM_PROCESSES"

# 1. Generate the rotation matrix
python preprocess.py --model_path $MODEL_PATH

# 2. Perform the quantizatin using multiple processes (on multiple NPUs)
id_npu=0
start_idx=1
end_idx=$NUM_PARTS_PER_PROCESS

for ((i = 0; i < NUM_PROCESSES; i++)); do
    echo "npu id $id_npu, start $start_idx, end $end_idx"
    python ./w8a8.py \
        --model_path $MODEL_PATH \
        --rot_model_path $QMODEL_PATH \
        --post_opt \
        --deactivate_head_rotate \
        --run_quant \
        --quant_mtp \
        --id_npu $id_npu \
        --file_idx_start $start_idx \
        --file_idx_end $end_idx &
    ((start_idx = end_idx + 1))
    ((end_idx = start_idx + NUM_PARTS_PER_PROCESS))
    ((id_npu = id_npu + 1))
done

wait
echo "Quantization done"

# 3. Rebuild the index json
echo "Rebuild index json"
python build_index.py --model_path $QMODEL_PATH

echo "Done"
