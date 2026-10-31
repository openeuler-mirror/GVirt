/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"

#ifdef __DAV_C220_VEC__

// C8 query quantizer for the V4 indexer path: quantize the rope'd (but NOT
// rotate_activation'd) BF16 indexer Q rows into the dense int8 q8 buffer that
// indexer_topk's C8 reader consumes, fusing the per-row de-scale with the
// per-(token, head) projection weight into fp16 scaledWeights. Row r is
// (token, head) = (r / heads, r % heads); all four buffers use that flat row
// order, so no slot mapping is needed. Per row (rotate NOT applied; chain
// identical to indexer_prepare_c8 / quant_cache_c8; zero row -> zeros):
//   q8_row           = round_nearest(q_row * 127/absmax)
//   scaledWeights[r] = fp16(headWeight[r] * absmax * rotateScale / 127)
// rotateScale (the rotate_activation factor, 1/sqrt(head_dim) for V4) is
// uniform per row, so it changes no int8 value and folds into the de-scale
// instead: the caller ropes with doRotate=false and skips the nope Muls;
// topk scores differ from the float path only by a per-query constant.

template <typename dtype>
__aicore__ inline void quant_q_c8(GM_ADDR q, GM_ADDR headWeight, GM_ADDR q8, GM_ADDR scaledWeights,
                                  uint32_t rows, uint32_t dim, float rotateScale)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint32_t calcRepeat = dim / VECTOR_MAX_NUM_OF_FP32;
    uint32_t quantRepeat = DIV_ROUND_UP(dim, VECTOR_MAX_NUM_OF_FP16);
    float descale = rotateScale / float(127);

    uintptr_t off = 0;
    __ubuf__ dtype *qRow[2];
    qRow[0] = reinterpret_cast<__ubuf__ dtype *>(off);
    off += ROUND_UP(dim * sizeof(dtype), VECTOR_MAX_BYTESIZE);
    qRow[1] = reinterpret_cast<__ubuf__ dtype *>(off);
    off += ROUND_UP(dim * sizeof(dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ dtype *headWeightHalf[2];
    headWeightHalf[0] = reinterpret_cast<__ubuf__ dtype *>(off);
    off += VECTOR_MAX_BYTESIZE;
    headWeightHalf[1] = reinterpret_cast<__ubuf__ dtype *>(off);
    off += VECTOR_MAX_BYTESIZE;
    auto *qRowF32 = reinterpret_cast<__ubuf__ float *>(off);
    off += ROUND_UP(dim * sizeof(float), VECTOR_MAX_BYTESIZE);
    auto *absValues = reinterpret_cast<__ubuf__ float *>(off);
    off += ROUND_UP(dim * sizeof(float), VECTOR_MAX_BYTESIZE);
    auto *rounded = reinterpret_cast<__ubuf__ int32_t *>(off);
    off += ROUND_UP(dim * sizeof(int32_t), VECTOR_MAX_BYTESIZE);
    auto *halfValues = reinterpret_cast<__ubuf__ half *>(off);
    off += ROUND_UP(dim * sizeof(half), VECTOR_MAX_BYTESIZE);
    __ubuf__ int8_t *quant[2];
    quant[0] = reinterpret_cast<__ubuf__ int8_t *>(off);
    off += ROUND_UP(dim, VECTOR_MAX_BYTESIZE);
    quant[1] = reinterpret_cast<__ubuf__ int8_t *>(off);
    off += ROUND_UP(dim, VECTOR_MAX_BYTESIZE);
    auto *headWeightF32 = reinterpret_cast<__ubuf__ float *>(off);
    off += VECTOR_MAX_BYTESIZE;
    auto *scaleF32 = reinterpret_cast<__ubuf__ float *>(off);
    off += VECTOR_MAX_BYTESIZE;
    auto *scaledWeightsF32 = reinterpret_cast<__ubuf__ float *>(off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ half *scaledWeightsF16[2];
    scaledWeightsF16[0] = reinterpret_cast<__ubuf__ half *>(off);
    off += VECTOR_MAX_BYTESIZE;
    scaledWeightsF16[1] = reinterpret_cast<__ubuf__ half *>(off);
    off += VECTOR_MAX_BYTESIZE;
    assert(off <= UB_SIZE);

    __gm__ dtype *qGm = reinterpret_cast<__gm__ dtype *>(q);
    __gm__ dtype *headWeightGm = reinterpret_cast<__gm__ dtype *>(headWeight);
    __gm__ int8_t *q8Gm = reinterpret_cast<__gm__ int8_t *>(q8);
    __gm__ half *scaledWeightsGm = reinterpret_cast<__gm__ half *>(scaledWeights);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    uint32_t curr = 0;
    for (uint32_t row = block_idx; row < rows; row += block_num) {
        // row + headWeight GM -> UB (slot curr returned two rows back)
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        CopyGmToUbufAligned(qRow[curr], qGm + static_cast<uint64_t>(row) * dim,
                            dim * sizeof(dtype));
        CopyGmToUbufAligned(headWeightHalf[curr], headWeightGm + row, sizeof(dtype));
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        convert_input<dtype>(qRowF32, qRow[curr], calcRepeat);
        convert_input<dtype>(headWeightF32, headWeightHalf[curr], 1);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);  // load slot free two rows ahead

        // absmax of the un-rotated row
        vabs(absValues, qRowF32, calcRepeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        ReduceMax(absValues, absValues, dim);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float absmax = *absValues;
        // zero row -> factor 0: int8 zeros, scaledWeights 0
        float factor = absmax != 0.0f ? float(127) / absmax : 0.0f;
        // round_nearest via the int32r round-trip (see indexer_prepare_c8)
        vmuls(scaleF32, absValues, descale, 1, 1, 1, 8, 8);
        vmuls(qRowF32, qRowF32, factor, calcRepeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(scaledWeightsF32, scaleF32, headWeightF32, 1, 1, 1, 1, 8, 8, 8);
        vconv_f322s32r(rounded, qRowF32, calcRepeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_s322f32(qRowF32, rounded, calcRepeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f16(halfValues, qRowF32, calcRepeat, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
        vconv_f322f16(scaledWeightsF16[curr], scaledWeightsF32, 1, 1, 1, 4, 8);
        vconv_f162s8(quant[curr], halfValues, quantRepeat, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);

        // dense stores: int8 row (dim % 64 == 0 keeps 32B block alignment),
        // fp16 scaledWeights at row (b16 path)
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        CopyUbufToGmAligned(q8Gm + static_cast<uint64_t>(row) * dim, quant[curr], dim);
        CopyUbufToGmAligned(scaledWeightsGm + row, scaledWeightsF16[curr], sizeof(half));
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);  // store slots free two rows ahead

        curr = 1 - curr;
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define QUANT_Q_C8_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void quant_q_c8_##dtype(                            \
        GM_ADDR q, GM_ADDR headWeight, GM_ADDR q8, GM_ADDR scaledWeights, uint32_t rows, \
        uint32_t dim, float rotateScale)                                                 \
    {                                                                                    \
        quant_q_c8<dtype>(q, headWeight, q8, scaledWeights, rows, dim, rotateScale);     \
    }
#else
#define QUANT_Q_C8_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void quant_q_c8_##dtype(                            \
        GM_ADDR q, GM_ADDR headWeight, GM_ADDR q8, GM_ADDR scaledWeights, uint32_t rows, \
        uint32_t dim, float rotateScale)                                                 \
    {                                                                                    \
    }
#endif
