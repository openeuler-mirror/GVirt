/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
// out = attn * sigmoid(gate_raw), elementwise.
// attn/gate/out: [num_tokens, dim], processed per token with tiled dim chunks.
template <typename Dtype>
__aicore__ void sigmoid_gate_mul(__gm__ Dtype *attn, __gm__ Dtype *gate, __gm__ Dtype *out,
                                 uint32_t numTokens, uint32_t dim)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_NUM_OF_FP32;
    // UB layout per pingpong slot: attn_dtype | gate_dtype | attn_f32 | gate_f32 | out_dtype
    // Keep tile small enough for pingpong.
    constexpr int maxTile = 2048;
    int tile = dim < static_cast<uint32_t>(maxTile) ? static_cast<int>(dim) : maxTile;
    tile = ROUND_DOWN(tile, calcPad);
    if (tile == 0) {
        tile = calcPad;
    }

    int actualTileBytes = tile * sizeof(Dtype);
    uint64_t dtypeLen = ROUND_UP(actualTileBytes, UB_BUF_ALIGN_SIZE);
    uint64_t f32Len = ROUND_UP(tile * sizeof(float), UB_BUF_ALIGN_SIZE);

    uint64_t off = 0;
    __ubuf__ Dtype *attnIn0 = (__ubuf__ Dtype *)off;
    off += dtypeLen;
    __ubuf__ Dtype *gateIn0 = (__ubuf__ Dtype *)off;
    off += dtypeLen;
    __ubuf__ float *attnF0 = (__ubuf__ float *)off;
    off += f32Len;
    __ubuf__ float *gateF0 = (__ubuf__ float *)off;
    off += f32Len;
    __ubuf__ Dtype *out0 = (__ubuf__ Dtype *)off;
    off += dtypeLen;

    __ubuf__ Dtype *attnIn1 = (__ubuf__ Dtype *)off;
    off += dtypeLen;
    __ubuf__ Dtype *gateIn1 = (__ubuf__ Dtype *)off;
    off += dtypeLen;
    __ubuf__ float *attnF1 = (__ubuf__ float *)off;
    off += f32Len;
    __ubuf__ float *gateF1 = (__ubuf__ float *)off;
    off += f32Len;
    __ubuf__ Dtype *out1 = (__ubuf__ Dtype *)off;
    off += dtypeLen;

    __ubuf__ float *ones = (__ubuf__ float *)off;
    vector_dup(ones, float(1), DIV_ROUND_UP(tile, calcPad), 1, 0, 8, 0);
    pipe_barrier(PIPE_V);

    __ubuf__ Dtype *attnIn[2] = {attnIn0, attnIn1};
    __ubuf__ Dtype *gateIn[2] = {gateIn0, gateIn1};
    __ubuf__ float *attnF[2] = {attnF0, attnF1};
    __ubuf__ float *gateF[2] = {gateF0, gateF1};
    __ubuf__ Dtype *outBuf[2] = {out0, out1};

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    int curr = 0;

    for (uint32_t token = block_idx; token < numTokens; token += uint32_t(block_num)) {
        for (uint32_t col = 0; col < dim; col += static_cast<uint32_t>(tile)) {
            uint32_t calcNum = dim - col;
            if (calcNum > static_cast<uint32_t>(tile)) {
                calcNum = static_cast<uint32_t>(tile);
            }
            int actualLen = calcNum * sizeof(Dtype);
            int repeat = DIV_ROUND_UP(static_cast<int>(calcNum), calcPad);

            __gm__ Dtype *attnPtr = attn + token * dim + col;
            __gm__ Dtype *gatePtr = gate + token * dim + col;
            __gm__ Dtype *outPtr = out + token * dim + col;

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            CopyGmToUbufAligned(attnIn[curr], attnPtr, actualLen);
            CopyGmToUbufAligned(gateIn[curr], gatePtr, actualLen);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
                vconv_bf162f32(attnF[curr], attnIn[curr], repeat, 1, 1, 8, 4);
                vconv_bf162f32(gateF[curr], gateIn[curr], repeat, 1, 1, 8, 4);
            } else {
                vconv_f162f32(attnF[curr], attnIn[curr], repeat, 1, 1, 8, 4);
                vconv_f162f32(gateF[curr], gateIn[curr], repeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

            // sigmoid(gate) = 1 / (1 + exp(-gate))
            vmuls(gateF[curr], gateF[curr], (float)-1.0, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vexp(gateF[curr], gateF[curr], repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vadds(gateF[curr], gateF[curr], (float)1.0, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vdiv(gateF[curr], ones, gateF[curr], repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            // attn * sigmoid(gate)
            vmul(attnF[curr], attnF[curr], gateF[curr], repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
            if constexpr (std::is_same_v<Dtype, float16_t>) {
                vconv_f322f16r(outBuf[curr], attnF[curr], repeat, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(outBuf[curr], attnF[curr], repeat, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            CopyUbufToGmAligned(outPtr, outBuf[curr], actualLen);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
            curr = 1 - curr;
        }
    }

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}

#define SIGMOID_GATE_MUL_FUNC_DEFINE(dtype)                                               \
    extern "C" __global__ __aicore__ void sigmoid_gate_mul_##dtype(                       \
        GM_ADDR attn, GM_ADDR gate, GM_ADDR out, uint32_t numTokens, uint32_t dim)        \
    {                                                                                     \
        sigmoid_gate_mul((__gm__ dtype *)attn, (__gm__ dtype *)gate, (__gm__ dtype *)out, \
                         numTokens, dim);                                                 \
    }
#else
#define SIGMOID_GATE_MUL_FUNC_DEFINE(dtype)                                        \
    extern "C" __global__ __aicore__ void sigmoid_gate_mul_##dtype(                \
        GM_ADDR attn, GM_ADDR gate, GM_ADDR out, uint32_t numTokens, uint32_t dim) \
    {                                                                              \
    }
#endif
