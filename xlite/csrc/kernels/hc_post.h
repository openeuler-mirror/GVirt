/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
// hc_post: post-attn/FFN merge (restores the H axis hc_pre collapsed).
//   x [m, D] bf16, residual [m, H, D] bf16, y [m, H, D] bf16
//   post [m, H] fp32, comb [m, H*H] fp32 (comb[h*H+k]: h=source, k=output)
//   y[k,d] = post[k]*x[d] + sum_h comb[h,k]*residual[h,d]
// term2 (K=H GEMM) runs as pure-Vector vaxpy accumulation. One kernel fuses term1 (vmuls)
// + term2 + bf16<->fp32 cast (fp32 compute). in-place (residual==y): read all H source
// streams before writing any output. Tiling: m across cores, D in dTile chunks, ping-pong.
template <typename Dtype>
__aicore__ void hc_post(__gm__ Dtype *x, __gm__ float *post, __gm__ float *comb,
                        __gm__ Dtype *residual, __gm__ Dtype *y, uint32_t m, uint32_t hcMult,
                        uint32_t hidden)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);

    // dTile<=2048: dual slot + residualFp32[H*dTile] (the big one) fits 192KB UB.
    const uint32_t dTile = hidden < 2048 ? hidden : 2048;

    const uint32_t postBytes = hcMult * sizeof(float);
    const uint32_t combBytes = hcMult * hcMult * sizeof(float);
    const uint64_t postLen = ROUND_UP(postBytes, UB_BUF_ALIGN_SIZE);
    const uint64_t combLen = ROUND_UP(combBytes, UB_BUF_ALIGN_SIZE);
    const uint64_t fp32Len = ROUND_UP(dTile * sizeof(float), UB_BUF_ALIGN_SIZE);
    const uint64_t dtypeLen = ROUND_UP(dTile * sizeof(Dtype), UB_BUF_ALIGN_SIZE);
    const uint64_t residualFp32Len = ROUND_UP(hcMult * dTile * sizeof(float), UB_BUF_ALIGN_SIZE);
    const uint64_t residualDtypeLen = ROUND_UP(hcMult * dTile * sizeof(Dtype), UB_BUF_ALIGN_SIZE);

    uint64_t off = 0;
    __ubuf__ float *postU = (__ubuf__ float *)off;
    off += postLen;
    __ubuf__ float *combU = (__ubuf__ float *)off;
    off += combLen;

    __ubuf__ float *residualFp32 = (__ubuf__ float *)off;
    off += residualFp32Len;
    __ubuf__ float *xFp32 = (__ubuf__ float *)off;
    off += fp32Len;
    __ubuf__ float *outFp32 = (__ubuf__ float *)off;
    off += fp32Len;

    // IO slot 0
    __ubuf__ Dtype *inXDtype_0 = (__ubuf__ Dtype *)off;
    off += dtypeLen;
    __ubuf__ Dtype *inResidDtype_0 = (__ubuf__ Dtype *)off;
    off += residualDtypeLen;
    __ubuf__ Dtype *outDtype_0 = (__ubuf__ Dtype *)off;
    off += dtypeLen;
    // IO slot 1
    __ubuf__ Dtype *inXDtype_1 = (__ubuf__ Dtype *)off;
    off += dtypeLen;
    __ubuf__ Dtype *inResidDtype_1 = (__ubuf__ Dtype *)off;
    off += residualDtypeLen;
    __ubuf__ Dtype *outDtype_1 = (__ubuf__ Dtype *)off;
    off += dtypeLen;

    __ubuf__ Dtype *inXDtypeArr[2] = {inXDtype_0, inXDtype_1};
    __ubuf__ Dtype *inResidDtypeArr[2] = {inResidDtype_0, inResidDtype_1};
    __ubuf__ Dtype *outDtypeArr[2] = {outDtype_0, outDtype_1};
    assert(off <= UB_SIZE);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    int curr = 0;
    for (uint32_t tok = block_idx; tok < m; tok += uint32_t(block_num)) {
        __gm__ float *postPtr = post + tok * hcMult;
        __gm__ float *combPtr = comb + tok * hcMult * hcMult;
        __gm__ Dtype *xPtr = x + tok * hidden;
        __gm__ Dtype *residBase = residual + tok * hcMult * hidden;

        // Load per-token scalar tables post[H]/comb[H*H].
        CopyGmToUbufAligned(postU, postPtr, postBytes);
        CopyGmToUbufAligned(combU, combPtr, combBytes);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID5);

        // D-tile loop: split hidden, pipeline one dTile per tile.
        for (uint32_t d = 0; d < hidden; d += dTile) {
            const uint32_t dLeft = hidden - d;
            const uint32_t dCur = dLeft < dTile ? dLeft : dTile;
            const int repeat = DIV_ROUND_UP(dCur, calcPad);
            const uint32_t dBytes = dCur * sizeof(Dtype);

            // IN: MTE2 loads x + H residual -> V casts to fp32. (cross-tile guard: wait V read
            // done)
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            CopyGmToUbufAligned(inXDtypeArr[curr], xPtr + d, dBytes);
            for (uint32_t h = 0; h < hcMult; h++) {
                CopyGmToUbufAligned(inResidDtypeArr[curr] + h * dTile, residBase + h * hidden + d,
                                    dBytes);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            convert_input(xFp32, inXDtypeArr[curr], repeat);
            for (uint32_t h = 0; h < hcMult; h++) {
                convert_input(residualFp32 + h * dTile, inResidDtypeArr[curr] + h * dTile, repeat);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

            // CALC: per output stream k: term1 (vmuls) + term2 (H x vaxpy) -> V writes outDtype ->
            // MTE3 stores GM
            if (d == 0) {
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID5);
            }
            for (uint32_t k = 0; k < hcMult; k++) {
                float postK = postU[k];
                vmuls(outFp32, xFp32, postK, repeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                for (uint32_t h = 0; h < hcMult; h++) {
                    float combHK = combU[h * hcMult + k];
                    vaxpy(outFp32, residualFp32 + h * dTile, combHK, repeat, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                }

                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
                convert_output(outDtypeArr[curr], outFp32, repeat);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                __gm__ Dtype *yPtr = y + tok * hcMult * hidden + k * hidden + d;
                CopyUbufToGmAligned(yPtr, outDtypeArr[curr], dBytes);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            }
            curr = 1 - curr;
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}

#define HC_POST_FUNC_DEFINE(dtype)                                                                 \
    extern "C" __global__ __aicore__ void hc_post_##dtype(GM_ADDR x, GM_ADDR post, GM_ADDR comb,   \
                                                          GM_ADDR residual, GM_ADDR y, uint32_t m, \
                                                          uint32_t hcMult, uint32_t hidden)        \
    {                                                                                              \
        hc_post((__gm__ dtype *)x, (__gm__ float *)post, (__gm__ float *)comb,                     \
                (__gm__ dtype *)residual, (__gm__ dtype *)y, m, hcMult, hidden);                   \
    }
#else
#define HC_POST_FUNC_DEFINE(dtype)                                                                 \
    extern "C" __global__ __aicore__ void hc_post_##dtype(GM_ADDR x, GM_ADDR post, GM_ADDR comb,   \
                                                          GM_ADDR residual, GM_ADDR y, uint32_t m, \
                                                          uint32_t hcMult, uint32_t hidden)        \
    {                                                                                              \
    }
#endif
