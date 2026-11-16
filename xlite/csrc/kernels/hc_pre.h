/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"
#include "ub_alloc.h"

// hc_pre: y[m,hidden] = Σ_h pre[h] * x[m,h,hidden]  (non-fused counterpart of hc_act's
// merge).  x bf16, pre fp32 (from GM, the hc_split_sinkhorn output), y bf16 out.
// pre is MTE2-loaded into preStage (ping-pong), ub2ub-relayed into preCalc, scalar-read by vaxpy.

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ inline void hc_pre(__gm__ float *pre, __gm__ Dtype *xResid, __gm__ Dtype *yOut,
                              uint32_t m, uint32_t hcMult, uint32_t hidden)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);  // fp32 elems per full vector

    const uint32_t preBytes = hcMult * sizeof(float);
    const uint64_t lenPre = ROUND_UP(preBytes, VECTOR_MAX_BYTESIZE);

    UbAlloc ub;
    __ubuf__ Dtype *inDtypeArr[2];
    __ubuf__ Dtype *outDtypeArr[2];
    __ubuf__ float *preStage[2];
    inDtypeArr[0] = ub.Alloc<Dtype>(hcMult * hidden);
    outDtypeArr[0] = ub.Alloc<Dtype>(hidden);
    inDtypeArr[1] = ub.Alloc<Dtype>(hcMult * hidden);
    outDtypeArr[1] = ub.Alloc<Dtype>(hidden);
    preStage[0] = ub.Alloc<float>(hcMult, VECTOR_MAX_BYTESIZE);
    preStage[1] = ub.Alloc<float>(hcMult, VECTOR_MAX_BYTESIZE);
    // ub2ub target; scalar-read by vaxpy
    __ubuf__ float *preCalc = ub.Alloc<float>(hcMult, VECTOR_MAX_BYTESIZE);
    __ubuf__ float *xFp32 = ub.Alloc<float>(hcMult * hidden);  // [hcMult, hidden] residual as fp32
    __ubuf__ float *yCalc = ub.Alloc<float>(hidden);           // accumulator: Σ_h pre[h]*x

    const int vecRep = DIV_ROUND_UP(hidden, calcPad);  // vector repeats over hidden
    const int totalRep = hcMult * vecRep;              // fp32 repeats over [hcMult, hidden]
    const uint32_t dBytes = hidden * sizeof(Dtype);
    const uint32_t preBlocks = lenPre / BLOCK_SIZE;  // ub2ub burst count (b32)

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);

    int curr = 0;
    for (uint32_t process = block_idx; process < m; process += uint32_t(block_num)) {
        __gm__ Dtype *xBase = xResid + (uint64_t)process * hcMult * hidden;
        __gm__ Dtype *yRow = yOut + (uint64_t)process * hidden;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        CopyGmToUbufAligned(preStage[curr], pre + process * hcMult, preBytes);
        for (uint32_t h = 0; h < hcMult; h++) {
            CopyGmToUbufAligned(inDtypeArr[curr] + h * hidden, xBase + h * hidden, dBytes);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);

        copy_ubuf_to_ubuf(preCalc, preStage[curr], 0, 1, preBlocks, 0, 0);
        for (int chunk = 0; chunk < totalRep; chunk += VECTOR_MAX_REPEAT) {
            const int rep =
                (totalRep - chunk) < VECTOR_MAX_REPEAT ? (totalRep - chunk) : VECTOR_MAX_REPEAT;
            convert_input(xFp32 + chunk * calcPad, inDtypeArr[curr] + chunk * calcPad, rep);
        }
        vector_dup(yCalc, 0.0f, vecRep, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

        set_flag(PIPE_V, PIPE_S, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID4);
        for (uint32_t h = 0; h < hcMult; h++) {
            float preH = preCalc[h];
            vaxpy(yCalc, xFp32 + h * hidden, preH, vecRep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }

        // y fp32 -> bf16 -> GM.
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        convert_output(outDtypeArr[curr], yCalc, vecRep);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
        CopyUbufToGmAligned(yRow, outDtypeArr[curr], dBytes);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);

        curr = 1 - curr;
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}

#define HC_PRE_FUNC_DEFINE(dtype)                                                                \
    extern "C" __global__ __aicore__ void hc_pre_##dtype(                                        \
        GM_ADDR pre, GM_ADDR xResid, GM_ADDR yOut, uint32_t m, uint32_t hcMult, uint32_t hidden) \
    {                                                                                            \
        hc_pre((__gm__ float *)pre, (__gm__ dtype *)xResid, (__gm__ dtype *)yOut, m, hcMult,     \
               hidden);                                                                          \
    }
#else
#define HC_PRE_FUNC_DEFINE(dtype)                                                                \
    extern "C" __global__ __aicore__ void hc_pre_##dtype(                                        \
        GM_ADDR pre, GM_ADDR xResid, GM_ADDR yOut, uint32_t m, uint32_t hcMult, uint32_t hidden) \
    {                                                                                            \
    }
#endif
