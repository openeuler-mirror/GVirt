/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * GLM-5.3-Flash head mean-collapse (weight-free). The MHC layer stream leaves a
 * residual of shape [m, hcMult*hidden] (hcMult blocks of `hidden` concatenated
 * along the column dim by ConcatCol/ForwardHcPre). GLM-5.3 has no hc_head
 * parameter, so the head is a plain unweighted mean over the hcMult streams:
 *
 *   y[m, j] = (1 / hcMult) * Σ_{h=0}^{hcMult-1} x[m, h*hidden + j]
 *
 * This is the unweighted counterpart of hc_pre (y = Σ_h pre[h] * x[m,h,*]).
 * Accumulation is done in fp32 (matching PyTorch's bf16 .mean() which upcasts),
 * then cast back to Dtype; the per-stream scale 1/hcMult is folded into the
 * vaxpy accumulator so no separate vmuls pass is needed. invHcMult is passed as
 * a host-computed float param (CCE forbids uint->float cast inside aicore).
 *
 * Synchronization: single (non-ping-pong) in/out UB buffers with self-contained
 * per-iteration set/wait pairs (mirrors indexer_kcache_store_glm5). This is
 * hang-free for any m >= 1 (including the small warmup batches where ping-pong
 * tail drains would deadlock). Two intra-iter syncs (MTE2->V before V reads
 * inDtype; V->MTE3 before MTE3 stores outDtype) and two inter-iter syncs at the
 * end of each iteration (MTE3->MTE2 protects next iter's inDtype load;
 * MTE3->V protects next iter's outDtype write) -each set is immediately
 * consumed by a wait in the same iteration, so no flags carry over and no tail
 * drain is needed.
 *
 * Layouts:
 *   xResid: [m, hcMult*hidden]  (Dtype, row-major; stream h at offset h*hidden)
 *   yOut:   [m, hidden]         (Dtype)
 *   Registered dtypes: bfloat16_t, float16_t (input is the model hidden-state
 *   dtype; fp32 hidden state is not a supported run config, and convert_input/
 *   convert_output have no float specialization).
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"

#ifdef __DAV_C220_VEC__

template <typename Dtype>
__aicore__ inline void hc_mean_collapse(__gm__ Dtype *xResid, __gm__ Dtype *yOut, uint32_t m,
                                        uint32_t hcMult, uint32_t hidden, float invHcMult)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);  // fp32 elems per full vector

    const uint64_t residDtypeLen = ROUND_UP(hcMult * hidden * sizeof(Dtype), UB_BUF_ALIGN_SIZE);
    const uint64_t yDtypeLen = ROUND_UP(hidden * sizeof(Dtype), UB_BUF_ALIGN_SIZE);
    const uint64_t residFp32Len = ROUND_UP(hcMult * hidden * sizeof(float), UB_BUF_ALIGN_SIZE);
    const uint64_t mergeFp32Len = ROUND_UP(hidden * sizeof(float), UB_BUF_ALIGN_SIZE);

    uint64_t off = 0;
    __ubuf__ Dtype *inDtype = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += residDtypeLen;
    __ubuf__ Dtype *outDtype = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += yDtypeLen;
    __ubuf__ float *xFp32 = reinterpret_cast<__ubuf__ float *>(off);  // [hcMult, hidden] as fp32
    off += residFp32Len;
    __ubuf__ float *yCalc = reinterpret_cast<__ubuf__ float *>(off);  // accumulator Σ_h x[h]
    off += mergeFp32Len;
    assert(off <= UB_SIZE);

    const int vecRep = DIV_ROUND_UP(hidden, calcPad);  // vector repeats over hidden (fp32)
    const int totalRep = hcMult * vecRep;              // fp32 repeats over [hcMult, hidden]
    const uint32_t dBytes = hidden * sizeof(Dtype);

    for (uint32_t process = GetBlockIdx(); process < m; process += GetBlockNum()) {
        __gm__ Dtype *xBase = xResid + (uint64_t)process * hcMult * hidden;
        __gm__ Dtype *yRow = yOut + (uint64_t)process * hidden;

        // MTE2: load all hcMult input rows into inDtype.
        for (uint32_t h = 0; h < hcMult; h++) {
            CopyGmToUbufAligned(inDtype + h * hidden, xBase + h * hidden, dBytes);
        }
        pipe_barrier(PIPE_MTE2);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // V: convert Dtype -> fp32 (hcMult*hidden elements, chunked over vector repeats).
        for (int chunk = 0; chunk < totalRep; chunk += VECTOR_MAX_REPEAT) {
            const int rep =
                (totalRep - chunk) < VECTOR_MAX_REPEAT ? (totalRep - chunk) : VECTOR_MAX_REPEAT;
            convert_input<Dtype>(xFp32 + chunk * calcPad, inDtype + chunk * calcPad, rep);
        }
        pipe_barrier(PIPE_V);

        // V: yCalc = 0, then yCalc = Σ_h invHcMult * xFp32[h]  ==  mean over hcMult streams.
        vector_dup(yCalc, 0.0f, vecRep, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        for (uint32_t h = 0; h < hcMult; h++) {
            vaxpy(yCalc, xFp32 + h * hidden, invHcMult, vecRep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }

        // V: fp32 yCalc -> Dtype outDtype.
        convert_output<Dtype>(outDtype, yCalc, vecRep);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // MTE3: store output row.
        CopyUbufToGmAligned(yRow, outDtype, dBytes);
        pipe_barrier(PIPE_MTE3);

        // Inter-iter handoffs: inDtype (next iter's MTE2 load) and outDtype (next
        // iter's V write) must both wait for this iter's MTE3 store to finish. The
        // set+wait is in the same iteration (self-draining), so no flag carries
        // over -hang-free for m >= 1 and no tail drain required.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
}

#define HC_MEAN_COLLAPSE_FUNC_DEFINE(dtype)                                               \
    extern "C" __global__ __aicore__ void hc_mean_collapse_##dtype(                       \
        GM_ADDR xResid, GM_ADDR yOut, uint32_t m, uint32_t hcMult, uint32_t hidden,       \
        float invHcMult)                                                                  \
    {                                                                                     \
        hc_mean_collapse((__gm__ dtype *)xResid, (__gm__ dtype *)yOut, m, hcMult, hidden, \
                         invHcMult);                                                      \
    }
#else
#define HC_MEAN_COLLAPSE_FUNC_DEFINE(dtype)                                         \
    extern "C" __global__ __aicore__ void hc_mean_collapse_##dtype(                 \
        GM_ADDR xResid, GM_ADDR yOut, uint32_t m, uint32_t hcMult, uint32_t hidden, \
        float invHcMult)                                                            \
    {                                                                               \
        (void)xResid;                                                               \
        (void)yOut;                                                                 \
        (void)m;                                                                    \
        (void)hcMult;                                                               \
        (void)hidden;                                                               \
        (void)invHcMult;                                                            \
    }
#endif
