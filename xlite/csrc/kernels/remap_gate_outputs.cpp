/*
 * Copyright (C) 2025 - 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * EPLB gate-output remap: scatter ForwardMoEGate outputs from logical [M, nRoutedExperts]
 * columns to physical [M, totalPhysicalExperts] columns via log2phy (logical→physical slot).
 *
 * Two concerns, fused in one AIV kernel, one token-tile per block-iteration:
 *   weights: wOut[t, log2phy[L]] = wIn[t, L]            (BF16/FP32, dtype-agnostic byte copy)
 *   routing: if bit (t, L) set in rIn → set bit (t, log2phy[L]) in rOut   (BIT1, bit-test-set)
 *
 * log2phy is a bijection logical→physical on this rank, so the scatter is collision-free.
 * rOut rows are zeroed in-UB before bit-set (bitmapSet is OR); padding rows [m, M) of rIn are
 * already zero (ForwardMoEGate memsets them), so rOut padding is naturally zero.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

__aicore__ inline void remap_gate_outputs_kernel(GM_ADDR wIn, GM_ADDR rIn, GM_ADDR wOut,
                                                 GM_ADDR rOut, GM_ADDR log2phy, uint32_t M,
                                                 uint32_t nRoutedExperts,
                                                 uint32_t totalPhysicalExperts, uint32_t elemBytes)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (M == 0 || nRoutedExperts == 0 || totalPhysicalExperts == 0 || elemBytes == 0) {
        return;
    }

    uint32_t log2phyBytes = nRoutedExperts * sizeof(uint32_t);
    uint32_t wInRowBytes = nRoutedExperts * elemBytes;
    uint32_t wOutRowBytes = totalPhysicalExperts * elemBytes;
    uint32_t rInRowBytes = DIV_ROUND_UP(nRoutedExperts, 8);
    uint32_t rOutRowBytes = DIV_ROUND_UP(totalPhysicalExperts, 8);

    // UB layout: log2phy | wInRow | wOutRow | rInRow | rOutRow. Each segment is rounded up to
    // VECTOR_MAX_BYTESIZE (256B) so the vector_dup zeroing (one 256B repeat) cannot overrun
    // into the next segment (same pattern as permutation.cpp).
    __ubuf__ uint8_t *base = (__ubuf__ uint8_t *)get_imm(0);
    __ubuf__ uint32_t *log2phyUbuf = (__ubuf__ uint32_t *)base;
    __ubuf__ uint8_t *wInRow = base + ROUND_UP(log2phyBytes, VECTOR_MAX_BYTESIZE);
    __ubuf__ uint8_t *wOutRow = wInRow + ROUND_UP(wInRowBytes, VECTOR_MAX_BYTESIZE);
    __ubuf__ uint8_t *rInRow = wOutRow + ROUND_UP(wOutRowBytes, VECTOR_MAX_BYTESIZE);
    __ubuf__ uint8_t *rOutRow = rInRow + ROUND_UP(rInRowBytes, VECTOR_MAX_BYTESIZE);

    // Load log2phy once per block (small: nRoutedExperts * 4B). The S-pipe reads it
    // inside the loop; a full barrier guarantees the MTE2 load completes first.
    CopyGmToUbufAligned((__ubuf__ uint8_t *)log2phyUbuf, (__gm__ uint8_t *)log2phy, log2phyBytes);
    pipe_barrier(PIPE_ALL);

    // Per-token phases are serialized with pipe_barrier(PIPE_ALL): the scalar S-pipe scatter
    // reads wInRow/rInRow (loaded by MTE2) and reads/writes wOutRow/rOutRow (zeroed by V), so
    // each phase needs a full drain. Heavier than set_flag/wait_flag pairing but trivially correct.
    for (uint32_t t = block_idx; t < M; t += block_num) {
        // ===== weights scatter: wOut[t, log2phy[L]] = wIn[t, L] =====
        CopyGmToUbufAligned(wInRow, (__gm__ uint8_t *)wIn + (uint64_t)t * wInRowBytes, wInRowBytes);
        // CCE vector_dup rejects uint8_t*; cast to a typed UB pointer (byte-equivalent zeroing).
        vector_dup((__ubuf__ uint32_t *)wOutRow, 0, DIV_ROUND_UP(wOutRowBytes, VECTOR_MAX_BYTESIZE),
                   1, 1, 8, 1);
        pipe_barrier(PIPE_ALL);  // wInRow loaded, wOutRow zeroed
        for (uint32_t L = 0; L < nRoutedExperts; L++) {
            uint32_t phy = log2phyUbuf[L];
            __ubuf__ uint8_t *dst = wOutRow + (uint64_t)phy * elemBytes;
            __ubuf__ uint8_t *src = wInRow + (uint64_t)L * elemBytes;
            for (uint32_t k = 0; k < elemBytes; k++) {
                dst[k] = src[k];
            }
        }
        pipe_barrier(PIPE_ALL);  // scatter done
        CopyUbufToGmAligned((__gm__ uint8_t *)wOut + (uint64_t)t * wOutRowBytes, wOutRow,
                            wOutRowBytes);
        pipe_barrier(PIPE_ALL);  // store done, UB reusable next iteration

        // ===== routing bit scatter: if bit(t,L) in rIn → set bit(t, log2phy[L]) in rOut =====
        CopyGmToUbufAligned(rInRow, (__gm__ uint8_t *)rIn + (uint64_t)t * rInRowBytes, rInRowBytes);
        vector_dup((__ubuf__ uint32_t *)rOutRow, 0, DIV_ROUND_UP(rOutRowBytes, VECTOR_MAX_BYTESIZE),
                   1, 1, 8, 1);
        pipe_barrier(PIPE_ALL);  // rInRow loaded, rOutRow zeroed
        __ubuf__ uint64_t *rInRow64 = (__ubuf__ uint64_t *)rInRow;
        __ubuf__ uint64_t *rOutRow64 = (__ubuf__ uint64_t *)rOutRow;
        for (uint32_t L = 0; L < nRoutedExperts; L++) {
            if (bitmapTest(rInRow64, L)) {
                bitmapSet(rOutRow64, log2phyUbuf[L]);
            }
        }
        pipe_barrier(PIPE_ALL);  // bit-set done
        CopyUbufToGmAligned((__gm__ uint8_t *)rOut + (uint64_t)t * rOutRowBytes, rOutRow,
                            rOutRowBytes);
        pipe_barrier(PIPE_ALL);  // store done, UB reusable next iteration
    }
}

extern "C" __global__ __aicore__ void remap_gate_outputs(GM_ADDR wIn, GM_ADDR rIn, GM_ADDR wOut,
                                                         GM_ADDR rOut, GM_ADDR log2phy, uint32_t M,
                                                         uint32_t nRoutedExperts,
                                                         uint32_t totalPhysicalExperts,
                                                         uint32_t elemBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    remap_gate_outputs_kernel(wIn, rIn, wOut, rOut, log2phy, M, nRoutedExperts,
                              totalPhysicalExperts, elemBytes);
}
#endif
