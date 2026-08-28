/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Repeat-interleave head expansion for linear attention (GDN Step 6).
 * Semantics (match ExpandLinearHeads):
 *   dst[t, h*expand + e, :] = src[t, h, :]   for all e in [0, expand)
 * Layouts:
 *   in:  [T, nKHeads * headBytes]  (token-major, per-head contiguous)
 *   out: [T, nVHeads * headBytes]  (nVHeads == nKHeads * expand)
 *
 * Pure byte-wise GM->GM copy (dtype-agnostic), one kernel launch replaces the
 * previous host-side loop of nKHeads*expand per-token aclrtMemcpyAsync calls
 * (96 tiny 256B copies per layer in decode), which dominated host time
 * (~1ms/layer, ~48ms/step). AIV blocks are scaled to the number of output
 * head segments so decode (48 segments) launches few cores while prefill
 * (24576 segments) saturates all cores.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

__aicore__ inline void repeat_interleave_kernel(GM_ADDR in, GM_ADDR out, uint32_t numTokens,
                                                uint32_t nKHeads, uint32_t nVHeads,
                                                uint32_t headBytes)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (numTokens == 0 || nKHeads == 0 || nVHeads == 0 || headBytes == 0) {
        return;
    }
    if (nVHeads % nKHeads != 0) {
        return;
    }
    uint32_t expand = nVHeads / nKHeads;
    if (expand == 0) {
        return;
    }
    uint64_t nkRowBytes = (uint64_t)nKHeads * headBytes;
    uint64_t nvRowBytes = (uint64_t)nVHeads * headBytes;
    uint64_t totalSegs = (uint64_t)numTokens * nVHeads;  // one segment per output head

    // UB ping-pong staging. headBytes is small (<=512B), so a single segment
    // always fits; we keep two halves for read/write overlap.
    constexpr uint32_t PINGPONG = 2;
    uint32_t halfBuf = UB_SIZE / PINGPONG;
    __ubuf__ uint8_t *dataBuf[PINGPONG] = {
        (__ubuf__ uint8_t *)((uintptr_t)0),
        (__ubuf__ uint8_t *)((uintptr_t)halfBuf),
    };
    // Cap a segment copy at halfBuf (defensive; headBytes <= halfBuf normally).
    uint32_t segBytes = headBytes;
    if (segBytes > halfBuf) {
        segBytes = halfBuf;
    }

    // Seed both MTE3->MTE2 flags so the first wait of each buffer can consume.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    uint32_t curr = 0;
    bool hasPending = false;
    uint64_t pendDst = 0;
    uint32_t pendSize = 0;

    for (uint64_t seg = (uint64_t)block_idx; seg < totalSegs; seg += (uint64_t)block_num) {
        uint64_t t = seg / nVHeads;
        uint32_t hv = (uint32_t)(seg % nVHeads);
        uint32_t kh = hv / expand;
        uint64_t dst = t * nvRowBytes + (uint64_t)hv * headBytes;
        uint64_t src = t * nkRowBytes + (uint64_t)kh * headBytes;

        // Write out the previous buffered segment from dataBuf[1-curr].
        if (hasPending) {
            uint32_t wbuf = 1 - curr;
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
            CopyUbufToGmAligned((__gm__ uint8_t *)out + pendDst, dataBuf[wbuf], pendSize);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
        }

        // Read this segment into dataBuf[curr] at offset 0.
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);
        CopyGmToUbufAligned(dataBuf[curr], (__gm__ uint8_t *)in + src, segBytes);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);

        pendDst = dst;
        pendSize = segBytes;
        hasPending = true;
        curr = 1 - curr;
    }

    // Flush the last buffered segment.
    if (hasPending) {
        uint32_t wbuf = 1 - curr;
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
        CopyUbufToGmAligned((__gm__ uint8_t *)out + pendDst, dataBuf[wbuf], pendSize);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ __aicore__ void repeat_interleave(GM_ADDR in, GM_ADDR out, uint32_t numTokens,
                                                        uint32_t nKHeads, uint32_t nVHeads,
                                                        uint32_t headBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    repeat_interleave_kernel(in, out, numTokens, nKHeads, nVHeads, headBytes);
}
#endif
