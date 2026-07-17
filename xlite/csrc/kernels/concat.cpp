/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

#ifndef XLITE_CONCAT_SPLIT_MAX_INPUTS
#define XLITE_CONCAT_SPLIT_MAX_INPUTS 8
#endif

__aicore__ inline void concat_kernel(GM_ADDR out, GM_ADDR in0, GM_ADDR in1, GM_ADDR in2,
                                     GM_ADDR in3, GM_ADDR in4, GM_ADDR in5, GM_ADDR in6,
                                     GM_ADDR in7, uint64_t s0, uint64_t s1, uint64_t s2,
                                     uint64_t s3, uint64_t s4, uint64_t s5, uint64_t s6,
                                     uint64_t s7, uint32_t nInputs, uint64_t totalBytes)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (totalBytes == 0 || nInputs == 0) {
        return;
    }

    GM_ADDR inputs[XLITE_CONCAT_SPLIT_MAX_INPUTS] = {in0, in1, in2, in3, in4, in5, in6, in7};
    uint64_t sizes[XLITE_CONCAT_SPLIT_MAX_INPUTS] = {s0, s1, s2, s3, s4, s5, s6, s7};

    // running offset at which each input starts in the output
    uint64_t inOff[XLITE_CONCAT_SPLIT_MAX_INPUTS] = {0};
    for (uint32_t i = 1; i < nInputs; i++) {
        inOff[i] = inOff[i - 1] + sizes[i - 1];
    }

    // Ping-pong UB staging: two halves. Each sub-segment is read into one half
    // at UB offset 0 (MTE2 writes UB destinations block-aligned, MTE3 reads UB
    // sources block-aligned, so each segment must start at offset 0 of its
    // half), then written out while the next segment is read into the other
    // half.
    constexpr uint32_t PINGPONG = 2;
    uint32_t halfBuf = UB_SIZE / PINGPONG;
    __ubuf__ uint8_t *dataBuf[PINGPONG] = {
        (__ubuf__ uint8_t *)((uintptr_t)0),
        (__ubuf__ uint8_t *)((uintptr_t)halfBuf),
    };
    uint32_t segBufSize = halfBuf;  // max bytes per sub-segment

    // Seed both MTE3->MTE2 flags so the first wait of each buffer can consume.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    // Walk the flat output range [0, totalBytes) in sub-segments, each <=
    // segBufSize and cut on input boundaries. Block-stride so all blocks share
    // the work. Sub-segment i is read into dataBuf[i%2]; the previous sub-segment
    // is written out from dataBuf[(i-1)%2] concurrently.
    uint32_t curr = 0;
    bool hasPending = false;
    uint64_t pendDstOff = 0;  // output offset of the sub-segment awaiting write
    uint32_t pendSize = 0;

    for (uint64_t bo = (uint64_t)block_idx * segBufSize; bo < totalBytes;
         bo += (uint64_t)block_num * segBufSize) {
        uint32_t remain = (uint32_t)((bo + segBufSize > totalBytes) ? totalBytes - bo : segBufSize);

        // Read sub-segments of [bo, bo+remain) from scattered inputs into
        // dataBuf[curr], one at a time (each cut on input boundary, <= segBufSize),
        // writing out the previous buffered sub-segment between reads.
        uint64_t segDstOff = bo;
        while (remain > 0) {
            uint32_t idx = 0;
            while (idx + 1 < nInputs && segDstOff >= inOff[idx + 1]) {
                idx++;
            }
            uint64_t localOff = segDstOff - inOff[idx];
            uint64_t avail = sizes[idx] - localOff;
            uint32_t take = (remain < avail) ? remain : (uint32_t)avail;
            if (take > segBufSize) {
                take = segBufSize;  // cap to UB half (single input exceeds it)
            }

            // Write out the previous buffered sub-segment from dataBuf[1-curr]
            // to the contiguous output.
            if (hasPending) {
                uint32_t wbuf = 1 - curr;
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
                CopyUbufToGmAligned((__gm__ uint8_t *)out + pendDstOff, dataBuf[wbuf], pendSize);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
            }

            // Read this sub-segment into dataBuf[curr] at offset 0.
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);
            CopyGmToUbufAligned(dataBuf[curr], (__gm__ uint8_t *)inputs[idx] + localOff, take);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);

            pendDstOff = segDstOff;
            pendSize = take;
            hasPending = true;
            curr = 1 - curr;

            segDstOff += take;
            remain -= take;
        }
    }

    // Flush the last buffered sub-segment.
    if (hasPending) {
        uint32_t wbuf = 1 - curr;
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
        CopyUbufToGmAligned((__gm__ uint8_t *)out + pendDstOff, dataBuf[wbuf], pendSize);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ __aicore__ void concat(GM_ADDR out, GM_ADDR in0, GM_ADDR in1, GM_ADDR in2,
                                             GM_ADDR in3, GM_ADDR in4, GM_ADDR in5, GM_ADDR in6,
                                             GM_ADDR in7, uint64_t s0, uint64_t s1, uint64_t s2,
                                             uint64_t s3, uint64_t s4, uint64_t s5, uint64_t s6,
                                             uint64_t s7, uint32_t nInputs, uint64_t totalBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    concat_kernel(out, in0, in1, in2, in3, in4, in5, in6, in7, s0, s1, s2, s3, s4, s5, s6, s7,
                  nInputs, totalBytes);
}
#endif
