/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

#ifndef XLITE_CONCAT_SPLIT_MAX_INPUTS
#define XLITE_CONCAT_SPLIT_MAX_INPUTS 8
#endif

__aicore__ inline void split_kernel(GM_ADDR in, GM_ADDR out0, GM_ADDR out1, GM_ADDR out2,
                                    GM_ADDR out3, GM_ADDR out4, GM_ADDR out5, GM_ADDR out6,
                                    GM_ADDR out7, uint64_t s0, uint64_t s1, uint64_t s2,
                                    uint64_t s3, uint64_t s4, uint64_t s5, uint64_t s6, uint64_t s7,
                                    uint32_t nOutputs, uint32_t numPackets, uint64_t totalSize)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (numPackets == 0 || nOutputs == 0 || totalSize == 0) {
        return;
    }

    GM_ADDR outputs[XLITE_CONCAT_SPLIT_MAX_INPUTS] = {out0, out1, out2, out3,
                                                      out4, out5, out6, out7};
    uint64_t sizes[XLITE_CONCAT_SPLIT_MAX_INPUTS] = {s0, s1, s2, s3, s4, s5, s6, s7};

    // running in-packet offset at which each output starts
    uint64_t outOff[XLITE_CONCAT_SPLIT_MAX_INPUTS] = {0};
    for (uint32_t i = 1; i < nOutputs; i++) {
        outOff[i] = outOff[i - 1] + sizes[i - 1];
    }

    // Ping-pong UB staging: two halves. Each sub-segment is read into one half
    // at UB offset 0 (MTE3 reads UB sources block-aligned, so we always start a
    // segment at offset 0 of its half), then written out from that half while
    // the next segment is read into the other half.
    constexpr uint32_t PINGPONG = 2;
    uint32_t halfBuf = UB_SIZE / PINGPONG;
    __ubuf__ uint8_t *dataBuf[PINGPONG] = {
        (__ubuf__ uint8_t *)((uintptr_t)0),
        (__ubuf__ uint8_t *)((uintptr_t)halfBuf),
    };
    uint32_t segBufSize = halfBuf;  // max bytes per sub-segment

    // Total bytes across all packets = numPackets * totalSize. Block-stride the
    // flat [0, totalBytes) range so all blocks share the work evenly (regardless
    // of how few packets / outputs there are). This avoids the old
    // totalJobs = numPackets * nOutputs distribution that left cores idle when
    // numPackets == 1 (e.g. 3 jobs on 48 cores).
    uint64_t totalBytes = (uint64_t)numPackets * totalSize;

    // Seed both MTE3->MTE2 flags so the first wait of each buffer can consume.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    // Walk the flat source range in sub-segments, each <= segBufSize and cut on
    // output/packet boundaries. Sub-segment i is read into dataBuf[i%2]; the
    // previous sub-segment is written out from dataBuf[(i-1)%2] concurrently.
    uint32_t curr = 0;
    // pending write state for the previous sub-segment:
    bool hasPending = false;
    uint64_t pendGbo = 0;
    uint32_t pendSize = 0;

    for (uint64_t gbo = (uint64_t)block_idx * segBufSize; gbo < totalBytes;
         gbo += (uint64_t)block_num * segBufSize) {
        uint32_t remain =
            (uint32_t)((gbo + segBufSize > totalBytes) ? totalBytes - gbo : segBufSize);

        // Read sub-segments of [gbo, gbo+remain) into dataBuf[curr], one at a
        // time (each cut on output/packet boundary, <= segBufSize), writing out
        // the previous buffered sub-segment between reads for ping-pong overlap.
        uint64_t segGbo = gbo;
        while (remain > 0) {
            uint64_t inPkt = segGbo % totalSize;
            uint32_t o = 0;
            while (o + 1 < nOutputs && inPkt >= outOff[o + 1]) {
                o++;
            }
            uint64_t localOff = inPkt - outOff[o];
            uint64_t avail = sizes[o] - localOff;
            uint32_t take = (remain < avail) ? remain : (uint32_t)avail;
            if (take > segBufSize) {
                take = segBufSize;  // cap to UB half (only when a single segment
                                    // exceeds it; source still contiguous within
                                    // one output, so we just chunk it)
            }

            // Write out the previous buffered sub-segment from dataBuf[1-curr].
            if (hasPending) {
                uint32_t wbuf = 1 - curr;
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
                // pendGbo maps to (pkt', o', localOff'); recompute (cheap, nOutputs<=8).
                uint32_t wpkt = (uint32_t)(pendGbo / totalSize);
                uint64_t winPkt = pendGbo % totalSize;
                uint32_t wo = 0;
                while (wo + 1 < nOutputs && winPkt >= outOff[wo + 1]) {
                    wo++;
                }
                uint64_t wlocalOff = winPkt - outOff[wo];
                CopyUbufToGmAligned(
                    (__gm__ uint8_t *)outputs[wo] + (uint64_t)wpkt * sizes[wo] + wlocalOff,
                    dataBuf[wbuf], pendSize);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
            }

            // Read this sub-segment into dataBuf[curr] at offset 0.
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);
            CopyGmToUbufAligned(dataBuf[curr], (__gm__ uint8_t *)in + segGbo, take);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);

            pendGbo = segGbo;
            pendSize = take;
            hasPending = true;
            curr = 1 - curr;

            segGbo += take;
            remain -= take;
        }
    }

    // Flush the last buffered sub-segment.
    if (hasPending) {
        uint32_t wbuf = 1 - curr;
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
        uint32_t wpkt = (uint32_t)(pendGbo / totalSize);
        uint64_t winPkt = pendGbo % totalSize;
        uint32_t wo = 0;
        while (wo + 1 < nOutputs && winPkt >= outOff[wo + 1]) {
            wo++;
        }
        uint64_t wlocalOff = winPkt - outOff[wo];
        CopyUbufToGmAligned((__gm__ uint8_t *)outputs[wo] + (uint64_t)wpkt * sizes[wo] + wlocalOff,
                            dataBuf[wbuf], pendSize);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ __aicore__ void split(GM_ADDR in, GM_ADDR out0, GM_ADDR out1, GM_ADDR out2,
                                            GM_ADDR out3, GM_ADDR out4, GM_ADDR out5, GM_ADDR out6,
                                            GM_ADDR out7, uint64_t s0, uint64_t s1, uint64_t s2,
                                            uint64_t s3, uint64_t s4, uint64_t s5, uint64_t s6,
                                            uint64_t s7, uint32_t nOutputs, uint32_t numPackets,
                                            uint64_t totalSize)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    split_kernel(in, out0, out1, out2, out3, out4, out5, out6, out7, s0, s1, s2, s3, s4, s5, s6, s7,
                 nOutputs, numPackets, totalSize);
}
#endif
