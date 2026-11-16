/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ void embed_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t dim, uint32_t batch_size,
                             uint32_t emb_start_idx, uint32_t emb_end_idx, uint32_t tp_size)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __gm__ Dtype *x_gm_addr = ((__gm__ Dtype *)x);
    __gm__ uint32_t *y_gm_addr = ((__gm__ uint32_t *)y);
    __gm__ Dtype *z_gm_addr = ((__gm__ Dtype *)z);
    // UB layout (dim elements each): [zero template | staging ping | staging
    // pong]. Rows are gathered by token id, so per-row transfers cannot be
    // merged; the win comes from pipelining: the MTE2 read of row i+1 into one
    // staging half overlaps the MTE3 write of row i from the other half. The
    // previous implementation fenced PIPE_ALL between every row's read and
    // write, serializing the copy engine phases row by row.
    // Capacity (3 * dim elements <= EMBED_UB_BUDGET, the shared kernel_param.h
    // constant equal to this arch's UB_SIZE) and dim % 16 == 0 are enforced
    // host-side in XliteOpEmbed; the assert below is debug-only (NDEBUG compiles
    // it out), so the host check is the real guard. Kernel and host assert the
    // same single constant, so the two checks cannot drift apart.
    assert((uint64_t)dim * 3 * sizeof(Dtype) <= EMBED_UB_BUDGET && dim % 16 == 0);
    __ubuf__ Dtype *zero_ub_addr = (__ubuf__ Dtype *)((uintptr_t)0);
    __ubuf__ Dtype *stage_ub_addr[2] = {
        (__ubuf__ Dtype *)((uintptr_t)(dim * 2)),
        (__ubuf__ Dtype *)((uintptr_t)(dim * 4)),
    };
    int start = block_idx;
    uint16_t n_burst = 1;
    uint16_t len_burst = dim / 16;  // Expected embedding dimension size could be divided by 16

    if (tp_size > 1) {
        vector_dup(zero_ub_addr, (Dtype)0, dim / VECTOR_MAX_NUM_OF_FP16, 1, 1, 8, 0);
    }
    // Make the zero template visible to MTE3 once, instead of fencing after
    // every row like the previous per-row pipe_barrier(PIPE_ALL) loop.
    pipe_barrier(PIPE_ALL);

    // Below a few rows per core the ping-pong flag chain costs more per row
    // than it wins back: there is no subsequent row whose read can overlap
    // the current write, so the loop degenerates to flag overhead on top of
    // the same serialized copy phases. Keep the barrier-per-row loop for that
    // regime (decode batches) and pipeline the rest (prefill / large batches).
    constexpr uint32_t PIPE_MIN_ROWS_PER_CORE = 4;
    if (batch_size < (uint32_t)block_num * PIPE_MIN_ROWS_PER_CORE) {
        for (int index = start; index < (int)batch_size; index += block_num) {
            uint32_t row = *(y_gm_addr + index);
            if ((row >= emb_end_idx) || (row < emb_start_idx)) {
                copy_ubuf_to_gm(z_gm_addr + (uint64_t)index * dim, zero_ub_addr, 0, n_burst,
                                len_burst, 0, 0);
            } else {
                auto *row_addr = x_gm_addr + (uint64_t)(row - emb_start_idx) * dim;
                copy_gm_to_ubuf(stage_ub_addr[0], row_addr, 0, n_burst, len_burst, 0, 0);
                pipe_barrier(PIPE_ALL);
                copy_ubuf_to_gm(z_gm_addr + (uint64_t)index * dim, stage_ub_addr[0], 0, n_burst,
                                len_burst, 0, 0);
            }
            pipe_barrier(PIPE_ALL);
        }
        return;
    }

    // Seed both MTE3->MTE2 flags so each staging half's first wait can consume.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    uint32_t curr = 0;
    bool hasPending = false;
    uint64_t pendIdx = 0;
    for (int index = start; index < (int)batch_size; index += block_num) {
        uint32_t row = *(y_gm_addr + index);
        if ((row >= emb_end_idx) || (row < emb_start_idx)) {
            // Out-of-range row: write the zero template straight to the output.
            // Disjoint output row and zero_ub is never rewritten after init,
            // so this write is independent of the ping-pong chain.
            copy_ubuf_to_gm(z_gm_addr + (uint64_t)index * dim, zero_ub_addr, 0, n_burst, len_burst,
                            0, 0);
        } else {
            auto *row_addr = x_gm_addr + (uint64_t)(row - emb_start_idx) * dim;
            // A staging half is free once the MTE3 write from two rows ago has
            // drained (flag set after that write).
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);
            copy_gm_to_ubuf(stage_ub_addr[curr], row_addr, 0, n_burst, len_burst, 0, 0);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);
            // Write out the previously staged row while this one is being read.
            if (hasPending) {
                uint32_t wbuf = 1 - curr;
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
                copy_ubuf_to_gm(z_gm_addr + pendIdx * dim, stage_ub_addr[wbuf], 0, n_burst,
                                len_burst, 0, 0);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
            }
            pendIdx = index;
            hasPending = true;
            curr = 1 - curr;
        }
    }

    // Flush the last staged row.
    if (hasPending) {
        uint32_t wbuf = 1 - curr;
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
        copy_ubuf_to_gm(z_gm_addr + pendIdx * dim, stage_ub_addr[wbuf], 0, n_burst, len_burst, 0,
                        0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define EMBED_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void embed_kernel_##dtype(                             \
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t dim, uint32_t batch_size,                 \
        uint32_t emb_start_idx, uint32_t emb_end_idx, uint32_t tp_size)                     \
    {                                                                                       \
        embed_kernel<dtype>(x, y, z, dim, batch_size, emb_start_idx, emb_end_idx, tp_size); \
    }
#else
#define EMBED_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void embed_kernel_##dtype(             \
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t dim, uint32_t batch_size, \
        uint32_t emb_start_idx, uint32_t emb_end_idx, uint32_t tp_size)     \
    {                                                                       \
    }
#endif
