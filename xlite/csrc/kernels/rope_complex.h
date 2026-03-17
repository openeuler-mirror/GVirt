/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
/*
 * type: 0x1 output new q_pe
 * type: 0x2 inplace write of q
 * type: 0x3 output new q_pe & inplace write of q
 */
template <typename Dtype>
__aicore__ __inline__ void rope_complex(uint32_t nTokens, uint32_t nLocalHeads, uint32_t qDim,
                                        uint32_t qkRopeHeadDim, GM_ADDR q_ptr, GM_ADDR freqs_ptr,
                                        GM_ADDR position, GM_ADDR q_pe_ptr, GM_ADDR indices,
                                        uint32_t type)
{
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __ubuf__ float *real = (__ubuf__ float *)get_imm(0);
    __ubuf__ float *img = (__ubuf__ float *)get_imm(1 << 10);
    __ubuf__ Dtype *input = (__ubuf__ Dtype *)get_imm(2 << 10);
    __ubuf__ Dtype *ouput = (__ubuf__ Dtype *)get_imm(3 << 10);
    __ubuf__ float *freqs = (__ubuf__ float *)get_imm(4 << 10);
    __ubuf__ float *t0 = (__ubuf__ float *)get_imm(5 << 10);
    __ubuf__ float *t1 = (__ubuf__ float *)get_imm(6 << 10);
    __ubuf__ float *t2 = (__ubuf__ float *)get_imm(7 << 10);
    __ubuf__ float *t3 = (__ubuf__ float *)get_imm(8 << 10);
    __ubuf__ float *t4 = (__ubuf__ float *)get_imm(9 << 10);
    __ubuf__ float *t5 = (__ubuf__ float *)get_imm(10 << 10);
    __ubuf__ float *t6 = (__ubuf__ float *)get_imm(11 << 10);
    __ubuf__ float *t7 = (__ubuf__ float *)get_imm(12 << 10);
    __ubuf__ float *t8 = (__ubuf__ float *)get_imm(13 << 10);
    __ubuf__ float *t9 = (__ubuf__ float *)get_imm(14 << 10);
    __ubuf__ uint32_t *vgather_indices = (__ubuf__ uint32_t *)get_imm(15 << 10);

    copy_gm_to_ubuf(vgather_indices, indices, 0, 1, 8, 0, 0);
    pipe_barrier(PIPE_ALL);

    int offset = qDim - qkRopeHeadDim;

    int rope_bytes = qkRopeHeadDim * sizeof(Dtype);
    int rope_blocks = rope_bytes >> 5;
    int rope_len = qkRopeHeadDim >> 1;
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (int index = block_idx; index < nTokens * nLocalHeads; index += block_num) {
        int token_idx = index / nLocalHeads;

        uint64_t freqs_idx = (uint64_t)(*((__gm__ int64_t *)position + token_idx));
        auto *freqs_gm = (__gm__ float *)(freqs_ptr) + freqs_idx * qkRopeHeadDim;

        auto *input_gm = (__gm__ Dtype *)(q_ptr) + index * qDim + offset;
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        copy_gm_to_ubuf(input, input_gm, 0, 1, rope_blocks, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        copy_gm_to_ubuf(freqs, freqs_gm, 0, 1, rope_blocks * 2, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(t0, input, 2, 1, 1, 8, 4);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(t0, input, 2, 1, 1, 8, 4);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_V);

        set_mask_count();
        set_vector_mask(0x0, qkRopeHeadDim);

        vreducev2((__ubuf__ uint32_t *)t1, (__ubuf__ uint32_t *)t0, (__ubuf__ uint32_t *)t0, 1, 1,
                  1, 8, 0);
        pipe_barrier(PIPE_V);
        vreducev2((__ubuf__ uint32_t *)t2, (__ubuf__ uint32_t *)t0, (__ubuf__ uint32_t *)t0, 1, 1,
                  2, 8, 0);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        vreducev2((__ubuf__ uint32_t *)t3, (__ubuf__ uint32_t *)freqs, (__ubuf__ uint32_t *)freqs,
                  1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vreducev2((__ubuf__ uint32_t *)t4, (__ubuf__ uint32_t *)freqs, (__ubuf__ uint32_t *)freqs,
                  1, 1, 2, 8, 0);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        // result->real
        vmul(t5, t1, t3, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(t6, t2, t4, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vsub(real, t5, t6, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // result->img
        vmul(t7, t1, t4, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(t8, t2, t3, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(img, t7, t8, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        set_mask_count();
        set_vector_mask(0x0, qkRopeHeadDim);
        vgather((__ubuf__ uint32_t *)t9, vgather_indices, (uint64_t)real, 0, 1);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f322f16(ouput, t9, 2, 1, 1, 4, 8);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_f322bf16r(ouput, t9, 2, 1, 1, 4, 8);
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if (type & 0x1) {
            auto *q_pe = ((__gm__ Dtype *)(q_pe_ptr)) + index * qkRopeHeadDim;
            copy_ubuf_to_gm(q_pe, ouput, 0, 1, rope_blocks, 0, 0);
        }
        if (type & 0x2) {
            auto *q_q_pe = ((__gm__ Dtype *)(q_ptr)) + index * qDim + offset;
            copy_ubuf_to_gm(q_q_pe, ouput, 0, 1, rope_blocks, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        // We can drop this later.
        pipe_barrier(PIPE_ALL);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_ALL);
}

#define ROPE_COMPLEX_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void rope_complex_##dtype(                                    \
        uint32_t nTokens, uint32_t nLocalHeads, uint32_t qDim, uint32_t qkRopeHeadDim,             \
        GM_ADDR q_ptr, GM_ADDR freqs_ptr, GM_ADDR position, GM_ADDR q_pe_ptr, GM_ADDR indices,     \
        uint32_t type)                                                                             \
    {                                                                                              \
        rope_complex<dtype>(nTokens, nLocalHeads, qDim, qkRopeHeadDim, q_ptr, freqs_ptr, position, \
                            q_pe_ptr, indices, type);                                              \
    }
#else
#define ROPE_COMPLEX_FUNC_DEFINE(dtype)                                                        \
    extern "C" __global__ __aicore__ void rope_complex_##dtype(                                \
        uint32_t nTokens, uint32_t nLocalHeads, uint32_t qDim, uint32_t qkRopeHeadDim,         \
        GM_ADDR q_ptr, GM_ADDR freqs_ptr, GM_ADDR position, GM_ADDR q_pe_ptr, GM_ADDR indices, \
        uint32_t type)                                                                         \
    {                                                                                          \
    }
#endif