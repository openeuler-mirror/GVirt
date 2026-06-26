/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#define UBA(T) __ubuf__ T *
#define GMA(T) __gm__ T *

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ __inline__ void rope_complex_and_cache(
    uint32_t nTokens, uint32_t nLocalHeads, uint32_t qDim, uint32_t qkRopeHeadDim, uint32_t offset,
    uint32_t kdim, uint32_t vdim, GM_ADDR q_ptr, GM_ADDR freqs_ptr, GM_ADDR position,
    uint32_t block_size, GM_ADDR key, GM_ADDR kcache, GM_ADDR vcache, GM_ADDR slot_mapping)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint64_t buf_size = (1 << 9);

    UBA(Dtype) input_buf0 = reinterpret_cast<UBA(Dtype)>((uintptr_t)0);
    UBA(Dtype) input_buf1 = reinterpret_cast<UBA(Dtype)>(input_buf0 + buf_size);
    UBA(float) freqs_buf0 = reinterpret_cast<UBA(float)>(input_buf1 + buf_size);
    UBA(float) freqs_buf1 = reinterpret_cast<UBA(float)>(freqs_buf0 + buf_size);
    UBA(Dtype) out_buf0 = reinterpret_cast<UBA(Dtype)>(freqs_buf1 + buf_size);
    UBA(Dtype) out_buf1 = reinterpret_cast<UBA(Dtype)>(out_buf0 + buf_size);
    UBA(Dtype) kinput_buf0 = reinterpret_cast<UBA(Dtype)>(out_buf1 + buf_size);
    UBA(Dtype) kinput_buf1 = reinterpret_cast<UBA(Dtype)>(kinput_buf0 + buf_size);

    UBA(float) out_f32_buf = reinterpret_cast<UBA(float)>(kinput_buf1 + buf_size);
    UBA(float) input_f32_buf = reinterpret_cast<UBA(float)>(out_f32_buf + buf_size);
    UBA(float) x_even_buf = reinterpret_cast<UBA(float)>(input_f32_buf + buf_size);
    UBA(float) x_odd_buf = reinterpret_cast<UBA(float)>(x_even_buf + buf_size);
    UBA(float) cos_buf = reinterpret_cast<UBA(float)>(x_odd_buf + buf_size);
    UBA(float) sin_buf = reinterpret_cast<UBA(float)>(cos_buf + buf_size);
    UBA(float) x_even_cos_buf = reinterpret_cast<UBA(float)>(sin_buf + buf_size);
    UBA(float) x_odd_sin_buf = reinterpret_cast<UBA(float)>(x_even_cos_buf + buf_size);
    UBA(float) x_even_sin_buf = reinterpret_cast<UBA(float)>(x_odd_sin_buf + buf_size);
    UBA(float) x_odd_cos_buf = reinterpret_cast<UBA(float)>(x_even_sin_buf + buf_size);

    uint64_t end_addr = reinterpret_cast<uint64_t>(x_odd_cos_buf + buf_size);
    assert(end_addr <= UB_SIZE);

    UBA(Dtype) input_bufs[2] = {input_buf0, input_buf1};
    UBA(float) freqs_bufs[2] = {freqs_buf0, freqs_buf1};
    UBA(Dtype) out_bufs[2] = {out_buf0, out_buf1};
    UBA(Dtype) kinput_bufs[2] = {kinput_buf0, kinput_buf1};

    bool need_k_cache = (block_size != 0) && (kcache != nullptr) && (slot_mapping != nullptr);
    bool need_v_cache = (block_size != 0) && (vcache != nullptr) && (slot_mapping != nullptr);

    int rope_blocks = (qkRopeHeadDim * sizeof(Dtype)) >> 5;
    int v_blocks = vdim * sizeof(Dtype) >> 5;
    int remain_blocks = vdim > qkRopeHeadDim ? ((vdim - qkRopeHeadDim) * sizeof(Dtype)) >> 5 : 0;

    int event_id = 0;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    for (int index = block_idx; index < nTokens * nLocalHeads; index += block_num) {
        int token_idx = index / nLocalHeads;

        uint32_t slot_idx, block, block_offset;
        if (need_k_cache || need_v_cache) {
            slot_idx = (uint32_t)(*((__gm__ uint32_t *)slot_mapping + token_idx));
            block = slot_idx / block_size;
            block_offset = slot_idx % block_size;
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0 + event_id);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0 + event_id);
        }
        if (need_k_cache) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + event_id);
            copy_gm_to_ubuf(kinput_bufs[event_id], (__gm__ Dtype *)(key) + index * kdim, 0, 1,
                            DIV_ROUND_UP(kdim * sizeof(Dtype), BLOCK_SIZE), 0, 0);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + event_id);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + event_id);
            auto *kcache_ptr = ((__gm__ Dtype *)(kcache)) +
                               block * nLocalHeads * block_size * kdim + block_offset * kdim;
            copy_ubuf_to_gm(kcache_ptr, kinput_bufs[event_id], 0, 1,
                            DIV_ROUND_UP(kdim * sizeof(Dtype), BLOCK_SIZE), 0, 0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + event_id);
        }

        uint64_t freqs_idx = (uint64_t)(*((__gm__ int64_t *)position + token_idx));
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + event_id);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + event_id);
        auto *freqs_gm = (__gm__ float *)(freqs_ptr) + freqs_idx * qkRopeHeadDim;

        auto *input_gm = (__gm__ Dtype *)(q_ptr) + index * qDim + offset;
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);
        // input GM -> UB
        copy_gm_to_ubuf(input_bufs[event_id], input_gm, 0, 1, need_v_cache ? v_blocks : rope_blocks,
                        0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + event_id);
        // freqs GM -> UB
        copy_gm_to_ubuf(freqs_bufs[event_id], freqs_gm, 0, 1, rope_blocks * 2, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + event_id);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(input_f32_buf, input_bufs[event_id], 2, 1, 1, 8, 4);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(input_f32_buf, input_bufs[event_id], 2, 1, 1, 8, 4);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);
        pipe_barrier(PIPE_V);

        set_mask_count();
        set_vector_mask(0x0, qkRopeHeadDim);

        // [x0, x2, x4, ...]
        vreducev2((UBA(uint32_t))x_even_buf, (UBA(uint32_t))input_f32_buf,
                  (UBA(uint32_t))input_f32_buf, 1, 1, 1, 8, 0);
        // [x1, x3, x5, ...]
        vreducev2((UBA(uint32_t))x_odd_buf, (UBA(uint32_t))input_f32_buf,
                  (UBA(uint32_t))input_f32_buf, 1, 1, 2, 8, 0);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + event_id);
        // [cos(k/(theta^(2t/dim))]
        vreducev2((UBA(uint32_t))cos_buf, (UBA(uint32_t))freqs_bufs[event_id],
                  (UBA(uint32_t))freqs_bufs[event_id], 1, 1, 1, 8, 0);
        // [sin(k/(theta^(2t/dim))]
        vreducev2((UBA(uint32_t))sin_buf, (UBA(uint32_t))freqs_bufs[event_id],
                  (UBA(uint32_t))freqs_bufs[event_id], 1, 1, 2, 8, 0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + event_id);
        pipe_barrier(PIPE_V);

        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        // result->real
        // x[0::2] * cos
        vmul(x_even_cos_buf, x_even_buf, cos_buf, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        // x[1::2] * sin
        vmul(x_odd_sin_buf, x_odd_buf, sin_buf, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);

        // result->img
        // x[0::2] * sin
        vmul(x_even_sin_buf, x_even_buf, sin_buf, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        // x[1::2] * cos
        vmul(x_odd_cos_buf, x_odd_buf, cos_buf, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // x[0::2] * cos - x[1::2] * sin
        vsub(out_f32_buf, x_even_cos_buf, x_odd_sin_buf, 1 /*repeats*/, 1, 1, 1, 8, 8, 8);
        // x[0::2] * sin + x[1::2] * cos
        vadd(out_f32_buf + qkRopeHeadDim / 2, x_even_sin_buf, x_odd_cos_buf, 1 /*repeats*/, 1, 1, 1,
             8, 8, 8);

        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);
        // out FP32 -> Dtype
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f322f16(out_bufs[event_id], out_f32_buf, 2, 1, 1, 4, 8);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_f322bf16r(out_bufs[event_id], out_f32_buf, 2, 1, 1, 4, 8);
        }

        if (need_v_cache && remain_blocks > 0) {
            copy_ubuf_to_ubuf(out_bufs[event_id] + qkRopeHeadDim,
                              input_bufs[event_id] + qkRopeHeadDim, 0, 1, remain_blocks, 0, 0);
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);
        auto *q_q_pe = ((__gm__ Dtype *)(q_ptr)) + index * qDim + offset;
        // out UB -> GM
        copy_ubuf_to_gm(q_q_pe, out_bufs[event_id], 0, 1, rope_blocks, 0, 0);
        if (need_v_cache) {
            auto *vcache_ptr = ((__gm__ Dtype *)(vcache)) +
                               block * nLocalHeads * block_size * vdim + block_offset * vdim;
            copy_ubuf_to_gm(vcache_ptr, out_bufs[event_id], 0, 1, v_blocks, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);

        event_id = 1 - event_id;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_ALL);
}

#define ROPE_COMPLEX_CACHE_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void rope_complex_and_cache_##dtype(                        \
        uint32_t nTokens, uint32_t nLocalHeads, uint32_t qDim, uint32_t qkRopeHeadDim,           \
        uint32_t offset, uint32_t kdim, uint32_t vdim, GM_ADDR q_ptr, GM_ADDR freqs_ptr,         \
        GM_ADDR position, uint32_t block_size, GM_ADDR key, GM_ADDR kcache, GM_ADDR vcache,      \
        GM_ADDR slot_mapping)                                                                    \
    {                                                                                            \
        rope_complex_and_cache<dtype>(nTokens, nLocalHeads, qDim, qkRopeHeadDim, offset, kdim,   \
                                      vdim, q_ptr, freqs_ptr, position, block_size, key, kcache, \
                                      vcache, slot_mapping);                                     \
    }
#else
#define ROPE_COMPLEX_CACHE_FUNC_DEFINE(dtype)                                               \
    extern "C" __global__ __aicore__ void rope_complex_##dtype(                             \
        uint32_t nTokens, uint32_t nLocalHeads, uint32_t qDim, uint32_t qkRopeHeadDim,      \
        uint32_t offset, uint32_t kdim, uint32_t vdim, GM_ADDR q_ptr, GM_ADDR freqs_ptr,    \
        GM_ADDR position, uint32_t block_size, GM_ADDR key, GM_ADDR kcache, GM_ADDR vcache, \
        GM_ADDR slot_mapping)                                                               \
    {                                                                                       \
    }
#endif