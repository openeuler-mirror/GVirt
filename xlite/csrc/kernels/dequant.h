/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#define UBA(T) __ubuf__ T *
#define GMA(T) __gm__ T *
#define PINGPONG 2

#ifdef __DAV_C220_VEC__

template <typename dtype>
__aicore__ inline void dequant(GM_ADDR in, GM_ADDR scale, GM_ADDR out, uint32_t m, uint32_t n,
                               bool hasScale)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    assert((n % (256 / sizeof(dtype))) == 0);

    GMA(dtype) in_gm_buf = reinterpret_cast<GMA(dtype)>(in);
    GMA(float32_t) scale_gm_buf = reinterpret_cast<GMA(float32_t)>(scale);
    GMA(dtype) out_gm_buf = reinterpret_cast<GMA(dtype)>(out);

    UBA(dtype) in_ub_buf0 = reinterpret_cast<UBA(dtype)>((uintptr_t)0);
    UBA(dtype) in_ub_buf1 = reinterpret_cast<UBA(dtype)>((uintptr_t)(in_ub_buf0 + n));

    UBA(float32_t) tmp_ub_buf0 = reinterpret_cast<UBA(float32_t)>((uintptr_t)(in_ub_buf1 + n));
    UBA(float32_t) tmp_ub_buf1 = reinterpret_cast<UBA(float32_t)>((uintptr_t)(tmp_ub_buf0 + n));

    UBA(float32_t) mul_ub_buf0 = reinterpret_cast<UBA(float32_t)>((uintptr_t)(tmp_ub_buf1 + n));
    UBA(float32_t) mul_ub_buf1 = reinterpret_cast<UBA(float32_t)>((uintptr_t)(mul_ub_buf0 + n));

    UBA(bfloat16_t) out_ub_buf0 = reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(mul_ub_buf1 + n));
    UBA(bfloat16_t) out_ub_buf1 = reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(out_ub_buf0 + n));

    UBA(float32_t) scale_ub_buf = reinterpret_cast<UBA(float32_t)>((uintptr_t)(out_ub_buf1 + n));

    UBA(dtype) in_ub_buf[PINGPONG] = {in_ub_buf0, in_ub_buf1};
    UBA(float32_t) tmp_ub_buf[PINGPONG] = {tmp_ub_buf0, tmp_ub_buf1};
    UBA(float32_t) mul_ub_buf[PINGPONG] = {mul_ub_buf0, mul_ub_buf1};
    UBA(bfloat16_t) out_ub_buf[PINGPONG] = {out_ub_buf0, out_ub_buf1};

    UBA(float32_t) tmp_ptr;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    int event_id = 0;

    // GM -> UB, scale_gm_buf -> scale_ub_buf
    if (hasScale) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        copy_gm_to_ubuf(scale_ub_buf, scale_gm_buf, 0, 1,
                        DIV_ROUND_UP(n * sizeof(float32_t), BLOCK_SIZE), 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    }

    for (uint32_t process = block_idx; process < m; process += uint32_t(block_num)) {
        // GM -> UB, in_gm_buf -> in_ub_buf
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);
        copy_gm_to_ubuf(in_ub_buf[event_id], in_gm_buf + process * n, 0, 1,
                        DIV_ROUND_UP(n * sizeof(dtype), BLOCK_SIZE), 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);

        // bf16 > fp32
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);
        vconv_f162f32(tmp_ub_buf[event_id], in_ub_buf[event_id],
                      DIV_ROUND_UP(n * sizeof(float32_t), VECTOR_MAX_BYTESIZE), 1, 1, 8, 4);
        pipe_barrier(PIPE_V);

        // VMUL, tmp_ub_buf -> VMUL(in * scale) -> tmp_ub_buf
        if (hasScale) {
            if (process < uint32_t(block_num)) {
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            }
            vmul(mul_ub_buf[event_id], tmp_ub_buf[event_id], scale_ub_buf,
                 DIV_ROUND_UP(n * sizeof(float32_t), VECTOR_MAX_BYTESIZE), 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if (process < uint32_t(block_num)) {
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            }
            tmp_ptr = mul_ub_buf[event_id];
        }
        tmp_ptr = hasScale ? mul_ub_buf[event_id] : tmp_ub_buf[event_id];

        // F32 -> BF16
        vconv_f322bf16r(out_ub_buf[event_id], tmp_ptr,
                        DIV_ROUND_UP(n * sizeof(float32_t), VECTOR_MAX_BYTESIZE), 1, 1, 4, 8);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);

        // UB -> GM, out_ub_buf -> out_gm_buf
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);
        copy_ubuf_to_gm(out_gm_buf + process * n, out_ub_buf[event_id], 0, 1,
                        DIV_ROUND_UP(n * sizeof(bfloat16_t), BLOCK_SIZE), 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);

        event_id = 1 - event_id;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    if (hasScale) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define DEQUANT_FUNC_DEFINE(dtype)                                                                \
    extern "C" __global__ __aicore__ void dequant_##dtype(GM_ADDR in, GM_ADDR scale, GM_ADDR out, \
                                                          uint32_t m, uint32_t n, bool has_scale) \
    {                                                                                             \
        dequant<dtype>(in, scale, out, m, n, has_scale);                                          \
    }
#else
#define DEQUANT_FUNC_DEFINE(dtype)                                                                \
    extern "C" __global__ __aicore__ void dequant_##dtype(GM_ADDR in, GM_ADDR scale, GM_ADDR out, \
                                                          uint32_t m, uint32_t n, bool has_scale) \
    {                                                                                             \
    }
#endif
