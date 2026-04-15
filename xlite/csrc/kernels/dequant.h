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

    uint32_t n_pad = ROUND_UP(n, (256 / sizeof(dtype)));

    GMA(dtype) in_gm_buf = reinterpret_cast<GMA(dtype)>(in);
    GMA(float32_t) scale_gm_buf = reinterpret_cast<GMA(float32_t)>(scale);
    GMA(dtype) out_gm_buf = reinterpret_cast<GMA(dtype)>(out);

    UBA(dtype) in_ub_buf0 = reinterpret_cast<UBA(dtype)>((uintptr_t)0);
    UBA(dtype) in_ub_buf1 = reinterpret_cast<UBA(dtype)>((uintptr_t)(in_ub_buf0 + n_pad));

    UBA(float32_t) tmp_ub_buf0 = reinterpret_cast<UBA(float32_t)>((uintptr_t)(in_ub_buf1 + n_pad));
    UBA(float32_t) tmp_ub_buf1 = reinterpret_cast<UBA(float32_t)>((uintptr_t)(tmp_ub_buf0 + n_pad));

    UBA(float32_t) mul_ub_buf0 = reinterpret_cast<UBA(float32_t)>((uintptr_t)(tmp_ub_buf1 + n_pad));
    UBA(float32_t) mul_ub_buf1 = reinterpret_cast<UBA(float32_t)>((uintptr_t)(mul_ub_buf0 + n_pad));

    UBA(bfloat16_t)
    out_ub_buf0 = reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(mul_ub_buf1 + n_pad));
    UBA(bfloat16_t)
    out_ub_buf1 = reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(out_ub_buf0 + n_pad));

    UBA(float32_t)
    scale_ub_buf = reinterpret_cast<UBA(float32_t)>((uintptr_t)(out_ub_buf1 + n_pad));
    UBA(float32_t) end_addr = reinterpret_cast<UBA(float32_t)>((uintptr_t)(scale_ub_buf + 1));
    assert((uint64_t)end_addr <= UB_SIZE);

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

    for (uint32_t process = block_idx; process < m; process += uint32_t(block_num)) {
        // GM -> UB, in_gm_buf -> in_ub_buf
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);
        copy_gm_to_ubuf_align_b16(in_ub_buf[event_id], in_gm_buf + process * n, 0, 1,
                                  n * sizeof(dtype), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);

        if (hasScale) {
            // GM -> UB, scale_gm_buf -> scale_ub_buf
            copy_gm_to_ubuf_align_b16(scale_ub_buf, scale_gm_buf + process, 0, 1, sizeof(float), 0,
                                      0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        }

        // bf16 > fp32
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);
        vconv_f162f32(tmp_ub_buf[event_id], in_ub_buf[event_id],
                      DIV_ROUND_UP(n, VECTOR_MAX_NUM_OF_FP32), 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);

        // VMULS, tmp_ub_buf -> VMULS(in * scale) -> tmp_ub_buf
        if (hasScale) {
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            vmuls(mul_ub_buf[event_id], tmp_ub_buf[event_id], float(*scale_ub_buf),
                  DIV_ROUND_UP(n, VECTOR_MAX_NUM_OF_FP32), 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            tmp_ptr = mul_ub_buf[event_id];
        }
        tmp_ptr = hasScale ? mul_ub_buf[event_id] : tmp_ub_buf[event_id];

        // F32 -> BF16
        vconv_f322bf16r(out_ub_buf[event_id], tmp_ptr, DIV_ROUND_UP(n, VECTOR_MAX_NUM_OF_FP32), 1,
                        1, 4, 8);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);

        // UB -> GM, out_ub_buf -> out_gm_buf
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);
        copy_ubuf_to_gm_align_b16(out_gm_buf + process * n, out_ub_buf[event_id], 0, 1,
                                  n * sizeof(bfloat16_t), 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);

        event_id = 1 - event_id;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
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
