/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#define UBA(T) (__ubuf__ T *)
#define GMA(T) (__gm__ T *)

#ifdef __DAV_C220_VEC__
template <typename SrcType, typename TarType>
__aicore__ void cast_kernel(GM_ADDR x, GM_ADDR y, uint32_t length)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint32_t max_num = MAX_REPEAT_TIMES * VECTOR_MAX_BYTESIZE / sizeof(float);
    __ubuf__ uint8_t *t1 = (__ubuf__ uint8_t *)get_imm(0);
    __ubuf__ uint8_t *t2 = (__ubuf__ uint8_t *)get_imm(
        max_num * ((std::is_same<SrcType, bfloat16_t>::value) ? sizeof(__bf16) : sizeof(float)));

    uint32_t tile_size =
        length <= max_num * get_block_num() ? DIV_ROUND_UP(length, get_block_num()) : max_num;
    uint32_t process_num = DIV_ROUND_UP(length, tile_size);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {
        uint32_t actual_len =
            length - process * tile_size >= tile_size ? tile_size : length - process * tile_size;
        // bf16 > fp32
        if ((std::is_same<SrcType, bfloat16_t>::value) && (std::is_same<TarType, float>::value)) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_ubuf(UBA(__bf16) t1, GMA(__bf16) x + process * tile_size, 0, 1,
                            DIV_ROUND_UP(actual_len * sizeof(__bf16), BLOCK_SIZE), 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            vconv_bf162f32(UBA(float) t2, UBA(__bf16) t1,
                           DIV_ROUND_UP(actual_len * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1, 8,
                           4);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            copy_ubuf_to_gm(GMA(float) y + process * tile_size, UBA(float) t2, 0, 1,
                            DIV_ROUND_UP(actual_len * sizeof(float), BLOCK_SIZE), 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }

        // fp32 > bf16
        if ((std::is_same<SrcType, float>::value) && (std::is_same<TarType, bfloat16_t>::value)) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_ubuf(UBA(float) t1, GMA(float) x + process * tile_size, 0, 1,
                            DIV_ROUND_UP(actual_len * sizeof(float), BLOCK_SIZE), 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            vconv_f322bf16r(UBA(__bf16) t2, UBA(float) t1,
                            DIV_ROUND_UP(actual_len * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1, 4,
                            8);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            copy_ubuf_to_gm(GMA(__bf16) y + process * tile_size, UBA(__bf16) t2, 0, 1,
                            DIV_ROUND_UP(actual_len * sizeof(__bf16), BLOCK_SIZE), 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

#define CAST_FUNC_DEFINE(SrcType, TarType)                                                 \
    extern "C" __global__ __aicore__ void cast_##SrcType##_##TarType(GM_ADDR x, GM_ADDR y, \
                                                                     uint32_t length)      \
    {                                                                                      \
        cast_kernel<SrcType, TarType>(x, y, length);                                       \
    }
#else
#define CAST_FUNC_DEFINE(SrcType, TarType)                                                 \
    extern "C" __global__ __aicore__ void cast_##SrcType##_##TarType(GM_ADDR x, GM_ADDR y, \
                                                                     uint32_t length)      \
    {                                                                                      \
    }
#endif