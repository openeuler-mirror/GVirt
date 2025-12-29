/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#define UBA(T) (__ubuf__ T*)
#define GMA(T) (__gm__ T*)

#ifdef __DAV_C220_VEC__
// 2维矩阵加法
template <typename Dtype>
__aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t x_numel, uint32_t y_numel)
{

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __ubuf__ uint8_t* t1_dtype = (__ubuf__ uint8_t*)get_imm(0);
    __ubuf__ uint8_t* t2_dtype = (__ubuf__ uint8_t*)get_imm(y_numel * 2);
    __ubuf__ uint8_t* t1_float = (__ubuf__ uint8_t*)get_imm(y_numel * 4);
    __ubuf__ uint8_t* t2_float = (__ubuf__ uint8_t*)get_imm(y_numel * 8);


    uint32_t process_num = x_numel;
    uint16_t vec_repeat_float = DIV_ROUND_UP(y_numel * sizeof(float), 256);

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for(uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {
        pipe_barrier(PIPE_ALL);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        copy_gm_to_ubuf(UBA(Dtype)t1_dtype,
                        GMA(Dtype)x + process * y_numel,
                        0,
                        1,
                        DIV_ROUND_UP(y_numel * sizeof(Dtype), BLOCK_SIZE), 0, 0);
        pipe_barrier(PIPE_ALL);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        copy_gm_to_ubuf(UBA(Dtype)t2_dtype,
                        GMA(Dtype)y + process * y_numel,
                        0,
                        1,
                        DIV_ROUND_UP(y_numel * sizeof(Dtype), BLOCK_SIZE), 0, 0);
        pipe_barrier(PIPE_ALL);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(UBA(float)t1_float, UBA(Dtype)t1_dtype, vec_repeat_float, 1, 1, 8, 4);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(UBA(float)t1_float, UBA(Dtype)t1_dtype, vec_repeat_float, 1, 1, 8, 4);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_ALL);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(UBA(float)t2_float, UBA(Dtype)t2_dtype, vec_repeat_float, 1, 1, 8, 4);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(UBA(float)t2_float, UBA(Dtype)t2_dtype, vec_repeat_float, 1, 1, 8, 4);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        pipe_barrier(PIPE_ALL);

        vadd(UBA(float)t1_float, UBA(float)t1_float, UBA(float)t2_float, vec_repeat_float, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_ALL);

        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f322f16r(UBA(Dtype)t2_float, UBA(float)t1_float, vec_repeat_float, 1, 1, 4, 8);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_f322bf16r(UBA(Dtype)t2_float, UBA(float)t1_float, vec_repeat_float, 1, 1, 4, 8);
        }
        pipe_barrier(PIPE_ALL);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(GMA(Dtype)z + process * y_numel, UBA(Dtype)t2_float, 0, 1, DIV_ROUND_UP(y_numel * sizeof(Dtype), BLOCK_SIZE), 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define ADD_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void add_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t x_numel, uint32_t y_numel) \
{ \
    add<dtype>(x, y, z, x_numel, y_numel); \
}

ADD_FUNC_DEFINE(float16_t);
ADD_FUNC_DEFINE(bfloat16_t);
#endif