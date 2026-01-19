/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

// 本算子由小艺团队贡献，参考论文《XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads》 [ASPLOS 2026]

template <typename Dtype>
__aicore__ inline void rmsnorm(GM_ADDR inout, GM_ADDR residual, GM_ADDR weight, GM_ADDR out,
                               uint32_t token_num, uint32_t norm_dim, float norm_eps,
                               uint32_t cnt_per_token, uint32_t step, uint32_t start_offset)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    bool has_bias = (residual != nullptr);
    float n = (float)1.0 / norm_dim;
    uint32_t total_dim = norm_dim * cnt_per_token;
    uint64_t len_burst = DIV_ROUND_UP(total_dim * sizeof(Dtype), BLOCK_SIZE);
    uint64_t len_burst_per_norm = DIV_ROUND_UP(norm_dim * sizeof(Dtype), BLOCK_SIZE);
    uint32_t inout_blocksize = ROUND_UP(total_dim * sizeof(Dtype), BLOCK_SIZE);
    uint32_t calc_blocksize = ROUND_UP(total_dim * sizeof(float), BLOCK_SIZE);
    int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    uint64_t repeat = DIV_ROUND_UP(total_dim, calcPad);
    uint64_t repeat_per_norm = DIV_ROUND_UP(norm_dim, calcPad);
    uint64_t len_burst_float_per_norm = DIV_ROUND_UP(norm_dim * sizeof(float), BLOCK_SIZE);
    uint64_t repeat_stride = DIV_ROUND_UP(norm_dim * sizeof(float), BLOCK_SIZE);
    float dupnum;

    uint32_t off = 0;
    // inout double buffer
    auto inout_addr0 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += inout_blocksize;
    auto inout_addr1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += inout_blocksize;
    __ubuf__ Dtype *inout_addr[PINGPONG_BUF_NUM] = {inout_addr0, inout_addr1};

    auto residual_addr0 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += inout_blocksize;
    auto residual_addr1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += inout_blocksize;
    __ubuf__ Dtype *residual_addr[PINGPONG_BUF_NUM] = {residual_addr0, residual_addr1};

    auto weight_addr = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += inout_blocksize;

    // calcBuf fp32
    auto calc0 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += calc_blocksize;
    auto calc1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += calc_blocksize;
    auto weight_calc = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);

    copy_gm_to_ubuf(weight_addr, (__gm__ Dtype *)weight, 0, 1, len_burst_per_norm, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (std::is_same_v<Dtype, float16_t>) {
        vconv_f162f32(weight_calc, weight_addr, repeat_per_norm, 1, 1, 8, 4);
    } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
        vconv_bf162f32(weight_calc, weight_addr, repeat_per_norm, 1, 1, 8, 4);
    }
    pipe_barrier(PIPE_V);

    for (uint32_t norm_idx = 1; norm_idx < cnt_per_token; norm_idx++) {
        auto weight_norm = weight_calc + norm_idx * norm_dim;
        copy_ubuf_to_ubuf(weight_norm, weight_calc, 0, 1, len_burst_float_per_norm, 0, 0);
        if (norm_idx == cnt_per_token - 1) {
            pipe_barrier(PIPE_V);
        }
    }

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    uint32_t loop_count = token_num;
    for (uint32_t loop = block_idx, ping = 1; loop < loop_count; loop += block_num) {
        uint32_t offset = start_offset + loop * step;
        auto event_id = ping == 1 ? EVENT_ID0 : EVENT_ID1;
        auto gm_inout = (__gm__ Dtype *)inout + offset;
        auto gm_out = (__gm__ Dtype *)out + offset;

        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        copy_gm_to_ubuf(inout_addr[event_id], gm_inout, 0, 1, len_burst, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, event_id);
        wait_flag(PIPE_MTE2, PIPE_V, event_id);

        // cast data type
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(calc0, inout_addr[event_id], repeat, 1, 1, 8, 4);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(calc0, inout_addr[event_id], repeat, 1, 1, 8, 4);
        }
        pipe_barrier(PIPE_V);

        if (has_bias) {
            auto gm_residual = (__gm__ Dtype *)residual + offset;
            auto inter_event_id = ping == 1 ? EVENT_ID2 : EVENT_ID3;
            wait_flag(PIPE_MTE3, PIPE_MTE2, inter_event_id);
            copy_gm_to_ubuf(residual_addr[event_id], gm_residual, 0, 1, len_burst, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, inter_event_id);
            wait_flag(PIPE_MTE2, PIPE_V, inter_event_id);

            // cast data type && input = input + residual
            if constexpr (std::is_same_v<Dtype, float16_t>) {
                vconv_f162f32(calc1, residual_addr[event_id], repeat, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                vadd(calc0, calc1, calc0, repeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_f322f16(residual_addr[event_id], calc0, repeat, 1, 1, 4, 8);
            } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
                vconv_bf162f32(calc1, residual_addr[event_id], repeat, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                vadd(calc0, calc1, calc0, repeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_f322bf16r(residual_addr[event_id], calc0, repeat, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, inter_event_id);
            wait_flag(PIPE_V, PIPE_MTE3, inter_event_id);
            copy_ubuf_to_gm(gm_inout, residual_addr[event_id], 0, 1, len_burst, 0, 0);
            set_flag(PIPE_MTE3, PIPE_MTE2, inter_event_id);
        }

        // rmsnorm
        // x ^ 2
        vmul(calc1, calc0, calc0, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // x ^ 2 / n
        vmuls(calc1, calc1, n, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // sum(x ^ 2)
        if (norm_dim == 128) {
            vadd(calc1, calc1, calc1 + 64, cnt_per_token, 1, 1, 1, 16, 16, 16);
            pipe_barrier(PIPE_V);
            for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
                auto calc_norm = calc1 + norm_idx * norm_dim;
                vcadd(calc_norm, calc_norm, 1, 1, 1, 8, 0);
            }
        } else {
            for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
                auto calc_norm = calc1 + norm_idx * norm_dim;
                ReduceSum(calc_norm, calc_norm, norm_dim);
            }
        }
        pipe_barrier(PIPE_V);

        SetMask(1);
        // sum(x ^ 2) + eps
        vadds(calc1, calc1, norm_eps, cnt_per_token, 1, 1, repeat_stride, repeat_stride);
        pipe_barrier(PIPE_V);

        // sqrt(sum(x ^ 2) + eps)
        vsqrt(calc1, calc1, cnt_per_token, 1, 1, repeat_stride, repeat_stride);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        set_flag(PIPE_V, PIPE_S, event_id);
        wait_flag(PIPE_V, PIPE_S, event_id);

        // duplicate item
        for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
            auto calc_norm = calc1 + norm_idx * norm_dim;
            dupnum = *calc_norm;
            set_flag(PIPE_S, PIPE_V, event_id);
            wait_flag(PIPE_S, PIPE_V, event_id);
            vector_dup(calc_norm, dupnum, repeat_per_norm, 1, 1, 8, 1);
        }
        pipe_barrier(PIPE_V);

        // x / sqrt(sum(x ^ 2) + eps)
        vdiv(calc1, calc0, calc1, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // x / sqrt(sum(x ^ 2) + eps) * weight
        vmul(calc1, weight_calc, calc1, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // cast data type
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f322f16(inout_addr[event_id], calc1, repeat, 1, 1, 4, 8);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_f322bf16r(inout_addr[event_id], calc1, repeat, 1, 1, 4, 8);
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, event_id);
        wait_flag(PIPE_V, PIPE_MTE3, event_id);

        copy_ubuf_to_gm(gm_out, inout_addr[event_id], 0, 1, len_burst, 0, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        ping ^= 1;
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    pipe_barrier(PIPE_ALL);
}


#define RMSNORM_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void rmsnorm_##dtype(GM_ADDR inout, GM_ADDR residual, GM_ADDR weight, GM_ADDR out, \
                                                      uint32_t token_num, uint32_t norm_dim, float norm_eps, \
                                                      uint32_t cnt_per_token, uint32_t step, uint32_t start_offset) \
{ \
    rmsnorm<dtype>(inout, residual, weight, out, token_num, norm_dim, norm_eps, cnt_per_token, step, start_offset); \
}
#else
#define RMSNORM_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void rmsnorm_##dtype(GM_ADDR inout, GM_ADDR residual, GM_ADDR weight, GM_ADDR out, \
                                                      uint32_t token_num, uint32_t norm_dim, float norm_eps, \
                                                      uint32_t cnt_per_token, uint32_t step, uint32_t start_offset) \
{ \
}
#endif