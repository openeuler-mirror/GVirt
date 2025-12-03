#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

#define PINGPONG_BUF_NUM 2

// 本算子由小艺团队贡献，参考论文《XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads》 [ASPLOS 2026]
inline __aicore__ void reduce_sum(__ubuf__ float *x, uint64_t repeat)
{
    uint64_t shift;
    while (repeat >= VECTOR_MAX_NUM_OF_FP32) {
        repeat = repeat / 2;
        shift = repeat * VECTOR_MAX_NUM_OF_FP32;
        vadd(x, x, (x + shift), repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
    vcadd(x, x, repeat, 1, 1, 8, false);
    pipe_barrier(PIPE_V);
    if (repeat > 1) {
        uint64_t one = 1;
        uint64_t mask = (one << repeat) - 1;
        set_mask_norm();
        set_vector_mask((uint64_t)0, (uint64_t)mask);
        vcadd(x, x, 1, 1, 1, 1, false);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
}

template <typename Dtype>
__aicore__ void rmsnorm(GM_ADDR inout, GM_ADDR residual, GM_ADDR weight, GM_ADDR out,
                        uint32_t token_num, uint32_t norm_dim, float norm_eps,
                        uint32_t cnt_per_token, uint32_t step, uint32_t start_offset)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    bool has_bias = (residual != nullptr);
    float n = (float)1.0 / norm_dim;

    // inout double buffer
    uint32_t buf_num = 0;
    uint32_t inout_blocksize = ROUND_UP(norm_dim * sizeof(Dtype), BLOCK_SIZE);
    auto inout_dtype_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)inout_blocksize * (buf_num++));
    auto inout_dtype_ubuf_addr1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)inout_blocksize * (buf_num++));
    __ubuf__ Dtype *inout_dtype_ubuf_addr[PINGPONG_BUF_NUM] = {inout_dtype_ubuf_addr0, inout_dtype_ubuf_addr1};

    auto inter_dtype_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)inout_blocksize * (buf_num++));
    auto inter_dtype_ubuf_addr1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)inout_blocksize * (buf_num++));
    __ubuf__ Dtype *inter_dtype_ubuf_addr[PINGPONG_BUF_NUM] = {inter_dtype_ubuf_addr0, inter_dtype_ubuf_addr1};

    auto weight_dtype_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)inout_blocksize * (buf_num++));

    // calcBuf fp32
    uint32_t calcbuf_start = inout_blocksize * buf_num;
    uint32_t calc_blocksize = norm_dim * sizeof(float);
    auto calc0_float_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>((uintptr_t)calcbuf_start);
    auto calc1_float_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(calcbuf_start + calc_blocksize));
    auto weight_float_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>((uintptr_t)(calcbuf_start + calc_blocksize * 2));

    // set vector config
    uint64_t repeatf16 = DIV_ROUND_UP(norm_dim, VECTOR_MAX_NUM_OF_FP16);
    uint64_t repeatf32 = DIV_ROUND_UP(norm_dim, VECTOR_MAX_NUM_OF_FP32);
    constexpr uint64_t dst_stride = 1;
    constexpr uint64_t src0_stride = 1;
    constexpr uint64_t src1_stride = 1;
    constexpr uint64_t dst_repeat_stride = VECTOR_MAX_BYTESIZE / BLOCK_SIZE;
    constexpr uint64_t src0_repeat_stride = VECTOR_MAX_BYTESIZE / BLOCK_SIZE;
    constexpr uint64_t src1_repeat_stride = VECTOR_MAX_BYTESIZE / BLOCK_SIZE;
    uint64_t vector_2src_f16config = set_vector_xt(dst_repeat_stride, src0_repeat_stride, src1_repeat_stride, dst_stride, src0_stride, src1_stride, repeatf16);
    uint64_t vector_2src_f32config = set_vector_xt(dst_repeat_stride, src0_repeat_stride, src1_repeat_stride, dst_stride, src0_stride, src1_stride, repeatf32);
    uint64_t vector_1src_f32config = set_vector_1src_xt(dst_repeat_stride, src0_repeat_stride, dst_stride, src0_stride, repeatf32);
    uint64_t vector_fp162fp32_config = set_vector_1src_xt(8, 4, dst_stride, src0_stride, repeatf32);
    uint64_t vector_fp322fp16_config = set_vector_1src_xt(4, 8, dst_stride, src0_stride, repeatf32);

    // set copy config
    uint64_t len_burst = DIV_ROUND_UP(norm_dim * sizeof(Dtype), BLOCK_SIZE);
    constexpr uint8_t sid = 0;
    constexpr uint16_t n_burst = 1;
    constexpr uint16_t src_gap = 0;
    constexpr uint16_t dst_gap = 0;
    uint64_t dmi_config = __set_dmi_config(sid, n_burst, len_burst, src_gap, dst_gap);
    float dupnum;

    copy_gm_to_ubuf(weight_dtype_ubuf_addr0, (__gm__ Dtype *)weight, dmi_config);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (std::is_same_v<Dtype, float16_t>) {
        vconv_f162f32(weight_float_ubuf_addr0, weight_dtype_ubuf_addr0, vector_fp162fp32_config);
    } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
        vconv_bf162f32(weight_float_ubuf_addr0, weight_dtype_ubuf_addr0, vector_fp162fp32_config);
    }
    pipe_barrier(PIPE_ALL);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    uint32_t loop_count = token_num * cnt_per_token;
    for (uint32_t loop = block_idx, ping = 1; loop < loop_count; loop += block_num)
    {
        uint32_t offset = start_offset + loop / cnt_per_token * step + loop % cnt_per_token * norm_dim;
        auto event_id = ping == 1 ? EVENT_ID0 : EVENT_ID1;
        auto gm_inout = (__gm__ Dtype *)inout + offset;
        auto gm_out = (__gm__ Dtype *)out + offset;

        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        copy_gm_to_ubuf(inout_dtype_ubuf_addr[event_id], gm_inout, dmi_config);
        set_flag(PIPE_MTE2, PIPE_V, event_id);
        wait_flag(PIPE_MTE2, PIPE_V, event_id);

        // cast data type
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(calc0_float_ubuf_addr0, inout_dtype_ubuf_addr[event_id], vector_fp162fp32_config);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(calc0_float_ubuf_addr0, inout_dtype_ubuf_addr[event_id], vector_fp162fp32_config);
        }
        pipe_barrier(PIPE_V);

        if (has_bias) {
            auto gm_residual = (__gm__ Dtype *)residual + offset;
            auto inter_event_id = ping == 1 ? EVENT_ID2 : EVENT_ID3;
            wait_flag(PIPE_MTE3, PIPE_MTE2, inter_event_id);
            copy_gm_to_ubuf(inter_dtype_ubuf_addr[event_id], gm_residual, dmi_config);
            set_flag(PIPE_MTE2, PIPE_V, inter_event_id);
            wait_flag(PIPE_MTE2, PIPE_V, inter_event_id);

            // cast data type && input = input + residual
            if constexpr (std::is_same_v<Dtype, float16_t>) {
                vconv_f162f32(calc1_float_ubuf_addr0, inter_dtype_ubuf_addr[event_id], vector_fp162fp32_config);
                pipe_barrier(PIPE_V);
                vadd(calc0_float_ubuf_addr0, calc1_float_ubuf_addr0, calc0_float_ubuf_addr0, vector_2src_f32config);
                pipe_barrier(PIPE_V);
                vconv_f322f16(inter_dtype_ubuf_addr[event_id], calc0_float_ubuf_addr0, vector_fp322fp16_config);
            } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
                vconv_bf162f32(calc1_float_ubuf_addr0, inter_dtype_ubuf_addr[event_id], vector_fp162fp32_config);
                pipe_barrier(PIPE_V);
                vadd(calc0_float_ubuf_addr0, calc1_float_ubuf_addr0, calc0_float_ubuf_addr0, vector_2src_f32config);
                pipe_barrier(PIPE_V);
                vconv_f322bf16r(inter_dtype_ubuf_addr[event_id], calc0_float_ubuf_addr0, vector_fp322fp16_config);
            }
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, inter_event_id);
            wait_flag(PIPE_V, PIPE_MTE3, inter_event_id);
            copy_ubuf_to_gm(gm_inout, inter_dtype_ubuf_addr[event_id], dmi_config);
            set_flag(PIPE_MTE3, PIPE_MTE2, inter_event_id);
        }

        // rmsnorm
        // x ^ 2
        vmul(calc1_float_ubuf_addr0, calc0_float_ubuf_addr0, calc0_float_ubuf_addr0, vector_2src_f32config);
        pipe_barrier(PIPE_V);

        // x ^ 2 / n
        vmuls(calc1_float_ubuf_addr0, calc1_float_ubuf_addr0, n, vector_1src_f32config);
        pipe_barrier(PIPE_V);

        // sum(x ^ 2)
        reduce_sum(calc1_float_ubuf_addr0, repeatf32);
        pipe_barrier(PIPE_V);

        // sum(x ^ 2) + eps
        vadds(calc1_float_ubuf_addr0, calc1_float_ubuf_addr0, norm_eps, 1, 1, 1, 1, 1);
        pipe_barrier(PIPE_V);

        // sqrt(sum(x ^ 2) + eps)
        vsqrt(calc1_float_ubuf_addr0, calc1_float_ubuf_addr0, 1, 1, 1, 1, 1);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_S, event_id);
        wait_flag(PIPE_V, PIPE_S, event_id);

        // duplicate item
        dupnum = *calc1_float_ubuf_addr0;
        set_flag(PIPE_S, PIPE_V, event_id);
        wait_flag(PIPE_S, PIPE_V, event_id);

        vector_dup(calc1_float_ubuf_addr0, dupnum, repeatf32, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);

        // x / sqrt(sum(x ^ 2) + eps)
        vdiv(calc1_float_ubuf_addr0, calc0_float_ubuf_addr0, calc1_float_ubuf_addr0, vector_2src_f32config);
        pipe_barrier(PIPE_V);

        // x / sqrt(sum(x ^ 2) + eps) * weight
        vmul(calc1_float_ubuf_addr0, weight_float_ubuf_addr0, calc1_float_ubuf_addr0, vector_2src_f32config);
        pipe_barrier(PIPE_V);

        // cast data type
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f322f16(inout_dtype_ubuf_addr[event_id], calc1_float_ubuf_addr0, vector_fp322fp16_config);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_f322bf16r(inout_dtype_ubuf_addr[event_id], calc1_float_ubuf_addr0, vector_fp322fp16_config);
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, event_id);
        wait_flag(PIPE_V, PIPE_MTE3, event_id);

        copy_ubuf_to_gm(gm_out, inout_dtype_ubuf_addr[event_id], dmi_config);
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

RMSNORM_FUNC_DEFINE(float16_t);
RMSNORM_FUNC_DEFINE(bfloat16_t);

#endif