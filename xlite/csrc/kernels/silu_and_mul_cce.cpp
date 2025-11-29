#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

#define GMA(T) (__gm__ T*)
#define UBA(T) (__ubuf__ T*)
#define UB_SIZE 196608
#define MAX_HIDDENSIZE_PER_PIECE 38912 // MAX_HIDDENSIZE_PER_PIECE = UB_SIZE(196608) / buffer_num(5)

template<typename SrcType, typename CalType>
__aicore__ void silu_and_mul(GM_ADDR x, GM_ADDR y, GM_ADDR pm, int32_t num_tokens, int32_t dim)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (pm) {
        uint32_t pm_val = *((__gm__ uint32_t *)pm);
        num_tokens = pm_val < num_tokens ? pm_val : num_tokens;
    }

    // split the matrix in a row-major manner such that each row (2 * dim elements) is contained in a single block
    int32_t block_dims = get_block_num();
    int32_t tokens_per_block = DIV_ROUND_UP(num_tokens, block_dims);

    // calculate actual batch size == how many tokens (rows) are handled in current block
    int32_t tokens_copied = tokens_per_block * get_block_idx();
    int32_t tokens_to_copy = tokens_per_block;
    if (tokens_copied + tokens_to_copy > num_tokens) {
        tokens_to_copy = num_tokens - tokens_copied;
    }

    if (tokens_to_copy <= 0) {
        return;
    }

    int32_t x_offset = tokens_copied * dim * 2;
    int32_t y_offset = tokens_copied * dim;
    __gm__ SrcType *x_start = (__gm__ SrcType *)x + x_offset;
    __gm__ SrcType *y_start = (__gm__ SrcType *)y + y_offset;

    // Double buffer, each needs 2 sub-buffer and needs 5 buffer in total
    uint64_t ubuf_unit_size = MAX_HIDDENSIZE_PER_PIECE;
    __ubuf__ CalType *ubuf0 = (__ubuf__ CalType *)(uintptr_t)0x00000;
    __ubuf__ CalType *ubuf1 = (__ubuf__ CalType *)(uintptr_t)(ubuf_unit_size);
    __ubuf__ CalType *ubuf2 = (__ubuf__ CalType *)(uintptr_t)(ubuf_unit_size * 2);
    __ubuf__ CalType *ubuf3 = (__ubuf__ CalType *)(uintptr_t)(ubuf_unit_size * 3);
    __ubuf__ CalType *cal_ubuf = (__ubuf__ CalType *)(uintptr_t)(ubuf_unit_size * 4);

    int32_t token_len = dim * sizeof(CalType);
    int32_t tokens_per_tile = ubuf_unit_size / token_len;

    // Config for vector operation
    constexpr uint64_t dst_stride = 1;
    constexpr uint64_t src0_stride = 1;
    constexpr uint64_t src1_stride = 1;
    constexpr uint64_t dst_repeat_stride = VECTOR_MAX_BYTESIZE / BLOCK_SIZE;
    constexpr uint64_t src0_repeat_stride = VECTOR_MAX_BYTESIZE / BLOCK_SIZE;
    constexpr uint64_t src1_repeat_stride = VECTOR_MAX_BYTESIZE / BLOCK_SIZE;
    uint64_t repeat_dtype = DIV_ROUND_UP(tokens_per_tile * dim, VECTOR_MAX_BYTESIZE / sizeof(CalType));
    uint64_t vector_config3ops = set_vector_xt(dst_repeat_stride, src0_repeat_stride, src1_repeat_stride, dst_stride, src0_stride, src1_stride, repeat_dtype);
    uint64_t vector_config2ops = set_vector_1src_xt(dst_repeat_stride, src0_repeat_stride, dst_stride, src0_stride, repeat_dtype);

    // Config for copy operation
    constexpr uint8_t sid = 0;
    constexpr uint16_t n_burst = 1;
    constexpr uint16_t src_gap = 0;
    constexpr uint16_t dst_gap = 0;
    uint64_t len_burst = DIV_ROUND_UP(tokens_per_tile * token_len, BLOCK_SIZE);
    uint64_t out_copy_config = __set_dmi_config(sid, n_burst, len_burst, src_gap, dst_gap);
    uint64_t token_len_burst = DIV_ROUND_UP(token_len, BLOCK_SIZE);
    uint64_t token_copy_config = __set_dmi_config(sid, tokens_per_tile, token_len_burst, token_len_burst, dst_gap);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    for (int32_t copied = 0, ping = 1; copied < tokens_to_copy; copied += tokens_per_tile) {
        __ubuf__ CalType *x_ubuf = ping ? ubuf0 : ubuf2;
        __ubuf__ CalType *y_ubuf = ping ? ubuf1 : ubuf3;
        auto event_id = ping ? EVENT_ID0 : EVENT_ID1;

        int32_t to_copy = tokens_per_tile;
        if (copied + to_copy > tokens_to_copy) {
            to_copy = tokens_to_copy - copied;
            len_burst = DIV_ROUND_UP(to_copy * token_len, BLOCK_SIZE);
            out_copy_config = __set_dmi_config(sid, n_burst, len_burst, src_gap, dst_gap);
            token_copy_config = __set_dmi_config(sid, to_copy, token_len_burst, token_len_burst, dst_gap);
        }

        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);

        copy_gm_to_ubuf(x_ubuf, x_start, token_copy_config);
        copy_gm_to_ubuf(y_ubuf, x_start + dim, token_copy_config);

        x_start += 2 * dim * to_copy;
        set_flag(PIPE_MTE2, PIPE_V, event_id);
        wait_flag(PIPE_MTE2, PIPE_V, event_id);

        // -x
        vmuls(cal_ubuf, x_ubuf, (CalType)-1.0, vector_config2ops);
        pipe_barrier(PIPE_V);

        // e^-x
        vexp(cal_ubuf, cal_ubuf, vector_config2ops);
        pipe_barrier(PIPE_V);

        // 1 + e^-x
        vadds(cal_ubuf, cal_ubuf, (CalType)1.0, vector_config2ops);
        pipe_barrier(PIPE_V);

        // silu = x / (1 + e^-x)
        vdiv(cal_ubuf, x_ubuf, cal_ubuf, vector_config3ops);
        pipe_barrier(PIPE_V);

        // x * silu(x) = y * x / (1 + e^-x)
        vmul(x_ubuf, y_ubuf, cal_ubuf, vector_config3ops);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, event_id);
        wait_flag(PIPE_V, PIPE_MTE3, event_id);

        copy_ubuf_to_gm(y_start, x_ubuf, out_copy_config);
        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);

        ping = 1 - ping;
        y_start += to_copy * dim;
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define SILU_AND_MUL_FUNC_DEFINE(dtype, cast_type) \
extern "C" __global__ __aicore__ void silu_and_mul_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR pm, int n_tokens, int dim) \
{ \
    silu_and_mul<dtype, cast_type>(x, y, pm, n_tokens, dim); \
}

SILU_AND_MUL_FUNC_DEFINE(float, float);
SILU_AND_MUL_FUNC_DEFINE(float16_t, float16_t);


extern "C" __global__ __aicore__ void silu_and_mul_bfloat16_t(GM_ADDR x, GM_ADDR y, GM_ADDR pm, int n_tokens, int dim)
{
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_atomic_none();

    if (pm) {
        uint32_t pm_val = *((__gm__ uint32_t *)pm);
        n_tokens = pm_val < n_tokens ? pm_val : n_tokens;
    }

    int step = dim * 2;
    int dim_split = dim;
    int max_dim = DIV_ROUND_UP(UB_SIZE, (6 * sizeof(float) + 4 * sizeof(bfloat16_t))); //支持ping-pong
    int split = 1;
    while (dim_split > max_dim) {
        dim_split = dim_split / 2;
        split = split * 2;
    }
    
    __ubuf__ float* x32_ub0 = reinterpret_cast<__ubuf__ float*>((uintptr_t)0);
    __ubuf__ float* x32_ub1 = reinterpret_cast<__ubuf__ float*>(x32_ub0 + 2 * dim_split);
    __ubuf__ float* tmp = reinterpret_cast<__ubuf__ float*>(x32_ub0 + 4 * dim_split);
    __ubuf__ float* output_ub = reinterpret_cast<__ubuf__ float*>(tmp + dim_split);
    __ubuf__ bfloat16_t* x32_ub0_bf16 = reinterpret_cast<__ubuf__ bfloat16_t*>(output_ub + dim_split);
    __ubuf__ bfloat16_t* x32_ub1_bf16 = reinterpret_cast<__ubuf__ bfloat16_t*>(x32_ub0_bf16 + 2 * dim_split);

    int burst_copy = DIV_ROUND_UP(dim_split * sizeof(bfloat16_t), BLOCK_SIZE);
    int repeat_cal = DIV_ROUND_UP(dim_split, VECTOR_MAX_NUM_OF_FP32);

    int event_id = 0;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (int index = block_idx; index < n_tokens * split; index += block_num) {
        int row = index / split;
        int line = index % split;
        __gm__ bfloat16_t* gm_x_first_half = GMA(bfloat16_t)(x) + row * step + line * dim_split;
        __gm__ bfloat16_t* gm_x_second_half = GMA(bfloat16_t)(x) + row * step + line * dim_split + dim;
        __gm__ bfloat16_t* gm_y = GMA(bfloat16_t)(y) + row * dim + line * dim_split;
        auto x32_ub = event_id == 0 ? x32_ub0 : x32_ub1;
        auto x32_ub_bf16 = event_id == 0 ? x32_ub0_bf16 : x32_ub1_bf16;

        wait_flag(PIPE_V, PIPE_MTE2, event_id);
        copy_gm_to_ubuf(x32_ub_bf16, gm_x_first_half, 0, 1, burst_copy, 0, 0);
        copy_gm_to_ubuf(x32_ub_bf16 + dim_split, gm_x_second_half, 0, 1, burst_copy, 0, 0);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        vconv_bf162f32(x32_ub, x32_ub_bf16, 2 * repeat_cal, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);

        // -x
        vmuls(tmp, x32_ub, float(-1.0000000000000000e+00f), repeat_cal, 1, 1, 8, 8); // 处理前dim个数
        pipe_barrier(PIPE_V);

        // exp(-x)
        vexp(tmp, tmp, repeat_cal, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // 1 + exp(-x)
        vadds(tmp, tmp, float(1.0), repeat_cal, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // x * sigmoid(x) = x / (1 + exp(-x))
        vdiv(tmp, x32_ub, tmp, repeat_cal, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // mul
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        vmul(output_ub, tmp, x32_ub + dim_split, repeat_cal, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322bf16r(UBA(bfloat16_t)output_ub, output_ub, repeat_cal, 1, 1, 4, 8);
        set_flag(PIPE_V, PIPE_MTE2, event_id);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(gm_y, UBA(bfloat16_t)output_ub, 0, 1, burst_copy, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        event_id = 1 - event_id;
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_ALL);
}

#endif