/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

#define HEAD_SIZE_64 64
#define HEAD_SIZE_128 128

// 本算子由小艺团队贡献，参考论文《XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM
// Workloads》 [ASPLOS 2026]
// Rotate x (staged in local_buf as Dtype) against the fp32 cos/sin buffers and
// leave the fp32 result in local_fp32_buf. COSSIN_FP32=true leaves the single
// final fp32->Dtype rounding to the caller (after its fp32 scale step);
// COSSIN_FP32=false [LEGACY, bf16 only] emulates torch_npu per-op bf16 rounding
// via an intermediate round-trip and rounds the result to local_buf.
template <typename Dtype, bool COSSIN_FP32>
__aicore__ inline void calc_cossin_cast(__gm__ Dtype *gm_buf_loop, __ubuf__ Dtype *local_buf,
                                        __ubuf__ Dtype *tmp_buf, __ubuf__ float *local_fp32_buf,
                                        __ubuf__ float *calc_fp32_buf, __ubuf__ float *sin_fp32_buf,
                                        __ubuf__ float *cos_fp32_buf, __ubuf__ float *sin_h_buf,
                                        __ubuf__ float *cos_h_buf, __ubuf__ float *sin_w_buf,
                                        __ubuf__ float *cos_w_buf, uint32_t embed_dim,
                                        uint32_t rot_dim, int event_id, uint32_t n_head,
                                        uint32_t head_size, uint32_t pos_dim, uint64_t mrope_mask_h,
                                        uint64_t mrope_mask_w, uint64_t copy_param)
{
    uint64_t calc_size = n_head * head_size;
    uint64_t head_stride = DIV_ROUND_UP(head_size * sizeof(float), BLOCK_SIZE);
    copy_gm_to_ubuf(local_buf, gm_buf_loop, copy_param);
    set_flag(PIPE_MTE2, PIPE_V, event_id);
    wait_flag(PIPE_MTE2, PIPE_V, event_id);

    convert_input<Dtype>(local_fp32_buf, local_buf,
                         DIV_ROUND_UP(calc_size * sizeof(float), VECTOR_MAX_BYTESIZE));
    pipe_barrier(PIPE_V);

    if (pos_dim > 1) {
        set_vector_mask(0x0, mrope_mask_h + mrope_mask_w);
        vector_dup(sin_fp32_buf, float(0), 1, 1, 1, 8, 0);
        vector_dup(cos_fp32_buf, float(0), 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);

        set_vector_mask(0x0, mrope_mask_h);
        vadd(sin_fp32_buf, sin_fp32_buf, sin_h_buf, 1, 1, 1, 1, 8, 8, 8);
        vadd(cos_fp32_buf, cos_fp32_buf, cos_h_buf, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        set_vector_mask(0x0, mrope_mask_w);
        vadd(sin_fp32_buf, sin_fp32_buf, sin_w_buf, 1, 1, 1, 1, 8, 8, 8);
        vadd(cos_fp32_buf, cos_fp32_buf, cos_w_buf, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        copy_ubuf_to_ubuf(sin_fp32_buf + embed_dim, sin_fp32_buf, 0, 1,
                          DIV_ROUND_UP(embed_dim * sizeof(float), BLOCK_SIZE), 1, 1);
        copy_ubuf_to_ubuf(cos_fp32_buf + embed_dim, cos_fp32_buf, 0, 1,
                          DIV_ROUND_UP(embed_dim * sizeof(float), BLOCK_SIZE), 1, 1);
        pipe_barrier(PIPE_V);
    }

    if (rot_dim == HEAD_SIZE_128) {
        // q * sin (q : numHeads * headDim  sin: 1 * headDim (headDim = 128))
        // fp32时，需计算的sin大小为 headDim(128) * 4 Byte = 512 Byte, 为VECTOR_MAX_BYTESIZE(256
        // Byte)的2倍, 所以计算时分两次计算 相邻两次执行，目的操作数相同block地址步长为 512 / 32
        // (1个datablock长度为32Byte) = 16
        vmul(calc_fp32_buf, local_fp32_buf, sin_fp32_buf, n_head, 1, 1, 1, head_stride, head_stride,
             0);
        vmul(calc_fp32_buf + embed_dim, local_fp32_buf + embed_dim, sin_fp32_buf, n_head, 1, 1, 1,
             head_stride, head_stride, 0);
        pipe_barrier(PIPE_V);
        // q * cos (q : numHeads * headDim  cos: 1 * headDim)
        vmul(local_fp32_buf, local_fp32_buf, cos_fp32_buf, n_head, 1, 1, 1, head_stride,
             head_stride, 0);
        vmul(local_fp32_buf + embed_dim, local_fp32_buf + embed_dim, cos_fp32_buf, n_head, 1, 1, 1,
             head_stride, head_stride, 0);
        pipe_barrier(PIPE_V);
    } else {
        // q * sin (q : numHeads * headDim  sin: 1 * headDim (headDim = 64))
        vmul(calc_fp32_buf, local_fp32_buf, sin_fp32_buf, n_head, 1, 1, 1, head_stride, head_stride,
             0);
        pipe_barrier(PIPE_V);

        // q * cos (q : numHeads * headDim  cos: 1 * headDim)
        vmul(local_fp32_buf, local_fp32_buf, cos_fp32_buf, n_head, 1, 1, 1, head_stride,
             head_stride, 0);
        pipe_barrier(PIPE_V);
    }

    // [LEGACY bf16 round-trip] 为了保证与torch_npu计算结果一致: round the fp32
    // products to bf16 and re-expand, emulating per-op bf16 rounding.
    if constexpr (!COSSIN_FP32 && std::is_same_v<Dtype, bfloat16_t>) {
        vconv_f322bf16r(local_buf, local_fp32_buf,
                        DIV_ROUND_UP(calc_size * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_f322bf16r(tmp_buf, calc_fp32_buf,
                        DIV_ROUND_UP(calc_size * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_bf162f32(local_fp32_buf, local_buf,
                       DIV_ROUND_UP(calc_size * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vconv_bf162f32(calc_fp32_buf, tmp_buf,
                       DIV_ROUND_UP(calc_size * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
    }

    if (rot_dim == HEAD_SIZE_64) {
        set_mask_norm();
        set_vector_mask((uint64_t)0, (uint64_t)0x00000000FFFFFFFF);
    }

    // q * cos - q * sin > q[:half] * cos - q[half:] * sin
    vsub(local_fp32_buf, local_fp32_buf, calc_fp32_buf + embed_dim, n_head, 1, 1, 1, head_stride,
         head_stride, head_stride);
    pipe_barrier(PIPE_V);

    // q * cos + q * sin > q[half:] * cos + q[:half] * sin
    vadd(local_fp32_buf + embed_dim, local_fp32_buf + embed_dim, calc_fp32_buf, n_head, 1, 1, 1,
         head_stride, head_stride, head_stride);
    pipe_barrier(PIPE_V);

    if (rot_dim == HEAD_SIZE_64) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }

    // [LEGACY bf16 final rounding] round the rotated result so the caller's
    // fp32 scale step re-expands it; in fp32 mode the caller owns the single
    // final rounding.
    if constexpr (!COSSIN_FP32 && std::is_same_v<Dtype, bfloat16_t>) {
        vconv_f322bf16r(local_buf, local_fp32_buf,
                        DIV_ROUND_UP(calc_size * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
    }
}

// [LEGACY fp16 direct-math rope]: fp16 arithmetic on a Dtype-staged cossin
// cache. TODO(deprecate-isFp32): remove with the COSSIN_FP32=false paths.
template <typename Dtype>
__aicore__ inline void calc_cossin(__gm__ Dtype *gm_buf_loop, __ubuf__ Dtype *local_buf,
                                   __ubuf__ Dtype *calc_buf, __ubuf__ Dtype *sin_buf,
                                   __ubuf__ Dtype *cos_buf, __ubuf__ Dtype *sin_h_buf,
                                   __ubuf__ Dtype *cos_h_buf, __ubuf__ Dtype *sin_w_buf,
                                   __ubuf__ Dtype *cos_w_buf, uint32_t embed_dim, uint32_t rot_dim,
                                   int event_id, uint32_t n_head, uint32_t head_size,
                                   uint32_t pos_dim, uint64_t mrope_mask_h, uint64_t mrope_mask_w,
                                   uint64_t copy_param)
{
    uint64_t head_stride = DIV_ROUND_UP(head_size * sizeof(float16_t), BLOCK_SIZE);
    copy_gm_to_ubuf(local_buf, gm_buf_loop, copy_param);
    set_flag(PIPE_MTE2, PIPE_V, event_id);
    wait_flag(PIPE_MTE2, PIPE_V, event_id);

    if (pos_dim > 1) {
        set_vector_mask(0x0, mrope_mask_h + mrope_mask_w);
        vector_dup(sin_buf, Dtype(0), 1, 1, 1, 8, 0);
        vector_dup(cos_buf, Dtype(0), 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);

        set_vector_mask(0x0, mrope_mask_h);
        vadd(sin_buf, sin_buf, sin_h_buf, 1, 1, 1, 1, 8, 8, 8);
        vadd(cos_buf, cos_buf, cos_h_buf, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        set_vector_mask(0x0, mrope_mask_w);
        vadd(sin_buf, sin_buf, sin_w_buf, 1, 1, 1, 1, 8, 8, 8);
        vadd(cos_buf, cos_buf, cos_w_buf, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        copy_ubuf_to_ubuf(sin_buf + embed_dim, sin_buf, 0, 1,
                          DIV_ROUND_UP(embed_dim * sizeof(Dtype), BLOCK_SIZE), 1, 1);
        copy_ubuf_to_ubuf(cos_buf + embed_dim, cos_buf, 0, 1,
                          DIV_ROUND_UP(embed_dim * sizeof(Dtype), BLOCK_SIZE), 1, 1);
        pipe_barrier(PIPE_V);
    }

    set_mask_norm();
    if (rot_dim == HEAD_SIZE_128) {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    } else {
        set_vector_mask((uint64_t)0, (uint64_t)-1);
    }
    // k * sin
    vmul(calc_buf, local_buf, sin_buf, n_head, 1, 1, 1, head_stride, head_stride, 0);
    pipe_barrier(PIPE_V);

    // k * cos
    vmul(local_buf, local_buf, cos_buf, n_head, 1, 1, 1, head_stride, head_stride, 0);
    pipe_barrier(PIPE_V);

    set_mask_norm();
    if (rot_dim == HEAD_SIZE_128) {
        set_vector_mask((uint64_t)0, (uint64_t)-1);
    } else {
        set_vector_mask((uint64_t)0, (uint64_t)0x00000000FFFFFFFF);
    }

    // k * cos - k * sin > k[:half] * cos - k[half:] * sin
    vsub(local_buf, local_buf, calc_buf + embed_dim, n_head, 1, 1, 1, head_stride, head_stride,
         head_stride);
    pipe_barrier(PIPE_V);

    // k * cos + k * sin > k[half:] * cos + k[:half] * sin
    vadd(local_buf + embed_dim, local_buf + embed_dim, calc_buf, n_head, 1, 1, 1, head_stride,
         head_stride, head_stride);
    pipe_barrier(PIPE_V);

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

// COSSIN_FP32: the cossin cache is fp32 (checked from the tensor on host);
// cos/sin are staged GM->UB directly with no cast, rotation runs in fp32 and
// takes a single final rounding back to Dtype. COSSIN_FP32=false is the
// legacy model-dtype cache path (bf16 round-trip emulation / fp16 direct
// math). TODO(deprecate-isFp32): remove the false paths.
template <typename Dtype, bool COSSIN_FP32>
__aicore__ inline void rope_and_cache(GM_ADDR positions, GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                      GM_ADDR cos_sin_cache, GM_ADDR key_cache, GM_ADDR value_cache,
                                      GM_ADDR slot_mapping, uint32_t num_tokens, uint32_t rot_dim,
                                      uint32_t query_stride, uint32_t key_stride,
                                      uint32_t value_stride, uint32_t num_heads,
                                      uint32_t num_kv_heads, uint32_t head_size,
                                      uint32_t block_size, float scale_, uint64_t mrope_mask_h,
                                      uint64_t mrope_mask_w)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    Dtype scale = (Dtype)scale_;
    float scale_fp32 = scale_;
    uint32_t embed_dim = rot_dim / 2;
    uint32_t q_size = num_heads * head_size;
    uint32_t kv_size = num_kv_heads * head_size;
    uint32_t pos_dim = (mrope_mask_h != 0 || mrope_mask_w != 0) ? 3 : 1;

    // input
    int ubuf_num = 0;
    uint32_t q_bytesize = q_size * sizeof(Dtype);
    auto query_dtype_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)0 * (ubuf_num++));
    auto query_dtype_ubuf_addr1 =
        reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)q_bytesize * (ubuf_num++));
    __ubuf__ Dtype *query_dtype_ubuf_addr[PINGPONG_BUF_NUM] = {query_dtype_ubuf_addr0,
                                                               query_dtype_ubuf_addr1};

    uint32_t kv_start = ubuf_num * q_bytesize;
    uint32_t kv_bytesize = kv_size * sizeof(Dtype);
    ubuf_num = 0;
    auto key_dtype_ubuf_addr0 =
        reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)kv_start + kv_bytesize * (ubuf_num++));
    auto key_dtype_ubuf_addr1 =
        reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)kv_start + kv_bytesize * (ubuf_num++));
    __ubuf__ Dtype *key_dtype_ubuf_addr[PINGPONG_BUF_NUM] = {key_dtype_ubuf_addr0,
                                                             key_dtype_ubuf_addr1};

    auto value_dtype_ubuf_addr0 =
        reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)(kv_start + kv_bytesize * (ubuf_num++)));
    auto value_dtype_ubuf_addr1 =
        reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)(kv_start + kv_bytesize * (ubuf_num++)));
    __ubuf__ Dtype *value_dtype_ubuf_addr[PINGPONG_BUF_NUM] = {value_dtype_ubuf_addr0,
                                                               value_dtype_ubuf_addr1};
    // cossin staged in Dtype (legacy paths; idle when the cache is fp32)
    uint32_t cossin_start = kv_start + ubuf_num * kv_bytesize;
    uint32_t cossin_blocksize = ROUND_UP(rot_dim * sizeof(Dtype), BLOCK_SIZE);
    __ubuf__ Dtype *cos_dtype_ubuf_addr[PINGPONG_BUF_NUM];
    __ubuf__ Dtype *sin_dtype_ubuf_addr[PINGPONG_BUF_NUM];
    __ubuf__ Dtype *cos_mrope_h_ubuf_addr[PINGPONG_BUF_NUM],
        *cos_mrope_w_ubuf_addr[PINGPONG_BUF_NUM];
    __ubuf__ Dtype *sin_mrope_h_ubuf_addr[PINGPONG_BUF_NUM],
        *sin_mrope_w_ubuf_addr[PINGPONG_BUF_NUM];
    ubuf_num = 0;
    auto cos_dtype_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>(
        (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
    auto cos_dtype_ubuf_addr1 = reinterpret_cast<__ubuf__ Dtype *>(
        (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
    cos_dtype_ubuf_addr[0] = cos_dtype_ubuf_addr0;
    cos_dtype_ubuf_addr[1] = cos_dtype_ubuf_addr1;
    auto sin_dtype_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>(
        (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
    auto sin_dtype_ubuf_addr1 = reinterpret_cast<__ubuf__ Dtype *>(
        (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
    sin_dtype_ubuf_addr[0] = sin_dtype_ubuf_addr0;
    sin_dtype_ubuf_addr[1] = sin_dtype_ubuf_addr1;
    if (pos_dim > 1) {
        auto cos_h_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>(
            (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
        auto cos_h_ubuf_addr1 = reinterpret_cast<__ubuf__ Dtype *>(
            (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
        auto cos_w_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>(
            (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
        auto cos_w_ubuf_addr1 = reinterpret_cast<__ubuf__ Dtype *>(
            (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
        cos_mrope_h_ubuf_addr[0] = cos_h_ubuf_addr0;
        cos_mrope_h_ubuf_addr[1] = cos_h_ubuf_addr1;
        cos_mrope_w_ubuf_addr[0] = cos_w_ubuf_addr0;
        cos_mrope_w_ubuf_addr[1] = cos_w_ubuf_addr1;
        auto sin_h_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>(
            (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
        auto sin_h_ubuf_addr1 = reinterpret_cast<__ubuf__ Dtype *>(
            (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
        auto sin_w_ubuf_addr0 = reinterpret_cast<__ubuf__ Dtype *>(
            (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
        auto sin_w_ubuf_addr1 = reinterpret_cast<__ubuf__ Dtype *>(
            (uintptr_t)(cossin_start + cossin_blocksize * (ubuf_num++)));
        sin_mrope_h_ubuf_addr[0] = sin_h_ubuf_addr0;
        sin_mrope_h_ubuf_addr[1] = sin_h_ubuf_addr1;
        sin_mrope_w_ubuf_addr[0] = sin_w_ubuf_addr0;
        sin_mrope_w_ubuf_addr[1] = sin_w_ubuf_addr1;
    }

    __ubuf__ Dtype *query_calc_dtype_ubuf_addr, *key_calc_dtype_ubuf_addr;
    __ubuf__ float *query_fp32_ubuf_addr[PINGPONG_BUF_NUM], *key_fp32_ubuf_addr[PINGPONG_BUF_NUM];
    __ubuf__ float *cos_fp32_ubuf_addr[PINGPONG_BUF_NUM], *sin_fp32_ubuf_addr[PINGPONG_BUF_NUM];
    __ubuf__ float *cos_fp32_mrope_h_ubuf_addr[PINGPONG_BUF_NUM],
        *cos_fp32_mrope_w_ubuf_addr[PINGPONG_BUF_NUM];
    __ubuf__ float *sin_fp32_mrope_h_ubuf_addr[PINGPONG_BUF_NUM],
        *sin_fp32_mrope_w_ubuf_addr[PINGPONG_BUF_NUM];
    __ubuf__ float *query_calc_fp32_ubuf, *key_calc_fp32_ubuf;

    uint32_t calcbuf_start = ROUND_UP(cossin_start + ubuf_num * cossin_blocksize, BLOCK_SIZE);
    if constexpr (COSSIN_FP32 || std::is_same_v<Dtype, bfloat16_t>) {
        // fp32 rotation pipeline: legacy bf16 round-trip emulation, or any
        // Dtype with a fp32 cossin cache. The fp32 cos/sin rows are the direct
        // GM staging targets in fp32 mode, else the vconv targets of the
        // staged Dtype rows.
        uint32_t q_fp32_blocksize = ROUND_UP(q_size * sizeof(float), BLOCK_SIZE);
        uint32_t kv_fp32_blocksize = ROUND_UP(kv_size * sizeof(float), BLOCK_SIZE);
        uint32_t cossin_fp32_blocksize = ROUND_UP(rot_dim * sizeof(float), BLOCK_SIZE);

        query_calc_dtype_ubuf_addr = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)calcbuf_start);
        calcbuf_start += MAX(q_bytesize, kv_bytesize);

        ubuf_num = 0;
        auto query_fp32_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>(
            (uintptr_t)(calcbuf_start + q_fp32_blocksize * (ubuf_num++)));
        auto query_fp32_ubuf_addr1 = reinterpret_cast<__ubuf__ float *>(
            (uintptr_t)(calcbuf_start + q_fp32_blocksize * (ubuf_num++)));
        query_fp32_ubuf_addr[0] = query_fp32_ubuf_addr0;
        query_fp32_ubuf_addr[1] = query_fp32_ubuf_addr1;
        calcbuf_start += ubuf_num * q_fp32_blocksize;

        ubuf_num = 0;
        auto key_fp32_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>(
            (uintptr_t)(calcbuf_start + kv_fp32_blocksize * (ubuf_num++)));
        auto key_fp32_ubuf_addr1 = reinterpret_cast<__ubuf__ float *>(
            (uintptr_t)(calcbuf_start + kv_fp32_blocksize * (ubuf_num++)));
        key_fp32_ubuf_addr[0] = key_fp32_ubuf_addr0;
        key_fp32_ubuf_addr[1] = key_fp32_ubuf_addr1;
        calcbuf_start += ubuf_num * kv_fp32_blocksize;

        // T-row cos/sin; ubuf_num continues into the mrope rows below (do NOT reset it there — that
        // would alias mrope onto cos/sin and the merge would zero the h/w lanes)
        ubuf_num = 0;
        auto cos_fp32_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>(
            (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
        auto cos_fp32_ubuf_addr1 = reinterpret_cast<__ubuf__ float *>(
            (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
        cos_fp32_ubuf_addr[0] = cos_fp32_ubuf_addr0;
        cos_fp32_ubuf_addr[1] = cos_fp32_ubuf_addr1;
        auto sin_fp32_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>(
            (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
        auto sin_fp32_ubuf_addr1 = reinterpret_cast<__ubuf__ float *>(
            (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
        sin_fp32_ubuf_addr[0] = sin_fp32_ubuf_addr0;
        sin_fp32_ubuf_addr[1] = sin_fp32_ubuf_addr1;
        if (pos_dim > 1) {
            auto cos_fp32_mrope_h_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>(
                (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
            auto cos_fp32_mrope_h_ubuf_addr1 = reinterpret_cast<__ubuf__ float *>(
                (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
            auto cos_fp32_mrope_w_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>(
                (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
            auto cos_fp32_mrope_w_ubuf_addr1 = reinterpret_cast<__ubuf__ float *>(
                (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
            cos_fp32_mrope_h_ubuf_addr[0] = cos_fp32_mrope_h_ubuf_addr0;
            cos_fp32_mrope_h_ubuf_addr[1] = cos_fp32_mrope_h_ubuf_addr1;
            cos_fp32_mrope_w_ubuf_addr[0] = cos_fp32_mrope_w_ubuf_addr0;
            cos_fp32_mrope_w_ubuf_addr[1] = cos_fp32_mrope_w_ubuf_addr1;

            auto sin_fp32_mrope_h_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>(
                (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
            auto sin_fp32_mrope_h_ubuf_addr1 = reinterpret_cast<__ubuf__ float *>(
                (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
            auto sin_fp32_mrope_w_ubuf_addr0 = reinterpret_cast<__ubuf__ float *>(
                (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
            auto sin_fp32_mrope_w_ubuf_addr1 = reinterpret_cast<__ubuf__ float *>(
                (uintptr_t)(calcbuf_start + cossin_fp32_blocksize * (ubuf_num++)));
            sin_fp32_mrope_h_ubuf_addr[0] = sin_fp32_mrope_h_ubuf_addr0;
            sin_fp32_mrope_h_ubuf_addr[1] = sin_fp32_mrope_h_ubuf_addr1;
            sin_fp32_mrope_w_ubuf_addr[0] = sin_fp32_mrope_w_ubuf_addr0;
            sin_fp32_mrope_w_ubuf_addr[1] = sin_fp32_mrope_w_ubuf_addr1;
        }
        calcbuf_start += ubuf_num * cossin_fp32_blocksize;

        query_calc_fp32_ubuf = reinterpret_cast<__ubuf__ float *>((uintptr_t)(calcbuf_start));
        key_calc_fp32_ubuf =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)(calcbuf_start + q_fp32_blocksize));
        calcbuf_start += (q_fp32_blocksize + kv_fp32_blocksize);

        // round-trip tmp; only written by the legacy bf16 path
        key_calc_dtype_ubuf_addr = query_calc_dtype_ubuf_addr;
    } else {
        // [LEGACY fp16 direct math] TODO(deprecate-isFp32)
        query_calc_dtype_ubuf_addr = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)calcbuf_start);
        key_calc_dtype_ubuf_addr =
            reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)(calcbuf_start + q_bytesize));
        calcbuf_start += 2 * q_bytesize;
    }

    // pos and slot
    uint32_t params_start = calcbuf_start;
    assert(params_start <= UB_SIZE);  // UB_SIZE - params_start below must not underflow
    // Round down to a multiple of 8 tokens (never round up; may cause ubuf overflow)
    uint32_t iter_posslot_num = ROUND_DOWN(UB_SIZE - params_start, pos_dim * BLOCK_SIZE) /
                                (sizeof(uint64_t) * pos_dim + sizeof(uint32_t));
    iter_posslot_num = MIN(ROUND_DOWN(iter_posslot_num, BLOCK_SIZE / sizeof(uint32_t)), num_tokens);
    assert(iter_posslot_num > 0);
    uint32_t posslot_iters = DIV_ROUND_UP(num_tokens, iter_posslot_num);

    uint32_t pos_size = ROUND_UP(iter_posslot_num * sizeof(uint64_t), BLOCK_SIZE);
    uint32_t slot_size = ROUND_UP(iter_posslot_num * sizeof(uint32_t), BLOCK_SIZE);
    auto pos_int_ubuf_addr0 = reinterpret_cast<__ubuf__ int64_t *>((uintptr_t)(params_start));
    __ubuf__ int64_t *pos_h_ubuf_addr0, *pos_w_ubuf_addr0;
    if (pos_dim > 1) {
        pos_h_ubuf_addr0 =
            reinterpret_cast<__ubuf__ int64_t *>((uintptr_t)(params_start + pos_size));
        pos_w_ubuf_addr0 =
            reinterpret_cast<__ubuf__ int64_t *>((uintptr_t)(params_start + 2 * pos_size));
    }
    auto slot_int_ubuf_addr0 =
        reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)(params_start + pos_dim * pos_size));

    uint64_t q_repeat;
    if constexpr (COSSIN_FP32 || std::is_same_v<Dtype, bfloat16_t>) {
        q_repeat = DIV_ROUND_UP(q_size, VECTOR_MAX_NUM_OF_FP32);
    } else {
        q_repeat = DIV_ROUND_UP(q_size, VECTOR_MAX_NUM_OF_FP16);
    }

    uint64_t lenBurst_q = DIV_ROUND_UP(q_bytesize, BLOCK_SIZE);
    uint64_t lenBurst_kv = DIV_ROUND_UP(kv_bytesize, BLOCK_SIZE);
    uint64_t lenBurst_cossin =
        DIV_ROUND_UP(embed_dim * (COSSIN_FP32 ? sizeof(float) : sizeof(Dtype)), BLOCK_SIZE);  // d/2
    uint64_t lenBurst_pos = DIV_ROUND_UP(pos_size, BLOCK_SIZE);
    uint64_t lenBurst_slot = DIV_ROUND_UP(slot_size, BLOCK_SIZE);
    constexpr uint8_t sid = 0;
    constexpr uint16_t n_burst = 1;
    constexpr uint16_t src_gap = 0;
    constexpr uint16_t dst_gap = 0;

    uint64_t dmi_cfg_q = __set_dmi_config(sid, n_burst, lenBurst_q, src_gap, dst_gap);
    uint64_t dmi_cfg_kv = __set_dmi_config(sid, n_burst, lenBurst_kv, src_gap, dst_gap);
    uint64_t dmi_cfg_cossin = __set_dmi_config(sid, n_burst, lenBurst_cossin, src_gap, dst_gap);
    uint64_t dmi_cfg_pos = __set_dmi_config(sid, n_burst, lenBurst_pos, src_gap, dst_gap);
    uint64_t dmi_cfg_slot = __set_dmi_config(sid, n_burst, lenBurst_slot, src_gap, dst_gap);

    uint32_t cos_shift;
    uint32_t sin_shift;
    uint64_t slot_startidx;
    uint32_t poslot_idx;
    for (uint32_t loop0 = 0; loop0 < posslot_iters; loop0 += 1) {
        uint32_t processed_num_tokens = loop0 * iter_posslot_num;
        auto gm_pos = (__gm__ int64_t *)positions + processed_num_tokens;
        auto gm_slot = (__gm__ int32_t *)slot_mapping + processed_num_tokens;
        if (loop0 == posslot_iters - 1) {
            iter_posslot_num = num_tokens - processed_num_tokens;
            lenBurst_pos = DIV_ROUND_UP(iter_posslot_num * sizeof(uint64_t), BLOCK_SIZE);
            lenBurst_slot = DIV_ROUND_UP(iter_posslot_num * sizeof(uint32_t), BLOCK_SIZE);
            dmi_cfg_pos = __set_dmi_config(sid, n_burst, lenBurst_pos, src_gap, dst_gap);
            dmi_cfg_slot = __set_dmi_config(sid, n_burst, lenBurst_slot, src_gap, dst_gap);
        }
        // copy positions and slotmapping
        copy_gm_to_ubuf(pos_int_ubuf_addr0, gm_pos, dmi_cfg_pos);
        if (pos_dim > 1) {
            copy_gm_to_ubuf(pos_h_ubuf_addr0, gm_pos + num_tokens, dmi_cfg_pos);
            copy_gm_to_ubuf(pos_w_ubuf_addr0, gm_pos + 2 * num_tokens, dmi_cfg_pos);
        }
        copy_gm_to_ubuf(slot_int_ubuf_addr0, gm_slot, dmi_cfg_slot);
        pipe_barrier(PIPE_ALL);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        for (uint32_t loop1 = (block_idx + processed_num_tokens), ping = 1;
             loop1 < (processed_num_tokens + iter_posslot_num); loop1 += block_num) {
            auto event_id = ping == 1 ? EVENT_ID0 : EVENT_ID1;
            poslot_idx = loop1 - processed_num_tokens;

            // qkv
            auto gm_query = (__gm__ Dtype *)query + loop1 * query_stride;
            auto gm_key = (__gm__ Dtype *)key + loop1 * key_stride;
            auto gm_value = (__gm__ Dtype *)value + loop1 * value_stride;
            // cossin (row = [cos(d/2) | sin(d/2)]; element offsets are dtype-independent)
            cos_shift = *(pos_int_ubuf_addr0 + poslot_idx) * rot_dim;
            sin_shift = cos_shift + embed_dim;
            // slot
            slot_startidx = *(slot_int_ubuf_addr0 + poslot_idx) * kv_size;
            auto gm_kcache = (__gm__ Dtype *)key_cache + slot_startidx;
            auto gm_vcache = (__gm__ Dtype *)value_cache + slot_startidx;

            set_flag(PIPE_S, PIPE_MTE2, event_id);
            wait_flag(PIPE_S, PIPE_MTE2, event_id);

            wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);
            // cache value
            copy_gm_to_ubuf(value_dtype_ubuf_addr[event_id], gm_value, dmi_cfg_kv);
            set_flag(PIPE_MTE2, PIPE_MTE3, event_id);
            wait_flag(PIPE_MTE2, PIPE_MTE3, event_id);
            copy_ubuf_to_gm(gm_vcache, value_dtype_ubuf_addr[event_id], dmi_cfg_kv);

            if constexpr (COSSIN_FP32) {
                // copy fp32 cos and sin, repeat for d/2
                auto gm_cos = (__gm__ float *)cos_sin_cache + cos_shift;
                auto gm_sin = (__gm__ float *)cos_sin_cache + sin_shift;
                copy_gm_to_ubuf(cos_fp32_ubuf_addr[event_id], gm_cos, dmi_cfg_cossin);
                copy_gm_to_ubuf(cos_fp32_ubuf_addr[event_id] + embed_dim, gm_cos, dmi_cfg_cossin);
                copy_gm_to_ubuf(sin_fp32_ubuf_addr[event_id], gm_sin, dmi_cfg_cossin);
                copy_gm_to_ubuf(sin_fp32_ubuf_addr[event_id] + embed_dim, gm_sin, dmi_cfg_cossin);
                set_flag(PIPE_MTE2, PIPE_V, event_id);
                wait_flag(PIPE_MTE2, PIPE_V, event_id);
                if (pos_dim > 1) {
                    uint32_t cos_h_shift = *(pos_h_ubuf_addr0 + poslot_idx) * rot_dim;
                    uint32_t sin_h_shift = cos_h_shift + embed_dim;
                    uint32_t cos_w_shift = *(pos_w_ubuf_addr0 + poslot_idx) * rot_dim;
                    uint32_t sin_w_shift = cos_w_shift + embed_dim;
                    copy_gm_to_ubuf(cos_fp32_mrope_h_ubuf_addr[event_id],
                                    (__gm__ float *)cos_sin_cache + cos_h_shift, dmi_cfg_cossin);
                    copy_gm_to_ubuf(sin_fp32_mrope_h_ubuf_addr[event_id],
                                    (__gm__ float *)cos_sin_cache + sin_h_shift, dmi_cfg_cossin);
                    copy_gm_to_ubuf(cos_fp32_mrope_w_ubuf_addr[event_id],
                                    (__gm__ float *)cos_sin_cache + cos_w_shift, dmi_cfg_cossin);
                    copy_gm_to_ubuf(sin_fp32_mrope_w_ubuf_addr[event_id],
                                    (__gm__ float *)cos_sin_cache + sin_w_shift, dmi_cfg_cossin);
                    set_flag(PIPE_MTE2, PIPE_V, event_id);
                    wait_flag(PIPE_MTE2, PIPE_V, event_id);
                }
            } else {
                auto gm_cos = (__gm__ Dtype *)cos_sin_cache + cos_shift;
                auto gm_sin = (__gm__ Dtype *)cos_sin_cache + sin_shift;
                // copy cos and sin, repeat for d/2
                copy_gm_to_ubuf(cos_dtype_ubuf_addr[event_id], gm_cos, dmi_cfg_cossin);
                copy_gm_to_ubuf(cos_dtype_ubuf_addr[event_id] + embed_dim, gm_cos, dmi_cfg_cossin);
                copy_gm_to_ubuf(sin_dtype_ubuf_addr[event_id], gm_sin, dmi_cfg_cossin);
                copy_gm_to_ubuf(sin_dtype_ubuf_addr[event_id] + embed_dim, gm_sin, dmi_cfg_cossin);
                set_flag(PIPE_MTE2, PIPE_V, event_id);
                wait_flag(PIPE_MTE2, PIPE_V, event_id);
                if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                    // staged T-row cossin -> fp32 for the cast pipeline
                    vconv_bf162f32(sin_fp32_ubuf_addr[event_id], sin_dtype_ubuf_addr[event_id],
                                   DIV_ROUND_UP(rot_dim * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1,
                                   8, 4);
                    vconv_bf162f32(cos_fp32_ubuf_addr[event_id], cos_dtype_ubuf_addr[event_id],
                                   DIV_ROUND_UP(rot_dim * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1,
                                   8, 4);
                    pipe_barrier(PIPE_V);
                }
                if (pos_dim > 1) {
                    uint32_t cos_h_shift = *(pos_h_ubuf_addr0 + poslot_idx) * rot_dim;
                    uint32_t sin_h_shift = cos_h_shift + embed_dim;
                    uint32_t cos_w_shift = *(pos_w_ubuf_addr0 + poslot_idx) * rot_dim;
                    uint32_t sin_w_shift = cos_w_shift + embed_dim;
                    copy_gm_to_ubuf(cos_mrope_h_ubuf_addr[event_id],
                                    (__gm__ Dtype *)cos_sin_cache + cos_h_shift, dmi_cfg_cossin);
                    copy_gm_to_ubuf(sin_mrope_h_ubuf_addr[event_id],
                                    (__gm__ Dtype *)cos_sin_cache + sin_h_shift, dmi_cfg_cossin);
                    copy_gm_to_ubuf(cos_mrope_w_ubuf_addr[event_id],
                                    (__gm__ Dtype *)cos_sin_cache + cos_w_shift, dmi_cfg_cossin);
                    copy_gm_to_ubuf(sin_mrope_w_ubuf_addr[event_id],
                                    (__gm__ Dtype *)cos_sin_cache + sin_w_shift, dmi_cfg_cossin);
                    set_flag(PIPE_MTE2, PIPE_V, event_id);
                    wait_flag(PIPE_MTE2, PIPE_V, event_id);
                    if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                        vconv_bf162f32(cos_fp32_mrope_h_ubuf_addr[event_id],
                                       cos_mrope_h_ubuf_addr[event_id],
                                       DIV_ROUND_UP(embed_dim * sizeof(float), VECTOR_MAX_BYTESIZE),
                                       1, 1, 8, 4);
                        vconv_bf162f32(sin_fp32_mrope_h_ubuf_addr[event_id],
                                       sin_mrope_h_ubuf_addr[event_id],
                                       DIV_ROUND_UP(embed_dim * sizeof(float), VECTOR_MAX_BYTESIZE),
                                       1, 1, 8, 4);
                        vconv_bf162f32(cos_fp32_mrope_w_ubuf_addr[event_id],
                                       cos_mrope_w_ubuf_addr[event_id],
                                       DIV_ROUND_UP(embed_dim * sizeof(float), VECTOR_MAX_BYTESIZE),
                                       1, 1, 8, 4);
                        vconv_bf162f32(sin_fp32_mrope_w_ubuf_addr[event_id],
                                       sin_mrope_w_ubuf_addr[event_id],
                                       DIV_ROUND_UP(embed_dim * sizeof(float), VECTOR_MAX_BYTESIZE),
                                       1, 1, 8, 4);
                    }
                    pipe_barrier(PIPE_V);
                }
            }

            if constexpr (!COSSIN_FP32 && !std::is_same_v<Dtype, bfloat16_t>) {
                // [LEGACY fp16 direct math] TODO(deprecate-isFp32)
                calc_cossin(gm_key, key_dtype_ubuf_addr[event_id], key_calc_dtype_ubuf_addr,
                            sin_dtype_ubuf_addr[event_id], cos_dtype_ubuf_addr[event_id],
                            sin_mrope_h_ubuf_addr[event_id], cos_mrope_h_ubuf_addr[event_id],
                            sin_mrope_w_ubuf_addr[event_id], cos_mrope_w_ubuf_addr[event_id],
                            embed_dim, rot_dim, event_id, num_kv_heads, head_size, pos_dim,
                            mrope_mask_h, mrope_mask_w, dmi_cfg_kv);
            } else {
                calc_cossin_cast<Dtype, COSSIN_FP32>(
                    gm_key, key_dtype_ubuf_addr[event_id], query_calc_dtype_ubuf_addr,
                    key_fp32_ubuf_addr[event_id], key_calc_fp32_ubuf, sin_fp32_ubuf_addr[event_id],
                    cos_fp32_ubuf_addr[event_id], sin_fp32_mrope_h_ubuf_addr[event_id],
                    cos_fp32_mrope_h_ubuf_addr[event_id], sin_fp32_mrope_w_ubuf_addr[event_id],
                    cos_fp32_mrope_w_ubuf_addr[event_id], embed_dim, rot_dim, event_id,
                    num_kv_heads, head_size, pos_dim, mrope_mask_h, mrope_mask_w, dmi_cfg_kv);
                if constexpr (COSSIN_FP32) {
                    // single final rounding of the rotated key
                    convert_output<Dtype>(
                        key_dtype_ubuf_addr[event_id], key_fp32_ubuf_addr[event_id],
                        DIV_ROUND_UP(kv_size * sizeof(float), VECTOR_MAX_BYTESIZE));
                    pipe_barrier(PIPE_V);
                }
            }
            set_flag(PIPE_V, PIPE_MTE3, event_id);
            wait_flag(PIPE_V, PIPE_MTE3, event_id);

            // Update key and cache key
            copy_ubuf_to_gm(gm_key, key_dtype_ubuf_addr[event_id], dmi_cfg_kv);
            copy_ubuf_to_gm(gm_kcache, key_dtype_ubuf_addr[event_id], dmi_cfg_kv);

            if constexpr (!COSSIN_FP32 && !std::is_same_v<Dtype, bfloat16_t>) {
                // [LEGACY fp16 direct math] TODO(deprecate-isFp32)
                calc_cossin(gm_query, query_dtype_ubuf_addr[event_id], query_calc_dtype_ubuf_addr,
                            sin_dtype_ubuf_addr[event_id], cos_dtype_ubuf_addr[event_id],
                            sin_mrope_h_ubuf_addr[event_id], cos_mrope_h_ubuf_addr[event_id],
                            sin_mrope_w_ubuf_addr[event_id], cos_mrope_w_ubuf_addr[event_id],
                            embed_dim, rot_dim, event_id, num_heads, head_size, pos_dim,
                            mrope_mask_h, mrope_mask_w, dmi_cfg_q);
                vmuls(query_dtype_ubuf_addr[event_id], query_dtype_ubuf_addr[event_id], scale,
                      q_repeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            } else {
                calc_cossin_cast<Dtype, COSSIN_FP32>(
                    gm_query, query_dtype_ubuf_addr[event_id], query_calc_dtype_ubuf_addr,
                    query_fp32_ubuf_addr[event_id], query_calc_fp32_ubuf,
                    sin_fp32_ubuf_addr[event_id], cos_fp32_ubuf_addr[event_id],
                    sin_fp32_mrope_h_ubuf_addr[event_id], cos_fp32_mrope_h_ubuf_addr[event_id],
                    sin_fp32_mrope_w_ubuf_addr[event_id], cos_fp32_mrope_w_ubuf_addr[event_id],
                    embed_dim, rot_dim, event_id, num_heads, head_size, pos_dim, mrope_mask_h,
                    mrope_mask_w, dmi_cfg_q);
                if constexpr (COSSIN_FP32) {
                    // fp32 scale, then the single final rounding back to Dtype
                    vmuls(query_fp32_ubuf_addr[event_id], query_fp32_ubuf_addr[event_id],
                          scale_fp32, q_repeat, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    convert_output<Dtype>(
                        query_dtype_ubuf_addr[event_id], query_fp32_ubuf_addr[event_id],
                        DIV_ROUND_UP(q_size * sizeof(float), VECTOR_MAX_BYTESIZE));
                    pipe_barrier(PIPE_V);
                } else {
                    // [LEGACY bf16] re-expand the rounded rotation, scale in
                    // fp32, round again (torch_npu per-op parity)
                    vconv_bf162f32(query_fp32_ubuf_addr[event_id], query_dtype_ubuf_addr[event_id],
                                   DIV_ROUND_UP(q_size * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1,
                                   8, 4);
                    pipe_barrier(PIPE_V);
                    vmuls(query_fp32_ubuf_addr[event_id], query_fp32_ubuf_addr[event_id],
                          scale_fp32, q_repeat, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    vconv_f322bf16r(query_dtype_ubuf_addr[event_id], query_fp32_ubuf_addr[event_id],
                                    DIV_ROUND_UP(q_size * sizeof(float), VECTOR_MAX_BYTESIZE), 1, 1,
                                    4, 8);
                    pipe_barrier(PIPE_V);
                }
            }
            set_flag(PIPE_V, PIPE_MTE3, event_id);
            wait_flag(PIPE_V, PIPE_MTE3, event_id);
            copy_ubuf_to_gm(gm_query, query_dtype_ubuf_addr[event_id], dmi_cfg_q);

            set_flag(PIPE_MTE3, PIPE_MTE2, event_id);
            ping = 1 - ping;
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    pipe_barrier(PIPE_ALL);
}

#define ROPEANDCACHE_FUNC_DEFINE(dtype)                                                           \
    extern "C" __global__ __aicore__ void rope_and_cache_##dtype(                                 \
        GM_ADDR positions, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR cossinCache,        \
        GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR slotMapping, uint32_t numTokens,            \
        uint32_t rotDim, uint32_t queryStride, uint32_t keyStride, uint32_t valueStride,          \
        uint32_t numHeads, uint32_t numKVHeads, uint32_t headDim, uint32_t blockSize,             \
        float scaleIn, uint64_t mropeMaskH, uint64_t mropeMaskW, bool cossinInFp32)               \
    {                                                                                             \
        if (cossinInFp32) {                                                                       \
            rope_and_cache<dtype, true>(positions, query, key, value, cossinCache, keyCache,      \
                                        valueCache, slotMapping, numTokens, rotDim, queryStride,  \
                                        keyStride, valueStride, numHeads, numKVHeads, headDim,    \
                                        blockSize, scaleIn, mropeMaskH, mropeMaskW);              \
        } else {                                                                                  \
            rope_and_cache<dtype, false>(positions, query, key, value, cossinCache, keyCache,     \
                                         valueCache, slotMapping, numTokens, rotDim, queryStride, \
                                         keyStride, valueStride, numHeads, numKVHeads, headDim,   \
                                         blockSize, scaleIn, mropeMaskH, mropeMaskW);             \
        }                                                                                         \
    }
#else
#define ROPEANDCACHE_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void rope_and_cache_##dtype(                          \
        GM_ADDR positions, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR cossinCache, \
        GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR slotMapping, uint32_t numTokens,     \
        uint32_t rotDim, uint32_t queryStride, uint32_t keyStride, uint32_t valueStride,   \
        uint32_t numHeads, uint32_t numKVHeads, uint32_t headDim, uint32_t blockSize,      \
        float scaleIn, uint64_t mropeMaskH, uint64_t mropeMaskW, bool cossinInFp32)        \
    {                                                                                      \
    }
#endif