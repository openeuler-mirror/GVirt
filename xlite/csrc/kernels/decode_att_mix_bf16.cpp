/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

inline __aicore__ void data_cache_clean_and_invalid(__gm__ void * __restrict__ gm)
{
    __asm__ __volatile__("");
    dcci(gm, 0 /*SINGLE_CACHE_LINE*/);
    __asm__ __volatile__("");
}

struct decode_att_mix_context {
    __gm__ uint32_t *v2a_gm;
    __gm__ uint32_t *a2v_gm;
    __gm__ int32_t *cum_prompt_len;
    __gm__ int32_t *mapping;
    __gm__ int32_t *decode_index;
    __gm__ int32_t *cached_lens;
    __gm__ bfloat16_t *q;
    __gm__ bfloat16_t *k;
    __gm__ bfloat16_t *v;
    __gm__ bfloat16_t *qk;
    __gm__ bfloat16_t *o;

#ifdef __DAV_C220_CUBE__
    __cbuf__ bfloat16_t *cbuf_addr_a;
    __cbuf__ bfloat16_t *cbuf_addr_b[2];
    __ca__ bfloat16_t *ca_addr;
    __cb__ bfloat16_t *cb_addr[2];
    __cc__ float *cc_addr;
#endif
    uint32_t num_tokens;
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t block_size;
    uint32_t mapping_len;
    uint32_t num_kv_heads;
    uint32_t max_context_len;
    uint32_t num_qkv_heads;
    uint32_t m_offset;
    uint32_t m_slice;
    uint32_t head_num_in_group;
    uint32_t block_mem_size;
    uint32_t kv_copy_step;
};

inline __aicore__ void init_decode_att_mix_context(decode_att_mix_context *ctx,
                                                   __gm__ uint8_t *a2v, __gm__ uint8_t *v2a,
                                                   __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v,
                                                   __gm__ uint8_t *cached_lens, __gm__ uint8_t *mapping,
                                                   __gm__ uint8_t *qk, __gm__ uint8_t *o,
                                                   __gm__ uint8_t *decode_index, __gm__ uint8_t *cum_prompt_len,
                                                   uint32_t num_tokens, uint32_t num_heads, uint32_t head_size,
                                                   uint32_t block_size, uint32_t mapping_len, uint32_t num_kv_heads,
                                                   uint32_t max_context_len, uint32_t num_qkv_heads, uint32_t m_offset,
                                                   uint32_t m_slice)
{
    ctx->a2v_gm = (__gm__ uint32_t*)a2v;
    ctx->v2a_gm = (__gm__ uint32_t*)v2a;
    ctx->cum_prompt_len = (__gm__ int32_t*)cum_prompt_len;
    ctx->cached_lens = (__gm__ int32_t*)cached_lens;
    ctx->mapping = (__gm__ int32_t*)mapping;
    ctx->decode_index = (__gm__ int32_t*)decode_index;

    ctx->q = (__gm__ bfloat16_t*)q;
    ctx->k = (__gm__ bfloat16_t*)k;
    ctx->v = (__gm__ bfloat16_t*)v;
    ctx->qk = (__gm__ bfloat16_t*)qk;
    ctx->o = (__gm__ bfloat16_t*)o;

#ifdef __DAV_C220_CUBE__
    ctx->cbuf_addr_a = reinterpret_cast<__cbuf__ bfloat16_t*>((uintptr_t)0);
    ctx->ca_addr = reinterpret_cast<__ca__ bfloat16_t*>((uintptr_t)0);
    ctx->cb_addr[0] = reinterpret_cast<__cb__ bfloat16_t*>((uintptr_t)0);
    ctx->cc_addr = reinterpret_cast<__cc__ float*>((uintptr_t)0);
#endif

    ctx->num_tokens = num_tokens;
    ctx->num_heads = num_heads;
    ctx->head_size = head_size;
    ctx->block_size = block_size;
    ctx->mapping_len = mapping_len;
    ctx->num_kv_heads = num_kv_heads;
    ctx->max_context_len = max_context_len;
    ctx->num_qkv_heads = num_qkv_heads;
    ctx->m_offset = m_offset;
    ctx->m_slice = m_slice;
    ctx->head_num_in_group = num_heads / num_kv_heads;
    ctx->block_mem_size = num_kv_heads * block_size * head_size;
    ctx->kv_copy_step = num_kv_heads * head_size;
}

inline __aicore__ void wait_aic_aiv_flag(__gm__ uint32_t *flag_gm, uint32_t head_num_in_group,
                                         uint32_t q_offset, uint32_t head_size)
{
    for (uint32_t sub_idx = 0; sub_idx < head_num_in_group; sub_idx++) {
        uint32_t head_idx = q_offset + sub_idx;
        __gm__ uint32_t *curr_addr = flag_gm + head_idx * head_size;
        data_cache_clean_and_invalid(curr_addr);
        uint32_t flag = *curr_addr;
        while (flag != head_idx + 1) {
            data_cache_clean_and_invalid(curr_addr);
            flag = *curr_addr;
        }
    }
}

inline __aicore__ void set_aic_aiv_flag(__gm__ uint32_t *flag_gm,
                                        uint32_t head_num_in_group, uint32_t q_offset, uint32_t head_size)
{
    for (int sub_idx = 0; sub_idx < head_num_in_group; sub_idx++) {
        uint32_t head_idx = q_offset + sub_idx;
        __gm__ uint32_t *curr_addr = flag_gm + head_idx * head_size;
        *curr_addr = head_idx + 1;
        data_cache_clean_and_invalid(curr_addr);
    }
}

inline __aicore__ void reset_aic_aiv_flag(__gm__ uint32_t *flag_gm,
                                          uint32_t head_num_in_group, uint32_t q_offset, uint32_t head_size)
{
    for (int sub_idx = 0; sub_idx < head_num_in_group; sub_idx++) {
        uint32_t head_idx = q_offset + sub_idx;
        __gm__ uint32_t *curr_addr = flag_gm + head_idx * head_size;
        *curr_addr = 0;
        data_cache_clean_and_invalid(curr_addr);
    }
}

// 256*128 64K * 128*256 64K = 256*256, 128k
#ifdef __DAV_C220_CUBE__

#define CUBE_SIZE 256 // 16 * 16 = 256
#define CUBE_BLOCK_SIZE 16 // 行列16
#define M_TILE_SIZE 16
const uint32_t MAX_CONTEXT_BLOCK_LEN = 8192;

inline __aicore__ void copy_to_l0a(__ca__ bfloat16_t *l0, __cbuf__ bfloat16_t *cbuf, uint32_t head_size, uint32_t offset = 0)
{
    load_cbuf_to_ca(l0,
                    cbuf + offset,
                    0 /* loadDataParams.startIndex */,
                    head_size / CUBE_BLOCK_SIZE /* loadDataParams.repeatTimes */,
                    1 /* loadDataParams.srcStride */,
                    0 /* loadDataParams.dstGap */,
                    0 /* loadDataParams.sid */,
                    0 /* transpose */,
                    inc);
}

inline __aicore__ void copy_to_l0b(__cb__ bfloat16_t *l0, __cbuf__ bfloat16_t *cbuf, uint32_t n_factor, uint32_t k_factor)
{
    for (int k = 0; k < k_factor; ++k) {
        int dst_offset = CUBE_SIZE * n_factor;
        int src_offset = CUBE_SIZE * n_factor;
        load_cbuf_to_cb(l0 + k * dst_offset,
                        cbuf + k * src_offset,
                        0 /* loadDataParams.startIndex */,
                        n_factor /* loadDataParams.repeatTimes */,
                        1 /* loadDataParams.srcStride */,
                        0 /* loadDataParams.dstGap */,
                        0 /* loadDataParams.sid */,
                        0 /* transpose */,
                        inc);
    }
}

inline __aicore__ void copy_to_l0b_gqa(__cb__ bfloat16_t *l0, __cbuf__ bfloat16_t *cbuf, uint32_t k_factor, uint32_t n_factor)
{
    if (n_factor == 1) {
        // LoadData2dParams loadDataParams;
        load_cbuf_to_cb(l0,
                        cbuf,
                        0 /* loadDataParams.startIndex */,
                        k_factor /* loadDataParams.repeatTimes */,
                        1 /* loadDataParams.srcStride */,
                        0 /* loadDataParams.dstGap */,
                        0 /* loadDataParams.sid */,
                        1 /* transpose */,
                        inc);
        return;
    }
    for (int k = 0; k < k_factor; ++k) {
        uint32_t dst_offset = n_factor * CUBE_SIZE;
        constexpr int src_offset = CUBE_SIZE;
        load_cbuf_to_cb(l0 + k * dst_offset,
                        cbuf + k * src_offset,
                        0 /* loadDataParams.startIndex */,
                        n_factor /* loadDataParams.repeatTimes */,
                        k_factor /* loadDataParams.srcStride */,
                        0 /* loadDataParams.dstGap */,
                        0 /* loadDataParams.sid */,
                        1 /* transpose */,
                        inc);
    }
}

inline __aicore__ void mmad(__cc__ float *cc, __ca__ bfloat16_t *ca, __cb__ bfloat16_t *cb,
                            uint32_t m, uint32_t n, uint32_t k, bool init_val, uint8_t unit_flag = 0)
{
    mad(cc, ca, cb, m, k, n, unit_flag, false /* kDirection Align */, 0, init_val);
}

inline __aicore__ void copy_to_l1_2d(__cbuf__ bfloat16_t *l1, __gm__ bfloat16_t *gm,
                                     uint32_t row, uint32_t col, uint32_t height, uint32_t width, uint32_t step)
{
    uint32_t offset = row * step + col;
    copy_gm_to_cbuf_multi_nd2nz_b16(l1,
                                    gm + offset,
                                    0,
                                    1 /* nd2nzParams.ndNum */,
                                    height /* nd2nzParams.nValue */,
                                    width /* nd2nzParams.dValue */,
                                    0 /* nd2nzParams.srcNdMatrixStride */,
                                    step /* nd2nzParams.srcDValue */,
                                    height /* nd2nzParams.dstNzC0Stride */,
                                    1 /* nd2nzParams.dstNzNStride */,
                                    0 /* nd2nzParams.dstNzMatrixStride */);
}

inline __aicore__ void copy_to_l1_gqa(__cbuf__ bfloat16_t *l1, __gm__ bfloat16_t *gm,
                                     uint32_t head_num_in_group, uint32_t sub_block_length, uint32_t blockLength)
{
#pragma unroll
    for (int i = 0; i < head_num_in_group; i++) {
        uint32_t offset = i * blockLength;
        copy_gm_to_cbuf(l1 + i * CUBE_BLOCK_SIZE, gm + offset,
                        0, sub_block_length / CUBE_BLOCK_SIZE, 1, 0, 16 - 1, (pad_t)0);
    }
}

inline __aicore__ void copy_out_to_gm(__gm__ bfloat16_t *gm, __cc__ float *cc,
                                      uint32_t n, uint32_t m, uint32_t kn, uint8_t unit_flag = 0)
{
    copy_matrix_cc_to_gm(gm,
                         cc,
                         0 /* fixpipeInfo.sid */,
                         n,
                         m,
                         kn /* fixpipeInfo.dstStride */,
                         M_TILE_SIZE /* fixpipeInfo.srcStride */,
                         unit_flag /* fixpipeInfo.unit_flag */,
                         QuantMode_t::F322BF16,
                         0 /* static_cast<uint8_t>(fixpipeInfo.reluEn) */,
                         0 /* fixpipeInfo.channelSplit */,
                         1 /* fixpipeInfo.nz2ndEn */);
}

inline __aicore__ void decode_mix_pre_copy_in_for_qk(decode_att_mix_context *ctx,
                                                     uint32_t gqa_head_idx,
                                                     uint32_t cumM,
                                                     uint32_t token_idx,
                                                     uint32_t q_head_idx_start,
                                                     uint32_t head_offset_len)
{
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    uint32_t head_size = ctx->head_size;
    uint32_t head_num_in_group = ctx->head_num_in_group;
    __gm__ bfloat16_t *q_gm_addr =
        ctx->q + cumM * ctx->num_qkv_heads * head_size + (q_head_idx_start % ctx->num_heads) * head_size;
    copy_gm_to_cbuf_multi_nd2nz_b16(ctx->cbuf_addr_a,
                                    q_gm_addr,
                                    0,
                                    1 /* nd2nzParams.ndNum */,
                                    head_num_in_group /* nd2nzParams.nValue */,
                                    head_size /* nd2nzParams.dValue */,
                                    0 /* nd2nzParams.srcNdMatrixStride */,
                                    head_size /* nd2nzParams.srcDValue */,
                                    CUBE_BLOCK_SIZE /* nd2nzParams.dstNzC0Stride */,
                                    1 /* nd2nzParams.dstNzNStride */,
                                    0 /* nd2nzParams.dstNzMatrixStride */);

    if (gqa_head_idx >= block_num) {
        set_aic_aiv_flag(ctx->a2v_gm, head_num_in_group, (gqa_head_idx - block_num) * head_num_in_group, head_size);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

    uint32_t block_table_id = (uint32_t)*(ctx->mapping + token_idx * ctx->mapping_len);
    __gm__ bfloat16_t *k_gm_addr = ctx->k + block_table_id * ctx->block_mem_size + head_offset_len;

    copy_to_l1_2d(ctx->cbuf_addr_b[0], k_gm_addr, 0, 0, ctx->block_size, head_size, ctx->kv_copy_step);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    copy_to_l0a(ctx->ca_addr, ctx->cbuf_addr_a, head_size);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
}

inline __aicore__ void decode_mix_compute_qk(decode_att_mix_context *ctx,
                                             uint32_t token_idx,
                                             uint32_t head_offset_len,
                                             uint32_t q_head_idx_start)
{
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID1);

    uint32_t context_len = (uint32_t)*(ctx->cached_lens + token_idx) + 1;
    int num_iters = DIV_ROUND_UP(context_len, ctx->block_size);
    uint32_t curr_idx = 0;
    uint32_t block_size = ctx->block_size;
    uint32_t head_size = ctx->head_size;
    for (int k_idx = 0; k_idx < num_iters; ++k_idx) {
        uint32_t next_idx = 1 - curr_idx;
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        if (k_idx + 1 < num_iters) {
            uint32_t loc_block_table_id = (uint32_t)*(ctx->mapping + token_idx * ctx->mapping_len + k_idx + 1);
            __gm__ bfloat16_t *loc_k_gm_addr = ctx->k + loc_block_table_id * ctx->block_mem_size + head_offset_len;
            copy_to_l1_2d(ctx->cbuf_addr_b[next_idx], loc_k_gm_addr, 0, 0,
                          block_size, head_size, ctx->kv_copy_step);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + next_idx);
        }

        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + curr_idx);
        copy_to_l0b(ctx->cb_addr[curr_idx], ctx->cbuf_addr_b[curr_idx], block_size / CUBE_BLOCK_SIZE,
                    block_size / CUBE_BLOCK_SIZE);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

        mmad(ctx->cc_addr, ctx->ca_addr, ctx->cb_addr[curr_idx], M_TILE_SIZE, head_size, block_size, true, 0);

        if (k_idx == num_iters - 1) {
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        }

        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        __gm__ bfloat16_t *kv_gm_addr = ctx->qk + q_head_idx_start * ctx->max_context_len + k_idx * block_size;

        wait_flag(PIPE_FIX, PIPE_S, EVENT_ID1);
        uint64_t config = 0x1;
        set_nd_para(config);
        copy_out_to_gm(kv_gm_addr, ctx->cc_addr, head_size, ctx->head_num_in_group, ctx->max_context_len);
        set_flag(PIPE_FIX, PIPE_S, EVENT_ID1);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        curr_idx = 1 - curr_idx;
    }
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID1);
}

inline __aicore__ void decode_mix_process_qk(decode_att_mix_context *ctx)
{
    uint32_t kSize = ctx->block_size * ctx->head_size * sizeof(bfloat16_t);
    uint32_t qSize = ctx->head_size * sizeof(bfloat16_t);

    ctx->cbuf_addr_b[0] = reinterpret_cast<__cbuf__ bfloat16_t*>((uintptr_t)qSize * CUBE_BLOCK_SIZE);
    ctx->cbuf_addr_b[1] = reinterpret_cast<__cbuf__ bfloat16_t*>((uintptr_t)(qSize * CUBE_BLOCK_SIZE + kSize));
    ctx->cb_addr[1] = reinterpret_cast<__cb__ bfloat16_t*>((uintptr_t)kSize);

    int total_task_num = ctx->num_tokens * ctx->num_kv_heads; // 按kvhead分配计算任务
    uint32_t head_num_in_group = ctx->head_num_in_group;
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    for (int gqa_head_idx = block_idx; gqa_head_idx < total_task_num; gqa_head_idx += block_num) {
        uint32_t q_head_idx_start = gqa_head_idx * head_num_in_group;
        uint32_t kv_head_idx = q_head_idx_start % ctx->num_heads / head_num_in_group;
        uint32_t token_idx = q_head_idx_start / ctx->num_heads;
        uint32_t real_token_idx = (uint32_t)*(ctx->decode_index + token_idx);
        uint32_t cumM = (uint32_t)*(ctx->cum_prompt_len + real_token_idx);
        if (ctx->m_slice > 0 && (cumM >= ctx->m_offset + ctx->m_slice || cumM < ctx->m_offset)) {
            continue;
        }

        uint32_t head_offset_len = kv_head_idx * ctx->head_size;
        decode_mix_pre_copy_in_for_qk(ctx, gqa_head_idx, cumM, real_token_idx, q_head_idx_start, head_offset_len);
        decode_mix_compute_qk(ctx, real_token_idx, head_offset_len, q_head_idx_start);

        if (gqa_head_idx + block_num >= total_task_num) {
            set_aic_aiv_flag(ctx->a2v_gm, head_num_in_group, q_head_idx_start, ctx->head_size);
        }
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    }
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

inline __aicore__ void decode_mix_pre_copy_in_for_pv(decode_att_mix_context *ctx,
                                                     uint32_t token_idx,
                                                     uint32_t head_offset_len,
                                                     uint32_t q_offset)
{
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    __gm__ bfloat16_t *qk_gm_addr = ctx->qk + q_offset * ctx->max_context_len;
    copy_to_l1_gqa(ctx->cbuf_addr_a, qk_gm_addr, ctx->head_num_in_group, MAX_CONTEXT_BLOCK_LEN, ctx->max_context_len);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    uint32_t block_table_id = (uint32_t)*(ctx->mapping + token_idx * ctx->mapping_len);
    __gm__ bfloat16_t *v_gm_addr = ctx->v + block_table_id * ctx->block_mem_size + head_offset_len;
    copy_to_l1_2d(ctx->cbuf_addr_b[0], v_gm_addr, 0, 0, ctx->block_size, ctx->head_size, ctx->kv_copy_step);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
}

inline __aicore__ void decode_mix_compute_pv(decode_att_mix_context *ctx,
                                             uint32_t token_idx,
                                             uint32_t head_offset_len,
                                             uint32_t q_offset)
{
    __gm__ int32_t *block_table_map = ctx->mapping + token_idx * ctx->mapping_len;
    __gm__ bfloat16_t *qk_gm_addr = ctx->qk + q_offset * ctx->max_context_len;

    uint32_t block_size = ctx->block_size;
    uint32_t head_size = ctx->head_size;
    uint32_t context_len = (uint32_t)*(ctx->cached_lens + token_idx) + 1;
    uint32_t cur_idx_max_context_block = 0;
    uint32_t max_context_block_len_iter_num = MAX_CONTEXT_BLOCK_LEN / block_size;
    uint32_t b_tile_factor = block_size / CUBE_BLOCK_SIZE;
    uint32_t h_tile_factor = head_size / CUBE_BLOCK_SIZE;

    uint32_t curr_idx = 0;
    uint8_t unit_flag = 2;
    bool init_val = true;

    int num_iters = DIV_ROUND_UP(context_len, block_size);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (int iter = 0; iter < num_iters; ++iter) {
        uint32_t next_idx = 1 - curr_idx;
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        if (iter + 1 < num_iters) {
            uint32_t block_table_id = (uint32_t)*(block_table_map + iter + 1);
            __gm__ bfloat16_t *v_gm_addr = ctx->v + block_table_id * ctx->block_mem_size + head_offset_len;
            copy_to_l1_2d(ctx->cbuf_addr_b[next_idx], v_gm_addr, 0, 0, block_size, head_size, ctx->kv_copy_step);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + next_idx);
        }

        if (iter == 0) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        }
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        copy_to_l0a(ctx->ca_addr, ctx->cbuf_addr_a, head_size,
                    CUBE_BLOCK_SIZE * (iter % max_context_block_len_iter_num) * block_size);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (iter + 1 < num_iters) {
            uint32_t next_iter_max_block = (iter + 1) * block_size / MAX_CONTEXT_BLOCK_LEN;
            if (next_iter_max_block != cur_idx_max_context_block) {
                cur_idx_max_context_block = next_iter_max_block;
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + next_idx);

                set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
                copy_to_l1_gqa(ctx->cbuf_addr_a, qk_gm_addr + MAX_CONTEXT_BLOCK_LEN * next_iter_max_block,
                               ctx->head_num_in_group, MAX_CONTEXT_BLOCK_LEN, ctx->max_context_len);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + next_idx);

                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
            }
        }

        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + curr_idx);
        copy_to_l0b_gqa(ctx->cb_addr[curr_idx], ctx->cbuf_addr_b[curr_idx], b_tile_factor, h_tile_factor);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);

        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        if (iter == num_iters - 1) {
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            unit_flag = 3;
        }
        mmad(ctx->cc_addr, ctx->ca_addr, ctx->cb_addr[curr_idx],
             M_TILE_SIZE, head_size, block_size, init_val, unit_flag);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        init_val = false;
        curr_idx = 1 - curr_idx;
    }
}

inline __aicore__ void decode_mix_process_pv(decode_att_mix_context *ctx)
{
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);

    uint32_t head_num_in_group = ctx->head_num_in_group;
    uint32_t head_size = ctx->head_size;
    uint32_t qk_size = MAX_CONTEXT_BLOCK_LEN * sizeof(bfloat16_t) * CUBE_BLOCK_SIZE; // 1,572,864
    uint32_t v_tile_size = ctx->block_size * head_size * sizeof(bfloat16_t);
    ctx->cbuf_addr_b[0] = reinterpret_cast<__cbuf__ bfloat16_t*>((uintptr_t)qk_size);
    ctx->cbuf_addr_b[1] = reinterpret_cast<__cbuf__ bfloat16_t*>((uintptr_t)(qk_size + v_tile_size));
    ctx->cb_addr[1] = reinterpret_cast<__cb__ bfloat16_t*>((uintptr_t)v_tile_size);

    int total_task_num = ctx->num_tokens * ctx->num_kv_heads;
    int start_offset = total_task_num % block_num;

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    for (int i = block_idx - start_offset; i < total_task_num; i += block_num) {
        if (i < 0) {
            continue;
        }
        uint32_t kv_head_idx = i % ctx->num_kv_heads;
        uint32_t token_idx = i / ctx->num_kv_heads;
        uint32_t real_token_idx = (uint32_t)*(ctx->decode_index + token_idx);
        uint32_t cumM = (uint32_t)*(ctx->cum_prompt_len + real_token_idx);
        if (ctx->m_slice > 0 && (cumM >= ctx->m_offset + ctx->m_slice || cumM < ctx->m_offset)) {
            continue;
        }
        uint32_t q_offset = i * head_num_in_group;
        uint32_t head_offset_len = kv_head_idx * head_size;
        wait_aic_aiv_flag(ctx->v2a_gm, head_num_in_group, q_offset, head_size);

        decode_mix_pre_copy_in_for_pv(ctx, real_token_idx, head_offset_len, q_offset);
        decode_mix_compute_pv(ctx, real_token_idx, head_offset_len, q_offset);

        set_flag(PIPE_FIX, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_S, EVENT_ID1);
        uint8_t unit_flag = 3;
        __gm__ bfloat16_t *qkv_gm_addr = ctx->o + cumM * ctx->num_heads * head_size +
            kv_head_idx * head_num_in_group * head_size;
        copy_out_to_gm(qkv_gm_addr, ctx->cc_addr, head_size, head_num_in_group, head_size, unit_flag);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        reset_aic_aiv_flag(ctx->v2a_gm, head_num_in_group, q_offset, head_size);
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);

    pipe_barrier(PIPE_ALL);
}

inline __aicore__ void decode_att_mix_aic(decode_att_mix_context *ctx)
{
    if (ctx == NULL) {
        return;
    }
    decode_mix_process_qk(ctx);
    decode_mix_process_pv(ctx);
}

#elif __DAV_C220_VEC__

const uint64_t BASE_IN_HIGH_BIT = 1L << 63;
const uint64_t BASE_IN_LOW_BIT = 1;
const uint32_t MAX_MASK0_BIT = 64;
const uint32_t DIVIDED_MASK_BIT = 64;
const uint64_t MASK_ALL = -1;
const uint64_t MASK_NONE = 0;
const uint32_t MAX_MASK_BIT = 128;

inline __aicore__ void reset_mask()
{
    set_vector_mask(MASK_ALL, MASK_ALL);
}

inline __aicore__ void set_mask0(uint32_t len)
{
    if (len >= MAX_MASK0_BIT) {
        set_vector_mask(MASK_NONE, MASK_ALL);
        return;
    }
    uint64_t mask = 0;
    uint64_t base = BASE_IN_LOW_BIT;
    uint64_t temp = len % DIVIDED_MASK_BIT;
    for (int64_t i = 0; i < temp; i++) {
        mask |= base << i;
    }
    set_vector_mask(MASK_NONE, mask);
}

inline __aicore__ void set_mask0_from_highbit(uint32_t len) {
    if (len >= MAX_MASK0_BIT) {
        set_vector_mask(MASK_NONE, MASK_ALL);
        return;
    }
    uint64_t mask = 0;
    uint64_t base = BASE_IN_HIGH_BIT;
    uint64_t temp = len % DIVIDED_MASK_BIT;
    for (int64_t i = 0; i < temp; i++) {
        mask |= base >> i;
    }
    set_vector_mask(MASK_NONE, mask);
}

inline __aicore__ void set_mask_from_highbit(uint32_t len) {
    if (len == 128) {
        set_vector_mask(MASK_ALL, MASK_ALL);
        return;
    }
    uint64_t mask = 0;
    uint64_t base = BASE_IN_HIGH_BIT;
    uint64_t temp = len % DIVIDED_MASK_BIT;
    for (int64_t i = 0; i < temp; i++) {
        mask |= base >> i;
    }
    if (len >= DIVIDED_MASK_BIT) {
        set_vector_mask(MASK_ALL, mask);
        return;
    }
    set_vector_mask(mask, MASK_NONE);
}

//128 * 4 * sizeof(float) for qk_ub_addr, need 3 * 128 * sizeof(float) for (48 * 1024 - 2 * 128) * sizeof(float) qk_result
const uint32_t MAX_UB_SIZE = 192 * 1024;
const uint32_t QK_RESULT_SIZE = 4 * VECTOR_MAX_BYTESIZE;
const uint32_t MAX_SUB_CONTEXT_SIZE = (MAX_UB_SIZE - 3 * QK_RESULT_SIZE) / 6;
const uint32_t MAX_SUB_CONTEXT_SIZE_F32 = MAX_SUB_CONTEXT_SIZE * 2;

const uint32_t MAX_BLOCK_NUM_ITER = MAX_SUB_CONTEXT_SIZE_F32 / VECTOR_MAX_BYTESIZE;
const uint32_t MAX_BLOCK_CONTEXT_LEN = MAX_SUB_CONTEXT_SIZE / sizeof(bfloat16_t);


inline __aicore__ void decode_att_mix_aiv(decode_att_mix_context *ctx)
{
    set_mask_norm();
    reset_mask();
    uint32_t addr = 0;
    auto *qk_reduce_ub_bf16 = reinterpret_cast<__ubuf__ bfloat16_t*>((uintptr_t)addr);
    addr += MAX_SUB_CONTEXT_SIZE;
    auto *qk_reduce_ub_f32 = reinterpret_cast<__ubuf__ float*>((uintptr_t)addr);
    addr += MAX_SUB_CONTEXT_SIZE_F32;
    auto *qk_out_ub_bf16 = reinterpret_cast<__ubuf__ bfloat16_t*>((uintptr_t)addr);
    addr += MAX_SUB_CONTEXT_SIZE;
    auto *qk_out_ub_f32 = reinterpret_cast<__ubuf__ float*>((uintptr_t)addr);
    addr += MAX_SUB_CONTEXT_SIZE_F32;
    auto *qk_ub_addr = reinterpret_cast<__ubuf__ float*>((uintptr_t)addr);
    addr += QK_RESULT_SIZE;
    auto *qk_reduce_sum_addr = reinterpret_cast<__ubuf__ float*>((uintptr_t)addr);
    addr += QK_RESULT_SIZE;
    auto *qk_max_qk_addr = reinterpret_cast<__ubuf__ float*>((uintptr_t)addr);

    float float_min = -3.4028235e+38;
    uint32_t process_num = ctx->num_tokens * ctx->num_heads;
    uint32_t aiv_block_num = block_num * 2;
    uint32_t id = get_block_idx() * 2 + get_subblockid();
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (uint32_t process = id; process < process_num; process += aiv_block_num) {
        uint32_t token_idx = process / ctx->num_heads;

        uint32_t real_token_idx = (uint32_t)*(ctx->decode_index + token_idx);
        uint32_t cumM = (uint32_t)*(ctx->cum_prompt_len + real_token_idx);
        if (ctx->m_slice > 0 && (cumM >= ctx->m_offset + ctx->m_slice || cumM < ctx->m_offset)) {
            continue;
        }
        uint32_t context_len = (uint32_t)*(ctx->cached_lens + real_token_idx) + 1;

        __gm__ bfloat16_t *qk_gm_addr = ctx->qk + process * ctx->max_context_len;

        wait_aic_aiv_flag(ctx->a2v_gm, 1, process, ctx->head_size);
        
        uint32_t num_iters = DIV_ROUND_UP(context_len, VECTOR_MAX_NUM_OF_FP32);
        uint32_t sub_block_number = DIV_ROUND_UP(context_len * sizeof(bfloat16_t), MAX_SUB_CONTEXT_SIZE);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        __ubuf__ float *max_qk[4] = {qk_max_qk_addr, qk_max_qk_addr + 64, qk_max_qk_addr + 128, qk_max_qk_addr + 192}; // 0~2 for sub_block`max_qk 3 for whole`max_qk
        int max_qk_flag[4] = {0, 0, 0, 0};
        __ubuf__ float *reduce_sum[4] = {qk_reduce_sum_addr, qk_reduce_sum_addr + 64, qk_reduce_sum_addr + 128, qk_reduce_sum_addr + 192};
        vector_dup(qk_max_qk_addr, float_min, 4, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vector_dup(qk_reduce_sum_addr, float(0), 4, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        for (uint32_t sub_block = 0; sub_block < sub_block_number; sub_block++) {
            uint32_t cur_iter_idx = sub_block * MAX_BLOCK_NUM_ITER;
            uint32_t cur_qk_offset = sub_block * MAX_BLOCK_CONTEXT_LEN;
            uint32_t cur_num_iters = (sub_block < sub_block_number - 1) ? MAX_BLOCK_NUM_ITER : (num_iters - cur_iter_idx);
            uint32_t cur_qk_context_len = (sub_block < sub_block_number - 1) ? MAX_BLOCK_CONTEXT_LEN : (context_len - cur_qk_offset);

            vector_dup(qk_reduce_ub_f32, float_min, MAX_BLOCK_NUM_ITER, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_ubuf_align_b16(qk_reduce_ub_bf16, qk_gm_addr + cur_qk_offset, 0, 1,
                                      cur_qk_context_len * sizeof(bfloat16_t),
                                      0, 0, 0, 0);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            vconv_bf162f32(qk_reduce_ub_f32, qk_reduce_ub_bf16, cur_num_iters, 1, 1, 8, 4);

            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);

            // 将多拷贝进来的数据进行置位
            uint32_t rest = cur_qk_context_len % VECTOR_MAX_NUM_OF_FP32;
            if (rest > 0) {
                uint32_t start = ROUND_DOWN(cur_qk_context_len, VECTOR_MAX_NUM_OF_FP32);
                set_mask0_from_highbit(VECTOR_MAX_NUM_OF_FP32 - rest);
                vector_dup(qk_reduce_ub_f32 + start, float_min, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                reset_mask();
            }

            // max = MAX(QK)
            ReduceMax(qk_ub_addr, qk_reduce_ub_f32, cur_qk_context_len);
            
            vbrcb((__ubuf__ uint32_t *) qk_ub_addr, (__ubuf__ uint32_t *) qk_ub_addr, 1, 8, 1);
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            float tmp_1 = *qk_ub_addr;
            float tmp_2 = *max_qk[3];
            if (tmp_1 > tmp_2) {
                for (int max_qk_idx = 0; max_qk_idx < sub_block; max_qk_idx++) {
                    max_qk_flag[max_qk_idx] = 1;
                }
                copy_ubuf_to_ubuf(max_qk[3], qk_ub_addr, 0, 1, 8, 1, 1);
                pipe_barrier(PIPE_V);
            }
            copy_ubuf_to_ubuf(max_qk[sub_block], max_qk[3], 0, 1, 8, 1, 1);
            copy_ubuf_to_ubuf(qk_ub_addr, max_qk[3], 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);

            vsub(qk_reduce_ub_f32, qk_reduce_ub_f32, qk_ub_addr, cur_num_iters, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            vexp(qk_reduce_ub_f32, qk_reduce_ub_f32, cur_num_iters, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // s = reduce_sum(EXP)
            ReduceSum(qk_ub_addr, qk_reduce_ub_f32, cur_qk_context_len);

            vbrcb((__ubuf__ uint32_t *) qk_ub_addr, (__ubuf__ uint32_t *) qk_ub_addr, 1, 8, 1);
            pipe_barrier(PIPE_V);
            // update reduce_sum
            if (sub_block) {
                if (tmp_1 > tmp_2) {
                    vsub(qk_ub_addr + VECTOR_MAX_NUM_OF_FP32 * 3, max_qk[sub_block - 1], max_qk[sub_block], 1, 1, 1, 1, 8, 8, 1);
                    pipe_barrier(PIPE_V);
                    vexp(qk_ub_addr + VECTOR_MAX_NUM_OF_FP32 * 3, qk_ub_addr + VECTOR_MAX_NUM_OF_FP32 * 3, 1, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    vmul(qk_ub_addr + VECTOR_MAX_NUM_OF_FP32 * 3, qk_ub_addr + VECTOR_MAX_NUM_OF_FP32 * 3, reduce_sum[sub_block - 1], 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                    vadd(qk_ub_addr, qk_ub_addr, qk_ub_addr + VECTOR_MAX_NUM_OF_FP32 * 3, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                } else {
                    vadd(qk_ub_addr, qk_ub_addr, reduce_sum[sub_block - 1], 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
            }
            copy_ubuf_to_ubuf(reduce_sum[sub_block], qk_ub_addr, 0, 1, 8, 1, 1);
            copy_ubuf_to_ubuf(reduce_sum[3], qk_ub_addr, 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            vconv_f322bf16r(qk_reduce_ub_bf16, qk_reduce_ub_f32, cur_num_iters, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);


            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm((qk_gm_addr + cur_qk_offset), qk_reduce_ub_bf16, 0, 1, 4 * cur_num_iters, 0, 0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);

        for (int sub_block = 0; sub_block < sub_block_number; sub_block++) {
            uint32_t cur_iter_idx = sub_block * MAX_BLOCK_NUM_ITER;
            uint32_t cur_qk_offset = sub_block * MAX_BLOCK_CONTEXT_LEN;
            uint32_t cur_num_iters = (sub_block < sub_block_number - 1) ? MAX_BLOCK_NUM_ITER : (num_iters - cur_iter_idx);
            uint32_t cur_qk_context_len = (sub_block < sub_block_number - 1) ? MAX_BLOCK_CONTEXT_LEN : (context_len - cur_qk_offset);


            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            copy_gm_to_ubuf_align_b16(qk_reduce_ub_bf16, qk_gm_addr + cur_qk_offset, 0, 1, cur_qk_context_len * 2, 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            vconv_bf162f32(qk_reduce_ub_f32, qk_reduce_ub_bf16,
                           DIV_ROUND_UP(cur_qk_context_len * sizeof(float), VECTOR_MAX_BYTESIZE),
                           1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

            uint32_t rest = cur_qk_context_len % VECTOR_MAX_NUM_OF_FP32;
            if (rest) {
                uint32_t start = ROUND_DOWN(cur_qk_context_len, VECTOR_MAX_NUM_OF_FP32);
                set_mask0_from_highbit(VECTOR_MAX_NUM_OF_FP32 - rest);
                vector_dup(qk_reduce_ub_f32 + start, float_min, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                reset_mask();
            }

            if (max_qk_flag[sub_block]) {
                vsub(qk_ub_addr + VECTOR_MAX_NUM_OF_FP32 * 3, max_qk[sub_block], max_qk[3], 1, 1, 1, 1, 8, 8, 1);
                pipe_barrier(PIPE_V);
                vexp(qk_ub_addr + VECTOR_MAX_NUM_OF_FP32 * 3, qk_ub_addr + VECTOR_MAX_NUM_OF_FP32 * 3, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vmul(qk_reduce_ub_f32, qk_reduce_ub_f32, qk_ub_addr + 3 * VECTOR_MAX_NUM_OF_FP32, cur_num_iters, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
            }
            vdiv(qk_out_ub_f32, qk_reduce_ub_f32, reduce_sum[3], cur_num_iters, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            vconv_f322bf16r(qk_out_ub_bf16, qk_out_ub_f32, cur_num_iters, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
            
            rest = cur_qk_context_len % VECTOR_MAX_NUM_OF_FP16;
            if (rest > 0) {
                uint32_t start = ROUND_DOWN(cur_qk_context_len, VECTOR_MAX_NUM_OF_FP16);
                set_mask_from_highbit(VECTOR_MAX_NUM_OF_FP16 - rest);
                vector_dup(qk_out_ub_bf16 + start, bfloat16_t(0), 1, 1, 1, 8, 0);
                reset_mask();
                pipe_barrier(PIPE_V);
            }

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            uint32_t offset = ROUND_UP(cur_qk_context_len * sizeof(bfloat16_t), VECTOR_MAX_BYTESIZE) / BLOCK_SIZE;
            copy_ubuf_to_gm(qk_gm_addr + cur_qk_offset, qk_out_ub_bf16, 0, 1, offset, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        }
        pipe_barrier(PIPE_ALL); // 此处的PIPE_ALL必须要，是用于核间同步的，保证结果写入到GM，如果用硬件同步可能可以去掉
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        reset_aic_aiv_flag(ctx->a2v_gm, 1, process, ctx->head_size);
        set_aic_aiv_flag(ctx->v2a_gm, 1, process, ctx->head_size);

    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}



#endif

extern "C" __global__ __aicore__ void decode_att_bf16(__gm__ uint8_t* a2v, __gm__ uint8_t* v2a,
                                                 __gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* v,
                                                 __gm__ uint8_t* cached_lens, __gm__ uint8_t* mapping,
                                                 __gm__ uint8_t* qk, __gm__ uint8_t* o,
                                                 __gm__ uint8_t* decode_index, __gm__ uint8_t* cum_prompt_len,
                                                 uint32_t num_tokens, uint32_t num_heads, uint32_t head_size,
                                                 uint32_t block_size, uint32_t mapping_len, uint32_t num_kv_heads,
                                                 uint32_t max_context_len, uint32_t num_qkv_heads,
                                                 uint32_t m_offset,
                                                 uint32_t m_slice)
{
    decode_att_mix_context ctx{};
    init_decode_att_mix_context(&ctx, a2v, v2a, q, k, v, cached_lens, mapping, qk,
                                o, decode_index, cum_prompt_len, num_tokens, num_heads, head_size,
                                block_size, mapping_len, num_kv_heads, max_context_len, num_qkv_heads,
                                m_offset, m_slice);
#ifdef __DAV_C220_CUBE__
    decode_att_mix_aic(&ctx);
#elif __DAV_C220_VEC__
    decode_att_mix_aiv(&ctx);
#endif
}
