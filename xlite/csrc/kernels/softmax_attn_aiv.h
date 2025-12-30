/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_SOFTMAX_ATTN_AIV_H_
#define _XLITE_SOFTMAX_ATTN_AIV_H_

#include "kernel_operator.h"
#include "kernel_macro.h"
using namespace AscendC;

#if __DAV_C220_VEC__
/* ROUND_UP(n * sizeof(Dtype), VECTOR_MAX_BYTESIZE) * 2 +
 *     ROUND_UP(n * sizeof(CalcDtype), VECTOR_MAX_BYTESIZE) +
 *     DIV_ROUND_UP(n * sizeof(CalcDtype), VECTOR_MAX_BYTESIZE) * sizeof(CalcDtype) <= ub size(196608B)
 * float16_t: n <= 32640
 * bfloat16_t: n <= 24320
 */
template<typename Dtype, typename CalcDtype>
inline __aicore__ void RunAivSoftmax(__gm__ Dtype *buf, uint32_t m, uint32_t n, uint32_t calcLen)
{
    CalcDtype min;
    if constexpr (std::is_same<CalcDtype, half>::value) {
        min = half(-65504);
    } else {
        min = -3.4028235e+38;
    }

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint64_t off = 0;
    __ubuf__ Dtype *in = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(n * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *out = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(n * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ CalcDtype *cal = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t)off);
    off += ROUND_UP(n * sizeof(CalcDtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ CalcDtype *temp = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t)off);
    __ubuf__ CalcDtype *ptr;

    int calPad = VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
    int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    for (int idx = 0; idx < m; ++idx) {
        int actualCalcLen = calcLen + idx; // 每一行开始mask的位置
        if (actualCalcLen > n) {
            actualCalcLen = n;
        }
        int padCachedTokens = ROUND_UP(actualCalcLen, calPad);
        int repeat = padCachedTokens / calPad;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        copy_gm_to_ubuf(in, buf + idx * n, 0, 1,
                        DIV_ROUND_UP(actualCalcLen * sizeof(Dtype), 32), 0, 0);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if constexpr (std::is_same<CalcDtype, half>::value) {
            ptr = in;
        } else {
            vconv_bf162f32((__ubuf__ float *)cal, (__ubuf__ Dtype *)in, repeat, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            ptr = cal;
        }

        // dup calPad
        if (actualCalcLen % calPad != 0) {
            SetMaskFromHighBit(calPad, calPad - actualCalcLen % calPad);
            vector_dup(ptr + ROUND_DOWN(actualCalcLen, calPad), min, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }

        // max
        ReduceMax(temp, ptr, actualCalcLen);

        // broadcast一个max标量为一个block大小的向量，避免使用scalar运算
        if constexpr (std::is_same<CalcDtype, half>::value) {
            vbrcb((__ubuf__ uint16_t*)temp, (__ubuf__ uint16_t*)temp, 0, 0, 1);
        } else {
            vbrcb((__ubuf__ uint32_t*)temp, (__ubuf__ uint32_t*)temp, 0, 0, 1);
        }
        pipe_barrier(PIPE_V);

        // QK - max
        vsub(cal, ptr, temp, repeat, 1, 1, 0, 8, 8, 0);
        pipe_barrier(PIPE_V);
        if constexpr (std::is_same<CalcDtype, half>::value) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        }

        // EXP = exp(QK-max)
        vexp(cal, cal, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // s = Reduce_sum(EXP)
        ReduceSum(temp, cal, actualCalcLen);

        // broadcast一个s = Reduce_sum(EXP)标量为一个block大小的向量，避免使用scalar运算
        if constexpr (std::is_same<CalcDtype, half>::value) {
            vbrcb((__ubuf__ uint16_t*)temp, (__ubuf__ uint16_t*)temp, 0, 0, 1);
        } else {
            vbrcb((__ubuf__ uint32_t*)temp, (__ubuf__ uint32_t*)temp, 0, 0, 1);
        }
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        if constexpr (std::is_same<CalcDtype, half>::value) {
            vdiv(out, cal, temp, repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
        } else {
            vdiv(cal, cal, temp, repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            vconv_f322bf16r(out, cal, repeat, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
        }

        if (n > actualCalcLen) {
            if (actualCalcLen % pad != 0) {
                SetMaskFromHighBit(pad, pad - actualCalcLen % pad);
                vector_dup(out + ROUND_DOWN(actualCalcLen, pad), Dtype(0), 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }
            int last = ROUND_UP(actualCalcLen, pad);
            if (n > last) {
                vector_dup(out + last, Dtype(0), (n - last) / pad, 1, 1, 8, 0);
            }
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if ((n * sizeof(Dtype)) % BLOCK_SIZE == 0) {
            copy_ubuf_to_gm(buf + idx * n, out, 0, 1, n * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
        } else {
            copy_ubuf_to_gm_align_b16(buf + idx * n, out, 0, 1, n * sizeof(Dtype), 0, 0, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    pipe_barrier(PIPE_ALL);
}

const uint64_t BASE_IN_HIGH_BIT = 1L << 63;
const uint64_t BASE_IN_LOW_BIT = 1;
const uint32_t MAX_MASK_BIT = 128;
const uint32_t DIVIDED_MASK_BIT = 64;
const uint64_t MASK_ALL = -1;
const uint64_t MASK_NONE = 0;

inline __aicore__ void set_mask(uint32_t len)
{
    if (len == MAX_MASK_BIT) {
        set_vector_mask(MASK_ALL, MASK_ALL);
        return;
    }
    uint64_t mask = 0;
    uint64_t base = BASE_IN_LOW_BIT;
    uint64_t temp = len % DIVIDED_MASK_BIT;
    for (int64_t i = 0; i < temp; i++) {
        mask |= base << i;
    }
    if (len >= DIVIDED_MASK_BIT) {
        set_vector_mask(mask, MASK_ALL);
        return;
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

inline __aicore__ void RunAivSoftmaxLongFP16(__gm__ float16_t *qk_gm_addr, uint32_t context_len)
{
    //128 * 4 * sizeof(half) for qk_ub_addr, need 3 * 128 * sizeof(half) for (48 * 1024 - 2 * 128) * sizeof(half) qk_result
    const uint32_t MAX_SUB_CONTEXT_SIZE = (192 * 1024 - 128 * 12 * sizeof(half)) / 2;
    set_mask_norm();
    set_vector_mask((uint64_t) -1, (uint64_t) -1);

    auto *qk_reduce_ub_addr = reinterpret_cast<__ubuf__ half *>((uintptr_t) 0);
    auto *qk_out_ub_addr = reinterpret_cast<__ubuf__ half *>((uintptr_t) MAX_SUB_CONTEXT_SIZE);
    auto *qk_ub_addr = reinterpret_cast<__ubuf__ half *>((uintptr_t) MAX_SUB_CONTEXT_SIZE * 2);
    auto *qk_reduce_sum_addr = reinterpret_cast<__ubuf__ half *>((uintptr_t) MAX_SUB_CONTEXT_SIZE * 2 + 128 * 4 * sizeof(half));
    auto *qk_max_qk_addr = reinterpret_cast<__ubuf__ half *>((uintptr_t) MAX_SUB_CONTEXT_SIZE * 2 + 128 * 8 * sizeof(half));

    int id = get_block_idx() * 2 + get_subblockid();
    uint32_t num_iters = DIV_ROUND_UP(context_len, VECTOR_MAX_NUM_OF_FP16);
    uint32_t max_block_num_iter = MAX_SUB_CONTEXT_SIZE / 256;
    int sub_block_number = (context_len * sizeof(half) + MAX_SUB_CONTEXT_SIZE - 1) / MAX_SUB_CONTEXT_SIZE;

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    __ubuf__ half *max_qk[4] = {qk_max_qk_addr, qk_max_qk_addr + 128, qk_max_qk_addr + 256, qk_max_qk_addr + 384}; // 0~2 for sub_block`max_qk 3 for whole`max_qk
    int max_qk_flag[4] = {0, 0, 0, 0};
    __ubuf__ half *reduce_sum[4] = {qk_reduce_sum_addr, qk_reduce_sum_addr + 128, qk_reduce_sum_addr + 256, qk_reduce_sum_addr + 384};
    vector_dup(qk_max_qk_addr, half(-65504), 4, 1, 1, 8, 0);
    vector_dup(qk_reduce_sum_addr, half(0), 4, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    for (int sub_block = 0; sub_block < sub_block_number; sub_block++) {
        uint32_t cur_iter_idx = sub_block * max_block_num_iter;
        uint32_t cur_num_iters = (sub_block < sub_block_number - 1) ? max_block_num_iter : (num_iters - cur_iter_idx);
        uint32_t cur_qk_context_len = (sub_block < sub_block_number - 1) ? max_block_num_iter * 128 : (context_len - cur_iter_idx * 128);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        vector_dup(qk_reduce_ub_addr, half(-65504), 255, 1, 1, 8, 0);
        vector_dup(qk_reduce_ub_addr + 255 * 128, half(-65504), max_block_num_iter - 255, 1, 1, 8, 0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); // 只用了一半空间，也许可以做pingpong
        copy_gm_to_ubuf_align_b16(qk_reduce_ub_addr, qk_gm_addr + cur_iter_idx * 128, 0, 1, cur_qk_context_len * 2, 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // 将多拷贝进来的数据进行置位
        uint32_t start = cur_qk_context_len * sizeof(half) / 32 * 32;
        uint32_t rest = cur_qk_context_len * sizeof(half) % 32 / sizeof(half);
        if (rest) {
            set_mask_from_highbit(128 - rest);
            vector_dup(qk_reduce_ub_addr + start / sizeof(half), half(-65504), 1, 1, 1, 8, 0);
            set_vector_mask((uint64_t) -1, (uint64_t) -1);
            pipe_barrier(PIPE_V);
        }

        // max = MAX(QK)
        if (cur_num_iters <= 255) {
            vcmax(qk_ub_addr, qk_reduce_ub_addr, cur_num_iters, 1, 1, 8, ONLY_VALUE);
            pipe_barrier(PIPE_V);
            if (cur_num_iters <= 128) {
                set_mask(cur_num_iters);
                vcmax(qk_ub_addr, qk_ub_addr, 1, 1, 1, 8, ONLY_VALUE);
                set_vector_mask((uint64_t) -1, (uint64_t) -1);
                pipe_barrier(PIPE_V);
            } else {
                set_mask_from_highbit(256 - cur_num_iters);
                vector_dup(qk_ub_addr + 128, half(-65504), 1, 1, 1, 8, 0);
                set_vector_mask((uint64_t) -1, (uint64_t) -1);
                pipe_barrier(PIPE_V);
                vcmax(qk_ub_addr, qk_ub_addr, 2, 1, 1, 8, ONLY_VALUE);
                pipe_barrier(PIPE_V);
                set_mask(2);
                vcmax(qk_ub_addr, qk_ub_addr, 1, 1, 1, 8, ONLY_VALUE);
                pipe_barrier(PIPE_V);
                set_vector_mask((uint64_t) -1, (uint64_t) -1);
            }
        } else {
            vcmax(qk_ub_addr, qk_reduce_ub_addr, 128, 1, 1, 8, ONLY_VALUE);
            vcmax(qk_ub_addr + 128, qk_reduce_ub_addr + 128 * 128, cur_num_iters - 128, 1, 1, 8, ONLY_VALUE);
            pipe_barrier(PIPE_V);

            set_mask_from_highbit(384 - cur_num_iters);
            vector_dup(qk_ub_addr + 256, half(-65504), 1, 1, 1, 8, 0);
            set_vector_mask((uint64_t) -1, (uint64_t) -1);
            pipe_barrier(PIPE_V);
            vcmax(qk_ub_addr, qk_ub_addr, 3, 1, 1, 8, ONLY_VALUE);
            pipe_barrier(PIPE_V);
            set_mask(3);
            vcmax(qk_ub_addr, qk_ub_addr, 1, 1, 1, 8, ONLY_VALUE);
            pipe_barrier(PIPE_V);
            set_vector_mask((uint64_t) -1, (uint64_t) -1);
        }
        vbrcb((__ubuf__ uint16_t *) qk_ub_addr, (__ubuf__ uint16_t *) qk_ub_addr, 1, 8, 1);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float tmp_1 = (float)*qk_ub_addr;
        float tmp_2 = (float)*max_qk[3];
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

        if (cur_num_iters <= 255) {
            vsub(qk_reduce_ub_addr, qk_reduce_ub_addr, qk_ub_addr, cur_num_iters, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            vexp(qk_reduce_ub_addr, qk_reduce_ub_addr, cur_num_iters, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // s = reduce_sum(EXP)
            vcadd(qk_ub_addr, qk_reduce_ub_addr, cur_num_iters, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            if (cur_num_iters <= 128) {
                set_mask(cur_num_iters); // 可以支持到128*128 = 16K
                vcadd(qk_ub_addr, qk_ub_addr, 1, 1, 1, 8, 0);
                set_vector_mask((uint64_t) -1, (uint64_t) -1);
                pipe_barrier(PIPE_V);
            } else {
                set_mask_from_highbit(256 - cur_num_iters);
                vector_dup(qk_ub_addr + 128, half(0), 1, 1, 1, 8, 0);
                set_vector_mask(MASK_ALL, MASK_ALL);
                pipe_barrier(PIPE_V);
                vcadd(qk_ub_addr, qk_ub_addr, 2, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_mask(2);
                vcadd(qk_ub_addr, qk_ub_addr, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_vector_mask(MASK_ALL, MASK_ALL);
            }
        } else {
            vsub(qk_reduce_ub_addr, qk_reduce_ub_addr, qk_ub_addr, 255, 1, 1, 0, 8, 8, 0); // 与上面不同，因为上面ub+255不对齐
            vsub(qk_reduce_ub_addr + 255 * 128, qk_reduce_ub_addr + 255 * 128, qk_ub_addr, cur_num_iters - 255, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            vexp(qk_reduce_ub_addr, qk_reduce_ub_addr, 255, 1, 1, 8, 8);
            vexp(qk_reduce_ub_addr + 255 * 128, qk_reduce_ub_addr + 255 * 128, cur_num_iters - 255, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // s = reduce_sum(EXP);
            vcadd(qk_ub_addr, qk_reduce_ub_addr, 128, 1, 1, 8, 0);
            vcadd(qk_ub_addr + 128, qk_reduce_ub_addr + 128 * 128, cur_num_iters - 128, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);

            set_mask_from_highbit(384 - cur_num_iters);
            vector_dup(qk_ub_addr + 256, half(0), 1, 1, 1, 8, 0);
            set_vector_mask(MASK_ALL, MASK_ALL);
            pipe_barrier(PIPE_V);
            vcadd(qk_ub_addr, qk_ub_addr, 3, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_mask(3);
            vcadd(qk_ub_addr, qk_ub_addr, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_vector_mask(MASK_ALL, MASK_ALL);
        }

        vbrcb((__ubuf__ uint16_t *) qk_ub_addr, (__ubuf__ uint16_t *) qk_ub_addr, 1, 8, 1);
        pipe_barrier(PIPE_V);
        // update reduce_sum
        if (sub_block) {
            if (tmp_1 > tmp_2) {
                vsub(qk_ub_addr + 128 * 3, max_qk[sub_block - 1], max_qk[sub_block], 1, 1, 1, 1, 8, 8, 1);
                pipe_barrier(PIPE_V);
                vexp(qk_ub_addr + 128 * 3, qk_ub_addr + 128 * 3, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vmul(qk_ub_addr + 128 * 3, qk_ub_addr + 128 * 3, reduce_sum[sub_block - 1], 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vadd(qk_ub_addr, qk_ub_addr, qk_ub_addr + 128 * 3, 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            } else {
                vadd(qk_ub_addr, qk_ub_addr, reduce_sum[sub_block - 1], 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }
        }
        copy_ubuf_to_ubuf(reduce_sum[sub_block], qk_ub_addr, 0, 1, 8, 1, 1);
        copy_ubuf_to_ubuf(reduce_sum[3], qk_ub_addr, 0, 1, 8, 1, 1);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm((qk_gm_addr + cur_iter_idx * 128), qk_reduce_ub_addr, 0, 1, 8 * cur_num_iters, 0, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);

    for (int sub_block = 0; sub_block < sub_block_number; sub_block++) {
        int cur_iter_idx = sub_block * max_block_num_iter;
        uint32_t cur_num_iters = (sub_block < sub_block_number - 1) ? max_block_num_iter : (num_iters - cur_iter_idx);
        uint32_t cur_qk_context_len = (sub_block < sub_block_number - 1) ? max_block_num_iter * 128 : (context_len - cur_iter_idx * 128);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        copy_gm_to_ubuf_align_b16(qk_reduce_ub_addr, qk_gm_addr + cur_iter_idx * 128, 0, 1, cur_qk_context_len * 2, 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        uint32_t start = ROUND_DOWN(cur_qk_context_len, VECTOR_MAX_NUM_OF_FP16);
        uint32_t rest = cur_qk_context_len % VECTOR_MAX_NUM_OF_FP16;
        if (rest > 0) {
            set_mask_from_highbit(VECTOR_MAX_NUM_OF_FP16 - rest);
            vector_dup(qk_reduce_ub_addr + start, half(0), 1, 1, 1, 8, 0);
            set_vector_mask(MASK_ALL, MASK_ALL);
            pipe_barrier(PIPE_V);
        }

        if (max_qk_flag[sub_block]) {
            vsub(qk_ub_addr + 128 * 3, max_qk[sub_block], max_qk[3], 1, 1, 1, 1, 8, 8, 1);
            pipe_barrier(PIPE_V);
            vexp(qk_ub_addr + 128 * 3, qk_ub_addr + 128 * 3, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            if (cur_num_iters <= 255) {
                vmul(qk_reduce_ub_addr, qk_reduce_ub_addr, qk_ub_addr + 3 * 128, cur_num_iters, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
            } else {
                vmul(qk_reduce_ub_addr, qk_reduce_ub_addr, qk_ub_addr + 3 * 128, 255, 1, 1, 0, 8, 8, 0);
                vmul(qk_reduce_ub_addr + 255 * 128, qk_reduce_ub_addr + 255 * 128, qk_ub_addr + 3 * 128, cur_num_iters - 255, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
            }
        }
        if (cur_num_iters <= 255) {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            vdiv(qk_out_ub_addr, qk_reduce_ub_addr, reduce_sum[3], cur_num_iters, 1, 1, 0, 8, 8, 0);
        } else {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            vdiv(qk_out_ub_addr, qk_reduce_ub_addr, reduce_sum[3], 255, 1, 1, 0, 8, 8, 0);
            vdiv(qk_out_ub_addr + 255 * 128, qk_reduce_ub_addr + 255 * 128, reduce_sum[3], cur_num_iters - 255, 1, 1, 0, 8, 8, 0);
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

        if (rest > 0) {
            set_mask_from_highbit(VECTOR_MAX_NUM_OF_FP16 - rest);
            vector_dup(qk_out_ub_addr + start, half(0), 1, 1, 1, 8, 0);
            set_vector_mask(MASK_ALL, MASK_ALL);
            pipe_barrier(PIPE_V);
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        copy_ubuf_to_gm((qk_gm_addr + cur_iter_idx * 128), qk_out_ub_addr, 0, 1, 8 * cur_num_iters, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    }
    pipe_barrier(PIPE_ALL); // 此处的PIPE_ALL必须要，是用于核间同步的，保证结果写入到GM，如果用硬件同步可能可以去掉
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    pipe_barrier(PIPE_ALL);
}


const uint32_t MAX_MASK0_BIT = 64;

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

inline __aicore__ void RunAivSoftmaxLongBF16(__gm__ bfloat16_t *qk_gm_addr, uint32_t context_len)
{
    //128 * 4 * sizeof(float) for qk_ub_addr, need 3 * 128 * sizeof(float) for (48 * 1024 - 2 * 128) * sizeof(float) qk_result
    const uint32_t MAX_UB_SIZE = 192 * 1024;
    const uint32_t QK_RESULT_SIZE = 4 * VECTOR_MAX_BYTESIZE;
    const uint32_t MAX_SUB_CONTEXT_SIZE = (MAX_UB_SIZE - 3 * QK_RESULT_SIZE) / 6;
    const uint32_t MAX_SUB_CONTEXT_SIZE_F32 = MAX_SUB_CONTEXT_SIZE * 2;

    const uint32_t MAX_BLOCK_NUM_ITER = MAX_SUB_CONTEXT_SIZE_F32 / VECTOR_MAX_BYTESIZE;
    const uint32_t MAX_BLOCK_CONTEXT_LEN = MAX_SUB_CONTEXT_SIZE / sizeof(bfloat16_t);
    set_mask_norm();
    set_vector_mask(MASK_ALL, MASK_ALL);
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
    uint32_t aiv_block_num = block_num * 2;
    uint32_t id = get_block_idx() * 2 + get_subblockid();
    uint32_t num_iters = DIV_ROUND_UP(context_len, VECTOR_MAX_NUM_OF_FP32);
    uint32_t sub_block_number = DIV_ROUND_UP(context_len * sizeof(bfloat16_t), MAX_SUB_CONTEXT_SIZE);

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
            set_vector_mask(MASK_ALL, MASK_ALL);
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
            set_vector_mask(MASK_ALL, MASK_ALL);
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
            set_vector_mask(MASK_ALL, MASK_ALL);
            pipe_barrier(PIPE_V);
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        uint32_t offset = ROUND_UP(cur_qk_context_len * sizeof(bfloat16_t), VECTOR_MAX_BYTESIZE) / BLOCK_SIZE;
        copy_ubuf_to_gm(qk_gm_addr + cur_qk_offset, qk_out_ub_bf16, 0, 1, offset, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    pipe_barrier(PIPE_ALL);
}

template<typename Dtype, typename CalcDtype>
inline __aicore__ void RunAivSoftmaxLong(__gm__ Dtype *qk, uint32_t contextLen)
{
    if constexpr (std::is_same<Dtype, float16_t>::value) {
        RunAivSoftmaxLongFP16(qk, contextLen);
    } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
        RunAivSoftmaxLongBF16(qk, contextLen);
    }
}
#endif
#endif