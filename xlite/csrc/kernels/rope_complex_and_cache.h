/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#define UBA(T) __ubuf__ T *

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ __inline__ void rope_complex_and_cache(
    uint32_t nTokens, uint32_t nLocalHeads, uint32_t shape1, uint32_t ropeDim, uint32_t offset,
    uint32_t vdim, GM_ADDR input_ptr, GM_ADDR freqs_ptr, GM_ADDR position, uint32_t block_size,
    GM_ADDR vcache, GM_ADDR slot_mapping, int coreOffset = 0, int *nextCoreOffset = nullptr)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    bool need_v_cache = (block_size != 0) && (vcache != nullptr) && (slot_mapping != nullptr);
    if (need_v_cache) {
        assert(nLocalHeads == 1);
    }

    int ropeDtypeBytes = ropeDim * sizeof(Dtype);
    int vCacheBytes = vdim * sizeof(Dtype);
    int ropeFPBytes = ropeDim * sizeof(float);
    int inputBytes = need_v_cache ? vCacheBytes : ropeDtypeBytes;
    int vRemainBytes = need_v_cache ? (vCacheBytes - ropeDtypeBytes) : 0;
    int freqsBytes = ropeFPBytes;
    int outputBytes = ropeDtypeBytes;
    int totalRopeFPBytes = ropeFPBytes * nLocalHeads;
    int totalInputBytes = inputBytes * nLocalHeads;
    int totalVRemainBytes = vRemainBytes * nLocalHeads;
    int totalOutputBytes = outputBytes * nLocalHeads;
    int totalShape1 = shape1 * nLocalHeads;
    int srcStride =
        DIV_ROUND_UP((shape1 - (need_v_cache ? vdim : ropeDim)) * sizeof(Dtype), BLOCK_SIZE);
    int remainStride = DIV_ROUND_UP(ropeDtypeBytes, BLOCK_SIZE);
    int vStride = nLocalHeads * block_size * vdim;
    int input_blocks = DIV_ROUND_UP(inputBytes, BLOCK_SIZE);
    int remain_blocks = DIV_ROUND_UP(vRemainBytes, BLOCK_SIZE);
    int rope_blocks = DIV_ROUND_UP(ropeFPBytes, BLOCK_SIZE);
    int half_rope_blocks = DIV_ROUND_UP(ropeFPBytes / 2, BLOCK_SIZE);
    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    int repeat = DIV_ROUND_UP(ropeDim, calcPad);
    int totalRepeat = DIV_ROUND_UP(ropeDim * nLocalHeads, calcPad);

    int maxCnt = 256;
    uint64_t off = 0;
    UBA(Dtype) input0 = reinterpret_cast<UBA(Dtype)>(off);
    off += ROUND_UP(totalInputBytes, VECTOR_MAX_BYTESIZE);
    UBA(Dtype) input1 = reinterpret_cast<UBA(Dtype)>(off);
    off += ROUND_UP(totalInputBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) freqs0 = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(freqsBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) freqs1 = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(freqsBytes, VECTOR_MAX_BYTESIZE);
    UBA(Dtype) out0 = reinterpret_cast<UBA(Dtype)>(off);
    off += ROUND_UP(totalInputBytes, VECTOR_MAX_BYTESIZE);
    UBA(Dtype) out1 = reinterpret_cast<UBA(Dtype)>(off);
    off += ROUND_UP(totalInputBytes, VECTOR_MAX_BYTESIZE);
    UBA(Dtype) vRemain = reinterpret_cast<UBA(Dtype)>(off);
    off += need_v_cache ? ROUND_UP(totalVRemainBytes, VECTOR_MAX_BYTESIZE) : 0;

    UBA(float) inOutFP32 = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(totalRopeFPBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) x_even = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(totalRopeFPBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) x_odd = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(totalRopeFPBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) cos = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(totalRopeFPBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) sin = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(totalRopeFPBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) x_even_cos = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(totalRopeFPBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) x_odd_sin = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(totalRopeFPBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) x_even_sin = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(totalRopeFPBytes, VECTOR_MAX_BYTESIZE);
    UBA(float) x_odd_cos = reinterpret_cast<UBA(float)>(off);
    off += ROUND_UP(totalRopeFPBytes, VECTOR_MAX_BYTESIZE);
    UBA(uint64_t) positionUB = reinterpret_cast<UBA(uint64_t)>(off);
    off += ROUND_UP((maxCnt) * sizeof(uint64_t), VECTOR_MAX_BYTESIZE);
    UBA(uint32_t) slotMappingUB = reinterpret_cast<UBA(uint32_t)>(off);
    off += need_v_cache ? ROUND_UP((maxCnt) * sizeof(uint32_t), VECTOR_MAX_BYTESIZE) : 0;
    assert(off <= UB_SIZE);

    UBA(Dtype) inputs[2] = {input0, input1};
    UBA(float) freqs[2] = {freqs0, freqs1};
    UBA(Dtype) outs[2] = {out0, out1};

    int taskNum = nTokens;
    if (nextCoreOffset) {
        *nextCoreOffset = (coreOffset + taskNum) % block_num;
    }

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    int curr = 0;
    int baseTokenIdx = -1;
    int first = (block_idx + block_num - coreOffset) % block_num;
    for (uint32_t index = first; index < taskNum; index += block_num) {
        int token_idx = index;
        if (baseTokenIdx == -1 || token_idx >= baseTokenIdx + maxCnt) {
            int remain = maxCnt;
            if (token_idx + maxCnt > nTokens) {
                remain = nTokens - token_idx;
            }
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            CopyGmToUbufAligned(positionUB, ((__gm__ uint64_t *)position) + token_idx,
                                remain * sizeof(uint64_t));
            if (need_v_cache) {
                CopyGmToUbufAligned(slotMappingUB, ((__gm__ uint32_t *)slot_mapping) + token_idx,
                                    remain * sizeof(uint32_t));
            }
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            baseTokenIdx = token_idx;
        }

        auto *input_gm = (__gm__ Dtype *)(input_ptr) + token_idx * totalShape1 + offset;
        auto *output_gm = input_gm;
        auto *freqs_gm =
            (__gm__ float *)(freqs_ptr) + positionUB[token_idx - baseTokenIdx] * ropeDim;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        // input GM -> UB
        copy_gm_to_ubuf(inputs[curr], input_gm, 0, nLocalHeads, input_blocks, srcStride, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);
        // freqs GM -> UB
        CopyGmToUbufAligned(freqs[curr], freqs_gm, freqsBytes);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(inOutFP32, inputs[curr], nLocalHeads, 1, 1, rope_blocks, input_blocks);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(inOutFP32, inputs[curr], nLocalHeads, 1, 1, rope_blocks, input_blocks);
        }
        if (need_v_cache && remain_blocks > 0) {
            copy_ubuf_to_ubuf(vRemain, inputs[curr] + ropeDim, 0, nLocalHeads, remain_blocks,
                              remainStride, 0);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
        SetMask(ropeDim);
        // [cos(k/(theta^(2t/dim))]
        vreducev2((UBA(uint32_t))cos, (UBA(uint32_t))freqs[curr], (UBA(uint32_t))freqs[curr],
                  repeat, 1, 1, 8, 0);
        // [sin(k/(theta^(2t/dim))]
        vreducev2((UBA(uint32_t))sin, (UBA(uint32_t))freqs[curr], (UBA(uint32_t))freqs[curr],
                  repeat, 1, 2, 8, 0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);

        // [x0, x2, x4, ...]
        vreducev2((UBA(uint32_t))x_even, (UBA(uint32_t))inOutFP32, (UBA(uint32_t))inOutFP32,
                  totalRepeat, 1, 1, 8, 0);
        // [x1, x3, x5, ...]
        vreducev2((UBA(uint32_t))x_odd, (UBA(uint32_t))inOutFP32, (UBA(uint32_t))inOutFP32,
                  totalRepeat, 1, 2, 8, 0);
        pipe_barrier(PIPE_V);

        SetMask(ropeDim / 2);
        // x[0::2] * cos
        vmul(x_even_cos, x_even, cos, nLocalHeads, 1, 1, 1, half_rope_blocks, half_rope_blocks, 0);
        // x[1::2] * sin
        vmul(x_odd_sin, x_odd, sin, nLocalHeads, 1, 1, 1, half_rope_blocks, half_rope_blocks, 0);
        // x[0::2] * sin
        vmul(x_even_sin, x_even, sin, nLocalHeads, 1, 1, 1, half_rope_blocks, half_rope_blocks, 0);
        // x[1::2] * cos
        vmul(x_odd_cos, x_odd, cos, nLocalHeads, 1, 1, 1, half_rope_blocks, half_rope_blocks, 0);
        pipe_barrier(PIPE_V);

        // real : x[0::2] * cos - x[1::2] * sin
        vsub(inOutFP32, x_even_cos, x_odd_sin, nLocalHeads, 1, 1, 1, rope_blocks, half_rope_blocks,
             half_rope_blocks);
        // img : x[0::2] * sin + x[1::2] * cos
        vadd(inOutFP32 + ropeDim / 2, x_even_sin, x_odd_cos, nLocalHeads, 1, 1, 1, rope_blocks,
             half_rope_blocks, half_rope_blocks);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
        // out FP32 -> Dtype
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f322f16(outs[curr], inOutFP32, nLocalHeads, 1, 1, input_blocks, rope_blocks);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_f322bf16r(outs[curr], inOutFP32, nLocalHeads, 1, 1, input_blocks, rope_blocks);
        }
        if (need_v_cache && remain_blocks > 0) {
            copy_ubuf_to_ubuf(outs[curr] + ropeDim, vRemain, 0, nLocalHeads, remain_blocks, 0,
                              remainStride);
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
        // out UB -> GM
        if (need_v_cache) {
            uint32_t slot_idx = slotMappingUB[token_idx - baseTokenIdx];
            uint32_t block = slot_idx / block_size;
            uint32_t block_offset = slot_idx % block_size;
            auto *vcache_ptr = ((__gm__ Dtype *)(vcache)) + block * vStride + block_offset * vdim;
            CopyUbufToGmAligned(vcache_ptr, outs[curr], vCacheBytes);
        } else {
            copy_ubuf_to_gm(output_gm, outs[curr], 0, nLocalHeads, input_blocks, 0, srcStride);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
        curr = 1 - curr;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_ALL);
}

#define ROPE_COMPLEX_CACHE_FUNC_DEFINE(dtype)                                                   \
    extern "C" __global__ __aicore__ void rope_complex_and_cache_##dtype(                       \
        uint32_t nTokens, uint32_t nLocalHeads, uint32_t shape1, uint32_t ropeDim,              \
        uint32_t offset, uint32_t vdim, GM_ADDR input_ptr, GM_ADDR freqs_ptr, GM_ADDR position, \
        uint32_t block_size, GM_ADDR vcache, GM_ADDR slot_mapping)                              \
    {                                                                                           \
        rope_complex_and_cache<dtype>(nTokens, nLocalHeads, shape1, ropeDim, offset, vdim,      \
                                      input_ptr, freqs_ptr, position, block_size, vcache,       \
                                      slot_mapping);                                            \
    }
#else
#define ROPE_COMPLEX_CACHE_FUNC_DEFINE(dtype)                                                   \
    extern "C" __global__ __aicore__ void rope_complex_##dtype(                                 \
        uint32_t nTokens, uint32_t nLocalHeads, uint32_t shape1, uint32_t ropeDim,              \
        uint32_t offset, uint32_t vdim, GM_ADDR input_ptr, GM_ADDR freqs_ptr, GM_ADDR position, \
        uint32_t block_size, GM_ADDR vcache, GM_ADDR slot_mapping)                              \
    {                                                                                           \
    }
#endif