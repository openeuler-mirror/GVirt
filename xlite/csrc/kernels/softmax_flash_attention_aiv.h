/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_SOFTMAX_FA_H_
#define _XLITE_SOFTMAX_FA_H_
#include "kernel_operator.h"
#include "kernel_macro.h"
using namespace AscendC;

#if __DAV_C220_VEC__
/*
 * Online Softmax Update Function - For incremental computation of attention softmax
 *
 * Algorithm:
 * When the KV sequence is too long to compute the full softmax at once, chunked processing is required.
 * The online softmax algorithm maintains two statistics (max and sum) for incremental updates:
 *
 * 1. New max: new_max = max(max_prev, max_curr)
 * 2. Scale factors: scale_prev = exp(max_prev - new_max), scale_curr = exp(max_curr - new_max)
 * 3. New sum: new_sum = sum_prev * scale_prev + sum_curr * scale_curr
 * 4. Updated sv: new_sv = sv_prev * scale_prev + sv_curr * scale_curr
 *
 * Parameters:
 *   currSv, currMax, currSum: Current chunk computation results
 *     - currSv = softmax * v_curr
 *     - currMax = local_max_curr
 *     - currSum = sum(exp(x_curr - local_max_curr))
 *   output: Accumulated result output, also serves as previous chunk data for next merge
 *     - First chunk: directly write currSv
 *     - Non-first chunk: write merged sv_prev * scale_prev + sv_curr * scale_curr
 *     - Last chunk: write normalized result (divided by new_sum)
 *   lastMax, lastSum: Global statistics output
 *     - First chunk: directly write currMax, currSum
 *     - Non-first chunk: write merged new_max, new_sum
 *   m: Batch size (number of softmax rows to process)
 *   subHeadOffset: Causal mask offset, starting row offset within query group
 *   nHeads: Number of query heads
 *   nKVHeads: Number of KV heads
 *   headSize: Head dimension
 *   isFirstKvTile: Whether this is the first KV chunk
 *   isLastKvTile: Whether this is the last KV chunk (normalize sv when true)
 */
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxUpdate(__gm__ Dtype *currSv, __gm__ float *currMax, __gm__ float *currSum,
                                           __gm__ Dtype *output, __gm__ float *lastMax, __gm__ float *lastSum,
                                           uint32_t m, uint32_t subHeadOffset, uint32_t nHeads, uint32_t nKVHeads, uint32_t headSize,
                                           int isFirstKvTile, int isLastKvTile)
{
#ifdef XLITE_KERNEL_DEBUG
    printf("RunAivSoftmaxUpdate: m=%u, headSize=%u, isFirstKvTile=%d, isLastKvTile=%d\n", m, headSize, isFirstKvTile, isLastKvTile);
#endif
    uint32_t headNumInGroup = nHeads / nKVHeads;
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    uint32_t repeat = DIV_ROUND_UP(headSize, calcPad);
    uint32_t nBytes = ROUND_UP(headSize * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    uint32_t nFloatBytes = ROUND_UP(headSize * sizeof(float), VECTOR_MAX_BYTESIZE);

    // UB buffer allocation using pointer arrays for Ping-Pong buffering
    uint64_t off = 0;
    __ubuf__ Dtype *svPrevUb[2], *svCurrUb[2];
    __ubuf__ float *svPrevFloatUb[2], *svCurrFloatUb[2];
    __ubuf__ Dtype *svOutUb[2];
    __ubuf__ float *svOutFloatUb[2];
    __ubuf__ float *maxPrevUb[2], *sumPrevUb[2], *maxCurrUb[2], *sumCurrUb[2];
    __ubuf__ float *scalePrevUb[2], *scaleCurrUb[2], *newMaxUb[2], *newSumUb[2], *tempUb[2];

    // SV data buffers (Dtype)
    for (int i = 0; i < 2; ++i) {
        svPrevUb[i] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += nBytes;
        svCurrUb[i] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += nBytes;
    }
    // Statistics buffers (max, sum, scale factors)
    for (int i = 0; i < 2; ++i) {
        maxPrevUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        sumPrevUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        maxCurrUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        sumCurrUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        scalePrevUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        scaleCurrUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        newMaxUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        newSumUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        tempUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
    }
    // Float computation buffers
    for (int i = 0; i < 2; ++i) {
        svPrevFloatUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += nFloatBytes;
        svCurrFloatUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += nFloatBytes;
    }
    // Output buffers
    for (int i = 0; i < 2; ++i) {
        svOutUb[i] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += nBytes;
        svOutFloatUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += nFloatBytes;
    }

    // Pipeline synchronization initialization
    // EVENT_ID0/1: Ping-Pong sync for input buffers (wait for previous computation)
    // EVENT_ID2/3: Ping-Pong sync for output buffers (wait for previous writeback)
    // EVENT_ID4/5: GM read/write sync
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    
    int curr = 0;
    for (uint32_t mIdx = 0; mIdx < m; ++mIdx) {
        uint32_t mOffset = ((mIdx + subHeadOffset) / headNumInGroup) * nHeads;
        uint32_t nOffset = ((mIdx + subHeadOffset) % headNumInGroup);

        if (isFirstKvTile && !isLastKvTile) {
            // First chunk (not last): Copy currSv to output, currMax/currSum to lastMax/lastSum
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(svCurrUb[curr], currSv + mIdx * headSize, 0, 1, headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(svCurrUb[curr], currSv + mIdx * headSize, 0, 1, headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            copy_gm_to_ubuf_align_b16(maxCurrUb[curr], currMax + mIdx, 0, 1, sizeof(float), 0, 0, 0, 0);
            copy_gm_to_ubuf_align_b16(sumCurrUb[curr], currSum + mIdx, 0, 1, sizeof(float), 0, 0, 0, 0);

            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4 + curr);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4 + curr);

            // Write output
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(output + (mOffset + nOffset) * headSize, svCurrUb[curr], 0, 1, headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(output + (mOffset + nOffset) * headSize, svCurrUb[curr], 0, 1, headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            copy_ubuf_to_gm_align_b16(lastMax + mOffset + nOffset, maxCurrUb[curr], 0, 1, sizeof(float), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b16(lastSum + mOffset + nOffset, sumCurrUb[curr], 0, 1, sizeof(float), 0, 0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
        } else if (isFirstKvTile && isLastKvTile) {
            // First and last chunk: Normalize and output directly
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(svCurrUb[curr], currSv + mIdx * headSize, 0, 1, headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(svCurrUb[curr], currSv + mIdx * headSize, 0, 1, headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            copy_gm_to_ubuf_align_b16(maxCurrUb[curr], currMax + mIdx, 0, 1, sizeof(float), 0, 0, 0, 0);
            copy_gm_to_ubuf_align_b16(sumCurrUb[curr], currSum + mIdx, 0, 1, sizeof(float), 0, 0, 0, 0);

            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4 + curr);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);

            // Convert to float32
            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f162f32(svCurrFloatUb[curr], svCurrUb[curr], repeat, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(svCurrFloatUb[curr], svCurrUb[curr], repeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);

            // Broadcast sum for normalization
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            vbrcb((__ubuf__ uint32_t *)newSumUb[curr], (__ubuf__ uint32_t *)sumCurrUb[curr], 0, 0, 1);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

            // Normalize: sv_out = sv_curr / sum
            vdiv(svOutFloatUb[curr], svCurrFloatUb[curr], newSumUb[curr], repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);

            // Convert back to Dtype
            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f322f16r(svOutUb[curr], svOutFloatUb[curr], repeat, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(svOutUb[curr], svOutFloatUb[curr], repeat, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);

            // Write output
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(output + (mOffset + nOffset) * headSize, svOutUb[curr], 0, 1, headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(output + (mOffset + nOffset) * headSize, svOutUb[curr], 0, 1, headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4 + curr);
            copy_ubuf_to_gm_align_b16(lastMax + mOffset + nOffset, maxCurrUb[curr], 0, 1, sizeof(float), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b16(lastSum + mOffset + nOffset, newSumUb[curr], 0, 1, sizeof(float), 0, 0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
        } else {
            // Non-first chunk: Merge with previous results using online softmax
            // Load previous and current statistics
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            copy_gm_to_ubuf_align_b16(maxPrevUb[curr], lastMax + mOffset + nOffset, 0, 1, sizeof(float), 0, 0, 0, 0);
            copy_gm_to_ubuf_align_b16(sumPrevUb[curr], lastSum + mOffset + nOffset, 0, 1, sizeof(float), 0, 0, 0, 0);
            copy_gm_to_ubuf_align_b16(maxCurrUb[curr], currMax + mIdx, 0, 1, sizeof(float), 0, 0, 0, 0);
            copy_gm_to_ubuf_align_b16(sumCurrUb[curr], currSum + mIdx, 0, 1, sizeof(float), 0, 0, 0, 0);
            // Load previous and current sv
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(svPrevUb[curr], output + (mOffset + nOffset) * headSize, 0, 1, headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
                copy_gm_to_ubuf(svCurrUb[curr], currSv + mIdx * headSize, 0, 1, headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(svPrevUb[curr], output + (mOffset + nOffset) * headSize, 0, 1, headSize * sizeof(Dtype), 0, 0, 0, 0);
                copy_gm_to_ubuf_align_b16(svCurrUb[curr], currSv + mIdx * headSize, 0, 1, headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);

            // Compute new_max = max(max_prev, max_curr)
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6 + curr);
            vmax(newMaxUb[curr], maxPrevUb[curr], maxCurrUb[curr], 1, 1, 1, 1, 8, 8, 1);
            pipe_barrier(PIPE_V);

            // Compute scale_prev = exp(max_prev - new_max)
            vsub(scalePrevUb[curr], maxPrevUb[curr], newMaxUb[curr], 1, 1, 1, 1, 8, 8, 1);
            pipe_barrier(PIPE_V);
            vexp(scalePrevUb[curr], scalePrevUb[curr], 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // Compute scale_curr = exp(max_curr - new_max)
            vsub(scaleCurrUb[curr], maxCurrUb[curr], newMaxUb[curr], 1, 1, 1, 1, 8, 8, 1);
            pipe_barrier(PIPE_V);
            vexp(scaleCurrUb[curr], scaleCurrUb[curr], 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // Compute new_sum = sum_prev * scale_prev + sum_curr * scale_curr
            vmul(tempUb[curr], sumPrevUb[curr], scalePrevUb[curr], 1, 1, 1, 1, 8, 8, 1);
            pipe_barrier(PIPE_V);
            vmul(newSumUb[curr], sumCurrUb[curr], scaleCurrUb[curr], 1, 1, 1, 1, 8, 8, 1);
            pipe_barrier(PIPE_V);
            vadd(newSumUb[curr], newSumUb[curr], tempUb[curr], 1, 1, 1, 1, 8, 8, 1);
            pipe_barrier(PIPE_V);

            // Convert sv to float32
            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f162f32(svPrevFloatUb[curr], svPrevUb[curr], repeat, 1, 1, 8, 4);
                vconv_f162f32(svCurrFloatUb[curr], svCurrUb[curr], repeat, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(svPrevFloatUb[curr], svPrevUb[curr], repeat, 1, 1, 8, 4);
                vconv_bf162f32(svCurrFloatUb[curr], svCurrUb[curr], repeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

            // Broadcast scale factors for vector multiplication
            vbrcb((__ubuf__ uint32_t *)scalePrevUb[curr], (__ubuf__ uint32_t *)scalePrevUb[curr], 0, 0, 1);
            vbrcb((__ubuf__ uint32_t *)scaleCurrUb[curr], (__ubuf__ uint32_t *)scaleCurrUb[curr], 0, 0, 1);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            // Apply scale factors: sv_prev *= scale_prev, sv_curr *= scale_curr
            vmul(svPrevFloatUb[curr], svPrevFloatUb[curr], scalePrevUb[curr], repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            vmul(svCurrFloatUb[curr], svCurrFloatUb[curr], scaleCurrUb[curr], repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);

            // Merge: sv_out = sv_prev + sv_curr
            vadd(svOutFloatUb[curr], svPrevFloatUb[curr], svCurrFloatUb[curr], repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            // Normalize if last chunk: sv_out /= new_sum
            if (isLastKvTile) {
                vbrcb((__ubuf__ uint32_t *)newSumUb[curr], (__ubuf__ uint32_t *)newSumUb[curr], 0, 0, 1);
                pipe_barrier(PIPE_V);
                vdiv(svOutFloatUb[curr], svOutFloatUb[curr], newSumUb[curr], repeat, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
            }

            // Convert back to Dtype
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f322f16r(svOutUb[curr], svOutFloatUb[curr], repeat, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(svOutUb[curr], svOutFloatUb[curr], repeat, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);

            // Write output
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(output + (mOffset + nOffset) * headSize, svOutUb[curr], 0, 1, headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(output + (mOffset + nOffset) * headSize, svOutUb[curr], 0, 1, headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);

            // Write updated statistics
            copy_ubuf_to_gm_align_b16(lastMax + mOffset + nOffset, newMaxUb[curr], 0, 1, sizeof(float), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b16(lastSum + mOffset + nOffset, newSumUb[curr], 0, 1, sizeof(float), 0, 0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6 + curr);
        }

        // Switch Ping-Pong buffer
        curr = 1 - curr;
    }

    // Wait for all pipeline operations to complete
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
}

#else
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxUpdate(__gm__ Dtype *currSv, __gm__ float *currMax, __gm__ float *currSum,
                                           __gm__ Dtype *output, __gm__ float *lastMax, __gm__ float *lastSum,
                                           uint32_t m, uint32_t subHeadOffset, uint32_t nHeads, uint32_t nKVHeads, uint32_t headSize,
                                           int isFirstKvTile, int isLastKvTile)
{
}
#endif
#endif
