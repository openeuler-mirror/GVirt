/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_SOFTMAX_ATTN_AIV_H_
#define _XLITE_SOFTMAX_ATTN_AIV_H_

#include "kernel_operator.h"
#include "kernel_macro.h"

// #define XLITE_KERNEL_DEBUG
#include "debug.h"

using namespace AscendC;

#if __DAV_C220_VEC__
/* ROUND_UP(n * sizeof(Dtype), VECTOR_MAX_BYTESIZE) * 4 +
 *     ROUND_UP(n * sizeof(float), VECTOR_MAX_BYTESIZE) +
 *     (DIV_ROUND_UP(n * sizeof(float), VECTOR_MAX_BYTESIZE) / 2) * VECTOR_MAX_BYTESIZE <= ub
 * size(196608B) float16_t or bfloat16_t: n <= 13952
 */
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxPingPong(__gm__ Dtype *buf, uint32_t m, uint32_t n, int calcLen,
                                             uint32_t outN = 0, bool seqHead = true,
                                             uint32_t maskOff = 0, uint32_t maskStride = 1,
                                             __gm__ float *maxBuf = nullptr,
                                             __gm__ float *sumBuf = nullptr, bool hasScale = false,
                                             float scale = 1.0f)
{
    if (outN == 0 || outN > n) {
        outN = n;
    }

    bool saveMaxSum = (maxBuf != nullptr && sumBuf != nullptr);

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint64_t off = 0;
    __ubuf__ Dtype *in1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *in2 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *in[PINGPONG_BUF_NUM] = {in1, in2};
    __ubuf__ Dtype *out1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *out2 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *out[PINGPONG_BUF_NUM] = {out1, out2};
    __ubuf__ float *cal = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(float), VECTOR_MAX_BYTESIZE);
    __ubuf__ float *temp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += (DIV_ROUND_UP(outN * sizeof(float), VECTOR_MAX_BYTESIZE) / 2) * VECTOR_MAX_BYTESIZE;
    __ubuf__ float *maxOut0 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ float *maxOut1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ float *maxOut[PINGPONG_BUF_NUM] = {maxOut0, maxOut1};
    __ubuf__ float *sumOut0 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ float *sumOut1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ float *sumOut[PINGPONG_BUF_NUM] = {sumOut0, sumOut1};

    constexpr int calPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
    int curr = 0;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    for (int idx = 0; idx < m; ++idx) {
        int actualCalcLen = seqHead ? calcLen + (idx + maskOff) / maskStride :
                                calcLen + (idx + maskOff) % maskStride;  // 每一行开始mask的位置
        if (actualCalcLen <= 0) {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            vector_dup(out[curr], Dtype(0), DIV_ROUND_UP(outN, pad), 1, 1, 8, 0);
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            if ((outN * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(buf + idx * n, out[curr], 0, 1, outN * sizeof(Dtype) / BLOCK_SIZE,
                                0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(buf + idx * n, out[curr], 0, 1, outN * sizeof(Dtype), 0,
                                          0, 0, 0);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        } else {
            if (actualCalcLen > outN) {
                actualCalcLen = outN;
            }
            int padCachedTokens = ROUND_UP(actualCalcLen, calPad);
            int repeat = padCachedTokens / calPad;

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);
            if (actualCalcLen * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(in[curr], buf + idx * n, 0, 1,
                                actualCalcLen * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(in[curr], buf + idx * n, 0, 1,
                                          actualCalcLen * sizeof(Dtype), 0, 0, 0, 0);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f162f32(cal, in[curr], repeat, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(cal, in[curr], repeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);

            if (hasScale) {
                vmuls(cal, cal, scale, repeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }

            // max
            ReduceMaxV2(temp, cal, actualCalcLen);
            pipe_barrier(PIPE_V);

            // broadcast一个max标量为一个block大小的向量，避免使用scalar运算
            vbrcb((__ubuf__ uint32_t *)temp, (__ubuf__ uint32_t *)temp, 0, 0, 1);
            pipe_barrier(PIPE_V);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            if (saveMaxSum) {
                copy_ubuf_to_ubuf(maxOut[curr], temp, 0, 1, 1, 1, 1);
                pipe_barrier(PIPE_V);
            }

            // QK - max
            vsub(cal, cal, temp, repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);

            // EXP = exp(QK-max)
            vexp(cal, cal, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // s = Reduce_sum(EXP)
            ReduceSumV2(temp, cal, actualCalcLen);
            pipe_barrier(PIPE_V);

            if (saveMaxSum) {
                copy_ubuf_to_ubuf(sumOut[curr], temp, 0, 1, 1, 1, 1);
                pipe_barrier(PIPE_V);
            }

            // broadcast一个s = Reduce_sum(EXP)标量为一个block大小的向量，避免使用scalar运算
            vbrcb((__ubuf__ uint32_t *)temp, (__ubuf__ uint32_t *)temp, 0, 0, 1);
            pipe_barrier(PIPE_V);

            vdiv(cal, cal, temp, repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);

            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f322f16r(out[curr], cal, repeat, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(out[curr], cal, repeat, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);

            if (outN > actualCalcLen) {
                if (actualCalcLen % pad != 0) {
                    SetMaskFromHighBit(pad, pad - actualCalcLen % pad);
                    vector_dup(out[curr] + ROUND_DOWN(actualCalcLen, pad), Dtype(0), 1, 1, 1, 8, 0);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                int last = ROUND_UP(actualCalcLen, pad);
                if (outN > last) {
                    vector_dup(out[curr] + last, Dtype(0), DIV_ROUND_UP(outN - last, pad), 1, 1, 8,
                               0);
                }
                pipe_barrier(PIPE_V);
            }

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            if ((outN * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(buf + idx * n, out[curr], 0, 1, outN * sizeof(Dtype) / BLOCK_SIZE,
                                0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(buf + idx * n, out[curr], 0, 1, outN * sizeof(Dtype), 0,
                                          0, 0, 0);
            }
            if (saveMaxSum) {
                copy_ubuf_to_gm_align_b16(maxBuf + idx, maxOut[curr], 0, 1, sizeof(float), 0, 0, 0,
                                          0);
                copy_ubuf_to_gm_align_b16(sumBuf + idx, sumOut[curr], 0, 1, sizeof(float), 0, 0, 0,
                                          0);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        }
        curr = 1 - curr;
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
}

/*
 * Online Softmax Update Function - Incremental computation for Flash Attention
 *
 * Background:
 * In Flash Attention, when the KV sequence length exceeds the capacity for single-pass softmax
 * computation, the sequence must be processed in chunks. This function implements the online
 * softmax algorithm to incrementally merge results from multiple KV chunks.
 *
 * Online Softmax Algorithm:
 * The algorithm maintains running statistics (max and sum) across chunks to enable numerically
 * stable incremental updates. For each new chunk:
 *
 *   1. new_max = max(max_prev, max_curr)
 *   2. scale_prev = exp(max_prev - new_max)
 *   3. scale_curr = exp(max_curr - new_max)
 *   4. new_sum = sum_prev * scale_prev + sum_curr * scale_curr
 *   5. new_sv = sv_prev * scale_prev + sv_curr * scale_curr
 *
 * When processing the final chunk, the accumulated result is normalized by dividing by new_sum.
 *
 * Parameters:
 *   @param currSv      GM pointer to current chunk's softmax-weighted values (softmax * v_curr)
 *   @param currMax     GM pointer to current chunk's local maximum values
 *   @param currSum     GM pointer to current chunk's sum of exponentials
 *                      sum(exp(x_curr - local_max_curr))
 *   @param output      GM pointer to accumulated output (also serves as sv_prev for next merge)
 *   @param lastMax     GM pointer to global maximum statistics across all processed chunks
 *   @param lastSum     GM pointer to global sum statistics across all processed chunks
 *   @param m           Number of query rows to process in this batch
 *   @param nHeads      Number of query attention heads
 *   @param headSize    Dimension of each attention head
 *   @param isFirstKvTile Flag indicating if this is the first KV chunk
 *   @param actualCalcSoftmaxLen Base length for softmax computation (adjusted by causal mask)
 *   @param seqHead     currSv/currMax/currSum format; true: (seq, head); false: (head, seq)
 *   @param maskOff     Offset for causal mask calculation
 *   @param maskStride  Stride for causal mask calculation
 *
 * Implementation Details:
 *   - Uses Ping-Pong buffering (double buffering) for pipeline parallelism
 *   - Allocates UB buffers for: sv_prev, sv_curr, sv_out, max/sum statistics, scale factors
 *   - Employs event-based synchronization between pipeline stages (MTE2, V, MTE3)
 *   - Converts Dtype (half/bfloat16) to float32 for numerical precision in computation
 *   - Supports both aligned and unaligned memory access patterns
 *
 * Pipeline Stages:
 *   - MTE2: Memory transfer from GM to UB (load data)
 *   - V: Vector computation (softmax update operations)
 *   - MTE3: Memory transfer from UB to GM (store results)
 *
 * Event IDs Usage:
 *   - EVENT_ID0/1: Ping-Pong sync for input buffer loads
 *   - EVENT_ID2/3: Ping-Pong sync for output buffer stores
 *   - EVENT_ID4/5: GM read/write synchronization for first chunk path
 */
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxUpdate(__gm__ Dtype *currSv, __gm__ float *currMax,
                                           __gm__ float *currSum, __gm__ Dtype *output,
                                           __gm__ float *lastMax, __gm__ float *lastSum, uint32_t m,
                                           uint32_t nHeads, uint32_t headSize, int isFirstKvTile,
                                           int actualCalcSoftmaxLen, bool seqHead, uint32_t maskOff,
                                           uint32_t maskStride)
{
    dbg_printf(
        "RunAivSoftmaxUpdate: m=%u, headSize=%u, isFirstKvTile=%d, actualCalcSoftmaxLen=%d\n", m,
        headSize, isFirstKvTile, actualCalcSoftmaxLen);
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    uint32_t repeat = DIV_ROUND_UP(headSize, calcPad);
    uint32_t nBytes = ROUND_UP(headSize * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    uint32_t nFloatBytes = ROUND_UP(headSize * sizeof(float), VECTOR_MAX_BYTESIZE);

    // UB buffer allocation for Ping-Pong buffering strategy
    // Ping-Pong buffering enables overlapping of computation and memory transfers
    // by using two sets of buffers that alternate between processing and I/O
    uint64_t off = 0;
    __ubuf__ Dtype *svPrevUb[2], *svCurrUb[2], *svOutUb[2];
    __ubuf__ float *newSumOutUb[2], *newMaxOutUb[2];
    __ubuf__ float *maxPrevUb[2], *sumPrevUb[2], *maxCurrUb[2], *sumCurrUb[2];
    __ubuf__ float *scalePrevUb, *scaleCurrUb, *tempUb, *svPrevFloatUb, *svCurrFloatUb,
        *svOutFloatUb, *newSumUb, *newMaxUb, *sumPrevCalUb, *sumCurrCalUb;

    // Allocate SV (softmax-weighted value) data buffers in Dtype precision
    // Each Ping-Pong set contains: previous sv, current sv, and output sv
    for (int i = 0; i < 2; ++i) {
        svPrevUb[i] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += nBytes;
        svCurrUb[i] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += nBytes;
        svOutUb[i] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += nBytes;
    }
    // Allocate statistics buffers: max values, sum values, scale factors
    // Each buffer is one VECTOR_MAX_BYTESIZE to hold a single scalar broadcast value
    for (int i = 0; i < 2; ++i) {
        maxPrevUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        sumPrevUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        maxCurrUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        sumCurrUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        newSumOutUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        newMaxOutUb[i] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
    }
    // Single buffers for scale factors and temporary computation
    scalePrevUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    scaleCurrUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    tempUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    newSumUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    newMaxUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    sumPrevCalUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    sumCurrCalUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    // Float32 buffers for high-precision computation of sv values
    svPrevFloatUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += nFloatBytes;
    svCurrFloatUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += nFloatBytes;
    svOutFloatUb = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += nFloatBytes;

    // Pipeline synchronization initialization using event flags
    // These flags coordinate between different pipeline stages to ensure correct execution order
    // EVENT_ID0/1: Ping-Pong sync for input buffer loads (MTE2 waits for V to finish)
    // EVENT_ID2/3: Ping-Pong sync for output buffer stores (V waits for MTE3 to finish)
    // EVENT_ID4/5: GM read/write synchronization for first chunk fast path
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);

    int curr = 0;
    for (uint32_t mIdx = 0; mIdx < m; ++mIdx) {
        // Calculate memory offsets for Grouped Query Attention (GQA)
        // seqIdx: seq Offset in output tensor
        // headIdx: head Offset in output tensor
        uint32_t seqIdx = seqHead ? (mIdx + maskOff) / maskStride : (mIdx + maskOff) % maskStride;
        uint32_t headIdx = seqHead ? (mIdx + maskOff) % maskStride : (mIdx + maskOff) / maskStride;
        uint64_t offset = seqIdx * nHeads + headIdx;
        // Adjust actual calculation length based on causal mask position
        int actualCalcLen = actualCalcSoftmaxLen + seqIdx;
        if (actualCalcLen <= 0) {
            continue;
        }
        if (isFirstKvTile) {
            // First chunk fast path: directly copy current results to output
            // No merge needed since there's no previous data to combine with
            // Load current chunk data from GM to UB
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(svCurrUb[curr], currSv + mIdx * headSize, 0, 1,
                                headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(svCurrUb[curr], currSv + mIdx * headSize, 0, 1,
                                          headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            copy_gm_to_ubuf_align_b16(maxCurrUb[curr], currMax + mIdx, 0, 1, sizeof(float), 0, 0, 0,
                                      0);
            copy_gm_to_ubuf_align_b16(sumCurrUb[curr], currSum + mIdx, 0, 1, sizeof(float), 0, 0, 0,
                                      0);

            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4 + curr);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4 + curr);

            // Store current results directly to output (first chunk initialization)
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(output + offset * headSize, svCurrUb[curr], 0, 1,
                                headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(output + offset * headSize, svCurrUb[curr], 0, 1,
                                          headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            copy_ubuf_to_gm_align_b16(lastMax + offset, maxCurrUb[curr], 0, 1, sizeof(float), 0, 0,
                                      0, 0);
            copy_ubuf_to_gm_align_b16(lastSum + offset, sumCurrUb[curr], 0, 1, sizeof(float), 0, 0,
                                      0, 0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
        } else {
            // Non-first chunk: merge with previous results using online softmax algorithm
            // Step 1: Load previous and current statistics from GM
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            copy_gm_to_ubuf_align_b16(maxPrevUb[curr], lastMax + offset, 0, 1, sizeof(float), 0, 0,
                                      0, 0);
            copy_gm_to_ubuf_align_b16(sumPrevUb[curr], lastSum + offset, 0, 1, sizeof(float), 0, 0,
                                      0, 0);
            copy_gm_to_ubuf_align_b16(maxCurrUb[curr], currMax + mIdx, 0, 1, sizeof(float), 0, 0, 0,
                                      0);
            copy_gm_to_ubuf_align_b16(sumCurrUb[curr], currSum + mIdx, 0, 1, sizeof(float), 0, 0, 0,
                                      0);
            // Step 2: Load previous and current sv values
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(svPrevUb[curr], output + offset * headSize, 0, 1,
                                headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
                copy_gm_to_ubuf(svCurrUb[curr], currSv + mIdx * headSize, 0, 1,
                                headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(svPrevUb[curr], output + offset * headSize, 0, 1,
                                          headSize * sizeof(Dtype), 0, 0, 0, 0);
                copy_gm_to_ubuf_align_b16(svCurrUb[curr], currSv + mIdx * headSize, 0, 1,
                                          headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);

            // Step 3: Compute new_max = max(max_prev, max_curr)
            vmax(newMaxUb, maxPrevUb[curr], maxCurrUb[curr], 1, 1, 1, 1, 8, 8, 1);
            pipe_barrier(PIPE_V);

            // Step 4: Compute scale_prev = exp(max_prev - new_max)
            // Step 5: Compute scale_curr = exp(max_curr - new_max)
            vsub(scalePrevUb, maxPrevUb[curr], newMaxUb, 1, 1, 1, 1, 8, 8, 1);
            vsub(scaleCurrUb, maxCurrUb[curr], newMaxUb, 1, 1, 1, 1, 8, 8, 1);
            pipe_barrier(PIPE_V);
            vexp(scalePrevUb, scalePrevUb, 1, 1, 1, 8, 8);
            vexp(scaleCurrUb, scaleCurrUb, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // Step 6: Compute new_sum = sum_prev * scale_prev + sum_curr * scale_curr
            vmul(tempUb, sumPrevUb[curr], scalePrevUb, 1, 1, 1, 1, 8, 8, 1);
            vmul(newSumUb, sumCurrUb[curr], scaleCurrUb, 1, 1, 1, 1, 8, 8, 1);
            vbrcb((__ubuf__ uint32_t *)sumPrevCalUb, (__ubuf__ uint32_t *)sumPrevUb[curr], 0, 0, 1);
            vbrcb((__ubuf__ uint32_t *)sumCurrCalUb, (__ubuf__ uint32_t *)sumCurrUb[curr], 0, 0, 1);
            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f162f32(svPrevFloatUb, svPrevUb[curr], repeat, 1, 1, 8, 4);
                vconv_f162f32(svCurrFloatUb, svCurrUb[curr], repeat, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(svPrevFloatUb, svPrevUb[curr], repeat, 1, 1, 8, 4);
                vconv_bf162f32(svCurrFloatUb, svCurrUb[curr], repeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

            vadd(newSumUb, newSumUb, tempUb, 1, 1, 1, 1, 8, 8, 1);
            // sv_prev = sv_prev * sum_prev, sv_curr *= sum_curr
            vbrcb((__ubuf__ uint32_t *)scalePrevUb, (__ubuf__ uint32_t *)scalePrevUb, 0, 0, 1);
            vbrcb((__ubuf__ uint32_t *)scaleCurrUb, (__ubuf__ uint32_t *)scaleCurrUb, 0, 0, 1);
            pipe_barrier(PIPE_V);
            vmul(svPrevFloatUb, svPrevFloatUb, sumPrevCalUb, repeat, 1, 1, 0, 8, 8, 0);
            vmul(svCurrFloatUb, svCurrFloatUb, sumCurrCalUb, repeat, 1, 1, 0, 8, 8, 0);
            vbrcb((__ubuf__ uint32_t *)newSumUb, (__ubuf__ uint32_t *)newSumUb, 0, 0, 1);
            pipe_barrier(PIPE_V);

            // Step 8: Apply scale factors to sv values
            // sv_prev *= scale_prev, sv_curr *= scale_curr
            vmul(svPrevFloatUb, svPrevFloatUb, scalePrevUb, repeat, 1, 1, 0, 8, 8, 0);
            vmul(svCurrFloatUb, svCurrFloatUb, scaleCurrUb, repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);

            // Step 9: Merge scaled values: sv_out = sv_prev + sv_curr
            vadd(svOutFloatUb, svPrevFloatUb, svCurrFloatUb, repeat, 1, 1, 1, 8, 8, 8);

            // Step 10: Normalize by new_sum (always performed for numerical stability)
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            copy_ubuf_to_ubuf(newMaxOutUb[curr], newMaxUb, 0, 1, 1, 1, 1);
            vbrcb((__ubuf__ uint32_t *)newSumOutUb[curr], (__ubuf__ uint32_t *)newSumUb, 0, 0, 1);
            pipe_barrier(PIPE_V);
            vdiv(svOutFloatUb, svOutFloatUb, newSumOutUb[curr], repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);

            // Step 11: Convert result back from float32 to Dtype
            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f322f16r(svOutUb[curr], svOutFloatUb, repeat, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(svOutUb[curr], svOutFloatUb, repeat, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);

            // Step 12: Write merged output to GM
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(output + offset * headSize, svOutUb[curr], 0, 1,
                                headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(output + offset * headSize, svOutUb[curr], 0, 1,
                                          headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            copy_ubuf_to_gm_align_b16(lastMax + offset, newMaxOutUb[curr], 0, 1, sizeof(float), 0,
                                      0, 0, 0);
            copy_ubuf_to_gm_align_b16(lastSum + offset, newSumOutUb[curr], 0, 1, sizeof(float), 0,
                                      0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        }

        // Toggle Ping-Pong buffer index for next iteration
        curr = 1 - curr;
    }

    // Final synchronization: wait for all pending pipeline operations to complete
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
}

/*
 * (VECTOR_MAX_REPEAT - 1) * VECTOR_MAX_BYTESIZE * 2 + VECTOR_MAX_BYTESIZE *
 * DIV_ROUND_UP(VECTOR_MAX_REPEAT - 1, calcPad) + VECTOR_MAX_BYTESIZE * (subBlockNum + 1) * 2 < UB
 * SIZE(196608) float16_t or bfloat16_t: subBlockNum <= 127 float16_t or bfloat16_t: n <= 2064512
 */
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxLong(__gm__ Dtype *buf, __gm__ float *expBuf, int32_t m,
                                         uint32_t n, uint32_t calcLen, uint32_t outN = 0,
                                         bool seqHead = true, uint32_t maskOff = 0, uint32_t maskStride = 1,
                                         bool hasScale = false, float scale = 1.0f)
{
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    float min = -3.4028235e+38;
    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);

    if (outN == 0 || outN > n) {
        outN = n;
    }

    // VECTOR_MAX_REPEAT - 1 for Dtype size 2, float size 4
    uint32_t maxRepeat = VECTOR_MAX_REPEAT - 1;
    const uint32_t MAX_SUB_CONTEXT_SIZE = maxRepeat * calcPad;
    uint32_t totalSubBlockNum = DIV_ROUND_UP(outN, MAX_SUB_CONTEXT_SIZE);

    uint32_t off = 0;
    auto *in = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += MAX_SUB_CONTEXT_SIZE * sizeof(Dtype);
    auto *out = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += MAX_SUB_CONTEXT_SIZE * sizeof(Dtype);
    __ubuf__ float *calc = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += MAX_SUB_CONTEXT_SIZE * sizeof(float);
    __ubuf__ float *calcDst = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    int useV2 = 0;
    if (totalSubBlockNum <= 65) {
        useV2 = 1;
        off += (VECTOR_MAX_REPEAT - 1) / 2 * VECTOR_MAX_BYTESIZE;
        ;
    } else {
        off += DIV_ROUND_UP(VECTOR_MAX_REPEAT - 1, calcPad) * VECTOR_MAX_BYTESIZE;
    }
    auto *max = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += (totalSubBlockNum + 1) * VECTOR_MAX_BYTESIZE;
    auto *sum = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += (totalSubBlockNum + 1) * VECTOR_MAX_BYTESIZE;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // for MTE2 in
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);  // for MTE2 calc
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);  // for V calc
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);  // for V out
    for (int idx = 0; idx < m; idx++) {
        int actualCalcLen = seqHead ? calcLen + (idx + maskOff) / maskStride :
                                calcLen + (idx + maskOff) % maskStride;  // 每一行开始mask的位置
        if (actualCalcLen > outN) {
            actualCalcLen = outN;
        }
        uint64_t calcOutLen = ROUND_UP(actualCalcLen, MAX_SUB_CONTEXT_SIZE);
        if (calcOutLen > outN) {
            calcOutLen = outN;
        }

        // stage1: max(x) & sum(exp(x - max(x)))
        uint32_t subBlockNum = DIV_ROUND_UP(actualCalcLen, MAX_SUB_CONTEXT_SIZE);
        auto *totalMax = max + subBlockNum * calcPad;
        auto *totalSum = sum + subBlockNum * calcPad;
        __gm__ Dtype *curBuf = buf + idx * n;
        bool maxFlag[128] = {false};
        int block;
        for (block = 0; block < subBlockNum; block++) {
            uint32_t offset = block * MAX_SUB_CONTEXT_SIZE;
            uint32_t curLen = MAX_SUB_CONTEXT_SIZE;
            uint32_t curRepeat = maxRepeat;
            if (block == subBlockNum - 1) {
                curLen = actualCalcLen - offset;
                curRepeat = DIV_ROUND_UP(curLen, calcPad);
            }
            float curMaxVal, totalMaxVal;

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            if (curLen * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(in, curBuf + offset, 0, 1, curLen * sizeof(Dtype) / BLOCK_SIZE, 0,
                                0);
            } else {
                copy_gm_to_ubuf_align_b16(in, curBuf + offset, 0, 1, curLen * sizeof(Dtype), 0, 0,
                                          0, 0);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                vconv_bf162f32(calc, in, curRepeat, 1, 1, 8, 4);
            } else {
                vconv_f162f32(calc, in, curRepeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

            if (hasScale) {
                vmuls(calc, calc, scale, curRepeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }

            if (useV2) {
                ReduceMaxV2(calcDst, calc, curLen);
                pipe_barrier(PIPE_V);
            } else {
                ReduceMax(calcDst, calc, curLen);
            }

            vbrcb((__ubuf__ uint32_t *)calcDst, (__ubuf__ uint32_t *)calcDst, 0, 0, 1);

            if (subBlockNum > 1 && block == 0) {
                vector_dup(max, min, subBlockNum + 1, 1, 1, 8, 0);
                vector_dup(sum, float(0), subBlockNum + 1, 1, 1, 8, 0);
            }
            pipe_barrier(PIPE_V);

            if (subBlockNum > 1) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID2);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
                curMaxVal = *calcDst;
                totalMaxVal = *totalMax;
                auto *curMax = max + block * calcPad;
                if (curMaxVal > totalMaxVal) {
                    for (int i = 0; i < block; i++) {
                        maxFlag[i] = 1;
                    }
                    copy_ubuf_to_ubuf(curMax, calcDst, 0, 1, 8, 1, 1);
                    copy_ubuf_to_ubuf(totalMax, calcDst, 0, 1, 8, 1, 1);
                } else {
                    copy_ubuf_to_ubuf(curMax, totalMax, 0, 1, 8, 1, 1);
                    copy_ubuf_to_ubuf(calcDst, totalMax, 0, 1, 8, 1, 1);
                    pipe_barrier(PIPE_V);
                }
            }

            vsub(calc, calc, calcDst, curRepeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            vexp(calc, calc, curRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            if (subBlockNum > 1) {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            }

            if (useV2) {
                ReduceSumV2(calcDst, calc, curLen);
                pipe_barrier(PIPE_V);
            } else {
                ReduceSum(calcDst, calc, curLen);
            }

            vbrcb((__ubuf__ uint32_t *)calcDst, (__ubuf__ uint32_t *)calcDst, 0, 0, 1);
            pipe_barrier(PIPE_V);

            if (subBlockNum > 1) {
                if (block > 0) {
                    auto *lastMax = max + (block - 1) * calcPad;
                    auto *lastSum = sum + (block - 1) * calcPad;
                    if (curMaxVal > totalMaxVal) {
                        auto *calcTemp = calcDst + calcPad;
                        vsub(calcTemp, lastMax, totalMax, 1, 1, 1, 1, 8, 8, 1);
                        pipe_barrier(PIPE_V);
                        vexp(calcTemp, calcTemp, 1, 1, 1, 8, 8);
                        pipe_barrier(PIPE_V);
                        vmul(calcTemp, lastSum, calcTemp, 1, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                        vadd(calcDst, calcDst, calcTemp, 1, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                    } else {
                        vadd(calcDst, calcDst, lastSum, 1, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                    }
                }
                auto *curSum = sum + block * calcPad;
                copy_ubuf_to_ubuf(curSum, calcDst, 0, 1, 8, 1, 1);

                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
                if (curLen * sizeof(float) % BLOCK_SIZE == 0) {
                    copy_ubuf_to_gm(expBuf + offset, calc, 0, 1,
                                    curLen * sizeof(float) / BLOCK_SIZE, 0, 0);
                } else {
                    copy_ubuf_to_gm_align_b16(expBuf + offset, calc, 0, 1, curLen * sizeof(float),
                                              0, 0, 0, 0);
                }
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            copy_ubuf_to_ubuf(totalSum, calcDst, 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);
        }

        if (subBlockNum > 1) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        }

        // stage2: exp(x - max(x)) / sum(exp(x - max(x)))
        for (block = 0; block < subBlockNum; block++) {
            uint32_t offset = block * MAX_SUB_CONTEXT_SIZE;
            uint32_t curLen = MAX_SUB_CONTEXT_SIZE;
            uint32_t curRepeat = maxRepeat;
            if (block == subBlockNum - 1) {
                curLen = actualCalcLen - offset;
                curRepeat = DIV_ROUND_UP(curLen, calcPad);
            }

            if (subBlockNum > 1) {
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                if (curLen * sizeof(float) % BLOCK_SIZE == 0) {
                    copy_gm_to_ubuf(calc, expBuf + offset, 0, 1,
                                    curLen * sizeof(float) / BLOCK_SIZE, 0, 0);
                } else {
                    copy_gm_to_ubuf_align_b16(calc, expBuf + offset, 0, 1, curLen * sizeof(float),
                                              0, 0, 0, 0);
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

                if (maxFlag[block]) {
                    auto *calcTemp = calcDst + calcPad;
                    auto *curMax = max + block * calcPad;
                    vsub(calcTemp, curMax, totalMax, 1, 1, 1, 1, 8, 8, 1);
                    pipe_barrier(PIPE_V);
                    vexp(calcTemp, calcTemp, 1, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    vmul(calc, calc, calcTemp, curRepeat, 1, 1, 0, 8, 8, 0);
                    pipe_barrier(PIPE_V);
                }
            }

            vdiv(calc, calc, totalSum, curRepeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                vconv_f322bf16r(out, calc, curRepeat, 1, 1, 4, 8);
            } else {
                vconv_f322f16r(out, calc, curRepeat, 1, 1, 4, 8);
            }
            if (subBlockNum > 1) {
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            }
            pipe_barrier(PIPE_V);

            if (block == subBlockNum - 1 && calcOutLen > actualCalcLen) {
                if (curLen % pad != 0) {
                    SetMaskFromHighBit(pad, pad - curLen % pad);
                    vector_dup(out + ROUND_DOWN(curLen, pad), Dtype(0), 1, 1, 1, 8, 0);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                int last = ROUND_UP(curLen, pad);
                if (calcOutLen - offset > last) {
                    vector_dup(out + last, Dtype(0), DIV_ROUND_UP(calcOutLen - offset - last, pad),
                               1, 1, 8, 0);
                }
                pipe_barrier(PIPE_V);
                curLen = calcOutLen - offset;
            }

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            if ((curLen * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype) / BLOCK_SIZE, 0,
                                0);
            } else {
                copy_ubuf_to_gm_align_b16(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype), 0, 0,
                                          0, 0);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        if (block < totalSubBlockNum) {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            vector_dup(out, Dtype(0), DIV_ROUND_UP(MAX_SUB_CONTEXT_SIZE, pad), 1, 1, 8, 0);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        }

        for (; block < totalSubBlockNum; block++) {
            uint32_t offset = block * MAX_SUB_CONTEXT_SIZE;
            uint32_t curLen = MAX_SUB_CONTEXT_SIZE;
            if (block == totalSubBlockNum - 1) {
                curLen = outN - offset;
            }
            if ((curLen * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype) / BLOCK_SIZE, 0,
                                0);
            } else {
                copy_ubuf_to_gm_align_b16(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype), 0, 0,
                                          0, 0);
            }
            if (block == totalSubBlockNum - 1) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}

template <typename Dtype>
inline __aicore__ void RunAivSoftmax(__gm__ Dtype *buf, __gm__ float *expBuf, uint32_t m,
                                     uint32_t n, uint32_t calcLen, uint32_t outN = 0,
                                     bool seqHead = true, uint32_t maskOff = 0, uint32_t maskStride = 1,
                                     bool hasScale = false, float scale = 1.0f)
{
    RunAivSoftmaxLong<Dtype>(buf, expBuf, m, n, calcLen, outN, seqHead, maskOff, maskStride, hasScale,
                             scale);
}

#else
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxPingPong(__gm__ Dtype *buf, uint32_t m, uint32_t n, int calcLen,
                                             uint32_t outN = 0, bool seqHead = true, uint32_t maskOff = 0,
                                             uint32_t maskStride = 1,
                                             __gm__ float *maxBuf = nullptr,
                                             __gm__ float *sumBuf = nullptr, bool hasScale = false,
                                             float scale = 1.0f)
{
}
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxLong(__gm__ Dtype *buf, __gm__ float *expBuf, uint32_t m,
                                         uint32_t n, uint32_t calcLen, uint32_t outN = 0,
                                         bool seqHead = true, uint32_t maskOff = 0, uint32_t maskStride = 1,
                                         bool hasScale = false, float scale = 1.0f)
{
}
#endif
#endif