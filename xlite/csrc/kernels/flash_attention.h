/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"
#include "flash_attention_param.h"
#include "softmax_attn_aiv.h"

#define MAX_M0 128
#define MBLOCKSIZE 16
#define NBLOCKSIZE 16

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
 *   @param subHeadOffset Starting row offset within the query group (for causal masking)
 *   @param nHeads      Number of query attention heads
 *   @param nKVHeads    Number of key-value attention heads (for GQA/MQA support)
 *   @param headSize    Dimension of each attention head
 *   @param isFirstKvTile Flag indicating if this is the first KV chunk
 *   @param actualCalcSoftmaxLen Base length for softmax computation (adjusted by causal mask)
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
                                           uint32_t subHeadOffset, uint32_t nHeads,
                                           uint32_t nKVHeads, uint32_t headSize, int isFirstKvTile,
                                           int actualCalcSoftmaxLen, uint32_t maskOff,
                                           uint32_t maskStride)
{
#ifdef XLITE_KERNEL_DEBUG
    printf("RunAivSoftmaxUpdate: m=%u, headSize=%u, isFirstKvTile=%d, actualCalcSoftmaxLen=%d\n", m,
           headSize, isFirstKvTile, actualCalcSoftmaxLen);
#endif
    uint32_t headNumInGroup = nHeads / nKVHeads;
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
        // mOffset: offset to the correct KV head group in the output tensor
        // nOffset: offset within the query head group
        uint32_t mOffset = ((mIdx + subHeadOffset) / headNumInGroup) * nHeads;
        uint32_t nOffset = ((mIdx + subHeadOffset) % headNumInGroup);
        // Adjust actual calculation length based on causal mask position
        int actualCalcLen = actualCalcSoftmaxLen + (mIdx + maskOff) / maskStride;
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
                copy_ubuf_to_gm(output + (mOffset + nOffset) * headSize, svCurrUb[curr], 0, 1,
                                headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(output + (mOffset + nOffset) * headSize, svCurrUb[curr],
                                          0, 1, headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            copy_ubuf_to_gm_align_b16(lastMax + mOffset + nOffset, maxCurrUb[curr], 0, 1,
                                      sizeof(float), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b16(lastSum + mOffset + nOffset, sumCurrUb[curr], 0, 1,
                                      sizeof(float), 0, 0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
        } else {
            // Non-first chunk: merge with previous results using online softmax algorithm
            // Step 1: Load previous and current statistics from GM
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            copy_gm_to_ubuf_align_b16(maxPrevUb[curr], lastMax + mOffset + nOffset, 0, 1,
                                      sizeof(float), 0, 0, 0, 0);
            copy_gm_to_ubuf_align_b16(sumPrevUb[curr], lastSum + mOffset + nOffset, 0, 1,
                                      sizeof(float), 0, 0, 0, 0);
            copy_gm_to_ubuf_align_b16(maxCurrUb[curr], currMax + mIdx, 0, 1, sizeof(float), 0, 0, 0,
                                      0);
            copy_gm_to_ubuf_align_b16(sumCurrUb[curr], currSum + mIdx, 0, 1, sizeof(float), 0, 0, 0,
                                      0);
            // Step 2: Load previous and current sv values
            if (headSize * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(svPrevUb[curr], output + (mOffset + nOffset) * headSize, 0, 1,
                                headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
                copy_gm_to_ubuf(svCurrUb[curr], currSv + mIdx * headSize, 0, 1,
                                headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(svPrevUb[curr], output + (mOffset + nOffset) * headSize,
                                          0, 1, headSize * sizeof(Dtype), 0, 0, 0, 0);
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
                copy_ubuf_to_gm(output + (mOffset + nOffset) * headSize, svOutUb[curr], 0, 1,
                                headSize * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(output + (mOffset + nOffset) * headSize, svOutUb[curr], 0,
                                          1, headSize * sizeof(Dtype), 0, 0, 0, 0);
            }
            copy_ubuf_to_gm_align_b16(lastMax + mOffset + nOffset, newMaxOutUb[curr], 0, 1,
                                      sizeof(float), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b16(lastSum + mOffset + nOffset, newSumOutUb[curr], 0, 1,
                                      sizeof(float), 0, 0, 0, 0);
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

template <typename Dtype>
class FlashAttention
{
public:
    __aicore__ inline FlashAttention()
    {
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk,
                                GM_ADDR sv, GM_ADDR max, GM_ADDR sum, GM_ADDR lastMax,
                                GM_ADDR lastSum, GM_ADDR sync, GM_ADDR output,
                                GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                                GM_ADDR blockTables, uint32_t nHeads, uint32_t nKVHeads,
                                uint32_t headSize, uint32_t blockSize, uint32_t batch,
                                uint32_t maxNumBlocks)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->input.SetGlobalBuffer((__gm__ Dtype *)input);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->vCache.SetGlobalBuffer((__gm__ Dtype *)vCache);
        this->output.SetGlobalBuffer((__gm__ Dtype *)output);

        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->blockTables = (__gm__ int32_t *)blockTables;

        this->nHeads = nHeads;
        this->nKVHeads = nKVHeads;
        this->nQKVHeads = nHeads + 2 * nKVHeads;
        this->headNumInGroup = nHeads / nKVHeads;
        this->headSize = headSize;
        this->blockSize = blockSize;
        this->batch = batch;
        this->maxNumBlocks = maxNumBlocks;
        this->maxSeqLen = maxNumBlocks * blockSize;
        this->qMemSize = nHeads * headSize;
        this->kvMemSize = nKVHeads * headSize;
        this->qkvMemSize = nQKVHeads * headSize;
        this->groupMemSize = headNumInGroup * headSize;
        this->blockMemSize = blockSize * kvMemSize;
        this->blockIdx = block_idx;
        this->subBlockIdx = get_subblockid();
        this->nextBlockIdx = (blockIdx + 1) % block_num;
        this->prevBlockIdx = blockIdx == 0 ? (block_num - 1) : (blockIdx - 1);
        this->setNextGeneration = 1;
        this->waitPrevGeneration = 1;

        this->qk[0].SetGlobalBuffer(((__gm__ Dtype *)qk) +
                                    block_idx * MAX_M0 * TILESIZE_OF_CACHED_KV);
        this->qk[1].SetGlobalBuffer(((__gm__ Dtype *)qk) +
                                    block_idx * MAX_M0 * TILESIZE_OF_CACHED_KV +
                                    block_num * MAX_M0 * TILESIZE_OF_CACHED_KV);
        this->sv[0].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * MAX_M0 * headSize);
        this->sv[1].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * MAX_M0 * headSize +
                                    block_num * MAX_M0 * headSize);
        this->max[0].SetGlobalBuffer(((__gm__ float *)max) + block_idx * MAX_M0 * 2 +
                                     subBlockIdx * MAX_M0);
        this->max[1].SetGlobalBuffer(((__gm__ float *)max) + block_idx * MAX_M0 * 2 +
                                     subBlockIdx * MAX_M0 + block_num * MAX_M0 * 2);
        this->sum[0].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * MAX_M0 * 2 +
                                     subBlockIdx * MAX_M0);
        this->sum[1].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * MAX_M0 * 2 +
                                     subBlockIdx * MAX_M0 + block_num * MAX_M0 * 2);
        this->prevSv[0].SetGlobalBuffer(((__gm__ Dtype *)sv) + prevBlockIdx * MAX_M0 * headSize);
        this->prevSv[1].SetGlobalBuffer(((__gm__ Dtype *)sv) + prevBlockIdx * MAX_M0 * headSize +
                                        block_num * MAX_M0 * headSize);
        this->prevMax[0].SetGlobalBuffer(((__gm__ float *)max) + prevBlockIdx * MAX_M0 * 2 +
                                         subBlockIdx * MAX_M0);
        this->prevMax[1].SetGlobalBuffer(((__gm__ float *)max) + prevBlockIdx * MAX_M0 * 2 +
                                         subBlockIdx * MAX_M0 + block_num * MAX_M0 * 2);
        this->prevSum[0].SetGlobalBuffer(((__gm__ float *)sum) + prevBlockIdx * MAX_M0 * 2 +
                                         subBlockIdx * MAX_M0);
        this->prevSum[1].SetGlobalBuffer(((__gm__ float *)sum) + prevBlockIdx * MAX_M0 * 2 +
                                         subBlockIdx * MAX_M0 + block_num * MAX_M0 * 2);
        this->lastMax.SetGlobalBuffer((__gm__ float *)lastMax);
        this->lastSum.SetGlobalBuffer((__gm__ float *)lastSum);
        this->setNextSync = (__gm__ int32_t *)sync + blockIdx * 2 + subBlockIdx;
        this->waitPrevSync = (__gm__ int32_t *)sync + prevBlockIdx * 2 + subBlockIdx;

        // 分配L1/L0
        uint64_t l1ATileBytes =
            MAX_M0 * (headSize > blockSize ? headSize : blockSize) * sizeof(Dtype);
        uint64_t l1BTileBytes = blockSize * headSize * sizeof(Dtype);
        uint64_t off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            l1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1ATileBytes;
        }
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B1);
            l1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1BTileBytes;
        }
        off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            l0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1ATileBytes;
        }
        off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1BTileBytes;
        }
        off = 0;
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    __aicore__ inline void SetNextCore()
    {
        __ubuf__ int32_t *val = (__ubuf__ int32_t *)(0ull);
#ifdef XLITE_KERNEL_DEBUG
        printf("block%d subblock%u set block%d subblock%u %u\n", blockIdx, subBlockIdx,
               nextBlockIdx, subBlockIdx, setNextGeneration);
#endif
        *val = setNextGeneration;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b16(setNextSync, val, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
        setNextGeneration++;
    }

    __aicore__ inline void WaitPrevCore()
    {
        __ubuf__ int32_t *val = (__ubuf__ int32_t *)(0ull);
#ifdef XLITE_KERNEL_DEBUG
        printf("block%d subblock%u wait block%d subblock%u %u\n", blockIdx, subBlockIdx,
               prevBlockIdx, subBlockIdx, waitPrevGeneration);
#endif
        do {
            copy_gm_to_ubuf_align_b16(val, waitPrevSync, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        } while (*val < waitPrevGeneration);
        waitPrevGeneration++;
    }

    __aicore__ inline void ResetPrevCore()
    {
        __ubuf__ int32_t *val = (__ubuf__ int32_t *)(0ull);
#ifdef XLITE_KERNEL_DEBUG
        printf("block%d subblock%u reset block%d subblock%u\n", blockIdx, subBlockIdx, prevBlockIdx,
               subBlockIdx);
#endif
        *val = 0;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b16(waitPrevSync, val, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
    }

    /*
     * m: tokens
     * n: cachedTokens
     * k: headSize
     */
    __aicore__ inline void RunAicQK(GlobalTensor<Dtype> query, int queryLen, int kvHeadIdx,
                                    __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                    GlobalTensor<Dtype> qk)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mActual = queryLen * headNumInGroup;
        int nIdxStart = kvOffset / blockSize;
        int nLoop = DIV_ROUND_UP(kvLen, blockSize);
        int mBlockPad = ROUND_UP(mActual, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nBlockPad = ROUND_UP(blockSize, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        int kBlockNum = DIV_ROUND_UP(headSize, kBlockSize);
        int kvHeadOffset = kvHeadIdx * headSize;

        Nd2NzParams nd2nzParams(1 /* NdNum */, queryLen /* nValue */, headSize /* dValue */,
                                0 /* srcNdMatrixStride */, qkvMemSize /* srcDValue */,
                                mBlockPad /* dstNzC0Stride */, headNumInGroup /* dstNzNStride */,
                                0 /* dstNzMatrixStride */);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        for (int h = 0; h < headNumInGroup; h++) {
            DataCopy(l1aBuf[0][MBLOCKSIZE * h], query[headSize * h], nd2nzParams);
        }
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        CopyToL0ACol(l0aBuf[0], l1aBuf[0], mBlockNum, 0, kBlockNum);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

        int curIdx = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {
            uint32_t blockIdx = blockTable[nIdx + nIdxStart];
            int nSize = blockSize;
            if (nIdx * blockSize + nSize > kvLen) {
                nSize = kvLen - nIdx * blockSize;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            CopyGmToL1Nd2Nz(l1bBuf[curIdx], kCache[blockIdx * blockMemSize + kvHeadOffset], nSize,
                            headSize, kvMemSize, nBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            CopyToL0BCol(l0bBuf[curIdx], l1bBuf[curIdx], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            CalMmad(l0cBuf, l0aBuf[0], l0bBuf[curIdx], mBlockPad, nBlockPad, headSize, true);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);

            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            CopyToGm(qk[nIdx * blockSize], l0cBuf, mActual, nSize, mBlockPad,
                     TILESIZE_OF_CACHED_KV);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            PipeBarrier<PIPE_M>();
            curIdx = 1 - curIdx;
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }

    /*
     * m: tokens
     * n: headSize
     * k: cachedTokens
     */
    __aicore__ inline void RunAicSV(GlobalTensor<Dtype> qk, int queryLen, int kvHeadIdx,
                                    __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                    GlobalTensor<Dtype> sv)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mActual = queryLen * headNumInGroup;
        int kIdxStart = kvOffset / blockSize;
        int kLoop = DIV_ROUND_UP(kvLen, blockSize);
        int mBlockPad = ROUND_UP(mActual, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nBlockNum = DIV_ROUND_UP(headSize, NBLOCKSIZE);
        int kBlockPad = ROUND_UP(blockSize, kBlockSize);
        int kBlockNum = kBlockPad / kBlockSize;
        int kvHeadOffset = kvHeadIdx * headSize;

        int curIdx = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int kIdx = 0; kIdx < kLoop; kIdx++) {
            int kSize = blockSize;
            if (kIdx * blockSize + kSize > kvLen) {
                kSize = kvLen - kIdx * blockSize;
                kBlockPad = ROUND_UP(kSize, kBlockSize);
                kBlockNum = kBlockPad / kBlockSize;
            }

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curIdx);
            CopyGmToL1Nd2Nz(l1aBuf[curIdx], qk[kIdx * blockSize], mActual, kSize,
                            TILESIZE_OF_CACHED_KV, mBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curIdx);

            uint32_t blockIdx = blockTable[kIdx + kIdxStart];
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            CopyGmToL1Nd2Nz(l1bBuf[curIdx], vCache[blockIdx * blockMemSize + kvHeadOffset], kSize,
                            headSize, kvMemSize, kBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curIdx);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curIdx);
            CopyToL0ACol(l0aBuf[curIdx], l1aBuf[curIdx], mBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curIdx);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curIdx);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            CopyToL0BTCol(l0bBuf[curIdx], l1bBuf[curIdx], nBlockNum, 0, kBlockNum, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curIdx);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);
            CalMmad(l0cBuf, l0aBuf[curIdx], l0bBuf[curIdx], mBlockPad, headSize, kBlockPad,
                    kIdx == 0);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curIdx);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            PipeBarrier<PIPE_M>();
            curIdx = 1 - curIdx;
        }
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);

        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        CopyToGm(sv, l0cBuf, mActual, headSize, mBlockPad, headSize);
    }

    __aicore__ inline void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);

        uint64_t flagIdx = 0;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t softmaxConfig = 1 | (mode << 4) | (flagIdx << 8);
        flagIdx = 1;
        uint64_t updateConfig = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskLen, lastkvHeadIdx, last, lastKvOffset, lastKvLen,
            lastQueryTaskOffset;
        __gm__ uint32_t *lastBlockTable;

        int needDoSV = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = MAX_M0;
            int queryTileSize = m0 / headNumInGroup;
            if (queryTileSize == 0) {
                queryTileSize = 1;
            }

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNum = DIV_ROUND_UP(cachedLen + queryLen, TILESIZE_OF_CACHED_KV);
            int taskNum = queryNum * nKVHeads * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int kvHeadIdx = (idx / kvNum) % nKVHeads;
                int queryIdx = idx / (kvNum * nKVHeads);
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                int kvOffset = kvIdx * TILESIZE_OF_CACHED_KV;
                if (calcLen <= kvOffset) {
                    continue;
                }
                if (totalIdx % block_num != block_idx) {
                    totalIdx++;
                    continue;
                }
                totalIdx++;

                int kvLen = TILESIZE_OF_CACHED_KV;
                if (kvOffset + kvLen > calcLen) {
                    kvLen = calcLen - kvOffset;
                }

                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;
                int kvHeadOffset = kvHeadIdx * groupMemSize;

                // do queryIdx & kvHeadIdx & kvIdx's QK
                uint32_t qOffset = queryTaskOffset * headSize * nQKVHeads + kvHeadOffset;

#ifdef XLITE_KERNEL_DEBUG
                printf("block%d: {batch %d, query [%u - %u), kvHeadIdx %u, "
                       "kv [%u - %u)} use %d temp buf: QK\n",
                       GetBlockIdx(), batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                       kvHeadIdx, kvOffset, kvOffset + kvLen, curr);
#endif
                RunAicQK(input[qOffset], queryTaskLen, kvHeadIdx, blockTable, kvOffset, kvLen,
                         qk[curr]);
                ffts_cross_core_sync(PIPE_FIX, softmaxConfig);

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(2);
                    // do softmax * V
#ifdef XLITE_KERNEL_DEBUG
                    printf("block%d: {batch %d, query [%u - %u), kvHeadIdx %u, "
                           "kv [%u - %u)} use %d temp buf: SV\n",
                           GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                           lastQueryTaskOffset + lastQueryTaskLen, lastkvHeadIdx, lastKvOffset,
                           lastKvOffset + lastKvLen, last);
#endif
                    RunAicSV(qk[last], lastQueryTaskLen, lastkvHeadIdx, lastBlockTable,
                             lastKvOffset, lastKvLen, sv[last]);
                    ffts_cross_core_sync(PIPE_FIX, updateConfig);
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastkvHeadIdx = kvHeadIdx;
                lastBlockTable = blockTable;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                last = curr;
                needDoSV = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
            cachedLen = -1;
        }

        // do last softmax * V
        if (needDoSV != 0) {
            wait_flag_dev(2);
#ifdef XLITE_KERNEL_DEBUG
            printf("block%d: {batch %d, query [%u - %u), kvHeadIdx %u, "
                   "kv [%u - %u)} use %d temp buf: SV\n",
                   GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                   lastQueryTaskOffset + lastQueryTaskLen, lastkvHeadIdx, lastKvOffset,
                   lastKvOffset + lastKvLen, last);
#endif
            RunAicSV(qk[last], lastQueryTaskLen, lastkvHeadIdx, lastBlockTable, lastKvOffset,
                     lastKvLen, sv[last]);
            ffts_cross_core_sync(PIPE_FIX, updateConfig);
        }
    }

    __aicore__ inline void RunAiv()
    {
        set_atomic_none();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        uint64_t flagIdx = 2;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskLen, lastkvHeadIdx, last, lastKvOffset, lastKvLen,
            lastQueryTaskOffset, lastWorkStart, lastWorkCurCore, lastActualCalcSoftmaxLen;
        int lastIsLastKvTile, lastSubHeadOffset;
        uint32_t lastOutOffset;
        __gm__ uint32_t *lastBlockTable;

        int needDoUpdate = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        int generation = 0;
        int resetNextCore = 0;
        int resetPrevCore = 0;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = MAX_M0;
            int queryTileSize = m0 / headNumInGroup;
            if (queryTileSize == 0) {
                queryTileSize = 1;
            }

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNum = DIV_ROUND_UP(cachedLen + queryLen, TILESIZE_OF_CACHED_KV);
            int taskNum = queryNum * nKVHeads * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int kvHeadIdx = (idx / kvNum) % nKVHeads;
                int queryIdx = idx / (kvNum * nKVHeads);
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                int kvOffset = kvIdx * TILESIZE_OF_CACHED_KV;
                if (calcLen <= kvOffset) {
                    continue;
                }
                if (totalIdx % block_num != block_idx) {
                    totalIdx++;
                    continue;
                }
                totalIdx++;

                int kvLen = TILESIZE_OF_CACHED_KV;
                if (kvOffset + kvLen > calcLen) {
                    kvLen = calcLen - kvOffset;
                }

                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;
                int kvHeadOffset = kvHeadIdx * groupMemSize;

                uint32_t qOffset = queryTaskOffset * headSize * nQKVHeads + kvHeadOffset;

                int isLastKvTile = (kvOffset + kvLen == calcLen) ? 1 : 0;

                int nWork = queryTaskLen * headNumInGroup;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                int nWorkStart = subBlockIdx * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                int subQOffset = nWorkStart / headNumInGroup;
                int subHeadOffset = nWorkStart % headNumInGroup;
                uint32_t outOffset =
                    (queryTaskOffset + subQOffset) * nHeads + kvHeadIdx * headNumInGroup;
                uint32_t calcSoftmaxLen =
                    cachedLen + queryTaskStart + nWorkStart / headNumInGroup + 1;
                int actualCalcSoftmaxLen = calcSoftmaxLen - kvOffset;
                uint32_t outN = ROUND_UP(cachedLen + queryTaskStart + queryTaskLen, blockSize);
                // wait aic qk done
                wait_flag_dev(0);

#ifdef XLITE_KERNEL_DEBUG
                printf("block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u - "
                       "%u), kvHeadIdx %u "
                       "kv [%u - %u)} use %d temp buf: SOFTMAX\n",
                       blockIdx, subBlockIdx, batchIdx, queryTaskOffset,
                       queryTaskOffset + queryTaskLen, nWorkStart, nWorkStart + nWorkCurCore,
                       kvHeadIdx, kvOffset, kvOffset + kvLen, curr);
#endif
                // TODO save max[curr][nWorkStart], sum[curr][nWorkStart]
                RunAivSoftmaxPingPong<Dtype>(
                    (__gm__ Dtype *)qk[curr][nWorkStart * TILESIZE_OF_CACHED_KV].GetPhyAddr(),
                    nWorkCurCore, TILESIZE_OF_CACHED_KV, actualCalcSoftmaxLen, outN,
                    nWorkStart % headNumInGroup, headNumInGroup,
                    (__gm__ float *)max[curr][nWorkStart].GetPhyAddr(),
                    (__gm__ float *)sum[curr][nWorkStart].GetPhyAddr());
                ffts_cross_core_sync(PIPE_MTE3, config);

                if (needDoUpdate != 0) {
                    // wait aic sv done
                    wait_flag_dev(1);
                    if (lastKvOffset != 0) {
                        WaitPrevCore();
                        resetPrevCore = 1;
                    }
#ifdef XLITE_KERNEL_DEBUG
                    printf("block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u "
                           "- %u), kvHeadIdx %u "
                           "kv [%u - %u)} use %d temp buf: UPDATE\n",
                           blockIdx, subBlockIdx, lastBatchIdx, lastQueryTaskOffset,
                           lastQueryTaskOffset + lastQueryTaskLen, lastWorkStart,
                           lastWorkStart + lastWorkCurCore, lastkvHeadIdx, lastKvOffset,
                           lastKvOffset + lastKvLen, last);
#endif
                    // do update with sv[last] & sum[last] & max[last] & prevcore's sum[last] &
                    // max[last]
                    RunAivSoftmaxUpdate<Dtype>(
                        (__gm__ Dtype *)sv[last][lastWorkStart * headSize].GetPhyAddr(),
                        (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ Dtype *)output[lastOutOffset * headSize].GetPhyAddr(),
                        (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                        (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(), lastWorkCurCore,
                        lastSubHeadOffset, nHeads, nKVHeads, headSize, lastKvOffset == 0,
                        lastActualCalcSoftmaxLen, lastWorkStart % headNumInGroup, headNumInGroup);
                    if (!lastIsLastKvTile) {
                        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        SetNextCore();
                    }
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastWorkStart = nWorkStart;
                lastWorkCurCore = nWorkCurCore;
                lastkvHeadIdx = kvHeadIdx;
                lastOutOffset = outOffset;
                lastSubHeadOffset = subHeadOffset;
                lastBlockTable = blockTable;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                lastIsLastKvTile = isLastKvTile;
                lastActualCalcSoftmaxLen = actualCalcSoftmaxLen;
                last = curr;
                needDoUpdate = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
            cachedLen = -1;
        }

        // do last update
        if (needDoUpdate != 0) {
            wait_flag_dev(1);
            if (lastKvOffset != 0) {
                WaitPrevCore();
                resetPrevCore = 1;
            }
#ifdef XLITE_KERNEL_DEBUG
            printf("block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u - %u), "
                   "kvHeadIdx %u "
                   "kv [%u - %u)} use %d temp buf: UPDATE\n",
                   blockIdx, subBlockIdx, lastBatchIdx, lastQueryTaskOffset,
                   lastQueryTaskOffset + lastQueryTaskLen, lastWorkStart,
                   lastWorkStart + lastWorkCurCore, lastkvHeadIdx, lastKvOffset,
                   lastKvOffset + lastKvLen, last);
#endif
            // do update with sv[last] & sum[last] & max[last] & prevcore's sum[last] & max[last]
            RunAivSoftmaxUpdate<Dtype>(
                (__gm__ Dtype *)sv[last][lastWorkStart * headSize].GetPhyAddr(),
                (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                (__gm__ Dtype *)output[lastOutOffset * headSize].GetPhyAddr(),
                (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(), lastWorkCurCore,
                lastSubHeadOffset, nHeads, nKVHeads, headSize, lastKvOffset == 0,
                lastActualCalcSoftmaxLen, lastWorkStart % headNumInGroup, headNumInGroup);
            if (!lastIsLastKvTile) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                SetNextCore();
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (resetPrevCore) {
            ResetPrevCore();
        }
    }

    __aicore__ inline void Run()
    {
#ifdef __DAV_C220_CUBE__
        RunAic();
#elif __DAV_C220_VEC__
        RunAiv();
#endif
    }

private:
    GlobalTensor<Dtype> input;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> vCache;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];
    GlobalTensor<Dtype> sv[PINGPONG_BUF_NUM];
    GlobalTensor<float> max[PINGPONG_BUF_NUM];
    GlobalTensor<float> sum[PINGPONG_BUF_NUM];
    GlobalTensor<Dtype> prevSv[PINGPONG_BUF_NUM];
    GlobalTensor<float> prevMax[PINGPONG_BUF_NUM];
    GlobalTensor<float> prevSum[PINGPONG_BUF_NUM];
    GlobalTensor<float> lastMax;
    GlobalTensor<float> lastSum;
    GlobalTensor<Dtype> output;
    __gm__ int32_t *setNextSync;
    __gm__ int32_t *waitPrevSync;

    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;

    uint32_t nHeads;
    uint32_t nKVHeads;
    uint32_t nQKVHeads;
    uint32_t headNumInGroup;
    uint32_t headSize;
    uint32_t blockSize;
    uint32_t batch;
    uint32_t maxNumBlocks;
    uint32_t maxSeqLen;
    uint32_t qMemSize;
    uint32_t kvMemSize;
    uint32_t qkvMemSize;
    uint32_t groupMemSize;
    uint32_t blockMemSize;
    int blockIdx;
    int subBlockIdx;
    int nextBlockIdx;
    int prevBlockIdx;
    uint32_t setNextGeneration;
    uint32_t waitPrevGeneration;

    LocalTensor<Dtype> l1aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l1bBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<float> l0cBuf;
};

#define FLASH_ATTN_FUNC_DEFINE(dtype)                                                              \
    extern "C" __global__ __aicore__ void flash_attention_##dtype(                                 \
        GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk, GM_ADDR sv, GM_ADDR max,        \
        GM_ADDR sum, GM_ADDR lastMax, GM_ADDR lastSum, GM_ADDR sync, GM_ADDR output,               \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,         \
        uint32_t nHeads, uint32_t nKVHeads, uint32_t headSize, uint32_t blockSize, uint32_t batch, \
        uint32_t maxNumBlocks)                                                                     \
    {                                                                                              \
        FlashAttention<dtype> op;                                                                  \
        op.Init(input, kCache, vCache, qk, sv, max, sum, lastMax, lastSum, sync, output,           \
                queryStartLoc, queryLens, cachedLens, blockTables, nHeads, nKVHeads, headSize,     \
                blockSize, batch, maxNumBlocks);                                                   \
        op.Run();                                                                                  \
    }