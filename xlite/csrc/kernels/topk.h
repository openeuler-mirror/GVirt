/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

/**
 * @brief Select topK elements from a batch of sequences.
 *
 * Constraints:
 *  - Only topK equals 2048 is supported
 *
 * @param[in] scores Scores values for each batch padded up to maxSeqLen [numBatches X maxSeqLen]
 * @param[in] indices torch.arange-like array [0..maxSeqLen]
 * @param[out] outIndices computed topK indices [numBatches X topK]
 * @param[in] lens Number of tokens in each batch [numBatches]
 * @param[in] maxSeqLen Maximal possible number of tokens in a batch
 * @param[in] numBatches Number of batches
 * @param[in] topK How many top elements to find
 */

#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"
#include <limits>

// #define XLITE_KERNEL_DEBUG
#include "debug.h"

#ifdef __DAV_C220_VEC__
constexpr uint32_t BIT_SIZE_OF_U32 = 32;
constexpr uint64_t SORT_BLOCK_SIZE = 32;
constexpr uint64_t SORT_RESULT_BLOCK_SIZE = SORT_BLOCK_SIZE * 2;
constexpr uint64_t MGR_SORT_VALID_BITS_OFFSET = 8;
constexpr uint64_t MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET = 12;
constexpr uint32_t CHUNK_SIZE = 2048;

static __aicore__ inline void DumpBufferIndex(__ubuf__ float *buf, const __gm__ char *name,
                                              int size, int step = 1)
{
    DumpBuffer(buf, name, size, step, 1, true);
}

template <typename Dtype>
class XliteTopK
{
public:
    __aicore__ inline XliteTopK() = default;

    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR indices, GM_ADDR outIndices,
                                GM_ADDR queryLens, GM_ADDR cachedLens, uint32_t maxSeqLen,
                                uint32_t numBatches, uint32_t topK)
    {
        set_mask_norm();
        this->scoresGm = (__gm__ Dtype *)scores;
        this->indicesGm = (__gm__ uint32_t *)indices;
        this->outIndicesGm = (__gm__ uint32_t *)outIndices;
        this->queryLensGm = (__gm__ uint32_t *)queryLens;
        this->cachedLensGm = (__gm__ uint32_t *)cachedLens;

        this->topK = topK;
        this->nBatches = numBatches;
        this->maxSeqLen = maxSeqLen;

        uint32_t pad = ROUND_UP(topK, VECTOR_MAX_NUM_OF_FP32) * sizeof(float);
        uint32_t padDtype = ROUND_UP(topK, VECTOR_MAX_BYTESIZE / sizeof(Dtype)) * sizeof(Dtype);
        uint64_t off = 0;

        scoresIn = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        dbg_printf("scoresIn: %lx\n", off);
        off += pad;
        indicesIn = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        dbg_printf("indicesIn: %lx\n", off);
        off += pad;
        this->indices = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        dbg_printf("indices: %lx\n", off);
        off += pad;

        this->topkBuf[0] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        dbg_printf("topk0: %lx\n", off);
        off += 4 * pad;
        this->topkBuf[1] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        dbg_printf("topk1: %lx\n", off);
        off += 4 * pad;
        this->scratch[0] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        dbg_printf("scratch0: %lx\n", off);
        off += 2 * pad;
        this->scratch[1] = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        dbg_printf("scratch1: %lx\n", off);
        off += 2 * pad;

        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            scoresInTmp = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            dbg_printf("scoresInTmp: %lx\n", off);
            off += pad / 2;
        }
        this->queryLensIn = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        dbg_printf("queryLensIn: %lx\n", off);
        off += ROUND_UP(numBatches * sizeof(uint32_t), UB_BUF_ALIGN_SIZE);
        this->cachedLensIn = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        dbg_printf("cachedLensIn: %lx\n", off);
        off += ROUND_UP(numBatches * sizeof(uint32_t), UB_BUF_ALIGN_SIZE);
        dbg_printf("Allocated in UB: %lu\n", off);
    }

    __aicore__ inline void Merge4(__ubuf__ float *dst, __ubuf__ float *src, uint64_t mrgRepeat,
                                  uint64_t blockSize)
    {
        auto resultBlockSize = 2 * blockSize;
        uint64_t validBits = 0xF;
        uint64_t ifExhaustedSuspension = 0;
        uint64_t config = mrgRepeat | validBits << MGR_SORT_VALID_BITS_OFFSET |
                          ifExhaustedSuspension << MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET;
        __ubuf__ float *addrArray[4] = {src, src + resultBlockSize, src + 2 * resultBlockSize,
                                        src + 3 * resultBlockSize};

        uint64_t lengths = blockSize | blockSize << 16 | blockSize << 32 | blockSize << 48;
        vmrgsort4(dst, addrArray, lengths, config);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void Merge2(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1,
                                  uint64_t mrgRepeat, uint64_t blockSize)
    {
        auto resultBlockSize = 2 * blockSize;
        uint64_t validBits = 0b11;
        uint64_t ifExhaustedSuspension = 0;
        uint64_t config = mrgRepeat | validBits << MGR_SORT_VALID_BITS_OFFSET |
                          ifExhaustedSuspension << MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET;
        __ubuf__ float *addrArray[4] = {src0, src1};

        uint64_t lengths = blockSize | blockSize << 16;
        vmrgsort4(dst, addrArray, lengths, config);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void Run()
    {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        copy_gm_to_ubuf_align_b32(queryLensIn, queryLensGm, 0, 1, nBatches * sizeof(uint32_t), 0, 0,
                                  0, 0);
        copy_gm_to_ubuf_align_b32(cachedLensIn, cachedLensGm, 0, 1, nBatches * sizeof(uint32_t), 0,
                                  0, 0, 0);
        pipe_barrier(PIPE_MTE2);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        uint32_t mIdx = 0;
        for (int batchIdx = 0; batchIdx < nBatches; batchIdx++) {
            uint32_t queryLen = queryLensIn[batchIdx];
            uint32_t cachedLen = cachedLensIn[batchIdx];
            uint32_t len = queryLen + cachedLen;
            for (int queryIdx = 0; queryIdx < queryLen; queryIdx++, mIdx++) {
                if (mIdx % block_num != block_idx) {
                    continue;
                }
                pipe_barrier(PIPE_ALL);

                initTopk();

                int current = 0;
                uint32_t offset = 0;
                for (uint32_t processed = 0; processed < len;) {
                    uint32_t length = min(len - processed, CHUNK_SIZE);

                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

                    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
                    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

                    CopyInIndices(offset, length);
                    CopyInScores(mIdx * maxSeqLen + offset, length);
                    PadInputs(length);

                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                    ConvertInput(CHUNK_SIZE);

                    Sort(CHUNK_SIZE, current);

                    processed += length;
                    current = 1 - current;
                    offset += length;
                }

                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

                FillOutScores(mIdx, current);
            }
        }

        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void CopyInIndices(int offset, uint32_t len)
    {
        copy_gm_to_ubuf_align_b32(indicesIn, indicesGm + offset, 0, 1, len * sizeof(uint32_t), 0, 0,
                                  0, 0);
        DumpBuffer(indicesIn, "indicesIn", 10, 1, 0, true);
    }

    __aicore__ inline void CopyInScores(int offset, uint32_t len)
    {
        __ubuf__ Dtype *buf;
        if constexpr (std::is_same<Dtype, float>::value) {
            buf = scoresIn;
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            buf = scoresInTmp;
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        copy_gm_to_ubuf_align_b16(buf, scoresGm + offset, 0, 1, len * sizeof(Dtype), 0, 0, 0, 0);
    }

    __aicore__ inline void PadInputs(uint32_t len)
    {
        __ubuf__ Dtype *buf;
        Dtype min;
        // Nuber of elements for vector_dup
        uint32_t vbatch;

        if constexpr (std::is_same<Dtype, float>::value) {
            buf = scoresIn;
            min = -std::numeric_limits<float>::infinity();
            vbatch = 64;

        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            buf = scoresInTmp;
            min = -3.4028235e+38;
            vbatch = 128;
        }
        if (len >= 2048) {
            return;
        }

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        uint32_t base = ROUND_DOWN(len, vbatch);
        uint32_t tail = len % vbatch;
        uint64_t top;
        uint64_t bottom;
        uint64_t chunk = DIV_ROUND_UP(len, vbatch) - 1;

        if (tail == 0) {
            top = 0;
            bottom = 0;
        } else if (tail < 64) {
            top = (uint64_t)-1;
            bottom = ((1UL << (tail)) - 1) ^ (uint64_t)-1;
        } else if (tail == 64) {
            top = (uint64_t)-1;
            bottom = 0;
        } else {
            top = ((1UL << (tail - 64)) - 1) ^ (uint64_t)-1;
            bottom = 0;
        }

        uint32_t repeat = (CHUNK_SIZE / vbatch) - DIV_ROUND_UP(len, vbatch);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        set_vector_mask(top, bottom);
        vector_dup(buf + chunk * vbatch, min, 1, 1, 1, 8, 0);

        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        vector_dup(buf + (chunk + 1) * vbatch, min, repeat, 1, 1, 8, 0);
    }

    __aicore__ inline void ConvertInput(uint32_t len)
    {
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            int calcRepeat = DIV_ROUND_UP(len, VECTOR_MAX_NUM_OF_FP32);
            uint64_t vconvConfig = set_vector_1src_xt(8, 4, 1, 1, calcRepeat);
            vconv_bf162f32(scoresIn, scoresInTmp, vconvConfig);
            DumpBuffer(scoresInTmp, "scoresInTmp", 10);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void Sort(uint32_t len, int current)
    {
        __ubuf__ float *scratch0 = scratch[current];
        __ubuf__ float *scratch1 = scratch[1 - current];

        __ubuf__ float *topk0 = topkBuf[current];
        __ubuf__ float *topk1 = topkBuf[1 - current];

        auto repeat = len / SORT_BLOCK_SIZE;
        vbitsort(scratch0, scoresIn, indicesIn, repeat);
        pipe_barrier(PIPE_V);

        uint64_t mrgRepeat = len / 128;
        Merge4(scratch1, scratch0, mrgRepeat, 32);

        mrgRepeat = len / 512;
        Merge4(scratch0, scratch1, mrgRepeat, 128);

        mrgRepeat = len / 2048;
        Merge4(scratch1, scratch0, mrgRepeat, 512);

        Merge2(topk1, topk0, scratch1, 1, 2048);
    }

    __aicore__ inline void initTopk()
    {
        uint64_t even = 0x5555555555555555;
        uint64_t odd = 0xAAAAAAAAAAAAAAAA;
        float min = -std::numeric_limits<float>::infinity();

        set_vector_mask(even, even);
        vector_dup(topkBuf[0], min, topK * 2 / 64, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);

        set_vector_mask(odd, odd);
        vector_dup(topkBuf[0], 0, topK * 2 / 64, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        DumpBuffer(topkBuf[0], "topkBuf[0]", 2048, 2, 0, false);
        DumpBuffer(topkBuf[0], "topkBuf[0]", 2048, 2, 1, true);
    }

    __aicore__ inline void FillOutScores(int offset, int current)
    {
        __ubuf__ uint32_t *src = (__ubuf__ uint32_t *)topkBuf[current];
        __ubuf__ uint32_t *dst = (__ubuf__ uint32_t *)topkBuf[1 - current];
        DumpBuffer(src, "Result", 10, 2, 1, true);

        vreducev2(dst, src, src, topK / 32, 1, 2, 8, 0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b32(outIndicesGm + offset * topK, dst, 0, 1, topK * sizeof(uint32_t),
                                  0, 0, 0, 0);
    }

private:
    uint32_t nBatches;
    uint32_t topK;
    uint32_t maxSeqLen;

    __gm__ Dtype *scoresGm;
    __gm__ uint32_t *indicesGm;
    __gm__ uint32_t *outIndicesGm;
    __gm__ uint32_t *queryLensGm;
    __gm__ uint32_t *cachedLensGm;

    __ubuf__ float *scoresIn;
    __ubuf__ uint32_t *indicesIn;
    __ubuf__ uint32_t *indices;
    __ubuf__ uint32_t *queryLensIn;
    __ubuf__ uint32_t *cachedLensIn;
    __ubuf__ float *topkBuf[2];
    __ubuf__ float *scratch[2];
    __ubuf__ Dtype *scoresInTmp;
};

#define TOPK_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void topk_##dtype(                                           \
        GM_ADDR scores, GM_ADDR indices, GM_ADDR outIndices, GM_ADDR queryLens,                   \
        GM_ADDR cachedLens, uint32_t maxSeqLen, uint32_t numBatches, uint32_t topK)               \
    {                                                                                             \
        XliteTopK<dtype> op;                                                                      \
        op.Init(scores, indices, outIndices, queryLens, cachedLens, maxSeqLen, numBatches, topK); \
        op.Run();                                                                                 \
    }
#else
#define TOPK_FUNC_DEFINE(dtype)                                                     \
    extern "C" __global__ __aicore__ void topk_##dtype(                             \
        GM_ADDR scores, GM_ADDR indices, GM_ADDR outIndices, GM_ADDR queryLens,     \
        GM_ADDR cachedLens, uint32_t maxSeqLen, uint32_t numBatches, uint32_t topK) \
    {                                                                               \
    }
#endif
