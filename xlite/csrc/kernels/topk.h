/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"

#ifdef __DAV_C220_VEC__
constexpr uint32_t BIT_SIZE_OF_U32 = 32;
constexpr uint64_t SORT_BLOCK_SIZE = 32;
constexpr uint64_t SORT_RESULT_BLOCK_SIZE = SORT_BLOCK_SIZE * 2;
constexpr uint64_t MGR_SORT_VALID_BITS_OFFSET = 8;
constexpr uint64_t MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET = 12;

//#define XLITE_KERNEL_DEBUG

template <typename T>
static __aicore__ inline void DumpBuffer(__ubuf__ T *buf, const __gm__ char *name, int size,
                                         int step = 1, int offset = 0, bool toInt = false)
{
#ifdef XLITE_KERNEL_DEBUG
    pipe_barrier(PIPE_V);
    printf("%s: [", name);
    for (int i = 0; i < size; i++) {
        if (i % 10 == 0) {
            printf("\n");
        }
        if (toInt) {
            printf("%u ", buf[i * step + offset]);
        } else {
            printf("%f ", buf[i * step + offset]);
        }
    }
    printf("]\n");
#endif
}

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
            uint32_t numTokens, uint32_t numRoutedExperts, uint32_t topK)
    {
        set_mask_norm();
        this->scoresGm = (__gm__ Dtype *)scores;
        this->indicesGm = (__gm__ uint32_t *)indices;
        this->outIndicesGm = (__gm__ uint32_t *)outIndices;

        this->nTokens = numTokens;
        this->nRoutedExperts = numRoutedExperts;
        this->topK = topK;
        this->calcRepeat = DIV_ROUND_UP(nRoutedExperts, VECTOR_MAX_NUM_OF_FP32);

        uint32_t pad = ROUND_UP(numRoutedExperts, VECTOR_MAX_NUM_OF_FP32) * sizeof(float);
        uint32_t padDtype =
            ROUND_UP(numRoutedExperts, VECTOR_MAX_BYTESIZE / sizeof(Dtype)) * sizeof(Dtype);
        uint64_t off = 0;
        scoresIn = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += pad;
        indicesIn = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += pad;
        this->indices = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += pad;
        sortTmp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 2 * pad;
        sortMrgTmp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 2 * pad;
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            scoresInTmp = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += pad / 2;
        }
    }

    __aicore__ inline void Merge4(__ubuf__ float *dst, __ubuf__ float *src,
            uint64_t mrgRepeat, uint64_t blockSize)
    {
        auto resultBlockSize = 2 * blockSize;
        uint64_t validBits = 0xF;
        uint64_t ifExhaustedSuspension = 0;
        uint64_t config = mrgRepeat | validBits << MGR_SORT_VALID_BITS_OFFSET |
            ifExhaustedSuspension << MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET;
        __ubuf__ float *addrArray[4] = {src, src + resultBlockSize,
                                        src + 2 * resultBlockSize,
                                        src + 3 * resultBlockSize};

        uint64_t lengths =
            blockSize | blockSize << 16 | blockSize << 32 | blockSize << 48;
        vmrgsort4(dst, addrArray, lengths, config);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void Merge3(__ubuf__ float *dst, __ubuf__ float *src,
            uint64_t mrgRepeat, uint64_t blockSize)
    {
        auto resultBlockSize = 2 * blockSize;
        uint64_t validBits = 0b111;
        uint64_t ifExhaustedSuspension = 0;
        uint64_t config = mrgRepeat | validBits << MGR_SORT_VALID_BITS_OFFSET |
            ifExhaustedSuspension << MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET;
        __ubuf__ float *addrArray[4] = {src, src + resultBlockSize,
                                        src + 2 * resultBlockSize};

        uint64_t lengths =
            blockSize | blockSize << 16 | blockSize << 32;
        vmrgsort4(dst, addrArray, lengths, config);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void Run()
    {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        copy_gm_to_ubuf_align_b32(indicesIn, indicesGm, 0, 1, nRoutedExperts * sizeof(uint32_t), 0,
                                  0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        for (int tokenIdx = block_idx; tokenIdx < nTokens; tokenIdx += block_num) {
            CopyInScores(tokenIdx);
            ConvertInput();
            Sort();
            FillOutScores(tokenIdx);
            pipe_barrier(PIPE_ALL);
        }

        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void CopyInScores(int tokenIdx)
    {
        if constexpr (std::is_same<Dtype, float>::value) {
            copy_gm_to_ubuf_align_b32(scoresIn, scoresGm + tokenIdx * nRoutedExperts, 0, 1,
                                      nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            copy_gm_to_ubuf_align_b32(scoresInTmp, scoresGm + tokenIdx * nRoutedExperts, 0, 1,
                                      nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
        }

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    }

    __aicore__ inline void ConvertInput()
    {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            uint64_t vector_bf162fp32_config = set_vector_1src_xt(8, 4, 1, 1, calcRepeat);
            vconv_bf162f32(scoresIn, scoresInTmp, vector_bf162fp32_config);
            DumpBuffer(scoresIn, "scoresIn", 10);
        }
    }

    __aicore__ inline void Sort()
    {
            // Assume no remainder
            auto repeat = nRoutedExperts / SORT_BLOCK_SIZE;
            vbitsort(sortTmp, scoresIn, indicesIn, repeat);
            DumpBuffer(indicesIn, "indicesIn", 10, 1, 0, true);

            // Sorted arrays of 128 elements
            uint64_t mrgRepeat = nRoutedExperts / 128;
            Merge4(sortMrgTmp, sortTmp, mrgRepeat, 32);

            // Sorted arrays of 512 elements
            mrgRepeat = nRoutedExperts / (128 * 4);
            Merge4(sortTmp, sortMrgTmp, mrgRepeat, 128);

            // Sorted arrays of 2048 elements
            mrgRepeat = nRoutedExperts / (512 * 4);
            Merge4(sortMrgTmp, sortTmp, mrgRepeat, 512);

            // Sorted array of 6144 elements
            mrgRepeat = 1;
            Merge3(sortTmp, sortMrgTmp, mrgRepeat, 2048);

            vreducev2(indices, (__ubuf__ uint32_t *)sortTmp, (__ubuf__ uint32_t *)sortTmp, topK / 32, 1, 2, 8, 0);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            DumpBuffer(indices, "indices", 10, 1, 0, true);
    }

    __aicore__ inline void FillOutScores(int tokenIdx) {
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm_align_b32(outIndicesGm + tokenIdx * topK,
                    indices, 0, 1, topK * sizeof(uint32_t), 0,0,0,0);
    }

private:
    uint32_t nTokens;
    uint32_t nRoutedExperts;
    uint32_t nGroup;
    uint32_t nTopkGroup;
    uint32_t nExpertsPerGroup;
    uint32_t nScores;
    uint32_t topK;
    bool normTopKProb;
    float scale;
    int calcRepeat;

    __gm__ Dtype *scoresGm;
    __gm__ uint32_t *indicesGm;
    __gm__ uint32_t *outIndicesGm;
    __gm__ uint32_t *routingMapGm;

    __ubuf__ float *scoresIn;
    __ubuf__ uint32_t *indicesIn;
    __ubuf__ uint32_t *indices;
    __ubuf__ float *sortTmp;
    __ubuf__ float *sortMrgTmp;
    __ubuf__ Dtype *scoresInTmp;
};

#define TOPK_FUNC_DEFINE(dtype)                                                             \
    extern "C" __global__ __aicore__ void topk_##dtype(                                     \
        GM_ADDR scores, GM_ADDR indices, GM_ADDR outIndices, uint32_t numTokens, uint32_t numRoutedExperts,     \
        uint32_t topK)                                                                      \
    {                                                                                       \
        XliteTopK<dtype> op;                                                                     \
        op.Init(scores, indices, outIndices, numTokens, numRoutedExperts, topK);                        \
        op.Run();                                                                            \
    }
#else
#define TOPK_FUNC_DEFINE(dtype)                                                             \
    extern "C" __global__ __aicore__ void topk_##dtype(                                     \
        GM_ADDR scores, GM_ADDR indices, GM_ADDR outIndices, uint32_t numTokens, uint32_t numRoutedExperts,     \
        uint32_t topK)                                                                      \
    {                                                                                       \
    }
#endif

