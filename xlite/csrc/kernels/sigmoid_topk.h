/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_macro.h"
#include "kernel_operator.h"

#ifdef __DAV_C220_VEC__
constexpr uint32_t BIT_SIZE_OF_U32 = 32;
constexpr uint64_t SORT_BLOCK_SIZE = 32;
constexpr uint64_t SORT_RESULT_BLOCK_SIZE = SORT_BLOCK_SIZE * 2;
constexpr uint64_t MGR_SORT_VALID_BITS_OFFSET = 8;
constexpr uint64_t MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET = 12;

template <typename T>
static __aicore__ inline void DumpBuffer(__ubuf__ T *buf, const __gm__ char *name, int size,
                                         int step=1, int offset=0, bool toInt=false)
{
#ifdef XLITE_KERNEL_DEBUG
    printf("%s: [", name);
    for (int i = 0; i < size; i ++) {
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

static __aicore__ inline void DumpBufferIndex(__ubuf__ float *buf, const __gm__ char *name, int size, int step=1)
{
    DumpBuffer(buf, name, size, step, 1, true);
}

template<typename Dtype>
class SigmoidTopK {
public:
    __aicore__ inline SigmoidTopK() = default;

    __aicore__ inline void Init(GM_ADDR socres, GM_ADDR indices, GM_ADDR bias,
                                float scale, GM_ADDR weightsMap, GM_ADDR routingMap,
                                uint32_t numTokens, uint32_t numRoutedExperts,
                                uint32_t topK, bool normTopKProb)
    {
        set_mask_norm();
        this->socresGm = (__gm__ Dtype*)socres;
        this->biasGm = (__gm__ float*)bias;
        this->weightsMapGm = (__gm__ Dtype*)weightsMap;
        this->indicesGm = (__gm__ uint32_t*)indices;
        this->routingMapGm = (__gm__ uint32_t*)routingMap;

        this->nTokens = numTokens;
        this->nRoutedExperts = numRoutedExperts;
        this->topK = topK;
        this->normTopKProb = normTopKProb;
        this->scale = scale;

        uint32_t pad = ROUND_UP(numRoutedExperts, VECTOR_MAX_NUM_OF_FP32) * sizeof(float);
        uint32_t padDtype = ROUND_UP(numRoutedExperts, VECTOR_MAX_BYTESIZE / sizeof(Dtype)) * sizeof(Dtype);
        uint64_t off = 0;
        socresIn = reinterpret_cast<__ubuf__ float*>((uintptr_t)off);
        off += pad;
        biasIn = reinterpret_cast<__ubuf__ float*>((uintptr_t)off);
        off += pad;
        indicesIn = reinterpret_cast<__ubuf__ uint32_t*>((uintptr_t)off);
        off += pad;
        routingMapOut = reinterpret_cast<__ubuf__ uint32_t*>((uintptr_t)off);
        off += pad;
        weightsOut = reinterpret_cast<__ubuf__ float*>((uintptr_t)off);
        off += pad;
        calc = reinterpret_cast<__ubuf__ float*>((uintptr_t)off);
        off += pad;
        reduceTmp = reinterpret_cast<__ubuf__ float*>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        sortTmp = reinterpret_cast<__ubuf__ float*>((uintptr_t)off);
        off += 2 * pad;
        sortMrgTmp = reinterpret_cast<__ubuf__ float*>((uintptr_t)off);
        off += 2 * pad;
        weightsTopK = reinterpret_cast<__ubuf__ float*>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        indicesTopK = reinterpret_cast<__ubuf__ uint32_t*>((uintptr_t)off);
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            off += pad;
            socresInTmp = reinterpret_cast<__ubuf__ Dtype*>((uintptr_t)off);
            off += padDtype;
            weightsOutTmp = reinterpret_cast<__ubuf__ Dtype*>((uintptr_t)off);
        }
    }

    __aicore__ inline void Run()
    {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        copy_gm_to_ubuf_align_b32(indicesIn, indicesGm, 0, 1, nRoutedExperts * sizeof(uint32_t), 0, 0, 0, 0);
        pipe_barrier(PIPE_ALL);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        for (int tokenIdx = block_idx; tokenIdx < nTokens; tokenIdx += block_num) {
            InitOutBuf();
            CopyInScores(tokenIdx);
            CalcSigmoid();
            SelectTopK();
            FillOutMap();
            CopyOutMap(tokenIdx);
        }
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void InitOutBuf()
    {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        vector_dup(routingMapOut, uint32_t(0), 1, 1, 0, 8, 0);
        DumpBuffer(routingMapOut, "routingMapOut [clean]", 5, 1, 0, true);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        vector_dup(weightsOut, float(0), DIV_ROUND_UP(nRoutedExperts, VECTOR_MAX_NUM_OF_FP32), 1, 0, 8, 0);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void CopyInScores(int tokenIdx)
    {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        if constexpr (std::is_same<Dtype, float>::value) {
            copy_gm_to_ubuf_align_b32(socresIn, socresGm + tokenIdx * nRoutedExperts,
                                      0, 1, nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
            pipe_barrier(PIPE_V);
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            copy_gm_to_ubuf_align_b32(socresInTmp, socresGm + tokenIdx * nRoutedExperts,
                                      0, 1, nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
            pipe_barrier(PIPE_V);
        }
        copy_gm_to_ubuf_align_b32(biasIn, biasGm,
                                  0, 1, nRoutedExperts * sizeof(float), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void CalcSigmoid()
    {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            uint64_t repeatf32 = DIV_ROUND_UP(nRoutedExperts, VECTOR_MAX_NUM_OF_FP32);
            uint64_t vector_bf162fp32_config = set_vector_1src_xt(8, 4, 1, 1, repeatf32);
            vconv_bf162f32(socresIn, socresInTmp, vector_bf162fp32_config);
            pipe_barrier(PIPE_V);
        }

        int repeat = DIV_ROUND_UP(nRoutedExperts, VECTOR_MAX_NUM_OF_FP32);

        // 0 - x
        vector_dup(calc, float(0), repeat, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vsub(calc, calc, socresIn, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // 提前设置标志，允许 MTE2 与后续计算并行
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        // 以下计算与下一轮数据搬入并行执行
        vexp(calc, calc, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // 1 + exp(-x)
        // 修复：dstRepStride 从 0 改为 8
        vector_dup(reduceTmp, float(1), repeat, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(calc, reduceTmp, calc, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // 1 / (1 + exp(-x)) —— 复用 reduceTmp（仍为 1），无需再次填充
        vdiv(calc, reduceTmp, calc, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Add bias
        vadd(calc, calc, biasIn, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void SelectTopK()
    {
        vbitsort(sortTmp, calc, indicesIn, nRoutedExperts / SORT_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
        uint64_t mrgRepeat = 1;
        uint64_t validBits = 0xF;
        uint64_t ifExhaustedSuspension = 0;
        uint64_t config = mrgRepeat | validBits << MGR_SORT_VALID_BITS_OFFSET |
                          ifExhaustedSuspension << MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET;
        uint64_t lengths = SORT_BLOCK_SIZE | SORT_BLOCK_SIZE << 16 |
                           SORT_BLOCK_SIZE << 32 | SORT_BLOCK_SIZE << 48;
        __ubuf__ float *addrArray[4] = {
            sortTmp,
            sortTmp + SORT_RESULT_BLOCK_SIZE,
            sortTmp + 2 * SORT_RESULT_BLOCK_SIZE,
            sortTmp + 3 * SORT_RESULT_BLOCK_SIZE
        };
        vmrgsort4(sortMrgTmp, addrArray, lengths, config);
        pipe_barrier(PIPE_V);

        uint64_t tailLen = nRoutedExperts / SORT_BLOCK_SIZE - 4;

        DumpBuffer(sortTmp, "sortTmp 0", 160, 2);

        copy_ubuf_to_ubuf(sortTmp, sortMrgTmp, 0, 1, 32, 0, 0);
        pipe_barrier(PIPE_V);
        DumpBuffer(sortMrgTmp, "sortMrgTmp 1", 160, 2);

        for (int i = 4; i <  4 + tailLen; i ++) {
            DumpBuffer(sortTmp, "sortTmp i", 160, 2);
            copy_ubuf_to_ubuf(addrArray[3], sortTmp + i * SORT_RESULT_BLOCK_SIZE, 0, 1, 2, 0, 0);
            pipe_barrier(PIPE_V);
            DumpBuffer(sortTmp, "sortTmp 2", 160, 2);

            vmrgsort4(sortMrgTmp, addrArray, lengths, config);
            pipe_barrier(PIPE_V);
            DumpBuffer(sortMrgTmp, "sortMrgTmp 3", 160, 2);
            DumpBuffer(sortMrgTmp, "sortMrgTmp 3", 160, 2, 1, true);
        }

        vreducev2(indicesTopK, (__ubuf__ uint32_t*)sortMrgTmp, (__ubuf__ uint32_t*)sortMrgTmp, 1, 1, 2, 8, 0);
        pipe_barrier(PIPE_V);
        DumpBuffer(indicesTopK, "indicesTopK 4", 160, 1, 0, true);

        vreducev2((__ubuf__ uint32_t*)weightsTopK, (__ubuf__ uint32_t*)sortMrgTmp, (__ubuf__ uint32_t*)sortMrgTmp, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        ReduceSum(reduceTmp, weightsTopK, topK);
        vbrcb((__ubuf__ uint32_t*)reduceTmp, (__ubuf__ uint32_t*)reduceTmp, 0, 0, 1);
        pipe_barrier(PIPE_V);
        if (normTopKProb) {
            vdiv(weightsTopK, weightsTopK, reduceTmp, 1, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
        }
        vmuls(weightsTopK, weightsTopK, scale, 1, 1, 1, 8, 8);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }

    __aicore__ inline void FillOutMap()
    {
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (int i = 0; i < topK; ++i) {
            uint32_t idx = *(indicesTopK + i);
            bitmapSet((__ubuf__ uint64_t*)routingMapOut, idx);
            *(weightsOut + idx) = *(weightsTopK + i);
        }
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            uint64_t repeatf32 = DIV_ROUND_UP(nRoutedExperts, VECTOR_MAX_NUM_OF_FP32);
            uint64_t vector_fp322bf16_config = set_vector_1src_xt(4, 8, 1, 1, repeatf32);
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            vconv_f322bf16r(weightsOutTmp, weightsOut, vector_fp322bf16_config);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        } else if constexpr (std::is_same<Dtype, float>::value) {
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        }
    }

    __aicore__ inline void CopyOutMap(int tokenIdx)
    {
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        } else if constexpr (std::is_same<Dtype, float>::value) {
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        }

        DumpBuffer(routingMapOut, "routingMapOut [final]", 5, 1, 0, true);
        copy_ubuf_to_gm_align_b32(routingMapGm + tokenIdx * nRoutedExperts / BIT_SIZE_OF_U32, routingMapOut,
                                  0, 1, nRoutedExperts * sizeof(uint32_t) / BIT_SIZE_OF_U32, 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if constexpr (std::is_same<Dtype, float>::value) {
            copy_ubuf_to_gm_align_b32(weightsMapGm + tokenIdx * nRoutedExperts, weightsOut,
                                      0, 1, nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            copy_ubuf_to_gm_align_b32(weightsMapGm + tokenIdx * nRoutedExperts, weightsOutTmp,
                                      0, 1, nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }

private:
    uint32_t nTokens;
    uint32_t nRoutedExperts;
    uint32_t topK;
    bool normTopKProb;
    float scale;

    __gm__ Dtype *socresGm;
    __gm__ float *biasGm;
    __gm__ Dtype *weightsMapGm;
    __gm__ uint32_t *indicesGm;
    __gm__ uint32_t *routingMapGm;

    __ubuf__ float *socresIn;
    __ubuf__ float *biasIn;
    __ubuf__ uint32_t *indicesIn;
    __ubuf__ uint32_t *routingMapOut;
    __ubuf__ float *weightsOut;
    __ubuf__ float *calc;
    __ubuf__ float *reduceTmp;
    __ubuf__ float *sortTmp;
    __ubuf__ float *sortMrgTmp;
    __ubuf__ float *weightsTopK;
    __ubuf__ uint32_t *indicesTopK;
    __ubuf__ Dtype *socresInTmp;
    __ubuf__ Dtype *weightsOutTmp;
};

#define SIGMOID_TOPK_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void sigmoid_topk_##dtype(GM_ADDR socres, GM_ADDR indices, \
                                                           GM_ADDR bias, float scale, \
                                                           GM_ADDR weightsMap, GM_ADDR routingMap, \
                                                           uint32_t numTokens, uint32_t numRoutedExperts, \
                                                           uint32_t topK, bool normTopKProb) \
{ \
    SigmoidTopK<dtype> op; \
    op.Init(socres, indices, bias, scale, weightsMap, routingMap, \
            numTokens, numRoutedExperts, topK, normTopKProb); \
    op.Run(); \
}
#else
#define SIGMOID_TOPK_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void sigmoid_topk_##dtype(GM_ADDR socres, GM_ADDR indices, \
                                                           GM_ADDR bias, float scale, \
                                                           GM_ADDR weightsMap, GM_ADDR routingMap, \
                                                           uint32_t numTokens, uint32_t numRoutedExperts, \
                                                           uint32_t topK, bool normTopKProb) \
{ \
}
#endif
