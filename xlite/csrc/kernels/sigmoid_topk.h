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

// #define XLITE_KERNEL_DEBUG
#include "debug.h"

#ifdef __DAV_C220_VEC__
constexpr uint32_t BIT_SIZE_OF_U32 = 32;
constexpr uint64_t SORT_RESULT_BLOCK_SIZE = SORT_BLOCK_SIZE * 2;

static __aicore__ inline void DumpBufferIndex(__ubuf__ float *buf, const __gm__ char *name,
                                              int size, int step = 1)
{
    DumpBuffer(buf, name, size, step, 1, true);
}

template <typename Dtype>
class SigmoidTopK
{
public:
    __aicore__ inline SigmoidTopK() = default;

    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR indices, GM_ADDR bias, float scale,
                                GM_ADDR weightsMap, GM_ADDR routingMap, uint32_t numTokens,
                                uint32_t numRoutedExperts, uint32_t nGroup, uint32_t nTopkGroup,
                                uint32_t topK, bool normTopKProb)
    {
        set_mask_norm();
        this->scoresGm = (__gm__ Dtype *)scores;
        this->biasGm = (__gm__ float *)bias;
        this->weightsMapGm = (__gm__ Dtype *)weightsMap;
        this->indicesGm = (__gm__ uint32_t *)indices;
        this->routingMapGm = (__gm__ uint32_t *)routingMap;

        this->nTokens = numTokens;
        this->nRoutedExperts = numRoutedExperts;
        this->nGroup = nGroup;
        this->nTopkGroup = nTopkGroup;
        this->nExpertsPerGroup = numRoutedExperts / nGroup;
        this->nScores = nGroup == 1 ? numRoutedExperts : (nTopkGroup * nExpertsPerGroup);
        // SelectTopK() unconditionally 4-way merges (vmrgsort4), so the bitonic sort needs at
        // least 4 * SORT_BLOCK_SIZE candidates. Models with few routed experts -- GLM-4.7-Flash
        // has 64 with nGroup=1, so nScores=64 -- would otherwise underflow `tailLen` and make
        // vmrgsort4 read and write past sortTmp/sortMrgTmp. Pad the sort width up and size every
        // buffer the sort touches for the padded width. For all previously supported shapes
        // sortPad <= pad, so the layout is unchanged (DeepSeek-V3: nScores=128 -> no regression).
        this->nSortScores = MAX(nScores, (uint32_t)(4 * SORT_BLOCK_SIZE));
        this->topK = topK;
        this->normTopKProb = normTopKProb;
        this->scale = scale;
        this->calcRepeat = DIV_ROUND_UP(nRoutedExperts, VECTOR_MAX_NUM_OF_FP32);

        uint32_t pad = ROUND_UP(numRoutedExperts, VECTOR_MAX_NUM_OF_FP32) * sizeof(float);
        uint32_t sortPad = ROUND_UP(nSortScores, VECTOR_MAX_NUM_OF_FP32) * sizeof(float);
        // indicesIn is DMA'd as numRoutedExperts entries but consumed as nSortScores sort keys,
        // so it (and the other sort-touched buffers) must cover the wider of the two.
        uint32_t idxPad = MAX(pad, sortPad);
        uint32_t padDtype =
            ROUND_UP(numRoutedExperts, VECTOR_MAX_BYTESIZE / sizeof(Dtype)) * sizeof(Dtype);
        uint64_t off = 0;
        scoresIn = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += pad;
        biasIn = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += pad;
        indicesIn = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += idxPad;
        routingMapOut = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += pad;
        weightsOut = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += pad;
        calc_unbiased = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += pad;
        calc = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += pad;
        this->scores = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += idxPad;
        this->indices = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += idxPad;
        groupScores = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        groupIndices = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        reduceTmp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += pad;
        sortTmp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 2 * idxPad;
        sortMrgTmp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 2 * idxPad;
        weightsTopK = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        indicesTopK = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            off += pad;
            scoresInTmp = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += padDtype;
            weightsOutTmp = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        }
    }

    __aicore__ inline void Run()
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        copy_gm_to_ubuf_align_b32(indicesIn, indicesGm, 0, 1, nRoutedExperts * sizeof(uint32_t), 0,
                                  0, 0, 0);
        pipe_barrier(PIPE_ALL);
        if (biasGm == nullptr) {
            // Models with no e_score_correction_bias (GLM-4.7-Flash routes on plain sigmoid) pass
            // a null bias. Zero the UB bias once so the unconditional bias adds in
            // CalcGroupScores() stay exact no-ops instead of DMA'ing an invalid GM address.
            vector_dup(biasIn, float(0), calcRepeat, 1, 0, 8, 0);
            pipe_barrier(PIPE_V);
        }
        InitSortTail();
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        for (int tokenIdx = block_idx; tokenIdx < nTokens; tokenIdx += block_num) {
            InitOutBuf();
            CopyInScores(tokenIdx);
            CalcSigmoid();
            CalcGroupScores();
            SelectTopK();
            FillOutMap();
            CopyOutMap(tokenIdx);
        }
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }

    // vmrgsort4 always merges 4 blocks, so when nScores < 4 * SORT_BLOCK_SIZE the sort width is
    // padded up to nSortScores. The score/indices padding lanes are pinned once here with
    // FLOAT_MIN / index 0: they always sort last, can never claim a top-K slot, and never feed
    // vgather an out-of-range index. The nGroup > 1 gather loop writes exactly nScores lanes, so
    // this one-time fill holds for the whole run; the nGroup == 1 vadd, however, spans
    // calcRepeat * VECTOR_MAX_NUM_OF_FP32 lanes and for a non-64-multiple nRoutedExperts (e.g. 96
    // -> 128 lanes) rewrites the padding every token -- PadScoresTail() re-pins it per token.
    // FLOAT_MIN (finite -3.4e38) rather than -inf, matching the padding convention already used by
    // topk.h and by groupScores above, so vbitsort's bitwise ordering stays well defined.
    // indicesIn is the sort key vector when nGroup == 1; the DMA in Run() only filled
    // nRoutedExperts entries, so its padding is zeroed here once (it is never rewritten).
    __aicore__ inline void InitSortTail()
    {
        if (nSortScores <= nScores) {
            return;
        }
        for (uint32_t i = nScores; i < nSortScores; i++) {
            *(scores + i) = FLOAT_MIN;
            *(indices + i) = 0;
        }
        for (uint32_t i = nRoutedExperts; i < nSortScores; i++) {
            *(indicesIn + i) = 0;
        }
        pipe_barrier(PIPE_ALL);
    }

    // nGroup == 1: CalcGroupScores()'s vadd rewrites `scores` every token over calcRepeat repeats,
    // which for a non-64-multiple nRoutedExperts reaches into the sort padding and clobbers the
    // one-time FLOAT_MIN fill from InitSortTail(). Re-pin those lanes per token, fenced against
    // the V-pipe writes (leading barrier) and before the V-pipe sort reads them (trailing
    // barrier). nGroup > 1 shapes never rewrite the padding and nSortScores <= nScore shapes have
    // none at all: both early-return at zero cost.
    __aicore__ inline void PadScoresTail()
    {
        if (nGroup != 1 || nSortScores <= nScores) {
            return;
        }
        pipe_barrier(PIPE_ALL);
        for (uint32_t i = nScores; i < nSortScores; i++) {
            *(scores + i) = FLOAT_MIN;
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void InitOutBuf()
    {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        vector_dup(routingMapOut, uint32_t(0), 1, 1, 0, 8, 0);
        pipe_barrier(PIPE_V);
        DumpBuffer(routingMapOut, "routingMapOut [clean]", 5, 1, 0, true);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        vector_dup(weightsOut, float(0), calcRepeat, 1, 0, 8, 0);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void CopyInScores(int tokenIdx)
    {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        if constexpr (std::is_same<Dtype, float>::value) {
            copy_gm_to_ubuf_align_b32(scoresIn, scoresGm + tokenIdx * nRoutedExperts, 0, 1,
                                      nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
            pipe_barrier(PIPE_V);
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            copy_gm_to_ubuf_align_b32(scoresInTmp, scoresGm + tokenIdx * nRoutedExperts, 0, 1,
                                      nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
            pipe_barrier(PIPE_V);
        }
        if (biasGm != nullptr) {
            copy_gm_to_ubuf_align_b32(biasIn, biasGm, 0, 1, nRoutedExperts * sizeof(float), 0, 0, 0,
                                      0);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void CalcSigmoid()
    {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            uint64_t vector_bf162fp32_config = set_vector_1src_xt(8, 4, 1, 1, calcRepeat);
            vconv_bf162f32(scoresIn, scoresInTmp, vector_bf162fp32_config);
            pipe_barrier(PIPE_V);
        }

        // 0 - x
        vector_dup(calc, float(0), calcRepeat, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vsub(calc, calc, scoresIn, calcRepeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // 提前设置标志，允许 MTE2 与后续计算并行
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        // 以下计算与下一轮数据搬入并行执行
        vexp(calc, calc, calcRepeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // 1 + exp(-x)
        // 修复：dstRepStride 从 0 改为 8
        vector_dup(reduceTmp, float(1), calcRepeat, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(calc, reduceTmp, calc, calcRepeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // 1 / (1 + exp(-x)) —— 复用 reduceTmp（仍为 1），无需再次填充
        vdiv(calc_unbiased, reduceTmp, calc, calcRepeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        DumpBuffer(calc_unbiased, "calc_unbiased", 160, 1, 0, false);
    }

    __aicore__ inline void CalcGroupScores()
    {
        if (nGroup == 1) {
            // Add bias
            vadd(scores, calc_unbiased, biasIn, calcRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            DumpBuffer(scores, "scores", 160, 1, 0, false);
            return;
        }
        float min = -3.4028235e+38;
        // Add bias
        vadd(calc, calc_unbiased, biasIn, calcRepeat, 1, 1, 1, 8, 8, 8);
        vector_dup(groupScores, min, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        for (int gIdx = 0; gIdx < nGroup; gIdx++) {
            __ubuf__ float *myGroupScores = calc + gIdx * nExpertsPerGroup;
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            vbitsort(sortTmp, myGroupScores, indicesIn,
                     DIV_ROUND_UP(nExpertsPerGroup, SORT_BLOCK_SIZE));
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            *(groupScores + gIdx) = sortTmp[0] + sortTmp[2];
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        }
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);

        // select nTopkGroup
        vbitsort(sortTmp, groupScores, indicesIn, DIV_ROUND_UP(nTopkGroup, SORT_BLOCK_SIZE));
        pipe_barrier(PIPE_V);
        SetMask(nTopkGroup);
        vreducev2(groupIndices, (__ubuf__ uint32_t *)sortTmp, (__ubuf__ uint32_t *)sortTmp, 1, 1, 2,
                  8, 0);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);

        int block = DIV_ROUND_UP(nExpertsPerGroup * sizeof(float), BLOCK_SIZE);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (int i = 0; i < nTopkGroup; i++) {
            uint32_t gIdx = groupIndices[i];
            copy_ubuf_to_ubuf(scores + i * nExpertsPerGroup, calc + gIdx * nExpertsPerGroup, 0, 1,
                              block, 0, 0);
            copy_ubuf_to_ubuf(indices + i * nExpertsPerGroup, indicesIn + gIdx * nExpertsPerGroup,
                              0, 1, block, 0, 0);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void SelectTopK()
    {
        // 排序前把补齐段重钉成 FLOAT_MIN（nGroup==1 且专家数非 64 整数倍时 vadd 会覆写它）
        PadScoresTail();
        vbitsort(sortTmp, scores, nGroup > 1 ? indices : indicesIn, nSortScores / SORT_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
        uint64_t mrgRepeat = 1;
        uint64_t validBits = 0xF;
        uint64_t ifExhaustedSuspension = 0;
        uint64_t config = mrgRepeat | validBits << MGR_SORT_VALID_BITS_OFFSET |
                          ifExhaustedSuspension << MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET;
        uint64_t lengths =
            SORT_BLOCK_SIZE | SORT_BLOCK_SIZE << 16 | SORT_BLOCK_SIZE << 32 | SORT_BLOCK_SIZE << 48;
        __ubuf__ float *addrArray[4] = {sortTmp, sortTmp + SORT_RESULT_BLOCK_SIZE,
                                        sortTmp + 2 * SORT_RESULT_BLOCK_SIZE,
                                        sortTmp + 3 * SORT_RESULT_BLOCK_SIZE};
        vmrgsort4(sortMrgTmp, addrArray, lengths, config);
        pipe_barrier(PIPE_V);

        // nSortScores >= 4 * SORT_BLOCK_SIZE by construction, so this cannot underflow.
        uint64_t nSortBlocks = nSortScores / SORT_BLOCK_SIZE;
        uint64_t tailLen = nSortBlocks > 4 ? nSortBlocks - 4 : 0;

        DumpBuffer(sortTmp, "sortTmp 0", 160, 2);

        copy_ubuf_to_ubuf(sortTmp, sortMrgTmp, 0, 1,
                          DIV_ROUND_UP(4 * SORT_RESULT_BLOCK_SIZE * sizeof(uint32_t), BLOCK_SIZE),
                          0, 0);
        pipe_barrier(PIPE_V);
        DumpBuffer(sortMrgTmp, "sortMrgTmp 1", 160, 2);

        for (int i = 4; i < 4 + tailLen; i++) {
            validBits = 3;  // The first two queues are valid.
            addrArray[1] = sortTmp + i * SORT_RESULT_BLOCK_SIZE;
            lengths = SORT_BLOCK_SIZE * i | SORT_BLOCK_SIZE << 16;
            config = mrgRepeat | validBits << MGR_SORT_VALID_BITS_OFFSET |
                     ifExhaustedSuspension << MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET;

            vmrgsort4(sortMrgTmp, addrArray, lengths, config);
            pipe_barrier(PIPE_V);
            copy_ubuf_to_ubuf(
                sortTmp, sortMrgTmp, 0, 1,
                DIV_ROUND_UP((i + 1) * SORT_RESULT_BLOCK_SIZE * sizeof(uint32_t), BLOCK_SIZE), 0,
                0);
            pipe_barrier(PIPE_V);
            DumpBuffer(sortMrgTmp, "sortMrgTmp 3", 160, 2);
            DumpBuffer(sortMrgTmp, "sortMrgTmp 3", 160, 2, 1, true);
        }

        vreducev2(indicesTopK, (__ubuf__ uint32_t *)sortMrgTmp, (__ubuf__ uint32_t *)sortMrgTmp, 1,
                  1, 2, 8, 0);
        pipe_barrier(PIPE_V);
        DumpBuffer(indicesTopK, "indicesTopK 4", 160, 1, 0, true);

        vmuls((__ubuf__ int32_t *)sortMrgTmp, (__ubuf__ int32_t *)indicesTopK, 4, 1, 1, 0, 0, 0);
        pipe_barrier(PIPE_V);
        vgather((__ubuf__ uint32_t *)weightsTopK, (__ubuf__ uint32_t *)sortMrgTmp,
                (uint64_t)calc_unbiased, 0, 1);
        pipe_barrier(PIPE_V);

        ReduceSum(reduceTmp, weightsTopK, topK);
        vbrcb((__ubuf__ uint32_t *)reduceTmp, (__ubuf__ uint32_t *)reduceTmp, 0, 0, 1);
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
            bitmapSet((__ubuf__ uint64_t *)routingMapOut, idx);
            *(weightsOut + idx) = *(weightsTopK + i);
        }
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            uint64_t vector_fp322bf16_config = set_vector_1src_xt(4, 8, 1, 1, calcRepeat);
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
        copy_ubuf_to_gm_align_b32(routingMapGm + tokenIdx * nRoutedExperts / BIT_SIZE_OF_U32,
                                  routingMapOut, 0, 1,
                                  nRoutedExperts * sizeof(uint32_t) / BIT_SIZE_OF_U32, 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if constexpr (std::is_same<Dtype, float>::value) {
            copy_ubuf_to_gm_align_b32(weightsMapGm + tokenIdx * nRoutedExperts, weightsOut, 0, 1,
                                      nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            copy_ubuf_to_gm_align_b32(weightsMapGm + tokenIdx * nRoutedExperts, weightsOutTmp, 0, 1,
                                      nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }

private:
    uint32_t nTokens;
    uint32_t nRoutedExperts;
    uint32_t nGroup;
    uint32_t nTopkGroup;
    uint32_t nExpertsPerGroup;
    uint32_t nScores;
    uint32_t nSortScores;  // nScores padded up to >= 4 * SORT_BLOCK_SIZE for vmrgsort4
    uint32_t topK;
    bool normTopKProb;
    float scale;
    int calcRepeat;

    __gm__ Dtype *scoresGm;
    __gm__ float *biasGm;
    __gm__ Dtype *weightsMapGm;
    __gm__ uint32_t *indicesGm;
    __gm__ uint32_t *routingMapGm;

    __ubuf__ float *scoresIn;
    __ubuf__ float *biasIn;
    __ubuf__ uint32_t *indicesIn;
    __ubuf__ uint32_t *routingMapOut;
    __ubuf__ float *weightsOut;
    __ubuf__ float *calc;
    __ubuf__ float *calc_unbiased;
    __ubuf__ float *scores;
    __ubuf__ uint32_t *indices;
    __ubuf__ float *groupScores;
    __ubuf__ uint32_t *groupIndices;
    __ubuf__ float *reduceTmp;
    __ubuf__ float *sortTmp;
    __ubuf__ float *sortMrgTmp;
    __ubuf__ float *weightsTopK;
    __ubuf__ uint32_t *indicesTopK;
    __ubuf__ Dtype *scoresInTmp;
    __ubuf__ Dtype *weightsOutTmp;
};

#define SIGMOID_TOPK_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void sigmoid_topk_##dtype(                                    \
        GM_ADDR scores, GM_ADDR indices, GM_ADDR bias, float scale, GM_ADDR weightsMap,            \
        GM_ADDR routingMap, uint32_t numTokens, uint32_t numRoutedExperts, uint32_t nGroup,        \
        uint32_t nTopkGroup, uint32_t topK, bool normTopKProb)                                     \
    {                                                                                              \
        SigmoidTopK<dtype> op;                                                                     \
        op.Init(scores, indices, bias, scale, weightsMap, routingMap, numTokens, numRoutedExperts, \
                nGroup, nTopkGroup, topK, normTopKProb);                                           \
        op.Run();                                                                                  \
    }
#else
#define SIGMOID_TOPK_FUNC_DEFINE(dtype)                                                     \
    extern "C" __global__ __aicore__ void sigmoid_topk_##dtype(                             \
        GM_ADDR scores, GM_ADDR indices, GM_ADDR bias, float scale, GM_ADDR weightsMap,     \
        GM_ADDR routingMap, uint32_t numTokens, uint32_t numRoutedExperts, uint32_t nGroup, \
        uint32_t nTopkGroup, uint32_t topK, bool normTopKProb)                              \
    {                                                                                       \
    }
#endif
