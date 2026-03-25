/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <tuple>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "model.h"

#define KVOFFSET_OF_QKBUFFER 4096  // the kv offset of qk buffer in different cycles of mla prefill

XModel::XModel(struct XModelConfig &c, uint32_t rankId) : _c(c), _rankId(rankId)
{
    attnNorm.resize(c.nLayers);
    attnOut.resize(c.nLayers);
    mhaQKV.resize(c.nLayers);
    mhaQKVBias.resize(c.nLayers);
    mhaQNorm.resize(c.nLayers);
    mhaKNorm.resize(c.nLayers);
    mlaQA.resize(c.nLayers);
    mlaQB.resize(c.nLayers);
    mlaQNorm.resize(c.nLayers);
    mlaKVA.resize(c.nLayers);
    mlaKVB.resize(c.nLayers);
    mlaKVNorm.resize(c.nLayers);
    mlpNorm.resize(c.nLayers);
    mlpUpGate.resize(c.nDenseLayers);
    mlpDown.resize(c.nDenseLayers);
    moeGate.resize(c.nLayers);
    moeGateBias.resize(c.nLayers);
    moeSEUpGate.resize(c.nLayers);
    moeSEDown.resize(c.nLayers);
    moeREUpGate.resize(c.nLayers);
    moeREUpGateScale.resize(c.nLayers);
    moeREDown.resize(c.nLayers);
    moeREDownScale.resize(c.nLayers);
    _moeREUpGate.resize(c.nLayers);
    _moeREUpGateScale.resize(c.nLayers);
    _moeREDown.resize(c.nLayers);
    _moeREDownScale.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        moeREUpGate[i].resize(c.nRoutedExperts);
        moeREUpGateScale[i].resize(c.nRoutedExperts);
        moeREDown[i].resize(c.nRoutedExperts);
        moeREDownScale[i].resize(c.nRoutedExperts);
    }
}

void XModel::Init(void)
{
    std::vector<uint32_t> gateIdx, vgatherIndices;
    std::vector<uint64_t> weights;
    size_t size;
    void *ptr;

    if (_c.nDenseLayers != _c.nLayers) {
        gateIdx.resize(_c.nRoutedExperts);
        size = _c.nRoutedExperts * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t i = 0; i < _c.nRoutedExperts; i++) {
            gateIdx[i] = i;
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, gateIdx.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _gateIndicts.Init({_c.nRoutedExperts}, INT32, ptr);
    }

    size = _c.nRoutedExperts * XDtypeBit(INT64) / 8;
    for (uint32_t i = _c.nDenseLayers; i < _c.nLayers; i++) {
        weights.resize(_c.nRoutedExperts);
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREUpGate[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREUpGate[i].Init({_c.nRoutedExperts}, INT64, ptr);

        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREUpGateScale[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREUpGateScale[i].Init({_c.nRoutedExperts}, INT64, ptr);

        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREDown[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREDown[i].Init({_c.nRoutedExperts}, INT64, ptr);

        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREDownScale[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREDownScale[i].Init({_c.nRoutedExperts}, INT64, ptr);
    }

    if (_c.attnType == XMODEL_ATTN_MLA) {
        size = _c.ropeHeadDim * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        vgatherIndices.resize(_c.ropeHeadDim);
        for (uint32_t i = 0; i < _c.ropeHeadDim / 2; i++) {
            vgatherIndices[i * 2] = i * 4;
            vgatherIndices[i * 2 + 1] = i * 4 + 1024;
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, vgatherIndices.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _vGather.Init({_c.ropeHeadDim}, INT32, ptr);
    }

    if (_c.attnType == XMODEL_ATTN_MHA) {
        size = _c.maxBatch * _c.nHeads * 512;
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        CHECK_ACL(aclrtMemset(ptr, size, 0, size));
        _a2v.Init({size / 4}, INT32, ptr);

        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        CHECK_ACL(aclrtMemset(ptr, size, 0, size));
        _v2a.Init({size / 4}, INT32, ptr);

        size = AIC_MAX_NUM;
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        CHECK_ACL(aclrtMemset(ptr, size, 0, size));
        _sync.Init({AIC_MAX_NUM}, INT32, ptr);
    }

    _mropeMaskH = 0;
    _mropeMaskW = 0;
    uint32_t mropeSectionH = _c.mropeSection.size() > 1 ? _c.mropeSection[1] : 0;
    uint32_t mropeSectionW = _c.mropeSection.size() > 2 ? _c.mropeSection[2] : 0;
    if (_c.mropeInterleaved) {
        for (uint32_t i = 1; i < mropeSectionH * 3; i += 3) {
            _mropeMaskH |= 1ULL << i;
        }
        for (uint32_t i = 2; i < mropeSectionW * 3; i += 3) {
            _mropeMaskW |= 1ULL << i;
        }
    } else {
        uint32_t sectionStartH = !_c.mropeSection.empty() ? _c.mropeSection[0] : 0;
        uint32_t sectionStartW = sectionStartH + mropeSectionH;
        for (uint32_t i = sectionStartH; i < sectionStartW; i++) {
            _mropeMaskH |= 1ULL << i;
        }
        for (uint32_t i = sectionStartW; i < sectionStartW + mropeSectionW; i++) {
            _mropeMaskW |= 1ULL << i;
        }
    }

    _isSharedExpertWeightFull =
        (_c.nSharedExperts != 0 && !moeSEUpGate[_c.nDenseLayers].shape.empty() &&
         moeSEUpGate[_c.nDenseLayers].shape[0] == _c.moeIntermediateSize * 2 &&
         moeSEDown[_c.nDenseLayers].shape.size() >= 2 &&
         moeSEDown[_c.nDenseLayers].shape[1] == _c.moeIntermediateSize);
}

XModel::~XModel(void)
{
    if (_c.nDenseLayers != _c.nLayers) {
        (void)aclrtFree(_gateIndicts.ptr);
    }
    for (uint32_t i = _c.nDenseLayers; i < _c.nLayers; i++) {
        (void)aclrtFree(_moeREUpGate[i].ptr);
        (void)aclrtFree(_moeREUpGateScale[i].ptr);
        (void)aclrtFree(_moeREDown[i].ptr);
        (void)aclrtFree(_moeREDownScale[i].ptr);
    }

    if (_c.attnType == XMODEL_ATTN_MLA) {
        (void)aclrtFree(_vGather.ptr);
    }

    if (_c.attnType == XMODEL_ATTN_MHA) {
        (void)aclrtFree(_a2v.ptr);
        (void)aclrtFree(_v2a.ptr);
        (void)aclrtFree(_sync.ptr);
    }
}

void XModel::ForwardParallelEmbed(XRuntime &rt, XTensor &input, XTensor &embed, XTensor &output)
{
    uint32_t vocabPerTp = DIV_ROUND_UP(_c.vocabSize, _c.defTpSize);
    uint32_t id = _rankId % _c.defTpSize;
    uint32_t start = id * vocabPerTp;
    uint32_t end = start + vocabPerTp;

    XliteOpEmbed(rt, input, embed, start, end, output);
    if (_c.defTpSize > 1) {
        XliteOpAllReduceSum(rt, output, output, TP);
    }
}

std::tuple<XTensor &, XTensor &, XTensor &> XModel::ForwardAttnMLACommon(
    XRuntime &rt, uint32_t layer, std::vector<std::pair<XTensor, XTensor>> &kvCache,
    XTensor &freqsCis, XTensor &hiddenState)
{
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;
    XTensor &kCache = kvCache[layer].first;
    XTensor &vCache = kvCache[layer].second;
    XTensor &attnQc = rt.pool->GetTensor({rt._realM, _c.qLoraRank}, hiddenState.dtype, DBG_LOC);
    XTensor &attnNormQc = rt.pool->GetTensor({rt._realM, _c.qLoraRank}, hiddenState.dtype, DBG_LOC);
    XTensor &attnQWithQr = rt.pool->GetTensor(
        {rt._realM, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &attnKvc =
        rt.pool->GetTensor({rt._realM, _c.kvLoraRank + _c.ropeHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &attnNormKvc =
        rt.pool->GetTensor({rt._realM, _c.kvLoraRank}, hiddenState.dtype, DBG_LOC);
    XTensor &attnKPe = rt.pool->GetTensor({rt._realM, _c.ropeHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &attnQPe =
        rt.pool->GetTensor({rt._realM, nLocalHeads, _c.ropeHeadDim}, hiddenState.dtype, DBG_LOC);

    XliteOpMatmul(rt, hiddenState, mlaQA[layer], attnQc, _c.weightNZ);
    XliteOpRmsNorm(rt, attnQc, mlaQNorm[layer], attnNormQc, _c.normEps, attnQc.shape[1]);
    XliteOpMatmul(rt, attnNormQc, mlaQB[layer], attnQWithQr, _c.weightNZ);
    XliteOpRopeComplex(rt, rt._realM, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim, _c.ropeHeadDim,
                       attnQWithQr, freqsCis, rt._attnPosition, _vGather, attnQPe,
                       rt._prefillBatch > 0 ? MIX : NORMAL);
    XliteOpMatmul(rt, hiddenState, mlaKVA[layer], attnKvc, _c.weightNZ);
    XliteOpRmsNorm(rt, attnKvc, mlaKVNorm[layer], attnNormKvc, _c.normEps, _c.kvLoraRank);
    XliteOpRopeComplexAndCache(rt, rt._realM, 1, _c.kvLoraRank + _c.ropeHeadDim, _c.ropeHeadDim,
                               attnKvc, freqsCis, rt._attnPosition, _vGather, attnKPe, NORMAL,
                               _c.blockSize, attnNormKvc, kCache, vCache, rt._attnSlotMapping);

    rt.pool->PutTensor(attnQc);
    rt.pool->PutTensor(attnNormQc);
    rt.pool->PutTensor(attnKvc);
    rt.pool->PutTensor(attnNormKvc);

    return {attnQWithQr, attnKPe, attnQPe};
}

XTensor &XModel::ForwardAttnMLAPrefillV2(XRuntime &rt, uint32_t layer,
                                         std::vector<std::pair<XTensor, XTensor>> &kvCache,
                                         XTensor &freqsCis, XTensor &attnQWithQr)
{
    XTensor &kCache = kvCache[layer].first;
    XTensor &vCache = kvCache[layer].second;
    uint32_t qHeads = _c.nHeads / _c.defTpSize;
    XTensor &output =
        rt.pool->GetTensor({attnQWithQr.shape[0], qHeads, _c.vHeadDim}, attnQWithQr.dtype, DBG_LOC);
    XTensor &qk = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, TILESIZE_OF_CACHED_KV},
                                     attnQWithQr.dtype, DBG_LOC);
    XTensor &sv = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, _c.vHeadDim},
                                     attnQWithQr.dtype, DBG_LOC);
    XTensor &max = rt.pool->GetTensor({rt.aivNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
    XTensor &sum = rt.pool->GetTensor({rt.aivNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
    XTensor &lastMax = rt.pool->GetTensor({attnQWithQr.shape[0], qHeads}, FP32, DBG_LOC);
    XTensor &lastSum = rt.pool->GetTensor({attnQWithQr.shape[0], qHeads}, FP32, DBG_LOC);
    XliteOpFlashMLA(rt, attnQWithQr, kCache, vCache, mlaKVB[layer], qk, sv, max, sum, lastMax,
                    lastSum, _sync, output, rt._cumPromptLens, rt._lens, rt._cachedLens,
                    rt._attnBlockTables, qHeads, _c.ropeHeadDim, _c.nopeHeadDim, _c.vHeadDim,
                    _c.kvLoraRank, _c.blockSize, rt._batch, rt._maxNumBlocks, _c.softmaxScale);
    rt.pool->PutTensor(lastSum);
    rt.pool->PutTensor(lastMax);
    rt.pool->PutTensor(sum);
    rt.pool->PutTensor(max);
    rt.pool->PutTensor(sv);
    rt.pool->PutTensor(qk);
    return output;
}

XTensor &XModel::ForwardAttnMLAPrefill(XRuntime &rt, uint32_t layer,
                                       std::vector<std::pair<XTensor, XTensor>> &kvCache,
                                       XTensor &freqsCis, XTensor &hiddenState,
                                       XTensor &attnQWithQr, XTensor &attnKPe, XTensor &attnQPe)
{
    XTensor &kCache = kvCache[layer].first;
    XTensor &vCache = kvCache[layer].second;
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;
    uint32_t mlaPaddingLen = ROUND_UP(rt._realM, _c.blockSize);
    uint32_t mlaPmSize = TILESIZE_OF_QUERY, mlaPerKvlen = KVOFFSET_OF_QKBUFFER, coreNum = rt.aicNum;
    XTensor &attnQkcAbsorb =
        rt.pool->GetTensor({rt._realM, nLocalHeads, _c.vHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &attnKRes = rt.pool->GetTensor(
        {mlaPaddingLen, nLocalHeads * (_c.nopeHeadDim + _c.vHeadDim)}, hiddenState.dtype, DBG_LOC);
    XTensor &attnKvFull = rt.pool->GetTensor(
        {mlaPaddingLen, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &attnV =
        rt.pool->GetTensor({mlaPaddingLen, nLocalHeads, _c.vHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &attnPrefillOut =
        rt.pool->GetTensor({rt._realM, nLocalHeads, _c.vHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &attnMlaOut =
        rt.pool->GetTensor({coreNum * mlaPmSize * _c.vHeadDim * 2}, FP32, DBG_LOC);
    XTensor &attnMlaMax = rt.pool->GetTensor({coreNum * mlaPmSize * 8}, FP32, DBG_LOC);
    XTensor &attnMlaSum = rt.pool->GetTensor({coreNum * mlaPmSize * 8 * 2}, FP32, DBG_LOC);
    XTensor &attnMlaAlpha = rt.pool->GetTensor({coreNum * mlaPmSize * 8 * 2}, FP32, DBG_LOC);
    XTensor &attnMlaQK = rt.pool->GetTensor({2, coreNum, mlaPmSize, mlaPerKvlen}, BF16, DBG_LOC);
    XTensor &attnMlaTmp = rt.pool->GetTensor({1}, INT32, DBG_LOC);

    XliteDsOpKvMatmul(rt, kCache, mlaKVB[layer], attnKRes, static_cast<int>(rt._prefillLenPad),
                      static_cast<int>(nLocalHeads * (_c.nopeHeadDim + _c.vHeadDim)),
                      static_cast<int>(_c.kvLoraRank), rt._attnBlockTables, true,
                      static_cast<int>(_c.blockSize), static_cast<int>(_c.kvLoraRank));
    XliteDsOpPrefillKvSplit(rt, attnKRes, attnKPe, vCache, rt._attnBlockTables, attnKvFull, attnV,
                            static_cast<int>(rt._prefillLen), static_cast<int>(rt._prefillLenPad),
                            static_cast<int>(nLocalHeads), 0, static_cast<int>(_c.ropeHeadDim),
                            static_cast<int>(_c.nopeHeadDim), static_cast<int>(_c.vHeadDim),
                            _c.blockSize);
    XliteDsOpPrefillMix(rt, attnMlaOut, attnMlaAlpha, attnMlaMax, attnMlaSum, attnQWithQr,
                        attnKvFull, attnMlaQK, attnMlaTmp, rt._cachedLens, attnV, attnPrefillOut,
                        attnQkcAbsorb, rt._lens, attnMlaTmp, attnMlaTmp, attnMlaTmp, rt._prefillIdx,
                        rt._cumPromptLens, _c.vHeadDim, nLocalHeads, nLocalHeads, _c.blockSize,
                        rt._prefillBatch, 0, 0, 0, 0, _c.softmaxScale);

    rt.pool->PutTensor(attnQWithQr);
    rt.pool->PutTensor(attnKPe);
    rt.pool->PutTensor(attnQPe);
    rt.pool->PutTensor(attnKRes);
    rt.pool->PutTensor(attnKvFull);
    rt.pool->PutTensor(attnV);
    rt.pool->PutTensor(attnPrefillOut);
    rt.pool->PutTensor(attnMlaOut);
    rt.pool->PutTensor(attnMlaMax);
    rt.pool->PutTensor(attnMlaSum);
    rt.pool->PutTensor(attnMlaAlpha);
    rt.pool->PutTensor(attnMlaQK);
    rt.pool->PutTensor(attnMlaTmp);
    return attnQkcAbsorb;
}

XTensor &XModel::ForwardAttnMLADecode(XRuntime &rt, uint32_t layer,
                                      std::vector<std::pair<XTensor, XTensor>> &kvCache,
                                      XTensor &freqsCis, XTensor &hiddenState, XTensor &attnQWithQr,
                                      XTensor &attnKPe, XTensor &attnQPe)
{
    uint32_t batch = hiddenState.shape[0];
    XTensor &kCache = kvCache[layer].first;
    XTensor &vCache = kvCache[layer].second;
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;
    XTensor &attnQkcAbsorb =
        rt.pool->GetTensor({rt._realM, nLocalHeads, _c.vHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &attnQAbsorb =
        rt.pool->GetTensor({rt._realM, nLocalHeads, _c.kvLoraRank}, hiddenState.dtype, DBG_LOC);
    XTensor &attnQk =
        rt.pool->GetTensor({batch, nLocalHeads, _c.maxSeqLen}, hiddenState.dtype, DBG_LOC);
    XTensor &attnQkc =
        rt.pool->GetTensor({rt._realM, nLocalHeads, _c.kvLoraRank}, hiddenState.dtype, DBG_LOC);

    XliteDsOpEinsumShdHdcShc(rt, static_cast<int>(rt._realM), static_cast<int>(_c.nopeHeadDim),
                             static_cast<int>(nLocalHeads),
                             static_cast<int>(_c.nopeHeadDim + _c.ropeHeadDim),
                             static_cast<int>(2 * _c.nopeHeadDim), static_cast<int>(_c.kvLoraRank),
                             attnQWithQr, mlaKVB[layer], attnQAbsorb);
    XliteDsOpDecodeAttn(rt, attnQAbsorb, kCache, attnQk, rt._cachedLens, rt._attnBlockTables,
                        rt._lens, rt._cumPromptLens, batch, nLocalHeads, 1, _c.kvLoraRank,
                        _c.blockSize, rt._maxNumBlocks, _c.maxSeqLen, false);
    XliteDsOpDecodeAttn(rt, attnQPe, vCache, attnQk, rt._cachedLens, rt._attnBlockTables, rt._lens,
                        rt._cumPromptLens, batch, nLocalHeads, 1, _c.ropeHeadDim, _c.blockSize,
                        rt._maxNumBlocks, _c.maxSeqLen, true);
    XliteDsOpSoftmax(rt, attnQk, rt._cachedLens, rt._lens, rt._cumPromptLens, _c.softmaxScale,
                     batch, nLocalHeads, _c.blockSize, _c.maxSeqLen);
    XliteDsOpEinsumShtTcShc(rt, static_cast<int>(batch), static_cast<int>(nLocalHeads),
                            static_cast<int>(_c.maxSeqLen), static_cast<int>(rt._maxNumBlocks),
                            static_cast<int>(kCache.shape[0]), static_cast<int>(_c.blockSize),
                            static_cast<int>(_c.kvLoraRank), attnQk, rt._cachedLens, rt._lens,
                            rt._cumPromptLens, rt._attnBlockTables, kCache, attnQkc);
    XliteDsOpEinsumShcHdcShd(rt, static_cast<int>(rt._realM), static_cast<int>(nLocalHeads),
                             static_cast<int>(_c.kvLoraRank), static_cast<int>(2 * _c.nopeHeadDim),
                             static_cast<int>(_c.vHeadDim), attnQkc, mlaKVB[layer], attnQkcAbsorb);

    rt.pool->PutTensor(attnQWithQr);
    rt.pool->PutTensor(attnKPe);
    rt.pool->PutTensor(attnQPe);
    rt.pool->PutTensor(attnQAbsorb);
    rt.pool->PutTensor(attnQk);
    rt.pool->PutTensor(attnQkc);
    return attnQkcAbsorb;
}

void XModel::ForwardAttnMLA(XRuntime &rt, uint32_t layer,
                            std::vector<std::pair<XTensor, XTensor>> &kvCache, XTensor &freqsCis,
                            XTensor &hiddenState)
{
    auto [attnQWithQr, attnKPe, attnQPe] =
        ForwardAttnMLACommon(rt, layer, kvCache, freqsCis, hiddenState);

    XTensor &attnOutput =
        (rt._prefillBatch > 0 ? ForwardAttnMLAPrefill(rt, layer, kvCache, freqsCis, hiddenState,
                                                      attnQWithQr, attnKPe, attnQPe)
                              : ForwardAttnMLADecode(rt, layer, kvCache, freqsCis, hiddenState,
                                                     attnQWithQr, attnKPe, attnQPe));

    XliteOpMatmul(rt, attnOutput, attnOut[layer], hiddenState, _c.weightNZ);

    if (_c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP);
        } else {
            XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP);
        }
    }
    rt.pool->PutTensor(attnOutput);
}

void XModel::ForwardAttnMHA(XRuntime &rt, uint32_t layer,
                            std::vector<std::pair<XTensor, XTensor>> &kvCache, XTensor &freqsCis,
                            XTensor &hiddenState)
{
    uint32_t qHeads = _c.nHeads / _c.defTpSize;
    uint32_t kHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
    XTensor &qkv =
        rt.pool->GetTensor({rt._realM, mhaQKV[layer].shape[0]}, hiddenState.dtype, DBG_LOC);
    if (_c.addBias) {
        XliteOpMatmul(rt, hiddenState, mhaQKV[layer], qkv, _c.weightNZ, mhaQKVBias[layer]);
    } else {
        XliteOpMatmul(rt, hiddenState, mhaQKV[layer], qkv, _c.weightNZ);
    }
    if (_c.qkNorm) {
        XliteOpRmsNorm(rt, qkv, mhaQNorm[layer], qkv, _c.normEps, _c.headDim, qHeads);
        XliteOpRmsNorm(rt, qkv, mhaKNorm[layer], qkv, _c.normEps, _c.headDim, kHeads,
                       qHeads * _c.headDim, qHeads * _c.headDim);
    }
    XliteOpRopeCache(rt, qkv, kvCache[layer].first, kvCache[layer].second, rt._attnPosition,
                     freqsCis, rt._attnSlotMapping, _c.nHeads, _c.nKvHeads, _c.headDim,
                     _c.ropeHeadDim, _c.blockSize, _c.ropeType == XMODEL_ROPE_NEOX, _mropeMaskH,
                     _mropeMaskW);

    XTensor &attn = rt.pool->GetTensor({hiddenState.shape[0], attnOut[layer].shape[1]},
                                       hiddenState.dtype, DBG_LOC);
    if (!rt.enableFlashAttention) {
        XTensor &qk =
            rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, rt._maxNumBlocks * _c.blockSize},
                               hiddenState.dtype, DBG_LOC);
        XliteOpAttention(rt, qkv, kvCache[layer].first, kvCache[layer].second, qk, attn,
                         rt._cumPromptLens, rt._lens, rt._cachedLens, rt._attnBlockTables, qHeads,
                         kHeads, _c.headDim, _c.blockSize, rt._batch, rt._maxNumBlocks);
        rt.pool->PutTensor(qk);
    } else {
        XTensor &qk = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, TILESIZE_OF_CACHED_KV},
                                         hiddenState.dtype, DBG_LOC);
        XTensor &sv = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, _c.headDim},
                                         hiddenState.dtype, DBG_LOC);
        XTensor &max = rt.pool->GetTensor({rt.aivNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.pool->GetTensor({rt.aivNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.pool->GetTensor({qkv.shape[0], qHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.pool->GetTensor({qkv.shape[0], qHeads}, FP32, DBG_LOC);
        XliteOpFlashAttention(rt, qkv, kvCache[layer].first, kvCache[layer].second, qk, sv, max,
                              sum, lastMax, lastSum, _sync, attn, rt._cumPromptLens, rt._lens,
                              rt._cachedLens, rt._attnBlockTables, qHeads, kHeads, _c.headDim,
                              _c.blockSize, rt._batch, rt._maxNumBlocks);
        rt.pool->PutTensor(lastSum);
        rt.pool->PutTensor(lastMax);
        rt.pool->PutTensor(sum);
        rt.pool->PutTensor(max);
        rt.pool->PutTensor(sv);
        rt.pool->PutTensor(qk);
    }

    XliteOpMatmul(rt, attn, attnOut[layer], hiddenState, _c.weightNZ);
    if (_c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP);
        } else {
            XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP);
        }
    }
    rt.pool->PutTensor(qkv);
    rt.pool->PutTensor(attn);
}

void XModel::ForwardAttn(XRuntime &rt, uint32_t layer,
                         std::vector<std::pair<XTensor, XTensor>> &kvCache, XTensor &freqsCis,
                         XTensor &hiddenState)
{
    if (_c.attnType == XMODEL_ATTN_MLA) {
        ForwardAttnMLA(rt, layer, kvCache, freqsCis, hiddenState);
    } else if (_c.attnType == XMODEL_ATTN_MHA) {
        ForwardAttnMHA(rt, layer, kvCache, freqsCis, hiddenState);
    } else {
        throw std::runtime_error(std::string(__func__) + ": TODO");
    }
}

void XModel::ForwardMLP(XRuntime &rt, XTensor &upGate, XTensor &down, XTensor &hiddenState,
                        bool withAllReduce)
{
    uint32_t m = hiddenState.shape[0];
    XTensor &h13 = rt.pool->GetTensor({m, upGate.shape[0]}, hiddenState.dtype, DBG_LOC);
    XTensor &h2 = rt.pool->GetTensor({m, down.shape[1]}, hiddenState.dtype, DBG_LOC);

    XliteOpMatmul(rt, hiddenState, upGate, h13, _c.weightNZ);
    XliteOpSiluAndMul(rt, h13, h2);
    XliteOpMatmul(rt, h2, down, hiddenState, _c.weightNZ);

    if (withAllReduce && _c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP);
        } else {
            XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP);
        }
    }
    rt.pool->PutTensor(h2);
    rt.pool->PutTensor(h13);
}

std::tuple<XTensor &, XTensor &> XModel::ForwardMoEGate(XRuntime &rt, uint32_t layer,
                                                        XTensor &input)
{
    uint32_t m = input.shape[0];
    XTensor &weights = rt.pool->GetTensor({m, _c.nRoutedExperts}, moeGate[layer].dtype, DBG_LOC);
    XTensor &routing = rt.pool->GetTensor({m, _c.nRoutedExperts}, BIT1, DBG_LOC);
    XTensor &scores =
        rt.pool->GetTensor({input.shape[0], _c.nRoutedExperts}, moeGate[layer].dtype, DBG_LOC);

    XliteOpMatmul(rt, input, moeGate[layer], scores, _c.weightNZ);

    if (_c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
        XliteOpSigmoidTopK(rt, scores, _gateIndicts, moeGateBias[layer], _c.routeScale, weights,
                           routing, _c.nExpertGroups, _c.nLimitedGroups, _c.nActExperts,
                           _c.normTopKProb);
    } else {
        XliteOpSoftmaxTopK(rt, scores, _gateIndicts, weights, routing, _c.nActExperts,
                           _c.normTopKProb);
    }

    rt.pool->PutTensor(scores);
    return {weights, routing};
}

std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &> XModel::ForwardMoEDispatch(
    XRuntime &rt, XTensor &tokenSorted, XTensor &weights, XTensor &routing)
{
    uint32_t m = tokenSorted.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;
    XTensor &unpIdx = rt.pool->GetTensor({_c.nRoutedExperts, mAllDp + 1}, INT32, DBG_LOC);
    XTensor &expertsSorted =
        rt.pool->GetTensor({mAllDp * _c.nActExperts, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
    XTensor &expertsCounts = rt.pool->GetTensor({_c.nRoutedExperts, 1}, INT32, DBG_LOC);

    if (_c.defDpSize > 1) {
        XTensor &inputPerDp = tokenSorted, &weightsPerDp = weights, &routingPerDp = routing;
        XTensor &inputAllDp =
            rt.pool->GetTensor({mAllDp, _c.hiddenSize}, inputPerDp.dtype, DBG_LOC);
        XTensor &weightsAllDp =
            rt.pool->GetTensor({mAllDp, _c.nRoutedExperts}, weightsPerDp.dtype, DBG_LOC);
        XTensor &routingAllDp =
            rt.pool->GetTensor({mAllDp, _c.nRoutedExperts}, routingPerDp.dtype, DBG_LOC);
        XliteOpAllGather(rt, inputPerDp, inputAllDp, DP);
        XliteOpAllGather(rt, weightsPerDp, weightsAllDp, DP);
        XliteOpAllGather(rt, routingPerDp, routingAllDp, DP);
        XliteOpPermutation(rt, inputAllDp, routingAllDp, start, end, expertsSorted, unpIdx,
                           expertsCounts);
        rt.pool->PutTensor(routingPerDp);
        rt.pool->PutTensor(weightsPerDp);
        rt.pool->PutTensor(inputAllDp);
        return {weightsAllDp, routingAllDp, unpIdx, expertsSorted, expertsCounts};
    } else {
        XliteOpPermutation(rt, tokenSorted, routing, start, end, expertsSorted, unpIdx,
                           expertsCounts);
        return {weights, routing, unpIdx, expertsSorted, expertsCounts};
    }
}

void XModel::ForwardMOECombine(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                               XTensor &routing, XTensor &unpIdx, XTensor &expertsSorted,
                               XTensor &expertsCounts)
{
    uint32_t m = tokenSorted.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;

    if (_c.defDpSize > 1) {
        XTensor &tokenSortedAllDp =
            rt.pool->GetTensor({mAllDp, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end,
                             tokenSortedAllDp);
        XliteOpReduceScatter(rt, tokenSortedAllDp, tokenSorted, DP);
        rt.pool->PutTensor(tokenSortedAllDp);
    } else {
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, tokenSorted);
    }

    rt.pool->PutTensor(expertsCounts);
    rt.pool->PutTensor(expertsSorted);
    rt.pool->PutTensor(unpIdx);
    rt.pool->PutTensor(routing);
    rt.pool->PutTensor(weights);
}

void XModel::ForwardMoE(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    uint32_t m = hiddenState.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t intermediateSize = _c.moeIntermediateSize / _c.moeTPSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;

    auto [w, r] = ForwardMoEGate(rt, layer, hiddenState);
    auto [weights, routing, unpIdx, expertsSorted, expertsCounts] =
        ForwardMoEDispatch(rt, hiddenState, w, r);

    // routed experts
    XTensor &h13 = rt.pool->GetTensor({mAllDp * _c.nActExperts, intermediateSize * 2},
                                      hiddenState.dtype, DBG_LOC);
    XTensor &h2 =
        rt.pool->GetTensor({mAllDp * _c.nActExperts, intermediateSize}, hiddenState.dtype, DBG_LOC);
    XliteOpGroupMatmul(rt, expertsSorted, _moeREUpGate[layer], _moeREUpGateScale[layer],
                       expertsCounts, start, end, moeREUpGate[layer][start].dtype,
                       intermediateSize * 2, _c.hiddenSize, h13, _c.weightNZ,
                       _c.expertsWeightTrans);
    XliteOpSiluAndMul(rt, h13, h2);
    XliteOpGroupMatmul(rt, h2, _moeREDown[layer], _moeREDownScale[layer], expertsCounts, start, end,
                       moeREDown[layer][start].dtype, _c.hiddenSize, intermediateSize,
                       expertsSorted, _c.weightNZ, _c.expertsWeightTrans);
    rt.pool->PutTensor(h13);
    rt.pool->PutTensor(h2);

    // Check if shared experts should be processed on this rank:
    // 1. Shared experts are enabled (nSharedExperts != 0)
    // 2. Either the weight is not full (all ranks process), or only rank 0 processes when weight is
    // full
    if (_c.nSharedExperts != 0 &&
        ((_isSharedExpertWeightFull && ((rt.rankId() % rt.tpSize()) == 0)) ||
         !_isSharedExpertWeightFull)) {
        XTensor &h = rt.pool->GetTensor({m, _c.hiddenSize}, hiddenState.dtype, DBG_LOC);
        ForwardMOECombine(rt, h, weights, routing, unpIdx, expertsSorted, expertsCounts);
        // share experts
        ForwardMLP(rt, moeSEUpGate[layer], moeSEDown[layer], hiddenState, false);
        XliteOpAdd(rt, hiddenState, h, hiddenState);
        rt.pool->PutTensor(h);
    } else {
        ForwardMOECombine(rt, hiddenState, weights, routing, unpIdx, expertsSorted, expertsCounts);
    }

    if (_c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP);
        } else {
            XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP);
        }
    }
}

void XModel::ForwardFFN(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    if (layer < _c.nDenseLayers) {
        ForwardMLP(rt, mlpUpGate[layer], mlpDown[layer], hiddenState, true);
    } else {
        ForwardMoE(rt, layer, hiddenState);
    }
}

void XModel::ForwardLayersCommOptimize(XRuntime &rt, XTensor &xPad,
                                       std::vector<std::pair<XTensor, XTensor>> &kvCache,
                                       std::vector<XTensor> &deepstackInputEmbeds,
                                       XTensor &freqsCis, XTensor &output)
{
    XTensor xSlice;
    size_t actualM = output.shape[0];
    size_t mPad = xPad.shape[0];
    size_t mPadPerTp = mPad / _c.defTpSize;
    size_t sizePerTp = mPad * _c.hiddenSize / _c.defTpSize * XDtypeBit(embed.dtype) / 8;
    XTensor &hiddenStatePad = rt.pool->GetTensor({mPad, _c.hiddenSize}, embed.dtype, DBG_LOC);
    XTensor x, h;
    x.Init({actualM, _c.hiddenSize}, embed.dtype, xPad.ptr);
    if (actualM == mPad) {
        h.Init({actualM, _c.hiddenSize}, embed.dtype, output.ptr);
        rt.hiddenStatePad.Init({mPad, _c.hiddenSize}, embed.dtype, output.ptr);
    } else {
        h.Init({actualM, _c.hiddenSize}, embed.dtype, hiddenStatePad.ptr);
        rt.hiddenStatePad.Init({mPad, _c.hiddenSize}, embed.dtype, hiddenStatePad.ptr);
    }
    void *slicePtr = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(rt.hiddenStatePad.ptr) +
                                              (rt.rankId() % _c.defTpSize) * sizePerTp);
    rt.hiddenStateSlice.Init({mPadPerTp, hiddenStatePad.shape[1]}, hiddenStatePad.dtype, slicePtr);
    slicePtr = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(xPad.ptr) +
                                        (rt.rankId() % _c.defTpSize) * sizePerTp);
    xSlice.Init({mPadPerTp, xPad.shape[1]}, xPad.dtype, slicePtr);

    for (uint32_t i = 0; i < _c.nLayers; i++) {
        if (i == 0) {
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1]);
        }
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XliteOpAddAndRmsNorm(rt, xSlice, rt.hiddenStateSlice, mlpNorm[i], _c.normEps,
                             rt.hiddenStateSlice);
        XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP);
        if (rt.multiTaskParallel) {
            rt.NotifyWaitPeerStream();
        }
        ForwardFFN(rt, i, h);
        if (i < _c.deepstackNumLevel) {
            XliteOpAdd(rt, h, deepstackInputEmbeds[i], h);
        }
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, xSlice, rt.hiddenStateSlice, attnNorm[i + 1], _c.normEps,
                                 rt.hiddenStateSlice);
            XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP);
        }
        if (rt.multiTaskParallel) {
            rt.NotifyWaitPeerStream();
        }
    }
    XliteOpAddAndRmsNorm(rt, xSlice, rt.hiddenStateSlice, norm, _c.normEps, rt.hiddenStateSlice);
    if (actualM == mPad) {
        XliteOpAllGather(rt, rt.hiddenStateSlice, output, TP);
    } else {
        XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP);
        aclrtMemcpyAsync(output.ptr, actualM * _c.hiddenSize * XDtypeBit(embed.dtype) / 8,
                         rt.hiddenStatePad.ptr,
                         actualM * _c.hiddenSize * XDtypeBit(embed.dtype) / 8,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream);
    }
    rt.pool->PutTensor(hiddenStatePad);
}

void XModel::ForwardLayersNaive(XRuntime &rt, XTensor &x,
                                std::vector<std::pair<XTensor, XTensor>> &kvCache,
                                std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                XTensor &output)
{
    XTensor &h = rt.pool->GetTensor({x.shape[0], _c.hiddenSize}, embed.dtype, DBG_LOC);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        if (i == 0) {
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1]);
        }
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XliteOpAddAndRmsNorm(rt, x, h, mlpNorm[i], _c.normEps, h);
        ForwardFFN(rt, i, h);
        if (i < _c.deepstackNumLevel) {
            XliteOpAdd(rt, h, deepstackInputEmbeds[i], h);
        }
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, x, h, attnNorm[i + 1], _c.normEps, h);
        }
    }
    XliteOpAddAndRmsNorm(rt, x, h, norm, _c.normEps, output);
    rt.pool->PutTensor(h);
}

void XModel::ForwardEmbedAndLayers(XRuntime &rt, XTensor &input,
                                   std::vector<std::pair<XTensor, XTensor>> &kvCache,
                                   std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                   XTensor &h)
{
    if (_c.defTpSize > 1 && (h.shape[0] >= rt.commOptimizeLen || rt.multiTaskParallel)) {
        XTensor &xPad = rt.pool->GetTensor({ROUND_UP(h.shape[0], _c.defTpSize), _c.hiddenSize},
                                           embed.dtype, DBG_LOC);
        XTensor x;
        x.Init({h.shape[0], _c.hiddenSize}, embed.dtype, xPad.ptr);
        ForwardParallelEmbed(rt, input, embed, x);
        rt.enableCommOptimize = true;
        ForwardLayersCommOptimize(rt, xPad, kvCache, deepstackInputEmbeds, freqsCis, h);
        rt.pool->PutTensor(xPad);
    } else {
        XTensor &x = rt.pool->GetTensor({input.shape[0], _c.hiddenSize}, embed.dtype, DBG_LOC);
        ForwardParallelEmbed(rt, input, embed, x);
        rt.enableCommOptimize = false;
        ForwardLayersNaive(rt, x, kvCache, deepstackInputEmbeds, freqsCis, h);
        rt.pool->PutTensor(x);
    }
}

void XModel::ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &output)
{
    uint32_t batch = rt._batch;
    XTensor localOutput({output.shape[1], output.shape[2]}, output.dtype, output.ptr);

    if (batch < input.shape[0]) {
        XTensor &x = rt.pool->GetTensor({batch, _c.hiddenSize}, input.dtype, DBG_LOC);
        XliteOpEmbed(rt, rt._prefillLastIdx, input, 0, rt._realM, x);
        XliteOpMatmul(rt, x, head, localOutput, _c.weightNZ);
        rt.pool->PutTensor(x);
    } else {
        XliteOpMatmul(rt, input, head, localOutput, _c.weightNZ);
    }

    if (_c.defTpSize > 1) {
        XliteOpAllGather(rt, localOutput, output, TP);
    }
}

void XModel::ForwardWithInputsEmbeds(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                     std::vector<std::pair<XTensor, XTensor>> &kvCache,
                                     std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                     XTensor &output)
{
    CheckForwardParam(rt, kvCache);
    rt.PrepareAttn(attnMeta, _c.maxM, _c.maxBatch, _c.maxSeqLen, _c.blockSize, _c.attnType);
    if (_c.defTpSize > 1 && output.shape[0] >= rt.commOptimizeLen) {
        rt.enableCommOptimize = true;
        if (output.shape[0] % _c.defTpSize != 0) {
            XTensor &xPad = rt.pool->GetTensor(
                {ROUND_UP(output.shape[0], _c.defTpSize), _c.hiddenSize}, embed.dtype, DBG_LOC);
            size_t copySize = output.shape[0] * _c.hiddenSize * XDtypeBit(embed.dtype) / 8;
            aclrtMemcpyAsync(xPad.ptr, copySize, input.ptr, copySize, ACL_MEMCPY_DEVICE_TO_DEVICE,
                             rt.stream);
            ForwardLayersCommOptimize(rt, xPad, kvCache, deepstackInputEmbeds, freqsCis, output);
            rt.pool->PutTensor(xPad);
        } else {
            ForwardLayersCommOptimize(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
        }
    } else {
        rt.enableCommOptimize = false;
        ForwardLayersNaive(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
    }
}

void XModel::Forward(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                     std::vector<std::pair<XTensor, XTensor>> &kvCache,
                     std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis, XTensor &output)
{
    CheckForwardParam(rt, kvCache);
    rt.PrepareAttn(attnMeta, _c.maxM, _c.maxBatch, _c.maxSeqLen, _c.blockSize, _c.attnType);
    ForwardEmbedAndLayers(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
}

void XModel::ComputeLogits(XRuntime &rt, XTensor &input, XTensor &output)
{
    if (!rt.pool) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": xlite runtime's tensor pool not inited");
    }

    XTensor localOutput({input.shape[0], head.shape[0]}, head.dtype, output.ptr);
    XliteOpMatmul(rt, input, head, localOutput, _c.weightNZ);
    if (_c.defTpSize > 1) {
        XliteOpAllGather(rt, localOutput, output, TP);
    }
}

void XModel::ForwardAndGetLogits(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::pair<XTensor, XTensor>> &kvCache,
                                 std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                 XTensor &output)
{
    uint32_t m = input.shape[0];
    CheckForwardParam(rt, kvCache);

    XTensor &h = rt.pool->GetTensor({m, _c.hiddenSize}, embed.dtype, DBG_LOC);
    rt.PrepareAttn(attnMeta, _c.maxM, _c.maxBatch, _c.maxSeqLen, _c.blockSize, _c.attnType);
    ForwardEmbedAndLayers(rt, input, kvCache, deepstackInputEmbeds, freqsCis, h);
    ForwardGetLogits(rt, h, output);
    rt.pool->PutTensor(h);
}

void XModel::CheckForwardParam(XRuntime &rt, std::vector<std::pair<XTensor, XTensor>> &kvCache)
{
    if (rt.rankId() != _rankId || rt.tpSize() != _c.defTpSize || rt.dpSize() != _c.defDpSize) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": check runtime communication setting failed");
    }

    if (!rt.pool) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": xlite runtime's tensor pool not inited");
    }

    uint32_t expectedKvHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
    if (kvCache[0].first.shape[1] != _c.blockSize || kvCache[0].second.shape[1] != _c.blockSize ||
        kvCache[0].first.shape[2] != expectedKvHeads ||
        kvCache[0].second.shape[2] != expectedKvHeads ||
        (_c.attnType == XMODEL_ATTN_MHA && kvCache[0].first.shape[3] != _c.headDim) ||
        (_c.attnType == XMODEL_ATTN_MLA && (kvCache[0].first.shape[3] != _c.kvLoraRank ||
                                            kvCache[0].second.shape[3] != _c.ropeHeadDim))) {
        throw std::runtime_error(
            std::string(__FILE__) + ":" + std::to_string(__LINE__) +
            ": kv cache's shape not match [block_num, block_size, kv_head_num, head_size]");
    }
}

size_t XModel::GetTensorPoolSize(int dbg)
{
    size_t dtypeSize = XDtypeBit(embed.dtype) / 8;
    size_t attnSize, ffnSize;
    size_t mlpBufSize = 0;
    size_t moeBufSize = 0;
    size_t moeGateSize = 0;
    size_t moeDispatchSize = 0;
    size_t scoresBufSize = 0;
    size_t moeDispatchDpBufSize = 0;
    size_t routedExpertsBufSize = 0;
    size_t shareExpertsBufSize = 0;
    size_t size = 0;
    size_t base = 0;
    const char *envFlashAttentionEnable = std::getenv("XLITE_FLASH_ATTENTION_ENABLE");

    // TODO
    if (_c.attnType != XMODEL_ATTN_MHA) {
        return 1024;
    }

    base = ROUND_UP(_c.maxM, _c.defTpSize) * _c.hiddenSize * 3 * dtypeSize;
    attnSize = _c.maxM * mhaQKV[0].shape[0] * dtypeSize;
    attnSize += _c.maxM * attnOut[0].shape[1] * dtypeSize;
    if (!isEnvironmentVariableTrue(envFlashAttentionEnable)) {
        attnSize += AIC_MAX_NUM * TILESIZE_OF_QUERY * 2 * _c.maxSeqLen * dtypeSize;
    } else {
        attnSize += AIC_MAX_NUM * TILESIZE_OF_QUERY * 2 * TILESIZE_OF_CACHED_KV * dtypeSize;
        attnSize += AIC_MAX_NUM * TILESIZE_OF_QUERY * 2 * _c.headDim * dtypeSize;
        attnSize += AIV_MAX_NUM * TILESIZE_OF_QUERY * 2 * sizeof(float);
        attnSize += AIV_MAX_NUM * TILESIZE_OF_QUERY * 2 * sizeof(float);
        attnSize += _c.maxM * _c.nHeads * sizeof(float);
        attnSize += _c.maxM * _c.nHeads * sizeof(float);
    }

    // FFN MLP
    if (_c.nDenseLayers > 0) {
        mlpBufSize += _c.maxM * mlpUpGate[0].shape[0] * dtypeSize;
        mlpBufSize += _c.maxM * mlpDown[0].shape[1] * dtypeSize;
    }
    // FFN MoE
    if (_c.nDenseLayers < _c.nLayers) {
        // MoE gate
        moeGateSize += _c.maxM * _c.nRoutedExperts * dtypeSize;
        moeGateSize += _c.maxM * _c.nRoutedExperts / 8;
        scoresBufSize += _c.maxM * _c.nRoutedExperts * dtypeSize;
        // MoE dispatch
        moeDispatchSize += (_c.maxM * _c.defDpSize + 1) * _c.nRoutedExperts * sizeof(int32_t);
        moeDispatchSize += _c.maxM * _c.defDpSize * _c.nActExperts * _c.hiddenSize * dtypeSize;
        moeDispatchSize += _c.nRoutedExperts * sizeof(int32_t);
        if (_c.defDpSize > 1) {
            moeDispatchDpBufSize += _c.maxM * _c.defDpSize * _c.hiddenSize * dtypeSize;
            moeDispatchDpBufSize += _c.maxM * _c.defDpSize * _c.nRoutedExperts * dtypeSize * 2;
        }
        // MoE routed experts
        uint32_t intermediateSize = _c.moeIntermediateSize / _c.moeTPSize;
        routedExpertsBufSize =
            _c.maxM * _c.defDpSize * _c.nActExperts * intermediateSize * 3 * dtypeSize;
        // MoE share experts
        if (_c.nSharedExperts != 0) {
            shareExpertsBufSize += _c.maxM * _c.hiddenSize * dtypeSize;
            shareExpertsBufSize += _c.maxM * intermediateSize * 3 * dtypeSize;
        }
        if (_c.defDpSize > 1) {
            shareExpertsBufSize += _c.maxM * _c.defDpSize * _c.hiddenSize * dtypeSize;
        }
        moeBufSize = moeGateSize + moeDispatchSize +
                     std::max({scoresBufSize, moeDispatchDpBufSize, routedExpertsBufSize,
                               shareExpertsBufSize});
    }
    ffnSize = std::max(mlpBufSize, moeBufSize);
    size = base + std::max(attnSize, ffnSize);

    if (_rankId == 0 && dbg) {
        std::cout << "[Tensor pool] base: " << base << " B, attn: " << attnSize
                  << " B, ffn: " << ffnSize << " B " << "{mlp: " << mlpBufSize
                  << " B, moe: " << moeBufSize << " B " << "(gate: " << moeGateSize
                  << " B, dispatch: " << moeDispatchSize << " B" << ", scores: " << scoresBufSize
                  << " B, dpBuf: " << moeDispatchDpBufSize << " B"
                  << ", routed experts: " << routedExpertsBufSize << " B"
                  << ", share experts: " << shareExpertsBufSize << " B)}" << std::endl;
    }
    return (size >> MB_BIT) + 128;
}
