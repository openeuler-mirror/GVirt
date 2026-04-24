/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <sstream>
#include <tuple>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "model.h"

XModel::XModel(struct XModelConfig &c, uint32_t rankId) : _c(c), _rankId(rankId)
{
    attnNorm.resize(c.nLayers);
    attnOut.resize(c.nLayers);
    mhaQKV.resize(c.nLayers);
    mhaQKVBias.resize(c.nLayers);
    mhaQNorm.resize(c.nLayers);
    mhaKNorm.resize(c.nLayers);
    mlaQKVA.resize(c.nLayers);
    mlaQB.resize(c.nLayers);
    mlaQNorm.resize(c.nLayers);
    mlaKVB.resize(c.nLayers);
    mlaKVNorm.resize(c.nLayers);
    mlpNorm.resize(c.nLayers);
    mlpUpGate.resize(c.nDenseLayers);
    mlpUpGateInputScale.resize(c.nDenseLayers);
    mlpUpGateInputOffset.resize(c.nDenseLayers);
    mlpUpGateQuantBias.resize(c.nDenseLayers);
    mlpUpGateDeqScale.resize(c.nDenseLayers);
    mlpDown.resize(c.nDenseLayers);
    mlpDownInputScale.resize(c.nDenseLayers);
    mlpDownInputOffset.resize(c.nDenseLayers);
    mlpDownQuantBias.resize(c.nDenseLayers);
    mlpDownDeqScale.resize(c.nDenseLayers);
    indexQB.resize(c.nLayers);
    indexK.resize(c.nLayers);
    indexKNorm.resize(c.nLayers);
    indexKNormBias.resize(c.nLayers);
    indexWeight.resize(c.nLayers);
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
        moeREDown[i].resize(c.nRoutedExperts);
    }

    // quantization
    attnOutInputScale.resize(c.nLayers);
    attnOutInputOffset.resize(c.nLayers);
    attnOutQuantBias.resize(c.nLayers);
    attnOutDeqScale.resize(c.nLayers);
    mhaQKVInputScale.resize(c.nLayers);
    mhaQKVInputOffset.resize(c.nLayers);
    mhaQKVQuantBias.resize(c.nLayers);
    mhaQKVDeqScale.resize(c.nLayers);
    attnNormBias.resize(c.nLayers);
    mhaQNormBias.resize(c.nLayers);
    mhaKNormBias.resize(c.nLayers);
    mlpNormBias.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        moeREUpGateScale[i].resize(c.nRoutedExperts);
        moeREDownScale[i].resize(c.nRoutedExperts);
    }
}

void XModel::Init(void)
{
    std::vector<uint32_t> gateIdx, vgatherIndices;
    std::vector<uint64_t> weights;
    size_t size;
    bool isWeightEmpty = false;
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
            weights[j] = reinterpret_cast<uint64_t>(moeREDown[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREDown[i].Init({_c.nRoutedExperts}, INT64, ptr);

        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREUpGateScale[i][j].ptr);
            if (weights[j] == 0) {
                isWeightEmpty = true;
                break;
            }
        }
        if (!isWeightEmpty) {
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREUpGateScale[i].Init({_c.nRoutedExperts}, INT64, ptr);
        }

        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREDownScale[i][j].ptr);
            if (weights[j] == 0) {
                isWeightEmpty = true;
                break;
            }
        }
        if (!isWeightEmpty) {
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREDownScale[i].Init({_c.nRoutedExperts}, INT64, ptr);
        }
    }

    if (_c.attnType == XMODEL_ATTN_MLA || _c.attnType == XMODEL_ATTN_DSA) {
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

    if (_c.attnType == XMODEL_ATTN_MLA || _c.attnType == XMODEL_ATTN_DSA) {
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

std::tuple<XTensor &, XTensor &> XModel::ForwardAttnMLACommon(
    XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
    XTensor &hiddenState)
{
    if (_c.defTpSize == 0 || _c.nHeads % _c.defTpSize != 0) {
        throw std::invalid_argument("nHeads must be divisible by defTpSize and defTpSize > 0");
    }
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;

    XTensor &kCache = kvCache[layer][0];
    XTensor &vCache = kvCache[layer][1];
    XTensor &attnQkvc =
        rt.GetTensor({hiddenState.shape[0], _c.qLoraRank + _c.kvLoraRank + _c.ropeHeadDim},
                     hiddenState.dtype, DBG_LOC);
    XTensor &attnNormQc =
        rt.GetTensor({hiddenState.shape[0], _c.qLoraRank}, hiddenState.dtype, DBG_LOC);
    XTensor &attnQWithQr =
        rt.GetTensor({hiddenState.shape[0], nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim},
                     hiddenState.dtype, DBG_LOC);
    XTensor &attnNormKvc =
        rt.GetTensor({hiddenState.shape[0], _c.kvLoraRank}, hiddenState.dtype, DBG_LOC);

    // TODO: support MLA quantization
    XliteOpMatmul(rt, hiddenState, mlaQKVA[layer], attnQkvc, _c.weightNZ);
    XliteOpRmsNorm(rt, attnQkvc, mlaQNorm[layer], attnNormQc, _c.normEps, _c.qLoraRank);
    XliteOpMatmul(rt, attnNormQc, mlaQB[layer], attnQWithQr, _c.weightNZ);
    XliteOpRopeComplex(rt, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim, _c.ropeHeadDim,
                       _c.nopeHeadDim, attnQWithQr, freqsCis, rt._attnPosition, _vGather);
    XliteOpRmsNorm(rt, attnQkvc, mlaKVNorm[layer], attnNormKvc, _c.normEps, _c.kvLoraRank, true,
                   XTensor(), 1, _c.qLoraRank);
    XliteOpRopeComplexAndCache(rt, 1, _c.qLoraRank + _c.kvLoraRank + _c.ropeHeadDim, _c.ropeHeadDim,
                               _c.qLoraRank + _c.kvLoraRank, _c.kvLoraRank, _c.ropeHeadDim,
                               attnQkvc, freqsCis, rt._attnPosition, _vGather, _c.blockSize,
                               attnNormKvc, kCache, vCache, rt._attnSlotMapping);
    rt.PutTensor(attnQkvc);
    rt.PutTensor(attnNormKvc);

    return {attnQWithQr, attnNormQc};
}

XTensor &XModel::ForwardAttnIndexer(XRuntime &rt, uint32_t layer, XTensor &hiddenState,
                                    XTensor &attnNormQc, XTensor &indexKCache, XTensor &freqsCis)
{
    XTensor &q = rt.GetTensor({hiddenState.shape[0], _c.indexNHeads * _c.indexHeadDim},
                              hiddenState.dtype, DBG_LOC);
    XTensor &k = rt.GetTensor({hiddenState.shape[0], _c.indexHeadDim}, hiddenState.dtype, DBG_LOC);

    XliteOpMuls(rt, attnNormQc, _c.indexSoftmaxScale, attnNormQc);
    XliteOpMatmul(rt, attnNormQc, indexQB[layer], q, _c.weightNZ);

    XliteOpMatmul(rt, hiddenState, indexK[layer], k, _c.weightNZ);
    XliteOpLayerNorm(rt, k, indexKNorm[layer], indexKNormBias[layer], k, _c.normEps,
                     _c.indexHeadDim);
    if (_c.indexRopeInterleaved) {
        XliteOpRopeComplex(rt, _c.indexNHeads, _c.indexHeadDim, _c.ropeHeadDim, 0, q, freqsCis,
                           rt._attnPosition, _vGather);
        XTensor key, kCache;
        XliteOpRopeComplexAndCache(rt, 1, _c.indexHeadDim, _c.ropeHeadDim, 0, 0, _c.indexHeadDim, k,
                                   freqsCis, rt._attnPosition, _vGather, _c.blockSize, key, kCache,
                                   indexKCache, rt._attnSlotMapping);
    } else {
        // 1.2 TODO
    }
    rt.PutTensor(k);

    XTensor &weights =
        rt.GetTensor({hiddenState.shape[0], _c.indexNHeads}, hiddenState.dtype, DBG_LOC);
    float weightScale = 1.0f / std::sqrt(static_cast<float>(_c.indexNHeads));
    XliteOpMatmul(rt, hiddenState, indexWeight[layer], weights, _c.weightNZ);
    XliteOpMuls(rt, weights, weightScale, weights);

    XTensor &scores = rt.GetTensor({hiddenState.shape[0], rt._maxNumBlocks * _c.blockSize},
                                   hiddenState.dtype, DBG_LOC);
    XliteOpIndexerScores(rt, q, indexKCache, weights, scores, rt._queryStartLoc, rt._lens,
                         rt._cachedLens, rt._attnBlockTables, _c.indexNHeads, _c.indexHeadDim,
                         _c.blockSize, rt._batch, rt._maxNumBlocks);
    rt.PutTensor(weights);
    rt.PutTensor(q);
    XTensor &topkIndices = rt.GetTensor({hiddenState.shape[0], _c.indexTopK}, INT32, DBG_LOC);
    // 1.4 TODO do topk on the result and return the indices
    // XliteOpTopK(rt, scores, topkIndices, _c.indexTopK);
    rt.PutTensor(scores);
    return topkIndices;
}

void XModel::ForwardAttnMLA(XRuntime &rt, uint32_t layer,
                            std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
                            XTensor &hiddenState)
{
    XTensor &kCache = kvCache[layer][0];
    XTensor &vCache = kvCache[layer][1];
    uint32_t qHeads = _c.nHeads / _c.defTpSize;

    auto [attnQWithQr, attnNormQc] =
        ForwardAttnMLACommon(rt, layer, kvCache, freqsCis, hiddenState);

    XTensor *topkIndices = nullptr;
    if (_c.attnType == XMODEL_ATTN_DSA) {
        topkIndices =
            &ForwardAttnIndexer(rt, layer, hiddenState, attnNormQc, kvCache[layer][2], freqsCis);
    }
    rt.PutTensor(attnNormQc);

    XTensor &attnOutput =
        rt.GetTensor({attnQWithQr.shape[0], qHeads * _c.vHeadDim}, attnQWithQr.dtype, DBG_LOC);
    uint32_t tileSizeOfCachedKV = GetTileSizeOfCachedKV(rt.aicNum);
    XTensor &qk = rt.GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, tileSizeOfCachedKV},
                               attnQWithQr.dtype, DBG_LOC);
    XTensor &sv =
        rt.GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, _c.vHeadDim}, attnQWithQr.dtype, DBG_LOC);
    XTensor &max = rt.GetTensor({rt.aivNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
    XTensor &sum = rt.GetTensor({rt.aivNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
    XTensor &lastMax = rt.GetTensor({attnQWithQr.shape[0], qHeads}, FP32, DBG_LOC);
    XTensor &lastSum = rt.GetTensor({attnQWithQr.shape[0], qHeads}, FP32, DBG_LOC);
    XliteOpFlashMLA(rt, attnQWithQr, kCache, vCache, mlaKVB[layer], qk, sv, max, sum, lastMax,
                    lastSum, _sync, attnOutput, rt._queryStartLoc, rt._lens, rt._cachedLens,
                    rt._attnBlockTables, qHeads, _c.ropeHeadDim, _c.nopeHeadDim, _c.vHeadDim,
                    _c.kvLoraRank, _c.blockSize, rt._batch, rt._maxNumBlocks, _c.softmaxScale);
    rt.PutTensor(attnQWithQr);
    rt.PutTensor(lastSum);
    rt.PutTensor(lastMax);
    rt.PutTensor(sum);
    rt.PutTensor(max);
    rt.PutTensor(sv);
    rt.PutTensor(qk);
    if (topkIndices != nullptr) {
        rt.PutTensor(*topkIndices);
    }

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
    rt.PutTensor(attnOutput);
}

void XModel::ForwardAttnMHA(XRuntime &rt, uint32_t layer,
                            std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
                            XTensor &hiddenState)
{
    XTensor &kCache = kvCache[layer][0];
    XTensor &vCache = kvCache[layer][1];
    uint32_t qHeads = _c.nHeads / _c.defTpSize;
    uint32_t kHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));

    XTensor &qkv = rt.GetTensor({hiddenState.shape[0], (qHeads + 2 * kHeads) * _c.headDim},
                                hiddenState.dtype, DBG_LOC);
    if (mhaQKV[layer].dtype == INT8) {
        XTensor &xQuanted = rt.GetTensor(hiddenState.shape, INT8, DBG_LOC);
        XTensor &qkvFp16 = rt.GetTensor(qkv.shape, FP16, DBG_LOC);
        XliteOpQuant(rt, hiddenState, mhaQKVInputScale[layer], mhaQKVInputOffset[layer], xQuanted);
        XliteOpMatmul(rt, xQuanted, mhaQKV[layer], qkvFp16, _c.weightNZ || _c.quantAttnWeightNz,
                      mhaQKVQuantBias[layer], mhaQKVDeqScale[layer], _c.quantAttnWeightTrans);
        XliteOpDeQuant(rt, qkvFp16, qkv, false);
        rt.PutTensor(xQuanted);
        rt.PutTensor(qkvFp16);
    } else if (_c.addBias) {
        XliteOpMatmul(rt, hiddenState, mhaQKV[layer], qkv, _c.weightNZ, mhaQKVBias[layer]);
    } else {
        XliteOpMatmul(rt, hiddenState, mhaQKV[layer], qkv, _c.weightNZ);
    }
    if (_c.qkNorm && !_c.qkNormFull) {
        XliteOpRmsNorm(rt, qkv, mhaQNorm[layer], qkv, _c.normEps, _c.headDim, true,
                       mhaQNormBias[layer], qHeads);
        XliteOpRmsNorm(rt, qkv, mhaKNorm[layer], qkv, _c.normEps, _c.headDim, true,
                       mhaKNormBias[layer], kHeads, qHeads * _c.headDim, qHeads * _c.headDim);
    }
    if (_c.qkNormFull) {
        XTensor &qLocalVariance = rt.GetTensor({qkv.shape[0], 1}, FP32, DBG_LOC);
        XTensor &kLocalVariance = rt.GetTensor({qkv.shape[0], 1}, FP32, DBG_LOC);
        XliteOpRmsNorm(rt, qkv, mhaQNorm[layer], qLocalVariance, _c.normEps, _c.headDim * qHeads,
                       false, XTensor());
        XliteOpRmsNorm(rt, qkv, mhaKNorm[layer], kLocalVariance, _c.normEps, _c.headDim * kHeads,
                       false, XTensor(), 1, qHeads * _c.headDim);
        if (_c.defTpSize > 1) {
            // Merge two variance tensors into a single AllReduceSum
            size_t bytesPerVar = qkv.shape[0] * 1 * XDtypeBit(FP32) / 8;
            XTensor &packedVar = rt.GetTensor({qkv.shape[0], 2}, FP32, DBG_LOC);

            // Concatenate two tensors
            std::vector<XTensor> inputs = {qLocalVariance, kLocalVariance};
            XliteOpConcat(rt, inputs, packedVar);

            // Single AllReduceSum operation
            XliteOpAllReduceSum(rt, packedVar, packedVar, TP);

            // Split back into two tensors
            std::vector<XTensor> outputs = {qLocalVariance, kLocalVariance};
            std::vector<size_t> sizes = {bytesPerVar, bytesPerVar};
            XliteOpSplit(rt, packedVar, outputs, sizes, 1);

            rt.PutTensor(packedVar);
        }
        XliteOpRmsNorm(rt, qkv, mhaQNorm[layer], qkv, _c.normEps, _c.headDim * qHeads, true,
                       XTensor(), 1, 0, 0, qLocalVariance);
        XliteOpRmsNorm(rt, qkv, mhaKNorm[layer], qkv, _c.normEps, _c.headDim * kHeads, true,
                       XTensor(), 1, qHeads * _c.headDim, qHeads * _c.headDim, kLocalVariance);
        rt.PutTensor(qLocalVariance);
        rt.PutTensor(kLocalVariance);
    }
    XliteOpRopeCache(rt, qkv, kCache, vCache, rt._attnPosition, freqsCis, rt._attnSlotMapping,
                     _c.nHeads, _c.nKvHeads, _c.headDim, _c.ropeHeadDim, _c.blockSize,
                     _c.ropeType == XMODEL_ROPE_NEOX, _mropeMaskH, _mropeMaskW);

    XTensor &attn =
        rt.GetTensor({hiddenState.shape[0], qHeads * _c.headDim}, hiddenState.dtype, DBG_LOC);
    if (!rt.enableFlashAttention) {
        XTensor &qk =
            rt.GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, rt._maxNumBlocks * _c.blockSize},
                         hiddenState.dtype, DBG_LOC);
        XliteOpAttention(rt, qkv, kCache, vCache, qk, attn, rt._queryStartLoc, rt._lens,
                         rt._cachedLens, rt._attnBlockTables, qHeads, kHeads, _c.headDim,
                         _c.blockSize, rt._batch, rt._maxNumBlocks);
        rt.PutTensor(qk);
    } else {
        uint32_t tileSizeOfCachedKV = GetTileSizeOfCachedKV(rt.aicNum);
        XTensor &qk = rt.GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, tileSizeOfCachedKV},
                                   hiddenState.dtype, DBG_LOC);
        XTensor &sv = rt.GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, _c.headDim},
                                   hiddenState.dtype, DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({qkv.shape[0], qHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({qkv.shape[0], qHeads}, FP32, DBG_LOC);
        XliteOpFlashAttention(rt, qkv, kCache, vCache, qk, sv, max, sum, lastMax, lastSum, _sync,
                              attn, rt._queryStartLoc, rt._lens, rt._cachedLens,
                              rt._attnBlockTables, qHeads, kHeads, _c.headDim, _c.blockSize,
                              rt._batch, rt._maxNumBlocks);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }

    if (attnOut[layer].dtype == INT8) {
        XTensor &attnQuant = rt.GetTensor(attn.shape, INT8, DBG_LOC);
        XTensor &tmpState = rt.GetTensor(hiddenState.shape, FP16, DBG_LOC);
        XliteOpQuant(rt, attn, attnOutInputScale[layer], attnOutInputOffset[layer], attnQuant);
        XliteOpMatmul(rt, attnQuant, attnOut[layer], tmpState, _c.weightNZ || _c.quantAttnWeightNz,
                      attnOutQuantBias[layer], attnOutDeqScale[layer], _c.quantAttnWeightTrans);
        XliteOpDeQuant(rt, tmpState, hiddenState, false);
        rt.PutTensor(attnQuant);
        rt.PutTensor(tmpState);
    } else {
        XliteOpMatmul(rt, attn, attnOut[layer], hiddenState, _c.weightNZ);
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
    rt.PutTensor(qkv);
    rt.PutTensor(attn);
}

void XModel::ForwardAttn(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                         XTensor &freqsCis, XTensor &hiddenState)
{
    if (_c.attnType == XMODEL_ATTN_MLA) {
        ForwardAttnMLA(rt, layer, kvCache, freqsCis, hiddenState);
    } else if (_c.attnType == XMODEL_ATTN_MHA) {
        ForwardAttnMHA(rt, layer, kvCache, freqsCis, hiddenState);
    } else if (_c.attnType == XMODEL_ATTN_DSA) {
        // TODO DSA
        ForwardAttnMLA(rt, layer, kvCache, freqsCis, hiddenState);
    } else {
        throw std::runtime_error(std::string(__func__) + ": TODO");
    }
}

void XModel::ForwardMLP(XRuntime &rt, XTensor &upGate, XTensor &down, XTensor &hiddenState,
                        bool withAllReduce, const XTensor &upGateInputScale,
                        const XTensor &upGateInputOffset, const XTensor &upGateQuantBias,
                        const XTensor &upGateDeqScale, const XTensor &downInputScale,
                        const XTensor &downInputOffset, const XTensor &downQuantBias,
                        const XTensor &downDeqScale)
{
    uint32_t m = hiddenState.shape[0], k = hiddenState.shape[1];
    uint32_t localIntermediateSize = (down.shape[0] == k) ? down.shape[1] : down.shape[0];
    XTensor &h13 = rt.GetTensor({m, localIntermediateSize * 2}, hiddenState.dtype, DBG_LOC);
    XTensor &h2 = rt.GetTensor({m, localIntermediateSize}, hiddenState.dtype, DBG_LOC);

    if (upGate.dtype == INT8) {
        XTensor &xQuanted = rt.GetTensor(hiddenState.shape, INT8, DBG_LOC);
        XTensor upScale = upGateInputScale, upOff = upGateInputOffset, upBias = upGateQuantBias,
                upDeq = upGateDeqScale;

        XliteOpQuant(rt, hiddenState, upScale, upOff, xQuanted);
        XTensor &outFp16 = rt.GetTensor(h13.shape, FP16, DBG_LOC);
        XliteOpMatmul(rt, xQuanted, upGate, outFp16, _c.weightNZ || _c.quantAttnWeightNz, upBias,
                      upDeq, _c.quantAttnWeightTrans);
        rt.PutTensor(xQuanted);
        XliteOpDeQuant(rt, outFp16, h13, false);
        rt.PutTensor(outFp16);
    } else {
        XliteOpMatmul(rt, hiddenState, upGate, h13, _c.weightNZ);
    }

    XliteOpSiluAndMul(rt, h13, h2);
    if (down.dtype == INT8) {
        XTensor downScale = downInputScale, downOff = downInputOffset, downBias = downQuantBias,
                downDeq = downDeqScale;

        XTensor &xQuanted = rt.GetTensor(h2.shape, INT8, DBG_LOC);
        XliteOpQuant(rt, h2, downScale, downOff, xQuanted);

        XTensor &outFp16 = rt.GetTensor(hiddenState.shape, FP16, DBG_LOC);
        XliteOpMatmul(rt, xQuanted, down, outFp16, _c.weightNZ || _c.quantAttnWeightNz, downBias,
                      downDeq, _c.quantAttnWeightTrans);
        rt.PutTensor(xQuanted);

        XliteOpDeQuant(rt, outFp16, hiddenState, false);
        rt.PutTensor(outFp16);
    } else {
        XliteOpMatmul(rt, h2, down, hiddenState, _c.weightNZ);
    }

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
    rt.PutTensor(h2);
    rt.PutTensor(h13);
}

std::tuple<XTensor &, XTensor &> XModel::ForwardMoEGate(XRuntime &rt, uint32_t layer,
                                                        XTensor &input)
{
    uint32_t m = input.shape[0];
    XTensor &weights = rt.GetTensor({m, _c.nRoutedExperts}, moeGate[layer].dtype, DBG_LOC);
    XTensor &routing = rt.GetTensor({m, _c.nRoutedExperts}, BIT1, DBG_LOC);
    XTensor &scores =
        rt.GetTensor({input.shape[0], _c.nRoutedExperts}, moeGate[layer].dtype, DBG_LOC);

    XliteOpMatmul(rt, input, moeGate[layer], scores, _c.weightNZ);

    if (_c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
        XliteOpSigmoidTopK(rt, scores, _gateIndicts, moeGateBias[layer], _c.routeScale, weights,
                           routing, _c.nExpertGroups, _c.nLimitedGroups, _c.nActExperts,
                           _c.normTopKProb);
    } else {
        XliteOpSoftmaxTopK(rt, scores, _gateIndicts, weights, routing, _c.nActExperts,
                           _c.normTopKProb);
    }

    rt.PutTensor(scores);
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

    if (_c.defDpSize > 1) {
        XTensor &inputPerDp = tokenSorted, &weightsPerDp = weights, &routingPerDp = routing;
        // 计算字节大小
        size_t bytesInput = m * _c.hiddenSize * XDtypeBit(inputPerDp.dtype) / 8;
        size_t bytesWeights = m * _c.nRoutedExperts * XDtypeBit(weightsPerDp.dtype) / 8;
        size_t bytesRouting = m * _c.nRoutedExperts / 8;
        size_t totalBytes = bytesInput + bytesWeights + bytesRouting;
        size_t allTotalBytes = totalBytes * _c.defDpSize;

        // 获取最终输出用的tensor
        XTensor &inputAllDp = rt.GetTensor({mAllDp, _c.hiddenSize}, inputPerDp.dtype, DBG_LOC);
        XTensor &weightsAllDp =
            rt.GetTensor({mAllDp, _c.nRoutedExperts}, weightsPerDp.dtype, DBG_LOC);
        XTensor &routingAllDp =
            rt.GetTensor({mAllDp, _c.nRoutedExperts}, routingPerDp.dtype, DBG_LOC);

        // 获取临时buffer
        XTensor &packedSend = rt.GetTensor({static_cast<uint32_t>(totalBytes)}, INT8, DBG_LOC);

        // 打包tensor
        std::vector<XTensor> inputs = {inputPerDp, weightsPerDp, routingPerDp};
        XliteOpConcat(rt, inputs, packedSend);
        rt.PutTensor(routingPerDp);
        rt.PutTensor(weightsPerDp);

        // 一次AllGather
        XTensor &packedRecv = rt.GetTensor({static_cast<uint32_t>(allTotalBytes)}, INT8, DBG_LOC);
        XliteOpAllGather(rt, packedSend, packedRecv, DP);
        rt.PutTensor(packedSend);

        // 拆包为多个Tensor
        std::vector<XTensor> outputs = {inputAllDp, weightsAllDp, routingAllDp};
        std::vector<size_t> sizes = {bytesInput, bytesWeights, bytesRouting};
        XliteOpSplit(rt, packedRecv, outputs, sizes, _c.defDpSize);
        rt.PutTensor(packedRecv);

        XTensor &unpIdx = rt.GetTensor({_c.nRoutedExperts, mAllDp + 1}, INT32, DBG_LOC);
        XTensor &expertsSorted =
            rt.GetTensor({mAllDp * _c.nActExperts, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XTensor &expertsCounts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
        XliteOpPermutation(rt, inputAllDp, routingAllDp, start, end, expertsSorted, unpIdx,
                           expertsCounts);
        rt.PutTensor(inputAllDp);
        return {weightsAllDp, routingAllDp, unpIdx, expertsSorted, expertsCounts};
    } else {
        XTensor &unpIdx = rt.GetTensor({_c.nRoutedExperts, mAllDp + 1}, INT32, DBG_LOC);
        XTensor &expertsSorted =
            rt.GetTensor({mAllDp * _c.nActExperts, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XTensor &expertsCounts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
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
            rt.GetTensor({mAllDp, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end,
                             tokenSortedAllDp);
        XliteOpReduceScatter(rt, tokenSortedAllDp, tokenSorted, DP);
        rt.PutTensor(tokenSortedAllDp);
    } else {
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, tokenSorted);
    }

    rt.PutTensor(expertsCounts);
    rt.PutTensor(expertsSorted);
    rt.PutTensor(unpIdx);
    rt.PutTensor(routing);
    rt.PutTensor(weights);
}

void XModel::ForwardMoE(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    uint32_t m = hiddenState.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t intermediateSize = _c.moeIntermediateSize / _c.moeTPSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;
    enum XDtype dtype = hiddenState.dtype;
    enum XDtype moeReDtype = moeREUpGate[layer][start].dtype;
    bool useQuant = moeReDtype != dtype;

    auto [w, r] = ForwardMoEGate(rt, layer, hiddenState);
    auto [weights, routing, unpIdx, expertsSorted, expertsCounts] =
        ForwardMoEDispatch(rt, hiddenState, w, r);
    // actual token num for current rank
    XTensor num = XTensor({1}, INT32, unpIdx.ptr);

    // routed experts
    XTensor *h13Ptr;
    if (useQuant) {
        // quant(x) -> xQuanted, perChannelScale
        XTensor &xQuanted = rt.GetTensor(expertsSorted.shape, moeReDtype, DBG_LOC);
        XTensor &scale = rt.GetTensor({expertsSorted.shape[0]}, FP32, DBG_LOC);
        XliteOpQuantDyn(rt, expertsSorted, scale, xQuanted);
        rt.PutTensor(expertsSorted);

        // group_matmul(xQuanted * w13 * w13Scale) -> h13Quanted
        XTensor &h13Quanted =
            rt.GetTensor({mAllDp * _c.nActExperts, intermediateSize * 2}, FP16, DBG_LOC);
        XliteOpGroupMatmul(rt, xQuanted, _moeREUpGate[layer], _moeREUpGateScale[layer],
                           expertsCounts, start, end, moeREUpGate[layer][start].dtype,
                           intermediateSize * 2, _c.hiddenSize, h13Quanted,
                           _c.weightNZ || _c.expertsWeightNZ, _c.expertsWeightTrans);
        rt.PutTensor(xQuanted);

        // dequant(h13Quanted, perChannelScale) -> h13
        XTensor &h13 = rt.GetTensor(h13Quanted.shape, dtype, DBG_LOC);
        XliteOpDeQuant(rt, h13Quanted, h13, true, scale);
        rt.PutTensor(h13Quanted);
        rt.PutTensor(scale);
        h13Ptr = &h13;
    } else {
        XTensor &h13 =
            rt.GetTensor({mAllDp * _c.nActExperts, intermediateSize * 2}, dtype, DBG_LOC);
        XliteOpGroupMatmul(rt, expertsSorted, _moeREUpGate[layer], _moeREUpGateScale[layer],
                           expertsCounts, start, end, moeREUpGate[layer][start].dtype,
                           intermediateSize * 2, _c.hiddenSize, h13,
                           _c.weightNZ || _c.expertsWeightNZ, _c.expertsWeightTrans);
        rt.PutTensor(expertsSorted);
        h13Ptr = &h13;
    }

    XTensor &h2 = rt.GetTensor({mAllDp * _c.nActExperts, intermediateSize}, dtype, DBG_LOC);
    XliteOpSiluAndMul(rt, *h13Ptr, h2, num);
    rt.PutTensor(*h13Ptr);

    XTensor *outPtr;
    if (useQuant) {
        // quant(x) -> xQuanted, perChannelScale
        XTensor &xQuanted = rt.GetTensor(h2.shape, moeReDtype, DBG_LOC);
        XTensor &scale = rt.GetTensor({h2.shape[0]}, FP32, DBG_LOC);
        XliteOpQuantDyn(rt, h2, scale, xQuanted);
        rt.PutTensor(h2);

        // group_matmul(xQuanted * w2 * w2Scale) -> outQuanted
        XTensor &outQuanted = rt.GetTensor({mAllDp * _c.nActExperts, _c.hiddenSize}, FP16, DBG_LOC);
        XliteOpGroupMatmul(rt, xQuanted, _moeREDown[layer], _moeREDownScale[layer], expertsCounts,
                           start, end, moeREDown[layer][start].dtype, _c.hiddenSize,
                           intermediateSize, outQuanted, _c.weightNZ || _c.expertsWeightNZ,
                           _c.expertsWeightTrans);
        rt.PutTensor(xQuanted);

        // dequant(outQuanted, perChannelScale) -> out
        XTensor &out = rt.GetTensor(outQuanted.shape, dtype, DBG_LOC);
        XliteOpDeQuant(rt, outQuanted, out, true, scale);
        rt.PutTensor(outQuanted);
        rt.PutTensor(scale);
        outPtr = &out;
    } else {
        XTensor &out = rt.GetTensor({mAllDp * _c.nActExperts, _c.hiddenSize}, dtype, DBG_LOC);
        XliteOpGroupMatmul(rt, h2, _moeREDown[layer], _moeREDownScale[layer], expertsCounts, start,
                           end, moeREDown[layer][start].dtype, _c.hiddenSize, intermediateSize, out,
                           _c.weightNZ || _c.expertsWeightNZ, _c.expertsWeightTrans);
        rt.PutTensor(h2);
        outPtr = &out;
    }

    // Check if shared experts should be processed on this rank:
    // 1. Shared experts are enabled (nSharedExperts != 0)
    // 2. Either the weight is not full (all ranks process), or only rank 0 processes when weight is
    // full
    if (_c.nSharedExperts != 0 &&
        ((_isSharedExpertWeightFull && ((rt.rankId() % rt.tpSize()) == 0)) ||
         !_isSharedExpertWeightFull)) {
        XTensor &h = rt.GetTensor({m, _c.hiddenSize}, dtype, DBG_LOC);
        ForwardMOECombine(rt, h, weights, routing, unpIdx, *outPtr, expertsCounts);
        // share experts
        ForwardMLP(rt, moeSEUpGate[layer], moeSEDown[layer], hiddenState, false);
        XliteOpAdd(rt, hiddenState, h, hiddenState);
        rt.PutTensor(h);
    } else {
        ForwardMOECombine(rt, hiddenState, weights, routing, unpIdx, *outPtr, expertsCounts);
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
        ForwardMLP(rt, mlpUpGate[layer], mlpDown[layer], hiddenState, true,
                   mlpUpGateInputScale[layer], mlpUpGateInputOffset[layer],
                   mlpUpGateQuantBias[layer], mlpUpGateDeqScale[layer], mlpDownInputScale[layer],
                   mlpDownInputOffset[layer], mlpDownQuantBias[layer], mlpDownDeqScale[layer]);
    } else {
        ForwardMoE(rt, layer, hiddenState);
    }
}

void XModel::ForwardLayersCommOptimize(XRuntime &rt, XTensor &xPad,
                                       std::vector<std::vector<XTensor>> &kvCache,
                                       std::vector<XTensor> &deepstackInputEmbeds,
                                       XTensor &freqsCis, XTensor &output)
{
    XTensor xSlice;
    size_t actualM = output.shape[0];
    size_t mPad = xPad.shape[0];
    size_t mPadPerTp = mPad / _c.defTpSize;
    size_t sizePerTp = mPad * _c.hiddenSize / _c.defTpSize * XDtypeBit(embed.dtype) / 8;
    XTensor &hiddenStatePad = rt.GetTensor({mPad, _c.hiddenSize}, embed.dtype, DBG_LOC);
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
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1], true, attnNormBias[i]);
        }
        XLITE_DEBUG_POINT(_rankId == 0 && (i == 0 || i == _c.nDenseLayers), rt, h,
                          ("layer" + std::to_string(i) + " in").c_str());
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XliteOpAddAndRmsNorm(rt, rt.hiddenStateSlice, xSlice, mlpNorm[i], _c.normEps,
                             rt.hiddenStateSlice, mlpNormBias[i]);
        XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP);
        if (rt.multiTaskParallel) {
            rt.NotifyWaitPeerStream();
        }
        XLITE_DEBUG_POINT(_rankId == 0 && (i == 0 || i == _c.nDenseLayers), rt, h,
                          ("layer" + std::to_string(i) + " after attn").c_str());
        ForwardFFN(rt, i, h);
        if (i < _c.deepstackNumLevel) {
            XliteOpAdd(rt, h, deepstackInputEmbeds[i], h);
        }
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, rt.hiddenStateSlice, xSlice, attnNorm[i + 1], _c.normEps,
                                 rt.hiddenStateSlice, attnNormBias[i + 1]);
            XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP);
        }
        if (rt.multiTaskParallel) {
            rt.NotifyWaitPeerStream();
        }
        XLITE_DEBUG_POINT(_rankId == 0 && (i == 0 || i == _c.nDenseLayers), rt, h,
                          ("layer" + std::to_string(i) + " after ffn").c_str());
    }
    XliteOpAddAndRmsNorm(rt, rt.hiddenStateSlice, xSlice, norm, _c.normEps, rt.hiddenStateSlice,
                         normBias);
    if (actualM == mPad) {
        XliteOpAllGather(rt, rt.hiddenStateSlice, output, TP);
    } else {
        XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP);
        aclrtMemcpyAsync(output.ptr, actualM * _c.hiddenSize * XDtypeBit(embed.dtype) / 8,
                         rt.hiddenStatePad.ptr,
                         actualM * _c.hiddenSize * XDtypeBit(embed.dtype) / 8,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream);
    }
    rt.PutTensor(hiddenStatePad);
}

void XModel::ForwardLayersNaive(XRuntime &rt, XTensor &x,
                                std::vector<std::vector<XTensor>> &kvCache,
                                std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                XTensor &output)
{
    XTensor &h = rt.GetTensor({x.shape[0], _c.hiddenSize}, embed.dtype, DBG_LOC);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        if (i == 0) {
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1], true, attnNormBias[i]);
        }
        XLITE_DEBUG_POINT(_rankId == 0 && (i == 0 || i == _c.nDenseLayers), rt, h,
                          ("layer" + std::to_string(i) + " in").c_str());
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XliteOpAddAndRmsNorm(rt, h, x, mlpNorm[i], _c.normEps, h, mlpNormBias[i]);
        XLITE_DEBUG_POINT(_rankId == 0 && (i == 0 || i == _c.nDenseLayers), rt, h,
                          ("layer" + std::to_string(i) + " after attn").c_str());
        ForwardFFN(rt, i, h);
        if (i < _c.deepstackNumLevel) {
            XliteOpAdd(rt, h, deepstackInputEmbeds[i], h);
        }
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, h, x, attnNorm[i + 1], _c.normEps, h, attnNormBias[i + 1]);
        }
        XLITE_DEBUG_POINT(_rankId == 0 && (i == 0 || i == _c.nDenseLayers), rt, h,
                          ("layer" + std::to_string(i) + " after ffn").c_str());
    }
    XliteOpAddAndRmsNorm(rt, h, x, norm, _c.normEps, output, normBias);
    rt.PutTensor(h);
}

void XModel::ForwardEmbedAndLayers(XRuntime &rt, XTensor &input,
                                   std::vector<std::vector<XTensor>> &kvCache,
                                   std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                   XTensor &h)
{
    if (_c.defTpSize > 1 && (h.shape[0] >= rt.commOptimizeLen || rt.multiTaskParallel)) {
        XTensor &xPad =
            rt.GetTensor({ROUND_UP(h.shape[0], _c.defTpSize), _c.hiddenSize}, embed.dtype, DBG_LOC);
        XTensor x;
        x.Init({h.shape[0], _c.hiddenSize}, embed.dtype, xPad.ptr);
        ForwardParallelEmbed(rt, input, embed, x);
        rt.enableCommOptimize = true;
        ForwardLayersCommOptimize(rt, xPad, kvCache, deepstackInputEmbeds, freqsCis, h);
        rt.PutTensor(xPad);
    } else {
        XTensor &x = rt.GetTensor({input.shape[0], _c.hiddenSize}, embed.dtype, DBG_LOC);
        ForwardParallelEmbed(rt, input, embed, x);
        rt.enableCommOptimize = false;
        ForwardLayersNaive(rt, x, kvCache, deepstackInputEmbeds, freqsCis, h);
        rt.PutTensor(x);
    }
}

void XModel::ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &output)
{
    uint32_t batch = rt._batch;
    XTensor localOutput({output.shape[1], output.shape[2]}, output.dtype, output.ptr);

    if (batch < input.shape[0]) {
        XTensor &x = rt.GetTensor({batch, _c.hiddenSize}, input.dtype, DBG_LOC);
        XliteOpEmbed(rt, rt._prefillLastIdx, input, 0, input.shape[0], x);
        XliteOpMatmul(rt, x, head, localOutput, _c.weightNZ);
        rt.PutTensor(x);
    } else {
        XliteOpMatmul(rt, input, head, localOutput, _c.weightNZ);
    }

    if (_c.defTpSize > 1) {
        XliteOpAllGather(rt, localOutput, output, TP);
    }
}

void XModel::ForwardWithInputsEmbeds(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                     std::vector<std::vector<XTensor>> &kvCache,
                                     std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                     XTensor &output)
{
    CheckForwardParam(rt, kvCache);
    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.blockSize);
    if (rt.dpSize() == 1 && rt.batchedTokens < input.shape[0]) {
        input.View({rt.batchedTokens});
        output.View({input.shape[0], output.shape[1]});
    }
    if (_c.defTpSize > 1 && output.shape[0] >= rt.commOptimizeLen) {
        rt.enableCommOptimize = true;
        if (output.shape[0] % _c.defTpSize != 0) {
            XTensor &xPad = rt.GetTensor({ROUND_UP(output.shape[0], _c.defTpSize), _c.hiddenSize},
                                         embed.dtype, DBG_LOC);
            size_t copySize = output.shape[0] * _c.hiddenSize * XDtypeBit(embed.dtype) / 8;
            aclrtMemcpyAsync(xPad.ptr, copySize, input.ptr, copySize, ACL_MEMCPY_DEVICE_TO_DEVICE,
                             rt.stream);
            ForwardLayersCommOptimize(rt, xPad, kvCache, deepstackInputEmbeds, freqsCis, output);
            rt.PutTensor(xPad);
        } else {
            ForwardLayersCommOptimize(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
        }
    } else {
        rt.enableCommOptimize = false;
        ForwardLayersNaive(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
    }
}

void XModel::Forward(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                     std::vector<std::vector<XTensor>> &kvCache,
                     std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis, XTensor &output)
{
    CheckForwardParam(rt, kvCache);
    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.blockSize);
    if (rt.dpSize() == 1 && rt.batchedTokens < input.shape[0]) {
        input.View({rt.batchedTokens});
        output.View({input.shape[0], output.shape[1]});
    }
    ForwardEmbedAndLayers(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
}

void XModel::ForwardAndGetLogits(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::vector<XTensor>> &kvCache,
                                 std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                 XTensor &output)
{
    CheckForwardParam(rt, kvCache);

    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.blockSize);
    if (rt.dpSize() == 1 && rt.batchedTokens < input.shape[0]) {
        input.View({rt.batchedTokens});
    }
    XTensor &h = rt.GetTensor({rt.batchedTokens, _c.hiddenSize}, embed.dtype, DBG_LOC);
    ForwardEmbedAndLayers(rt, input, kvCache, deepstackInputEmbeds, freqsCis, h);
    ForwardGetLogits(rt, h, output);
    rt.PutTensor(h);
}

void XModel::CheckForwardParam(XRuntime &rt, std::vector<std::vector<XTensor>> &kvCache)
{
    XTensor &kCache = kvCache[0][0];
    XTensor &vCache = kvCache[0][1];
    if (rt.rankId() != _rankId || rt.tpSize() != _c.defTpSize || rt.dpSize() != _c.defDpSize) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": check runtime communication setting failed");
    }

    if (!rt.Inited()) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": xlite runtime not inited");
    }

    uint32_t expectedKvHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
    if (kCache.shape[1] != _c.blockSize || vCache.shape[1] != _c.blockSize ||
        kCache.shape[2] != expectedKvHeads || vCache.shape[2] != expectedKvHeads ||
        (_c.attnType == XMODEL_ATTN_MHA && kCache.shape[3] != _c.headDim) ||
        (_c.attnType == XMODEL_ATTN_MLA &&
         (kCache.shape[3] != _c.kvLoraRank || vCache.shape[3] != _c.ropeHeadDim))) {
        throw std::runtime_error(
            std::string(__FILE__) + ":" + std::to_string(__LINE__) +
            ": kv cache's shape not match [block_num, block_size, kv_head_num, head_size]");
    }
    if (_c.attnType == XMODEL_ATTN_DSA) {
        XTensor &indexKCache = kvCache[0][2];
        if (indexKCache.shape[1] != _c.blockSize || indexKCache.shape[2] != 1 ||
            indexKCache.shape[3] != _c.indexHeadDim) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": DSA index k cache's shape not match [block_num, "
                                     "block_size, 1, index_head_size]");
        }
    }
}

size_t XModel::GetTensorPoolSize(int dbg)
{
    XDummyRuntime rt(0, 0, _rankId, _c.defTpSize, _c.defDpSize);
    rt.InitDummyRuntime(1ull << 40);

    XModelAttnMeta attnMeta;
    attnMeta.version = 0;
    uint32_t batchSize = 1;
    uint32_t seqLen = _c.maxBatchedTokens;
    uint32_t maxNumBlocks = DIV_ROUND_UP(_c.maxSeqLen, _c.blockSize);

    for (uint32_t i = 0; i < batchSize; i++) {
        attnMeta.lens.push_back(seqLen);
        attnMeta.cachedLens.push_back(_c.maxSeqLen > seqLen ? _c.maxSeqLen - seqLen : 0);
        attnMeta.isPrefills.push_back(true);
        std::vector<uint32_t> blockTable(maxNumBlocks);
        for (uint32_t j = 0; j < maxNumBlocks; j++) {
            blockTable[j] = i * maxNumBlocks + j;
        }
        attnMeta.blockTables.push_back(blockTable);
    }

    std::vector<std::vector<XTensor>> kvCache(_c.nLayers);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        uint32_t expectedKvHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
        if (_c.attnType == XMODEL_ATTN_MHA) {
            XTensor kCache({_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.headDim},
                           embed.dtype, nullptr);
            XTensor vCache({_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.headDim},
                           embed.dtype, nullptr);
            kvCache[i] = {kCache, vCache};
        } else if (_c.attnType == XMODEL_ATTN_MLA) {
            XTensor kCache(
                {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.kvLoraRank},
                embed.dtype, nullptr);
            XTensor vCache(
                {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.ropeHeadDim},
                embed.dtype, nullptr);
            kvCache[i] = {kCache, vCache};
        } else if (_c.attnType == XMODEL_ATTN_DSA) {
            XTensor kCache(
                {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.kvLoraRank},
                embed.dtype, nullptr);
            XTensor vCache(
                {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.ropeHeadDim},
                embed.dtype, nullptr);
            XTensor indexKCache({_c.maxBatch * maxNumBlocks, _c.blockSize, 1, _c.indexHeadDim},
                                embed.dtype, nullptr);
            kvCache[i] = {kCache, vCache, indexKCache};
        } else {
            throw std::runtime_error(std::string(__func__) + ": TODO");
        }
    }

    std::vector<XTensor> deepstackInputEmbeds(_c.deepstackNumLevel);
    for (uint32_t i = 0; i < _c.deepstackNumLevel; i++) {
        XTensor deepstackEmbed({_c.maxBatchedTokens, _c.hiddenSize}, embed.dtype, nullptr);
        deepstackInputEmbeds[i] = deepstackEmbed;
    }

    XTensor freqsCis({_c.maxBatchedTokens, _c.ropeHeadDim}, embed.dtype, nullptr);
    XTensor input({_c.maxBatchedTokens}, INT32, nullptr);
    XTensor output({_c.maxBatchedTokens, _c.hiddenSize}, embed.dtype, nullptr);
    XTensor logits({_c.defTpSize, _c.maxBatch, _c.vocabSize / _c.defTpSize}, embed.dtype, nullptr);

    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.blockSize);
    Forward(rt, input, attnMeta, kvCache, deepstackInputEmbeds, freqsCis, output);
    ForwardGetLogits(rt, output, logits);
    size_t size = rt.maxUsedSize();

    if (_rankId == 0 && dbg) {
        std::cout << "[Tensor pool] calculated size: " << size << " bytes ("
                  << DIV_ROUND_UP(size, 1ull << MB_BIT) << " MB)" << std::endl;
    }
    return DIV_ROUND_UP(size, 1ull << MB_BIT);
}
