/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "model.h"
#include "debug.h"

// Per-seq cached-KV token threshold for routing decode+DSA to the dense
// (sparse gather) path vs the paged path; see ForwardAttnMLAV2.
#define XLITE_MLA_DENSE_THRESHOLD 280
// Max query length for still routing a non-decode (MTP / small prefill) step through
// the dense (sparse gather) path.
#define XLITE_SMALL_PREFILL_THRESHOLD 6

XModel::XModel(struct XModelConfig &c, uint32_t rankId) : _c(c), _rankId(rankId)
{
    // Physical slot count = logical + redundant; equals nRoutedExperts when EPLB is off.
    uint32_t totalPhysicalExperts = c.nRoutedExperts + c.numRedundantExperts;
    attnNorm.resize(c.nLayers);
    attnOut.resize(c.nLayers);
    mhaQKV.resize(c.nLayers);
    mhaQKVBias.resize(c.nLayers);
    mhaQNorm.resize(c.nLayers);
    mhaKNorm.resize(c.nLayers);
    mlaQKVA.resize(c.nLayers);
    mlaQB.resize(c.nLayers);
    mlaQNorm.resize(c.nLayers);
    mlaQNormBias.resize(c.nLayers);
    mlaWUV.resize(c.nLayers);
    mlaWUKT.resize(c.nLayers);
    mlaKVNorm.resize(c.nLayers);
    mlaKVNormBias.resize(c.nLayers);
    mlpNorm.resize(c.nLayers);
    mlpUpGate.resize(c.nDenseLayers);
    mlpDown.resize(c.nDenseLayers);
    indexQB.resize(c.nLayers);
    indexKWeightsProj.resize(c.nLayers);
    indexKNorm.resize(c.nLayers);
    indexKNormBias.resize(c.nLayers);
    linearInProjQKV.resize(c.nLayers);
    linearInProjZ.resize(c.nLayers);
    linearInProjB.resize(c.nLayers);
    linearInProjA.resize(c.nLayers);
    linearConv1d.resize(c.nLayers);
    linearALog.resize(c.nLayers);
    linearDtBias.resize(c.nLayers);
    linearNorm.resize(c.nLayers);
    linearOutProj.resize(c.nLayers);
    moeGate.resize(c.nLayers);
    moeGateBias.resize(c.nLayers);
    moeTid2Eid.resize(c.nHashLayers);
    moeSEUpGate.resize(c.nLayers);
    moeSEDown.resize(c.nLayers);
    moeSEGate.resize(c.nLayers);
    moeREUpGate.resize(c.nLayers);
    moeREUpGateDeqScale.resize(c.nLayers);
    moeREDown.resize(c.nLayers);
    moeREDownDeqScale.resize(c.nLayers);
    _moeREUpGate.resize(c.nLayers);
    _moeREUpGateDeqScale.resize(c.nLayers);
    _moeREDown.resize(c.nLayers);
    _moeREDownDeqScale.resize(c.nLayers);
    _moeREUpGateScaleBias.resize(c.nLayers);
    _moeREDownScaleBias.resize(c.nLayers);
    log2phy.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        // Indexed by GLOBAL physical slot; each EP rank fills its [start, end) slice.
        moeREUpGate[i].resize(totalPhysicalExperts);
        moeREDown[i].resize(totalPhysicalExperts);
    }

    attnSink.resize(c.nLayers);
    attnWqA.resize(c.nLayers);
    attnWoA.resize(c.nLayers);
    attnWoB.resize(c.nLayers);
    attnWKv.resize(c.nLayers);
    compApe.resize(c.nLayers);
    compWKv.resize(c.nLayers);
    compWGate.resize(c.nLayers);
    compNorm.resize(c.nLayers);
    idxWqB.resize(c.nLayers);
    idxWeightsProj.resize(c.nLayers);
    idxCompApe.resize(c.nLayers);
    idxCompWKv.resize(c.nLayers);
    idxCompWGate.resize(c.nLayers);
    idxCompNorm.resize(c.nLayers);
    hcAttnFn.resize(c.nLayers);
    hcFfnFn.resize(c.nLayers);
    hcAttnBase.resize(c.nLayers);
    hcFfnBase.resize(c.nLayers);
    hcAttnScale.resize(c.nLayers);
    hcFfnScale.resize(c.nLayers);

    attnNormBias.resize(c.nLayers);
    mhaQNormBias.resize(c.nLayers);
    mhaKNormBias.resize(c.nLayers);
    mlpNormBias.resize(c.nLayers);
    moeREUpGateScaleBias.resize(c.nLayers);
    moeREDownScaleBias.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        moeREUpGateDeqScale[i].resize(totalPhysicalExperts);
        moeREDownDeqScale[i].resize(totalPhysicalExperts);
        moeREUpGateScaleBias[i].resize(totalPhysicalExperts);
        moeREDownScaleBias[i].resize(totalPhysicalExperts);
    }

    if (IsHybridAttn(c.attnType)) {
        if (c.fullAttentionInterval == 0) {
            throw std::invalid_argument("fullAttentionInterval must be > 0 for hybrid attention");
        }
        _layerTypes.resize(c.nLayers);
        for (uint32_t i = 0; i < c.nLayers; i++) {
            _layerTypes[i] = ((i + 1) % c.fullAttentionInterval == 0) ? XMODEL_LAYER_ATTN_FULL
                                                                      : XMODEL_LAYER_ATTN_LINEAR;
        }
    }
    if (_c.blockSizes.empty()) {
        _c.blockSizes.push_back(_c.blockSize);
    }
}

void XModel::Init(void)
{
    std::vector<uint32_t> vbitsortIndices;
    std::vector<uint64_t> weights;
    size_t size;
    bool isWeightEmpty = true;
    void *ptr;
    uint32_t totalPhysicalExperts = _c.nRoutedExperts + _c.numRedundantExperts;
    uint32_t nLocalPhysicalExperts = totalPhysicalExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalPhysicalExperts;
    uint32_t end = start + nLocalPhysicalExperts;

    if (_c.nDenseLayers != _c.nLayers || _c.attnType == XMODEL_ATTN_DSA ||
        _c.attnType == XMODEL_ATTN_CXA) {
        // generate the identity index table 0.1.2...max(maxSeqLen,topK)-1
        uint64_t maxSeqLen = ROUND_UP(_c.maxSeqLen, _c.blockSizes[0]);
        maxSeqLen = INIT_MIN_SEQ_POS > maxSeqLen ? INIT_MIN_SEQ_POS : maxSeqLen;
        uint64_t maxIndices = maxSeqLen > _c.nRoutedExperts ? maxSeqLen : _c.nRoutedExperts;
        vbitsortIndices.resize(maxIndices);
        size = maxIndices * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t i = 0; i < maxIndices; i++) {
            vbitsortIndices[i] = i;
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, vbitsortIndices.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        if (_c.nDenseLayers != _c.nLayers) {
            _gateIndices.Init({_c.nRoutedExperts}, INT32, ptr);
        }
        if (_c.attnType == XMODEL_ATTN_DSA || _c.attnType == XMODEL_ATTN_CXA) {
            _dsaSeqPositions.Init({maxSeqLen}, INT32, ptr);
        }
    }

    if (_c.attnType == XMODEL_ATTN_DSA || _c.attnType == XMODEL_ATTN_CXA) {
        if (!_c.indexC8Mask.empty()) {
            // Like compress_ratios, a trailing MTP-draft entry is tolerated; the
            // mask must still cover every runtime layer.
            if (_c.indexC8Mask.size() < static_cast<size_t>(_c.nLayers)) {
                throw std::invalid_argument(
                    "indexC8Mask must describe every DSA/CXA runtime layer");
            }
            if (_c.attnType == XMODEL_ATTN_DSA) {
                for (uint32_t i = 0; i < _c.nLayers; ++i) {
                    if (_c.indexC8Mask[i] && i < _c.indexFullMask.size() && !_c.indexFullMask[i]) {
                        throw std::invalid_argument("LI-C8 requires a full Indexer layer");
                    }
                }
            }
        }
    }
    if (_c.attnType == XMODEL_ATTN_DSA || _c.attnType == XMODEL_ATTN_CXA) {
        _dsaIndexerScale = 1.0f / std::sqrt(static_cast<float>(_c.indexNHeads));
        _dsaIndexerScale *= 1.0f / std::sqrt(static_cast<float>(_c.indexHeadDim));
    } else if (!_c.indexC8Mask.empty()) {
        throw std::invalid_argument("indexC8Mask requires DSA/CXA");
    }

    if (_c.attnType == XMODEL_ATTN_CXA) {
        if (_c.oGroups == 0 || _c.oLoraRank == 0) {
            throw std::invalid_argument("CxA: o_groups and o_lora_rank must be set");
        }
        if (_c.windowSize == 0) {
            throw std::invalid_argument("CxA: window_size must be set");
        }
        if (!_c.compressRatios.empty() &&
            _c.compressRatios.size() < static_cast<size_t>(_c.nLayers)) {
            throw std::invalid_argument(
                "CxA: compress_ratios size must be 0 or greater than n_layers");
        }
        if (_c.hcMult == 0) {
            throw std::invalid_argument("CxA: hc_mult must be set");
        }
    }

    size = totalPhysicalExperts * XDtypeBit(INT64) / 8;
    for (uint32_t i = _c.nDenseLayers; i < _c.nLayers; i++) {
        weights.resize(totalPhysicalExperts);
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < totalPhysicalExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREUpGate[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREUpGate[i].Init({totalPhysicalExperts}, INT64, ptr);

        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < totalPhysicalExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREDown[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREDown[i].Init({totalPhysicalExperts}, INT64, ptr);

        isWeightEmpty = true;
        for (uint32_t j = 0; j < totalPhysicalExperts; j++) {
            if (j >= start && j < end) {
                weights[j] = reinterpret_cast<uint64_t>(moeREUpGateDeqScale[i][j].ptr);
                if (weights[j] != 0) {
                    isWeightEmpty = false;
                }
            } else {
                weights[j] = 0;
            }
        }
        if (!isWeightEmpty) {
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREUpGateDeqScale[i].Init({totalPhysicalExperts}, INT64, ptr);
        }

        isWeightEmpty = true;
        for (uint32_t j = 0; j < totalPhysicalExperts; j++) {
            if (j >= start && j < end) {
                weights[j] = reinterpret_cast<uint64_t>(moeREDownDeqScale[i][j].ptr);
                if (weights[j] != 0) {
                    isWeightEmpty = false;
                }
            } else {
                weights[j] = 0;
            }
        }

        if (!isWeightEmpty) {
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREDownDeqScale[i].Init({totalPhysicalExperts}, INT64, ptr);
        }

        // MSD W4A8 is auto-detected from weight dtype (INT4) and scale_bias presence.
        // When weights are INT4-packed and scale_bias tensors are bound, build the
        // scale_bias pointer arrays for the merge_dequant kernel.
        if (moeREUpGate[i][start].dtype == INT4 && moeREUpGateScaleBias[i][start].ptr != nullptr) {
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            for (uint32_t j = 0; j < totalPhysicalExperts; j++) {
                if (j >= start && j < end) {
                    weights[j] = reinterpret_cast<uint64_t>(moeREUpGateScaleBias[i][j].ptr);
                } else {
                    weights[j] = 0;
                }
            }
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREUpGateScaleBias[i].Init({totalPhysicalExperts}, INT64, ptr);

            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            for (uint32_t j = 0; j < totalPhysicalExperts; j++) {
                if (j >= start && j < end) {
                    weights[j] = reinterpret_cast<uint64_t>(moeREDownScaleBias[i][j].ptr);
                } else {
                    weights[j] = 0;
                }
            }
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREDownScaleBias[i].Init({totalPhysicalExperts}, INT64, ptr);
        }
    }

    size = DIV_ROUND_UP(AIV_MAX_NUM * XDtypeBit(INT32), 8);
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMemset(ptr, size, 0, size));
    _sync.Init({AIV_MAX_NUM}, INT32, ptr);

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

    if (_c.nDenseLayers < _c.nLayers && _c.nSharedExperts != 0 &&
        !moeSEUpGate[_c.nDenseLayers].weight.shape.empty() &&
        !moeSEDown[_c.nDenseLayers].weight.shape.empty()) {
        if (_c.quantAttnWeightTrans) {
            _isSharedExpertWeightFull =
                moeSEUpGate[_c.nDenseLayers].weight.shape.size() >= 2 &&
                moeSEUpGate[_c.nDenseLayers].weight.shape[1] == _c.moeIntermediateSize * 2 &&
                moeSEDown[_c.nDenseLayers].weight.shape[0] == _c.moeIntermediateSize;
        } else {
            _isSharedExpertWeightFull =
                moeSEDown[_c.nDenseLayers].weight.shape.size() >= 2 &&
                moeSEUpGate[_c.nDenseLayers].weight.shape[0] == _c.moeIntermediateSize * 2 &&
                moeSEDown[_c.nDenseLayers].weight.shape[1] == _c.moeIntermediateSize;
        }
    }
}

XModel::~XModel(void)
{
    auto freeAllocated = [](void *ptr) {
        if (ptr != nullptr) {
            (void)aclrtFree(ptr);
        }
    };
    if (_gateIndices.ptr != nullptr) {
        freeAllocated(_gateIndices.ptr);
        // both views wrap the same identity-index allocation; free it exactly once
        _dsaSeqPositions.ptr = nullptr;
    }
    freeAllocated(_dsaSeqPositions.ptr);
    for (uint32_t i = _c.nDenseLayers; i < _c.nLayers; i++) {
        freeAllocated(_moeREUpGate[i].ptr);
        freeAllocated(_moeREUpGateDeqScale[i].ptr);
        freeAllocated(_moeREDown[i].ptr);
        freeAllocated(_moeREDownDeqScale[i].ptr);
        freeAllocated(_moeREUpGateScaleBias[i].ptr);
        freeAllocated(_moeREDownScaleBias[i].ptr);
    }
    freeAllocated(_sync.ptr);
}

static void DispatchAllReduceSum(XRuntime &rt, XTensor &t, enum commType type, DebugSrcLoc loc)
{
    uint32_t arM = 0;
    bool inGraph = rt.AllReduceSumInGraphActive(type) &&
                   rt.AllReduceSumInGraphMatch(type, t.dtype, t.numel, &arM);
    if (inGraph) {
        rt.AllReduceSumInGraph(t.ptr, t.ptr, arM, static_cast<int>(XDtype2HcclDtype(t.dtype)),
                               type);
    } else {
        XliteOpAllReduceSum(rt, t, t, type, false, loc);
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
        DispatchAllReduceSum(rt, output, TP, DBG_LOC);
    }
}

// TODO: optimize XTensor allocation and release order
void XModel::ForwardLinear(XRuntime &rt, uint32_t layer, XTensor &x,
                           std::vector<MatmulWeight> &weights, XTensor &out,
                           const std::vector<XTensor> &weightBias)
{
    MatmulWeight &w = weights[layer];
    enum QuantType quantType = w.GetQuantType();
    bool isNz = _c.weightNZ;
    bool isTransposed = false;

    if (quantType == NO_QUANT) {
        if (layer < weightBias.size()) {
            XliteOpMatmul(rt, x, w.weight, out, isNz, weightBias[layer]);
        } else {
            XliteOpMatmul(rt, x, w.weight, out, isNz);
        }
        return;
    }

    isNz = _c.weightNZ || _c.quantAttnWeightNz;
    isTransposed = _c.quantAttnWeightTrans;

    XTensor &xQuanted = rt.GetTensor(x.shape, INT8, DBG_LOC);
    if (quantType == DYNAMIC_QUANT) {
        // quant(x) -> xQuanted, perChannelScale
        XTensor &scale = rt.GetTensor({x.shape[0]}, FP32, DBG_LOC);
        XliteOpQuantDyn(rt, x, scale, xQuanted);
        // matmulDequant(xQuanted, weight, quantBias, deqScale, perChannelScale) -> out
        XliteOpMatmulDeQuant(rt, xQuanted, w.weight, out, w.quantBias, w.deqScale, isNz,
                             isTransposed, scale);
        rt.PutTensor(scale);
    } else {
        // quant(x) -> xQuanted
        XliteOpQuant(rt, x, w.inputScale, w.inputOffset, xQuanted);
        // matmul(xQuanted, weight, quantBias, deqScale) -> outQuanted
        XliteOpMatmulDeQuant(rt, xQuanted, w.weight, out, w.quantBias, w.deqScale, isNz,
                             isTransposed);
    }
    rt.PutTensor(xQuanted);
}

void XModel::ForwardAttnIndexer(XRuntime &rt, uint32_t layer, XTensor &hiddenState,
                                XTensor &attnNormQc, XTensor &indexKCache, XTensor &freqsCis,
                                const XTensor &indexKScaleCache)
{
    // TODO not interleaved case
    if (!_c.indexRopeInterleaved) {
        throw std::runtime_error(std::string(__func__) + ": TODO");
    }

    XTensor &kw = rt.GetTensor({hiddenState.shape[0], _c.indexHeadDim + _c.indexNHeads},
                               hiddenState.dtype, DBG_LOC);
    XliteOpMatmul(rt, hiddenState, indexKWeightsProj[layer], kw, _c.weightNZ);

    const uint32_t maxNumBlocks = rt._attnBlockTables[0].shape[1];
    // only use sparse attention when the sequence length is long enough
    bool isLong = maxNumBlocks * _c.blockSizes[0] > _c.indexTopK;
    const bool c8 = !_c.indexC8Mask.empty() && _c.indexC8Mask[layer];
    XTensor *qPtr = nullptr;
    XTensor *q8 = nullptr;
    XTensor *qScale = nullptr;
    XTensor *scaledWeights = nullptr;
    if (isLong) {
        qPtr = &rt.GetTensor({hiddenState.shape[0], _c.indexNHeads * _c.indexHeadDim},
                             hiddenState.dtype, DBG_LOC);
        ForwardLinear(rt, layer, attnNormQc, indexQB, *qPtr);
        if (c8) {
            q8 = &rt.GetTensor(qPtr->shape, INT8, DBG_LOC);
            qScale = &rt.GetTensor({hiddenState.shape[0], _c.indexNHeads}, FP16, DBG_LOC);
            scaledWeights =
                &rt.GetTensor(qScale->shape, FP16,
                              DBG_LOC);  // Fold Q dequantization scales into head weights for C8.
        }
    }
    XliteOpIndexerPrepare(rt, kw, indexKNorm[layer], indexKNormBias[layer], freqsCis,
                          rt._attnPosition, _c.indexHeadDim, _c.indexNHeads, _c.ropeHeadDim,
                          _c.blockSizes[0], indexKCache, rt._attnSlotMapping[0], _c.normEps,
                          qPtr == nullptr ? XTensor() : *qPtr, _dsaIndexerScale, _c.indexTopK,
                          isLong, 1, indexKScaleCache, q8 == nullptr ? XTensor() : *q8,
                          qScale == nullptr ? XTensor() : *qScale,
                          scaledWeights == nullptr ? XTensor() : *scaledWeights);

    if (!isLong) {
        rt.PutTensor(kw);
        rt.dsaPerLayerTopk = nullptr;
        return;
    }

    XTensor &scores = rt.GetTensor({2 * rt.aicNum * XLITE_MAX_M0, MAX_INDEXER_KV_TILE_LEN},
                                   c8 ? FP32 : hiddenState.dtype, DBG_LOC);
    XTensor &lastTopk = rt.GetTensor({hiddenState.shape[0], 2 * _c.indexTopK}, INT32, DBG_LOC);
    rt._dsaTopkBuffer.View(hiddenState.shape[0]);
    rt.dsaPerLayerTopk = &rt._dsaTopkBuffer;
    XliteOpIndexerTopK(rt, c8 ? *q8 : *qPtr, indexKCache, c8 ? *scaledWeights : kw, scores,
                       lastTopk, _dsaSeqPositions, rt._dsaTopkBuffer, rt._attnQueryStartLoc,
                       rt._attnLens, rt._attnCachedLens, rt._attnBlockTables[0], _sync,
                       _c.indexNHeads, _c.indexHeadDim, _c.blockSizes[0], rt._batch, _c.indexTopK,
                       0, indexKScaleCache);
    rt.PutTensor(kw);
    rt.PutTensor(*qPtr);
    rt.PutTensor(lastTopk);
    rt.PutTensor(scores);
    if (c8) {
        rt.PutTensor(*scaledWeights);
        rt.PutTensor(*qScale);
        rt.PutTensor(*q8);
    }
}

std::tuple<XTensor &, XTensor &, XTensor &> XModel::ForwardAttnMLACommonV2(
    XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
    XTensor &hiddenState)
{
    if (_c.defTpSize == 0 || _c.nHeads % _c.defTpSize != 0) {
        throw std::invalid_argument("nHeads must be divisible by defTpSize and defTpSize > 0");
    }
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;

    XTensor &kCache = kvCache[layer][0];
    XTensor &peCache = kvCache[layer][1];
    XTensor &attnQkvc =
        rt.GetTensor({hiddenState.shape[0], _c.qLoraRank + _c.kvLoraRank + _c.ropeHeadDim},
                     hiddenState.dtype, DBG_LOC);
    XTensor &attnNormQc =
        rt.GetTensor({hiddenState.shape[0], _c.qLoraRank}, hiddenState.dtype, DBG_LOC);
    XTensor &attnQWithQr =
        rt.GetTensor({hiddenState.shape[0], nLocalHeads * (_c.nopeHeadDim + _c.ropeHeadDim)},
                     hiddenState.dtype, DBG_LOC);

    ForwardLinear(rt, layer, hiddenState, mlaQKVA, attnQkvc);

    XliteOpMlaPrepare(rt, attnQkvc, mlaQNorm[layer], mlaQNormBias[layer], attnNormQc,
                      mlaKVNorm[layer], mlaKVNormBias[layer], freqsCis, rt._attnPosition,
                      _c.qLoraRank, _c.kvLoraRank, _c.ropeHeadDim, _c.blockSizes[0], kCache,
                      peCache, rt._attnSlotMapping[0], _c.normEps);
    rt.PutTensor(attnQkvc);

    ForwardLinear(rt, layer, attnNormQc, mlaQB, attnQWithQr);

    XTensor &qPe = rt.GetTensor({hiddenState.shape[0], nLocalHeads * _c.ropeHeadDim},
                                hiddenState.dtype, DBG_LOC);
    XliteOpRopeComplex(rt, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim, _c.ropeHeadDim,
                       _c.ropeHeadDim, _c.nopeHeadDim, 0, attnQWithQr, freqsCis, rt._attnPosition,
                       qPe);

    return {attnQWithQr, qPe, attnNormQc};
}

void XModel::ForwardAttnMLAV2(XRuntime &rt, uint32_t layer,
                              std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
                              XTensor &hiddenState)
{
    XTensor &kCache = kvCache[layer][0];
    XTensor &peCache = kvCache[layer][1];
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;
    const uint32_t maxNumBlocks = rt._attnBlockTables[0].shape[1];

    auto [attnQWithQr, qPe, attnNormQc] =
        ForwardAttnMLACommonV2(rt, layer, kvCache, freqsCis, hiddenState);

    if (_c.attnType == XMODEL_ATTN_DSA &&
        (layer >= _c.indexFullMask.size() || _c.indexFullMask[layer])) {
        // For shared indexer, only update the current topk for a full indexer layer
        ForwardAttnIndexer(rt, layer, hiddenState, attnNormQc, kvCache[layer][2], freqsCis,
                           kvCache[layer].size() == 4 ? kvCache[layer][3] : XTensor());
    }
    XTensor *topkIndices = rt.dsaPerLayerTopk;
    rt.PutTensor(attnNormQc);

    XTensor &qAbsorb = rt.GetTensor({hiddenState.shape[0], nLocalHeads * _c.kvLoraRank},
                                    hiddenState.dtype, DBG_LOC);
    // for MLA, `mlaWUKT` could be in NZ; for DSA, `mlaWUKT` is in ND format
    // TODO: remove `_c.attnType != XMODEL_ATTN_DSA` when vllm-ascend configures `mlaWuktWeightNZ`
    XliteOpEinsumMhtHtdMhd(rt, attnQWithQr, mlaWUKT[layer], qAbsorb, hiddenState.shape[0],
                           nLocalHeads, _c.nopeHeadDim, _c.kvLoraRank,
                           _c.weightNZ && (_c.mlaWuktWeightNZ || _c.attnType != XMODEL_ATTN_DSA),
                           static_cast<int>(_c.nopeHeadDim + _c.ropeHeadDim));
    rt.PutTensor(attnQWithQr);

    XTensor &oAbsorb = rt.GetTensor({hiddenState.shape[0], nLocalHeads * _c.kvLoraRank},
                                    hiddenState.dtype, DBG_LOC);
    // Route DSA to the dense (sparse gather) path only when it pays off. Below ~280
    // tokens/seq the gather overhead dominates and the dense path regresses by up to 9%
    // on short-prompt + large-batch cases (see paged vs dense perf sweep).
    // Cond 1 (cached KV > tile): long-seq fallback where dense is consistently faster.
    // Cond 2 (per-seq cached KV > 280 * batch): perf-derived threshold that steers
    //   degenerate short-prompt + large-batch cases back to paged to avoid regression.
    // Decode always qualifies; a non-decode (MTP / small prefill) step also qualifies
    // when its max query length is small enough that the per-query-token dense cache
    // (forced queryTileSize=1) still beats the paged path on long sequences.
    if (_c.attnType == XMODEL_ATTN_DSA && topkIndices != nullptr &&
        (maxNumBlocks * _c.blockSizes[0] > rt._tileSizeOfCachedKV ||
         maxNumBlocks * _c.blockSizes[0] > XLITE_MLA_DENSE_THRESHOLD * rt._batch) &&
        rt._maxQueryLen <= XLITE_SMALL_PREFILL_THRESHOLD) {
        XTensor &kDense = rt.GetTensor({hiddenState.shape[0], _c.indexTopK, _c.kvLoraRank},
                                       hiddenState.dtype, DBG_LOC);
        XTensor &peDense = rt.GetTensor({hiddenState.shape[0], _c.indexTopK, _c.ropeHeadDim},
                                        hiddenState.dtype, DBG_LOC);
        XliteOpGatherSparseKVCache(rt, kCache, peCache, rt._attnBlockTables[0], *topkIndices,
                                   rt._attnQueryStartLoc, rt._attnLens, rt._attnCachedLens, kDense,
                                   peDense, rt._batch, _c.indexTopK, _c.blockSizes[0],
                                   _c.kvLoraRank, _c.ropeHeadDim, _c.nKvHeads);
        XTensor &qkDense =
            rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, _c.indexTopK}, hiddenState.dtype, DBG_LOC);
        XliteOpMLAV2(rt, qAbsorb, qPe, kDense, peDense, qkDense, oAbsorb, rt._attnQueryStartLoc,
                     rt._attnLens, rt._attnCachedLens, rt._attnBlockTables[0], nLocalHeads,
                     _c.ropeHeadDim, _c.kvLoraRank, _c.blockSizes[0], rt._batch, _c.indexTopK,
                     _c.softmaxScale, 0, XTensor(), true);
        rt.PutTensor(qkDense);
        rt.PutTensor(peDense);
        rt.PutTensor(kDense);
    } else if (maxNumBlocks * _c.blockSizes[0] <= rt._tileSizeOfCachedKV) {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, maxNumBlocks * _c.blockSizes[0]},
                                   hiddenState.dtype, DBG_LOC);
        XliteOpMLAV2(rt, qAbsorb, qPe, kCache, peCache, qk, oAbsorb, rt._attnQueryStartLoc,
                     rt._attnLens, rt._attnCachedLens, rt._attnBlockTables[0], nLocalHeads,
                     _c.ropeHeadDim, _c.kvLoraRank, _c.blockSizes[0], rt._batch,
                     maxNumBlocks * _c.blockSizes[0], _c.softmaxScale, _c.indexTopK,
                     topkIndices == nullptr ? XTensor() : *topkIndices);
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, rt._tileSizeOfCachedKV},
                                   hiddenState.dtype, DBG_LOC);
        XTensor &sv =
            rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, _c.kvLoraRank}, hiddenState.dtype, DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({hiddenState.shape[0], nLocalHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({hiddenState.shape[0], nLocalHeads}, FP32, DBG_LOC);
        XliteOpFlashMLAV2(rt, qAbsorb, qPe, kCache, peCache, qk, sv, max, sum, lastMax, lastSum,
                          _sync, oAbsorb, rt._attnQueryStartLoc, rt._attnLens, rt._attnCachedLens,
                          rt._attnBlockTables[0], nLocalHeads, _c.ropeHeadDim, _c.kvLoraRank,
                          _c.blockSizes[0], rt._batch, _c.softmaxScale, rt._tileSizeOfCachedKV,
                          _c.indexTopK, topkIndices == nullptr ? XTensor() : *topkIndices);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }
    rt.PutTensor(qPe);
    rt.PutTensor(qAbsorb);

    XTensor &attnOutput =
        rt.GetTensor({hiddenState.shape[0], nLocalHeads * _c.vHeadDim}, hiddenState.dtype, DBG_LOC);
    XliteOpEinsumMhtHtdMhd(rt, oAbsorb, mlaWUV[layer], attnOutput, hiddenState.shape[0],
                           nLocalHeads, _c.kvLoraRank, _c.vHeadDim, _c.mlaWuvWeightNZ);
    rt.PutTensor(oAbsorb);

    ForwardLinear(rt, layer, attnOutput, attnOut, hiddenState);
    if (_c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false, DBG_LOC);
        } else {
            DispatchAllReduceSum(rt, hiddenState, TP, DBG_LOC);
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
    const uint32_t maxNumBlocks = rt._attnBlockTables[0].shape[1];
    uint32_t qDim = qHeads * _c.headDim;
    // With output gate: fused mhaQKV layout is [Q | K | V | Gate] so a single
    // SplitCol yields contiguous [Q|K|V] for Rope/Attention (no ConcatCol).
    uint32_t qkvDim = (qHeads + 2 * kHeads) * _c.headDim;
    uint32_t projOutDim = _c.attnOutputGate ? (qkvDim + qDim) : qkvDim;

    XTensor &projOut = rt.GetTensor({hiddenState.shape[0], projOutDim}, hiddenState.dtype, DBG_LOC);
    ForwardLinear(rt, layer, hiddenState, mhaQKV, projOut, mhaQKVBias);

    XTensor *gatePtr = nullptr;
    XTensor *qkvPtr = nullptr;
    if (_c.attnOutputGate) {
        XTensor &qkv = rt.GetTensor({hiddenState.shape[0], qkvDim}, hiddenState.dtype, DBG_LOC);
        XTensor &gate = rt.GetTensor({hiddenState.shape[0], qDim}, hiddenState.dtype, DBG_LOC);
        std::vector<XTensor> splitOut = {qkv, gate};
        XliteOpSplitCol(rt, projOut, splitOut);
        rt.PutTensor(projOut);
        gatePtr = &gate;
        qkvPtr = &qkv;
    } else {
        qkvPtr = &projOut;
    }
    XTensor &qkv = *qkvPtr;

    if (_c.qkNorm && !_c.qkNormFull) {
        XliteOpQkRmsNorm(rt, qkv, mhaQNorm[layer], mhaQNormBias[layer], mhaKNorm[layer],
                         mhaKNormBias[layer], qkv, _c.normEps, _c.headDim, qHeads, _c.headDim,
                         kHeads, qHeads * _c.headDim, true);
    }
    if (_c.qkNormFull) {
        size_t rows = qkv.shape[0];
        size_t bytesPerVar = rows * XDtypeBit(FP32) / 8;
        XTensor &packedVar = rt.GetTensor({rows * 2, 1}, FP32, DBG_LOC);
        XTensor qLocalVariance;
        XTensor kLocalVariance;
        qLocalVariance.Init({rows, 1}, FP32, packedVar.ptr);
        kLocalVariance.Init(
            {rows, 1}, FP32,
            reinterpret_cast<void *>(reinterpret_cast<uint64_t>(packedVar.ptr) + bytesPerVar));
        XliteOpRmsNorm(rt, qkv, XTensor(), qLocalVariance, _c.normEps, _c.headDim * qHeads, false,
                       XTensor());
        XliteOpRmsNorm(rt, qkv, XTensor(), kLocalVariance, _c.normEps, _c.headDim * kHeads, false,
                       XTensor(), 1, qHeads * _c.headDim);
        if (_c.defTpSize > 1) {
            // Single AllReduceSum over the packed buffer reduces both variances in place.
            DispatchAllReduceSum(rt, packedVar, TP, DBG_LOC);
        }
        XliteOpQkRmsNorm(rt, qkv, mhaQNorm[layer], XTensor(), mhaKNorm[layer], XTensor(), qkv,
                         _c.normEps, _c.headDim * qHeads, 1, _c.headDim * kHeads, 1,
                         qHeads * _c.headDim, true, qLocalVariance, kLocalVariance);
        rt.PutTensor(packedVar);
    }
    XliteOpRopeCache(rt, qkv, kCache, vCache, rt._attnPosition, freqsCis, rt._attnSlotMapping[0],
                     _c.nHeads, _c.nKvHeads, _c.headDim, _c.ropeHeadDim, _c.blockSizes[0],
                     _c.ropeType == XMODEL_ROPE_NEOX, _mropeMaskH, _mropeMaskW);

    XTensor &attn =
        rt.GetTensor({hiddenState.shape[0], qHeads * _c.headDim}, hiddenState.dtype, DBG_LOC);
    if (maxNumBlocks * _c.blockSizes[0] <= rt._tileSizeOfCachedKV) {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, maxNumBlocks * _c.blockSizes[0]},
                                   hiddenState.dtype, DBG_LOC);
        XliteOpAttention(rt, qkv, kCache, vCache, qk, attn, rt._attnQueryStartLoc, rt._attnLens,
                         rt._attnCachedLens, rt._attnBlockTables[0], qHeads, kHeads, _c.headDim,
                         _c.blockSizes[0], rt._batch);
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, rt._tileSizeOfCachedKV},
                                   hiddenState.dtype, DBG_LOC);
        XTensor &sv =
            rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, _c.headDim}, hiddenState.dtype, DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({qkv.shape[0], qHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({qkv.shape[0], qHeads}, FP32, DBG_LOC);
        XliteOpFlashAttention(rt, qkv, kCache, vCache, qk, sv, max, sum, lastMax, lastSum, _sync,
                              attn, rt._attnQueryStartLoc, rt._attnLens, rt._attnCachedLens,
                              rt._attnBlockTables[0], qHeads, kHeads, _c.headDim, _c.blockSizes[0],
                              rt._batch, rt._tileSizeOfCachedKV);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }

    if (gatePtr != nullptr) {
        XliteOpSigmoidGateMul(rt, attn, *gatePtr, attn);
        rt.PutTensor(*gatePtr);
    }

    ForwardLinear(rt, layer, attn, attnOut, hiddenState);

    if (_c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false, DBG_LOC);
        } else {
            DispatchAllReduceSum(rt, hiddenState, TP, DBG_LOC);
        }
    }
    rt.PutTensor(qkv);
    rt.PutTensor(attn);
}

void XModel::ForwardAttnLinear(XRuntime &rt, uint32_t layer,
                               std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
                               XTensor &hiddenState)
{
    (void)freqsCis;
    if (kvCache.size() <= layer || kvCache[layer].size() < 2) {
        throw std::runtime_error(std::string(__func__) +
                                 ": linear-attention state cache missing at layer " +
                                 std::to_string(layer));
    }
    XTensor &convState = kvCache[layer][0];
    XTensor &ssmState = kvCache[layer][1];

    uint32_t m = hiddenState.shape[0];
    uint32_t batch = rt._batch;
    if (batch == 0) {
        throw std::runtime_error("ForwardAttnLinear: empty batch");
    }
    if (rt._hostLens.size() != batch || rt._cachedLensHost.size() != batch) {
        throw std::runtime_error(
            "ForwardAttnLinear: host lens not populated; PrepareAttn must run first");
    }
    // Pure decode = every request has one query token and cached state (host-side
    // _linearDecodeStep). Multi-token steps, including speculative verification, take the
    // packed prefill path; the check below only rejects decode metadata whose token count
    // is not one per request.
    bool prefill = !rt._linearDecodeStep;
    if (!rt.IsDummyRuntime() && !prefill && m != batch) {
        throw std::runtime_error("multi-token native GDN requires prefill metadata");
    }

    uint32_t nLocalKHeads = _c.linearNumKHeads / _c.defTpSize;
    uint32_t nLocalVHeads = _c.linearNumVHeads / _c.defTpSize;
    uint32_t localKeyDim = nLocalKHeads * _c.linearKeyHeadDim;
    uint32_t localValueDim = nLocalVHeads * _c.linearValueHeadDim;
    uint32_t convDim = localKeyDim * 2 + localValueDim;
    uint32_t qkvDim = localKeyDim * 2 + localValueDim;
    uint32_t zDim = localValueDim;
    uint32_t bDim = nLocalVHeads;
    uint32_t aDim = nLocalVHeads;

    // Step 1: Direct per-weight ND projections (no runtime weight concat).
    // The old path Concat'd W_qkv/W_z/W_b/W_a into a 169MB temp W every step
    // (48 layers x 169MB D2D per step) before one fused matmul + SplitCol.
    // Weights are ND here (NZ weights cannot be Concat'd; tiny NZ matmul
    // tiles over-read), so each projection can consume its own weight
    // directly: zero D2D, no temp W, no SplitCol. Numerically identical:
    // same matmul split along N, no summation reordering.
    XTensor &mixQkv = rt.GetTensor({m, qkvDim}, hiddenState.dtype, DBG_LOC);
    XTensor &z = rt.GetTensor({m, zDim}, hiddenState.dtype, DBG_LOC);
    XTensor &b = rt.GetTensor({m, bDim}, hiddenState.dtype, DBG_LOC);
    XTensor &a = rt.GetTensor({m, aDim}, hiddenState.dtype, DBG_LOC);
    XliteOpMatmul(rt, hiddenState, linearInProjQKV[layer].weight, mixQkv, false);
    XliteOpMatmul(rt, hiddenState, linearInProjZ[layer].weight, z, false);
    XliteOpMatmul(rt, hiddenState, linearInProjB[layer].weight, b, false);
    XliteOpMatmul(rt, hiddenState, linearInProjA[layer].weight, a, false);

    // Step 2: beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
    XTensor &beta = rt.GetTensor({m, nLocalVHeads}, hiddenState.dtype, DBG_LOC);
    XTensor &g = rt.GetTensor({m, nLocalVHeads}, FP32, DBG_LOC);
    XliteOpBetaDecay(rt, b, a, linearALog[layer], linearDtBias[layer], beta, g, m, 1, nLocalVHeads);
    rt.PutTensor(b);
    rt.PutTensor(a);

    // Native pages already contain prefill/restored state. Never clear a batch row.
    XTensor &convPacked = rt.GetTensor({m, convDim}, hiddenState.dtype, DBG_LOC);
    if (!rt.IsDummyRuntime()) {
        if (prefill) {
            XliteOpConv1dAndSiLUPrefillPaged(rt, convState, mixQkv, linearConv1d[layer], convPacked,
                                             rt._gdnConvIndices.at(layer),
                                             rt._gdnQueryStarts.at(layer),
                                             rt._gdnConvInitial.at(layer));
        } else {
            XliteOpConv1dAndSiLUPaged(rt, convState, mixQkv, linearConv1d[layer], convPacked,
                                      rt._gdnConvIndices.at(layer));
        }
    }
    rt.PutTensor(mixQkv);

    // Step 4-7: split / L2 / grouped-head gated delta
    XTensor &query = rt.GetTensor({m, localKeyDim}, hiddenState.dtype, DBG_LOC);
    XTensor &key = rt.GetTensor({m, localKeyDim}, hiddenState.dtype, DBG_LOC);
    XTensor &value = rt.GetTensor({m, localValueDim}, hiddenState.dtype, DBG_LOC);
    XTensor &coreAttn = rt.GetTensor({m, localValueDim}, hiddenState.dtype, DBG_LOC);
    {
        std::vector<XTensor> qkvSplit = {query, key, value};
        XliteOpSplitCol(rt, convPacked, qkvSplit);
        rt.PutTensor(convPacked);

        XliteOpL2Norm(rt, query, query, _c.normEps, _c.linearKeyHeadDim, nLocalKHeads);
        XliteOpL2Norm(rt, key, key, _c.normEps, _c.linearKeyHeadDim, nLocalKHeads);

        if (prefill && !rt.IsDummyRuntime()) {
            XliteOpRecurrentGatedDeltaRulePrefillPaged(
                rt, query, key, value, beta, g, ssmState, coreAttn, rt._gdnSsmIndices.at(layer),
                rt._gdnQueryStarts.at(layer), rt._gdnSsmInitial.at(layer), nLocalKHeads,
                nLocalVHeads, _c.linearKeyHeadDim, _c.linearValueHeadDim);
        } else {
            XliteOpRecurrentGatedDeltaRulePaged(
                rt, query, key, value, beta, g, ssmState, coreAttn,
                rt.IsDummyRuntime() ? nullptr : &rt._gdnSsmIndices.at(layer), batch, nLocalKHeads,
                nLocalVHeads, _c.linearKeyHeadDim, _c.linearValueHeadDim);
        }
        rt.PutTensor(query);
        rt.PutTensor(key);
        rt.PutTensor(beta);
        rt.PutTensor(g);
        rt.PutTensor(value);
    }

    // Step 8: RMSNormGated(core, z) = rmsnorm(core) * silu(z)
    XTensor &gated = rt.GetTensor({m, localValueDim}, hiddenState.dtype, DBG_LOC);
    {
        XTensor &normed = rt.GetTensor({m, localValueDim}, FP32, DBG_LOC);
        XliteOpRmsNorm(rt, coreAttn, linearNorm[layer], normed, _c.normEps, _c.linearValueHeadDim,
                       true, XTensor(), nLocalVHeads);
        XliteOpSiluGateMul(rt, normed, z, gated);
        rt.PutTensor(z);
        rt.PutTensor(normed);
        rt.PutTensor(coreAttn);
    }

    // Step 9: output projection + TP all-reduce
    {
        ForwardLinear(rt, layer, gated, linearOutProj, hiddenState);
        rt.PutTensor(gated);

        if (_c.defTpSize > 1) {
            if (rt.multiTaskParallel) {
                rt.NotifyRecordPeerStream();
            }
            if (rt.enableCommOptimize) {
                XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false,
                                     DBG_LOC);
            } else {
                DispatchAllReduceSum(rt, hiddenState, TP, DBG_LOC);
            }
        }
    }
}

// CXA per-layer table access: the layer's own row when populated, else the
// flat default (V1 / DummyRun). An empty (numel==0) row plane is EXACT data
// for this step — it must NOT fall back to flat (which would serve another
// ratio's layout).
static inline XTensor &LayerPlane(std::vector<std::vector<XTensor>> &rows,
                                  std::vector<XTensor> &flat, uint32_t layer, size_t plane)
{
    if (!rows.empty() && layer < rows.size() && plane < rows[layer].size()) {
        return rows[layer][plane];
    }
    return flat[plane];
}

// Compressor positions for a ratio (layers of one ratio share one buffer).
// Only called for ratio>0; throws if unregistered (V1 / non-CXA — CXA is V2-only).
static inline const XTensor &CompressorPositions(XRuntime &rt, uint32_t ratio)
{
    auto it =
        std::find(rt._distinctCompressRatios.begin(), rt._distinctCompressRatios.end(), ratio);
    if (it == rt._distinctCompressRatios.end()) {
        throw std::runtime_error(std::string(__func__) + ": no compressor positions for ratio " +
                                 std::to_string(ratio));
    }
    return rt._compressorPosBufs[std::distance(rt._distinctCompressRatios.begin(), it)];
}

std::tuple<XTensor &, XTensor &> XModel::ForwardAttnCXACommon(XRuntime &rt, uint32_t layer,
                                                              std::vector<XTensor> &kvCache,
                                                              XTensor &freqsCis,
                                                              XTensor &hiddenState)
{
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;
    XTensor &qr = rt.GetTensor({hiddenState.shape[0], _c.qLoraRank}, hiddenState.dtype, DBG_LOC);
    XTensor &q =
        rt.GetTensor({hiddenState.shape[0], nLocalHeads, _c.headDim}, hiddenState.dtype, DBG_LOC);
    XTensor &kv = rt.GetTensor({hiddenState.shape[0], _c.headDim}, hiddenState.dtype, DBG_LOC);

    ForwardLinear(rt, layer, hiddenState, attnWqA, qr);
    ForwardLinear(rt, layer, hiddenState, attnWKv, kv);

    XliteOpRmsNorm(rt, qr, mlaQNorm[layer], qr, _c.normEps, _c.qLoraRank);
    XliteOpRmsNorm(rt, kv, mlaKVNorm[layer], kv, _c.normEps, _c.headDim);

    XliteOpRopeComplexAndCache(
        rt, 1, _c.headDim, _c.ropeHeadDim, _c.headDim - _c.ropeHeadDim, _c.headDim, kv, freqsCis,
        rt._attnPosition, _c.blockSizes[CXA_SWA_KV], kvCache[CXA_SWA_KV],
        LayerPlane(rt._attnSlotMappingPerLayer, rt._attnSlotMapping, layer, CXA_SWA_KV), true);
    if (_c.compressRatios[layer] > 0) {
        uint32_t ratio = _c.compressRatios[layer];
        uint32_t overlap = (ratio == 4) ? 1 : 0;
        uint32_t coff = 1 + overlap;
        uint32_t fullDim = coff * _c.headDim;
        uint32_t batch = rt._batch;
        uint32_t batchedTokens = hiddenState.shape[0];

        XTensor &compKv = rt.GetTensor({batchedTokens, fullDim}, FP32, DBG_LOC);
        XTensor &compScore = rt.GetTensor({batchedTokens, fullDim}, FP32, DBG_LOC);
        // compWKv/wGate are FP32 (never NZ-converted); matmul_float reads ND, so
        // do NOT forward global weightNZ — it faults on ND fp32.
        XliteOpMatmul(rt, hiddenState, compWKv[layer], compKv, false);
        XliteOpMatmul(rt, hiddenState, compWGate[layer], compScore, false);

        const XTensor &compPosTensor = CompressorPositions(rt, ratio);
        XTensor &compSlotTensor =
            LayerPlane(rt._attnSlotMappingPerLayer, rt._attnSlotMapping, layer, CXA_COMPRESS_KV);
        uint32_t nTotalBlocks = compPosTensor.shape[0];

        XTensor &weightedSum = rt.GetTensor({nTotalBlocks, _c.headDim}, BF16, DBG_LOC);
        // State pools are paged per-ratio by the host (block size differs per
        // ratio); the single _c.blockSizes[CXA_STATE] can't represent both, so
        // read the actual block size from the pool tensor (shape[1]).
        const XTensor &stateCache = kvCache[CXA_STATE];
        uint32_t stateBs = (stateCache.ptr != nullptr && stateCache.shape.size() > 1)
                               ? static_cast<uint32_t>(stateCache.shape[1])
                               : _c.blockSizes[CXA_STATE];
        XliteOpCompressor(
            rt, compKv, compScore, compApe[layer], compNorm[layer], freqsCis, weightedSum,
            rt._attnQueryStartLoc, rt._attnLens, rt._attnCachedLens, batch, nTotalBlocks, ratio,
            overlap, _c.headDim, _c.ropeHeadDim, _c.normEps, kvCache[CXA_COMPRESS_KV],
            compPosTensor, compSlotTensor, _c.blockSizes[CXA_COMPRESS_KV], kvCache[CXA_STATE],
            LayerPlane(rt._attnSlotMappingPerLayer, rt._attnSlotMapping, layer, CXA_STATE),
            LayerPlane(rt._attnBlockTablesPerLayer, rt._attnBlockTables, layer, CXA_STATE), stateBs,
            kvCache[CXA_STATE].stride.empty() ? 0 : kvCache[CXA_STATE].stride[0]);
        rt.PutTensor(weightedSum);
        rt.PutTensor(compKv);
        rt.PutTensor(compScore);
    }

    ForwardLinear(rt, layer, qr, mlaQB, q);

    q.View({hiddenState.shape[0], nLocalHeads * _c.headDim});
    XliteOpRmsNorm(rt, q, XTensor(), q, _c.normEps, _c.headDim, true, XTensor(), nLocalHeads);
    q.View({hiddenState.shape[0], nLocalHeads, _c.headDim});

    XliteOpRopeComplex(rt, nLocalHeads, _c.headDim, _c.headDim, _c.ropeHeadDim,
                       _c.headDim - _c.ropeHeadDim, _c.headDim - _c.ropeHeadDim, q, freqsCis,
                       rt._attnPosition, q, false, true);
    rt.PutTensor(kv);
    return {qr, q};
}

XTensor *XModel::ForwardAttnCXAIndexer(XRuntime &rt, uint32_t layer, XTensor &hiddenState,
                                       XTensor &qr, std::vector<XTensor> &kvCache,
                                       XTensor &freqsCis)
{
    uint32_t ratio = _c.compressRatios[layer];
    uint32_t indexerHeadDim = _c.indexHeadDim;
    uint32_t overlap = (ratio == 4) ? 1 : 0;
    uint32_t coff = 1 + overlap;
    uint32_t fullDim = coff * indexerHeadDim;
    uint32_t batch = rt._batch;
    uint32_t batchedTokens = hiddenState.shape[0];
    // LI-C8 (Path B): compressor ropes weightedSum in place (uniform row scale
    // absorbed by per-row quant scale, so no rotate); Q's rotate factor folds
    // into scaledWeights via quant_q_c8; topk runs its int8 branch.
    const bool c8 = !_c.indexC8Mask.empty() && _c.indexC8Mask[layer];

    // wkv / wgate GEMMs
    XTensor &idxCompKv = rt.GetTensor({batchedTokens, fullDim}, FP32, DBG_LOC);
    XTensor &idxCompScore = rt.GetTensor({batchedTokens, fullDim}, FP32, DBG_LOC);
    // FP32 compressor weights, ND layout — see ForwardAttnCXACommon: weightNZ must be false here.
    XliteOpMatmul(rt, hiddenState, idxCompWKv[layer], idxCompKv, false);
    XliteOpMatmul(rt, hiddenState, idxCompWGate[layer], idxCompScore, false);

    // nTotalBlocks from the row's host-computed INDEXER_K slot_mapping length.
    uint32_t nTotalBlocks =
        LayerPlane(rt._attnSlotMappingPerLayer, rt._attnSlotMapping, layer, CXA_INDEXER_K).shape[0];

    XTensor &idxWeightedSum = rt.GetTensor({nTotalBlocks, indexerHeadDim}, BF16, DBG_LOC);
    // C8 mode: absent compressKv (nullptr) — in-place rope leaves weightedSum
    // un-rotated for quant_cache_c8.
    XTensor emptyCompressKv;
    XliteOpCompressor(
        rt, idxCompKv, idxCompScore, idxCompApe[layer], idxCompNorm[layer], freqsCis,
        idxWeightedSum, rt._attnQueryStartLoc, rt._attnLens, rt._attnCachedLens, batch,
        nTotalBlocks, ratio, overlap, indexerHeadDim, _c.ropeHeadDim, _c.normEps,
        c8 ? emptyCompressKv : kvCache[CXA_INDEXER_K], CompressorPositions(rt, ratio),
        LayerPlane(rt._attnSlotMappingPerLayer, rt._attnSlotMapping, layer, CXA_INDEXER_K),
        _c.blockSizes[CXA_INDEXER_K], kvCache[CXA_INDEXER_STATE],
        LayerPlane(rt._attnSlotMappingPerLayer, rt._attnSlotMapping, layer, CXA_INDEXER_STATE),
        LayerPlane(rt._attnBlockTablesPerLayer, rt._attnBlockTables, layer, CXA_INDEXER_STATE),
        _c.blockSizes[CXA_INDEXER_STATE],
        kvCache[CXA_INDEXER_STATE].stride.empty() ? 0 : kvCache[CXA_INDEXER_STATE].stride[0], !c8);
    if (c8) {
        XliteOpQuantCacheC8(
            rt, idxWeightedSum,
            LayerPlane(rt._attnSlotMappingPerLayer, rt._attnSlotMapping, layer, CXA_INDEXER_K),
            kvCache[CXA_INDEXER_K], kvCache[CXA_INDEXER_K_SCALE], _c.blockSizes[CXA_INDEXER_K]);
    }

    rt.PutTensor(idxWeightedSum);
    rt.PutTensor(idxCompKv);
    rt.PutTensor(idxCompScore);

    const uint32_t idxNHeads = _c.indexNHeads;
    const uint32_t maxNumBlocks =
        LayerPlane(rt._attnBlockTablesPerLayer, rt._attnBlockTables, layer, CXA_INDEXER_K).shape[1];
    const bool isLong =
        static_cast<uint64_t>(maxNumBlocks) * _c.blockSizes[CXA_INDEXER_K] > _c.indexTopK;
    if (!isLong) {
        // Short seq: nullptr → XliteOpCXA dense branch. Indexer K was still
        // written by the compressor above.
        return nullptr;
    }

    // Indexer Q. C8: un-rotated (doRotate=false, no nope Muls) — the uniform
    // 1/sqrt(head_dim) factor folds into scaledWeights, so int8 scores differ
    // from float only by a per-query constant (ranking/ReLU unaffected).
    XTensor &idxQ =
        rt.GetTensor({batchedTokens, idxNHeads * indexerHeadDim}, hiddenState.dtype, DBG_LOC);
    ForwardLinear(rt, layer, qr, idxWqB, idxQ);
    const float rotateScale = 1.0f / std::sqrt(static_cast<float>(indexerHeadDim));
    XliteOpRopeComplex(rt, idxNHeads, indexerHeadDim, indexerHeadDim, _c.ropeHeadDim,
                       indexerHeadDim - _c.ropeHeadDim, indexerHeadDim - _c.ropeHeadDim, idxQ,
                       freqsCis, rt._attnPosition, idxQ, false, true, 0, !c8);
    idxQ.View({static_cast<size_t>(batchedTokens) * idxNHeads, indexerHeadDim});
    if (!c8) {
        // Scale the nope region; the rope region is scaled by doRotate inside the rope op.
        const uint32_t nopeDim = indexerHeadDim - _c.ropeHeadDim;
        XliteOpMuls(rt, idxQ, rotateScale, idxQ, 0, nopeDim);
        idxQ.View(
            {static_cast<size_t>(batchedTokens), static_cast<size_t>(idxNHeads) * indexerHeadDim});
    }

    XTensor &topkIndices = rt.GetTensor({batchedTokens, _c.indexTopK}, INT32, DBG_LOC);
    XTensor &lastTopk = rt.GetTensor({batchedTokens, 2 * _c.indexTopK}, INT32, DBG_LOC);
    // headWeight = weights_proj(h) * indexer score scale. weights_proj is NZ-
    // converted under weight_nz_mode=2 (NZ applied globally to bf16 weights),
    // so pass the real flag here — unlike the fp32 compressor weights / wo_a.
    XTensor &idxWeight = rt.GetTensor({batchedTokens, idxNHeads}, hiddenState.dtype, DBG_LOC);
    XliteOpMatmul(rt, hiddenState, idxWeightsProj[layer], idxWeight, _c.weightNZ);
    XliteOpMuls(rt, idxWeight, _dsaIndexerScale, idxWeight);

    XTensor *q8 = nullptr;
    XTensor *scaledWeights = nullptr;
    if (c8) {
        // Explicit vector form: View({n}) binds View(size_t, dim=0) and inflates numel.
        idxWeight.View(std::vector<size_t>{static_cast<size_t>(batchedTokens) * idxNHeads});
        q8 = &rt.GetTensor(idxQ.shape, INT8, DBG_LOC);
        scaledWeights = &rt.GetTensor({batchedTokens, idxNHeads}, FP16, DBG_LOC);
        XliteOpQuantQC8(rt, idxQ, idxWeight, *q8, *scaledWeights, rotateScale);
    }

    // The int8 topk branch requires fp32 score workspace.
    XTensor &scores = rt.GetTensor({2 * rt.aicNum * XLITE_MAX_M0, MAX_INDEXER_KV_TILE_LEN},
                                   c8 ? FP32 : hiddenState.dtype, DBG_LOC);
    XliteOpIndexerTopK(
        rt, c8 ? *q8 : idxQ, kvCache[CXA_INDEXER_K], c8 ? *scaledWeights : idxWeight, scores,
        lastTopk, _dsaSeqPositions, topkIndices, rt._attnQueryStartLoc, rt._attnLens,
        rt._attnCachedLens,
        LayerPlane(rt._attnBlockTablesPerLayer, rt._attnBlockTables, layer, CXA_INDEXER_K), _sync,
        idxNHeads, indexerHeadDim, _c.blockSizes[CXA_INDEXER_K], batch, _c.indexTopK, 0,
        c8 ? kvCache[CXA_INDEXER_K_SCALE] : XTensor(), ratio);
    rt.PutTensor(idxQ);
    rt.PutTensor(idxWeight);
    rt.PutTensor(scores);
    rt.PutTensor(lastTopk);
    if (q8 != nullptr) {
        rt.PutTensor(*q8);
        rt.PutTensor(*scaledWeights);
    }

    return &topkIndices;
}

void XModel::ForwardAttnCXA(XRuntime &rt, uint32_t layer, std::vector<XTensor> &kvCache,
                            XTensor &freqsCis, XTensor &hiddenState)
{
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;
    uint32_t nLocalGroups = _c.oGroups / _c.defTpSize;
    auto [qr, q] = ForwardAttnCXACommon(rt, layer, kvCache, freqsCis, hiddenState);
    XTensor *pTopkIndices = nullptr;
    if (_c.compressRatios[layer] == 4) {
        pTopkIndices = ForwardAttnCXAIndexer(rt, layer, hiddenState, qr, kvCache, freqsCis);
    }

    XTensor &o =
        rt.GetTensor({hiddenState.shape[0], nLocalHeads, _c.headDim}, hiddenState.dtype, DBG_LOC);
    uint32_t swaSegWidth = _c.windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B;
    uint32_t kvsize = _c.compressRatios[layer] == 0
                          ? 0
                          : ROUND_UP(rt._maxTotalLens / _c.compressRatios[layer], 4 * CXA_SVCK0);
    XTensor &scores = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, swaSegWidth + kvsize},
                                   hiddenState.dtype, DBG_LOC);
    XTensor emptyBt;
    // Read path walks the idx2 paged block_table (host-carried, no per-ratio
    // lookup): the kernel bounds its walk by compressRatio, so one padded pool
    // table serves every ratio.
    XTensor *compressBt =
        (_c.compressRatios[layer] > 0)
            ? &LayerPlane(rt._attnBlockTablesPerLayer, rt._attnBlockTables, layer, CXA_COMPRESS_KV)
            : nullptr;
    XliteOpCXA(rt, q, kvCache[CXA_SWA_KV], kvCache[CXA_COMPRESS_KV],
               LayerPlane(rt._attnBlockTablesPerLayer, rt._attnBlockTables, layer, CXA_SWA_KV),
               compressBt ? *compressBt : emptyBt, _c.blockSizes[CXA_SWA_KV],
               _c.blockSizes[CXA_COMPRESS_KV], attnSink[layer], scores, o, rt._batch,
               rt._attnQueryStartLoc, rt._attnLens, rt._attnCachedLens, nLocalHeads, _c.headDim,
               _c.softmaxScale, _c.windowSize, kvsize, _c.compressRatios[layer], _c.indexTopK,
               pTopkIndices == nullptr ? XTensor() : *pTopkIndices, false);
    rt.PutTensor(scores);

    if (pTopkIndices) {
        rt.PutTensor(*pTopkIndices);
    }
    rt.PutTensor(q);
    rt.PutTensor(qr);

    XliteOpRopeComplex(rt, nLocalHeads, _c.headDim, _c.headDim, _c.ropeHeadDim,
                       _c.headDim - _c.ropeHeadDim, _c.headDim - _c.ropeHeadDim, o, freqsCis,
                       rt._attnPosition, o, true, true);

    o.View({hiddenState.shape[0], nLocalGroups, nLocalHeads * _c.headDim / nLocalGroups});
    XTensor &oa = rt.GetTensor({hiddenState.shape[0], nLocalGroups * _c.oLoraRank},
                               hiddenState.dtype, DBG_LOC);
    // wo_a is ND (skip_weight_nz_conversion=True) — pass false even when
    // weightNZ is on, else the bf16 kernel reads ND as NZ → garbage. wo_b
    // below is NZ-converted, so _c.weightNZ stays correct for it.
    XliteOpMatmul(rt, o, attnWoA[layer], oa, false);
    rt.PutTensor(o);

    XliteOpMatmul(rt, oa, attnWoB[layer], hiddenState, _c.weightNZ);
    rt.PutTensor(oa);

    if (_c.defTpSize > 1) {
        DispatchAllReduceSum(rt, hiddenState, TP, DBG_LOC);
    }
}

void XModel::ForwardAttn(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                         XTensor &freqsCis, XTensor &hiddenState)
{
    if (IsHybridAttn(_c.attnType)) {
        if (_layerTypes[layer] == XMODEL_LAYER_ATTN_FULL) {
            ForwardAttnMHA(rt, layer, kvCache, freqsCis, hiddenState);
        } else {
            ForwardAttnLinear(rt, layer, kvCache, freqsCis, hiddenState);
        }
    } else if (_c.attnType == XMODEL_ATTN_MLA || _c.attnType == XMODEL_ATTN_DSA) {
        ForwardAttnMLAV2(rt, layer, kvCache, freqsCis, hiddenState);
    } else if (_c.attnType == XMODEL_ATTN_MHA) {
        ForwardAttnMHA(rt, layer, kvCache, freqsCis, hiddenState);
    } else {
        throw std::runtime_error(std::string(__func__) + ": TODO");
    }
}

void XModel::ForwardMLP(XRuntime &rt, uint32_t layer, XTensor &hiddenState,
                        std::vector<MatmulWeight> &upGate, std::vector<MatmulWeight> &down,
                        bool withAllReduce)
{
    size_t m = hiddenState.shape[0], k = hiddenState.shape[1];
    std::vector<size_t> &downShape = down[layer].weight.shape;
    size_t localIntermediateSize = (downShape[0] == k) ? downShape[1] : downShape[0];
    XTensor &h13 = rt.GetTensor({m, localIntermediateSize * 2}, hiddenState.dtype, DBG_LOC);
    XTensor &h2 = rt.GetTensor({m, localIntermediateSize}, hiddenState.dtype, DBG_LOC);

    ForwardLinear(rt, layer, hiddenState, upGate, h13);
    XliteOpSiluAndMul(rt, h13, h2, XTensor(), _c.swigluLimit);
    ForwardLinear(rt, layer, h2, down, hiddenState);

    if (withAllReduce && _c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (!rt.enableMoEAllToAll) {
            if (rt.enableCommOptimize) {
                XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false,
                                     DBG_LOC);
            } else {
                DispatchAllReduceSum(rt, hiddenState, TP, DBG_LOC);
            }
        }
    }
    rt.PutTensor(h2);
    rt.PutTensor(h13);
}

std::tuple<XTensor &, XTensor &> XModel::ForwardMoEGate(XRuntime &rt, uint32_t layer,
                                                        XTensor &input)
{
    uint32_t m = input.shape[0], M = rt.maxTokensDp;  // input may or may not be padded to match DPs
    XTensor &weights = rt.GetTensor({M, _c.nRoutedExperts}, moeGate[layer].dtype, DBG_LOC).View(m);
    XTensor &routing = rt.GetTensor({M, _c.nRoutedExperts}, BIT1, DBG_LOC);
    if (!rt.IsDummyRuntime()) {
        (void)routing.Memset(0, rt.stream, m * _c.nRoutedExperts);
    }
    (void)routing.View(m);
    XTensor &scores = rt.GetTensor({M, _c.nRoutedExperts}, moeGate[layer].dtype, DBG_LOC).View(m);

    XliteOpMatmul(rt, input, moeGate[layer], scores, _c.gateCaptured && _c.weightNZ);

    if (_c.scoringFunc == XMODEL_SCORING_FUNC_SQRTSOFTPLUS) {
        uint32_t moELayer = layer - _c.nDenseLayers;
        bool isHash = moELayer < _c.nHashLayers;
        const XTensor &tid2eid = isHash ? moeTid2Eid[moELayer] : XTensor();
        if (!rt.IsDummyRuntime() && isHash && (_inputIds.ptr == nullptr || _inputIds.numel == 0)) {
            throw std::runtime_error(
                std::string(__func__) + ": hash-gated MoE layer " + std::to_string(layer) +
                " requires input_ids (token ids); pass them to forward_with_inputs_embeds[/_v2]");
        }
        XliteOpSqrtsoftplusHashTopK(rt, scores, _gateIndices, moeGateBias[layer], _inputIds,
                                    tid2eid, weights, routing, _c.routeScale, _c.nActExperts,
                                    isHash);
    } else if (_c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
        XliteOpSigmoidTopK(rt, scores, _gateIndices, moeGateBias[layer], _c.routeScale, weights,
                           routing, _c.nExpertGroups, _c.nLimitedGroups, _c.nActExperts,
                           _c.normTopKProb);
    } else {
        XliteOpSoftmaxTopK(rt, scores, _gateIndices, weights, routing, _c.nActExperts,
                           _c.normTopKProb);
    }

    rt.PutTensor(scores);
    return {weights.View(M), routing.View(M)};
}

std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, XTensor *, MoEAlltoAllMeta>
    XModel::ForwardMoEDispatch(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                               XTensor &routing, enum XDtype moeReDtype)
{
    // NOTE: the input XTensor's must already be viewed/padded to match DPs, if needed
    uint32_t m = rt.maxTokensDp;  // should equal to `tokenSorted.shape[0]`
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t totalPhysicalExperts = _c.nRoutedExperts + _c.numRedundantExperts;
    uint32_t nLocalPhysicalExperts = totalPhysicalExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalPhysicalExperts;
    uint32_t end = start + nLocalPhysicalExperts;
    bool hoistQuant = (moeReDtype == INT8) && (_c.hiddenSize % 32 == 0) && !rt.disableMoeQuantHoist;
    XTensor *permInput = nullptr, *permRouting = nullptr, *permScale = nullptr;
    XTensor *retWeights = nullptr, *retRouting = nullptr;

    XTensor &unpIdx = rt.GetTensor({totalPhysicalExperts, mAllDp + 1}, INT32, DBG_LOC);
    XTensor &expertsCounts = rt.GetTensor({totalPhysicalExperts}, INT32, DBG_LOC);
    size_t shape0 = mAllDp * _c.nActExperts;
    if (mAllDp >= XLITE_ACTIVE_TOKENS_RATIO_PER_EP_THRESHOLD) {
        shape0 = static_cast<size_t>(mAllDp * _c.nActExperts * rt.activeTokensRatioPerEp);
    }
    enum XDtype sortedDtype = hoistQuant ? INT8 : tokenSorted.dtype;
    XTensor &expertsSorted = rt.GetTensor({shape0, _c.hiddenSize}, sortedDtype, DBG_LOC);

    if (_c.defDpSize > 1) {
        XTensor &inputPerDp = tokenSorted, &weightsPerDp = weights, &routingPerDp = routing;

        XTensor *xInt8PerDp = nullptr;
        XTensor *scalePerDp = nullptr;
        XTensor *scaleAllDp = nullptr;
        if (hoistQuant) {
            xInt8PerDp = &rt.GetTensor(inputPerDp.shape, INT8, DBG_LOC);
            scalePerDp = &rt.GetTensor({inputPerDp.shape[0]}, FP32, DBG_LOC);
            scaleAllDp = &rt.GetTensor({mAllDp}, FP32, DBG_LOC);
            XliteOpQuantDyn(rt, inputPerDp, *scalePerDp, *xInt8PerDp);
        }

        size_t actBytes = hoistQuant ? (xInt8PerDp->bytes + scalePerDp->bytes) : inputPerDp.bytes;
        size_t totalBytes = actBytes + weightsPerDp.bytes + routingPerDp.bytes;
        size_t allTotalBytes = totalBytes * _c.defDpSize;

        enum XDtype actDtypeAllDp = hoistQuant ? INT8 : inputPerDp.dtype;
        XTensor &inputAllDp = rt.GetTensor({mAllDp, _c.hiddenSize}, actDtypeAllDp, DBG_LOC);
        XTensor &weightsAllDp =
            rt.GetTensor({mAllDp, totalPhysicalExperts}, weightsPerDp.dtype, DBG_LOC);
        XTensor &routingAllDp =
            rt.GetTensor({mAllDp, totalPhysicalExperts}, routingPerDp.dtype, DBG_LOC);

        bool inGraph = rt.AllGatherInGraphActive(DP) && (m <= _c.maxBatch);

        // packedSend: graph → bind fixed _agSendBuf; eager → pool GetTensor (returned below).
        XTensor sendFixed;
        XTensor *sendOut = nullptr;
        if (inGraph) {
            sendFixed.Init({totalBytes}, INT8, rt._agSendBuf.ptr);
            sendOut = &sendFixed;
        } else {
            sendOut = &rt.GetTensor({totalBytes}, INT8, DBG_LOC);
        }

        std::vector<XTensor> inputs;
        if (hoistQuant) {
            inputs = {*xInt8PerDp, *scalePerDp, weightsPerDp, routingPerDp};
        } else {
            inputs = {inputPerDp, weightsPerDp, routingPerDp};
        }
        XliteOpConcat(rt, inputs, *sendOut);
        rt.PutTensor(routingPerDp);
        rt.PutTensor(weightsPerDp);
        if (hoistQuant) {
            rt.PutTensor(*xInt8PerDp);
            rt.PutTensor(*scalePerDp);
        }

        // packedRecv: graph → bind fixed _agRecvBuf; eager → pool GetTensor.
        XTensor recvFixed;
        XTensor *recvOut = nullptr;
        if (inGraph) {
            recvFixed.Init({allTotalBytes}, INT8, rt._agRecvBuf.ptr);
            recvOut = &recvFixed;
        } else {
            recvOut = &rt.GetTensor({allTotalBytes}, INT8, DBG_LOC);
        }
        if (inGraph) {
            rt.AllGatherInGraph(sendOut->ptr, recvOut->ptr, m,
                                static_cast<int>(XDtype2HcclDtype(INT8)), DP);
        } else {
            XliteOpAllGather(rt, *sendOut, *recvOut, DP, true, DBG_LOC);
        }

        std::vector<XTensor> outputs;
        std::vector<size_t> sizes;
        if (hoistQuant) {
            outputs = {inputAllDp, *scaleAllDp, weightsAllDp, routingAllDp};
            sizes = {xInt8PerDp->bytes, scalePerDp->bytes, weightsPerDp.bytes, routingPerDp.bytes};
        } else {
            outputs = {inputAllDp, weightsAllDp, routingAllDp};
            sizes = {inputPerDp.bytes, weightsPerDp.bytes, routingPerDp.bytes};
        }
        XliteOpSplit(rt, *recvOut, outputs, sizes, _c.defDpSize);

        if (!inGraph) {
            rt.PutTensor(*sendOut);
            rt.PutTensor(*recvOut);
        }

        permInput = &inputAllDp;
        permRouting = &routingAllDp;
        permScale = scaleAllDp;  // nullptr unless hoistQuant
        retWeights = &weightsAllDp;
        retRouting = &routingAllDp;
    } else {
        if (hoistQuant) {
            XTensor &xInt8 = rt.GetTensor(tokenSorted.shape, INT8, DBG_LOC);
            XTensor &scale = rt.GetTensor({tokenSorted.shape[0]}, FP32, DBG_LOC);
            XliteOpQuantDyn(rt, tokenSorted, scale, xInt8, XTensor());
            permInput = &xInt8;
            permScale = &scale;
        } else {
            permInput = &tokenSorted;
        }
        permRouting = &routing;
        retWeights = &weights;
        retRouting = &routing;
    }

    XTensor *expertsScalePtr = nullptr;
    if (hoistQuant) {
        XTensor &expertsScale = rt.GetTensor({shape0}, FP32, DBG_LOC);
        expertsScalePtr = &expertsScale;
        XliteOpPermutation(rt, *permInput, *permRouting, start, end, expertsSorted, unpIdx,
                           expertsCounts, *permScale, expertsScale);
    } else {
        XliteOpPermutation(rt, *permInput, *permRouting, start, end, expertsSorted, unpIdx,
                           expertsCounts);
    }
    if (hoistQuant || _c.defDpSize > 1) {
        rt.PutTensor(*permInput);
    }
    if (hoistQuant) {
        rt.PutTensor(*permScale);
    }
    return {*retWeights, *retRouting, unpIdx, expertsSorted, expertsCounts, expertsScalePtr, {}};
}

void XModel::ForwardMOECombine(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                               XTensor &routing, XTensor &unpIdx, XTensor &expertsSorted,
                               XTensor &expertsCounts)
{
    // NOTE: the input XTensor's must already be viewed/padded to match DPs, if needed
    uint32_t m = rt.maxTokensDp, mAllDp = rt.maxTokensDp * _c.defDpSize;
    uint32_t nLocalPhysicalExperts = (_c.nRoutedExperts + _c.numRedundantExperts) / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalPhysicalExperts;
    uint32_t end = start + nLocalPhysicalExperts;

    if (_c.defDpSize > 1) {
        bool inGraph = rt.ReduceScatterInGraphActive(DP) && (m <= _c.maxBatch);
        XTensor rsSendFixed;  // graph: bound to _rsSendBuf; eager: unused
        XTensor *sendOut = nullptr;
        if (inGraph) {
            rsSendFixed.Init({mAllDp, _c.hiddenSize}, tokenSorted.dtype, rt._rsSendBuf.ptr);
            sendOut = &rsSendFixed;
        } else {
            sendOut = &rt.GetTensor({mAllDp, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        }
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, *sendOut);
        if (inGraph) {
            rt.ReduceScatterInGraph(rt._rsSendBuf.ptr, rt._rsRecvBuf.ptr, m,
                                    static_cast<int>(XDtype2HcclDtype(tokenSorted.dtype)), DP);
            size_t recvBytes =
                static_cast<size_t>(m) * _c.hiddenSize * XDtypeBit(tokenSorted.dtype) / 8;
            CHECK_ACL(aclrtMemcpyAsync(tokenSorted.ptr, recvBytes, rt._rsRecvBuf.ptr, recvBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        } else {
            XliteOpReduceScatter(rt, *sendOut, tokenSorted, DP, true, DBG_LOC);
        }
        if (!inGraph) {
            rt.PutTensor(*sendOut);  // eager: return pool tensor; graph: fixed buffer, skip
        }
    } else {
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, tokenSorted);
    }

    rt.PutTensor(expertsCounts);
    rt.PutTensor(expertsSorted);
    rt.PutTensor(unpIdx);
    rt.PutTensor(routing);
    rt.PutTensor(weights);
}

std::vector<int64_t> ComputePrefixSum(const std::vector<int64_t> &counts)
{
    std::vector<int64_t> displs(counts.size());
    displs[0] = 0;
    for (size_t i = 1; i < counts.size(); i++) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    return displs;
}

MoEAlltoAllMeta XModel::MoeComputeAlltoAllVMeta(const int32_t *tokensPerEpGroupAllEpHost,
                                                uint32_t moeEpSize, uint32_t moeTpSize,
                                                uint32_t hiddenSize, uint32_t rankId,
                                                uint32_t nRoutedExperts)
{
    uint32_t currentEp = rankId / moeTpSize;

    std::vector<int64_t> sendCountsData(moeEpSize);
    std::vector<int64_t> recvCountsData(moeEpSize);
    for (uint32_t ep = 0; ep < moeEpSize; ep++) {
        sendCountsData[ep] =
            static_cast<int64_t>(tokensPerEpGroupAllEpHost[currentEp * moeEpSize + ep]) *
            hiddenSize;
        recvCountsData[ep] =
            static_cast<int64_t>(tokensPerEpGroupAllEpHost[ep * moeEpSize + currentEp]) *
            hiddenSize;
    }

    std::vector<int64_t> sdisplsData = ComputePrefixSum(sendCountsData);
    std::vector<int64_t> rdisplsData = ComputePrefixSum(recvCountsData);

    uint64_t totalRecvElements = 0;
    for (auto v : recvCountsData) {
        totalRecvElements += v;
    }

    MoEAlltoAllMeta meta;
    meta.sendCountsData = sendCountsData;
    meta.recvCountsData = recvCountsData;
    meta.sdisplsData = sdisplsData;
    meta.rdisplsData = rdisplsData;
    meta.sendCounts.Init({moeEpSize}, INT64, meta.sendCountsData.data());
    meta.recvCounts.Init({moeEpSize}, INT64, meta.recvCountsData.data());
    meta.sdispls.Init({moeEpSize}, INT64, meta.sdisplsData.data());
    meta.rdispls.Init({moeEpSize}, INT64, meta.rdisplsData.data());
    meta.totalRecvElements = totalRecvElements;
    meta.nRoutedExperts = nRoutedExperts;
    return meta;
}

MoEAlltoAllMeta XModel::MoeComputeReverseAlltoAllVMeta(const MoEAlltoAllMeta &sendMeta,
                                                       uint32_t moeEpSize)
{
    uint64_t totalSendElements = 0;

    for (auto v : sendMeta.sendCountsData) {
        totalSendElements += v;
    }

    MoEAlltoAllMeta meta;
    meta.sendCountsData = sendMeta.recvCountsData;
    meta.recvCountsData = sendMeta.sendCountsData;
    meta.sdisplsData = sendMeta.rdisplsData;
    meta.rdisplsData = sendMeta.sdisplsData;
    meta.sendCounts.Init({moeEpSize}, INT64, meta.sendCountsData.data());
    meta.recvCounts.Init({moeEpSize}, INT64, meta.recvCountsData.data());
    meta.sdispls.Init({moeEpSize}, INT64, meta.sdisplsData.data());
    meta.rdispls.Init({moeEpSize}, INT64, meta.rdisplsData.data());
    meta.totalRecvElements = totalSendElements;
    meta.nRoutedExperts = sendMeta.nRoutedExperts;
    meta.expertsCountsAllEpDevice = sendMeta.expertsCountsAllEpDevice;
    return meta;
}

std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, MoEAlltoAllMeta>
    XModel::ForwardMoEDispatchAllToAll(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                                       XTensor &routing)
{
    uint32_t m = tokenSorted.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t start = 0;
    uint32_t end = start + _c.nRoutedExperts;

    if (_c.defDpSize > 1) {
        // step 1: permutation — route tokens to experts, output expertsCounts per rank
        XTensor &unpIdx = rt.GetTensor({_c.nRoutedExperts, m + 1}, INT32, DBG_LOC);
        XTensor &expertsSorted =
            rt.GetTensor({m * _c.nActExperts, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XTensor &expertsCounts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
        XliteOpPermutation(rt, tokenSorted, routing, start, end, expertsSorted, unpIdx,
                           expertsCounts);

        // step 2: allgather expertsCounts across EP ranks — each rank knows all ranks' token counts
        XTensor &expertsCountsAllEp =
            rt.GetTensor({_c.moeEpSize, _c.nRoutedExperts}, INT32, DBG_LOC);
        XliteOpAllGather(rt, expertsCounts, expertsCountsAllEp, EP, false, DBG_LOC);
        rt.PutTensor(expertsCounts);

        // step 3: compute tokensPerEpGroupAllEp[ep][ep] (tokens each EP rank sends to each EP rank)
        //         and tokensPerExperts[expert] (total tokens per expert across all EP ranks)
        XTensor &tokensPerEpGroupAllEp = rt.GetTensor({_c.moeEpSize, _c.moeEpSize}, INT32, DBG_LOC);
        XTensor &tokensPerExperts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
        XliteOpExpertsCountsSum(rt, expertsCountsAllEp, tokensPerEpGroupAllEp, tokensPerExperts,
                                _c.nRoutedExperts);

        // step 4: copy tokensPerEpGroupAllEp to host, compute meta
        if (rt.IsDummyRuntime()) {
            int32_t *host = static_cast<int32_t *>(rt._tokensPerEpGroupAllEpHost.ptr);
            uint32_t currentEp = _rankId / _c.moeTPSize;
            int32_t perEpUpper = static_cast<int32_t>(m * _c.nActExperts);
            uint32_t moeEp = _c.moeEpSize > 0 ? _c.moeEpSize : 1;
            size_t total = static_cast<size_t>(moeEp) * moeEp;
            for (size_t i = 0; i < total; i++) {
                host[i] = 0;
            }
            for (uint32_t ep = 0; ep < moeEp; ep++) {
                host[ep * moeEp + currentEp] = perEpUpper;
            }
        } else {
            rt.MemcpyD2HAsync(rt._tokensPerEpGroupAllEpHost.ptr, tokensPerEpGroupAllEp.ptr,
                              tokensPerEpGroupAllEp.numel * sizeof(int32_t));
            rt.Synchronize();
        }

        MoEAlltoAllMeta meta = MoeComputeAlltoAllVMeta(
            static_cast<int32_t *>(rt._tokensPerEpGroupAllEpHost.ptr), _c.moeEpSize, _c.moeTPSize,
            _c.hiddenSize, rt.rankId(), _c.nRoutedExperts);
        meta.expertsCountsAllEpDevice = &expertsCountsAllEp;

        uint32_t moeEpSafe = _c.moeEpSize > 0 ? _c.moeEpSize : 1;
        uint32_t nLocalRoutedExperts = _c.nRoutedExperts / moeEpSafe;
        uint32_t localStart = moeEpSafe == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
        uint32_t localEnd = localStart + nLocalRoutedExperts;

        // step 5: alltoallv — dispatch token data across EP ranks according to meta
        XTensor &recvBuffer = rt.GetTensor({meta.totalRecvElements / _c.hiddenSize, _c.hiddenSize},
                                           tokenSorted.dtype, DBG_LOC);
        XliteOpAlltoAllV(rt, expertsSorted, recvBuffer, meta.sendCounts, meta.recvCounts,
                         meta.sdispls, meta.rdispls, EP, DBG_LOC);

        // step 6: reorder source-grouped → expert-grouped for GroupMatmul
        XTensor &expertGrouped = rt.GetTensor(
            {meta.totalRecvElements / _c.hiddenSize, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XliteOpReorderMoE(rt, recvBuffer, expertGrouped, expertsCountsAllEp, _c.hiddenSize,
                          localStart, localEnd, true);
        rt.PutTensor(recvBuffer);
        rt.PutTensor(tokensPerEpGroupAllEp);
        rt.PutTensor(expertsSorted);

        return {weights, routing, unpIdx, expertGrouped, tokensPerExperts, meta};
    } else {
        XTensor &unpIdx = rt.GetTensor({_c.nRoutedExperts, mAllDp + 1}, INT32, DBG_LOC);
        size_t shape0 = mAllDp * _c.nActExperts;
        if (mAllDp >= XLITE_ACTIVE_TOKENS_RATIO_PER_EP_THRESHOLD) {
            shape0 = static_cast<size_t>(mAllDp * _c.nActExperts * rt.activeTokensRatioPerEp);
        }
        XTensor &expertsSorted = rt.GetTensor({shape0, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XTensor &expertsCounts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
        XliteOpPermutation(rt, tokenSorted, routing, start, end, expertsSorted, unpIdx,
                           expertsCounts);

        MoEAlltoAllMeta meta;
        return {weights, routing, unpIdx, expertsSorted, expertsCounts, meta};
    }
}

void XModel::ForwardMoECombineAllToAll(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                                       XTensor &routing, XTensor &unpIdx, XTensor &expertsSorted,
                                       XTensor &expertsCounts, const MoEAlltoAllMeta &meta)
{
    if (_c.defDpSize > 1) {
        if (!meta.expertsCountsAllEpDevice) {
            throw std::runtime_error(std::string(__func__) +
                                     ": meta.expertsCountsAllEpDevice is nullptr");
        }
        uint32_t moeEpSafe = _c.moeEpSize > 0 ? _c.moeEpSize : 1;
        uint32_t nLocalRoutedExperts = _c.nRoutedExperts / moeEpSafe;
        uint32_t localStart = moeEpSafe == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
        uint32_t localEnd = localStart + nLocalRoutedExperts;

        // reorder expert-grouped → source-grouped for reverse AlltoAllV
        XTensor &sourceGrouped = rt.GetTensor({expertsSorted.shape[0], expertsSorted.shape[1]},
                                              expertsSorted.dtype, DBG_LOC);
        XliteOpReorderMoE(rt, expertsSorted, sourceGrouped, *meta.expertsCountsAllEpDevice,
                          _c.hiddenSize, localStart, localEnd, false);
        MoEAlltoAllMeta revMeta = MoeComputeReverseAlltoAllVMeta(meta, _c.moeEpSize);
        XTensor &backBuffer =
            rt.GetTensor({revMeta.totalRecvElements / _c.hiddenSize, _c.hiddenSize},
                         expertsSorted.dtype, DBG_LOC);
        XliteOpAlltoAllV(rt, sourceGrouped, backBuffer, revMeta.sendCounts, revMeta.recvCounts,
                         revMeta.sdispls, revMeta.rdispls, EP, DBG_LOC);
        XliteOpUnpermutation(rt, backBuffer, unpIdx, routing, weights, 0, _c.nRoutedExperts,
                             tokenSorted);

        rt.PutTensor(sourceGrouped);
        rt.PutTensor(backBuffer);
    } else {
        uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
        uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
        uint32_t end = start + nLocalRoutedExperts;
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, tokenSorted);
    }

    rt.PutTensor(expertsCounts);
    rt.PutTensor(expertsSorted);
    rt.PutTensor(unpIdx);
    rt.PutTensor(routing);
    rt.PutTensor(weights);
}

void XModel::ForwardMoEMSD(XRuntime &rt, uint32_t layer, XTensor &expertsSorted, XTensor &counts,
                           XTensor &num, uint32_t start, uint32_t end, uint32_t outDim,
                           std::vector<XTensor> &weights, std::vector<XTensor> &deqScales,
                           std::vector<XTensor> &scaleBias, XTensor &output)
{
    // MSD W4A8 routed-expert three-stage pipeline, group form:
    //   1. QuantDyn  : bf16 activation → xINT8 + perTokenScale[m] (FP32)
    //   2. Unpack    : xINT8 [m, k] → xINT8 [2m, k/2] interleaved (token r → row 2r low, 2r+1 high)
    //   3. GroupMatmul xInt4 [2m, k/2] × w_int4 [k/2, n] (single pass, grouped by 2×counts)
    //      → yMerged [2m, n] FP16, where rows (2r, 2r+1) hold the low/high result pair for token r
    //   4. MergeDequant: (yMerged [2m, n] + scaleBias[m, n]) * perTokenScale[m] → [m, n] BF16
    uint32_t m = expertsSorted.shape[0];
    uint32_t k = expertsSorted.shape[1];

    // Step 1: dynamic per-token quantization bf16 → int8
    XTensor &xInt8 = rt.GetTensor(expertsSorted.shape, INT8, DBG_LOC);
    XTensor &perTokenScale = rt.GetTensor({m}, FP32, DBG_LOC);
    XliteOpQuantDyn(rt, expertsSorted, perTokenScale, xInt8, num);
    rt.PutTensor(expertsSorted);

    // Step 2: unpack INT8 → interleaved double-row INT4-packed layout [2m, k/2]
    // For input token r, row 2r = low nibbles, row 2r+1 = high nibbles
    XTensor &xUnpacked = rt.GetTensor({2 * m, k}, INT4, DBG_LOC);
    XliteOpUnpackActivation(rt, xInt8, xUnpacked);
    rt.PutTensor(xInt8);

    // Step 3: INT4 group_matmul → FP16 mid-stage [2m, outDim] (single pass, interleaved rows)
    XTensor &yMerged = rt.GetTensor({2 * m, outDim}, FP16, DBG_LOC);
    XliteOpGroupMatmul(rt, xUnpacked, weights[layer], deqScales[layer], counts, start, end, INT4,
                       outDim, k, yMerged, _c.weightNZ || _c.expertsWeightNZ,
                       _c.expertsWeightTrans);
    rt.PutTensor(xUnpacked);

    // Step 4: (yMerged [2m, n] + scaleBias[m, n]) * perTokenScale[m] for target experts
    XliteOpMSDMergeDequant(rt, yMerged, scaleBias[layer], counts, start, end, perTokenScale,
                           output);
    rt.PutTensor(yMerged);
    rt.PutTensor(perTokenScale);
}

void XModel::ForwardMoE(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    // the original shape of hiddenState must not be smaller than (maxTokensDp, hiddenSize)
    size_t m = rt.currTokens, M = rt.maxTokensDp;
    uint32_t intermediateSize = _c.moeIntermediateSize / _c.moeTPSize;
    uint32_t totalPhysicalExperts = _c.nRoutedExperts + _c.numRedundantExperts;
    uint32_t nLocalPhysicalExperts = totalPhysicalExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalPhysicalExperts;
    uint32_t end = start + nLocalPhysicalExperts;
    enum XDtype moeReDtype = moeREUpGate[layer][start].dtype;

    // EPLB + MoE AllToAll is not yet supported (the AllToAll path is un-adapted for
    // totalPhysicalExperts). EPLB + DP>1 (non-AllToAll) is supported.
    if (_c.numRedundantExperts > 0 && rt.enableMoEAllToAll) {
        throw std::runtime_error("EPLB + MoE AllToAll is not yet supported");
    }

    auto [w, r] = ForwardMoEGate(rt, layer, hiddenState);
    // Remap gate outputs from logical to physical columns via log2phy so Dispatch indexes the
    // correct physical weight slots. Skipped (zero overhead) when log2phy is absent (non-EPLB).
    XTensor *wIn = &w, *rIn = &r;
    if (log2phy[layer].ptr != nullptr) {
        XTensor &wPhy = rt.GetTensor({M, totalPhysicalExperts}, w.dtype, DBG_LOC);
        XTensor &rPhy = rt.GetTensor({M, totalPhysicalExperts}, BIT1, DBG_LOC);
        XliteOpRemapGateOutputs(rt, w, r, wPhy, rPhy, log2phy[layer]);
        rt.PutTensor(w);
        rt.PutTensor(r);
        wIn = &wPhy;
        rIn = &rPhy;
    }
    // expertsScale is nullptr on non-hoist paths (AllToAll / non-INT8 / hiddenSize not a multiple
    // of 32 / graph-comm-enabled / env-disabled); a non-null pointer marks the hoisted per-token
    // scale the caller consumes.
    XTensor *weightsPtr;
    XTensor *routingPtr;
    XTensor *unpIdxPtr;
    XTensor *expertsSortedPtr;
    XTensor *expertsCountsPtr;
    XTensor *expertsScalePtr = nullptr;
    MoEAlltoAllMeta meta;
    if (rt.enableMoEAllToAll) {
        auto [w, r, ui, es, ec, m] = ForwardMoEDispatchAllToAll(rt, hiddenState, *wIn, *rIn);
        weightsPtr = &w;
        routingPtr = &r;
        unpIdxPtr = &ui;
        expertsSortedPtr = &es;
        expertsCountsPtr = &ec;
        meta = m;
    } else {
        auto [w, r, ui, es, ec, sc, m] =
            ForwardMoEDispatch(rt, hiddenState.View(M), *wIn, *rIn, moeReDtype);
        weightsPtr = &w;
        routingPtr = &r;
        unpIdxPtr = &ui;
        expertsSortedPtr = &es;
        expertsCountsPtr = &ec;
        expertsScalePtr = sc;
        meta = m;
    }
    XTensor &weights = *weightsPtr;
    XTensor &routing = *routingPtr;
    XTensor &unpIdx = *unpIdxPtr;
    XTensor &expertsSorted = *expertsSortedPtr;
    XTensor &expertsCounts = *expertsCountsPtr;
    // actual token num for current rank
    XTensor num = rt.enableMoEAllToAll ? XTensor() : XTensor({1}, INT32, unpIdx.ptr);
    bool quantHoisted = (expertsSorted.dtype == INT8) && (expertsScalePtr != nullptr);

    // routed experts
    XTensor *h13Ptr;
    if (moeReDtype == INT4) {
        // MSD W4A8: QuantDyn → Unpack → INT4 group_matmul → merge_dequant
        XTensor &h13 = rt.GetTensor({expertsSorted.shape[0], intermediateSize * 2},
                                    hiddenState.dtype, DBG_LOC);
        // NOTE: expertsSorted is released inside ForwardMoEMSD
        ForwardMoEMSD(rt, layer, expertsSorted, expertsCounts, num, start, end,
                      intermediateSize * 2, _moeREUpGate, _moeREUpGateDeqScale,
                      _moeREUpGateScaleBias, h13);
        h13Ptr = &h13;
    } else if (moeReDtype == INT8) {
        XTensor *xQuantedPtr;
        XTensor *scalePtr;
        if (quantHoisted) {
            xQuantedPtr = &expertsSorted;
            scalePtr =
                expertsScalePtr;  // non-null: quantHoisted implies expertsScalePtr != nullptr
        } else {
            XTensor &xQuanted = rt.GetTensor(expertsSorted.shape, moeReDtype, DBG_LOC);
            XTensor &scale = rt.GetTensor({expertsSorted.shape[0]}, FP32, DBG_LOC);
            XliteOpQuantDyn(rt, expertsSorted, scale, xQuanted, num);
            rt.PutTensor(expertsSorted);
            xQuantedPtr = &xQuanted;
            scalePtr = &scale;
        }

        // group_matmul(xQuanted * w13 * w13Scale) * perTokenScale -> h13
        XTensor &h13 = rt.GetTensor({expertsSorted.shape[0], intermediateSize * 2},
                                    hiddenState.dtype, DBG_LOC);
        XliteOpGroupMatmulDeQuant(
            rt, *xQuantedPtr, _moeREUpGate[layer], _moeREUpGateDeqScale[layer], expertsCounts,
            start, end, moeREUpGate[layer][start].dtype, intermediateSize * 2, _c.hiddenSize, h13,
            *scalePtr, num, _c.weightNZ || _c.expertsWeightNZ, _c.expertsWeightTrans);
        rt.PutTensor(*xQuantedPtr);
        rt.PutTensor(*scalePtr);
        h13Ptr = &h13;
    } else {
        XTensor &h13 = rt.GetTensor({expertsSorted.shape[0], intermediateSize * 2},
                                    hiddenState.dtype, DBG_LOC);
        XliteOpGroupMatmul(rt, expertsSorted, _moeREUpGate[layer], _moeREUpGateDeqScale[layer],
                           expertsCounts, start, end, moeREUpGate[layer][start].dtype,
                           intermediateSize * 2, _c.hiddenSize, h13,
                           _c.weightNZ || _c.expertsWeightNZ, _c.expertsWeightTrans);
        rt.PutTensor(expertsSorted);
        h13Ptr = &h13;
    }

    XTensor &h2 =
        rt.GetTensor({expertsSorted.shape[0], intermediateSize}, hiddenState.dtype, DBG_LOC);
    XliteOpSiluAndMul(rt, *h13Ptr, h2, num, _c.swigluLimit);
    rt.PutTensor(*h13Ptr);

    XTensor *outPtr;
    if (moeReDtype == INT4) {
        // MSD W4A8: QuantDyn → Unpack → INT4 group_matmul → merge_dequant (w2/down)
        XTensor &out =
            rt.GetTensor({expertsSorted.shape[0], _c.hiddenSize}, hiddenState.dtype, DBG_LOC);
        // NOTE: h2 is released inside ForwardMoEMSD
        ForwardMoEMSD(rt, layer, h2, expertsCounts, num, start, end, _c.hiddenSize, _moeREDown,
                      _moeREDownDeqScale, _moeREDownScaleBias, out);
        outPtr = &out;
    } else if (moeReDtype == INT8) {
        // quant(x) -> xQuanted, perChannelScale
        XTensor &xQuanted = rt.GetTensor(h2.shape, moeReDtype, DBG_LOC);
        XTensor &scale = rt.GetTensor({h2.shape[0]}, FP32, DBG_LOC);
        XliteOpQuantDyn(rt, h2, scale, xQuanted, num);
        rt.PutTensor(h2);

        // group_matmul(xQuanted * w2 * w2Scale) * perTokenScale -> out
        XTensor &out =
            rt.GetTensor({expertsSorted.shape[0], _c.hiddenSize}, hiddenState.dtype, DBG_LOC);
        XliteOpGroupMatmulDeQuant(rt, xQuanted, _moeREDown[layer], _moeREDownDeqScale[layer],
                                  expertsCounts, start, end, moeREDown[layer][start].dtype,
                                  _c.hiddenSize, intermediateSize, out, scale, num,
                                  _c.weightNZ || _c.expertsWeightNZ, _c.expertsWeightTrans);
        rt.PutTensor(xQuanted);
        rt.PutTensor(scale);
        outPtr = &out;
    } else {
        XTensor &out =
            rt.GetTensor({expertsSorted.shape[0], _c.hiddenSize}, hiddenState.dtype, DBG_LOC);
        XliteOpGroupMatmul(rt, h2, _moeREDown[layer], _moeREDownDeqScale[layer], expertsCounts,
                           start, end, moeREDown[layer][start].dtype, _c.hiddenSize,
                           intermediateSize, out, _c.weightNZ || _c.expertsWeightNZ,
                           _c.expertsWeightTrans);
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
        XTensor &h = rt.GetTensor(hiddenState.shape, hiddenState.dtype, DBG_LOC);
        if (rt.enableMoEAllToAll) {  // hiddenState of shape (currTokens = m, hiddenSize)
            ForwardMoECombineAllToAll(rt, h, weights, routing, unpIdx, *outPtr, expertsCounts,
                                      meta);
        } else {  // hiddenState of shape (maxTokensDp, hiddenSize)
            ForwardMOECombine(rt, h, weights, routing, unpIdx, *outPtr, expertsCounts);
        }
        // share experts: y = routed + sigmoid(se_gate(x)) * shared(x)
        // ForwardMLP overwrites hiddenState, so compute the scalar gate from x first.
        // se_gate is Linear(H->1) in ND (never NZ-converted); do not pass weightNZ.
        XTensor *seGateLogits = nullptr;
        if (moeSEGate[layer].ptr != nullptr) {
            seGateLogits = &rt.GetTensor({m, 1}, hiddenState.dtype, DBG_LOC);
            XliteOpMatmul(rt, hiddenState.View(m), moeSEGate[layer], *seGateLogits, false);
        }
        ForwardMLP(rt, layer, hiddenState.View(m), moeSEUpGate, moeSEDown, false);
        if (seGateLogits != nullptr) {
            XliteOpSigmoidGateMul(rt, hiddenState, *seGateLogits, hiddenState);
            rt.PutTensor(*seGateLogits);
        }
        XliteOpAdd(rt, hiddenState, h.View(m), hiddenState);
        rt.PutTensor(h);
    } else {
        if (rt.enableMoEAllToAll) {  // hiddenState of shape (currTokens = m, hiddenSize)
            ForwardMoECombineAllToAll(rt, hiddenState, weights, routing, unpIdx, *outPtr,
                                      expertsCounts, meta);
        } else {  // hiddenState of shape (maxTokensDp, hiddenSize)
            ForwardMOECombine(rt, hiddenState, weights, routing, unpIdx, *outPtr, expertsCounts);
            hiddenState.View(m);
        }
    }

    if (rt.enableMoEAllToAll && meta.expertsCountsAllEpDevice != nullptr) {
        rt.PutTensor(*meta.expertsCountsAllEpDevice);
    }

    if (_c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (!rt.enableMoEAllToAll) {
            if (rt.enableCommOptimize) {
                XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false,
                                     DBG_LOC);
            } else {
                DispatchAllReduceSum(rt, hiddenState, TP, DBG_LOC);
            }
        }
    }
}

void XModel::ForwardFFN(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    if (layer < _c.nDenseLayers) {
        ForwardMLP(rt, layer, hiddenState, mlpUpGate, mlpDown, true);
    } else {
        ForwardMoE(rt, layer, hiddenState);
    }
}

void XModel::MaybeCaptureAuxHidden(XRuntime &rt, uint32_t layer, XTensor &layerOut)
{
    if (_auxOutputs.empty() || _c.auxLayerIds.empty()) {
        return;
    }
    for (size_t k = 0; k < _c.auxLayerIds.size(); ++k) {
        if (_c.auxLayerIds[k] != layer) {
            continue;
        }
        size_t copyBytes = rt.currTokens * _c.hiddenSize * XDtypeBit(layerOut.dtype) / 8;
        aclrtMemcpyAsync(_auxOutputs[k].ptr, _auxOutputs[k].bytes, layerOut.ptr, copyBytes,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream);
        break;
    }
}

bool XModel::IsAuxLayer(uint32_t layer) const
{
    return std::any_of(_c.auxLayerIds.begin(), _c.auxLayerIds.end(),
                       [layer](uint32_t id) { return id == layer; });
}

void XModel::ForwardLayersCommOptimize(XRuntime &rt, XTensor &xPad,
                                       std::vector<std::vector<XTensor>> &kvCache,
                                       std::vector<XTensor> &deepstackInputEmbeds,
                                       XTensor &freqsCis, XTensor &output)
{
    // m: actual token num (x, h, output)
    // mPad: token num padded to match TP size (xPad, hiddenStatePad)
    // mPadPerTp: mPad / TP size (xSlice, hiddenStateSlice)
    // M: max possible token num considering both TP and DP (outputPad, original h/hiddenStatePad)
    XTensor xSlice;
    size_t m = rt.currTokens, mPad = xPad.shape[0], M = std::max(mPad, rt.maxTokensDp);
    size_t mPadPerTp = mPad / _c.defTpSize;
    size_t sizePerTp = mPad * _c.hiddenSize / _c.defTpSize * XDtypeBit(embed.dtype) / 8;
    XTensor x, h;
    x.Init({m, _c.hiddenSize}, embed.dtype, xPad.ptr);

    XTensor &outputPad = rt.GetTensor({M, _c.hiddenSize}, embed.dtype, DBG_LOC);
    h.Init({M, _c.hiddenSize}, embed.dtype, outputPad.ptr).View(m);
    rt.hiddenStatePad.Init({M, _c.hiddenSize}, embed.dtype, outputPad.ptr).View(mPad);

    void *slicePtr = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(rt.hiddenStatePad.ptr) +
                                              (rt.rankId() % _c.defTpSize) * sizePerTp);
    rt.hiddenStateSlice.Init({mPadPerTp, _c.hiddenSize}, rt.hiddenStatePad.dtype, slicePtr);
    slicePtr = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(xPad.ptr) +
                                        (rt.rankId() % _c.defTpSize) * sizePerTp);
    xSlice.Init({mPadPerTp, _c.hiddenSize}, xPad.dtype, slicePtr);

    for (uint32_t i = 0; i < _c.nLayers; i++) {
        if (i == 0) {
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1], true, attnNormBias[i]);
        }
        if (IsAuxLayer(i)) {
            XliteOpAllGather(rt, xSlice, xPad, TP, false, DBG_LOC);
            MaybeCaptureAuxHidden(rt, i, xPad);
        }
        XDEBUG_SET_STATE(_rankId == 0 && (i == 0 || i == _c.nDenseLayers));
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " in").c_str(), 1e6f);
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after attn").c_str(), 1e6f);
        XliteOpAddAndRmsNorm(rt, rt.hiddenStateSlice, xSlice, mlpNorm[i], _c.normEps,
                             rt.hiddenStateSlice, mlpNormBias[i]);
        if (!rt.enableMoEAllToAll) {
            XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP, false, DBG_LOC);
        }
        if (rt.multiTaskParallel) {
            rt.NotifyWaitPeerStream();
        }
        ForwardFFN(rt, i, rt.enableMoEAllToAll ? rt.hiddenStateSlice : h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after ffn").c_str(), 1e6f);
        if (i < _c.deepstackNumLevel) {
            XliteOpAdd(rt, h, deepstackInputEmbeds[i], h);
        }
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, rt.hiddenStateSlice, xSlice, attnNorm[i + 1], _c.normEps,
                                 rt.hiddenStateSlice, attnNormBias[i + 1]);
            XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP, false, DBG_LOC);
        }
        if (rt.multiTaskParallel) {
            rt.NotifyWaitPeerStream();
        }
    }
    XliteOpAddAndRmsNorm(rt, rt.hiddenStateSlice, xSlice, norm, _c.normEps, rt.hiddenStateSlice,
                         normBias);
    if (IsAuxLayer(_c.nLayers)) {
        XliteOpAllGather(rt, xSlice, xPad, TP, false, DBG_LOC);
        MaybeCaptureAuxHidden(rt, _c.nLayers, xPad);
    }
    XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP, false, DBG_LOC);
    aclrtMemcpyAsync(output.ptr, output.bytes, rt.hiddenStatePad.ptr, output.bytes,
                     ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream);
    rt.PutTensor(outputPad);
}

void XModel::ForwardLayersNaive(XRuntime &rt, XTensor &x,
                                std::vector<std::vector<XTensor>> &kvCache,
                                std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                XTensor &output)
{
    XTensor &h =
        rt.GetTensor({rt.maxTokensDp, _c.hiddenSize}, embed.dtype, DBG_LOC).View(rt.currTokens);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        if (i == 0) {
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1], true, attnNormBias[i]);
        }
        MaybeCaptureAuxHidden(rt, i, x);
        XDEBUG_SET_STATE(_rankId == 0 && (i == 0 || i == _c.nDenseLayers));
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " in").c_str(), 1e6f);
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after attn").c_str(), 1e6f);
        XliteOpAddAndRmsNorm(rt, h, x, mlpNorm[i], _c.normEps, h, mlpNormBias[i]);
        ForwardFFN(rt, i, h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after ffn").c_str(), 1e6f);
        if (i < _c.deepstackNumLevel) {
            XliteOpAdd(rt, h, deepstackInputEmbeds[i], h);
        }
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, h, x, attnNorm[i + 1], _c.normEps, h, attnNormBias[i + 1]);
        }
    }
    XliteOpAddAndRmsNorm(rt, h, x, norm, _c.normEps, output, normBias);
    MaybeCaptureAuxHidden(rt, _c.nLayers, x);
    rt.PutTensor(h);
}

void XModel::ForwardHcPre(XRuntime &rt, XTensor &input, XTensor &hcFn, XTensor &hcScale,
                          XTensor &hcBase, XTensor &output, XTensor &post, XTensor &comb)
{
    const uint32_t hcMult = _c.hcMult;
    const uint32_t hiddenSize = _c.hiddenSize;
    const uint32_t hcDim = hcMult * hiddenSize;
    const uint32_t outFeatures = hcFn.shape[0];  // mix_hc for attn/ffn, hc_mult for head
    const bool headOnly = (outFeatures == hcMult);

    XTensor &mixes = rt.GetTensor({input.shape[0], outFeatures}, FP32, DBG_LOC);
    XTensor &xFlat = input.View({input.shape[0], hcDim});
    XTensor &xNorm = rt.GetTensor({input.shape[0], hcDim}, FP32, DBG_LOC);
    XliteOpRmsNorm(rt, xFlat, XTensor(), xNorm, _c.normEps, hcDim, true);
    XliteOpMatmul(rt, xNorm, hcFn, mixes, false);
    input.ResetView();
    rt.PutTensor(xNorm);
    XliteOpHcAct(rt, mixes, hcScale, hcBase, post, comb, hcMult, _c.hcEps, _c.hcSinkhornIters,
                 headOnly, input, output);
    rt.PutTensor(mixes);
}

void XModel::ForwardHcPost(XRuntime &rt, XTensor &input, XTensor &post, XTensor &comb,
                           XTensor &residual, XTensor &output)
{
    const uint32_t m = post.shape[0];
    XliteOpHcPost(rt, input, post, comb, residual, output, m, _c.hcMult, _c.hiddenSize);
}

void XModel::ForwardLayersMhc(XRuntime &rt, XTensor &x, std::vector<std::vector<XTensor>> &kvCache,
                              std::vector<XTensor> &freqsCis, XTensor &output)
{
    XTensor &residual = rt.GetTensor({x.shape[0], _c.hcMult * _c.hiddenSize}, embed.dtype, DBG_LOC);
    XTensor &h =
        rt.GetTensor({rt.maxTokensDp, _c.hiddenSize}, embed.dtype, DBG_LOC).View(rt.currTokens);
    XTensor &post = rt.GetTensor({x.shape[0], _c.hcMult}, FP32, DBG_LOC);
    XTensor &comb = rt.GetTensor({x.shape[0], _c.hcMult * _c.hcMult}, FP32, DBG_LOC);
    if (x.shape.size() == 2) {
        XliteOpConcatCol(rt, std::vector<XTensor>(_c.hcMult, x), residual);
    }
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        XDEBUG_SET_STATE(_rankId == 0 && (i == 0 || i == _c.nDenseLayers));
        XDEBUG_PRINT_X(rt, i == 0 ? x : residual, ("L" + std::to_string(i) + " in").c_str(), 1e6f);
        ForwardHcPre(rt, residual, hcAttnFn[i], hcAttnScale[i], hcAttnBase[i], h, post, comb);
        XliteOpRmsNorm(rt, h, attnNorm[i], h, _c.normEps, _c.hiddenSize, true, attnNormBias[i]);
        ForwardAttnCXA(rt, i, kvCache[i], freqsCis[i], h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after attn").c_str(), 1e6f);
        ForwardHcPost(rt, h, post, comb, residual, residual);

        ForwardHcPre(rt, residual, hcFfnFn[i], hcFfnScale[i], hcFfnBase[i], h, post, comb);
        XliteOpRmsNorm(rt, h, mlpNorm[i], h, _c.normEps, _c.hiddenSize, true, mlpNormBias[i]);
        ForwardFFN(rt, i, h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after ffn").c_str(), 1e6f);
        ForwardHcPost(rt, h, post, comb, residual, residual);
    }
    ForwardHcPre(rt, residual, hcHeadFn, hcHeadScale, hcHeadBase, h, post, comb);
    XliteOpRmsNorm(rt, h, norm, output, _c.normEps, _c.hiddenSize, true, normBias);
    rt.PutTensor(comb);
    rt.PutTensor(post);
    rt.PutTensor(h);
    rt.PutTensor(residual);
}

void XModel::ForwardEmbedAndLayers(XRuntime &rt, XTensor &input,
                                   std::vector<std::vector<XTensor>> &kvCache,
                                   std::vector<XTensor> &deepstackInputEmbeds,
                                   std::vector<XTensor> &freqsCis, XTensor &h)
{
    _inputIds.Init(input.shape, input.dtype, input.ptr);
    if (_c.hcMult == 0) {
        // under DP, different ranks may routed differently with `enableCommOptimize`
        // enabled/disabled;
        // thus, `ForwardFNN` from both paths must be guarded for correct DP communications
        if (rt.enableCommOptimize) {
            size_t m = rt.currTokens, mPad = ROUND_UP(rt.currTokens, _c.defTpSize);
            XTensor &xPad = rt.GetTensor({mPad, _c.hiddenSize}, embed.dtype, DBG_LOC);
            ForwardParallelEmbed(rt, input, embed, xPad.View(m));
            ForwardLayersCommOptimize(rt, xPad.View(mPad), kvCache, deepstackInputEmbeds,
                                      freqsCis[0], h);
            rt.PutTensor(xPad);
        } else {
            XTensor &x = rt.GetTensor({rt.currTokens, _c.hiddenSize}, embed.dtype, DBG_LOC);
            ForwardParallelEmbed(rt, input, embed, x);
            ForwardLayersNaive(rt, x, kvCache, deepstackInputEmbeds, freqsCis[0], h);
            rt.PutTensor(x);
        }
    } else {
        XTensor &x = rt.GetTensor({rt.currTokens, _c.hiddenSize}, embed.dtype, DBG_LOC);
        ForwardParallelEmbed(rt, input, embed, x);
        ForwardLayersMhc(rt, x, kvCache, freqsCis, h);
        rt.PutTensor(x);
    }
}

void XModel::RegisterForwardHandlers(XRuntime &rt)
{
    auto wrap = [this](auto fn) {
        return [this, fn](XRuntime &rt, ForwardTask &t) {
            if (!t.auxOutputs.empty()) {
                SetAuxOutputs(t.auxOutputs);
            }
            fn(rt, t);
            if (!t.auxOutputs.empty()) {
                ClearAuxOutputs();
            }
        };
    };
    rt.RegisterForwardTaskHandler(ForwardTask::FORWARD, wrap([this](XRuntime &rt, ForwardTask &t) {
                                      Forward(rt, t.input, t.attnMeta, t.kvCache,
                                              t.deepstackInputEmbeds, t.freqsCis, t.output);
                                  }));
    rt.RegisterForwardTaskHandler(
        ForwardTask::FORWARD_AND_GET_LOGITS, wrap([this](XRuntime &rt, ForwardTask &t) {
            ForwardAndGetLogits(rt, t.input, t.attnMeta, t.kvCache, t.deepstackInputEmbeds,
                                t.freqsCis, t.indices, t.output);
        }));
    rt.RegisterForwardTaskHandler(
        ForwardTask::FORWARD_WITH_INPUTS_EMBEDS, wrap([this](XRuntime &rt, ForwardTask &t) {
            ForwardWithInputsEmbeds(rt, t.input, t.attnMeta, t.kvCache, t.deepstackInputEmbeds,
                                    t.freqsCisSingle, t.inputIds, t.output);
        }));
    rt.RegisterForwardTaskHandler(ForwardTask::FORWARD_GET_LOGITS,
                                  wrap([this](XRuntime &rt, ForwardTask &t) {
                                      ForwardGetLogits(rt, t.input, t.indices, t.output);
                                  }));
}

void XModel::ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &indices, XTensor &output)
{
    uint32_t batch = rt._batch;
    XTensor localOutput({output.shape[1], output.shape[2]}, output.dtype, output.ptr);

    // TODO: drop the vocabSize cutoff once CopyGmToL1 handles srcStride > 65535 for NZ.
    if (batch < input.shape[0]) {
        XTensor &x = rt.GetTensor({batch, _c.hiddenSize}, input.dtype, DBG_LOC);
        XliteOpEmbed(rt, indices, input, 0, input.shape[0], x);
        XliteOpMatmul(rt, x, head, localOutput, _c.weightNZ && _c.vocabSize <= 65536);
        rt.PutTensor(x);
    } else {
        XliteOpMatmul(rt, input, head, localOutput, _c.weightNZ && _c.vocabSize <= 65536);
    }

    if (_c.defTpSize > 1) {
        XliteOpAllGather(rt, localOutput, output, TP, false, DBG_LOC);
    }
}

void XModel::ForwardWithInputsEmbeds(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                     std::vector<std::vector<XTensor>> &kvCache,
                                     std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                     XTensor &inputIds, XTensor &output)
{
    CheckForwardParam(rt, kvCache);
    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads, _c.nKvHeads,
                   _c.blockSizes, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                   static_cast<int>(embed.dtype),
                   (_c.nDenseLayers < _c.nLayers) ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                                                  : static_cast<int>(embed.dtype),
                   _c.indexTopK, _layerTypes, _c.compressRatios);
    if (rt.batchedTokens < input.shape[0]) {
        input.View(rt.batchedTokens);
        inputIds.View(rt.batchedTokens);
    }
    rt.currTokens = input.shape[0];
    rt.maxTokensDp = PadForDp(rt) ? output.OrigShape()[0] : rt.currTokens;
    output.View(rt.currTokens);

    ConfigRtCommOptimize(rt, rt.currTokens);
    if (inputIds.ptr != nullptr && inputIds.numel != 0) {
        _inputIds.Init(inputIds.shape, inputIds.dtype, inputIds.ptr);
    }
    if (rt.enableCommOptimize) {
        size_t mPad = ROUND_UP(input.shape[0], _c.defTpSize);
        XTensor *xPadPtr, xPad;
        if (input.OrigShape()[0] >= mPad) {
            xPad.Init({mPad, _c.hiddenSize}, embed.dtype, input.ptr);
            xPadPtr = &xPad;
        } else {
            xPadPtr = &rt.GetTensor({mPad, _c.hiddenSize}, embed.dtype, DBG_LOC);
            aclrtMemcpyAsync(xPadPtr->ptr, input.bytes, input.ptr, input.bytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream);
        }
        ForwardLayersCommOptimize(rt, *xPadPtr, kvCache, deepstackInputEmbeds, freqsCis, output);
        rt.PutTensor(*xPadPtr);
    } else {
        ForwardLayersNaive(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
    }
}

void XModel::Forward(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                     std::vector<std::vector<XTensor>> &kvCache,
                     std::vector<XTensor> &deepstackInputEmbeds, std::vector<XTensor> &freqsCis,
                     XTensor &output)
{
    CheckForwardParam(rt, kvCache);
    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads, _c.nKvHeads,
                   _c.blockSizes, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                   static_cast<int>(embed.dtype),
                   (_c.nDenseLayers < _c.nLayers) ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                                                  : static_cast<int>(embed.dtype),
                   _c.indexTopK, _layerTypes, _c.compressRatios);
    if (rt.batchedTokens < input.shape[0]) {
        input.View(rt.batchedTokens);
    }
    rt.currTokens = input.shape[0];
    rt.maxTokensDp = PadForDp(rt) ? output.OrigShape()[0] : rt.currTokens;
    output.View(rt.currTokens);

    ConfigRtCommOptimize(rt, rt.currTokens);
    ForwardEmbedAndLayers(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
}

void XModel::ForwardAndGetLogits(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::vector<XTensor>> &kvCache,
                                 std::vector<XTensor> &deepstackInputEmbeds,
                                 std::vector<XTensor> &freqsCis, XTensor &indices, XTensor &output)
{
    CheckForwardParam(rt, kvCache);
    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads, _c.nKvHeads,
                   _c.blockSizes, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                   static_cast<int>(embed.dtype),
                   (_c.nDenseLayers < _c.nLayers) ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                                                  : static_cast<int>(embed.dtype),
                   _c.indexTopK, _layerTypes, _c.compressRatios);
    if (rt.batchedTokens < input.shape[0]) {
        input.View(rt.batchedTokens);
    }
    rt.currTokens = input.shape[0];
    // shape of output: world_size, numTokens, vocab_size // world_size (numTokens = maxTokensDp)
    rt.maxTokensDp = PadForDp(rt) ? output.OrigShape()[output.shape.size() - 2] : rt.currTokens;

    ConfigRtCommOptimize(rt, rt.currTokens);
    XTensor &h =
        rt.GetTensor({rt.maxTokensDp, _c.hiddenSize}, embed.dtype, DBG_LOC).View(rt.currTokens);
    ForwardEmbedAndLayers(rt, input, kvCache, deepstackInputEmbeds, freqsCis, h);
    // here, output is not contiguous by `numTokens` dimension, thus we must view `h` instead
    // TODO: change output's shape to be of `numTokens, world_size, vocab_size // world_size`
    ForwardGetLogits(rt, h.View(rt.maxTokensDp), indices, output);
    rt.PutTensor(h);
}

void XModel::CheckForwardParam(XRuntime &rt, std::vector<std::vector<XTensor>> &kvCache)
{
    if (rt.rankId() != _rankId || rt.tpSize() != _c.defTpSize || rt.dpSize() != _c.defDpSize) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": check runtime communication setting failed");
    }

    if (!rt.Inited()) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": xlite runtime not inited");
    }

    if (kvCache.size() != _c.nLayers) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": state cache size must equal nLayers");
    }

    if (IsHybridAttn(_c.attnType)) {
        uint32_t expectedKvHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
        uint32_t nLocalKHeads = _c.linearNumKHeads / _c.defTpSize;
        uint32_t nLocalVHeads = _c.linearNumVHeads / _c.defTpSize;
        if (_c.linearNumKHeads % _c.defTpSize != 0 || _c.linearNumVHeads % _c.defTpSize != 0 ||
            nLocalKHeads == 0 || nLocalVHeads == 0 || nLocalVHeads % nLocalKHeads != 0) {
            throw std::runtime_error("hybrid GDN requires divisible TP and grouped heads");
        }
        uint32_t convDim =
            nLocalKHeads * _c.linearKeyHeadDim * 2 + nLocalVHeads * _c.linearValueHeadDim;
        for (uint32_t i = 0; i < _c.nLayers; i++) {
            if (kvCache[i].size() != 2) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": each layer must provide two state tensors");
            }
            const XTensor &c0 = kvCache[i][0];
            const XTensor &c1 = kvCache[i][1];
            if (_layerTypes[i] == XMODEL_LAYER_ATTN_FULL) {
                if (c0.shape.size() != 4 || c1.shape.size() != 4 ||
                    c0.shape[1] != _c.blockSizes[0] || c1.shape[1] != _c.blockSizes[0] ||
                    c0.shape[2] != expectedKvHeads || c1.shape[2] != expectedKvHeads ||
                    c0.shape[3] != _c.headDim || c1.shape[3] != _c.headDim) {
                    throw std::runtime_error(
                        std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                        ": full-attention cache shape mismatch at layer " + std::to_string(i));
                }
            } else {
                if (!rt.IsDummyRuntime()) {
                    uint32_t width = _c.linearConvKernelDim;
                    if (width < 2 || width > 4 || convDim % 64 != 0 ||
                        _c.linearKeyHeadDim % 16 != 0 || _c.linearValueHeadDim % 64 != 0 ||
                        _c.linearKeyHeadDim > 128 || _c.linearValueHeadDim > 128 ||
                        c0.shape.size() != 3 || c0.shape[1] != width - 1 ||
                        c0.shape[2] != convDim || c0.stride.size() != 3 || c0.stride[2] != 1 ||
                        c0.stride[1] != convDim || c0.stride[0] < (width - 1) * convDim ||
                        c1.shape.size() != 4 || c1.shape[1] != nLocalVHeads ||
                        c1.shape[2] != _c.linearValueHeadDim ||
                        c1.shape[3] != _c.linearKeyHeadDim || c1.stride.size() != 4 ||
                        c1.stride[3] != 1 || c1.stride[2] != _c.linearKeyHeadDim ||
                        c1.stride[1] != _c.linearKeyHeadDim * _c.linearValueHeadDim ||
                        c1.stride[0] < nLocalVHeads * _c.linearKeyHeadDim * _c.linearValueHeadDim ||
                        c0.shape[0] < 2 || c1.shape[0] < 2 || c0.dtype != embed.dtype ||
                        (c1.dtype != embed.dtype && c1.dtype != FP32)) {
                        throw std::runtime_error(
                            "unsupported native GDN cache descriptor at layer " +
                            std::to_string(i));
                    }
                }
            }
        }
    } else if (_c.attnType == XMODEL_ATTN_MHA) {
        XTensor &kCache = kvCache[0][0];
        XTensor &vCache = kvCache[0][1];
        uint32_t expectedKvHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
        if (kCache.shape[1] != _c.blockSizes[0] || vCache.shape[1] != _c.blockSizes[0] ||
            kCache.shape[2] != expectedKvHeads || vCache.shape[2] != expectedKvHeads ||
            kCache.shape[3] != _c.headDim || vCache.shape[3] != _c.headDim) {
            throw std::runtime_error(
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                ": kv cache's shape not match [block_num, block_size, kv_head_num, head_size]");
        }
    } else if (_c.attnType == XMODEL_ATTN_MLA || _c.attnType == XMODEL_ATTN_DSA) {
        XTensor &kCache = kvCache[0][0];
        XTensor &vCache = kvCache[0][1];
        uint32_t expectedKvHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
        if (kCache.shape[1] != _c.blockSizes[0] || kCache.shape[2] != expectedKvHeads ||
            kCache.shape[3] != _c.kvLoraRank) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": k nope cache's shape not match [block_num, block_size, "
                                     "kv_head_num, kv_lora_rank]");
        }
        if (vCache.shape[1] != _c.blockSizes[0] || vCache.shape[2] != expectedKvHeads ||
            vCache.shape[3] != _c.ropeHeadDim) {
            throw std::runtime_error(
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                ": pe cache's shape not match [block_num, block_size, kv_head_num, rope_head_dim]");
        }
        if (_c.attnType == XMODEL_ATTN_DSA) {
            for (uint32_t i = 0; i < _c.nLayers; ++i) {
                if (_c.indexC8Mask.empty() || !_c.indexC8Mask[i]) {
                    continue;
                }
                if (kvCache[i].size() != 4) {
                    throw std::invalid_argument("DSA cache tuple must match indexC8Mask");
                }
                if (i < _c.indexFullMask.size() && !_c.indexFullMask[i]) {
                    continue;
                }
                const XTensor &ik = kvCache[i][2];
                if (ik.shape.size() != 4 || ik.shape[1] != _c.blockSizes[0] || ik.shape[2] != 1 ||
                    ik.shape[3] != _c.indexHeadDim || ik.dtype != INT8) {
                    throw std::invalid_argument("DSA index K cache shape/dtype mismatch");
                }
                if (embed.dtype != BF16 || kvCache[i][3].dtype != FP16 ||
                    kvCache[i][3].shape != std::vector<size_t>{ik.shape[0], ik.shape[1], 1, 1}) {
                    throw std::invalid_argument(
                        "LI-C8 requires BF16 input and a paired FP16 scale cache");
                }
            }
            XTensor &indexKCache = kvCache[0][2];
            if (indexKCache.shape[1] != _c.blockSizes[0] || indexKCache.shape[2] != 1 ||
                indexKCache.shape[3] != _c.indexHeadDim) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": DSA index k cache's shape not match [block_num, "
                                         "block_size, 1, index_head_size]");
            }
        }
    } else if (_c.attnType == XMODEL_ATTN_CXA) {
        // Per-layer tuple (see CxaCacheIndex): indexer_* on ratio==4 layers,
        // compress_kv/state on ratio!=0 layers, swa_kv on every layer; the 6th
        // scale slot is LI-C8 layers only.
        bool hasC8 = false;
        for (uint32_t i = 0; i < _c.nLayers; ++i) {
            bool c8Layer = !_c.indexC8Mask.empty() && _c.indexC8Mask[i];
            if (kvCache[i].size() != (c8Layer ? 6 : 5)) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA kv cache tuple size must match indexC8Mask");
            }
            hasC8 = hasC8 || c8Layer;
        }
        // check layer 2
        uint32_t checkLayer = _c.nLayers <= 2 ? 0 : 2;
        uint32_t ratio = _c.compressRatios[checkLayer];
        bool hasIndexer = (ratio == 4);
        bool hasCompress = (ratio != 0);
        uint32_t coff = 1 + (hasIndexer ? 1 : 0);  // overlap == (ratio == 4)
        bool c8CheckLayer = !_c.indexC8Mask.empty() && _c.indexC8Mask[checkLayer];
        if (hasC8 && !hasIndexer) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": CXA LI-C8 requires a compress_ratio==4 check layer");
        }

        auto isEmpty = [](XTensor &t) { return t.numel == 0 || t.shape.empty(); };
        auto checkShape = [](XTensor &t, uint32_t blockSize, uint32_t headNum, size_t dim,
                             const char *what) {
            if (t.shape.size() != 4 || t.shape[1] != blockSize || t.shape[2] != headNum ||
                t.shape[3] != dim) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA " + what +
                                         " shape not match [block_num, "
                                         "block_size, head_num, dim]");
            }
        };

        XTensor &indexerState = kvCache[checkLayer][0];
        XTensor &indexerK = kvCache[checkLayer][1];
        if (hasIndexer) {
            if (isEmpty(indexerState) || isEmpty(indexerK)) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA indexer cache must be non-empty on ratio==4 layer");
            }
            checkShape(indexerState, _c.blockSizes[CXA_INDEXER_STATE], 1,
                       2 * coff * _c.indexHeadDim, "indexer state cache");
            checkShape(indexerK, _c.blockSizes[CXA_INDEXER_K], 1, _c.indexHeadDim,
                       "indexer k cache");
            if (c8CheckLayer) {
                // int8 k + fp16 scale packed into one page (block-strided,
                // stride[0] >= blockSize*dim).
                const XTensor &indexerKScale = kvCache[checkLayer][CXA_INDEXER_K_SCALE];
                if (indexerK.dtype != INT8 || indexerKScale.dtype != FP16 ||
                    indexerKScale.shape !=
                        std::vector<size_t>{indexerK.shape[0], indexerK.shape[1], 1, 1}) {
                    throw std::runtime_error(
                        std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                        ": CXA LI-C8 requires an int8 indexer k cache and a paired fp16 "
                        "scale cache [block_num, block_size, 1, 1]");
                }
            } else if (indexerK.dtype != BF16) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA non-C8 indexer k cache must be bf16");
            }
        } else {
            if (!isEmpty(indexerState) || !isEmpty(indexerK)) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA indexer cache must be empty on non-ratio==4 layer");
            }
        }

        XTensor &compressKv = kvCache[checkLayer][2];
        XTensor &state = kvCache[checkLayer][3];
        if (hasCompress) {
            if (isEmpty(compressKv) || isEmpty(state)) {
                throw std::runtime_error(
                    std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                    ": CXA compress cache must be non-empty on ratio!=0 layer");
            }
            checkShape(compressKv, _c.blockSizes[CXA_COMPRESS_KV], 1, _c.headDim,
                       "compress kv cache");
            // State block_size is per-ratio (host-paged), not pinned to
            // _c.blockSizes[CXA_STATE] — validate other dims + non-zero only;
            // the forward reads the actual size from shape[1].
            if (state.shape.size() != 4 || state.shape[1] == 0 || state.shape[2] != 1 ||
                state.shape[3] != 2 * coff * _c.headDim) {
                throw std::runtime_error(
                    std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                    ": CXA state cache shape not match [block_num, block_size(per-ratio), 1, "
                    "2*coff*head_dim]");
            }
        } else {
            if (!isEmpty(compressKv) || !isEmpty(state)) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA compress cache must be empty on ratio==0 layer");
            }
        }

        XTensor &swaKv = kvCache[checkLayer][4];
        if (isEmpty(swaKv)) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": CXA swa kv cache must be non-empty");
        }
        checkShape(swaKv, _c.blockSizes[CXA_SWA_KV], 1, _c.headDim, "swa kv cache");
    }
}

size_t XModel::DummyRun()
{
    XDummyRuntime rt(0, 0, _rankId, _c.defTpSize, _c.defDpSize, _c.moeTPSize, _c.moeEpSize);
    rt.InitDummyRuntime(1ull << 40);

    _c.maxBatchedTokens = ROUND_UP(_c.maxBatchedTokens, _c.defTpSize);

    auto buildKvCache = [&](uint32_t maxNumBlocks) {
        std::vector<std::vector<XTensor>> kvCache(_c.nLayers);
        for (uint32_t i = 0; i < _c.nLayers; i++) {
            uint32_t expectedKvHeads =
                std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
            if (IsHybridAttn(_c.attnType)) {
                if (_layerTypes[i] == XMODEL_LAYER_ATTN_FULL) {
                    XTensor kCache(
                        {_c.maxBatch * maxNumBlocks, _c.blockSizes[0], expectedKvHeads, _c.headDim},
                        embed.dtype, nullptr);
                    XTensor vCache(
                        {_c.maxBatch * maxNumBlocks, _c.blockSizes[0], expectedKvHeads, _c.headDim},
                        embed.dtype, nullptr);
                    kvCache[i] = {kCache, vCache};
                } else {
                    uint32_t nLocalKHeads = _c.linearNumKHeads / _c.defTpSize;
                    uint32_t nLocalVHeads = _c.linearNumVHeads / _c.defTpSize;
                    uint32_t convDim = nLocalKHeads * _c.linearKeyHeadDim * 2 +
                                       nLocalVHeads * _c.linearValueHeadDim;
                    XTensor convState({_c.maxBatch + 1, _c.linearConvKernelDim - 1, convDim},
                                      embed.dtype, nullptr);
                    XTensor ssmState(
                        {_c.maxBatch + 1, nLocalVHeads, _c.linearValueHeadDim, _c.linearKeyHeadDim},
                        embed.dtype, nullptr);
                    kvCache[i] = {convState, ssmState};
                }
            } else if (_c.attnType == XMODEL_ATTN_MHA) {
                XTensor kCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSizes[0], expectedKvHeads, _c.headDim},
                    embed.dtype, nullptr);
                XTensor vCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSizes[0], expectedKvHeads, _c.headDim},
                    embed.dtype, nullptr);
                kvCache[i] = {kCache, vCache};
            } else if (_c.attnType == XMODEL_ATTN_MLA) {
                XTensor kCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSizes[0], expectedKvHeads, _c.kvLoraRank},
                    embed.dtype, nullptr);
                XTensor vCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSizes[0], expectedKvHeads, _c.ropeHeadDim},
                    embed.dtype, nullptr);
                kvCache[i] = {kCache, vCache};
            } else if (_c.attnType == XMODEL_ATTN_DSA) {
                XTensor kCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSizes[0], expectedKvHeads, _c.kvLoraRank},
                    embed.dtype, nullptr);
                XTensor vCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSizes[0], expectedKvHeads, _c.ropeHeadDim},
                    embed.dtype, nullptr);
                XTensor indexKCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSizes[0], 1, _c.indexHeadDim},
                    !_c.indexC8Mask.empty() && _c.indexC8Mask[i] ? INT8 : embed.dtype, nullptr);
                kvCache[i] = {kCache, vCache, indexKCache};
                if (!_c.indexC8Mask.empty() && _c.indexC8Mask[i]) {
                    kvCache[i].emplace_back(  // Cache for scale
                        std::vector<size_t>{_c.maxBatch * maxNumBlocks, _c.blockSizes[0], 1, 1},
                        FP16, nullptr);
                }
            } else if (_c.attnType == XMODEL_ATTN_CXA) {
                uint32_t ratio = _c.compressRatios.empty() ? 0 : _c.compressRatios[i];
                bool hasIndexer = (ratio == 4);
                bool hasCompress = (ratio != 0);
                uint32_t coff = 1 + (hasIndexer ? 1 : 0);
                uint32_t swaBlocks =
                    _c.maxBatch * DIV_ROUND_UP(_c.maxSeqLen, _c.blockSizes[CXA_SWA_KV]);
                uint32_t compKvBlocks =
                    hasCompress ? _c.maxBatch * DIV_ROUND_UP(_c.maxSeqLen / ratio,
                                                             _c.blockSizes[CXA_COMPRESS_KV])
                                : 0;
                uint32_t idxKvBlocks =
                    hasIndexer ? _c.maxBatch * DIV_ROUND_UP(_c.maxSeqLen / ratio,
                                                            _c.blockSizes[CXA_INDEXER_K])
                               : 0;
                uint32_t compStateBlocks =
                    hasCompress ? _c.maxBatch * DIV_ROUND_UP(_c.maxSeqLen, _c.blockSizes[CXA_STATE])
                                : 0;
                uint32_t idxStateBlocks =
                    hasIndexer
                        ? _c.maxBatch * DIV_ROUND_UP(_c.maxSeqLen, _c.blockSizes[CXA_INDEXER_STATE])
                        : 0;

                auto emptyTensor = [&]() -> XTensor { return XTensor({0}, embed.dtype, nullptr); };

                // LI-C8 layers carry a 6th fp16 scale slot; dummy tuples must
                // match registration (5 vs 6) or the pool-size walk below
                // indexes a missing slot.
                bool c8Layer = !_c.indexC8Mask.empty() && _c.indexC8Mask[i];
                XTensor indexerState =
                    hasIndexer ? XTensor({idxStateBlocks, _c.blockSizes[CXA_INDEXER_STATE], 1,
                                          2 * coff * _c.indexHeadDim},
                                         embed.dtype, nullptr)
                               : emptyTensor();
                XTensor indexerK =
                    hasIndexer
                        ? XTensor({idxKvBlocks, _c.blockSizes[CXA_INDEXER_K], 1, _c.indexHeadDim},
                                  c8Layer ? INT8 : embed.dtype, nullptr)
                        : emptyTensor();
                XTensor compressKv =
                    hasCompress
                        ? XTensor({compKvBlocks, _c.blockSizes[CXA_COMPRESS_KV], 1, _c.headDim},
                                  embed.dtype, nullptr)
                        : emptyTensor();
                XTensor state = hasCompress ? XTensor({compStateBlocks, _c.blockSizes[CXA_STATE], 1,
                                                       2 * coff * _c.headDim},
                                                      embed.dtype, nullptr)
                                            : emptyTensor();
                XTensor swaKv({swaBlocks, _c.blockSizes[CXA_SWA_KV], 1, _c.headDim}, embed.dtype,
                              nullptr);
                kvCache[i] = {indexerState, indexerK, compressKv, state, swaKv};
                if (c8Layer) {
                    kvCache[i].emplace_back(  // Cache for scale
                        std::vector<size_t>{idxKvBlocks, _c.blockSizes[CXA_INDEXER_K], 1, 1}, FP16,
                        nullptr);
                }
            }
        }
        return kvCache;
    };

    // v2 attnMeta for DummyRun: device tensors only need shape to pass CheckAttnMetaV2
    // (dummy runtime never dereferences their nullptr payloads). Family slots: one main-KV
    // block table + single slot mapping, plus a mamba state-index table for hybrid models.
    // Query lens fill the token budget as the scheduler would: queryLen per seq with a
    // clamped tail; zero-length seqs are dropped, so the batch may shrink.
    auto buildV2AttnMeta = [&](uint32_t batchSize, uint32_t queryLen, uint32_t cachedLen) {
        XModelAttnMeta attnMeta;
        attnMeta.version = 2;
        attnMeta.attnType = _c.attnType;
        uint64_t batchedTokens = 0;
        uint32_t totalLens = queryLen + cachedLen;
        for (uint32_t i = 0; i < batchSize; i++) {
            uint32_t len = 0;
            if (batchedTokens < _c.maxBatchedTokens) {
                len = static_cast<uint32_t>(
                    std::min<uint64_t>(queryLen, _c.maxBatchedTokens - batchedTokens));
            }
            if (len == 0) {
                break;
            }
            attnMeta.lensCpu.push_back(len);
            attnMeta.cachedLensCpu.push_back(cachedLen);
            batchedTokens += len;
        }
        uint32_t batch = static_cast<uint32_t>(attnMeta.lensCpu.size());
        attnMeta.lens.Init({batch}, INT32, nullptr);
        attnMeta.cachedLens.Init({batch}, INT32, nullptr);
        attnMeta.queryStartLoc.Init({batch + 1}, INT32, nullptr);
        attnMeta.position.Init({batchedTokens}, INT64, nullptr);
        for (uint32_t bs : _c.blockSizes) {
            uint32_t maxNumBlocks = DIV_ROUND_UP(totalLens, bs);
            attnMeta.blockTables.emplace_back(std::vector<size_t>{batch, maxNumBlocks}, INT32,
                                              nullptr);
        }
        if (IsHybridAttn(_c.attnType) && attnMeta.blockTables.size() < 2) {
            attnMeta.blockTables.emplace_back(std::vector<size_t>{batch, 1}, INT32, nullptr);
        }
        // slotMapping: one per blockSizes entry (per-cache) for non-hybrid; hybrid
        // carries one main-KV table (family slots live in blockTables[1..]).
        const size_t nSlotMappings = IsHybridAttn(_c.attnType) ? 1 : _c.blockSizes.size();
        for (size_t s = 0; s < nSlotMappings; s++) {
            attnMeta.slotMapping.emplace_back(std::vector<size_t>{batchedTokens}, INT32, nullptr);
        }
        // CXA DummyRun (shapes only): override Indexer-K — the generic loop
        // sizes it as per-token, but it's a per-block cache; size it ratio==4
        // per-block so pool sizing matches the real forward.
        if (_c.attnType == XMODEL_ATTN_CXA &&
            std::find(_c.compressRatios.begin(), _c.compressRatios.end(), 4u) !=
                _c.compressRatios.end()) {
            const uint32_t compBs = _c.blockSizes[CXA_COMPRESS_KV];
            uint32_t rowsPerReq4 = _c.maxSeqLen / 4;
            uint32_t compBlocksPerReq4 = DIV_ROUND_UP(rowsPerReq4, compBs);
            uint32_t nBlocksPerReq4 = ((cachedLen + queryLen) / 4) - (cachedLen / 4);
            size_t nBlocks4 = static_cast<size_t>(batchSize) * nBlocksPerReq4;
            attnMeta.slotMapping[CXA_INDEXER_K].Init(std::vector<size_t>{nBlocks4}, INT32, nullptr);
            attnMeta.blockTables[CXA_INDEXER_K].Init(
                std::vector<size_t>{batchSize, compBlocksPerReq4}, INT32, nullptr);
        }
        return attnMeta;
    };
    uint32_t spanMax =
        static_cast<uint32_t>(DIV_ROUND_UP(_c.maxSeqLen, _c.blockSizes[0]) * _c.blockSizes[0]);

    auto runPrefill = [&]() -> size_t {
        uint32_t batchSize = 1;
        uint32_t queryLen = _c.maxBatchedTokens;
        uint32_t cachedLen = _c.maxSeqLen > queryLen ? _c.maxSeqLen - queryLen : 0;
        auto attnMeta = buildV2AttnMeta(batchSize, queryLen, cachedLen);
        // kvCache is sized off blockSizes[0] (matches _attnBlockTables[0].shape[1]).
        uint32_t maxNumBlocks = DIV_ROUND_UP(queryLen + cachedLen, _c.blockSizes[0]);
        auto kvCache = buildKvCache(maxNumBlocks);
        std::vector<XTensor> deepstackInputEmbeds(_c.deepstackNumLevel);
        for (uint32_t i = 0; i < _c.deepstackNumLevel; i++) {
            XTensor deepstackEmbed({_c.maxBatchedTokens, _c.hiddenSize}, embed.dtype, nullptr);
            deepstackInputEmbeds[i] = deepstackEmbed;
        }
        XTensor freqsCis({_c.maxBatchedTokens, _c.ropeHeadDim}, embed.dtype, nullptr);
        std::vector<XTensor> freqsCisVec = {freqsCis};
        XTensor input({_c.maxBatchedTokens}, INT32, nullptr);
        XTensor output({_c.maxBatchedTokens, _c.hiddenSize}, embed.dtype, nullptr);
        XTensor logits({_c.defTpSize, _c.maxBatch, _c.vocabSize / _c.defTpSize}, embed.dtype,
                       nullptr);
        rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads,
                       _c.nKvHeads, _c.blockSizes, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                       static_cast<int>(embed.dtype),
                       (_c.nDenseLayers < _c.nLayers)
                           ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                           : static_cast<int>(embed.dtype),
                       _c.indexTopK, _layerTypes, _c.compressRatios);
        Forward(rt, input, attnMeta, kvCache, deepstackInputEmbeds, freqsCisVec, output);
        XTensor indices;
        indices.Init({batchSize}, INT32, nullptr);
        ForwardGetLogits(rt, output, indices, logits);
        return rt.maxUsedSize();
    };

    auto runDecodeAs = [&](uint32_t batchSize, uint32_t queryLen) -> size_t {
        uint32_t cachedLen =
            _c.maxSeqLen > queryLen ? static_cast<uint32_t>(_c.maxSeqLen - queryLen) : 0;
        auto attnMeta = buildV2AttnMeta(batchSize, queryLen, cachedLen);
        uint32_t batchedTokens = static_cast<uint32_t>(attnMeta.position.shape[0]);
        uint32_t batch = static_cast<uint32_t>(attnMeta.lens.shape[0]);
        // kvCache is sized off blockSizes[0] (matches _attnBlockTables[0].shape[1]).
        uint32_t maxNumBlocks = DIV_ROUND_UP(queryLen + cachedLen, _c.blockSizes[0]);
        auto kvCache = buildKvCache(maxNumBlocks);
        std::vector<XTensor> deepstackInputEmbeds(_c.deepstackNumLevel);
        for (uint32_t i = 0; i < _c.deepstackNumLevel; i++) {
            XTensor deepstackEmbed({batchedTokens, _c.hiddenSize}, embed.dtype, nullptr);
            deepstackInputEmbeds[i] = deepstackEmbed;
        }
        XTensor freqsCis({batchedTokens, _c.ropeHeadDim}, embed.dtype, nullptr);
        std::vector<XTensor> freqsCisVec = {freqsCis};
        XTensor input({batchedTokens}, INT32, nullptr);
        XTensor output({batchedTokens, _c.hiddenSize}, embed.dtype, nullptr);
        XTensor logits({_c.defTpSize, _c.maxBatch, _c.vocabSize / _c.defTpSize}, embed.dtype,
                       nullptr);
        rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads,
                       _c.nKvHeads, _c.blockSizes, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                       static_cast<int>(embed.dtype),
                       (_c.nDenseLayers < _c.nLayers)
                           ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                           : static_cast<int>(embed.dtype),
                       _c.indexTopK, _layerTypes, _c.compressRatios);
        Forward(rt, input, attnMeta, kvCache, deepstackInputEmbeds, freqsCisVec, output);
        XTensor indices;
        indices.Init({batch}, INT32, nullptr);
        ForwardGetLogits(rt, output, indices, logits);
        return rt.maxUsedSize();
    };

    auto runDecode = [&]() -> size_t {
        if (_c.maxBatch == 0) {
            return 0;
        }
        // Unset (-1): budget-derived worst case, kept inside the dense-path query gate.
        uint32_t queryLen = 1u;
        if (_c.numSpecTokens >= 0) {
            queryLen = static_cast<uint32_t>(_c.numSpecTokens) + 1;
        } else if (_c.maxBatchedTokens >= _c.maxBatch) {
            queryLen = static_cast<uint32_t>(std::min<uint64_t>(XLITE_SMALL_PREFILL_THRESHOLD,
                                                                _c.maxBatchedTokens / _c.maxBatch));
        }
        return runDecodeAs(static_cast<uint32_t>(_c.maxBatch), queryLen);
    };

    auto runDecodeDense = [&]() -> size_t {
        if (_c.maxBatch == 0 || _c.attnType != XMODEL_ATTN_DSA) {
            return 0;
        }
        if (_c.numSpecTokens >= XLITE_SMALL_PREFILL_THRESHOLD) {
            return 0;
        }
        bool hasFullIndexer =
            _c.indexFullMask.empty() ||
            std::any_of(_c.indexFullMask.begin(), _c.indexFullMask.end(), [](bool v) { return v; });
        if (!hasFullIndexer || spanMax <= _c.indexTopK || spanMax <= XLITE_MLA_DENSE_THRESHOLD) {
            return 0;
        }
        uint32_t batch0 = static_cast<uint32_t>(
            std::min<uint64_t>(_c.maxBatch, DIV_ROUND_UP(spanMax, XLITE_MLA_DENSE_THRESHOLD) - 1));
        uint32_t queryLen = _c.numSpecTokens >= 0
                                ? static_cast<uint32_t>(_c.numSpecTokens) + 1
                                : static_cast<uint32_t>(std::max<uint64_t>(
                                      std::min<uint64_t>(XLITE_SMALL_PREFILL_THRESHOLD,
                                                         _c.maxBatchedTokens / batch0),
                                      1));
        return runDecodeAs(batch0, queryLen);
    };

    return std::max<size_t>({runPrefill(), runDecode(), runDecodeDense()});
}

size_t XModel::GetTensorPoolSize(int dbg)
{
    size_t size = DummyRun();

    if (_rankId == 0 && dbg) {
        XDebugStream s(_rankId, __func__);
        s << "calculated size: " << size << " bytes (" << DIV_ROUND_UP(size, 1ull << MB_BIT)
          << " MB)" << std::endl;
    }
    return DIV_ROUND_UP(size, 1ull << MB_BIT);
}
