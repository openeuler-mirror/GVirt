/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <tuple>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "model.h"

#define KVOFFSET_OF_QKBUFFER 4096   // the kv offset of qk buffer in different cycles of mla prefill

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
    long size;
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

    size = _c.maxM * XDtypeBit(INT64) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _position.Init({_c.maxM}, INT64, ptr);

    size = _c.maxM * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _slotMapping.Init({_c.maxM}, INT32, ptr);

    size = _c.maxBatch * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _cachedLens.Init({_c.maxBatch}, INT32, ptr);

    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _lens.Init({_c.maxBatch}, INT32, ptr);

    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _cumPromptLens.Init({_c.maxBatch}, INT32, ptr);

    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _prefillIdx.Init({_c.maxBatch}, INT32, ptr);

    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _prefillLastIdx.Init({_c.maxBatch}, INT32, ptr);

    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _decodeIdx.Init({_c.maxBatch}, INT32, ptr);

    size = _c.maxBatch * DIV_ROUND_UP(_c.maxSeqLen, _c.blockSize) * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _blockTables.Init({_c.maxBatch * DIV_ROUND_UP(_c.maxSeqLen, _c.blockSize)}, INT32, ptr);

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
    }
}

XModel::~XModel(void)
{
    if (_c.nDenseLayers != _c.nLayers) {
        CHECK_ACL(aclrtFree(_gateIndicts.ptr));
    }
    for (uint32_t i = _c.nDenseLayers; i < _c.nLayers; i++) {
        CHECK_ACL(aclrtFree(_moeREUpGate[i].ptr));
        CHECK_ACL(aclrtFree(_moeREUpGateScale[i].ptr));
        CHECK_ACL(aclrtFree(_moeREDown[i].ptr));
        CHECK_ACL(aclrtFree(_moeREDownScale[i].ptr));
    }
    CHECK_ACL(aclrtFree(_position.ptr));
    CHECK_ACL(aclrtFree(_slotMapping.ptr));
    CHECK_ACL(aclrtFree(_cachedLens.ptr));
    CHECK_ACL(aclrtFree(_lens.ptr));
    CHECK_ACL(aclrtFree(_cumPromptLens.ptr));
    CHECK_ACL(aclrtFree(_prefillIdx.ptr));
    CHECK_ACL(aclrtFree(_prefillLastIdx.ptr));
    CHECK_ACL(aclrtFree(_decodeIdx.ptr));
    CHECK_ACL(aclrtFree(_blockTables.ptr));

    if (_c.attnType == XMODEL_ATTN_MLA) {
        CHECK_ACL(aclrtFree(_vGather.ptr));
    }

    if (_c.attnType == XMODEL_ATTN_MHA) {
        CHECK_ACL(aclrtFree(_a2v.ptr));
        CHECK_ACL(aclrtFree(_v2a.ptr));
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

void XModel::PrepareAttn(XRuntime &rt, XModelAttnMeta& attnMeta)
{
    uint32_t batch = attnMeta.lens.size();
    std::vector<uint32_t> lens(batch);
    std::vector<uint32_t> cachedLens(batch);
    std::vector<uint32_t> prefillIdx(batch);
    std::vector<uint32_t> decodeIdx(batch);
    std::vector<uint32_t> prefillLastIdx(batch);
    std::vector<uint32_t> cumPromptLens(batch);
    std::vector<uint32_t> slotMapping, blockTables;
    std::vector<uint64_t> position;
    uint32_t blockNum, cumPromptLen, blockId, id, k;
    size_t size;

    _realM = 0;
    _maxNumBlocks = 0;
    _prefillBatch = 0;
    _decodeBatch = 0;
    cumPromptLen = 0;
    for (uint32_t i = 0; i < batch; i++) {
        lens[i] = attnMeta.lens[i];
        cachedLens[i] = attnMeta.cachedLens[i];
        cumPromptLens[i] = cumPromptLen;
        cumPromptLen += lens[i];
        blockNum = DIV_ROUND_UP(lens[i] + cachedLens[i], _c.blockSize);
        _maxNumBlocks = blockNum > _maxNumBlocks ? blockNum : _maxNumBlocks;
        _realM += lens[i];
        prefillLastIdx[i] = _realM - 1;
        if (attnMeta.isPrefills[i]) {
            prefillIdx[_prefillBatch] = i;
            _prefillBatch++;
            _prefillLen = lens[i];
            _prefillLenPad = ROUND_UP(_prefillLen, _c.blockSize);
        } else if (lens[i] <= 2) {
            decodeIdx[_decodeBatch] = i;
            _decodeBatch++;
        } else {
            std::cerr << __FILE__ << ":" << __LINE__ << ": invalid attnMeta" << i << ", decode len too long" << lens[i] << std::endl;
            return;
        }
    }

    if (_realM > _c.maxM) {
        std::cerr << __FILE__ << ":" << __LINE__ <<
            ": invalid attnMeta realM(" << _realM << ") > maxM(" << _c.maxM << ")" << std::endl;
        return;
    }

    size = batch * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMemcpy(_lens.ptr, size, lens.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(_cachedLens.ptr, size, cachedLens.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(_cumPromptLens.ptr, size, cumPromptLens.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    if (_prefillBatch > 0) {
        CHECK_ACL(aclrtMemcpy(_prefillIdx.ptr, size, prefillIdx.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(_prefillLastIdx.ptr, size, prefillLastIdx.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    if (_decodeBatch > 0) {
        CHECK_ACL(aclrtMemcpy(_decodeIdx.ptr, size, decodeIdx.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    switch (attnMeta.version) {
        case 0:
            position.resize(_realM);
            slotMapping.resize(_realM);
            k = 0;
            for (uint32_t i = 0; i < batch; i++) {
                for (uint32_t j = 0; j < lens[i]; j++) {
                    position[k] = cachedLens[i] + j;
                    blockId = position[k] / _c.blockSize;
                    id = position[k] % _c.blockSize;
                    slotMapping[k++] = attnMeta.blockTables[i][blockId] * _c.blockSize + id;
                }
            }
            size = _realM * XDtypeBit(INT64) / 8;
            CHECK_ACL(aclrtMemcpy(_position.ptr, size, position.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            size = _realM * XDtypeBit(INT32) / 8;
            CHECK_ACL(aclrtMemcpy(_slotMapping.ptr, size, slotMapping.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));

            blockTables.resize(batch * _maxNumBlocks);
            for (uint32_t i = 0; i < batch; i++) {
                blockNum = DIV_ROUND_UP(lens[i] + cachedLens[i], _c.blockSize);
                for (uint32_t j = 0; j < blockNum; j++) {
                    blockTables[i * _maxNumBlocks + j] = attnMeta.blockTables[i][j];
                }
            }
            size = batch * _maxNumBlocks * XDtypeBit(INT32) / 8;
            CHECK_ACL(aclrtMemcpy(_blockTables.ptr, size, blockTables.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _attnPosition = _position;
            _attnBlockTables = _blockTables;
            _attnSlotMapping = _slotMapping;
            break;
        case 1:
            _maxNumBlocks = attnMeta.vllmBlockTables.shape[1];
            _attnPosition = attnMeta.vllmPosition;
            _attnBlockTables = attnMeta.vllmBlockTables;
            _attnSlotMapping = attnMeta.vllmSlotMapping;
            break;
        default:
            std::cerr << __FILE__ << ":" << __LINE__ << ": invalid attnMeta version: " << attnMeta.version << std::endl;
            return;
    }
}

std::tuple<XTensor &, XTensor &, XTensor &> XModel::ForwardAttnMLACommon(XRuntime &rt, uint32_t layer,
                                                                    std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                                                    XTensor &freqsCis, XTensor &hiddenState)
{
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;
    XTensor &kCache = kvCache[layer].first;
    XTensor &vCache = kvCache[layer].second;
    XTensor &attnQc = rt.pool->GetTensor({_realM, _c.qLoraRank}, hiddenState.dtype);
    XTensor &attnNormQc = rt.pool->GetTensor({_realM, _c.qLoraRank}, hiddenState.dtype);
    XTensor &attnQWithQr = rt.pool->GetTensor({_realM, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim}, hiddenState.dtype);
    XTensor &attnKvc = rt.pool->GetTensor({_realM, _c.kvLoraRank + _c.ropeHeadDim}, hiddenState.dtype);
    XTensor &attnNormKvc = rt.pool->GetTensor({_realM, _c.kvLoraRank}, hiddenState.dtype);
    XTensor &attnKPe = rt.pool->GetTensor({_realM, _c.ropeHeadDim}, hiddenState.dtype);
    XTensor &attnQPe = rt.pool->GetTensor({_realM, nLocalHeads, _c.ropeHeadDim}, hiddenState.dtype);

    XliteOpMatmul(rt, hiddenState, mlaQA[layer], attnQc, _c.weightNZ);
    XliteOpRmsNorm(rt, attnQc, mlaQNorm[layer], attnNormQc, _c.normEps, attnQc.shape[1]);
    XliteOpMatmul(rt, attnNormQc, mlaQB[layer], attnQWithQr, _c.weightNZ);
    XliteDsOpRopeBatch(rt, _realM, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim, _c.ropeHeadDim,
                       attnQWithQr, freqsCis, _attnPosition, _vGather, attnQPe, _prefillBatch > 0 ? MIX : NORMAL);
    XliteOpMatmul(rt, hiddenState, mlaKVA[layer], attnKvc, _c.weightNZ);
    XliteDsOpStridedRmsnorm(rt, attnKvc, mlaKVNorm[layer], attnNormKvc, _realM, _c.kvLoraRank,
                            _c.kvLoraRank + _c.ropeHeadDim, _c.normEps);
    XliteDsOpRopeBatch(rt, _realM, 1, _c.kvLoraRank + _c.ropeHeadDim, _c.ropeHeadDim,
                       attnKvc, freqsCis, _attnPosition, _vGather, attnKPe, NORMAL);
    XliteDsOpReshapeAndCache(rt, attnNormKvc, attnKPe, kCache, vCache, _attnSlotMapping, _realM, _c.kvLoraRank,
                             _c.ropeHeadDim, 1, _c.kvLoraRank, _c.ropeHeadDim, _c.blockSize, kCache.shape[0]);

    rt.pool->PutTensor(attnQc);
    rt.pool->PutTensor(attnNormQc);
    rt.pool->PutTensor(attnKvc);
    rt.pool->PutTensor(attnNormKvc);

    return {attnQWithQr, attnKPe, attnQPe};
}

XTensor& XModel::ForwardAttnMLAPrefill(XRuntime &rt, uint32_t layer,
                                       std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                       XTensor &freqsCis, XTensor &hiddenState,
                                       XTensor &attnQWithQr, XTensor &attnKPe, XTensor &attnQPe)
{
    XTensor &kCache = kvCache[layer].first;
    XTensor &vCache = kvCache[layer].second;
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;
    uint32_t mlaPaddingLen = ROUND_UP(_realM, _c.blockSize);
    uint32_t mlaPmSize = TILESIZE_OF_QUERY, mlaPerKvlen = KVOFFSET_OF_QKBUFFER, coreNum = rt.aicNum;
    XTensor &attnQkcAbsorb = rt.pool->GetTensor({_realM, nLocalHeads, _c.vHeadDim}, hiddenState.dtype);
    XTensor &attnKRes = rt.pool->GetTensor({mlaPaddingLen, nLocalHeads * (_c.nopeHeadDim + _c.vHeadDim)}, hiddenState.dtype);
    XTensor &attnKvFull = rt.pool->GetTensor({mlaPaddingLen, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim}, hiddenState.dtype);
    XTensor &attnV = rt.pool->GetTensor({mlaPaddingLen, nLocalHeads, _c.vHeadDim}, hiddenState.dtype);
    XTensor &attnPrefillOut = rt.pool->GetTensor({_realM, nLocalHeads, _c.vHeadDim}, hiddenState.dtype);
    XTensor &attnMlaOut = rt.pool->GetTensor({coreNum * mlaPmSize * _c.vHeadDim * 2}, FP32);
    XTensor &attnMlaMax = rt.pool->GetTensor({coreNum * mlaPmSize * 8}, FP32);
    XTensor &attnMlaSum = rt.pool->GetTensor({coreNum * mlaPmSize * 8 * 2}, FP32);
    XTensor &attnMlaAlpha = rt.pool->GetTensor({coreNum * mlaPmSize * 8 * 2}, FP32);
    XTensor &attnMlaQK = rt.pool->GetTensor({2, coreNum, mlaPmSize, mlaPerKvlen}, BF16);
    XTensor &attnMlaTmp = rt.pool->GetTensor({1}, INT32);

    XliteDsOpKvMatmul(rt, kCache, mlaKVB[layer], attnKRes, _prefillLenPad, nLocalHeads * (_c.nopeHeadDim + _c.vHeadDim),
                      _c.kvLoraRank, _attnBlockTables, true, _c.blockSize, _c.kvLoraRank);
    XliteDsOpPrefillKvSplit(rt, attnKRes, attnKPe, vCache, _attnBlockTables, attnKvFull, attnV, _prefillLen,
                            _prefillLenPad, nLocalHeads, 0, _c.ropeHeadDim, _c.nopeHeadDim, _c.vHeadDim, _c.blockSize);
    XliteDsOpPrefillMix(rt, attnMlaOut, attnMlaAlpha, attnMlaMax, attnMlaSum, attnQWithQr, attnKvFull,
                        attnMlaQK, attnMlaTmp, _cachedLens, attnV, attnPrefillOut, attnQkcAbsorb,
                        _lens, attnMlaTmp, attnMlaTmp, attnMlaTmp, _prefillIdx, _cumPromptLens,
                        _c.vHeadDim, nLocalHeads, nLocalHeads, _c.blockSize, _prefillBatch, 0, 0, 0, 0, _c.softmaxScale);

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

XTensor& XModel::ForwardAttnMLADecode(XRuntime &rt, uint32_t layer,
                                      std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                      XTensor &freqsCis, XTensor &hiddenState,
                                      XTensor &attnQWithQr, XTensor &attnKPe, XTensor &attnQPe)
{
    uint32_t batch = hiddenState.shape[0];
    XTensor &kCache = kvCache[layer].first;
    XTensor &vCache = kvCache[layer].second;
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;
    XTensor &attnQkcAbsorb = rt.pool->GetTensor({_realM, nLocalHeads, _c.vHeadDim}, hiddenState.dtype);
    XTensor &attnQAbsorb = rt.pool->GetTensor({_realM, nLocalHeads, _c.kvLoraRank}, hiddenState.dtype);
    XTensor &attnQk = rt.pool->GetTensor({batch, nLocalHeads, _c.maxSeqLen}, hiddenState.dtype);
    XTensor &attnQkc = rt.pool->GetTensor({_realM, nLocalHeads, _c.kvLoraRank}, hiddenState.dtype);

    XliteDsOpEinsumShdHdcShc(rt, _realM, _c.nopeHeadDim, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim,
                             2 * _c.nopeHeadDim, _c.kvLoraRank, attnQWithQr, mlaKVB[layer], attnQAbsorb);
    XliteDsOpDecodeAttn(rt, attnQAbsorb, kCache, attnQk, _cachedLens, _attnBlockTables, _lens, _cumPromptLens, batch,
                        nLocalHeads, 1, _c.kvLoraRank, _c.blockSize, _maxNumBlocks, _c.maxSeqLen, false);
    XliteDsOpDecodeAttn(rt, attnQPe, vCache, attnQk, _cachedLens, _attnBlockTables, _lens, _cumPromptLens, batch,
                        nLocalHeads, 1, _c.ropeHeadDim, _c.blockSize, _maxNumBlocks, _c.maxSeqLen, true);
    XliteDsOpSoftmax(rt, attnQk, _cachedLens, _lens, _cumPromptLens, _c.softmaxScale, batch, nLocalHeads,
                     _c.blockSize, _c.maxSeqLen);
    XliteDsOpEinsumShtTcShc(rt, batch, nLocalHeads, _c.maxSeqLen, _maxNumBlocks, kCache.shape[0], _c.blockSize,
                            _c.kvLoraRank, attnQk, _cachedLens, _lens, _cumPromptLens, _attnBlockTables, kCache, attnQkc);
    XliteDsOpEinsumShcHdcShd(rt, _realM, nLocalHeads, _c.kvLoraRank, 2 * _c.nopeHeadDim, _c.vHeadDim, attnQkc,
                             mlaKVB[layer], attnQkcAbsorb);

    rt.pool->PutTensor(attnQWithQr);
    rt.pool->PutTensor(attnKPe);
    rt.pool->PutTensor(attnQPe);
    rt.pool->PutTensor(attnQAbsorb);
    rt.pool->PutTensor(attnQk);
    rt.pool->PutTensor(attnQkc);
    return attnQkcAbsorb;
}

void XModel::ForwardAttnMLA(XRuntime &rt, uint32_t layer,
                            std::vector<std::pair<XTensor, XTensor>>& kvCache,
                            XTensor &freqsCis, XTensor &hiddenState)
{
    auto [attnQWithQr, attnKPe, attnQPe] = ForwardAttnMLACommon(rt, layer, kvCache, freqsCis, hiddenState);

    XTensor &attnOutput = (_prefillBatch > 0
                           ? ForwardAttnMLAPrefill(rt, layer, kvCache, freqsCis, hiddenState, attnQWithQr, attnKPe, attnQPe)
                           : ForwardAttnMLADecode(rt, layer, kvCache, freqsCis, hiddenState, attnQWithQr, attnKPe, attnQPe));

    XliteOpMatmul(rt, attnOutput, attnOut[layer], hiddenState, _c.weightNZ);

    if (_c.defTpSize > 1) {
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, hiddenState, rt.hiddenStateSlice, TP);
        } else {
            XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP);
        }
    }
    rt.pool->PutTensor(attnOutput);
}

void XModel::XliteOpAttention(XRuntime &rt, uint32_t layer, XTensor &kCache, XTensor &vCache,
                                XTensor &input, XTensor &output)
{
    if (_prefillBatch > 0) {
        XTensor &qk = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, _maxNumBlocks * _c.blockSize}, input.dtype);
        XliteOpPrefillAttention(rt, input, kCache, qk, _attnBlockTables, _cachedLens,
                                vCache, output, _lens, _prefillIdx, _cumPromptLens,
                                _c.headDim, _c.nHeads, _c.nKvHeads, _c.blockSize,
                                _prefillBatch, _maxNumBlocks);
        rt.pool->PutTensor(qk);
    }
    if (_decodeBatch > 0) {
        XTensor &qk = rt.pool->GetTensor({_decodeBatch, _c.nHeads / _c.defTpSize, _maxNumBlocks * _c.blockSize}, input.dtype);
        XliteOpDecodeAttention(rt, _a2v, _v2a, input, kCache, vCache, _cachedLens, _attnBlockTables, qk,
                               output, _decodeIdx, _cumPromptLens, _decodeBatch, _c.nHeads, _c.headDim,
                               _c.blockSize, _maxNumBlocks, _c.nKvHeads);
        rt.pool->PutTensor(qk);
    }
}

void XModel::ForwardAttnMHA(XRuntime &rt, uint32_t layer,
                            std::vector<std::pair<XTensor, XTensor>>& kvCache,
                            XTensor &freqsCis, XTensor &hiddenState)
{
    XTensor &qkv = rt.pool->GetTensor({_realM, mhaQKV[layer].shape[0]}, hiddenState.dtype);
    XliteOpMatmul(rt, hiddenState, mhaQKV[layer], qkv, _c.weightNZ);
    if (_c.addBias) {
        XliteOpAddBias(rt, qkv, mhaQKVBias[layer], qkv);
    }
    if (_c.qkNorm) {
        uint32_t qHeads = _c.nHeads / _c.defTpSize;
        uint32_t kHeads = std::max(_c.nKvHeads / _c.defTpSize, uint32_t(1));
        XliteOpRmsNorm(rt, qkv, mhaQNorm[layer], qkv, _c.normEps, _c.headDim, qHeads);
        XliteOpRmsNorm(rt, qkv, mhaKNorm[layer], qkv, _c.normEps, _c.headDim, kHeads, qHeads * _c.headDim);
    }
    XliteOpRopeCache(rt, qkv, kvCache[layer].first, kvCache[layer].second, _attnPosition, freqsCis,
                     _attnSlotMapping, _c.nHeads, _c.nKvHeads, _c.headDim,
                     _c.ropeHeadDim, _c.blockSize, _c.ropeType == XMODEL_ROPE_NEOX);
    XTensor &attn = rt.pool->GetTensor({hiddenState.shape[0], attnOut[layer].shape[1]}, hiddenState.dtype);
    XliteOpAttention(rt, layer, kvCache[layer].first, kvCache[layer].second, qkv, attn);
    XliteOpMatmul(rt, attn, attnOut[layer], hiddenState, _c.weightNZ);
    if (_c.defTpSize > 1) {
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, hiddenState, rt.hiddenStateSlice, TP);
        } else {
            XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP);
        }
    }
    rt.pool->PutTensor(qkv);
    rt.pool->PutTensor(attn);
}

void XModel::ForwardAttn(XRuntime &rt, uint32_t layer,
                         std::vector<std::pair<XTensor, XTensor>>& kvCache,
                         XTensor &freqsCis, XTensor &hiddenState)
{
    if (_c.attnType == XMODEL_ATTN_MLA) {
        ForwardAttnMLA(rt, layer, kvCache, freqsCis, hiddenState);
    } else if (_c.attnType == XMODEL_ATTN_MHA) {
        ForwardAttnMHA(rt, layer, kvCache, freqsCis, hiddenState);
    } else {
        std::cout << __func__ << ": TODO" << std::endl;
    }
}

void XModel::ForwardMLP(XRuntime &rt, XTensor &upGate, XTensor &down, XTensor &hiddenState, bool withAllReduce)
{
    uint32_t m = hiddenState.shape[0];
    XTensor &h13 = rt.pool->GetTensor({m, upGate.shape[0]}, hiddenState.dtype);
    XTensor &h2 = rt.pool->GetTensor({m, down.shape[1]}, hiddenState.dtype);

    XliteOpMatmul(rt, hiddenState, upGate, h13, _c.weightNZ);
    XliteOpSiluAndMul(rt, h13, h2);
    XliteOpMatmul(rt, h2, down, hiddenState, _c.weightNZ);

    if (withAllReduce && _c.defTpSize > 1) {
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, hiddenState, rt.hiddenStateSlice, TP);
        } else {
            XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP);
        }
    }
    rt.pool->PutTensor(h2);
    rt.pool->PutTensor(h13);
}

std::tuple<XTensor &, XTensor &> XModel::ForwardMoEGate(XRuntime &rt, uint32_t layer, XTensor &input)
{
    uint32_t m = input.shape[0];
    XTensor &weights = rt.pool->GetTensor({m, _c.nRoutedExperts}, moeGate[layer].dtype);
    XTensor &routing = rt.pool->GetTensor({m, _c.nRoutedExperts}, BIT1);
    XTensor &scores = rt.pool->GetTensor({input.shape[0], _c.nRoutedExperts}, moeGate[layer].dtype);

    XliteOpMatmul(rt, input, moeGate[layer], scores, _c.weightNZ);

    if (_c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
        XliteOpSigmoidTopK(rt, scores, moeGateBias[layer], _gateIndicts, _c.nExpertGroups, _c.nLimitedGroups,
                           _c.nActExperts, _c.routeScale, weights, routing);
    } else {
        XliteOpSoftmaxTopK(rt, scores, _gateIndicts, weights, routing, _c.nActExperts, _c.normTopKProb);
    }

    rt.pool->PutTensor(scores);
    return {weights, routing};
}

std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &> XModel::ForwardMoEDispatch(XRuntime &rt,
    XTensor &tokenSorted, XTensor &weights, XTensor &routing)
{
    uint32_t m = tokenSorted.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;
    XTensor &unpIdx = rt.pool->GetTensor({_c.nRoutedExperts, mAllDp + 1}, INT32);
    XTensor &expertsSorted = rt.pool->GetTensor({mAllDp * _c.nActExperts, _c.hiddenSize}, tokenSorted.dtype);
    XTensor &expertsCounts = rt.pool->GetTensor({_c.nRoutedExperts, 1}, INT32);

    if (_c.defDpSize > 1) {
        XTensor &inputPerDp = tokenSorted, &weightsPerDp = weights, &routingPerDp = routing;
        XTensor &inputAllDp = rt.pool->GetTensor({mAllDp, _c.hiddenSize}, inputPerDp.dtype);
        XTensor &weightsAllDp = rt.pool->GetTensor({mAllDp, _c.nRoutedExperts}, weightsPerDp.dtype);
        XTensor &routingAllDp = rt.pool->GetTensor({mAllDp, _c.nRoutedExperts}, routingPerDp.dtype);
        XliteOpAllGather(rt, inputPerDp, inputAllDp, DP);
        XliteOpAllGather(rt, weightsPerDp, weightsAllDp, DP);
        XliteOpAllGather(rt, routingPerDp, routingAllDp, DP);
        XliteOpPermutation(rt, inputAllDp, routingAllDp, start, end, expertsSorted, unpIdx, expertsCounts);
        rt.pool->PutTensor(routingPerDp);
        rt.pool->PutTensor(weightsPerDp);
        rt.pool->PutTensor(inputAllDp);
        return {weightsAllDp, routingAllDp, unpIdx, expertsSorted, expertsCounts};
    } else {
        XliteOpPermutation(rt, tokenSorted, routing, start, end, expertsSorted, unpIdx, expertsCounts);
        return {weights, routing, unpIdx, expertsSorted, expertsCounts};
    }
}

void XModel::ForwardMOECombine(XRuntime &rt, XTensor &tokenSorted, XTensor &weights, XTensor &routing, XTensor &unpIdx,
                               XTensor &expertsSorted, XTensor &expertsCounts)
{
    uint32_t m = tokenSorted.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;
    

    if (_c.defDpSize > 1) {
        XTensor &tokenSortedAllDp = rt.pool->GetTensor({mAllDp, _c.hiddenSize}, tokenSorted.dtype);
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, tokenSortedAllDp);
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
    auto [weights, routing, unpIdx, expertsSorted, expertsCounts] = ForwardMoEDispatch(rt, hiddenState, w, r);

    // routed experts
    XTensor &h13 = rt.pool->GetTensor({mAllDp * _c.nActExperts, intermediateSize * 2}, hiddenState.dtype);
    XTensor &h2 = rt.pool->GetTensor({mAllDp * _c.nActExperts, intermediateSize}, hiddenState.dtype);
    XliteOpGroupMatmul(rt, expertsSorted, _moeREUpGate[layer], _moeREUpGateScale[layer], expertsCounts,
        start, end, moeREUpGate[layer][start].dtype, intermediateSize * 2, _c.hiddenSize, h13, _c.weightNZ, _c.expertsWeightTrans);
    XliteOpSiluAndMul(rt, h13, h2);
    XliteOpGroupMatmul(rt, h2, _moeREDown[layer], _moeREDownScale[layer], expertsCounts,
        start, end, moeREDown[layer][start].dtype, _c.hiddenSize, intermediateSize, expertsSorted, _c.weightNZ, _c.expertsWeightTrans);
    rt.pool->PutTensor(h13);
    rt.pool->PutTensor(h2);

    if (_c.nSharedExperts != 0) {
        XTensor &h = rt.pool->GetTensor({m, _c.hiddenSize}, hiddenState.dtype);
        ForwardMOECombine(rt, h, weights, routing, unpIdx, expertsSorted, expertsCounts);
        // share experts
        ForwardMLP(rt, moeSEUpGate[layer], moeSEDown[layer], hiddenState, false);
        XliteOpAdd(rt, hiddenState, h, hiddenState);
        rt.pool->PutTensor(h);
    } else {
        ForwardMOECombine(rt, hiddenState, weights, routing, unpIdx, expertsSorted, expertsCounts);
    }

    if (_c.defTpSize > 1) {
        if (rt.enableCommOptimize) {
            XliteOpReduceScatter(rt, hiddenState, rt.hiddenStateSlice, TP);
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

void XModel::ForwardLayersCommOptimize(XRuntime &rt, XTensor &x,
                                       std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                       XTensor &freqsCis, XTensor &output)
{
    XTensor xSlice;
    XTensor &h = rt.pool->GetTensor({x.shape[0], _c.hiddenSize}, embed.dtype);
    void *slicePtr = (void *)((uint64_t)h.ptr + rt.rankId() * h.numel / _c.defTpSize * XDtypeBit(h.dtype) / 8);
    rt.hiddenStateSlice.Init({h.shape[0] / _c.defTpSize, h.shape[1]}, h.dtype, slicePtr);
    slicePtr = (void *)((uint64_t)x.ptr + rt.rankId() * x.numel / _c.defTpSize * XDtypeBit(x.dtype) / 8);
    xSlice.Init({x.shape[0] / _c.defTpSize, x.shape[1]}, x.dtype, slicePtr);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        if (i == 0) {
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1]);
        }
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XliteOpAddAndRmsNorm(rt, xSlice, rt.hiddenStateSlice, mlpNorm[i], _c.normEps, rt.hiddenStateSlice);
        XliteOpAllGather(rt, rt.hiddenStateSlice, h, TP);
        ForwardFFN(rt, i, h);
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, xSlice, rt.hiddenStateSlice, attnNorm[i + 1], _c.normEps, rt.hiddenStateSlice);
            XliteOpAllGather(rt, rt.hiddenStateSlice, h, TP);
        }  
    }
    XliteOpAddAndRmsNorm(rt, xSlice, rt.hiddenStateSlice, norm, _c.normEps, rt.hiddenStateSlice);
    XliteOpAllGather(rt, rt.hiddenStateSlice, output, TP);
    rt.pool->PutTensor(h);
}

void XModel::ForwardLayersNaive(XRuntime &rt, XTensor &x,
                                std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                XTensor &freqsCis, XTensor &output)
{
    XTensor &h = rt.pool->GetTensor({x.shape[0], _c.hiddenSize}, embed.dtype);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        if (i == 0) {
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1]);
        }
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XliteOpAddAndRmsNorm(rt, x, h, mlpNorm[i], _c.normEps, h);
        ForwardFFN(rt, i, h);
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, x, h, attnNorm[i + 1], _c.normEps, h);
        }  
    }
    XliteOpAddAndRmsNorm(rt, x, h, norm, _c.normEps, output);
    rt.pool->PutTensor(h);
}

void XModel::ForwardLayers(XRuntime &rt, XTensor &x,
                           std::vector<std::pair<XTensor, XTensor>>& kvCache,
                           XTensor &freqsCis, XTensor &h)
{
    if (_c.defTpSize > 1 && h.shape[0] >= rt.commOptimizeLen &&
        h.shape[0] % _c.defTpSize == 0) {
        rt.enableCommOptimize = true;
        ForwardLayersCommOptimize(rt, x, kvCache, freqsCis, h);
    } else {
        rt.enableCommOptimize = false;
        ForwardLayersNaive(rt, x, kvCache, freqsCis, h);
    }
}

void XModel::ForwardEmbedAndLayers(XRuntime &rt, XTensor &input,
                                   std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                   XTensor &freqsCis, XTensor &h)
{
    XTensor &x = rt.pool->GetTensor({input.shape[0], _c.hiddenSize}, embed.dtype);
    ForwardParallelEmbed(rt, input, embed, x);
    ForwardLayers(rt, x, kvCache, freqsCis, h);
    rt.pool->PutTensor(x);
}

void XModel::ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &output)
{
    uint32_t batch = _prefillBatch + _decodeBatch;
    XTensor localOutput({output.shape[1], output.shape[2]}, output.dtype, output.ptr);

    if (batch < input.shape[0]) {
        XTensor &x = rt.pool->GetTensor({batch, _c.hiddenSize}, input.dtype);
        XliteOpEmbed(rt, _prefillLastIdx, input, 0, _realM, x);
        XliteOpMatmul(rt, x, head, localOutput, _c.weightNZ);
        rt.pool->PutTensor(x);
    } else {
        XliteOpMatmul(rt, input, head, localOutput, _c.weightNZ);
    }

    if (_c.defTpSize > 1) {
        XliteOpAllGather(rt, localOutput, output, TP);
    }
}

void XModel::ForwardWithInputsEmbeds(XRuntime &rt, XTensor &input,
                                     XModelAttnMeta& attnMeta,
                                     std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                     XTensor &freqsCis, XTensor &output)
{
    if (rt.rankId() != _rankId || rt.tpSize() != _c.defTpSize || rt.dpSize() != _c.defDpSize) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": check runtime communication setting failed" << std::endl;
        return;
    }
    if (!rt.pool) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": xlite runtime's tensor pool not inited" << std::endl;
        return;
    }

    PrepareAttn(rt, attnMeta);
    ForwardLayers(rt, input, kvCache, freqsCis, output);
}

void XModel::Forward(XRuntime &rt, XTensor &input,
                     XModelAttnMeta& attnMeta,
                     std::vector<std::pair<XTensor, XTensor>>& kvCache,
                     XTensor &freqsCis, XTensor &output)
{
    if (rt.rankId() != _rankId || rt.tpSize() != _c.defTpSize || rt.dpSize() != _c.defDpSize) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": check runtime communication setting failed" << std::endl;
        return;
    }
    if (!rt.pool) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": xlite runtime's tensor pool not inited" << std::endl;
        return;
    }

    PrepareAttn(rt, attnMeta);
    ForwardEmbedAndLayers(rt, input, kvCache, freqsCis, output);
}

void XModel::ComputeLogits(XRuntime &rt, XTensor &input, XTensor &output)
{
    if (!rt.pool) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": xlite runtime's tensor pool not inited" << std::endl;
        return;
    }

    XTensor localOutput({input.shape[0], head.shape[0]}, head.dtype, output.ptr);
    XliteOpMatmul(rt, input, head, localOutput, _c.weightNZ);
    if (_c.defTpSize > 1) {
        XliteOpAllGather(rt, localOutput, output, TP);
    }
}

void XModel::ForwardAndGetLogits(XRuntime &rt, XTensor &input,
                                 XModelAttnMeta& attnMeta,
                                 std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                 XTensor &freqsCis, XTensor &output)
{
    uint32_t m = input.shape[0];

    if (rt.rankId() != _rankId || rt.tpSize() != _c.defTpSize || rt.dpSize() != _c.defDpSize) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": check runtime communication setting failed" << std::endl;
        return;
    }
    if (!rt.pool) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": xlite runtime's tensor pool not inited" << std::endl;
        return;
    }

    XTensor &h = rt.pool->GetTensor({m, _c.hiddenSize}, embed.dtype);
    PrepareAttn(rt, attnMeta);
    ForwardEmbedAndLayers(rt, input, kvCache, freqsCis, h);
    ForwardGetLogits(rt, h, output);
    rt.pool->PutTensor(h);
}

size_t XModel::GetTensorPoolSize(void)
{
    int dtypeSize = XDtypeBit(embed.dtype) / 8;
    size_t attnSize, ffnSize, prefillBufSize, decodeBufSize;
    size_t mlpBufSize = 0;
    size_t moeBufSize = 0;
    size_t scoresBufSize = 0;
    size_t moeDispatchDpBufSize = 0;
    size_t routedExpertsBufSize = 0;
    size_t shareExpertsBufSize = 0;
    size_t size = 0;

    // TODO
    if (_c.attnType != XMODEL_ATTN_MHA) {
        return 1024;
    }

    size = _c.maxM * _c.hiddenSize * 3 * dtypeSize;
    attnSize = _c.maxM * mhaQKV[0].shape[0] * dtypeSize;
    attnSize += _c.maxM * attnOut[0].shape[1] * dtypeSize;
    prefillBufSize = AIC_MAX_NUM * TILESIZE_OF_QUERY * 2 * _c.maxSeqLen * dtypeSize;
    decodeBufSize = _c.maxBatch * _c.nHeads / _c.defTpSize * _c.maxSeqLen * dtypeSize;
    attnSize += std::max(prefillBufSize, decodeBufSize);

    // FFN MLP
    if (_c.nDenseLayers > 0) {
        mlpBufSize += _c.maxM * mlpUpGate[0].shape[0] * dtypeSize;
        mlpBufSize += _c.maxM * mlpDown[0].shape[1] * dtypeSize;
    }
    // FFN MoE
    if (_c.nDenseLayers < _c.nLayers) {
        // MoE gate
        moeBufSize += _c.maxM * _c.nRoutedExperts * dtypeSize;
        moeBufSize += _c.maxM * _c.nRoutedExperts / 8;
        scoresBufSize += _c.maxM * _c.nRoutedExperts * dtypeSize;
        // MoE dispatch
        moeBufSize += (_c.maxM * _c.defDpSize + 1) * _c.nRoutedExperts * sizeof(int32_t);
        moeBufSize += _c.maxM * _c.defDpSize * _c.nActExperts * _c.hiddenSize * dtypeSize;
        moeBufSize += _c.nRoutedExperts * sizeof(int32_t);
        if (_c.defDpSize > 1) {
            moeDispatchDpBufSize += _c.maxM * _c.defDpSize * _c.hiddenSize * dtypeSize;
            moeDispatchDpBufSize += _c.maxM * _c.defDpSize * _c.nRoutedExperts * dtypeSize * 2;
        }
        // MoE routed experts
        uint32_t intermediateSize = _c.moeIntermediateSize / _c.moeTPSize;
        routedExpertsBufSize = _c.maxM * _c.defDpSize * _c.nActExperts * intermediateSize * 3 * dtypeSize;
        // MoE share experts
        if (_c.nSharedExperts != 0) {
            shareExpertsBufSize += _c.maxM * _c.hiddenSize * dtypeSize;
            shareExpertsBufSize += _c.maxM * moeSEUpGate[0].shape[0] * dtypeSize;
            shareExpertsBufSize += _c.maxM * moeSEDown[0].shape[1] * dtypeSize;
        }
        if (_c.defDpSize > 1) {
            shareExpertsBufSize += _c.maxM * _c.defDpSize * _c.hiddenSize * dtypeSize;
        }
        moeBufSize += std::max({scoresBufSize, moeDispatchDpBufSize, routedExpertsBufSize, shareExpertsBufSize});
    }
    ffnSize = std::max(mlpBufSize, moeBufSize);

    size += std::max(attnSize, ffnSize);

    return (size >> MB_BIT) + 128;
}
