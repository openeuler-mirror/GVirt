/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_MODEL_H_
#define _XLITE_MODEL_H_

#include "base.h"

enum XModelAttnType {
    XMODEL_ATTN_MHA,
    XMODEL_ATTN_MLA,
    XMODEL_ATTN_MAX_TYPE,
};

enum XModelRopeType {
    XMODEL_ROPE_NEOX,
    XMODEL_ROPE_GPTJ,
    XMODEL_ROPE_MAX_TYPE,
};

enum XModelScoringFuncType {
    XMODEL_SCORING_FUNC_SOFTMAX,
    XMODEL_SCORING_FUNC_SIGMOID,
    XMODEL_SCORING_FUNC_MAX_TYPE,
};

struct XModelConfig {
    // global config
    uint32_t vocabSize;
    uint32_t hiddenSize;
    uint32_t nLayers;

    // attention config
    enum XModelAttnType attnType = XMODEL_ATTN_MHA;
    enum XModelRopeType ropeType = XMODEL_ROPE_NEOX;
    bool addBias = false;
    bool qkNorm = false;
    uint32_t nHeads;
    uint32_t nKvHeads;
    uint32_t headDim;
    uint32_t ropeHeadDim;
    uint32_t nopeHeadDim;
    uint32_t vHeadDim;
    uint32_t qLoraRank;
    uint32_t kvLoraRank;
    uint32_t maxM;
    uint32_t blockSize;
    uint32_t maxBatch;
    uint32_t maxSeqLen;
    float normEps;
    float ropeTheta;
    float softmaxScale;

    // mlp
    uint32_t nDenseLayers;
    uint32_t nRoutedExperts;
    uint32_t nSharedExperts;
    uint32_t nExpertGroups;
    uint32_t nLimitedGroups;
    uint32_t nActExperts;
    uint32_t intermediateSize;
    uint32_t moeIntermediateSize;
    enum XModelScoringFuncType scoringFunc = XMODEL_SCORING_FUNC_SOFTMAX;
    float routeScale;
    bool normTopKProb;
    bool expertsWeightTrans = false;

    // parallel config
    uint32_t defTpSize;
    uint32_t defDpSize;
    uint32_t moeEpSize;
    uint32_t moeTPSize;

    bool weightNZ = false;
};

struct XModelAttnMeta {
    int version = 0;

    std::vector<uint32_t> lens;
    std::vector<uint32_t> cachedLens;
    std::vector<bool> isPrefills;

    /* only for version 0 */
    std::vector<std::vector<uint32_t>> blockTables;

    /* only for version 1 */
    XTensor vllmBlockTables;
    XTensor vllmSlotMapping;
    XTensor vllmPosition;
};

#define TILESIZE_OF_QUERY 128 // the tile size of query
#define AIC_MAX_NUM 25

class XModel {
public:
    XModel(struct XModelConfig &c, uint32_t rankId);
    void Init(void);
    ~XModel(void);
    void Forward(XRuntime &rt, XTensor &input,
                 XModelAttnMeta& attnMeta,
                 std::vector<std::pair<XTensor, XTensor>>& kvCache,
                 XTensor &freqsCis, XTensor &output);
    void ComputeLogits(XRuntime &rt, XTensor &input,
                       XTensor &output);
    void ForwardAndGetLogits(XRuntime &rt, XTensor &input,
                             XModelAttnMeta& attnMeta,
                             std::vector<std::pair<XTensor, XTensor>>& kvCache,
                             XTensor &freqsCis, XTensor &output);
    void ForwardWithInputsEmbeds(XRuntime &rt, XTensor &input,
                                 XModelAttnMeta& attnMeta,
                                 std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                 XTensor &freqsCis, XTensor &output);
    size_t GetTensorPoolSize(void);

    // weights
    XTensor embed;
    XTensor norm;
    XTensor head;

    std::vector<XTensor> attnNorm;
    std::vector<XTensor> attnOut;
    std::vector<XTensor> mhaQKV;
    std::vector<XTensor> mhaQKVBias;
    std::vector<XTensor> mhaQNorm;
    std::vector<XTensor> mhaKNorm;
    std::vector<XTensor> mlaQA;
    std::vector<XTensor> mlaQB;
    std::vector<XTensor> mlaQNorm;
    std::vector<XTensor> mlaKVA;
    std::vector<XTensor> mlaKVB;
    std::vector<XTensor> mlaKVNorm;

    std::vector<XTensor> mlpNorm;
    std::vector<XTensor> mlpUpGate;
    std::vector<XTensor> mlpDown;

    std::vector<XTensor> moeGate;
    std::vector<XTensor> moeGateBias;
    std::vector<XTensor> moeSEUpGate;
    std::vector<XTensor> moeSEDown;
    std::vector<std::vector<XTensor>> moeREUpGate;
    std::vector<std::vector<XTensor>> moeREUpGateScale;
    std::vector<std::vector<XTensor>> moeREDown;
    std::vector<std::vector<XTensor>> moeREDownScale;

private:
    void ForwardParallelEmbed(XRuntime &rt, XTensor &input, XTensor &embed, XTensor &output);
    void PrepareAttn(XRuntime &rt, XModelAttnMeta& attnMeta);
    std::tuple<XTensor &, XTensor &, XTensor &> ForwardAttnMLACommon(XRuntime &rt, uint32_t layer,
                                                                     std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                                                     XTensor &freqsCis, XTensor &hiddenState);
    XTensor& ForwardAttnMLAPrefill(XRuntime &rt, uint32_t layer,
                                   std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                   XTensor &freqsCis, XTensor &hiddenState,
                                   XTensor &attnQWithQr, XTensor &attnKPe, XTensor &attnQPe);
    XTensor& ForwardAttnMLADecode(XRuntime &rt, uint32_t layer,
                                  std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                  XTensor &freqsCis, XTensor &hiddenState,
                                  XTensor &attnQWithQr, XTensor &attnKPe, XTensor &attnQPe);
    void ForwardAttnMLA(XRuntime &rt, uint32_t layer,
                        std::vector<std::pair<XTensor, XTensor>>& kvCache,
                        XTensor &freqsCis, XTensor &hiddenState);
    void XliteOpQKNorm(XRuntime &rt, uint32_t layer, XTensor &qkv);
    void XliteOpAttention(XRuntime &rt, uint32_t layer, XTensor &kCache, XTensor &vCache,
                          XTensor &input, XTensor &output);
    void ForwardAttnMHA(XRuntime &rt, uint32_t layer,
                        std::vector<std::pair<XTensor, XTensor>>& kvCache,
                        XTensor &freqsCis, XTensor &hiddenState);
    void ForwardAttn(XRuntime &rt, uint32_t layer,
                     std::vector<std::pair<XTensor, XTensor>>& kvCache,
                     XTensor &freqsCis, XTensor &hiddenState);
    void ForwardMLP(XRuntime &rt, XTensor &upGate, XTensor &down, XTensor &hiddenState, bool withAllReduce);
    std::tuple<XTensor &, XTensor &> ForwardMoEGate(XRuntime &rt, uint32_t layer, XTensor &input);
    std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &> ForwardMoEDispatch(XRuntime &rt,
                                                                                         XTensor &tokenSorted,
                                                                                         XTensor &weights,
                                                                                         XTensor &routing);
    void ForwardMOECombine(XRuntime &rt, XTensor &tokenSorted, XTensor &weights, XTensor &routing, XTensor &unpIdx,
                           XTensor &expertsSorted, XTensor &expertsCounts);
    void ForwardMoE(XRuntime &rt, uint32_t layer, XTensor &hiddenState);
    void ForwardFFN(XRuntime &rt, uint32_t layer, XTensor &hiddenState);
    void ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &output);
    void ForwardLayers(XRuntime &rt, XTensor &input,
                       std::vector<std::pair<XTensor, XTensor>>& kvCache,
                       XTensor &freqsCis, XTensor &output);
    void ForwardLayersWithInputsEmbeds(XRuntime &rt, XTensor &x,
                                       std::vector<std::pair<XTensor, XTensor>>& kvCache,
                                       XTensor &freqsCis, XTensor &h);
    struct XModelConfig _c;
    uint32_t _rankId;

    // FFN
    XTensor _gateIndicts;
    std::vector<XTensor> _moeREUpGate;
    std::vector<XTensor> _moeREUpGateScale;
    std::vector<XTensor> _moeREDown;
    std::vector<XTensor> _moeREDownScale;

    // ATTN
    uint32_t _realM;
    uint32_t _maxNumBlocks;
    int _prefillBatch;
    int _decodeBatch;
    int _prefillLen;
    int _prefillLenPad;
    XTensor _attnPosition;
    XTensor _attnBlockTables;
    XTensor _attnSlotMapping;
    XTensor _position;
    XTensor _blockTables;
    XTensor _slotMapping;
    XTensor _prefillIdx;
    XTensor _prefillLastIdx;
    XTensor _decodeIdx;
    XTensor _cachedLens;
    XTensor _lens;
    XTensor _cumPromptLens;
    XTensor _vGather;
    XTensor _a2v;
    XTensor _v2a;
};

#endif
