/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_MODEL_H_
#define _XLITE_MODEL_H_

#include "runtime.h"
#include "base.h"

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
    uint32_t nLayers = 0;

    // attention config
    enum XModelAttnType attnType = XMODEL_ATTN_MHA;
    enum XModelRopeType ropeType = XMODEL_ROPE_NEOX;
    bool addBias = false;
    bool qkNorm = false;
    bool qkNormFull = false;
    // Qwen3.5 full-attention output gate: attn *= sigmoid(gate).
    // When true, fused mhaQKV layout is [Q | K | V | Gate].
    bool attnOutputGate = false;
    uint32_t nHeads = 0;
    uint32_t nKvHeads = 1;
    uint32_t headDim;
    uint32_t ropeHeadDim;
    uint32_t nopeHeadDim;
    uint32_t vHeadDim;
    uint32_t qLoraRank;
    uint32_t kvLoraRank;
    uint32_t blockSize;
    uint32_t deepstackNumLevel = 0;
    uint64_t maxBatchedTokens;
    uint64_t maxBatch;
    uint64_t maxSeqLen;
    bool quantAttnWeightTrans = false;
    bool quantAttnWeightNz = false;
    float normEps;
    float ropeTheta;
    float softmaxScale;
    std::vector<uint32_t> mropeSection;
    bool mropeInterleaved = false;
    uint32_t indexHeadDim;
    uint32_t indexNHeads;
    uint32_t indexTopK;
    float indexSoftmaxScale;
    bool indexRopeInterleaved = false;

    // mlp
    uint32_t nDenseLayers = 0;
    uint32_t nRoutedExperts = 0;
    uint32_t nSharedExperts = 0;
    uint32_t nExpertGroups = 1;
    uint32_t nLimitedGroups = 1;
    uint32_t nActExperts = 0;
    uint32_t intermediateSize;
    uint32_t moeIntermediateSize;
    enum XModelScoringFuncType scoringFunc = XMODEL_SCORING_FUNC_SOFTMAX;
    float routeScale = 1.0f;
    bool normTopKProb;
    bool expertsWeightTrans = false;
    bool expertsWeightNZ = false;
    // For GLM4/GLM5, vllm-ascend doesn't capture the Gate layer, so its gate won't use NZ format
    bool gateCaptured = true;

    // parallel config
    uint32_t defTpSize;
    uint32_t defDpSize;
    uint32_t moeEpSize;
    uint32_t moeTPSize;

    bool weightNZ = false;
};

struct MoEAlltoAllMeta {
    std::vector<int64_t> sendCountsData;
    std::vector<int64_t> recvCountsData;
    std::vector<int64_t> sdisplsData;
    std::vector<int64_t> rdisplsData;
    XTensor sendCounts;
    XTensor recvCounts;
    XTensor sdispls;
    XTensor rdispls;
    uint64_t totalRecvElements = 0;
    // per-source per-expert counts for reorder, pointer to device tensor (not value copy)
    XTensor *expertsCountsAllEpDevice = nullptr;
    uint32_t nRoutedExperts = 0;
};

#define AIC_MAX_NUM 25
#define AIV_MAX_NUM 50

class XModel
{
public:
    XModel(struct XModelConfig &c, uint32_t rankId);
    void Init(void);
    ~XModel(void);
    void Forward(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                 std::vector<std::vector<XTensor>> &kvCache,
                 std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis, XTensor &output);
    void ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &indices, XTensor &output);
    void ForwardAndGetLogits(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                             std::vector<std::vector<XTensor>> &kvCache,
                             std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                             XTensor &indices, XTensor &output);
    void ForwardWithInputsEmbeds(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::vector<XTensor>> &kvCache,
                                 std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                 XTensor &output);
    size_t GetTensorPoolSize(int dbg);

    // weights
    XTensor embed;
    XTensor norm;
    XTensor normBias;
    XTensor head;

    std::vector<XTensor> attnNorm;
    std::vector<XTensor> attnNormBias;
    std::vector<MatmulWeight> attnOut;
    std::vector<MatmulWeight> mhaQKV;
    std::vector<XTensor> mhaQKVBias;
    std::vector<XTensor> mhaQNorm;
    std::vector<XTensor> mhaQNormBias;
    std::vector<XTensor> mhaKNorm;
    std::vector<XTensor> mhaKNormBias;

    std::vector<MatmulWeight> mlaQKVA;
    std::vector<MatmulWeight> mlaQB;
    std::vector<XTensor> mlaQNorm;
    std::vector<XTensor> mlaQNormBias;
    std::vector<XTensor> mlaKVB;
    std::vector<XTensor> mlaKVNorm;
    std::vector<XTensor> mlaKVNormBias;

    std::vector<MatmulWeight> indexQB;
    std::vector<XTensor> indexKWeightsProj;
    std::vector<XTensor> indexKNorm;
    std::vector<XTensor> indexKNormBias;

    std::vector<XTensor> mlpNorm;
    std::vector<XTensor> mlpNormBias;
    std::vector<MatmulWeight> mlpUpGate;
    std::vector<MatmulWeight> mlpDown;

    std::vector<XTensor> moeGate;
    std::vector<XTensor> moeGateBias;
    std::vector<MatmulWeight> moeSEUpGate;
    std::vector<MatmulWeight> moeSEDown;
    std::vector<std::vector<XTensor>> moeREUpGate;
    std::vector<std::vector<XTensor>> moeREUpGateDeqScale;
    std::vector<std::vector<XTensor>> moeREDown;
    std::vector<std::vector<XTensor>> moeREDownDeqScale;

private:
    void ForwardParallelEmbed(XRuntime &rt, XTensor &input, XTensor &embed, XTensor &output);
    std::tuple<XTensor &, XTensor &> ForwardAttnMLACommon(
        XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
        XTensor &hiddenState);
    XTensor *ForwardAttnIndexer(XRuntime &rt, uint32_t layer, XTensor &hiddenState,
                                XTensor &attnNormQc, XTensor &indexKCache, XTensor &freqsCis);
    void ForwardAttnMLA(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                        XTensor &freqsCis, XTensor &hiddenState);
    void XliteOpQKNorm(XRuntime &rt, uint32_t layer, XTensor &qkv);
    void ForwardAttnMHA(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                        XTensor &freqsCis, XTensor &hiddenState);
    void ForwardAttn(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                     XTensor &freqsCis, XTensor &hiddenState);
    void ForwardMLP(XRuntime &rt, uint32_t layer, XTensor &hiddenState,
                    std::vector<MatmulWeight> &upGate, std::vector<MatmulWeight> &down,
                    bool withAllReduce);
    std::tuple<XTensor &, XTensor &> ForwardMoEGate(XRuntime &rt, uint32_t layer, XTensor &input);
    std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, MoEAlltoAllMeta>
        ForwardMoEDispatch(XRuntime &rt, XTensor &tokenSorted, XTensor &weights, XTensor &routing);
    void ForwardMOECombine(XRuntime &rt, XTensor &tokenSorted, XTensor &weights, XTensor &routing,
                           XTensor &unpIdx, XTensor &expertsSorted, XTensor &expertsCounts);
    MoEAlltoAllMeta MoeComputeAlltoAllVMeta(const int32_t *tokensPerEpGroupAllEpHost,
                                            uint32_t moeEpSize, uint32_t moeTpSize,
                                            uint32_t hiddenSize, uint32_t rankId,
                                            uint32_t nRoutedExperts);
    MoEAlltoAllMeta MoeComputeReverseAlltoAllVMeta(const MoEAlltoAllMeta &meta, uint32_t moeEpSize);
    std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, MoEAlltoAllMeta>
        ForwardMoEDispatchAllToAll(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                                   XTensor &routing);
    void ForwardMoECombineAllToAll(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                                   XTensor &routing, XTensor &unpIdx, XTensor &expertsSorted,
                                   XTensor &expertsCounts, const MoEAlltoAllMeta &meta);
    void ForwardMoE(XRuntime &rt, uint32_t layer, XTensor &hiddenState);
    void ForwardFFN(XRuntime &rt, uint32_t layer, XTensor &hiddenState);
    void ForwardEmbedAndLayers(XRuntime &rt, XTensor &input,
                               std::vector<std::vector<XTensor>> &kvCache,
                               std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                               XTensor &h);
    void ForwardLayers(XRuntime &rt, XTensor &x, std::vector<std::vector<XTensor>> &kvCache,
                       std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis, XTensor &h);
    void ForwardLayersNaive(XRuntime &rt, XTensor &x, std::vector<std::vector<XTensor>> &kvCache,
                            std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                            XTensor &output);
    void ForwardLayersCommOptimize(XRuntime &rt, XTensor &x,
                                   std::vector<std::vector<XTensor>> &kvCache,
                                   std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                   XTensor &output);
    void CheckForwardParam(XRuntime &rt, std::vector<std::vector<XTensor>> &kvCache);
    void ForwardLinear(XRuntime &rt, uint32_t layer, XTensor &x, std::vector<MatmulWeight> &weights,
                       XTensor &out, const std::vector<XTensor> &weightBias = {});

    struct XModelConfig _c;
    uint32_t _rankId;

    // FFN
    XTensor _gateIndices;
    std::vector<XTensor> _moeREUpGate;
    std::vector<XTensor> _moeREUpGateDeqScale;
    std::vector<XTensor> _moeREDown;
    std::vector<XTensor> _moeREDownDeqScale;
    bool _isSharedExpertWeightFull = false;

    // ATTN
    uint64_t _mropeMaskH;
    uint64_t _mropeMaskW;
    XTensor _sync;
    XTensor _dsaTopkIndices;
    float _dsaIndexerScale = 1.0f;
};

#endif
