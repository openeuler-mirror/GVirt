/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include "base.h"
#include "core_assigner.h"
#include "op.h"
#include "runtime.h"
#include "model.h"

namespace py = pybind11;

struct CModelAttnMeta {
    std::vector<uint32_t> lens;
    std::vector<uint32_t> cachedLens;
    std::vector<bool> isPrefills;
    std::vector<std::vector<uint32_t>> blockTablesList;
    at::Tensor positions;
};

class _CModel
{
public:
    _CModel() {};
    ~_CModel();
    void Init(struct XModelConfig &c, uint32_t rankId);
    void Forward(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                 std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache, at::Tensor &freqsCis,
                 at::Tensor &output, uint64_t currStream);
    void ForwardV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                   std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache, at::Tensor &freqsCis,
                   at::Tensor &output, uint64_t currStream);
    void ComputeLogits(XRuntime &rt, at::Tensor &input, at::Tensor &output, uint64_t currStream);
    void ForwardAndGetLogits(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                             std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache,
                             at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream);
    void ForwardAndGetLogitsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                               std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache,
                               at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream);
    void ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache,
                                 at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                 std::vector<at::Tensor> &deepstackInput);
    void ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                   std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache,
                                   at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                   std::vector<at::Tensor> &deepstackInput);
    size_t GetTensorPoolSize(int dbg);

    // weights
    at::Tensor embed;
    at::Tensor norm;
    at::Tensor head;

    std::vector<at::Tensor> attnNorm;
    std::vector<at::Tensor> attnOut;
    std::vector<at::Tensor> mhaQKV;
    std::vector<at::Tensor> mhaQKVBias;
    std::vector<at::Tensor> mhaQNorm;
    std::vector<at::Tensor> mhaKNorm;
    std::vector<at::Tensor> mlaQA;
    std::vector<at::Tensor> mlaQB;
    std::vector<at::Tensor> mlaQNorm;
    std::vector<at::Tensor> mlaKVA;
    std::vector<at::Tensor> mlaKVB;
    std::vector<at::Tensor> mlaKVNorm;

    std::vector<at::Tensor> mlpNorm;
    std::vector<at::Tensor> mlpUpGate;
    std::vector<at::Tensor> mlpDown;

    std::vector<at::Tensor> moeGate;
    std::vector<at::Tensor> moeGateBias;
    std::vector<at::Tensor> moeSEUpGate;
    std::vector<at::Tensor> moeSEDown;
    std::vector<at::Tensor> moeREUpGate;
    std::vector<at::Tensor> moeREUpGateScale;
    std::vector<at::Tensor> moeREDown;
    std::vector<at::Tensor> moeREDownScale;

private:
    XModel *_model = nullptr;
    std::vector<std::pair<XTensor, XTensor>> _kv;
    std::vector<XTensor> _deepstackInputEmbeds;
};

static inline enum XDtype XDtype(at::Tensor &t)
{
    switch (t.scalar_type()) {
        case at::ScalarType::Char:
            return INT8;
        case at::ScalarType::Int:
            return INT32;
        case at::ScalarType::Long:
            return INT64;
        case at::ScalarType::Half:
            return FP16;
        case at::ScalarType::BFloat16:
            return BF16;
        case at::ScalarType::Float:
            return FP32;
        case at::ScalarType::ComplexFloat:
            return CPLXF;
        default:
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": unknown data type " +
                                     std::to_string(static_cast<int>(t.scalar_type())));
    }
}

static inline void *TensorPtr(at::Tensor &t)
{
    return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(t.storage().data_ptr().get()) +
                                    t.storage_offset() * t.dtype().itemsize());
}

static inline void InitXTensor(XTensor &out, at::Tensor &in)
{
    auto sizesVec = in.sizes().vec();
    std::vector<size_t> sizes;
    sizes.reserve(sizesVec.size());

    for (auto s : sizesVec) {
        if (s < 0) {
            throw std::runtime_error("Negative size detected: " + std::to_string(s));
        }
        sizes.push_back(static_cast<size_t>(s));
    }

    out.Init(sizes, XDtype(in), TensorPtr(in));
}

void _CModel::Init(struct XModelConfig &c, uint32_t rankId)
{
    uint32_t idx = 0, moe_idx = 0;
    uint32_t nLocalRoutedExperts = c.nRoutedExperts / c.moeEpSize;
    uint32_t expertsStartIdx = c.moeEpSize == 1 ? 0 : rankId / c.moeTPSize * nLocalRoutedExperts;
    uint32_t expertsEndIdx = expertsStartIdx + nLocalRoutedExperts;
    uint32_t nRE = (c.nLayers - c.nDenseLayers) * nLocalRoutedExperts;

    if (c.nRoutedExperts % c.moeEpSize != 0) {
        std::cerr << __FILE__ << ":" << __LINE__
                  << ": num of routed experts per expert parallel group: " << nLocalRoutedExperts
                  << std::endl;
        throw std::invalid_argument(
            "num of routed experts must be divisible by moe expert parallel size");
    }

    if (c.nLayers < c.nDenseLayers) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << c.nLayers
                  << ", num of dense layers: " << c.nDenseLayers << std::endl;
        throw std::invalid_argument(
            "num of layers must be greater than or equal to num of dense layers");
    }

    if (attnNorm.size() != c.nLayers || attnOut.size() != c.nLayers ||
        mlpNorm.size() != c.nLayers) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << attnNorm.size()
                  << std::endl;
        throw std::invalid_argument(
            "Mismatched number of layers attention norm or attention out or mlp norm parameters");
    }

    if (c.attnType == XMODEL_ATTN_MLA) {
        if (mlaQA.size() != c.nLayers || mlaQB.size() != c.nLayers ||
            mlaQNorm.size() != c.nLayers || mlaKVA.size() != c.nLayers ||
            mlaKVB.size() != c.nLayers || mlaKVNorm.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mlaQA.size()
                      << std::endl;
            throw std::invalid_argument("Mismatched number of layers MLA attention QA/QB/QA "
                                        "norm/KVA/KVB/KV norm parameters");
        }
    } else if (c.attnType == XMODEL_ATTN_MHA) {
        if (mhaQKV.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mhaQKV.size()
                      << std::endl;
            throw std::invalid_argument("Mismatched number of layers MHA attention QKV parameters");
        }
        if (c.addBias && mhaQKVBias.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mhaQKVBias.size()
                      << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MHA attention QKV bias parameters");
        }
        if (c.qkNorm && (mhaQNorm.size() != c.nLayers || mhaKNorm.size() != c.nLayers)) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mhaQNorm.size()
                      << ", " << mhaKNorm.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MHA attention Q/K norm parameters");
        }
    }

    if (mlpUpGate.size() != c.nDenseLayers || mlpDown.size() != c.nDenseLayers) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": num of dense layers: " << mlpUpGate.size()
                  << std::endl;
        throw std::invalid_argument("Mismatched number of dense layers up gate or down parameters");
    }

    if (moeREUpGate.size() != nRE || moeREDown.size() != nRE) {
        std::cerr << __FILE__ << ":" << __LINE__
                  << ": num of routed experts: " << moeREUpGate.size() << std::endl;
        throw std::invalid_argument(
            "Mismatched number of routed experts up gate or down parameters");
    }

    if (moeGate.size() != (c.nLayers - c.nDenseLayers)) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": num of moe layers: " << moeGate.size()
                  << std::endl;
        throw std::invalid_argument("Mismatched number of moe layers gate parameters");
    }

    if (c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID &&
        moeGateBias.size() != (c.nLayers - c.nDenseLayers)) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": num of moe layers: " << moeGateBias.size()
                  << std::endl;
        throw std::invalid_argument("Mismatched number of moe layers gate bias parameters");
    }

    if (c.nSharedExperts != 0 && (moeSEUpGate.size() != (c.nLayers - c.nDenseLayers) ||
                                  moeSEDown.size() != (c.nLayers - c.nDenseLayers))) {
        std::cerr << __FILE__ << ":" << __LINE__
                  << ": num of moe layers with shared experts: " << moeSEUpGate.size() << std::endl;
        throw std::invalid_argument(
            "Mismatched number of moe layers with shared experts parameters");
    }

    _model = new XModel(c, rankId);

    InitXTensor(_model->embed, embed);
    InitXTensor(_model->norm, norm);
    InitXTensor(_model->head, head);

    for (uint32_t i = 0; i < c.nLayers; i++) {
        InitXTensor(_model->attnNorm[i], attnNorm[i]);
        InitXTensor(_model->attnOut[i], attnOut[i]);
        InitXTensor(_model->mlpNorm[i], mlpNorm[i]);
        if (c.attnType == XMODEL_ATTN_MLA) {
            InitXTensor(_model->mlaQA[i], mlaQA[i]);
            InitXTensor(_model->mlaQB[i], mlaQB[i]);
            InitXTensor(_model->mlaQNorm[i], mlaQNorm[i]);
            InitXTensor(_model->mlaKVA[i], mlaKVA[i]);
            InitXTensor(_model->mlaKVB[i], mlaKVB[i]);
            InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i]);
        } else if (c.attnType == XMODEL_ATTN_MHA) {
            InitXTensor(_model->mhaQKV[i], mhaQKV[i]);
            if (c.addBias) {
                InitXTensor(_model->mhaQKVBias[i], mhaQKVBias[i]);
            }
            if (c.qkNorm) {
                InitXTensor(_model->mhaQNorm[i], mhaQNorm[i]);
                InitXTensor(_model->mhaKNorm[i], mhaKNorm[i]);
            }
        }
    }

    for (uint32_t i = 0; i < c.nDenseLayers; i++) {
        InitXTensor(_model->mlpUpGate[i], mlpUpGate[i]);
        InitXTensor(_model->mlpDown[i], mlpDown[i]);
    }

    for (uint32_t i = c.nDenseLayers; i < c.nLayers; i++) {
        InitXTensor(_model->moeGate[i], moeGate[moe_idx]);
        if (c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
            InitXTensor(_model->moeGateBias[i], moeGateBias[moe_idx]);
        }
        if (c.nSharedExperts != 0) {
            InitXTensor(_model->moeSEUpGate[i], moeSEUpGate[moe_idx]);
            InitXTensor(_model->moeSEDown[i], moeSEDown[moe_idx]);
        }

        for (uint32_t j = expertsStartIdx; j < expertsEndIdx; j++) {
            InitXTensor(_model->moeREUpGate[i][j], moeREUpGate[idx]);
            if (moeREUpGate[idx].scalar_type() == at::ScalarType::Char) {
                InitXTensor(_model->moeREUpGateScale[i][j], moeREUpGateScale[idx]);
            }
            InitXTensor(_model->moeREDown[i][j], moeREDown[idx]);
            if (moeREDown[idx].scalar_type() == at::ScalarType::Char) {
                InitXTensor(_model->moeREDownScale[i][j], moeREDownScale[idx]);
            }
            idx++;
        }
        moe_idx++;
    }

    _model->Init();

    if (rankId == 0) {
        std::cout << "Euler Xlite Model Inited! [tensor paralled(" << c.defTpSize
                  << "), data parallel(" << c.defDpSize << "), expert parallel (" << c.moeEpSize
                  << ")]" << std::endl;
    }

    _kv.resize(c.nLayers);
    _deepstackInputEmbeds.resize(c.deepstackNumLevel);
}

_CModel::~_CModel(void)
{
    if (_model != nullptr) {
        delete _model;
        _model = nullptr;
    }
}

void _CModel::Forward(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                      std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache, at::Tensor &freqsCis,
                      at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _output, _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);
    InitXTensor(_freqsCis, freqsCis);

    if (kvCache.size() != _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        InitXTensor(_kv[i].first, kvCache[i].first);
        InitXTensor(_kv[i].second, kvCache[i].second);
    }

    if (currStream != 0 && rt.taskId == 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    if (rt.multiTaskParallel && rt.taskId == 1) {
        rt.NotifyWaitPeerStream();
    }

    _model->Forward(rt, _input, attnMeta, _kv, _deepstackInputEmbeds, _freqsCis, _output);

    if (rt.multiTaskParallel) {
        if (rt.taskId == 0) {
            rt.NotifyRecordPeerStream();
            rt.NotifyWaitPeerStream();
        }
        if (rt.taskId == 1) {
            rt.NotifyRecordPeerStream();
            return;
        }
    }

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                        std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache,
                        at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.lens = attnMeta.lens;
    _attnMeta.cachedLens = attnMeta.cachedLens;
    _attnMeta.isPrefills = attnMeta.isPrefills;
    _attnMeta.blockTables = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.vllmPosition, attnMeta.positions);
    Forward(rt, input, _attnMeta, kvCache, freqsCis, output, currStream);
}

void _CModel::ComputeLogits(XRuntime &rt, at::Tensor &input, at::Tensor &output,
                            uint64_t currStream)
{
    XTensor _input, _output;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);

    if (currStream != 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    _model->ComputeLogits(rt, _input, _output);

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardAndGetLogits(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                  std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache,
                                  at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _output, _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);
    InitXTensor(_freqsCis, freqsCis);

    if (kvCache.size() != _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        InitXTensor(_kv[i].first, kvCache[i].first);
        InitXTensor(_kv[i].second, kvCache[i].second);
    }

    if (currStream != 0 && rt.taskId == 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    if (rt.multiTaskParallel && rt.taskId == 1) {
        rt.NotifyWaitPeerStream();
    }

    _model->ForwardAndGetLogits(rt, _input, attnMeta, _kv, _deepstackInputEmbeds, _freqsCis,
                                _output);

    if (rt.multiTaskParallel) {
        if (rt.taskId == 0) {
            rt.NotifyRecordPeerStream();
            rt.NotifyWaitPeerStream();
        }
        if (rt.taskId == 1) {
            rt.NotifyRecordPeerStream();
            return;
        }
    }

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardAndGetLogitsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                    std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache,
                                    at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.lens = attnMeta.lens;
    _attnMeta.cachedLens = attnMeta.cachedLens;
    _attnMeta.isPrefills = attnMeta.isPrefills;
    _attnMeta.blockTables = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.vllmPosition, attnMeta.positions);
    ForwardAndGetLogits(rt, input, _attnMeta, kvCache, freqsCis, output, currStream);
}

void _CModel::ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                      std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache,
                                      at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                      std::vector<at::Tensor> &deepstackInput)
{
    XModelAttnMeta _attnMeta;
    XTensor _input, _output, _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);
    InitXTensor(_freqsCis, freqsCis);

    if (kvCache.size() != _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    if (deepstackInput.size() != _deepstackInputEmbeds.size()) {
        throw std::runtime_error(std::string(__func__) + ": check deepstack input failed");
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        InitXTensor(_kv[i].first, kvCache[i].first);
        InitXTensor(_kv[i].second, kvCache[i].second);
    }

    for (uint32_t i = 0; i < deepstackInput.size(); i++) {
        InitXTensor(_deepstackInputEmbeds[i], deepstackInput[i]);
    }

    if (currStream != 0 && rt.taskId == 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    if (rt.multiTaskParallel && rt.taskId == 1) {
        rt.NotifyWaitPeerStream();
    }

    _model->ForwardWithInputsEmbeds(rt, _input, attnMeta, _kv, _deepstackInputEmbeds, _freqsCis,
                                    _output);

    if (rt.multiTaskParallel) {
        if (rt.taskId == 0) {
            rt.NotifyRecordPeerStream();
            rt.NotifyWaitPeerStream();
        }
        if (rt.taskId == 1) {
            rt.NotifyRecordPeerStream();
            return;
        }
    }

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                        std::vector<std::pair<at::Tensor, at::Tensor>> &kvCache,
                                        at::Tensor &freqsCis, at::Tensor &output,
                                        uint64_t currStream,
                                        std::vector<at::Tensor> &deepstackInput)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.lens = attnMeta.lens;
    _attnMeta.cachedLens = attnMeta.cachedLens;
    _attnMeta.isPrefills = attnMeta.isPrefills;
    _attnMeta.blockTables = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.vllmPosition, attnMeta.positions);
    ForwardWithInputsEmbeds(rt, input, _attnMeta, kvCache, freqsCis, output, currStream,
                            deepstackInput);
}

size_t _CModel::GetTensorPoolSize(int dbg)
{
    return _model->GetTensorPoolSize(dbg);
}

void AllGather(XRuntime &rt, at::Tensor &out, at::Tensor &in)
{
    XTensor _in, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    XliteOpAllGather(rt, _in, _out, TP);
    rt.Synchronize();
}

void ReduceScatter(XRuntime &rt, at::Tensor &out, at::Tensor &in)
{
    XTensor _in, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    XliteOpReduceScatter(rt, _in, _out, TP);
    rt.Synchronize();
}

void AllReduce(XRuntime &rt, at::Tensor &out, at::Tensor &in)
{
    XTensor _in, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    XliteOpAllReduceSum(rt, _in, _out, TP);
    rt.Synchronize();
}

void Add(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z)
{
    XTensor _x, _y, _z;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    XliteOpAdd(rt, _x, _y, _z);
    rt.Synchronize();
}

void Print(at::Tensor &x)
{
    XTensor _x;

    InitXTensor(_x, x);
    _x.Print();
}

void Matmul(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z, bool weightNZ,
            bool transpose)
{
    XTensor _x, _y, _z, _bias;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias, transpose);
    rt.Synchronize();
}

void MatmulWithBias(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z, at::Tensor &bias,
                    bool weightNZ)
{
    XTensor _x, _y, _z, _bias;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    InitXTensor(_bias, bias);
    XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias);
    rt.Synchronize();
}

void Embed(XRuntime &rt, at::Tensor &weight, at::Tensor &in, at::Tensor &out, uint32_t start,
           uint32_t end)
{
    XTensor _in, _out, _weight;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_weight, weight);
    XliteOpEmbed(rt, _in, _weight, start, end, _out);
    rt.Synchronize();
}

void RMSNorm(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &out, float normEps,
             uint32_t normDim, uint32_t cntPerToken, uint32_t startOffset)
{
    XTensor _in, _out, _norm;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    XliteOpRmsNorm(rt, _in, _norm, _out, normEps, normDim == 0 ? _in.shape[1] : normDim,
                   cntPerToken, startOffset);
    rt.Synchronize();
}

void AddBias(XRuntime &rt, at::Tensor &in, at::Tensor &weight, at::Tensor &out)
{
    XTensor _in, _out, _weight;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_weight, weight);
    XliteOpAddBias(rt, _in, _weight, _out);
    rt.Synchronize();
}

void SiluAndMul(XRuntime &rt, at::Tensor &in, at::Tensor &out)
{
    XTensor _in, _out;
    InitXTensor(_in, in);
    InitXTensor(_out, out);

    XliteOpSiluAndMul(rt, _in, _out);
    rt.Synchronize();
}

void RopeAndCache(XRuntime &rt, at::Tensor &inout, at::Tensor &kCache, at::Tensor &vCache,
                  at::Tensor &position, at::Tensor &cossin, at::Tensor &slotMapping,
                  uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim, uint32_t rotDim,
                  uint32_t blockSize, bool isNeox, uint64_t mropeMaskH, uint64_t mropeMaskW)
{
    XTensor _inout, _kCache, _vCache, _position, _cossin, _slotMapping;

    InitXTensor(_inout, inout);
    InitXTensor(_kCache, kCache);
    InitXTensor(_vCache, vCache);
    InitXTensor(_position, position);
    InitXTensor(_cossin, cossin);
    InitXTensor(_slotMapping, slotMapping);
    XliteOpRopeCache(rt, _inout, _kCache, _vCache, _position, _cossin, _slotMapping, nHeads,
                     nKvHeads, headDim, rotDim, blockSize, isNeox, mropeMaskH, mropeMaskW);
    rt.Synchronize();
}

void Attention(XRuntime &rt, at::Tensor &qkv, at::Tensor &kCache, at::Tensor &vCache,
               at::Tensor &output, at::Tensor &cumPromptLens, at::Tensor &lens,
               at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads, uint32_t nKvHeads,
               uint32_t headDim, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)
{
    XTensor _qkv, _kCache, _vCache, _qk, _output, _cumPromptLens, _lens, _cachedLens, _blockTables;

    InitXTensor(_qkv, qkv);
    InitXTensor(_kCache, kCache);
    InitXTensor(_vCache, vCache);
    InitXTensor(_output, output);
    InitXTensor(_cumPromptLens, cumPromptLens);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);

    if (!std::getenv("FA_TEST")) {
        XTensor &qk = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, maxNumBlock * blockSize},
                                         XDtype(qkv), DBG_LOC);
        XliteOpAttention(rt, _qkv, _kCache, _vCache, qk, _output, _cumPromptLens, _lens, _cachedLens,
                        _blockTables, nHeads, nKvHeads, headDim, blockSize, batch, maxNumBlock);
        rt.Synchronize();
        rt.pool->PutTensor(qk);
    } else {
        XTensor &qk = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, TILESIZE_OF_CACHED_KV},
                                        XDtype(qkv), DBG_LOC);
        XTensor &sv = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2, headDim},
                                        XDtype(qkv), DBG_LOC);
        XTensor &max = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.pool->GetTensor({rt.aicNum * TILESIZE_OF_QUERY * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.pool->GetTensor({_qkv.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.pool->GetTensor({_qkv.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &sync = rt.pool->GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
        sync.Memset(0);
        XliteOpFlashAttention(rt, _qkv, _kCache, _vCache, qk, sv, max, sum, lastMax, lastSum, sync, _output, _cumPromptLens, _lens, _cachedLens,
                        _blockTables, nHeads, nKvHeads, headDim, blockSize, batch, maxNumBlock);
        rt.Synchronize();
        sync.Print("Flash Attention Sync", 1, rt.aivNum);
        rt.pool->PutTensor(sync);
        rt.pool->PutTensor(lastSum);
        rt.pool->PutTensor(lastMax);
        rt.pool->PutTensor(sum);
        rt.pool->PutTensor(max);
        rt.pool->PutTensor(sv);
        rt.pool->PutTensor(qk);
    }
}

void AddAndRMSNorm(XRuntime &rt, at::Tensor &in1, at::Tensor &in2, at::Tensor &norm,
                   at::Tensor &out, float normEps)
{
    XTensor _in1, _in2, _out, _norm;

    InitXTensor(_in1, in1);
    InitXTensor(_in2, in2);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    XliteOpAddAndRmsNorm(rt, _in1, _in2, _norm, normEps, _out);
    rt.Synchronize();
}

void SoftmaxTopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &outWeights,
                 at::Tensor &outRouting, uint32_t topK, bool normTopKProb)
{
    XTensor _scores, _indices, _outWeights, _outRouting;

    InitXTensor(_scores, scores);
    InitXTensor(_indices, indices);
    InitXTensor(_outWeights, outWeights);
    std::vector<size_t> sizes(scores.sizes().vec().begin(), scores.sizes().vec().end());
    _outRouting.Init(sizes, BIT1, TensorPtr(outRouting));
    XliteOpSoftmaxTopK(rt, _scores, _indices, _outWeights, _outRouting, topK, normTopKProb);
    rt.Synchronize();
}

void SigmoidTopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &bias,
                 float scale, at::Tensor &outWeights, at::Tensor &outRouting, uint32_t nGroup,
                 uint32_t nTopkGroup, uint32_t topK, bool normTopKProb)
{
    XTensor _scores, _indices, _bias, _outWeights, _outRouting;

    InitXTensor(_scores, scores);
    InitXTensor(_bias, bias);
    InitXTensor(_indices, indices);
    InitXTensor(_outWeights, outWeights);
    std::vector<size_t> sizes(scores.sizes().vec().begin(), scores.sizes().vec().end());
    _outRouting.Init(sizes, BIT1, TensorPtr(outRouting));
    XliteOpSigmoidTopK(rt, _scores, _indices, _bias, scale, _outWeights, _outRouting, nGroup,
                       nTopkGroup, topK, normTopKProb);
    rt.Synchronize();
}

void CastUp(XRuntime &rt, at::Tensor &in, at::Tensor &out)
{
    XTensor _in, _out;
    XTensor &inScale = rt.pool->GetTensor({1}, XDtype(in), DBG_LOC);

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    XliteOpCastUp(rt, _in, inScale, _out);
    rt.Synchronize();
    rt.pool->PutTensor(inScale);
}

void Permutation(XRuntime &rt, at::Tensor &in, at::Tensor &routing, uint32_t start, uint32_t end,
                 at::Tensor &out, at::Tensor &unpIdx, at::Tensor &counts)
{
    XTensor _in, _routing, _out, _unpIdx, _counts;

    InitXTensor(_in, in);
    InitXTensor(_routing, routing);
    InitXTensor(_out, out);
    InitXTensor(_unpIdx, unpIdx);
    InitXTensor(_counts, counts);
    XliteOpPermutation(rt, _in, _routing, start, end, _out, _unpIdx, _counts);
    rt.Synchronize();
}

void UnPermutation(XRuntime &rt, at::Tensor &in, at::Tensor &routing, at::Tensor &weights,
                   uint32_t start, uint32_t end, at::Tensor &out, at::Tensor &unpIdx)
{
    XTensor _in, _routing, _weights, _out, _unpIdx;
    InitXTensor(_in, in);
    InitXTensor(_routing, routing);
    InitXTensor(_weights, weights);
    InitXTensor(_out, out);
    InitXTensor(_unpIdx, unpIdx);

    XliteOpUnpermutation(rt, _in, _unpIdx, _routing, _weights, start, end, _out);
    rt.Synchronize();
}

void GroupMatmul(XRuntime &rt, at::Tensor &in, std::vector<at::Tensor> &weights,
                 std::vector<at::Tensor> &scales, at::Tensor &counts, uint32_t start, uint32_t end,
                 long outDim, long inDim, at::Tensor &output, bool weightNZ, bool transpose)
{
    XTensor _in, _counts, _output;
    std::vector<void *> p;
    uint32_t i, num = counts.size(0);

    InitXTensor(_in, in);
    InitXTensor(_counts, counts);
    InitXTensor(_output, output);
    XTensor &_weights = rt.pool->GetTensor({num}, INT64, DBG_LOC);
    XTensor &_scales = rt.pool->GetTensor({num}, INT64, DBG_LOC);

    p.resize(num);
    for (i = 0; i < num; i++) {
        p[i] = TensorPtr(weights[i]);
    }
    rt.MemcpyH2D(_weights.ptr, reinterpret_cast<void *>(p.data()), num * sizeof(void *));

    if (scales.size() == num) {
        for (i = 0; i < num; i++) {
            p[i] = TensorPtr(scales[i]);
        }
        rt.MemcpyH2D(_scales.ptr, reinterpret_cast<void *>(p.data()), num * sizeof(void *));
    }

    XliteOpGroupMatmul(rt, _in, _weights, _scales, _counts, start, end, XDtype(weights[0]), outDim,
                       inDim, _output, weightNZ, transpose);
    rt.Synchronize();
    rt.pool->PutTensor(_weights);
    rt.pool->PutTensor(_scales);
}

void Softmax(XRuntime &rt, at::Tensor &x, uint32_t calcLen, bool isLong)
{
    XTensor _x;
    InitXTensor(_x, x);
    if (isLong) {
        XTensor &expBuf = rt.pool->GetTensor({1, _x.shape[1]}, FP32, DBG_LOC);
        XliteOpSoftmaxLong(rt, calcLen, _x, expBuf);
        rt.pool->PutTensor(expBuf);
    } else {
        XliteOpSoftmax(rt, calcLen, _x);
    }
    rt.Synchronize();
}

void RopeComplex(XRuntime &rt, uint32_t numTokens, uint32_t nLocalHeads, uint32_t stepDim,
                 uint32_t ropeDim, at::Tensor &inputWithR, at::Tensor &freqs, at::Tensor &position,
                 at::Tensor &vGather, at::Tensor &outputPe, uint32_t ropeType)
{
    XTensor _inputWithR, _freqs, _position, _vGather, _outputPe;
    InitXTensor(_inputWithR, inputWithR);
    InitXTensor(_freqs, freqs);
    InitXTensor(_position, position);
    InitXTensor(_vGather, vGather);
    InitXTensor(_outputPe, outputPe);
    XliteOpRopeComplex(rt, numTokens, nLocalHeads, stepDim, ropeDim, _inputWithR, _freqs, _position,
                       _vGather, _outputPe, static_cast<enum XRopeType>(ropeType));
    rt.Synchronize();
}

PYBIND11_MODULE(_C, m)
{
    py::class_<XRuntime>(m, "Runtime")
        .def(py::init<uint32_t, size_t, uint32_t, uint32_t, uint32_t>(), py::arg("devid"),
             py::arg("size") = 0, py::arg("rank") = 0, py::arg("tp_size") = 1,
             py::arg("dp_size") = 1)
        .def_readwrite("task_id", &XRuntime::taskId)
        .def_readwrite("notify", &XRuntime::notify)
        .def_readwrite("peer_notify", &XRuntime::peerNotify)
        .def_readwrite("multi_task_parallel", &XRuntime::multiTaskParallel)
        .def("update_core_num", &XRuntime::UpdateCoreNum)
        .def("init_tensor_pool", &XRuntime::InitTensorPool)
        .def("set_current_context", &XRuntime::SetCurrentContext);

    py::class_<XModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &XModelConfig::vocabSize)
        .def_readwrite("hidden_size", &XModelConfig::hiddenSize)
        .def_readwrite("n_layers", &XModelConfig::nLayers)
        .def_readwrite("attn_type", &XModelConfig::attnType)
        .def_readwrite("n_heads", &XModelConfig::nHeads)
        .def_readwrite("n_kv_heads", &XModelConfig::nKvHeads)
        .def_readwrite("head_dim", &XModelConfig::headDim)
        .def_readwrite("nope_head_dim", &XModelConfig::nopeHeadDim)
        .def_readwrite("rope_head_dim", &XModelConfig::ropeHeadDim)
        .def_readwrite("v_head_dim", &XModelConfig::vHeadDim)
        .def_readwrite("q_lora_rank", &XModelConfig::qLoraRank)
        .def_readwrite("kv_lora_rank", &XModelConfig::kvLoraRank)
        .def_readwrite("norm_eps", &XModelConfig::normEps)
        .def_readwrite("rope_theta", &XModelConfig::ropeTheta)
        .def_readwrite("softmax_scale", &XModelConfig::softmaxScale)
        .def_readwrite("n_dense_layers", &XModelConfig::nDenseLayers)
        .def_readwrite("n_routed_experts", &XModelConfig::nRoutedExperts)
        .def_readwrite("n_shared_experts", &XModelConfig::nSharedExperts)
        .def_readwrite("n_expert_groups", &XModelConfig::nExpertGroups)
        .def_readwrite("n_limited_groups", &XModelConfig::nLimitedGroups)
        .def_readwrite("n_act_experts", &XModelConfig::nActExperts)
        .def_readwrite("intermediate_size", &XModelConfig::intermediateSize)
        .def_readwrite("moe_intermediate_size", &XModelConfig::moeIntermediateSize)
        .def_readwrite("route_scale", &XModelConfig::routeScale)
        .def_readwrite("def_tp_size", &XModelConfig::defTpSize)
        .def_readwrite("def_dp_size", &XModelConfig::defDpSize)
        .def_readwrite("moe_ep_size", &XModelConfig::moeEpSize)
        .def_readwrite("moe_tp_size", &XModelConfig::moeTPSize)
        .def_readwrite("max_seq_len", &XModelConfig::maxSeqLen)
        .def_readwrite("max_batch_size", &XModelConfig::maxBatch)
        .def_readwrite("max_m", &XModelConfig::maxM)
        .def_readwrite("block_size", &XModelConfig::blockSize)
        .def_readwrite("weight_nz", &XModelConfig::weightNZ)
        .def_readwrite("experts_weight_transpose", &XModelConfig::expertsWeightTrans)
        .def_readwrite("qkv_bias", &XModelConfig::addBias)
        .def_readwrite("qk_norm", &XModelConfig::qkNorm)
        .def_readwrite("scoring_func", &XModelConfig::scoringFunc)
        .def_readwrite("norm_topk_prob", &XModelConfig::normTopKProb)
        .def_readwrite("mrope_section", &XModelConfig::mropeSection)
        .def_readwrite("mrope_interleaved", &XModelConfig::mropeInterleaved)
        .def_readwrite("deepstack_num_level", &XModelConfig::deepstackNumLevel);

    py::class_<XModelAttnMeta>(m, "ModelAttnMeta")
        .def(py::init<>())
        .def_readwrite("lens", &XModelAttnMeta::lens)
        .def_readwrite("cached_lens", &XModelAttnMeta::cachedLens)
        .def_readwrite("is_prefills", &XModelAttnMeta::isPrefills)
        .def_readwrite("block_tables", &XModelAttnMeta::blockTables);

    py::class_<CModelAttnMeta>(m, "AttnMeta")
        .def(py::init<>())
        .def_readwrite("lens", &CModelAttnMeta::lens)
        .def_readwrite("cached_lens", &CModelAttnMeta::cachedLens)
        .def_readwrite("is_prefills", &CModelAttnMeta::isPrefills)
        .def_readwrite("block_tables_cpu", &CModelAttnMeta::blockTablesList)
        .def_readwrite("positions", &CModelAttnMeta::positions);

    py::enum_<XModelAttnType>(m, "AttnType")
        .value("AttnMHA", XModelAttnType::XMODEL_ATTN_MHA)
        .value("AttnMLA", XModelAttnType::XMODEL_ATTN_MLA)
        .export_values();

    py::enum_<XModelScoringFuncType>(m, "ScoringFuncType")
        .value("ScoringFuncSoftmax", XModelScoringFuncType::XMODEL_SCORING_FUNC_SOFTMAX)
        .value("ScoringFuncSigmoid", XModelScoringFuncType::XMODEL_SCORING_FUNC_SIGMOID)
        .export_values();

    py::class_<_CModel>(m, "Model")
        .def(py::init<>())
        .def_readwrite("embed", &_CModel::embed)
        .def_readwrite("norm", &_CModel::norm)
        .def_readwrite("head", &_CModel::head)
        .def_readwrite("attn_norm", &_CModel::attnNorm)
        .def_readwrite("attn_out", &_CModel::attnOut)
        .def_readwrite("mha_qkv", &_CModel::mhaQKV)
        .def_readwrite("mha_qkv_bias", &_CModel::mhaQKVBias)
        .def_readwrite("mha_q_norm", &_CModel::mhaQNorm)
        .def_readwrite("mha_k_norm", &_CModel::mhaKNorm)
        .def_readwrite("mla_q_a", &_CModel::mlaQA)
        .def_readwrite("mla_q_b", &_CModel::mlaQB)
        .def_readwrite("mla_q_norm", &_CModel::mlaQNorm)
        .def_readwrite("mla_kv_a", &_CModel::mlaKVA)
        .def_readwrite("mla_kv_b", &_CModel::mlaKVB)
        .def_readwrite("mla_kv_norm", &_CModel::mlaKVNorm)
        .def_readwrite("mlp_norm", &_CModel::mlpNorm)
        .def_readwrite("mlp_up_gate", &_CModel::mlpUpGate)
        .def_readwrite("mlp_down", &_CModel::mlpDown)
        .def_readwrite("gate", &_CModel::moeGate)
        .def_readwrite("gate_bias", &_CModel::moeGateBias)
        .def_readwrite("se_up_gate", &_CModel::moeSEUpGate)
        .def_readwrite("se_down", &_CModel::moeSEDown)
        .def_readwrite("re_up_gate", &_CModel::moeREUpGate)
        .def_readwrite("re_up_gate_scale", &_CModel::moeREUpGateScale)
        .def_readwrite("re_down", &_CModel::moeREDown)
        .def_readwrite("re_down_scale", &_CModel::moeREDownScale)
        .def("init", &_CModel::Init, "model init", py::arg("config"), py::arg("rank") = 0)
        .def("forward", &_CModel::Forward, "forward", py::arg("rt"), py::arg("input"),
             py::arg("attn_meta"), py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
        .def("forward", &_CModel::ForwardV1, "forward", py::arg("rt"), py::arg("input"),
             py::arg("attn_meta"), py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
        .def("compute_logits", &_CModel::ComputeLogits, "compute_logits", py::arg("rt"),
             py::arg("input"), py::arg("output"), py::arg("curr_stream") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_and_get_logits", &_CModel::ForwardAndGetLogits, "forward_and_get_logits",
             py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
             py::arg("freqs_cis"), py::arg("output"), py::arg("curr_stream") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_and_get_logits", &_CModel::ForwardAndGetLogitsV1, "forward_and_get_logits",
             py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
             py::arg("freqs_cis"), py::arg("output"), py::arg("curr_stream") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_with_inputs_embeds", &_CModel::ForwardWithInputsEmbeds,
             "forward_with_inputs_embeds", py::arg("rt"), py::arg("input"), py::arg("attn_meta"),
             py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::arg("deepstack_input") = std::vector<at::Tensor>{},
             py::call_guard<py::gil_scoped_release>())
        .def("forward_with_inputs_embeds", &_CModel::ForwardWithInputsEmbedsV1,
             "forward_with_inputs_embeds", py::arg("rt"), py::arg("input"), py::arg("attn_meta"),
             py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::arg("deepstack_input") = std::vector<at::Tensor>{},
             py::call_guard<py::gil_scoped_release>())
        .def("get_tensor_pool_size", &_CModel::GetTensorPoolSize, "get_tensor_pool_size",
             py::arg("dbg") = 0);

    py::class_<XCoreAssigner>(m, "CoreAssigner")
        .def(py::init<float>(), py::arg("prefillRatio"))
        .def("assign_core", &XCoreAssigner::AssignCore, py::call_guard<py::gil_scoped_release>())
        .def("release_core", &XCoreAssigner::ReleaseCore, py::call_guard<py::gil_scoped_release>());

    // kernels
    m.def("all_gather", &AllGather);
    m.def("reduce_scatter", &ReduceScatter);
    m.def("all_reduce", &AllReduce);
    m.def("add", &Add);
    m.def("matmul", &Matmul, "matmul", py::arg("rt"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("weight_nz") = false, py::arg("transpose") = false);
    m.def("matmul_with_bias", &MatmulWithBias, "matmul_with_bias", py::arg("rt"), py::arg("x"),
          py::arg("y"), py::arg("z"), py::arg("bias"), py::arg("weight_nz") = false);
    m.def("embed", &Embed);
    m.def("rmsnorm", &RMSNorm, "rmsnorm", py::arg("rt"), py::arg("in"), py::arg("norm"),
          py::arg("out"), py::arg("norm_eps"), py::arg("norm_dim") = 0,
          py::arg("cnt_per_token") = 1, py::arg("start_offset") = 0);
    m.def("add_bias", &AddBias);
    m.def("silu_and_mul", &SiluAndMul);
    m.def("rope_and_cache", &RopeAndCache, "rope_and_cache", py::arg("rt"), py::arg("inout"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("position"), py::arg("cosin"),
          py::arg("slot_mapping"), py::arg("n_heads"), py::arg("n_kv_heads"), py::arg("head_dim"),
          py::arg("rot_dim"), py::arg("block_size"), py::arg("is_neox"),
          py::arg("mrope_mask_h") = 0, py::arg("mrope_mask_w") = 0);
    m.def("attention", &Attention);
    m.def("add_and_rmsnorm", &AddAndRMSNorm);
    m.def("softmax_topk", &SoftmaxTopK);
    m.def("sigmoid_topk", &SigmoidTopK);
    m.def("cast_up", &CastUp);
    m.def("permutation", &Permutation);
    m.def("unpermutation", &UnPermutation);
    m.def("group_matmul", &GroupMatmul);
    m.def("softmax", &Softmax);
    m.def("rope_complex", &RopeComplex, "rope_complex", py::arg("rt"), py::arg("num_tokens"),
          py::arg("n_local_heads"), py::arg("step_dim"), py::arg("rope_dim"),
          py::arg("input_with_r"), py::arg("freqs"), py::arg("position"), py::arg("v_gather"),
          py::arg("output_pe"), py::arg("rope_type"));

    // funcs
    m.def("print", &Print);
}
