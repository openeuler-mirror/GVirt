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
    at::Tensor blockTables;
    at::Tensor slotMapping;
    at::Tensor positions;
};

class _CModel {
public:
    _CModel() {};
    ~_CModel();
    void Init(struct XModelConfig &c, uint32_t rankId);
    void Forward(XRuntime &rt, at::Tensor &input,
                 XModelAttnMeta& attnMeta,
                 std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                 at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream);
    void ForwardV1(XRuntime &rt, at::Tensor &input,
                 CModelAttnMeta& attnMeta,
                 std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                 at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream);
    void ComputeLogits(XRuntime &rt, at::Tensor &input, at::Tensor &output, uint64_t currStream);
    void ForwardAndGetLogits(XRuntime &rt, at::Tensor &input,
                             XModelAttnMeta& attnMeta,
                             std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                             at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream);
    void ForwardAndGetLogitsV1(XRuntime &rt, at::Tensor &input,
                             CModelAttnMeta& attnMeta,
                             std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                             at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream);
    void ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input,
                                 XModelAttnMeta& attnMeta,
                                 std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                                 at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream);
    void ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input,
                                   CModelAttnMeta& attnMeta,
                                   std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                                   at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream);
    size_t GetTensorPoolSize(void);

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
    XModel *_model;
    std::vector<std::pair<XTensor, XTensor>> _kv;
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
            std::cerr << __FILE__ << ":" << __LINE__ << ": unknown data type " << t.scalar_type() << std::endl;
            return MAX_XDTYPE;
    }
}

static inline void *TensorPtr(at::Tensor &t)
{
    return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(t.storage().data_ptr().get()) +
        t.storage_offset() * t.dtype().itemsize());
}

static inline void InitXTensor(XTensor &out, at::Tensor &in)
{
    out.Init(in.sizes().vec(), XDtype(in), TensorPtr(in));
}

void _CModel::Init(struct XModelConfig &c, uint32_t rankId)
{
    uint32_t idx = 0, moe_idx = 0;
    uint32_t nLocalRoutedExperts = c.nRoutedExperts / c.moeEpSize;
    uint32_t expertsStartIdx = c.moeEpSize == 1 ? 0 : rankId * nLocalRoutedExperts;
    uint32_t expertsEndIdx = expertsStartIdx + nLocalRoutedExperts;
    uint32_t nRE = (c.nLayers - c.nDenseLayers) * nLocalRoutedExperts;

    if (moeREUpGate.size() != nRE || moeREDown.size() != nRE) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": num of routed experts: " << moeREUpGate.size() << std::endl;
        return;
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
        InitXTensor(_model->moeGateBias[i], moeGateBias[moe_idx]);
        InitXTensor(_model->moeSEUpGate[i], moeSEUpGate[moe_idx]);
        InitXTensor(_model->moeSEDown[i], moeSEDown[moe_idx]);
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
        std::cout << "Euler Xlite Model Inited! [tensor paralled(" << c.defTpSize <<
            "), data parallel(" << c.defDpSize << "), expert parallel (" << c.moeEpSize << ")]" << std::endl;
    }

    _kv.resize(c.nLayers);
}

_CModel::~_CModel(void)
{
    delete _model;
}

void _CModel::Forward(XRuntime &rt, at::Tensor &input,
                      XModelAttnMeta& attnMeta,
                      std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                      at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _output, _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);
    InitXTensor(_freqsCis, freqsCis);

    if (kvCache.size() != _kv.size()) {
        std::cerr << __func__ << ": check kv cache failed!" << std::endl;
        return;
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        _kv[i].first.Init(kvCache[i].first.sizes().vec(), XDtype(kvCache[i].first), TensorPtr(kvCache[i].first));
        _kv[i].second.Init(kvCache[i].second.sizes().vec(), XDtype(kvCache[i].second), TensorPtr(kvCache[i].second));
    }

    if (currStream != 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    _model->Forward(rt, _input, attnMeta, _kv, _freqsCis, _output);

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardV1(XRuntime &rt, at::Tensor &input,
                        CModelAttnMeta& attnMeta,
                        std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                        at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.lens = attnMeta.lens;
    _attnMeta.cachedLens = attnMeta.cachedLens;
    _attnMeta.isPrefills = attnMeta.isPrefills;
    InitXTensor(_attnMeta.vllmBlockTables, attnMeta.blockTables);
    InitXTensor(_attnMeta.vllmSlotMapping, attnMeta.slotMapping);
    InitXTensor(_attnMeta.vllmPosition, attnMeta.positions);
    Forward(rt, input, _attnMeta, kvCache, freqsCis, output, currStream);
}

void _CModel::ComputeLogits(XRuntime &rt, at::Tensor &input, at::Tensor &output, uint64_t currStream)
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

void _CModel::ForwardAndGetLogits(XRuntime &rt, at::Tensor &input,
                                  XModelAttnMeta& attnMeta,
                                  std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                                  at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _output, _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);
    InitXTensor(_freqsCis, freqsCis);

    if (kvCache.size() != _kv.size()) {
        std::cerr << __func__ << ": check kv cache failed!" << std::endl;
        return;
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        _kv[i].first.Init(kvCache[i].first.sizes().vec(), XDtype(kvCache[i].first), TensorPtr(kvCache[i].first));
        _kv[i].second.Init(kvCache[i].second.sizes().vec(), XDtype(kvCache[i].second), TensorPtr(kvCache[i].second));
    }

    if (currStream != 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    _model->ForwardAndGetLogits(rt, _input, attnMeta, _kv, _freqsCis, _output);

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardAndGetLogitsV1(XRuntime &rt, at::Tensor &input,
                                    CModelAttnMeta& attnMeta,
                                    std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                                    at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.lens = attnMeta.lens;
    _attnMeta.cachedLens = attnMeta.cachedLens;
    _attnMeta.isPrefills = attnMeta.isPrefills;
    InitXTensor(_attnMeta.vllmBlockTables, attnMeta.blockTables);
    InitXTensor(_attnMeta.vllmSlotMapping, attnMeta.slotMapping);
    InitXTensor(_attnMeta.vllmPosition, attnMeta.positions);
    ForwardAndGetLogits(rt, input, _attnMeta, kvCache, freqsCis, output, currStream);
}

void _CModel::ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input,
                                      XModelAttnMeta& attnMeta,
                                      std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                                      at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    XTensor _input, _output, _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);
    InitXTensor(_freqsCis, freqsCis);

    if (kvCache.size() != _kv.size()) {
        std::cerr << __func__ << ": check kv cache failed!" << std::endl;
        return;
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        _kv[i].first.Init(kvCache[i].first.sizes().vec(), XDtype(kvCache[i].first), TensorPtr(kvCache[i].first));
        _kv[i].second.Init(kvCache[i].second.sizes().vec(), XDtype(kvCache[i].second), TensorPtr(kvCache[i].second));
    }

    if (currStream != 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    _model->ForwardWithInputsEmbeds(rt, _input, attnMeta, _kv, _freqsCis, _output);

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input,
                                        CModelAttnMeta& attnMeta,
                                        std::vector<std::pair<at::Tensor, at::Tensor>>& kvCache,
                                        at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.lens = attnMeta.lens;
    _attnMeta.cachedLens = attnMeta.cachedLens;
    _attnMeta.isPrefills = attnMeta.isPrefills;
    InitXTensor(_attnMeta.vllmBlockTables, attnMeta.blockTables);
    InitXTensor(_attnMeta.vllmSlotMapping, attnMeta.slotMapping);
    InitXTensor(_attnMeta.vllmPosition, attnMeta.positions);
    ForwardWithInputsEmbeds(rt, input, _attnMeta, kvCache, freqsCis, output, currStream);
}

size_t _CModel::GetTensorPoolSize(void)
{
    return _model->GetTensorPoolSize();
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

void Matmul(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z, bool weightNZ)
{
    XTensor _x, _y, _z;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    XliteOpMatmul(rt, _x, _y, _z, weightNZ);
    rt.Synchronize();
}

void Embed(XRuntime &rt, at::Tensor &weight, at::Tensor &in, at::Tensor &out, uint32_t start, uint32_t end)
{
    XTensor _in, _out, _weight;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_weight, weight);
    XliteOpEmbed(rt, _in, _weight, start, end, _out);
    rt.Synchronize();
}

void RMSNorm(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &out, float normEps)
{
    XTensor _in, _out, _norm;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    XliteOpRmsNorm(rt, _in, _norm, _out, normEps, _in.shape[0], _in.shape[1]);
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
                  uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim,
                  uint32_t rotDim, uint32_t blockSize, bool isNeox)
{
    XTensor _inout, _kCache, _vCache, _position, _cossin, _slotMapping;

    InitXTensor(_inout, inout);
    InitXTensor(_kCache, kCache);
    InitXTensor(_vCache, vCache);
    InitXTensor(_position, position);
    InitXTensor(_cossin, cossin);
    InitXTensor(_slotMapping, slotMapping);
    XliteOpRopeCache(rt, _inout, _kCache, _vCache, _position, _cossin, _slotMapping,
                     nHeads, nKvHeads, headDim, rotDim, blockSize, isNeox);
    rt.Synchronize();
}

void AttentionPrefill(XRuntime &rt, at::Tensor &qkv, at::Tensor &kCache, at::Tensor &qk, at::Tensor &blockTables,
                      at::Tensor &cachedLens, at::Tensor &vCache, at::Tensor &output,
                      at::Tensor &lens, at::Tensor &prefillIndex, at::Tensor &cumPromptLens, uint32_t headDim,
                      uint32_t nHeads, uint32_t nKvHeads, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)
{
    XTensor _qkv, _kCache, _qk, _blockTables, _cachedLens, _vCache, _output, _lens, _prefillIndex, _cumPromptLens;

    InitXTensor(_qkv, qkv);
    InitXTensor(_kCache, kCache);
    InitXTensor(_qk, qk);
    InitXTensor(_blockTables, blockTables);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_vCache, vCache);
    InitXTensor(_output, output);
    InitXTensor(_lens, lens);
    InitXTensor(_prefillIndex, prefillIndex);
    InitXTensor(_cumPromptLens, cumPromptLens);
    XliteOpPrefillAttention(rt, _qkv, _kCache, _qk, _blockTables, _cachedLens,
                            _vCache, _output, _lens, _prefillIndex, _cumPromptLens,
                            headDim, nHeads, nKvHeads, blockSize, batch, maxNumBlock);
    rt.Synchronize();
}

void DecodeAttentionMix(XRuntime &rt, at::Tensor &a2v, at::Tensor &v2a, at::Tensor &qkv,
                        at::Tensor &kCache, at::Tensor &vCache, at::Tensor &cachedLens,
                        at::Tensor &blockTables, at::Tensor &output, at::Tensor &decodeIdx,
                        at::Tensor &cumPromptLens, uint32_t batch, uint32_t nHeads,
                        uint32_t headDim, uint32_t blockSize, uint32_t maxNumBlock,
                        uint32_t nKvHeads, uint32_t maxM)
{
    XTensor _a2v, _v2a, _qkv, _kCache, _vCache, _cachedLens,
            _blockTables, _output, _decodeIdx, _cumPromptLens;
    XTensor &qk = rt.pool->GetTensor({batch, nHeads / rt.tpSize(), maxM}, XDtype(qkv));

    InitXTensor(_a2v, a2v);
    InitXTensor(_v2a, v2a);
    InitXTensor(_qkv, qkv);
    InitXTensor(_kCache, kCache);
    InitXTensor(_vCache, vCache);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);
    InitXTensor(_output, output);
    InitXTensor(_decodeIdx, decodeIdx);
    InitXTensor(_cumPromptLens, cumPromptLens);

    XliteOpDecodeAttention(rt, _a2v, _v2a, _qkv, _kCache, _vCache, _cachedLens,
                           _blockTables, qk, _output, _decodeIdx, _cumPromptLens, batch, nHeads,
                           headDim, blockSize, maxNumBlock, nKvHeads, maxM);
    rt.Synchronize();

    rt.pool->PutTensor(qk);
}

void AddAndRMSNorm(XRuntime &rt, at::Tensor &in1, at::Tensor &in2, at::Tensor &norm, at::Tensor &out, float normEps)
{
    XTensor _in1, _in2, _out, _norm;

    InitXTensor(_in1, in1);
    InitXTensor(_in2, in2);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    XliteOpAddAndRmsNorm(rt, _in1, _in2, _norm, normEps, _out);
    rt.Synchronize();
}

PYBIND11_MODULE(_C, m) {
    py::class_<XRuntime>(m, "Runtime")
        .def(py::init<uint32_t, size_t, uint32_t, uint32_t, uint32_t>(),
            py::arg("devid"), py::arg("size") = 0, py::arg("rank") = 0,
            py::arg("tp_size") = 1, py::arg("dp_size") = 1)
        .def("update_core_num", &XRuntime::UpdateCoreNum)
        .def("init_tensor_pool", &XRuntime::InitTensorPool);

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
        .def_readwrite("qkv_bias", &XModelConfig::addBias)
        .def_readwrite("qk_norm", &XModelConfig::qkNorm);

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
        .def_readwrite("block_tables", &CModelAttnMeta::blockTables)
        .def_readwrite("slot_mapping", &CModelAttnMeta::slotMapping)
        .def_readwrite("positions", &CModelAttnMeta::positions);

    py::enum_<XModelAttnType>(m, "AttnType")
        .value("AttnMHA", XModelAttnType::XMODEL_ATTN_MHA)
        .value("AttnMLA", XModelAttnType::XMODEL_ATTN_MLA)
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
        .def("init", &_CModel::Init, "model init",
            py::arg("config"), py::arg("rank") = 0)
        .def("forward", &_CModel::Forward, "forward",
            py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
            py::arg("freqs_cis"), py::arg("output"), py::arg("curr_stream") = 0,
        py::call_guard<py::gil_scoped_release>())
        .def("forward", &_CModel::ForwardV1, "forward",
            py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
            py::arg("freqs_cis"), py::arg("output"), py::arg("curr_stream") = 0,
        py::call_guard<py::gil_scoped_release>())
        .def("compute_logits", &_CModel::ComputeLogits, "compute_logits",
            py::arg("rt"), py::arg("input"), py::arg("output"), py::arg("curr_stream") = 0,
        py::call_guard<py::gil_scoped_release>())
        .def("forward_and_get_logits", &_CModel::ForwardAndGetLogits, "forward_and_get_logits",
            py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
            py::arg("freqs_cis"), py::arg("output"), py::arg("curr_stream") = 0,
        py::call_guard<py::gil_scoped_release>())
        .def("forward_and_get_logits", &_CModel::ForwardAndGetLogitsV1, "forward_and_get_logits",
            py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
            py::arg("freqs_cis"), py::arg("output"), py::arg("curr_stream") = 0,
        py::call_guard<py::gil_scoped_release>())
        .def("forward_with_inputs_embeds", &_CModel::ForwardWithInputsEmbeds, "forward_with_inputs_embeds",
            py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
            py::arg("freqs_cis"), py::arg("output"), py::arg("curr_stream") = 0,
        py::call_guard<py::gil_scoped_release>())
        .def("forward_with_inputs_embeds", &_CModel::ForwardWithInputsEmbedsV1, "forward_with_inputs_embeds",
            py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
            py::arg("freqs_cis"), py::arg("output"), py::arg("curr_stream") = 0,
        py::call_guard<py::gil_scoped_release>())
        .def("get_tensor_pool_size", &_CModel::GetTensorPoolSize);

    py::class_<XCoreAssigner>(m, "CoreAssigner")
        .def(py::init<float>(), py::arg("prefillRatio"))
        .def("assign_core", &XCoreAssigner::AssignCore, py::call_guard<py::gil_scoped_release>())
        .def("release_core", &XCoreAssigner::ReleaseCore, py::call_guard<py::gil_scoped_release>());

    // kernels
    m.def("all_gather", &AllGather);
    m.def("reduce_scatter", &ReduceScatter);
    m.def("all_reduce", &AllReduce);
    m.def("add", &Add);
    m.def("matmul", &Matmul, "matmul",
          py::arg("rt"), py::arg("x"), py::arg("y"), py::arg("z"), py::arg("weight_nz") = false);
    m.def("embed", &Embed);
    m.def("rmsnorm", &RMSNorm);
    m.def("add_bias", &AddBias);
    m.def("silu_and_mul", &SiluAndMul);
    m.def("rope_and_cache", &RopeAndCache);
    m.def("attention_prefill", &AttentionPrefill);
    m.def("decode_attention_mix", &DecodeAttentionMix);
    m.def("add_and_rmsnorm", &AddAndRMSNorm);

    // funcs
    m.def("print", &Print);
}
