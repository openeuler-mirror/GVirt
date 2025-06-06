/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include "base.h"
#include "op.h"
#include "runtime.h"
#include "model.h"

namespace py = pybind11;

class _CModel {
public:
    _CModel() {};
    ~_CModel();
    void Init(struct XModelConfig &c, uint32_t rankId);
    void Forward(XRuntime &rt, at::Tensor &input, at::Tensor &output);

    // weights
    at::Tensor embed;
    at::Tensor norm;
    at::Tensor head;

    std::vector<at::Tensor> attnNorm;
    std::vector<at::Tensor> attnOut;
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
        default:
            std::cerr << __FILE__ << ":" << __LINE__ << "unknown data type " << t.scalar_type() << std::endl;
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
        std::cerr << __FILE__ << ":" << __LINE__ << " num of routed experts: " << moeREUpGate.size() << std::endl;
        return;
    }

    _model = new XModel(c, rankId);

    InitXTensor(_model->embed, embed);
    InitXTensor(_model->norm, norm);
    InitXTensor(_model->head, head);

    for (uint32_t i = 0; i < c.nLayers; i++) {
        InitXTensor(_model->attnNorm[i], attnNorm[i]);
        InitXTensor(_model->attnOut[i], attnOut[i]);
        InitXTensor(_model->mlaQA[i], mlaQA[i]);
        InitXTensor(_model->mlaQB[i], mlaQB[i]);
        InitXTensor(_model->mlaQNorm[i], mlaQNorm[i]);
        InitXTensor(_model->mlaKVA[i], mlaKVA[i]);
        InitXTensor(_model->mlaKVB[i], mlaKVB[i]);
        InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i]);
        InitXTensor(_model->mlpNorm[i], mlpNorm[i]);
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

    if (rankId == 0) {
        std::cout << "Euler Xlite Model Inited! [tensor paralled(" << c.defTpSize <<
            "), data parallel(" << c.defDpSize << "), expert parallel (" << c.moeEpSize << ")]" << std::endl;
    }
}

_CModel::~_CModel(void)
{
    delete _model;
}

void _CModel::Forward(XRuntime &rt, at::Tensor &input, at::Tensor &output)
{
    XTensor _input(input.sizes().vec(), XDtype(input), TensorPtr(input));
    XTensor _output(output.sizes().vec(), XDtype(output), TensorPtr(output));
    _model->Forward(rt, _input, _output);
}

void AllGather(XRuntime &rt, at::Tensor &out, at::Tensor &in)
{
    XTensor _in(in.sizes().vec(), XDtype(in), TensorPtr(in));
    XTensor _out(out.sizes().vec(), XDtype(out), TensorPtr(out));
    XliteOpAllGather(rt, _in, _out, TP);
}

void ReduceScatter(XRuntime &rt, at::Tensor &out, at::Tensor &in)
{
    XTensor _in(in.sizes().vec(), XDtype(in), TensorPtr(in));
    XTensor _out(out.sizes().vec(), XDtype(out), TensorPtr(out));
    XliteOpReduceScatter(rt, _in, _out, TP);
}

void AllReduce(XRuntime &rt, at::Tensor &out, at::Tensor &in)
{
    XTensor _in(in.sizes().vec(), XDtype(in), TensorPtr(in));
    XTensor _out(out.sizes().vec(), XDtype(out), TensorPtr(out));
    XliteOpAllReduceSum(rt, _in, _out, TP);
}

void Add(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z)
{
    XTensor _x(x.sizes().vec(), XDtype(x), TensorPtr(x));
    XTensor _y(y.sizes().vec(), XDtype(y), TensorPtr(y));
    XTensor _z(z.sizes().vec(), XDtype(z), TensorPtr(z));
    XliteOpAdd(rt, _x, _y, _z);
    rt.Synchronize();
}

void Print(at::Tensor &x)
{
    XTensor _x(x.sizes().vec(), XDtype(x), TensorPtr(x));
    _x.Print();
}

PYBIND11_MODULE(_C, m) {
    py::class_<XRuntime>(m, "runtime")
        .def(py::init<uint32_t, size_t, uint32_t, uint32_t, uint32_t>(),
            py::arg("devid"), py::arg("size"), py::arg("rank") = 0, py::arg("tp_size") = 1, py::arg("dp_size") = 1);

    py::class_<XModelConfig>(m, "model_config")
        .def(py::init<>())
        .def_readwrite("vocab_size", &XModelConfig::vocabSize)
        .def_readwrite("hidden_size", &XModelConfig::hiddenSize)
        .def_readwrite("n_layers", &XModelConfig::nLayers)
        .def_readwrite("n_heads", &XModelConfig::nHeads)
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
        .def_readwrite("moe_tp_size", &XModelConfig::moeTPSize);

    py::class_<_CModel>(m, "model")
        .def(py::init<>())
        .def_readwrite("embed", &_CModel::embed)
        .def_readwrite("norm", &_CModel::norm)
        .def_readwrite("head", &_CModel::head)
        .def_readwrite("attn_norm", &_CModel::attnNorm)
        .def_readwrite("attn_out", &_CModel::attnOut)
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
        .def("init", &_CModel::Init)
        .def("forward", &_CModel::Forward);

    // kernels
    m.def("all_gather", &AllGather);
    m.def("reduce_scatter", &ReduceScatter);
    m.def("all_reduce", &AllReduce);
    m.def("add", &Add);

    // funcs
    m.def("print", &Print);
}