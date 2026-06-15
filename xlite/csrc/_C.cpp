/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <optional>
#include "base.h"
#include "core_assigner.h"
#include "op.h"
#include "runtime.h"
#include "model.h"
#include "auto_tuner.h"

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
                 std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                 at::Tensor &output, uint64_t currStream);
    void ForwardV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                   std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                   at::Tensor &output, uint64_t currStream);
    void ForwardGetLogits(XRuntime &rt, at::Tensor &input, at::Tensor &indices, at::Tensor &output,
                          uint64_t currStream);
    void ForwardAndGetLogits(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                             std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                             at::Tensor &indices, at::Tensor &output, uint64_t currStream);
    void ForwardAndGetLogitsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                               std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                               at::Tensor &indices, at::Tensor &output, uint64_t currStream);
    void ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::vector<at::Tensor>> &kvCache,
                                 at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                 std::vector<at::Tensor> &deepstackInput);
    void ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                   std::vector<std::vector<at::Tensor>> &kvCache,
                                   at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                   std::vector<at::Tensor> &deepstackInput);
    size_t GetTensorPoolSize(int dbg);

    // weights
    at::Tensor embed;
    at::Tensor norm;
    at::Tensor normBias;
    at::Tensor head;

    std::vector<at::Tensor> attnNorm;
    std::vector<at::Tensor> attnNormBias;
    std::vector<at::Tensor> attnOut;
    std::vector<at::Tensor> attnOutInputScale;
    std::vector<at::Tensor> attnOutInputOffset;
    std::vector<at::Tensor> attnOutQuantBias;
    std::vector<at::Tensor> attnOutDeqScale;
    std::vector<at::Tensor> mhaQKV;
    std::vector<at::Tensor> mhaQKVBias;
    std::vector<at::Tensor> mhaQKVInputScale;
    std::vector<at::Tensor> mhaQKVInputOffset;
    std::vector<at::Tensor> mhaQKVQuantBias;
    std::vector<at::Tensor> mhaQKVDeqScale;
    std::vector<at::Tensor> mhaQNorm;
    std::vector<at::Tensor> mhaQNormBias;
    std::vector<at::Tensor> mhaKNorm;
    std::vector<at::Tensor> mhaKNormBias;
    std::vector<at::Tensor> mlaQKVA;
    std::vector<at::Tensor> mlaQKVAInputScale;
    std::vector<at::Tensor> mlaQKVAInputOffset;
    std::vector<at::Tensor> mlaQKVAQuantBias;
    std::vector<at::Tensor> mlaQKVADeqScale;
    std::vector<at::Tensor> mlaQB;
    std::vector<at::Tensor> mlaQBInputScale;
    std::vector<at::Tensor> mlaQBInputOffset;
    std::vector<at::Tensor> mlaQBQuantBias;
    std::vector<at::Tensor> mlaQBDeqScale;
    std::vector<at::Tensor> mlaQNorm;
    std::vector<at::Tensor> mlaQNormBias;
    std::vector<at::Tensor> mlaKVB;
    std::vector<at::Tensor> mlaKVNorm;
    std::vector<at::Tensor> mlaKVNormBias;
    std::vector<at::Tensor> indexQB;
    std::vector<at::Tensor> indexQBInputScale;
    std::vector<at::Tensor> indexQBInputOffset;
    std::vector<at::Tensor> indexQBQuantBias;
    std::vector<at::Tensor> indexQBDeqScale;
    std::vector<at::Tensor> indexKWeightsProj;
    std::vector<at::Tensor> indexKNorm;
    std::vector<at::Tensor> indexKNormBias;

    std::vector<at::Tensor> mlpNorm;
    std::vector<at::Tensor> mlpNormBias;
    std::vector<at::Tensor> mlpUpGate;
    std::vector<at::Tensor> mlpUpGateInputScale;
    std::vector<at::Tensor> mlpUpGateInputOffset;
    std::vector<at::Tensor> mlpUpGateQuantBias;
    std::vector<at::Tensor> mlpUpGateDeqScale;
    std::vector<at::Tensor> mlpDown;
    std::vector<at::Tensor> mlpDownInputScale;
    std::vector<at::Tensor> mlpDownInputOffset;
    std::vector<at::Tensor> mlpDownQuantBias;
    std::vector<at::Tensor> mlpDownDeqScale;

    std::vector<at::Tensor> moeGate;
    std::vector<at::Tensor> moeGateBias;
    std::vector<at::Tensor> moeSEUpGate;
    std::vector<at::Tensor> moeSEUpGateDeqScale;
    std::vector<at::Tensor> moeSEDown;
    std::vector<at::Tensor> moeSEDownDeqScale;
    std::vector<at::Tensor> moeREUpGate;
    std::vector<at::Tensor> moeREUpGateDeqScale;
    std::vector<at::Tensor> moeREDown;
    std::vector<at::Tensor> moeREDownDeqScale;

private:
    XModel *_model = nullptr;
    std::vector<std::vector<XTensor>> _kv;
    std::vector<XTensor> _deepstackInputEmbeds;
    void InitMatmulWeight(const std::string &name, std::vector<at::Tensor> &w,
                          std::vector<at::Tensor> &iScale, std::vector<at::Tensor> &iOffset,
                          std::vector<at::Tensor> &qBias, std::vector<at::Tensor> &dScale,
                          std::vector<MatmulWeight> &weightsXT, uint32_t currLayer,
                          bool isRowParallel, uint32_t tpRank, uint32_t layerOffset = 0);
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
    uint32_t idx = 0;
    uint32_t nLocalRoutedExperts = c.nRoutedExperts / c.moeEpSize;
    uint32_t expertsStartIdx = c.moeEpSize == 1 ? 0 : rankId / c.moeTPSize * nLocalRoutedExperts;
    uint32_t expertsEndIdx = expertsStartIdx + nLocalRoutedExperts;
    uint32_t numMoeLayers = c.nLayers - c.nDenseLayers;
    uint32_t nRE = numMoeLayers * nLocalRoutedExperts;
    uint32_t tpRank = rankId % c.defTpSize;

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

    if (!attnNormBias.empty() && attnNormBias.size() != c.nLayers) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << attnNormBias.size()
                  << std::endl;
        throw std::invalid_argument("Mismatched number of layers attention norm bias parameters");
    }
    if (!mlpNormBias.empty() && mlpNormBias.size() != c.nLayers) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mlpNormBias.size()
                  << std::endl;
        throw std::invalid_argument("Mismatched number of layers MLP norm bias parameters");
    }

    if (c.attnType == XMODEL_ATTN_MHA) {
        if (!attnOutInputScale.empty() && attnOutInputScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << attnOutInputScale.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers attention out input scale parameters");
        }
        if (!attnOutInputOffset.empty() && attnOutInputOffset.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << attnOutInputOffset.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers attention out input offset parameters");
        }
        if (!attnOutQuantBias.empty() && attnOutQuantBias.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << attnOutQuantBias.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers attention out quant bias parameters");
        }
        if (!attnOutDeqScale.empty() && attnOutDeqScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << attnOutDeqScale.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers attention out dequant scale parameters");
        }
        if (!mhaQKVInputScale.empty() && mhaQKVInputScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mhaQKVInputScale.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MHA QKV input scale parameters");
        }
        if (!mhaQKVInputOffset.empty() && mhaQKVInputOffset.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mhaQKVInputOffset.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MHA QKV input offset parameters");
        }
        if (!mhaQKVQuantBias.empty() && mhaQKVQuantBias.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mhaQKVQuantBias.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MHA QKV quant bias parameters");
        }
        if (!mhaQKVDeqScale.empty() && mhaQKVDeqScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mhaQKVDeqScale.size()
                      << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MHA QKV dequant scale parameters");
        }
    }

    if (c.attnType == XMODEL_ATTN_MLA) {
        if (mlaQKVA.size() != c.nLayers || mlaQB.size() != c.nLayers ||
            mlaQNorm.size() != c.nLayers || mlaKVB.size() != c.nLayers ||
            mlaKVNorm.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mlaQKVA.size()
                      << std::endl;
            throw std::invalid_argument("Mismatched number of layers MLA attention QA/QB/QA "
                                        "norm/KVA/KVB/KV norm parameters");
        }
        if (!mlaQKVAInputScale.empty() && mlaQKVAInputScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQKVAInputScale.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MLA QKVA input scale parameters");
        }
        if (!mlaQKVAInputOffset.empty() && mlaQKVAInputOffset.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQKVAInputOffset.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MLA QKVA input offset parameters");
        }
        if (!mlaQKVAQuantBias.empty() && mlaQKVAQuantBias.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQKVAQuantBias.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MLA QKVA quant bias parameters");
        }
        if (!mlaQKVADeqScale.empty() && mlaQKVADeqScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQKVADeqScale.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MLA QKVA dequant scale parameters");
        }
        if (!mlaQBInputScale.empty() && mlaQBInputScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQBInputScale.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MLA QB input scale parameters");
        }
        if (!mlaQBInputOffset.empty() && mlaQBInputOffset.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQBInputOffset.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MLA QB input offset parameters");
        }
        if (!mlaQBQuantBias.empty() && mlaQBQuantBias.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mlaQBQuantBias.size()
                      << std::endl;
            throw std::invalid_argument("Mismatched number of layers MLA QB quant bias parameters");
        }
        if (!mlaQBDeqScale.empty() && mlaQBDeqScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mlaQBDeqScale.size()
                      << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MLA QB dequant scale parameters");
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
        if (c.qkNorm && ((!mhaQNormBias.empty() && mhaQNormBias.size() != c.nLayers) ||
                         (!mhaKNormBias.empty() && mhaKNormBias.size() != c.nLayers))) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mhaQNormBias.size()
                      << ", " << mhaKNormBias.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers MHA attention Q/K norm bias parameters");
        }
    } else if (c.attnType == XMODEL_ATTN_DSA) {
        if (mlaQKVA.size() != c.nLayers || mlaQB.size() != c.nLayers ||
            mlaQNorm.size() != c.nLayers || mlaKVB.size() != c.nLayers ||
            mlaKVNorm.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mlaQKVA.size()
                      << std::endl;
            throw std::invalid_argument("Mismatched number of layers DSA attention QA/QB/QA "
                                        "norm/KVA/KVB/KV norm parameters");
        }
        if ((!mlaQNormBias.empty() && mlaQNormBias.size() != c.nLayers) ||
            (!mlaKVNormBias.empty() && mlaKVNormBias.size() != c.nLayers)) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mlaQNormBias.size()
                      << std::endl;
            throw std::invalid_argument("Mismatched number of layers DSA attention Q/KV norm "
                                        "bias parameters");
        }
        if (indexQB.size() != c.nLayers || indexKWeightsProj.size() != c.nLayers ||
            indexKNorm.size() != c.nLayers || indexKNormBias.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << indexQB.size()
                      << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers DSA attention index QB/KWeightsProj/KNorm parameters");
        }
        if (!mlaQKVAInputScale.empty() && mlaQKVAInputScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQKVAInputScale.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers DSA QKVA input scale parameters");
        }
        if (!mlaQKVAInputOffset.empty() && mlaQKVAInputOffset.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQKVAInputOffset.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers DSA QKVA input offset parameters");
        }
        if (!mlaQKVAQuantBias.empty() && mlaQKVAQuantBias.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQKVAQuantBias.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers DSA QKVA quant bias parameters");
        }
        if (!mlaQKVADeqScale.empty() && mlaQKVADeqScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQKVADeqScale.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers DSA QKVA dequant scale parameters");
        }
        if (!mlaQBInputScale.empty() && mlaQBInputScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQBInputScale.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers DSA QB input scale parameters");
        }
        if (!mlaQBInputOffset.empty() && mlaQBInputOffset.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": num of layers: " << mlaQBInputOffset.size() << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers DSA QB input offset parameters");
        }
        if (!mlaQBQuantBias.empty() && mlaQBQuantBias.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mlaQBQuantBias.size()
                      << std::endl;
            throw std::invalid_argument("Mismatched number of layers DSA QB quant bias parameters");
        }
        if (!mlaQBDeqScale.empty() && mlaQBDeqScale.size() != c.nLayers) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": num of layers: " << mlaQBDeqScale.size()
                      << std::endl;
            throw std::invalid_argument(
                "Mismatched number of layers DSA QB dequant scale parameters");
        }
    } else {
        std::cerr << __FILE__ << ":" << __LINE__ << ": invalid attention type: " << c.attnType
                  << std::endl;
        throw std::invalid_argument("Invalid attention type");
    }

    if (mlpUpGate.size() != c.nDenseLayers || mlpDown.size() != c.nDenseLayers) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": num of dense layers: " << mlpUpGate.size()
                  << std::endl;
        throw std::invalid_argument("Mismatched number of dense layers up gate or down parameters");
    }

    if ((!mlpUpGateInputScale.empty() && mlpUpGateInputScale.size() != c.nDenseLayers) ||
        (!mlpUpGateInputOffset.empty() && mlpUpGateInputOffset.size() != c.nDenseLayers) ||
        (!mlpUpGateQuantBias.empty() && mlpUpGateQuantBias.size() != c.nDenseLayers) ||
        (!mlpUpGateDeqScale.empty() && mlpUpGateDeqScale.size() != c.nDenseLayers)) {
        std::cerr << __FILE__ << ":" << __LINE__
                  << ": num of dense layers: " << mlpUpGateInputScale.size() << std::endl;
        throw std::invalid_argument("Mismatched number of dense layers up gate quanted parameters");
    }

    if ((!mlpDownInputScale.empty() && mlpDownInputScale.size() != c.nDenseLayers) ||
        (!mlpDownInputOffset.empty() && mlpDownInputOffset.size() != c.nDenseLayers) ||
        (!mlpDownQuantBias.empty() && mlpDownQuantBias.size() != c.nDenseLayers) ||
        (!mlpDownDeqScale.empty() && mlpDownDeqScale.size() != c.nDenseLayers)) {
        std::cerr << __FILE__ << ":" << __LINE__
                  << ": num of dense layers: " << mlpDownInputScale.size() << std::endl;
        throw std::invalid_argument("Mismatched number of dense layers down quanted parameters");
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

    if (c.nSharedExperts != 0 && ((!moeSEUpGateDeqScale.empty() &&
                                   moeSEUpGateDeqScale.size() != (c.nLayers - c.nDenseLayers)) ||
                                  (!moeSEDownDeqScale.empty() &&
                                   moeSEDownDeqScale.size() != (c.nLayers - c.nDenseLayers)))) {
        std::cerr << __FILE__ << ":" << __LINE__
                  << ": num of moe layers: " << moeSEUpGateDeqScale.size() << std::endl;
        throw std::invalid_argument(
            "Mismatched number of moe layers with shared experts quantization parameters");
    }

    _model = new XModel(c, rankId);

    InitXTensor(_model->embed, embed);
    InitXTensor(_model->norm, norm);
    InitXTensor(_model->head, head);
    if (normBias.defined()) {
        InitXTensor(_model->normBias, normBias);
    }

    for (uint32_t i = 0; i < c.nLayers; i++) {
        InitXTensor(_model->attnNorm[i], attnNorm[i]);
        InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                         attnOutQuantBias, attnOutDeqScale, _model->attnOut, i, true, tpRank);
        if (!attnNormBias.empty()) {
            InitXTensor(_model->attnNormBias[i], attnNormBias[i]);
        }
        if (!mlpNormBias.empty()) {
            InitXTensor(_model->mlpNormBias[i], mlpNormBias[i]);
        }
        InitXTensor(_model->mlpNorm[i], mlpNorm[i]);
        if (c.attnType == XMODEL_ATTN_MLA) {
            InitMatmulWeight("mlaQKVA", mlaQKVA, mlaQKVAInputScale, mlaQKVAInputOffset,
                             mlaQKVAQuantBias, mlaQKVADeqScale, _model->mlaQKVA, i, false, tpRank);
            InitMatmulWeight("mlaQB", mlaQB, mlaQBInputScale, mlaQBInputOffset, mlaQBQuantBias,
                             mlaQBDeqScale, _model->mlaQB, i, false, tpRank);
            InitXTensor(_model->mlaQNorm[i], mlaQNorm[i]);
            if (!mlaQNormBias.empty()) {
                InitXTensor(_model->mlaQNormBias[i], mlaQNormBias[i]);
            }
            InitXTensor(_model->mlaKVB[i], mlaKVB[i]);
            InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i]);
            if (!mlaKVNormBias.empty()) {
                InitXTensor(_model->mlaKVNormBias[i], mlaKVNormBias[i]);
            }
        } else if (c.attnType == XMODEL_ATTN_MHA) {
            InitMatmulWeight("mhaQKV", mhaQKV, mhaQKVInputScale, mhaQKVInputOffset, mhaQKVQuantBias,
                             mhaQKVDeqScale, _model->mhaQKV, i, false, tpRank);
            if (c.addBias) {
                InitXTensor(_model->mhaQKVBias[i], mhaQKVBias[i]);
            }
            if (c.qkNorm) {
                InitXTensor(_model->mhaQNorm[i], mhaQNorm[i]);
                InitXTensor(_model->mhaKNorm[i], mhaKNorm[i]);
                if (!mhaQNormBias.empty()) {
                    InitXTensor(_model->mhaQNormBias[i], mhaQNormBias[i]);
                }
                if (!mhaKNormBias.empty()) {
                    InitXTensor(_model->mhaKNormBias[i], mhaKNormBias[i]);
                }
            }
        } else if (c.attnType == XMODEL_ATTN_DSA) {
            InitMatmulWeight("mlaQKVA", mlaQKVA, mlaQKVAInputScale, mlaQKVAInputOffset,
                             mlaQKVAQuantBias, mlaQKVADeqScale, _model->mlaQKVA, i, false, tpRank);
            InitMatmulWeight("mlaQB", mlaQB, mlaQBInputScale, mlaQBInputOffset, mlaQBQuantBias,
                             mlaQBDeqScale, _model->mlaQB, i, false, tpRank);
            InitXTensor(_model->mlaQNorm[i], mlaQNorm[i]);
            if (!mlaQNormBias.empty()) {
                InitXTensor(_model->mlaQNormBias[i], mlaQNormBias[i]);
            }
            InitXTensor(_model->mlaKVB[i], mlaKVB[i]);
            InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i]);
            if (!mlaKVNormBias.empty()) {
                InitXTensor(_model->mlaKVNormBias[i], mlaKVNormBias[i]);
            }
            InitMatmulWeight("indexQB", indexQB, indexQBInputScale, indexQBInputOffset,
                             indexQBQuantBias, indexQBDeqScale, _model->indexQB, i, false, tpRank);
            InitXTensor(_model->indexKWeightsProj[i], indexKWeightsProj[i]);
            InitXTensor(_model->indexKNorm[i], indexKNorm[i]);
            InitXTensor(_model->indexKNormBias[i], indexKNormBias[i]);
        }
    }

    for (uint32_t i = 0; i < c.nDenseLayers; i++) {
        InitMatmulWeight("mlpUpGate", mlpUpGate, mlpUpGateInputScale, mlpUpGateInputOffset,
                         mlpUpGateQuantBias, mlpUpGateDeqScale, _model->mlpUpGate, i, false,
                         tpRank);
        InitMatmulWeight("mlpDown", mlpDown, mlpDownInputScale, mlpDownInputOffset,
                         mlpDownQuantBias, mlpDownDeqScale, _model->mlpDown, i, true, tpRank);
    }

    for (uint32_t i = c.nDenseLayers; i < c.nLayers; i++) {
        InitXTensor(_model->moeGate[i], moeGate[i - c.nDenseLayers]);
        if (c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
            InitXTensor(_model->moeGateBias[i], moeGateBias[i - c.nDenseLayers]);
        }
        std::vector<at::Tensor> emptyWeights = {};
        if (c.nSharedExperts != 0) {
            InitMatmulWeight("moeSEUpGate", moeSEUpGate, emptyWeights, emptyWeights, emptyWeights,
                             moeSEUpGateDeqScale, _model->moeSEUpGate, i, false, tpRank,
                             c.nDenseLayers);
            InitMatmulWeight("moeSEDown", moeSEDown, emptyWeights, emptyWeights, emptyWeights,
                             moeSEDownDeqScale, _model->moeSEDown, i, true, tpRank, c.nDenseLayers);
        }

        for (uint32_t j = expertsStartIdx; j < expertsEndIdx; j++) {
            InitXTensor(_model->moeREUpGate[i][j], moeREUpGate[idx]);
            InitXTensor(_model->moeREDown[i][j], moeREDown[idx]);
            if (!moeREUpGateDeqScale.empty()) {
                InitXTensor(_model->moeREUpGateDeqScale[i][j], moeREUpGateDeqScale[idx]);
            }
            if (!moeREDownDeqScale.empty()) {
                InitXTensor(_model->moeREDownDeqScale[i][j], moeREDownDeqScale[idx]);
            }
            idx++;
        }
    }

    _model->Init();

    if (rankId == 0) {
        std::cout << "Euler Xlite Model Inited! [tensor paralled(" << c.defTpSize
                  << "), data parallel(" << c.defDpSize << "), expert parallel(" << c.moeEpSize
                  << ")]" << std::endl;
    }

    _kv.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        _kv[i].resize(c.attnType == XMODEL_ATTN_DSA ? 3 : 2);
    }
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
                      std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                      at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _output, _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);
    InitXTensor(_freqsCis, freqsCis);

    if (kvCache.size() < _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    if (input.size(0) > output.size(0)) {
        throw std::runtime_error(std::string(__func__) + ": input's size 0 > output's size 0");
    }

    if (input.size(0) == 0) {
        return;
    }

    // only calculate the region [_input.shape[0], _output.shape[1]]
    _output.View({_input.shape[0], _output.shape[1]});

    for (uint64_t i = 0; i < _kv.size(); i++) {
        if (kvCache[i].size() != _kv[i].size()) {
            throw std::runtime_error(std::string(__func__) + ": check kv cache failed at layer " +
                                     std::to_string(i));
        }
        for (uint64_t j = 0; j < _kv[i].size(); j++) {
            InitXTensor(_kv[i][j], kvCache[i][j]);
        }
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
                        std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                        at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.lens = attnMeta.lens;
    _attnMeta.cachedLens = attnMeta.cachedLens;
    _attnMeta.blockTables = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.vllmPosition, attnMeta.positions);
    Forward(rt, input, _attnMeta, kvCache, freqsCis, output, currStream);
}

void _CModel::ForwardGetLogits(XRuntime &rt, at::Tensor &input, at::Tensor &indices,
                               at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _indices, _output;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_indices, indices);
    InitXTensor(_output, output);

    if (input.size(0) == 0) {
        return;
    }

    if (currStream != 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    _model->ForwardGetLogits(rt, _input, _indices, _output);

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardAndGetLogits(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                  std::vector<std::vector<at::Tensor>> &kvCache,
                                  at::Tensor &freqsCis, at::Tensor &indices, at::Tensor &output,
                                  uint64_t currStream)
{
    XTensor _input, _indices, _output, _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_indices, indices);
    InitXTensor(_output, output);
    InitXTensor(_freqsCis, freqsCis);

    if (kvCache.size() != _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    if (input.size(0) == 0) {
        return;
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        if (kvCache[i].size() != _kv[i].size()) {
            throw std::runtime_error(std::string(__func__) + ": check kv cache failed at layer " +
                                     std::to_string(i));
        }
        for (uint64_t j = 0; j < _kv[i].size(); j++) {
            InitXTensor(_kv[i][j], kvCache[i][j]);
        }
    }

    if (currStream != 0 && rt.taskId == 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    if (rt.multiTaskParallel && rt.taskId == 1) {
        rt.NotifyWaitPeerStream();
    }

    _model->ForwardAndGetLogits(rt, _input, attnMeta, _kv, _deepstackInputEmbeds, _freqsCis,
                                _indices, _output);

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
                                    std::vector<std::vector<at::Tensor>> &kvCache,
                                    at::Tensor &freqsCis, at::Tensor &indices, at::Tensor &output,
                                    uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.lens = attnMeta.lens;
    _attnMeta.cachedLens = attnMeta.cachedLens;
    _attnMeta.blockTables = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.vllmPosition, attnMeta.positions);
    ForwardAndGetLogits(rt, input, _attnMeta, kvCache, freqsCis, indices, output, currStream);
}

void _CModel::ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                      std::vector<std::vector<at::Tensor>> &kvCache,
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

    if (input.size(0) > output.size(0)) {
        throw std::runtime_error(std::string(__func__) + ": input's size 0 > output's size 0");
    }

    if (input.size(0) == 0) {
        return;
    }

    // only calculate the region [_input.shape[0], _output.shape[1]]
    _output.View({_input.shape[0], _output.shape[1]});

    if (deepstackInput.size() != _deepstackInputEmbeds.size()) {
        throw std::runtime_error(std::string(__func__) + ": check deepstack input failed");
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        if (kvCache[i].size() != _kv[i].size()) {
            throw std::runtime_error(std::string(__func__) + ": check kv cache failed at layer " +
                                     std::to_string(i));
        }
        for (uint64_t j = 0; j < _kv[i].size(); j++) {
            InitXTensor(_kv[i][j], kvCache[i][j]);
        }
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
                                        std::vector<std::vector<at::Tensor>> &kvCache,
                                        at::Tensor &freqsCis, at::Tensor &output,
                                        uint64_t currStream,
                                        std::vector<at::Tensor> &deepstackInput)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.lens = attnMeta.lens;
    _attnMeta.cachedLens = attnMeta.cachedLens;
    _attnMeta.blockTables = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.vllmPosition, attnMeta.positions);
    ForwardWithInputsEmbeds(rt, input, _attnMeta, kvCache, freqsCis, output, currStream,
                            deepstackInput);
}

size_t _CModel::GetTensorPoolSize(int dbg)
{
    return _model->GetTensorPoolSize(dbg);
}

void _CModel::InitMatmulWeight(const std::string &name, std::vector<at::Tensor> &w,
                               std::vector<at::Tensor> &iScale, std::vector<at::Tensor> &iOffset,
                               std::vector<at::Tensor> &qBias, std::vector<at::Tensor> &dScale,
                               std::vector<MatmulWeight> &weightsXT, uint32_t currLayer,
                               bool isRowParallel, uint32_t tpRank, uint32_t layerOffset)
{
    MatmulWeight &wXT = weightsXT[currLayer];
    uint32_t weightLayer = currLayer - layerOffset;

    wXT.name = name + "[" + std::to_string(currLayer) + "]";
    InitXTensor(wXT.weight, w[weightLayer]);
    if (wXT.weight.dtype == INT8 && dScale.size() <= weightLayer) {
        std::string errStr = DBG_PREFIX + ": " + name +
                             "dequant scale parameters are "
                             "required when weights are quantized";
        throw std::invalid_argument(errStr);
    }
    if (weightLayer < iScale.size()) {
        InitXTensor(wXT.inputScale, iScale[weightLayer]);
    }
    if (weightLayer < iOffset.size()) {
        InitXTensor(wXT.inputOffset, iOffset[weightLayer]);
    }
    // Notice: only tpRank == 0 in RowParallelLinear need to add quant_bias
    if (weightLayer < qBias.size() && (!isRowParallel || tpRank == 0)) {
        InitXTensor(wXT.quantBias, qBias[weightLayer]);
    }
    if (weightLayer < dScale.size()) {
        InitXTensor(wXT.deqScale, dScale[weightLayer]);
    }
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

void AlltoAllV(XRuntime &rt, at::Tensor &out, at::Tensor &in, at::Tensor &sendCounts,
               at::Tensor &recvCounts, at::Tensor &sdispls, at::Tensor &rdispls,
               uint32_t commType = 0)
{
    XTensor _in, _out, _sendCounts, _recvCounts, _sdispls, _rdispls;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_sendCounts, sendCounts);
    InitXTensor(_recvCounts, recvCounts);
    InitXTensor(_sdispls, sdispls);
    InitXTensor(_rdispls, rdispls);

    enum commType type = TP;
    if (commType == 1) {
        type = DP;
    } else if (commType == 2) {
        type = EP;
    }
    XliteOpAlltoAllV(rt, _in, _out, _sendCounts, _recvCounts, _sdispls, _rdispls, type);
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

void Print(at::Tensor &x, const char *name, uint32_t nRow, uint32_t nCol)
{
    XTensor _x;

    InitXTensor(_x, x);
    _x.Print(name, nRow, nCol);
}

void Matmul(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z, bool weightNZ,
            bool transpose)
{
    XTensor _x, _y, _z, _bias, _deqScale;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias, _deqScale, transpose);
}

uint64_t MatmulBench(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z,
                     at::Tensor &x_warmup, at::Tensor &y_warmup, at::Tensor &z_warmup,
                     int iterations, int warmup_iterations, bool weightNZ, bool transpose)
{
    XTensor _x, _y, _z, _x_warmup, _y_warmup, _z_warmup, _bias, _deqScale;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);

    InitXTensor(_x_warmup, x_warmup);
    InitXTensor(_y_warmup, y_warmup);
    InitXTensor(_z_warmup, z_warmup);

    for (int i = 0; i < warmup_iterations; i++) {
        XliteOpMatmul(rt, _x_warmup, _y_warmup, _z_warmup, weightNZ, _bias, _deqScale, transpose);
    }
    rt.Synchronize();

    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias, _deqScale, transpose);
    }
    rt.Synchronize();
    auto end = std::chrono::steady_clock::now();

    auto res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    return res / iterations;
}

void MatmulWithBias(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z, at::Tensor &bias,
                    bool weightNZ)
{
    XTensor _x, _y, _z, _bias, _deqSacle;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    InitXTensor(_bias, bias);
    XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias, _deqSacle);
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
             uint32_t normDim, uint32_t cntPerToken, uint32_t inStartOffset,
             uint32_t outStartOffset, bool useNorm, std::optional<at::Tensor> variance)
{
    XTensor _in, _out, _norm, _normBias, _variance;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    if (variance.has_value()) {
        std::cout << "variance: " << variance.value() << std::endl;
        InitXTensor(_variance, variance.value());
    }
    XliteOpRmsNorm(rt, _in, _norm, _out, normEps, normDim == 0 ? _in.shape[1] : normDim, useNorm,
                   _normBias, cntPerToken, inStartOffset, outStartOffset, _variance);
    rt.Synchronize();
}

void RMSNormWithBias(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &normBias,
                     at::Tensor &out, float normEps, uint32_t normDim, uint32_t cntPerToken,
                     uint32_t inStartOffset, uint32_t outStartOffset)
{
    XTensor _in, _out, _norm, _normBias;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    InitXTensor(_normBias, normBias);
    XliteOpRmsNorm(rt, _in, _norm, _out, normEps, normDim == 0 ? _in.shape[1] : normDim, true,
                   _normBias, cntPerToken, inStartOffset, outStartOffset);
    rt.Synchronize();
}

void LayerNorm(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &normBias,
               at::Tensor &out, float normEps, uint32_t normDim)
{
    XTensor _in, _out, _norm, _normBias;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    InitXTensor(_normBias, normBias);
    XliteOpLayerNorm(rt, _in, _norm, _normBias, _out, normEps, normDim);
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
               at::Tensor &output, at::Tensor &queryStartLoc, at::Tensor &lens,
               at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads, uint32_t nKvHeads,
               uint32_t headDim, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock,
               bool enableFlashAttention, uint32_t tileSizeOfCachedKV)
{
    XTensor _qkv, _kCache, _vCache, _qk, _output, _queryStartLoc, _lens, _cachedLens, _blockTables;

    InitXTensor(_qkv, qkv);
    InitXTensor(_kCache, kCache);
    InitXTensor(_vCache, vCache);
    InitXTensor(_output, output);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);

    if (!enableFlashAttention) {
        XTensor &qk =
            rt.GetTensor({rt.aicNum * XLITE_ATTENTION_MAX_M0 * 2, maxNumBlock * blockSize},
                         XDtype(qkv), DBG_LOC);
        XliteOpAttention(rt, _qkv, _kCache, _vCache, qk, _output, _queryStartLoc, _lens,
                         _cachedLens, _blockTables, nHeads, nKvHeads, headDim, blockSize, batch,
                         maxNumBlock);
        rt.Synchronize();
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_ATTENTION_MAX_M0 * 2, tileSizeOfCachedKV},
                                   XDtype(qkv), DBG_LOC);
        XTensor &sv =
            rt.GetTensor({rt.aicNum * XLITE_ATTENTION_MAX_M0 * 2, headDim}, XDtype(qkv), DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_ATTENTION_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_ATTENTION_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({_qkv.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({_qkv.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &sync = rt.GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
        sync.Memset(0);
        XliteOpFlashAttention(rt, _qkv, _kCache, _vCache, qk, sv, max, sum, lastMax, lastSum, sync,
                              _output, _queryStartLoc, _lens, _cachedLens, _blockTables, nHeads,
                              nKvHeads, headDim, blockSize, batch, maxNumBlock, tileSizeOfCachedKV);
        rt.Synchronize();
        rt.PutTensor(sync);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }
}

void MLA(XRuntime &rt, at::Tensor &qWithQr, at::Tensor &kCache, at::Tensor &vCache,
         at::Tensor &wkvb, at::Tensor &output, at::Tensor &queryStartLoc, at::Tensor &lens,
         at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
         uint32_t nopeHeadDim, uint32_t vHeadDim, uint32_t kvLoraRank, uint32_t blockSize,
         uint32_t batch, uint32_t maxNumBlock, float scale, bool weightNz,
         bool enableFlashAttention, uint32_t tileSizeOfCachedKV)
{
    XTensor _qWithQr, _kCache, _vCache, _wkvb, _output, _queryStartLoc, _lens, _cachedLens,
        _blockTables;
    uint32_t qHeads = nHeads;

    InitXTensor(_qWithQr, qWithQr);
    InitXTensor(_kCache, kCache);
    InitXTensor(_vCache, vCache);
    InitXTensor(_wkvb, wkvb);
    InitXTensor(_output, output);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);

    if (!enableFlashAttention) {
        XTensor &qk =
            rt.GetTensor({rt.aicNum * XLITE_ATTENTION_MAX_M0 * 2, maxNumBlock * blockSize},
                         XDtype(qWithQr), DBG_LOC);
        XliteOpMLA(rt, _qWithQr, _kCache, _vCache, _wkvb, qk, _output, _queryStartLoc, _lens,
                   _cachedLens, _blockTables, qHeads, ropeHeadDim, nopeHeadDim, vHeadDim,
                   kvLoraRank, blockSize, batch, maxNumBlock, scale, weightNz);
        rt.Synchronize();
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_ATTENTION_MAX_M0 * 2, tileSizeOfCachedKV},
                                   XDtype(qWithQr), DBG_LOC);
        XTensor &sv = rt.GetTensor({rt.aicNum * XLITE_ATTENTION_MAX_M0 * 2, vHeadDim},
                                   XDtype(qWithQr), DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_ATTENTION_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_ATTENTION_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({_qWithQr.shape[0], qHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({_qWithQr.shape[0], qHeads}, FP32, DBG_LOC);
        XTensor &sync = rt.GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
        sync.Memset(0);

        XliteOpFlashMLA(rt, _qWithQr, _kCache, _vCache, _wkvb, qk, sv, max, sum, lastMax, lastSum,
                        sync, _output, _queryStartLoc, _lens, _cachedLens, _blockTables, qHeads,
                        ropeHeadDim, nopeHeadDim, vHeadDim, kvLoraRank, blockSize, batch,
                        maxNumBlock, scale, weightNz, tileSizeOfCachedKV);

        rt.Synchronize();
        rt.PutTensor(sync);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }
}

void MLAWithIndices(XRuntime &rt, at::Tensor &qWithQr, at::Tensor &kCache, at::Tensor &vCache,
                    at::Tensor &wkvb, at::Tensor &output, at::Tensor &queryStartLoc,
                    at::Tensor &lens, at::Tensor &cachedLens, at::Tensor &blockTables,
                    uint32_t nHeads, uint32_t ropeHeadDim, uint32_t nopeHeadDim, uint32_t vHeadDim,
                    uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock,
                    float scale, uint32_t topK, at::Tensor &topkIndices, bool weightNz,
                    bool enableFlashAttention)
{
    XTensor _qWithQr, _kCache, _vCache, _wkvb, _output, _queryStartLoc, _lens, _cachedLens,
        _blockTables, _topkIndices;
    uint32_t qHeads = nHeads;

    InitXTensor(_qWithQr, qWithQr);
    InitXTensor(_kCache, kCache);
    InitXTensor(_vCache, vCache);
    InitXTensor(_wkvb, wkvb);
    InitXTensor(_output, output);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);
    InitXTensor(_topkIndices, topkIndices);

    XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_ATTENTION_MAX_M0 * 2, maxNumBlock * blockSize},
                               XDtype(qWithQr), DBG_LOC);
    XliteOpMLA(rt, _qWithQr, _kCache, _vCache, _wkvb, qk, _output, _queryStartLoc, _lens,
               _cachedLens, _blockTables, qHeads, ropeHeadDim, nopeHeadDim, vHeadDim, kvLoraRank,
               blockSize, batch, maxNumBlock, scale, weightNz, topK, _topkIndices);
    rt.Synchronize();
    rt.PutTensor(qk);
}

void AddAndRMSNorm(XRuntime &rt, at::Tensor &in, at::Tensor &addInOut, at::Tensor &norm,
                   at::Tensor &out, float normEps)
{
    XTensor _in, _addInOut, _out, _norm;

    InitXTensor(_in, in);
    InitXTensor(_addInOut, addInOut);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    XliteOpAddAndRmsNorm(rt, _in, _addInOut, _norm, normEps, _out);
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

void TopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &outIndices,
          at::Tensor &queryLens, at::Tensor &cachedLens, size_t k)
{
    XTensor _scores, _indices, _outIndices, _queryLens, _cachedLens;

    InitXTensor(_scores, scores);
    InitXTensor(_indices, indices);
    InitXTensor(_outIndices, outIndices);
    InitXTensor(_queryLens, queryLens);
    InitXTensor(_cachedLens, cachedLens);

    XliteOpTopK(rt, _scores, _indices, _outIndices, _queryLens, _cachedLens, k);

    rt.Synchronize();
}

void CastUp(XRuntime &rt, at::Tensor &in, at::Tensor &out)
{
    XTensor _in, _out;
    XTensor &inScale = rt.GetTensor({1}, XDtype(in), DBG_LOC);

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    XliteOpCastUp(rt, _in, inScale, _out);
    rt.Synchronize();
    rt.PutTensor(inScale);
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
    XTensor _in, _counts, _output, _scalesTensor;
    XTensor *_scales = &_scalesTensor;
    std::vector<void *> p;
    uint32_t i, num = counts.size(0);
    bool hasScale = (scales.size() == num);

    InitXTensor(_in, in);
    InitXTensor(_counts, counts);
    InitXTensor(_output, output);
    XTensor &_weights = rt.GetTensor({num}, INT64, DBG_LOC);

    p.resize(num);
    for (i = 0; i < num; i++) {
        p[i] = TensorPtr(weights[i]);
    }
    rt.MemcpyH2D(_weights.ptr, reinterpret_cast<void *>(p.data()), num * sizeof(void *));

    if (hasScale) {
        _scales = &rt.GetTensor({num}, INT64, DBG_LOC);
        for (i = 0; i < num; i++) {
            p[i] = TensorPtr(scales[i]);
        }
        rt.MemcpyH2D(_scales->ptr, reinterpret_cast<void *>(p.data()), num * sizeof(void *));
    }

    XliteOpGroupMatmul(rt, _in, _weights, *_scales, _counts, start, end, XDtype(weights[0]), outDim,
                       inDim, _output, weightNZ, transpose);
    rt.Synchronize();
    rt.PutTensor(_weights);
    if (hasScale) {
        rt.PutTensor(*_scales);
    }
}

void Softmax(XRuntime &rt, at::Tensor &x, uint32_t calcLen, bool isLong)
{
    XTensor _x;
    InitXTensor(_x, x);
    if (isLong) {
        XTensor &expBuf = rt.GetTensor({1, _x.shape[1]}, FP32, DBG_LOC);
        XliteOpSoftmaxLong(rt, calcLen, _x, expBuf);
        rt.PutTensor(expBuf);
    } else {
        XliteOpSoftmax(rt, calcLen, _x);
    }
    rt.Synchronize();
}

void RopeComplex(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim, uint32_t ropeDim,
                 at::Tensor &inputWithR, at::Tensor &freqs, at::Tensor &position)
{
    XTensor _inputWithR, _freqs, _position, _outputPe;
    InitXTensor(_inputWithR, inputWithR);
    InitXTensor(_freqs, freqs);
    InitXTensor(_position, position);
    XliteOpRopeComplex(rt, nLocalHeads, stepDim, ropeDim, stepDim - ropeDim, _inputWithR, _freqs,
                       _position);
    rt.Synchronize();
}

void Quant(XRuntime &rt, at::Tensor &x, at::Tensor &scaleReciprocal, at::Tensor &offset,
           at::Tensor &out)
{
    XTensor _x, _scaleRec, _offset, _out;
    InitXTensor(_x, x);
    InitXTensor(_scaleRec, scaleReciprocal);
    InitXTensor(_offset, offset);
    InitXTensor(_out, out);
    XliteOpQuant(rt, _x, _scaleRec, _offset, _out);
    rt.Synchronize();
}

void QuantDyn(XRuntime &rt, at::Tensor &x, at::Tensor &scale, at::Tensor &out)
{
    XTensor _x, _scale, _out;
    InitXTensor(_x, x);
    InitXTensor(_scale, scale);
    InitXTensor(_out, out);
    XliteOpQuantDyn(rt, _x, _scale, _out);
    rt.Synchronize();
}

void MatmulDeQuant(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &bias,
                   at::Tensor &deqScale, at::Tensor &z, bool weightNZ, bool transpose)
{
    XTensor _x, _y, _z, _bias, _deqScale;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_bias, bias);
    InitXTensor(_deqScale, deqScale);
    InitXTensor(_z, z);
    XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias, _deqScale, transpose);
    rt.Synchronize();
}

void DeQuant(XRuntime &rt, at::Tensor &in, at::Tensor &scale, at::Tensor &out, bool hasScale)
{
    XTensor _in, _scale, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    if (hasScale) {
        InitXTensor(_scale, scale);
    }
    XliteOpDeQuant(rt, _in, _out, _scale);
    rt.Synchronize();
}

void IndexerScores(XRuntime &rt, at::Tensor &q, at::Tensor &kCache, at::Tensor &weight,
                   at::Tensor &scores, at::Tensor &queryStartLoc, at::Tensor &lens,
                   at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads,
                   uint32_t headDim, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)
{
    XTensor _q, _kCache, _weight, _scores, _queryStartLoc, _lens, _cachedLens, _blockTables;

    InitXTensor(_q, q);
    InitXTensor(_kCache, kCache);
    InitXTensor(_weight, weight);
    InitXTensor(_scores, scores);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);
    XliteOpIndexerScores(rt, _q, _kCache, _weight, _scores, _queryStartLoc, _lens, _cachedLens,
                         _blockTables, nHeads, headDim, blockSize, batch, maxNumBlock);
    rt.Synchronize();
}

void Muls(XRuntime &rt, at::Tensor &input, float scale, at::Tensor &output)
{
    XTensor _input, _output;
    InitXTensor(_input, input);
    InitXTensor(_output, output);
    XliteOpMuls(rt, _input, scale, _output);
    rt.Synchronize();
}

void ExpertsCountsSum(XRuntime &rt, at::Tensor &expertsCountsInput, at::Tensor &tokensPerEpgroup,
                      at::Tensor &expertsCountsOutput, uint32_t nRoutedExperts)
{
    XTensor _expertsCountsInput, _tokensPerEpgroup, _expertsCountsOutput;

    InitXTensor(_expertsCountsInput, expertsCountsInput);
    InitXTensor(_tokensPerEpgroup, tokensPerEpgroup);
    InitXTensor(_expertsCountsOutput, expertsCountsOutput);
    XliteOpExpertsCountsSum(rt, _expertsCountsInput, _tokensPerEpgroup, _expertsCountsOutput,
                            nRoutedExperts);
    rt.Synchronize();
}

void ReorderMoE(XRuntime &rt, at::Tensor &in, at::Tensor &out, at::Tensor &counts,
                uint32_t hiddenSize, uint32_t localStart, uint32_t localEnd, bool forward)
{
    XTensor _in, _out, _counts;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_counts, counts);
    XliteOpReorderMoE(rt, _in, _out, _counts, hiddenSize, localStart, localEnd, forward);
    rt.Synchronize();
}

void LinearAttProj(XRuntime &rt, at::Tensor &x, at::Tensor &W_qkv, at::Tensor &W_z, at::Tensor &W_b,
                   at::Tensor &W_a, at::Tensor &mix_qkv, at::Tensor &z, at::Tensor &b,
                   at::Tensor &a, uint32_t m, uint32_t n, uint32_t v, uint32_t h, uint32_t k)
{
    XTensor _x, _W_qkv, _W_z, _W_b, _W_a;
    XTensor _mix_qkv, _z, _b, _a;
    XTensor bias, deqScale;

    InitXTensor(_x, x);
    InitXTensor(_W_qkv, W_qkv);
    InitXTensor(_W_z, W_z);
    InitXTensor(_W_b, W_b);
    InitXTensor(_W_a, W_a);

    InitXTensor(_mix_qkv, mix_qkv);
    InitXTensor(_z, z);
    InitXTensor(_b, b);
    InitXTensor(_a, a);

    std::vector<XTensor> inputs = {_W_qkv, _W_z, _W_b, _W_a};
    XTensor &W = rt.GetTensor({k, n + v + h + h}, XDtype(x), DBG_LOC);
    XliteOpConcatCol(rt, inputs, W);

    XTensor &out = rt.GetTensor({m, n + v + h + h}, XDtype(x), DBG_LOC);
    XliteOpMatmul(rt, _x, W, out, false, bias, deqScale, true);
    rt.PutTensor(W);

    std::vector<XTensor> outputs = {_mix_qkv, _z, _b, _a};
    XliteOpSplitCol(rt, out, outputs);
    rt.PutTensor(out);
    rt.Synchronize();
}

void Transpose_1_2(XRuntime &rt, at::Tensor &input, at::Tensor &output)
{
    XTensor _input, _output;
    InitXTensor(_input, input);
    InitXTensor(_output, output);
    XliteOpTranspose_1_2(rt, _input, _output);
    rt.Synchronize();
}

void LinearAttConv1dAndSiLU(XRuntime &rt, at::Tensor &mix_qkv, at::Tensor &conv_state,
                            at::Tensor &weight, at::Tensor &output)
{
    XTensor _mix_qkv, _conv_state, _weight, _output;
    InitXTensor(_mix_qkv, mix_qkv);
    InitXTensor(_conv_state, conv_state);
    InitXTensor(_weight, weight);
    InitXTensor(_output, output);
    std::vector<XTensor> inputs = {_conv_state, _mix_qkv};
    XTensor &x = rt.GetTensor(
        {_mix_qkv.shape[0], _mix_qkv.shape[1], _mix_qkv.shape[2] + _conv_state.shape[2]},
        XDtype(mix_qkv), DBG_LOC);
    XliteOpConcatCol(rt, inputs, x);
    XliteOpConv1dAndSiLU(rt, x, _weight, _output);
    rt.PutTensor(x);
    rt.Synchronize();
}

void SplitCol(XRuntime &rt, at::Tensor &in, std::vector<at::Tensor> &outputs)
{
    XTensor _in;
    std::vector<XTensor> _outputs;
    InitXTensor(_in, in);
    uint32_t totalWidth = 0;
    for (auto &output : outputs) {
        XTensor x;
        InitXTensor(x, output);
        _outputs.push_back(x);
        totalWidth += x.shape.back();
    }
    if (totalWidth != _in.shape.back()) {
        throw std::runtime_error("input last dim != sum(outputs last dim)");
    }
    XliteOpSplitCol(rt, _in, _outputs);
    rt.Synchronize();
}

void BetaDecay(XRuntime &rt, at::Tensor &b, at::Tensor &a, at::Tensor &A_log, at::Tensor &dt_bias,
               at::Tensor &beta, at::Tensor &g, uint32_t bsz, uint32_t seqlen, uint32_t num_v_heads)
{
    XTensor _b, _a, _A_log, _dt_bias, _beta, _g;
    InitXTensor(_b, b);
    InitXTensor(_a, a);
    InitXTensor(_A_log, A_log);
    InitXTensor(_dt_bias, dt_bias);
    InitXTensor(_beta, beta);
    InitXTensor(_g, g);

    XliteOpBetaDecay(rt, _b, _a, _A_log, _dt_bias, _beta, _g, bsz, seqlen, num_v_heads);
    rt.Synchronize();
}

PYBIND11_MODULE(_C, m)
{
    py::class_<XRuntime>(m, "Runtime")
        .def(py::init<uint32_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("devid"), py::arg("size") = 0, py::arg("rank") = 0, py::arg("tp_size") = 1,
             py::arg("dp_size") = 1, py::arg("moe_tp_size") = 1, py::arg("moe_ep_size") = 1)
        .def_readwrite("task_id", &XRuntime::taskId)
        .def_readwrite("notify", &XRuntime::notify)
        .def_readwrite("peer_notify", &XRuntime::peerNotify)
        .def_readwrite("multi_task_parallel", &XRuntime::multiTaskParallel)
        .def("update_core_num", &XRuntime::UpdateCoreNum, py::arg("util"))
        .def("init_tensor_pool", &XRuntime::InitTensorPool, py::arg("size"))
        .def("set_current_context", &XRuntime::SetCurrentContext)
        .def("configure_swizzle", &XRuntime::ConfigureSwizzle, py::arg("swizzle"),
             py::arg("use_swizzle_table"));

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
        .def_readwrite("quant_attn_weight_transpose",
                       &XModelConfig::quantAttnWeightTrans)  // only for quantization
        .def_readwrite("quant_attn_weight_nz",
                       &XModelConfig::quantAttnWeightNz)  // only for quantization
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
        .def_readwrite("max_m", &XModelConfig::maxBatchedTokens)
        .def_readwrite("max_num_batched_tokens", &XModelConfig::maxBatchedTokens)
        .def_readwrite("block_size", &XModelConfig::blockSize)
        .def_readwrite("weight_nz", &XModelConfig::weightNZ)
        .def_readwrite("experts_weight_transpose", &XModelConfig::expertsWeightTrans)
        .def_readwrite("experts_weight_nz", &XModelConfig::expertsWeightNZ)
        .def_readwrite("gate_captured", &XModelConfig::gateCaptured)
        .def_readwrite("qkv_bias", &XModelConfig::addBias)
        .def_readwrite("qk_norm", &XModelConfig::qkNorm)
        .def_readwrite("qk_norm_full", &XModelConfig::qkNormFull)
        .def_readwrite("scoring_func", &XModelConfig::scoringFunc)
        .def_readwrite("norm_topk_prob", &XModelConfig::normTopKProb)
        .def_readwrite("mrope_section", &XModelConfig::mropeSection)
        .def_readwrite("mrope_interleaved", &XModelConfig::mropeInterleaved)
        .def_readwrite("deepstack_num_level", &XModelConfig::deepstackNumLevel)
        .def_readwrite("index_head_dim", &XModelConfig::indexHeadDim)
        .def_readwrite("index_n_heads", &XModelConfig::indexNHeads)
        .def_readwrite("index_topk", &XModelConfig::indexTopK)
        .def_readwrite("index_softmax_scale", &XModelConfig::indexSoftmaxScale)
        .def_readwrite("index_rope_interleaved", &XModelConfig::indexRopeInterleaved);

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
        .value("AttnDSA", XModelAttnType::XMODEL_ATTN_DSA)
        .export_values();

    py::enum_<XModelScoringFuncType>(m, "ScoringFuncType")
        .value("ScoringFuncSoftmax", XModelScoringFuncType::XMODEL_SCORING_FUNC_SOFTMAX)
        .value("ScoringFuncSigmoid", XModelScoringFuncType::XMODEL_SCORING_FUNC_SIGMOID)
        .export_values();

    py::class_<_CModel>(m, "Model")
        .def(py::init<>())
        .def_readwrite("embed", &_CModel::embed)
        .def_readwrite("norm", &_CModel::norm)
        .def_readwrite("norm_bias", &_CModel::normBias)
        .def_readwrite("head", &_CModel::head)
        .def_readwrite("attn_norm", &_CModel::attnNorm)
        .def_readwrite("attn_norm_bias", &_CModel::attnNormBias)
        .def_readwrite("attn_out", &_CModel::attnOut)
        .def_readwrite("attn_out_input_scale", &_CModel::attnOutInputScale)
        .def_readwrite("attn_out_input_offset", &_CModel::attnOutInputOffset)
        .def_readwrite("attn_out_quant_bias", &_CModel::attnOutQuantBias)
        .def_readwrite("attn_out_deq_scale", &_CModel::attnOutDeqScale)
        .def_readwrite("mha_qkv", &_CModel::mhaQKV)
        .def_readwrite("mha_qkv_bias", &_CModel::mhaQKVBias)
        .def_readwrite("mha_qkv_input_scale", &_CModel::mhaQKVInputScale)
        .def_readwrite("mha_qkv_input_offset", &_CModel::mhaQKVInputOffset)
        .def_readwrite("mha_qkv_quant_bias", &_CModel::mhaQKVQuantBias)
        .def_readwrite("mha_qkv_deq_scale", &_CModel::mhaQKVDeqScale)
        .def_readwrite("mha_q_norm", &_CModel::mhaQNorm)
        .def_readwrite("mha_q_norm_bias", &_CModel::mhaQNormBias)
        .def_readwrite("mha_k_norm", &_CModel::mhaKNorm)
        .def_readwrite("mha_k_norm_bias", &_CModel::mhaKNormBias)
        .def_readwrite("mla_qkv_a", &_CModel::mlaQKVA)
        .def_readwrite("mla_q_b", &_CModel::mlaQB)
        .def_readwrite("mla_q_norm", &_CModel::mlaQNorm)
        .def_readwrite("mla_q_norm_bias", &_CModel::mlaQNormBias)
        .def_readwrite("mla_kv_b", &_CModel::mlaKVB)
        .def_readwrite("mla_kv_norm", &_CModel::mlaKVNorm)
        .def_readwrite("mla_kv_norm_bias", &_CModel::mlaKVNormBias)
        .def_readwrite("mla_qkv_a_input_scale", &_CModel::mlaQKVAInputScale)
        .def_readwrite("mla_qkv_a_input_offset", &_CModel::mlaQKVAInputOffset)
        .def_readwrite("mla_qkv_a_quant_bias", &_CModel::mlaQKVAQuantBias)
        .def_readwrite("mla_qkv_a_deq_scale", &_CModel::mlaQKVADeqScale)
        .def_readwrite("mla_q_b_input_scale", &_CModel::mlaQBInputScale)
        .def_readwrite("mla_q_b_input_offset", &_CModel::mlaQBInputOffset)
        .def_readwrite("mla_q_b_quant_bias", &_CModel::mlaQBQuantBias)
        .def_readwrite("mla_q_b_deq_scale", &_CModel::mlaQBDeqScale)
        .def_readwrite("index_q_b", &_CModel::indexQB)
        .def_readwrite("index_q_b_input_scale", &_CModel::indexQBInputScale)
        .def_readwrite("index_q_b_input_offset", &_CModel::indexQBInputOffset)
        .def_readwrite("index_q_b_quant_bias", &_CModel::indexQBQuantBias)
        .def_readwrite("index_q_b_deq_scale", &_CModel::indexQBDeqScale)
        .def_readwrite("index_k_weights_proj", &_CModel::indexKWeightsProj)
        .def_readwrite("index_k_norm", &_CModel::indexKNorm)
        .def_readwrite("index_k_norm_bias", &_CModel::indexKNormBias)
        .def_readwrite("mlp_norm", &_CModel::mlpNorm)
        .def_readwrite("mlp_norm_bias", &_CModel::mlpNormBias)
        .def_readwrite("mlp_up_gate", &_CModel::mlpUpGate)
        .def_readwrite("mlp_up_gate_input_scale", &_CModel::mlpUpGateInputScale)
        .def_readwrite("mlp_up_gate_input_offset", &_CModel::mlpUpGateInputOffset)
        .def_readwrite("mlp_up_gate_quant_bias", &_CModel::mlpUpGateQuantBias)
        .def_readwrite("mlp_up_gate_deq_scale", &_CModel::mlpUpGateDeqScale)
        .def_readwrite("mlp_down", &_CModel::mlpDown)
        .def_readwrite("mlp_down_input_scale", &_CModel::mlpDownInputScale)
        .def_readwrite("mlp_down_input_offset", &_CModel::mlpDownInputOffset)
        .def_readwrite("mlp_down_quant_bias", &_CModel::mlpDownQuantBias)
        .def_readwrite("mlp_down_deq_scale", &_CModel::mlpDownDeqScale)
        .def_readwrite("gate", &_CModel::moeGate)
        .def_readwrite("gate_bias", &_CModel::moeGateBias)
        .def_readwrite("se_up_gate", &_CModel::moeSEUpGate)
        .def_readwrite("se_up_gate_deq_scale", &_CModel::moeSEUpGateDeqScale)
        .def_readwrite("se_down", &_CModel::moeSEDown)
        .def_readwrite("se_down_deq_scale", &_CModel::moeSEDownDeqScale)
        .def_readwrite("re_up_gate", &_CModel::moeREUpGate)
        .def_readwrite("re_up_gate_scale", &_CModel::moeREUpGateDeqScale)
        .def_readwrite("re_up_gate_deq_scale", &_CModel::moeREUpGateDeqScale)
        .def_readwrite("re_down", &_CModel::moeREDown)
        .def_readwrite("re_down_scale", &_CModel::moeREDownDeqScale)
        .def_readwrite("re_down_deq_scale", &_CModel::moeREDownDeqScale)
        .def("init", &_CModel::Init, "model init", py::arg("config"), py::arg("rank") = 0)
        .def("forward", &_CModel::Forward, "forward", py::arg("rt"), py::arg("input"),
             py::arg("attn_meta"), py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
        .def("forward", &_CModel::ForwardV1, "forward", py::arg("rt"), py::arg("input"),
             py::arg("attn_meta"), py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
        .def("forward_get_logits", &_CModel::ForwardGetLogits, "forward_get_logits", py::arg("rt"),
             py::arg("input"), py::arg("indices"), py::arg("output"), py::arg("curr_stream") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_and_get_logits", &_CModel::ForwardAndGetLogits, "forward_and_get_logits",
             py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
             py::arg("freqs_cis"), py::arg("indices"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
        .def("forward_and_get_logits", &_CModel::ForwardAndGetLogitsV1, "forward_and_get_logits",
             py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
             py::arg("freqs_cis"), py::arg("indices"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
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
        .def(py::init<float>(), py::arg("prefill_ratio"))
        .def("assign_core", &XCoreAssigner::AssignCore, py::arg("is_decode"),
             py::call_guard<py::gil_scoped_release>())
        .def("release_core", &XCoreAssigner::ReleaseCore, py::arg("is_decode"),
             py::call_guard<py::gil_scoped_release>());

    // kernels
    m.def("all_gather", &AllGather, py::arg("rt"), py::arg("out"), py::arg("in_"));
    m.def("reduce_scatter", &ReduceScatter, py::arg("rt"), py::arg("out"), py::arg("in_"));
    m.def("all_reduce", &AllReduce, py::arg("rt"), py::arg("out"), py::arg("in_"));
    m.def("alltoallv", &AlltoAllV, py::arg("rt"), py::arg("out"), py::arg("in_"),
          py::arg("send_counts"), py::arg("recv_counts"), py::arg("sdispls"), py::arg("rdispls"),
          py::arg("comm_type") = 0);
    m.def("add", &Add, py::arg("rt"), py::arg("x"), py::arg("y"), py::arg("z"));
    m.def("matmul", &Matmul, "matmul", py::arg("rt"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("weight_nz") = false, py::arg("transpose") = false);
    m.def("matmul_bench", &MatmulBench, py::arg("rt"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("x_warmup"), py::arg("y_warmup"), py::arg("z_warmup"), py::arg("iterations"),
          py::arg("warmup_iterations"), py::arg("weight_nz") = false, py::arg("transpose") = false);
    m.def("matmul_with_bias", &MatmulWithBias, "matmul_with_bias", py::arg("rt"), py::arg("x"),
          py::arg("y"), py::arg("z"), py::arg("bias"), py::arg("weight_nz") = false);
    m.def("embed", &Embed, py::arg("rt"), py::arg("weight"), py::arg("in_"), py::arg("out"),
          py::arg("start"), py::arg("end"));
    m.def("rmsnorm", &RMSNorm, "rmsnorm", py::arg("rt"), py::arg("in_"), py::arg("norm"),
          py::arg("out"), py::arg("norm_eps"), py::arg("norm_dim") = 0,
          py::arg("cnt_per_token") = 1, py::arg("in_start_offset") = 0,
          py::arg("out_start_offset") = 0, py::arg("use_norm") = true,
          py::arg("variance") = std::nullopt);
    m.def("rmsnorm_with_bias", &RMSNormWithBias, "rmsnorm_with_bias", py::arg("rt"), py::arg("in_"),
          py::arg("norm"), py::arg("norm_bias"), py::arg("out"), py::arg("norm_eps"),
          py::arg("norm_dim") = 0, py::arg("cnt_per_token") = 1, py::arg("in_start_offset") = 0,
          py::arg("out_start_offset") = 0);
    m.def("layernorm", &LayerNorm, py::arg("rt"), py::arg("in_"), py::arg("norm"),
          py::arg("norm_bias"), py::arg("out"), py::arg("norm_eps"), py::arg("norm_dim"));
    m.def("add_bias", &AddBias, py::arg("rt"), py::arg("in_"), py::arg("weight"), py::arg("out"));
    m.def("silu_and_mul", &SiluAndMul, py::arg("rt"), py::arg("in_"), py::arg("out"));
    m.def("rope_and_cache", &RopeAndCache, "rope_and_cache", py::arg("rt"), py::arg("inout"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("position"), py::arg("cosin"),
          py::arg("slot_mapping"), py::arg("n_heads"), py::arg("n_kv_heads"), py::arg("head_dim"),
          py::arg("rot_dim"), py::arg("block_size"), py::arg("is_neox"),
          py::arg("mrope_mask_h") = 0, py::arg("mrope_mask_w") = 0);
    m.def("attention", &Attention, py::arg("rt"), py::arg("qkv"), py::arg("k_cache"),
          py::arg("v_cache"), py::arg("output"), py::arg("query_start_loc"), py::arg("lens"),
          py::arg("cached_lens"), py::arg("block_tables"), py::arg("n_heads"),
          py::arg("n_kv_heads"), py::arg("head_dim"), py::arg("block_size"), py::arg("batch"),
          py::arg("max_num_block"), py::arg("enable_flash_attention") = false,
          py::arg("tile_size_of_cached_kv") = 8192);
    m.def("add_and_rmsnorm", &AddAndRMSNorm, py::arg("rt"), py::arg("in_"), py::arg("add_in_out"),
          py::arg("norm"), py::arg("out"), py::arg("norm_eps"));
    m.def("softmax_topk", &SoftmaxTopK, py::arg("rt"), py::arg("scores"), py::arg("indices"),
          py::arg("out_weights"), py::arg("out_routing"), py::arg("top_k"),
          py::arg("norm_top_k_prob"));
    m.def("sigmoid_topk", &SigmoidTopK, py::arg("rt"), py::arg("scores"), py::arg("indices"),
          py::arg("bias"), py::arg("scale"), py::arg("out_weights"), py::arg("out_routing"),
          py::arg("n_group"), py::arg("n_topk_group"), py::arg("top_k"),
          py::arg("norm_top_k_prob"));
    m.def("topk", &TopK, py::arg("rt"), py::arg("scores"), py::arg("indices"),
          py::arg("outIndices"), py::arg("query_lens"), py::arg("cached_lens"), py::arg("k"));
    m.def("cast_up", &CastUp, py::arg("rt"), py::arg("in_"), py::arg("out"));
    m.def("permutation", &Permutation, py::arg("rt"), py::arg("in_"), py::arg("routing"),
          py::arg("start"), py::arg("end"), py::arg("out"), py::arg("unp_idx"), py::arg("counts"));
    m.def("unpermutation", &UnPermutation, py::arg("rt"), py::arg("in_"), py::arg("routing"),
          py::arg("weights"), py::arg("start"), py::arg("end"), py::arg("out"), py::arg("unp_idx"));
    m.def("group_matmul", &GroupMatmul, py::arg("rt"), py::arg("in_"), py::arg("weights"),
          py::arg("scales"), py::arg("counts"), py::arg("start"), py::arg("end"),
          py::arg("out_dim"), py::arg("in_dim"), py::arg("output"), py::arg("weight_nz"),
          py::arg("transpose"));
    m.def("softmax", &Softmax, py::arg("rt"), py::arg("x"), py::arg("calc_len"),
          py::arg("is_long"));
    m.def("rope_complex", &RopeComplex, "rope_complex", py::arg("rt"), py::arg("n_local_heads"),
          py::arg("step_dim"), py::arg("rope_dim"), py::arg("input_with_r"), py::arg("freqs"),
          py::arg("position"));
    m.def("quant", &Quant, py::arg("rt"), py::arg("x"), py::arg("scale_reciprocal"),
          py::arg("offset"), py::arg("out"));
    m.def("quant_dynamic", &QuantDyn, py::arg("rt"), py::arg("x"), py::arg("scale"),
          py::arg("out"));
    m.def("matmul_dequant", &MatmulDeQuant, "matmul_dequant", py::arg("rt"), py::arg("x"),
          py::arg("y"), py::arg("bias"), py::arg("deq_scale"), py::arg("z"),
          py::arg("weight_nz") = false, py::arg("transpose") = false);
    m.def("dequant", &DeQuant, py::arg("rt"), py::arg("in_"), py::arg("scale"), py::arg("out"),
          py::arg("has_scale"));
    m.def("mla", &MLA, py::arg("rt"), py::arg("q_with_qr"), py::arg("k_cache"), py::arg("v_cache"),
          py::arg("wkvb"), py::arg("output"), py::arg("query_start_loc"), py::arg("lens"),
          py::arg("cached_lens"), py::arg("block_tables"), py::arg("n_heads"),
          py::arg("rope_head_dim"), py::arg("nope_head_dim"), py::arg("v_head_dim"),
          py::arg("kv_lora_rank"), py::arg("block_size"), py::arg("batch"),
          py::arg("max_num_block"), py::arg("scale"), py::arg("nz") = false,
          py::arg("enable_flash_attention") = false, py::arg("tile_size_of_cached_kv") = 8192);
    m.def("mla_with_indices", &MLAWithIndices, py::arg("rt"), py::arg("q_with_qr"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("wkvb"), py::arg("output"),
          py::arg("query_start_loc"), py::arg("lens"), py::arg("cached_lens"),
          py::arg("block_tables"), py::arg("n_heads"), py::arg("rope_head_dim"),
          py::arg("nope_head_dim"), py::arg("v_head_dim"), py::arg("kv_lora_rank"),
          py::arg("block_size"), py::arg("batch"), py::arg("max_num_block"), py::arg("scale"),
          py::arg("top_k"), py::arg("topk_indices"), py::arg("nz") = false,
          py::arg("enable_flash_attention") = false);
    m.def("indexer_scores", &IndexerScores, py::arg("rt"), py::arg("q"), py::arg("k_cache"),
          py::arg("weight"), py::arg("scores"), py::arg("query_start_loc"), py::arg("lens"),
          py::arg("cached_lens"), py::arg("block_tables"), py::arg("n_heads"), py::arg("head_dim"),
          py::arg("block_size"), py::arg("batch"), py::arg("max_num_block"));
    m.def("muls", &Muls, py::arg("rt"), py::arg("input"), py::arg("scale"), py::arg("output"));
    m.def("experts_counts_sum", &ExpertsCountsSum, py::arg("rt"), py::arg("experts_counts_input"),
          py::arg("tokens_per_epgroup"), py::arg("experts_counts_output"),
          py::arg("n_routed_experts"));
    m.def("reorder_moe", &ReorderMoE, py::arg("rt"), py::arg("in_"), py::arg("out"),
          py::arg("counts"), py::arg("hidden_size"), py::arg("local_start"), py::arg("local_end"),
          py::arg("forward"));
    m.def("linear_att_proj", &LinearAttProj, py::arg("rt"), py::arg("x"), py::arg("W_qkv"),
          py::arg("W_z"), py::arg("W_b"), py::arg("W_a"), py::arg("mix_qkv"), py::arg("z"),
          py::arg("b"), py::arg("a"), py::arg("m"), py::arg("n"), py::arg("v"), py::arg("h"),
          py::arg("k"));
    m.def("transpose_1_2", &Transpose_1_2, py::arg("rt"), py::arg("input"), py::arg("output"));
    m.def("linear_att_conv_and_silu", &LinearAttConv1dAndSiLU, py::arg("rt"), py::arg("mix_qkv"),
          py::arg("conv_state"), py::arg("weight"), py::arg("output"));
    m.def("split_col", &SplitCol, py::arg("rt"), py::arg("in"), py::arg("outputs"));
    m.def("beta_decay", &BetaDecay, py::arg("rt"), py::arg("b"), py::arg("a"), py::arg("A_log"),
          py::arg("dt_bias"), py::arg("beta"), py::arg("g"), py::arg("bsz"), py::arg("seqlen"),
          py::arg("num_v_heads"));

    // funcs
    m.def("print", &Print, "print", py::arg("x"), py::arg("name") = "", py::arg("row") = 6,
          py::arg("col") = 6);
    m.def("get_tile_size_of_cached_kv", &GetTileSizeOfCachedKV,
          "Get optimal tile size for cached KV based on workload", py::arg("cached_lens"),
          py::arg("query_lens"), py::arg("head_num_in_group"), py::arg("n_kv_heads"),
          py::arg("block_size"), py::arg("aic_num"));
}
