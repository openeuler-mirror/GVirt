/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * _C.cpp — the abi3 SABI binding module for xlite (single self-contained TU).
 *
 * Built via `Python3_add_library(... USE_SABI 3.11 WITH_SOABI)` (CMakeLists.txt),
 * producing xlite_sabi.abi3.so. Registers torch.ops.xlite_sabi.* /
 * torch.classes.xlite_sabi.* through torch's dispatcher — no direct Python
 * C-API contact — so a single wheel installs on any CPython 3.11+. The
 * pure-Python xlite/_C.py shim loads this .so and re-exports the names so
 * existing `from xlite._C import ...` call sites work unchanged.
 *
 * Layout (top to bottom):
 *   1. includes
 *   2. internal structs CModelAttnMeta / CModelAttnMetaV2 and class _CModel
 *      (inherit torch::CustomClassHolder — internal to the binding layer)
 *   3. wrapper free functions (XRuntime& + at::Tensor& -> XTensor -> XliteOp*)
 *   4. holders for public C++ classes (RuntimeHolder / ConfigHolder /
 *      CoreAssignerHolder) + TORCH_LIBRARY registration of all classes + ops
 *
 * Binding-boundary rules (verified by PoC):
 *   - integral args are int64_t (IValue maps schema "int"->int64_t; uint32_t
 *     fails a static_assert). Cast to uint32_t at the wrapper call.
 *   - vector<uint32_t> struct fields are NOT supported by IValue; exposed via
 *     opreg::Props::def_rw with int64 conversion (field type stays uint32_t
 *     in C++).
 *   - custom-class args arrive as c10::intrusive_ptr<Holder>.
 *   - at::Tensor args arrive by value; in-place outputs work (shared storage).
 *   - ops register through opreg::OpDefs: .def(name, op<&Fn>, {torch::arg...})
 *     infers argument types from the adapter signature and merges the
 *     torch::arg names/defaults into the schema (auto-inference alone names
 *     args _0, _1, ... and cannot express defaults).
 */
#include <ATen/core/op_registration/infer_schema.h>
#include <torch/library.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <cstdint>
#include <initializer_list>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "xlite.h"          // InitXTensor, XTensor
#include "runtime.h"        // XRuntime (public; wrapped, not inherited)
#include "model.h"          // XModel, XModelConfig, MatmulWeight, XModelAttnMeta
#include "dsa_cp_weight.h"  // DsaCpWeightStorageShape
#include "op.h"             // XliteOp* declarations
#include "core_assigner.h"  // XCoreAssigner (public; wrapped)
#include "auto_tuner.h"     // GetTileSizeOfCachedKV (vector<uint32_t> args)
#include "debug.h"          // DBG_LOC

struct CModelAttnMeta : torch::CustomClassHolder {
    std::vector<uint32_t> lens;
    std::vector<uint32_t> cachedLens;
    std::vector<std::vector<uint32_t>> blockTablesList;
    at::Tensor positions;
};

struct CModelAttnMetaV2 : torch::CustomClassHolder {
    std::vector<uint32_t> lensCpu;
    std::vector<uint32_t> cachedLensCpu;
    at::Tensor positions;
    at::Tensor lens;
    at::Tensor cachedLens;
    at::Tensor queryStartLoc;
    std::vector<at::Tensor> slotMapping;
    std::vector<at::Tensor> blockTables;
    // TODO: device int32 [num_seqs]; seq i reads block-table row [i] instead of row i
    std::optional<at::Tensor> blockTablesSeqMapping;
    // CXA per-layer (v2): [n_layers] rows of 5-plane references. One template
    // row per distinct ratio>0 (shared); per-layer row for ratio==0 layers.
    // See XModelAttnMeta::blockTablesPerLayer for the full layout contract.
    std::vector<std::vector<at::Tensor>> blockTablesPerLayer;  // [n_layers][5]
    std::vector<std::vector<at::Tensor>> slotMappingPerLayer;  // [n_layers][5]
};

void CheckBlockTablesSeqMapping(const CModelAttnMetaV2 &attnMeta);

class _CModel : public torch::CustomClassHolder
{
public:
    _CModel() {};
    ~_CModel() override;
    void Init(struct XModelConfig &c, uint32_t rankId);
    void ForwardV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                   std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                   at::Tensor &output, uint64_t currStream);
    void ForwardV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                   std::vector<std::vector<at::Tensor>> &kvCache, std::vector<at::Tensor> &freqsCis,
                   at::Tensor &output, uint64_t currStream);
    void ForwardGetLogits(XRuntime &rt, at::Tensor &input, at::Tensor &indices, at::Tensor &output,
                          uint64_t currStream);
    void ForwardAndGetLogitsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                               std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                               at::Tensor &indices, at::Tensor &output, uint64_t currStream);
    void ForwardAndGetLogitsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                               std::vector<std::vector<at::Tensor>> &kvCache,
                               std::vector<at::Tensor> &freqsCis, at::Tensor &indices,
                               at::Tensor &output, uint64_t currStream);
    void ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                   std::vector<std::vector<at::Tensor>> &kvCache,
                                   at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                   std::vector<at::Tensor> &deepstackInput,
                                   const std::optional<at::Tensor> &inputIds);
    void ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                   std::vector<std::vector<at::Tensor>> &kvCache,
                                   at::Tensor &freqsCis, at::Tensor &output,
                                   std::vector<at::Tensor> &auxOutputs, uint64_t currStream,
                                   std::vector<at::Tensor> &deepstackInput,
                                   const std::optional<at::Tensor> &inputIds);
    void ForwardWithInputsEmbedsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                                   std::vector<std::vector<at::Tensor>> &kvCache,
                                   at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                   std::vector<at::Tensor> &deepstackInput,
                                   const std::optional<at::Tensor> &inputIds);
    void ForwardWithInputsEmbedsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                                   std::vector<std::vector<at::Tensor>> &kvCache,
                                   at::Tensor &freqsCis, at::Tensor &output,
                                   std::vector<at::Tensor> &auxOutputs, uint64_t currStream,
                                   std::vector<at::Tensor> &deepstackInput,
                                   const std::optional<at::Tensor> &inputIds);
    void ForwardV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                   std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                   at::Tensor &output, std::vector<at::Tensor> &auxOutputs, uint64_t currStream);
    void ForwardV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                   std::vector<std::vector<at::Tensor>> &kvCache, std::vector<at::Tensor> &freqsCis,
                   at::Tensor &output, std::vector<at::Tensor> &auxOutputs, uint64_t currStream);
    size_t GetTensorPoolSize(int dbg);

    enum XModelAttnType attnType = XMODEL_ATTN_MHA;

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
    std::vector<at::Tensor> mlaWUV;
    std::vector<at::Tensor> mlaWUKT;
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

    std::vector<at::Tensor> linearInProjQKV;
    std::vector<at::Tensor> linearInProjZ;
    std::vector<at::Tensor> linearInProjB;
    std::vector<at::Tensor> linearInProjA;
    std::vector<at::Tensor> linearConv1d;
    std::vector<at::Tensor> linearALog;
    std::vector<at::Tensor> linearDtBias;
    std::vector<at::Tensor> linearNorm;
    std::vector<at::Tensor> linearOutProj;

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
    std::vector<at::Tensor> moeTid2Eid;
    std::vector<at::Tensor> moeSEUpGate;
    std::vector<at::Tensor> moeSEUpGateDeqScale;
    std::vector<at::Tensor> moeSEDown;
    std::vector<at::Tensor> moeSEDownDeqScale;
    std::vector<at::Tensor> moeSEGate;
    std::vector<at::Tensor> moeREUpGate;
    std::vector<at::Tensor> moeREUpGateDeqScale;
    std::vector<at::Tensor> moeREDown;
    std::vector<at::Tensor> moeREDownDeqScale;
    std::vector<at::Tensor> moeREUpGateScaleBias;
    std::vector<at::Tensor> moeREDownScaleBias;
    // EPLB: per-MoE-layer log2phy registered buffer (one per MoE layer, MoE-layer-major).
    // Undefined/empty for non-EPLB layers; InitOptionalXTensor skips those.
    std::vector<at::Tensor> log2phy;

    // DeepSeek-V4 (CxA)
    std::vector<at::Tensor> attnSink;
    std::vector<at::Tensor> attnWqA;
    std::vector<at::Tensor> attnWqAInputScale;
    std::vector<at::Tensor> attnWqAInputOffset;
    std::vector<at::Tensor> attnWqAQuantBias;
    std::vector<at::Tensor> attnWqADeqScale;
    std::vector<at::Tensor> attnWoA;
    std::vector<at::Tensor> attnWoB;
    std::vector<at::Tensor> attnWKv;
    std::vector<at::Tensor> attnWKvInputScale;
    std::vector<at::Tensor> attnWKvInputOffset;
    std::vector<at::Tensor> attnWKvQuantBias;
    std::vector<at::Tensor> attnWKvDeqScale;
    std::vector<at::Tensor> compApe;
    std::vector<at::Tensor> compWKv;
    std::vector<at::Tensor> compWGate;
    std::vector<at::Tensor> compNorm;
    std::vector<at::Tensor> idxWqB;
    std::vector<at::Tensor> idxWqBInputScale;
    std::vector<at::Tensor> idxWqBInputOffset;
    std::vector<at::Tensor> idxWqBQuantBias;
    std::vector<at::Tensor> idxWqBDeqScale;
    std::vector<at::Tensor> idxWeightsProj;
    std::vector<at::Tensor> idxCompApe;
    std::vector<at::Tensor> idxCompWKv;
    std::vector<at::Tensor> idxCompWGate;
    std::vector<at::Tensor> idxCompNorm;
    std::vector<at::Tensor> hcAttnFn;
    std::vector<at::Tensor> hcFfnFn;
    std::vector<at::Tensor> hcAttnBase;
    std::vector<at::Tensor> hcFfnBase;
    std::vector<at::Tensor> hcAttnScale;
    std::vector<at::Tensor> hcFfnScale;
    at::Tensor hcHeadFn;
    at::Tensor hcHeadBase;
    at::Tensor hcHeadScale;

    // GLM-5.3-Flash (GLM5). KDA layers: eight projections + per-stream conv1d + A_log/dt_bias/
    // o_norm + wo. Sparse-MLA layers reuse the mlaQKVA/mlaQB/mlaWUV/mlaWUKT/mlaQNorm/mlaKVNorm
    // slots plus the indexer group (fp32 wk / weights_proj, LayerNorm k_norm, kpool gate+ape).
    std::vector<at::Tensor> linearInProjQ;
    std::vector<at::Tensor> linearInProjK;
    std::vector<at::Tensor> linearInProjV;
    std::vector<at::Tensor> linearInProjFA;
    std::vector<at::Tensor> linearFBProj;
    std::vector<at::Tensor> linearInProjGA;
    std::vector<at::Tensor> linearGBProj;
    std::vector<at::Tensor> linearQConv1d;
    std::vector<at::Tensor> linearKConv1d;
    std::vector<at::Tensor> linearVConv1d;
    std::vector<at::Tensor> indexWK;
    std::vector<at::Tensor> indexWeightsProj;
    std::vector<at::Tensor> indexKpoolGate;
    std::vector<at::Tensor> indexKpoolApe;

private:
    XModel *_model = nullptr;
    // Original weights own contiguous views. Retain only materialized local
    // shards here when their physical blocks are not adjacent in the source.
    std::vector<at::Tensor> dsaCpDecodeWeightStorage;
    void PrepareDsaCpDecodeWeights(const XModelConfig &c, uint32_t tpRank, uint32_t layer);
    std::vector<std::vector<XTensor>> _kv;
    std::vector<XTensor> _deepstackInputEmbeds;
    void Forward(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                 std::vector<std::vector<at::Tensor>> &kvCache, std::vector<at::Tensor> &freqsCis,
                 at::Tensor &output, uint64_t currStream);
    void ForwardAndGetLogits(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                             std::vector<std::vector<at::Tensor>> &kvCache,
                             std::vector<at::Tensor> &freqsCis, at::Tensor &indices,
                             at::Tensor &output, uint64_t currStream);
    void ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::vector<at::Tensor>> &kvCache,
                                 at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                 std::vector<at::Tensor> &deepstackInput, at::Tensor &inputIds);
    void EnsureForwardHandlers(XRuntime &rt);
    void InitMatmulWeight(const std::string &name, std::vector<at::Tensor> &w,
                          std::vector<at::Tensor> &iScale, std::vector<at::Tensor> &iOffset,
                          std::vector<at::Tensor> &qBias, std::vector<at::Tensor> &dScale,
                          std::vector<MatmulWeight> &weightsXT, uint32_t currLayer,
                          bool isRowParallel, uint32_t tpRank, uint32_t layerOffset = 0);
};

static bool TensorUsable(const at::Tensor &t)
{
    return t.defined() && t.numel() > 0;
}

static void InitOptionalXTensor(XTensor &out, at::Tensor &in, DebugSrcLoc loc,
                                bool allowPitched = false)
{
    if (TensorUsable(in)) {
        InitXTensor(out, in, loc, allowPitched);
    }
}

static std::vector<uint32_t> ResolveLayerTypes(const XModelConfig &c)
{
    return DeriveLayerTypes(c);
}

static bool IsConfigLayerFullAttention(XModelConfig &c, uint32_t layer,
                                       const std::vector<uint32_t> &layerTypes)
{
    if (IsHybridAttn(c.attnType)) {
        return layerTypes[layer] == XMODEL_LAYER_ATTN_FULL;
    }
    return c.attnType == XMODEL_ATTN_MHA;
}

namespace
{
using TensorsRef = std::reference_wrapper<std::vector<at::Tensor>>;
struct LayersCheckEntry {
    TensorsRef weights;
    std::string name;
    bool allowEmpty;
    bool predicate;
};

void checkLayersDims(const std::vector<LayersCheckEntry> &table, uint32_t size, uint32_t rankId,
                     const std::string &prefix)
{
    for (const auto &entry : table) {
        const auto &w = entry.weights.get();
        if (!entry.predicate) {
            continue;
        }
        if (entry.allowEmpty && w.empty()) {
            continue;
        }

        if (w.size() != size) {
            XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
            s << "num of layers: " << w.size() << std::endl;
            std::string msg = prefix + " " + entry.name + " parameters";
            throw std::invalid_argument(msg);
        }
    }
}
}  // namespace

void _CModel::Init(struct XModelConfig &c, uint32_t rankId)
{
    uint32_t idx = 0;
    uint32_t totalPhysicalExperts = c.nRoutedExperts + c.numRedundantExperts;
    uint32_t nLocalPhysicalExperts = totalPhysicalExperts / c.moeEpSize;
    uint32_t expertsStartIdx = c.moeEpSize == 1 ? 0 : rankId / c.moeTPSize * nLocalPhysicalExperts;
    uint32_t expertsEndIdx = expertsStartIdx + nLocalPhysicalExperts;
    uint32_t numMoeLayers = c.nLayers - c.nDenseLayers;
    uint32_t nRE = numMoeLayers * nLocalPhysicalExperts;
    uint32_t tpRank = rankId % c.defTpSize;

    if (totalPhysicalExperts % c.moeEpSize != 0) {
        {
            XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
            s << "num of physical experts per expert parallel group: " << nLocalPhysicalExperts
              << std::endl;
        }
        throw std::invalid_argument(
            "num of physical experts must be divisible by moe expert parallel size");
    }

    attnType = c.attnType;

    if (c.nLayers < c.nDenseLayers) {
        {
            XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
            s << "num of layers: " << c.nLayers << ", num of dense layers: " << c.nDenseLayers
              << std::endl;
        }
        throw std::invalid_argument(
            "num of layers must be greater than or equal to num of dense layers");
    }

    if (IsHybridGdn(c.attnType)) {
        if (c.linearNumKHeads == 0 || c.linearNumVHeads == 0 || c.linearKeyHeadDim == 0 ||
            c.linearValueHeadDim == 0 || c.linearConvKernelDim == 0) {
            throw std::invalid_argument("Linear attention config is incomplete");
        }
        if (c.fullAttentionInterval == 0) {
            throw std::invalid_argument("fullAttentionInterval must be > 0 for hybrid attention");
        }
    } else if (IsHybridKda(c.attnType)) {
        if (c.linearNumKHeads == 0 || c.linearKeyHeadDim == 0 || c.linearConvKernelDim == 0) {
            throw std::invalid_argument(
                "HYBRID_KDA requires non-zero linearNumKHeads, linearKeyHeadDim and "
                "linearConvKernelDim");
        }
        if (c.hcMult == 0) {
            throw std::invalid_argument("HYBRID_KDA requires hcMult != 0 (MHC wrapping)");
        }
        if (!c.indexKpoolCompress || c.indexKpool <= 1) {
            throw std::invalid_argument(
                "HYBRID_KDA requires indexKpoolCompress=true and indexKpool>1");
        }
        if (c.indexTopK > 0 && !c.indexKpoolAlwaysSelectTail) {
            throw std::invalid_argument(
                "HYBRID_KDA kpool indexer requires indexKpoolAlwaysSelectTail=true");
        }
        if (c.blockSizes.size() < 2) {
            throw std::invalid_argument(
                "HYBRID_KDA kpool path requires blockSizes = [mainBlockSize, indexerBlockSize]");
        }
        // Compress and topk paths agree on pool-block mapping only under this identity.
        if (static_cast<uint64_t>(c.blockSizes[1]) * c.indexKpool != c.blockSizes[0]) {
            throw std::invalid_argument(
                "HYBRID_KDA kpool path requires blockSizes[1] * indexKpool == blockSizes[0] (got " +
                std::to_string(c.blockSizes[1]) + " * " + std::to_string(c.indexKpool) +
                " != " + std::to_string(c.blockSizes[0]) + ")");
        }
    } else if (c.attnType >= XMODEL_ATTN_MAX_TYPE) {
        XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
        s << "invalid attention type: " << c.attnType << std::endl;
        throw std::invalid_argument("Invalid attention type");
    }

    if (c.attnType == XMODEL_ATTN_CXA && c.compressRatios.size() < c.nLayers) {
        throw std::invalid_argument(
            "compressRatios must have at least nLayers elements for CxA attention");
    }

    bool isAttnMLA = c.attnType == XMODEL_ATTN_MLA;
    bool isAttnMHA = c.attnType == XMODEL_ATTN_MHA;
    bool isAttnDSA = c.attnType == XMODEL_ATTN_DSA;
    bool isAttnHybridGdn = IsHybridGdn(c.attnType);
    bool isAttnHybridKda = IsHybridKda(c.attnType);
    bool isAttnCxA = c.attnType == XMODEL_ATTN_CXA;
    const std::vector<LayersCheckEntry> layersTable = {
        // Tensors, Message, allowEmpty, Predicate
        // Common
        {std::ref(attnNorm), "attention norm", false, true},
        {std::ref(attnOut), "attention out", false, !isAttnCxA},
        {std::ref(mlpNorm), "mlp norm", false, true},
        {std::ref(attnNormBias), "attention norm bias", true, true},
        {std::ref(mlpNormBias), "MLP norm bias", true, true},
        // MHA
        {std::ref(attnOutInputScale), "attention out input scale", true, isAttnMHA},
        {std::ref(attnOutInputOffset), "attention out input offset", true, isAttnMHA},
        {std::ref(attnOutQuantBias), "attention out quant bias", true, isAttnMHA},
        {std::ref(attnOutDeqScale), "attention out dequant scale", true, isAttnMHA},
        {std::ref(mhaQKVInputScale), "MHA QKV input scale", true, isAttnMHA},
        {std::ref(mhaQKVInputOffset), "MHA QKV input offset", true, isAttnMHA},
        {std::ref(mhaQKVQuantBias), "MHA QKV quant bias", true, isAttnMHA},
        {std::ref(mhaQKVDeqScale), "MHA QKV dequant scale", true, isAttnMHA},
        // MLA
        {std::ref(mlaQKVA), "MLA attention QKVA", false, isAttnMLA},
        {std::ref(mlaQB), "MLA attention QB", false, isAttnMLA},
        {std::ref(mlaQNorm), "MLA attention Q Norm", false, isAttnMLA},
        {std::ref(mlaKVNorm), "MLA attention KV Norm", false, isAttnMLA},
        {std::ref(mlaWUV), "MLA WUV", false, isAttnMLA},
        {std::ref(mlaWUKT), "MLA WUK_T", false, isAttnMLA},
        {std::ref(mlaQKVAInputScale), "MLA QKVA input scale", true, isAttnMLA},
        {std::ref(mlaQKVAInputOffset), "MLA QKVA input offset", true, isAttnMLA},
        {std::ref(mlaQKVAQuantBias), "MLA QKVA quant bias", true, isAttnMLA},
        {std::ref(mlaQKVADeqScale), "MLA QKVA dequant scale", true, isAttnMLA},
        {std::ref(mlaQBInputScale), "MLA QB input scale", true, isAttnMLA},
        {std::ref(mlaQBInputOffset), "MLA QB input offset", true, isAttnMLA},
        {std::ref(mlaQBQuantBias), "MLA QB quant bias", true, isAttnMLA},
        {std::ref(mlaQBDeqScale), "MLA QB dequant scale", true, isAttnMLA},
        // MHA
        {std::ref(mhaQKV), "MHA QKV", false, isAttnMHA},
        {std::ref(mhaQKVBias), "MHA QKV bias", false, isAttnMHA && c.addBias},
        {std::ref(mhaQNorm), "MHA Q norm", false, isAttnMHA && c.qkNorm},
        {std::ref(mhaKNorm), "MHA K norm", false, isAttnMHA && c.qkNorm},
        {std::ref(mhaQNormBias), "MHA Q norm bias", true, isAttnMHA && c.qkNorm},
        {std::ref(mhaKNormBias), "MHA K norm bias", true, isAttnMHA && c.qkNorm},
        // DSA
        {std::ref(mlaQKVA), "DSA attention QKVA", false, isAttnDSA},
        {std::ref(mlaQB), "DSA attention QB", false, isAttnDSA},
        {std::ref(mlaQNorm), "DSA attention Q Norm", false, isAttnDSA},
        {std::ref(mlaKVNorm), "DSA attention KV norm", false, isAttnDSA},
        {std::ref(mlaWUV), "DSA WUV", false, isAttnDSA},
        {std::ref(mlaWUKT), "DSA WUK_T", false, isAttnDSA},
        {std::ref(mlaQNormBias), "DSA attention Q norm bias", true, isAttnDSA},
        {std::ref(mlaKVNormBias), "DSA attention KV norm bias", true, isAttnDSA},
        {std::ref(indexQB), "DSA attention index QB", false, isAttnDSA || isAttnHybridKda},
        {std::ref(indexKWeightsProj), "DSA attention index KWeightsProj", false, isAttnDSA},
        {std::ref(indexKNorm), "DSA attention index KNorm", false, isAttnDSA || isAttnHybridKda},
        {std::ref(indexKNormBias), "DSA attention index KNorm bias", false,
         isAttnDSA || isAttnHybridKda},
        {std::ref(indexQBInputScale), "DSA index QB input scale", true, isAttnDSA},
        {std::ref(indexQBInputOffset), "DSA index QB input offset", true, isAttnDSA},
        {std::ref(indexQBQuantBias), "DSA index QB quant bias", true, isAttnDSA},
        {std::ref(indexQBDeqScale), "DSA index QB dequant scale", true, isAttnDSA},
        {std::ref(mlaQKVAInputScale), "DSA QKVA input scale", true, isAttnDSA},
        {std::ref(mlaQKVAInputOffset), "DSA QKVA input offset", true, isAttnDSA},
        {std::ref(mlaQKVAQuantBias), "DSA QKVA quant bias", true, isAttnDSA},
        {std::ref(mlaQKVADeqScale), "DSA QKVA dequant scale", true, isAttnDSA},
        {std::ref(mlaQBInputScale), "DSA QB input scale", true, isAttnDSA},
        {std::ref(mlaQBInputOffset), "DSA QB input offset", true, isAttnDSA},
        {std::ref(mlaQBQuantBias), "DSA QB quant bias", true, isAttnDSA},
        {std::ref(mlaQBDeqScale), "DSA QB dequant scale", true, isAttnDSA},
        // Hybrid GDN (Qwen3.5); the linear_* member names are shared with the KDA adapter
        {std::ref(mhaQKV), "Hybrid QKV", false, isAttnHybridGdn},
        {std::ref(linearInProjQKV), "Linear QKV proj", false, isAttnHybridGdn},
        {std::ref(linearInProjZ), "Linear Z proj", false, isAttnHybridGdn},
        {std::ref(linearInProjB), "Linear B proj", false, isAttnHybridGdn || isAttnHybridKda},
        {std::ref(linearInProjA), "Linear A proj", false, isAttnHybridGdn},
        {std::ref(linearOutProj), "Linear out proj", false, isAttnHybridGdn || isAttnHybridKda},
        {std::ref(linearConv1d), "Linear conv1d", false, isAttnHybridGdn},
        {std::ref(linearALog), "Linear A log", false, isAttnHybridGdn || isAttnHybridKda},
        {std::ref(linearDtBias), "Linear Dt bias", false, isAttnHybridGdn || isAttnHybridKda},
        {std::ref(linearNorm), "Linear norm", false, isAttnHybridGdn || isAttnHybridKda},
        {std::ref(mhaQNorm), "Hybrid Q norm", false, isAttnHybridGdn && c.qkNorm},
        {std::ref(mhaKNorm), "Hybrid K norm", false, isAttnHybridGdn && c.qkNorm},
        // Hybrid KDA (GLM5: linear attention + sparse-MLA indexer); b/a_log/dt_bias/norm/wo
        // bind into the GDN-named members above, index_q_b/k_norm/k_norm_bias into the DSA ones
        {std::ref(linearInProjQ), "Linear in proj q", false, isAttnHybridKda},
        {std::ref(linearInProjK), "Linear in proj k", false, isAttnHybridKda},
        {std::ref(linearInProjV), "Linear in proj v", false, isAttnHybridKda},
        {std::ref(linearInProjFA), "Linear in proj f_a", false, isAttnHybridKda},
        {std::ref(linearFBProj), "Linear f_b proj", false, isAttnHybridKda},
        {std::ref(linearInProjGA), "Linear in proj g_a", false, isAttnHybridKda},
        {std::ref(linearGBProj), "Linear g_b proj", false, isAttnHybridKda},
        {std::ref(linearQConv1d), "Linear q conv1d", false, isAttnHybridKda},
        {std::ref(linearKConv1d), "Linear k conv1d", false, isAttnHybridKda},
        {std::ref(linearVConv1d), "Linear v conv1d", false, isAttnHybridKda},
        {std::ref(indexWK), "Index wk", false, isAttnHybridKda},
        {std::ref(indexWeightsProj), "Index weights proj", false, isAttnHybridKda},
        {std::ref(indexKpoolGate), "Index kpool gate", false, isAttnHybridKda},
        {std::ref(indexKpoolApe), "Index kpool ape", false, isAttnHybridKda},
        // CxA (DeepSeek-V4)
        {std::ref(attnSink), "v4 attn sink", false, isAttnCxA},
        {std::ref(attnWqA), "v4 attn wq_a", false, isAttnCxA},
        {std::ref(attnWqAInputScale), "v4 attn wq_a input scale", true, isAttnCxA},
        {std::ref(attnWqAInputOffset), "v4 attn wq_a input offset", true, isAttnCxA},
        {std::ref(attnWqAQuantBias), "v4 attn wq_a quant bias", true, isAttnCxA},
        {std::ref(attnWqADeqScale), "v4 attn wq_a deq scale", true, isAttnCxA},
        {std::ref(mlaQB), "DSA attention QB", false, isAttnCxA},
        {std::ref(mlaQBInputScale), "DSA QB input scale", true, isAttnCxA},
        {std::ref(mlaQBInputOffset), "DSA QB input offset", true, isAttnCxA},
        {std::ref(mlaQBQuantBias), "DSA QB quant bias", true, isAttnCxA},
        {std::ref(mlaQBDeqScale), "DSA QB dequant scale", true, isAttnCxA},
        {std::ref(mlaQNorm), "DSA attention Q Norm", false, isAttnCxA},
        {std::ref(mlaKVNorm), "DSA attention KV norm", false, isAttnCxA},
        {std::ref(attnWoA), "v4 attn wo_a", false, isAttnCxA},
        {std::ref(attnWoB), "v4 attn wo_b", false, isAttnCxA},
        {std::ref(attnWKv), "v4 attn wkv", false, isAttnCxA},
        {std::ref(attnWKvInputScale), "v4 attn wkv input scale", true, isAttnCxA},
        {std::ref(attnWKvInputOffset), "v4 attn wkv input offset", true, isAttnCxA},
        {std::ref(attnWKvQuantBias), "v4 attn wkv quant bias", true, isAttnCxA},
        {std::ref(attnWKvDeqScale), "v4 attn wkv deq scale", true, isAttnCxA},
        {std::ref(compApe), "v4 compressor ape", false, isAttnCxA},
        {std::ref(compWKv), "v4 compressor wkv", false, isAttnCxA},
        {std::ref(compWGate), "v4 compressor wgate", false, isAttnCxA},
        {std::ref(compNorm), "v4 compressor norm", false, isAttnCxA},
        {std::ref(idxWeightsProj), "v4 indexer weights_proj", false, isAttnCxA},
        {std::ref(idxWqB), "v4 indexer wq_b", false, isAttnCxA},
        {std::ref(idxWqBInputScale), "v4 indexer wq_b input scale", true, isAttnCxA},
        {std::ref(idxWqBInputOffset), "v4 indexer wq_b input offset", true, isAttnCxA},
        {std::ref(idxWqBQuantBias), "v4 indexer wq_b quant bias", true, isAttnCxA},
        {std::ref(idxWqBDeqScale), "v4 indexer wq_b deq scale", true, isAttnCxA},
        {std::ref(idxCompApe), "v4 idx compressor ape", false, isAttnCxA},
        {std::ref(idxCompWKv), "v4 idx compressor wkv", false, isAttnCxA},
        {std::ref(idxCompWGate), "v4 idx compressor wgate", false, isAttnCxA},
        {std::ref(idxCompNorm), "v4 idx compressor norm", false, isAttnCxA},
        {std::ref(hcAttnFn), "v4 hc attn fn", false, isAttnCxA},
        {std::ref(hcFfnFn), "v4 hc ffn fn", false, isAttnCxA},
        {std::ref(hcAttnBase), "v4 hc attn base", false, isAttnCxA},
        {std::ref(hcFfnBase), "v4 hc ffn base", false, isAttnCxA},
        {std::ref(hcAttnScale), "v4 hc attn scale", false, isAttnCxA},
        {std::ref(hcFfnScale), "v4 hc ffn scale", false, isAttnCxA},
    };
    checkLayersDims(layersTable, c.nLayers, rankId, "Mismatched number of layers");

    const std::vector<LayersCheckEntry> denseLayersTable = {
        {std::ref(mlpUpGate), "up gate", false, true},
        {std::ref(mlpDown), "down", false, true},
        {std::ref(mlpUpGateInputScale), "up gate input scale", true, true},
        {std::ref(mlpUpGateInputOffset), "up gate input offset", true, true},
        {std::ref(mlpUpGateQuantBias), "up gate quant bias", true, true},
        {std::ref(mlpUpGateDeqScale), "up gate deq scale", true, true},
        {std::ref(mlpDownInputScale), "down gate input scale", true, true},
        {std::ref(mlpDownInputOffset), "down gate input offset", true, true},
        {std::ref(mlpDownQuantBias), "down gate quant bias", true, true},
        {std::ref(mlpDownDeqScale), "down gate deq scale", true, true},
    };
    checkLayersDims(denseLayersTable, c.nDenseLayers, rankId, "Mismatched number of dense layers");

    const std::vector<LayersCheckEntry> moeLayersTable = {
        {std::ref(moeGate), "gate", false, true},
        {std::ref(moeGateBias), "gate bias", false, c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID},
        {std::ref(moeSEUpGate), "SE up gate", false, c.nSharedExperts != 0},
        {std::ref(moeSEDown), "SE down", false, c.nSharedExperts != 0},
        {std::ref(moeSEGate), "SE gate", true, c.nSharedExperts != 0},
        {std::ref(moeSEUpGateDeqScale), "SE up gate deq scale", true, c.nSharedExperts != 0},
        {std::ref(moeSEDownDeqScale), "SE down deq scale", true, c.nSharedExperts != 0},
    };
    checkLayersDims(moeLayersTable, numMoeLayers, rankId, "Mismatched number of moe layers");
    if (c.scoringFunc == XMODEL_SCORING_FUNC_SQRTSOFTPLUS && c.nHashLayers > 0 &&
        moeTid2Eid.size() != c.nHashLayers) {
        XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
        s << "tid2eid num of layers: " << moeTid2Eid.size() << std::endl;
        throw std::invalid_argument("Mismatched number of tid2eid parameters");
    }

    if (moeREUpGate.size() != nRE || moeREDown.size() != nRE) {
        {
            XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
            s << "num of routed experts: " << moeREUpGate.size() << std::endl;
        }
        throw std::invalid_argument(
            "Mismatched number of routed experts up gate or down parameters");
    }

    _model = new XModel(c, rankId);

    InitXTensor(_model->embed, embed, DBG_LOC);
    InitXTensor(_model->norm, norm, DBG_LOC);
    InitXTensor(_model->head, head, DBG_LOC);
    if (normBias.defined()) {
        InitXTensor(_model->normBias, normBias, DBG_LOC);
    }

    std::vector<uint32_t> layerTypes = ResolveLayerTypes(c);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        InitXTensor(_model->attnNorm[i], attnNorm[i], DBG_LOC);
        if (!attnNormBias.empty()) {
            InitXTensor(_model->attnNormBias[i], attnNormBias[i], DBG_LOC);
        }
        if (!mlpNormBias.empty()) {
            InitXTensor(_model->mlpNormBias[i], mlpNormBias[i], DBG_LOC);
        }
        InitXTensor(_model->mlpNorm[i], mlpNorm[i], DBG_LOC);

        bool isFullLayer = IsConfigLayerFullAttention(c, i, layerTypes);
        if (IsHybridGdn(c.attnType)) {
            if (isFullLayer) {
                InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                                 attnOutQuantBias, attnOutDeqScale, _model->attnOut, i, true,
                                 tpRank);
                InitMatmulWeight("mhaQKV", mhaQKV, mhaQKVInputScale, mhaQKVInputOffset,
                                 mhaQKVQuantBias, mhaQKVDeqScale, _model->mhaQKV, i, false, tpRank);
                if (c.addBias) {
                    InitOptionalXTensor(_model->mhaQKVBias[i], mhaQKVBias[i], DBG_LOC);
                }
                if (c.qkNorm) {
                    InitOptionalXTensor(_model->mhaQNorm[i], mhaQNorm[i], DBG_LOC);
                    InitOptionalXTensor(_model->mhaKNorm[i], mhaKNorm[i], DBG_LOC);
                    if (!mhaQNormBias.empty()) {
                        InitOptionalXTensor(_model->mhaQNormBias[i], mhaQNormBias[i], DBG_LOC);
                    }
                    if (!mhaKNormBias.empty()) {
                        InitOptionalXTensor(_model->mhaKNormBias[i], mhaKNormBias[i], DBG_LOC);
                    }
                }
            } else {
                InitMatmulWeight("linearOutProj", linearOutProj, attnOutInputScale,
                                 attnOutInputOffset, attnOutQuantBias, attnOutDeqScale,
                                 _model->linearOutProj, i, true, tpRank);
                InitMatmulWeight("linearInProjQKV", linearInProjQKV, mhaQKVInputScale,
                                 mhaQKVInputOffset, mhaQKVQuantBias, mhaQKVDeqScale,
                                 _model->linearInProjQKV, i, false, tpRank);
                InitMatmulWeight("linearInProjZ", linearInProjZ, mhaQKVInputScale,
                                 mhaQKVInputOffset, mhaQKVQuantBias, mhaQKVDeqScale,
                                 _model->linearInProjZ, i, false, tpRank);
                InitMatmulWeight("linearInProjB", linearInProjB, mhaQKVInputScale,
                                 mhaQKVInputOffset, mhaQKVQuantBias, mhaQKVDeqScale,
                                 _model->linearInProjB, i, false, tpRank);
                InitMatmulWeight("linearInProjA", linearInProjA, mhaQKVInputScale,
                                 mhaQKVInputOffset, mhaQKVQuantBias, mhaQKVDeqScale,
                                 _model->linearInProjA, i, false, tpRank);
                InitOptionalXTensor(_model->linearConv1d[i], linearConv1d[i], DBG_LOC);
                InitOptionalXTensor(_model->linearALog[i], linearALog[i], DBG_LOC);
                InitOptionalXTensor(_model->linearDtBias[i], linearDtBias[i], DBG_LOC);
                InitOptionalXTensor(_model->linearNorm[i], linearNorm[i], DBG_LOC);
            }
        } else {
            if (c.attnType == XMODEL_ATTN_MLA) {
                InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                                 attnOutQuantBias, attnOutDeqScale, _model->attnOut, i, true,
                                 tpRank);
                InitMatmulWeight("mlaQKVA", mlaQKVA, mlaQKVAInputScale, mlaQKVAInputOffset,
                                 mlaQKVAQuantBias, mlaQKVADeqScale, _model->mlaQKVA, i, false,
                                 tpRank);
                InitMatmulWeight("mlaQB", mlaQB, mlaQBInputScale, mlaQBInputOffset, mlaQBQuantBias,
                                 mlaQBDeqScale, _model->mlaQB, i, false, tpRank);
                InitXTensor(_model->mlaQNorm[i], mlaQNorm[i], DBG_LOC);
                if (!mlaQNormBias.empty()) {
                    InitXTensor(_model->mlaQNormBias[i], mlaQNormBias[i], DBG_LOC);
                }
                InitXTensor(_model->mlaWUV[i], mlaWUV[i], DBG_LOC);
                InitXTensor(_model->mlaWUKT[i], mlaWUKT[i], DBG_LOC);
                InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i], DBG_LOC);
                if (!mlaKVNormBias.empty()) {
                    InitXTensor(_model->mlaKVNormBias[i], mlaKVNormBias[i], DBG_LOC);
                }
            } else if (c.attnType == XMODEL_ATTN_MHA) {
                InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                                 attnOutQuantBias, attnOutDeqScale, _model->attnOut, i, true,
                                 tpRank);
                InitMatmulWeight("mhaQKV", mhaQKV, mhaQKVInputScale, mhaQKVInputOffset,
                                 mhaQKVQuantBias, mhaQKVDeqScale, _model->mhaQKV, i, false, tpRank);
                if (c.addBias) {
                    InitXTensor(_model->mhaQKVBias[i], mhaQKVBias[i], DBG_LOC);
                }
                if (c.qkNorm) {
                    InitXTensor(_model->mhaQNorm[i], mhaQNorm[i], DBG_LOC);
                    InitXTensor(_model->mhaKNorm[i], mhaKNorm[i], DBG_LOC);
                    if (!mhaQNormBias.empty()) {
                        InitXTensor(_model->mhaQNormBias[i], mhaQNormBias[i], DBG_LOC);
                    }
                    if (!mhaKNormBias.empty()) {
                        InitXTensor(_model->mhaKNormBias[i], mhaKNormBias[i], DBG_LOC);
                    }
                }
            } else if (c.attnType == XMODEL_ATTN_DSA) {
                InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                                 attnOutQuantBias, attnOutDeqScale, _model->attnOut, i,
                                 !c.enableDsaCp, tpRank);
                InitMatmulWeight("mlaQKVA", mlaQKVA, mlaQKVAInputScale, mlaQKVAInputOffset,
                                 mlaQKVAQuantBias, mlaQKVADeqScale, _model->mlaQKVA, i, false,
                                 tpRank);
                InitMatmulWeight("mlaQB", mlaQB, mlaQBInputScale, mlaQBInputOffset, mlaQBQuantBias,
                                 mlaQBDeqScale, _model->mlaQB, i, false, tpRank);
                InitXTensor(_model->mlaQNorm[i], mlaQNorm[i], DBG_LOC);
                if (!mlaQNormBias.empty()) {
                    InitXTensor(_model->mlaQNormBias[i], mlaQNormBias[i], DBG_LOC);
                }
                InitXTensor(_model->mlaWUV[i], mlaWUV[i], DBG_LOC);
                InitXTensor(_model->mlaWUKT[i], mlaWUKT[i], DBG_LOC);
                InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i], DBG_LOC);
                if (!mlaKVNormBias.empty()) {
                    InitXTensor(_model->mlaKVNormBias[i], mlaKVNormBias[i], DBG_LOC);
                }
                if (c.enableDsaCp) {
                    PrepareDsaCpDecodeWeights(c, tpRank, i);
                }
                // DSA top-k sharing: skip indexer weight binding on shared layers.
                if (i >= c.indexFullMask.size() || c.indexFullMask[i]) {
                    InitMatmulWeight("indexQB", indexQB, indexQBInputScale, indexQBInputOffset,
                                     indexQBQuantBias, indexQBDeqScale, _model->indexQB, i, false,
                                     tpRank);
                    InitXTensor(_model->indexKWeightsProj[i], indexKWeightsProj[i], DBG_LOC);
                    InitXTensor(_model->indexKNorm[i], indexKNorm[i], DBG_LOC);
                    InitXTensor(_model->indexKNormBias[i], indexKNormBias[i], DBG_LOC);
                }
            } else if (c.attnType == XMODEL_ATTN_CXA) {
                InitMatmulWeight("attnWqA", attnWqA, attnWqAInputScale, attnWqAInputOffset,
                                 attnWqAQuantBias, attnWqADeqScale, _model->attnWqA, i, false,
                                 tpRank);
                InitMatmulWeight("attnWKv", attnWKv, attnWKvInputScale, attnWKvInputOffset,
                                 attnWKvQuantBias, attnWKvDeqScale, _model->attnWKv, i, false,
                                 tpRank);
                // wq_b reuses MLA's mlaQB field (same structure as v3 MLA).
                InitMatmulWeight("mlaQB", mlaQB, mlaQBInputScale, mlaQBInputOffset, mlaQBQuantBias,
                                 mlaQBDeqScale, _model->mlaQB, i, false, tpRank);
                // q_norm/kv_norm reuse MLA's mlaQNorm/mlaKVNorm fields.
                InitXTensor(_model->mlaQNorm[i], mlaQNorm[i], DBG_LOC);
                if (!mlaQNormBias.empty()) {
                    InitXTensor(_model->mlaQNormBias[i], mlaQNormBias[i], DBG_LOC);
                }
                InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i], DBG_LOC);
                if (!mlaKVNormBias.empty()) {
                    InitXTensor(_model->mlaKVNormBias[i], mlaKVNormBias[i], DBG_LOC);
                }
                // v4 attention output projection (replaces v3 attnOut).
                InitXTensor(_model->attnWoA[i], attnWoA[i], DBG_LOC);
                InitXTensor(_model->attnWoB[i], attnWoB[i], DBG_LOC);
                InitXTensor(_model->attnSink[i], attnSink[i], DBG_LOC);
                // Compressor (per-layer; may be empty when compress_ratio == 0).
                InitOptionalXTensor(_model->compWKv[i], compWKv[i], DBG_LOC);
                InitOptionalXTensor(_model->compWGate[i], compWGate[i], DBG_LOC);
                InitOptionalXTensor(_model->compApe[i], compApe[i], DBG_LOC);
                InitOptionalXTensor(_model->compNorm[i], compNorm[i], DBG_LOC);
                // Indexer's weights (only on compress_ratio==4 layers)
                if (c.compressRatios[i] == 4) {
                    InitMatmulWeight("idxWqB", idxWqB, idxWqBInputScale, idxWqBInputOffset,
                                     idxWqBQuantBias, idxWqBDeqScale, _model->idxWqB, i, false,
                                     tpRank);
                    InitXTensor(_model->idxWeightsProj[i], idxWeightsProj[i], DBG_LOC);
                    InitXTensor(_model->idxCompWKv[i], idxCompWKv[i], DBG_LOC);
                    InitXTensor(_model->idxCompWGate[i], idxCompWGate[i], DBG_LOC);
                    InitXTensor(_model->idxCompApe[i], idxCompApe[i], DBG_LOC);
                    InitXTensor(_model->idxCompNorm[i], idxCompNorm[i], DBG_LOC);
                }
                // MHC (per-layer).
                InitXTensor(_model->hcAttnFn[i], hcAttnFn[i], DBG_LOC);
                InitXTensor(_model->hcFfnFn[i], hcFfnFn[i], DBG_LOC);
                InitXTensor(_model->hcAttnBase[i], hcAttnBase[i], DBG_LOC);
                InitXTensor(_model->hcFfnBase[i], hcFfnBase[i], DBG_LOC);
                InitXTensor(_model->hcAttnScale[i], hcAttnScale[i], DBG_LOC);
                InitXTensor(_model->hcFfnScale[i], hcFfnScale[i], DBG_LOC);
            } else if (c.attnType == XMODEL_ATTN_HYBRID_KDA) {
                // GLM5 weights are BF16/FP32 (unquantized): InitMatmulWeight skips empty
                // quant vectors, so all quant args share this local empty vector.
                std::vector<at::Tensor> noQuant;
                const bool isKda = layerTypes[i] == XMODEL_LAYER_ATTN_LINEAR;
                if (isKda) {
                    // KDA layer: q/k/v/b column-parallel; f_a/g_a replicated (full head_dim,
                    // see the python row slices); f_b/g_b consume the full f_a/g_a (head-shard
                    // output); wo reduces to full hidden.
                    InitMatmulWeight("linearInProjQ", linearInProjQ, noQuant, noQuant, noQuant,
                                     noQuant, _model->linearInProjQ, i, false, tpRank);
                    InitMatmulWeight("linearInProjK", linearInProjK, noQuant, noQuant, noQuant,
                                     noQuant, _model->linearInProjK, i, false, tpRank);
                    InitMatmulWeight("linearInProjV", linearInProjV, noQuant, noQuant, noQuant,
                                     noQuant, _model->linearInProjV, i, false, tpRank);
                    InitMatmulWeight("linearInProjB", linearInProjB, noQuant, noQuant, noQuant,
                                     noQuant, _model->linearInProjB, i, false, tpRank);
                    InitMatmulWeight("linearInProjFA", linearInProjFA, noQuant, noQuant, noQuant,
                                     noQuant, _model->linearInProjFA, i, false, tpRank);
                    InitMatmulWeight("linearFBProj", linearFBProj, noQuant, noQuant, noQuant,
                                     noQuant, _model->linearFBProj, i, false, tpRank);
                    InitMatmulWeight("linearInProjGA", linearInProjGA, noQuant, noQuant, noQuant,
                                     noQuant, _model->linearInProjGA, i, false, tpRank);
                    InitMatmulWeight("linearGBProj", linearGBProj, noQuant, noQuant, noQuant,
                                     noQuant, _model->linearGBProj, i, false, tpRank);
                    // Per-stream causal conv1d + recurrence scalars + gated RMSNorm.
                    InitXTensor(_model->linearQConv1d[i], linearQConv1d[i], DBG_LOC);
                    InitXTensor(_model->linearKConv1d[i], linearKConv1d[i], DBG_LOC);
                    InitXTensor(_model->linearVConv1d[i], linearVConv1d[i], DBG_LOC);
                    InitXTensor(_model->linearALog[i], linearALog[i], DBG_LOC);
                    InitXTensor(_model->linearDtBias[i], linearDtBias[i], DBG_LOC);
                    InitXTensor(_model->linearNorm[i], linearNorm[i], DBG_LOC);
                    InitMatmulWeight("linearOutProj", linearOutProj, noQuant, noQuant, noQuant,
                                     noQuant, _model->linearOutProj, i, true, tpRank);
                } else {
                    // Sparse-MLA layer: reuse the MLA weight slots (same bindings as the MLA
                    // branch) + the GLM5 indexer group.
                    InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                                     attnOutQuantBias, attnOutDeqScale, _model->attnOut, i, true,
                                     tpRank);
                    InitMatmulWeight("mlaQKVA", mlaQKVA, mlaQKVAInputScale, mlaQKVAInputOffset,
                                     mlaQKVAQuantBias, mlaQKVADeqScale, _model->mlaQKVA, i, false,
                                     tpRank);
                    InitMatmulWeight("mlaQB", mlaQB, mlaQBInputScale, mlaQBInputOffset,
                                     mlaQBQuantBias, mlaQBDeqScale, _model->mlaQB, i, false,
                                     tpRank);
                    InitXTensor(_model->mlaQNorm[i], mlaQNorm[i], DBG_LOC);
                    if (!mlaQNormBias.empty()) {
                        InitXTensor(_model->mlaQNormBias[i], mlaQNormBias[i], DBG_LOC);
                    }
                    InitXTensor(_model->mlaWUV[i], mlaWUV[i], DBG_LOC);
                    InitXTensor(_model->mlaWUKT[i], mlaWUKT[i], DBG_LOC);
                    InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i], DBG_LOC);
                    if (!mlaKVNormBias.empty()) {
                        InitXTensor(_model->mlaKVNormBias[i], mlaKVNormBias[i], DBG_LOC);
                    }
                    // Indexer: wq_b matmul; wk / weights_proj are FP32 (replicated); LayerNorm
                    // k_norm; kpool gate+ape.
                    InitMatmulWeight("indexQB", indexQB, noQuant, noQuant, noQuant, noQuant,
                                     _model->indexQB, i, false, tpRank);
                    InitXTensor(_model->indexWK[i], indexWK[i], DBG_LOC);
                    InitXTensor(_model->indexWeightsProj[i], indexWeightsProj[i], DBG_LOC);
                    InitXTensor(_model->indexKNorm[i], indexKNorm[i], DBG_LOC);
                    InitXTensor(_model->indexKNormBias[i], indexKNormBias[i], DBG_LOC);
                    InitXTensor(_model->indexKpoolGate[i], indexKpoolGate[i], DBG_LOC);
                    InitXTensor(_model->indexKpoolApe[i], indexKpoolApe[i], DBG_LOC);
                }
                // MHC (per-layer; GLM5 uses the shared MHC pre/post pipeline).
                InitXTensor(_model->hcAttnFn[i], hcAttnFn[i], DBG_LOC);
                InitXTensor(_model->hcFfnFn[i], hcFfnFn[i], DBG_LOC);
                InitXTensor(_model->hcAttnBase[i], hcAttnBase[i], DBG_LOC);
                InitXTensor(_model->hcFfnBase[i], hcFfnBase[i], DBG_LOC);
                InitXTensor(_model->hcAttnScale[i], hcAttnScale[i], DBG_LOC);
                InitXTensor(_model->hcFfnScale[i], hcFfnScale[i], DBG_LOC);
            }
        }
    }

    for (uint32_t i = 0; i < c.nDenseLayers; i++) {
        InitMatmulWeight("mlpUpGate", mlpUpGate, mlpUpGateInputScale, mlpUpGateInputOffset,
                         mlpUpGateQuantBias, mlpUpGateDeqScale, _model->mlpUpGate, i, false,
                         tpRank);
        InitMatmulWeight("mlpDown", mlpDown, mlpDownInputScale, mlpDownInputOffset,
                         mlpDownQuantBias, mlpDownDeqScale, _model->mlpDown, i, true, tpRank);
    }

    // EPLB (numRedundantExperts>0) requires a log2phy map for every MoE layer.
    if (c.numRedundantExperts > 0 && log2phy.size() < numMoeLayers) {
        throw std::invalid_argument(
            "EPLB active (num_redundant_experts>0) but log2phy not provided for all MoE layers");
    }

    for (uint32_t i = c.nDenseLayers; i < c.nLayers; i++) {
        InitXTensor(_model->moeGate[i], moeGate[i - c.nDenseLayers], DBG_LOC);
        if (i - c.nDenseLayers < log2phy.size()) {
            InitOptionalXTensor(_model->log2phy[i], log2phy[i - c.nDenseLayers], DBG_LOC);
        }
        if (c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
            InitXTensor(_model->moeGateBias[i], moeGateBias[i - c.nDenseLayers], DBG_LOC);
        }
        if (c.scoringFunc == XMODEL_SCORING_FUNC_SQRTSOFTPLUS) {
            InitOptionalXTensor(_model->moeGateBias[i], moeGateBias[i - c.nDenseLayers], DBG_LOC);
        }
        if (c.scoringFunc == XMODEL_SCORING_FUNC_SQRTSOFTPLUS &&
            i < c.nDenseLayers + c.nHashLayers) {
            InitOptionalXTensor(_model->moeTid2Eid[i - c.nDenseLayers],
                                moeTid2Eid[i - c.nDenseLayers], DBG_LOC);
        }
        std::vector<at::Tensor> emptyWeights = {};
        if (c.nSharedExperts != 0) {
            InitMatmulWeight("moeSEUpGate", moeSEUpGate, emptyWeights, emptyWeights, emptyWeights,
                             moeSEUpGateDeqScale, _model->moeSEUpGate, i, false, tpRank,
                             c.nDenseLayers);
            InitMatmulWeight("moeSEDown", moeSEDown, emptyWeights, emptyWeights, emptyWeights,
                             moeSEDownDeqScale, _model->moeSEDown, i, true, tpRank, c.nDenseLayers);
            uint32_t seIdx = i - c.nDenseLayers;
            if (seIdx < moeSEGate.size()) {
                InitOptionalXTensor(_model->moeSEGate[i], moeSEGate[seIdx], DBG_LOC);
            }
        }

        for (uint32_t j = expertsStartIdx; j < expertsEndIdx; j++) {
            InitXTensor(_model->moeREUpGate[i][j], moeREUpGate[idx], DBG_LOC);
            InitXTensor(_model->moeREDown[i][j], moeREDown[idx], DBG_LOC);
            // for w4a8 quantization: weights will arrive as int4pack (int32 dtype)
            if (_model->moeREUpGate[i][j].dtype == INT32) {
                _model->moeREUpGate[i][j].View(INT4);
            }
            if (_model->moeREDown[i][j].dtype == INT32) {
                _model->moeREDown[i][j].View(INT4);
            }
            if (!moeREUpGateDeqScale.empty()) {
                InitXTensor(_model->moeREUpGateDeqScale[i][j], moeREUpGateDeqScale[idx], DBG_LOC);
            }
            if (!moeREDownDeqScale.empty()) {
                InitXTensor(_model->moeREDownDeqScale[i][j], moeREDownDeqScale[idx], DBG_LOC);
            }
            if (!moeREUpGateScaleBias.empty()) {
                InitXTensor(_model->moeREUpGateScaleBias[i][j], moeREUpGateScaleBias[idx], DBG_LOC);
            }
            if (!moeREDownScaleBias.empty()) {
                InitXTensor(_model->moeREDownScaleBias[i][j], moeREDownScaleBias[idx], DBG_LOC);
            }
            idx++;
        }
    }

    if (c.attnType == XMODEL_ATTN_CXA) {
        InitXTensor(_model->hcHeadFn, hcHeadFn, DBG_LOC);
        InitXTensor(_model->hcHeadBase, hcHeadBase, DBG_LOC);
        InitXTensor(_model->hcHeadScale, hcHeadScale, DBG_LOC);
    }

    _model->Init();

    if (rankId % c.defTpSize == 0) {
        XDebugStream s(rankId, "");
        s << "Euler Xlite Model Inited! [tensor paralled(" << c.defTpSize << "), data parallel("
          << c.defDpSize << "), expert parallel(" << c.moeEpSize << ")]" << std::endl;
    }

    _kv.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        if (c.attnType == XMODEL_ATTN_HYBRID_KDA) {
            // GLM-5.3-Flash hybrid: KDA layers carry (conv, ssm) state, sparse MLA
            // layers carry (k, pe, indexK, tail) caches.
            const bool isKda = layerTypes[i] == XMODEL_LAYER_ATTN_LINEAR;
            _kv[i].resize(isKda ? 2 : 4);
        } else if (c.attnType == XMODEL_ATTN_DSA) {
            _kv[i].resize(i < c.indexC8Mask.size() && c.indexC8Mask[i] ? 4 : 3);
        } else if (c.attnType == XMODEL_ATTN_CXA) {
            // LI-C8 layers carry vllm's fp16 indexer-k scale pool as a 6th slot.
            _kv[i].resize(i < c.indexC8Mask.size() && c.indexC8Mask[i] ? 6 : 5);
        } else {
            _kv[i].resize(2);
        }
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

void _CModel::EnsureForwardHandlers(XRuntime &rt)
{
    if (!rt.ForwardTaskHandlersReady()) {
        _model->RegisterForwardHandlers(rt);
    }
}

void _CModel::Forward(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                      std::vector<std::vector<at::Tensor>> &kvCache,
                      std::vector<at::Tensor> &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _output;
    std::vector<XTensor> _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    _freqsCis.resize(freqsCis.size());
    for (size_t i = 0; i < freqsCis.size(); i++) {
        InitXTensor(_freqsCis[i], freqsCis[i], DBG_LOC);
    }

    if (kvCache.size() < _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    if (input.size(0) > output.size(0)) {
        throw std::runtime_error(std::string(__func__) + ": input's size 0 > output's size 0");
    }

    if (input.size(0) == 0) {
        return;  // for DP dummy run with MoE, add a padding token to avoid empty input
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        if (kvCache[i].size() != _kv[i].size()) {
            throw std::runtime_error(std::string(__func__) + ": check kv cache failed at layer " +
                                     std::to_string(i));
        }
        // allowPitch: cache consumers that address a dim-0 pitch (hybrid GDN
        // state, MHA/GQA k/v, MLA k/pe, CXA, DSA). CXA caches are paged-addressed;
        // LI-C8 writers/readers consume stride[0] directly (K + scales packed
        // page-strided in one alloc).
        bool allowPitch = IsHybridAttn(attnType) || attnType == XMODEL_ATTN_MHA ||
                          attnType == XMODEL_ATTN_MLA || attnType == XMODEL_ATTN_CXA ||
                          attnType == XMODEL_ATTN_DSA;
        for (uint64_t j = 0; j < _kv[i].size(); j++) {
            InitXTensor(_kv[i][j], kvCache[i][j], DBG_LOC, /*allowPitched=*/allowPitch);
        }
    }

    bool async = rt.ForwardThreadEnabled() && !rt.multiTaskParallel && currStream != 0;
    if (async) {
        ForwardTask task;
        task.kind = ForwardTask::FORWARD;
        task.input = _input;
        task.output = _output;
        task.attnMeta = attnMeta;
        task.kvCache = _kv;
        task.deepstackInputEmbeds = _deepstackInputEmbeds;
        task.freqsCis = _freqsCis;
        task.auxOutputs = _model->StealAuxOutputs();
        EnsureForwardHandlers(rt);
        task.seq = rt.AllocFwdSeq();
        rt.RecordInputEvent(reinterpret_cast<aclrtStream>(currStream), task.seq);
        rt.WaitOutputEvent(reinterpret_cast<aclrtStream>(currStream), task.seq);
        rt.SubmitForward(std::move(task));
        return;
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
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lens;
    _attnMeta.cachedLensCpu = attnMeta.cachedLens;
    _attnMeta.blockTablesCpu = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.position, attnMeta.positions, DBG_LOC, /*allowPitched=*/true);
    std::vector<at::Tensor> freqsCisVec = {freqsCis};
    Forward(rt, input, _attnMeta, kvCache, freqsCisVec, output, currStream);
}

// Copy the CXA per-layer (block_tables/slot_mapping) vectors from the
// Python-facing view (at::Tensor) to the runtime view (XTensor). Shared by all
// ForwardV2 variants. No-op when the host did not populate them (V1, non-CXA,
// or DummyRun) — CXA consumers then fall back to the flat
// blockTables/slotMapping.
static void InitAttnMetaV2PerLayer(XModelAttnMeta &dst, const CModelAttnMetaV2 &src)
{
    dst.blockTablesPerLayer.resize(src.blockTablesPerLayer.size());
    for (size_t l = 0; l < src.blockTablesPerLayer.size(); l++) {
        dst.blockTablesPerLayer[l].resize(src.blockTablesPerLayer[l].size());
        for (size_t p = 0; p < src.blockTablesPerLayer[l].size(); p++) {
            InitXTensor(dst.blockTablesPerLayer[l][p], src.blockTablesPerLayer[l][p]);
        }
    }
    dst.slotMappingPerLayer.resize(src.slotMappingPerLayer.size());
    for (size_t l = 0; l < src.slotMappingPerLayer.size(); l++) {
        dst.slotMappingPerLayer[l].resize(src.slotMappingPerLayer[l].size());
        for (size_t p = 0; p < src.slotMappingPerLayer[l].size(); p++) {
            InitXTensor(dst.slotMappingPerLayer[l][p], src.slotMappingPerLayer[l][p]);
        }
    }
}

void CheckBlockTablesSeqMapping(const CModelAttnMetaV2 &attnMeta)
{
    if (attnMeta.blockTablesSeqMapping.has_value()) {
        throw std::runtime_error(
            "AttnMetaV2.block_tables_seq_mapping is not supported yet, expect None");
    }
}

void _CModel::ForwardV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                        std::vector<std::vector<at::Tensor>> &kvCache,
                        std::vector<at::Tensor> &freqsCis, at::Tensor &output, uint64_t currStream)
{
    CheckBlockTablesSeqMapping(attnMeta);
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 2;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lensCpu;
    _attnMeta.cachedLensCpu = attnMeta.cachedLensCpu;
    InitXTensor(_attnMeta.position, attnMeta.positions, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_attnMeta.lens, attnMeta.lens, DBG_LOC);
    InitXTensor(_attnMeta.cachedLens, attnMeta.cachedLens, DBG_LOC);
    InitXTensor(_attnMeta.queryStartLoc, attnMeta.queryStartLoc, DBG_LOC);
    _attnMeta.slotMapping.resize(attnMeta.slotMapping.size());
    for (size_t i = 0; i < attnMeta.slotMapping.size(); i++) {
        InitXTensor(_attnMeta.slotMapping[i], attnMeta.slotMapping[i], DBG_LOC);
    }
    _attnMeta.blockTables.resize(attnMeta.blockTables.size());
    for (size_t i = 0; i < attnMeta.blockTables.size(); i++) {
        InitXTensor(_attnMeta.blockTables[i], attnMeta.blockTables[i], DBG_LOC);
    }
    InitAttnMetaV2PerLayer(_attnMeta, attnMeta);
    Forward(rt, input, _attnMeta, kvCache, freqsCis, output, currStream);
}

void _CModel::ForwardAndGetLogitsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                                    std::vector<std::vector<at::Tensor>> &kvCache,
                                    std::vector<at::Tensor> &freqsCis, at::Tensor &indices,
                                    at::Tensor &output, uint64_t currStream)
{
    CheckBlockTablesSeqMapping(attnMeta);
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 2;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lensCpu;
    _attnMeta.cachedLensCpu = attnMeta.cachedLensCpu;
    InitXTensor(_attnMeta.position, attnMeta.positions, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_attnMeta.lens, attnMeta.lens, DBG_LOC);
    InitXTensor(_attnMeta.cachedLens, attnMeta.cachedLens, DBG_LOC);
    InitXTensor(_attnMeta.queryStartLoc, attnMeta.queryStartLoc, DBG_LOC);
    _attnMeta.slotMapping.resize(attnMeta.slotMapping.size());
    for (size_t i = 0; i < attnMeta.slotMapping.size(); i++) {
        InitXTensor(_attnMeta.slotMapping[i], attnMeta.slotMapping[i], DBG_LOC);
    }
    _attnMeta.blockTables.resize(attnMeta.blockTables.size());
    for (size_t i = 0; i < attnMeta.blockTables.size(); i++) {
        InitXTensor(_attnMeta.blockTables[i], attnMeta.blockTables[i], DBG_LOC);
    }
    InitAttnMetaV2PerLayer(_attnMeta, attnMeta);
    ForwardAndGetLogits(rt, input, _attnMeta, kvCache, freqsCis, indices, output, currStream);
}

void _CModel::ForwardWithInputsEmbedsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                                        std::vector<std::vector<at::Tensor>> &kvCache,
                                        at::Tensor &freqsCis, at::Tensor &output,
                                        uint64_t currStream,
                                        std::vector<at::Tensor> &deepstackInput,
                                        const std::optional<at::Tensor> &inputIds)
{
    CheckBlockTablesSeqMapping(attnMeta);
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 2;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lensCpu;
    _attnMeta.cachedLensCpu = attnMeta.cachedLensCpu;
    InitXTensor(_attnMeta.position, attnMeta.positions, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_attnMeta.lens, attnMeta.lens, DBG_LOC);
    InitXTensor(_attnMeta.cachedLens, attnMeta.cachedLens, DBG_LOC);
    InitXTensor(_attnMeta.queryStartLoc, attnMeta.queryStartLoc, DBG_LOC);
    _attnMeta.slotMapping.resize(attnMeta.slotMapping.size());
    for (size_t i = 0; i < attnMeta.slotMapping.size(); i++) {
        InitXTensor(_attnMeta.slotMapping[i], attnMeta.slotMapping[i], DBG_LOC);
    }
    _attnMeta.blockTables.resize(attnMeta.blockTables.size());
    for (size_t i = 0; i < attnMeta.blockTables.size(); i++) {
        InitXTensor(_attnMeta.blockTables[i], attnMeta.blockTables[i], DBG_LOC);
    }
    InitAttnMetaV2PerLayer(_attnMeta, attnMeta);
    at::Tensor ids = inputIds.value_or(at::Tensor());
    ForwardWithInputsEmbeds(rt, input, _attnMeta, kvCache, freqsCis, output, currStream,
                            deepstackInput, ids);
}

void _CModel::ForwardWithInputsEmbedsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                                        std::vector<std::vector<at::Tensor>> &kvCache,
                                        at::Tensor &freqsCis, at::Tensor &output,
                                        std::vector<at::Tensor> &auxOutputs, uint64_t currStream,
                                        std::vector<at::Tensor> &deepstackInput,
                                        const std::optional<at::Tensor> &inputIds)
{
    std::vector<XTensor> auxXs(auxOutputs.size());
    for (size_t i = 0; i < auxOutputs.size(); i++) {
        InitXTensor(auxXs[i], auxOutputs[i], DBG_LOC);
    }
    struct AuxGuard {
        XModel *m;
        ~AuxGuard()
        {
            m->ClearAuxOutputs();
        }
    } guard{_model};
    _model->SetAuxOutputs(auxXs);
    ForwardWithInputsEmbedsV2(rt, input, attnMeta, kvCache, freqsCis, output, currStream,
                              deepstackInput, inputIds);
}

void _CModel::ForwardGetLogits(XRuntime &rt, at::Tensor &input, at::Tensor &indices,
                               at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _indices, _output;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input, DBG_LOC);
    InitXTensor(_indices, indices, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);

    if (input.size(0) == 0) {
        return;
    }

    bool async = rt.ForwardThreadEnabled() && !rt.multiTaskParallel && currStream != 0;
    if (async) {
        ForwardTask task;
        task.kind = ForwardTask::FORWARD_GET_LOGITS;
        task.input = _input;
        task.indices = _indices;
        task.output = _output;
        EnsureForwardHandlers(rt);
        task.seq = rt.AllocFwdSeq();
        rt.RecordInputEvent(reinterpret_cast<aclrtStream>(currStream), task.seq);
        rt.WaitOutputEvent(reinterpret_cast<aclrtStream>(currStream), task.seq);
        rt.SubmitForward(std::move(task));
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
                                  std::vector<at::Tensor> &freqsCis, at::Tensor &indices,
                                  at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _indices, _output;
    std::vector<XTensor> _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input, DBG_LOC);
    InitXTensor(_indices, indices, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    _freqsCis.resize(freqsCis.size());
    for (size_t i = 0; i < freqsCis.size(); i++) {
        InitXTensor(_freqsCis[i], freqsCis[i], DBG_LOC);
    }

    if (kvCache.size() != _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    if (input.size(0) == 0) {
        return;  // for DP dummy run with MoE, add a padding token to avoid empty input
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        if (kvCache[i].size() != _kv[i].size()) {
            throw std::runtime_error(std::string(__func__) + ": check kv cache failed at layer " +
                                     std::to_string(i));
        }
        // allowPitch: cache consumers that address a dim-0 pitch (hybrid GDN
        // state, MHA/GQA k/v, MLA k/pe, CXA, DSA). CXA caches are paged-addressed;
        // LI-C8 writers/readers consume stride[0] directly (K + scales packed
        // page-strided in one alloc).
        bool allowPitch = IsHybridAttn(attnType) || attnType == XMODEL_ATTN_MHA ||
                          attnType == XMODEL_ATTN_MLA || attnType == XMODEL_ATTN_CXA ||
                          attnType == XMODEL_ATTN_DSA;
        for (uint64_t j = 0; j < _kv[i].size(); j++) {
            InitXTensor(_kv[i][j], kvCache[i][j], DBG_LOC, /*allowPitched=*/allowPitch);
        }
    }

    bool async = rt.ForwardThreadEnabled() && !rt.multiTaskParallel && currStream != 0;
    if (async) {
        ForwardTask task;
        task.kind = ForwardTask::FORWARD_AND_GET_LOGITS;
        task.input = _input;
        task.output = _output;
        task.attnMeta = attnMeta;
        task.kvCache = _kv;
        task.deepstackInputEmbeds = _deepstackInputEmbeds;
        task.freqsCis = _freqsCis;
        task.indices = _indices;
        task.auxOutputs = _model->StealAuxOutputs();
        EnsureForwardHandlers(rt);
        task.seq = rt.AllocFwdSeq();
        rt.RecordInputEvent(reinterpret_cast<aclrtStream>(currStream), task.seq);
        rt.WaitOutputEvent(reinterpret_cast<aclrtStream>(currStream), task.seq);
        rt.SubmitForward(std::move(task));
        return;
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
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lens;
    _attnMeta.cachedLensCpu = attnMeta.cachedLens;
    _attnMeta.blockTablesCpu = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.position, attnMeta.positions, DBG_LOC, /*allowPitched=*/true);
    std::vector<at::Tensor> freqsCisVec = {freqsCis};
    ForwardAndGetLogits(rt, input, _attnMeta, kvCache, freqsCisVec, indices, output, currStream);
}

void _CModel::ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                      std::vector<std::vector<at::Tensor>> &kvCache,
                                      at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                      std::vector<at::Tensor> &deepstackInput, at::Tensor &inputIds)
{
    XModelAttnMeta _attnMeta;
    XTensor _input, _output, _freqsCis, _inputIds;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    InitXTensor(_freqsCis, freqsCis, DBG_LOC);
    InitOptionalXTensor(_inputIds, inputIds, DBG_LOC);

    if (kvCache.size() != _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    if (input.size(0) > output.size(0)) {
        throw std::runtime_error(std::string(__func__) + ": input's size 0 > output's size 0");
    }

    if (input.size(0) == 0) {
        return;  // for DP dummy run with MoE, add a padding token to avoid empty input
    }

    if (deepstackInput.size() != _deepstackInputEmbeds.size()) {
        throw std::runtime_error(std::string(__func__) + ": check deepstack input failed");
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        if (kvCache[i].size() != _kv[i].size()) {
            throw std::runtime_error(std::string(__func__) + ": check kv cache failed at layer " +
                                     std::to_string(i));
        }
        // allowPitch: cache consumers that address a dim-0 pitch (hybrid GDN
        // state, MHA/GQA k/v, MLA k/pe, CXA, DSA). CXA caches are paged-addressed;
        // LI-C8 writers/readers consume stride[0] directly (K + scales packed
        // page-strided in one alloc).
        bool allowPitch = IsHybridAttn(attnType) || attnType == XMODEL_ATTN_MHA ||
                          attnType == XMODEL_ATTN_MLA || attnType == XMODEL_ATTN_CXA ||
                          attnType == XMODEL_ATTN_DSA;
        for (uint64_t j = 0; j < _kv[i].size(); j++) {
            InitXTensor(_kv[i][j], kvCache[i][j], DBG_LOC, /*allowPitched=*/allowPitch);
        }
    }

    for (uint32_t i = 0; i < deepstackInput.size(); i++) {
        InitXTensor(_deepstackInputEmbeds[i], deepstackInput[i], DBG_LOC);
    }

    bool async = rt.ForwardThreadEnabled() && !rt.multiTaskParallel && currStream != 0;
    if (async) {
        ForwardTask task;
        task.kind = ForwardTask::FORWARD_WITH_INPUTS_EMBEDS;
        task.input = _input;
        task.output = _output;
        task.attnMeta = attnMeta;
        task.kvCache = _kv;
        task.deepstackInputEmbeds = _deepstackInputEmbeds;
        task.freqsCisSingle = _freqsCis;
        task.inputIds = _inputIds;
        task.auxOutputs = _model->StealAuxOutputs();
        EnsureForwardHandlers(rt);
        task.seq = rt.AllocFwdSeq();
        rt.RecordInputEvent(reinterpret_cast<aclrtStream>(currStream), task.seq);
        rt.WaitOutputEvent(reinterpret_cast<aclrtStream>(currStream), task.seq);
        rt.SubmitForward(std::move(task));
        return;
    }

    if (currStream != 0 && rt.taskId == 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    if (rt.multiTaskParallel && rt.taskId == 1) {
        rt.NotifyWaitPeerStream();
    }

    _model->ForwardWithInputsEmbeds(rt, _input, attnMeta, _kv, _deepstackInputEmbeds, _freqsCis,
                                    _inputIds, _output);

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
                                        std::vector<at::Tensor> &deepstackInput,
                                        const std::optional<at::Tensor> &inputIds)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lens;
    _attnMeta.cachedLensCpu = attnMeta.cachedLens;
    _attnMeta.blockTablesCpu = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.position, attnMeta.positions, DBG_LOC, /*allowPitched=*/true);
    at::Tensor ids = inputIds.value_or(at::Tensor());
    ForwardWithInputsEmbeds(rt, input, _attnMeta, kvCache, freqsCis, output, currStream,
                            deepstackInput, ids);
}

void _CModel::ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                        std::vector<std::vector<at::Tensor>> &kvCache,
                                        at::Tensor &freqsCis, at::Tensor &output,
                                        std::vector<at::Tensor> &auxOutputs, uint64_t currStream,
                                        std::vector<at::Tensor> &deepstackInput,
                                        const std::optional<at::Tensor> &inputIds)
{
    std::vector<XTensor> auxXs(auxOutputs.size());
    for (size_t i = 0; i < auxOutputs.size(); i++) {
        InitXTensor(auxXs[i], auxOutputs[i], DBG_LOC);
    }
    struct AuxGuard {
        XModel *m;
        ~AuxGuard()
        {
            m->ClearAuxOutputs();
        }
    } guard{_model};
    _model->SetAuxOutputs(auxXs);
    ForwardWithInputsEmbedsV1(rt, input, attnMeta, kvCache, freqsCis, output, currStream,
                              deepstackInput, inputIds);
}

void _CModel::ForwardV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                        std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                        at::Tensor &output, std::vector<at::Tensor> &auxOutputs,
                        uint64_t currStream)
{
    std::vector<XTensor> auxXs(auxOutputs.size());
    for (size_t i = 0; i < auxOutputs.size(); i++) {
        InitXTensor(auxXs[i], auxOutputs[i], DBG_LOC);
    }
    struct AuxGuard {
        XModel *m;
        ~AuxGuard()
        {
            m->ClearAuxOutputs();
        }
    } guard{_model};
    _model->SetAuxOutputs(auxXs);
    ForwardV1(rt, input, attnMeta, kvCache, freqsCis, output, currStream);
}

void _CModel::ForwardV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                        std::vector<std::vector<at::Tensor>> &kvCache,
                        std::vector<at::Tensor> &freqsCis, at::Tensor &output,
                        std::vector<at::Tensor> &auxOutputs, uint64_t currStream)
{
    std::vector<XTensor> auxXs(auxOutputs.size());
    for (size_t i = 0; i < auxOutputs.size(); i++) {
        InitXTensor(auxXs[i], auxOutputs[i], DBG_LOC);
    }
    struct AuxGuard {
        XModel *m;
        ~AuxGuard()
        {
            m->ClearAuxOutputs();
        }
    } guard{_model};
    _model->SetAuxOutputs(auxXs);
    ForwardV2(rt, input, attnMeta, kvCache, freqsCis, output, currStream);
}

size_t _CModel::GetTensorPoolSize(int dbg)
{
    return _model->GetTensorPoolSize(dbg);
}

static void DsaCpMaterializeNzShard(const uint8_t *src, uint8_t *dst, size_t outer,
                                    size_t shardBytes, size_t sourceStride)
{
    CHECK_ACL(aclrtSynchronizeDevice());
    for (size_t i = 0; i < outer; ++i) {
        CHECK_ACL(aclrtMemcpy(dst + i * shardBytes, shardBytes, src + i * sourceStride, shardBytes,
                              ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
}

void _CModel::PrepareDsaCpDecodeWeights(const XModelConfig &c, uint32_t tpRank, uint32_t layer)
{
    at::NoGradGuard noGrad;
    auto headShard = [&](XTensor &out, const at::Tensor &weight, size_t dim) {
        XTensor source;
        InitXTensor(source, weight, DBG_LOC);
        const bool nz = source.nz;
        auto shape = source.shape;
        if (shape[dim] % c.defTpSize != 0) {
            throw std::invalid_argument("DSA-CP decode weight head axis must be divisible by TP");
        }
        shape[dim] /= c.defTpSize;
        if (!nz) {
            // Only a non-contiguous ND column slice needs a local copy.
            auto shard =
                weight
                    .narrow(static_cast<int64_t>(dim), static_cast<int64_t>(tpRank * shape[dim]),
                            static_cast<int64_t>(shape[dim]))
                    .contiguous();
            InitXTensor(out, shard, DBG_LOC);
            dsaCpDecodeWeightStorage.push_back(std::move(shard));
            return;
        }
        const auto storageShape = DsaCpWeightStorageShape(source, true);
        size_t storageDim = dim;
        size_t shardLen = shape[dim];
        if (dim + 2 >= shape.size()) {
            const bool columns = dim + 1 == shape.size();
            const size_t block = columns ? storageShape.back() : DSA_CP_NZ_ROWS;
            if (shardLen % block != 0) {
                throw std::invalid_argument(
                    "DSA-CP decode head shard must contain whole NZ blocks");
            }
            shardLen /= block;
            storageDim = columns ? dim - 1 : dim + 1;
        }
        size_t outer = 1, inner = 1;
        for (size_t i = 0; i < storageDim; ++i) {
            outer *= storageShape[i];
        }
        for (size_t i = storageDim + 1; i < storageShape.size(); ++i) {
            inner *= storageShape[i];
        }
        const size_t shardBytes = shardLen * inner * weight.element_size();
        auto *src = static_cast<uint8_t *>(source.ptr) + tpRank * shardBytes;
        if (outer == 1) {
            // W8A8 transposed QB and leading-head WUKT/WUV slices share storage.
            out.Init(shape, source.dtype, src);
            out.nz = true;
            return;
        }
        // Non-transposed NZ QB slices row blocks inside each column block.
        // Materialize only this rank's blocks, keeping their NZ byte layout.
        auto shard = at::empty({static_cast<int64_t>(outer * shardLen * inner)}, weight.options());
        auto *dst = static_cast<uint8_t *>(TensorPtr(shard));
        const size_t sourceStride = storageShape[storageDim] * inner * weight.element_size();
        DsaCpMaterializeNzShard(src, dst, outer, shardBytes, sourceStride);
        out.Init(shape, source.dtype, dst);
        out.nz = true;
        dsaCpDecodeWeightStorage.push_back(std::move(shard));
    };
    auto outputParameterShard = [&](XTensor &parameter, size_t elementsPerChannel) {
        if (parameter.ptr != nullptr && parameter.numel > elementsPerChannel) {
            const size_t shardLen = parameter.numel / c.defTpSize;
            parameter.View(std::vector<size_t>{parameter.numel});
            parameter.SliceView(tpRank * shardLen, (tpRank + 1) * shardLen);
        }
    };

    InitMatmulWeight("mlaQBDecodeTp", mlaQB, mlaQBInputScale, mlaQBInputOffset, mlaQBQuantBias,
                     mlaQBDeqScale, _model->mlaQBDecodeTp, layer, false, tpRank);
    auto &qb = _model->mlaQBDecodeTp[layer];
    const bool transposed = mlaQB[layer].scalar_type() == at::kChar && c.quantAttnWeightTrans;
    headShard(qb.weight, mlaQB[layer], transposed ? 1 : 0);
    headShard(_model->mlaWUKTDecodeTp[layer], mlaWUKT[layer], 0);
    headShard(_model->mlaWUVDecodeTp[layer], mlaWUV[layer], 0);
    outputParameterShard(qb.quantBias, 1);
    // Each dequantization channel is packed as two FP32 elements.
    outputParameterShard(qb.deqScale, 2);
    // O is already TP-sharded; decode uses the same native storage with bias
    // on rank zero only, while prefill needs the full bias on every token shard.
    InitMatmulWeight("attnOutDecodeTp", attnOut, attnOutInputScale, attnOutInputOffset,
                     attnOutQuantBias, attnOutDeqScale, _model->attnOutDecodeTp, layer, true,
                     tpRank);
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
    if (weightLayer >= w.size() || !TensorUsable(w[weightLayer])) {
        throw std::invalid_argument(DBG_PREFIX + ": " + wXT.name +
                                    " weight tensor is required but not provided");
    }
    InitXTensor(wXT.weight, w[weightLayer], DBG_LOC);
    if (wXT.weight.dtype ==
        INT32) {  // for int4 matmul: weights will arrive as int4pack (int32 dtype)
        wXT.weight.View(INT4);
    }
    if (wXT.weight.dtype == INT8 && dScale.size() <= weightLayer) {
        std::string errStr = DBG_PREFIX + ": " + name +
                             "dequant scale parameters are "
                             "required when weights are quantized";
        throw std::invalid_argument(errStr);
    }
    if (weightLayer < iScale.size()) {
        InitXTensor(wXT.inputScale, iScale[weightLayer], DBG_LOC);
    }
    if (weightLayer < iOffset.size()) {
        InitXTensor(wXT.inputOffset, iOffset[weightLayer], DBG_LOC);
    }
    // Notice: only tpRank == 0 in RowParallelLinear need to add quant_bias
    if (weightLayer < qBias.size() && (!isRowParallel || tpRank == 0)) {
        InitXTensor(wXT.quantBias, qBias[weightLayer], DBG_LOC);
    }
    if (weightLayer < dScale.size()) {
        InitXTensor(wXT.deqScale, dScale[weightLayer], DBG_LOC);
    }
}

void DsaCpMetadata(XRuntime &rt, at::Tensor &queryStartLoc, at::Tensor &queryLens,
                   at::Tensor &cachedLens, at::Tensor &localQueryStartLoc,
                   at::Tensor &localQueryLens, at::Tensor &localCachedLens,
                   at::Tensor &localSlotMapping, uint32_t shardBegin, uint32_t shardEnd)
{
    XTensor queryStartLocXT, queryLensXT, cachedLensXT, localQueryStartLocXT, localQueryLensXT,
        localCachedLensXT, localSlotMappingXT;
    InitXTensor(queryStartLocXT, queryStartLoc);
    InitXTensor(queryLensXT, queryLens);
    InitXTensor(cachedLensXT, cachedLens);
    InitXTensor(localQueryStartLocXT, localQueryStartLoc);
    InitXTensor(localQueryLensXT, localQueryLens);
    InitXTensor(localCachedLensXT, localCachedLens);
    InitXTensor(localSlotMappingXT, localSlotMapping);
    XliteOpDsaCpMetadata(rt, queryStartLocXT, queryLensXT, cachedLensXT, localQueryStartLocXT,
                         localQueryLensXT, localCachedLensXT, localSlotMappingXT, queryLens.numel(),
                         shardBegin, shardEnd);
    rt.Synchronize();
}

void DsaCpWriteKVCacheBySlot(XRuntime &rt, at::Tensor &input, at::Tensor &cache,
                             at::Tensor &slotMapping, uint32_t numTokens, uint32_t blockSize)
{
    XTensor inputXT, cacheXT, slotMappingXT;
    InitXTensor(inputXT, input, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(cacheXT, cache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(slotMappingXT, slotMapping);
    XliteOpDsaCpWriteKVCacheBySlot(rt, inputXT, cacheXT, slotMappingXT, numTokens, blockSize);
    rt.Synchronize();
}

void DsaCpReorderWo(XRuntime &rt, at::Tensor &input, at::Tensor &output, uint32_t ranks,
                    uint32_t columnBlocks, uint64_t blockBytes)
{
    XTensor inputXT, outputXT;
    InitXTensor(inputXT, input);
    InitXTensor(outputXT, output);
    XliteOpDsaCpReorderWo(rt, inputXT, outputXT, ranks, columnBlocks, blockBytes);
    rt.Synchronize();
}

void AllGather(XRuntime &rt, at::Tensor &out, at::Tensor &in, uint32_t commType = 0)
{
    XTensor _in, _out;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);

    enum commType type = commType == 1 ? DP : TP;
    XliteOpAllGather(rt, _in, _out, type, type == DP, DBG_LOC);
    rt.Synchronize();
}

void ReduceScatter(XRuntime &rt, at::Tensor &out, at::Tensor &in, uint32_t commType = 0)
{
    XTensor _in, _out;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);

    enum commType type = commType == 1 ? DP : TP;
    XliteOpReduceScatter(rt, _in, _out, type, type == DP, DBG_LOC);
    rt.Synchronize();
}

void AllReduce(XRuntime &rt, at::Tensor &out, at::Tensor &in, uint32_t commType = 0)
{
    XTensor _in, _out;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);

    enum commType type = commType == 1 ? DP : TP;
    XliteOpAllReduceSum(rt, _in, _out, type, type == DP, DBG_LOC);
    rt.Synchronize();
}

void AlltoAllV(XRuntime &rt, at::Tensor &out, at::Tensor &in, at::Tensor &sendCounts,
               at::Tensor &recvCounts, at::Tensor &sdispls, at::Tensor &rdispls,
               uint32_t commType = 0)
{
    XTensor _in, _out, _sendCounts, _recvCounts, _sdispls, _rdispls;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_sendCounts, sendCounts, DBG_LOC);
    InitXTensor(_recvCounts, recvCounts, DBG_LOC);
    InitXTensor(_sdispls, sdispls, DBG_LOC);
    InitXTensor(_rdispls, rdispls, DBG_LOC);

    enum commType type = TP;
    if (commType == 1) {
        type = DP;
    } else if (commType == 2) {
        type = EP;
    }
    XliteOpAlltoAllV(rt, _in, _out, _sendCounts, _recvCounts, _sdispls, _rdispls, type, DBG_LOC);
    rt.Synchronize();
}

void Add(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z)
{
    XTensor _x, _y, _z;

    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_y, y, DBG_LOC);
    InitXTensor(_z, z, DBG_LOC);
    XliteOpAdd(rt, _x, _y, _z);
    rt.Synchronize();
}

// Untimed micro-benchmark dispatch (variant-based, mirrors the timed family).
// Return the block count actually launched with.
uint32_t BenchEmpty(XRuntime &rt, const std::string &variant, uint32_t blockNum)
{
    uint32_t n = XliteOpBenchEmpty(rt, variant, blockNum);
    rt.Synchronize();
    return n;
}

uint32_t BenchScalar(XRuntime &rt, const std::string &variant, at::Tensor &outTensor, double seed,
                     uint32_t iters, uint32_t blockNum)
{
    XTensor out;
    InitXTensor(out, outTensor);
    uint32_t n = XliteOpBenchScalar(rt, variant, out, seed, iters, blockNum);
    rt.Synchronize();
    return n;
}

#ifdef MICROBENCH_TIMED
// Timed scalar micro-benchmarks (SYS_CNT sampling via the BENCH_TICK* macros
// in csrc/kernels/kernel_macro.h): same dependent
// chains as the scalar_* family, plus in-kernel SYS_CNT sampling written to
// the caller-pre-allocated `ticks` int64 tensor as [start, issued, drained]
// cycles. One entry for the whole timed family; `variant` picks kernel and
// core type. Callers read `ticks` back after this returns (Synchronize inside).
// Compiled only in XLITE_BENCH_TIMED=ON builds.
uint32_t BenchTimed(XRuntime &rt, const std::string &variant, at::Tensor &outTensor,
                    at::Tensor &ticksTensor, double seed, uint32_t iters, uint32_t blockNum)
{
    XTensor out, ticks;
    InitXTensor(out, outTensor);
    InitXTensor(ticks, ticksTensor);
    uint32_t n = XliteOpBenchTimed(rt, variant, out, ticks, seed, iters, blockNum);
    rt.Synchronize();
    return n;
}

// MTE memory micro-benchmarks (lmbench-style bandwidth / latency vs size, see
// the MTE section of csrc/kernels/microbench.h): read / write / copy stream
// `bytes` per repetition, chase walks a dependent 64B-granular chain. One
// entry for the whole family; `variant` picks the kernel. Per-block [start,
// issued, drained] cycles land in the caller-pre-allocated `ticks` int64
// tensor (3 per block), verification data in `dst`. Compiled only in
// XLITE_BENCH_TIMED=ON builds.
uint32_t BenchMte(XRuntime &rt, const std::string &variant, at::Tensor &srcTensor,
                  at::Tensor &dstTensor, at::Tensor &ticksTensor, uint64_t bytes, uint32_t iters,
                  uint32_t blockNum)
{
    XTensor src, dst, ticks;
    InitXTensor(src, srcTensor);
    InitXTensor(dst, dstTensor);
    InitXTensor(ticks, ticksTensor);
    uint32_t n = XliteOpBenchMte(rt, variant, src, dst, ticks, bytes, iters, blockNum);
    rt.Synchronize();
    return n;
}

// Vector-op micro-benchmarks (vector-pipe throughput vs vector length, see
// the VEC section of csrc/kernels/microbench.h): a dependent chain of `iters`
// op applications over `elems` UB-resident elements per block. One entry for
// the whole family; `variant` picks the kernel. Per-block [start, issued,
// drained] cycles land in the caller-pre-allocated `ticks` int64 tensor,
// every block's final chain value in `out`. Compiled only in
// XLITE_BENCH_TIMED=ON builds.
uint32_t BenchVec(XRuntime &rt, const std::string &variant, at::Tensor &srcTensor,
                  at::Tensor &outTensor, at::Tensor &ticksTensor, uint64_t elems, uint32_t iters,
                  uint32_t blockNum)
{
    XTensor src, out, ticks;
    InitXTensor(src, srcTensor);
    InitXTensor(out, outTensor);
    InitXTensor(ticks, ticksTensor);
    uint32_t n = XliteOpBenchVec(rt, variant, src, out, ticks, elems, iters, blockNum);
    rt.Synchronize();
    return n;
}

// Cube matmul micro-benchmark (single A@B.T per launch via the Matmul class):
// FP16 inputs A[m0,k0] and B[n0,k0], FP16 output C[m0,n0]. Per-block [start,
// issued, drained] cycles land in `ticks`. Compiled only in XLITE_BENCH_TIMED=ON.
uint32_t BenchCube(XRuntime &rt, const std::string &variant, at::Tensor &aTensor,
                   at::Tensor &bTensor, at::Tensor &cTensor, at::Tensor &ticksTensor, uint64_t m0,
                   uint64_t n0, uint64_t k0, uint32_t blockNum)
{
    XTensor a, b, c, ticks;
    InitXTensor(a, aTensor);
    InitXTensor(b, bTensor);
    InitXTensor(c, cTensor);
    InitXTensor(ticks, ticksTensor);
    uint32_t n = XliteOpBenchCube(rt, variant, a, b, c, ticks, m0, n0, k0, blockNum);
    rt.Synchronize();
    return n;
}
#endif  // MICROBENCH_TIMED

void Print(at::Tensor &x, const char *name, uint32_t nRow, uint32_t nCol)
{
    XTensor _x;

    // debug output reads pitched layouts correctly
    InitXTensor(_x, x, DBG_LOC, /*allowPitched=*/true);
    _x.Print(name, nRow, nCol);
}

void Matmul(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z, bool weightNZ,
            bool transpose)
{
    (void)weightNZ;  // deprecated no-op
    XTensor _x, _y, _z, _bias, _deqScale;

    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_y, y, DBG_LOC);
    InitXTensor(_z, z, DBG_LOC);
    if (_x.dtype == INT32 && _y.dtype == INT32) {  // for int4 matmul
        _x.View(INT4);
        _y.View(INT4);
    }
    XliteOpMatmul(rt, _x, _y, _z, _bias, _deqScale, transpose);
}

uint64_t MatmulBench(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z,
                     at::Tensor &x_warmup, at::Tensor &y_warmup, at::Tensor &z_warmup,
                     int iterations, int warmup_iterations, bool weightNZ, bool transpose)
{
    (void)weightNZ;  // deprecated no-op
    XTensor _x, _y, _z, _x_warmup, _y_warmup, _z_warmup, _bias, _deqScale;

    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_y, y, DBG_LOC);
    InitXTensor(_z, z, DBG_LOC);

    InitXTensor(_x_warmup, x_warmup, DBG_LOC);
    InitXTensor(_y_warmup, y_warmup, DBG_LOC);
    InitXTensor(_z_warmup, z_warmup, DBG_LOC);

    for (int i = 0; i < warmup_iterations; i++) {
        XliteOpMatmul(rt, _x_warmup, _y_warmup, _z_warmup, _bias, _deqScale, transpose);
    }
    rt.Synchronize();

    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        XliteOpMatmul(rt, _x, _y, _z, _bias, _deqScale, transpose);
    }
    rt.Synchronize();
    auto end = std::chrono::steady_clock::now();

    auto res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    return res / iterations;
}

void MatmulWithBias(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z, at::Tensor &bias,
                    bool weightNZ)
{
    (void)weightNZ;  // deprecated no-op
    XTensor _x, _y, _z, _bias, _deqSacle;

    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_y, y, DBG_LOC);
    InitXTensor(_z, z, DBG_LOC);
    InitXTensor(_bias, bias, DBG_LOC);
    if (_x.dtype == INT32 && _y.dtype == INT32) {  // for int4 matmul
        _x.View(INT4);
        _y.View(INT4);
    }
    XliteOpMatmul(rt, _x, _y, _z, _bias, _deqSacle);
    rt.Synchronize();
}

void Embed(XRuntime &rt, at::Tensor &weight, at::Tensor &in, at::Tensor &out, uint32_t start,
           uint32_t end)
{
    XTensor _in, _out, _weight;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_weight, weight, DBG_LOC);
    XliteOpEmbed(rt, _in, _weight, start, end, _out);
    rt.Synchronize();
}

void RMSNormVarianceOnly(XRuntime &rt, at::Tensor &in, at::Tensor &out, float normEps,
                         uint32_t normDim, uint32_t cntPerToken, uint32_t inStartOffset,
                         uint32_t outStartOffset)
{
    XTensor _in, _out;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    XliteOpRmsNorm(rt, _in, XTensor(), _out, normEps, normDim == 0 ? _in.shape[1] : normDim, false,
                   XTensor(), cntPerToken, inStartOffset, outStartOffset, XTensor());
    rt.Synchronize();
}

void RMSNorm(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &out, float normEps,
             uint32_t normDim, uint32_t cntPerToken, uint32_t inStartOffset,
             uint32_t outStartOffset, std::optional<at::Tensor> variance)
{
    XTensor _in, _out, _norm, _normBias, _variance;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitOptionalXTensor(_norm, norm, DBG_LOC);
    if (variance.has_value()) {
#ifdef XLITE_DEBUG_ON_MISC
        {
            XDebugStream s(rt.rankId(), __func__, rt.rankId() == 0);
            s << "variance: " << variance.value() << std::endl;
        }
#endif
        InitXTensor(_variance, variance.value(), DBG_LOC);
    }
    XliteOpRmsNorm(rt, _in, _norm, _out, normEps, normDim == 0 ? _in.shape[1] : normDim, true,
                   _normBias, cntPerToken, inStartOffset, outStartOffset, _variance);
    rt.Synchronize();
}

void RMSNormWithBias(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &normBias,
                     at::Tensor &out, float normEps, uint32_t normDim, uint32_t cntPerToken,
                     uint32_t inStartOffset, uint32_t outStartOffset)
{
    XTensor _in, _out, _norm, _normBias;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_norm, norm, DBG_LOC);
    InitXTensor(_normBias, normBias, DBG_LOC);
    XliteOpRmsNorm(rt, _in, _norm, _out, normEps, normDim == 0 ? _in.shape[1] : normDim, true,
                   _normBias, cntPerToken, inStartOffset, outStartOffset);
    rt.Synchronize();
}

void L2Norm(XRuntime &rt, at::Tensor &in, at::Tensor &out, float normEps, uint32_t normDim)
{
    XTensor _in, _out;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    XliteOpL2Norm(rt, _in, _out, normEps, normDim == 0 ? _in.shape[1] : normDim);

    rt.Synchronize();
}

void LayerNorm(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &normBias,
               at::Tensor &out, float normEps, uint32_t normDim)
{
    XTensor _in, _out, _norm, _normBias;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_norm, norm, DBG_LOC);
    InitXTensor(_normBias, normBias, DBG_LOC);
    XliteOpLayerNorm(rt, _in, _norm, _normBias, _out, normEps, normDim);
    rt.Synchronize();
}

void AddBias(XRuntime &rt, at::Tensor &in, at::Tensor &weight, at::Tensor &out)
{
    XTensor _in, _out, _weight;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_weight, weight, DBG_LOC);
    XliteOpAddBias(rt, _in, _weight, _out);
    rt.Synchronize();
}

void SiluAndMul(XRuntime &rt, at::Tensor &in, at::Tensor &out, float swigluLimit)
{
    XTensor _in, _out;
    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);

    XliteOpSiluAndMul(rt, _in, _out, XTensor(), swigluLimit);
    rt.Synchronize();
}

void SigmoidGateMul(XRuntime &rt, at::Tensor &attn, at::Tensor &gate, at::Tensor &out,
                    const std::optional<at::Tensor> &add)
{
    XTensor _attn, _gate, _out;
    InitXTensor(_attn, attn, DBG_LOC);
    InitXTensor(_gate, gate, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);

    XTensor _add;
    XTensor *addPtr = nullptr;
    if (add.has_value()) {
        InitXTensor(_add, *add, DBG_LOC);
        addPtr = &_add;
    }
    XliteOpSigmoidGateMul(rt, _attn, _gate, _out, addPtr);
    rt.Synchronize();
}

void RopeAndCache(XRuntime &rt, at::Tensor &inout, at::Tensor &kCache, at::Tensor &vCache,
                  at::Tensor &position, at::Tensor &cossin, at::Tensor &slotMapping,
                  uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim, uint32_t rotDim,
                  uint32_t blockSize, bool isNeox, uint64_t mropeMaskH, uint64_t mropeMaskW)
{
    XTensor _inout, _kCache, _vCache, _position, _cossin, _slotMapping;

    InitXTensor(_inout, inout, DBG_LOC);
    InitXTensor(_kCache, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_vCache, vCache, DBG_LOC, /*allowPitched=*/true);
    // Pitched positions are only consumed on the mrope path (H/W rows at the dim-0 pitch);
    // elsewhere the kernel reads positions flat, so contiguity is still required there.
    InitXTensor(_position, position, DBG_LOC,
                /*allowPitched=*/position.dim() == 2 && (mropeMaskH != 0 || mropeMaskW != 0));
    InitXTensor(_cossin, cossin, DBG_LOC);
    InitXTensor(_slotMapping, slotMapping, DBG_LOC);
    XliteOpRopeCache(rt, _inout, _kCache, _vCache, _position, _cossin, _slotMapping, nHeads,
                     nKvHeads, headDim, rotDim, blockSize, isNeox, mropeMaskH, mropeMaskW);
    rt.Synchronize();
}

// max(lens + cachedLens) over the batch (sync D2H; standalone/test path)
static uint32_t DeriveMaxSeqLen(const XTensor &lens, const XTensor &cachedLens)
{
    uint32_t batch = static_cast<uint32_t>(lens.shape[0]);
    std::vector<int32_t> l(batch), c(batch);
    size_t bytes = batch * sizeof(int32_t);
    CHECK_ACL(aclrtMemcpy(l.data(), bytes, lens.ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(c.data(), bytes, cachedLens.ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    uint32_t maxSeqLen = 0;
    for (uint32_t i = 0; i < batch; i++) {
        maxSeqLen = std::max(maxSeqLen, static_cast<uint32_t>(l[i] + c[i]));
    }
    return maxSeqLen;
}

// Narrow a persistent block table to the lens-derived block count; returns it
static uint32_t SliceBlockTableToLens(XTensor &blockTables, const XTensor &lens,
                                      const XTensor &cachedLens, uint32_t blockSize,
                                      uint32_t compressRatio = 1)
{
    blockTables.AtLeast2D();
    uint32_t blocks = DIV_ROUND_UP(DeriveMaxSeqLen(lens, cachedLens) / compressRatio, blockSize);
    const size_t batch = lens.numel;
    if (blockTables.shape.size() != 2 || blockTables.shape[0] < batch) {
        throw std::runtime_error(std::string(__func__) +
                                 ": blockTables must be 1-D single-row or 2-D with rows >= batch");
    }
    if (blocks > 0) {  // 0 = no complete compressed block: nothing valid to narrow
        blockTables.SliceView(0, blocks, 1);
    }
    return blocks;
}

void Attention(XRuntime &rt, at::Tensor &qkv, at::Tensor &kCache, at::Tensor &vCache,
               at::Tensor &output, at::Tensor &queryStartLoc, at::Tensor &lens,
               at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads, uint32_t nKvHeads,
               uint32_t headDim, uint32_t blockSize, uint32_t batch, bool enableFlashAttention,
               uint32_t tileSizeOfCachedKV)
{
    XTensor _qkv, _kCache, _vCache, _qk, _output, _queryStartLoc, _lens, _cachedLens, _blockTables;

    InitXTensor(_qkv, qkv, DBG_LOC);
    InitXTensor(_kCache, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_vCache, vCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_output, output, DBG_LOC);
    InitXTensor(_queryStartLoc, queryStartLoc, DBG_LOC);
    InitXTensor(_lens, lens, DBG_LOC);
    InitXTensor(_cachedLens, cachedLens, DBG_LOC);
    InitXTensor(_blockTables, blockTables, DBG_LOC, /*allowPitched=*/true);
    uint32_t maxNumBlock = SliceBlockTableToLens(_blockTables, _lens, _cachedLens, blockSize);

    if (!enableFlashAttention) {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, maxNumBlock * blockSize},
                                   XDtypeOf(qkv), DBG_LOC);
        XliteOpAttention(rt, _qkv, _kCache, _vCache, qk, _output, _queryStartLoc, _lens,
                         _cachedLens, _blockTables, nHeads, nKvHeads, headDim, blockSize, batch);
        rt.Synchronize();
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, tileSizeOfCachedKV},
                                   XDtypeOf(qkv), DBG_LOC);
        XTensor &sv = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, headDim}, XDtypeOf(qkv), DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({_qkv.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({_qkv.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &sync = rt.GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
        sync.Memset(0, rt.stream);
        XliteOpFlashAttention(rt, _qkv, _kCache, _vCache, qk, sv, max, sum, lastMax, lastSum, sync,
                              _output, _queryStartLoc, _lens, _cachedLens, _blockTables, nHeads,
                              nKvHeads, headDim, blockSize, batch, tileSizeOfCachedKV);
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

void MLAV2(XRuntime &rt, at::Tensor &qWithQr, at::Tensor &qr, at::Tensor &kCache,
           at::Tensor &peCache, at::Tensor &wukT, at::Tensor &wuv, at::Tensor &output,
           at::Tensor &queryStartLoc, at::Tensor &lens, at::Tensor &cachedLens,
           at::Tensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim, uint32_t nopeHeadDim,
           uint32_t vHeadDim, uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, float scale,
           at::Tensor &topkIndices, uint32_t topK, bool weightNz, bool enableFlashAttention,
           uint32_t tileSizeOfCachedKV, bool dense, const std::optional<at::Tensor> &kScaleCache)
{
    (void)weightNz;  // deprecated no-op: nz is auto-detected from the weight format
    XTensor _qWithQr, _qr, _kCache, _peCache, _wukT, _wuv, _output, _queryStartLoc, _lens,
        _cachedLens, _blockTables, _topkIndices, _kScaleCache;
    InitXTensor(_kScaleCache, kScaleCache, DBG_LOC);
    InitXTensor(_qWithQr, qWithQr, DBG_LOC);
    InitXTensor(_qr, qr, DBG_LOC);
    InitXTensor(_kCache, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_peCache, peCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_wukT, wukT, DBG_LOC);
    InitXTensor(_wuv, wuv, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    InitXTensor(_queryStartLoc, queryStartLoc, DBG_LOC);
    InitXTensor(_lens, lens, DBG_LOC);
    InitXTensor(_cachedLens, cachedLens, DBG_LOC);
    InitXTensor(_blockTables, blockTables, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_topkIndices, topkIndices, DBG_LOC);
    uint32_t maxNumBlocks =
        dense ? 0 : SliceBlockTableToLens(_blockTables, _lens, _cachedLens, blockSize);

    XTensor &qAbsorb =
        rt.GetTensor({_qWithQr.shape[0], nHeads * kvLoraRank}, XDtypeOf(qWithQr), DBG_LOC);
    XliteOpEinsumMhtHtdMhd(rt, _qWithQr, _wukT, qAbsorb, _qWithQr.shape[0], nHeads, nopeHeadDim,
                           kvLoraRank, static_cast<int>(nopeHeadDim + ropeHeadDim));

    XTensor &oAbsorb =
        rt.GetTensor({_qWithQr.shape[0], nHeads * kvLoraRank}, XDtypeOf(qWithQr), DBG_LOC);
    const bool c8 = _kCache.dtype == INT8;
    XTensor *k = &_kCache;
    XTensor *pe = &_peCache;
    XTensor noBlockTables;
    XTensor *table = &_blockTables;
    if (dense) {
        if (topK == 0) {
            throw std::runtime_error(std::string(__func__) +
                                     ": dense=True requires topK > 0 (dense cache length)");
        }
        XTensor &kDense =
            rt.GetTensor({_qWithQr.shape[0], topK, kvLoraRank}, XDtypeOf(qWithQr), DBG_LOC);
        XTensor &peDense =
            rt.GetTensor({_qWithQr.shape[0], topK, ropeHeadDim}, XDtypeOf(qWithQr), DBG_LOC);
        XliteOpGatherSparseKVCache(rt, _kCache, _peCache, _blockTables, _topkIndices,
                                   _queryStartLoc, _lens, _cachedLens, kDense, peDense, batch, topK,
                                   blockSize, kvLoraRank, ropeHeadDim, 1, 1, GM_UB_BURST_SIZE,
                                   _kScaleCache);
        XTensor &qk =
            rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, topK}, XDtypeOf(qWithQr), DBG_LOC);
        XliteOpMLAV2(rt, qAbsorb, _qr, kDense, peDense, qk, oAbsorb, _queryStartLoc, _lens,
                     _cachedLens, _blockTables, nHeads, ropeHeadDim, kvLoraRank, blockSize, batch,
                     topK, scale, 0, XTensor(), dense, true);
        rt.PutTensor(qk);
        rt.PutTensor(peDense);
        rt.PutTensor(kDense);
    } else if (!enableFlashAttention) {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, maxNumBlocks * blockSize},
                                   XDtypeOf(qWithQr), DBG_LOC);
        if (c8) {
            uint32_t capacity = maxNumBlocks * blockSize;
            k = &rt.GetTensor({batch, capacity, kvLoraRank}, BF16, DBG_LOC);
            pe = &rt.GetTensor({batch, capacity, ropeHeadDim}, BF16, DBG_LOC);
            table = &noBlockTables;
            XTensor noIndices;
            XliteOpGatherSparseKVCache(rt, _kCache, _peCache, _blockTables, noIndices,
                                       _queryStartLoc, _lens, _cachedLens, *k, *pe, batch, capacity,
                                       blockSize, kvLoraRank, ropeHeadDim, 1, 1, GM_UB_BURST_SIZE,
                                       _kScaleCache);
        }
        XliteOpMLAV2(rt, qAbsorb, _qr, *k, *pe, qk, oAbsorb, _queryStartLoc, _lens, _cachedLens,
                     *table, nHeads, ropeHeadDim, kvLoraRank, blockSize, batch,
                     maxNumBlocks * blockSize, scale, topK, _topkIndices, c8, false);
        if (c8) {
            rt.PutTensor(*pe);
            rt.PutTensor(*k);
        }
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, tileSizeOfCachedKV},
                                   XDtypeOf(qWithQr), DBG_LOC);
        XTensor &sv =
            rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, kvLoraRank}, XDtypeOf(qWithQr), DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({_qWithQr.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({_qWithQr.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &sync = rt.GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
        sync.Memset(0, rt.stream);

        if (c8) {
            uint32_t capacity = maxNumBlocks * blockSize;
            k = &rt.GetTensor({batch, capacity, kvLoraRank}, BF16, DBG_LOC);
            pe = &rt.GetTensor({batch, capacity, ropeHeadDim}, BF16, DBG_LOC);
            table = &noBlockTables;
            XTensor noIndices;
            XliteOpGatherSparseKVCache(rt, _kCache, _peCache, _blockTables, noIndices,
                                       _queryStartLoc, _lens, _cachedLens, *k, *pe, batch, capacity,
                                       blockSize, kvLoraRank, ropeHeadDim, 1, 1, GM_UB_BURST_SIZE,
                                       _kScaleCache);
        }
        XliteOpFlashMLAV2(rt, qAbsorb, _qr, *k, *pe, qk, sv, max, sum, lastMax, lastSum, sync,
                          oAbsorb, _queryStartLoc, _lens, _cachedLens, *table, nHeads, ropeHeadDim,
                          kvLoraRank, blockSize, batch, scale, tileSizeOfCachedKV, topK,
                          _topkIndices, c8);
        if (c8) {
            rt.PutTensor(*pe);
            rt.PutTensor(*k);
        }
        rt.PutTensor(sync);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }

    XliteOpEinsumMhtHtdMhd(rt, oAbsorb, _wuv, _output, _qWithQr.shape[0], nHeads, kvLoraRank,
                           vHeadDim);
    rt.Synchronize();
    rt.PutTensor(qAbsorb);
    rt.PutTensor(oAbsorb);
}

void DsaCpGatherMLA(XRuntime &rt, at::Tensor &qAbsorb, at::Tensor &qr, at::Tensor &kCache,
                    at::Tensor &peCache, at::Tensor &output, at::Tensor &queryStartLoc,
                    at::Tensor &queryLens, at::Tensor &cachedLens, at::Tensor &blockTables,
                    at::Tensor &topkIndices, float scale)
{
    XTensor qAbsorbXT, qPeXT, kCacheXT, peCacheXT, outputXT, queryStartLocXT, queryLensXT,
        cachedLensXT, blockTablesXT, topkIndicesXT;
    InitXTensor(qAbsorbXT, qAbsorb);
    InitXTensor(qPeXT, qr);
    InitXTensor(kCacheXT, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(peCacheXT, peCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(outputXT, output);
    InitXTensor(queryStartLocXT, queryStartLoc);
    InitXTensor(queryLensXT, queryLens);
    InitXTensor(cachedLensXT, cachedLens);
    InitXTensor(blockTablesXT, blockTables, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(topkIndicesXT, topkIndices);
    if (qAbsorbXT.shape.size() != 3 || qPeXT.shape.size() != 3 || kCacheXT.shape.size() != 3 ||
        topkIndicesXT.shape.size() != 2) {
        throw std::invalid_argument("DsaCpGatherMLA: expected 3-D Q/KV and 2-D Top-K indices");
    }
    XliteOpDsaCpGatherMLA(rt, qAbsorbXT, qPeXT, kCacheXT, peCacheXT, outputXT, queryStartLocXT,
                          queryLensXT, cachedLensXT, blockTablesXT, topkIndicesXT,
                          qAbsorbXT.shape[1], qPeXT.shape[2], qAbsorbXT.shape[2], kCacheXT.shape[1],
                          queryLensXT.numel, topkIndicesXT.shape[1], scale);
    rt.Synchronize();
}

void GatherSparseKVCache(XRuntime &rt, at::Tensor &kCache, at::Tensor &peCache,
                         at::Tensor &blockTables, at::Tensor &topkIndices,
                         at::Tensor &queryStartLoc, at::Tensor &lens, at::Tensor &cachedLens,
                         at::Tensor &kDenseCache, at::Tensor &peDenseCache, uint32_t batch,
                         uint32_t indexTopK, uint32_t blockSize, uint32_t kvLoraRank,
                         uint32_t ropeHeadDim, uint32_t kvHeads, uint32_t compressRatio,
                         uint32_t burstCopySize, const std::optional<at::Tensor> &kScaleCache)
{
    XTensor _kCache, _peCache, _blockTables, _topkIndices, _queryStartLoc, _lens, _cachedLens,
        _kDenseCache, _peDenseCache, _kScaleCache;
    InitXTensor(_kScaleCache, kScaleCache, DBG_LOC);
    InitXTensor(_kCache, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_peCache, peCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_blockTables, blockTables, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_topkIndices, topkIndices, DBG_LOC);
    InitXTensor(_queryStartLoc, queryStartLoc, DBG_LOC);
    InitXTensor(_lens, lens, DBG_LOC);
    InitXTensor(_cachedLens, cachedLens, DBG_LOC);
    InitXTensor(_kDenseCache, kDenseCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_peDenseCache, peDenseCache, DBG_LOC, /*allowPitched=*/true);
    XliteOpGatherSparseKVCache(rt, _kCache, _peCache, _blockTables, _topkIndices, _queryStartLoc,
                               _lens, _cachedLens, _kDenseCache, _peDenseCache, batch, indexTopK,
                               blockSize, kvLoraRank, ropeHeadDim, kvHeads, compressRatio,
                               burstCopySize, _kScaleCache);
    rt.Synchronize();
}

void CXA(XRuntime &rt, at::Tensor &q, at::Tensor &swaKCache, at::Tensor &compressKCache,
         at::Tensor &swaBlockTables, at::Tensor &compressBlockTables, uint32_t swaBlockSize,
         uint32_t compressBlockSize, at::Tensor &attnSink, at::Tensor &output, uint32_t batch,
         at::Tensor &queryStartLoc, at::Tensor &lens, at::Tensor &cachedLens, uint32_t nHeads,
         uint32_t headDim, float scale, uint32_t windowSize, uint32_t compressRatio,
         uint32_t indexTopK, at::Tensor &topkIndices, bool enableFlashAttention,
         uint32_t tileSizeOfCachedKV, bool dense)
{
    XTensor _q, _swaKCache, _compressKCache, _swaBlockTables, _compressBlockTables, _attnSink,
        _output, _queryStartLoc, _lens, _cachedLens, _topkIndices;
    InitXTensor(_q, q, DBG_LOC);
    InitXTensor(_swaKCache, swaKCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_compressKCache, compressKCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_swaBlockTables, swaBlockTables, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_compressBlockTables, compressBlockTables, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_attnSink, attnSink, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    InitXTensor(_queryStartLoc, queryStartLoc, DBG_LOC);
    InitXTensor(_lens, lens, DBG_LOC);
    InitXTensor(_cachedLens, cachedLens, DBG_LOC);
    InitXTensor(_topkIndices, topkIndices, DBG_LOC);
    uint32_t swaSegWidth = windowSize == 0 ? 0 : windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B;
    // kvSize = compress-segment capacity of the scores workspace (compressed tokens):
    // dense mode holds indexTopK contiguous compressed tokens per batch; sparse mode
    // walks the paged cache (compressMaxNumBlocks * compressBlockSize per batch).
    uint32_t kvSize;
    if (dense) {
        kvSize = compressRatio == 0 ? 0 : ROUND_UP((uint64_t)indexTopK, 4 * CXA_SVCK0);
        XTensor &compressKCacheDense =
            rt.GetTensor({_q.shape[0], indexTopK, headDim}, XDtypeOf(q), DBG_LOC);
        XliteOpGatherSparseKVCache(rt, _compressKCache, XTensor(), _compressBlockTables,
                                   _topkIndices, _queryStartLoc, _lens, _cachedLens,
                                   compressKCacheDense, XTensor(), batch, indexTopK,
                                   compressBlockSize, headDim, 0, 1, compressRatio);
        XTensor &scores = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, swaSegWidth + kvSize},
                                       XDtypeOf(q), DBG_LOC);
        XliteOpCXA(rt, _q, _swaKCache, compressKCacheDense, _swaBlockTables, _compressBlockTables,
                   swaBlockSize, compressBlockSize, _attnSink, scores, _output, batch,
                   _queryStartLoc, _lens, _cachedLens, nHeads, headDim, scale, windowSize, kvSize,
                   compressRatio, indexTopK, XTensor(), true);
        rt.PutTensor(compressKCacheDense);
        rt.PutTensor(scores);
    } else if (enableFlashAttention && compressRatio != 0) {
        uint32_t qkWidth = swaSegWidth + tileSizeOfCachedKV;
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, qkWidth}, XDtypeOf(q), DBG_LOC);
        XTensor &sv = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, headDim}, XDtypeOf(q), DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({_q.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({_q.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &sync = rt.GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
        sync.Memset(0, rt.stream);

        XliteOpFlashCXA(rt, _q, _swaKCache, _compressKCache, _swaBlockTables, _compressBlockTables,
                        swaBlockSize, compressBlockSize, _attnSink, qk, sv, max, sum, lastMax,
                        lastSum, sync, _output, batch, _queryStartLoc, _lens, _cachedLens, nHeads,
                        headDim, scale, windowSize, compressRatio, indexTopK, _topkIndices,
                        tileSizeOfCachedKV);
        rt.PutTensor(sync);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    } else {
        kvSize = compressRatio == 0
                     ? 0
                     : ROUND_UP(DeriveMaxSeqLen(_lens, _cachedLens) / compressRatio, 4 * CXA_SVCK0);
        XTensor &scores = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, swaSegWidth + kvSize},
                                       XDtypeOf(q), DBG_LOC);
        XliteOpCXA(rt, _q, _swaKCache, _compressKCache, _swaBlockTables, _compressBlockTables,
                   swaBlockSize, compressBlockSize, _attnSink, scores, _output, batch,
                   _queryStartLoc, _lens, _cachedLens, nHeads, headDim, scale, windowSize, kvSize,
                   compressRatio, indexTopK, _topkIndices, false);
        rt.PutTensor(scores);
    }
    rt.Synchronize();
}

void AddAndRMSNorm(XRuntime &rt, at::Tensor &in, at::Tensor &addInOut, at::Tensor &norm,
                   at::Tensor &out, float normEps)
{
    XTensor _in, _addInOut, _out, _norm;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_addInOut, addInOut, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_norm, norm, DBG_LOC);
    XliteOpAddAndRmsNorm(rt, _in, _addInOut, _norm, normEps, _out);
    rt.Synchronize();
}

void SoftmaxTopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &outWeights,
                 at::Tensor &outRouting, uint32_t topK, bool normTopKProb)
{
    XTensor _scores, _indices, _outWeights, _outRouting;

    InitXTensor(_scores, scores, DBG_LOC);
    InitXTensor(_indices, indices, DBG_LOC);
    InitXTensor(_outWeights, outWeights, DBG_LOC);
    // .vec() returns a temporary; taking begin()/end() from two separate temporaries is UB
    // (garbage distance -> length_error). Materialise one named vector first.
    auto scoresSizesVec = scores.sizes().vec();
    std::vector<size_t> sizes(scoresSizesVec.begin(), scoresSizesVec.end());
    _outRouting.Init(sizes, BIT1, TensorPtr(outRouting));
    XliteOpSoftmaxTopK(rt, _scores, _indices, _outWeights, _outRouting, topK, normTopKProb);
    rt.Synchronize();
}

void SigmoidTopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &bias,
                 float scale, at::Tensor &outWeights, at::Tensor &outRouting, uint32_t nGroup,
                 uint32_t nTopkGroup, uint32_t topK, bool normTopKProb)
{
    XTensor _scores, _indices, _bias, _outWeights, _outRouting;

    InitXTensor(_scores, scores, DBG_LOC);
    InitXTensor(_bias, bias, DBG_LOC);
    InitXTensor(_indices, indices, DBG_LOC);
    InitXTensor(_outWeights, outWeights, DBG_LOC);
    // Same UB fix as SoftmaxTopK: one named vector, not two temporaries' iterators.
    auto scoresSizesVec = scores.sizes().vec();
    std::vector<size_t> sizes(scoresSizesVec.begin(), scoresSizesVec.end());
    _outRouting.Init(sizes, BIT1, TensorPtr(outRouting));
    XliteOpSigmoidTopK(rt, _scores, _indices, _bias, scale, _outWeights, _outRouting, nGroup,
                       nTopkGroup, topK, normTopKProb);
    rt.Synchronize();
}

void SqrtsoftplusHashTopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &bias,
                          at::Tensor &inputIds, at::Tensor &tid2eid, at::Tensor &outWeights,
                          at::Tensor &routingMap, float scale, uint32_t topK, bool hash)
{
    XTensor _scores, _indices, _bias, _inputIds, _tid2eid, _outWeights, _routingMap;

    InitXTensor(_scores, scores, DBG_LOC);
    InitXTensor(_indices, indices, DBG_LOC);
    InitXTensor(_bias, bias, DBG_LOC);
    InitXTensor(_inputIds, inputIds, DBG_LOC);
    InitXTensor(_tid2eid, tid2eid, DBG_LOC);
    InitXTensor(_outWeights, outWeights, DBG_LOC);
    auto scoresSizesVec = scores.sizes().vec();
    std::vector<size_t> sizes(scoresSizesVec.begin(), scoresSizesVec.end());
    _routingMap.Init(sizes, BIT1, TensorPtr(routingMap));
    XliteOpSqrtsoftplusHashTopK(rt, _scores, _indices, _bias, _inputIds, _tid2eid, _outWeights,
                                _routingMap, scale, topK, hash);
    rt.Synchronize();
}

void TopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &outIndices,
          at::Tensor &queryLens, at::Tensor &cachedLens, size_t k)
{
    XTensor _scores, _indices, _outIndices, _queryLens, _cachedLens;

    InitXTensor(_scores, scores, DBG_LOC);
    InitXTensor(_indices, indices, DBG_LOC);
    InitXTensor(_outIndices, outIndices, DBG_LOC);
    InitXTensor(_queryLens, queryLens, DBG_LOC);
    InitXTensor(_cachedLens, cachedLens, DBG_LOC);

    XliteOpTopK(rt, _scores, _indices, _outIndices, _queryLens, _cachedLens, _queryLens.shape[0],
                k);

    rt.Synchronize();
}

void CastUp(XRuntime &rt, at::Tensor &in, at::Tensor &out)
{
    XTensor _in, _out;
    XTensor &inScale = rt.GetTensor({1}, XDtypeOf(in), DBG_LOC);

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    XliteOpCastUp(rt, _in, inScale, _out);
    rt.Synchronize();
    rt.PutTensor(inScale);
}

void CastDown(XRuntime &rt, at::Tensor &in, at::Tensor &out)
{
    XTensor _in, _out;
    XTensor &outScale = rt.GetTensor({1}, XDtypeOf(in), DBG_LOC);

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    XliteOpCastDown(rt, _in, _out, outScale);
    rt.Synchronize();
    rt.PutTensor(outScale);
}

void Permutation(XRuntime &rt, at::Tensor &in, at::Tensor &routing, uint32_t start, uint32_t end,
                 at::Tensor &out, at::Tensor &unpIdx, at::Tensor &counts,
                 std::optional<at::Tensor> scaleIn, std::optional<at::Tensor> scaleOut)
{
    XTensor _in, _routing, _out, _unpIdx, _counts;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_routing, routing, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_unpIdx, unpIdx, DBG_LOC);
    InitXTensor(_counts, counts, DBG_LOC);
    XTensor _scaleIn, _scaleOut;
    if (scaleIn && scaleIn->defined()) {
        InitXTensor(_scaleIn, *scaleIn, DBG_LOC);
    }
    if (scaleOut && scaleOut->defined()) {
        InitXTensor(_scaleOut, *scaleOut, DBG_LOC);
    }
    XliteOpPermutation(rt, _in, _routing, start, end, _out, _unpIdx, _counts, _scaleIn, _scaleOut);
    rt.Synchronize();
}

void UnPermutation(XRuntime &rt, at::Tensor &in, at::Tensor &routing, at::Tensor &weights,
                   uint32_t start, uint32_t end, at::Tensor &out, at::Tensor &unpIdx)
{
    XTensor _in, _routing, _weights, _out, _unpIdx;
    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_routing, routing, DBG_LOC);
    InitXTensor(_weights, weights, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_unpIdx, unpIdx, DBG_LOC);

    XliteOpUnpermutation(rt, _in, _unpIdx, _routing, _weights, start, end, _out);
    rt.Synchronize();
}

void RemapGateOutputs(XRuntime &rt, at::Tensor &wIn, at::Tensor &rIn, at::Tensor &wOut,
                      at::Tensor &rOut, at::Tensor &log2phy)
{
    XTensor _wIn, _rIn, _wOut, _rOut, _log2phy;
    InitXTensor(_wIn, wIn, DBG_LOC);
    InitXTensor(_rIn, rIn, DBG_LOC);
    InitXTensor(_wOut, wOut, DBG_LOC);
    InitXTensor(_rOut, rOut, DBG_LOC);
    InitXTensor(_log2phy, log2phy, DBG_LOC);
    XliteOpRemapGateOutputs(rt, _wIn, _rIn, _wOut, _rOut, _log2phy);
    rt.Synchronize();
}

void GroupMatmul(XRuntime &rt, at::Tensor &in, std::vector<at::Tensor> &weights,
                 std::vector<at::Tensor> &scales, at::Tensor &counts, uint32_t start, uint32_t end,
                 long outDim, long inDim, at::Tensor &output, bool weightNZ, bool transpose)
{
    (void)weightNZ;  // deprecated no-op
    if (weights.empty()) {
        throw std::invalid_argument(DBG_PREFIX + ": weights must not be empty");
    }
    XTensor _in, _counts, _output, _scalesTensor;
    XTensor *_scales = &_scalesTensor;
    std::vector<void *> p;
    uint32_t i, num = counts.size(0);
    bool hasScale = (scales.size() == num);

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_counts, counts, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    if (!weights.empty() && XDtypeOf(weights[0]) == INT32 &&
        (_in.dtype == INT32 || _in.dtype == INT8)) {
        // for the w4a8 matmul, activation and weight will be viewed as int4
        _in.View(INT4);
    }
    XTensor &_weights = rt.GetTensor({num}, INT64, DBG_LOC);
    const bool expertNZ = IsNpuNzFormat(weights[0]);

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

    enum XDtype weightDtype = XDtypeOf(weights[0]);
    if (weightDtype == INT32) {
        weightDtype = INT4;
    }
    XliteOpGroupMatmul(rt, _in, _weights, *_scales, _counts, start, end, weightDtype, outDim, inDim,
                       _output, expertNZ, transpose);
    rt.Synchronize();
    rt.PutTensor(_weights);
    if (hasScale) {
        rt.PutTensor(*_scales);
    }
}

void Softmax(XRuntime &rt, at::Tensor &x, uint32_t calcLen, bool isLong)
{
    XTensor _x;
    InitXTensor(_x, x, DBG_LOC);
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
                 at::Tensor &inputWithR, at::Tensor &freqs, at::Tensor &position,
                 at::Tensor &output, bool inverse, bool outInterleaved, uint64_t minPosition = 0,
                 bool doRotate = false)
{
    XTensor _inputWithR, _freqs, _position, _output;
    InitXTensor(_inputWithR, inputWithR, DBG_LOC);
    InitXTensor(_freqs, freqs, DBG_LOC);
    InitXTensor(_position, position, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    // outStepDim = ropeDim: the standalone op writes only the rope region. offset/outOffset keep
    // the rope region at the head tail ([nope | rope]); the caller scales the nope region via muls.
    uint32_t outStepDim = ropeDim;
    XliteOpRopeComplex(rt, nLocalHeads, stepDim, outStepDim, ropeDim, stepDim - ropeDim, 0,
                       _inputWithR, _freqs, _position, _output, inverse, outInterleaved,
                       minPosition, doRotate);
    rt.Synchronize();
}

void RopeComplexAndCache(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim, uint32_t ropeDim,
                         uint32_t offset, uint32_t vdim, at::Tensor &inputWithR, at::Tensor &freqs,
                         at::Tensor &position, uint32_t blockSize, at::Tensor &vCache,
                         at::Tensor &slotMapping, bool outInterleaved, bool doRotate = false)
{
    XTensor _inputWithR, _freqs, _position, _vCache, _slotMapping;
    InitXTensor(_inputWithR, inputWithR, DBG_LOC);
    InitXTensor(_freqs, freqs, DBG_LOC);
    InitXTensor(_position, position, DBG_LOC);
    InitXTensor(_vCache, vCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_slotMapping, slotMapping, DBG_LOC);
    XliteOpRopeComplexAndCache(rt, nLocalHeads, stepDim, ropeDim, offset, vdim, _inputWithR, _freqs,
                               _position, blockSize, _vCache, _slotMapping, outInterleaved,
                               doRotate);
    rt.Synchronize();
}

void MlaPrepare(XRuntime &rt, at::Tensor &attnQkvc, at::Tensor &qNorm, at::Tensor &qNormBias,
                at::Tensor &attnNormQc, at::Tensor &kvNorm, at::Tensor &kvNormBias,
                at::Tensor &attnNormKvc, at::Tensor &freqs, at::Tensor &position,
                uint32_t qLoraRank, uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t blockSize,
                at::Tensor &kCache, at::Tensor &peCache, at::Tensor &slotMapping, float normEps,
                const std::optional<at::Tensor> &kScaleCache)
{
    XTensor _attnQkvc, _qNorm, _qNormBias, _attnNormQc, _kvNorm, _kvNormBias, _attnNormKvc, _freqs,
        _position, _kCache, _peCache, _slotMapping, _kScaleCache;
    InitXTensor(_kScaleCache, kScaleCache, DBG_LOC);
    InitXTensor(_attnQkvc, attnQkvc, DBG_LOC);
    InitXTensor(_qNorm, qNorm, DBG_LOC);
    InitXTensor(_qNormBias, qNormBias, DBG_LOC);
    InitXTensor(_attnNormQc, attnNormQc, DBG_LOC);
    InitXTensor(_kvNorm, kvNorm, DBG_LOC);
    InitXTensor(_kvNormBias, kvNormBias, DBG_LOC);
    InitXTensor(_attnNormKvc, attnNormKvc, DBG_LOC);
    InitXTensor(_freqs, freqs, DBG_LOC);
    InitXTensor(_position, position, DBG_LOC);
    InitXTensor(_kCache, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_peCache, peCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_slotMapping, slotMapping, DBG_LOC);
    XliteOpMlaPrepare(rt, _attnQkvc, _qNorm, _qNormBias, _attnNormQc, _kvNorm, _kvNormBias, _freqs,
                      _position, qLoraRank, kvLoraRank, ropeHeadDim, blockSize, _kCache, _peCache,
                      _slotMapping, normEps, _attnNormKvc, _kScaleCache);
    rt.Synchronize();
}

void MlaCacheC8(XRuntime &rt, at::Tensor &latent, at::Tensor &pe, at::Tensor &slotMapping,
                at::Tensor &cache, at::Tensor &peCache, at::Tensor &kScaleCache)
{
    XTensor _latent, _pe, _slots, _cache, _peCache, _kScaleCache;
    InitXTensor(_latent, latent);
    InitXTensor(_pe, pe);
    InitXTensor(_slots, slotMapping);
    InitXTensor(_cache, cache);
    InitXTensor(_peCache, peCache);
    InitXTensor(_kScaleCache, kScaleCache);
    XliteOpMlaCacheC8(rt, _latent, _pe, _slots, _cache, _peCache, _kScaleCache);
    rt.Synchronize();
}

void IndexerPrepare(XRuntime &rt, at::Tensor &kw, at::Tensor &kNorm, at::Tensor &kNormBias,
                    at::Tensor &freqs, at::Tensor &position, uint32_t indexHeadDim,
                    uint32_t indexNHeads, uint32_t ropeHeadDim, uint32_t blockSize,
                    at::Tensor &indexKCache, at::Tensor &slotMapping, float normEps, at::Tensor &q,
                    float scale, uint32_t topK, bool isLong,
                    const std::optional<at::Tensor> &kScaleCache = std::nullopt,
                    const std::optional<at::Tensor> &q8 = std::nullopt,
                    const std::optional<at::Tensor> &qScale = std::nullopt,
                    const std::optional<at::Tensor> &scaledWeights = std::nullopt)
{
    XTensor _kw, _kNorm, _kNormBias, _freqs, _position, _indexKCache, _slotMapping, _q;
    XTensor _ks, _q8, _qs, _sw;
    InitXTensor(_ks, kScaleCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_q8, q8, DBG_LOC);
    InitXTensor(_qs, qScale, DBG_LOC);
    InitXTensor(_sw, scaledWeights, DBG_LOC);
    InitXTensor(_kw, kw, DBG_LOC);
    InitXTensor(_kNorm, kNorm, DBG_LOC);
    InitXTensor(_kNormBias, kNormBias, DBG_LOC);
    InitXTensor(_freqs, freqs, DBG_LOC);
    InitXTensor(_position, position, DBG_LOC);
    InitXTensor(_indexKCache, indexKCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_slotMapping, slotMapping, DBG_LOC);
    InitXTensor(_q, q, DBG_LOC);
    XliteOpIndexerPrepare(rt, _kw, _kNorm, _kNormBias, _freqs, _position, indexHeadDim, indexNHeads,
                          ropeHeadDim, blockSize, _indexKCache, _slotMapping, normEps, _q, scale,
                          topK, isLong, 1, _ks, _q8, _qs, _sw);
    rt.Synchronize();
}

void IndexerKCacheC8(XRuntime &rt, const at::Tensor &k, const at::Tensor &kNorm,
                     const at::Tensor &kNormBias, const at::Tensor &freqs,
                     const at::Tensor &position, const at::Tensor &slotMapping, at::Tensor &kCache,
                     at::Tensor &scaleCache, float normEps)
{
    XTensor x, w, b, f, pTensor, sTensor, kc, sc;
    InitXTensor(x, k, DBG_LOC);
    InitXTensor(w, kNorm, DBG_LOC);
    InitXTensor(b, kNormBias, DBG_LOC);
    InitXTensor(f, freqs, DBG_LOC);
    InitXTensor(pTensor, position, DBG_LOC);
    InitXTensor(sTensor, slotMapping, DBG_LOC);
    InitXTensor(kc, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(sc, scaleCache, DBG_LOC, /*allowPitched=*/true);
    XliteOpIndexerKCacheC8(rt, x, w, b, f, pTensor, sTensor, kc, sc, normEps);
    rt.Synchronize();
}

void QuantCacheC8(XRuntime &rt, const at::Tensor &x, const at::Tensor &slotMapping,
                  at::Tensor &kCache, at::Tensor &kScaleCache, uint32_t blockSize)
{
    XTensor _x, _slotMapping, _kCache, _kScaleCache;
    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_slotMapping, slotMapping, DBG_LOC);
    // The DSV4 pool hands us page-strided as_strided views (K and scales packed
    // per page); quant_cache_c8 addresses both via stride[0].
    InitXTensor(_kCache, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_kScaleCache, kScaleCache, DBG_LOC, /*allowPitched=*/true);
    XliteOpQuantCacheC8(rt, _x, _slotMapping, _kCache, _kScaleCache, blockSize);
    rt.Synchronize();
}

void QuantQC8(XRuntime &rt, at::Tensor &q, at::Tensor &headWeight, at::Tensor &q8,
              at::Tensor &scaledWeights, float rotateScale)
{
    XTensor _q, _headWeight, _q8, _scaledWeights;
    InitXTensor(_q, q, DBG_LOC);
    InitXTensor(_headWeight, headWeight, DBG_LOC);
    InitXTensor(_q8, q8, DBG_LOC);
    InitXTensor(_scaledWeights, scaledWeights, DBG_LOC);
    XliteOpQuantQC8(rt, _q, _headWeight, _q8, _scaledWeights, rotateScale);
    rt.Synchronize();
}

void Compressor(XRuntime &rt, at::Tensor &kv, at::Tensor &score, std::optional<at::Tensor> ape,
                std::optional<at::Tensor> norm, std::optional<at::Tensor> freqs,
                at::Tensor &weightedSum, at::Tensor &queryStartLoc, at::Tensor &queryLens,
                std::optional<at::Tensor> cachedLens, uint32_t batch, uint32_t nTotalBlocks,
                uint32_t ratio, uint32_t overlap, uint32_t headDim, uint32_t ropeHeadDim,
                float normEps, std::optional<at::Tensor> compressKv, at::Tensor &compressPositions,
                at::Tensor &compressSlotMapping, uint32_t compressBlockSize,
                std::optional<at::Tensor> state, std::optional<at::Tensor> stateSlotMapping,
                std::optional<at::Tensor> stateBlockTable, uint32_t stateBlockSize,
                bool doRotate = false)
{
    XTensor _kv, _score, _ape, _norm, _freqs, _weightedSum, _compressPositions,
        _compressSlotMapping, _queryStartLoc, _queryLens, _cachedLens, _compressKv, _state,
        _stateSlotMapping, _stateBlockTable;
    InitXTensor(_kv, kv, DBG_LOC);
    InitXTensor(_score, score, DBG_LOC);
    if (ape.has_value()) {
        InitOptionalXTensor(_ape, ape.value(), DBG_LOC);
    }
    if (norm.has_value()) {
        InitOptionalXTensor(_norm, norm.value(), DBG_LOC);
    }
    if (freqs.has_value()) {
        InitOptionalXTensor(_freqs, freqs.value(), DBG_LOC);
    }
    InitXTensor(_weightedSum, weightedSum, DBG_LOC);
    InitXTensor(_compressPositions, compressPositions, DBG_LOC);
    InitXTensor(_compressSlotMapping, compressSlotMapping, DBG_LOC);
    InitXTensor(_queryStartLoc, queryStartLoc, DBG_LOC);
    InitXTensor(_queryLens, queryLens, DBG_LOC);
    if (cachedLens.has_value()) {
        InitOptionalXTensor(_cachedLens, cachedLens.value(), DBG_LOC);
    }
    if (compressKv.has_value()) {
        InitOptionalXTensor(_compressKv, compressKv.value(), DBG_LOC, /*allowPitched=*/true);
    }
    if (state.has_value()) {
        InitOptionalXTensor(_state, state.value(), DBG_LOC, /*allowPitched=*/true);
    }
    if (stateSlotMapping.has_value()) {
        InitOptionalXTensor(_stateSlotMapping, stateSlotMapping.value(), DBG_LOC);
    }
    if (stateBlockTable.has_value()) {
        InitOptionalXTensor(_stateBlockTable, stateBlockTable.value(), DBG_LOC);
    }
    XliteOpCompressor(rt, _kv, _score, _ape, _norm, _freqs, _weightedSum, _queryStartLoc,
                      _queryLens, _cachedLens, batch, nTotalBlocks, ratio, overlap, headDim,
                      ropeHeadDim, normEps, _compressKv, _compressPositions, _compressSlotMapping,
                      compressBlockSize, _state, _stateSlotMapping, _stateBlockTable,
                      stateBlockSize, doRotate);
    rt.Synchronize();
}

void Quant(XRuntime &rt, at::Tensor &x, at::Tensor &scaleReciprocal, at::Tensor &offset,
           at::Tensor &out)
{
    XTensor _x, _scaleRec, _offset, _out;
    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_scaleRec, scaleReciprocal, DBG_LOC);
    InitXTensor(_offset, offset, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    XliteOpQuant(rt, _x, _scaleRec, _offset, _out);
    rt.Synchronize();
}

void QuantDyn(XRuntime &rt, at::Tensor &x, at::Tensor &scale, at::Tensor &out)
{
    XTensor _x, _scale, _out;
    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_scale, scale, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    XliteOpQuantDyn(rt, _x, _scale, _out);
    rt.Synchronize();
}

void MSDMergeDequant(XRuntime &rt, at::Tensor &yMerged, std::vector<at::Tensor> &scaleBiases,
                     at::Tensor &counts, at::Tensor &perTokenScale, at::Tensor &out)
{
    XTensor _yMerged, _perTokenScale, _out, _counts;
    InitXTensor(_yMerged, yMerged, DBG_LOC);
    InitXTensor(_counts, counts, DBG_LOC);
    InitXTensor(_perTokenScale, perTokenScale, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);

    // Build an INT64 [numExperts] pointer array of per-expert scale_bias pointers
    uint32_t numExperts = scaleBiases.size();
    if (numExperts == 0) {
        throw std::invalid_argument("group matmul need at least one expert");
    }
    XTensor &_scaleBiasPtrs = rt.GetTensor({numExperts}, INT64, DBG_LOC);
    std::vector<void *> p(numExperts);
    for (uint32_t i = 0; i < numExperts; i++) {
        p[i] = TensorPtr(scaleBiases[i]);
    }
    rt.MemcpyH2D(_scaleBiasPtrs.ptr, reinterpret_cast<void *>(p.data()),
                 numExperts * sizeof(void *));

    XliteOpMSDMergeDequant(rt, _yMerged, _scaleBiasPtrs, _counts, 0, numExperts, _perTokenScale,
                           _out);
    rt.Synchronize();
    rt.PutTensor(_scaleBiasPtrs);
}

void MatmulDeQuant(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &bias,
                   at::Tensor &deqScale, at::Tensor &z, bool weightNZ, bool transpose)
{
    (void)weightNZ;  // deprecated no-op
    XTensor _x, _y, _z, _bias, _deqScale;

    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_y, y, DBG_LOC);
    InitXTensor(_bias, bias, DBG_LOC);
    InitXTensor(_deqScale, deqScale, DBG_LOC);
    InitXTensor(_z, z, DBG_LOC);
    if (_x.dtype == INT32 && _y.dtype == INT32) {  // for int4 matmul
        _x.View(INT4);
        _y.View(INT4);
    }
    XliteOpMatmul(rt, _x, _y, _z, _bias, _deqScale, transpose);
    rt.Synchronize();
}

void DeQuant(XRuntime &rt, at::Tensor &in, at::Tensor &scale, at::Tensor &out, bool hasScale)
{
    XTensor _in, _scale, _out;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    if (hasScale) {
        InitXTensor(_scale, scale, DBG_LOC);
    }
    XliteOpDeQuant(rt, _in, _out, _scale);
    rt.Synchronize();
}

void FusionOperatorMatmulDequantPipeline(XRuntime &rt, at::Tensor &x, at::Tensor &weight,
                                         at::Tensor &out, at::Tensor &bias, at::Tensor &deqScale,
                                         bool weightNZ, bool transpose, at::Tensor &outScale,
                                         at::Tensor &num)
{
    (void)weightNZ;  // deprecated no-op
    XTensor _x, _weight, _out, _bias, _deqScale, _outScale, _num;

    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_weight, weight, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_bias, bias, DBG_LOC);
    InitXTensor(_deqScale, deqScale, DBG_LOC);
    if (outScale.numel() > 0) {
        InitXTensor(_outScale, outScale, DBG_LOC);
    }
    if (num.numel() > 0) {
        InitXTensor(_num, num, DBG_LOC);
    }
    XliteOpFusionOperatorMatmulDequantPipeline(rt, _x, _weight, _out, _bias, _deqScale, transpose,
                                               _outScale, _num);
    rt.Synchronize();
}
void FusedQuantMatmulDequant(XRuntime &rt, at::Tensor &x,
                             const std::optional<at::Tensor> &quantScaleRec,
                             const std::optional<at::Tensor> &quantOffset, at::Tensor &weight,
                             at::Tensor &out, const std::optional<at::Tensor> &bias,
                             const std::optional<at::Tensor> &deqScale, at::Tensor &xQuant,
                             const std::optional<at::Tensor> &outScale,
                             const std::optional<at::Tensor> &num, bool dynQuant, bool weightNZ,
                             bool transpose)
{
    (void)weightNZ;  // deprecated no-op
    XTensor _x, _quantScaleRec, _quantOffset, _weight, _out, _bias, _deqScale, _xQuant, _outScale,
        _num;

    // The std::optional overload of InitXTensor is a no-op for nullopt, which
    // leaves the XTensor ptr-null ("absent") for the op-level gates.
    InitXTensor(_x, x);
    InitXTensor(_quantScaleRec, quantScaleRec);
    InitXTensor(_quantOffset, quantOffset);
    InitXTensor(_weight, weight);
    InitXTensor(_out, out);
    InitXTensor(_xQuant, xQuant);
    InitXTensor(_bias, bias);
    InitXTensor(_deqScale, deqScale);
    InitXTensor(_outScale, outScale);
    InitXTensor(_num, num);
    XliteOpFusedQuantMatmulDequant(rt, _x, _quantScaleRec, _quantOffset, _weight, _out, _bias,
                                   _deqScale, _xQuant, _outScale, _num, dynQuant, transpose);
    rt.Synchronize();
}

void FusedGroupMatmulDequant(XRuntime &rt, at::Tensor &in, std::vector<at::Tensor> &weights,
                             std::vector<at::Tensor> &scales, at::Tensor &counts, uint32_t start,
                             uint32_t end, long outDim, long inDim, at::Tensor &output,
                             const std::optional<at::Tensor> &outScale, bool weightNZ,
                             bool transpose)
{
    if (weights.empty()) {
        throw std::invalid_argument(DBG_PREFIX + ": weights must not be empty");
    }
    XTensor _in, _counts, _output, _outScale;
    XTensor _scalesTensor;
    XTensor *_scales = &_scalesTensor;
    std::vector<void *> p;
    uint32_t i, num = counts.size(0);
    bool hasScale = (scales.size() == num);

    InitXTensor(_in, in);
    InitXTensor(_counts, counts);
    InitXTensor(_output, output);
    // Same contract as fused_quant_matmul_dequant's optional tensors: the
    // std::optional InitXTensor overload is a no-op for nullopt, leaving the
    // XTensor ptr-null — the kernel's no-per-token-scale (hasScale=false)
    // path; an explicitly passed empty tensor reaches the same state.
    InitXTensor(_outScale, outScale);
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

    enum XDtype weightDtype = XDtypeOf(weights[0]);
    if (weightDtype == INT32) {
        weightDtype = INT4;
    }
    XliteOpFusedGroupMatmulDequant(rt, _in, _weights, *_scales, _counts, start, end, weightDtype,
                                   outDim, inDim, _output, _outScale, weightNZ, transpose);
    rt.Synchronize();
    rt.PutTensor(_weights);
    if (hasScale) {
        rt.PutTensor(*_scales);
    }
}

void IndexerScores(XRuntime &rt, at::Tensor &q, at::Tensor &kCache, at::Tensor &weight,
                   at::Tensor &scores, at::Tensor &queryStartLoc, at::Tensor &lens,
                   at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads,
                   uint32_t headDim, uint32_t blockSize, uint32_t batch)
{
    XTensor _q, _kCache, _weight, _scores, _queryStartLoc, _lens, _cachedLens, _blockTables;

    InitXTensor(_q, q, DBG_LOC);
    InitXTensor(_kCache, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_weight, weight, DBG_LOC);
    InitXTensor(_scores, scores, DBG_LOC);
    InitXTensor(_queryStartLoc, queryStartLoc, DBG_LOC);
    InitXTensor(_lens, lens, DBG_LOC);
    InitXTensor(_cachedLens, cachedLens, DBG_LOC);
    InitXTensor(_blockTables, blockTables, DBG_LOC, /*allowPitched=*/true);
    SliceBlockTableToLens(_blockTables, _lens, _cachedLens, blockSize);
    XliteOpIndexerScores(rt, _q, _kCache, _weight, _scores, _queryStartLoc, _lens, _cachedLens,
                         _blockTables, nHeads, headDim, blockSize, batch);
    rt.Synchronize();
}

void IndexerTopK(XRuntime &rt, at::Tensor &q, at::Tensor &kCache, at::Tensor &weight,
                 at::Tensor &topkIndices, at::Tensor &queryStartLoc, at::Tensor &lens,
                 at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads, uint32_t headDim,
                 uint32_t blockSize, uint32_t batch, uint32_t topK,
                 const std::optional<at::Tensor> &kScaleCache = std::nullopt,
                 uint32_t compressRatio = 1, uint8_t kvSplit = 0, uint8_t qTile = 0,
                 uint8_t packTail = 0, uint8_t reluQk = 1, uint32_t outStride = 0)
{
    XTensor _q, _kCache, _weight, _topkIndices, _queryStartLoc, _lens, _cachedLens, _blockTables,
        _ks;
    // The int8 branch reads the fp16 scale pool page-strided (DSV4 packs K and
    // scales per page); dense pools keep the contiguous fast path.
    InitXTensor(_ks, kScaleCache, DBG_LOC, /*allowPitched=*/true);

    InitXTensor(_q, q, DBG_LOC);
    InitXTensor(_kCache, kCache, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_weight, weight, DBG_LOC);
    InitXTensor(_topkIndices, topkIndices, DBG_LOC);
    InitXTensor(_queryStartLoc, queryStartLoc, DBG_LOC);
    InitXTensor(_lens, lens, DBG_LOC);
    InitXTensor(_cachedLens, cachedLens, DBG_LOC);
    InitXTensor(_blockTables, blockTables, DBG_LOC, /*allowPitched=*/true);

    if (qTile > MAX_INDEXER_Q_TILE_LEN) {
        throw std::runtime_error(std::string(__func__) + ": qTile must be in [0, " +
                                 std::to_string(MAX_INDEXER_Q_TILE_LEN) + "], got " +
                                 std::to_string(qTile));
    }

    uint32_t maxNumBlock =
        SliceBlockTableToLens(_blockTables, _lens, _cachedLens, blockSize, compressRatio);
    // generate the identity index table (0..max(maxSeqLen,topK)-1)
    XTensor seqPositions;
    std::vector<uint32_t> host;
    size_t maxSeqLen = static_cast<size_t>(blockSize) * maxNumBlock;
    maxSeqLen = INIT_MIN_SEQ_POS > maxSeqLen ? INIT_MIN_SEQ_POS : maxSeqLen;
    size_t len = maxSeqLen > topK ? maxSeqLen : topK;
    size_t size = len * sizeof(uint32_t);
    host.resize(len);
    for (uint32_t i = 0; i < len; i++) {
        host[i] = i;
    }
    void *ptr;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMemcpy(ptr, size, host.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    seqPositions.Init({len}, INT32, ptr);

    XTensor &scores =
        rt.GetTensor({2 * rt.aicNum * MAX_INDEXER_Q_TILE_LEN, MAX_INDEXER_KV_TILE_LEN},
                     q.scalar_type() == at::kChar ? FP32 : XDtypeOf(q), DBG_LOC);
    XTensor &lastTopk = rt.GetTensor({_q.shape[0], 2 * topK}, INT32, DBG_LOC);
    XTensor &sync = rt.GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
    sync.Memset(0, rt.stream);

    XliteOpIndexerTopK(rt, _q, _kCache, _weight, scores, lastTopk, seqPositions, _topkIndices,
                       _queryStartLoc, _lens, _cachedLens, _blockTables, sync, nHeads, headDim,
                       blockSize, batch, topK, 0, _ks, compressRatio, kvSplit, qTile, packTail,
                       reluQk, outStride);
    rt.Synchronize();
    rt.PutTensor(sync);
    rt.PutTensor(lastTopk);
    rt.PutTensor(scores);
    (void)aclrtFree(seqPositions.ptr);
}

void Muls(XRuntime &rt, at::Tensor &input, float scale, at::Tensor &output, uint32_t calcOffset = 0,
          uint32_t calcNum = UINT32_MAX)
{
    XTensor _input, _output;
    InitXTensor(_input, input, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    XliteOpMuls(rt, _input, scale, _output, calcOffset, calcNum);
    rt.Synchronize();
}

void ExpertsCountsSum(XRuntime &rt, at::Tensor &expertsCountsInput, at::Tensor &tokensPerEpgroup,
                      at::Tensor &expertsCountsOutput, uint32_t nRoutedExperts)
{
    XTensor _expertsCountsInput, _tokensPerEpgroup, _expertsCountsOutput;

    InitXTensor(_expertsCountsInput, expertsCountsInput, DBG_LOC);
    InitXTensor(_tokensPerEpgroup, tokensPerEpgroup, DBG_LOC);
    InitXTensor(_expertsCountsOutput, expertsCountsOutput, DBG_LOC);
    XliteOpExpertsCountsSum(rt, _expertsCountsInput, _tokensPerEpgroup, _expertsCountsOutput,
                            nRoutedExperts);
    rt.Synchronize();
}

void ReorderMoE(XRuntime &rt, at::Tensor &in, at::Tensor &out, at::Tensor &counts,
                uint32_t hiddenSize, uint32_t localStart, uint32_t localEnd, bool forward)
{
    XTensor _in, _out, _counts;

    InitXTensor(_in, in, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    InitXTensor(_counts, counts, DBG_LOC);
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

    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_W_qkv, W_qkv, DBG_LOC);
    InitXTensor(_W_z, W_z, DBG_LOC);
    InitXTensor(_W_b, W_b, DBG_LOC);
    InitXTensor(_W_a, W_a, DBG_LOC);

    InitXTensor(_mix_qkv, mix_qkv, DBG_LOC);
    InitXTensor(_z, z, DBG_LOC);
    InitXTensor(_b, b, DBG_LOC);
    InitXTensor(_a, a, DBG_LOC);

    std::vector<XTensor> inputs = {_W_qkv, _W_z, _W_b, _W_a};
    XTensor &W = rt.GetTensor({k, n + v + h + h}, XDtypeOf(x), DBG_LOC);
    XliteOpConcatCol(rt, inputs, W);

    XTensor &out = rt.GetTensor({m, n + v + h + h}, XDtypeOf(x), DBG_LOC);
    XliteOpMatmul(rt, _x, W, out, bias, deqScale, true);
    rt.PutTensor(W);

    std::vector<XTensor> outputs = {_mix_qkv, _z, _b, _a};
    XliteOpSplitCol(rt, out, outputs);
    rt.PutTensor(out);
    rt.Synchronize();
}

void Transpose_1_2(XRuntime &rt, at::Tensor &input, at::Tensor &output)
{
    XTensor _input, _output;
    InitXTensor(_input, input, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    XliteOpTranspose_1_2(rt, _input, _output);
    rt.Synchronize();
}

static void CheckGdnDevices(XRuntime &rt, std::initializer_list<const at::Tensor *> tensors)
{
    rt.SetCurrentContext();
    int32_t device;
    CHECK_ACL(aclrtGetDevice(&device));
    for (const at::Tensor *tensor : tensors) {
        if (tensor != nullptr && (tensor->device().type() != c10::DeviceType::PrivateUse1 ||
                                  tensor->device().index() != device)) {
            throw std::runtime_error("paged GDN tensors must be on the runtime NPU");
        }
    }
}

void LinearAttConv1dAndSiLUPaged(XRuntime &rt, at::Tensor &input, at::Tensor &state,
                                 at::Tensor &weight, at::Tensor &output, at::Tensor &indices,
                                 const std::optional<at::Tensor> &cachedLens,
                                 uint32_t stateBlockSize, bool aligned, uint32_t stateLayout,
                                 uint32_t chanOffset)
{
    CheckGdnDevices(rt, {&input, &state, &weight, &output, &indices,
                         cachedLens.has_value() ? &cachedLens.value() : nullptr});
    if (!input.is_contiguous() || !weight.is_contiguous() || !output.is_contiguous()) {
        throw std::runtime_error("paged conv requires contiguous activation, weight and output");
    }
    XTensor x, c, w, y, pages;
    InitXTensor(x, input);
    InitXTensor(c, state, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(w, weight);
    InitXTensor(y, output);
    InitXTensor(pages, indices, DBG_LOC, /*allowPitched=*/true);
    XTensor cached;
    if (cachedLens.has_value()) {
        InitXTensor(cached, cachedLens.value());
    }
    XliteOpConv1dAndSiLUPaged(rt, c, x, w, y, pages, cachedLens.has_value() ? &cached : nullptr,
                              stateBlockSize, aligned, stateLayout, chanOffset);
    rt.Synchronize();
}

void SplitCol(XRuntime &rt, at::Tensor &in, std::vector<at::Tensor> &outputs)
{
    XTensor _in;
    std::vector<XTensor> _outputs;
    InitXTensor(_in, in, DBG_LOC);
    uint32_t totalWidth = 0;
    for (auto &output : outputs) {
        XTensor x;
        InitXTensor(x, output, DBG_LOC);
        _outputs.push_back(x);
        totalWidth += x.shape.back();
    }
    if (totalWidth != _in.shape.back()) {
        throw std::runtime_error("input last dim != sum(outputs last dim)");
    }
    XliteOpSplitCol(rt, _in, _outputs);
    rt.Synchronize();
}

void Concat(XRuntime &rt, std::vector<at::Tensor> &inputs, at::Tensor &out)
{
    if (inputs.empty()) {
        return;
    }
    std::vector<XTensor> _inputs;
    for (auto &input : inputs) {
        XTensor x;
        InitXTensor(x, input, DBG_LOC);
        _inputs.push_back(x);
    }
    XTensor _out;
    InitXTensor(_out, out, DBG_LOC);

    // Validate byte sizes: the output must hold exactly the concatenated bytes.
    size_t inBytes = 0;
    for (auto &x : _inputs) {
        inBytes += x.bytes;
    }
    size_t outBytes = _out.bytes;
    if (inBytes != outBytes) {
        throw std::runtime_error(std::string(__func__) + ": output bytes (" +
                                 std::to_string(outBytes) + ") != sum(inputs bytes) (" +
                                 std::to_string(inBytes) + ")");
    }
    XliteOpConcat(rt, _inputs, _out);
    rt.Synchronize();
}

void ConcatCol(XRuntime &rt, std::vector<at::Tensor> &inputs, at::Tensor &out)
{
    if (inputs.empty()) {
        return;
    }
    std::vector<XTensor> _inputs;
    size_t totalWidth = 0;
    for (auto &input : inputs) {
        XTensor x;
        InitXTensor(x, input, DBG_LOC);
        _inputs.push_back(x);
        totalWidth += x.shape.back();
    }
    XTensor _out;
    InitXTensor(_out, out, DBG_LOC);
    if (totalWidth != _out.shape.back()) {
        throw std::runtime_error(std::string(__func__) + ": out last dim (" +
                                 std::to_string(_out.shape.back()) + ") != sum(inputs last dim) (" +
                                 std::to_string(totalWidth) + ")");
    }
    XliteOpConcatCol(rt, _inputs, _out);
    rt.Synchronize();
}

void Split(XRuntime &rt, at::Tensor &in, std::vector<at::Tensor> &outputs,
           std::vector<uint32_t> &sizes, uint32_t numPackets)
{
    if (outputs.empty()) {
        return;
    }
    if (outputs.size() != sizes.size()) {
        throw std::runtime_error(std::string(__func__) + ": outputs count (" +
                                 std::to_string(outputs.size()) + ") != sizes count (" +
                                 std::to_string(sizes.size()) + ")");
    }
    XTensor _in;
    std::vector<XTensor> _outputs;
    InitXTensor(_in, in, DBG_LOC);
    for (auto &output : outputs) {
        XTensor x;
        InitXTensor(x, output, DBG_LOC);
        _outputs.push_back(x);
    }
    std::vector<size_t> _sizes(sizes.begin(), sizes.end());

    size_t totalSize = 0;
    for (size_t s : _sizes) {
        totalSize += s;
    }
    size_t inBytes = _in.bytes;
    if (totalSize * numPackets != inBytes) {
        throw std::runtime_error(std::string(__func__) + ": input bytes (" +
                                 std::to_string(inBytes) + ") != totalSize*numPackets (" +
                                 std::to_string(totalSize * numPackets) + ")");
    }
    // Validate each output holds numPackets * sizes[j] bytes; otherwise the
    // kernel writes out-of-bounds and corrupts adjacent device memory.
    for (size_t j = 0; j < _outputs.size(); j++) {
        size_t need = _sizes[j] * numPackets;
        size_t cap = _outputs[j].bytes;
        if (need > cap) {
            throw std::runtime_error(std::string(__func__) + ": output[" + std::to_string(j) +
                                     "] bytes (" + std::to_string(cap) + ") < numPackets*sizes[" +
                                     std::to_string(j) + "] (" + std::to_string(need) + ")");
        }
    }
    XliteOpSplit(rt, _in, _outputs, _sizes, numPackets);
    rt.Synchronize();
}

void BetaDecay(XRuntime &rt, at::Tensor &b, at::Tensor &a, at::Tensor &A_log, at::Tensor &dt_bias,
               at::Tensor &beta, at::Tensor &g, uint32_t bsz, uint32_t seqlen, uint32_t num_v_heads)
{
    XTensor _b, _a, _A_log, _dt_bias, _beta, _g;
    InitXTensor(_b, b, DBG_LOC);
    InitXTensor(_a, a, DBG_LOC);
    InitXTensor(_A_log, A_log, DBG_LOC);
    InitXTensor(_dt_bias, dt_bias, DBG_LOC);
    InitXTensor(_beta, beta, DBG_LOC);
    InitXTensor(_g, g, DBG_LOC);

    XliteOpBetaDecay(rt, _b, _a, _A_log, _dt_bias, _beta, _g, bsz, seqlen, num_v_heads);
    rt.Synchronize();
}

void RecurrentGatedDeltaRulePaged(XRuntime &rt, at::Tensor &query, at::Tensor &key,
                                  at::Tensor &value, at::Tensor &beta, at::Tensor &g,
                                  at::Tensor &state, at::Tensor &out, at::Tensor &state_indices,
                                  uint32_t num_key_heads, uint32_t num_value_heads, uint32_t k_dim,
                                  uint32_t v_dim, const std::optional<at::Tensor> &cachedLens,
                                  uint32_t stateBlockSize, bool aligned)
{
    CheckGdnDevices(rt, {&query, &key, &value, &beta, &g, &state, &out, &state_indices,
                         cachedLens.has_value() ? &cachedLens.value() : nullptr});
    XTensor q, k, v, b, decay, s, o, indices;
    InitXTensor(q, query);
    InitXTensor(k, key);
    InitXTensor(v, value);
    InitXTensor(b, beta);
    InitXTensor(decay, g);
    InitXTensor(s, state, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(o, out);
    InitXTensor(indices, state_indices, DBG_LOC, /*allowPitched=*/true);
    XTensor cached;
    if (cachedLens.has_value()) {
        InitXTensor(cached, cachedLens.value());
    }
    uint32_t batch = query.dim() == 2 ? static_cast<uint32_t>(query.size(0)) : 0;
    XliteOpRecurrentGatedDeltaRulePaged(
        rt, q, k, v, b, decay, s, o, &indices, batch, num_key_heads, num_value_heads, k_dim, v_dim,
        cachedLens.has_value() ? &cached : nullptr, stateBlockSize, aligned);
    rt.Synchronize();
}

// KDA safe-gate decode: raw beta/gate sigmoid'd in-kernel, per-k decay,
// q/k L2-normed in-kernel. aLog/dtBias are fp32 parameters.
void RecurrentGatedDeltaRuleKdaPaged(
    XRuntime &rt, at::Tensor &query, at::Tensor &key, at::Tensor &value, at::Tensor &beta,
    at::Tensor &gate, at::Tensor &aLog, at::Tensor &dtBias, at::Tensor &state, at::Tensor &out,
    at::Tensor &state_indices, uint32_t num_key_heads, uint32_t num_value_heads, uint32_t k_dim,
    uint32_t v_dim, double scale, double lowerBound, double normEps,
    const std::optional<at::Tensor> &cachedLens, uint32_t stateBlockSize, bool aligned)
{
    CheckGdnDevices(rt, {&query, &key, &value, &beta, &gate, &aLog, &dtBias, &state, &out,
                         &state_indices, cachedLens.has_value() ? &cachedLens.value() : nullptr});
    XTensor q, k, v, b, g, al, db, s, o, indices;
    InitXTensor(q, query);
    InitXTensor(k, key);
    InitXTensor(v, value);
    InitXTensor(b, beta);
    InitXTensor(g, gate);
    InitXTensor(al, aLog);
    InitXTensor(db, dtBias);
    InitXTensor(s, state, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(o, out);
    InitXTensor(indices, state_indices, DBG_LOC, /*allowPitched=*/true);
    XTensor cached;
    if (cachedLens.has_value()) {
        InitXTensor(cached, cachedLens.value());
    }
    uint32_t batch = query.dim() == 2 ? static_cast<uint32_t>(query.size(0)) : 0;
    XliteOpRecurrentGatedDeltaRuleKdaPaged(
        rt, q, k, v, b, g, al, db, s, o, &indices, batch, num_key_heads, num_value_heads, k_dim,
        v_dim, static_cast<float>(scale), static_cast<float>(lowerBound),
        static_cast<float>(normEps), cachedLens.has_value() ? &cached : nullptr, stateBlockSize,
        aligned);
    rt.Synchronize();
}

void EinsumMhtHdtMhd(XRuntime &rt, at::Tensor &mht, at::Tensor &hdt, at::Tensor &mhd, uint32_t m,
                     uint32_t h, uint32_t t, uint32_t d, bool weightNZ)
{
    (void)weightNZ;  // deprecated no-op
    XTensor _mht, _hdt, _mhd;
    InitXTensor(_mht, mht, DBG_LOC);
    InitXTensor(_hdt, hdt, DBG_LOC);
    InitXTensor(_mhd, mhd, DBG_LOC);
    XliteOpEinsumMhtHdtMhd(rt, _mht, _hdt, _mhd, m, h, t, d);
    rt.Synchronize();
}

void EinsumMhtHtdMhd(XRuntime &rt, at::Tensor &mht, at::Tensor &htd, at::Tensor &mhd, uint32_t m,
                     uint32_t h, uint32_t t, uint32_t d, bool weightNZ)
{
    (void)weightNZ;  // deprecated no-op
    XTensor _mht, _htd, _mhd;
    InitXTensor(_mht, mht, DBG_LOC);
    InitXTensor(_htd, htd, DBG_LOC);
    InitXTensor(_mhd, mhd, DBG_LOC);
    XliteOpEinsumMhtHtdMhd(rt, _mht, _htd, _mhd, m, h, t, d);
    rt.Synchronize();
}

void UnpackActivation(XRuntime &rt, at::Tensor &input, at::Tensor &output)
{
    XTensor _input, _output;
    InitXTensor(_input, input, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);

    XliteOpUnpackActivation(rt, _input, _output);
    rt.Synchronize();
}

void HcAct(XRuntime &rt, at::Tensor &mixes, at::Tensor &hcScale, at::Tensor &hcBase,
           at::Tensor &post, at::Tensor &comb, uint32_t hcMult, float eps, uint32_t sinkhornIters,
           at::Tensor &xResid, at::Tensor &output)
{
    XTensor _mixes, _hcScale, _hcBase, _post, _comb, _xResid, _output;

    InitXTensor(_mixes, mixes, DBG_LOC);
    InitXTensor(_hcScale, hcScale, DBG_LOC);
    InitXTensor(_hcBase, hcBase, DBG_LOC);
    InitXTensor(_post, post, DBG_LOC);
    InitXTensor(_comb, comb, DBG_LOC);
    if (TensorUsable(xResid)) {
        InitXTensor(_xResid, xResid, DBG_LOC);
    }
    if (TensorUsable(output)) {
        InitXTensor(_output, output, DBG_LOC);
    }
    XliteOpHcAct(rt, _mixes, _hcScale, _hcBase, _post, _comb, hcMult, eps, sinkhornIters, false,
                 _xResid, _output);
    rt.Synchronize();
}

void HcPost(XRuntime &rt, at::Tensor &x, at::Tensor &post, at::Tensor &comb, at::Tensor &residual,
            at::Tensor &y, uint32_t m, uint32_t hcMult, uint32_t hidden)
{
    XTensor _x, _post, _comb, _residual, _y;
    InitXTensor(_x, x, DBG_LOC);
    InitXTensor(_post, post, DBG_LOC);
    InitXTensor(_comb, comb, DBG_LOC);
    InitXTensor(_residual, residual, DBG_LOC);
    InitXTensor(_y, y, DBG_LOC);
    XliteOpHcPost(rt, _x, _post, _comb, _residual, _y, m, hcMult, hidden);
    rt.Synchronize();
}

void HcSplitSinkhorn(XRuntime &rt, at::Tensor &mixes, at::Tensor &hcScale, at::Tensor &hcBase,
                     at::Tensor &pre, at::Tensor &post, at::Tensor &comb, uint32_t hcMult,
                     float eps, uint32_t sinkhornIters)
{
    XTensor _mixes, _hcScale, _hcBase, _pre, _post, _comb;
    InitXTensor(_mixes, mixes, DBG_LOC);
    InitXTensor(_hcScale, hcScale, DBG_LOC);
    InitXTensor(_hcBase, hcBase, DBG_LOC);
    InitXTensor(_pre, pre, DBG_LOC);
    if (TensorUsable(post)) {
        InitXTensor(_post, post, DBG_LOC);
    }
    if (TensorUsable(comb)) {
        InitXTensor(_comb, comb, DBG_LOC);
    }
    // hc_split_sinkhorn has no head path; headOnly is ignored (always non-head).
    XliteOpHcSplitSinkhorn(rt, _mixes, _hcScale, _hcBase, _pre, _post, _comb, hcMult, eps,
                           sinkhornIters, false);
    rt.Synchronize();
}

void HcPre(XRuntime &rt, at::Tensor &xResid, at::Tensor &pre, at::Tensor &output, uint32_t m,
           uint32_t hcMult, uint32_t hidden)
{
    XTensor _xResid, _pre, _output;
    InitXTensor(_xResid, xResid, DBG_LOC);
    InitXTensor(_pre, pre, DBG_LOC);
    InitXTensor(_output, output, DBG_LOC);
    XliteOpHcPre(rt, _xResid, _pre, _output, m, hcMult, hidden);
    rt.Synchronize();
}

void SiluGateMul(XRuntime &rt, at::Tensor &normed, at::Tensor &gate, at::Tensor &out)
{
    XTensor n, g, o;
    InitXTensor(n, normed);
    InitXTensor(g, gate);
    InitXTensor(o, out);
    XliteOpSiluGateMul(rt, n, g, o);
    rt.Synchronize();
}

void KpoolTailCompressAndSpill(XRuntime &rt, at::Tensor &tailCache, at::Tensor &indexerCache,
                               at::Tensor &k, at::Tensor &gate, at::Tensor &ape,
                               at::Tensor &positions, at::Tensor &queryStartLoc, at::Tensor &lens,
                               at::Tensor &cachedLens, at::Tensor &slotMapping,
                               at::Tensor &tailBlockTable, uint32_t headDim, uint32_t kpool,
                               uint32_t mainBlockSize)
{
    XTensor _tailCache, _indexerCache, _k, _gate, _ape, _positions, _queryStartLoc, _lens,
        _cachedLens, _slotMapping, _tailBlockTable;
    // Paged-pool cache views arrive dim0-pitched; kernels consume the pitch via stride.
    InitXTensor(_tailCache, tailCache, UNKNOWN_DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_indexerCache, indexerCache, UNKNOWN_DBG_LOC, /*allowPitched=*/true);
    InitXTensor(_k, k);
    InitXTensor(_gate, gate);
    InitXTensor(_ape, ape);
    InitXTensor(_positions, positions);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_slotMapping, slotMapping);
    InitXTensor(_tailBlockTable, tailBlockTable);
    XliteOpKpoolTailCompressAndSpill(rt, _tailCache, _indexerCache, _k, _gate, _ape, _positions,
                                     _queryStartLoc, _lens, _cachedLens, _slotMapping,
                                     _tailBlockTable, headDim, kpool, mainBlockSize);
    rt.Synchronize();
}

void RecurrentGatedDeltaRulePrefillPaged(XRuntime &rt, at::Tensor &query, at::Tensor &key,
                                         at::Tensor &value, at::Tensor &beta, at::Tensor &g,
                                         at::Tensor &state, at::Tensor &out, at::Tensor &indices,
                                         at::Tensor &starts, at::Tensor &initial, uint32_t kh,
                                         uint32_t vh, uint32_t kd, uint32_t vd)
{
    CheckGdnDevices(rt,
                    {&query, &key, &value, &beta, &g, &state, &out, &indices, &starts, &initial});
    XTensor q, k, v, b, decay, st, o, ids, bounds, init;
    InitXTensor(q, query);
    InitXTensor(k, key);
    InitXTensor(v, value);
    InitXTensor(b, beta);
    InitXTensor(decay, g);
    InitXTensor(st, state, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(o, out);
    InitXTensor(ids, indices);
    InitXTensor(bounds, starts);
    InitXTensor(init, initial);
    XliteOpRecurrentGatedDeltaRulePrefillPaged(rt, q, k, v, b, decay, st, o, ids, bounds, init, kh,
                                               vh, kd, vd);
    rt.Synchronize();
}

// KDA safe-gate prefill: same in-kernel contracts as the decode variant.
void RecurrentGatedDeltaRuleKdaPrefillPaged(XRuntime &rt, at::Tensor &query, at::Tensor &key,
                                            at::Tensor &value, at::Tensor &beta, at::Tensor &gate,
                                            at::Tensor &aLog, at::Tensor &dtBias, at::Tensor &state,
                                            at::Tensor &out, at::Tensor &indices,
                                            at::Tensor &starts, at::Tensor &initial, uint32_t kh,
                                            uint32_t vh, uint32_t kd, uint32_t vd, double scale,
                                            double lowerBound, double normEps)
{
    CheckGdnDevices(rt, {&query, &key, &value, &beta, &gate, &aLog, &dtBias, &state, &out, &indices,
                         &starts, &initial});
    XTensor q, k, v, b, g, al, db, st, o, ids, bounds, init;
    InitXTensor(q, query);
    InitXTensor(k, key);
    InitXTensor(v, value);
    InitXTensor(b, beta);
    InitXTensor(g, gate);
    InitXTensor(al, aLog);
    InitXTensor(db, dtBias);
    InitXTensor(st, state, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(o, out);
    InitXTensor(ids, indices);
    InitXTensor(bounds, starts);
    InitXTensor(init, initial);
    XliteOpRecurrentGatedDeltaRuleKdaPrefillPaged(
        rt, q, k, v, b, g, al, db, st, o, ids, bounds, init, kh, vh, kd, vd,
        static_cast<float>(scale), static_cast<float>(lowerBound), static_cast<float>(normEps));
    rt.Synchronize();
}

void HcMeanCollapse(XRuntime &rt, at::Tensor &input, at::Tensor &output, uint32_t m,
                    uint32_t hcMult, uint32_t hidden)
{
    CheckGdnDevices(rt, {&input, &output});
    XTensor _input, _output;
    InitXTensor(_input, input);
    InitXTensor(_output, output);
    XliteOpHcMeanCollapse(rt, _input, _output, m, hcMult, hidden);
    rt.Synchronize();
}

void Conv1dAndSiLUPrefillPaged(XRuntime &rt, at::Tensor &input, at::Tensor &state,
                               at::Tensor &weight, at::Tensor &output, at::Tensor &indices,
                               at::Tensor &starts, at::Tensor &initial, uint32_t stateLayout,
                               uint32_t chanOffset)
{
    CheckGdnDevices(rt, {&input, &state, &weight, &output, &indices, &starts, &initial});
    XTensor x, st, w, o, ids, bounds, init;
    InitXTensor(x, input);
    InitXTensor(st, state, DBG_LOC, /*allowPitched=*/true);
    InitXTensor(w, weight);
    InitXTensor(o, output);
    InitXTensor(ids, indices);
    InitXTensor(bounds, starts);
    InitXTensor(init, initial);
    XliteOpConv1dAndSiLUPrefillPaged(rt, st, x, w, o, ids, bounds, init, stateLayout, chanOffset);
    rt.Synchronize();
}

// ===========================================================================
// Cast<Dst>::go(Src): convert a dispatcher arg (Src) to the op's arg type
// (Dst). Defined early so hand-written adapters (RuntimeHolder ctor,
// configureSwizzle, setTaskId, model_init, op wrappers) share the same
// guarded narrowing as the op<&Fn> path. RuntimeHolder/_rt-dependent
// specializations (XRuntime&, const char*) stay further down.
// ===========================================================================
template <typename Dst>
struct Cast {
    template <typename V>
    static auto go(V &&v) -> decltype(auto)
    {
        return std::forward<V>(v);
    }
};
// Guarded int64->uint narrowing: pybind11 rejected out-of-range at the
// boundary; a negative (e.g. -1 sentinel) would wrap to ~4e9 and feed
// n_heads/head_dim/block_size as a huge allocation or loop bound.
template <>
struct Cast<uint32_t> {
    static uint32_t go(int64_t v)
    {
        TORCH_CHECK(v >= 0 && v <= static_cast<int64_t>(UINT32_MAX),
                    "argument out of range for uint32: ", v);
        return static_cast<uint32_t>(v);
    }
};
template <>
struct Cast<uint64_t> {
    static uint64_t go(int64_t v)
    {
        TORCH_CHECK(v >= 0, "argument out of range for uint64 (negative): ", v);
        return static_cast<uint64_t>(v);
    }
};
template <>
struct Cast<uint8_t> {
    static uint8_t go(int64_t v)
    {
        TORCH_CHECK(v >= 0 && v <= static_cast<int64_t>(UINT8_MAX),
                    "argument out of range for uint8: ", v);
        return static_cast<uint8_t>(v);
    }
};
// Cast<int>/Cast<long>: signed, same width or wider than int64 — no wrap risk,
// and EinsumMhtHtdMhd uses `int T = -1, int D = -1` sentinels, so no guard.
template <>
struct Cast<int> {
    static int go(int64_t v)
    {
        return static_cast<int>(v);
    }
};
template <>
struct Cast<long> {
    static long go(int64_t v)
    {
        return static_cast<long>(v);
    }
};
template <>
struct Cast<float> {
    static float go(double v)
    {
        return static_cast<float>(v);
    }
};

// ===========================================================================
// Holders for public C++ classes (cannot inherit CustomClassHolder — would
// leak torch deps into the public headers used by downstream C++ consumers).
// ===========================================================================
struct RuntimeHolder : torch::CustomClassHolder {
    std::unique_ptr<XRuntime> rt;

    // XRuntime ctor: (devid, sizeMB=0, rank=0, tp=1, dp=1, moeTp=1, moeEp=1)
    RuntimeHolder(int64_t devid, double sizeMB, int64_t rank, int64_t tpSize, int64_t dpSize,
                  int64_t moeTpSize, int64_t moeEpSize)
    {
        rt = std::make_unique<XRuntime>(Cast<uint32_t>::go(devid), static_cast<size_t>(sizeMB),
                                        Cast<uint32_t>::go(rank), Cast<uint32_t>::go(tpSize),
                                        Cast<uint32_t>::go(dpSize), Cast<uint32_t>::go(moeTpSize),
                                        Cast<uint32_t>::go(moeEpSize));
    }

    void synchronize()
    {
        if (rt) {
            rt->Synchronize();
        }
    }
    void updateCoreNum(double util)
    {
        if (rt) {
            rt->UpdateCoreNum(static_cast<float>(util));
        }
    }
    int64_t initTensorPool(double sizeMB)
    {
        return rt ? static_cast<int64_t>(rt->InitTensorPool(static_cast<size_t>(sizeMB))) : -1;
    }
    void setCurrentContext()
    {
        if (rt) {
            rt->SetCurrentContext();
        }
    }
    void configureSwizzle(int64_t swizzle, bool useSwizzleTable)
    {
        if (rt) {
            rt->ConfigureSwizzle(Cast<uint32_t>::go(swizzle), useSwizzleTable);
        }
    }
    int64_t getTaskId() const
    {
        return rt ? static_cast<int64_t>(rt->taskId) : 0;
    }
    void setTaskId(int64_t v)
    {
        if (rt) {
            rt->taskId = Cast<uint32_t>::go(v);
        }
    }
    bool getMultiTaskParallel() const
    {
        return rt ? rt->multiTaskParallel : false;
    }
    void setMultiTaskParallel(bool v)
    {
        if (rt) {
            rt->multiTaskParallel = v;
        }
    }
    // notify/peerNotify are opaque aclrtNotify (void*) handles: notify is
    // created in XRuntime::Init, peerNotify is only ever set from Python —
    // MULTI_TASK_PARALLEL cross-wires them (rt0.peer_notify = rt1.notify).
    // Expose the raw pointer value as an int64 round-trip; Python has no type
    // that can hold the handle itself.
    int64_t getNotify() const
    {
        return rt ? reinterpret_cast<int64_t>(rt->notify) : 0;
    }
    void setNotify(int64_t v)
    {
        if (rt) {
            rt->notify = reinterpret_cast<aclrtNotify>(v);
        }
    }
    int64_t getPeerNotify() const
    {
        return rt ? reinterpret_cast<int64_t>(rt->peerNotify) : 0;
    }
    void setPeerNotify(int64_t v)
    {
        if (rt) {
            rt->peerNotify = reinterpret_cast<aclrtNotify>(v);
        }
    }
};

struct ConfigHolder : torch::CustomClassHolder {
    XModelConfig cfg;

    ConfigHolder() = default;
};

struct CoreAssignerHolder : torch::CustomClassHolder {
    std::unique_ptr<XCoreAssigner> ca;
    CoreAssignerHolder(double prefillRatio)
    {
        ca = std::make_unique<XCoreAssigner>(static_cast<float>(prefillRatio));
    }
    double assignCore(bool isDecode)
    {
        return ca ? ca->AssignCore(isDecode) : 0.0;
    }
    void releaseCore(bool isDecode)
    {
        if (ca) {
            ca->ReleaseCore(isDecode);
        }
    }
};

// ===========================================================================
// Op call bridge: call_op<&Fn>(...) adapts dispatcher-level argument types
// (intrusive_ptr<RuntimeHolder>, int64_t, double) to the op implementations'
// types (XRuntime&, uint32_t, float) via the OpWrap template below. The
// opreg::op<&Fn> adapters and the three hand-written wrappers further down
// all funnel through this.
// ===========================================================================
static XRuntime &_rt(c10::intrusive_ptr<RuntimeHolder> &h)
{
    return *h->rt;
}

// FnTraits: extract return type and arg-type tuple from a function pointer.
template <typename T>
struct FnTraits;
template <typename R, typename... Args>
struct FnTraits<R (*)(Args...)> {
    using Ret = R;
    using ArgsTup = std::tuple<Args...>;
    static constexpr size_t N = sizeof...(Args);
};

// Cast specializations depending on RuntimeHolder/_rt (defined above);
// scalar specializations live earlier, before the holders.
template <>
struct Cast<XRuntime &> {
    static XRuntime &go(c10::intrusive_ptr<RuntimeHolder> h)
    {
        return _rt(h);
    }
};
template <>
struct Cast<const char *> {
    static const char *go(const std::string &v)
    {
        return v.c_str();
    }
};

// RetCast<Src, Dst>: convert an op's return value to the dispatcher return type.
template <typename Src, typename Dst>
struct RetCast {
    static Dst go(Src v)
    {
        return static_cast<Dst>(std::move(v));
    }
};

// OpWrap<Fn, Idx, DRet>::run(args...) forwards dispatcher args to Fn, casting
// each arg via Cast<Fn's arg type>::go, and casts the return via RetCast. Dst
// arg types and Src arg types are both deduced (Dst from &Fn, Src from args),
// so call sites need no explicit type list.
template <auto Fn, typename Idx, typename DRet>
struct OpWrap;
template <auto Fn, size_t... I, typename DRet>
struct OpWrap<Fn, std::index_sequence<I...>, DRet> {
    using FT = FnTraits<decltype(Fn)>;
    using Dst = typename FT::ArgsTup;
    static_assert(sizeof...(I) == FT::N, "OpWrap arg count mismatch");
    template <typename... A>
    static auto run(A &&...a) -> DRet
    {
        auto t = std::forward_as_tuple(std::forward<A>(a)...);
        if constexpr (std::is_void_v<typename FT::Ret>) {
            Fn(Cast<std::tuple_element_t<I, Dst>>::go(std::get<I>(t))...);
        } else {
            return RetCast<typename FT::Ret, DRet>::go(
                Fn(Cast<std::tuple_element_t<I, Dst>>::go(std::get<I>(t))...));
        }
    }
};

// call_op<Fn, DRet>(args...): convenience wrapper that forwards to OpWrap.
// DRet defaults to the op's own return type; pass int64_t for ops whose
// uint32_t/uint64_t return must be widened for the dispatcher.
template <auto Fn, typename DRet = typename FnTraits<decltype(Fn)>::Ret, typename... A>
auto call_op(A &&...a) -> DRet
{
    return OpWrap<Fn, std::make_index_sequence<FnTraits<decltype(Fn)>::N>, DRet>::run(
        std::forward<A>(a)...);
}

// Hand-written wrappers for the three ops whose argument prep goes beyond
// call_op's cast layer: muls remaps the calcNum sentinel (INT64_MAX ->
// UINT32_MAX), split narrows vector<int64_t> to the vector<uint32_t> that
// Split mutates in place, and get_tile_size_of_cached_kv narrows int64 lists
// to the host uint32 vectors GetTileSizeOfCachedKV takes. All other ops
// register via op<&Fn>.
static void muls_op(c10::intrusive_ptr<RuntimeHolder> rt, at::Tensor input, double scale,
                    at::Tensor output, int64_t calcOffset, int64_t calcNum)
{
    // The C++ side takes uint32_t; pybind11 rejected out-of-range values at the
    // boundary, so guard the narrowing here instead of wrapping silently.
    // INT64_MAX stays valid as the "all elements" sentinel mapped below.
    TORCH_CHECK(calcOffset >= 0, "muls: calc_offset must be non-negative, got ", calcOffset);
    TORCH_CHECK(calcNum >= 0 && (calcNum <= static_cast<int64_t>(UINT32_MAX) ||
                                 calcNum == std::numeric_limits<int64_t>::max()),
                "muls: calc_num out of range for uint32, got ", calcNum);
    Muls(_rt(rt), input, static_cast<float>(scale), output, static_cast<uint32_t>(calcOffset),
         static_cast<uint32_t>(calcNum == std::numeric_limits<int64_t>::max()
                                   ? UINT32_MAX
                                   : static_cast<uint64_t>(calcNum)));
}
static void split_op(c10::intrusive_ptr<RuntimeHolder> rt, at::Tensor in_,
                     std::vector<at::Tensor> outputs, const std::vector<int64_t> &sizes,
                     int64_t numPackets)
{
    // Split takes std::vector<uint32_t>& (lvalue ref, mutated in place); the
    // dispatcher passes std::vector<int64_t>, so build the narrowed vector here.
    // Guard the narrowing: pybind11 rejected out-of-range values at the
    // boundary instead of wrapping silently.
    std::vector<uint32_t> sizes32;
    sizes32.reserve(sizes.size());
    for (auto s : sizes) {
        TORCH_CHECK(s >= 0 && s <= static_cast<int64_t>(UINT32_MAX),
                    "split: sizes element out of range for uint32, got ", s);
        sizes32.push_back(static_cast<uint32_t>(s));
    }
    TORCH_CHECK(numPackets >= 0 && numPackets <= static_cast<int64_t>(UINT32_MAX),
                "split: num_packets out of range for uint32, got ", numPackets);
    Split(_rt(rt), in_, outputs, sizes32, static_cast<uint32_t>(numPackets));
}
static int64_t get_tile_size_of_cached_kv_op(const std::vector<int64_t> &cachedLens,
                                             const std::vector<int64_t> &queryLens,
                                             int64_t headNumInGroup, int64_t nKVHeads,
                                             int64_t blockSize, int64_t aicNum)
{
    // GetTileSizeOfCachedKV takes host uint32 vectors; narrow the int64 lists
    // the Python side passes (pybind11 accepted List[int] directly; torch's
    // dispatcher passes std::vector<int64_t>). Guard the narrowing like
    // split_op — pybind11 rejected out-of-range values at the boundary.
    std::vector<uint32_t> cl, ql;
    cl.reserve(cachedLens.size());
    ql.reserve(queryLens.size());
    for (auto v : cachedLens) {
        TORCH_CHECK(v >= 0 && v <= static_cast<int64_t>(UINT32_MAX),
                    "get_tile_size_of_cached_kv: cached_lens element out of range for uint32, got ",
                    v);
        cl.push_back(static_cast<uint32_t>(v));
    }
    for (auto v : queryLens) {
        TORCH_CHECK(v >= 0 && v <= static_cast<int64_t>(UINT32_MAX),
                    "get_tile_size_of_cached_kv: query_lens element out of range for uint32, got ",
                    v);
        ql.push_back(static_cast<uint32_t>(v));
    }
    return static_cast<int64_t>(GetTileSizeOfCachedKV(
        cl, ql, Cast<uint32_t>::go(headNumInGroup), Cast<uint32_t>::go(nKVHeads),
        Cast<uint32_t>::go(blockSize), Cast<uint32_t>::go(aicNum)));
}

// ===========================================================================
// Model.forward method adapters (torch::class_ has no overloading — each
// distinct signature gets its own method; the primary V1 variant keeps the
// plain `forward` name, the others carry suffixes).
// ===========================================================================
// forward (V1, no aux): rt, input, attnMeta, kvCache, freqsCis, output, currStream
static void model_forward_v1(const c10::intrusive_ptr<_CModel> &self,
                             const c10::intrusive_ptr<RuntimeHolder> &rt, at::Tensor input,
                             const c10::intrusive_ptr<CModelAttnMeta> &attnMeta,
                             std::vector<std::vector<at::Tensor>> kvCache, at::Tensor freqsCis,
                             at::Tensor output,
                             const std::optional<std::vector<at::Tensor>> &auxOutputs,
                             int64_t currStream)
{
    // An omitted or empty aux list means "no capture"; the aux-taking ForwardV1
    // overload handles it the same way as the no-aux overload (Set/Clear on an
    // empty vector is a no-op).
    std::vector<at::Tensor> aux = auxOutputs.value_or(std::vector<at::Tensor>{});
    self->ForwardV1(*rt->rt, input, *attnMeta, kvCache, freqsCis, output, aux,
                    static_cast<uint64_t>(currStream));
}
static void model_forward_get_logits(const c10::intrusive_ptr<_CModel> &self,
                                     const c10::intrusive_ptr<RuntimeHolder> &rt, at::Tensor input,
                                     at::Tensor indices, at::Tensor output, int64_t currStream)
{
    self->ForwardGetLogits(*rt->rt, input, indices, output, static_cast<uint64_t>(currStream));
}
static void model_forward_and_get_logits(const c10::intrusive_ptr<_CModel> &self,
                                         const c10::intrusive_ptr<RuntimeHolder> &rt,
                                         at::Tensor input,
                                         const c10::intrusive_ptr<CModelAttnMeta> &attnMeta,
                                         std::vector<std::vector<at::Tensor>> kvCache,
                                         at::Tensor freqsCis, at::Tensor indices, at::Tensor output,
                                         int64_t currStream)
{
    self->ForwardAndGetLogitsV1(*rt->rt, input, *attnMeta, kvCache, freqsCis, indices, output,
                                static_cast<uint64_t>(currStream));
}
static void model_forward_with_inputs_embeds_v1(
    const c10::intrusive_ptr<_CModel> &self, const c10::intrusive_ptr<RuntimeHolder> &rt,
    at::Tensor input, const c10::intrusive_ptr<CModelAttnMeta> &attnMeta,
    std::vector<std::vector<at::Tensor>> kvCache, at::Tensor freqsCis, at::Tensor output,
    const std::optional<std::vector<at::Tensor>> &auxOutputs, int64_t currStream,
    std::vector<at::Tensor> deepstackInput, const std::optional<at::Tensor> &inputIds)
{
    std::vector<at::Tensor> aux = auxOutputs.value_or(std::vector<at::Tensor>{});
    self->ForwardWithInputsEmbedsV1(*rt->rt, input, *attnMeta, kvCache, freqsCis, output, aux,
                                    static_cast<uint64_t>(currStream), deepstackInput,
                                    inputIds.value_or(at::Tensor()));
}
static void model_forward_v2(const c10::intrusive_ptr<_CModel> &self,
                             const c10::intrusive_ptr<RuntimeHolder> &rt, at::Tensor input,
                             const c10::intrusive_ptr<CModelAttnMetaV2> &attnMeta,
                             std::vector<std::vector<at::Tensor>> kvCache,
                             std::vector<at::Tensor> freqsCis, at::Tensor output,
                             const std::optional<std::vector<at::Tensor>> &auxOutputs,
                             int64_t currStream)
{
    std::vector<at::Tensor> aux = auxOutputs.value_or(std::vector<at::Tensor>{});
    self->ForwardV2(*rt->rt, input, *attnMeta, kvCache, freqsCis, output, aux,
                    static_cast<uint64_t>(currStream));
}
static void model_forward_and_get_logits_v2(const c10::intrusive_ptr<_CModel> &self,
                                            const c10::intrusive_ptr<RuntimeHolder> &rt,
                                            at::Tensor input,
                                            const c10::intrusive_ptr<CModelAttnMetaV2> &attnMeta,
                                            std::vector<std::vector<at::Tensor>> kvCache,
                                            std::vector<at::Tensor> freqsCis, at::Tensor indices,
                                            at::Tensor output, int64_t currStream)
{
    self->ForwardAndGetLogitsV2(*rt->rt, input, *attnMeta, kvCache, freqsCis, indices, output,
                                static_cast<uint64_t>(currStream));
}
static void model_forward_with_inputs_embeds_v2(
    const c10::intrusive_ptr<_CModel> &self, const c10::intrusive_ptr<RuntimeHolder> &rt,
    at::Tensor input, const c10::intrusive_ptr<CModelAttnMetaV2> &attnMeta,
    std::vector<std::vector<at::Tensor>> kvCache, at::Tensor freqsCis, at::Tensor output,
    const std::optional<std::vector<at::Tensor>> &auxOutputs, int64_t currStream,
    std::vector<at::Tensor> deepstackInput, const std::optional<at::Tensor> &inputIds)
{
    std::vector<at::Tensor> aux = auxOutputs.value_or(std::vector<at::Tensor>{});
    self->ForwardWithInputsEmbedsV2(*rt->rt, input, *attnMeta, kvCache, freqsCis, output, aux,
                                    static_cast<uint64_t>(currStream), deepstackInput,
                                    inputIds.value_or(at::Tensor()));
}
static void model_init(const c10::intrusive_ptr<_CModel> &self,
                       const c10::intrusive_ptr<ConfigHolder> &config, int64_t rank)
{
    self->Init(config->cfg, Cast<uint32_t>::go(rank));
}
static int64_t model_get_tensor_pool_size(const c10::intrusive_ptr<_CModel> &self, int64_t dbg)
{
    return static_cast<int64_t>(self->GetTensorPoolSize(static_cast<int>(dbg)));
}

// ===========================================================================
// Class registration
// ===========================================================================
// Op registration in the TORCH_LIBRARY(xlite_sabi, m) block below.
// opreg::OpDefs(m).def(name, fn, args...) mirrors torch::class_::def's
// initializer_list<torch::arg> API: argument types and returns are inferred
// from the registered function's C++ signature, while the optional torch::arg
// entries supply the two things a signature cannot express — argument names
// (auto-inferred schemas name them positionally _0, _1, ...) and default
// values. Types therefore cannot drift from the implementation. The schema is
// unqualified; the library namespace xlite_sabi is prepended automatically.
// Leave the arg list out for positional-only ops.
namespace opreg
{

// Builds the registration schema: infers types/returns from the registered
// function's C++ signature, then applies argument names and defaults from the
// torch::arg list (the mechanism torch::class_::def uses internally via
// class_base::withNewArguments). std::optional<T> parameters already infer as
// `T?`, so None defaults need no type annotation. Return names are dropped
// (canonical unnamed form, as in hand-written schemas). The arity check
// doubles as a drift guard: changing the function signature without updating
// the torch::arg list fails at library load with a clear message instead of
// registering a mismatched schema.
template <typename FuncType>
std::string makeSchema(const char *name, const std::vector<torch::arg> &args)
{
    auto inferred = c10::inferFunctionSchemaFlattenedReturns<FuncType>();
    const auto &oldArgs = inferred.arguments();
    std::vector<torch::arg> all = args;
    if (all.empty()) {
        for (const auto &a : oldArgs) {
            all.push_back(torch::arg(a.name()));
        }
    }
    TORCH_CHECK(all.size() == oldArgs.size(), name, ": got ", all.size(), " torch::arg entries",
                " for ", oldArgs.size(), " C++ parameters; keep the torch::arg list in sync with",
                " the registered function signature");
    std::vector<c10::Argument> patched;
    patched.reserve(oldArgs.size());
    bool seen_default = false;
    for (size_t i = 0; i < oldArgs.size(); ++i) {
        // Torch schemas require non-defaulted arguments to precede defaulted
        // ones (FunctionSchema::checkSchema), but the C++ signatures ported
        // from pybind11 interleave them (e.g. compressor's required
        // weighted_sum follows the optional ape/norm/freqs). Keep such
        // signatures loadable by marking every non-defaulted argument after
        // the first defaulted one keyword-only — the same exemption
        // checkSchema gives list-typed arguments. Call sites must pass those
        // by keyword, which the .pyi stubs document with a `*` marker.
        const bool defaulted = all[i].value_.has_value();
        const bool kwonly =
            seen_default && !defaulted && oldArgs[i].type()->kind() != c10::ListType::Kind;
        seen_default = seen_default || defaulted;
        patched.emplace_back(all[i].name_, oldArgs[i].type(), oldArgs[i].N(), all[i].value_,
                             kwonly || oldArgs[i].kwarg_only());
    }
    std::vector<c10::Argument> returns;
    returns.reserve(inferred.returns().size());
    for (const auto &r : inferred.returns()) {
        returns.emplace_back("", r.type(), r.N(), std::nullopt, r.kwarg_only());
    }
    c10::FunctionSchema schema(name, "", std::move(patched), std::move(returns),
                               inferred.is_vararg(), inferred.is_varret());
    return c10::toString(schema);
}

// Chainable op registration over torch::Library, giving TORCH_LIBRARY blocks
// the same .def(name, fn, {torch::arg(...) ...}) shape as torch::class_.
class OpDefs
{
public:
    explicit OpDefs(torch::Library &m) : m_(m)
    {
    }

    template <typename FuncType>
    OpDefs &def(const char *name, FuncType fn, const std::vector<torch::arg> &args = {})
    {
        m_.def(makeSchema<FuncType>(name, args).c_str(), fn);
        return *this;
    }

private:
    // Single-use registration builder chained through *this; never copied or
    // assigned, so the reference member's disabled copy semantics are moot.
    torch::Library &m_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

// IValue-side type for an op implementation parameter: the dispatcher only
// carries int64_t/double/bool/std::string/at::Tensor/custom classes, so
// narrow types widen to their dispatcher forms — the same mapping the
// hand-written _op wrappers used to encode. Types the dispatcher can
// represent (at::Tensor, bool, std::string, std::optional/std::vector of
// tensors) pass through unchanged.
template <typename T>
struct IValueArg {
    using D = std::remove_cv_t<std::remove_reference_t<T>>;
    using type = std::conditional_t<
        std::is_same_v<D, XRuntime>, c10::intrusive_ptr<RuntimeHolder>,
        std::conditional_t<
            std::is_same_v<D, const char *>, std::string,
            std::conditional_t<std::is_same_v<D, float>, double,
                               std::conditional_t<(std::is_integral_v<D> || std::is_enum_v<D>) &&
                                                      !std::is_same_v<D, bool>,
                                                  int64_t, D>>>>;
};

// IValue-side type for an op implementation return value (same mapping).
template <typename T>
struct IValueRet {
    using D = std::remove_cv_t<std::remove_reference_t<T>>;
    using type = std::conditional_t<
        std::is_same_v<D, const char *>, std::string,
        std::conditional_t<std::is_same_v<D, float>, double,
                           std::conditional_t<(std::is_integral_v<D> || std::is_enum_v<D>) &&
                                                  !std::is_same_v<D, bool>,
                                              int64_t, D>>>;
};

// op<&Fn> replaces the hand-written xxx_op wrapper functions: a functor whose
// operator() carries the IValue-mapped signature of Fn (so schema inference
// yields the same schema the wrappers used to provide), forwarding through
// call_op, which narrows each argument back to Fn's types.
template <auto Fn, typename ArgsTup>
class OpAdapterT;
template <auto Fn, typename... Params>
class OpAdapterT<Fn, std::tuple<Params...>>
{
public:
    typename IValueRet<typename FnTraits<decltype(Fn)>::Ret>::type operator()(
        typename IValueArg<Params>::type... args) const
    {
        using Ret = typename IValueRet<typename FnTraits<decltype(Fn)>::Ret>::type;
        return call_op<Fn, Ret>(args...);
    }
};
template <auto Fn>
using OpAdapter = OpAdapterT<Fn, typename FnTraits<decltype(Fn)>::ArgsTup>;

template <auto Fn>
inline constexpr OpAdapter<Fn> op{};

// Decomposes a pointer-to-data-member `T C::*` into Class=C, Field=T.
template <typename M>
struct MemberPtrTraits;
template <typename C, typename T>
struct MemberPtrTraits<T C::*> {
    using Class = C;
    using Field = T;
};

// A def_rw-bound field lives either directly on the holder (AttnMeta/
// AttnMetaV2) or inside ConfigHolder::cfg; the member pointer's class says
// which.
template <typename Holder, typename C, typename T>
static T &fieldRef(const c10::intrusive_ptr<Holder> &h, T C::*field)
{
    if constexpr (std::is_same_v<C, Holder>) {
        return (*h).*field;
    } else {
        static_assert(std::is_same_v<C, XModelConfig>,
                      "def_rw fields must live on the holder or in ConfigHolder::cfg");
        return h->cfg.*field;
    }
}

// IValue-side type for a property field: the dispatcher only carries
// int64_t/double/bool/std::string/at::Tensor and lists of those, so other
// field types convert at the boundary —
//   vector<bool>                    <-> ListBool (bool preserved element-wise)
//   vector<uint32_t>                <-> ListInt (std::vector<int64_t>)
//   vector<vector<uint32_t>>        <-> ListListInt (per-row conversion)
//   float                           <-> double
//   bool                            <-> bool  (before the integral branch)
//   integral/enum                   <-> int64_t
// Anything else passes through and must be dispatcher-native (at::Tensor,
// std::optional<at::Tensor>, std::vector<at::Tensor>, ...); a type IValue
// cannot hold fails to compile at the def_rw site. vector<bool> stays on the
// IValue-native List<bool> path so Python sees [True, False, ...] rather than
// [0, 1, ...].
template <typename T>
struct IValueField {
    using type = std::conditional_t<
        std::is_same_v<T, std::vector<bool>>, c10::List<bool>,
        std::conditional_t<
            std::is_same_v<T, std::vector<uint32_t>>, std::vector<int64_t>,
            std::conditional_t<
                std::is_same_v<T, std::vector<std::vector<uint32_t>>>,
                std::vector<std::vector<int64_t>>,
                std::conditional_t<std::is_same_v<T, float>, double,
                                   std::conditional_t<std::is_same_v<T, bool>, bool,
                                                      std::conditional_t<std::is_integral_v<T> ||
                                                                             std::is_enum_v<T>,
                                                                         int64_t, T>>>>>>;
};

// Getter/setter for one field. Unlike op<&Fn> (compile-time function pointer
// as template argument), these capture a runtime pointer-to-member in the
// returned lambda — the same stateful-functor pattern torch's own
// class_::def_readwrite uses — which is what lets def_rw take &Class::field
// instead of a macro token. One instantiation per field type, so all
// properties of the same type share the functor code.
template <typename Holder, typename C, typename T>
static auto propGetter(T C::*field)
{
    using D = typename IValueField<T>::type;
    return [field](const c10::intrusive_ptr<Holder> &h) -> D {
        auto &f = fieldRef(h, field);
        if constexpr (std::is_same_v<D, c10::List<bool>>) {
            c10::List<bool> o;
            o.reserve(f.size());
            for (auto v : f) {
                o.push_back(v);
            }
            return o;
        } else if constexpr (std::is_same_v<D, std::vector<int64_t>>) {
            std::vector<int64_t> o;
            o.reserve(f.size());
            for (auto v : f) {
                o.push_back(static_cast<int64_t>(v));
            }
            return o;
        } else if constexpr (std::is_same_v<D, std::vector<std::vector<int64_t>>>) {
            std::vector<std::vector<int64_t>> o;
            for (auto &r : f) {
                std::vector<int64_t> row;
                row.reserve(r.size());
                for (auto v : r) {
                    row.push_back(static_cast<int64_t>(v));
                }
                o.push_back(std::move(row));
            }
            return o;
        } else {
            return static_cast<D>(f);
        }
    };
}

template <typename Holder, typename C, typename T>
static auto propSetter(T C::*field)
{
    using D = typename IValueField<T>::type;
    return [field](const c10::intrusive_ptr<Holder> &h, const D &v) {
        auto &f = fieldRef(h, field);
        // The list/unsigned narrowings below used to be rejected by pybind11's
        // int->uint32 casters; guard them so out-of-range values fail loudly
        // instead of wrapping. vector<bool> maps to c10::List<bool> (not
        // std::vector<bool>) because the boxed-kernel signature only accepts the
        // IValue-native List<bool>.
        if constexpr (std::is_same_v<D, c10::List<bool>>) {
            f.clear();
            for (auto x : v) {
                f.push_back(static_cast<bool>(x));
            }
        } else if constexpr (std::is_same_v<D, std::vector<int64_t>>) {
            f.clear();
            for (auto x : v) {
                if constexpr (std::is_same_v<typename T::value_type, uint32_t>) {
                    TORCH_CHECK(x >= 0 && x <= static_cast<int64_t>(UINT32_MAX),
                                "uint32 list element out of range: ", x);
                }
                f.push_back(static_cast<typename T::value_type>(x));
            }
        } else if constexpr (std::is_same_v<D, std::vector<std::vector<int64_t>>>) {
            f.clear();
            for (auto &r : v) {
                std::vector<typename T::value_type::value_type> row;
                for (auto x : r) {
                    TORCH_CHECK(x >= 0 && x <= static_cast<int64_t>(UINT32_MAX),
                                "uint32 list element out of range: ", x);
                    row.push_back(static_cast<typename T::value_type::value_type>(x));
                }
                f.push_back(std::move(row));
            }
        } else {
            if constexpr (std::is_integral_v<T> && !std::is_signed_v<T> &&
                          !std::is_same_v<T, bool>) {
                TORCH_CHECK(v >= 0 && static_cast<uint64_t>(v) <=
                                          static_cast<uint64_t>(std::numeric_limits<T>::max()),
                            "value out of range for unsigned field: ", v);
            }
            f = static_cast<T>(v);
        }
    };
}

// Chainable property registration, giving custom-class blocks the same
// .def_rw("name", &Class::field) shape as torch::class_::def_readwrite.
// def_readwrite only covers IValue-native field types; def_rw routes the
// numeric/list conversions above. The bound field may live on the holder
// itself or in ConfigHolder::cfg.
template <typename Holder>
class Props
{
public:
    explicit Props(torch::class_<Holder> &c) : c_(c)
    {
    }

    template <typename C, typename T>
    Props &def_rw(const char *name, T C::*field)
    {
        c_.def_property(name, propGetter<Holder>(field), propSetter<Holder>(field));
        return *this;
    }

private:
    // Single-use registration builder chained through *this; never copied or
    // assigned, so the reference member's disabled copy semantics are moot.
    torch::class_<Holder> &c_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

}  // namespace opreg

void EngramGate(XRuntime &rt, at::Tensor &hidden, at::Tensor &kv, at::Tensor &qWeight,
                at::Tensor &kWeight, at::Tensor &mask, at::Tensor &out, uint32_t hcMult,
                uint32_t dim, double eps, std::optional<at::Tensor> hiddenResidual)
{
    // xlite's XDtypeOf (xlite.h:21) does not map torch.bool/uint8; cast to int32
    // (nonzero = active). The kernel reads it as __gm__ int32_t.
    at::Tensor maskI32 = mask.to(at::kInt);
    XTensor _hidden, _hiddenResid, _kv, _q, _k, _mask, _out;
    InitXTensor(_hidden, hidden, DBG_LOC);
    // hiddenResidual nullopt/undefined → aliases hidden (unrotated path).
    if (hiddenResidual && hiddenResidual->defined() && hiddenResidual->numel() > 0) {
        InitXTensor(_hiddenResid, *hiddenResidual, DBG_LOC);
    }
    InitXTensor(_kv, kv, DBG_LOC);
    InitXTensor(_q, qWeight, DBG_LOC);
    InitXTensor(_k, kWeight, DBG_LOC);
    InitXTensor(_mask, maskI32, DBG_LOC);
    InitXTensor(_out, out, DBG_LOC);
    XliteOpEngramGate(rt, _hidden, _hiddenResid, _kv, _q, _k, _mask, _out, hcMult, dim,
                      static_cast<float>(eps));
    rt.Synchronize();
}

TORCH_LIBRARY(xlite_sabi, m)
{
    m.class_<RuntimeHolder>("Runtime")
        .def(torch::init<int64_t, double, int64_t, int64_t, int64_t, int64_t, int64_t>(), "",
             {torch::arg("devid"), torch::arg("size") = 0.0, torch::arg("rank") = 0,
              torch::arg("tp_size") = 1, torch::arg("dp_size") = 1, torch::arg("moe_tp_size") = 1,
              torch::arg("moe_ep_size") = 1})
        .def("synchronize", &RuntimeHolder::synchronize)
        .def("update_core_num", &RuntimeHolder::updateCoreNum, "", {torch::arg("util")})
        .def("init_tensor_pool", &RuntimeHolder::initTensorPool, "", {torch::arg("size")})
        .def("set_current_context", &RuntimeHolder::setCurrentContext)
        .def("configure_swizzle", &RuntimeHolder::configureSwizzle, "",
             {torch::arg("swizzle"), torch::arg("use_swizzle_table")})
        .def_property("task_id", &RuntimeHolder::getTaskId, &RuntimeHolder::setTaskId)
        .def_property("multi_task_parallel", &RuntimeHolder::getMultiTaskParallel,
                      &RuntimeHolder::setMultiTaskParallel)
        .def_property("notify", &RuntimeHolder::getNotify, &RuntimeHolder::setNotify)
        .def_property("peer_notify", &RuntimeHolder::getPeerNotify, &RuntimeHolder::setPeerNotify);

    // Fields live in ConfigHolder::cfg; def_rw resolves that from the member
    // pointer's class and converts non-IValue field types at the boundary.
    opreg::Props(m.class_<ConfigHolder>("ModelConfig").def(torch::init<>()))
        .def_rw("vocab_size", &XModelConfig::vocabSize)
        .def_rw("hidden_size", &XModelConfig::hiddenSize)
        .def_rw("n_layers", &XModelConfig::nLayers)
        .def_rw("attn_type", &XModelConfig::attnType)
        .def_rw("n_heads", &XModelConfig::nHeads)
        .def_rw("n_kv_heads", &XModelConfig::nKvHeads)
        .def_rw("head_dim", &XModelConfig::headDim)
        .def_rw("nope_head_dim", &XModelConfig::nopeHeadDim)
        .def_rw("rope_head_dim", &XModelConfig::ropeHeadDim)
        .def_rw("v_head_dim", &XModelConfig::vHeadDim)
        .def_rw("q_lora_rank", &XModelConfig::qLoraRank)
        .def_rw("kv_lora_rank", &XModelConfig::kvLoraRank)
        .def_rw("quant_attn_weight_transpose", &XModelConfig::quantAttnWeightTrans)
        .def_rw("quant_attn_weight_nz", &XModelConfig::quantAttnWeightNz)
        .def_rw("norm_eps", &XModelConfig::normEps)
        .def_rw("rope_theta", &XModelConfig::ropeTheta)
        .def_rw("softmax_scale", &XModelConfig::softmaxScale)
        .def_rw("n_dense_layers", &XModelConfig::nDenseLayers)
        .def_rw("n_routed_experts", &XModelConfig::nRoutedExperts)
        .def_rw("num_redundant_experts", &XModelConfig::numRedundantExperts)
        .def_rw("n_shared_experts", &XModelConfig::nSharedExperts)
        .def_rw("n_expert_groups", &XModelConfig::nExpertGroups)
        .def_rw("n_limited_groups", &XModelConfig::nLimitedGroups)
        .def_rw("n_act_experts", &XModelConfig::nActExperts)
        .def_rw("intermediate_size", &XModelConfig::intermediateSize)
        .def_rw("moe_intermediate_size", &XModelConfig::moeIntermediateSize)
        .def_rw("route_scale", &XModelConfig::routeScale)
        .def_rw("def_tp_size", &XModelConfig::defTpSize)
        .def_rw("def_dp_size", &XModelConfig::defDpSize)
        .def_rw("moe_ep_size", &XModelConfig::moeEpSize)
        .def_rw("moe_tp_size", &XModelConfig::moeTPSize)
        .def_rw("max_seq_len", &XModelConfig::maxSeqLen)
        .def_rw("max_batch_size", &XModelConfig::maxBatch)
        .def_rw("max_m", &XModelConfig::maxBatchedTokens)
        .def_rw("max_num_batched_tokens", &XModelConfig::maxBatchedTokens)
        .def_rw("num_spec_tokens", &XModelConfig::numSpecTokens)
        .def_rw("block_size", &XModelConfig::blockSize)
        .def_rw("block_sizes", &XModelConfig::blockSizes)
        .def_rw("experts_weight_transpose", &XModelConfig::expertsWeightTrans)
        .def_rw("weight_nz", &XModelConfig::weightNZ)
        .def_rw("experts_weight_nz", &XModelConfig::expertsWeightNZ)
        .def_rw("gate_captured", &XModelConfig::gateCaptured)
        .def_rw("mla_wuv_weight_nz", &XModelConfig::mlaWuvWeightNZ)
        .def_rw("mla_wukt_weight_nz", &XModelConfig::mlaWuktWeightNZ)
        .def_rw("qkv_bias", &XModelConfig::addBias)
        .def_rw("qk_norm", &XModelConfig::qkNorm)
        .def_rw("qk_norm_full", &XModelConfig::qkNormFull)
        .def_rw("attn_output_gate", &XModelConfig::attnOutputGate)
        .def_rw("scoring_func", &XModelConfig::scoringFunc)
        .def_rw("norm_topk_prob", &XModelConfig::normTopKProb)
        .def_rw("mrope_section", &XModelConfig::mropeSection)
        .def_rw("mrope_interleaved", &XModelConfig::mropeInterleaved)
        .def_rw("deepstack_num_level", &XModelConfig::deepstackNumLevel)
        .def_rw("index_head_dim", &XModelConfig::indexHeadDim)
        .def_rw("index_n_heads", &XModelConfig::indexNHeads)
        .def_rw("index_topk", &XModelConfig::indexTopK)
        .def_rw("index_softmax_scale", &XModelConfig::indexSoftmaxScale)
        .def_rw("index_rope_interleaved", &XModelConfig::indexRopeInterleaved)
        .def_rw("index_full_mask", &XModelConfig::indexFullMask)
        .def_rw("index_c8_mask", &XModelConfig::indexC8Mask)
        .def_rw("enable_dsa_cp", &XModelConfig::enableDsaCp)
        .def_rw("linear_num_k_heads", &XModelConfig::linearNumKHeads)
        .def_rw("linear_num_v_heads", &XModelConfig::linearNumVHeads)
        .def_rw("linear_key_head_dim", &XModelConfig::linearKeyHeadDim)
        .def_rw("linear_value_head_dim", &XModelConfig::linearValueHeadDim)
        .def_rw("linear_conv_kernel_dim", &XModelConfig::linearConvKernelDim)
        .def_rw("full_attention_interval", &XModelConfig::fullAttentionInterval)
        .def_rw("o_groups", &XModelConfig::oGroups)
        .def_rw("o_lora_rank", &XModelConfig::oLoraRank)
        .def_rw("window_size", &XModelConfig::windowSize)
        .def_rw("compress_rope_theta", &XModelConfig::compressRopeTheta)
        .def_rw("original_seq_len", &XModelConfig::originalSeqLen)
        .def_rw("rope_factor", &XModelConfig::ropeFactor)
        .def_rw("beta_fast", &XModelConfig::betaFast)
        .def_rw("beta_slow", &XModelConfig::betaSlow)
        .def_rw("hc_mult", &XModelConfig::hcMult)
        .def_rw("hc_sinkhorn_iters", &XModelConfig::hcSinkhornIters)
        .def_rw("hc_eps", &XModelConfig::hcEps)
        .def_rw("swiglu_limit", &XModelConfig::swigluLimit)
        .def_rw("n_hash_layers", &XModelConfig::nHashLayers)
        .def_rw("compress_ratios", &XModelConfig::compressRatios)
        .def_rw("aux_layer_ids", &XModelConfig::auxLayerIds)
        .def_rw("gate_lower_bound", &XModelConfig::gateLowerBound)
        .def_rw("index_kpool", &XModelConfig::indexKpool)
        .def_rw("index_kpool_compress", &XModelConfig::indexKpoolCompress)
        .def_rw("index_kpool_always_select_tail", &XModelConfig::indexKpoolAlwaysSelectTail)
        .def_rw("kda_layers", &XModelConfig::kdaLayers)
        .def_rw("full_attn_layers", &XModelConfig::fullAttnLayers);

    opreg::Props(m.class_<CModelAttnMeta>("AttnMeta").def(torch::init<>()))
        .def_rw("lens", &CModelAttnMeta::lens)
        .def_rw("cached_lens", &CModelAttnMeta::cachedLens)
        .def_rw("block_tables_cpu", &CModelAttnMeta::blockTablesList)
        .def_rw("positions", &CModelAttnMeta::positions);

    opreg::Props(m.class_<CModelAttnMetaV2>("AttnMetaV2").def(torch::init<>()))
        .def_rw("lens", &CModelAttnMetaV2::lens)
        .def_rw("cached_lens", &CModelAttnMetaV2::cachedLens)
        .def_rw("positions", &CModelAttnMetaV2::positions)
        .def_rw("lens_cpu", &CModelAttnMetaV2::lensCpu)
        .def_rw("cached_lens_cpu", &CModelAttnMetaV2::cachedLensCpu)
        .def_rw("query_start_loc", &CModelAttnMetaV2::queryStartLoc)
        .def_rw("slot_mapping", &CModelAttnMetaV2::slotMapping)
        .def_rw("block_tables", &CModelAttnMetaV2::blockTables)
        .def_rw("block_tables_seq_mapping", &CModelAttnMetaV2::blockTablesSeqMapping)
        .def_rw("block_tables_per_layer", &CModelAttnMetaV2::blockTablesPerLayer)
        .def_rw("slot_mapping_per_layer", &CModelAttnMetaV2::slotMappingPerLayer);

    m.class_<_CModel>("Model")
        .def(torch::init<>())
        // Weight tensor fields (at::Tensor / std::vector<at::Tensor>) — IValue
        // supports these directly, so def_readwrite ports the pybind11
        // .def_readwrite lines verbatim. Field names mirror xlite/_C.pyi.
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
        .def_readwrite("mla_wuv", &_CModel::mlaWUV)
        .def_readwrite("mla_wuk_t", &_CModel::mlaWUKT)
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
        .def_readwrite("linear_in_proj_qkv", &_CModel::linearInProjQKV)
        .def_readwrite("linear_in_proj_z", &_CModel::linearInProjZ)
        .def_readwrite("linear_in_proj_b", &_CModel::linearInProjB)
        .def_readwrite("linear_in_proj_a", &_CModel::linearInProjA)
        .def_readwrite("linear_conv1d", &_CModel::linearConv1d)
        .def_readwrite("linear_a_log", &_CModel::linearALog)
        .def_readwrite("linear_dt_bias", &_CModel::linearDtBias)
        .def_readwrite("linear_norm", &_CModel::linearNorm)
        .def_readwrite("linear_out_proj", &_CModel::linearOutProj)
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
        .def_readwrite("tid2eid", &_CModel::moeTid2Eid)
        .def_readwrite("se_up_gate", &_CModel::moeSEUpGate)
        .def_readwrite("se_up_gate_deq_scale", &_CModel::moeSEUpGateDeqScale)
        .def_readwrite("se_down", &_CModel::moeSEDown)
        .def_readwrite("se_down_deq_scale", &_CModel::moeSEDownDeqScale)
        .def_readwrite("se_gate", &_CModel::moeSEGate)
        .def_readwrite("re_up_gate", &_CModel::moeREUpGate)
        .def_readwrite("re_up_gate_scale", &_CModel::moeREUpGateDeqScale)
        .def_readwrite("re_up_gate_deq_scale", &_CModel::moeREUpGateDeqScale)
        .def_readwrite("re_down", &_CModel::moeREDown)
        .def_readwrite("re_down_scale", &_CModel::moeREDownDeqScale)
        .def_readwrite("re_down_deq_scale", &_CModel::moeREDownDeqScale)
        .def_readwrite("re_up_gate_scale_bias", &_CModel::moeREUpGateScaleBias)
        .def_readwrite("re_down_scale_bias", &_CModel::moeREDownScaleBias)
        .def_readwrite("log2phy", &_CModel::log2phy)
        // DeepSeek-V4 (CxA)
        .def_readwrite("attn_sink", &_CModel::attnSink)
        .def_readwrite("attn_wq_a", &_CModel::attnWqA)
        .def_readwrite("attn_wq_a_input_scale", &_CModel::attnWqAInputScale)
        .def_readwrite("attn_wq_a_input_offset", &_CModel::attnWqAInputOffset)
        .def_readwrite("attn_wq_a_quant_bias", &_CModel::attnWqAQuantBias)
        .def_readwrite("attn_wq_a_deq_scale", &_CModel::attnWqADeqScale)
        .def_readwrite("attn_wo_a", &_CModel::attnWoA)
        .def_readwrite("attn_wo_b", &_CModel::attnWoB)
        .def_readwrite("attn_wkv", &_CModel::attnWKv)
        .def_readwrite("attn_wkv_input_scale", &_CModel::attnWKvInputScale)
        .def_readwrite("attn_wkv_input_offset", &_CModel::attnWKvInputOffset)
        .def_readwrite("attn_wkv_quant_bias", &_CModel::attnWKvQuantBias)
        .def_readwrite("attn_wkv_deq_scale", &_CModel::attnWKvDeqScale)
        .def_readwrite("comp_ape", &_CModel::compApe)
        .def_readwrite("comp_w_kv", &_CModel::compWKv)
        .def_readwrite("comp_w_gate", &_CModel::compWGate)
        .def_readwrite("comp_norm", &_CModel::compNorm)
        .def_readwrite("idx_wq_b", &_CModel::idxWqB)
        .def_readwrite("idx_wq_b_input_scale", &_CModel::idxWqBInputScale)
        .def_readwrite("idx_wq_b_input_offset", &_CModel::idxWqBInputOffset)
        .def_readwrite("idx_wq_b_quant_bias", &_CModel::idxWqBQuantBias)
        .def_readwrite("idx_wq_b_deq_scale", &_CModel::idxWqBDeqScale)
        .def_readwrite("idx_weights_proj", &_CModel::idxWeightsProj)
        .def_readwrite("idx_comp_ape", &_CModel::idxCompApe)
        .def_readwrite("idx_comp_w_kv", &_CModel::idxCompWKv)
        .def_readwrite("idx_comp_w_gate", &_CModel::idxCompWGate)
        .def_readwrite("idx_comp_norm", &_CModel::idxCompNorm)
        .def_readwrite("hc_attn_fn", &_CModel::hcAttnFn)
        .def_readwrite("hc_ffn_fn", &_CModel::hcFfnFn)
        .def_readwrite("hc_attn_base", &_CModel::hcAttnBase)
        .def_readwrite("hc_ffn_base", &_CModel::hcFfnBase)
        .def_readwrite("hc_attn_scale", &_CModel::hcAttnScale)
        .def_readwrite("hc_ffn_scale", &_CModel::hcFfnScale)
        .def_readwrite("hc_head_fn", &_CModel::hcHeadFn)
        .def_readwrite("hc_head_base", &_CModel::hcHeadBase)
        .def_readwrite("hc_head_scale", &_CModel::hcHeadScale)
        .def_readwrite("linear_in_proj_q", &_CModel::linearInProjQ)
        .def_readwrite("linear_in_proj_k", &_CModel::linearInProjK)
        .def_readwrite("linear_in_proj_v", &_CModel::linearInProjV)
        .def_readwrite("linear_in_proj_f_a", &_CModel::linearInProjFA)
        .def_readwrite("linear_f_b_proj", &_CModel::linearFBProj)
        .def_readwrite("linear_in_proj_g_a", &_CModel::linearInProjGA)
        .def_readwrite("linear_g_b_proj", &_CModel::linearGBProj)
        .def_readwrite("linear_q_conv1d", &_CModel::linearQConv1d)
        .def_readwrite("linear_k_conv1d", &_CModel::linearKConv1d)
        .def_readwrite("linear_v_conv1d", &_CModel::linearVConv1d)
        .def_readwrite("index_wk", &_CModel::indexWK)
        .def_readwrite("index_weights_proj", &_CModel::indexWeightsProj)
        .def_readwrite("index_kpool_gate", &_CModel::indexKpoolGate)
        .def_readwrite("index_kpool_ape", &_CModel::indexKpoolApe)
        // Methods. torch::class_ has no overloading, so each forward variant
        // gets a distinct name; the V1 adapter keeps the plain `forward` name.
        //
        // GIL: pybind11 used call_guard<gil_scoped_release> on these (and
        // CoreAssigner below). torch::class_ has no call_guard, but the invoke
        // bridge (runAndInsertCall, pybind_utils.h) runs callee.run(stack) under
        // gil_scoped_release — equivalent; verified by assign_core blocking on
        // cv.wait without stalling other Python threads. If torch changes that
        // bridge, blocking methods (assign_core, forward) could hold the GIL.
        .def("init", &model_init, "",
             {torch::arg("config"), torch::arg("rank") = static_cast<int64_t>(0)})
        .def("forward", &model_forward_v1, "",
             {torch::arg("rt"), torch::arg("input"), torch::arg("attn_meta"),
              torch::arg("kv_cache"), torch::arg("freqs_cis"), torch::arg("output"),
              torch::arg("aux_outputs") = torch::arg::none(),
              torch::arg("curr_stream") = static_cast<int64_t>(0)})
        .def("forward_get_logits", &model_forward_get_logits, "",
             {torch::arg("rt"), torch::arg("input"), torch::arg("indices"), torch::arg("output"),
              torch::arg("curr_stream") = static_cast<int64_t>(0)})
        .def("forward_and_get_logits", &model_forward_and_get_logits, "",
             {torch::arg("rt"), torch::arg("input"), torch::arg("attn_meta"),
              torch::arg("kv_cache"), torch::arg("freqs_cis"), torch::arg("indices"),
              torch::arg("output"), torch::arg("curr_stream") = static_cast<int64_t>(0)})
        .def("forward_with_inputs_embeds", &model_forward_with_inputs_embeds_v1, "",
             {torch::arg("rt"), torch::arg("input"), torch::arg("attn_meta"),
              torch::arg("kv_cache"), torch::arg("freqs_cis"), torch::arg("output"),
              torch::arg("aux_outputs") = torch::arg::none(),
              torch::arg("curr_stream") = static_cast<int64_t>(0),
              torch::arg("deepstack_input") = std::vector<at::Tensor>{},
              torch::arg("input_ids") = torch::arg::none()})
        .def("forward_v2", &model_forward_v2, "",
             {torch::arg("rt"), torch::arg("input"), torch::arg("attn_meta"),
              torch::arg("kv_cache"), torch::arg("freqs_cis"), torch::arg("output"),
              torch::arg("aux_outputs") = torch::arg::none(),
              torch::arg("curr_stream") = static_cast<int64_t>(0)})
        .def("forward_and_get_logits_v2", &model_forward_and_get_logits_v2, "",
             {torch::arg("rt"), torch::arg("input"), torch::arg("attn_meta"),
              torch::arg("kv_cache"), torch::arg("freqs_cis"), torch::arg("indices"),
              torch::arg("output"), torch::arg("curr_stream") = static_cast<int64_t>(0)})
        .def("forward_with_inputs_embeds_v2", &model_forward_with_inputs_embeds_v2, "",
             {torch::arg("rt"), torch::arg("input"), torch::arg("attn_meta"),
              torch::arg("kv_cache"), torch::arg("freqs_cis"), torch::arg("output"),
              torch::arg("aux_outputs") = torch::arg::none(),
              torch::arg("curr_stream") = static_cast<int64_t>(0),
              torch::arg("deepstack_input") = std::vector<at::Tensor>{},
              torch::arg("input_ids") = torch::arg::none()})
        .def("get_tensor_pool_size", &model_get_tensor_pool_size, "", {torch::arg("dbg") = 0});

    m.class_<CoreAssignerHolder>("CoreAssigner")
        .def(torch::init<double>(), "", {torch::arg("prefill_ratio")})
        .def("assign_core", &CoreAssignerHolder::assignCore, "", {torch::arg("is_decode")})
        .def("release_core", &CoreAssignerHolder::releaseCore, "", {torch::arg("is_decode")});
    using opreg::op;
    opreg::OpDefs ops(m);
    ops.def("add", op<&Add>, {torch::arg("rt"), torch::arg("x"), torch::arg("y"), torch::arg("z")})
        .def("all_gather", op<&AllGather>,
             {torch::arg("rt"), torch::arg("out"), torch::arg("in_"),
              torch::arg("comm_type") = static_cast<int64_t>(0)})
        .def("reduce_scatter", op<&ReduceScatter>,
             {torch::arg("rt"), torch::arg("out"), torch::arg("in_"),
              torch::arg("comm_type") = static_cast<int64_t>(0)})
        .def("all_reduce", op<&AllReduce>,
             {torch::arg("rt"), torch::arg("out"), torch::arg("in_"),
              torch::arg("comm_type") = static_cast<int64_t>(0)})
        .def("alltoallv", op<&AlltoAllV>,
             {torch::arg("rt"), torch::arg("out"), torch::arg("in_"), torch::arg("send_counts"),
              torch::arg("recv_counts"), torch::arg("sdispls"), torch::arg("rdispls"),
              torch::arg("comm_type") = static_cast<int64_t>(0)})
        .def("dsa_cp_metadata", op<&DsaCpMetadata>,
             {torch::arg("rt"), torch::arg("query_start_loc"), torch::arg("query_lens"),
              torch::arg("cached_lens"), torch::arg("local_query_start_loc"),
              torch::arg("local_query_lens"), torch::arg("local_cached_lens"),
              torch::arg("local_slot_mapping"), torch::arg("shard_begin"), torch::arg("shard_end")})
        .def("dsa_cp_write_kv_cache_by_slot", op<&DsaCpWriteKVCacheBySlot>,
             {torch::arg("rt"), torch::arg("input"), torch::arg("cache"),
              torch::arg("slot_mapping"), torch::arg("num_tokens"), torch::arg("block_size")})
        .def("dsa_cp_reorder_wo", op<&DsaCpReorderWo>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("out"), torch::arg("ranks"),
              torch::arg("column_blocks"), torch::arg("block_bytes")})
        .def("bench_empty", op<&BenchEmpty>,
             {torch::arg("rt"), torch::arg("variant"),
              torch::arg("block_num") = static_cast<int64_t>(0)})
        .def("bench_scalar", op<&BenchScalar>,
             {torch::arg("rt"), torch::arg("variant"), torch::arg("out"), torch::arg("seed") = 3.0,
              torch::arg("iters") = static_cast<int64_t>(1000000),
              torch::arg("block_num") = static_cast<int64_t>(0)});
#ifdef MICROBENCH_TIMED
    ops.def("bench_timed", op<&BenchTimed>,
            {torch::arg("rt"), torch::arg("variant"), torch::arg("out"), torch::arg("ticks"),
             torch::arg("seed") = 3.0, torch::arg("iters") = static_cast<int64_t>(1000000),
             torch::arg("block_num") = static_cast<int64_t>(0)})
        .def("bench_mte", op<&BenchMte>,
             {torch::arg("rt"), torch::arg("variant"), torch::arg("src"), torch::arg("dst"),
              torch::arg("ticks"), torch::arg("bytes"),
              torch::arg("iters") = static_cast<int64_t>(1),
              torch::arg("block_num") = static_cast<int64_t>(0)})
        .def("bench_vec", op<&BenchVec>,
             {torch::arg("rt"), torch::arg("variant"), torch::arg("src"), torch::arg("out"),
              torch::arg("ticks"), torch::arg("elems"),
              torch::arg("iters") = static_cast<int64_t>(1),
              torch::arg("block_num") = static_cast<int64_t>(0)})
        .def("bench_cube", op<&BenchCube>,
             {torch::arg("rt"), torch::arg("variant"), torch::arg("a"), torch::arg("b"),
              torch::arg("c"), torch::arg("ticks"), torch::arg("m0"), torch::arg("n0"),
              torch::arg("k0"), torch::arg("block_num") = static_cast<int64_t>(0)});
#endif  // MICROBENCH_TIMED
    ops.def("matmul", op<&Matmul>,
            {torch::arg("rt"), torch::arg("x"), torch::arg("y"), torch::arg("z"),
             torch::arg("weight_nz") = false, torch::arg("transpose") = false})
        .def("matmul_bench", op<&MatmulBench>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("y"), torch::arg("z"),
              torch::arg("x_warmup"), torch::arg("y_warmup"), torch::arg("z_warmup"),
              torch::arg("iterations"), torch::arg("warmup_iterations"),
              torch::arg("weight_nz") = false, torch::arg("transpose") = false})
        .def("matmul_with_bias", op<&MatmulWithBias>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("y"), torch::arg("z"),
              torch::arg("bias"), torch::arg("weight_nz") = false})
        .def("embed", op<&Embed>,
             {torch::arg("rt"), torch::arg("weight"), torch::arg("in_"), torch::arg("out"),
              torch::arg("start"), torch::arg("end")})
        .def("rmsnorm", op<&RMSNorm>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("norm"), torch::arg("out"),
              torch::arg("norm_eps"), torch::arg("norm_dim") = static_cast<int64_t>(0),
              torch::arg("cnt_per_token") = static_cast<int64_t>(1),
              torch::arg("in_start_offset") = static_cast<int64_t>(0),
              torch::arg("out_start_offset") = static_cast<int64_t>(0),
              torch::arg("variance") = torch::arg::none()})
        .def("rmsnorm_variance_only", op<&RMSNormVarianceOnly>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("out"), torch::arg("norm_eps"),
              torch::arg("norm_dim") = static_cast<int64_t>(0),
              torch::arg("cnt_per_token") = static_cast<int64_t>(1),
              torch::arg("in_start_offset") = static_cast<int64_t>(0),
              torch::arg("out_start_offset") = static_cast<int64_t>(0)})
        .def("rmsnorm_with_bias", op<&RMSNormWithBias>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("norm"), torch::arg("norm_bias"),
              torch::arg("out"), torch::arg("norm_eps"),
              torch::arg("norm_dim") = static_cast<int64_t>(0),
              torch::arg("cnt_per_token") = static_cast<int64_t>(1),
              torch::arg("in_start_offset") = static_cast<int64_t>(0),
              torch::arg("out_start_offset") = static_cast<int64_t>(0)})
        .def("layernorm", op<&LayerNorm>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("norm"), torch::arg("norm_bias"),
              torch::arg("out"), torch::arg("norm_eps"), torch::arg("norm_dim")})
        .def("l2norm", op<&L2Norm>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("out"), torch::arg("norm_eps"),
              torch::arg("norm_dim") = static_cast<int64_t>(0)})
        .def("add_bias", op<&AddBias>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("weight"), torch::arg("out")})
        .def("silu_and_mul", op<&SiluAndMul>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("out"),
              torch::arg("swiglu_limit") = 0.0})
        .def("sigmoid_gate_mul", op<&SigmoidGateMul>,
             {torch::arg("rt"), torch::arg("attn"), torch::arg("gate"), torch::arg("out"),
              torch::arg("add") = torch::arg::none()})
        .def("rope_and_cache", op<&RopeAndCache>,
             {torch::arg("rt"), torch::arg("inout"), torch::arg("k_cache"), torch::arg("v_cache"),
              torch::arg("position"), torch::arg("cosin"), torch::arg("slot_mapping"),
              torch::arg("n_heads"), torch::arg("n_kv_heads"), torch::arg("head_dim"),
              torch::arg("rot_dim"), torch::arg("block_size"), torch::arg("is_neox"),
              torch::arg("mrope_mask_h") = static_cast<int64_t>(0),
              torch::arg("mrope_mask_w") = static_cast<int64_t>(0)})
        .def("attention", op<&Attention>,
             {torch::arg("rt"), torch::arg("qkv"), torch::arg("k_cache"), torch::arg("v_cache"),
              torch::arg("output"), torch::arg("query_start_loc"), torch::arg("lens"),
              torch::arg("cached_lens"), torch::arg("block_tables"), torch::arg("n_heads"),
              torch::arg("n_kv_heads"), torch::arg("head_dim"), torch::arg("block_size"),
              torch::arg("batch"), torch::arg("enable_flash_attention") = false,
              torch::arg("tile_size_of_cached_kv") = static_cast<int64_t>(8192)})
        .def("add_and_rmsnorm", op<&AddAndRMSNorm>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("add_in_out"), torch::arg("norm"),
              torch::arg("out"), torch::arg("norm_eps")})
        .def("softmax_topk", op<&SoftmaxTopK>,
             {torch::arg("rt"), torch::arg("scores"), torch::arg("indices"),
              torch::arg("out_weights"), torch::arg("out_routing"), torch::arg("top_k"),
              torch::arg("norm_top_k_prob")})
        .def("sigmoid_topk", op<&SigmoidTopK>,
             {torch::arg("rt"), torch::arg("scores"), torch::arg("indices"), torch::arg("bias"),
              torch::arg("scale"), torch::arg("out_weights"), torch::arg("out_routing"),
              torch::arg("n_group"), torch::arg("n_topk_group"), torch::arg("top_k"),
              torch::arg("norm_top_k_prob")})
        .def("sqrtsoftplus_hash_topk", op<&SqrtsoftplusHashTopK>,
             {torch::arg("rt"), torch::arg("scores"), torch::arg("indices"), torch::arg("bias"),
              torch::arg("input_ids"), torch::arg("tid2eid"), torch::arg("out_weights"),
              torch::arg("routing_map"), torch::arg("scale"), torch::arg("top_k"),
              torch::arg("hash")})
        .def("topk", op<&TopK>,
             {torch::arg("rt"), torch::arg("scores"), torch::arg("indices"),
              torch::arg("outIndices"), torch::arg("query_lens"), torch::arg("cached_lens"),
              torch::arg("k")})
        .def("cast_up", op<&CastUp>, {torch::arg("rt"), torch::arg("in_"), torch::arg("out")})
        .def("cast_down", op<&CastDown>, {torch::arg("rt"), torch::arg("in_"), torch::arg("out")})
        .def("permutation", op<&Permutation>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("routing"), torch::arg("start"),
              torch::arg("end"), torch::arg("out"), torch::arg("unp_idx"), torch::arg("counts"),
              torch::arg("scale_in") = torch::arg::none(),
              torch::arg("scale_out") = torch::arg::none()})
        .def("unpermutation", op<&UnPermutation>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("routing"), torch::arg("weights"),
              torch::arg("start"), torch::arg("end"), torch::arg("out"), torch::arg("unp_idx")})
        .def("remap_gate_outputs", op<&RemapGateOutputs>,
             {torch::arg("rt"), torch::arg("w_in"), torch::arg("r_in"), torch::arg("w_out"),
              torch::arg("r_out"), torch::arg("log2phy")})
        .def("group_matmul", op<&GroupMatmul>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("weights"), torch::arg("scales"),
              torch::arg("counts"), torch::arg("start"), torch::arg("end"), torch::arg("out_dim"),
              torch::arg("in_dim"), torch::arg("output"), torch::arg("weight_nz"),
              torch::arg("transpose")})
        .def("softmax", op<&Softmax>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("calc_len"), torch::arg("is_long")})
        .def(
            "rope_complex", op<&RopeComplex>,
            {torch::arg("rt"), torch::arg("n_local_heads"), torch::arg("step_dim"),
             torch::arg("rope_dim"), torch::arg("input_with_r"), torch::arg("freqs"),
             torch::arg("position"), torch::arg("output"), torch::arg("inverse") = false,
             torch::arg("out_interleaved") = false,
             torch::arg("min_position") = static_cast<int64_t>(0), torch::arg("do_rotate") = false})
        .def("rope_complex_and_cache", op<&RopeComplexAndCache>,
             {torch::arg("rt"), torch::arg("n_local_heads"), torch::arg("step_dim"),
              torch::arg("rope_dim"), torch::arg("offset"), torch::arg("vdim"),
              torch::arg("input_with_r"), torch::arg("freqs"), torch::arg("position"),
              torch::arg("block_size"), torch::arg("v_cache"), torch::arg("slot_mapping"),
              torch::arg("out_interleaved") = false, torch::arg("do_rotate") = false})
        .def("mla_prepare", op<&MlaPrepare>,
             {torch::arg("rt"), torch::arg("attn_qkvc"), torch::arg("q_norm"),
              torch::arg("q_norm_bias"), torch::arg("attn_norm_qc"), torch::arg("kv_norm"),
              torch::arg("kv_norm_bias"), torch::arg("attn_norm_kvc"), torch::arg("freqs"),
              torch::arg("position"), torch::arg("q_lora_rank"), torch::arg("kv_lora_rank"),
              torch::arg("rope_head_dim"), torch::arg("block_size"), torch::arg("k_cache"),
              torch::arg("pe_cache"), torch::arg("slot_mapping"), torch::arg("norm_eps"),
              torch::arg("k_scale_cache") = torch::arg::none()})
        .def("mla_cache_c8", op<&MlaCacheC8>,
             {torch::arg("rt"), torch::arg("latent"), torch::arg("pe"), torch::arg("slot_mapping"),
              torch::arg("cache"), torch::arg("pe_cache"), torch::arg("k_scale_cache")})
        .def("indexer_prepare", op<&IndexerPrepare>,
             {torch::arg("rt"),
              torch::arg("kw"),
              torch::arg("k_norm"),
              torch::arg("k_norm_bias"),
              torch::arg("freqs"),
              torch::arg("position"),
              torch::arg("index_head_dim"),
              torch::arg("index_n_heads"),
              torch::arg("rope_head_dim"),
              torch::arg("block_size"),
              torch::arg("index_k_cache"),
              torch::arg("slot_mapping"),
              torch::arg("norm_eps"),
              torch::arg("q"),
              torch::arg("scale"),
              torch::arg("top_k"),
              torch::arg("is_long"),
              torch::arg("k_scale_cache") = torch::arg::none(),
              torch::arg("q8") = torch::arg::none(),
              torch::arg("q_scale") = torch::arg::none(),
              torch::arg("scaled_weights") = torch::arg::none()})
        .def("indexer_k_cache_c8", op<&IndexerKCacheC8>,
             {torch::arg("rt"), torch::arg("k"), torch::arg("k_norm"), torch::arg("k_norm_bias"),
              torch::arg("freqs"), torch::arg("position"), torch::arg("slot_mapping"),
              torch::arg("k_cache"), torch::arg("scale_cache"), torch::arg("norm_eps") = 1e-06})
        .def("quant_cache_c8", op<&QuantCacheC8>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("slot_mapping"), torch::arg("k_cache"),
              torch::arg("k_scale_cache"), torch::arg("block_size")})
        .def("quant_q_c8", op<&QuantQC8>,
             {torch::arg("rt"), torch::arg("q"), torch::arg("head_weight"), torch::arg("q8"),
              torch::arg("scaled_weights"), torch::arg("rotate_scale")})
        .def("compressor", op<&Compressor>,
             {torch::arg("rt"),
              torch::arg("kv"),
              torch::arg("score"),
              torch::arg("ape") = torch::arg::none(),
              torch::arg("norm") = torch::arg::none(),
              torch::arg("freqs") = torch::arg::none(),
              torch::arg("weighted_sum"),
              torch::arg("query_start_loc"),
              torch::arg("query_lens"),
              torch::arg("cached_lens") = torch::arg::none(),
              torch::arg("batch"),
              torch::arg("n_total_blocks"),
              torch::arg("ratio"),
              torch::arg("overlap"),
              torch::arg("head_dim"),
              torch::arg("rope_head_dim"),
              torch::arg("norm_eps"),
              torch::arg("compress_kv") = torch::arg::none(),
              torch::arg("compress_positions"),
              torch::arg("compress_slot_mapping"),
              torch::arg("compress_block_size"),
              torch::arg("state") = torch::arg::none(),
              torch::arg("state_slot_mapping") = torch::arg::none(),
              torch::arg("state_block_table") = torch::arg::none(),
              torch::arg("state_block_size"),
              torch::arg("do_rotate") = false})
        .def("quant", op<&Quant>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("scale_reciprocal"),
              torch::arg("offset"), torch::arg("out")})
        .def("quant_dynamic", op<&QuantDyn>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("scale"), torch::arg("out")})
        .def("msd_merge_dequant", op<&MSDMergeDequant>,
             {torch::arg("rt"), torch::arg("y_merged"), torch::arg("scale_biases"),
              torch::arg("counts"), torch::arg("per_token_scale"), torch::arg("out")})
        .def("matmul_dequant", op<&MatmulDeQuant>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("y"), torch::arg("bias"),
              torch::arg("deq_scale"), torch::arg("z"), torch::arg("weight_nz") = false,
              torch::arg("transpose") = false})
        .def("fusion_operator_matmul_dequant_pipeline", op<&FusionOperatorMatmulDequantPipeline>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("weight"), torch::arg("out"),
              torch::arg("bias"), torch::arg("deq_scale"), torch::arg("weight_nz") = false,
              torch::arg("transpose") = false, torch::arg("out_scale"), torch::arg("num")})
        .def("fused_quant_matmul_dequant", op<&FusedQuantMatmulDequant>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("quant_scale_rec") = torch::arg::none(),
              torch::arg("quant_offset") = torch::arg::none(), torch::arg("weight"),
              torch::arg("out"), torch::arg("bias") = torch::arg::none(),
              torch::arg("deq_scale") = torch::arg::none(), torch::arg("x_quant"),
              torch::arg("out_scale") = torch::arg::none(), torch::arg("num") = torch::arg::none(),
              torch::arg("dyn_quant"), torch::arg("weight_nz") = false,
              torch::arg("transpose") = false})
        .def("fused_group_matmul_dequant", op<&FusedGroupMatmulDequant>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("weights"), torch::arg("scales"),
              torch::arg("counts"), torch::arg("start"), torch::arg("end"), torch::arg("out_dim"),
              torch::arg("in_dim"), torch::arg("output"),
              torch::arg("out_scale") = torch::arg::none(), torch::arg("weight_nz") = false,
              torch::arg("transpose") = false})
        .def("dequant", op<&DeQuant>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("scale"), torch::arg("out"),
              torch::arg("has_scale")})
        .def("mla_v2", op<&MLAV2>,
             {torch::arg("rt"),
              torch::arg("q_with_qr"),
              torch::arg("qr"),
              torch::arg("k_cache"),
              torch::arg("pe_cache"),
              torch::arg("wuk_t"),
              torch::arg("wuv"),
              torch::arg("output"),
              torch::arg("query_start_loc"),
              torch::arg("lens"),
              torch::arg("cached_lens"),
              torch::arg("block_tables"),
              torch::arg("n_heads"),
              torch::arg("rope_head_dim"),
              torch::arg("nope_head_dim"),
              torch::arg("v_head_dim"),
              torch::arg("kv_lora_rank"),
              torch::arg("block_size"),
              torch::arg("batch"),
              torch::arg("scale"),
              torch::arg("topk_indices"),
              torch::arg("top_k") = static_cast<int64_t>(0),
              torch::arg("nz") = false,
              torch::arg("enable_flash_attention") = false,
              torch::arg("tile_size_of_cached_kv") = static_cast<int64_t>(8192),
              torch::arg("dense") = false,
              torch::arg("k_scale_cache") = torch::arg::none()})
        .def("dsa_cp_gather_mla", op<&DsaCpGatherMLA>,
             {torch::arg("rt"), torch::arg("q_absorb"), torch::arg("qr"), torch::arg("k_cache"),
              torch::arg("pe_cache"), torch::arg("output"), torch::arg("query_start_loc"),
              torch::arg("query_lens"), torch::arg("cached_lens"), torch::arg("block_tables"),
              torch::arg("topk_indices"), torch::arg("scale")})
        .def("gather_sparse_kv_cache", op<&GatherSparseKVCache>,
             {torch::arg("rt"), torch::arg("k_cache"), torch::arg("pe_cache"),
              torch::arg("block_tables"), torch::arg("topk_indices"), torch::arg("query_start_loc"),
              torch::arg("lens"), torch::arg("cached_lens"), torch::arg("k_dense_cache"),
              torch::arg("pe_dense_cache"), torch::arg("batch"), torch::arg("index_topk"),
              torch::arg("block_size"), torch::arg("kv_lora_rank"), torch::arg("rope_head_dim"),
              torch::arg("kv_heads") = static_cast<int64_t>(1),
              torch::arg("compress_ratio") = static_cast<int64_t>(1),
              torch::arg("burst_copy_size") = static_cast<int64_t>(GM_UB_BURST_SIZE),
              torch::arg("k_scale_cache") = torch::arg::none()})
        .def("cxa", op<&CXA>,
             {torch::arg("rt"),
              torch::arg("q"),
              torch::arg("swa_k_cache"),
              torch::arg("compress_k_cache"),
              torch::arg("swa_block_tables"),
              torch::arg("compress_block_tables"),
              torch::arg("swa_block_size"),
              torch::arg("compress_block_size"),
              torch::arg("attn_sink"),
              torch::arg("output"),
              torch::arg("batch"),
              torch::arg("query_start_loc"),
              torch::arg("lens"),
              torch::arg("cached_lens"),
              torch::arg("n_heads"),
              torch::arg("head_dim"),
              torch::arg("scale"),
              torch::arg("window_size"),
              torch::arg("compress_ratio"),
              torch::arg("index_topk"),
              torch::arg("topk_indices"),
              torch::arg("enable_flash_attention") = false,
              torch::arg("tile_size_of_cached_kv") = static_cast<int64_t>(8192),
              torch::arg("dense") = false})
        .def("indexer_scores", op<&IndexerScores>,
             {torch::arg("rt"), torch::arg("q"), torch::arg("k_cache"), torch::arg("weight"),
              torch::arg("scores"), torch::arg("query_start_loc"), torch::arg("lens"),
              torch::arg("cached_lens"), torch::arg("block_tables"), torch::arg("n_heads"),
              torch::arg("head_dim"), torch::arg("block_size"), torch::arg("batch")})
        .def("indexer_topk", op<&IndexerTopK>,
             {torch::arg("rt"),
              torch::arg("q"),
              torch::arg("k_cache"),
              torch::arg("weight"),
              torch::arg("topk_indices"),
              torch::arg("query_start_loc"),
              torch::arg("lens"),
              torch::arg("cached_lens"),
              torch::arg("block_tables"),
              torch::arg("n_heads"),
              torch::arg("head_dim"),
              torch::arg("block_size"),
              torch::arg("batch"),
              torch::arg("top_k"),
              torch::arg("k_scale_cache") = torch::arg::none(),
              torch::arg("compress_ratio") = static_cast<int64_t>(1),
              torch::arg("kv_split") = static_cast<int64_t>(0),
              torch::arg("q_tile") = static_cast<int64_t>(0),
              torch::arg("pack_tail") = static_cast<int64_t>(0),
              torch::arg("relu_qk") = static_cast<int64_t>(1),
              torch::arg("out_stride") = static_cast<int64_t>(0)})
        .def("muls", muls_op,
             {torch::arg("rt"), torch::arg("input"), torch::arg("scale"), torch::arg("output"),
              torch::arg("calc_offset") = static_cast<int64_t>(0),
              torch::arg("calc_num") = static_cast<int64_t>(4294967295)})
        .def(
            "experts_counts_sum", op<&ExpertsCountsSum>,
            {torch::arg("rt"), torch::arg("experts_counts_input"), torch::arg("tokens_per_epgroup"),
             torch::arg("experts_counts_output"), torch::arg("n_routed_experts")})
        .def("reorder_moe", op<&ReorderMoE>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("out"), torch::arg("counts"),
              torch::arg("hidden_size"), torch::arg("local_start"), torch::arg("local_end"),
              torch::arg("forward")})
        .def("linear_att_proj", op<&LinearAttProj>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("W_qkv"), torch::arg("W_z"),
              torch::arg("W_b"), torch::arg("W_a"), torch::arg("mix_qkv"), torch::arg("z"),
              torch::arg("b"), torch::arg("a"), torch::arg("m"), torch::arg("n"), torch::arg("v"),
              torch::arg("h"), torch::arg("k")})
        .def("transpose_1_2", op<&Transpose_1_2>,
             {torch::arg("rt"), torch::arg("input"), torch::arg("output")})
        .def("linear_att_conv_and_silu_paged", op<&LinearAttConv1dAndSiLUPaged>,
             {torch::arg("rt"), torch::arg("mix_qkv"), torch::arg("conv_state"),
              torch::arg("weight"), torch::arg("output"), torch::arg("state_indices"),
              torch::arg("cached_lens") = torch::arg::none(),
              torch::arg("state_block_size") = static_cast<int64_t>(0),
              torch::arg("aligned") = false, torch::arg("state_layout") = static_cast<int64_t>(1),
              torch::arg("chan_offset") = static_cast<int64_t>(0)})
        .def("split_col", op<&SplitCol>,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("outputs")})
        .def("concat", op<&Concat>, {torch::arg("rt"), torch::arg("inputs"), torch::arg("out")})
        .def("concat_col", op<&ConcatCol>,
             {torch::arg("rt"), torch::arg("inputs"), torch::arg("out")})
        .def("split", split_op,
             {torch::arg("rt"), torch::arg("in_"), torch::arg("outputs"), torch::arg("sizes"),
              torch::arg("num_packets")})
        .def("beta_decay", op<&BetaDecay>,
             {torch::arg("rt"), torch::arg("b"), torch::arg("a"), torch::arg("A_log"),
              torch::arg("dt_bias"), torch::arg("beta"), torch::arg("g"), torch::arg("bsz"),
              torch::arg("seqlen"), torch::arg("num_v_heads")})
        .def("einsum_mht_hdt_mhd", op<&EinsumMhtHdtMhd>,
             {torch::arg("rt"), torch::arg("mht"), torch::arg("hdt"), torch::arg("mhd"),
              torch::arg("m"), torch::arg("h"), torch::arg("t"), torch::arg("d"),
              torch::arg("weight_nz") = false})
        .def("einsum_mht_htd_mhd", op<&EinsumMhtHtdMhd>,
             {torch::arg("rt"), torch::arg("mht"), torch::arg("htd"), torch::arg("mhd"),
              torch::arg("m"), torch::arg("h"), torch::arg("t"), torch::arg("d"),
              torch::arg("weight_nz") = false})
        .def("unpack_activation", op<&UnpackActivation>,
             {torch::arg("rt"), torch::arg("input"), torch::arg("output")})
        .def("hc_act", op<&HcAct>,
             {torch::arg("rt"), torch::arg("mixes"), torch::arg("hc_scale"), torch::arg("hc_base"),
              torch::arg("post"), torch::arg("comb"), torch::arg("hc_mult"), torch::arg("eps"),
              torch::arg("sinkhorn_iters"), torch::arg("x_resid"), torch::arg("output")})
        .def("hc_post", op<&HcPost>,
             {torch::arg("rt"), torch::arg("x"), torch::arg("post"), torch::arg("comb"),
              torch::arg("residual"), torch::arg("y"), torch::arg("m"), torch::arg("hc_mult"),
              torch::arg("hidden")})
        .def("hc_split_sinkhorn", op<&HcSplitSinkhorn>,
             {torch::arg("rt"), torch::arg("mixes"), torch::arg("hc_scale"), torch::arg("hc_base"),
              torch::arg("pre"), torch::arg("post"), torch::arg("comb"), torch::arg("hc_mult"),
              torch::arg("eps"), torch::arg("sinkhorn_iters")})
        .def("hc_pre", op<&HcPre>,
             {torch::arg("rt"), torch::arg("x_resid"), torch::arg("pre"), torch::arg("output"),
              torch::arg("m"), torch::arg("hc_mult"), torch::arg("hidden")})
        .def("silu_gate_mul", op<&SiluGateMul>,
             {torch::arg("rt"), torch::arg("normed"), torch::arg("gate"), torch::arg("out")})
        .def("recurrent_gated_delta_rule_prefill_paged", op<&RecurrentGatedDeltaRulePrefillPaged>,
             {torch::arg("rt"), torch::arg("query"), torch::arg("key"), torch::arg("value"),
              torch::arg("beta"), torch::arg("g"), torch::arg("state"), torch::arg("out"),
              torch::arg("indices"), torch::arg("starts"), torch::arg("initial"),
              torch::arg("num_key_heads"), torch::arg("num_value_heads"), torch::arg("k_dim"),
              torch::arg("v_dim")})
        .def("conv1d_and_silu_prefill_paged", op<&Conv1dAndSiLUPrefillPaged>,
             {torch::arg("rt"), torch::arg("input"), torch::arg("state"), torch::arg("weight"),
              torch::arg("output"), torch::arg("indices"), torch::arg("starts"),
              torch::arg("initial"), torch::arg("state_layout") = static_cast<int64_t>(1),
              torch::arg("chan_offset") = static_cast<int64_t>(0)})
        .def("print", op<&Print>,
             {torch::arg("x"), torch::arg("name") = std::string(""),
              torch::arg("row") = static_cast<int64_t>(6),
              torch::arg("col") = static_cast<int64_t>(6)})
        .def("get_tile_size_of_cached_kv", get_tile_size_of_cached_kv_op,
             {torch::arg("cached_lens"), torch::arg("query_lens"), torch::arg("head_num_in_group"),
              torch::arg("n_kv_heads"), torch::arg("block_size"), torch::arg("aic_num")})
        .def("recurrent_gated_delta_rule_paged", op<&RecurrentGatedDeltaRulePaged>,
             {torch::arg("rt"), torch::arg("query"), torch::arg("key"), torch::arg("value"),
              torch::arg("beta"), torch::arg("g"), torch::arg("state"), torch::arg("out"),
              torch::arg("state_indices"), torch::arg("num_key_heads"),
              torch::arg("num_value_heads"), torch::arg("k_dim"), torch::arg("v_dim"),
              torch::arg("cached_lens") = torch::arg::none(),
              torch::arg("state_block_size") = static_cast<int64_t>(0),
              torch::arg("aligned") = false});
    ops.def("recurrent_gated_delta_rule_kda_paged", op<&RecurrentGatedDeltaRuleKdaPaged>,
            {torch::arg("rt"),
             torch::arg("query"),
             torch::arg("key"),
             torch::arg("value"),
             torch::arg("beta"),
             torch::arg("gate"),
             torch::arg("a_log"),
             torch::arg("dt_bias"),
             torch::arg("state"),
             torch::arg("out"),
             torch::arg("state_indices"),
             torch::arg("num_key_heads"),
             torch::arg("num_value_heads"),
             torch::arg("k_dim"),
             torch::arg("v_dim"),
             torch::arg("scale"),
             torch::arg("lower_bound"),
             torch::arg("norm_eps"),
             torch::arg("cached_lens") = torch::arg::none(),
             torch::arg("state_block_size") = static_cast<int64_t>(0),
             torch::arg("aligned") = false});
    ops.def("recurrent_gated_delta_rule_kda_prefill_paged",
            op<&RecurrentGatedDeltaRuleKdaPrefillPaged>,
            {torch::arg("rt"),          torch::arg("query"),         torch::arg("key"),
             torch::arg("value"),       torch::arg("beta"),          torch::arg("gate"),
             torch::arg("a_log"),       torch::arg("dt_bias"),       torch::arg("state"),
             torch::arg("out"),         torch::arg("indices"),       torch::arg("starts"),
             torch::arg("initial"),     torch::arg("num_key_heads"), torch::arg("num_value_heads"),
             torch::arg("k_dim"),       torch::arg("v_dim"),         torch::arg("scale"),
             torch::arg("lower_bound"), torch::arg("norm_eps")});
    ops.def("hc_mean_collapse", op<&HcMeanCollapse>,
            {torch::arg("rt"), torch::arg("input"), torch::arg("output"), torch::arg("m"),
             torch::arg("hc_mult"), torch::arg("hidden")});
    ops.def("kpool_tail_compress_and_spill", op<&KpoolTailCompressAndSpill>,
            {torch::arg("rt"), torch::arg("tail_cache"), torch::arg("indexer_cache"),
             torch::arg("k"), torch::arg("gate"), torch::arg("ape"), torch::arg("positions"),
             torch::arg("query_start_loc"), torch::arg("lens"), torch::arg("cached_lens"),
             torch::arg("slot_mapping"), torch::arg("tail_block_table"), torch::arg("head_dim"),
             torch::arg("kpool"), torch::arg("main_block_size")});
    ops.def(
        "engram_gate", op<&EngramGate>,
        {torch::arg("rt"), torch::arg("hidden"), torch::arg("kv"), torch::arg("q_weight"),
         torch::arg("k_weight"), torch::arg("mask"), torch::arg("out"), torch::arg("hc_mult"),
         torch::arg("dim"), torch::arg("eps"), torch::arg("hidden_residual") = torch::arg::none()});
}
