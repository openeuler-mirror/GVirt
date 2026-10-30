/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_OP_H_
#define _XLITE_OP_H_

#include "base.h"
#include "runtime.h"
#include "ascend.h"
#include "auto_tuner.h"
#include "kernels/kernel_param.h"

static_assert(MAX_KV_TILE_SIZE <= MAX_SOFTMAX_PINGPONG_LEN);

// Derive maxNumBlocks from blockTables: 2-D [batch, maxNumBlocks] -> shape[1];
// 1-D [batch * maxNumBlocks] -> len / batch. Avoids shape[1] UB on 1-D input.
static inline uint32_t DeriveMaxNumBlocks(const XTensor &blockTables, uint32_t batch)
{
    const auto &shape = blockTables.shape;
    if (shape.size() == 2) {
        return static_cast<uint32_t>(shape[1]);
    }
    if (shape.size() == 1) {
        if (batch == 0) {
            throw std::runtime_error(
                std::string(__func__) +
                ": batch == 0 when deriving maxNumBlocks from 1-D blockTables");
        }
        if (shape[0] % batch != 0) {
            throw std::runtime_error(std::string(__func__) + ": 1-D blockTables length (" +
                                     std::to_string(shape[0]) + ") is not divisible by batch (" +
                                     std::to_string(batch) + ")");
        }
        return static_cast<uint32_t>(shape[0] / batch);
    }
    throw std::runtime_error(std::string(__func__) + ": blockTables must be 1-D or 2-D, got rank " +
                             std::to_string(shape.size()));
}

HcclDataType XDtype2HcclDtype(enum XDtype dtype);
void XliteOpAllGather(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                      bool fetchOffset = false, DebugSrcLoc loc = UNKNOWN_DBG_LOC,
                      uint32_t copySize = COPY_SIZE);
void XliteOpReduceScatter(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                          bool fetchOffset = false, DebugSrcLoc loc = UNKNOWN_DBG_LOC,
                          uint32_t copySize = COPY_SIZE);
void XliteOpAllReduceSum(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                         bool fetchOffset = false, DebugSrcLoc loc = UNKNOWN_DBG_LOC,
                         uint32_t copySize = COPY_SIZE);
void XliteOpAlltoAllV(XRuntime &rt, XTensor &in, XTensor &out, XTensor &sendCounts,
                      XTensor &recvCounts, XTensor &sdispls, XTensor &rdispls, enum commType type,
                      DebugSrcLoc loc = UNKNOWN_DBG_LOC);

void XliteOpEmbed(XRuntime &rt, XTensor &in, XTensor &embed, uint32_t start, uint32_t end,
                  XTensor &out);
void XliteOpRmsNorm(XRuntime &rt, XTensor &in, const XTensor &norm, XTensor &out, float normEps,
                    uint32_t normDim, bool useNorm = true, const XTensor &normBias = XTensor(),
                    uint32_t cntPerToken = 1, uint32_t inStartOffset = 0,
                    uint32_t outStartOffset = 0, const XTensor &variance = XTensor());
void XliteOpLayerNorm(XRuntime &rt, XTensor &in, XTensor &norm, XTensor &normBias, XTensor &out,
                      float normEps, uint32_t normDim, uint32_t cntPerToken = 1,
                      uint32_t inStartOffset = 0, uint32_t outStartOffset = 0);
void XliteOpL2Norm(XRuntime &rt, XTensor &in, XTensor &out, float normEps, uint32_t normDim,
                   uint32_t cntPerToken = 1, uint32_t inStartOffset = 0,
                   uint32_t outStartOffset = 0);
void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out);
void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, bool weightNZ = false,
                   const XTensor &bias = XTensor(), const XTensor &deqScale = XTensor(),
                   bool transpose = false, uint64_t m0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
                   uint64_t n0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
                   uint64_t k0 = MATMUL_M0_N0_K0_DEFAULT_VALUE);

void XliteOpSiluAndMul(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &num = XTensor(),
                       float swigluLimit = 0.0f);
void XliteOpCastDown(XRuntime &rt, XTensor &in, XTensor &out, XTensor &outScale);
void XliteOpCastUp(XRuntime &rt, XTensor &in, XTensor &inScale, XTensor &out);
void XliteOpPermutation(XRuntime &rt, XTensor &in, XTensor &routing, uint32_t start, uint32_t end,
                        XTensor &out, XTensor &unpIdx, XTensor &counts);
void XliteOpUnpermutation(XRuntime &rt, XTensor &in, XTensor &unpIdx, XTensor &routing,
                          XTensor &weights, uint32_t start, uint32_t end, XTensor &out);
// EPLB: scatter gate outputs from logical [M, nRoutedExperts] to physical [M, totalPhysicalExperts]
// columns via log2phy (logical→physical). wIn/wOut are BF16/FP32; rIn/rOut are BIT1. rOut is
// zeroed by the op. log2phy is [nRoutedExperts] INT32.
void XliteOpRemapGateOutputs(XRuntime &rt, XTensor &wIn, XTensor &rIn, XTensor &wOut, XTensor &rOut,
                             const XTensor &log2phy);
void XliteOpGroupMatmul(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &deqScales,
                        XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                        long outDim, long inDim, XTensor &output, bool weightNZ = false,
                        bool transpose = false);
void XliteOpRopeCache(XRuntime &rt, XTensor &inout, XTensor &kCache, XTensor &vCache,
                      XTensor &position, XTensor &cossin, XTensor &slotMapping, uint32_t nHeads,
                      uint32_t nKvHeads, uint32_t headDim, uint32_t rotDim, uint32_t blockSize,
                      bool isNeox, uint64_t mropeMaskH, uint64_t mropeMaskW);
void XliteOpAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache, XTensor &qk,
                      XTensor &output, XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                      XTensor &blockTables, uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim,
                      uint32_t blockSize, uint32_t batch);
void XliteOpFlashAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache,
                           XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum, XTensor &lastMax,
                           XTensor &lastSum, XTensor &sync, XTensor &output, XTensor &queryStartLoc,
                           XTensor &lens, XTensor &cachedLens, XTensor &blockTables,
                           uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim, uint32_t blockSize,
                           uint32_t batch, uint32_t tileSizeOfCachedKV = MAX_KV_TILE_SIZE);
void XliteOpCXA(XRuntime &rt, XTensor &q, XTensor &swaKCache, XTensor &compressKCache,
                XTensor &swaBlockTables, XTensor &compressBlockTables, uint32_t swaBlockSize,
                uint32_t compressBlockSize, XTensor &attnSink, XTensor &scores, XTensor &output,
                uint32_t batch, XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                uint32_t nHeads, uint32_t headDim, float scale, uint32_t windowSize,
                uint32_t kvSize, uint32_t compressRatio, uint32_t indexTopK,
                const XTensor &topkIndices = XTensor(), bool dense = false);
// Flash (online-softmax) variant of CXA: tiles the compress-token (KV-len)
// dimension into tileSizeOfCachedKV chunks merged with online softmax. Sparse
// (paged) path only; dense decode uses the non-flash cxa kernel.
void XliteOpFlashCXA(XRuntime &rt, XTensor &q, XTensor &swaKCache, XTensor &compressKCache,
                     XTensor &swaBlockTables, XTensor &compressBlockTables, uint32_t swaBlockSize,
                     uint32_t compressBlockSize, XTensor &attnSink, XTensor &qk, XTensor &sv,
                     XTensor &max, XTensor &sum, XTensor &lastMax, XTensor &lastSum, XTensor &sync,
                     XTensor &output, uint32_t batch, XTensor &queryStartLoc, XTensor &lens,
                     XTensor &cachedLens, uint32_t nHeads, uint32_t headDim, float scale,
                     uint32_t windowSize, uint32_t compressRatio, uint32_t indexTopK,
                     const XTensor &topkIndices = XTensor(),
                     uint32_t tileSizeOfCachedKV = MAX_KV_TILE_SIZE);
void XliteOpMLAV2(XRuntime &rt, XTensor &qAbsorb, XTensor &qr, XTensor &kCache, XTensor &peCache,
                  XTensor &qk, XTensor &oAbsorb, XTensor &queryStartLoc, XTensor &lens,
                  XTensor &cachedLens, XTensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                  uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, uint32_t maxSeqLen,
                  float scale, uint32_t topK = 0, const XTensor &topkIndices = XTensor(),
                  bool dense = false);
void XliteOpFlashMLAV2(XRuntime &rt, XTensor &qAbsorb, XTensor &qr, XTensor &kCache,
                       XTensor &peCache, XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum,
                       XTensor &lastMax, XTensor &lastSum, XTensor &sync, XTensor &oAbsorb,
                       XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                       XTensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                       uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, float scale,
                       uint32_t tileSizeOfCachedKV = MAX_KV_TILE_SIZE, uint32_t topK = 0,
                       const XTensor &topkIndices = XTensor());
void XliteOpGatherSparseKVCache(XRuntime &rt, XTensor &kCache, const XTensor &peCache,
                                XTensor &blockTables, XTensor &topkIndices, XTensor &queryStartLoc,
                                XTensor &queryLens, XTensor &cachedLens, XTensor &kDenseCache,
                                const XTensor &peDenseCache, uint32_t batch, uint32_t indexTopK,
                                uint32_t blockSize, uint32_t kvLoraRank, uint32_t ropeHeadDim,
                                uint32_t kvHeads, uint32_t compressRatio = 1,
                                uint32_t burstCopySize = GM_UB_BURST_SIZE);
void XliteOpAddBias(XRuntime &rt, XTensor &input, XTensor &weight, XTensor &output);
void XliteOpAddAndRmsNorm(XRuntime &rt, XTensor &in, XTensor &addInOut, XTensor &norm,
                          float normEps, XTensor &out, const XTensor &normBias = XTensor());
void XliteOpSoftmaxTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outWeights,
                        XTensor &outRouting, uint32_t topK, bool normTopKProb);
void XliteOpSigmoidTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &bias, float scale,
                        XTensor &outWeights, XTensor &outRouting, uint32_t nGroup,
                        uint32_t nTopkGroup, uint32_t topK, bool normTopKProb);
void XliteOpSqrtsoftplusHashTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &bias,
                                 XTensor &inputIds, const XTensor &tid2eid, XTensor &outWeights,
                                 XTensor &routingMap, float scale, uint32_t topK, bool hash);
void XliteOpTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outIndices,
                 XTensor &queryLens, XTensor &cachedLens, uint32_t batch, size_t k);
void XliteOpSoftmax(XRuntime &rt, uint32_t calcLen, XTensor &x);
void XliteOpSoftmaxLong(XRuntime &rt, uint32_t calcLen, XTensor &x, XTensor &expBuf);
void XliteOpRopeComplex(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim, uint32_t outStepDim,
                        uint32_t ropeDim, uint32_t offset, uint32_t outOffset, XTensor &inputWithR,
                        XTensor &freqs, XTensor &position, XTensor &output, bool inverse = false,
                        bool outInterleaved = false, uint64_t minPosition = 0,
                        bool doRotate = false);
void XliteOpRopeComplexAndCache(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim,
                                uint32_t ropeDim, uint32_t offset, uint32_t vdim,
                                XTensor &inputWithR, XTensor &freqs, XTensor &position,
                                uint32_t blockSize, XTensor &vCache, XTensor &slotMapping,
                                bool outInterleaved = false, bool doRotate = false);
void XliteOpMlaPrepare(XRuntime &rt, XTensor &attnQkvc, const XTensor &qNorm,
                       const XTensor &qNormBias, XTensor &attnNormQc, const XTensor &kvNorm,
                       const XTensor &kvNormBias, const XTensor &freqs, const XTensor &position,
                       uint32_t qLoraRank, uint32_t kvLoraRank, uint32_t ropeHeadDim,
                       uint32_t blockSize, XTensor &kCache, XTensor &peCache,
                       const XTensor &slotMapping, float normEps,
                       const XTensor &attnNormKvc = XTensor());
void XliteOpQkRmsNorm(XRuntime &rt, XTensor &in, const XTensor &qNorm, const XTensor &qNormBias,
                      const XTensor &kNorm, const XTensor &kNormBias, XTensor &out, float normEps,
                      uint32_t qNormDim, uint32_t qCntPerToken, uint32_t kNormDim,
                      uint32_t kCntPerToken, uint32_t kStartOffset, bool useNorm,
                      const XTensor &qVariance = XTensor(), const XTensor &kVariance = XTensor());
void XliteOpIndexerPrepare(XRuntime &rt, XTensor &kw, const XTensor &kNorm,
                           const XTensor &kNormBias, const XTensor &freqs, const XTensor &position,
                           uint32_t indexHeadDim, uint32_t indexNHeads, uint32_t ropeHeadDim,
                           uint32_t blockSize, XTensor &indexKCache, const XTensor &slotMapping,
                           float normEps, const XTensor &q = XTensor(), float scale = 1.0f,
                           uint32_t topK = 2048, bool isLong = false, uint32_t tpSize = 1,
                           const XTensor &kScaleCache = XTensor(), const XTensor &q8 = XTensor(),
                           const XTensor &qScale = XTensor(),
                           const XTensor &scaledWeights = XTensor());

// LI-C8 writer only. Inputs are contiguous; K may have fused head-weight columns
// after its first 128 elements. Valid slots must be unique, or -1 for padding.
// Native callers validate indices and order producers/consumers on rt.stream.
void XliteOpIndexerKCacheC8(XRuntime &rt, const XTensor &k, const XTensor &kNorm,
                            const XTensor &kNormBias, const XTensor &freqs, const XTensor &position,
                            const XTensor &slotMapping, XTensor &kCache, const XTensor &scaleCache,
                            float normEps = 1e-6f);
void XliteOpCompressor(XRuntime &rt, XTensor &kv, const XTensor &score, const XTensor &ape,
                       const XTensor &norm, const XTensor &freqs, XTensor &weightedSum,
                       const XTensor &queryStartLoc, const XTensor &queryLens,
                       const XTensor &cachedLens, uint32_t batch, uint32_t nTotalBlocks,
                       uint32_t ratio, uint32_t overlap, uint32_t headDim, uint32_t ropeHeadDim,
                       float normEps, XTensor &compressKv, const XTensor &compressPositions,
                       const XTensor &compressSlotMapping, uint32_t compressBlockSize,
                       XTensor &state, const XTensor &stateSlotMapping,
                       const XTensor &stateBlockTable, uint32_t stateBlockSize,
                       uint64_t stateCacheStrideDim0, bool doRotate = false);

void XliteOpQuant(XRuntime &rt, XTensor &x, XTensor &scale_reciprocal, XTensor &offset,
                  XTensor &out);
void XliteOpQuantDyn(XRuntime &rt, XTensor &x, XTensor &scale, XTensor &out,
                     const XTensor &num = XTensor());
void XliteOpDeQuant(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &scale = XTensor(),
                    const XTensor &num = XTensor());
void XliteOpMSDMergeDequant(XRuntime &rt, XTensor &yMerged, XTensor &scaleBiasPtrs, XTensor &counts,
                            uint32_t start, uint32_t end, XTensor &perTokenScale, XTensor &out);
void XliteOpMatmulDeQuant(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out,
                          const XTensor &quantBias = XTensor(),
                          const XTensor &weightScale = XTensor(), bool weightNZ = false,
                          bool transpose = false, const XTensor &outScale = XTensor(),
                          const XTensor &num = XTensor());
void XliteOpFusionOperatorMatmulDequantPipeline(
    XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, const XTensor &quantBias = XTensor(),
    const XTensor &weightScale = XTensor(), bool weightNZ = false, bool transpose = false,
    const XTensor &outScale = XTensor(), const XTensor &num = XTensor(),
    uint64_t m0 = MATMUL_M0_N0_K0_DEFAULT_VALUE, uint64_t n0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
    uint64_t k0 = MATMUL_M0_N0_K0_DEFAULT_VALUE);
// Fused bf16 quant (static per-channel or dynamic per-token) -> int8 matmul
// (+fixpipe deqScale) -> dequant.  xQuant ([m,k] INT8) is the quant output
// workspace the AIVs fill and the AICs consume; outScale ([m] FP32) carries
// the per-token scales in dynQuant mode (workspace, written by quant and read
// back by dequant inside the kernel).
void XliteOpFusedQuantMatmulDequant(XRuntime &rt, XTensor &x, const XTensor &quantScaleRec,
                                    const XTensor &quantOffset, XTensor &weight, XTensor &out,
                                    const XTensor &quantBias, const XTensor &weightScale,
                                    XTensor &xQuant, XTensor &outScale, const XTensor &num,
                                    bool dynQuant, bool weightNZ = false, bool transpose = false,
                                    uint64_t m0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
                                    uint64_t n0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
                                    uint64_t k0 = MATMUL_M0_N0_K0_DEFAULT_VALUE);
void XliteOpGroupMatmulDeQuant(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &deqScales,
                               XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                               long outDim, long inDim, XTensor &output, XTensor &outScale,
                               XTensor &num, bool weightNZ = false, bool transpose = false);
void XliteOpConcat(XRuntime &rt, const std::vector<XTensor> &inputs, XTensor &out);
void XliteOpConcatCol(XRuntime &rt, const std::vector<XTensor> &inputs, XTensor &out);
void XliteOpSplitCol(XRuntime &rt, XTensor &in, const std::vector<XTensor> &outputs);
void XliteOpRepeatInterleave(XRuntime &rt, XTensor &in, XTensor &out, uint32_t numTokens,
                             uint32_t nKHeads, uint32_t nVHeads, uint32_t headBytes);
void XliteOpSplit(XRuntime &rt, XTensor &in, const std::vector<XTensor> &outputs,
                  const std::vector<size_t> &sizes, uint32_t numPackets);
void XliteOpSigmoidGateMul(XRuntime &rt, XTensor &attn, XTensor &gate, XTensor &out);
void XliteOpIndexerScores(XRuntime &rt, XTensor &q, XTensor &kCache, XTensor &weight,
                          XTensor &scores, XTensor &queryStartLoc, XTensor &lens,
                          XTensor &cachedLens, XTensor &blockTables, uint32_t nHeads,
                          uint32_t headDim, uint32_t blockSize, uint32_t batch);
void XliteOpIndexerTopK(XRuntime &rt, XTensor &q, XTensor &kCache, XTensor &weight, XTensor &scores,
                        XTensor &lastTopk, XTensor &seqPositions, XTensor &topkIndices,
                        XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                        XTensor &blockTables, XTensor &sync, uint32_t nHeads, uint32_t headDim,
                        uint32_t blockSize, uint32_t batch, uint32_t topK,
                        uint8_t skipDenseTopk = 0, const XTensor &kScaleCache = XTensor(),
                        uint32_t compressRatio = 1);
void XliteOpMuls(XRuntime &rt, XTensor &input, float scale, XTensor &output,
                 uint32_t calcOffset = 0, uint32_t calcNum = UINT32_MAX);
void XliteOpExpertsCountsSum(XRuntime &rt, XTensor &expertsCountsInput, XTensor &tokensPerEpgroup,
                             XTensor &expertsCountsOutput, uint32_t nRoutedExperts);
void XliteOpReorderMoE(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &counts,
                       uint32_t hiddenSize, uint32_t localStart, uint32_t localEnd, bool forward);
void XliteOpTranspose_1_2(XRuntime &rt, XTensor &input, XTensor &output);
void XliteOpConv1dAndSiLUPaged(XRuntime &rt, XTensor &state, XTensor &input, XTensor &weight,
                               XTensor &output, XTensor &stateIndices,
                               XTensor *cachedLens = nullptr, uint32_t stateBlockSize = 0,
                               bool aligned = false);
void XliteOpBetaDecay(XRuntime &rt, XTensor &b, XTensor &a, XTensor &A_log, XTensor &dt_bias,
                      XTensor &beta, XTensor &g, uint32_t bsz, uint32_t seqlen,
                      uint32_t num_v_heads);
// Single-token decode on native [P, Hv, V, K] pages; Q/K retain their Hk heads.
void XliteOpRecurrentGatedDeltaRulePaged(XRuntime &rt, XTensor &query, XTensor &key, XTensor &value,
                                         XTensor &beta, XTensor &g, XTensor &state, XTensor &out,
                                         XTensor *stateIndices, uint32_t batch, uint32_t kHeads,
                                         uint32_t vHeads, uint32_t kDim, uint32_t vDim,
                                         XTensor *cachedLens = nullptr, uint32_t stateBlockSize = 0,
                                         bool aligned = false);
void XliteOpEinsumMhtHdtMhd(XRuntime &rt, XTensor &mht, XTensor &hdt, XTensor &mhd, uint32_t m,
                            uint32_t h, uint32_t t, uint32_t d, bool weightNZ, int T = -1,
                            int D = -1);
void XliteOpEinsumMhtHtdMhd(XRuntime &rt, XTensor &mht, XTensor &htd, XTensor &mhd, uint32_t m,
                            uint32_t h, uint32_t t, uint32_t d, bool weightNZ, int T = -1,
                            int D = -1);
void XliteOpUnpackActivation(XRuntime &rt, XTensor &input, XTensor &output);
void XliteOpHcAct(XRuntime &rt, XTensor &mixes, const XTensor &hcScale, const XTensor &hcBase,
                  XTensor &post, XTensor &comb, uint32_t hcMult, float eps, uint32_t sinkhornIters,
                  bool headOnly, XTensor &xResid, XTensor &output);
void XliteOpHcPost(XRuntime &rt, XTensor &x, XTensor &post, XTensor &comb, XTensor &residual,
                   XTensor &y, uint32_t m, uint32_t hcMult, uint32_t hidden);
void XliteOpHcSplitSinkhorn(XRuntime &rt, XTensor &mixes, const XTensor &hcScale,
                            const XTensor &hcBase, XTensor &pre, XTensor &post, XTensor &comb,
                            uint32_t hcMult, float eps, uint32_t sinkhornIters, bool headOnly);
void XliteOpHcPre(XRuntime &rt, XTensor &xResid, const XTensor &pre, XTensor &output, uint32_t m,
                  uint32_t hcMult, uint32_t hidden);
// Benchmarks
uint32_t XliteOpBenchEmpty(XRuntime &rt, const std::string &variant, uint32_t blockNum = 0);
uint32_t XliteOpBenchScalar(XRuntime &rt, const std::string &variant, XTensor &out, double seed,
                            uint32_t iters, uint32_t blockNum = 0);
#ifdef MICROBENCH_TIMED
uint32_t XliteOpBenchTimed(XRuntime &rt, const std::string &variant, XTensor &out, XTensor &ticks,
                           double seed, uint32_t iters, uint32_t blockNum = 0);
uint32_t XliteOpBenchMte(XRuntime &rt, const std::string &variant, const XTensor &src, XTensor &dst,
                         XTensor &ticks, uint64_t bytes, uint32_t iters, uint32_t blockNum = 0);
uint32_t XliteOpBenchVec(XRuntime &rt, const std::string &variant, const XTensor &src, XTensor &out,
                         XTensor &ticks, uint64_t elems, uint32_t iters, uint32_t blockNum = 0);
uint32_t XliteOpBenchCube(XRuntime &rt, const std::string &variant, const XTensor &a,
                          const XTensor &b, XTensor &c, XTensor &ticks, uint64_t m0, uint64_t n0,
                          uint64_t k0, uint32_t blockNum = 0);
#endif  // MICROBENCH_TIMED

void XliteOpSiluGateMul(XRuntime &rt, XTensor &normed, XTensor &gate, XTensor &out);
void XliteOpConv1dAndSiLUPrefillPaged(XRuntime &rt, XTensor &state, XTensor &input, XTensor &weight,
                                      XTensor &output, XTensor &indices, XTensor &starts,
                                      XTensor &initial);
void XliteOpRecurrentGatedDeltaRulePrefillPaged(XRuntime &rt, XTensor &query, XTensor &key,
                                                XTensor &value, XTensor &beta, XTensor &g,
                                                XTensor &state, XTensor &out, XTensor &indices,
                                                XTensor &starts, XTensor &initial, uint32_t kHeads,
                                                uint32_t vHeads, uint32_t kDim, uint32_t vDim);
#endif
