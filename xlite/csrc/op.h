/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_OP_H_
#define _XLITE_OP_H_

#include "base.h"
#include "runtime.h"
#include "kernels/ccl_param.h"

#define MATMUL_M0_N0_K0_DEFAULT_VALUE ((uint64_t)(-1))
#define MATMUL_SWIZZLE_DEFAULT_VALUE (0x600)

void XliteOpAllGather(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                      uint32_t copySize = COPY_SIZE);
void XliteOpReduceScatter(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                          uint32_t copySize = COPY_SIZE);
void XliteOpAllReduceSum(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                         uint32_t copySize = COPY_SIZE);

void XliteOpEmbed(XRuntime &rt, XTensor &in, XTensor &embed, uint32_t start, uint32_t end,
                  XTensor &out);
void XliteOpRmsNorm(XRuntime &rt, XTensor &in, XTensor &norm, XTensor &out, float normEps,
                    uint32_t normDim, uint32_t cntPerToken = 1, uint32_t startOffset = 0);
void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out);
void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, bool weightNZ = false,
                   const XTensor &bias = XTensor(), bool transpose = false,
                   uint64_t m0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
                   uint64_t n0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
                   uint64_t k0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
                   uint64_t swizzle = MATMUL_SWIZZLE_DEFAULT_VALUE);

void XliteOpSiluAndMul(XRuntime &rt, XTensor &in, XTensor &out);
void XliteOpCastDown(XRuntime &rt, XTensor &in, XTensor &out, XTensor &outScale);
void XliteOpCastUp(XRuntime &rt, XTensor &in, XTensor &inScale, XTensor &out);
void XliteOpPermutation(XRuntime &rt, XTensor &in, XTensor &routing, uint32_t start, uint32_t end,
                        XTensor &out, XTensor &unpIdx, XTensor &counts);
void XliteOpUnpermutation(XRuntime &rt, XTensor &in, XTensor &unpIdx, XTensor &routing,
                          XTensor &weights, uint32_t start, uint32_t end, XTensor &out);
void XliteOpGroupMatmul(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &scales,
                        XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                        long outDim, long inDim, XTensor &output, bool weightNZ = false,
                        bool transpose = false);
void XliteOpRopeCache(XRuntime &rt, XTensor &inout, XTensor &kCache, XTensor &vCache,
                      XTensor &position, XTensor &cossin, XTensor &slotMapping, uint32_t nHeads,
                      uint32_t nKvHeads, uint32_t headDim, uint32_t rotDim, uint32_t blockSize,
                      bool isNeox, uint64_t mropeMaskH, uint64_t mropeMaskW);
void XliteOpAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache, XTensor &qk,
                      XTensor &output, XTensor &cumPromptLens, XTensor &lens, XTensor &cachedLens,
                      XTensor &blockTables, uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim,
                      uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock);
void XliteDsOpStridedRmsnorm(XRuntime &rt, XTensor &input, XTensor &w, XTensor &output,
                             uint32_t numTokens, uint32_t normDim, uint32_t stepDim, float normEps);
void XliteDsOpReshapeAndCache(XRuntime &rt, XTensor &key, XTensor &value, XTensor &kCache,
                              XTensor &vCache, XTensor &slotMapping, int32_t numTokens,
                              int32_t keyStride, int32_t valueStride, int32_t numKvHeads,
                              int32_t kHeadSize, int32_t vHeadSize, int32_t blockSize,
                              int32_t blockNum);
void XliteDsOpKvMatmul(XRuntime &rt, XTensor &input, XTensor &w, XTensor &output, int m, int n,
                       int k, XTensor &blockTable, bool nt, int blockSize, int headSize);
void XliteDsOpPrefillKvSplit(XRuntime &rt, XTensor &kv, XTensor &kPe, XTensor &cache,
                             XTensor &blockTable, XTensor &kvFull, XTensor &v, int nTokens,
                             int nTokensPad, int nLocalHeads, int kvLoraRank, int rotDim,
                             int headSize, int vDim, uint32_t blockSize);
void XliteDsOpPrefillMix(XRuntime &rt, XTensor &out, XTensor &alpha, XTensor &max, XTensor &sum,
                         XTensor &q, XTensor &k, XTensor &qk, XTensor &blockTables,
                         XTensor &cachedLens, XTensor &v, XTensor &mixOut, XTensor &mixOutFinal,
                         XTensor &promptLens, XTensor &attnMask, XTensor &attnMaskAddr,
                         XTensor &speculateLens, XTensor &prefillIndex, XTensor &cumPromptLens,
                         uint32_t headSize, uint32_t numHeads, uint32_t numKVHeads,
                         uint32_t blockSize, uint32_t batchSize, uint32_t mappingLen,
                         uint32_t doTreeAttnMask, uint32_t offsetM, uint32_t mSlice, float scale);
void XliteDsOpEinsumShdHdcShc(XRuntime &rt, int numTokens, int headSize, int nLocalHeads,
                              int qStepDim, int kvUpWeightStepDim, int kvLoraRank, XTensor &qWithQr,
                              XTensor &kvUpWeight, XTensor &qAbsorb);
void XliteDsOpDecodeAttn(XRuntime &rt, XTensor &q, XTensor &k, XTensor &o, XTensor &cachedLens,
                         XTensor &mapping, XTensor &promptLens, XTensor &promptLensCum,
                         uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                         uint32_t headSize, uint32_t blockSize, uint32_t mappingLen,
                         uint32_t maxContextLen, bool add);
void XliteDsOpSoftmax(XRuntime &rt, XTensor &qk, XTensor &cachedLens, XTensor &promptLens,
                      XTensor &promptLensCum, float scale, uint32_t numTokens, uint32_t numHeads,
                      uint32_t blockSize, uint32_t maxContextLen);
void XliteDsOpEinsumShtTcShc(XRuntime &rt, int numTokens, int nLocalHeads, int maxTokens,
                             int maxBlocksPerQuery, int numBlocks, int blockSize, int kvLoraRank,
                             XTensor &scores, XTensor &cachedLens, XTensor &promptLens,
                             XTensor &promptLensCum, XTensor &blockTables, XTensor &cCache,
                             XTensor &result);
void XliteDsOpEinsumShcHdcShd(XRuntime &rt, int numTokens, int nLocalHeads, int kvLoraRank,
                              int wkvbStep, int vDim, XTensor &scores, XTensor &kvUpWeight,
                              XTensor &result);
void XliteOpAddBias(XRuntime &rt, XTensor &input, XTensor &weight, XTensor &output);
void XliteOpAddAndRmsNorm(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &norm, float normEps,
                          XTensor &out);

void XliteOpSoftmaxTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outWeights,
                        XTensor &outRouting, uint32_t topK, bool normTopKProb);
void XliteOpSigmoidTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &bias, float scale,
                        XTensor &outWeights, XTensor &outRouting, uint32_t nGroup,
                        uint32_t nTopkGroup, uint32_t topK, bool normTopKProb);
void XliteOpSoftmax(XRuntime &rt, uint32_t calcLen, XTensor &x);
void XliteOpSoftmaxLong(XRuntime &rt, uint32_t calcLen, XTensor &x, XTensor &expBuf);
void XliteOpRopeComplex(XRuntime &rt, uint32_t numTokens, uint32_t nLocalHeads, uint32_t stepDim,
                        uint32_t ropeDim, XTensor &inputWithR, XTensor &freqs, XTensor &position,
                        XTensor &vGather, XTensor &outputPe, enum XRopeType ropeType);
#endif
