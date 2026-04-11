/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_OP_H_
#define _XLITE_OP_H_

#include "base.h"
#include "runtime.h"
#include "kernels/kernel_param.h"

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
                    uint32_t normDim, uint32_t cntPerToken = 1, uint32_t inStartOffset = 0,
                    uint32_t outStartOffset = 0);
void XliteOpLayerNorm(XRuntime &rt, XTensor &in, XTensor &norm, XTensor &normBias, XTensor &out,
                      float normEps, uint32_t normDim, uint32_t cntPerToken = 1,
                      uint32_t inStartOffset = 0, uint32_t outStartOffset = 0);
void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out);
void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, bool weightNZ = false,
                   const XTensor &bias = XTensor(), const XTensor &deqScale = XTensor(),
                   bool transpose = false, uint64_t m0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
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
void XliteOpGroupMatmul(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &deqScales,
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
void XliteOpFlashAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache,
                           XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum, XTensor &lastMax,
                           XTensor &lastSum, XTensor &sync, XTensor &output, XTensor &cumPromptLens,
                           XTensor &lens, XTensor &cachedLens, XTensor &blockTables,
                           uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim, uint32_t blockSize,
                           uint32_t batch, uint32_t maxNumBlock);
void XliteOpFlashMLA(XRuntime &rt, XTensor &qWithQr, XTensor &kCache, XTensor &vCache,
                     XTensor &wkvb, XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum,
                     XTensor &lastMax, XTensor &lastSum, XTensor &sync, XTensor &output,
                     XTensor &cumPromptLens, XTensor &lens, XTensor &cachedLens,
                     XTensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                     uint32_t nopeHeadDim, uint32_t vHeadDim, uint32_t kvLoraRank,
                     uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock, float scale);
void XliteOpAddBias(XRuntime &rt, XTensor &input, XTensor &weight, XTensor &output);
void XliteOpAddAndRmsNorm(XRuntime &rt, XTensor &in, XTensor &addInOut, XTensor &norm,
                          float normEps, XTensor &out);

void XliteOpSoftmaxTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outWeights,
                        XTensor &outRouting, uint32_t topK, bool normTopKProb);
void XliteOpSigmoidTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &bias, float scale,
                        XTensor &outWeights, XTensor &outRouting, uint32_t nGroup,
                        uint32_t nTopkGroup, uint32_t topK, bool normTopKProb);
void XliteOpSoftmax(XRuntime &rt, uint32_t calcLen, XTensor &x);
void XliteOpSoftmaxLong(XRuntime &rt, uint32_t calcLen, XTensor &x, XTensor &expBuf);
void XliteOpRopeComplex(XRuntime &rt, uint32_t numTokens, uint32_t nLocalHeads, uint32_t stepDim,
                        uint32_t ropeDim, uint32_t offset, XTensor &inputWithR, XTensor &freqs,
                        XTensor &position, XTensor &vGather);
void XliteOpRopeComplexAndCache(XRuntime &rt, uint32_t numTokens, uint32_t nLocalHeads,
                                uint32_t stepDim, uint32_t ropeDim, uint32_t offset, uint32_t vdim,
                                XTensor &inputWithR, XTensor &freqs, XTensor &position,
                                XTensor &vGather, uint32_t blockSize, XTensor &key, XTensor &kCache,
                                XTensor &vCache, XTensor &slotMapping);

void XliteOpQuant(XRuntime &rt, XTensor &x, XTensor &scale_reciprocal, XTensor &offset,
                  XTensor &out);

void XliteOpQuantDyn(XRuntime &rt, XTensor &x, XTensor &scale, XTensor &out);

void XliteOpDeQuant(XRuntime &rt, XTensor &in, XTensor &scale, XTensor &out, bool hasScale);

void XliteOpConcat3(XRuntime &rt, XTensor &in0, XTensor &in1, XTensor &in2, XTensor &out);
void XliteOpSplit3(XRuntime &rt, XTensor &in, XTensor &out0, XTensor &out1, XTensor &out2,
                   size_t size0, size_t size1, size_t size2, uint32_t numPackets);
void XliteOpIndexerScores(XRuntime &rt, XTensor &q, XTensor &kCache, XTensor &weight,
                          XTensor &scores, XTensor &cumPromptLens, XTensor &lens,
                          XTensor &cachedLens, XTensor &blockTables, uint32_t nHeads,
                          uint32_t headDim, uint32_t blockSize, uint32_t batch,
                          uint32_t maxNumBlock);
void XliteOpMuls(XRuntime &rt, XTensor &input, float scale, XTensor &output);
#endif
