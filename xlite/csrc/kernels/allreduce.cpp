/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"
#include "kernel_macro.h"
#include "ccl_param.h"

using namespace AscendC;

#define PINGPONG_BUF_NUM 2

template<typename Dtype>
class AllReduce {
public:
    __aicore__ inline AllReduce() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint64_t count,
        uint32_t rankId, uint32_t rankSize, uint64_t generation, GM_ADDR param)
    {
        __gm__ struct XcclParam *xcclParam = (__gm__ struct XcclParam *)param;
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

        for (uint32_t r = 0; r < rankSize; r++) {
            this->param.ipcMems[r] = xcclParam->ipcMems[r];
            this->param.ipcXTensorMems[r] = xcclParam->ipcXTensorMems[r];
        }
        this->count = count;
        this->myRankId = rankId;
        this->rankSize = rankSize;
        this->coreIdx = block_idx;
        this->coreNum = block_num;
        uint64_t countPerRank = DIV_ROUND_UP(count, rankSize);
        this->countPerBlock = DIV_ROUND_UP(countPerRank, rankSize - 1);
        this->offsetCurrRank = countPerRank * myRankId;
        this->generation = generation;

        uint64_t off = 0;
        flagBuf.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
        flagBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += UB_BUF_ALIGN_SIZE;
        uint64_t ubBufBytes = COPY_SIZE;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            ubBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
            ubBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += ubBufBytes;
        }

        for (uint32_t r = 0; r < rankSize; r++) {
            this->ipcFlagBuf[r].SetGlobalBuffer((__gm__ uint32_t *)(this->param.ipcMems[r] + XLITE_IPC_MEM_FLAG_OFFSET));
        }

        if (coreIdx == 0) {
            __gm__ uint64_t *inputOffset = (__gm__ uint64_t *)(this->param.ipcMems[myRankId]);
            __gm__ uint64_t *outputOffset = (__gm__ uint64_t *)(this->param.ipcMems[myRankId] + sizeof(uint64_t));
            *inputOffset = (uint64_t)input - (uint64_t)this->param.ipcXTensorMems[myRankId];
            *outputOffset = (uint64_t)output - (uint64_t)this->param.ipcXTensorMems[myRankId];
            DataCacheCleanAndInvalid(inputOffset);
            DataCacheCleanAndInvalid(outputOffset);
            SetIpcFlag(0, generation);
            for (uint32_t r = 0; r < rankSize; r++) {
                if (r == myRankId) {
                    continue;
                }
                WaitIpcFlag(r, 0, generation);
            }
        }

        CrossCoreSetFlag<0x0, PIPE_MTE3>(1);
        CrossCoreWaitFlag(1);

        uint32_t idx = 0;
        for (uint32_t r = 0; r < rankSize; r++) {
            if (r == myRankId) {
                inputBuf[r].SetGlobalBuffer((__gm__ Dtype *)input);
                outputBuf[r].SetGlobalBuffer((__gm__ Dtype *)output);
                continue;
            }
            __gm__ uint64_t *inputOffset = (__gm__ uint64_t *)(this->param.ipcMems[r]);
            __gm__ uint64_t *outputOffset = (__gm__ uint64_t *)(this->param.ipcMems[r] + sizeof(uint64_t));
            struct XcclIpcMemData localIpcMemData;
            DataCacheCleanAndInvalid(inputOffset);
            DataCacheCleanAndInvalid(outputOffset);
            localIpcMemData.inputOffset = *inputOffset;
            localIpcMemData.outputOffset = *outputOffset;
            inputBuf[r].SetGlobalBuffer((__gm__ Dtype *)(this->param.ipcXTensorMems[r] + localIpcMemData.inputOffset));
            outputBuf[r].SetGlobalBuffer((__gm__ Dtype *)(this->param.ipcXTensorMems[r] + localIpcMemData.outputOffset));
            rankIdxMapping[idx++] = r;
        }

        // each rank process countPerRank elements
        countCurrRank = countPerRank;
        if (myRankId * countPerRank + countCurrRank > count) {
            countCurrRank = count - myRankId * countPerRank;
        }
    }

    __aicore__ inline void SetIpcFlag(uint32_t flagId, uint32_t value)
    {
        PipeBarrier<PIPE_ALL>();
        flagBuf.SetValue(0, value);
        PipeBarrier<PIPE_ALL>();
        DataCopyParams copyParams;
        copyParams.blockLen = sizeof(uint32_t);
        copyParams.blockCount = 1;
        DataCopyPad(ipcFlagBuf[myRankId][flagId], flagBuf, copyParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WaitIpcFlag(uint32_t rankId, uint32_t flagId, uint32_t expectValue)
    {
        uint32_t flagValue = 0;
        PipeBarrier<PIPE_ALL>();
        do {
            DataCopyParams copyParams;
            copyParams.blockLen = sizeof(uint32_t);
            copyParams.blockCount = 1;
            DataCopyPadParams padParams;
            padParams.isPad = false;
            DataCopyPad(flagBuf, ipcFlagBuf[rankId][flagId], copyParams, padParams);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
            flagValue = flagBuf.GetValue(0);
        } while (flagValue < expectValue);
        PipeBarrier<PIPE_ALL>();
    }

    // split work among cores
    __aicore__ inline void WorkSplit(uint32_t workNum, uint32_t *start, uint32_t *end)
    {
        uint32_t remain = workNum % coreNum;
        uint32_t avg = workNum / coreNum;
        if (coreIdx < remain) {
            *start = coreIdx * avg + coreIdx;
            *end = *start + avg + 1;
        } else {
            *start = coreIdx * avg + remain;
            *end = *start + avg;
        }
        if (*end > workNum) {
            *end = workNum;
        }
    }

    __aicore__ inline void CopyGMtoUbuf(LocalTensor<Dtype> dst, GlobalTensor<Dtype> src, uint64_t count)
    {
        if (count * sizeof(Dtype) % BLOCK_SIZE == 0) {
            DataCopy(dst, src, count);
        } else {
            DataCopyParams copyParams;
            copyParams.blockLen = count * sizeof(Dtype);
            copyParams.blockCount = 1;
            DataCopyPadParams padParams;
            padParams.isPad = false;
            DataCopyPad(dst, src, copyParams, padParams);
        }
    }

    __aicore__ inline void CopyUbufToGM(GlobalTensor<Dtype> dst, LocalTensor<Dtype> src, uint64_t count)
    {
        if (count * sizeof(Dtype) % BLOCK_SIZE == 0) {
            DataCopy(dst, src, count);
        } else {
            DataCopyParams copyParams;
            copyParams.blockLen = count * sizeof(Dtype);
            copyParams.blockCount = 1;
            DataCopyPad(dst, src, copyParams);
        }
    }

    __aicore__ inline void Run()
    {
        uint32_t corePerBlock = coreNum / (rankSize - 1);
        int curr = 0;
        // reduce-scatter phase
        for  (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + i);
        }
        uint32_t workStart = 0;
        uint32_t workEnd = 0;
        uint32_t workNum = coreNum <= rankSize - 1 ? rankSize - 1 : ROUND_DOWN(coreNum, rankSize - 1);
        WorkSplit(workNum, &workStart, &workEnd);
        uint32_t countPerWork = coreNum <= rankSize - 1 ? countPerBlock : DIV_ROUND_UP(countCurrRank, workNum);
        for (uint32_t r = 0; r < rankSize; r++) {
            if (r == 1) {
                SetAtomicAdd<Dtype>();
                PipeBarrier<PIPE_ALL>();
            }
            for (uint32_t workIdx = workStart; workIdx < workEnd; workIdx++) {
                uint32_t blockIdx = coreNum <= rankSize - 1 ? workIdx : workIdx / corePerBlock;
                uint32_t processRankIdx = r == 0 ? myRankId : rankIdxMapping[(blockIdx + (r - 1)) % (rankSize - 1)];
                uint64_t workOffset = workIdx * countPerWork;
                uint64_t workCount = countPerWork;
                if (workOffset + workCount > countCurrRank) {
                    workCount = countCurrRank - workOffset;
                }
                uint32_t copyCount = COPY_SIZE / sizeof(Dtype);
                uint32_t copyNum = DIV_ROUND_UP(workCount, copyCount);
                for (uint32_t copyIdx = 0; copyIdx < copyNum; copyIdx++) {
                    uint64_t copyOffset = copyIdx * copyCount;
                    uint64_t currCopyCount = copyCount;
                    if (copyOffset + currCopyCount > workCount) {
                        currCopyCount = workCount - copyOffset;
                    }
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                    CopyGMtoUbuf(ubBuf[curr], inputBuf[processRankIdx][offsetCurrRank + workOffset + copyOffset], currCopyCount);
                    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                    CopyUbufToGM(outputBuf[myRankId][offsetCurrRank + workOffset + copyOffset], ubBuf[curr], currCopyCount);
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                    curr = (curr + 1) % PINGPONG_BUF_NUM;
                }
            }
        }

        // clear atomic
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();

        // allgather phase
        curr = 0;
        for (uint32_t r = 0; r < rankSize - 1; r++) {
            for (uint32_t workIdx = workStart; workIdx < workEnd; workIdx++) {
                uint32_t blockIdx = coreNum <= rankSize - 1 ? workIdx : workIdx / corePerBlock;
                uint32_t processRankIdx = rankIdxMapping[(blockIdx + r) % (rankSize - 1)];
                uint64_t workOffset = workIdx * countPerWork;
                uint64_t workCount = countPerWork;
                if (workOffset + workCount > countCurrRank) {
                    workCount = countCurrRank - workOffset;
                }
                uint32_t copyCount = COPY_SIZE / sizeof(Dtype);
                uint32_t copyNum = DIV_ROUND_UP(workCount, copyCount);
                for (uint32_t copyIdx = 0; copyIdx < copyNum; copyIdx++) {
                    uint64_t copyOffset = copyIdx * copyCount;
                    uint64_t currCopyCount = copyCount;
                    if (copyOffset + currCopyCount > workCount) {
                        currCopyCount = workCount - copyOffset;
                    }
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                    CopyGMtoUbuf(ubBuf[curr], outputBuf[myRankId][offsetCurrRank + workOffset + copyOffset], currCopyCount);
                    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + curr);
                    CopyUbufToGM(outputBuf[processRankIdx][offsetCurrRank + workOffset + copyOffset], ubBuf[curr], currCopyCount);
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + curr);
                    curr = (curr + 1) % PINGPONG_BUF_NUM;
                }
            }
        }
        for  (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + i);
        }

        // inter-rank sync
        CrossCoreSetFlag<0x0, PIPE_MTE3>(1);
        CrossCoreWaitFlag(1);

        // outer-rank sync
        if (coreIdx == 0) {
            SetIpcFlag(1, generation);
            for (uint32_t r = 0; r < rankSize; r++) {
                if (r == myRankId) {
                    continue;
                }
                WaitIpcFlag(r, 1, generation);
            }
        }

        PipeBarrier<PIPE_ALL>();
    }

private:
    GlobalTensor<Dtype> inputBuf[XLITE_CCL_MAX_RANK_SIZE];
    GlobalTensor<Dtype> outputBuf[XLITE_CCL_MAX_RANK_SIZE];
    GlobalTensor<uint32_t> ipcFlagBuf[XLITE_CCL_MAX_RANK_SIZE];
    LocalTensor<uint32_t> flagBuf;
    LocalTensor<Dtype> ubBuf[PINGPONG_BUF_NUM];
    uint64_t count;
    uint64_t offsetCurrRank;
    uint64_t countCurrRank;
    uint64_t countPerBlock;
    uint64_t generation;
    uint32_t myRankId;
    uint32_t rankSize;
    uint32_t coreIdx;
    uint32_t coreNum;
    uint32_t rankIdxMapping[XLITE_CCL_MAX_RANK_SIZE];
    struct XcclParam param;
};

#define ALLREDUCE_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void allreduce_##dtype(GM_ADDR input, GM_ADDR output, uint64_t count, \
    uint32_t rankId, uint32_t rankSize, uint64_t generation, GM_ADDR param) \
{ \
    AllReduce<dtype> op; \
    op.Init(input, output, count, rankId, rankSize, generation, param); \
    op.Run(); \
}

ALLREDUCE_FUNC_DEFINE(int8_t);
ALLREDUCE_FUNC_DEFINE(int16_t);
ALLREDUCE_FUNC_DEFINE(int32_t);
ALLREDUCE_FUNC_DEFINE(float16_t);
ALLREDUCE_FUNC_DEFINE(bfloat16_t);
ALLREDUCE_FUNC_DEFINE(float);