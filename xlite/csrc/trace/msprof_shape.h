/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "base.h"

// Msprof host-side shape record, gated by XLITE_MSPROF_RECODE_SHAPE (see
// CMakeLists.txt); without it this header and all call sites compile to
// nothing.
#ifdef XLITE_MSPROF_RECODE_SHAPE

extern "C" {
uint64_t MsprofSysCycleTime(void);
uint64_t MsprofGetHashId(const char *data, uint32_t len);
int32_t MsprofReportAdditionalInfo(uint32_t nonPersistantFlag, void *data, uint32_t len);
}

namespace msprofshape
{

// The aclrtlaunch_* stubs stamp their node record with a timestamp only they
// know (StartAscendProf's). The parser merges additional records into that
// node only when they carry the exact same (opName, timestamp, threadId);
// anything else becomes a second host-task descriptor and drops the task from
// op_summary. scripts/patch_host_stub.py injects a weak hook into the stubs
// that hands us the timestamp: stage tensors before the launch, and the hook
// reports them stamped to merge into the stub's single node record.

constexpr uint16_t kMagicNumber = 0x5a5a;
constexpr uint16_t kNodeLevel = 10000;      // MSPROF_REPORT_NODE_LEVEL
constexpr uint32_t kTensorInfoType = 1;     // MSPROF_REPORT_NODE_TENSOR_INFO_TYPE
constexpr uint32_t kTensorTypeInput = 0;    // MSPROF_GE_TENSOR_TYPE_INPUT
constexpr uint32_t kTensorTypeOutput = 1;   // MSPROF_GE_TENSOR_TYPE_OUTPUT
constexpr uint32_t kShapeLen = 8;           // MSPROF_GE_TENSOR_DATA_SHAPE_LEN
constexpr uint32_t kTensorDataNum = 5;      // MSPROF_GE_TENSOR_DATA_NUM
constexpr uint32_t kAdditionDataLen = 232;  // MSPROF_ADDTIONAL_INFO_DATA_LENGTH
constexpr uint32_t kFormatND = 2;           // ACL_FORMAT_ND

// One reported tensor (ptr kept for parity with msprof's record layout).
struct TensorDesc {
    const void *ptr;
    enum XDtype dtype;
    const std::vector<size_t> &shape;
};

#pragma pack(1)
struct MsrofTensorData {
    uint32_t tensorType;
    uint32_t format;
    uint32_t dataType;
    uint32_t shape[kShapeLen];
};

struct MsprofTensorInfo {
    uint64_t opName;
    uint32_t tensorNum;
    MsrofTensorData tensorData[kTensorDataNum];
};

struct MsprofAdditionalInfo {
    uint16_t magicNumber;
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    uint8_t data[kAdditionDataLen];
};
#pragma pack()

static_assert(sizeof(MsprofTensorInfo) == kAdditionDataLen);

// aclDataType values for the XDtype mapping.
enum : uint32_t {
    kAclBit1 = 30,  // ACL_UINT1, matches BIT1 packed-bit storage
    kAclInt4 = 29,
    kAclBf16 = 27,
};

inline uint32_t XDtypeToAcl(enum XDtype dtype)
{
    switch (dtype) {
        case BIT1:
            return kAclBit1;
        case INT4:
            return kAclInt4;
        case INT8:
            return 2;
        case INT32:
            return 3;
        case INT64:
            return 9;
        case FP16:
            return 1;
        case BF16:
            return kAclBf16;
        case FP32:
            return 0;
        case CPLXF:
            return 17;  // ACL_COMPLEX128 (2 x fp64, matches CPLXF storage)
        default:
            return 0xFFFFFFFF;
    }
}

// Stage the shapes of the next launch of `kernelName` on this thread. Must be
// called BEFORE the launch, from the same thread that launches; consumed by
// XLiteMsprofShapeHook inside the launch stub (discarded at the next staging
// if the hook never fires, e.g. profiling off). kernelName must be the exact
// string the stub passes to StartAscendProf (e.g. "matmul_float16_t").
void StageLaunchShapes(const char *kernelName, const TensorDesc *inputs, uint32_t numInputs,
                       const TensorDesc *outputs, uint32_t numOutputs);

}  // namespace msprofshape

#endif  // XLITE_MSPROF_RECODE_SHAPE
