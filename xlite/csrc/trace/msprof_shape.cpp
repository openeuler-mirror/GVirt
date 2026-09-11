/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */

#include "msprof_shape.h"

#ifdef XLITE_MSPROF_RECODE_SHAPE

#include <cstring>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

namespace msprofshape
{
namespace
{

// One staged launch: tensors to report once the stub hands us its timestamp.
struct StagedLaunch {
    std::string kernelName;
    std::vector<TensorDesc> tensors;  // inputs first, then outputs
    std::vector<uint32_t> types;      // kTensorTypeInput / kTensorTypeOutput
};

// The stub runs on the staging thread, so a thread_local slot carries the
// tensors from StageLaunchShapes into XLiteMsprofShapeHook with no locks.
thread_local StagedLaunch g_staged = {};

uint32_t CurrentThreadId()
{
    return static_cast<uint32_t>(syscall(SYS_gettid));
}

void FillTensorData(MsrofTensorData &t, uint32_t tensorType, const TensorDesc &desc)
{
    t.tensorType = tensorType;
    t.format = kFormatND;
    t.dataType = XDtypeToAcl(desc.dtype);
    // Keep the innermost kShapeLen dims, as GE does.
    const size_t skip = desc.shape.size() > kShapeLen ? desc.shape.size() - kShapeLen : 0;
    uint32_t n = 0;
    for (size_t i = skip; i < desc.shape.size() && n < kShapeLen; ++i) {
        t.shape[n++] = static_cast<uint32_t>(desc.shape[i]);
    }
}

// Report one tensor; returns false if msprof rejects the record.
bool ReportTensor(uint64_t opName, uint32_t threadId, uint64_t timeStamp, uint32_t tensorType,
                  const TensorDesc &desc)
{
    MsprofTensorInfo info = {};
    info.opName = opName;
    info.tensorNum = 1;
    FillTensorData(info.tensorData[0], tensorType, desc);

    MsprofAdditionalInfo addition = {};
    addition.magicNumber = kMagicNumber;
    addition.level = kNodeLevel;
    addition.type = kTensorInfoType;
    addition.threadId = threadId;
    addition.dataLen = sizeof(info);
    addition.timeStamp = timeStamp;
    memcpy(addition.data, &info, sizeof(info));
    return MsprofReportAdditionalInfo(1, &addition, sizeof(addition)) == 0;
}

}  // namespace

void StageLaunchShapes(const char *kernelName, const TensorDesc *inputs, uint32_t numInputs,
                       const TensorDesc *outputs, uint32_t numOutputs)
{
    if (kernelName == nullptr) {
        return;
    }
    StagedLaunch &staged = g_staged;
    staged.kernelName = kernelName;
    staged.tensors.clear();
    staged.types.clear();
    for (uint32_t i = 0; i < numInputs && i < kTensorDataNum; ++i) {
        staged.tensors.push_back(inputs[i]);
        staged.types.push_back(kTensorTypeInput);
    }
    for (uint32_t i = 0; i < numOutputs && i < kTensorDataNum; ++i) {
        staged.tensors.push_back(outputs[i]);
        staged.types.push_back(kTensorTypeOutput);
    }
}

}  // namespace msprofshape

// The launch-stub hook injected by scripts/patch_host_stub.py; fires right
// after StartAscendProf captured startTime (the stamp the stub's node record
// carries). Defined outside the anonymous namespace because the patched stub
// calls it as a C symbol; with the macro off it is never defined and the
// stub's weak reference stays null.
extern "C" void XLiteMsprofShapeHook(const char *name, uint64_t startTime)
{
    if (name == nullptr) {
        return;
    }
    msprofshape::StagedLaunch &staged = msprofshape::g_staged;
    if (staged.kernelName != name || staged.tensors.empty()) {
        return;
    }
    const uint64_t opName = MsprofGetHashId(name, static_cast<uint32_t>(strlen(name)));
    if (opName == 0) {
        return;
    }
    const uint32_t threadId = msprofshape::CurrentThreadId();
    // ReportAscendProf stamps the node record at startTime + 1 tick, so the
    // tensor records must carry the same +1 to hit the exact merge key.
    const uint64_t nodeStamp = startTime + 1;
    for (size_t i = 0; i < staged.tensors.size(); ++i) {
        if (!msprofshape::ReportTensor(opName, threadId, nodeStamp, staged.types[i],
                                       staged.tensors[i])) {
            break;
        }
    }
    staged.tensors.clear();
    staged.types.clear();
    staged.kernelName.clear();
}

#endif  // XLITE_MSPROF_RECODE_SHAPE
