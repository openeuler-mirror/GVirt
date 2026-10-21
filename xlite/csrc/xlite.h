/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_H_
#define _XLITE_H_

// Umbrella header exposing the public C++ API.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "base.h"
#include "runtime.h"
#include "model.h"

inline enum XDtype XDtypeOf(const torch::Tensor &t)
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
                                     ": unknown torch dtype for xlite: " +
                                     std::to_string(static_cast<int>(t.scalar_type())));
    }
}

inline void *TensorPtr(const torch::Tensor &t)
{
    return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(t.storage().data_ptr().get()) +
                                    t.storage_offset() * t.dtype().itemsize());
}

// Wrap a torch tensor as an XTensor. Torch-contiguous tensors wrap as contiguous directly:
// is_contiguous() skips size-1 dims, but such strides never affect element addressing, so
// the memory layout is exactly row-major; zero-element tensors carry no data and wrap the
// same way. Any other tensor needs allowPitched, which forwards torch's strides to XTensor
// for validation (dim-0 row pitch only): kv caches (MHA/GQA/MLA/DSA/CXA) and debug output
// (Print/Save). Without allowPitched, non-contiguous tensors are rejected at wrap time.
inline void InitXTensor(XTensor &out, const torch::Tensor &in, bool allowPitched = false)
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

    if (in.numel() == 0 || in.is_contiguous()) {
        out.Init(sizes, XDtypeOf(in), TensorPtr(in));
        return;
    }

    auto stridesVec = in.strides().vec();
    std::vector<size_t> strides;
    strides.reserve(stridesVec.size());
    for (auto s : stridesVec) {
        if (s < 0) {
            throw std::runtime_error("Negative stride detected: " + std::to_string(s));
        }
        strides.push_back(static_cast<size_t>(s));
    }

    if (!allowPitched) {
        auto contigStride = ContiguousStride(sizes);
        std::string strideStr, contigStr;
        for (size_t i = 0; i < strides.size(); i++) {
            strideStr += (i == 0 ? "" : ", ") + std::to_string(strides[i]);
            contigStr += (i == 0 ? "" : ", ") + std::to_string(contigStride[i]);
        }
        throw std::invalid_argument(
            std::string(__func__) +
            ": torch tensor must be contiguous for this operation: tensor stride [" + strideStr +
            "] does not match the row-major layout [" + contigStr + "]; call .contiguous() first");
    }
    out.Init(sizes, XDtypeOf(in), TensorPtr(in), std::move(strides));
}

inline void InitXTensor(XTensor &out, const std::optional<torch::Tensor> &in,
                        bool allowPitched = false)
{
    if (in) {
        InitXTensor(out, *in, allowPitched);
    }
}

#endif  // _XLITE_H_
