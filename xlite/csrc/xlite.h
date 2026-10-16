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

// Wrap a torch tensor as an XTensor. The layout must be exactly contiguous (torch's
// is_contiguous() is looser around size-1 dims, so the strides are checked directly);
// callers must .contiguous() anything else. Zero-element tensors carry no data and are
// wrapped as contiguous regardless of stride. allowPitched additionally accepts a dim-0
// row pitch for consumers that address memory through stride[0]: kv caches
// (MHA/GQA/MLA/DSA/CXA) and debug output (Print/Save).
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

    auto stridesVec = in.strides().vec();
    std::vector<size_t> strides;
    strides.reserve(stridesVec.size());

    for (auto s : stridesVec) {
        if (s < 0) {
            throw std::runtime_error("Negative stride detected: " + std::to_string(s));
        }
        strides.push_back(static_cast<size_t>(s));
    }

    if (in.numel() == 0) {
        out.Init(sizes, XDtypeOf(in), TensorPtr(in));
        return;
    }
    auto contigStride = ContiguousStride(sizes);
    if (!allowPitched && strides != contigStride) {
        std::string strideStr, contigStr;
        for (size_t i = 0; i < strides.size(); i++) {
            strideStr += (i == 0 ? "" : ", ") + std::to_string(strides[i]);
            contigStr += (i == 0 ? "" : ", ") + std::to_string(contigStride[i]);
        }
        // torch's is_contiguous() accepts size-1-dim layouts this check rejects, and
        // .contiguous() is a no-op for those, so the remedy is to re-materialize.
        std::string remedy = in.is_contiguous()
                                 ? "torch reports it contiguous (size-1 dims); re-materialize it, "
                                   "e.g. x = x.reshape(-1).reshape(x.shape)"
                                 : "call .contiguous() first";
        throw std::invalid_argument(
            std::string(__func__) +
            ": torch tensor must be contiguous for this operation: stride [" + strideStr +
            "] does not match the row-major layout [" + contigStr + "]; " + remedy);
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
