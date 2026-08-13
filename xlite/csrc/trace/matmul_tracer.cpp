/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */

#include <algorithm>
#include <vector>

#include "matmul_tracer.h"
#include "runtime.h"

namespace xlite_trace
{
std::string MatmulShape::ToCsv() const
{
    return std::string(XDtypeStr(dtype)) + "," + std::to_string(transpose) + "," +
           std::to_string(nz) + "," + std::to_string(m) + "," + std::to_string(n) + "," +
           std::to_string(k);
}

bool MatmulShape::operator<(const MatmulShape &o) const
{
    return std::tie(dtype, transpose, nz, m, n, k) <
           std::tie(o.dtype, o.transpose, o.nz, o.m, o.n, o.k);
}

bool MatmulShape::operator==(const MatmulShape &o) const
{
    return std::tie(dtype, transpose, nz, m, n, k) ==
           std::tie(o.dtype, o.transpose, o.nz, o.m, o.n, o.k);
}

const char *MatmulShape::CsvHeader()
{
    return "dtype,transpose,nz,m,n,k";
}

MatmulTracer &MatmulTracer::Instance()
{
    static MatmulTracer t;
    return t;
}

void MatmulTracer::Flush(uint32_t rankId)
{
    auto s = std::string("matmul_trace_r") + std::to_string(rankId) + ".csv";
    CsvSink sink(s);
    if (!sink.Open()) {
        return;
    }
    sink.WriteHeader(std::string("rank,") + MatmulShape::CsvHeader() + ",count");
    for (const auto &kv : histogram_) {
        sink.WriteRow(rankId, kv.first, kv.second);
    }
    sink.Commit();
}

void MatmulTracer::Record(uint32_t rankId, const MatmulShape &s)
{
    std::scoped_lock<std::mutex> lk(mutex_);
    ++histogram_[s];
    Flush(rankId);
}

void MatmulTracer::Record(uint32_t rankId, enum XDtype dtype, bool transpose, bool nz, uint64_t m,
                          uint64_t n, uint64_t k)
{
    Record(rankId, MatmulShape(dtype, transpose, nz, m, n, k));
}

void MatmulTracer::RecordGroup(XRuntime &rt, XTensor &counts, uint32_t start, uint32_t end,
                               enum XDtype dtype, bool transpose, bool nz, uint64_t n, uint64_t k)
{
    uint64_t nExperts = counts.shape[0];
    uint32_t startClamped = static_cast<uint32_t>(std::min<uint64_t>(start, nExperts));
    uint32_t endClamped = static_cast<uint32_t>(std::min<uint64_t>(end, nExperts));
    if (endClamped <= startClamped) {
        return;
    }
    uint32_t numExperts = endClamped - startClamped;
    std::vector<uint32_t> countsHost(numExperts);
    rt.Synchronize();
    rt.MemcpyD2H(countsHost.data(),
                 reinterpret_cast<uint8_t *>(counts.ptr) +
                     static_cast<size_t>(startClamped) * sizeof(uint32_t),
                 static_cast<size_t>(numExperts) * sizeof(uint32_t));

    std::scoped_lock<std::mutex> lk(mutex_);
    for (uint32_t i = 0; i < numExperts; i++) {
        uint32_t m = countsHost[i];
        if (m == 0) {
            continue;
        }
        ++histogram_[MatmulShape(dtype, transpose, nz, m, n, k)];
    }
    Flush(rt.rankId());
}

}  // namespace xlite_trace
