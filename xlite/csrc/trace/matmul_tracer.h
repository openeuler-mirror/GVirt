/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "base.h"
#include "csv_sink.h"

class XRuntime;

namespace xlite_trace
{

// Value type identifying one aggregated matmul shape.
struct MatmulShape : public ICsvRecord {
    enum XDtype dtype;
    bool transpose;
    bool nz;
    uint64_t m;
    uint64_t n;
    uint64_t k;

    MatmulShape() = default;
    MatmulShape(enum XDtype d, bool t, bool z, uint64_t mm, uint64_t nn, uint64_t kk)
        : dtype(d), transpose(t), nz(z), m(mm), n(nn), k(kk)
    {
    }

    bool operator<(const MatmulShape &o) const;
    bool operator==(const MatmulShape &o) const;

    [[nodiscard]] std::string ToCsv() const override;

    // Column names matching ToCsv(), used to build the CSV header line.
    static const char *CsvHeader();
};

// Aggregates matmul shapes into a histogram and flushes it via a CsvSink.
class MatmulTracer
{
public:
    MatmulTracer(const MatmulTracer &) = delete;
    MatmulTracer &operator=(const MatmulTracer &) = delete;

    static MatmulTracer &Instance();

    void Record(uint32_t rankId, enum XDtype dtype, bool transpose, bool nz, uint64_t m, uint64_t n,
                uint64_t k);
    void RecordGroup(XRuntime &rt, XTensor &counts, uint32_t start, uint32_t end, enum XDtype dtype,
                     bool transpose, bool nz, uint64_t n, uint64_t k);

private:
    MatmulTracer() = default;
    ~MatmulTracer() = default;

    void Flush(uint32_t rankId);
    void Record(uint32_t rankId, const MatmulShape &s);

    std::mutex mutex_;
    std::map<MatmulShape, uint64_t> histogram_;
};

}  // namespace xlite_trace
