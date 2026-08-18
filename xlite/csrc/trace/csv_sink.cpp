/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <utility>
#include <cinttypes>

#include "csv_sink.h"

namespace xlite_trace
{

CsvSink::CsvSink(std::string path) : path_(std::move(path)), tmp_(path_ + ".tmp")
{
}

bool CsvSink::Open()
{
    file_ = std::fopen(tmp_.c_str(), "w");
    return file_ != nullptr;
}

void CsvSink::WriteHeader(const std::string &columns)
{
    (void)std::fprintf(file_, "%s\n", columns.c_str());
}

void CsvSink::WriteRow(uint32_t rankId, const ICsvRecord &r, uint64_t count)
{
    std::string row = r.ToCsv();
    (void)std::fprintf(file_, "%" PRIu32 ",%s,%llu\n", rankId, row.c_str(),
                       static_cast<unsigned long long>(count));
}

void CsvSink::Commit()
{
    if (file_ == nullptr) {
        return;
    }
    (void)std::fclose(file_);
    file_ = nullptr;
    (void)std::rename(tmp_.c_str(), path_.c_str());
}

}  // namespace xlite_trace
