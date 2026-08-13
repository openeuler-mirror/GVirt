/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace xlite_trace
{

// Interface for values that can serialize themselves to a CSV row.
class ICsvRecord
{
public:
    virtual ~ICsvRecord() = default;

    // CSV row body (columns between "rank" and "count").
    [[nodiscard]] virtual std::string ToCsv() const = 0;
};

// Crash-safe CSV output: writes to "<path>.tmp" then atomically renames it over
// <path> so the on-disk file is always complete (only the in-flight write is lost
// on SIGKILL). Decoupled from concrete record types via ICsvRecord.
class CsvSink
{
public:
    explicit CsvSink(std::string path);
    CsvSink(const CsvSink &) = delete;
    CsvSink &operator=(const CsvSink &) = delete;

    bool Open();
    void WriteHeader(const std::string &columns);
    void WriteRow(uint32_t rankId, const ICsvRecord &r, uint64_t count);
    void Commit();

private:
    std::string path_;
    std::string tmp_;
    FILE *file_ = nullptr;
};

}  // namespace xlite_trace
