/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "debug.h"

#include <iostream>
#ifdef XLITE_DEBUG_ON
#include <atomic>
#endif

/* ---- atomic print primitives -------------------------------------------- */

static std::mutex &DebugOutputMutex()
{
    static std::mutex m;
    return m;
}

void XDebugLog(bool hasRank, uint32_t rankId, const std::string &info, const std::string &body,
               XDebugColor overrideColor)
{
    std::string line;
    line.reserve(body.size() + 32);
    bool useOverride = (overrideColor != XDebugColor::Reset);
    const char *prefixColor = "";
    if (useOverride) {
        prefixColor = AnsiColor(overrideColor);
    } else if (hasRank) {
        prefixColor = AnsiColor(RankColor(rankId));
    }
    if (hasRank) {
        if (prefixColor[0] != '\0') {
            line += prefixColor;
        }
        line += RankTag(rankId);
        line += " ";
        if (!info.empty()) {
            line += info + ": ";
        }
        if (prefixColor[0] != '\0') {
            line += AnsiColor(XDebugColor::Reset);
        }
    } else if (!info.empty()) {
        if (prefixColor[0] != '\0') {
            line += prefixColor;
        }
        line += info + ": ";
        if (prefixColor[0] != '\0') {
            line += AnsiColor(XDebugColor::Reset);
        }
    }
    line += body;
    if (body.empty() || body.back() != '\n') {
        line += '\n';
    }

    std::scoped_lock lk(DebugOutputMutex());
    std::cout << line << std::flush;
}

/* ---- debug tensor ops ---------------------------------------------------- */

#ifdef XLITE_DEBUG_ON
static std::atomic<bool> &DebugPrintFlag()
{
    static std::atomic<bool> flag{true};
    return flag;
}

static bool DebugPrintEnabled()
{
    return DebugPrintFlag().load(std::memory_order_relaxed);
}

void XDebugSetState(bool condition)
{
    DebugPrintFlag().store(condition, std::memory_order_relaxed);
}

void XDebugPrintLog(XRuntime &rt, const char *header, const char *str)
{
    if (!DebugPrintEnabled() || rt.IsDummyRuntime()) {
        return;
    }
    {
        XDebugStream s(rt.rankId(), header);
        s << str << std::endl;
    }
}

void XDebugPrintOffset(XRuntime &rt, XTensor &h, const char *header)
{
    if (!DebugPrintEnabled() || rt.IsDummyRuntime()) {
        return;
    }
    {
        XDebugStream s(rt.rankId(), header);
        int64_t offset = rt.GetTensorOffset(h);
        if (offset < 0) {
            s << "tensor not in pool";
        } else {
            s << "offset in pool: " << offset;
        }
        s << ", shape=(";
        for (uint32_t i = 0; i < h.shape.size(); i++) {
            s << h.shape[i];
            if (i != h.shape.size() - 1) {
                s << ", ";
            }
        }
        s << "), dtype=" << XDtypeStr(h.dtype) << std::endl;
    }
}

void XDebugPrint(XRuntime &rt, XTensor &h, const char *header, float threshold)
{
    if (!DebugPrintEnabled() || rt.IsDummyRuntime()) {
        return;
    }
    rt.Synchronize();
    // Print and CheckNanInf share one stream so their output is gathered and
    // flushed as a single atomic line (rank-tagged, colored).
    {
        XDebugStream s(rt.rankId(), header);
        std::string offsetNote = "offset in pool: " + std::to_string(rt.GetTensorOffset(h));
        h.Print("", 6, 6, s.stream(), offsetNote.c_str());
        h.CheckNanInf(header, threshold, s.stream());
    }
}

void XDebugPrintRowsCols(XRuntime &rt, XTensor &h, const char *header, uint32_t rows, uint32_t cols,
                         float threshold)
{
    if (!DebugPrintEnabled() || rt.IsDummyRuntime()) {
        return;
    }
    rt.Synchronize();
    {
        XDebugStream s(rt.rankId(), header);
        std::string offsetNote = "offset in pool: " + std::to_string(rt.GetTensorOffset(h));
        h.Print("", rows, cols, s.stream(), offsetNote.c_str());
        h.CheckNanInf(header, threshold, s.stream());
    }
}

void XDebugPrintPtr(XRuntime &rt, XTensor &h, const char *header, std::vector<size_t> &subShape,
                    enum XDtype subDtype, float threshold)
{
    if (!DebugPrintEnabled() || rt.IsDummyRuntime()) {
        return;
    }
    rt.Synchronize();
    {
        XDebugStream s(rt.rankId(), header);
        std::string offsetNote = "offset in pool: " + std::to_string(rt.GetTensorOffset(h));
        h.PrintPtr(header, subShape, subDtype, 6, 6, s.stream(), offsetNote.c_str());
        h.CheckNanInf(header, threshold, s.stream());
    }
}

void XDebugCheckNanInf(XRuntime &rt, XTensor &h, const char *header, float threshold)
{
    if (!DebugPrintEnabled() || rt.IsDummyRuntime()) {
        return;
    }
    rt.Synchronize();
    {
        XDebugStream s(rt.rankId(), header);
        std::string offsetNote = "offset in pool: " + std::to_string(rt.GetTensorOffset(h));
        h.CheckNanInf("", threshold, s.stream(), offsetNote.c_str());
    }
}

void XDebugDumpXTensor(XRuntime &rt, XTensor &h, const std::string &path)
{
    if (!DebugPrintEnabled() || rt.IsDummyRuntime()) {
        return;
    }
    rt.Synchronize();
    h.Save(path);
}
#endif
