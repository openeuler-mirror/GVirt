/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Centralized debug module for xlite.
 *
 * Responsibilities:
 *   1. Provide rank-aware, color-coded, single-flush (atomic) printing so that
 *      multi-rank / multi-thread output never interleaves mid-line.
 *   2. Host the XDEBUG_* tensor inspection ops so they can consult
 *      Runtime.debug at run time.
 */
#ifndef _XLITE_DEBUG_H_
#define _XLITE_DEBUG_H_

#ifdef XLITE_DEBUG_ON_ALL
#ifndef XLITE_DEBUG_ON
#define XLITE_DEBUG_ON
#endif
#ifndef XLITE_DEBUG_ON_FORWARD
#define XLITE_DEBUG_ON_FORWARD
#endif
#ifndef XLITE_DEBUG_ON_TUNER
#define XLITE_DEBUG_ON_TUNER
#endif
#ifndef XLITE_DEBUG_ON_GETTENSOR
#define XLITE_DEBUG_ON_GETTENSOR
#endif
#ifndef XLITE_DEBUG_ON_MISC
#define XLITE_DEBUG_ON_MISC
#endif
#endif

#ifdef XLITE_DEBUG_ON_BASE
#ifndef XLITE_DEBUG_ON
#define XLITE_DEBUG_ON
#endif
#endif

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#ifdef XLITE_DEBUG_ON
#include "runtime.h"
#endif

/* ---- color helpers ------------------------------------------------------- */

// Each non-Reset enumerator's underlying value IS its xterm 256-color palette
// index (a light/"younger" shade chosen to read well on a black background).
// AnsiColor builds the escape from that value directly, so the index lives in
// exactly one place: the enum definition.
enum class XDebugColor {
    Red = 210,      // light red
    Orange = 215,   // light orange
    Yellow = 221,   // light yellow
    Green = 156,    // light green
    Cyan = 153,     // light cyan
    Blue = 117,     // light blue (brighter than the old ANSI 34)
    Magenta = 183,  // light magenta
    Pink = 218,     // light pink
    Reset = 0,      // terminal reset; not a rank color
};

// Ordered palette of the real rank colors (every enumerator except Reset).
// This is the single source of truth for the rank-color set and its size.
inline constexpr XDebugColor kXDebugPalette[] = {
    XDebugColor::Red,  XDebugColor::Orange, XDebugColor::Yellow,  XDebugColor::Green,
    XDebugColor::Cyan, XDebugColor::Blue,   XDebugColor::Magenta, XDebugColor::Pink};

// Number of available rank colors. Derived from the palette array, so adding
// or removing a color only requires editing the enum + kXDebugPalette -- no
// manual count to keep in sync.
constexpr size_t XDebugColorCount()
{
    return sizeof(kXDebugPalette) / sizeof(kXDebugPalette[0]);
}

// Return the ANSI escape for a color. For non-Reset colors the underlying enum
// value is the xterm index, so the escape is built from it directly (the
// static buffer makes it safe to return a const char*).
inline const char *AnsiColor(XDebugColor c)
{
    if (c == XDebugColor::Reset) {
        return "\033[0m";
    }
    // 256-color foreground: "\033[38;5;<idx>m" -- idx is the enum value (0..255).
    static thread_local char buf[16];
    int idx = static_cast<int>(c);
    // Bounds: idx is one of the enum palette indices, always 0..255.
    int n = std::snprintf(buf, sizeof(buf), "\033[38;5;%dm", idx);
    return (n > 0) ? buf : "";
}

// Pick a per-rank color by rankId modulo the number of available colors. The
// divisor is XDebugColorCount() (derived from kXDebugPalette), so it stays
// correct automatically if colors are added/removed.
inline XDebugColor RankColor(uint32_t rankId)
{
    return kXDebugPalette[rankId % XDebugColorCount()];
}

/* "R<rank>" tag (no color codes). */
inline std::string RankTag(uint32_t rankId)
{
    return "(R" + std::to_string(rankId) + ")";
}

/* ---- atomic print primitives -------------------------------------------- */

/*
 * Emit one debug line, atomically (single write + flush, guarded by a process-wide
 * mutex so concurrent threads/ranks do not interleave).
 *
 * When hasRank is true, the line is prefixed with "R<rank>" (plus optional info):
 *   R0 attn: <body>
 * The prefix is colored by rank (RankColor) unless overrideColor is set (i.e. not
 * Reset), in which case that color is used instead -- handy to force, say, red for
 * an error regardless of rank.
 * When hasRank is false, no R-tag is emitted; only the optional "<info>: " prefix
 * precedes the body, colored only if overrideColor is set.
 * A trailing newline is ensured.
 */
void XDebugLog(bool hasRank, uint32_t rankId, const std::string &info, const std::string &body,
               XDebugColor overrideColor = XDebugColor::Reset);

/*
 * RAII stream: accumulate with <<, then flush atomically on destruction.
 *
 * rank id is OPTIONAL -- use the no-rank constructor when no rank is available
 * (e.g. inside free functions with no runtime context):
 *   XDebugStream s(rankId, "attn");    // rank-aware: colored "R<rank> attn: " prefix
 *   XDebugStream s("auto_tuner");      // no rank:    plain "<info>: " prefix (no color)
 *   XDebugStream s;                    // no rank, no info: bare content
 *
 * To force a specific color instead of the rank-derived one, chain .color(...):
 *   XDebugStream s(rankId, "error").color(XDebugColor::Red) << "boom";
 * Reset is not allowed (it is the "no override" sentinel).
 *
 * If the content stream is empty on destruction, nothing is emitted.
 */
class XDebugStream
{
public:
    // Rank-aware: output prefixed with colored "R<rank> <info>: ".
    XDebugStream(uint32_t rankId, std::string info = "", bool condition = true)
        : _hasRank(true), _rankId(rankId), _info(std::move(info)), _condition(condition)
    {
    }
    // No rank available: output prefixed with "<info>: " (no color, no R-tag).
    explicit XDebugStream(std::string info = "", bool condition = true)
        : _hasRank(false), _rankId(0), _info(std::move(info)), _condition(condition)
    {
    }
    ~XDebugStream()
    {
        if (_condition && !_oss.str().empty()) {
            XDebugLog(_hasRank, _rankId, _info, _oss.str(), _overrideColor);
        }
    }

    // non-copyable, non-movable (flushes mutex-protected output on destruction)
    XDebugStream(const XDebugStream &) = delete;
    XDebugStream &operator=(const XDebugStream &) = delete;
    XDebugStream(XDebugStream &&) = delete;
    XDebugStream &operator=(XDebugStream &&) = delete;

    // Override the rank-derived color. `c` must not be Reset (Reset is the
    // "no override" sentinel; passing it is a no-op).
    XDebugStream &color(XDebugColor c)
    {
        if (c != XDebugColor::Reset) {
            _overrideColor = c;
        }
        return *this;
    }

    template <typename T>
    std::ostringstream &operator<<(const T &v)
    {
        _oss << v;
        return _oss;
    }

    // Access the underlying stream so callers can pass it to functions that take
    // std::ostream& (e.g. XTensor::Print(..., os)). Output written this way is
    // captured and flushed atomically (with rank tag + color) on destruction.
    std::ostringstream &stream()
    {
        return _oss;
    }

private:
    bool _hasRank;
    uint32_t _rankId;
    std::string _info;
    bool _condition = true;                           // default: print
    XDebugColor _overrideColor = XDebugColor::Reset;  // Reset => use rank color
    std::ostringstream _oss;
};

/* ---------- debug tensor ops -------------------------------- */
/*
 * Debug inspection entry points, used across model.cpp/op.cpp via the
 * XDEBUG_* macros below. Each op, when active:
 *   - gates on a process-wide debug flag (defaulted OFF when compiled in; set
 *     per-call with XDebugSetState / XDEBUG_SET_STATE to fold a dynamic
 *     condition -- e.g. rank/layer narrowing -- into the gate),
 *   - synchronizes the runtime stream before reading device memory,
 *   - runs the inspection (Print / PrintPtr / Save) and, into the SAME stream,
 *     a CheckNanInf pass -- so the print and any NaN/Inf summary flush together
 *     as one atomic, rank-tagged, colored line.
 *
 * The optional `threshold` (forwarded to CheckNanInf; -1.0 disables the
 * large-value check) is a default argument on the XDebug* functions, so the
 * macros have a single form each -- callers may omit threshold.
 *
 * Categorized control (driven by the XLITE_DEBUG_ON env var; CMakeLists.txt
 * parses it into macros, this header maps macros to behavior):
 *   XLITE_DEBUG_ON              base debug -- defined for any non-falsy env
 *                              value. Compiles the SHARED debug infra:
 *                              XDebugStream/XDebugLog, the XDebug* impls in
 *                              debug.cpp (incl. the runtime flag), <torch/torch.h>,
 *                              ToScalarType/Save in base.*. Every category
 *                              needs the base macro.
 *   XLITE_DEBUG_ON_FORWARD      the XDEBUG_PRINT* / XDEBUG_SET_STATE macros are
 *                              real (else empty). The forward intermediate-
 *                              tensor prints in model.cpp are LIVE (uncommented)
 *                              but compile to nothing unless this is defined, so
 *                              forward is INDEPENDENT of tuner/gettensor.
 *   XLITE_DEBUG_ON_TUNER        #ifdef block in auto_tuner.cpp (tile-size logs).
 *   XLITE_DEBUG_ON_GETTENSOR    #ifdef blocks in base.cpp (dummy-pool alloc logs).
 *   XLITE_DEBUG_ON_MISC         #ifdef blocks in _C.cpp (variance logs).
 *
 * Env grammar (see CMakeLists.txt): XLITE_DEBUG_ON is unset/empty/falsy -> no
 * macros; =<tok>[,<tok>...] -> XLITE_DEBUG_ON + XLITE_DEBUG_ON_<TOK> per token,
 * e.g. XLITE_DEBUG_ON=forward,tuner -> XLITE_DEBUG_ON, XLITE_DEBUG_ON_FORWARD,
 * XLITE_DEBUG_ON_TUNER.
 */
#ifdef XLITE_DEBUG_ON
void XDebugSetState(bool condition);
void XDebugPrint(XRuntime &rt, XTensor &h, const char *str, float threshold = -1.0f);
void XDebugPrintRowsCols(XRuntime &rt, XTensor &h, const char *str, uint32_t rows, uint32_t cols,
                         float threshold = -1.0f);
void XDebugPrintPtr(XRuntime &rt, XTensor &h, const char *str, std::vector<size_t> &subShape,
                    enum XDtype subDtype, float threshold = -1.0f);
void XDebugCheckNanInf(XRuntime &rt, XTensor &h, const char *str, float threshold = -1.0f);
void XDebugDumpXTensor(XRuntime &rt, XTensor &h, const std::string &path);
#define XDEBUG_LOG(rt, str) XDebugLog(true, (rt).rankId(), "", (str))
#else
#define XDEBUG_LOG(rt, str)
#endif

/* ---- forward category: print/inspect macros ----------------------------- *
 * These are the inspection ops used by the forward pass in model.cpp. They are
 * real only when BOTH XLITE_DEBUG_ON (base, for the XDebug* impls they call)
 * and XLITE_DEBUG_ON_FORWARD are defined.
 */
#if defined(XLITE_DEBUG_ON) && defined(XLITE_DEBUG_ON_FORWARD)
#define XDEBUG_SET_STATE(condition) XDebugSetState((condition))
#define XDEBUG_PRINT(rt, h, str) XDebugPrint((rt), (h), (str))
#define XDEBUG_PRINT_X(rt, h, str, threshold) XDebugPrint((rt), (h), (str), (threshold))
#define XDEBUG_PRINT_ROWS_COLS(rt, h, str, rows, cols) \
    XDebugPrintRowsCols((rt), (h), (str), (rows), (cols))
#define XDEBUG_PRINT_ROWS_COLS_X(rt, h, str, rows, cols, threshold) \
    XDebugPrintRowsCols((rt), (h), (str), (rows), (cols), (threshold))
#define XDEBUG_PRINT_PTR(rt, h, str, subShape, subDtype) \
    XDebugPrintPtr((rt), (h), (str), (subShape), (subDtype))
#define XDEBUG_PRINT_PTR_X(rt, h, str, subShape, subDtype, threshold) \
    XDebugPrintPtr((rt), (h), (str), (subShape), (subDtype), (threshold))
#define XDEBUG_CHECK_NAN_INF(rt, h, str) XDebugCheckNanInf((rt), (h), (str))
#define XDEBUG_CHECK_NAN_INF_X(rt, h, str, threshold) \
    XDebugCheckNanInf((rt), (h), (str), (threshold))
#define XDEBUG_DUMP_XTENSOR(rt, h, path) XDebugDumpXTensor((rt), (h), (path))
#else
#define XDEBUG_SET_STATE(condition)
#define XDEBUG_PRINT(rt, h, str)
#define XDEBUG_PRINT_X(rt, h, str, threshold)
#define XDEBUG_PRINT_ROWS_COLS(rt, h, str, rows, cols)
#define XDEBUG_PRINT_ROWS_COLS_X(rt, h, str, rows, cols, threshold)
#define XDEBUG_PRINT_PTR(rt, h, str, subShape, subDtype)
#define XDEBUG_PRINT_PTR_X(rt, h, str, subShape, subDtype, threshold)
#define XDEBUG_CHECK_NAN_INF(rt, h, str)
#define XDEBUG_CHECK_NAN_INF_X(rt, h, str, threshold)
#define XDEBUG_DUMP_XTENSOR(rt, h, path)
#endif

#endif
