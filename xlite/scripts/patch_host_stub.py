#!/usr/bin/env python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Patch CANN's generated host_stub.cpp files.

Two independent, idempotent patches:

1. Overflow-status buffer removal (always applied). The stock generator
   emits, in every launch wrapper, an 8-byte device allocation plus a
   matching free around the kernel launch:

       constexpr uint32_t __ascendc_overflow_status_size = 8;
       AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow),
                            __ascendc_overflow_status_size);
       ... launch ...
       FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);

   This buffer is a debug facility: the CCE-compiled operator checks, at
   its exit, whether the pointer argument is non-null and then stores the
   STATUS register's overflow bits into it (each launch gets its own
   buffer so the overflowing operator can be identified). The host never
   reads it back -- the stock stub frees it right after the launch call
   returns. Per review feedback (#619) the per-launch device malloc/free
   is pure launch-path overhead for that debug facility, so the patch
   passes null instead: the kernel-side write is null-guarded, which
   merely turns overflow-status reporting off until it is needed again.
   The slot stays in the launch-argument struct -- its layout is part of
   the host/device ABI the generator lays out on both sides -- but it is
   set to null and never allocated. Real out-of-bounds access remains an
   AICore hardware trap, and the public API for explicit checks is
   aclrtGetOverflowStatus; neither goes through this buffer.

   If a future CANN version changes the emitted pattern, the script warns
   and leaves that stub unpatched (stock behavior: correct, just slower).

2. Msprof shape hook (only with --msprof). The stubs pass StartAscendProf's
   startTime to ReportAscendProf, so it is the exact stamp of the node record
   msprof keys the launch on. Inject a weak hook
   XLiteMsprofShapeHook(name, startTime) right after StartAscendProf so
   xlite's shape reporter can stamp its tensor records with the same
   timestamp and merge into that node's descriptor. Inert when libxlite does
   not define the hook.

Files already containing the current-version marker are left untouched;
stubs carrying an older overflow-cache block are upgraded in place.
"""

import re
import sys
from pathlib import Path

# ---- patch 1: overflow-status buffer removal --------------------------------

# Stock generator text (constexpr line included: it would otherwise be left
# unused after the replacement).
CONSTEXPR_LINE = "    constexpr uint32_t __ascendc_overflow_status_size = 8;\n"
OVERFLOW_STOCK_BLOCK = (
    CONSTEXPR_LINE + "    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);\n"
)

# Stock alloc line without the constexpr (fallback if the pair differs).
OVERFLOW_ALLOC_ONLY = (
    "    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);\n"
)

OVERFLOW_FREE_LINE = "    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);\n"

# Replacement: the buffer is removed entirely. The argument slot stays (its
# layout is part of the generator's host/device ABI) but is set to null: the
# CCE-compiled operator null-checks the pointer at its exit and only writes
# the overflow status when it is non-null, so passing null just disables the
# debug-only overflow-status reporting (see the module docstring).
OVERFLOW_REMOVED_ASSIGN = (
    "    __ascendc_args.__ascendc_overflow = nullptr; "
    "/* xlite: overflow-status buffer removed (debug-only,\n"
    "     * kernel-side write is null-guarded), see patch_host_stub.py. */\n"
)

# Marker proving the file already carries the removal (idempotency).
OVERFLOW_MARKER = "overflow-status buffer removed"

# Legacy epoch declarations (v2 weak function / v3 weak data symbol) that
# earlier versions of this script inserted at file scope: stripped when an
# old cache block is upgraded to the removal.
OVERFLOW_EPOCH_DECL = (
    "/* xlite: overflow-cache generation counter, defined by libxlite\n"
    " * (weak data symbol, read-only in the stub); see\n"
    " * scripts/patch_host_stub.py. */\n"
    'extern "C" unsigned long long gXliteOverflowEpoch '
    "__attribute__((weak));\n"
)
OVERFLOW_V2_EPOCH_DECL = (
    "/* xlite: overflow-cache generation counter, defined by libxlite\n"
    " (weak: the stub also works standalone); see scripts/patch_host_stub.py. */\n"
    'extern "C" unsigned long long xliteOverflowEpoch() '
    "__attribute__((weak));\n"
)

# v1 of this patch (magic-static init, no epoch validation): upgraded when
# found, e.g. in a build dir patched before this version.
OVERFLOW_V1_BLOCK = (
    "    /* Cached overflow-status buffer: host-side write-only scratch, the\n"
    "     * stock stub frees it right after launch without reading it back.\n"
    "     * One allocation here replaces a device malloc/free pair per launch. */\n"
    "    static void *__ascendc_overflow_cache = [] {\n"
    "        void *p = nullptr;\n"
    "        AllocAscendMemDevice(&p, 512);\n"
    "        return p;\n"
    "    }();\n"
    "    __ascendc_args.__ascendc_overflow = __ascendc_overflow_cache;\n"
)

# v2 of this patch (epoch check via a weak-symbol FUNCTION call): upgraded
# when found, together with its file-scope declaration above.
OVERFLOW_V2_CACHE_BLOCK = (
    "    /* Cached overflow-status buffer: host write-only scratch, the stock\n"
    "     * stub frees it right after launch without reading it back, so one\n"
    "     * allocation here replaces a device malloc/free pair per launch.\n"
    "     * Validated against the xliteOverflowEpoch() generation counter\n"
    "     * (weak, defined by libxlite): a runtime Init/teardown bumps it and\n"
    "     * the cache is re-allocated instead of holding an address stale\n"
    "     * across aclrtResetDevice/aclFinalize/device switches. Allocation\n"
    "     * failure retries on the next launch. Without libxlite the epoch\n"
    "     * reads as 0 and the cache persists for the process. */\n"
    "    static void *__ascendc_overflow_cache = nullptr;\n"
    "    static unsigned long long __ascendc_overflow_epoch = 0;\n"
    "    unsigned long long __xlite_ovf_epoch = "
    "&xliteOverflowEpoch != nullptr ? xliteOverflowEpoch() : 0;\n"
    "    if (__xlite_ovf_epoch != __ascendc_overflow_epoch) {\n"
    "        __ascendc_overflow_cache = nullptr;\n"
    "        __ascendc_overflow_epoch = __xlite_ovf_epoch;\n"
    "    }\n"
    "    if (__ascendc_overflow_cache == nullptr) {\n"
    "        void *__xlite_ovf_buf = nullptr;\n"
    "        if (AllocAscendMemDevice(&__xlite_ovf_buf, 512) == 0) {\n"
    "            __ascendc_overflow_cache = __xlite_ovf_buf;\n"
    "        }\n"
    "    }\n"
    "    __ascendc_args.__ascendc_overflow = __ascendc_overflow_cache;\n"
)

# v3 of this patch (epoch check via a weak DATA symbol, no call in the hot
# path): upgraded when found, together with its file-scope declaration
# above.
OVERFLOW_CACHE_BLOCK = (
    "    /* Cached overflow-status buffer: host write-only scratch, the stock\n"
    "     * stub frees it right after launch without reading it back, so one\n"
    "     * allocation here replaces a device malloc/free pair per launch.\n"
    "     * Validated against the gXliteOverflowEpoch generation counter\n"
    "     * (weak data symbol defined by libxlite): a runtime Init/teardown\n"
    "     * bumps it and the cache is re-allocated instead of holding an\n"
    "     * address stale across aclrtResetDevice/aclFinalize/device\n"
    "     * switches. Allocation failure retries on the next launch.\n"
    "     * Without libxlite the weak symbol is null and the epoch reads\n"
    "     * as 0, so the cache persists for the process. */\n"
    "    static void *__ascendc_overflow_cache = nullptr;\n"
    "    static unsigned long long __ascendc_overflow_epoch = 0;\n"
    "    unsigned long long __xlite_ovf_epoch =\n"
    "        &gXliteOverflowEpoch != nullptr\n"
    "            ? __atomic_load_n(&gXliteOverflowEpoch, __ATOMIC_ACQUIRE)\n"
    "            : 0;\n"
    "    if (__builtin_expect(\n"
    "            __ascendc_overflow_cache == nullptr ||\n"
    "            __ascendc_overflow_epoch != __xlite_ovf_epoch, 0)) {\n"
    "        __ascendc_overflow_cache = nullptr;\n"
    "        __ascendc_overflow_epoch = __xlite_ovf_epoch;\n"
    "        void *__xlite_ovf_buf = nullptr;\n"
    "        if (AllocAscendMemDevice(&__xlite_ovf_buf, 512) == 0) {\n"
    "            __ascendc_overflow_cache = __xlite_ovf_buf;\n"
    "        }\n"
    "    }\n"
    "    __ascendc_args.__ascendc_overflow = __ascendc_overflow_cache;\n"
)


def patch_overflow(path: Path, text: str) -> str:
    import os

    if os.environ.get("XLITE_NO_OVERFLOW_PATCH"):
        return text  # A/B probe: keep stock per-launch alloc/free
    if OVERFLOW_MARKER in text:
        return text  # already patched (buffer removed)

    # Legacy file-scope epoch declarations are stripped when upgrading.
    for decl in (OVERFLOW_V2_EPOCH_DECL, OVERFLOW_EPOCH_DECL):
        if decl in text:
            text = text.replace(decl, "")

    if OVERFLOW_STOCK_BLOCK in text:
        count = text.count(OVERFLOW_STOCK_BLOCK)
        text = text.replace(OVERFLOW_STOCK_BLOCK, OVERFLOW_REMOVED_ASSIGN)
    elif OVERFLOW_ALLOC_ONLY in text:
        count = text.count(OVERFLOW_ALLOC_ONLY)
        text = text.replace(OVERFLOW_ALLOC_ONLY, OVERFLOW_REMOVED_ASSIGN)
    else:
        # Legacy cache blocks (v1/v2/v3) upgrade to the removal in place.
        count = 0
        for legacy in (
            CONSTEXPR_LINE + OVERFLOW_V1_BLOCK,
            OVERFLOW_V1_BLOCK,
            OVERFLOW_V2_CACHE_BLOCK,
            OVERFLOW_CACHE_BLOCK,
        ):
            if legacy in text:
                count += text.count(legacy)
                text = text.replace(legacy, OVERFLOW_REMOVED_ASSIGN)
        if count == 0:
            print(
                f"warning: {path}: overflow alloc/free pattern not found; "
                "stub left unpatched (CANN generator changed?)",
                file=sys.stderr,
            )
            return text

    frees = text.count(OVERFLOW_FREE_LINE)
    text = text.replace(OVERFLOW_FREE_LINE, "")
    print(f"patched {path}: removed {count} overflow buffer alloc(s), {frees} free(s)")
    return text


# ---- patch 2: msprof shape hook ---------------------------------------------

HOOK_DECL = "void XLiteMsprofShapeHook(const char *name, uint64_t startTime) __attribute__((weak));\n"
# Call through a local copy of the address: a weak symbol that xlite does not
# define (macro off) resolves to NULL, so calling it directly would crash the
# process once profiling is on.
HOOK_CALL = (
    "        void (*hook)(const char *, uint64_t) = XLiteMsprofShapeHook;\n"
    "        if (hook) {\n"
    "            hook(name, startTime);\n"
    "        }\n"
)

# The exact block the generator emits in every launch_and_profiling_<name>():
#     if (profStatus) {
#         StartAscendProf(name, &startTime);
#     }
LAUNCH_BLOCK = re.compile(
    r"^(?P<indent>[ ]{4}if \(profStatus\) \{\n"
    r"[ ]{8}StartAscendProf\(name, &startTime\);\n)"
    r"(?P<close>[ ]{4}\}\n)",
    re.M,
)


def patch_msprof(path: Path, text: str) -> str:
    if "XLiteMsprofShapeHook" in text:
        return text  # already patched

    m = LAUNCH_BLOCK.search(text)
    if not m:
        raise SystemExit(f"{path}: StartAscendProf block not found")

    # Insert the declaration next to the other prof externs.
    anchor = "void StartAscendProf(const char *name, uint64_t *startTime);\n"
    if anchor not in text:
        raise SystemExit(f"{path}: StartAscendProf declaration not found")
    text = text.replace(anchor, anchor + HOOK_DECL, 1)

    # Add the hook call after StartAscendProf in EVERY launch wrapper, so any
    # kernel whose caller staged shapes gets reported.
    text, count = LAUNCH_BLOCK.subn(lambda mm: mm.group("indent") + HOOK_CALL + mm.group("close"), text)
    if count == 0:
        raise SystemExit(f"{path}: no launch block patched")

    print(f"patched {path}: {count} launch wrapper(s)")
    return text


def patch_file(path: Path, msprof: bool) -> None:
    text = path.read_text(encoding="utf-8")
    patched = patch_overflow(path, text)
    if msprof:
        patched = patch_msprof(path, patched)
    if patched != text:
        path.write_text(patched, encoding="utf-8")


def main(argv):
    args = argv[1:]
    msprof = "--msprof" in args
    dirs = [a for a in args if a != "--msprof"]
    if not dirs:
        raise SystemExit("usage: patch_host_stub.py [--msprof] <auto_gen_dir> [<auto_gen_dir> ...]")
    for arg in dirs:
        for stub in sorted(Path(arg).glob("*/host_stub.cpp")):
            patch_file(stub, msprof)


if __name__ == "__main__":
    main(sys.argv)
