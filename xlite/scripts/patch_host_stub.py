#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Patch CANN's generated host_stub.cpp files.

Two independent, idempotent patches:

1. Overflow-status buffer cache (always applied). The stock generator emits,
   in every launch wrapper, an 8-byte device allocation plus a matching free
   around the kernel launch:

       constexpr uint32_t __ascendc_overflow_status_size = 8;
       AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow),
                            __ascendc_overflow_status_size);
       ... launch ...
       FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);

   The host never reads this buffer: the stock stub frees it immediately
   after the launch call returns, i.e. before the kernel has even started.
   Replacing the per-launch alloc/free pair with one cached buffer removes
   two device-allocator round-trips from the host submission path of every
   kernel launch. Measured (GLM-4.7-Flash, 910B2 x4, vllm bench serve,
   192x2048/256 conc 8): the removed pair accounted for ~4.8-6.9 ms/step of
   host aclrtMalloc+aclrtFree time; host cost per op call 27.2 -> 22.5 us.

   The cache is lifetime-safe: it is validated against the
   gXliteOverflowEpoch generation counter that libxlite bumps on every
   XRuntime::Init / teardown, so the buffer is re-allocated instead of going
   stale across aclrtSetDevice / aclrtResetDevice / aclFinalize (runtime
   destroyed and re-created within one process, sequential per-device
   runtimes). Allocation failure is retried on the next launch. The counter
   is a weakly-referenced data symbol, so the per-launch check is a guarded
   load plus one predicted branch -- no function call, no PLT indirection (a
   cross-TU weak-symbol function call here clobbers the launch wrapper's
   register allocation and regressed host-bound benches; see PR #619
   review).

   If a future CANN version changes the emitted pattern, the script warns
   and leaves that stub unpatched (stock behavior: correct, just slower).

2. Msprof shape hook (only with --msprof). The stubs pass StartAscendProf's
   startTime to ReportAscendProf, so it is the exact stamp of the node record
   msprof keys the launch on. Inject a weak hook
   XLiteMsprofShapeHook(name, startTime) right after StartAscendProf so
   xlite's shape reporter can stamp its tensor records with the same
   timestamp and merge into that node's descriptor. Inert when libxlite does
   not define the hook.

Files already containing the current-version markers are left untouched;
stubs carrying an older overflow-cache block are upgraded in place.
"""

import re
import sys
from pathlib import Path

# ---- patch 1: cached overflow-status buffer --------------------------------

# Stock generator text (constexpr line included: it would otherwise be left
# unused after the replacement).
CONSTEXPR_LINE = '    constexpr uint32_t __ascendc_overflow_status_size = 8;\n'
OVERFLOW_STOCK_BLOCK = (
    CONSTEXPR_LINE
    + '    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), '
    '__ascendc_overflow_status_size);\n'
)

# Stock alloc line without the constexpr (fallback if the pair differs).
OVERFLOW_ALLOC_ONLY = (
    '    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), '
    '__ascendc_overflow_status_size);\n'
)

OVERFLOW_FREE_LINE = '    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);\n'

# Declaration of libxlite's overflow-cache generation counter, inserted once
# per stub file next to the other runtime externs. A weakly-referenced DATA
# symbol, not a function: reading it is a load (through the GOT) plus one
# predicted branch -- no call, no PLT indirection, no caller-saved-register
# clobber in the launch wrapper. (A cross-TU weak-symbol function call here
# regressed host-bound benches by 0.7%; see PR #619 review.) Weak because
# libxlite_kernels_*.so do not link against libxlite (a non-weak reference
# would need a circular DT_NEEDED); at runtime the kernels .so and libxlite
# are in the same dlopen group, so the reference resolves, and without
# libxlite it reads as null and the stub treats the epoch as 0.
OVERFLOW_EPOCH_DECL = (
    '/* xlite: overflow-cache generation counter, defined by libxlite\n'
    ' * (weak data symbol, read-only in the stub); see\n'
    ' * scripts/patch_host_stub.py. */\n'
    'extern "C" unsigned long long gXliteOverflowEpoch '
    '__attribute__((weak));\n'
)
OVERFLOW_EPOCH_ANCHOR = 'uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);\n'

# v1 of this patch (magic-static init, no epoch validation): upgraded when
# found, e.g. in a build dir patched before this version.
OVERFLOW_V1_BLOCK = (
    '    /* Cached overflow-status buffer: host-side write-only scratch, the\n'
    '     * stock stub frees it right after launch without reading it back.\n'
    '     * One allocation here replaces a device malloc/free pair per launch. */\n'
    '    static void *__ascendc_overflow_cache = [] {\n'
    '        void *p = nullptr;\n'
    '        AllocAscendMemDevice(&p, 512);\n'
    '        return p;\n'
    '    }();\n'
    '    __ascendc_args.__ascendc_overflow = __ascendc_overflow_cache;\n'
)

# v2 of this patch (epoch check via a weak-symbol FUNCTION call): upgraded
# when found. Kept verbatim so build dirs patched by the previous script
# version are rewritten in place; the weak function declaration it inserted
# at file scope is swapped for the v3 data-symbol declaration.
OVERFLOW_V2_EPOCH_DECL = (
    '/* xlite: overflow-cache generation counter, defined by libxlite\n'
    ' (weak: the stub also works standalone); see scripts/patch_host_stub.py. */\n'
    'extern "C" unsigned long long xliteOverflowEpoch() '
    '__attribute__((weak));\n'
)
OVERFLOW_V2_CACHE_BLOCK = (
    '    /* Cached overflow-status buffer: host write-only scratch, the stock\n'
    '     * stub frees it right after launch without reading it back, so one\n'
    '     * allocation here replaces a device malloc/free pair per launch.\n'
    '     * Validated against the xliteOverflowEpoch() generation counter\n'
    '     * (weak, defined by libxlite): a runtime Init/teardown bumps it and\n'
    '     * the cache is re-allocated instead of holding an address stale\n'
    '     * across aclrtResetDevice/aclFinalize/device switches. Allocation\n'
    '     * failure retries on the next launch. Without libxlite the epoch\n'
    '     * reads as 0 and the cache persists for the process. */\n'
    '    static void *__ascendc_overflow_cache = nullptr;\n'
    '    static unsigned long long __ascendc_overflow_epoch = 0;\n'
    '    unsigned long long __xlite_ovf_epoch = '
    '&xliteOverflowEpoch != nullptr ? xliteOverflowEpoch() : 0;\n'
    '    if (__xlite_ovf_epoch != __ascendc_overflow_epoch) {\n'
    '        __ascendc_overflow_cache = nullptr;\n'
    '        __ascendc_overflow_epoch = __xlite_ovf_epoch;\n'
    '    }\n'
    '    if (__ascendc_overflow_cache == nullptr) {\n'
    '        void *__xlite_ovf_buf = nullptr;\n'
    '        if (AllocAscendMemDevice(&__xlite_ovf_buf, 512) == 0) {\n'
    '            __ascendc_overflow_cache = __xlite_ovf_buf;\n'
    '        }\n'
    '    }\n'
    '    __ascendc_args.__ascendc_overflow = __ascendc_overflow_cache;\n'
)

# v3: cached buffer + generation check + retry on allocation failure. One
# static pair per launch wrapper (a few hundred bytes of device memory per
# kernel library in total). Padded well past the stock 8B in case the kernel
# side writes per-block status entries. Stale buffers are dropped, not freed:
# after a device reset the address is already invalid, and on a device switch
# FreeAscendMemDevice would target the wrong device (see runtime.cpp). The
# retry makes a failed allocation retry on the next launch instead of caching
# nullptr forever; concurrent first launches may race and leak one 512B
# buffer, which is harmless.
#
# v3 vs v2: the generation check reads the weak data symbol
# gXliteOverflowEpoch directly (guarded acquire load; without libxlite it
# resolves to null and the epoch reads as 0) and folds cache-null and
# epoch-change into ONE __builtin_expect(..., 0) branch. The v2 form called
# the weak-symbol function xliteOverflowEpoch() per launch; an opaque
# cross-TU call cannot be inlined, clobbers the caller-saved registers the
# launch setup lives in (forcing spills) and regressed host-bound benches by
# ~0.7% (PR #619 review). The fast path is now loads + predicted branch
# only, no call.
OVERFLOW_CACHE_BLOCK = (
    '    /* Cached overflow-status buffer: host write-only scratch, the stock\n'
    '     * stub frees it right after launch without reading it back, so one\n'
    '     * allocation here replaces a device malloc/free pair per launch.\n'
    '     * Validated against the gXliteOverflowEpoch generation counter\n'
    '     * (weak data symbol defined by libxlite): a runtime Init/teardown\n'
    '     * bumps it and the cache is re-allocated instead of holding an\n'
    '     * address stale across aclrtResetDevice/aclFinalize/device\n'
    '     * switches. Allocation failure retries on the next launch.\n'
    '     * Without libxlite the weak symbol is null and the epoch reads\n'
    '     * as 0, so the cache persists for the process. */\n'
    '    static void *__ascendc_overflow_cache = nullptr;\n'
    '    static unsigned long long __ascendc_overflow_epoch = 0;\n'
    '    unsigned long long __xlite_ovf_epoch =\n'
    '        &gXliteOverflowEpoch != nullptr\n'
    '            ? __atomic_load_n(&gXliteOverflowEpoch, __ATOMIC_ACQUIRE)\n'
    '            : 0;\n'
    '    if (__builtin_expect(\n'
    '            __ascendc_overflow_cache == nullptr ||\n'
    '            __ascendc_overflow_epoch != __xlite_ovf_epoch, 0)) {\n'
    '        __ascendc_overflow_cache = nullptr;\n'
    '        __ascendc_overflow_epoch = __xlite_ovf_epoch;\n'
    '        void *__xlite_ovf_buf = nullptr;\n'
    '        if (AllocAscendMemDevice(&__xlite_ovf_buf, 512) == 0) {\n'
    '            __ascendc_overflow_cache = __xlite_ovf_buf;\n'
    '        }\n'
    '    }\n'
    '    __ascendc_args.__ascendc_overflow = __ascendc_overflow_cache;\n'
)

# Marker proving the file already carries the current (v3) cache block: the
# v3 data-symbol name. (v2 files carry the function name and are upgraded.)
OVERFLOW_MARKER = 'gXliteOverflowEpoch'


def patch_overflow(path: Path, text: str) -> str:
    import os
    if os.environ.get("XLITE_NO_OVERFLOW_PATCH"):
        return text  # A/B probe: keep stock per-launch alloc/free
    if OVERFLOW_MARKER in text:
        return text  # already current version (v3)

    # File-scope epoch declaration: insert next to the other runtime externs,
    # or swap the v2 weak-function declaration for the v3 data symbol.
    if OVERFLOW_V2_EPOCH_DECL in text:
        text = text.replace(OVERFLOW_V2_EPOCH_DECL, OVERFLOW_EPOCH_DECL)
    elif OVERFLOW_EPOCH_ANCHOR in text:
        text = text.replace(OVERFLOW_EPOCH_ANCHOR,
                            OVERFLOW_EPOCH_ANCHOR + OVERFLOW_EPOCH_DECL, 1)
    else:
        print(f"warning: {path}: AllocAscendMemDevice declaration not found; "
              "stub left unpatched (CANN generator changed?)", file=sys.stderr)
        return text

    if OVERFLOW_STOCK_BLOCK in text:
        count = text.count(OVERFLOW_STOCK_BLOCK)
        text = text.replace(OVERFLOW_STOCK_BLOCK, OVERFLOW_CACHE_BLOCK)
    elif OVERFLOW_ALLOC_ONLY in text:
        count = text.count(OVERFLOW_ALLOC_ONLY)
        text = text.replace(OVERFLOW_ALLOC_ONLY, OVERFLOW_CACHE_BLOCK)
    elif OVERFLOW_V1_BLOCK in text:
        # v1 replaced only the alloc line, so its files carry the stock
        # constexpr line right above the v1 block; consume it too.
        composite = CONSTEXPR_LINE + OVERFLOW_V1_BLOCK
        if composite in text:
            count = text.count(composite)
            text = text.replace(composite, OVERFLOW_CACHE_BLOCK)
        else:
            count = text.count(OVERFLOW_V1_BLOCK)
            text = text.replace(OVERFLOW_V1_BLOCK, OVERFLOW_CACHE_BLOCK)
    elif OVERFLOW_V2_CACHE_BLOCK in text:
        count = text.count(OVERFLOW_V2_CACHE_BLOCK)
        text = text.replace(OVERFLOW_V2_CACHE_BLOCK, OVERFLOW_CACHE_BLOCK)
    else:
        print(f"warning: {path}: overflow alloc/free pattern not found; "
              "stub left unpatched (CANN generator changed?)", file=sys.stderr)
        return text

    frees = text.count(OVERFLOW_FREE_LINE)
    text = text.replace(OVERFLOW_FREE_LINE, '')
    print(f"patched {path}: {count} overflow buffer cache(s), "
          f"{frees} free(s) removed")
    return text


# ---- patch 2: msprof shape hook ---------------------------------------------

HOOK_DECL = (
    'void XLiteMsprofShapeHook(const char *name, uint64_t startTime) '
    '__attribute__((weak));\n'
)
# Call through a local copy of the address: a weak symbol that xlite does not
# define (macro off) resolves to NULL, so calling it directly would crash the
# process once profiling is on.
HOOK_CALL = (
    '        void (*hook)(const char *, uint64_t) = XLiteMsprofShapeHook;\n'
    '        if (hook) {\n'
    '            hook(name, startTime);\n'
    '        }\n'
)

# The exact block the generator emits in every launch_and_profiling_<name>():
#     if (profStatus) {
#         StartAscendProf(name, &startTime);
#     }
LAUNCH_BLOCK = re.compile(
    r'^(?P<indent>[ ]{4}if \(profStatus\) \{\n'
    r'[ ]{8}StartAscendProf\(name, &startTime\);\n)'
    r'(?P<close>[ ]{4}\}\n)',
    re.M,
)


def patch_msprof(path: Path, text: str) -> str:
    if 'XLiteMsprofShapeHook' in text:
        return text  # already patched

    m = LAUNCH_BLOCK.search(text)
    if not m:
        raise SystemExit(f"{path}: StartAscendProf block not found")

    # Insert the declaration next to the other prof externs.
    anchor = 'void StartAscendProf(const char *name, uint64_t *startTime);\n'
    if anchor not in text:
        raise SystemExit(f"{path}: StartAscendProf declaration not found")
    text = text.replace(anchor, anchor + HOOK_DECL, 1)

    # Add the hook call after StartAscendProf in EVERY launch wrapper, so any
    # kernel whose caller staged shapes gets reported.
    text, count = LAUNCH_BLOCK.subn(
        lambda mm: mm.group('indent') + HOOK_CALL + mm.group('close'), text
    )
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
    msprof = '--msprof' in args
    dirs = [a for a in args if a != '--msprof']
    if not dirs:
        raise SystemExit("usage: patch_host_stub.py [--msprof] "
                         "<auto_gen_dir> [<auto_gen_dir> ...]")
    for arg in dirs:
        for stub in sorted(Path(arg).glob('*/host_stub.cpp')):
            patch_file(stub, msprof)


if __name__ == '__main__':
    main(sys.argv)
