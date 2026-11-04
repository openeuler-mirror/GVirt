#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Unit test for scripts/patch_host_stub.py (pure host, no NPU needed).

Synthesizes CANN-style host_stub.cpp files (stock, a stub carrying the v1
cache block, and one carrying the v2 block with the weak-symbol epoch
FUNCTION call), runs the patch script on them, and asserts the transformed
text: v3 cache blocks reading the hidden data symbol gXliteOverflowEpoch
(no call in the launch path), the hidden extern declaration, no leftover
stock alloc/free, no orphaned constexpr, no v1/v2 leftovers, and idempotent
re-runs.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "patch_host_stub.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("patch_host_stub", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load_script_module()

EXTERN_DECLS = """uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
void StartAscendProf(const char *name, uint64_t *startTime);
"""

STOCK_WRAPPER = """void launch_and_profiling_k(float *x, float *y, float *z,
    uint32_t xNumel, uint32_t yNumel, uint32_t numBlocks, void *stream)
{
    alignas(8) struct __Args {
        void *x;
        void *y;
        void *z;
        void *__ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.x = x;
    __ascendc_args.__ascendc_args_y = y;
    __ascendc_ret = launch_and_profiling_k(0, numBlocks, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}
"""

V1_CACHE_BLOCK = """    /* Cached overflow-status buffer: host-side write-only scratch, the
     * stock stub frees it right after launch without reading it back.
     * One allocation here replaces a device malloc/free pair per launch. */
    static void *__ascendc_overflow_cache = [] {
        void *p = nullptr;
        AllocAscendMemDevice(&p, 512);
        return p;
    }();
    __ascendc_args.__ascendc_overflow = __ascendc_overflow_cache;
"""

STOCK_BLOCK = (
    "    constexpr uint32_t __ascendc_overflow_status_size = 8;\n"
    "    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), "
    "__ascendc_overflow_status_size);\n"
)


def make_stub_dir(base, name, cache_block=None, decls=""):
    d = base / name
    d.mkdir(parents=True)
    body = STOCK_WRAPPER
    if cache_block is not None:
        body = body.replace(STOCK_BLOCK, cache_block)
    (d / "host_stub.cpp").write_text(EXTERN_DECLS + decls + body)
    return d.parent


def run_patch(auto_gen):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(auto_gen)],
        check=True, capture_output=True, text=True,
    )


def check_stub(path):
    text = path.read_text()
    assert "gXliteOverflowEpoch" in text, "v3 cache block missing"
    assert text.count("__atomic_load_n(&gXliteOverflowEpoch") == 1, \
        "expected one v3 block (single guarded acquire load)"
    assert 'extern "C" unsigned long long gXliteOverflowEpoch' in text, \
        "weak data-symbol decl missing"
    assert "__attribute__((weak))" in text, "weak attribute missing"
    assert "xliteOverflowEpoch()" not in text, "v2 weak function left"
    assert "&xliteOverflowEpoch" not in text, "v2 weak guard left"
    assert "AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow)" not in text, "stock alloc left"
    assert "FreeAscendMemDevice(__ascendc_args.__ascendc_overflow)" not in text, "stock free left"
    assert "__ascendc_overflow_status_size = 8" not in text, "orphaned constexpr left"
    assert "__ascendc_overflow_cache = []" not in text, "v1 block left"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Stock -> v3.
        auto_gen = make_stub_dir(base, "stock", cache_block=None)
        out = run_patch(auto_gen)
        assert "overflow buffer cache(s)" in out.stdout, out.stdout
        check_stub(auto_gen / "stock" / "host_stub.cpp")

        # Idempotent re-run: no changes, no warnings.
        before = (auto_gen / "stock" / "host_stub.cpp").read_text()
        run_patch(auto_gen)
        assert (auto_gen / "stock" / "host_stub.cpp").read_text() == before, "not idempotent"

        # v1 upgrade path.
        v1_gen = make_stub_dir(base / "v1case", "v1", cache_block=V1_CACHE_BLOCK)
        run_patch(v1_gen)
        check_stub(v1_gen / "v1" / "host_stub.cpp")

        # v2 upgrade path: weak-function epoch call -> v3 data symbol.
        v2_gen = make_stub_dir(
            base / "v2case", "v2",
            cache_block=MOD.OVERFLOW_V2_CACHE_BLOCK,
            decls=MOD.OVERFLOW_V2_EPOCH_DECL,
        )
        out = run_patch(v2_gen)
        assert "overflow buffer cache(s)" in out.stdout, out.stdout
        check_stub(v2_gen / "v2" / "host_stub.cpp")

        # Unknown pattern (future CANN): warn, leave file untouched.
        unk = base / "unknown" / "auto_gen" / "lib"
        unk.mkdir(parents=True)
        (unk / "host_stub.cpp").write_text("int f() { return 0; }\n")
        out = run_patch(unk.parent)
        assert ("pattern not found" in out.stderr
                or "declaration not found" in out.stderr), out.stderr
        assert (unk / "host_stub.cpp").read_text() == "int f() { return 0; }\n"

    print("PASS: patch_host_stub stock/v1/v2/idempotent/unknown cases")


if __name__ == "__main__":
    main()
