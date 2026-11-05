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

Synthesizes CANN-style host_stub.cpp files (stock, stubs carrying the v1 /
v2 / v3 cache blocks), runs the patch script on them, and asserts the
transformed text: the overflow-status buffer is removed (argument slot set
to null, no device alloc/free left, no orphaned constexpr, no leftover
epoch declaration or cache block), and re-runs are idempotent. Also checks
the XLITE_NO_OVERFLOW_PATCH probe (file left stock) and the unknown-pattern
warning path.
"""

import importlib.util
import os
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


def run_patch(auto_gen, env=None):
    run_env = None
    if env:
        run_env = os.environ.copy()
        run_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(auto_gen)],
        check=True, capture_output=True, text=True, env=run_env,
    )


def check_stub(path):
    text = path.read_text()
    assert "__ascendc_args.__ascendc_overflow = nullptr" in text, \
        "null assignment missing"
    assert "overflow-status buffer removed" in text, "removal marker missing"
    assert "AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow)" not in text, "stock alloc left"
    assert "FreeAscendMemDevice(__ascendc_args.__ascendc_overflow)" not in text, "stock free left"
    assert "__ascendc_overflow_status_size = 8" not in text, "orphaned constexpr left"
    assert "__ascendc_overflow_cache = []" not in text, "v1 block left"
    assert "__ascendc_overflow_cache" not in text, "cache block left"
    assert "gXliteOverflowEpoch" not in text, "v3 epoch decl left"
    assert "xliteOverflowEpoch" not in text, "v2 epoch decl left"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Stock -> removal.
        auto_gen = make_stub_dir(base, "stock", cache_block=None)
        out = run_patch(auto_gen)
        assert "removed 1 overflow buffer alloc(s)" in out.stdout, out.stdout
        check_stub(auto_gen / "stock" / "host_stub.cpp")

        # Idempotent re-run: no changes, no warnings.
        before = (auto_gen / "stock" / "host_stub.cpp").read_text()
        run_patch(auto_gen)
        assert (auto_gen / "stock" / "host_stub.cpp").read_text() == before, "not idempotent"

        # A/B probe: stock file left untouched.
        probe_gen = make_stub_dir(base / "probe", "probe", cache_block=None)
        before = (probe_gen / "probe" / "host_stub.cpp").read_text()
        run_patch(probe_gen, env={"XLITE_NO_OVERFLOW_PATCH": "1"})
        assert (probe_gen / "probe" / "host_stub.cpp").read_text() == before, \
            "probe modified the file"
        check_stub_probe = (probe_gen / "probe" / "host_stub.cpp").read_text()
        assert "AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow)" \
            in check_stub_probe, "probe should keep stock alloc"

        # v1 upgrade path.
        v1_gen = make_stub_dir(base / "v1case", "v1", cache_block=V1_CACHE_BLOCK)
        run_patch(v1_gen)
        check_stub(v1_gen / "v1" / "host_stub.cpp")

        # v2 upgrade path: weak-function epoch block -> removal.
        v2_gen = make_stub_dir(
            base / "v2case", "v2",
            cache_block=MOD.OVERFLOW_V2_CACHE_BLOCK,
            decls=MOD.OVERFLOW_V2_EPOCH_DECL,
        )
        out = run_patch(v2_gen)
        assert "removed 1 overflow buffer alloc(s)" in out.stdout, out.stdout
        check_stub(v2_gen / "v2" / "host_stub.cpp")

        # v3 upgrade path: weak-data-symbol epoch block -> removal.
        v3_gen = make_stub_dir(
            base / "v3case", "v3",
            cache_block=MOD.OVERFLOW_CACHE_BLOCK,
            decls=MOD.OVERFLOW_EPOCH_DECL,
        )
        run_patch(v3_gen)
        check_stub(v3_gen / "v3" / "host_stub.cpp")

        # Unknown pattern (future CANN): warn, leave file untouched.
        unk = base / "unknown" / "auto_gen" / "lib"
        unk.mkdir(parents=True)
        (unk / "host_stub.cpp").write_text("int f() { return 0; }\n")
        out = run_patch(unk.parent)
        assert "pattern not found" in out.stderr, out.stderr
        assert (unk / "host_stub.cpp").read_text() == "int f() { return 0; }\n"

    print("PASS: patch_host_stub stock/v1/v2/v3/probe/idempotent/unknown cases")


if __name__ == "__main__":
    main()
