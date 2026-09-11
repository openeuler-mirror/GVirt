#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Patch CANN's generated host_stub.cpp files to expose the launch timestamp.

The stubs pass StartAscendProf's startTime to ReportAscendProf, so it is the
exact stamp of the node record msprof keys the launch on. Inject a weak hook
XLiteMsprofShapeHook(name, startTime) right after StartAscendProf so xlite's
shape reporter can stamp its tensor records with the same timestamp and merge
into that node's descriptor. Inert when libxlite does not define the hook.

Idempotent: files already containing the hook are left untouched.
"""

import re
import sys
from pathlib import Path

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


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if 'XLiteMsprofShapeHook' in text:
        return False

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

    path.write_text(text, encoding="utf-8")
    print(f"patched {path}: {count} launch wrapper(s)")
    return True


def main(argv):
    if len(argv) < 1:
        raise SystemExit("usage: patch_host_stub.py <auto_gen_dir> [<auto_gen_dir> ...]")
    for arg in argv[1:]:
        for stub in sorted(Path(arg).glob('*/host_stub.cpp')):
            patch_file(stub)


if __name__ == '__main__':
    main(sys.argv)
