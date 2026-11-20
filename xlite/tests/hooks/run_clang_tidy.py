#!/usr/bin/env python3
"""clang-tidy hook (CI / manual stage).

Requires a build tree with compile_commands.json; configures the project when
the tree is missing and always brings it current (kernel launch headers are
build-time artifacts). Kernel sources under csrc/kernels and csrc/ascend.h are
excluded (CCE intrinsics produce intentional warnings).

Analysis runs one clang-tidy process per file in parallel (XLITE_CI_TIDY_JOBS
workers, default min(16, cpus) — the pre-pre-commit checker's jobs=16): the
serial loop made a full pass cost the SUM of all per-file analyses instead of
the slowest few.
"""

import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

XLITE_ROOT = Path(__file__).resolve().parents[2]
SKIP_PATHS = ("csrc/kernels/", "csrc/ascend.h")
# The static analyzer (clang-analyzer-*) explores every registration lambda in
# the TORCH_LIBRARY table — 428 s on the current _C.cpp, 96% of the file's
# analysis cost and most of a full pass — while never having reported a
# finding there; these files keep every other check family.
ANALYZER_EXEMPT_FILES = ("_C.cpp",)
TIDY_FINDING = re.compile(r":\d+:\d+: (?:warning|error): ")


def ensure_build_tree() -> bool:
    build_dir = XLITE_ROOT / "build"
    if not (build_dir / "compile_commands.json").exists():
        # A first configure can fail transiently (compiler ABI try_compile flake)
        # and leave a partial tree; wipe and retry once before giving up.
        for attempt in (1, 2):
            configure = subprocess.run(
                ["cmake", "-S", str(XLITE_ROOT), "-B", str(build_dir), "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
                capture_output=True,
                text=True,
            )
            if configure.returncode == 0:
                break
            shutil.rmtree(build_dir, ignore_errors=True)
            if attempt == 2:
                print(f"cmake configure failed:\n{configure.stderr.strip()}")
                return False
            print("cmake configure failed transiently; retrying with a clean build tree ...")

    # Build even when the tree already exists: the kernel launch headers
    # op.cpp includes are build-time artifacts, so a tree predating a kernel
    # change parses stale stubs and tidy dies on aclrtlaunch_* signature
    # errors. cmake no-ops here when everything is current.
    # Numbered -j: a bare -j leaks unbounded MAKEFLAGS into the kernel sub-makes (see setup.py).
    jobs = os.environ.get("MAX_JOBS") or os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", jobs or str(min(os.cpu_count() or 4, 64))],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        print(f"cmake build failed:\n{build.stderr.strip()[-4000:]}")
        return False

    return (build_dir / "compile_commands.json").exists()


def main() -> int:
    def should_check(raw: str) -> bool:
        path = Path(raw).resolve()
        return path.is_relative_to(XLITE_ROOT) and not any(s in path.as_posix() for s in SKIP_PATHS)

    files = sorted(Path(raw).resolve() for raw in sys.argv[1:] if should_check(raw))

    if not files:
        return 0
    if not ensure_build_tree():
        return 1

    def analyze_one(path: Path) -> tuple[bool, str]:
        rel = path.relative_to(XLITE_ROOT)
        cmd = ["clang-tidy", rel.as_posix(), f"--config-file={XLITE_ROOT / '.clang-tidy'}", "-p=build"]
        if path.name in ANALYZER_EXEMPT_FILES:
            cmd.append("--checks=-clang-analyzer-*")
        result = subprocess.run(
            cmd,
            cwd=XLITE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        hits = [line for line in output.splitlines() if TIDY_FINDING.search(line)]
        if result.returncode != 0:
            text = f"clang-tidy failed for {rel} (exit {result.returncode}):\n" + (
                "\n".join(hits) if hits else output.strip()[-2000:]
            )
        elif hits:
            text = f"clang-tidy issues in {rel}:\n" + "\n".join(hits)
        else:
            text = ""
        return result.returncode == 0 and not hits, text

    jobs = int(os.environ.get("XLITE_CI_TIDY_JOBS") or min(16, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        # map keeps the output in input (sorted) order: deterministic logs.
        results = list(executor.map(analyze_one, files))
    failed = [text for clean, text in results if not clean]
    if failed:
        for text in failed:
            print(text)
        print("Run `clang-tidy -p=build --fix <file>` under xlite/ to apply suggested fixes.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
