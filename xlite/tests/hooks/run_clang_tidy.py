#!/usr/bin/env python3
"""clang-tidy hook (CI / manual stage).

Requires a build tree with compile_commands.json; configures and builds the
project first when the tree is missing. Kernel sources under csrc/kernels and
csrc/ascend.h are excluded (CCE intrinsics produce intentional warnings).
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

XLITE_ROOT = Path(__file__).resolve().parents[2]
SKIP_PATHS = ("csrc/kernels/", "csrc/ascend.h")
TIDY_FINDING = re.compile(r":\d+:\d+: (?:warning|error): ")


def ensure_build_tree() -> bool:
    build_dir = XLITE_ROOT / "build"
    if (build_dir / "compile_commands.json").exists():
        return True

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

    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "-j"],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        print(f"cmake build failed:\n{build.stderr.strip()}")
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

    failed = False
    for path in sorted(files):
        rel = path.relative_to(XLITE_ROOT)
        result = subprocess.run(
            ["clang-tidy", rel.as_posix(), f"--config-file={XLITE_ROOT / '.clang-tidy'}", "-p=build"],
            cwd=XLITE_ROOT,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        findings = [line for line in output.splitlines() if TIDY_FINDING.search(line)]
        if result.returncode != 0:
            failed = True
            print(f"clang-tidy failed for {rel} (exit {result.returncode}):")
            print("\n".join(findings) if findings else output.strip()[-2000:])
        elif findings:
            failed = True
            print(f"clang-tidy issues in {rel}:")
            print("\n".join(findings))

    if failed:
        print("Run `clang-tidy -p=build --fix <file>` under xlite/ to apply suggested fixes.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
