#!/usr/bin/env python3
"""Copyright and header-guard checks for xlite C/C++ sources."""

import argparse
import re
from pathlib import Path

COPYRIGHT_PATTERN = re.compile(r"Copyright.*Huawei Technologies")
HEADER_GUARD_IFNDEF = re.compile(r"^#ifndef\s+(\S+)\s*$")


def check_copyright(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            first_lines = [f.readline().strip() for _ in range(6)]
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path}: failed to read file: {exc}"]

    if not any(COPYRIGHT_PATTERN.search(line) for line in first_lines):
        return [f"{path}: missing copyright header"]
    return []


def check_header_guard(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path}: failed to read file: {exc}"]

    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#pragma once"):
            return []
        match = HEADER_GUARD_IFNDEF.match(stripped)
        if match:
            guard_macro = match.group(1)
            define_pattern = re.compile(rf"#define\s+{re.escape(guard_macro)}\b")
            if any(define_pattern.match(line.strip()) for line in lines):
                return []
            return [f"{path}: missing matching #define for {guard_macro}"]

    return [f"{path}: missing header guard (#pragma once or #ifndef/#define)"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check C/C++ headers and copyright")
    parser.add_argument("--check-copyright", action="store_true")
    parser.add_argument("--check-header-guard", action="store_true")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    if not args.check_copyright and not args.check_header_guard:
        parser.error("at least one check must be enabled")

    errors: list[str] = []
    for raw in args.files:
        path = Path(raw)
        if args.check_copyright:
            errors.extend(check_copyright(path))
        if args.check_header_guard:
            errors.extend(check_header_guard(path))

    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
