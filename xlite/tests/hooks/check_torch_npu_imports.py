#!/usr/bin/env python3
"""Screen test scripts for proper torch_npu injection.

`torch.npu` only exists after `import torch_npu`, so any test using it must
have `import torch` followed by `import torch_npu`.
"""

import argparse
import ast
from pathlib import Path


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return []  # syntax errors are reported by the syntax/lint hooks

    uses_npu = False
    torch_line = None
    torch_npu_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" and torch_line is None:
                    torch_line = node.lineno
                if alias.name.split(".")[0] == "torch_npu" and torch_npu_line is None:
                    torch_npu_line = node.lineno
                    uses_npu = True
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "torch_npu":
            if torch_npu_line is None:
                torch_npu_line = node.lineno
            uses_npu = True
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "torch" and node.attr == "npu":
                uses_npu = True

    if not uses_npu:
        return []
    if torch_npu_line is None:
        return [f"{path}: uses torch.npu/torch_npu but never imports torch_npu"]
    if torch_line is None:
        return [f"{path}:{torch_npu_line}: imports torch_npu without `import torch`"]
    if torch_npu_line < torch_line:
        return [f"{path}: `import torch_npu` (line {torch_npu_line}) must follow `import torch` (line {torch_line})"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Check torch/torch_npu import order in test scripts")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    errors: list[str] = []
    for raw in args.files:
        errors.extend(check_file(Path(raw)))

    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
