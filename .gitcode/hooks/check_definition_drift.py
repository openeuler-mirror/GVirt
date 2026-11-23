#!/usr/bin/env python3
"""Fail when the checked tree's check definitions differ from this hook's
own tree.

Rule files that hooks read via per-file discovery (clang-format, ruff) or
relative paths resolve against the checked worktree, so the CI harness
runs this hook from the trusted .gitcode copy: a non-owner PR cannot
relax any check definition without failing here. In a developer's own
checkout both trees are identical and the hook is a no-op. pyproject.toml
is compared per tool section, so packaging edits do not trip it.

Discovery cannot be listed exhaustively — a tool reads rule files nearest-up
from each checked file, and any path is plantable — so beyond the frozen
top-level files, every discovery-based config file anywhere in the checked
tree must match a byte-identical file at the same path in this hook's own
tree: a planted nested config has no trusted counterpart and fails here.
"""

from pathlib import Path

CHECKED_ROOT = Path.cwd()  # pre-commit runs hooks with the checked tree as cwd
OWN_ROOT = Path(__file__).resolve().parents[2]

BYTE_FILES = (
    "xlite/.pre-commit-config.yaml",
    "xlite/.clang-format",
    "xlite/.clang-tidy",
    "xlite/.markdownlint.yaml",
    # absent from the trusted tree, so any planted copy is drift too
    "xlite/.ruff.toml",
    "xlite/ruff.toml",
)
PYPROJECT_SECTIONS = ("tool.ruff", "tool.mypy", "tool.codespell")
# Names the discovery-based tools read nearest-up from each checked file:
# clang-format (also its alternate _clang-format name), ruff, and
# codespell's per-directory exclusion file (not discovered by the pinned
# 2.4.3, but by newer releases — frozen so a bump cannot open the hole).
# Everything else runs with an explicit config argument.
DISCOVERY_CONFIG_NAMES = (
    ".clang-format",
    "_clang-format",
    ".ruff.toml",
    "ruff.toml",
    "pyproject.toml",
    "codespellignore",
)
FROZEN_PATHS = frozenset(BYTE_FILES) | {"xlite/pyproject.toml"}


def read_bytes(root: Path, rel: str) -> bytes | None:
    path = root / rel
    return path.read_bytes() if path.exists() else None


def pyproject_section(root: Path, section: str) -> str:
    """The textual slice of one [tool.*] section: a header collects while
    its bracket-stripped key equals the wanted table or extends it with a
    dot — nested subtables ([tool.ruff.lint]) and array-of-tables forms
    ([[tool.mypy.overrides]]) included, since those hold the real rules and
    must not escape the comparison. Siblings sharing the prefix
    ([tool.mypy_pack], [tool.mypy-overrides]) are not collected."""
    path = root / "xlite" / "pyproject.toml"
    if not path.exists():
        return "<missing>"
    lines, inside, collected = path.read_text().splitlines(), False, []
    for line in lines:
        if line.startswith("["):
            s = line.lstrip("[")
            inside = s == f"{section}]" or s.startswith(f"{section}.")
        if inside:
            collected.append(line)
    return "\n".join(collected)


def nested_definition_drift(checked_root: Path, own_root: Path) -> list[str]:
    """Discovery-based config files planted outside the frozen top-level
    set: each must have a byte-identical same-path counterpart in the own
    tree, so a PR can neither relax a check with a nested config nor add
    one without an owner-approved harness change."""
    drift = []
    for path in sorted(checked_root.rglob("*")):
        if path.name not in DISCOVERY_CONFIG_NAMES or not path.is_file():
            continue
        rel = path.relative_to(checked_root).as_posix()
        if rel in FROZEN_PATHS:
            continue
        if read_bytes(own_root, rel) != path.read_bytes():
            drift.append(rel)
    return drift


def definition_drift(checked_root: Path, own_root: Path) -> list[str]:
    drift = []
    for rel in BYTE_FILES:
        if read_bytes(checked_root, rel) != read_bytes(own_root, rel):
            drift.append(rel)
    for section in PYPROJECT_SECTIONS:
        if pyproject_section(checked_root, section) != pyproject_section(
            own_root, section
        ):
            drift.append(f"xlite/pyproject.toml [{section}]")
    drift.extend(nested_definition_drift(checked_root, own_root))
    return drift


def main() -> int:
    drift = definition_drift(CHECKED_ROOT, OWN_ROOT)
    if not drift:
        return 0
    print("check definitions differ from this hook's own tree:")
    for rel in drift:
        print(f"  {rel}")
    print(
        "Definitions are frozen to the trusted copy: rebase onto latest master, "
        "or ask a `.gitcode` owner to comment `/ci` to re-check with this PR's harness."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
