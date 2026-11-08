# xlite Static Checker Documentation

## Overview

All xlite lint checks run through [pre-commit](https://pre-commit.com/), driven by a single
configuration file at `xlite/.pre-commit-config.yaml` (next to the other lint configs).
The same hooks run locally (on every commit) and in CI, so a PR that passes locally passes
the bot. A symlink at the repo root (`.pre-commit-config.yaml` → `xlite/.pre-commit-config.yaml`)
keeps the standard pre-commit entry points working from any checkout; edit the file under
`xlite/`.

## Hook inventory

| Hook | Scope | Stage |
| --- | --- | --- |
| `trailing-whitespace`, `end-of-file-fixer`, `mixed-line-ending` | all files | pre-commit |
| `check-merge-conflict`, `check-yaml`, `check-json`, `check-toml`, `check-added-large-files`, `detect-private-key`, `check-ast`, `debug-statements` | all files | pre-commit |
| `ruff-check` (`--fix`) + `ruff-format` | Python | pre-commit |
| `clang-format` | `xlite/csrc/**` C/C++ | pre-commit |
| `markdownlint` (`--fix`) | Markdown (`xlite/.markdownlint.yaml`) | pre-commit |
| `codespell` | all text files (`xlite/pyproject.toml` `[tool.codespell]`) | pre-commit |
| `check-cpp-copyright`, `check-cpp-header-guard` | `xlite/csrc/**` | pre-commit |
| `check-torch-npu-imports` | `xlite/tests/**` (enforces `import torch` before `import torch_npu`) | pre-commit |
| `mypy` | `xlite/xlite/**` package | manual (CI) |
| `clang-tidy` | `xlite/csrc/**` (skips `csrc/kernels/` and `csrc/ascend.h`; needs a build tree) | manual (CI) |
| `signoff-commit` | appends `Signed-off-by` when missing | prepare-commit-msg |

Stages: `pre-commit` hooks run on every local commit; `manual` hooks run only in CI
(`python xlite/tests/run_static_checks.py` or `pre-commit run --all-files --hook-stage manual`).

## Usage

One-time setup (from any directory in the repo):

```bash
pre-commit install
```

After that, hooks run automatically on `git commit`. Hooks that modify files
(`ruff-check`, `ruff-format`, `markdownlint`) fail the commit and leave the fixes
in the working tree — `git add` the touched files and commit again.

Full pre-PR verification (CI parity, includes mypy and clang-tidy):

```bash
python xlite/tests/run_static_checks.py            # or: pre-commit run --all-files --hook-stage manual
python xlite/tests/run_static_checks.py --cpp-only
python xlite/tests/run_static_checks.py --python-only
```

Run specific hooks on specific files:

```bash
pre-commit run ruff-check --files xlite/tests/pr_ci_bot.py
```

## Fixing findings

- Python lint/format: most findings are fixed by `pre-commit run ruff-check ruff-format --files <files>`; the rest need manual edits.
- C++ format: `clang-format -i <file>` under `xlite/` (or `git clang-format` for staged changes).
- clang-tidy: `clang-tidy -p=build --fix <file>` under `xlite/` (requires a build tree).
- Markdown: `pre-commit run markdownlint --files <file>` auto-fixes spacing/structure issues.
- Spelling: `codespell -i 3 <file>` interactive fixes; project vocabulary lives in `[tool.codespell]` of `xlite/pyproject.toml`.

The checking hooks for clang-format and clang-tidy run from pre-commit-managed environments
(no local install needed). For the manual fix commands above, install the binaries once with
`pip install clang-format==22.1.8 clang-tidy==22.1.8` — the versions must match the
`mirrors-clang-format` rev and the `clang-tidy` hook's `additional_dependencies` in
`.pre-commit-config.yaml`.

## Configuration files

- `xlite/.pre-commit-config.yaml` — hook sources, versions, scopes, stages; owns the
  clang-format and clang-tidy versions. A symlink at the repo root points to it so the
  standard pre-commit commands (and editor integrations) keep working.
- `xlite/pyproject.toml` — `[tool.ruff]` rules, `[tool.codespell]` vocabulary and skips, `[tool.mypy]` settings.
- `xlite/.markdownlint.yaml` — markdownlint rule overrides.
- `xlite/.clang-format`, `xlite/.clang-tidy` — C++ style and analysis configuration.

Line-length policy: Python enforces 120 columns (test scripts under `xlite/tests/` are exempt
via `per-file-ignores`); C++ stays at the `.clang-format` limit of 100 columns.

## CI

`tests/pr_ci_bot.py` polls open GitCode PRs, runs `tests/run_static_checks.py` plus the
vllm-ascend integration mypy check in an isolated worktree per PR, runs the accuracy tests
when the static checks pass, and posts one concise comment per commit with per-check statuses
and failure output only.
