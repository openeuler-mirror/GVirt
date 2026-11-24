# xlite Static Checker Documentation

## Overview

All xlite lint checks run through [pre-commit](https://pre-commit.com/), driven by a single
configuration file at `xlite/.pre-commit-config.yaml` (next to the other lint configs; covered
by the CI harness version digest via `.gitcode/ci.toml`).
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
| `markdownlint` (`--fix`) | Markdown (`xlite/.markdownlint.yaml`) | manual (CI) |
| `codespell` | all text files (`xlite/pyproject.toml` `[tool.codespell]`) | pre-commit |
| `check-cpp-copyright`, `check-cpp-header-guard` | `xlite/csrc/**` | pre-commit |
| `check-torch-npu-imports` | `xlite/tests/**` (enforces `import torch` before `import torch_npu`) | pre-commit |
| `mypy` | `xlite/xlite/**` package | manual (CI) |
| `clang-tidy` | `xlite/csrc/**` (skips `csrc/kernels/` and `csrc/ascend.h`; needs a build tree) | manual (CI) |
| `binding-contract` | binding-layer probes (`tests/funcs/binding_contract.py`) against the most local built xlite — the checked tree's build if present, the installed copy otherwise; skipped when no module is importable (the CI device phase probes the freshly built PR package; the probe itself does no device work) | manual (CI) |
| `check-definition-drift` | check-definition files, incl. any nested discovery-based config (no-op in one's own checkout) | pre-commit + manual |
| `signoff-commit` | appends `Signed-off-by` when missing | prepare-commit-msg |

Stages: `pre-commit` hooks run on every local commit; `manual` hooks run only in CI
(`python .gitcode/run_static_checks.py` or `pre-commit run --all-files --hook-stage manual`).

## Usage

One-time setup (from any directory in the repo). `--install-hooks` also builds every hook environment up front — the first commit then needs no network, and env-build failures (e.g. behind a proxy) surface here with a clear message instead of stalling the first commit. On node-less machines use plain `pre-commit install`: environments build lazily per stage, and the CI-only hooks (markdownlint) never build locally at all:

```bash
pre-commit install --install-hooks   # node-less machines: pre-commit install
```

After that, hooks run automatically on `git commit`. Hooks that modify files
(`ruff-check`, `ruff-format`) fail the commit and leave the fixes
in the working tree — `git add` the touched files and commit again.

Fast whole-tree check of the commit-time set (no CANN build tree, no mypy, no clang-tidy, no node):

```bash
pre-commit run --all-files                     # commit-stage hooks over the whole tree
```

Full pre-PR verification (CI parity, includes mypy and clang-tidy). Runs from the repository root or from `xlite/` (whose `.gitcode` symlinks to `../.gitcode`):

```bash
python .gitcode/run_static_checks.py            # or: pre-commit run --all-files --hook-stage manual
python .gitcode/run_static_checks.py --cpp-only
python .gitcode/run_static_checks.py --python-only
```

Run specific hooks on specific files:

```bash
pre-commit run ruff-check --files .gitcode/run_ci_checks.py
```

## Fixing findings

- Python lint/format: most findings are fixed by `pre-commit run ruff-check ruff-format --files <files>`; the rest need manual edits.
- C++ format: `clang-format -i <file>` under `xlite/` (or `git clang-format` for staged changes).
- clang-tidy: `clang-tidy -p=build --fix <file>` under `xlite/` (requires a build tree).
- Markdown: `pre-commit run markdownlint --hook-stage manual --files <file>` auto-fixes spacing/structure issues (needs node ≥18 locally — the hook is CI-only).
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

.gitcode/pr_ci_bot.py polls open GitCode PRs and checks each in an isolated worktree with the
CI harness from `.gitcode/` (the master copy by default; a PR's own copy only when its author
is a `.gitcode` owner or an owner commented `/ci`). The harness runs the static checks, the
vllm-ascend integration mypy check, and — when those pass — the accuracy tests, posting one
concise comment per commit with per-check statuses and failure output only.
