# xlite development guidelines

The core of the project is `csrc/`, containing the C++ model/runtime code and kernel implementations under `csrc/kernels/`.

`csrc/kernels/kernel_macro.h` contains shared kernel macros, wrappers, constants, and utilities. `csrc/kernels/cce_stub.h` contains lower-level CCE stub interfaces. Consult existing implementations before adding new kernel infrastructure.

## Development requirements

- Work primarily under `<repo_root>/xlite` unless specified otherwise.
- Do not preserve obsolete internal APIs, paths, fallbacks, or compatibility layers unless explicitly required. Preserve documented public APIs/ABIs unless a breaking change is intentional.
- Prefer existing kernels, utilities, abstractions, and dependencies over reimplementing equivalent functionality.
- Keep changes narrowly scoped. Do not modify unrelated code or discard pre-existing user changes.

## Build environment

- Use a container with the appropriate CANN installation for NPU development. If on host, check if an existing container is specified and available before proceeding in the host environment.
- Use `pip install --force-reinstall --no-deps --no-build-isolation -v -e .` to install the package in editable mode (if failed due to missing dependencies, run `pip install -r requirements-build.txt --extra-index-url https://download.pytorch.org/whl/cpu`).
- For a git worktree (no `.git` at repo root) or clean CMake build, use `cmake -B build && cmake --build build -j && cmake --install build` for isolated builds.

## Testing

- `tests/kernels/` contains kernel correctness tests. Update relevant tests when modifying or adding kernels.
- `tests/funcs/`, `tests/models/`, and `tests/e2e/` contain functional, model, and end-to-end tests.
- `tests/performance/` contains performance tooling.
- TODO: `tests/perf/` corresponds to `tests/kernels/` for `msprof` related performance tests. Ignore for now unless specified otherwise.
- Test the narrowest relevant target first, then expand coverage as appropriate.
- Do not claim tests or benchmarks passed unless they were actually run.
- For performance changes, measure before/after performance rather than relying on code inspection.

## Static checks

Run before submitting a PR: `python tests/run_static_checks.py`. Use `--cpp-only` or `--python-only` for targeted checks when appropriate.

## Documentation

- `doc/` contains project documentation; kernel documentation is under `doc/kernels/`.
- Update documentation when behavior, interfaces, configuration, or supported functionality changes.
- Prefer function names and descriptions over exact source line numbers.
- Comments should explain non-obvious logic, invariants, hardware assumptions, or synchronization requirements. Avoid comments that merely restate the code.

## Commit/PR conventions

PR title format: `xlite: <type>: <short description>`. Typical types include `feat`, `bugfix`, `refactor`, `test`, `doc`, `perf`, and `ci`.
