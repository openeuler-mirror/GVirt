# Running xlite kernels on the CANN simulator (no NPU)

This guide shows how to build xlite and run kernel tests on the CANN
performance simulator (`msprof op simulator`, camodel) on a machine without an
Ascend NPU. It is intended for correctness experiments and single-kernel
profiling, not for full model or vLLM runs.

## Prerequisites

- CANN toolkit with simulator assets, e.g. 9.2.0-beta.1, including
  `<CANN>/tools/simulator/<SoC>/lib/libruntime_camodel.so`.
- `uv` (the script uses it to create the venv).
- A C++17 host compiler: `clang` / `clang++` (the script default) or
  `gcc` / `g++` (override with `CC` / `CXX`).
- Python 3.9-3.12. The script defaults to 3.11 because `torch-npu` wheels are
  available for it.

## Quick start

```bash
cd xlite

tests/run_sim.sh setup     # simulator shim + venv (torch CPU, torch-npu, ...)
tests/run_sim.sh build     # build kernels + runtime for $XLITE_SIM_SOC
tests/run_sim.sh run       # run tests/sim_smoke.py under the simulator
```

`run` accepts another test script:

```bash
tests/run_sim.sh run tests/kernels/add.py
tests/run_sim.sh run tests/kernels/muls.py
```

Keep test cases small: camodel is orders of magnitude slower than hardware.
`tests/kernels/muls.py` performs >100 launches and can take longer than the
usual timeout.

Existing kernel tests that create their inputs on the device
(`torch.randn(..., dtype=torch.float16/bfloat16, device="npu:0")`, as
`add.py` and `muls.py` do) go through the `torch-npu` aclnn op
`aclnnInplaceNormal` before any xlite kernel runs. Whether that op works on
camodel depends on the CANN and `torch-npu` versions: it runs on CANN
9.2.0-beta.1 with `torch-npu` 2.9.1, while on CANN 8.3.RC2 with `torch-npu`
2.9.0 it fails with error `561103 Parse dynamic kernel config fail` and the
xlite kernel is never executed. On such versions generate the data on the CPU
and move it to the device (`torch.randn(...).to("npu:0")`), as
`tests/sim_smoke.py` does.

## Running an already-built tree

`run` never builds: it imports the `xlite` package from the source tree
(`xlite/*.so`, `xlite/lib/*.so`) and only needs the venv and the shim. So if
xlite was built some other way — for example
`pip install --no-build-isolation -e .`, which stages the artifacts in place —
the simulator can use that build without going through `build`:

```bash
tests/run_sim.sh setup     # venv + shim; skips installation if the venv exists
tests/run_sim.sh run tests/sim_smoke.py
```

To reuse an existing Python environment instead of creating the venv, point
`XLITE_SIM_VENV` at it; `setup` then creates only the shim:

```bash
XLITE_SIM_VENV=/path/to/env tests/run_sim.sh setup
```

## What the script does

1. **Shim** (`setup`): CANN 9.2.0-beta.1 has a bug where `msprof op simulator`
   injects the hardware runtime, so `aclrtSetDevice` takes the TSD/HDC path and
   fails with `aclError:507033` on the stub driver. The script copies
   `libacl_rt_impl.so` into a shim dir and points `libruntime.so` next to it to
   `libruntime_camodel.so`. With the shim first on `LD_LIBRARY_PATH`, the
   injection resolves the simulator runtime and stays on the `no TsdOpen` path.
2. **Venv** (`setup`): creates `~/.cache/xlite-sim/venv`, installs CPU torch,
   `torch-npu`, cmake and build dependencies.
3. **Build** (`build`): configures CMake with `SOC_VERSION=$XLITE_SIM_SOC`,
   builds with clang in `$XLITE_SIM_BUILD` (default
   `~/.cache/xlite-sim/build`), then stages the built `.so` files into the
   source-tree `xlite/` package so it can be imported without installing a
   wheel. The build is out of tree, so it
   never touches the repository `build/` directory and can coexist with a
   regular NPU build in the same checkout. The build directory is wiped before
   each build by default: incremental AscendC kernel rebuilds are unreliable
   (`ld.lld: error: ... unknown file type`), and CMake caches the imported
   `Torch_DIR`. Set `XLITE_SIM_INCREMENTAL=1` to reuse an existing build
   directory.
4. **Run** (`run`): writes a wrapper that prepends the venv torch lib to
   `LD_LIBRARY_PATH`, sets `PYTHONPATH` to the repo and runs the test through
   `tests/sim_launch.py` (a `runpy` launcher, see below) with
   `python -u` (unbuffered), then invokes:

   ```bash
   LD_LIBRARY_PATH=<shim>:$LD_LIBRARY_PATH \
       msprof op simulator --soc-version=<SoC> --output=<out> <wrapper>
   ```

   Results are written to `OPPROF_*` under the output directory, including
   `simulator/core*.veccore*/trace.json` and instruction/cycle CSVs.

   The launcher runs the test and then `os._exit()`s from an `atexit` hook
   registered last (so it runs first). The `torch-npu`/camodel teardown path
   segfaults on the stub driver after the test has finished (`Child process
   killed by signal 11`); exiting before interpreter finalization keeps the
   `msprof` run clean and still propagates the test's exit status.

   `msprof` does not forward that status: a failing test only produces
   `[WARN] Child process exited with status 256` while `msprof` itself exits
   `0`. The generated wrapper therefore writes the test's real exit status to
   a file, and `run` returns it, so `tests/run_sim.sh run` exits non-zero when
   the test fails. A non-zero `msprof` exit status is returned as well when the
   test never reported one.

## Expected output

```text
avail True count 16
set_device ok
muls torch.float16: ok
muls torch.bfloat16: ok
smoke test: PASS
[INFO]  Profiling results saved in .../OPPROF_...
```

The first two lines (`avail True count 16`, `set_device ok`) are emitted by
`torch-npu`/camodel and depend on the CANN build; they may be missing on other
versions, which is not an error. The lines from `muls ...` onward are the
test's own output.

Some `torch-npu`/camodel paths need source-level stacks
(`Failed to parse pc to code ... recompile kernel with -g`); that only affects
the optional per-instruction call-stack annotation, not the timings.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `XLITE_SIM_SOC` | `Ascend910B3` | target SoC; must match a `tools/simulator/<SoC>` dir |
| `XLITE_SIM_WORK` | `~/.cache/xlite-sim` | work dir for shim, venv, build and output |
| `XLITE_SIM_VENV` | `$XLITE_SIM_WORK/venv` | Python venv |
| `XLITE_SIM_PYTHON` | `3.11` | interpreter for `uv venv` |
| `XLITE_SIM_TORCH` | `2.9.1` | torch version (xlite requires `<=2.10.0`) |
| `XLITE_SIM_TORCH_NPU` | `2.9.1` | must match the torch version |
| `XLITE_SIM_OUT` | `$XLITE_SIM_WORK/out` | `msprof` output dir |
| `XLITE_SIM_BUILD` | `$XLITE_SIM_WORK/build` | CMake build dir (out of tree) |
| `XLITE_SIM_JOBS` | `nproc` | parallel build jobs |
| `XLITE_SIM_CLEAN` | unset | force a clean build (already the default) |
| `XLITE_SIM_INCREMENTAL` | unset | set to `1` to reuse the build dir |
| `CANN_SET_ENV` | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | CANN env script |

## Troubleshooting

- `aclError:507033` / `TsdOpen failed` — the shim is missing or not first on
  `LD_LIBRARY_PATH`. Re-run `tests/run_sim.sh setup` and use `run`.
- `libruntime_camodel.so: cannot open shared object file` — the SoC does not
  match; check `$CANN/tools/simulator/<SoC>/lib`.
- `Failed to load the backend extension: torch_npu` — install `pyyaml` (the
  script does) or a `torch-npu` build matching torch.
- `ld.lld: error: ... unknown file type` during the kernel preprocess step —
  an incremental AscendC kernel rebuild; wipe the build dir, e.g.
  `rm -rf ~/.cache/xlite-sim/build` (or your `$XLITE_SIM_BUILD`). The script
  does this by default, so do not set `XLITE_SIM_INCREMENTAL=1` unless you
  need it.
- `run` exits non-zero although the profile was written — the test itself
  failed; look at its output above the `msprof` summary. Version of the script
  before this behaviour was added returned `msprof`'s status, which stays `0`
  even for a failing test.
- No `OPPROF_*` produced — the test failed before launching kernels; check the
  test stdout above the `msprof` summary.
