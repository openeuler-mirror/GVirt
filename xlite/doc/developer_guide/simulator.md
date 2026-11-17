# Running xlite kernels on the CANN simulator (no NPU)

This guide shows how to build xlite and run kernel tests on the CANN
performance simulator (`msprof op simulator`, camodel) on a machine without an
Ascend NPU. It is intended for correctness experiments and single-kernel
profiling, not for full model or vLLM runs.

## Prerequisites

- CANN toolkit with simulator assets, e.g. 9.2.0-beta.1, including
  `<CANN>/tools/simulator/<SoC>/lib/libruntime_camodel.so`.
- `uv` (the script uses it to create the venv).
- A host compiler that supports `__fp16` on x86_64: `clang` / `clang++`
  (gcc does not; xlite uses `__fp16` in `csrc/base.cpp`).
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

## What the script does

1. **Shim** (`setup`): CANN 9.2.0-beta.1 has a bug where `msprof op simulator`
   injects the hardware runtime, so `aclrtSetDevice` takes the TSD/HDC path and
   fails with `aclError:507033` on the stub driver. The script copies
   `libacl_rt_impl.so` into a shim dir and points `libruntime.so` next to it to
   `libruntime_camodel.so`. With the shim first on `LD_LIBRARY_PATH`, the
   injection resolves the simulator runtime and stays on the `no TsdOpen` path.
2. **Venv** (`setup`): creates `~/.cache/xlite-sim/venv`, installs CPU torch,
   `torch-npu`, cmake, pybind11 and build dependencies.
3. **Build** (`build`): configures CMake with `SOC_VERSION=$XLITE_SIM_SOC`,
   builds with clang, then stages the built `.so` files into the source-tree
   `xlite/` package so it can be imported without installing a wheel. The
   `build/` directory is wiped before each build by default: incremental
   AscendC kernel rebuilds are unreliable (`ld.lld: error: ... unknown file
   type`), and CMake caches `Torch_DIR`/`pybind11_DIR`. Set
   `XLITE_SIM_INCREMENTAL=1` to reuse an existing `build/`.
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

## Expected output

```text
avail True count 16
set_device ok
muls torch.float16: ok
muls torch.bfloat16: ok
smoke test: PASS
[INFO]  Profiling results saved in .../OPPROF_...
```

Some `torch-npu`/camodel paths need source-level stacks
(`Failed to parse pc to code ... recompile kernel with -g`); that only affects
the optional per-instruction call-stack annotation, not the timings.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `XLITE_SIM_SOC` | `Ascend910B3` | target SoC; must match a `tools/simulator/<SoC>` dir |
| `XLITE_SIM_WORK` | `~/.cache/xlite-sim` | work dir for shim, venv and output |
| `XLITE_SIM_VENV` | `$XLITE_SIM_WORK/venv` | Python venv |
| `XLITE_SIM_PYTHON` | `3.11` | interpreter for `uv venv` |
| `XLITE_SIM_TORCH` | `2.9.1` | torch version (xlite requires `<=2.10.0`) |
| `XLITE_SIM_TORCH_NPU` | `2.9.1` | must match the torch version |
| `XLITE_SIM_OUT` | `$XLITE_SIM_WORK/out` | `msprof` output dir |
| `XLITE_SIM_CLEAN` | unset | force a clean build (already the default) |
| `XLITE_SIM_INCREMENTAL` | unset | set to `1` to reuse `build/` |
| `CANN_SET_ENV` | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | CANN env script |

## Troubleshooting

- `aclError:507033` / `TsdOpen failed` — the shim is missing or not first on
  `LD_LIBRARY_PATH`. Re-run `tests/run_sim.sh setup` and use `run`.
- `libruntime_camodel.so: cannot open shared object file` — the SoC does not
  match; check `$CANN/tools/simulator/<SoC>/lib`.
- `Failed to load the backend extension: torch_npu` — install `pyyaml` (the
  script does) or a `torch-npu` build matching torch.
- `__fp16 was not declared` — use clang, not gcc, and clean the CMake cache
  (`XLITE_SIM_CLEAN=1 tests/run_sim.sh build`).
- `ld.lld: error: ... unknown file type` during the kernel preprocess step —
  an incremental AscendC kernel rebuild; wipe `build/` (the script does this by
  default, so do not set `XLITE_SIM_INCREMENTAL=1` unless you need it).
- No `OPPROF_*` produced — the test failed before launching kernels; check the
  test stdout above the `msprof` summary.
