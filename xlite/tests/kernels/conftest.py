#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Shared fixtures for the pytest kernel suite.

Migration into the suite is opt-in by filename: pytest collects only
``test_*.py``, so the legacy loop-based scripts are never imported (their
module-level code would run kernels at collection time) and keep working via
``python tests/kernels/<script>.py``. A file joins the default run (the
"stable gate") only after it passes repeatedly on the CI container; see
doc/developer_guide/contributing.md for the migration recipe and the
flaky/broken/multi_npu opt-in markers.

Device selection follows ASCEND_RT_VISIBLE_DEVICES: fixtures use device 0,
the first *visible* NPU.
"""

import os
import zlib

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.fixture(scope="session", autouse=True)
def npu_session():
    """Session-wide torch_npu setup shared by every collected test.

    ``allow_internal_format`` matches the vllm-ascend model runner, which
    sets it process-wide in production: with it off, ``npu_format_cast(y, 29)``
    writes a different physical NZ layout than the one xlite NZ kernels
    consume (the contract documented in test_matmul.py).
    """
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    yield


@pytest.fixture(scope="session", autouse=True)
def default_device(npu_session):
    """Default tensor device for the suite.

    Factories without an explicit ``device=`` allocate on the first visible
    NPU, so tests omit the ``device="npu:0"`` boilerplate. Explicit
    ``device=`` arguments (e.g. ``device="cpu"`` reference tensors) override.
    """
    with torch.device("npu:0"):
        yield


@pytest.fixture(scope="session")
def runtime(npu_session):
    """Session-wide xlite Runtime for single-NPU kernel tests.

    Tests exercising Runtime lifecycle itself (runtime_recycle.py) create
    their own Runtime instead of using this fixture. Raise the pool size for
    a whole run via XLITE_TEST_RUNTIME_MB if a migrated test needs more
    device memory.
    """
    from xlite._C import Runtime

    return Runtime(0, size=int(os.environ.get("XLITE_TEST_RUNTIME_MB", "512")))


@pytest.fixture(autouse=True)
def reproducible_rng(request):
    """Seed the RNG per test case so any failure reproduces on rerun.

    Legacy scripts call ``torch.randn`` unseeded, so tolerance near-ties (see
    ATOL_RTOL in test_matmul.py) fire at random between runs. The seed derives
    from the test id via CRC32, which is stable across processes unlike
    hash(). Tests needing specific data reseed inside the test body afterwards.
    """
    torch.manual_seed(zlib.crc32(request.node.nodeid.encode()))
