#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Unit tests for the per-communicator HCCL buffer sizing policy.

The policy is pure arithmetic (payload bytes + rank count -> buffer MB), so these tests
need no device: they exercise the helper the runtime uses when it creates the TP / DP / EP
communicators. Run with pytest or as a script.
"""

from xlite._C import hccl_buffer_mb

MB = 1 << 20


def test_unknown_payload_configures_nothing():
    """A caller that reports no payload keeps the upstream behaviour: nothing is configured."""
    assert hccl_buffer_mb(0, 8) == 0
    assert hccl_buffer_mb(0, 1) == 0


def test_tiny_payload_floors_at_one_mb():
    """HCCL rejects a zero-sized buffer, so anything below 1 MB rounds up to 1 MB."""
    assert hccl_buffer_mb(1024, 8) == 1  # 1 KB over 8 ranks
    assert hccl_buffer_mb(MB, 8) == 1  # 1 MB over 8 ranks -> 128 KB * 2 -> 1 MB
    assert hccl_buffer_mb(MB, 1) == 2  # single rank: 1 MB * 2 directions -> 2 MB


def test_rounding_is_per_rank_and_bidirectional():
    """needMb = ceil(2 * msgBytes / ranks / 1MB); both directions are counted."""
    assert hccl_buffer_mb(8 * MB, 8) == 2  # 1 MB per rank, doubled
    assert hccl_buffer_mb(16 * MB, 8) == 4  # 2 MB per rank, doubled
    assert hccl_buffer_mb(12 * MB, 8) == 3  # 1.5 MB per rank -> rounds up to 3


def test_more_ranks_shrink_the_buffer():
    """The same total payload is split across more ranks, so the buffer must shrink."""
    assert hccl_buffer_mb(16 * MB, 4) == 8
    assert hccl_buffer_mb(16 * MB, 8) == 4
    assert hccl_buffer_mb(16 * MB, 16) == 2


def test_large_payload_is_honoured():
    """Sizes are never clamped: a future model may legitimately need more than 256 MB."""
    assert hccl_buffer_mb(1 << 30, 8) == 256  # 128 MB/rank * 2 -> 256 MB
    assert hccl_buffer_mb(1 << 31, 8) == 512  # 256 MB/rank * 2 -> 512 MB, NOT clamped
    assert hccl_buffer_mb(1 << 34, 8) == 4096  # 2 GB/rank * 2 -> 4 GB, still honoured


def test_zero_ranks_is_treated_as_one():
    """A rank count of 0 must not divide by zero; it behaves like a single rank."""
    assert hccl_buffer_mb(MB, 0) == 2


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} HCCL buffer sizing tests passed!")


if __name__ == "__main__":
    main()
