#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Unit tests for the per-collective payload arithmetic.

``compute_collective_msg_bytes`` is the pure-function form of the TP/DP/EP payload
derivation used to size each HCCL communicator. It takes plain scalars, so the formulas can
be pinned here without loading a model or touching a device.
"""

from xlite._C import compute_collective_msg_bytes

MB = 1 << 20


def reference(tokens, hidden, n_experts, topk, ep_size, dp_size, elem):
    """Independent restatement of the formulas, kept deliberately literal."""
    tp = tokens * hidden * elem
    ep = tp * topk if (topk > 0 and ep_size > 1) else 0
    dp = 0
    if dp_size > 1:
        weights = tokens * n_experts * elem
        routing = tokens * n_experts // 8  # BIT1: one bit per expert
        dp = (tp + weights + routing) * dp_size
    return [tp, dp, ep]


def check(name, tokens, hidden, n_experts, topk, ep_size, dp_size, elem):
    got = list(compute_collective_msg_bytes(tokens, hidden, n_experts, topk, ep_size, dp_size, elem))
    want = reference(tokens, hidden, n_experts, topk, ep_size, dp_size, elem)
    assert got == want, f"{name}: got {got}, want {want}"
    return got


def test_dense_model_only_has_tp():
    """A dense model under TP has no EP dispatch and no DP all-gather."""
    assert check("dense tp8", 32, 4096, 0, 0, 1, 1, 2) == [32 * 4096 * 2, 0, 0]


def test_moe_ep_scales_by_topk():
    """EP dispatch sends each token to topK experts, so ep is a multiple of tp."""
    tp, dp, ep = check("moe ep2 topk8", 32, 2048, 128, 8, 2, 1, 2)
    assert dp == 0
    assert ep == tp * 8


def test_ep_is_zero_without_cross_rank_experts():
    """topK alone is not enough: expert parallelism must actually span ranks."""
    assert check("ep size 1", 32, 2048, 128, 8, 1, 1, 2)[2] == 0
    assert check("dense topk 0", 32, 2048, 128, 0, 2, 1, 2)[2] == 0


def test_dp_packs_three_terms():
    """dp is activations + gate weights + BIT1 routing mask, replicated per DP rank."""
    tp, dp, ep = check("dp2 with experts", 32, 2048, 128, 8, 1, 2, 2)
    weights = 32 * 128 * 2
    routing = 32 * 128 // 8
    assert ep == 0
    assert dp == (tp + weights + routing) * 2


def test_routing_is_one_bit_per_expert():
    """Guards the BIT1 term: sizing the mask like an activation would be 16x too large."""
    tp, dp, _ = check("routing size", 32, 2048, 128, 8, 1, 2, 2)
    mask_as_bits = (dp // 2) - tp - 32 * 128 * 2
    assert mask_as_bits == 32 * 128 // 8
    assert mask_as_bits != 32 * 128 * 2


def test_moe_with_dp_and_ep():
    """All three communicators active at once."""
    tp, dp, ep = check("ep2 dp2", 64, 2048, 32, 4, 2, 2, 2)
    assert ep == tp * 4
    assert dp == (tp + 64 * 32 * 2 + 64 * 32 // 8) * 2


def test_element_size_follows_dtype():
    """fp32 activations double every payload that carries hidden states."""
    bf16 = check("bf16", 16, 1024, 64, 8, 2, 2, 2)
    fp32 = check("fp32", 16, 1024, 64, 8, 2, 2, 4)
    assert bf16[0] * 2 == fp32[0]


def test_negative_inputs_are_treated_as_zero():
    """The binding clamps negatives rather than wrapping into huge unsigned values."""
    assert list(compute_collective_msg_bytes(-1, -1, -1, -1, -1, -1, -1)) == [0, 0, 0]


def test_matches_measured_e2e_values():
    """Values observed in the per-comm HCCL buffer log lines on real hardware."""
    # Qwen3-8B tp8 decodeonly: max_num_seqs=32 rounded up to 8 -> 32
    assert check("qwen3-8b tp8", 32, 4096, 0, 0, 1, 1, 2) == [262144, 0, 0]
    # Qwen3-30B-A3B tp4 + dp2 + EP: hidden 2048, 128 experts, topK 8
    assert check("qwen3-30b-a3b", 32, 2048, 128, 8, 2, 2, 2) == [131072, 279552, 1048576]
    # GLM-5.2 16-card full mode: 8192 tokens, hidden 6144, topK 8
    assert check("glm-5.2 full tp", 8192, 6144, 0, 0, 1, 1, 2) == [100663296, 0, 0]
    assert check("glm-5.2 full ep", 8192, 6144, 160, 8, 16, 1, 2)[2] == 805306368


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} collective payload tests passed!")


if __name__ == "__main__":
    main()
