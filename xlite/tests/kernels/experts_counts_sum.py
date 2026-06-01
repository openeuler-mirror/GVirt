#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
# Test for the experts_counts_sum kernel, which computes two reductions over
# a per-DP-rank per-expert token count matrix:
#
#   1. tokens_per_epgroup[dp_idx, ep_id]:
#      Total token count from DP rank ``dp_idx`` destined for experts in
#      EP group ``ep_id``.
#
#   2. experts_counts_output[expert]:
#      Total token count for expert ``expert``, summed across all DP ranks.
#
# The input ``experts_counts_input`` has shape [ep_size, n_routed_experts]
# where each row corresponds to one DP rank and each column to one expert.
# ===============================================================================

import logging
import torch
import numpy as np
from xlite._C import Runtime, experts_counts_sum

logging.getLogger().setLevel(logging.INFO)

# Single-card test: device 0, 500 MB tensor pool.
rt = Runtime(0, 500)
torch.npu.set_device(0)

for n_routed_experts in [160, 256]:
    for ep_size in [8, 16]:
        if n_routed_experts % ep_size != 0:
            continue

        # Number of experts handled by each EP group.
        experts_ep_group = n_routed_experts // ep_size

        # Input: per-DP-rank per-expert token counts.
        # Shape: [ep_size, n_routed_experts].
        experts_counts_input = torch.randint(
            0, 20, (ep_size, n_routed_experts,), dtype=torch.int32, device="npu:0"
        )

        # Output buffer 1: per-DP-rank per-EP-group token sums.
        # Shape: [ep_size, ep_size].
        tokens_per_epgroup = torch.ones(
            ep_size, ep_size, dtype=torch.int32, device="npu:0"
        )

        # Output buffer 2: per-expert total token counts (summed over all DP ranks).
        # Shape: [n_routed_experts].
        experts_counts_output = torch.zeros(
            n_routed_experts, dtype=torch.int32, device="npu:0"
        )

        torch.npu.synchronize()

        experts_counts_sum(
            rt,
            experts_counts_input,
            tokens_per_epgroup,
            experts_counts_output,
            n_routed_experts
        )
        torch.npu.synchronize()

        # ---- CPU reference computation ----

        c = experts_counts_input.cpu().numpy().reshape(ep_size, n_routed_experts)

        # Reference 1: sum per-DP-rank counts within each EP group.
        ref_tokens_per_epgroup = np.zeros((ep_size, ep_size), dtype=np.int32)
        for dp_idx in range(ep_size):
            for curr_expert in range(n_routed_experts):
                ep_id = curr_expert // experts_ep_group
                ref_tokens_per_epgroup[dp_idx, ep_id] += c[dp_idx, curr_expert]

        # Reference 2: sum per-expert counts across all DP ranks.
        ref_experts_counts = np.zeros(n_routed_experts, dtype=np.int32)
        for curr_expert in range(n_routed_experts):
            for dp_idx in range(ep_size):
                ref_experts_counts[curr_expert] += c[dp_idx, curr_expert]

        ref_tokens_per_epgroup_t = torch.from_numpy(ref_tokens_per_epgroup)
        ref_experts_counts_t = torch.from_numpy(ref_experts_counts)

        logging.info(
            f"n_routed_experts={n_routed_experts} ep_size={ep_size} "
            f"experts_counts_sum executed!"
        )

        experts_counts_output_cpu = experts_counts_output.cpu()
        tokens_per_epgroup_cpu = tokens_per_epgroup.cpu()
        torch.npu.synchronize()

        # Validate tokens_per_epgroup output.
        try:
            torch.testing.assert_close(
                tokens_per_epgroup_cpu, ref_tokens_per_epgroup_t, atol=0, rtol=0
            )
        except AssertionError as e:
            logging.error(f"[tokens_per_epgroup] {e}")
            logging.error(f"  ref:  {ref_tokens_per_epgroup_t}")
            logging.error(f"  got:  {tokens_per_epgroup_cpu}")

        # Validate experts_counts_output.
        try:
            torch.testing.assert_close(
                experts_counts_output_cpu, ref_experts_counts_t, atol=0, rtol=0
            )
        except AssertionError as e:
            logging.error(f"[experts_counts_output] {e}")
            logging.error(f"  ref:  {ref_experts_counts_t}")
            logging.error(f"  got:  {experts_counts_output_cpu}")
