#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# Shared helpers for building xlite AttnMetaV2 metadata.
# Imports only numpy + torch: AttnMetaV2 (xlite._C) is only available under the
# xlite backend, so the caller passes it in.
import numpy as np
import torch


def prepare_xlite_attnmeta_v2(
    AttnMetaV2,
    tokens: torch.Tensor,
    start_pos: int,
    max_seq_len: int,
    block_sizes,
    gdn_state_pages: torch.Tensor | None = None,
):
    """Build AttnMetaV2 with device-tensor query_start_loc / slot_mapping /
    block_tables. The C++ side skips host computation and H2D copies.

    Uniform batch (lens==seqlen, cached_lens==start_pos): query_start_loc ==
    arange(batch+1)*seqlen (0-prefixed prefix sum), and the contiguous block
    table makes the slot gather collapse to i*step*block_size + pos.

    One block_tables / slot_mapping entry per block_sizes element, matching
    C++'s per-kv-cache slotMapping/blockTables vectors. Hybrid models pass
    gdn_state_pages ([batch] int32, nonzero) which is appended as
    block_tables[1] = mamba state indices [batch, 1] (block id == state slot).
    """
    batch = tokens.size(0)
    seqlen = tokens.size(1)

    meta = AttnMetaV2()
    meta.lens_cpu = [seqlen] * batch
    meta.cached_lens_cpu = [start_pos] * batch
    meta.lens = torch.full((batch,), seqlen, dtype=torch.int32) \
        .to(tokens.device, non_blocking=True)
    meta.cached_lens = torch.full((batch,), start_pos, dtype=torch.int32) \
        .to(tokens.device, non_blocking=True)

    # 0-prefixed cumsum of lens for a uniform batch == arange(batch+1)*seqlen.
    meta.query_start_loc = (torch.arange(batch + 1, dtype=torch.int32) * seqlen) \
        .to(tokens.device, non_blocking=True)

    meta.positions = torch.arange(start_pos, start_pos + seqlen, dtype=torch.int64) \
        .repeat(batch).to(tokens.device, non_blocking=True)

    batch_indices = np.arange(batch, dtype=np.int32).reshape(-1, 1)
    positions_per_sample = np.arange(start_pos, start_pos + seqlen, dtype=np.int32)

    # block_tables/slot_mapping are pybind def_readwrite vectors: assignment
    # copies the list in, but later append() is NOT reflected back to C++.
    # Build locally, assign once at the end.
    block_tables = []
    slot_mapping = []
    for bs in block_sizes:
        step = (max_seq_len + bs - 1) // bs
        block_num = (seqlen + start_pos + bs - 1) // bs

        # Persistent-width block table [batch, step]: only the first block_num
        # columns per row are valid (block_tables[i, k] = i*step + k); the trail
        # stays stale garbage (in-bounds id 0, points at real data of row 0) that
        # must never be followed.
        block_tables_2d = np.zeros((batch, step), dtype=np.int32)
        block_tables_2d[:, :block_num] = batch_indices * step + np.arange(block_num, dtype=np.int32)
        block_tables.append(
            torch.from_numpy(block_tables_2d).to(tokens.device, non_blocking=True)
        )

        # Contiguous table => slot gather collapses to i*step*bs + pos.
        slot_stride = np.int32(step * bs)
        slot_mapping_2d = batch_indices * slot_stride + positions_per_sample  # (batch, seqlen)
        slot_mapping.append(
            torch.from_numpy(slot_mapping_2d.reshape(-1)).to(tokens.device, non_blocking=True)
        )

    # Hybrid family-slot contract: block_tables[1] = mamba state indices
    # [batch, 1] int32 (block id == state slot, page 0 reserved). The C++
    # side derives the per-layer paged GDN operands from this, cachedLens
    # and queryStartLoc — no separate gdn_* metadata fields anymore.
    # NOTE: pybind def_readwrite assignment copies the list in — the table
    # list must be complete BEFORE assigning meta.block_tables.
    if gdn_state_pages is not None:
        block_tables.append(
            gdn_state_pages.to(torch.int32).to(tokens.device, non_blocking=True).reshape(-1, 1)
        )
    meta.block_tables = block_tables
    meta.slot_mapping = slot_mapping
    return meta
