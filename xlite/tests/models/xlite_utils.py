#!/usr/bin/python3
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
    compress_ratios=None,
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

    compress_ratios: the PER-LAYER CXA compress ratio list (DeepSeek-V4:
    44 entries incl. the trailing MTP phantom 0). When given, the CXA planes
    are overridden (state planes 0/3 -> per-token abs pos; indexer_k plane 1
    -> ratio==4 per-block table; compress_kv plane 2 -> per-block slots per
    distinct ratio, byte-identical to what the C++ runtime used to derive
    from the same identity table) and per-layer rows are attached — C++ CXA
    consumers read their own row (LayerPlane), the flats stay the V1/Dummy/
    Check/Verify surface. The compress_kv pool is sized to the generic stride
    in init_xlite_kvcache so no ratio overflows. None/empty => no CXA.
    """
    batch = tokens.size(0)
    seqlen = tokens.size(1)

    meta = AttnMetaV2()
    meta.lens_cpu = [seqlen] * batch
    meta.cached_lens_cpu = [start_pos] * batch
    meta.lens = torch.full((batch,), seqlen, dtype=torch.int32).to(tokens.device, non_blocking=True)
    meta.cached_lens = torch.full((batch,), start_pos, dtype=torch.int32).to(tokens.device, non_blocking=True)

    # 0-prefixed cumsum of lens for a uniform batch == arange(batch+1)*seqlen.
    meta.query_start_loc = (torch.arange(batch + 1, dtype=torch.int32) * seqlen).to(tokens.device, non_blocking=True)

    meta.positions = (
        torch.arange(start_pos, start_pos + seqlen, dtype=torch.int64)
        .repeat(batch)
        .to(tokens.device, non_blocking=True)
    )

    batch_indices = np.arange(batch, dtype=np.int32).reshape(-1, 1)
    positions_per_sample = np.arange(start_pos, start_pos + seqlen, dtype=np.int32)

    # Pybind def_readwrite vectors: build locally, assign once at the end.
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
        block_tables.append(torch.from_numpy(block_tables_2d).to(tokens.device, non_blocking=True))

        slot_stride = np.int32(step * bs)
        slot_mapping_2d = batch_indices * slot_stride + positions_per_sample
        slot_mapping.append(torch.from_numpy(slot_mapping_2d.reshape(-1)).to(tokens.device, non_blocking=True))

    # Hybrid family-slot contract: block_tables[1] = mamba state indices
    # [batch, 1] int32 (block id == state slot, page 0 reserved). The C++
    # side derives the per-layer paged GDN operands from this, cachedLens
    # and queryStartLoc — no separate gdn_* metadata fields anymore.
    # NOTE: pybind def_readwrite assignment copies the list in — the table
    # list must be complete BEFORE assigning meta.block_tables.
    if gdn_state_pages is not None:
        block_tables.append(gdn_state_pages.to(torch.int32).to(tokens.device, non_blocking=True).reshape(-1, 1))
    meta.block_tables = block_tables
    meta.slot_mapping = slot_mapping

    # CXA (DeepSeek-V4): per-plane overrides + per-layer rows. compress_ratios
    # is the PER-LAYER ratio list. Uniform identity pools make every layer's
    # row able to reference the same tensors; only the per-block planes'
    # content/length depends on the layer's ratio. Plane order matches the C++
    # CXA enum: INDEXER_STATE=0, INDEXER_K=1, COMPRESS_KV=2, STATE=3, SWA_KV=4.
    if compress_ratios and 4 in compress_ratios and len(slot_mapping) >= 5:
        comp_bs = block_sizes[2]
        empty = torch.zeros(0, dtype=torch.int32, device=tokens.device)

        # State planes (0, 3): per-token absolute position (write path derives
        # each token's state slot from its absolute position).
        state_pos = np.arange(start_pos, start_pos + seqlen, dtype=np.int32)
        state_slot = torch.from_numpy(np.tile(state_pos, (batch, 1)).reshape(-1)).to(tokens.device, non_blocking=True)
        slot_mapping[0] = state_slot
        slot_mapping[3] = state_slot

        # Indexer-K (plane 1, ratio==4 layers only): per-block identity table.
        # slot = bt1[b][row//bs]*bs + row%bs == b*comp_rows_stride + row under
        # the identity layout (comp_rows_stride = comp_blocks_per_req*bs).
        r = 4
        idx_bs = block_sizes[1]
        rows_per_req = max_seq_len // r
        comp_rows_stride = ((rows_per_req + idx_bs - 1) // idx_bs) * idx_bs
        comp_blocks_per_req = comp_rows_stride // idx_bs
        n_blocks_per_req = ((start_pos + seqlen) // r) - (start_pos // r)

        # Tile (b, j) to match the kernel's totalBlockIdx stride.
        j_arange = np.arange(n_blocks_per_req, dtype=np.int64)
        slot_per_req = start_pos // r + j_arange.astype(np.int32)
        slot_2d = batch_indices * np.int32(comp_rows_stride) + slot_per_req
        slot_mapping[1] = torch.from_numpy(slot_2d.reshape(-1).astype(np.int32)).to(tokens.device, non_blocking=True)

        bt_k = np.arange(comp_blocks_per_req, dtype=np.int32)
        bt_2d = batch_indices * np.int32(comp_blocks_per_req) + bt_k
        block_tables[1] = torch.from_numpy(bt_2d.astype(np.int32)).to(tokens.device, non_blocking=True)

        # Compress_kv plane (2): per-block write slots per distinct ratio —
        # byte-identical to the slots the C++ runtime used to derive from the
        # same generic identity block_tables[2] (stride ceil(max_seq_len/
        # comp_bs)): slot = bt2[b][row//comp_bs]*comp_bs + row%comp_bs
        #        == b*stride2*comp_bs + row. Its numel must equal the
        # C++-derived positions count (XliteOpCompressor checks).
        stride2 = (max_seq_len + comp_bs - 1) // comp_bs
        comp_sm = {}
        for ratio in sorted({x for x in compress_ratios if x > 0}):
            n_blocks = ((start_pos + seqlen) // ratio) - (start_pos // ratio)
            if n_blocks > 0:
                row = np.arange(start_pos // ratio, start_pos // ratio + n_blocks, dtype=np.int32)
                sm_2d = batch_indices * np.int32(stride2 * comp_bs) + row
                comp_sm[ratio] = torch.from_numpy(sm_2d.reshape(-1).astype(np.int32)).to(
                    tokens.device, non_blocking=True
                )
            else:
                comp_sm[ratio] = empty

        # Per-layer rows: ratio==0 layers keep only the swa plane (planes they
        # never consume stay _empty); ratio>0 layers share their ratio's row
        # (all pools identically laid out, so one row of references serves the
        # whole group). The trailing MTP phantom layer (ratio==0, beyond
        # n_layers) gets an all-_empty row C++ never consumes.
        bt_rows, sm_rows = [], []
        for ratio in compress_ratios:
            if ratio == 0:
                bt_rows.append([empty, empty, empty, empty, block_tables[4]])
                sm_rows.append([empty, empty, empty, empty, slot_mapping[4]])
            else:
                has_indexer = ratio == 4
                bt_rows.append(
                    [
                        block_tables[0] if has_indexer else empty,
                        block_tables[1] if has_indexer else empty,
                        block_tables[2],
                        block_tables[3],
                        block_tables[4],
                    ]
                )
                sm_rows.append(
                    [
                        state_slot if has_indexer else empty,
                        slot_mapping[1] if has_indexer else empty,
                        comp_sm[ratio],
                        state_slot,
                        slot_mapping[4],
                    ]
                )

        # Re-assign whole lists (pybind: index assign does not propagate).
        meta.slot_mapping = slot_mapping
        meta.block_tables = block_tables
        meta.block_tables_per_layer = bt_rows
        meta.slot_mapping_per_layer = sm_rows

    return meta
