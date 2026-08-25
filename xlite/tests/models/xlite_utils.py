#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# Shared helpers for building xlite AttnMetaV2 metadata.
#
# This module deliberately imports only numpy + torch at the top level:
# AttnMetaV2 (and the xlite._C package) is only available when the xlite
# backend is active, and is therefore passed in by the caller.
import numpy as np
import torch


def prepare_xlite_attnmeta_v2(
    AttnMetaV2,
    tokens: torch.Tensor,
    start_pos: int,
    max_seq_len: int,
    block_size: int,
):
    """Build AttnMetaV2 with device-tensor query_start_loc / slot_mapping /
    block_tables. The C++ side skips host computation and H2D copies.

    All samples share lens==seqlen / cached_lens==start_pos (uniform batch),
    so query_start_loc == arange(batch)*seqlen and the contiguous block table
    lets the slot gather collapse to i*step*block_size + pos.
    """
    batch = tokens.size(0)
    seqlen = tokens.size(1)
    step = (max_seq_len + block_size - 1) // block_size
    block_num = (seqlen + start_pos + block_size - 1) // block_size

    meta = AttnMetaV2()
    meta.lens_cpu = [seqlen] * batch
    meta.cached_lens_cpu = [start_pos] * batch
    meta.lens = torch.full((batch,), seqlen, dtype=torch.int32) \
        .to(tokens.device, non_blocking=True)
    meta.cached_lens = torch.full((batch,), start_pos, dtype=torch.int32) \
        .to(tokens.device, non_blocking=True)

    # Uniform batch: cumsum([seqlen]*batch) == arange(batch)*seqlen.
    meta.query_start_loc = (torch.arange(batch, dtype=torch.int32) * seqlen) \
        .to(tokens.device, non_blocking=True)

    meta.positions = torch.arange(start_pos, start_pos + seqlen, dtype=torch.int64) \
        .repeat(batch).to(tokens.device, non_blocking=True)

    # Contiguous block table: block_tables[i, k] = i*step + k.
    batch_indices = np.arange(batch, dtype=np.int32).reshape(-1, 1)
    block_indices = np.arange(block_num, dtype=np.int32)
    block_tables_2d = batch_indices * step + block_indices  # (batch, block_num)
    meta.block_tables = [torch.from_numpy(block_tables_2d) \
        .to(tokens.device, non_blocking=True)]

    # Contiguous table => slot gather collapses:
    # block_tables[i,pos//bs]*bs + pos%bs == i*step*bs + pos.
    positions_per_sample = np.arange(start_pos, start_pos + seqlen, dtype=np.int32)
    slot_stride = np.int32(step * block_size)
    slot_mapping_2d = batch_indices * slot_stride + positions_per_sample  # (batch, seqlen)
    meta.slot_mapping = [torch.from_numpy(slot_mapping_2d.reshape(-1)) \
        .to(tokens.device, non_blocking=True)]

    return meta
