---
name: data-levels-and-transfer
description: Ascend 910B (2201) data levels and the pipes that move data between them — GM/L2/L1/L0A/L0B/L0C/UB/BiasTable/Fixpipe sizes, and which MTE/FixPipe/API moves each path. Use when deciding which buffer holds what, how to move it, and whether a path exists (L0C→GM→UB vs L0C→L1).
---

# Data levels and data transfer

Targets **Ascend 910B (NpuArch 2201)**; `kernel_macro.h` only reserves a 3510 UB-layout branch — there are no 3510 kernels.

## Storage levels

| Level | Size (2201) | Buffer tag | What it's for |
|---|---|---|---|
| GM | device HBM | `__gm__` (`GM_ADDR`) | kernel I/O; also Cube↔Vector staging |
| L2 | — (off-core) | — | caches GM traffic; hit rate shaped by swizzle, not sized/hinted directly in xlite |
| L1 | 512 KB | `TPosition::A1/B1/C1` | AIC staging GM→L0; bias/deqScale (C1). **Cube-only** |
| L0A / L0B | 64 KB each | `TPosition::A2/B2` | Cube left/right operands |
| L0C | 128 KB (16 × 8 KB) | `TPosition::CO1` | Mmad accumulate (`4*m0*n0` bytes, fp32) |
| UB | 192 KB (48 × 4 KB) | `__ubuf__` | Vector/Scalar data + scratch; 32 B blocks |
| BiasTable (C2) | 1 KB, 64 B block | `TPosition::C2` | matmul bias |
| Fixpipe buffer | 2 KB, 128 B block | `TPosition::C2PIPE2GM` | per-N dequant scale |

Sizes from `kernel_macro.h` (`ASCEND_L0_BYTES`, `C2_DATABLOCK`, `FIXPIPE_DATABLOCK`), `kernel_param.h` (`UB_SIZE`, `GM_UB_BURST_SIZE`), and the CANN memory-structure docs (L1 512 KB, L0A/L0B 64 KB, L0C 128 KB on 2201).

## Movement units

| Unit | Paths | API |
|---|---|---|
| MTE2 | GM→L1 (ND→NZ), GM→UB | `DataCopy`+`Nd2NzParams`/`DataCopyParams`; `CopyGmToL1Nd2Nz`/`CopyGmToL1`; `copy_gm_to_ubuf*` |
| MTE1 | L1→L0A/L0B, L1→BiasTable | `LoadData` (`CopyToL0ACol`/`CopyToL0BCol`); `LoadDataWithTranspose` (`CopyToL0BTCol`, int8/int4 B^T); `DataCopy(l0BiasBuf, l1BiasBuf, …)` (64B C2 blocks) |
| MTE3 | UB→GM | `copy_ubuf_to_gm*` |
| FixPipe | L0C→GM (NZ→ND + dequant), L0C→L1, L1→Fixpipe buffer | `CopyToGm`/`CopyToGmWithDequant`; `CopyL0CToL1`; `DataCopy(fixpipeBuf, …)` |
| DMI | UB→UB | `copy_ubuf_to_ubuf` |

The hardware/CANN on 2201 also has **GM→L0A/L0B** (MTE2, `LoadData` from GM) and **L1→GM** (MTE3) paths, but xlite never uses them: it always stages Cube operands **GM→L1→L0A/L0B** and writes results **L0C→GM** (FixPipe) — never GM→L0 directly, never L1→GM. (On 3510 both paths are removed anyway.)

## Data paths that matter

- Vector kernel: `GM → UB → Vector → UB → GM`.
- Cube kernel: `GM → L1 → L0A/L0B → L0C → GM`.
- Cube→Vector (fused): **no direct L0C→UB on 2201** — L0C writes a GM workspace (`CopyToGm`), then `GM→UB`; ordered by `cross-core-sync`.
- Cube→Cube: skip GM via `L0C → L1 → L0B` (`CopyL0CToL1` + `CopyToL0BCol`, `indexer_scores.h`).

## Transfer sizing

- All MTE copies are in **32 B blocks** (`BLOCK_SIZE`); `count*sizeof(T) % 32 == 0`.
- `CopyGmToUbufAligned`/`CopyUbufToGmAligned` pick `copy_*` / `*_align_b16` / `*_align_b8` by alignment.
- ~16 KB per transfer reaches peak bandwidth (`GM_UB_BURST_SIZE`); see `transfer-larger-blocks`.
