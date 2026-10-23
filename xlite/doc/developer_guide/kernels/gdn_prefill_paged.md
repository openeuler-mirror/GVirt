# Native paged GDN Prefill

`XliteOpConv1dAndSiLUPrefillPaged` and
`XliteOpRecurrentGatedDeltaRulePrefillPaged` process packed variable-length
queries directly against vLLM-owned state pages. They complement the existing
single-token paged Decode operators.

## Layout and metadata

- Conv state: `[pages, width - 1, channels]`, with contiguous inner dimensions.
- Recurrent state: `[pages, value_heads, value_dim, key_dim]`, with K contiguous.
- Each page may have a padded leading stride and a nonzero storage offset.
- Inputs and outputs are contiguous packed `[tokens, features]` matrices.
- Conv and SSM have independent `int32[batch]` page indices and initial-state
  flags. `int32[batch + 1]` cumulative query starts describe the token segments.
- A false initial-state flag resets that sequence's working state before its
  first token. True retains the existing page, including chunk continuations
  and Decode rows in a mixed batch.
- Page zero and negative indices produce zero outputs without changing state.
  An empty segment leaves its page untouched. Active requests must own distinct
  pages, as guaranteed by the upstream cache manager.

The Conv kernel assigns each sequence/channel tile to one vector core, retains
its sliding window in UB, and commits the final history once. Prefill rounds
the convolution result to activation dtype before SiLU, matching the current
native `conv1d` followed by `SiLU` implementation. Decode preserves its existing
fused rounding contract. The recurrent
kernel assigns sequence/head/V tiles, retains an FP32 state tile in UB across
all query tokens, and writes the final state to the same native page. Grouped
K heads are addressed directly; no expanded Q/K or transposed SSM allocation is
needed. BF16, FP16 and FP32 activations are supported, with activation-dtype or
FP32 SSM state; decay gates may be activation dtype or FP32. Model execution
uses FP32 decay gates and reads `A_log` and `dt_bias` with their actual dtypes.
The GDN output normalization remains FP32 until the SiLU gate product; the
shared gate implementation avoids a separate BF16 rounding and packed concat.

## Scope and performance

This is a recurrent Prefill implementation using the native VK layout. It does
not implement FLA's chunked WY/matrix-multiplication algorithm. Token recurrence
is sequential within each state tile, so removing state conversion does not
imply faster long-prompt Prefill. Compare measured latency before choosing a
production mode.

## Full-attention score precision

The standalone attention binding accepts `fp32_scores=True` for BF16/FP16
and at most 8192 KV tokens. QK accumulation stays FP32 through causal softmax;
probabilities round once to activation dtype for SV. Tiled Flash attention
cannot be enabled simultaneously. Model attention dispatch retains its existing
policy. Both variants use upstream K/V page strides and zero-buffer tail protection.

The FP32 score and activation-dtype probability workspaces are transient,
profiled by DummyRun, and reused across layers. At 24 Cube cores and 8192 KV
tokens their combined size is approximately 288 MiB. This is additional compute
scratch, not persistent KV state. Callers choose the existing attention paths for other configurations.

`tests/kernels/attention_fp32_scores.py` compares against CPU FP32 QK/softmax
with activation-dtype probability rounding, including ragged mixed queries,
continuation, shuffled physical pages, head dimension 256 and the 8192-token bound.
Model-level numerical alignment is a separate acceptance check; isolated kernel
correctness does not establish end-to-end Full-mode precision.

## Validation

`tests/kernels/gdn_prefill_paged.py` checks fresh state, continuation, ragged and
empty segments, mixed one-token/multi-token queries, null pages, grouped heads,
padded strides/storage offsets, long queries and Prefill-to-Decode handoff.
Compare against an FP32 CPU recurrence and verify unowned pages and guards
exactly. Use the NPU build environment described in `AGENTS.md`.
