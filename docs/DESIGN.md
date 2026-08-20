# Design notes

## Computation path

Each decoder block is pre-normalized: `RMSNorm -> GQA -> residual`, followed by
`RMSNorm -> SwiGLU -> residual`. Rotary position embeddings are applied to query
and key vectors before cached keys are concatenated. Optional QK normalization
and logit soft-capping make the small-model training path easier to probe under
aggressive learning rates.

## Attention Bridge

The bridge carries the preceding block's attention update into the next
block. Before mixing, its per-token RMS is matched to the current attention
update. A learned sigmoid gate controls reuse. This differs from simply adding
all prior attention tensors: it is local, bounded, observable, and inexpensive.

Both `standard` and `bridge` modes allocate the same gate parameters. Standard
mode bypasses those values, so comparisons have exactly matched parameter
counts instead of quietly giving the experimental arm extra capacity.

The `inspect` command reports residual RMS and bridge gates. Set
`residual_mode: standard` for a controlled ablation with identical surrounding
architecture.

The mechanism is an educational research hypothesis, not a claim of state of
the art. The ablation runner repeats matched trials across seeds and reports
mean, spread, token budget, and raw artifacts so results remain falsifiable.

## Cache contract

The cache stores unexpanded grouped keys and values. GQA expansion happens only
for the attention computation, reducing cache memory by `n_heads / n_kv_heads`.
The test suite verifies token-by-token cached logits against a full causal pass.
