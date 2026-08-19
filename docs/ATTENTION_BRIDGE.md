# Attention Bridge

Attention Bridge is an experimental cross-layer residual path. Block `l` may
reuse block `l-1`'s attention update after matching its per-token RMS to the
current update:

```text
g = sigmoid(gate_logit)
previous_balanced = previous * RMS(current) / RMS(previous)
bridged = sqrt(1 - g²) * current + g * previous_balanced
```

The square-root coefficient limits scale growth while preserving a differentiable
gate. The bridge is local—only the preceding attention update is reused—so state
does not grow with depth.

## Falsifiable hypothesis

At a matched parameter count and token budget, a bounded cross-layer attention
path may improve optimization stability or validation loss in small decoder-only
models. It may also do nothing, or hurt. AstraLM treats all three outcomes as
valid experimental results.

## Matched baseline

Every block allocates its bridge gate in both `standard` and `attention_bridge`
modes. Standard mode bypasses the operation. Parameter counts therefore match
exactly; `astralm ablate` asserts this before writing a report.

## Measurements

- validation loss across at least three seeds;
- gradient norm and elapsed time;
- per-layer residual RMS and learned gate values;
- cached decoding throughput and maximum cache logit error.

Do not generalize a pico-scale result to large language models without evidence.

