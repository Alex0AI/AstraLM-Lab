# AstraLM Lab

An inspectable decoder-only Transformer built for experiments that should fit on
one machine and survive close technical review. AstraLM implements the important
parts directly in PyTorch: grouped-query causal attention, RoPE, RMSNorm,
SwiGLU, AdamW training, incremental KV caching, and autoregressive sampling.

Its original feature is **Attention Bridge**: a learned, RMS-balanced connection
between adjacent layers' attention updates. It can be ablated with one config
field and inspected per layer, turning a residual-stream idea into a measurable
experiment rather than a hidden architectural tweak.

## Why this project is different

- **Correctness first:** cached and full-sequence decoding are compared at the
  logit level; causality, RoPE norm preservation, and diagnostics are tested.
- **Modern small-model stack:** GQA reduces KV memory, RoPE supplies positions,
  and tied embeddings keep the parameter count honest.
- **Offline by default:** the byte tokenizer and sample corpus need no model hub.
- **Public-data friendly:** an optional streaming helper reuses TinyStories or
  FineWeb-Edu without downloading a full dataset to a laptop.
- **Ablation ready:** switch between `standard` and `attention_bridge` while
  leaving the rest of the architecture unchanged.
- **Evidence over claims:** `benchmark` reports cached/uncached throughput on the
  current hardware; `inspect` emits layer-level JSON diagnostics.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

astralm train --steps 200 --batch-size 8 --seq-len 96
astralm generate checkpoints/demo/model.pt "Small models" --tokens 80
astralm benchmark checkpoints/demo/model.pt
astralm inspect checkpoints/demo/model.pt "Follow the residual stream"
pytest
```

On a low-power laptop, skip the local install entirely: open **Actions → Cloud
TinyStories Experiment → Run workflow**. The runner streams a bounded public-data
sample, trains, benchmarks cached decoding, and uploads only JSON results.

## Architecture

```text
bytes -> embedding -> [ RMSNorm -> GQA + RoPE -> Attention Bridge -> residual
                     -> RMSNorm -> SwiGLU -----------------------> residual ] x N
      -> RMSNorm -> tied language-model head

incremental decode: grouped K/V -> layer cache -> repeat to query heads -> SDPA
```

The default tiny configuration has 6 query heads but only 2 KV heads, so its KV
cache is one third the size of standard multi-head attention at the same width.
See [design notes](docs/DESIGN.md) for the bridge and cache invariants and
[public data recipes](docs/DATA.md) for low-disk streaming experiments.

## Experiment recipe

Train two runs with the same seed and data, changing only `residual_mode` between
`standard` and `attention_bridge`. Compare validation loss, gradient norm, block
RMS, bridge gates, and cached decoding throughput. Metrics and train settings are
saved beside each checkpoint for reproducibility.

## Scope

This is an educational research implementation, not a distributed pretraining
framework. It deliberately favors readable contracts and targeted experiments
over kernel fusion and multi-node complexity.

## License

MIT
