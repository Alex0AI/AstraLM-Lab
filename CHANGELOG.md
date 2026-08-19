# Changelog

All notable changes are documented here. AstraLM follows semantic versioning
for its Python package and report schema.

## 0.2.0 - 2026-08-20

### Added

- Equal-parameter, multi-seed Attention Bridge ablation runner
- JSON, Markdown, and standalone HTML experiment reports
- Paired-seed deltas, win counts, runtime provenance, and an honest results ledger
- Warmup/cosine training, validation, atomic checkpoints, and resume
- Deterministic dependency-free byte-level BPE
- Static parameter and KV-cache estimator
- QK normalization and configurable logit soft-capping
- Interactive browser-only AstraScope studio
- Pico and small presets, cloud ablation workflow, and Colab tour

### Changed

- Diagnostics now expose bridge gates while standard mode retains matched
  parameter allocation.
- Checkpoints remain loadable when training uses `torch.compile`.

## 0.1.0 - 2026-08-19

- Initial decoder-only Transformer, GQA, RoPE, RMSNorm, SwiGLU, KV cache,
  generation, diagnostics, training example, and correctness tests.
