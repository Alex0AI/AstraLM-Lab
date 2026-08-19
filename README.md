<div align="center">

![AstraLM architecture hero](assets/astralm-hero.png)

# AstraLM Lab

### Build a modern decoder-only Transformer. Prove it works. Run a falsifiable architecture experiment.

[![CI](https://github.com/Alex0AI/AstraLM-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Alex0AI/AstraLM-Lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/Alex0AI/AstraLM-Lab)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Alex0AI/AstraLM-Lab?style=social)](https://github.com/Alex0AI/AstraLM-Lab/stargazers)

[Live AstraScope](https://alex0ai.github.io/AstraLM-Lab/) · [Open in Colab](https://colab.research.google.com/github/Alex0AI/AstraLM-Lab/blob/main/notebooks/quickstart.ipynb) · [First result](docs/RESULTS.md) · [Experiments](docs/EXPERIMENTS.md) · [中文](README.zh-CN.md)

</div>

AstraLM is a compact research and teaching lab for the parts of language models
that are easy to implement incorrectly and hard to explain honestly. It includes
a modern PyTorch decoder, a verified incremental KV cache, a dependency-free BPE,
a reproducible training engine, and matched multi-seed ablations that generate
their own Markdown, JSON, and HTML reports.

The project also asks one original, falsifiable question: can a bounded,
RMS-balanced connection between adjacent attention updates help a small decoder?
The answer is not hard-coded. **Attention Bridge** is compared with an exactly
equal-parameter baseline, and the tooling is designed to preserve negative
results as carefully as positive ones.

## Why another from-scratch LLM?

| Typical demo | AstraLM Lab |
|---|---|
| “It generates text” | Cached and full logits must agree numerically |
| One training loss | Held-out loss, raw JSONL metrics, best checkpoints |
| Architecture tweak | Equal-parameter, equal-token, multi-seed ablation |
| Notebook-only result | CLI, CI, machine report, presentation-ready HTML |
| GPU required to learn | Browser-only estimator and pico CPU preset |
| Tokenizer hidden in a library | Deterministic UTF-8 byte BPE implemented here |

## Three-minute tour

### 1. Explore without installing anything

Open **[AstraScope](https://alex0ai.github.io/AstraLM-Lab/)**. Change layer count,
GQA ratio, context, batch, and precision; the parameter and KV-cache budgets
update immediately.

### 2. Install the small core

```bash
git clone https://github.com/Alex0AI/AstraLM-Lab
cd AstraLM-Lab
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 3. Prove the cache before training

```bash
astralm doctor
astralm estimate --preset pico --batch-size 2 --sequence-length 128
astralm verify-cache --preset pico
```

`verify-cache` compares every token's cached logits with a full causal forward
pass and exits non-zero if the configured tolerance is exceeded.

### 4. Run a matched experiment

```bash
astralm ablate \
  --preset pico \
  --corpus examples/tiny_corpus.txt \
  --steps 100 \
  --seeds 17,42,73 \
  --output runs/ablation
```

Open `runs/ablation/report.html`, or commit `report.md` with your experiment.
Every run retains raw metrics and resumable checkpoints.

## What is implemented

```text
UTF-8 bytes / learned BPE
          │
          ▼
tied token embedding
          │
  ┌───────┴──────────────────────────────────────────────┐
  │ RMSNorm → GQA + RoPE + QK-Norm → Attention Bridge ─┤ residual
  │ RMSNorm → SwiGLU ──────────────────────────────────┤ residual
  └────────────────────────────── repeated N times ─────┘
          │
          ▼
RMSNorm → soft-capped tied LM head

incremental path: grouped K/V → per-layer cache → query-head expansion → SDPA
```

- grouped-query causal attention with an unexpanded KV cache;
- rotary embeddings with offset-correct incremental decoding;
- RMSNorm, SwiGLU, QK-Norm, logit softcapping, and tied embeddings;
- bounded cross-layer Attention Bridge with observable gates;
- byte tokenizer plus a deterministic byte-level BPE trainer;
- AdamW, warmup + cosine decay, gradient accumulation, mixed precision;
- train/validation split, periodic evaluation, atomic checkpoints, resume;
- parameter/KV memory estimation, cache benchmark, layer diagnostics;
- matched multi-seed experiments and self-contained reports.

## Attention Bridge, in one equation

```text
g = sigmoid(gate_logit)
bridge(current, previous) = √(1 − g²) · current
                          + g · RMSMatch(previous, current)
```

The gate exists in both standard and bridge blocks, so the two modes have exactly
the same parameter count. Standard mode bypasses it. Read the full
[hypothesis, invariants, and measurement plan](docs/ATTENTION_BRIDGE.md).

## Evidence, not decoration

CI checks causality, RoPE norm preservation, GQA cache equivalence, Unicode BPE
round-trips, static-vs-runtime parameter counts, equal-parameter baselines,
training/checkpoint smoke behavior, and report generation.

This repository does **not** claim that Attention Bridge universally improves
LLMs. A report is evidence for its exact data, scale, token budget, hardware, and
seeds—not a theorem about larger models.

## Laptop-friendly, cloud-ready

| Path | Laptop impact | Purpose |
|---|---:|---|
| AstraScope | zero install | architecture and memory intuition |
| `pico` preset | CPU friendly | correctness and training smoke tests |
| `tiny` preset | modest | local experiments |
| Cloud workflow | no local Torch/data | pinned public-text ablation |
| `small` preset | GPU recommended | scaling experiments |

Use **Actions → Cloud Public Text Ablation → Run workflow** to download a bounded,
commit-pinned Tiny Shakespeare sample, train on a GitHub runner, and retrieve only
the reports. The downloader uses Python's standard library—no native data stack.

## Learn, reproduce, contribute

- Follow the eight-step [learning path](docs/LEARNING_PATH.md).
- Use the [experiment checklist](docs/EXPERIMENTS.md).
- Read the [design notes](docs/DESIGN.md) and [data policy](docs/DATA.md).
- Pick an independently useful item from the public [roadmap](docs/ROADMAP.md).
- Share negative results under the same standard as positive ones.
- See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing an architecture record.

## Provenance

AstraLM is original source code built in conversation with the open LLM ecosystem.
The ideas it learned from—and the exact boundary of its original contribution—are
listed in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

## License

MIT. Public datasets and externally trained weights retain their own licenses.
