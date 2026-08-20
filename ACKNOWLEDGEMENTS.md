# Acknowledgements and provenance

AstraLM's source code is original to this repository. Its engineering taste is
informed by excellent open projects:

- [nanoGPT](https://github.com/karpathy/nanoGPT) — readable end-to-end training.
- [llm.c](https://github.com/karpathy/llm.c) — reference agreement tests and honest performance evidence.
- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) — fixed-budget experimentation and public records.
- [LitGPT](https://github.com/Lightning-AI/litgpt) — explicit recipes and approachable scaling paths.
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) — stepwise pedagogy and laptop-aware learning.

No source files from those projects are copied into AstraLM. Concepts such as
RoPE, GQA, RMSNorm, SwiGLU, QK-Norm, and logit softcapping are established public
techniques; their inclusion is not presented as original work. AstraLM's
experimental contribution is the bounded, RMS-balanced **Attention Bridge** and
the matched-ablation workflow around it.

Public data helpers preserve upstream dataset names. Check each dataset card and
license before redistributing text or trained weights.

