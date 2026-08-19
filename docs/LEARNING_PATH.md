# Learning path

1. Open [`studio/index.html`](../studio/index.html) and change query/KV heads.
2. Run `astralm estimate --preset pico` and verify the same memory formula.
3. Read `layers.py`: RMSNorm → GQA/RoPE/QK-Norm → residual → SwiGLU.
4. Run `astralm verify-cache --preset pico`; inspect the maximum logit error.
5. Train a byte-level BPE with `astralm bpe-train` and examine its merge file.
6. Train the pico model for a few steps and inspect JSONL metrics.
7. Run a matched Attention Bridge ablation across multiple seeds.
8. Change one assumption, record it, and contribute the negative or positive result.

The goal is not merely to obtain fluent samples. It is to build the habit of
turning architecture claims into reproducible tests.

