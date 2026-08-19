# Roadmap

AstraLM grows by making small-model experiments easier to understand, falsify,
and reproduce—not by accumulating every feature used by production LLM stacks.

## Now: verifiable foundations

- [x] Full-pass versus incremental KV-cache equivalence test
- [x] Equal-parameter Attention Bridge ablations across multiple seeds
- [x] Deterministic UTF-8 byte BPE
- [x] Atomic and resumable training checkpoints
- [x] Browser-only parameter and KV-memory explorer
- [x] Cloud experiment workflow with public streamed data

## Next: stronger evidence

- [ ] Publish a fixed-token-budget TinyStories baseline table
- [ ] Add bootstrap confidence intervals and paired seed comparisons
- [ ] Record throughput, peak memory, environment, and commit SHA in reports
- [ ] Add a second public corpus to test whether conclusions transfer
- [ ] Export report cards as standalone SVG figures

## Later: teaching and portability

- [ ] Annotated forward-pass trace from UTF-8 bytes to next-token logits
- [ ] ONNX export for the pico preset
- [ ] CPU quantized inference recipe
- [ ] Translated walkthroughs maintained by native speakers

## Explicit non-goals

- Competing with distributed pretraining frameworks
- Claiming that a small-scale win automatically transfers to large models
- Shipping weights or datasets without clear provenance and licenses

If you want to contribute, choose one measurable item, open an issue with its
acceptance criteria, and keep the first pull request narrow.
