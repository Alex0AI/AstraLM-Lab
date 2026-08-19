# Public data recipes

AstraLM does not claim originality over training text. It deliberately reuses
well-documented public datasets while keeping model code, experimental design,
checkpoints, measurements, and conclusions original.

The bounded downloader currently supports Tiny Shakespeare, mirrored by
[`karpathy/char-rnn`](https://github.com/karpathy/char-rnn). The source URL is
pinned to commit `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`, and the text is public-domain
Shakespeare. The script records the URL, byte count, and SHA-256 digest beside
the generated corpus.

```bash
python scripts/prepare_public_data.py tinyshakespeare --max-bytes 750000
astralm train --corpus data/public_sample.txt --output runs/tinyshakespeare
```

The helper uses only Python's standard library and reads at most the requested
byte count. Generated corpora and checkpoints are gitignored so a laptop and the
repository stay small. Keep the generated metadata with published measurements,
and re-check upstream provenance before distributing trained weights.
