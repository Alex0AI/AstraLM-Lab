# Public data recipes

AstraLM does not claim originality over training text. It deliberately reuses
well-documented public datasets while keeping model code, experimental design,
checkpoints, measurements, and conclusions original.

The optional streaming helper supports:

- `roneneldan/TinyStories` for fast language-model sanity checks.
- `HuggingFaceFW/fineweb-edu` for higher-quality web-text experiments.

```bash
pip install -e ".[data]"
python scripts/prepare_public_data.py tinystories --documents 10000
astralm train --corpus data/public_sample.txt --output checkpoints/tinystories
```

Streaming reads only the requested document count. Generated corpora and
checkpoints are gitignored so a laptop and the Git repository stay small. Record
the upstream dataset revision and license before publishing any trained weights.
