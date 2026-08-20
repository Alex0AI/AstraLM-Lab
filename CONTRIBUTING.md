# Contributing

Contributions are welcome, especially correctness tests, matched ablations,
teaching notes, and reproducible performance measurements.

## Ground rules

1. Open an issue for architectural changes before writing a large patch.
2. Keep model comparisons matched on data, token budget, seed, parameter count,
   precision, and evaluation batches.
3. Add a focused test for every correctness-sensitive change.
4. Never commit datasets, credentials, proprietary weights, or generated claims.
5. Report unsuccessful experiments; negative results are useful evidence.

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check src tests
pytest
astralm verify-cache --preset pico
```

## Experiment pull requests

Include the exact command, commit SHA, hardware, Torch version, data revision,
configuration, random seeds, raw metrics, and a scoped interpretation. A small
table is better than an unsupported superlative.

