# Reproducible experiments

## Fast local smoke experiment

```bash
astralm ablate --preset pico --steps 30 --seeds 17,42 --output runs/smoke
```

This creates one directory per mode and seed plus:

- `report.json` for machines;
- `report.md` for GitHub;
- `report.html` for presentations;
- raw JSONL metrics and resumable checkpoints.

## Public-data cloud experiment

Use **Actions → Cloud Public Text Ablation → Run workflow**. The workflow
downloads a bounded, commit-pinned Tiny Shakespeare sample on the runner,
performs a matched ablation, and uploads reports. No dataset or Torch installation
is required on your laptop.

## Comparison checklist

- [ ] identical corpus revision and split
- [ ] identical model configuration and parameter count
- [ ] identical optimizer, precision, batch size, and token budget
- [ ] identical seeds and number of evaluation batches
- [ ] raw metrics retained
- [ ] hardware and Torch version recorded
- [ ] conclusion scoped to the observed setting
