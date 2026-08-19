from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

from .config import ModelConfig
from .reporting import summarize_runs, write_report
from .training import TrainConfig, train_run


def run_ablation(
    corpus: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output: str | Path,
    seeds: Iterable[int] = (17, 42, 73),
    device: str | None = None,
) -> dict:
    output_path = Path(output)
    seed_list = list(seeds)
    runs = []
    for mode in ("standard", "attention_bridge"):
        for seed in seed_list:
            run_model = replace(model_config, residual_mode=mode)
            run_train = replace(train_config, seed=seed)
            run_output = output_path / f"{mode}-seed-{seed}"
            result = train_run(corpus, run_model, run_train, run_output, device)
            run = result.to_dict() | {"mode": mode, "seed": seed}
            runs.append(run)
    parameter_counts = {run["parameter_count"] for run in runs}
    if len(parameter_counts) != 1:
        raise AssertionError("ablation parameter counts must match")
    payload = {
        "schema": "astralm-ablation-v1",
        "corpus": str(corpus),
        "seeds": seed_list,
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    write_report(payload, output_path)
    (output_path / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
