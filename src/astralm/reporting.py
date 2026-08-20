from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for mode in sorted({run["mode"] for run in runs}):
        selected = [run for run in runs if run["mode"] == mode]
        values = [float(run["best_val_loss"]) for run in selected]
        summaries.append(
            {
                "mode": mode,
                "runs": len(values),
                "mean_best_val_loss": mean(values),
                "std_best_val_loss": stdev(values) if len(values) > 1 else 0.0,
                "parameters": selected[0]["parameter_count"],
                "mean_duration_seconds": mean(float(run["duration_seconds"]) for run in selected),
            }
        )
    return summaries


def paired_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {
        mode: {int(run["seed"]): float(run["best_val_loss"]) for run in runs if run["mode"] == mode}
        for mode in ("standard", "attention_bridge")
    }
    seeds = sorted(set(by_mode["standard"]) & set(by_mode["attention_bridge"]))
    if not seeds:
        raise ValueError("paired comparison requires matching standard and bridge seeds")
    deltas = [by_mode["attention_bridge"][seed] - by_mode["standard"][seed] for seed in seeds]
    return {
        "definition": "attention_bridge_best_val_loss - standard_best_val_loss",
        "seeds": seeds,
        "deltas": deltas,
        "mean_delta": mean(deltas),
        "std_delta": stdev(deltas) if len(deltas) > 1 else 0.0,
        "attention_bridge_wins": sum(delta < 0 for delta in deltas),
        "standard_wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
    }


def write_report(payload: dict[str, Any], output: str | Path) -> tuple[Path, Path, Path]:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "report.json"
    markdown_path = target / "report.md"
    html_path = target / "report.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [
        f"| {row['mode']} | {row['runs']} | {row['parameters']:,} | {row['mean_best_val_loss']:.4f} ± {row['std_best_val_loss']:.4f} | {row['mean_duration_seconds']:.1f}s |"
        for row in payload["summary"]
    ]
    paired = payload["paired_comparison"]
    markdown = "\n".join(
        [
            "# AstraLM ablation report",
            "",
            "> Generated from matched configurations and seeds. Lower validation loss is better.",
            "",
            "| Residual mode | Runs | Parameters | Best validation loss | Mean duration |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            f"Corpus: `{payload['corpus']}`  ",
            f"Seeds: `{', '.join(map(str, payload['seeds']))}`  ",
            f"Training steps per run: `{payload['train_config']['steps']}`",
            f"Paired bridge − standard delta: `{paired['mean_delta']:+.6f} ± {paired['std_delta']:.6f}`  ",
            f"Seed wins (bridge / standard / ties): `{paired['attention_bridge_wins']} / {paired['standard_wins']} / {paired['ties']}`",
            "",
            "A negative paired delta favors Attention Bridge. Compare its magnitude with seed variation before interpreting it.",
            "",
            "## Environment",
            "",
            f"Python: `{payload['environment']['python']}`  ",
            f"PyTorch: `{payload['environment']['torch']}`  ",
            f"Platform: `{payload['environment']['platform']}`  ",
            f"Commit: `{payload['environment']['commit_sha'] or 'not recorded'}`",
            "",
            "## Interpretation guardrail",
            "",
            "These results apply only to the recorded scale, data, token budget, and seeds. They do not establish a universal architecture improvement.",
        ]
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    cards = "".join(
        f"<article><h2>{html.escape(row['mode'])}</h2><strong>{row['mean_best_val_loss']:.4f}</strong><p>± {row['std_best_val_loss']:.4f} · {row['parameters']:,} params</p></article>"
        for row in payload["summary"]
    )
    comparison = (
        f"Bridge − standard: {paired['mean_delta']:+.6f} ± {paired['std_delta']:.6f} · "
        f"wins {paired['attention_bridge_wins']}/{paired['standard_wins']}/{paired['ties']}"
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>AstraLM report</title><style>
body{{font:16px system-ui;background:#071426;color:#dbeafe;max-width:1000px;margin:0 auto;padding:48px}}h1{{font-size:44px}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}}article{{background:#0d2038;border:1px solid #1d4ed8;border-radius:18px;padding:24px}}strong{{font-size:38px;color:#67e8f9}}footer{{margin-top:32px;color:#94a3b8}}
</style></head><body><h1>AstraLM ablation report</h1><p>Matched architecture · matched token budget · multi-seed comparison</p><main>{cards}</main><h2>Paired comparison</h2><p>{html.escape(comparison)}</p><footer>Generated by AstraLM. Lower validation loss is better. Scope conclusions to this exact experiment.</footer></body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return json_path, markdown_path, html_path
