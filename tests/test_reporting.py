from math import isclose

from astralm.reporting import paired_comparison, summarize_runs, write_report


def test_report_contains_matched_modes(tmp_path) -> None:
    runs = [
        {"mode": "standard", "seed": 1, "best_val_loss": 2.0, "parameter_count": 100, "duration_seconds": 1.0},
        {"mode": "standard", "seed": 2, "best_val_loss": 2.2, "parameter_count": 100, "duration_seconds": 1.2},
        {"mode": "attention_bridge", "seed": 1, "best_val_loss": 1.9, "parameter_count": 100, "duration_seconds": 1.1},
        {"mode": "attention_bridge", "seed": 2, "best_val_loss": 2.1, "parameter_count": 100, "duration_seconds": 1.3},
    ]
    comparison = paired_comparison(runs)
    payload = {
        "corpus": "sample.txt",
        "seeds": [1, 2],
        "train_config": {"steps": 10},
        "summary": summarize_runs(runs),
        "paired_comparison": comparison,
        "environment": {
            "python": "3.test",
            "torch": "2.test",
            "platform": "test-platform",
            "commit_sha": "abc123",
        },
    }
    paths = write_report(payload, tmp_path)
    assert all(path.exists() for path in paths)
    assert "Interpretation guardrail" in paths[1].read_text(encoding="utf-8")
    assert isclose(comparison["mean_delta"], -0.1)
    assert comparison["attention_bridge_wins"] == 2
