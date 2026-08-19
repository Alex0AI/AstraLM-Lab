from astralm.config import ModelConfig
from astralm.training import TrainConfig, train_run


def test_training_writes_best_checkpoint_and_metrics(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("small verified models make experiments reproducible. " * 20, encoding="utf-8")
    result = train_run(
        corpus,
        ModelConfig.preset("pico"),
        TrainConfig(
            steps=2,
            batch_size=2,
            seq_len=8,
            warmup_steps=1,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            precision="fp32",
        ),
        tmp_path / "run",
        device="cpu",
    )
    assert result.checkpoint.exists()
    assert result.best_checkpoint.exists()
    assert result.metrics_file.read_text(encoding="utf-8").count("\n") == 2
    assert result.best_val_loss < float("inf")
