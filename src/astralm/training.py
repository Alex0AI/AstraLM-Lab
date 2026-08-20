from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import ModelConfig
from .model import DecoderLM
from .tokenizer import ByteTokenizer


@dataclass(slots=True)
class TrainConfig:
    steps: int = 500
    batch_size: int = 16
    seq_len: int = 128
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation: int = 1
    eval_interval: int = 50
    eval_batches: int = 10
    val_fraction: float = 0.1
    checkpoint_interval: int = 250
    precision: str = "auto"
    compile_model: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.seq_len, self.gradient_accumulation) < 1:
            raise ValueError("steps, batch_size, seq_len, and gradient_accumulation must be positive")
        if min(self.eval_interval, self.eval_batches) < 1 or self.warmup_steps < 0:
            raise ValueError("evaluation values must be positive and warmup_steps cannot be negative")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be between 0 and 0.5")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16, or fp16")


@dataclass(frozen=True, slots=True)
class TrainResult:
    checkpoint: Path
    best_checkpoint: Path
    metrics_file: Path
    best_val_loss: float
    final_val_loss: float
    tokens_processed: int
    parameter_count: int
    duration_seconds: float
    device: str

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "checkpoint": str(self.checkpoint),
            "best_checkpoint": str(self.best_checkpoint),
            "metrics_file": str(self.metrics_file),
            "best_val_loss": self.best_val_loss,
            "final_val_loss": self.final_val_loss,
            "tokens_processed": self.tokens_processed,
            "parameter_count": self.parameter_count,
            "duration_seconds": self.duration_seconds,
            "device": self.device,
        }


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch(
    data: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - cfg.seq_len - 1, (cfg.batch_size,), generator=generator)
    sequences = torch.stack([data[start : start + cfg.seq_len + 1] for start in starts])
    return sequences[:, :-1].to(device), sequences[:, 1:].to(device)


def _split_tokens(tokens: list[int], cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if not tokens:
        raise ValueError("corpus must contain at least one UTF-8 byte")
    minimum = 2 * (cfg.seq_len + 2)
    if len(tokens) < minimum:
        tokens *= math.ceil(minimum / max(1, len(tokens)))
    split = min(
        len(tokens) - cfg.seq_len - 2,
        max(cfg.seq_len + 2, int(len(tokens) * (1 - cfg.val_fraction))),
    )
    return torch.tensor(tokens[:split], dtype=torch.long), torch.tensor(tokens[split:], dtype=torch.long)


def _learning_rate(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32" or device.type not in {"cuda", "cpu"}:
        return nullcontext()
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    if precision in {"auto", "bf16"} and (device.type == "cuda" or precision == "bf16"):
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate(
    model: DecoderLM,
    data: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        inputs, targets = _batch(data, cfg, device, generator)
        with _autocast(device, cfg.precision):
            loss = model(inputs, targets).loss
        assert loss is not None
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def _save_checkpoint(
    path: Path,
    model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    best_val_loss: float,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model_config.to_dict(),
            "train_config": asdict(train_config),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        temporary,
    )
    temporary.replace(path)


def train_run(
    corpus: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output: str | Path,
    device: str | None = None,
    resume: str | Path | None = None,
) -> TrainResult:
    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    target_device = torch.device(device) if device else detect_device()
    tokens = ByteTokenizer().encode(Path(corpus).read_text(encoding="utf-8"))
    train_data, val_data = _split_tokens(tokens, train_config)
    model = DecoderLM(model_config).to(target_device)
    parameter_count = model.parameter_count
    start_step = 0
    best_val_loss = float("inf")
    resume_payload = None
    if resume:
        resume_payload = torch.load(resume, map_location=target_device, weights_only=False)
        model.load_state_dict(resume_payload["model"])
        start_step = int(resume_payload["step"]) + 1
        best_val_loss = float(resume_payload.get("best_val_loss", best_val_loss))

    train_model = model
    if train_config.compile_model and hasattr(torch, "compile"):
        train_model = torch.compile(model)  # type: ignore[assignment]
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=train_config.weight_decay,
    )
    if resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = output_path / "model.pt"
    best_checkpoint = output_path / "best.pt"
    metrics_file = output_path / "metrics.jsonl"
    if not resume:
        metrics_file.write_text("", encoding="utf-8")
    generator = torch.Generator().manual_seed(train_config.seed)
    tokens_per_step = train_config.batch_size * train_config.seq_len * train_config.gradient_accumulation
    started = time.perf_counter()
    final_val_loss = float("inf")
    train_model.train()
    for step in range(start_step, train_config.steps):
        lr = _learning_rate(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(train_config.gradient_accumulation):
            inputs, targets = _batch(train_data, train_config, target_device, generator)
            with _autocast(target_device, train_config.precision):
                loss = train_model(inputs, targets).loss
                assert loss is not None
                scaled_loss = loss / train_config.gradient_accumulation
            scaled_loss.backward()
            accumulated_loss += float(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), train_config.grad_clip)
        optimizer.step()

        should_evaluate = step % train_config.eval_interval == 0 or step == train_config.steps - 1
        if should_evaluate:
            final_val_loss = evaluate(train_model, val_data, train_config, target_device, generator)
            row = {
                "step": step,
                "train_loss": accumulated_loss / train_config.gradient_accumulation,
                "val_loss": final_val_loss,
                "learning_rate": lr,
                "grad_norm": float(grad_norm),
                "tokens_processed": (step + 1) * tokens_per_step,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                _save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    model_config,
                    train_config,
                    step,
                    best_val_loss,
                )
        if train_config.checkpoint_interval and (step + 1) % train_config.checkpoint_interval == 0:
            _save_checkpoint(checkpoint, model, optimizer, model_config, train_config, step, best_val_loss)

    _save_checkpoint(
        checkpoint,
        model,
        optimizer,
        model_config,
        train_config,
        train_config.steps - 1,
        best_val_loss,
    )
    duration = time.perf_counter() - started
    summary = TrainResult(
        checkpoint=checkpoint,
        best_checkpoint=best_checkpoint,
        metrics_file=metrics_file,
        best_val_loss=best_val_loss,
        final_val_loss=final_val_loss,
        tokens_processed=train_config.steps * tokens_per_step,
        parameter_count=parameter_count,
        duration_seconds=duration,
        device=str(target_device),
    )
    (output_path / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    return summary


def train(
    corpus: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output: str | Path,
    device: str | None = None,
) -> Path:
    """Compatibility wrapper returning the final checkpoint path."""
    return train_run(corpus, model_config, train_config, output, device).checkpoint
