from __future__ import annotations

import json
import math
import random
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
    warmup_steps: int = 50
    grad_clip: float = 1.0
    seed: int = 42


def _batch(data: torch.Tensor, cfg: TrainConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - cfg.seq_len - 1, (cfg.batch_size,))
    sequences = torch.stack([data[start : start + cfg.seq_len + 1] for start in starts])
    return sequences[:, :-1].to(device), sequences[:, 1:].to(device)


def train(
    corpus: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output: str | Path,
    device: str | None = None,
) -> Path:
    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    text = Path(corpus).read_text(encoding="utf-8")
    tokens = ByteTokenizer().encode(text)
    if len(tokens) <= train_config.seq_len + 1:
        repeats = math.ceil((train_config.seq_len + 2) / max(1, len(tokens)))
        tokens *= repeats
    data = torch.tensor(tokens, dtype=torch.long)
    model = DecoderLM(model_config).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.95))
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(train_config.steps):
        warmup = min(1.0, (step + 1) / max(1, train_config.warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * step / max(1, train_config.steps)))
        lr = train_config.learning_rate * warmup * cosine
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = _batch(data, train_config, target_device)
        loss = model(inputs, targets).loss
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        if step % 10 == 0 or step == train_config.steps - 1:
            history.append({"step": step, "loss": float(loss), "lr": lr, "grad_norm": float(grad_norm)})
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = output_path / "model.pt"
    torch.save({"model": model.state_dict(), "config": model_config.to_dict()}, checkpoint)
    (output_path / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_path / "train_config.json").write_text(json.dumps(asdict(train_config), indent=2), encoding="utf-8")
    return checkpoint

