from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 256
    dim: int = 192
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    hidden_dim: int = 512
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_base: float = 10_000.0
    residual_mode: str = "attention_bridge"
    bridge_init: float = -2.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if self.residual_mode not in {"standard", "attention_bridge"}:
            raise ValueError("residual_mode must be standard or attention_bridge")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

