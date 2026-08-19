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
    qk_norm: bool = True
    logit_softcap: float | None = 30.0
    norm_eps: float = 1e-6
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if min(self.dim, self.n_heads, self.n_kv_heads, self.hidden_dim) < 1:
            raise ValueError("dimensions and head counts must be positive")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if self.residual_mode not in {"standard", "attention_bridge"}:
            raise ValueError("residual_mode must be standard or attention_bridge")
        if self.vocab_size < 2 or self.n_layers < 1 or self.max_seq_len < 2:
            raise ValueError("vocab_size, n_layers, and max_seq_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.logit_softcap is not None and self.logit_softcap <= 0:
            raise ValueError("logit_softcap must be positive or null")
        if self.norm_eps <= 0 or self.init_std <= 0:
            raise ValueError("norm_eps and init_std must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def preset(cls, name: str) -> ModelConfig:
        """Return a documented scale preset; all presets fit the same code path."""
        presets: dict[str, dict[str, Any]] = {
            "pico": {
                "dim": 64,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 1,
                "hidden_dim": 192,
                "max_seq_len": 128,
            },
            "tiny": {
                "dim": 192,
                "n_layers": 6,
                "n_heads": 6,
                "n_kv_heads": 2,
                "hidden_dim": 512,
                "max_seq_len": 256,
            },
            "small": {
                "dim": 512,
                "n_layers": 12,
                "n_heads": 8,
                "n_kv_heads": 2,
                "hidden_dim": 1_408,
                "max_seq_len": 1_024,
            },
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name!r}; choose from {', '.join(presets)}")
        return cls(**presets[name])

