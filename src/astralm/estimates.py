from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig


@dataclass(frozen=True, slots=True)
class ModelEstimate:
    parameters: int
    parameter_bytes: int
    kv_cache_bytes: int
    kv_reduction_vs_mha: float
    tokens_per_batch: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "parameters": self.parameters,
            "parameter_bytes": self.parameter_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "kv_reduction_vs_mha": self.kv_reduction_vs_mha,
            "tokens_per_batch": self.tokens_per_batch,
        }


def estimate(
    config: ModelConfig,
    batch_size: int = 1,
    sequence_length: int | None = None,
    bytes_per_element: int = 2,
) -> ModelEstimate:
    if batch_size < 1 or bytes_per_element not in {1, 2, 4}:
        raise ValueError("batch_size must be positive and bytes_per_element must be 1, 2, or 4")
    sequence_length = sequence_length or config.max_seq_len
    if not 1 <= sequence_length <= config.max_seq_len:
        raise ValueError("sequence_length must be within the configured context window")
    head_dim = config.dim // config.n_heads
    embedding = config.vocab_size * config.dim
    output = 0 if config.tie_embeddings else embedding
    attention = config.n_layers * (
        config.dim * config.n_heads * head_dim
        + 2 * config.dim * config.n_kv_heads * head_dim
        + config.dim * config.dim
    )
    feed_forward = config.n_layers * 3 * config.dim * config.hidden_dim
    norms = config.n_layers * 2 * config.dim + config.dim
    bridges = config.n_layers
    parameters = embedding + output + attention + feed_forward + norms + bridges
    cache_elements = 2 * batch_size * config.n_layers * config.n_kv_heads * sequence_length * head_dim
    return ModelEstimate(
        parameters=parameters,
        parameter_bytes=parameters * bytes_per_element,
        kv_cache_bytes=cache_elements * bytes_per_element,
        kv_reduction_vs_mha=config.n_heads / config.n_kv_heads,
        tokens_per_batch=batch_size * sequence_length,
    )


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")
