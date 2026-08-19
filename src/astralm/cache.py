from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


class KVCache:
    """Layer-addressed cache used by incremental decoding.

    Tensors use [batch, kv_head, time, head_dim]. The cache intentionally keeps
    a tiny API so cached and uncached execution can be tested against each other.
    """

    def __init__(self, n_layers: int) -> None:
        self.layers: list[LayerKV | None] = [None] * n_layers

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> LayerKV:
        previous = self.layers[layer]
        if previous is None:
            current = LayerKV(key, value)
        else:
            current = LayerKV(
                torch.cat((previous.key, key), dim=2),
                torch.cat((previous.value, value), dim=2),
            )
        self.layers[layer] = current
        return current

    @property
    def length(self) -> int:
        first = self.layers[0]
        return 0 if first is None else first.key.size(2)

    def layer_length(self, layer: int) -> int:
        """Return one layer's length while a multi-layer append is in progress."""
        state = self.layers[layer]
        return 0 if state is None else state.key.size(2)

    def clear(self) -> None:
        self.layers = [None] * len(self.layers)
