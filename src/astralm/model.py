from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .cache import KVCache
from .config import ModelConfig
from .layers import DecoderBlock, RMSNorm


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    diagnostics: dict[str, float] | None = None


class DecoderLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(DecoderBlock(config, index) for index in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    @property
    def parameter_count(self) -> int:
        """Count unique parameters (tied embeddings are counted once)."""
        return sum(parameter.numel() for parameter in self.parameters())

    def kv_cache_bytes(self, batch_size: int, sequence_length: int, bytes_per_element: int = 2) -> int:
        head_dim = self.config.dim // self.config.n_heads
        elements = 2 * batch_size * self.config.n_layers * self.config.n_kv_heads * sequence_length * head_dim
        return elements * bytes_per_element

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        collect_diagnostics: bool = False,
    ) -> ModelOutput:
        if tokens.size(1) + (cache.length if cache else 0) > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        x = self.embedding(tokens)
        previous_attention = None
        diagnostics: dict[str, float] = {}
        for index, block in enumerate(self.blocks):
            x, previous_attention = block(x, previous_attention, cache)
            if collect_diagnostics:
                diagnostics[f"block_{index}.residual_rms"] = float(x.detach().float().pow(2).mean().sqrt())
                if block.use_bridge:
                    diagnostics[f"block_{index}.bridge_gate"] = float(
                        block.bridge.gate_logit.detach().sigmoid()
                    )
        logits = self.output(self.norm(x))
        if self.config.logit_softcap is not None:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return ModelOutput(logits, loss, diagnostics if collect_diagnostics else None)

    @torch.inference_mode()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.eval()
        cache = KVCache(self.config.n_layers) if use_cache else None
        generated = tokens
        logits = self(tokens, cache=cache).logits[:, -1]
        for step in range(max_new_tokens):
            if temperature <= 0:
                next_token = logits.argmax(-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < values[:, [-1]]] = float("-inf")
                next_token = torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if step == max_new_tokens - 1:
                break
            context = next_token if cache is not None else generated[:, -self.config.max_seq_len :]
            logits = self(context, cache=cache).logits[:, -1]
        return generated

    @torch.inference_mode()
    def cache_error(self, tokens: torch.Tensor) -> float:
        """Return max absolute logit error between full and token-wise decode."""
        self.eval()
        expected = self(tokens).logits
        cache = KVCache(self.config.n_layers)
        pieces = [self(tokens[:, index : index + 1], cache=cache).logits for index in range(tokens.size(1))]
        actual = torch.cat(pieces, dim=1)
        return float((expected - actual).abs().max())
