from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .cache import KVCache
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.size(2)
        cos = self.cos[offset : offset + length][None, None, :, :]
        sin = self.sin[offset : offset + length][None, None, :, :]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.repeats = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = config.dropout
        self.qk_norm = config.qk_norm
        self.qk_norm_eps = config.norm_eps
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        # Each layer is appended at a different point in the forward pass, so a
        # global cache length would be one token ahead for layers after layer 0.
        offset = cache.layer_length(self.layer_id) if cache is not None else 0
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = q * torch.rsqrt(
                q.float().pow(2).mean(-1, keepdim=True) + self.qk_norm_eps
            ).type_as(q)
            k = k * torch.rsqrt(
                k.float().pow(2).mean(-1, keepdim=True) + self.qk_norm_eps
            ).type_as(k)
        q, k = self.rope(q, k, offset)
        if cache is not None:
            cached = cache.append(self.layer_id, k, v)
            k, v = cached.key, cached.value
        k = k.repeat_interleave(self.repeats, dim=1)
        v = v.repeat_interleave(self.repeats, dim=1)

        query_positions = torch.arange(offset, offset + length, device=x.device)
        key_positions = torch.arange(k.size(2), device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        mask = torch.zeros((length, k.size(2)), dtype=q.dtype, device=x.device)
        mask.masked_fill_(~allowed, float("-inf"))
        attention = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out_proj(attention)


class AttentionBridge(nn.Module):
    """Learned cross-layer attention residual with RMS-preserving fusion.

    A layer can reuse the preceding layer's attention update without allowing its
    magnitude to dominate. A single interpretable gate makes ablations cheap.
    """

    def __init__(self, init: float) -> None:
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(float(init)))

    def forward(self, current: torch.Tensor, previous: torch.Tensor | None) -> torch.Tensor:
        if previous is None:
            return current
        gate = torch.sigmoid(self.gate_logit)
        previous_scale = previous.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        current_scale = current.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        balanced_previous = previous * (current_scale / previous_scale).type_as(previous)
        return torch.sqrt(1.0 - gate.square()) * current + gate * balanced_previous


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = GroupedQueryAttention(config, layer_id)
        self.ffn = SwiGLU(config)
        # The scalar exists in both modes so an ablation has exactly equal
        # parameter count. Standard mode simply bypasses its computation.
        self.bridge = AttentionBridge(config.bridge_init)
        self.use_bridge = config.residual_mode == "attention_bridge"

    def forward(
        self, x: torch.Tensor, previous_attention: torch.Tensor | None, cache: KVCache | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self.attn(self.attn_norm(x), cache)
        if self.use_bridge:
            attention = self.bridge(attention, previous_attention)
        x = x + attention
        x = x + self.ffn(self.ffn_norm(x))
        return x, attention
