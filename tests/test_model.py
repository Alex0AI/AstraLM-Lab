import torch

from astralm.cache import KVCache
from astralm.config import ModelConfig
from astralm.layers import RMSNorm, RotaryEmbedding
from astralm.model import DecoderLM


def tiny_config(residual_mode: str = "attention_bridge") -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        max_seq_len=32,
        residual_mode=residual_mode,
    )


def test_rms_norm_has_unit_rms() -> None:
    output = RMSNorm(16)(torch.randn(3, 5, 16))
    assert torch.allclose(output.pow(2).mean(-1), torch.ones(3, 5), atol=1e-4)


def test_rope_preserves_vector_norm() -> None:
    rope = RotaryEmbedding(8, 16, 10_000)
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 2, 6, 8)
    q_rotated, k_rotated = rope(q, k, 0)
    assert torch.allclose(q.norm(dim=-1), q_rotated.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rotated.norm(dim=-1), atol=1e-5)


def test_future_tokens_do_not_change_past_logits() -> None:
    torch.manual_seed(7)
    model = DecoderLM(tiny_config()).eval()
    prefix = torch.randint(0, 64, (1, 5))
    longer = torch.cat((prefix, torch.randint(0, 64, (1, 3))), dim=1)
    assert torch.allclose(model(prefix).logits, model(longer).logits[:, :5], atol=1e-5)


def test_incremental_cache_matches_full_decode() -> None:
    torch.manual_seed(9)
    model = DecoderLM(tiny_config()).eval()
    tokens = torch.randint(0, 64, (1, 9))
    expected = model(tokens).logits
    cache = KVCache(model.config.n_layers)
    pieces = [model(tokens[:, index : index + 1], cache=cache).logits for index in range(tokens.size(1))]
    actual = torch.cat(pieces, dim=1)
    assert torch.allclose(expected, actual, atol=2e-5)
    assert cache.length == tokens.size(1)


def test_attention_bridge_is_observable() -> None:
    model = DecoderLM(tiny_config())
    output = model(torch.randint(0, 64, (1, 5)), collect_diagnostics=True)
    assert output.diagnostics is not None
    assert "block_1.bridge_gate" in output.diagnostics


def test_standard_and_bridge_have_equal_parameter_count() -> None:
    standard = DecoderLM(tiny_config("standard"))
    bridge = DecoderLM(tiny_config("attention_bridge"))
    assert standard.parameter_count == bridge.parameter_count


def test_model_cache_error_helper() -> None:
    torch.manual_seed(11)
    model = DecoderLM(tiny_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, 7))
    assert model.cache_error(tokens) < 2e-5


def test_standard_diagnostics_hide_inactive_bridge() -> None:
    model = DecoderLM(tiny_config("standard"))
    output = model(torch.randint(0, 64, (1, 5)), collect_diagnostics=True)
    assert output.diagnostics is not None
    assert not any(key.endswith("bridge_gate") for key in output.diagnostics)
