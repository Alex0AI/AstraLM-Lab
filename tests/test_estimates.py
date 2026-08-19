from astralm.config import ModelConfig
from astralm.estimates import estimate, human_bytes
from astralm.model import DecoderLM


def test_static_parameter_estimate_matches_model() -> None:
    config = ModelConfig.preset("pico")
    assert estimate(config).parameters == DecoderLM(config).parameter_count


def test_gqa_cache_reduction_is_reported() -> None:
    config = ModelConfig.preset("tiny")
    result = estimate(config, batch_size=2, sequence_length=128)
    assert result.kv_reduction_vs_mha == config.n_heads / config.n_kv_heads
    assert result.kv_cache_bytes > 0
    assert "KiB" in human_bytes(result.kv_cache_bytes) or "MiB" in human_bytes(result.kv_cache_bytes)
