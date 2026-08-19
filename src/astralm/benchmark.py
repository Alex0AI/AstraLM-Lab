from __future__ import annotations

import time

import torch

from .model import DecoderLM


def benchmark_decode(
    model: DecoderLM, prompt_length: int = 32, new_tokens: int = 64, repeats: int = 3
) -> dict[str, float]:
    device = next(model.parameters()).device
    prompt = torch.randint(0, model.config.vocab_size, (1, prompt_length), device=device)
    timings: dict[str, float] = {}
    for cached in (False, True):
        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.generate(prompt, new_tokens, temperature=0, use_cache=cached)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append(new_tokens / (time.perf_counter() - start))
        timings["cached_tokens_per_second" if cached else "uncached_tokens_per_second"] = max(samples)
    timings["speedup"] = timings["cached_tokens_per_second"] / timings["uncached_tokens_per_second"]
    return timings

