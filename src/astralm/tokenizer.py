from __future__ import annotations


class ByteTokenizer:
    """Deterministic, download-free UTF-8 tokenizer with a 256-token vocabulary."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(token & 0xFF for token in tokens).decode("utf-8", errors="replace")

