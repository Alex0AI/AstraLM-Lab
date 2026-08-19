from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _merge_pair(tokens: list[int], pair: tuple[int, int], replacement: int) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            merged.append(replacement)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged


class BytePairTokenizer:
    """A deterministic byte-level BPE implementation with no runtime dependency."""

    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges = merges or []
        self.vocabulary: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for index, (left, right) in enumerate(self.merges, start=256):
            self.vocabulary[index] = self.vocabulary[left] + self.vocabulary[right]

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, min_frequency: int = 2) -> BytePairTokenizer:
        if not 256 <= vocab_size <= 65_535:
            raise ValueError("vocab_size must be between 256 and 65535")
        tokens = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        while len(merges) + 256 < vocab_size and len(tokens) > 1:
            counts = Counter(zip(tokens, tokens[1:]))
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_frequency:
                break
            tokens = _merge_pair(tokens, pair, 256 + len(merges))
            merges.append(pair)
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        for index, pair in enumerate(self.merges, start=256):
            tokens = _merge_pair(tokens, pair, index)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        try:
            payload = b"".join(self.vocabulary[token] for token in tokens)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error
        return payload.decode("utf-8", errors="replace")

    def compression_ratio(self, text: str) -> float:
        encoded = self.encode(text)
        return len(text.encode("utf-8")) / max(1, len(encoded))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "astralm-bpe-v1", "merges": [list(pair) for pair in self.merges]}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> BytePairTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "astralm-bpe-v1":
            raise ValueError("unsupported tokenizer format")
        return cls([tuple(pair) for pair in payload["merges"]])
