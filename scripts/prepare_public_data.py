"""Stream a bounded slice of a public text dataset into a local UTF-8 corpus.

This script is intentionally optional. It reuses public training data while
keeping AstraLM's architecture, experiments, checkpoints, and conclusions
independent. Streaming prevents a multi-gigabyte dataset from filling a laptop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


PRESETS = {
    "tinystories": ("roneneldan/TinyStories", "train", "text"),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "train", "text"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument("--documents", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("data/public_sample.txt"))
    args = parser.parse_args()
    dataset_name, split, field = PRESETS[args.preset]
    stream = load_dataset(dataset_name, split=split, streaming=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(stream):
            if index >= args.documents:
                break
            text = str(record.get(field, "")).strip()
            if text:
                handle.write(text + "\n\n")


if __name__ == "__main__":
    main()
