"""Download a bounded, pinned public text corpus with only the standard library.

The default source is hosted on GitHub and pinned to a commit, avoiding native
dataset runtimes and multi-gigabyte downloads on laptops or ephemeral runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


SOURCES = {
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt",
        "description": "Public-domain Shakespeare text mirrored by karpathy/char-rnn",
    }
}


def download(source: str, max_bytes: int, output: Path) -> dict[str, str | int]:
    if max_bytes < 1_024:
        raise ValueError("max_bytes must be at least 1024")
    metadata = SOURCES[source]
    request = Request(metadata["url"], headers={"User-Agent": "AstraLM-Lab/0.2"})
    with urlopen(request, timeout=60) as response:
        payload = response.read(max_bytes)
    if len(payload) < 1_024:
        raise RuntimeError("public corpus response was unexpectedly small")
    text = payload.decode("utf-8", errors="ignore")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    result: dict[str, str | int] = {
        "source": source,
        "source_url": metadata["url"],
        "description": metadata["description"],
        "bytes": len(text.encode("utf-8")),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    output.with_suffix(".meta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument("--max-bytes", type=int, default=750_000)
    parser.add_argument("--output", type=Path, default=Path("data/public_sample.txt"))
    args = parser.parse_args()
    print(json.dumps(download(args.source, args.max_bytes, args.output), indent=2))


if __name__ == "__main__":
    main()
