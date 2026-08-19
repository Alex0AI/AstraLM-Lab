from __future__ import annotations

import argparse
import json

import torch

from .benchmark import benchmark_decode
from .config import ModelConfig
from .model import DecoderLM
from .tokenizer import ByteTokenizer
from .training import TrainConfig, train


def _load(checkpoint: str, device: str) -> DecoderLM:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DecoderLM(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(prog="astralm", description="AstraLM experiment laboratory")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", default="configs/tiny.yaml")
    train_parser.add_argument("--corpus", default="examples/tiny_corpus.txt")
    train_parser.add_argument("--output", default="checkpoints/demo")
    train_parser.add_argument("--steps", type=int, default=500)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--seq-len", type=int, default=128)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("checkpoint")
    generate_parser.add_argument("prompt")
    generate_parser.add_argument("--tokens", type=int, default=80)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("checkpoint")
    benchmark_parser.add_argument("--tokens", type=int, default=64)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("checkpoint")
    inspect_parser.add_argument("text")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.command == "train":
        checkpoint = train(
            args.corpus,
            ModelConfig.from_yaml(args.config),
            TrainConfig(steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len),
            args.output,
            device,
        )
        print(checkpoint)
    elif args.command == "generate":
        tokenizer = ByteTokenizer()
        prompt = torch.tensor([tokenizer.encode(args.prompt)], device=device)
        result = _load(args.checkpoint, device).generate(prompt, args.tokens, args.temperature)
        print(tokenizer.decode(result[0].tolist()))
    elif args.command == "benchmark":
        print(json.dumps(benchmark_decode(_load(args.checkpoint, device), new_tokens=args.tokens), indent=2))
    else:
        tokenizer = ByteTokenizer()
        tokens = torch.tensor([tokenizer.encode(args.text)], device=device)
        output = _load(args.checkpoint, device)(tokens, collect_diagnostics=True)
        print(json.dumps(output.diagnostics, indent=2))


if __name__ == "__main__":
    main()
