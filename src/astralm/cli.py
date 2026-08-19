from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark import benchmark_decode
from .bpe import BytePairTokenizer
from .config import ModelConfig
from .estimates import estimate, human_bytes
from .experiments import run_ablation
from .model import DecoderLM
from .tokenizer import ByteTokenizer
from .training import TrainConfig, detect_device, train_run


def _load(checkpoint: str, device: str) -> DecoderLM:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DecoderLM(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device)


def _model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig.from_yaml(args.config) if getattr(args, "config", None) else ModelConfig.preset(args.preset)


def _add_model_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML config; overrides --preset")
    parser.add_argument("--preset", choices=("pico", "tiny", "small"), default="tiny")


def _train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        warmup_steps=min(args.warmup_steps, args.steps),
        gradient_accumulation=args.gradient_accumulation,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        precision=args.precision,
        compile_model=args.compile,
        seed=args.seed,
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astralm",
        description="A verifiable laboratory for decoder-only Transformer experiments",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train with validation and resumable checkpoints")
    _add_model_selector(train_parser)
    _add_training_arguments(train_parser)
    train_parser.add_argument("--corpus", default="examples/tiny_corpus.txt")
    train_parser.add_argument("--output", default="runs/demo")
    train_parser.add_argument("--resume")

    ablate_parser = commands.add_parser("ablate", help="Run matched standard-vs-bridge experiments")
    _add_model_selector(ablate_parser)
    _add_training_arguments(ablate_parser)
    ablate_parser.set_defaults(preset="pico", steps=100, batch_size=8, seq_len=64, eval_interval=20, eval_batches=5)
    ablate_parser.add_argument("--corpus", default="examples/tiny_corpus.txt")
    ablate_parser.add_argument("--output", default="runs/ablation")
    ablate_parser.add_argument("--seeds", default="17,42,73")

    estimate_parser = commands.add_parser("estimate", help="Estimate parameters and KV-cache memory")
    _add_model_selector(estimate_parser)
    estimate_parser.add_argument("--batch-size", type=int, default=1)
    estimate_parser.add_argument("--sequence-length", type=int)
    estimate_parser.add_argument("--bytes-per-element", type=int, choices=(1, 2, 4), default=2)

    verify_parser = commands.add_parser("verify-cache", help="Numerically compare cached and full logits")
    _add_model_selector(verify_parser)
    verify_parser.set_defaults(preset="pico")
    verify_parser.add_argument("--length", type=int, default=16)
    verify_parser.add_argument("--seed", type=int, default=7)
    verify_parser.add_argument("--tolerance", type=float, default=3e-5)

    generate_parser = commands.add_parser("generate", help="Generate UTF-8 text from a checkpoint")
    generate_parser.add_argument("checkpoint")
    generate_parser.add_argument("prompt")
    generate_parser.add_argument("--tokens", type=int, default=80)
    generate_parser.add_argument("--temperature", type=float, default=0.8)

    benchmark_parser = commands.add_parser("benchmark", help="Compare cached and uncached decode speed")
    benchmark_parser.add_argument("checkpoint")
    benchmark_parser.add_argument("--tokens", type=int, default=64)

    inspect_parser = commands.add_parser("inspect", help="Emit residual RMS and bridge gates as JSON")
    inspect_parser.add_argument("checkpoint")
    inspect_parser.add_argument("text")

    bpe_parser = commands.add_parser("bpe-train", help="Train the dependency-free byte-level BPE")
    bpe_parser.add_argument("corpus")
    bpe_parser.add_argument("--vocab-size", type=int, default=512)
    bpe_parser.add_argument("--output", default="runs/tokenizer.json")

    commands.add_parser("doctor", help="Show the active Torch runtime and accelerator")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = str(detect_device())
    if args.command == "train":
        result = train_run(
            args.corpus,
            _model_config(args),
            _train_config(args),
            args.output,
            device,
            args.resume,
        )
        print(json.dumps(result.to_dict(), indent=2))
    elif args.command == "ablate":
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
        payload = run_ablation(
            args.corpus,
            _model_config(args),
            _train_config(args),
            args.output,
            seeds,
            device,
        )
        print(json.dumps(payload["summary"], indent=2))
    elif args.command == "estimate":
        result = estimate(
            _model_config(args),
            args.batch_size,
            args.sequence_length,
            args.bytes_per_element,
        )
        payload = result.to_dict() | {
            "parameter_memory_human": human_bytes(result.parameter_bytes),
            "kv_cache_human": human_bytes(result.kv_cache_bytes),
        }
        print(json.dumps(payload, indent=2))
    elif args.command == "verify-cache":
        torch.manual_seed(args.seed)
        config = _model_config(args)
        if args.length > config.max_seq_len:
            raise SystemExit("--length exceeds max_seq_len")
        model = DecoderLM(config).to(device)
        tokens = torch.randint(0, config.vocab_size, (1, args.length), device=device)
        error = model.cache_error(tokens)
        print(json.dumps({"max_absolute_logit_error": error, "tolerance": args.tolerance, "passed": error <= args.tolerance}, indent=2))
        raise SystemExit(0 if error <= args.tolerance else 1)
    elif args.command == "generate":
        tokenizer = ByteTokenizer()
        prompt = torch.tensor([tokenizer.encode(args.prompt)], device=device)
        result = _load(args.checkpoint, device).generate(prompt, args.tokens, args.temperature)
        print(tokenizer.decode(result[0].tolist()))
    elif args.command == "benchmark":
        print(json.dumps(benchmark_decode(_load(args.checkpoint, device), new_tokens=args.tokens), indent=2))
    elif args.command == "inspect":
        tokenizer = ByteTokenizer()
        tokens = torch.tensor([tokenizer.encode(args.text)], device=device)
        output = _load(args.checkpoint, device)(tokens, collect_diagnostics=True)
        print(json.dumps(output.diagnostics, indent=2))
    elif args.command == "bpe-train":
        text = Path(args.corpus).read_text(encoding="utf-8")
        tokenizer = BytePairTokenizer.train(text, args.vocab_size)
        target = tokenizer.save(args.output)
        print(json.dumps({"path": str(target), "vocab_size": tokenizer.vocab_size, "compression_ratio": tokenizer.compression_ratio(text)}, indent=2))
    else:
        print(json.dumps({"torch": torch.__version__, "device": device, "sdpa": hasattr(torch.nn.functional, "scaled_dot_product_attention")}, indent=2))


if __name__ == "__main__":
    main()
