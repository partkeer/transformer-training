"""Standalone quality evaluation CLI with early offline configuration."""

import argparse
import json
import os
from pathlib import Path
from .implementations import add_implementation_arguments


def build_parser():
    """Return a checkpoint-evaluation parser with execution overrides, validation
    selection, and generation controls. Keep generation seed independent of the
    saved validation selection seed.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TinyStories checkpoint and generate story samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Training checkpoint (best.pt or latest.pt)")
    add_implementation_arguments(parser, evaluation=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh output directory; default <checkpoint-dir>/evaluations/<UTC-id>")
    parser.add_argument("--device", help="Override saved device: cpu, cuda or cuda:<index>")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"),
                        help="Override saved evaluation precision")
    parser.add_argument("--batch-size", type=int, help="Override saved microbatch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--offline", action="store_true", help="Use cached data/tokenizer only")
    parser.add_argument("--full-validation", action="store_true",
                        help="Evaluate the full official split instead of the saved selection")
    parser.add_argument("--prompt", action="append", dest="prompts",
                        help="Story opening; repeat to replace the three default prompts")
    parser.add_argument("--skip-generation", action="store_true", help="Report metrics only")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation token limit")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature; 0 is greedy")
    parser.add_argument("--top-k", type=int, default=50, help="Sampling candidates; 0 is full vocabulary")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed; validation uses saved seed")
    return parser


def main(argv=None):
    """Parse options, configure offline mode before data imports, and run checkpoint
    evaluation. Print metrics and continuations without duplicating complete text,
    returning zero on success or reporting a CLI error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    options = vars(args).copy()
    options.pop("offline")
    implementation_options = {key: options.pop(key) for key in
                              ("attention", "sdpa_backend", "compile", "compile_mode")}
    options["implementation_overrides"] = {k: v for k, v in implementation_options.items() if v is not None}
    try:
        from .quality import evaluate_checkpoint, _samples_text

        report = evaluate_checkpoint(**options)
        print(json.dumps({"checkpoint": report["checkpoint"],
                          "selection": report["evaluation"]["selection"],
                          **report["metrics"], "output_dir": report["output_dir"]},
                         indent=2, allow_nan=False))
        print(_samples_text(report["samples"], include_complete_text=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, FloatingPointError) as error:
        parser.exit(2, f"error: {error}\n")
