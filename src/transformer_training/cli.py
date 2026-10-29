"""Training CLI; imports data libraries only after configuring offline mode."""

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

from .config import ModelConfig, TrainingConfig


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a small TinyStories decoder on one GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    for cls in (ModelConfig, TrainingConfig):
        defaults = cls()
        for field in fields(cls):
            # Model vocabulary is obtained from the pinned tokenizer, never guessed for training.
            if field.name == "vocab_size":
                continue
            if field.name == "validation_loss_targets":
                parser.add_argument("--validation-loss-target", dest=field.name,
                                    type=float, action="append", default=[],
                                    help="Observe first validation loss at/below this target; repeatable")
                continue
            default = getattr(defaults, field.name)
            kind = int if field.name in {"max_steps", "supervised_token_budget"} else type(default)
            options = {"type": kind, "default": default,
                       "help": field.name.replace("_", " ").capitalize()}
            if field.name == "precision":
                options["choices"] = ["fp32", "bf16", "fp16"]
            parser.add_argument("--" + field.name.replace("_", "-"), **options)
    parser.add_argument("--revision", help="Dataset revision (default: pinned EDA commit)")
    parser.add_argument("--tokenizer-revision", help="Tokenizer revision (default: pinned GPT-2 commit)")
    parser.add_argument("--max-stories", type=int, help="Seeded sample per split, after full leakage audit")
    parser.add_argument("--offline", action="store_true", help="Cached data and tokenizer only")
    parser.add_argument("--output-dir", type=Path, help="Fresh run directory; default runs/<UTC-id>")
    parser.add_argument("--resume", type=Path, help="Restore an optimizer-boundary checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration and CPU model size; no data or optimizer steps")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    explicit = set()
    # argparse normally loses whether a default was explicitly requested. Resume
    # must inherit saved settings while still rejecting explicit incompatible flags.
    for argument in (sys.argv[1:] if argv is None else argv):
        if argument.startswith("--"):
            explicit.add(argument.split("=", 1)[0][2:].replace("-", "_"))
    if "validation_loss_target" in explicit:
        explicit.add("validation_loss_targets")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        if args.max_stories is not None and args.max_stories < 1:
            raise ValueError("max_stories must be positive")
        from .checkpoint import load_checkpoint

        saved = load_checkpoint(args.resume) if args.resume else None
        configs = []
        for cls, key in ((ModelConfig, "model_config"), (TrainingConfig, "training_config")):
            values = saved[key].copy() if saved else asdict(cls())
            for field in fields(cls):
                if field.name != "vocab_size" and (not saved or field.name in explicit):
                    values[field.name] = getattr(args, field.name)
            configs.append(cls(**values))
        model_config, training_config = configs
        if saved is not None:
            from .training import validate_resume_settings

            validate_resume_settings(saved, model_config, training_config)
        if args.dry_run:
            from .model import DecoderOnlyTransformer

            model = DecoderOnlyTransformer(model_config)
            print(json.dumps({
                "dry_run": True, "model": asdict(model_config),
                "training": asdict(training_config),
                "parameters": sum(p.numel() for p in model.parameters()),
                "effective_batch_size": training_config.effective_batch_size,
                "maximum_padded_tokens_per_update": training_config.effective_batch_size
                    * model_config.sequence_length,
                "note": "CPU model construction only; no data access, GPU check or optimizer steps",
            }, indent=2))
            return 0
        from .training import Trainer, validate_device, validate_output_directory

        output = args.output_dir or (args.resume.parent if args.resume else
                                    Path("runs") / datetime.now(timezone.utc).strftime(
                                        "%Y%m%dT%H%M%S%fZ"))
        validate_output_directory(output, resume=args.resume)
        validate_device(training_config)  # Fail before downloads if CUDA setup is missing.

        options = {
            "sequence_length": model_config.sequence_length,
            "batch_size": training_config.batch_size, "seed": training_config.seed,
            "num_workers": training_config.num_workers,
            "pin_memory": training_config.device.startswith("cuda"),
            "exclude_validation_overlap": True,
            "max_stories": args.max_stories,
        }
        for option, metadata_key in (("revision", "revision"),
                                     ("tokenizer_revision", "tokenizer_revision"),
                                     ("max_stories", "max_stories")):
            if saved and option not in explicit:
                options[option] = saved["data_signature"].get(metadata_key)
            elif getattr(args, option) is not None:
                options[option] = getattr(args, option)
            if saved is not None:
                value = options.get(option)
                # Symbolic Hub revisions still need resolving before comparison.
                immutable = isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
                if (option == "max_stories" or immutable) and (
                    value != saved["data_signature"].get(metadata_key)
                ):
                    raise ValueError(f"Incompatible data setting for resume: {option}")
        from .data import load_tinystories

        preparation_start = time.perf_counter()
        loaders = load_tinystories(**options)
        preparation_seconds = time.perf_counter() - preparation_start
        model_config = replace(model_config, vocab_size=loaders.metadata["vocab_size"])
        trainer = Trainer(model_config, training_config, loaders, output, resume=args.resume,
                          preparation_seconds=preparation_seconds)
        trainer.fit()
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, FloatingPointError) as error:
        parser.exit(2, f"error: {error}\n")
