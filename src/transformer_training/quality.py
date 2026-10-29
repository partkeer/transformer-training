"""Importable checkpoint-quality evaluation; no training or optimizer creation."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import re

from .checkpoint import atomic_json, atomic_text, load_checkpoint
from .config import ModelConfig, TrainingConfig
from .evaluation import evaluate
from .generation import DEFAULT_PROMPTS, encode_prompt, generate, validate_generation_options
from .model import DecoderOnlyTransformer
from .implementations import resolve_implementation, execution_report
from .training import CHECKPOINT_VERSION, environment, validate_device


def loss_metrics(result):
    """Extend the shared evaluator result with JSON-safe perplexity."""
    if not math.isfinite(result["loss"]):
        raise FloatingPointError("Nonfinite validation loss")
    try:
        perplexity = math.exp(result["loss"])
    except OverflowError:
        perplexity = None
    return {**result, "perplexity": perplexity,
            "perplexity_overflow": perplexity is None}


def verify_validation_metadata(saved, actual, *, full_validation=False):
    """Compare validation identity only; train preparation is unnecessary."""
    for key in ("dataset", "revision", "tokenizer", "tokenizer_revision",
                "sequence_length", "vocab_size", "eos_id", "pad_id", "ignore_index"):
        if saved[key] != actual[key]:
            raise ValueError(f"Validation metadata mismatch: {key}")
    expected = saved["splits"]["validation"]
    found = actual["splits"]["validation"]
    if full_validation:
        if found["original_stories"] != expected["original_stories"]:
            raise ValueError("Validation metadata mismatch: original_stories")
        # If training already used the full split, fingerprints must still match.
        if saved["max_stories"] is None or expected["selected_row_indices"] is None:
            if expected != found:
                raise ValueError("Validation provenance/fingerprints changed")
    elif saved["max_stories"] != actual["max_stories"] or expected != found:
        raise ValueError("Validation selection, provenance or fingerprints changed; "
                         "use the checkpoint's data and preprocessing environment")


def _samples_text(samples, *, include_complete_text=True):
    if not samples:
        return "Generation skipped.\n"
    return "\n\n".join(
        f"Sample {index + 1} | seed {sample['seed']} | "
        f"{sample['generated_tokens']} generated tokens | {sample['stop_reason']}\n"
        f"Prompt: {sample['prompt']}\n"
        f"Continuation:\n{sample['continuation']}"
        + (f"\nComplete text:\n{sample['text']}" if include_complete_text else "")
        for index, sample in enumerate(samples)
    ) + "\n"


def evaluate_checkpoint(checkpoint, *, output_dir=None, device=None, precision=None,
                        batch_size=None, num_workers=0, full_validation=False,
                        prompts=None, skip_generation=False, max_new_tokens=128,
                        temperature=0.8, top_k=50, seed=42, implementation_overrides=None):
    """Evaluate a training checkpoint and atomically write new quality artifacts.

    Configure HF_HOME/offline mode before importing Hugging Face libraries (the
    CLI does this). Returns the report dictionary and creates no optimizer.
    Operational overrides affect evaluation only, not the input checkpoint.
    """
    checkpoint = Path(checkpoint).resolve()
    payload = load_checkpoint(checkpoint)
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    model_config = ModelConfig(**payload["model_config"])
    training_config = TrainingConfig(**payload["training_config"])
    import argparse
    allowed = {"attention", "sdpa_backend", "compile", "compile_mode"}
    if set(implementation_overrides or {}) - allowed:
        raise ValueError("Unsupported evaluation implementation override")
    implementation = resolve_implementation(argparse.Namespace(**(implementation_overrides or {})), payload)
    overrides = {"num_workers": num_workers}
    for key, value in (("device", device), ("precision", precision), ("batch_size", batch_size)):
        if value is not None:
            overrides[key] = value
    config = replace(training_config, **overrides)
    target_device = validate_device(config)  # Fail before any Hub requests.
    signature = payload["data_signature"]
    for key in ("revision", "tokenizer_revision"):
        if not isinstance(signature[key], str) or not re.fullmatch(r"[0-9a-f]{40}", signature[key]):
            raise ValueError(f"Checkpoint {key} must be an immutable commit")
    if (signature["sequence_length"] != model_config.sequence_length
            or signature["vocab_size"] != model_config.vocab_size):
        raise ValueError("Checkpoint model context/vocabulary does not match data signature")
    selected_prompts = list(DEFAULT_PROMPTS if prompts is None else prompts)
    validate_generation_options(max_new_tokens=max_new_tokens, temperature=temperature,
                                top_k=top_k, seed=seed)
    if not skip_generation:
        if not selected_prompts or any(not isinstance(p, str) or not p.strip()
                                       for p in selected_prompts):
            raise ValueError("At least one nonblank prompt is required")
        if seed + len(selected_prompts) - 1 >= 2**63:
            raise ValueError("Per-prompt generation seed exceeds supported range")
    output = Path(output_dir) if output_dir is not None else (
        checkpoint.parent / "evaluations" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Evaluation output directory must be empty: {output}")

    # Instantiate/load strictly before downloads; do not retain optimizer tensors.
    model = DecoderOnlyTransformer(model_config, implementation)
    model.load_state_dict(payload["model"], strict=True)
    checkpoint_updates = payload["state"]["updates"]
    del payload
    model.to(target_device)
    model.eval()
    from .data import load_tinystories_validation

    validation = load_tinystories_validation(
        revision=signature["revision"], tokenizer_revision=signature["tokenizer_revision"],
        sequence_length=model_config.sequence_length, batch_size=config.batch_size,
        selection_seed=signature["splits"]["validation"]["selection_seed"],
        num_workers=config.num_workers, pin_memory=target_device.type == "cuda",
        max_stories=None if full_validation else signature["max_stories"],
    )
    verify_validation_metadata(signature, validation.metadata, full_validation=full_validation)
    if not skip_generation:
        for prompt in selected_prompts:
            encode_prompt(validation.tokenizer, prompt, model_config.sequence_length)
    shapes = {} if implementation.attention == "sdpa" else None
    metrics = loss_metrics(evaluate(model, validation.validation, device=target_device,
                                   precision=config.precision, shape_examples=shapes))
    metrics["evaluated_windows"] = len(validation.validation.dataset)
    samples = [] if skip_generation else [
        generate(model, validation.tokenizer, prompt, max_new_tokens=max_new_tokens,
                 temperature=temperature, top_k=top_k, seed=seed + index,
                 precision=config.precision)
        for index, prompt in enumerate(selected_prompts)
    ]
    if implementation.attention == "sdpa":
        from .attention_evidence import capture_attention
        # Evaluation capture runs after metric computation and creates no optimizer.
        model.attention_evidence = capture_attention(model, list(shapes.values()), device=target_device,
            precision=config.precision, environment=environment(target_device), backward=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_updates": checkpoint_updates,
        "model_config": asdict(model_config),
        "evaluation": {"device": str(target_device), "precision": config.precision,
                       "batch_size": config.batch_size, "num_workers": config.num_workers,
                       "selection": "full_validation" if full_validation else "checkpoint",
                       "selection_seed": signature["splits"]["validation"]["selection_seed"]},
        "data": validation.metadata,
        "training_overlap_exclusion": signature.get("exclude_validation_overlap"),
        "environment": environment(target_device), "metrics": metrics,
        "implementation": asdict(implementation), "execution": dict(execution_report(model),
            optimizer="not_applicable", activation_checkpointing_active=False),
        "generation": {"skipped": skip_generation,
                       "prompts": [] if skip_generation else selected_prompts,
                       "max_new_tokens": max_new_tokens, "temperature": temperature,
                       "top_k": top_k, "seed": seed},
        "samples": samples, "output_dir": str(output.resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    atomic_text(output / "samples.txt", _samples_text(samples))
    return report
