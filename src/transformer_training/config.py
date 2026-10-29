"""Explicit, JSON-serialisable model and training settings."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    sequence_length: int = 512
    layers: int = 6
    d_model: int = 160
    heads: int = 4
    d_ff: int = 640
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("vocab_size", "sequence_length", "layers", "d_model", "heads", "d_ff"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    seed: int = 42
    precision: str = "fp32"
    device: str = "cuda"
    num_workers: int = 2
    eval_every: int = 2000
    checkpoint_every: int = 500
    log_every: int = 10
    timing_warmup_steps: int = 10
    supervised_token_budget: int | None = None
    validation_loss_targets: tuple[float, ...] = ()

    def __post_init__(self):
        for name in ("batch_size", "gradient_accumulation_steps", "epochs",
                     "eval_every", "checkpoint_every", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or None")
        if self.supervised_token_budget is not None and self.supervised_token_budget < 1:
            raise ValueError("supervised_token_budget must be positive or None")
        if any(not math.isfinite(x) or x < 0 for x in self.validation_loss_targets):
            raise ValueError("validation loss targets must be finite and nonnegative")
        object.__setattr__(self, "validation_loss_targets", tuple(sorted({float(x) for x in self.validation_loss_targets})))
        if self.num_workers < 0 or self.timing_warmup_steps < 0:
            raise ValueError("num_workers and timing_warmup_steps must be nonnegative")
        if self.seed < 0 or self.seed >= 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        for name in ("learning_rate", "min_learning_rate", "weight_decay", "eps",
                     "max_grad_norm", "warmup_ratio", "beta1", "beta2"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError("Require 0 < min_learning_rate <= learning_rate")
        if self.weight_decay < 0 or self.eps <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid weight decay, optimizer epsilon or gradient norm")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        if self.device != "cpu" and self.device != "cuda" and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("device must be cpu, cuda or cuda:<index>")

    @property
    def effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation_steps
