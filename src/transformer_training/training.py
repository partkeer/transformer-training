"""Single-GPU reference training, token-weighted accumulation and exact data resume."""

from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

import torch
from torch.utils.data import DataLoader, Sampler

from .checkpoint import atomic_json, capture_rng, load_checkpoint, restore_rng, save_checkpoint
from .config import ModelConfig, TrainingConfig
from .evaluation import IGNORE_INDEX, autocast_context, evaluate, move_batch, next_token_loss
from .model import DecoderOnlyTransformer
from .measurements import Invocation, Phases, DeviceMemorySampler, source_identity, unavailable

CHECKPOINT_VERSION = 1


class NumericalFailure(FloatingPointError):
    """Keep the existing exception contract while identifying the numerical failure."""
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def validate_output_directory(output_dir, *, resume=None):
    """Reject invalid paths before data preparation, without creating anything."""
    output = Path(output_dir)
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output path is not a directory: {output}")
        if resume is None and any(output.iterdir()):
            raise ValueError("Output directory is not empty; use a new directory or --resume")
    for parent in output.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f"Output parent is not a directory: {parent}")
            break


def validate_resume_settings(payload, model_config, config):
    """Check compatibility that does not depend on preparing the dataset."""
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    current = asdict(config)
    saved = asdict(TrainingConfig(**payload["training_config"]))
    for key in ("num_workers", "log_every", "eval_every", "checkpoint_every",
                "validation_loss_targets", "supervised_token_budget", "timing_warmup_steps"):
        current.pop(key, None)
        saved.pop(key, None)
    if payload["model_config"] != asdict(model_config) or saved != current:
        raise ValueError("Incompatible model/training settings for resume")


def clip_gradients(parameters, max_norm, *, allow_nonfinite_gradients=False):
    """Clip finite gradients; leave AMP overflow untouched for GradScaler to skip.

    A nonfinite norm with finite gradient elements is a separate reduction
    overflow. Abort before clipping can zero those gradients or step the optimizer.
    """
    parameters = list(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(gradients)
    if not torch.isfinite(total_norm):
        nonfinite = any(not torch.isfinite(gradient).all() for gradient in gradients)
        if allow_nonfinite_gradients and nonfinite:
            return total_norm
        raise NumericalFailure("nonfinite_gradients" if nonfinite else "gradient_norm_overflow",
                               "Nonfinite gradient norm; stopping before optimizer update")
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm)
    return total_norm


def validate_device(config):
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Install CUDA-enabled PyTorch on the GPU host; "
                             "CPU execution requires --device cpu explicitly.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device index is unavailable")
        with torch.cuda.device(device):
            if config.precision == "bf16" and not torch.cuda.is_bf16_supported(
                including_emulation=False
            ):
                raise ValueError("BF16 requires native GPU BF16 support; select fp32 or fp16")
    elif config.precision != "fp32":
        raise ValueError("This reference supports CPU only with fp32 precision")
    return device


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    # Use one precision-control API consistently (PyTorch >= 2.9).
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False


class EpochSampler(Sampler):
    """Epoch permutation sliced at the trainer's consumed cursor, never prefetch state."""

    def __init__(self, size, seed, epoch, start=0):
        if not 0 <= start <= size:
            raise ValueError("Consumed cursor is outside the dataset")
        self.size, self.seed, self.epoch, self.start = size, seed, epoch, start

    def __iter__(self):
        generator = torch.Generator().manual_seed((self.seed + self.epoch) % 2**63)
        order = torch.randperm(self.size, generator=generator)
        # Tensor iteration unbinds millions of scalar tensors at once. A Python
        # integer list is much smaller and matches PyTorch RandomSampler's approach.
        return iter(order[self.start:].tolist())

    def __len__(self):
        return self.size - self.start


class WarmupCosine:
    """LR for the next successful update; skipped FP16 updates keep the same LR."""

    def __init__(self, optimizer, config, total_updates):
        if total_updates < 1:
            raise ValueError("Schedule requires at least one update")
        self.optimizer, self.config = optimizer, config
        self.total_updates = total_updates
        self.completed_updates = 0
        self._apply()

    def learning_rate(self, index):
        c, total = self.config, self.total_updates
        if total == 1:
            return c.learning_rate
        if index >= total - 1:
            return c.min_learning_rate
        warmup = min(total - 1, max(1, int(total * c.warmup_ratio))) if c.warmup_ratio else 0
        if index < warmup:
            return c.learning_rate * (index + 1) / warmup
        progress = min(1.0, (index - warmup) / max(1, total - warmup - 1))
        return c.min_learning_rate + 0.5 * (c.learning_rate - c.min_learning_rate) * (
            1 + math.cos(math.pi * progress))

    def _apply(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate(self.completed_updates)

    def step(self):
        self.completed_updates += 1
        self._apply()

    def state_dict(self):
        return {"completed_updates": self.completed_updates, "total_updates": self.total_updates}

    def load_state_dict(self, state):
        if state["total_updates"] != self.total_updates:
            raise ValueError("Checkpoint schedule horizon changed")
        self.completed_updates = state["completed_updates"]
        self._apply()


def make_optimizer(model, config):
    # model.parameters() removes aliases, so tied embeddings appear only once.
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps,
        foreach=False, fused=False,
    )


def accumulate_gradients(model, batches, *, device="cpu", precision="fp32", scaler=None,
                         phases=None):
    """Backpropagate sum(losses) / sum(valid targets) without retaining microbatch graphs."""
    phases = phases or Phases()
    with phases.range("data_preparation"):
        tokens = sum(int((batch["labels"] != IGNORE_INDEX).sum()) for batch in batches)
    if not tokens:
        raise ValueError("Accumulation group contains no supervised tokens")
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    for cpu_batch in batches:
        with phases.range("transfer"):
            batch = move_batch(cpu_batch, device)
        with phases.range("forward_loss"), autocast_context(device, precision):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = next_token_loss(logits, batch["labels"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; stopping before optimizer update")
        normalised = loss / tokens
        with phases.range("backward"):
            (scaler.scale(normalised) if scaler is not None else normalised).backward()
        loss_sum += loss.detach().double()
        del logits, loss, normalised, batch
    return {"loss_sum": loss_sum.item(), "supervised_tokens": tokens,
            "slots": sum(batch["labels"].numel() for batch in batches),
            "examples": sum(len(batch["labels"]) for batch in batches)}


def optimizer_attempt(model, optimizer, scaler, scheduler, batches, config, device, *, phases=None):
    """The single update path used by training, benchmarks, and every profiler."""
    phases = phases or Phases()
    with phases.range("zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    result = accumulate_gradients(model, batches, device=device, precision=config.precision,
                                  scaler=scaler, phases=phases)
    with phases.range("optimizer"):
        with phases.range("unscale_clip"):
            scaler.unscale_(optimizer)
            norm = clip_gradients(model.parameters(), config.max_grad_norm,
                                  allow_nonfinite_gradients=config.precision == "fp16")
        lr, previous_scale = optimizer.param_groups[0]["lr"], scaler.get_scale()
        with phases.range("scaler_step"):
            scaler.step(optimizer)
            scaler.update()
        scale = scaler.get_scale()
        skipped = scale < previous_scale
        if not skipped:
            scheduler.step()
    return dict(result, skipped=skipped, learning_rate=lr, scaler_before=previous_scale,
                scaler_after=scale, nonfinite_gradients=skipped)


def data_signature(metadata):
    # Worker and hardware settings may change on resume; corpus identity may not.
    keys = ("dataset", "revision", "tokenizer", "tokenizer_revision", "sequence_length",
            "max_stories", "seed", "vocab_size", "eos_id", "pad_id", "ignore_index",
            "exclude_validation_overlap", "policy", "splits", "audit")
    return {key: metadata[key] for key in keys if key in metadata}


def environment(device):
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: str(importlib.metadata.version(name)) for name in
                     ("torch", "datasets", "tokenizers", "huggingface-hub")},
        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": str(device), "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "torch_threads": torch.get_num_threads(),
        "source": source_identity(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, gpu_memory_bytes=props.total_memory,
                      compute_capability=list(torch.cuda.get_device_capability(device)),
                      gpu_uuid=str(props.uuid) if getattr(props, "uuid", None) else None)
        try:
            result["nvidia_driver"] = subprocess.check_output(
                ["nvidia-smi", "--id=" + result["gpu_uuid"],
                 "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError, TypeError):
            result["nvidia_driver"] = None
    return result


class Trainer:
    """Reference runner. Only tests should pass synthetic loaders directly.

    stop_after_updates on fit() is a test hook for an orderly interruption; it
    does not change the configured schedule horizon or checkpoint compatibility.
    """

    def __init__(self, model_config: ModelConfig, config: TrainingConfig, loaders,
                 output_dir, *, resume=None, preparation_seconds=None, diagnostic=False):
        self.invocation = Invocation(config.timing_warmup_steps)
        self.preparation_seconds = preparation_seconds
        self.diagnostic = diagnostic
        self.phases = Phases()
        self.target_observations = {}
        self.prior_active_seconds = 0.0
        self.model_config, self.config, self.loaders = model_config, config, loaders
        self.device = validate_device(config)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if (loaders.metadata["sequence_length"] != model_config.sequence_length
                or loaders.metadata["vocab_size"] != model_config.vocab_size):
            raise ValueError("Model context/vocabulary does not match data metadata")
        if loaders.train.batch_size != config.batch_size:
            raise ValueError("Training batch size does not match loader")
        self.output_dir = Path(output_dir)
        validate_output_directory(self.output_dir, resume=resume)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(config.seed)
        self.model = DecoderOnlyTransformer(model_config).to(self.device)
        self.optimizer = make_optimizer(self.model, config)
        batches = math.ceil(len(loaders.train.dataset) / config.batch_size)
        self.total_updates = config.epochs * math.ceil(batches / config.gradient_accumulation_steps)
        if config.max_steps is not None:
            self.total_updates = min(self.total_updates, config.max_steps)
        self.scheduler = WarmupCosine(self.optimizer, config, self.total_updates)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        self.state = {"epoch": 0, "consumed_examples": 0, "updates": 0, "attempts": 0,
                      "skipped_updates": 0, "supervised_tokens": 0, "best_loss": None,
                      "timed_updates": 0, "timed_tokens": 0, "timed_seconds": 0.0}
        self.state["applied_supervised_tokens"] = 0
        self.pending_rng = None
        self.signature = data_signature(loaders.metadata)
        if resume is not None:
            self._restore(resume)
        record = {"model": asdict(model_config), "training": asdict(config),
                  "effective_batch_size": config.effective_batch_size,
                  "schedule_updates": self.total_updates,
                  "parameters": sum(p.numel() for p in self.model.parameters()),
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "resume": str(resume) if resume is not None else None}
        if resume is None:
            atomic_json(self.output_dir / "config.json", record)
            atomic_json(self.output_dir / "data.json", loaders.metadata)
            if "audit" in loaders.metadata:
                atomic_json(self.output_dir / "audit.json", loaders.metadata["audit"])
        else:
            self._log({"event": "resume", **record})
        env = environment(self.device)
        atomic_json(self.output_dir / "environment.json", env)
        atomic_json(self.output_dir / "invocations" / self.invocation.id / "environment.json", env)
        self.env = env
        self.memory_sampler = DeviceMemorySampler(env.get("gpu_uuid"))
        self._log({"event": "invocation_start", "measurement": self.invocation.snapshot(),
                   "preparation_seconds": preparation_seconds,
                   "resumed_from": str(resume) if resume else None,
                   "trajectory_start": self.state.copy()})
        atomic_json(self.output_dir / "invocations" / self.invocation.id / "measurement.json",
                    self.invocation.snapshot())

    def _restore(self, path):
        payload = load_checkpoint(path)
        validate_resume_settings(payload, self.model_config, self.config)
        if payload["data_signature"] != self.signature:
            raise ValueError("Data fingerprints, revisions or preprocessing policy changed")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.state = payload["state"]
        self.state.setdefault("applied_supervised_tokens", None)
        prior = payload.get("measurements", {})
        self.prior_active_seconds = prior.get("trajectory_active_seconds", 0.0)
        self.history_known = bool(prior.get("history_known", False))
        self.target_observations = prior.get("target_observations", {})
        self.pending_rng = payload["rng"]
        size = len(self.loaders.train.dataset)
        if not (0 <= self.state["epoch"] <= self.config.epochs
                and 0 <= self.state["consumed_examples"] <= size
                and self.state["updates"] == self.scheduler.completed_updates):
            raise ValueError("Invalid checkpoint counters")

    def _log(self, record, *, console=True):
        record = dict(record, updates=self.state["updates"], measurement_version=1,
                      invocation_id=self.invocation.id)
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if console:
            print(json.dumps(record, allow_nan=False), flush=True)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _checkpoint(self, name):
        self._sync()
        start = time.perf_counter()
        save_checkpoint(self.output_dir / name, {
            "version": CHECKPOINT_VERSION, "model_config": asdict(self.model_config),
            "training_config": asdict(self.config), "data_signature": self.signature,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
            "rng": capture_rng(), "state": self.state,
            "measurements": {"trajectory_active_seconds": self.prior_active_seconds +
                             time.perf_counter() - self.invocation.started,
                             "history_known": getattr(self, "history_known", True),
                             "timing_coverage": "checkpoint-boundary lower bound; excludes downtime",
                             "target_observations": self.target_observations},
        })
        self._log({"event": "checkpoint", "file": name,
                   "checkpoint_seconds": time.perf_counter() - start})

    def _validate(self):
        self._sync()
        start = time.perf_counter()
        result = evaluate(self.model, self.loaders.validation,
                          device=self.device, precision=self.config.precision)
        self._sync()
        elapsed = time.perf_counter() - self.invocation.started
        quality = {"training_processed_tokens": self.state["supervised_tokens"],
                   "training_applied_tokens": self.state["applied_supervised_tokens"],
                   "invocation_processed_tokens": self.invocation.processed_tokens,
                   "invocation_applied_tokens": self.invocation.applied_tokens,
                   "invocation_lifecycle_seconds": elapsed,
                   "invocation_training_seconds": self.invocation.training_seconds,
                   "trajectory_active_seconds": self.prior_active_seconds + elapsed
                   if getattr(self, "history_known", True) else None,
                   "trajectory_timing_coverage": "checkpoint-boundary lower bound across resume",
                   "validation_tokens": result["supervised_tokens"]}
        for target in self.config.validation_loss_targets:
            key = str(target)
            if key not in self.target_observations and result["loss"] <= target:
                self.target_observations[key] = dict(quality, loss=result["loss"],
                    updates=self.state["updates"], invocation_id=self.invocation.id)
        self._log({"event": "validation", **result, **quality,
                   "evaluation_seconds": time.perf_counter() - start})
        if self.state["best_loss"] is None or result["loss"] < self.state["best_loss"]:
            self.state["best_loss"] = result["loss"]
            self._checkpoint("best.pt")
        return result

    def _epoch_loader(self):
        c, state = self.config, self.state
        return DataLoader(
            self.loaders.train.dataset, batch_size=c.batch_size,
            sampler=EpochSampler(len(self.loaders.train.dataset), c.seed,
                                 state["epoch"], state["consumed_examples"]),
            collate_fn=self.loaders.train.collate_fn,
            num_workers=c.num_workers, pin_memory=self.device.type == "cuda",
            generator=torch.Generator().manual_seed((c.seed + state["epoch"] + 100000) % 2**63),
            drop_last=False,
        )

    def fit(self, *, stop_after_updates=None):
        self.memory_sampler.start()
        try:
            return self._fit(stop_after_updates=stop_after_updates)
        except BaseException as error:
            if isinstance(error, FloatingPointError):
                self.invocation.numerical_failures += 1
                message = str(error)
                kind = getattr(error, "kind", "validation_loss" if "validation" in message else "training_loss")
                self._log({"event": "numerical_failure", "kind": kind, "reason": message,
                           "measurement": self.invocation.snapshot(status="failed")})
            self._finish_measurements("failed", error=str(error))
            raise
        finally:
            if self.memory_sampler.process is not None:
                self.memory_sampler.stop()

    def _finish_measurements(self, status, *, error=None):
        # End the lifecycle clock before measurement/report serialization.
        result = self.invocation.snapshot(status=status)
        result.update(preparation_seconds=self.preparation_seconds, error=error,
                      timing_complete=status == "complete",
                      historical_measurements_known=getattr(self, "history_known", True),
                      quality_targets={str(t): self.target_observations.get(str(t))
                                       for t in self.config.validation_loss_targets},
                      quality_target_status={str(t): "reached" if str(t) in self.target_observations
                                             else "unreached" for t in self.config.validation_loss_targets})
        result["device_memory"] = self.memory_sampler.stop()
        samples = result["device_memory"].pop("samples")
        sample_path = self.output_dir / "invocations" / self.invocation.id / "device_memory_samples.json"
        atomic_json(sample_path, samples)
        result["device_memory"]["samples_path"] = sample_path.relative_to(self.output_dir).as_posix()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None
        result["allocator_measurement"] = {"status": "available", "scope": "training lifecycle"} if self.device.type == "cuda" else unavailable("CUDA allocator measurements are unavailable on CPU")
        atomic_json(self.output_dir / "invocations" / self.invocation.id / "measurement.json", result)
        self._log({"event": "invocation_end", "measurement": result})
        return result

    def _fit(self, *, stop_after_updates=None):
        c, state = self.config, self.state
        self.model.train()
        interval = {"loss_sum": 0.0, "tokens": 0, "slots": 0, "seconds": 0.0,
                    "attempts": 0, "successful_updates": 0}
        last_validation_update = None
        validation = None
        interrupted = False
        while state["epoch"] < c.epochs and state["updates"] < self.total_updates:
            if c.supervised_token_budget is not None and state["supervised_tokens"] >= c.supervised_token_budget:
                break
            iterator = iter(self._epoch_loader())
            # Iterator creation uses a dedicated generator. Restore dropout RNG last.
            if self.pending_rng is not None:
                restore_rng(self.pending_rng)
                self.pending_rng = None
            while state["consumed_examples"] < len(self.loaders.train.dataset):
                self._sync()
                start = time.perf_counter()
                with self.phases.range("data_acquisition"):
                    batches = list(itertools.islice(iterator, c.gradient_accumulation_steps))
                if not batches:
                    raise RuntimeError("DataLoader ended before the consumed cursor reached its end")
                result = optimizer_attempt(self.model, self.optimizer, self.scaler, self.scheduler,
                                           batches, c, self.device, phases=self.phases)
                skipped, lr = result["skipped"], result["learning_rate"]
                state["attempts"] += 1
                state["consumed_examples"] += result["examples"]
                state["supervised_tokens"] += result["supervised_tokens"]
                if skipped:
                    state["skipped_updates"] += 1
                else:
                    state["updates"] += 1
                    if state["applied_supervised_tokens"] is not None:
                        state["applied_supervised_tokens"] += result["supervised_tokens"]
                self._sync()
                elapsed = time.perf_counter() - start
                self.invocation.record(result["supervised_tokens"], elapsed, skipped)
                if result["nonfinite_gradients"]:
                    self.invocation.numerical_failures += 1
                    self._log({"event": "numerical_failure", "kind": "nonfinite_gradients",
                               "action": "skipped_update" if skipped else "updated",
                               "processed_tokens": result["supervised_tokens"],
                               "scaler_before": result["scaler_before"], "scaler_after": result["scaler_after"]})
                if self.invocation.attempts > c.timing_warmup_steps:
                    state["timed_updates"] += int(not skipped)
                    state["timed_tokens"] += result["supervised_tokens"]
                    state["timed_seconds"] += elapsed
                self._log({"event": "attempt", "attempt_seconds": elapsed,
                           "processed_tokens": result["supervised_tokens"],
                           "applied_tokens": 0 if skipped else result["supervised_tokens"],
                           "skipped": skipped, "scaler_before": result["scaler_before"],
                           "scaler_after": result["scaler_after"],
                           "nonfinite_gradients": result["nonfinite_gradients"],
                           "attempt": state["attempts"],
                           "invocation_attempt": self.invocation.attempts}, console=False)
                interval["loss_sum"] += result["loss_sum"]
                interval["tokens"] += result["supervised_tokens"]
                interval["slots"] += result["slots"]
                interval["seconds"] += elapsed
                interval["attempts"] += 1
                interval["successful_updates"] += int(not skipped)
                epoch_end = state["consumed_examples"] == len(self.loaders.train.dataset)
                if epoch_end:
                    state["epoch"] += 1
                    state["consumed_examples"] = 0
                budget_done = c.supervised_token_budget is not None and state["supervised_tokens"] >= c.supervised_token_budget
                done = state["updates"] >= self.total_updates or state["epoch"] >= c.epochs or budget_done
                interrupted = (stop_after_updates is not None
                               and state["updates"] >= stop_after_updates and not done)
                if skipped or state["updates"] % c.log_every == 0 or done or interrupted:
                    seconds = interval["seconds"]
                    self._log({
                        "event": "train", "loss": interval["loss_sum"] / interval["tokens"],
                        "learning_rate": lr, "supervised_tokens": interval["tokens"],
                        "total_supervised_tokens": state["supervised_tokens"],
                        "padding_utilisation": interval["tokens"] / interval["slots"],
                        "mean_update_seconds": seconds / interval["attempts"],
                        "supervised_tokens_per_second": interval["tokens"] / seconds,
                        "invocation_measurement": self.invocation.snapshot(),
                        "attempts": interval["attempts"],
                        "successful_updates": interval["successful_updates"],
                        "skipped_updates_total": state["skipped_updates"],
                        "epoch": state["epoch"], "consumed_examples": state["consumed_examples"],
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
                        if self.device.type == "cuda" else None,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)
                        if self.device.type == "cuda" else None,
                        "measurement": "training loop including data delivery and transfers",
                    })
                    interval = {key: 0 for key in interval}
                if not skipped and state["updates"] % c.eval_every == 0:
                    validation = self._validate()
                    last_validation_update = state["updates"]
                if not skipped and state["updates"] % c.checkpoint_every == 0:
                    self._checkpoint("latest.pt")
                if done or interrupted or epoch_end:
                    break
            del iterator
            if interrupted or done:
                break
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)
            self.pending_rng = None
        if not interrupted and last_validation_update != state["updates"]:
            validation = self._validate()
        self._checkpoint("latest.pt")
        measurement = self._finish_measurements("interrupted" if interrupted else "complete")
        summary = {
            "status": "interrupted" if interrupted else "complete",
            "measurements": measurement,
            "requested_supervised_token_budget": c.supervised_token_budget,
            "actual_supervised_tokens": state["supervised_tokens"],
            "token_budget_overshoot": max(0, state["supervised_tokens"] - c.supervised_token_budget)
                if c.supervised_token_budget is not None else None,
            "state": state.copy(), "final_validation": validation,
            "steady_state_supervised_tokens_per_second":
                measurement["warmed_processed_tokens_per_second"],
            "steady_state_mean_attempt_seconds": self.invocation.measured_seconds /
                self.invocation.measured_attempts if self.invocation.measured_attempts else None,
            "timing_warmup_attempts": c.timing_warmup_steps,
            "measurement": "training loop including data delivery and transfers; "
                           "excluding preprocessing, epoch iterator setup, evaluation and checkpoints",
        }
        atomic_json(self.output_dir / "summary.json", summary)
        return summary
