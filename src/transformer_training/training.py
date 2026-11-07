"""Single-GPU reference training, token-weighted accumulation and exact data resume."""

from contextlib import nullcontext
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
from .config import ImplementationConfig, ModelConfig, TrainingConfig
from .implementations import (compiler_snapshot, compilation_changed, execution_report,
                              configure_restored_optimizer, file_sha256, optimizer_flags)
from .evaluation import IGNORE_INDEX, autocast_context, evaluate, move_batch, next_token_loss
from .loss import model_loss_backend, validate_loss_backend
from .model import DecoderOnlyTransformer
from .measurements import Invocation, Phases, DeviceMemorySampler, source_identity, unavailable

CHECKPOINT_VERSION = 1


class NumericalFailure(FloatingPointError):
    """Keep the existing exception contract while identifying the numerical failure."""
    def __init__(self, kind, message):
        """Attach a machine-readable failure kind to a floating-point exception while
        retaining its human-readable error message.
        """
        super().__init__(message)
        self.kind = kind


def validate_output_directory(output_dir, *, resume=None):
    """Reject file paths, file-valued parents, and occupied fresh-run directories.

    Resume may reuse an existing directory. Check before data preparation and
    create no files or directories here.
    """
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


def validate_resume_settings(payload, model_config, config, implementation=None, *, diagnostic=False):
    """Reject incompatible checkpoint versions, model, or training settings.

    Allow the explicitly listed operational overrides; diagnostic runs may
    additionally change precision and implementation. Dataset identity is checked
    separately after preparation.
    """
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    current = asdict(config)
    saved = asdict(TrainingConfig(**payload["training_config"]))
    if diagnostic:
        current.pop("precision", None)
        saved.pop("precision", None)
    for key in ("num_workers", "log_every", "eval_every", "checkpoint_every",
                "validation_loss_targets", "supervised_token_budget", "timing_warmup_steps"):
        current.pop(key, None)
        saved.pop(key, None)
    if payload["model_config"] != asdict(model_config) or saved != current:
        raise ValueError("Incompatible model/training settings for resume")
    if not diagnostic and ImplementationConfig(**payload.get("implementation_config", {})) != (
        implementation or ImplementationConfig()
    ):
        raise ValueError("Incompatible implementation settings for resume")


def clip_gradients(parameters, max_norm, *, allow_nonfinite_gradients=False):
    """Clip finite gradients; leave AMP overflow untouched for GradScaler to skip.

    A nonfinite norm with finite gradient elements is a separate reduction
    overflow. Abort before clipping can zero those gradients or step the optimizer.
    Use the public PyTorch 2.6 clipping API, which checks the norm before mutation.
    Recompute a failed norm only to classify overflow and preserve AMP handling.
    """
    parameters = list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError:
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients:
            raise
        total_norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(gradient).to(gradients[0].device) for gradient in gradients
        ]))
        if torch.isfinite(total_norm):
            raise
        nonfinite = any(not torch.isfinite(gradient).all() for gradient in gradients)
        if allow_nonfinite_gradients and nonfinite:
            return total_norm
        raise NumericalFailure("nonfinite_gradients" if nonfinite else "gradient_norm_overflow",
                               "Nonfinite gradient norm; stopping before optimizer update") from None


def validate_device(config):
    """Return the requested torch device after checking CUDA availability, index, and
    native BF16 support. Reject mixed precision on CPU rather than silently
    substituting another execution mode.
    """
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
    """Seed Python and PyTorch CPU/CUDA generators and configure IEEE FP32 math without
    cuDNN autotuning. These settings support reproducible comparisons but do not
    promise cross-platform bitwise identity. Select the precision-control API
    available in PyTorch 2.6 or newer without mixing old and new controls.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class EpochSampler(Sampler):
    """Epoch permutation sliced at the trainer's consumed cursor, never prefetch state."""

    def __init__(self, size, seed, epoch, start=0):
        """Store dataset size, base seed, epoch, and consumed-example cursor; reject a
        cursor outside the dataset.
        """
        if not 0 <= start <= size:
            raise ValueError("Consumed cursor is outside the dataset")
        self.size, self.seed, self.epoch, self.start = size, seed, epoch, start

    def __iter__(self):
        """Return the epoch's seeded permutation starting at the consumed cursor,
        independently of global RNG state or how far DataLoader workers prefetched.
        """
        generator = torch.Generator().manual_seed((self.seed + self.epoch) % 2**63)
        order = torch.randperm(self.size, generator=generator)
        # Tensor iteration unbinds millions of scalar tensors at once. A Python
        # integer list is much smaller and matches PyTorch RandomSampler's approach.
        return iter(order[self.start:].tolist())

    def __len__(self):
        """Return the number of examples remaining after the consumed cursor, including
        any final partial batch.
        """
        return self.size - self.start


class WarmupCosine:
    """LR for the next successful update; skipped FP16 updates keep the same LR."""

    def __init__(self, optimizer, config, total_updates):
        """Create a schedule over a positive number of successful updates and
        immediately apply the rate for the first update.
        """
        if total_updates < 1:
            raise ValueError("Schedule requires at least one update")
        self.optimizer, self.config = optimizer, config
        self.total_updates = total_updates
        self.completed_updates = 0
        self._apply()

    def learning_rate(self, index):
        """Return the rate for a zero-based update index: optional linear warm-up
        followed by cosine decay. A single-update run uses the peak rate; indices at
        or beyond the final update use the minimum rate.
        """
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
        """Set every optimiser parameter group's learning rate for the next successful
        update without advancing the schedule.
        """
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate(self.completed_updates)

    def step(self):
        """Advance the successful-update counter once and apply the next rate; callers
        omit this when AMP skips an update.
        """
        self.completed_updates += 1
        self._apply()

    def state_dict(self):
        """Return the completed-update count and fixed horizon needed to resume the
        learning-rate trajectory.
        """
        return {"completed_updates": self.completed_updates, "total_updates": self.total_updates}

    def load_state_dict(self, state):
        """Restore the completed-update count and current rate, rejecting a checkpoint
        whose schedule horizon differs from this run.
        """
        if state["total_updates"] != self.total_updates:
            raise ValueError("Checkpoint schedule horizon changed")
        self.completed_updates = state["completed_updates"]
        self._apply()


def make_optimizer(model, config, implementation=None):
    # model.parameters() removes aliases, so tied embeddings appear only once.
    """Return AdamW with weight decay on matrices and embeddings but none on biases or
    normalisation vectors. Deduplicate tied parameters through model.parameters()
    and use the requested standard, foreach, or fused implementation.
    """
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps,
        **optimizer_flags(implementation or getattr(model, "implementation", ImplementationConfig())),
    )


def accumulate_gradients(model, batches, *, device="cpu", precision="fp32", scaler=None,
                         phases=None, distributed=None):
    """Accumulate gradients weighted by all supervised tokens in the batch group.

    Divide each microbatch's summed loss by the group's total valid-target count,
    so short or padded batches are not overweighted. Optionally scale for AMP,
    reject empty/nonfinite objectives, and return loss/token/example statistics
    without retaining microbatch graphs. Inherit the model's loss backend; plain
    modules use PyTorch. An explicit distributed context normalises against global
    tokens, compensates for DDP gradient averaging, and synchronises only the last
    forward/backward pair. Dummy forwards carry no supervision or workload counts.
    The caller clears gradients and steps.
    """
    phases = phases or Phases()
    with phases.range("data_preparation"):
        tokens = sum(int((batch["labels"] != IGNORE_INDEX).sum()) for batch in batches)
        if distributed is not None:
            tokens = int(distributed.sum([tokens])[0])
    if not tokens:
        raise ValueError("Accumulation group contains no supervised tokens")
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    for index, cpu_batch in enumerate(batches):
        with phases.range("transfer"):
            batch = move_batch(cpu_batch, device)
        context = (distributed.accumulation_context(model, index == len(batches) - 1)
                   if distributed is not None else nullcontext())
        with context:
            with phases.range("forward_loss"), autocast_context(device, precision):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = next_token_loss(logits, batch["labels"], backend=model_loss_backend(model))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; stopping before optimizer update")
            normalised = loss / tokens
            if distributed is not None and distributed.world_size > 1:
                normalised = normalised * distributed.world_size
            with phases.range("backward"):
                (scaler.scale(normalised) if scaler is not None else normalised).backward()
        loss_sum += loss.detach().double()
        del logits, loss, normalised, batch
    examples = sum(int((~batch['_scaling_dummy']).sum()) if '_scaling_dummy' in batch
                   else len(batch['labels']) for batch in batches)
    slots = sum((int((~batch['_scaling_dummy']).sum()) * batch['labels'].shape[1])
                if '_scaling_dummy' in batch else batch['labels'].numel() for batch in batches)
    stats = [loss_sum.item(), slots, examples]
    if distributed is not None:
        stats = distributed.sum(stats)
    return {"loss_sum": stats[0], "supervised_tokens": tokens,
            "slots": int(stats[1]), "examples": int(stats[2])}


def optimizer_attempt(model, optimizer, scaler, scheduler, batches, config, device, *,
                      phases=None, distributed=None):
    """Attempt one shared accumulated optimiser update and return its statistics.

    Clear gradients, backpropagate token-weighted loss, unscale and clip, then
    invoke the scaler/optimiser. Advance the schedule only when AMP applies the
    update; return skip/scale metadata so callers account for consumed work.
    Training, benchmarks, and profilers all use this path. Opted-in distributed
    attempts return global workload statistics and check rank agreement on AMP
    skip/scaler/scheduler state before proceeding to another attempt.
    """
    phases = phases or Phases()
    with phases.range("zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    result = accumulate_gradients(model, batches, device=device, precision=config.precision,
                                  scaler=scaler, phases=phases, distributed=distributed)
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
        if distributed is not None:
            distributed.check_update(skipped, scale, scheduler.completed_updates)
    return dict(result, skipped=skipped, learning_rate=lr, scaler_before=previous_scale,
                scaler_after=scale, nonfinite_gradients=skipped)


def data_signature(metadata):
    # Worker and hardware settings may change on resume; corpus identity may not.
    """Extract corpus, tokenizer, selection, and preprocessing identity for checkpoint
    compatibility while leaving operational worker and hardware settings out.
    """
    keys = ("dataset", "revision", "tokenizer", "tokenizer_revision", "sequence_length",
            "max_stories", "seed", "vocab_size", "eos_id", "pad_id", "ignore_index",
            "exclude_validation_overlap", "policy", "splits", "audit")
    return {key: metadata[key] for key in keys if key in metadata}


def environment(device, *, source_exclude=()):
    """Return software versions, source identity, precision controls, and device
    details for interpreting measurements. Include CUDA GPU properties and attempt
    to read the driver version; an unavailable driver query yields None. Report the
    optional Triton distribution without importing it or requiring its presence.
    Express the legacy PyTorch 2.6 TF32 switch using the same precision label as
    newer releases so reports describe the applied math policy consistently.
    Optional source_exclude paths omit generated benchmark outputs from Git
    status/content identity, including custom nonignored destinations.
    """
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: str(importlib.metadata.version(name)) for name in
                     ("torch", "datasets", "tokenizers", "huggingface-hub")},
        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": str(device), "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_fp32_precision": (
            torch.backends.cuda.matmul.fp32_precision
            if hasattr(torch.backends.cuda.matmul, "fp32_precision") else
            "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"),
        "torch_threads": torch.get_num_threads(),
        "source": source_identity(exclude_paths=source_exclude),
    }
    try:
        result["packages"]["triton"] = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        result["packages"]["triton"] = None
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
                 output_dir, *, resume=None, preparation_seconds=None, diagnostic=False,
                 implementation=None):
        """Validate prepared loaders and execution settings, create run artifacts, and
        construct the model, optimiser, scheduler, scaler, and counters. Validate
        the loss runtime before output creation. Optionally
        restore a compatible checkpoint while deferring RNG restoration until loader
        setup.
        """
        self.invocation = Invocation(config.timing_warmup_steps)
        self.preparation_seconds = preparation_seconds
        self.diagnostic = diagnostic
        self.implementation = implementation or ImplementationConfig(**(
            load_checkpoint(resume).get("implementation_config", {}) if resume else {}))
        self.checkpoint_sha256 = file_sha256(resume) if resume else None
        self.compilation_contaminated = False
        self.compile_attempts = []
        self.representative_batches = {}
        self.first_forward_backward_attempt_seconds = None
        self.warmup_seconds = 0.0
        self.phases = Phases()
        self.target_observations = {}
        self.prior_active_seconds = 0.0
        self.model_config, self.config, self.loaders = model_config, config, loaders
        self.device = validate_device(config)
        validate_loss_backend(self.implementation.loss_backend, self.device, model_config.vocab_size)
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
        self.model = DecoderOnlyTransformer(model_config, self.implementation).to(self.device)
        self.optimizer = make_optimizer(self.model, config, self.implementation)
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
                  "implementation": asdict(self.implementation),
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
        """Restore compatible model, optimiser, schedule, scaler, counters, and
        measurement history. Verify data identity and counter bounds, and defer RNG
        restoration so iterator setup cannot disturb resumed dropout state.
        """
        payload = load_checkpoint(path)
        validate_resume_settings(payload, self.model_config, self.config, self.implementation,
                                 diagnostic=self.diagnostic)
        if payload["data_signature"] != self.signature:
            raise ValueError("Data fingerprints, revisions or preprocessing policy changed")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        # load_state_dict restores saved implementation flags as well as moments.
        configure_restored_optimizer(self.optimizer, self.implementation)
        self.scheduler.load_state_dict(payload["scheduler"])
        saved_precision = payload["training_config"]["precision"]
        if saved_precision == self.config.precision:
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
        """Append a JSON metrics record tagged with the current update and invocation,
        optionally also printing it. Reject nonfinite JSON values to keep artifacts
        machine-readable.
        """
        record = dict(record, updates=self.state["updates"], measurement_version=1,
                      invocation_id=self.invocation.id)
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if console:
            print(json.dumps(record, allow_nan=False), flush=True)

    def _sync(self):
        """Wait for outstanding work on this trainer's CUDA device; do nothing for CPU
        execution.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _checkpoint(self, name):
        """Synchronise and atomically save complete resumable state under the supplied
        filename, including RNG, data cursor, and measurement history. Call only at
        accumulation boundaries and log checkpoint duration separately.
        """
        self._sync()
        start = time.perf_counter()
        save_checkpoint(self.output_dir / name, {
            "version": CHECKPOINT_VERSION, "model_config": asdict(self.model_config),
            "training_config": asdict(self.config), "data_signature": self.signature,
            "implementation_config": asdict(self.implementation),
            "execution": execution_report(self.model),
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
        """Evaluate token-weighted validation loss, record first crossings of quality
        targets, and save best.pt when loss improves. Return the evaluator result
        and log validation time outside training-attempt timing.
        """
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
        """Return a loader for the current epoch starting at the consumed cursor. Use
        separate seeded sampler and worker generators so prefetch and iterator setup
        do not alter resumed training order or dropout RNG.
        """
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
        """Run training with device-memory sampling and return the final summary.
        Record failed measurements before re-raising errors, always stop sampling,
        and allow an optional successful-update cap for orderly interruption tests.
        """
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
        """Close the invocation measurement window, stop memory sampling, and write
        artifacts for the given status. Preserve unavailable values explicitly and
        collect optional attention-dispatch evidence after timing has ended.
        """
        result = self.invocation.snapshot(status=status)
        result.update(preparation_seconds=self.preparation_seconds, error=error,
                      execution=execution_report(self.model),
                      compilation_contaminated=self.compilation_contaminated,
                      compilation_attempts=self.compile_attempts,
                      first_forward_backward_attempt_seconds=self.first_forward_backward_attempt_seconds,
                      warmup_seconds=self.warmup_seconds,
                      timings_eligible_for_comparison=not self.compilation_contaminated and
                          (not self.implementation.compile or compiler_snapshot(True)["status"] == "available"),
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
        if status in {"complete", "interrupted"} and self.representative_batches and self.implementation.attention == "sdpa":
            from .attention_evidence import capture_attention
            self.model.attention_evidence = capture_attention(self.model, list(self.representative_batches.values()),
                device=self.device, precision=self.config.precision, environment=self.env)
            result["execution"]["attention_evidence"] = self.model.attention_evidence
        atomic_json(self.output_dir / "invocations" / self.invocation.id / "measurement.json", result)
        self._log({"event": "invocation_end", "measurement": result})
        return result

    def _fit(self, *, stop_after_updates=None):
        """Execute accumulated attempts until the epoch, update, or processed-token
        budget is reached. Advance the consumed cursor even on AMP skips, validate
        and checkpoint at boundaries, and return a persisted summary; the last
        accumulation group can exceed the token budget.
        """
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
                for batch in batches:
                    self.representative_batches.setdefault(tuple(batch["input_ids"].shape), batch)
                compiler_before = compiler_snapshot(c_enabled := self.implementation.compile)
                result = optimizer_attempt(self.model, self.optimizer, self.scaler, self.scheduler,
                                           batches, c, self.device, phases=self.phases)
                compiler_after = compiler_snapshot(c_enabled)
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
                if self.invocation.attempts == 1:
                    self.first_forward_backward_attempt_seconds = elapsed
                if self.invocation.attempts <= c.timing_warmup_steps:
                    self.warmup_seconds += elapsed
                if compilation_changed(compiler_before, compiler_after):
                    entry = {"attempt": self.invocation.attempts, "attempt_seconds": elapsed,
                             "compiler": compiler_after}
                    self.compile_attempts.append(entry)
                    self.compilation_contaminated |= self.invocation.attempts > c.timing_warmup_steps
                    self._log({"event": "compilation", **entry})
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
