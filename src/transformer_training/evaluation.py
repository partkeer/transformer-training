"""Shared next-token objective and valid-token-weighted validation."""

from contextlib import nullcontext

import torch
from .loss import IGNORE_INDEX, cross_entropy_sum, model_loss_backend, triton_launch_settings


def autocast_context(device, precision):
    """Return a no-op context for FP32 or a device-specific autocast context for
    BF16/FP16. The caller validates precision names and hardware support.
    """
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def move_batch(batch, device):
    """Return a batch dictionary with tensors moved to the requested device, requesting
    nonblocking copies so pinned CPU inputs can transfer asynchronously where
    supported.
    """
    return {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}


def next_token_loss(logits, labels, *, backend="pytorch"):
    """Return summed cross-entropy over already-aligned next-token targets, ignoring
    -100 padding. Never shift labels again; callers divide by the total supervised-
    token count for their accumulation or evaluation scope. The default preserves
    PyTorch behaviour; explicit Triton selection requires contiguous aligned inputs
    and returns an FP32 sum. Do not silently copy unsupported Triton layouts.
    """
    if backend == "triton" and (not logits.is_contiguous() or not labels.is_contiguous()):
        raise ValueError("Triton logits and labels must be contiguous; select --loss-backend pytorch.")
    if backend == "triton" and (logits.ndim < 2 or labels.shape != logits.shape[:-1]):
        raise ValueError("Triton labels must match the leading logits dimensions.")
    if backend == "triton":
        triton_launch_settings(logits.size(-1))
    return cross_entropy_sum(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), backend=backend)


@torch.no_grad()
def evaluate(model, loader, *, device="cpu", precision="fp32", shape_examples=None):
    """Return mean next-token loss and supervised-token count over the loader,
    including genuine EOS and excluding -100 padding. Sum losses before dividing to
    avoid overweighting short batches, restore model mode even on failure, and
    optionally retain a CPU batch for each observed shape. Inherit the model's loss
    backend, defaulting modules without implementation settings to PyTorch.
    """
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = 0
    try:
        for cpu_batch in loader:
            if shape_examples is not None:
                shape_examples.setdefault(tuple(cpu_batch["input_ids"].shape), cpu_batch)
            total_tokens += int((cpu_batch["labels"] != IGNORE_INDEX).sum())
            batch = move_batch(cpu_batch, device)
            with autocast_context(device, precision):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = next_token_loss(logits, batch["labels"], backend=model_loss_backend(model))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            total_loss += loss.double()
            del logits, loss, batch
        if total_tokens == 0:
            raise ValueError("Validation contains no supervised tokens")
        return {"loss": total_loss.item() / total_tokens,
                "supervised_tokens": total_tokens}
    finally:
        model.train(was_training)
