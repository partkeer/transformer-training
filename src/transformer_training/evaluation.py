"""Shared next-token objective and valid-token-weighted validation."""

from contextlib import nullcontext

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def move_batch(batch, device):
    return {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}


def next_token_loss(logits, labels):
    """Sum already-aligned targets; never shift labels again."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=IGNORE_INDEX, reduction="sum")


@torch.no_grad()
def evaluate(model, loader, *, device="cpu", precision="fp32"):
    """Divide summed token loss by all nonignored targets, including real EOS.

    Averaging batch means would overweight short or heavily padded batches.
    """
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = 0
    try:
        for cpu_batch in loader:
            total_tokens += int((cpu_batch["labels"] != IGNORE_INDEX).sum())
            batch = move_batch(cpu_batch, device)
            with autocast_context(device, precision):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = next_token_loss(logits, batch["labels"])
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
