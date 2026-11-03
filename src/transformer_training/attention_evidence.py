"""Untimed dispatch evidence; profiling is never inside benchmark intervals."""

from dataclasses import asdict
from contextlib import nullcontext

import torch

from .checkpoint import capture_rng, restore_rng
from .evaluation import autocast_context, move_batch, next_token_loss
from .measurements import digest


def capture_attention(model, batches, *, device, precision, environment, backward=True):
    """Profile representative batch shapes outside measured timing to identify
    attention operators, optionally including backward but no optimiser step. Return
    evidence tied to settings and environment, reject contradictory forced-backend
    observations, and restore original gradients, RNG, and model mode.
    """
    if model.implementation.attention == "reference":
        return {"status": "reference", "observed": ["explicit"], "scope": "training forward/backward" if backward else "evaluation forward"}
    rng = capture_rng()
    gradients = [p.grad for p in model.parameters()]
    was_training = model.training
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    per_shape = []
    try:
        model.train(backward)
        for cpu_batch in batches:
            model.zero_grad(set_to_none=True)
            batch = move_batch(cpu_batch, device)
            with torch.profiler.profile(activities=activities) as profiler, (nullcontext() if backward else torch.no_grad()):
                with autocast_context(device, precision):
                    logits = model(batch["input_ids"], batch["attention_mask"])
                    loss = next_token_loss(logits, batch["labels"])
                if backward:
                    loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            names = sorted({event.key for event in profiler.key_averages()} |
                           {event.name for event in profiler.events()})
            observed = set()
            for name in names:
                lower = name.lower()
                if "scaled_dot_product" in lower:
                    for key in ("flash", "efficient", "cudnn", "math"):
                        if key in lower:
                            observed.add(key)
                if model.implementation.compile and ("triton_" in lower or "cpp_fused_" in lower):
                    observed.add("compiled_generated")
            per_shape.append({"shape": list(cpu_batch["input_ids"].shape),
                              "observed": sorted(observed), "operators": names})
            del logits, loss, batch
        observed = sorted({name for entry in per_shape for name in entry["observed"]})
        requested = model.implementation.sdpa_backend
        concrete = set(observed) - {"compiled_generated"}
        if requested != "auto" and concrete - {requested}:
            raise RuntimeError(f"Observed SDPA backend {concrete} differs from requested {requested}")
        verified = bool(per_shape) and all(entry["observed"] for entry in per_shape)
        return {"status": "observed" if verified else "unverified", "observed": observed,
                "requested": requested, "shapes": per_shape,
                "scope": "untimed training forward/backward after measurement; no optimizer step" if backward
                    else "untimed evaluation forward after metric computation",
                "identity": digest({"implementation": asdict(model.implementation),
                    "precision": precision, "environment": environment,
                    "shapes": [entry["shape"] for entry in per_shape]})}
    finally:
        for parameter, gradient in zip(model.parameters(), gradients):
            parameter.grad = gradient
        model.train(was_training)
        restore_rng(rng)
