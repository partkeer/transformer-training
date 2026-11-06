"""Shared execution options and version-isolated compiler instrumentation.

Compiler metrics use the pinned PyTorch diagnostic API. Absence of telemetry
never turns an unverified compiled interval into an eligible warmed result.
"""

import argparse
from dataclasses import asdict, fields
import hashlib
from pathlib import Path

from .config import ImplementationConfig


def add_implementation_arguments(parser, *, evaluation=False):
    """Add attention, loss backend, and compilation options to a parser, plus
    optimiser/checkpointing switches for training. Leave defaults unset so omitted
    flags can inherit checkpoint settings.
    """
    parser.add_argument("--attention", choices=("reference", "sdpa"))
    parser.add_argument("--loss-backend", choices=("pytorch", "triton"),
                        help="Summed cross-entropy implementation; Triton is explicit and never falls back")
    parser.add_argument("--sdpa-backend", choices=("auto", "math", "flash", "efficient", "cudnn"))
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune",
                                                  "max-autotune-no-cudagraphs"))
    if not evaluation:
        parser.add_argument("--optimizer-implementation", choices=("standard", "foreach", "fused"))
        parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=None)


def resolve_implementation(args, saved=None):
    """Return validated execution settings by combining saved values with explicit
    flags. Choosing reference attention or disabling compilation also clears
    incompatible inherited backend or compile-mode settings unless explicitly
    overridden.
    """
    values = dict((saved or {}).get("implementation_config", {}))
    for field in fields(ImplementationConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            values[field.name] = value
    # Choosing the reference explicitly also clears an inherited forced backend.
    if getattr(args, "attention", None) == "reference" and getattr(args, "sdpa_backend", None) is None:
        values["sdpa_backend"] = "auto"
    if getattr(args, "compile", None) is False and getattr(args, "compile_mode", None) is None:
        values["compile_mode"] = "default"
    return ImplementationConfig(**values)


def optimizer_flags(implementation):
    """Return explicit foreach and fused flags for the requested AdamW implementation,
    disabling both for the standard path.
    """
    return {"foreach": implementation.optimizer_implementation == "foreach",
            "fused": implementation.optimizer_implementation == "fused"}


def configure_restored_optimizer(optimizer, implementation):
    """Reapply the requested algorithm after loading shared Adam moments.

    PyTorch casts step counters using the *saved* group flags during load.
    Switching to fused therefore needs explicit device/dtype placement too.
    """
    import torch

    flags = optimizer_flags(implementation)
    # Keep GradScaler's protocol explicit after restoring and overriding groups.
    optimizer._step_supports_amp_scaling = flags["fused"]
    for group in optimizer.param_groups:
        group.update(flags)
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            step = state.get("step")
            if step is not None:
                on_device = flags["fused"] or group.get("capturable", False)
                state["step"] = step.to(
                    device=parameter.device if on_device else "cpu",
                    dtype=torch.float32 if on_device else step.dtype)


def file_sha256(path):
    """Return a file's SHA-256 checksum using bounded chunks, allowing large checkpoint
    and artifact identities without loading the whole file into memory.
    """
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def compiler_snapshot(enabled):
    """Return JSON-safe compiler events when telemetry is available, or an explicit
    disabled/unavailable status. Missing telemetry is not evidence that an interval
    contains no compilation.
    """
    if not enabled:
        return {"status": "disabled", "events": []}
    try:
        from torch._dynamo.utils import get_compilation_metrics

        events = []
        for metric in get_compilation_metrics():
            events.append({key: getattr(metric, key, None) for key in
                ("compile_id", "is_forward", "start_time", "end_time_us", "entire_frame_compile_time_s",
                 "duration_us", "backend_compile_time_s", "backward_cumulative_compile_time_us",
                 "fail_type", "fail_reason", "graph_node_count", "restart_reasons", "recompile_reason",
                 "num_graph_breaks", "cudagraph_skip_reason", "runtime_cudagraphify_time_us")})
        # Some diagnostic fields are sets or compiler-specific objects.
        return {"status": "available", "events": [
            {key: value if value is None or isinstance(value, (str, int, float, bool)) else str(value)
             for key, value in event.items()} for event in events]}
    except (ImportError, AttributeError):
        return {"status": "unavailable", "events": []}


def compilation_changed(before, after):
    """Return whether compiler status or event contents changed between snapshots,
    identifying intervals that may include compilation work.
    """
    return before["status"] != after["status"] or before["events"] != after["events"]


def execution_report(model):
    """Return requested execution settings, compile-wrapper time, compiler telemetry,
    and observed attention evidence; label uncaptured SDPA dispatch as unverified.
    Loss metadata describes configured execution and deterministic launch settings,
    not profiler-observed dispatch. Successful Triton calls cannot fall back.
    """
    from .loss import triton_launch_settings

    loss = {"backend": model.implementation.loss_backend, "fallback": "explicit_pytorch_selection",
            "evidence": "configuration; not a profiler observation"}
    if model.implementation.loss_backend == "triton":
        loss["launch"] = triton_launch_settings(model.config.vocab_size)
    return {"requested": asdict(model.implementation), "loss": loss,
            "compile_wrapper_seconds": model.compile_wrapper_seconds,
            "compiler": compiler_snapshot(model.implementation.compile),
            "attention_evidence": getattr(model, "attention_evidence", {
                "status": "unverified" if model.implementation.attention == "sdpa" else "reference",
                "observed": [], "reason": "Use a diagnostic capture to observe dispatched operators"})}
