"""Bounded performance trials using the same training attempt implementation."""

from contextlib import nullcontext
from dataclasses import asdict
import itertools
import math
import os
from pathlib import Path
import statistics
import time

import torch

from .checkpoint import atomic_json, restore_rng
from .measurements import Phases, digest, unavailable
from .memory_profile import MemoryAdapter, SavedActivations, static_memory
from .training import optimizer_attempt
from .traces import analyse_trace, load_chrome
from .implementations import compiler_snapshot, compilation_changed, execution_report
from .config import ImplementationConfig


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values):
    return {"count": len(values), "median": statistics.median(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "p50": percentile(values, .5), "p90": percentile(values, .9),
            "p95": percentile(values, .95), "p99": percentile(values, .99)}


def comparison_identity(trainer, mode):
    config = asdict(trainer.config)
    for key in ("log_every", "eval_every", "checkpoint_every", "validation_loss_targets",
                "timing_warmup_steps", "supervised_token_budget"):
        config.pop(key, None)
    env = trainer.env.copy()
    return {"model": asdict(trainer.model_config), "training": config,
            "implementation": asdict(trainer.implementation),
            "checkpoint_sha256": trainer.checkpoint_sha256,
            "data": trainer.signature, "schedule_updates": trainer.scheduler.total_updates,
            "starting_state": trainer.state.copy(), "environment": env, "input_mode": mode}


def field_differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [path for key in sorted(left.keys() | right.keys())
                for path in field_differences(left.get(key), right.get(key),
                                              f"{prefix}.{key}" if prefix else key)]
    return [] if left == right else [prefix]


def compare_results(baseline, candidate, *, allowed_changes=()):
    left, right = baseline["comparison"], candidate["comparison"]
    differences = field_differences(left, right)
    allowed = {"training.precision", *["implementation." + key for key in asdict(ImplementationConfig())]}
    if set(allowed_changes) - allowed:
        raise ValueError("Only precision and implementation fields can be declared experimental changes")
    unexpected = sorted(set(differences) - set(allowed_changes))
    if unexpected:
        return {"status": "conditions_differ", "differences": differences, "speedup": None,
                "unexpected_differences": unexpected,
                "reason": "Review changed conditions before interpreting a performance difference"}
    if not baseline.get("timings_eligible_for_comparison", True) or not candidate.get("timings_eligible_for_comparison", True):
        return {"status": "timings_ineligible", "differences": differences, "speedup": None}
    if "implementation.sdpa_backend" in differences:
        for report in (baseline, candidate):
            for trial in report.get("trials", []):
                if trial.get("comparison", {}).get("implementation", {}).get("attention") == "sdpa" and (
                    trial.get("execution", {}).get("attention_evidence", {}).get("status") != "observed"
                ):
                    return {"status": "backend_unverified", "differences": differences, "speedup": None}
    a = baseline["throughput"]["median"]
    b = candidate["throughput"]["median"]
    ma, mb = baseline.get("peak_allocated_bytes", {}).get("median"), candidate.get("peak_allocated_bytes", {}).get("median")
    ta, tb = baseline.get("latency_seconds", {}).get("median"), candidate.get("latency_seconds", {}).get("median")
    return {"status": "matched", "differences": differences, "speedup": b / a if a and b is not None else None,
            "memory_reduction": 1 - mb / ma if ma and mb is not None else None,
            "update_time_change_percent": 100 * (tb / ta - 1) if ta and tb is not None else None}


def run_trial(trainer, *, warmup=10, measured=50, mode="loader", backend="benchmark"):
    """Mutates only this isolated trial's model; never saves training checkpoints."""
    if warmup < 0 or measured < 1 or mode not in {"loader", "fixed"}:
        raise ValueError("Invalid trial workload")
    output, c, state = trainer.output_dir, trainer.config, trainer.state
    comparison = comparison_identity(trainer, mode)
    comparison["benchmark_protocol"] = {"warmup": warmup, "measured": measured}
    initial_state = state.copy()
    invocation = trainer.invocation
    invocation.warmup = warmup
    trainer.model.train()
    iterator, fixed = None, None
    phases = Phases(enabled=backend != "benchmark", nvtx=backend in {"nsys", "ncu"})
    records = []
    warmup_records = []
    representative_batches = {}
    adapter = MemoryAdapter(trainer.device)
    saved = SavedActivations(trainer.model)
    profiler = None
    sampler = trainer.memory_sampler.start()
    trace_path = output / "trace.json"

    def group():
        nonlocal iterator, fixed
        if state["epoch"] >= c.epochs or state["updates"] >= trainer.total_updates:
            raise ValueError("Insufficient data/schedule remains for all warm-up and measured attempts")
        if mode == "fixed" and fixed is not None:
            return fixed
        if iterator is None:
            iterator = iter(trainer._epoch_loader())
            if trainer.pending_rng is not None:
                restore_rng(trainer.pending_rng)
                trainer.pending_rng = None
        batches = list(itertools.islice(iterator, c.gradient_accumulation_steps))
        if not batches:
            raise ValueError("Insufficient data remains for trial")
        if mode == "fixed":
            fixed = batches
        return batches

    def attempt():
        nonlocal iterator
        trainer._sync()
        started = time.perf_counter()
        with phases.range("update"):
            with phases.range("data_acquisition"):
                batches = group()
            for batch in batches:
                representative_batches.setdefault(tuple(batch["input_ids"].shape), batch)
            compiler_before = compiler_snapshot(trainer.implementation.compile)
            result = optimizer_attempt(trainer.model, trainer.optimizer, trainer.scaler,
                                       trainer.scheduler, batches, c, trainer.device, phases=phases)
            trainer._sync()
        seconds = time.perf_counter() - started
        compiler_after = compiler_snapshot(trainer.implementation.compile)
        state["attempts"] += 1
        state["updates"] += int(not result["skipped"])
        state["skipped_updates"] += int(result["skipped"])
        state["supervised_tokens"] += result["supervised_tokens"]
        if state["applied_supervised_tokens"] is not None and not result["skipped"]:
            state["applied_supervised_tokens"] += result["supervised_tokens"]
        if mode == "loader":
            state["consumed_examples"] += result["examples"]
            if state["consumed_examples"] == len(trainer.loaders.train.dataset):
                state["epoch"] += 1
                state["consumed_examples"] = 0
                iterator = None
        invocation.record(result["supervised_tokens"], seconds, result["skipped"])
        invocation.numerical_failures += int(result["nonfinite_gradients"])
        return dict(result, seconds=seconds, compilation_detected=compilation_changed(compiler_before, compiler_after))

    history_status = None
    try:
        # Warm-up is outside every profiler/counter capture, but uses the same state progression.
        original_enabled, original_nvtx = phases.enabled, phases.nvtx
        phases.enabled = phases.nvtx = False
        for _ in range(warmup):
            warmup_records.append(attempt())
        phases.enabled, phases.nvtx = original_enabled, original_nvtx
        phases.host_seconds.clear()
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        if backend in {"pytorch", "memory"}:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if trainer.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(activities=activities,
                record_shapes=True, with_stack=backend == "memory", profile_memory=backend == "memory")
        if backend == "memory":
            history_status = adapter.start()
        with (profiler if profiler else nullcontext()), (saved if backend == "memory" else nullcontext()):
            if backend in {"nsys", "ncu"}:
                if trainer.device.type != "cuda":
                    raise ValueError("Nsight capture requires CUDA")
                with torch.cuda.device(trainer.device):
                    torch.cuda.cudart().cudaProfilerStart()
            try:
                trainer._sync()
                measured_start = time.perf_counter()
                for _ in range(measured):
                    records.append(attempt())
                    if profiler:
                        profiler.step()
                trainer._sync()
                invocation.measured_seconds = time.perf_counter() - measured_start
            finally:
                if backend in {"nsys", "ncu"} and trainer.device.type == "cuda":
                    with torch.cuda.device(trainer.device):
                        torch.cuda.cudart().cudaProfilerStop()
        measurement = invocation.snapshot(status="complete")
        measurement.update(kind=backend, timing_complete=True,
            scope="diagnostic trial setup, warm-up and measured attempts; no validation/checkpointing")
        result = {"version": 1, "kind": backend, "input_mode": mode, "process_id": os.getpid(),
                  "measurement": measurement, "comparison": comparison,
                  "comparison_sha256": digest(comparison), "starting_state": initial_state,
                  "measured_attempts": records,
                  "latency_seconds": distribution([r["seconds"] for r in records]),
                  "host_phase_seconds": phases.host_seconds,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type == "cuda" else None,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(trainer.device) if trainer.device.type == "cuda" else None,
                  "memory_peak_scope": "measured attempts after warm-up",
                  "allocator_measurement": {"status": "available"} if trainer.device.type == "cuda"
                      else unavailable("CUDA allocator measurements are unavailable on CPU"),
                  "static_memory_end": static_memory(trainer.model, trainer.optimizer),
                  "device_memory": sampler.stop(),
                  "timings_eligible_for_comparison": backend == "benchmark" and
                      not any(r["compilation_detected"] for r in records) and
                      (not trainer.implementation.compile or compiler_after_measurement_available(trainer)),
                  "execution": execution_report(trainer.model),
                  "warmup_records": warmup_records,
                  "warmup_seconds": sum(r["seconds"] for r in warmup_records),
                  "first_forward_backward_attempt_seconds": (warmup_records or records)[0]["seconds"],
                  "compilation_contaminated": any(r["compilation_detected"] for r in records),
                  "timing_scope": "complete training attempts including delivery and transfers; excludes validation/checkpoints",
                  "warmup_attempts": warmup, "captured_attempts": measured}
        atomic_json(output / "device_memory_samples.json", result["device_memory"].pop("samples"))
        result["device_memory"]["samples_path"] = "device_memory_samples.json"
        if profiler:
            profiler.export_chrome_trace(str(trace_path))
            result["trace"] = trace_path.name
            result["timeline"] = analyse_trace(load_chrome(trace_path))
            result["operators"] = [{"name": e.key, "calls": e.count,
                "self_cpu_us": e.self_cpu_time_total,
                "self_device_us": getattr(e, "self_device_time_total", None),
                "shapes": e.input_shapes} for e in profiler.key_averages(group_by_input_shape=True)]
            result["operators"].sort(key=lambda e: e["self_device_us"] or e["self_cpu_us"], reverse=True)
        if backend == "memory":
            result["memory"] = adapter.finish(profiler, output)
            result["memory"]["history_start"] = history_status
            result["memory"]["saved_for_backward"] = saved.report()
        # The capture is after all timed intervals, peak queries and sampler stop.
        from .attention_evidence import capture_attention
        evidence = (capture_attention(trainer.model, list(representative_batches.values()),
            device=trainer.device, precision=c.precision, environment=trainer.env)
            if backend not in {"nsys", "ncu"} else {"status": "unverified", "observed": [],
                "reason": "External Nsight capture owns profiling; use a separate PyTorch capture for dispatch evidence"})
        trainer.model.attention_evidence = evidence
        result["execution"]["attention_evidence"] = evidence
        measurement.update(peak_allocated_bytes=result["peak_allocated_bytes"],
                           peak_reserved_bytes=result["peak_reserved_bytes"],
                           allocator_measurement=result["allocator_measurement"],
                           device_memory=result["device_memory"])
        atomic_json(output / "invocations" / invocation.id / "measurement.json", measurement)
        trainer._log({"event": "invocation_end", "measurement": measurement})
        atomic_json(output / "trial.json", result)
        return result
    finally:
        sampler.stop()
        if adapter.history:
            torch.cuda.memory._record_memory_history(enabled=None, device=trainer.device)
        if iterator is not None:
            del iterator


def aggregate_trials(trials):
    if not trials:
        raise ValueError("No trials to aggregate")
    if any(t["kind"] != "benchmark" for t in trials):
        raise ValueError("Profiled trials cannot be aggregated as benchmark timings")
    if any(t["comparison"] != trials[0]["comparison"] for t in trials[1:]):
        raise ValueError("Trial conditions differ; refusing to combine measurements")
    comparison = dict(trials[0]["comparison"], repeats=len(trials))
    return {"version": 1, "kind": "benchmark", "status": "complete",
            "timings_eligible_for_comparison": all(t.get("timings_eligible_for_comparison", True) for t in trials),
            "comparison": comparison, "comparison_sha256": digest(comparison),
            "throughput": distribution([t["measurement"]["warmed_useful_tokens_per_second"] for t in trials]),
            "processed_throughput": distribution([t["measurement"]["warmed_processed_tokens_per_second"] for t in trials]),
            "latency_seconds": distribution([r["seconds"] for t in trials for r in t["measured_attempts"]]),
            "peak_allocated_bytes": distribution([t["peak_allocated_bytes"] for t in trials if t["peak_allocated_bytes"] is not None]),
            "peak_reserved_bytes": distribution([t["peak_reserved_bytes"] for t in trials if t["peak_reserved_bytes"] is not None]),
            "trials": trials}


def compiler_after_measurement_available(trainer):
    return compiler_snapshot(trainer.implementation.compile)["status"] == "available"
