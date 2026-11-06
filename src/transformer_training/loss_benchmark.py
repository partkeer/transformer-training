"""Fresh-process loss microbenchmarks, separate correctness and memory evidence."""

import argparse
import gc
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .checkpoint import atomic_json, atomic_text
from .diagnostics import distribution
from .evaluation import autocast_context
from .loss import cross_entropy_sum, triton_launch_settings, validate_loss_backend
from .subprocesses import run_bounded
from .training import environment, validate_output_directory

GRADIENT_TOLERANCES = {"fp32": (1e-5, 1e-6), "fp16": (2e-3, 2e-5), "bf16": (2e-2, 2e-4)}


def check_loss_and_gradient(logits, targets, backend, precision, *, upstream=1.0):
    """Compare a backend to FP32 PyTorch on identical quantised values, outside timing.

    Return error magnitudes on success; raise on tolerance failures. Release large
    intermediate graphs before benchmarking. The returned gradient of the FP32
    objective is cast to the original dtype by autograd, matching AMP training.
    """
    reference_input = logits.detach().clone().requires_grad_(True)
    expected = cross_entropy_sum(reference_input.float(), targets)
    expected_gradient, = torch.autograd.grad(expected, reference_input, grad_outputs=expected.new_tensor(upstream))
    actual_input = logits.detach().requires_grad_(True)
    with autocast_context(logits.device, precision):
        actual = cross_entropy_sum(actual_input, targets, backend=backend)
    actual_gradient, = torch.autograd.grad(actual, actual_input, grad_outputs=actual.new_tensor(upstream))
    valid = int((targets != -100).sum())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5 * max(valid, 1))
    rtol, atol = GRADIENT_TOLERANCES[precision]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=rtol, atol=atol * abs(upstream))
    ignored = actual_gradient[targets == -100]
    if torch.count_nonzero(ignored).item():
        raise AssertionError("Ignored targets must have exactly zero gradients")
    return {"status": "passed", "reference": "FP32 PyTorch on identical quantised logits",
            "loss_absolute_error": abs(actual.item() - expected.item()),
            "max_gradient_absolute_error": float((actual_gradient.float() - expected_gradient.float()).abs().max())
                if logits.numel() else 0.0,
            "upstream_gradient": upstream}


def _timed_iterations(prepare, operation, cleanup, *, device, count):
    """Time synchronised operations; setup and cleanup, including backward graph setup, are excluded."""
    samples = []
    for _ in range(count):
        state = prepare()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = operation(state)
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
        cleanup(state, result)
        del state, result
    return samples


def benchmark_case(request):
    """Run one warmed CUDA workload in a fresh process, returning correctness/timing/memory.

    Compare native AMP PyTorch with Triton for the same logits, labels and upstream
    scale. Compilation and correctness checks precede all timing. Allocator peaks
    are measured in separate passes, never in the timed iteration loop.
    """
    device = torch.device(request["device"])
    backend, precision = request["backend"], request["precision"]
    validate_loss_backend(backend, device, request["vocab"])
    with torch.cuda.device(device):
        if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            return {"status": "unavailable", "reason": "Native BF16 is unavailable", "request": request}
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    generator = torch.Generator(device=device).manual_seed(request["seed"])
    logits = torch.randn(request["rows"], request["vocab"], generator=generator, device=device, dtype=dtype,
                         requires_grad=True)
    targets = torch.randint(request["vocab"], (request["rows"],), generator=generator, device=device)
    ignored_count = round(request["rows"] * request["ignored_fraction"])
    order = torch.randperm(request["rows"], generator=generator, device=device)
    targets[order[:ignored_count]] = -100
    del order
    correctness = check_loss_and_gradient(logits, targets, backend, precision,
                                          upstream=request["upstream"])
    valid = request["rows"] - ignored_count
    upstream = torch.tensor(request["upstream"], dtype=torch.float32, device=device)

    def objective():
        """Invoke the public loss including input validation under the repository's AMP policy."""
        with autocast_context(device, precision):
            return cross_entropy_sum(logits, targets, backend=backend)

    def prepare_backward():
        """Build a new backward graph outside the timed backward-only interval."""
        return objective()

    def forward(_):
        """Measure inference forward without saving an autograd graph."""
        with torch.no_grad():
            return objective()

    def backward(loss):
        """Consume the prepared graph, applying the configured scalar upstream gradient."""
        loss.backward(upstream)

    def combined(_):
        """Measure a complete loss forward/backward, including graph construction."""
        loss = objective()
        loss.backward(upstream)
        return loss.detach()

    def cleanup(_state, _result):
        """Release leaf gradients between iterations without charging reset to operation time."""
        logits.grad = None

    operations = {"forward": (lambda: None, forward),
                  "backward": (prepare_backward, backward),
                  "forward_backward": (lambda: None, combined)}
    timings = {}
    for name, (prepare, operation) in operations.items():
        _timed_iterations(prepare, operation, cleanup, device=device, count=request["warmup"])
        samples = _timed_iterations(prepare, operation, cleanup, device=device, count=request["measured"])
        timings[name] = {"latency_seconds": distribution(samples),
                         "supervised_tokens_per_second": valid / distribution(samples)["median"]}

    memory = {}
    for name, (prepare, operation) in operations.items():
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        state = prepare()
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = operation(state)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        memory[name] = {"baseline_allocated_bytes": baseline, "baseline_reserved_bytes": baseline_reserved,
                        "peak_allocated_bytes": peak, "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "incremental_peak_allocated_bytes": peak - baseline}
        cleanup(state, result)
        del state, result
    with torch.no_grad():
        loss_sum = objective().item()
    return {"status": "complete", "request": request, "environment": environment(device),
            "correctness": correctness, "timings": timings, "memory": memory,
            "supervised_tokens": valid, "loss_sum": loss_sum,
            "mean_loss": loss_sum / valid if valid else None,
            "launch": triton_launch_settings(request["vocab"]) if backend == "triton" else None,
            "timing_scope": "synchronised public operation, including validation; excludes setup/reset and compilation",
            "memory_scope": "separate warmed pass; backward baseline includes its prepared forward graph"}


def summarize_cases(cases):
    """Aggregate eligible repeats and pair speed/memory within each identical workload.

    Preserve failures/unavailable cases separately. Never compare incomplete repeat
    groups or collapse memory bytes into a speed claim.
    """
    grouped = {}
    for case in cases:
        request = case["request"]
        key = tuple(request[k] for k in ("rows", "vocab", "precision", "ignored_fraction", "backend"))
        grouped.setdefault(key, []).append(case)
    groups = []
    for key, repeats in grouped.items():
        rows, vocab, precision, ignored, backend = key
        complete = [r for r in repeats if r["status"] == "complete"]
        eligible = len(complete) == len(repeats)
        summary = {"rows": rows, "vocab": vocab, "precision": precision, "ignored_fraction": ignored,
                   "backend": backend, "status": "complete" if eligible else "incomplete",
                   "complete_repeats": len(complete), "requested_repeats": len(repeats), "operations": {}}
        if eligible:
            for name in ("forward", "backward", "forward_backward"):
                summary["operations"][name] = {
                    "repeat_median_seconds": distribution([r["timings"][name]["latency_seconds"]["median"] for r in complete]),
                    "supervised_tokens_per_second": distribution([r["timings"][name]["supervised_tokens_per_second"] for r in complete]),
                    "peak_allocated_bytes": distribution([r["memory"][name]["peak_allocated_bytes"] for r in complete]),
                    "peak_reserved_bytes": distribution([r["memory"][name]["peak_reserved_bytes"] for r in complete]),
                    "incremental_peak_allocated_bytes": distribution([r["memory"][name]["incremental_peak_allocated_bytes"] for r in complete])}
        groups.append(summary)
    comparisons = []
    for candidate in groups:
        if candidate["backend"] != "triton" or candidate["status"] != "complete":
            continue
        baseline = next((r for r in groups if r["backend"] == "pytorch" and r["status"] == "complete" and
                         all(r[k] == candidate[k] for k in ("rows", "vocab", "precision", "ignored_fraction"))), None)
        if baseline is None:
            continue
        entry = {k: candidate[k] for k in ("rows", "vocab", "precision", "ignored_fraction")}
        entry["operations"] = {}
        for name, value in candidate["operations"].items():
            reference = baseline["operations"][name]
            a, b = reference["peak_allocated_bytes"]["median"], value["peak_allocated_bytes"]["median"]
            entry["operations"][name] = {
                "speedup": reference["repeat_median_seconds"]["median"] / value["repeat_median_seconds"]["median"],
                "peak_allocated_bytes_saved": a - b,
                "peak_allocated_reduction_fraction": 1 - b / a if a else None}
        comparisons.append(entry)
    return {"groups": groups, "comparisons": comparisons}


def write_loss_report(report, output):
    """Write machine-readable evidence and a concise Markdown report, retaining failure reasons."""
    output = Path(output)
    atomic_json(output / "report.json", report)
    lines = ["# Cross-entropy microbenchmark", "", f"Status: {report['status']}", "",
             "Correctness checks, operation timings, and separate allocator measurements are reported independently.",
             "Complete optimiser updates use the dedicated experiment plan; these are loss-only measurements.", ""]
    checked = [c for c in report.get("cases", []) if c.get("correctness", {}).get("status") == "passed"]
    if checked:
        lines += [f"Correctness: {len(checked)} case/repeat checks passed before timing; see JSON for error magnitudes.", "",
                  "Timings below are combined forward/backward. JSON also contains forward-only and backward-only distributions.",
                  "Allocator measurements come from separate warmed passes, with baselines and incremental peaks in JSON.", ""]
    else:
        lines += ["No successful correctness/timing/memory measurements are available.", ""]
    if report.get("reason"):
        lines += [report["reason"], "", "GPU correctness, speed and memory: **not run**.", ""]
    lines += ["| Rows | Precision | Ignored | Backend | Status | F+B median ms | Peak allocated MiB |",
              "|---:|---|---:|---|---|---:|---:|"]
    for group in report.get("groups", []):
        operation = group["operations"].get("forward_backward")
        latency = f"{1000 * operation['repeat_median_seconds']['median']:.4f}" if operation else "unavailable"
        memory = f"{operation['peak_allocated_bytes']['median'] / 2**20:.2f}" if operation else "unavailable"
        lines.append(f"| {group['rows']} | {group['precision']} | {group['ignored_fraction']} | {group['backend']} | "
                     f"{group['status']} | {latency} | {memory} |")
    for comparison in report.get("comparisons", []):
        value = comparison["operations"]["forward_backward"]
        lines += ["", f"{comparison['precision']}, N={comparison['rows']}, ignored={comparison['ignored_fraction']}: "
                  f"F+B speedup {value['speedup']:.3f}x; peak allocation saved "
                  f"{value['peak_allocated_bytes_saved'] / 2**20:.2f} MiB."]
    for case in report.get("cases", []):
        if case["status"] != "complete":
            lines += ["", f"{case['request']['backend']} {case['request']['precision']} N={case['request']['rows']}: "
                      f"{case['status']}: {case.get('reason', 'see worker log')}"]
    atomic_text(output / "report.md", "\n".join(lines) + "\n")


def main(argv=None):
    """Resolve a side-effect-free dry-run or orchestrate isolated CUDA benchmark workers.

    Unavailable hardware writes an explicit report and returns 2. Worker failures
    remain failures, while independent cases continue and partial reports persist.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 257])
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--precisions", choices=list(GRADIENT_TOLERANCES), nargs="+", default=["fp32", "fp16", "bf16"])
    parser.add_argument("--ignored-fractions", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upstream", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        request = json.loads(args.worker.read_text())
        destination = args.worker.parent / "result.json"
        try:
            result = benchmark_case(request)
        except Exception as error:
            atomic_json(destination, {"status": "failed", "reason": f"{type(error).__name__}: {error}", "request": request})
            raise
        atomic_json(destination, result)
        return 0
    import math

    if (min(args.rows) < 1 or args.vocab < 1 or args.repeats < 1 or args.warmup < 1 or
            args.measured < 1 or args.timeout_seconds < 1 or not 0 <= args.seed < 2**63 or
            not math.isfinite(args.upstream) or any(not 0 <= x <= 1 for x in args.ignored_fractions)):
        parser.error("Invalid dimensions, fractions, counts, seed or upstream gradient")
    protocol = {key: value for key, value in vars(args).items() if key not in {"worker", "output_dir", "dry_run"}}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "backends": ["pytorch", "triton"], **protocol}, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is used")
    validate_output_directory(args.output_dir)
    report = {"version": 1, "kind": "loss_microbenchmark", "protocol": protocol, "cases": [],
              "status": "running", "groups": [], "comparisons": []}
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        report.update(status="unavailable", reason="CUDA is unavailable; GPU correctness, timing and memory were not measured.",
                      environment=environment(torch.device("cpu")))
        write_loss_report(report, args.output_dir)
        return 2
    report["environment"] = environment(torch.device(args.device))
    index = 0
    for rows, precision, ignored, backend, repeat in itertools.product(
            args.rows, args.precisions, args.ignored_fractions, ("pytorch", "triton"), range(args.repeats)):
        index += 1
        request = {"rows": rows, "vocab": args.vocab, "precision": precision, "ignored_fraction": ignored,
                   "backend": backend, "repeat": repeat, "seed": args.seed, "device": args.device,
                   "warmup": args.warmup, "measured": args.measured, "upstream": args.upstream}
        path = args.output_dir / f"case-{index:03d}"
        atomic_json(path / "request.json", request)
        print(f"Case {index}: {backend} {precision} N={rows}, ignored={ignored}, repeat={repeat + 1}", flush=True)
        try:
            validate_loss_backend(backend, args.device, args.vocab)
        except ValueError as error:
            result = {"status": "unavailable", "reason": str(error), "request": request}
        else:
            with (path / "worker.log").open("w", encoding="utf-8") as log:
                try:
                    process = run_bounded([sys.executable, "-m", "transformer_training.loss_benchmark",
                                           "--worker", str((path / "request.json").resolve())],
                                          timeout=args.timeout_seconds, stdout=log, stderr=subprocess.STDOUT)
                    if process.returncode:
                        result = (json.loads((path / "result.json").read_text()) if (path / "result.json").is_file()
                                  else {"status": "failed", "reason": f"Worker exited {process.returncode}; see {path / 'worker.log'}", "request": request})
                    else:
                        result = json.loads((path / "result.json").read_text())
                except subprocess.TimeoutExpired:
                    result = {"status": "failed", "reason": "Worker timed out", "request": request}
        atomic_json(path / "result.json", result)
        report["cases"].append(result)
        report.update(summarize_cases(report["cases"]))
        write_loss_report(report, args.output_dir)
    report["status"] = "complete" if all(c["status"] == "complete" for c in report["cases"]) else "incomplete"
    write_loss_report(report, args.output_dir)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
