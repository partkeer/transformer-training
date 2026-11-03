"""Readable, machine-readable and plotted reports for instrumented runs only."""

import argparse
import json
import os
from pathlib import Path

from .checkpoint import atomic_json, atomic_text


def write_report(report, output):
    """Atomically write JSON and readable Markdown for benchmark, training, profile, or
    imported evidence, including scope limitations and links to available raw
    artifacts.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    lines = ["# Training measurement report", "", f"Kind: {report['kind']}", ""]
    if report["kind"] == "benchmark":
        stats = report["throughput"]
        lines.extend([f"Useful throughput median: {stats['median']:.3f} tokens/s",
            f"Mean: {stats['mean']:.3f}; sample standard deviation: {stats['sample_stddev']}",
            f"Range: {stats['min']:.3f}–{stats['max']:.3f} tokens/s", "",
            "Numerator: non-padding tokens in successful updates. Denominator includes skipped attempts.",
            "Scope: warmed complete update attempts; includes loader delivery and transfers unless fixed mode is selected.",
            "Validation and checkpointing are excluded. Each trial is a separate process.", "",
            f"Comparison fingerprint: `{report['comparison_sha256']}`", "",
            "| Trial | Useful tokens/s | Processed tokens/s | Skipped updates (includes warm-up) | Raw artifacts |",
            "|---|---:|---:|---:|---|"])
        for i, trial in enumerate(report["trials"], 1):
            m = trial["measurement"]
            lines.append(f"| {i} | {m['warmed_useful_tokens_per_second']:.3f} | {m['warmed_processed_tokens_per_second']:.3f} | {m['skipped_updates']} | [trial](trial-{i}/trial.json), [log](trial-{i}/worker.log) |")
        if "baseline_comparison" in report:
            lines.extend(["", "Baseline comparison: " + json.dumps(report["baseline_comparison"])])
    elif report["kind"] == "training":
        lines.extend(["## Invocation measurements", ""])
        for item in report["invocations"]:
            lines.extend([f"### {item['invocation_id']}", "",
                f"Status: {item['status']}; timing complete: {item.get('timing_complete', False)}",
                f"Warmed useful throughput: {item.get('warmed_useful_tokens_per_second')} tokens/s",
                f"Whole-invocation useful throughput: {item.get('whole_run_useful_tokens_per_second')} tokens/s",
                f"Processed/applied tokens: {item['processed_tokens']} / {item['applied_tokens']}",
                f"Peak allocated/reserved: {item.get('peak_allocated_bytes')} / {item.get('peak_reserved_bytes')} bytes",
                f"Device-wide sampled memory: {item.get('device_memory', {}).get('value')} bytes",
                f"Quality targets: {json.dumps(item.get('quality_target_status', {}))}", ""])
        for path in report.get("plots", []):
            lines.extend([f"![Validation quality]({path})", ""])
    elif report["kind"] == "profile":
        for backend, capture in report["captures"].items():
            lines.extend([f"## {backend}", "", "Status: " + capture.get("status", "available"), ""])
            if capture.get("reason"):
                lines.extend([capture["reason"], ""])
            timeline = capture.get("timeline", capture.get("analysis", {}))
            if timeline.get("status") == "available":
                lines.extend([f"GPU busy / idle: {timeline['gpu_busy_us']} / {timeline['gpu_idle_us']} us",
                    f"Copy/compute overlap: {timeline['copy_compute_overlap_us']} us; exposed copy: {timeline['exposed_copy_us']} us", "",
                    "| Kernel | Aggregate GPU microseconds | Launches |", "|---|---:|---:|"])
                lines.extend(f"| {k['name']} | {k['gpu_us']:.3f} | {k['launches']} |" for k in timeline["kernels"][:10])
                lines.append("")
            if "memory" in capture:
                memory = capture["memory"]
                lines.extend(["Tensor peak: " + json.dumps(memory.get("tensor_peak")), "",
                    "Saved activations: " + json.dumps({k: v for k, v in memory["saved_for_backward"].items() if k != "events"}), ""])
            if "captures" in capture:
                for kernel in capture["captures"]:
                    lines.extend(["Kernel: " + kernel["kernel"], "", "```json",
                                  json.dumps(kernel["result"].get("analysis", kernel["result"]), indent=2), "```", ""])
            lines.extend([f"Raw capture directory: [{backend}/]({backend}/)", ""])
    else:
        lines.extend(["```json", json.dumps(report, indent=2), "```", ""])
    lines.extend(["## Interpretation limits", "",
        "Detailed profiles perturb execution and are excluded from benchmark comparisons.",
        "Phase durations can overlap. Kernel-time sums are not elapsed step time.",
        "Device-wide memory is sampled and can include other processes; reserved minus allocated memory is not a fragmentation diagnosis.",
        "Missing measurements are unavailable, not zero. A short benchmark does not establish convergence.", ""])
    atomic_text(output / "report.md", "\n".join(lines))


def report_training(run_dir, output):
    """Read invocation records and matching metrics, plot available validation
    observations, and write a report. Keep invocation histories separate and reject
    runs with no recorded measurements instead of reconstructing historical timing.
    """
    run_dir, output = Path(run_dir), Path(output)
    paths = sorted((run_dir / "invocations").glob("*/measurement.json"))
    if not paths:
        raise ValueError("No instrumented invocations found; historical reports are not reconstructed")
    invocations = [json.loads(p.read_text()) for p in paths]
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    valid_ids = {m["invocation_id"] for m in invocations}
    events = [e for e in events if e.get("invocation_id") in valid_ids]
    output.mkdir(parents=True, exist_ok=True)
    plots = []
    validation = [e for e in events if e["event"] == "validation"]
    if validation:
        os.environ.setdefault("MPLCONFIGDIR", str((output / ".matplotlib").resolve()))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, axis, label in (("validation_time.png", "invocation_lifecycle_seconds", "Invocation lifecycle seconds"),
                                  ("validation_tokens.png", "training_processed_tokens", "Processed training tokens")):
            fig, ax = plt.subplots(figsize=(9, 5))
            for item in invocations:
                points = [e for e in validation if e["invocation_id"] == item["invocation_id"]]
                if points:
                    ax.plot([e[axis] for e in points], [e["loss"] for e in points], marker="o", label=item["invocation_id"][:8])
                for target, observation in item.get("quality_targets", {}).items():
                    ax.axhline(float(target), linestyle="--", color="grey", alpha=.4)
                    if observation and observation["invocation_id"] == item["invocation_id"]:
                        ax.scatter(observation[axis], observation["loss"], marker="*", s=100)
            ax.set(xlabel=label, ylabel="Validation loss", title="Validation quality (invocations kept separate)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / name, dpi=150)
            plt.close(fig)
            plots.append(name)
    report = {"version": 1, "kind": "training", "invocations": invocations,
              "validation": validation, "numerical_failures": [e for e in events if e["event"] == "numerical_failure"],
              "plots": plots,
              "limitations": "Invocations are separate execution histories; replayed branches are not merged into one trajectory"}
    write_report(report, output)
    return report


def main(argv=None):
    """Parse a training-report command, validate a fresh destination, and generate
    artifacts without executing training. Return zero on success or report a CLI
    error.
    """
    parser = argparse.ArgumentParser(description="Report new instrumented training runs", allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    from .training import validate_output_directory
    try:
        output = args.output_dir or args.run_dir / "measurements-report"
        validate_output_directory(output)
        report_training(args.run_dir, output)
        print(output / "report.md")
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
