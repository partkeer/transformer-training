"""Compact comparison tables with links to complete per-trial evidence."""

import csv
import io
import json

from .checkpoint import atomic_json, atomic_text


def paired_quality(candidate, reference, quality):
    """Compare candidate and reference loss per seed, checking equal work, budget
    coverage, skipped updates, and validation identity. Report loss differences
    against an optional margin only for eligible pairs; do not infer statistical
    equivalence.
    """
    pairs = []
    for seed in quality["seeds"]:
        a, b = reference.get(str(seed)), candidate.get(str(seed))
        if not a or not b:
            pairs.append({"seed": seed, "status": "incomplete"})
            continue
        checks = {
            "processed_tokens_match": a["processed_tokens"] == b["processed_tokens"],
            "applied_tokens_match": a["applied_tokens"] == b["applied_tokens"],
            "budget_reached": min(a["processed_tokens"], b["processed_tokens"]) >= quality["supervised_token_budget"],
            "no_skipped_updates": a["skipped_updates"] == b["skipped_updates"] == 0,
            "validation_matches": a["validation_identity"] == b["validation_identity"],
        }
        eligible = all(checks.values())
        delta = b["loss"] - a["loss"]
        margin = quality.get("degradation_margin")
        pairs.append({"seed": seed, "status": "matched" if eligible else "conditions_differ",
            "checks": checks, "reference_loss": a["loss"], "candidate_loss": b["loss"],
            "loss_difference": delta,
            "within_declared_margin": delta <= margin if eligible and margin is not None else None,
            "reference_budget_overshoot": a["budget_overshoot"], "candidate_budget_overshoot": b["budget_overshoot"]})
    return {"pairs": pairs, "degradation_margin": quality.get("degradation_margin"),
            "interpretation": "Paired finite-budget observations; no statistical equivalence claim"}


def write_suite_report(output, report):
    """Atomically write JSON, CSV, and Markdown views of a report into the existing
    output directory, preserving missing metrics and linking raw evidence and
    incomplete jobs.
    """
    atomic_json(output / "report.json", report)
    columns = ["variant", "reference", "status", "changes", "observed_backends", "useful_tokens_per_second",
        "processed_tokens_per_second", "throughput_stddev", "median_update_seconds", "p95_update_seconds",
        "peak_allocated_bytes", "peak_reserved_bytes", "compile_wrapper_seconds", "compiler_reported_seconds",
        "first_forward_backward_attempt_seconds", "warmup_seconds", "skipped_updates", "numerical_failures",
        "speedup", "memory_reduction", "update_time_change_percent", "validation_loss", "perplexity", "artifacts"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    lines = ["# Transformer implementation comparisons", "", f"Status: {report['status']}", "",
        "Useful throughput counts successfully applied tokens; skipped attempts still cost time.",
        "All timings cover complete updates. Fixed input still includes transfers.", "",
        "| Variant | Control | Status | Tokens/s | Update ms | Peak allocated MiB | Speedup | Memory reduction | Backends |",
        "|---|---|---|---:|---:|---:|---:|---:|---|"]
    def display(value, scale=1):
        """Format a numeric metric with an optional unit scale, or render None as
        unavailable rather than zero.
        """
        return "unavailable" if value is None else f"{value * scale:.3f}"
    for row in report["rows"]:
        writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), (list, dict)) else row.get(key)
                         for key in columns})
        memory_reduction = display(row.get('memory_reduction'), 100)
        if row.get("memory_reduction") is not None:
            memory_reduction += "%"
        lines.append(f"| [{row['variant']}]({row['artifacts']}) | {row['reference'] or '-'} | {row['status']} | "
            f"{display(row.get('useful_tokens_per_second'))} | {display(row.get('median_update_seconds'), 1000)} | "
            f"{display(row.get('peak_allocated_bytes'), 1 / 2**20)} | {display(row.get('speedup'))} | "
            f"{memory_reduction} | {', '.join(row['observed_backends']) or 'unverified'} |")
    lines.extend(["", "## Configuration and startup", ""])
    for row in report["rows"]:
        lines.extend([f"### {row['variant']}", "", "Declared changes: " + json.dumps(row["changes"]),
            "Requested: `" + json.dumps(row["implementation"], sort_keys=True) + "`",
            f"Precision: {row['precision']}; comparison: {json.dumps(row.get('comparison'))}",
            f"First forward/backward attempt: {row.get('first_forward_backward_attempt_seconds')} s; "
            f"warm-up: {row.get('warmup_seconds')} s; compiler-reported time: {row.get('compiler_reported_seconds')} s.",
            f"Skipped updates: {row.get('skipped_updates')}; numerical failures: {row.get('numerical_failures')}.",
            f"Quality: {json.dumps(row.get('quality', {}))}",
            f"Paired quality: {json.dumps(row.get('paired_quality', {}))}", ""])
    failures = [(key, job) for key, job in report["jobs"].items() if job["status"] != "complete"]
    if failures:
        lines.extend(["## Incomplete jobs", ""])
        for key, job in failures:
            lines.extend([f"### {key}", "", f"Status: {job['status']}", "", "```text",
                          job.get("reason", "Job was interrupted; resume to retry"), "```", ""])
    lines.extend(["## Interpretation", "",
        "Compiler timings can overlap and depend on telemetry coverage. First-attempt latency includes execution.",
        "Compilation in measured attempts makes warmed comparisons ineligible; raw trial records remain available.",
        "Backend evidence comes from separate untimed captures after measurement; inspect each trial for shape coverage.",
        "Unavailable memory is not zero. Unsupported/failed entries are never retried with different settings.",
        "Quality uses final token-budget checkpoints evaluated with FP32, eager reference attention on full validation.", ""])
    atomic_text(output / "report.csv", stream.getvalue())
    atomic_text(output / "report.md", "\n".join(lines))
