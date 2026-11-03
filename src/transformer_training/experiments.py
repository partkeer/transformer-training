"""Sequential, resumable suites over existing isolated benchmark workers."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import TrainingConfig
from .experiment_plan import benchmark_request, load_plan


@contextmanager
def suite_lock(output):
    """Hold a nonblocking exclusive OS lock for one suite output directory.

    Raise if another runner holds it, and release on context exit or process
    termination so resume does not rely on guessing whether a saved PID is stale.
    The output directory must already exist.
    """
    with (output / ".suite.lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if stream.read(1) == b"":
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another process holds the suite lock") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another process holds the suite lock") from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def artifact_manifest(directory, root):
    """Return root-relative checksums for completed job files, excluding compiler
    caches so resume verifies durable artifacts rather than transient generated
    code.
    """
    from .implementations import file_sha256
    return {path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()
            and "compiler-cache" not in path.relative_to(root).parts}


def verify_manifest(manifest, output):
    """Verify saved artifacts exist under the suite output with recorded checksums;
    raise on missing or changed completed work before resuming.
    """
    from .implementations import file_sha256
    for name, expected in manifest.items():
        path = output / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Completed artifact changed or is missing: {name}")


def worker_environment(cache):
    """Return a copy of the environment with worker-local Inductor/Triton caches,
    remote compiler caches disabled, and graph-break/recompilation logging enabled.
    """
    env = os.environ.copy()
    env.update(TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE="0", TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE="0",
        TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE="0",
        TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE="0", TORCH_LOGS="graph_breaks,recompiles")
    return env


def _launch_benchmark(request, destination, timeout):
    """Run one isolated benchmark worker with the requested timeout and return its
    trial.json path.
    """
    from .diagnostics_cli import launch_worker
    launch_worker(request, destination, timeout=timeout)
    return destination / "trial.json"


def _launch_quality(request, destination, timeout):
    """Run an isolated quality worker in a new directory with a saved request, log, and
    local compiler cache. Return quality.json on success or raise with exit code and
    log tail.
    """
    from .checkpoint import atomic_json
    from .subprocesses import run_bounded
    destination.mkdir(parents=True, exist_ok=False)
    request = dict(request, output=str(destination))
    path = destination / "request.json"
    atomic_json(path, request)
    with (destination / "worker.log").open("w", encoding="utf-8") as log:
        result = run_bounded([sys.executable, "-m", "transformer_training.experiment_worker", str(path)],
            stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
            env=worker_environment(destination / "compiler-cache"))
    if result.returncode:
        tail = (destination / "worker.log").read_text(encoding="utf-8")[-3000:]
        raise RuntimeError(f"Quality exit {result.returncode}; {destination / 'worker.log'}\n{tail}")
    return destination / "quality.json"


def _compiler_seconds(trial):
    """Sum reported compiler-event durations in seconds, returning zero when disabled
    and None when durations are unavailable. Events can overlap, so this is not
    exclusive wall time.
    """
    compiler = trial["execution"]["compiler"]
    if compiler["status"] == "disabled":
        return 0.0
    durations = [e["duration_us"] for e in compiler["events"] if e.get("duration_us") is not None]
    return sum(durations) / 1e6 if durations else None


def build_report(plan, state, output):
    """Read completed artifacts to assemble per-variant benchmark, reference-
    comparison, and paired-quality summaries. Retain incomplete or ineligible
    statuses and return a report without launching work.
    """
    from .diagnostics import aggregate_trials, compare_results, distribution
    from .experiment_report import paired_quality

    rows, benchmarks, qualities = [], {}, {}
    for variant in plan["variants"]:
        name = variant["id"]
        trials, quality_results = [], {}
        for repeat in range(plan["benchmark"]["repeats"]):
            job = state["jobs"].get(f"{name}/benchmark/{repeat}", {})
            if job.get("status") == "complete":
                trials.append(json.loads((output / job["result"]).read_text()))
        if len(trials) == plan["benchmark"]["repeats"]:
            benchmarks[name] = aggregate_trials(trials)
        if plan["quality"]:
            for seed in plan["quality"]["seeds"]:
                job = state["jobs"].get(f"{name}/quality/{seed}", {})
                if job.get("status") == "complete":
                    quality_results[str(seed)] = json.loads((output / job["result"]).read_text())
        qualities[name] = quality_results
        benchmark = benchmarks.get(name)
        reference = variant["reference"]
        comparison = None
        if benchmark and reference:
            comparison = (compare_results(benchmarks[reference], benchmark, allowed_changes=variant["changes"])
                if reference in benchmarks else {"status": "reference_unavailable", "speedup": None})
        complete = benchmark is not None and (not plan["quality"] or len(quality_results) == len(plan["quality"]["seeds"]))
        row = {"variant": name, "reference": reference, "status": "complete" if complete else "incomplete",
            "changes": variant["changes"], "precision": variant["training"]["precision"],
            "implementation": variant["implementation"], "artifacts": name + "/",
            "comparison": comparison, "quality": quality_results,
            "observed_backends": sorted({backend for trial in trials for backend in
                                         trial["execution"]["attention_evidence"]["observed"]})}
        if benchmark:
            row.update(useful_tokens_per_second=benchmark["throughput"]["median"],
                processed_tokens_per_second=benchmark["processed_throughput"]["median"],
                throughput_stddev=benchmark["throughput"]["sample_stddev"],
                median_update_seconds=benchmark["latency_seconds"]["median"],
                p95_update_seconds=benchmark["latency_seconds"]["p95"],
                peak_allocated_bytes=benchmark["peak_allocated_bytes"]["median"],
                peak_reserved_bytes=benchmark["peak_reserved_bytes"]["median"],
                compile_wrapper_seconds=distribution([t["execution"]["compile_wrapper_seconds"] for t in trials])["median"],
                compiler_reported_seconds=distribution([s for t in trials if (s := _compiler_seconds(t)) is not None])["median"],
                first_forward_backward_attempt_seconds=distribution([t["first_forward_backward_attempt_seconds"] for t in trials])["median"],
                warmup_seconds=distribution([t["warmup_seconds"] for t in trials])["median"],
                skipped_updates=sum(t["measurement"]["skipped_updates"] for t in trials),
                numerical_failures=sum(t["measurement"]["numerical_failures"] for t in trials),
                timings_eligible_for_comparison=benchmark["timings_eligible_for_comparison"])
            if not benchmark["timings_eligible_for_comparison"]:
                row["status"] = "timings_ineligible"
        if comparison:
            row.update({key: comparison.get(key) for key in ("speedup", "memory_reduction", "update_time_change_percent")})
            if comparison["status"] != "matched":
                row["status"] = comparison["status"]
        if quality_results:
            row["validation_loss"] = distribution([v["loss"] for v in quality_results.values()])["median"]
            perplexities = [v["perplexity"] for v in quality_results.values() if v["perplexity"] is not None]
            row["perplexity"] = distribution(perplexities)["median"]
            if any(not item["budget_reached"] for item in quality_results.values()):
                row["status"] = "quality_budget_unreached"
        if plan["quality"] and reference:
            row["paired_quality"] = paired_quality(quality_results, qualities.get(reference, {}), plan["quality"])
        rows.append(row)
    return {"version": 1, "kind": "implementation_suite", "status": "complete" if
            all(row["status"] == "complete" for row in rows) else "incomplete",
            "rows": rows, "jobs": state["jobs"], "plan": plan, "identity": state["identity"]}


def run_suite(plan, output, *, resume=False):
    """Execute resolved variants sequentially, persist job state and reports, and
    return the summary. Resume only with matching plan/source/environment identity
    and verified artifacts; retry unfinished jobs in new attempt directories and
    reuse available quality checkpoints.
    """
    from .checkpoint import atomic_json
    from .implementations import file_sha256
    from .measurements import digest
    from .training import environment, validate_device, validate_output_directory
    from .experiment_report import write_suite_report

    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[2]
    if output.is_relative_to(root):
        ignored = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", "check-ignore", "-q", "--",
                                  str(output / "suite-state.json")], cwd=root, capture_output=True)
        if ignored.returncode != 0:
            raise ValueError("Suite outputs inside the checkout must be Git-ignored; use runs/ or an external directory")
    config = TrainingConfig(**plan["shared"]["training"])
    if not resume:
        validate_output_directory(output)
    elif not (output / "suite-state.json").is_file():
        raise ValueError("--resume requires an existing suite-state.json")
    identity = {"plan_sha256": digest(plan), "environment": environment(validate_device(config))}
    if identity["environment"]["source"].get("status") == "unavailable":
        raise ValueError("Suite execution requires a verified source identity")
    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        if not resume and (output / "suite-state.json").exists():
            raise ValueError("A suite already exists here; use --resume or a new output directory")
        if resume:
            state = json.loads((output / "suite-state.json").read_text())
            if state["identity"] != identity:
                raise ValueError("Plan, source or environment changed; use a new output directory and baseline")
            if file_sha256(output / "resolved-plan.json") != state["resolved_plan_sha256"]:
                raise ValueError("Resolved plan artifact changed")
            for job in state["jobs"].values():
                if job["status"] == "complete":
                    verify_manifest(job["manifest"], output)
        else:
            atomic_json(output / "resolved-plan.json", plan)
            state = {"version": 1, "identity": identity, "jobs": {},
                     "resolved_plan_sha256": file_sha256(output / "resolved-plan.json")}
            atomic_json(output / "suite-state.json", state)

        for variant in plan["variants"]:
            jobs = [("benchmark", i) for i in range(plan["benchmark"]["repeats"])]
            if plan["quality"]:
                jobs += [("quality", seed) for seed in plan["quality"]["seeds"]]
            for stage, index in jobs:
                key = f"{variant['id']}/{stage}/{index}"
                previous = state["jobs"].get(key, {})
                if previous.get("status") == "complete":
                    continue
                attempt = previous.get("attempt", 0) + 1
                destination = output / key / f"attempt-{attempt}"
                job = {"status": "running", "attempt": attempt}
                if stage == "quality":
                    prior_dir = previous.get("training_dir")
                    checkpoint = output / prior_dir / "latest.pt" if prior_dir else None
                    training_dir = output / prior_dir if checkpoint and checkpoint.is_file() else destination / "training"
                    job["training_dir"] = training_dir.relative_to(output).as_posix()
                state["jobs"][key] = job
                atomic_json(output / "suite-state.json", state)
                print(f"{key} (attempt {attempt})", flush=True)
                try:
                    if stage == "benchmark":
                        result = _launch_benchmark(benchmark_request(plan, variant), destination,
                                                  plan["benchmark"]["timeout_seconds"])
                    else:
                        training = dict(variant["training"], seed=index,
                            supervised_token_budget=plan["quality"]["supervised_token_budget"])
                        request = {"model": plan["shared"]["model"], "training": training,
                            "implementation": variant["implementation"], "data": plan["shared"]["data"],
                            "offline": plan["shared"]["offline"], "training_dir": str(training_dir),
                            "resume": str(checkpoint) if checkpoint and checkpoint.is_file() else None}
                        result = _launch_quality(request, destination, plan["quality"]["timeout_seconds"])
                    job.update(status="complete", result=result.relative_to(output).as_posix(),
                               manifest=artifact_manifest(destination, output))
                    if stage == "quality" and not training_dir.is_relative_to(destination):
                        job["manifest"].update(artifact_manifest(training_dir, output))
                except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                    timed_out = isinstance(error, subprocess.TimeoutExpired) or isinstance(error.__cause__, subprocess.TimeoutExpired)
                    job.update(status="timeout" if timed_out else "failed",
                               reason=str(error))
                atomic_json(output / "suite-state.json", state)
                write_suite_report(output, build_report(plan, state, output))
        report = build_report(plan, state, output)
        write_suite_report(output, report)
        return report


def main(argv=None):
    """Parse a plan, print resolved settings for dry-run or execute/resume the suite,
    and return zero for completion or one for incomplete jobs. Report invalid
    configuration as a CLI error.
    """
    parser = argparse.ArgumentParser(description="Sequential transformer implementation comparisons", allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.dry_run:
            print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
            return 0
        if args.output_dir is None:
            raise ValueError("Execution requires --output-dir")
        report = run_suite(plan, args.output_dir, resume=args.resume)
        print(f"Report: {args.output_dir / 'report.md'}")
        return 0 if report["status"] == "complete" else 1
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
