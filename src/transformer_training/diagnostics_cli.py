"""Fresh-process benchmark/profiling orchestration; dry-run performs no data/device work."""

from dataclasses import asdict, fields, replace
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time

from .cli import build_parser
from .config import ModelConfig, TrainingConfig
from .implementations import resolve_implementation


def resolve_request(args, argv):
    """Resolve CLI and optional checkpoint settings into a serialisable request without
    loading data or initialising CUDA. Inherit unspecified saved options and reject
    incompatible explicit model, training, or data changes.
    """
    from .checkpoint import load_checkpoint
    from .training import validate_resume_settings

    explicit = {x.split("=", 1)[0][2:].replace("-", "_") for x in argv if x.startswith("--")}
    if "validation_loss_target" in explicit:
        explicit.add("validation_loss_targets")
    saved = load_checkpoint(args.resume) if args.resume else None
    implementation = resolve_implementation(args, saved)
    configs = []
    for cls, key in ((ModelConfig, "model_config"), (TrainingConfig, "training_config")):
        values = saved[key].copy() if saved else asdict(cls())
        for f in fields(cls):
            if f.name != "vocab_size" and (not saved or f.name in explicit):
                values[f.name] = getattr(args, f.name)
        configs.append(cls(**values))
    model, training = configs
    if saved:
        validate_resume_settings(saved, model, training, implementation, diagnostic=True)
    options = {"sequence_length": model.sequence_length, "batch_size": training.batch_size,
        "seed": training.seed, "num_workers": training.num_workers,
        "pin_memory": training.device.startswith("cuda"), "exclude_validation_overlap": True,
        "max_stories": args.max_stories}
    for key in ("revision", "tokenizer_revision", "max_stories"):
        if saved and key not in explicit:
            options[key] = saved["data_signature"].get(key)
        elif getattr(args, key) is not None:
            options[key] = getattr(args, key)
        if saved and key in explicit and options.get(key) != saved["data_signature"].get(key):
            raise ValueError("Conflicting checkpoint data setting: " + key)
    if args.max_stories is not None and args.max_stories < 1:
        raise ValueError("max_stories must be positive")
    return {"model": asdict(model), "training": asdict(training),
            "implementation": asdict(implementation), "data_options": options,
            "checkpoint": str(args.resume.resolve()) if args.resume else None,
            "offline": args.offline, "warmup": args.warmup_attempts,
            "measured": args.measured_attempts, "input_mode": args.input_mode}


def worker(request):
    """Configure offline mode, prepare data, build an isolated trainer, and execute one
    resolved trial request. Write completion/failure status for trial execution and
    return its result.
    """
    if request["offline"]:
        os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
    from .checkpoint import atomic_json
    from .data import load_tinystories
    from .diagnostics import run_trial
    from .training import Trainer, validate_device
    from .config import ImplementationConfig

    config = TrainingConfig(**request["training"])
    validate_device(config)
    start = time.perf_counter()
    loaders = load_tinystories(**request["data_options"])
    preparation = time.perf_counter() - start
    model = replace(ModelConfig(**request["model"]), vocab_size=loaders.metadata["vocab_size"])
    trainer = Trainer(model, config, loaders, request["output"], resume=request["checkpoint"],
                      preparation_seconds=preparation, diagnostic=True,
                      implementation=ImplementationConfig(**request.get("implementation", {})))
    try:
        result = run_trial(trainer, warmup=request["warmup"], measured=request["measured"],
                           mode=request["input_mode"], backend=request["backend"])
        atomic_json(Path(request["output"]) / "worker_status.json", {"status": "complete"})
        return result
    except BaseException as error:
        atomic_json(Path(request["output"]) / "worker_status.json", {"status": "failed", "reason": str(error)})
        raise


def launch_worker(request, output, *, timeout=1800):
    """Write a request in a new directory and launch a bounded fresh Python worker with
    isolated compiler caches and a captured log. Return trial result and command, or
    raise with timeout/failure context.
    """
    from .checkpoint import atomic_json
    from .subprocesses import run_bounded

    output.mkdir(parents=True, exist_ok=False)
    request = dict(request, output=str((output / "artifacts").resolve()))
    path = output / "request.json"
    atomic_json(path, request)
    command = [sys.executable, "-m", "transformer_training.diagnostics_cli", "--worker", str(path.resolve())]
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        try:
            from .experiments import worker_environment
            process = run_bounded(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                  env=worker_environment(output / "compiler-cache"))
        except subprocess.TimeoutExpired as error:
            atomic_json(output / "worker_status.json", {"status": "timeout", "reason": str(error)})
            raise RuntimeError(f"Trial timed out; see {output / 'worker.log'}") from error
    if process.returncode:
        tail = (output / "worker.log").read_text(encoding="utf-8")[-4000:]
        raise RuntimeError(f"Trial failed with exit {process.returncode}; see {output / 'worker.log'}\n{tail}")
    result = json.loads((output / "artifacts" / "trial.json").read_text())
    atomic_json(output / "trial.json", result)
    return result, command


def main(kind="benchmark", argv=None):
    """Parse benchmark/profile commands and return an exit status after running
    isolated trials, importing captures, or printing a side-effect-free dry-run
    request. Protect occupied destinations and keep profiler captures separate from
    benchmark statistics.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    parser.description = "Repeated complete-update benchmarks" if kind == "benchmark" else "Separate short profiling captures"
    parser.add_argument("--checkpoint", dest="resume", type=Path, help="Inherit a saved training state; original checkpoint is never written")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-attempts", type=int, default=10)
    parser.add_argument("--measured-attempts", type=int, default=50 if kind == "benchmark" else 5)
    parser.add_argument("--input-mode", choices=["loader", "fixed"], default="loader")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--backend", choices=["pytorch", "memory", "nsys", "ncu", "all"], default="pytorch")
    parser.add_argument("--timeline", type=Path, help="Prior profile JSON with ranked kernels, required by ncu alone")
    parser.add_argument("--followup-counters", action="store_true", help="Opt in to occupancy and stall diagnostics")
    parser.add_argument("--compare", type=Path, help="Baseline benchmark JSON")
    parser.add_argument("--allow-change", action="append", default=[],
                        help="Declared difference for --compare, e.g. implementation.attention or training.precision")
    parser.add_argument("--import-path", type=Path)
    parser.add_argument("--import-kind", choices=["chrome", "nsys", "ncu"])
    args = parser.parse_args(argv)
    created_output = False
    try:
        if args.warmup_attempts < 0 or args.measured_attempts < 1 or args.repeats < 1 or args.timeout_seconds < 1:
            raise ValueError("Counts/timeouts must be positive; warm-up may be zero")
        if kind == "benchmark" and any(x.split("=", 1)[0] in {"--backend", "--timeline", "--followup-counters"} for x in argv):
            raise ValueError("Profiler options belong to profile_training.py; benchmark trials always disable profiling")
        if kind == "profile" and any(x.split("=", 1)[0] == "--repeats" for x in argv):
            raise ValueError("Profiles run once per selected backend; use --repeats only for benchmarks")
        from .checkpoint import atomic_json
        from .training import validate_output_directory
        from .reporting import write_report

        output = args.output_dir or Path("runs") / (kind + "-" + str(time.time_ns()))
        validate_output_directory(output)
        if args.import_path:
            if not args.import_kind:
                raise ValueError("--import-path requires --import-kind")
            if args.dry_run:
                print(json.dumps({"dry_run": True, "import_kind": args.import_kind, "input": str(args.import_path)}))
                return 0
            from .nsight import parse_ncu_csv
            from .traces import analyse_trace, load_chrome, load_nsys_sqlite
            try:
                analysis = (parse_ncu_csv(args.import_path.read_text()) if args.import_kind == "ncu" else
                    analyse_trace(load_nsys_sqlite(args.import_path) if args.import_kind == "nsys" else load_chrome(args.import_path)))
            except (OSError, ValueError, KeyError, TypeError, AttributeError, sqlite3.Error) as error:
                analysis = {"status": "malformed_export", "value": None, "reason": str(error)}
            output.mkdir(parents=True, exist_ok=True)
            created_output = True
            write_report({"version": 1, "kind": "import", "source": str(args.import_path.resolve()),
                          "backend": args.import_kind, "analysis": analysis}, output)
            return 0
        request = resolve_request(args, argv)
        if args.dry_run:
            print(json.dumps({"dry_run": True, "kind": kind, "repeats": args.repeats if kind == "benchmark" else 1,
                              "backend": args.backend if kind == "profile" else "benchmark", **request}, indent=2))
            return 0
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "1"
        from .training import validate_device
        validate_device(TrainingConfig(**request["training"]))
        if kind == "profile" and args.backend in {"nsys", "ncu", "all"} and request["training"]["device"] == "cpu":
            raise ValueError("Nsight backends require CUDA; select pytorch or memory for CPU diagnostics")
        output.mkdir(parents=True, exist_ok=True)
        created_output = True
        if kind == "benchmark":
            from .diagnostics import aggregate_trials, compare_results
            trials = []
            for i in range(args.repeats):
                print(f"Trial {i + 1}/{args.repeats}", flush=True)
                trial, _ = launch_worker(dict(request, backend="benchmark"), output / f"trial-{i + 1}", timeout=args.timeout_seconds)
                trials.append(trial)
            report = aggregate_trials(trials)
            if args.compare:
                report["baseline_comparison"] = compare_results(json.loads(args.compare.read_text()), report,
                                                               allowed_changes=args.allow_change)
        else:
            from .nsight import capture_compute, capture_systems
            backends = ["pytorch", "memory", "nsys", "ncu"] if args.backend == "all" else [args.backend]
            report = {"version": 1, "kind": "profile", "captures": {}}
            timeline = json.loads(args.timeline.read_text()) if args.timeline else {}
            for backend in backends:
                destination = output / backend
                if backend in {"pytorch", "memory"}:
                    trial, _ = launch_worker(dict(request, backend=backend), destination, timeout=args.timeout_seconds)
                    report["captures"][backend] = trial
                    if backend == "pytorch":
                        timeline = trial.get("timeline", {})
                else:
                    destination.mkdir()
                    worker_dir = destination / "workload"
                    worker_request = dict(request, output=str(worker_dir.resolve()), backend=backend)
                    path = destination / "request.json"
                    atomic_json(path, worker_request)
                    command = [sys.executable, "-m", "transformer_training.diagnostics_cli", "--worker", str(path.resolve())]
                    if backend == "nsys":
                        captured = capture_systems(command, destination, timeout=args.timeout_seconds)
                        if captured.get("analysis", {}).get("status") == "available":
                            timeline = captured["analysis"]
                    else:
                        kernels = timeline.get("kernels", timeline.get("timeline", {}).get("kernels", []))
                        captured = capture_compute(command, destination, kernels, followup=args.followup_counters,
                                                   timeout=args.timeout_seconds)
                    report["captures"][backend] = captured
        write_report(report, output)
        print(f"Report: {output / 'report.md'}", flush=True)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        if created_output:
            # Do not overwrite pre-existing reports when destination validation failed.
            if not (output / "report.json").exists():
                from .checkpoint import atomic_json
                atomic_json(output / "failure.json", {"status": "failed", "reason": str(error)})
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(json.loads(Path(sys.argv[2]).read_text()))
    else:
        raise SystemExit(main())
