"""Optional local NVIDIA CLI capture and unit-aware report import."""

import csv
import io
import math
from pathlib import Path
import re
import shutil
import json
import sqlite3
import subprocess

from .measurements import unavailable
from .traces import analyse_trace, load_nsys_sqlite
from .subprocesses import run_bounded


METRICS = ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
           "dram__throughput.avg.pct_of_peak_sustained_elapsed",
           "sm__throughput.avg.pct_of_peak_sustained_elapsed",
           "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
           "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
           "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"]
TENSOR_METRICS = [f"sm__ops_path_tensor_src_{src}_dst_{dst}_sparsity_off.sum"
                  for src, dst in (("fp16", "fp32"), ("fp16", "fp16"),
                                   ("bf16", "fp32"), ("tf32", "fp32"), ("fp64", "fp64"))]


def normalize(value, unit):
    """Parse a finite Nsight CSV value, remove thousands separators, and convert
    supported time, byte, and rate units to base units. Preserve percentages and
    reject unknown units.
    """
    value = float(value.replace(",", ""))
    if not math.isfinite(value):
        raise ValueError("Nonfinite Nsight metric")
    factors = {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1,
               "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
               "byte/second": 1, "Kbyte/second": 1e3, "Mbyte/second": 1e6,
               "Gbyte/second": 1e9, "%": 1, "inst": 1, "op": 1, "FLOP": 1, "": 1}
    if unit not in factors:
        raise ValueError("Unsupported Nsight metric unit: " + unit)
    return value * factors[unit]


def parse_ncu_csv(text):
    """Parse per-launch Nsight Compute metrics into normalised kernel records,
    retaining raw rows and marking invalid metrics unavailable. Derive traffic,
    bandwidth, and supported arithmetic rates; bottleneck labels are heuristic
    candidates, not proven diagnoses.
    """
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if "Metric Name" in line and "Metric Value" in line), None)
    if header is None:
        raise ValueError("Malformed Nsight Compute CSV: metric header missing")
    kernels = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        if not row.get("Metric Name") or row["Metric Name"] == "Metric Name":
            continue
        name, launch = row.get("Kernel Name", "unknown"), row.get("ID", "unknown")
        key = (row.get("Process ID"), row.get("Context"), row.get("Stream"), launch, name)
        item = kernels.setdefault(key, {"name": name, "launch_id": launch,
            "process_id": row.get("Process ID"), "context": row.get("Context"),
            "stream": row.get("Stream"), "metrics": {}, "raw_metrics": []})
        item["raw_metrics"].append(row)
        try:
            value = normalize(row["Metric Value"], row.get("Metric Unit", ""))
            item["metrics"][row["Metric Name"]] = value
        except (ValueError, TypeError):
            item["metrics"][row["Metric Name"]] = None
    if not kernels:
        raise ValueError("Nsight Compute CSV contains no kernel measurements")
    for item in kernels.values():
        m = item["metrics"]
        seconds = m.get("gpu__time_duration.sum")
        reads, writes = m.get("dram__bytes_read.sum"), m.get("dram__bytes_write.sum")
        traffic = reads + writes if reads is not None and writes is not None else None
        item["duration_seconds"] = seconds
        item["dram_bytes"] = traffic
        item["bandwidth_bytes_per_second"] = traffic / seconds if traffic is not None and seconds else None
        counts = [m.get("smsp__sass_thread_inst_executed_op_" + op + "_pred_on.sum")
                  for op in ("ffma", "fadd", "fmul")]
        flops = 2 * counts[0] + counts[1] + counts[2] if all(x is not None for x in counts) else None
        item["fp32_scalar_flops_per_second"] = flops / seconds if flops is not None and seconds else None
        item["fp32_scalar_arithmetic_intensity"] = flops / traffic if flops is not None and traffic else None
        item["compute_scope"] = "FP32 scalar instruction counters only; excludes tensor cores and other arithmetic"
        item["tensor_operations_per_second"] = {metric: m[metric] / seconds for metric in TENSOR_METRICS
                                                if m.get(metric) is not None and seconds}
        memory = m.get("dram__throughput.avg.pct_of_peak_sustained_elapsed")
        compute = m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        item["evidence"] = {"dram_percent_sustained": memory, "sm_percent_sustained": compute,
                            "classification": "inconclusive",
                            "reason": "Saturation, instruction path and roofline evidence require joint interpretation; no utilization-only diagnosis"}
        if traffic is not None and seconds and memory is not None and compute is not None:
            candidate = ("mixed bandwidth/compute candidate" if memory >= 80 and compute >= 80 else
                         "memory-bandwidth candidate" if memory >= 80 and compute < 60 else
                         "compute candidate" if compute >= 80 and memory < 60 and
                         (flops is not None or item["tensor_operations_per_second"]) else "inconclusive")
            item["evidence"].update(classification=candidate,
                reason="Heuristic triage using measured traffic/time and hardware pipeline saturation; not a proven bottleneck",
                heuristic_thresholds={"high_sustained_percent": 80, "low_sustained_percent": 60})
        item["missing_metrics"] = [metric for metric in METRICS if m.get(metric) is None]
    return {"status": "available", "kernels": list(kernels.values()),
            "scope": "separate instrumented kernel capture; excluded from benchmark timings"}


def run_tool(arguments, *, timeout=1800, stdout_path=None):
    """Run an installed NVIDIA tool with a timeout and return structured evidence.

    Optionally save stdout, and distinguish missing executables, denied counter
    access, and failed captures. Preserve command/error details without changing
    privileges or invoking a shell.
    """
    if not shutil.which(arguments[0]):
        return unavailable(arguments[0] + " is not installed", "missing_tool")
    try:
        result = run_bounded(arguments, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return unavailable(str(error), "capture_failed")
    if stdout_path:
        Path(stdout_path).write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        reason = result.stderr + result.stdout
        status = "permission_denied" if "ERR_NVGPUCTRPERM" in reason or "permission" in reason.lower() else "capture_failed"
        return dict(unavailable(reason[-8000:], status), command=arguments, returncode=result.returncode)
    return {"status": "available", "stdout": result.stdout, "stderr": result.stderr, "command": arguments}


def capture_systems(command, directory, *, timeout=1800):
    """Run a bounded Nsight Systems capture and SQLite export in the directory, then
    return overlap-aware timeline analysis and artifact paths or an explicit
    failure.
    """
    directory = Path(directory)
    prefix = directory / "systems"
    result = run_tool(["nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
        "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--output=" + str(prefix), *command], timeout=timeout)
    if result["status"] != "available":
        return result
    exported = run_tool(["nsys", "export", "--type=sqlite", "--output=" + str(prefix) + ".sqlite",
                         str(prefix) + ".nsys-rep"], timeout=timeout)
    if exported["status"] != "available":
        return exported
    try:
        return {"status": "available", "capture": result,
                "analysis": analyse_trace(load_nsys_sqlite(str(prefix) + ".sqlite")),
                "raw_report": "systems.nsys-rep", "sqlite": "systems.sqlite"}
    except (ValueError, OSError, sqlite3.Error) as error:
        return unavailable(str(error), "unsupported_export")


def capture_compute(command, directory, kernels, *, followup=False, timeout=1800):
    """Capture supported counters for up to five dominant kernel names from a prior
    timeline, using fresh workload outputs per kernel. Optionally request
    occupancy/stall sections; return per-capture evidence and partial failures
    separately from benchmark timings.
    """
    directory = Path(directory)
    if not kernels:
        return unavailable("A timeline with dominant kernel signatures is required", "missing_timeline")
    query = run_tool(["ncu", "--query-metrics", "--query-metrics-mode", "suffix"])
    if query["status"] != "available":
        return query
    (directory / "available_metrics.txt").write_text(query["stdout"], encoding="utf-8")
    sections = run_tool(["ncu", "--list-sections"])
    if sections["status"] != "available":
        return sections
    (directory / "available_sections.txt").write_text(sections["stdout"], encoding="utf-8")
    metrics = [m for m in METRICS + TENSOR_METRICS if m in query["stdout"]]
    if not metrics:
        return unavailable("Requested hardware counters are not supported", "unsupported_metrics")
    captures = []
    for index, kernel in enumerate(kernels[:5]):
        prefix = directory / f"compute-{index}"
        from .checkpoint import atomic_json
        original_request = json.loads(Path(command[-1]).read_text())
        original_request["output"] = str((directory / f"workload-{index}").resolve())
        request_path = directory / f"request-{index}.json"
        atomic_json(request_path, original_request)
        kernel_command = [*command[:-1], str(request_path.resolve())]
        # Exact regex match, three representative launches per dominant name.
        arguments = ["ncu", "--profile-from-start", "off", "--kernel-name-base", "demangled",
            "--kernel-name", "regex:^" + re.escape(kernel["name"]) + "$", "--launch-count", "3",
            "--metrics", ",".join(metrics), "--export", str(prefix)]
        for section in ("SpeedOfLight_RooflineChart", "MemoryWorkloadAnalysis"):
            if section in sections["stdout"]:
                arguments.extend(["--section", section])
        if followup:
            for section in ("Occupancy", "WarpStateStats"):
                if section in sections["stdout"]:
                    arguments.extend(["--section", section])
        captured = run_tool([*arguments, *kernel_command], timeout=timeout)
        if captured["status"] == "available":
            exported = run_tool(["ncu", "--import", str(prefix) + ".ncu-rep", "--page", "raw", "--csv"],
                                stdout_path=prefix.with_suffix(".csv"))
            if exported["status"] == "available":
                try:
                    captured["analysis"] = parse_ncu_csv(exported["stdout"])
                except ValueError as error:
                    captured = unavailable(str(error), "malformed_export")
            else:
                captured = exported
        captures.append({"kernel": kernel["name"], "result": captured})
    return {"status": "available" if all(c["result"]["status"] == "available" for c in captures) else "partial",
            "captures": captures, "unsupported_metrics": [m for m in METRICS if m not in metrics],
            "timing_scope": "instrumented kernel replay; never a training-throughput measurement"}
