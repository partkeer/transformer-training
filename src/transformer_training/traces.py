"""Overlap-aware Chrome/CUPTI timeline analysis. All internal intervals use microseconds."""

import json
from pathlib import Path
import sqlite3


def union(intervals):
    result = []
    for start, end in sorted((a, b) for a, b in intervals if b > a):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(b - a for a, b in union(intervals))


def intersection(left, right):
    left, right = union(left), union(right)
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if b > a:
            result.append((a, b))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def subtract(windows, busy):
    remaining = []
    for start, end in union(windows):
        cursor = start
        for a, b in union(busy):
            if b <= cursor or a >= end:
                continue
            if a > cursor:
                remaining.append((cursor, a))
            cursor = max(cursor, min(b, end))
        if cursor < end:
            remaining.append((cursor, end))
    return remaining


def analyse_trace(events):
    events = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) >= 0]
    intervals = lambda rows: [(e["ts"], e["ts"] + e["dur"]) for e in rows]
    windows = intervals([e for e in events if e.get("name") == "training/update"])
    kernels = [e for e in events if e.get("cat") in {"kernel", "cuda_kernel"}]
    copies = [e for e in events if "memcpy" in e.get("cat", "").lower()]
    phases = [e for e in events if e.get("name", "").startswith("training/")]
    if not windows:
        return {"status": "unavailable", "value": None,
                "reason": "No complete training/update ranges; refusing to guess a measurement window"}
    if not kernels:
        return {"status": "unavailable", "value": None, "reason": "No GPU kernel activities were captured",
                "host_phase_us": {name: duration(intervals([e for e in phases if e["name"] == name]))
                                  for name in sorted({e["name"] for e in phases})}}
    windows = union(windows)
    kernel_intervals = intersection(intervals(kernels), windows)
    copy_intervals = intersection(intervals(copies), windows)
    busy = union(kernel_intervals + copy_intervals)
    gaps = subtract(windows, busy)
    ranked = {}
    for event in kernels:
        elapsed = duration(intersection(intervals([event]), windows))
        if not elapsed:
            continue
        item = ranked.setdefault(event["name"], {"name": event["name"], "gpu_us": 0.0,
                                                "launches": 0, "examples": []})
        item["gpu_us"] += elapsed
        item["launches"] += 1
        if len(item["examples"]) < 3:
            item["examples"].append({"ts": event["ts"], "dur": event["dur"],
                                     "args": event.get("args", {})})
    phase_host = {}
    for name in sorted({e["name"] for e in phases}):
        phase_host[name] = duration(intersection(intervals([e for e in phases if e["name"] == name]), windows))
    # Kernel attribution requires launch correlation, not just overlap with host ranges.
    launches = {str(e.get("args", {}).get("correlation")): e for e in events
                if e.get("cat") in {"cuda_runtime", "cuda_driver"} and
                "correlation" in e.get("args", {})}
    phase_gpu = {}
    unattributed = []
    for kernel in kernels:
        launch = launches.get(str(kernel.get("args", {}).get("correlation")))
        candidates = [] if launch is None else [e for e in phases
            if e["name"] != "training/update" and e.get("tid") == launch.get("tid")
            and e["ts"] <= launch["ts"] < e["ts"] + e["dur"]]
        if candidates:
            name = min(candidates, key=lambda e: e["dur"])["name"]
            phase_gpu.setdefault(name, []).extend(intervals([kernel]))
        else:
            unattributed.extend(intervals([kernel]))
    explained = []
    for gap in gaps:
        candidates = sorted({e["name"] for e in phases if
            e["name"] in {"training/data_acquisition", "training/data_preparation"} and
            duration(intersection([gap], intervals([e]))) > 0})
        sync = [e for e in events if "synchroniz" in e.get("name", "").lower() and
                e.get("cat") in {"cuda_runtime", "cuda_driver"} and
                duration(intersection([gap], intervals([e]))) > 0]
        explained.append({"start_us": gap[0], "end_us": gap[1],
                          "overlapping_host_ranges": candidates,
                          "overlapping_synchronization": [e["name"] for e in sync],
                          "attribution": "correlated candidate, not proof of causality"
                          if candidates or sync else "unclassified"})
    return {"status": "available", "units": "microseconds", "window_us": duration(windows),
            "idle_fraction": duration(gaps) / duration(windows) if duration(windows) else None,
            "kernel_busy_us": duration(kernel_intervals), "copy_busy_us": duration(copy_intervals),
            "copy_compute_overlap_us": duration(intersection(kernel_intervals, copy_intervals)),
            "exposed_copy_us": duration(subtract(copy_intervals, kernel_intervals)),
            "gpu_busy_us": duration(busy), "gpu_idle_us": duration(gaps), "idle_gaps": explained,
            "host_phase_us": phase_host,
            "correlated_gpu_phase_us": {k: duration(intersection(v, windows)) for k, v in phase_gpu.items()},
            "unattributed_gpu_us": duration(intersection(unattributed, windows)),
            "kernels": sorted(ranked.values(), key=lambda x: x["gpu_us"], reverse=True),
            "limitations": "Kernel sums can exceed elapsed time across streams. Host overlaps are not causal proof."}


def load_chrome(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("Malformed Chrome trace: expected traceEvents list")
    return events


def load_nsys_sqlite(path):
    """Read the documented CUPTI/NVTX tables; unknown schemas fail explicitly."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "NVTX_EVENTS" not in tables or "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            raise ValueError("Unsupported Nsight Systems export: NVTX/kernel tables missing")
        strings = dict(connection.execute("SELECT id,value FROM StringIds")) if "StringIds" in tables else {}
        events = []
        mappings = {"NVTX_EVENTS": "nvtx", "CUPTI_ACTIVITY_KIND_KERNEL": "kernel",
                    "CUPTI_ACTIVITY_KIND_MEMCPY": "gpu_memcpy", "CUPTI_ACTIVITY_KIND_RUNTIME": "cuda_runtime"}
        for table, category in mappings.items():
            if table not in tables:
                continue
            for row in connection.execute(f'SELECT * FROM "{table}"'):
                row = dict(row)
                if row.get("end") is None or row.get("start") is None:
                    continue
                name = row.get("text") or strings.get(row.get("textId")) or strings.get(
                    row.get("demangledName", row.get("shortName", row.get("nameId")))) or category
                events.append({"ph": "X", "cat": category, "name": name,
                    "ts": row["start"] / 1000, "dur": (row["end"] - row["start"]) / 1000,
                    "tid": row.get("globalTid", row.get("streamId", 0)),
                    "args": {"correlation": row.get("correlationId"),
                             "stream": row.get("streamId"), "device": row.get("deviceId")}})
        return events
    finally:
        connection.close()
