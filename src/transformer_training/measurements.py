"""Versioned invocation-local accounting; no training or device initialization on import."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

MEASUREMENT_VERSION = 1


def unavailable(reason, status="unavailable"):
    return {"status": status, "value": None, "reason": reason}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", *args],
            cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    try:
        revision, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
        # Include untracked source contents, not just the tracked diff.
        files = sorted(set(git("ls-files").splitlines() +
                           git("ls-files", "--others", "--exclude-standard").splitlines()))
        contents = hashlib.sha256()
        for name in files:
            path = root / name
            contents.update(name.encode())
            contents.update(path.read_bytes() if path.is_file() else b"<deleted>")
        return {"revision": revision, "dirty": bool(status), "content_sha256": contents.hexdigest()}
    except (OSError, subprocess.SubprocessError):
        return unavailable("Git source identity could not be read")


@dataclass
class Invocation:
    warmup: int
    clock: object = time.perf_counter
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attempts: int = 0
    updates: int = 0
    processed_tokens: int = 0
    applied_tokens: int = 0
    training_seconds: float = 0.0
    measured_attempts: int = 0
    measured_processed_tokens: int = 0
    measured_applied_tokens: int = 0
    measured_seconds: float = 0.0
    numerical_failures: int = 0

    def __post_init__(self):
        self.started = self.clock()

    def record(self, tokens, seconds, skipped):
        self.attempts += 1
        self.updates += int(not skipped)
        self.processed_tokens += tokens
        self.applied_tokens += 0 if skipped else tokens
        self.training_seconds += seconds
        if self.attempts > self.warmup:
            self.measured_attempts += 1
            self.measured_processed_tokens += tokens
            self.measured_applied_tokens += 0 if skipped else tokens
            self.measured_seconds += seconds

    def snapshot(self, *, elapsed=None, status="running"):
        elapsed = self.clock() - self.started if elapsed is None else elapsed
        measured = self.measured_seconds
        return {
            "version": MEASUREMENT_VERSION, "invocation_id": self.id, "status": status,
            "lifecycle_seconds": elapsed, "training_seconds": self.training_seconds,
            "attempts": self.attempts, "successful_updates": self.updates,
            "skipped_updates": self.attempts - self.updates,
            "processed_tokens": self.processed_tokens, "applied_tokens": self.applied_tokens,
            "skipped_tokens": self.processed_tokens - self.applied_tokens,
            "warmup_attempts": self.warmup, "measured_attempts": self.measured_attempts,
            "measured_seconds": measured,
            "measured_processed_tokens": self.measured_processed_tokens,
            "measured_applied_tokens": self.measured_applied_tokens,
            "warmed_useful_tokens_per_second": self.measured_applied_tokens / measured if measured else None,
            "warmed_processed_tokens_per_second": self.measured_processed_tokens / measured if measured else None,
            "warmed_status": "available" if measured else "unavailable",
            "warmed_reason": None if measured else "No complete attempts remain after warm-up",
            "whole_run_useful_tokens_per_second": self.applied_tokens / elapsed if elapsed > 0 else None,
            "whole_run_processed_tokens_per_second": self.processed_tokens / elapsed if elapsed > 0 else None,
            "numerical_failures": self.numerical_failures,
            "scope": "after data preparation, before trainer construction through final checkpoint",
        }


class Phases:
    """Optional host ranges and NVTX. Never synchronize within a phase."""
    def __init__(self, enabled=False, nvtx=False):
        self.enabled, self.nvtx = enabled, nvtx
        self.host_seconds = {}

    @contextmanager
    def range(self, name):
        if not self.enabled:
            yield
            return
        import torch
        start = time.perf_counter()
        with torch.profiler.record_function("training/" + name):
            if self.nvtx:
                torch.cuda.nvtx.range_push("training/" + name)
            try:
                yield
            finally:
                if self.nvtx:
                    torch.cuda.nvtx.range_pop()
                self.host_seconds[name] = self.host_seconds.get(name, 0.0) + time.perf_counter() - start


class DeviceMemorySampler:
    """One persistent nvidia-smi process, selected by CUDA UUID, sampled every 200 ms."""
    def __init__(self, gpu_uuid=None, interval_ms=200):
        self.uuid, self.interval_ms = gpu_uuid, interval_ms
        self.samples, self.process, self.thread = [], None, None
        self.status = unavailable("CUDA UUID unavailable; device-wide sampling was not started")

    def start(self):
        if not self.uuid:
            return self
        try:
            self.process = subprocess.Popen([
                "nvidia-smi", "--id=" + self.uuid,
                "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits",
                "--loop-ms=" + str(self.interval_ms)], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True)
            self.status = {"status": "collecting", "value": None, "reason": None}
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except OSError as error:
            self.status = unavailable(str(error))
        return self

    def _read(self):
        for line in self.process.stdout:
            try:
                gpu, mib = next(csv.reader([line]))
                if gpu.strip() == self.uuid:
                    self.samples.append({"monotonic_seconds": time.perf_counter(),
                                         "device_used_bytes": int(float(mib) * 1024**2)})
            except (ValueError, StopIteration):
                continue

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.thread.join(timeout=5)
            self.process.stdout.close()
        if self.samples:
            self.status = {"status": "available", "value": max(x["device_used_bytes"] for x in self.samples),
                           "reason": None}
        elif self.process is not None:
            self.status = unavailable("nvidia-smi returned no valid samples")
        return {**self.status, "sample_interval_ms": self.interval_ms,
                "sample_count": len(self.samples), "samples": self.samples,
                "first_sample_monotonic_seconds": self.samples[0]["monotonic_seconds"] if self.samples else None,
                "last_sample_monotonic_seconds": self.samples[-1]["monotonic_seconds"] if self.samples else None,
                "scope": "sampled device-wide usage, including other processes; not an exact peak"}
