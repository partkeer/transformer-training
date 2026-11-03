"""Pinned-version memory adapter and storage-deduplicated autograd lifetime tracking."""

import json
from pathlib import Path
import time

import torch

from .checkpoint import atomic_json
from .measurements import unavailable


def storage_record(tensor):
    """Return a tensor storage's device, address, and byte size, allowing views and
    aliases to be counted as one allocation.
    """
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def unique_storage_bytes(tensors):
    """Sum bytes across distinct tensor storages, ignoring None entries and counting
    shared storage only once.
    """
    return sum(key[2] for key in {storage_record(t) for t in tensors if t is not None})


def static_memory(model, optimizer):
    """Return separate live-storage totals for parameters, gradients, and materialised
    optimiser state. Deduplicate aliases within each category and distinguish
    optimiser state held on other devices.
    """
    params = list(model.parameters())
    def tensors(value):
        """Yield tensors recursively from nested dictionaries, tuples, and lists in
        optimiser state, ignoring nontensor metadata.
        """
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
    state = list(tensors(optimizer.state))
    device = params[0].device if params else torch.device("cpu")
    return {"device": str(device), "parameters_bytes": unique_storage_bytes(params),
            "gradients_bytes": unique_storage_bytes(p.grad for p in params),
            "optimizer_state_bytes": unique_storage_bytes(t for t in state if t.device == device),
            "optimizer_state_other_device_bytes": unique_storage_bytes(t for t in state if t.device != device),
            "scope": "live storage at observation, includes only materialized optimiser state"}


class _Saved:
    def __init__(self, tensor, tracker):
        # The packed value owns storage, not the original grad_fn (which can create a cycle).
        """Retain a detached tensor for autograd's saved-tensor hook and register its
        storage, avoiding ownership of the original grad_fn.
        """
        self.tensor, self.tracker = tensor.detach(), tracker
        self.key = storage_record(tensor)
        tracker.add(self.key)

    def __del__(self):
        """Release this packed tensor's storage reference count when autograd no longer
        owns it, allowing lifetimes to end without backward as well.
        """
        self.tracker.remove(self.key)


class SavedActivations:
    """Count storage once while autograd owns packed tensors; no extra tensor references."""
    def __init__(self, model):
        """Record parameter storage identities to exclude from activation totals and
        initialise reference counts, event history, and peak observations.
        """
        self.parameters = {storage_record(p) for p in model.parameters()}
        self.live, self.events, self.peak, self.peak_storages = {}, [], 0, []

    def add(self, key):
        """Increment ownership of a saved storage key and record the live nonparameter
        storage total without double-counting aliases.
        """
        self.live[key] = self.live.get(key, 0) + 1
        self._record("save", key)

    def remove(self, key):
        """Release one ownership of a storage key, remove it after the final reference,
        and record the resulting live total.
        """
        self.live[key] -= 1
        if not self.live[key]:
            del self.live[key]
        self._record("release", key)

    def _record(self, action, key):
        """Append a timestamped save/release event and update the maximum simultaneous
        nonparameter storage and its identities.
        """
        keys = [k for k in self.live if k not in self.parameters]
        size = sum(k[2] for k in keys)
        if size > self.peak:
            self.peak, self.peak_storages = size, keys.copy()
        self.events.append({"time_ns": time.time_ns(), "action": action, "storage": key,
                            "saved_nonparameter_bytes": size})

    def __enter__(self):
        """Install autograd hooks that wrap saved tensors in tracked detached owners
        and return this tracker for context-managed reporting.
        """
        self.hooks = torch.autograd.graph.saved_tensors_hooks(lambda t: _Saved(t, self), lambda obj: obj.tensor)
        self.hooks.__enter__()
        return self

    def __exit__(self, *args):
        """Remove saved-tensor hooks on context exit, forwarding exception information
        and the underlying context manager's result.
        """
        return self.hooks.__exit__(*args)

    def report(self):
        """Return the peak and event history for saved nonparameter storage. This can
        include inputs and is separate from total tensor or CUDA allocator memory.
        """
        return {"peak_saved_nonparameter_storage_bytes": self.peak,
                "storages_at_saved_peak": self.peak_storages, "events": self.events,
                "scope": "saved-for-backward storage excluding parameters; may include inputs; separate from allocator peak"}


class MemoryAdapter:
    """Contain private APIs required by PyTorch 2.14; never silently invent attribution."""
    def __init__(self, device):
        """Resolve the requested torch device, making an unindexed CUDA device
        explicit, and initialise allocation-history recording as disabled.
        """
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.history = False

    def start(self):
        """Enable bounded CUDA allocator history and capture its initial snapshot.
        Return availability metadata, explicitly reporting unsupported CPU execution
        or private-API failures.
        """
        if self.device.type != "cuda":
            return unavailable("CUDA allocation history is unavailable on CPU")
        try:
            torch.cuda.memory._record_memory_history(max_entries=200000, device=self.device)
            self.history = True
            self.initial = torch.cuda.memory._snapshot(device=self.device)
            return {"status": "available", "max_entries": 200000}
        except (RuntimeError, AttributeError) as error:
            return unavailable(str(error))

    def finish(self, profiler, directory):
        """Export allocator and profiler memory artifacts into the directory and return
        available peak/category evidence. Disable active allocation-history
        recording and report missing attribution explicitly rather than estimating
        it.
        """
        directory = Path(directory)
        result = {}
        if self.history:
            try:
                snapshot = torch.cuda.memory._snapshot(device=self.device)
                torch.cuda.memory._dump_snapshot(str(directory / "allocator_snapshot.pickle"))
                atomic_json(directory / "allocator_snapshot.json", snapshot)
                allocator_peak = replay_allocator_peak(self.initial, snapshot, self.device.index
                    if self.device.index is not None else torch.cuda.current_device())
                result["allocator_peak_live_allocations"] = allocator_peak
                result["allocator_snapshot"] = "allocator_snapshot.pickle"
                result["allocator_history"] = {"status": "available", "max_entries": 200000,
                    "coverage": "bounded ring history; oldest events can be evicted"}
            except (RuntimeError, AttributeError, TypeError) as error:
                result["allocator_history"] = unavailable(str(error))
            finally:
                torch.cuda.memory._record_memory_history(enabled=None, device=self.device)
                self.history = False
        else:
            result["allocator_history"] = unavailable("CUDA allocator recording was not enabled")
        try:
            # Plain JSON avoids PyTorch's Windows NamedTemporaryFile gzip locking bug.
            path = directory / "memory_timeline.json"
            profiler.export_memory_timeline(str(path), device=str(self.device))
            times, sizes = json.loads(path.read_text())
            from torch.profiler._memory_profiler import _CATEGORY_TO_INDEX
            index = max(range(len(sizes)), key=lambda i: sum(sizes[i]))
            categories = {c.name.lower() if c else "unattributed": sizes[index][i + 1]
                          for c, i in _CATEGORY_TO_INDEX.items()}
            result["tensor_peak"] = {"status": "available", "timestamp_us": times[index],
                "bytes": sum(sizes[index]), "categories": categories,
                "scope": "simultaneous profiler-inferred tensor categories, not reserved allocator memory"}
            # Allocation identities complement source stacks in the CUDA allocator snapshot.
            profile = profiler._memory_profile()
            live = {}
            peak_live, maximum = [], 0
            from torch.profiler._memory_profiler import Action
            for timestamp, action, (key, version), numbytes in profile.timeline:
                if key.device != self.device:
                    continue
                identity = str(key)
                if action in {Action.CREATE, Action.PREEXISTING}:
                    live[identity] = {"identity": identity, "bytes": numbytes,
                                      "created_ns": timestamp, "version": version}
                elif action == Action.DESTROY:
                    live.pop(identity, None)
                if sum(x["bytes"] for x in live.values()) > maximum:
                    maximum = sum(x["bytes"] for x in live.values())
                    peak_live = [dict(x) for x in live.values()]
            result["live_allocations_at_tensor_peak"] = peak_live
            result["timeline"] = path.name
            if self.device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(self.device)
                result["allocator_minus_categorized_peak_bytes"] = peak - sum(sizes[index])
                result["reconciliation_note"] = "Independent peaks; difference is not a simultaneous temporary allocation estimate"
        except (RuntimeError, ValueError, AttributeError, IndexError, KeyError) as error:
            result["tensor_peak"] = unavailable(str(error))
        return result


def replay_allocator_peak(initial, final, device_index):
    """Return simultaneous peak active bytes and allocations for one CUDA device.

    Replay final allocation events from the initial snapshot, retaining pending
    frees until completion. Preserve stack evidence and preexisting allocations;
    bounded or evicted history can make the reconstruction incomplete.
    """
    live = {}
    for segment in initial.get("segments", []):
        if segment["device"] != device_index:
            continue
        address = segment["address"]
        for block in segment["blocks"]:
            if block["state"] in {"active_allocated", "active_awaiting_free"}:
                live[address] = {"address": address, "size": block["size"],
                                 "frames": block.get("frames", []), "preexisting": True}
            address += block["size"]
    peak = sum(x["size"] for x in live.values())
    peak_live = list(live.values())
    traces = final.get("device_traces", [])
    events = traces[device_index] if device_index < len(traces) else []
    for event in events:
        if event["action"] == "alloc":
            live[event["addr"]] = {"address": event["addr"], "size": event["size"],
                                    "frames": event.get("frames", []), "preexisting": False}
        elif event["action"] == "free_completed":
            live.pop(event["addr"], None)
        size = sum(x["size"] for x in live.values())
        if size > peak:
            peak, peak_live = size, list(live.values())
    return {"active_bytes": peak, "allocations": peak_live,
            "scope": "allocator history reconstruction including pending frees; bounded history may be incomplete",
            "history_events": len(events)}
