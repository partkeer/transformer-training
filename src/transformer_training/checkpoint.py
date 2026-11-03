"""Atomic artifacts and portable optimizer-boundary RNG snapshots."""

import json
import os
from pathlib import Path
import random
import tempfile

import torch


def _atomic_write(path, writer):
    """Pass a binary temporary stream to writer, flush it to disk, then atomically
    replace the destination. Create parent directories and clean up the temporary
    file if writing fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    """Atomically write indented UTF-8 JSON with a trailing newline, rejecting NaN and
    infinity so reports remain valid JSON.
    """
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(encoded))


def atomic_text(path, value):
    """Atomically replace a UTF-8 text artifact."""
    encoded = value.encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(encoded))


def save_checkpoint(path, payload):
    """Serialise the supplied payload with torch.save and atomically replace its
    destination, avoiding a partially written checkpoint.
    """
    _atomic_write(path, lambda stream: torch.save(payload, stream))


def load_checkpoint(path):
    """Load a checkpoint onto CPU with PyTorch's weights-only loader, leaving device
    placement and compatibility checks to the caller.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def capture_rng():
    """Return Python, PyTorch CPU, and available CUDA generator states for saving at an
    accumulation boundary.
    """
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    """Restore saved Python and PyTorch generator states, rejecting CUDA state when the
    available device count differs. This does not guarantee identical results across
    hardware or library versions.
    """
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG state does not match available devices")
        torch.cuda.set_rng_state_all(state["cuda"])
