"""Capture/import bounded PyTorch and optional NVIDIA profiles."""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
from transformer_training.diagnostics_cli import main

if __name__ == "__main__":
    raise SystemExit(main("profile"))
