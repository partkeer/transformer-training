"""Evaluate a saved TinyStories checkpoint without starting training."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from transformer_training.evaluation_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
