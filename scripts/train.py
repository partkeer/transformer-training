"""Train TinyStories, or inspect its configuration safely with --dry-run."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from transformer_training.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
