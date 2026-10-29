"""Run a declared implementation suite sequentially in isolated processes."""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
from transformer_training.experiments import main

if __name__ == "__main__":
    raise SystemExit(main())
