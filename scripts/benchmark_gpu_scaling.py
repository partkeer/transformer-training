"""Compare fixed-global-batch training performance on one GPU and two-GPU DDP."""
import os
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1] / '.cache' / 'huggingface'))
from transformer_training.gpu_scaling import main

if __name__ == '__main__':
    raise SystemExit(main())
