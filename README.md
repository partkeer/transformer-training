# transformer-training

Train a small PyTorch transformer on TinyStories and compare training speed,
memory use, and validation quality across implementations.

## Setup

Use **Python 3.12** (3.13 is also allowed) with **PyTorch 2.6.0**. Run commands
from the repository root in an activated environment. GPU commands require an
NVIDIA GPU and a CUDA-enabled PyTorch build. Triton requires **Linux or WSL2**.

Create and activate the environment:

```bash
# Linux / WSL2
# Ubuntu prerequisite if ensurepip is missing: sudo apt install python3.12-venv
python3.12 -m venv .venv-linux
source .venv-linux/bin/activate
```

```powershell
# Windows / PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For GPU use, install PyTorch first. This CUDA 12.4 build supports the RTX 2070;
use a compatible NVIDIA driver. Other builds are listed in the
[PyTorch 2.6 installation instructions](https://pytorch.org/get-started/previous-versions/#v260):

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install the project, then check GPU access if applicable:

```bash
python -m pip install -e .
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

On rented machines, keep the project and `HF_HOME` cache on persistent storage.
For Triton on Linux/WSL2 x86-64, install the pinned optional backend with
`python -m pip install -e '.[triton]'` (Triton 3.2.0). Use a fresh environment
when switching from the previous Python 3.14/PyTorch 2.14 setup.

## Train and resume

Inspect settings without downloading data or starting training:

```bash
python scripts/train.py --dry-run
```

Start a baseline, a shorter sampled run, or resume a saved run:

```bash
python scripts/train.py --output-dir runs/baseline
python scripts/train.py --max-stories 1000 --max-steps 20 --output-dir runs/sample
python scripts/train.py --resume runs/baseline/latest.pt
```

Defaults: FP32, eager reference attention, standard AdamW, no activation
checkpointing, one epoch, batch size 8 and 8 accumulation steps. The first run
downloads and prepares data; even a sampled run checks both full splits for
overlap. Add `--offline` to use an existing cache.

Use a fresh output directory for each new run. It contains `latest.pt` for
resume, `best.pt` for evaluation, `metrics.jsonl`, and `summary.json`.
Resume inherits saved settings and continues from the last checkpoint.

## Evaluate a checkpoint

Report validation loss/perplexity and generate story continuations:

```bash
python scripts/evaluate.py --checkpoint runs/baseline/best.pt
```

Useful options: `--skip-generation`, `--full-validation`, and repeatable
`--prompt "Your story opening"`. Evaluation inherits the checkpoint's settings.
For a CPU-only machine, override them explicitly:

```bash
python scripts/evaluate.py --checkpoint runs/baseline/best.pt --device cpu --precision fp32 --attention reference --no-compile --batch-size 1
```

Outputs: `<checkpoint-directory>/evaluations/<UTC-id>/report.json` and
`samples.txt`. Use `--output-dir` to choose another empty directory.

## Switch implementations and compare them

These switches work with training, benchmarking, and profiling:

| Option | Values |
|---|---|
| `--precision` | `fp32`, `bf16`, `fp16` |
| `--loss-backend` | `pytorch` (default), `triton` (Linux/WSL2 NVIDIA SM75+, explicit opt-in) |
| `--attention` | `reference`, `sdpa` |
| `--sdpa-backend` | `auto`, `math`, `flash`, `efficient`, `cudnn` |
| `--compile` | Enable compilation; undo with `--no-compile` |
| `--compile-mode` | `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs` |
| `--optimizer-implementation` | `standard`, `foreach`, `fused` |
| `--activation-checkpointing` | Enable checkpointing; undo with `--no-activation-checkpointing` |

For example, benchmark BF16 with SDPA:

```bash
python scripts/benchmark_training.py --precision bf16 --attention sdpa --output-dir runs/bf16-sdpa
```

For fused cross-entropy, use `--loss-backend triton` with training or evaluation.
Unsupported requests fail with instructions to select `--loss-backend pytorch`.
See [kernel support, installation and correctness checks](docs/fused-cross-entropy.md).

```bash
python scripts/train.py --loss-backend triton --precision bf16 --output-dir runs/triton-training
python scripts/benchmark_loss.py --output-dir runs/loss-benchmark
python scripts/run_experiments.py --plan experiments/fused-cross-entropy.json --output-dir runs/loss-updates
```

On an RTX 2070, use FP32 or FP16; BF16 requires newer hardware. Start with a
smaller microbatch; this example keeps the default effective batch of 64:

```bash
python scripts/train.py --loss-backend triton --precision fp16 --batch-size 1 --gradient-accumulation-steps 64 --output-dir runs/2070-triton
```

Run the GPU correctness checks in the [kernel guide](docs/fused-cross-entropy.md)
before relying on the backend; RTX 2070 execution remains unverified here.

The first benchmark measures loss operations; the experiment plan compares
complete optimiser updates. Both write `report.json` and `report.md`.

For sequential comparisons, edit a copy of the
[example plan](experiments/implementations.json), then inspect, run, or resume it:

```bash
python scripts/run_experiments.py --plan experiments/implementations.json --dry-run
python scripts/run_experiments.py --plan experiments/implementations.json --output-dir runs/implementations
python scripts/run_experiments.py --plan experiments/implementations.json --output-dir runs/implementations --resume
```

The supplied plan runs **16 configurations with five repeats each**.
Open `runs/implementations/report.md` for comparisons; `report.csv` and
`report.json` provide the same results for further analysis. Use a new suite
directory after changing the plan, source, or environment. See the
[measurement guide](docs/measurements.md#implementation-experiment-plans)
for plan configuration and optional quality runs.

## Compare one and two GPUs

On one Linux/WSL2 machine with two NVIDIA GPUs and CUDA/NCCL-enabled PyTorch:

```bash
# Inspect settings without data access, CUDA initialisation or output creation.
python scripts/benchmark_gpu_scaling.py --dry-run

# Run real updates, keeping the global batch at 64 examples in both cases.
CUDA_VISIBLE_DEVICES=0,1 python scripts/benchmark_gpu_scaling.py \
  --batch-size 8 --gradient-accumulation-steps 8 \
  --repeats 5 --warmup-attempts 10 --measured-attempts 50 \
  --validation-batches 32 --output-dir runs/gpu-scaling
```

Accumulation is the **one-GPU** setting; two GPUs use half as many accumulation
steps per GPU with the same microbatch size. It must be even and at least two.
Existing precision and implementation switches, including `--loss-backend triton`,
are supported. Reduce `--batch-size` and increase accumulation if needed for VRAM.

Open `runs/gpu-scaling/report.md` for speedup, efficiency, memory and loss checks;
JSON/CSV and per-rank logs retain the measurements. Validation runs after timing;
use `--validation-batches 0` to skip it or `--full-validation` for the selected split.
This command benchmarks updates; ordinary training/resume remains single-GPU.
See [comparison rules and metrics](docs/measurements.md#one-versus-two-gpu-scaling).

## Measure speed and memory

These diagnostics perform real training updates and need fresh output directories:

```bash
# Benchmark warmed training updates: tokens/s, update time, and peak GPU memory.
python scripts/benchmark_training.py --output-dir runs/benchmark

# Profile CPU/GPU activity and training phases to locate bottlenecks.
python scripts/profile_training.py --backend pytorch --output-dir runs/profile-timeline

# Capture memory usage over time and break it down by tensor category.
python scripts/profile_training.py --backend memory --output-dir runs/profile-memory
```

Each writes `report.md`, `report.json`, and supporting artifacts. Benchmark
defaults are five repeats, ten warm-up attempts, and fifty measured attempts;
change them with `--repeats`, `--warmup-attempts`, and `--measured-attempts`.
Use benchmarks for speed comparisons; profiling affects execution time.

To report an existing training run without starting more training:

```bash
python scripts/report_measurements.py --run-dir runs/baseline
```

Output: `runs/baseline/measurements-report/report.md` and plots. See the
[measurement guide](docs/measurements.md) for metric definitions, controlled
comparisons, checkpoint-based diagnostics, and Nsight commands.

## Tests and further help

Run offline tests with synthetic data; GPU checks skip when CUDA is unavailable:

```bash
python -m unittest discover -s tests -v
```

Add `--help` to any script for its full options. Training, benchmarking,
profiling, and experiment suites support `--dry-run` to inspect settings.
Additional data-loading and evaluation commands are in [scripts/README.md](scripts/README.md).
