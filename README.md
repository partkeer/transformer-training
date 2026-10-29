# transformer-training

Train a small decoder-only language model on TinyStories with PyTorch, then
measure training speed, GPU memory use, and model quality. Training, checkpoint
evaluation, benchmarking, and profiling commands are available. A custom GPU
kernel remains planned; profiling will determine which bottleneck to optimise.

## Setup

Run all commands from the repository root. Use **Python 3.14** and, for GPU
training, an NVIDIA GPU with a compatible CUDA-enabled **PyTorch 2.14.1** build.

### Linux / GPU

Create and activate an environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

Choose a CUDA wheel index using the [official PyTorch installation guidance](https://pytorch.org/get-started/locally/).
Replace the placeholder below with an index compatible with PyTorch 2.14.1,
Python 3.14, your GPU, and the host driver. Then install PyTorch and this project:

```bash
export PYTORCH_CUDA_INDEX_URL='<compatible-official-CUDA-wheel-index>'
python -m pip install 'torch==2.14.1' --index-url "$PYTORCH_CUDA_INDEX_URL"
python -m pip install -e .
python -m pip check
```

Check that this environment can access the GPU before starting training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

On rented machines, keep the repository (including `runs/`) on persistent
storage, and set `HF_HOME` to a persistent cache directory before running the
scripts. Otherwise, downloads and checkpoints may be lost with the instance.

### Windows / PowerShell

For local inspection and CPU evaluation, create and activate an environment
using Python 3.14, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

For GPU use, install a compatible CUDA-enabled PyTorch 2.14.1 build using the
installation guidance above, then run the same GPU check. All commands below
use the activated environment's `python` and work with forward-slash paths.

## Train and resume

Inspect settings and model size without downloading data, creating a run
directory, or running training. This also works on a CPU-only machine:

```bash
python scripts/train.py --dry-run
```

Start a baseline run on the GPU:

```bash
python scripts/train.py --output-dir runs/baseline
```

Defaults: approximately **10 million parameters**, a **512-token context**,
**FP32** precision, batches of **8** with **8 accumulation steps** (64 windows
per full update), and **one training epoch**.

The first run downloads the data and tokenizer, checks both full data splits
for exact overlap, and prepares training inputs. Allow several GB of disk space
and time for preparation. Subsequent runs reuse the cache, which defaults to
`.cache/huggingface/` unless `HF_HOME` is set.

For a shorter trial, sample up to 1,000 stories per split and train for at most
20 optimiser updates. The first download and full overlap check still apply:

```bash
python scripts/train.py --max-stories 1000 --max-steps 20 --output-dir runs/sample
```

Use a new or empty output directory for each new run. Training writes checkpoints,
settings, metrics, and a summary into that directory:

| File | Purpose |
|---|---|
| `latest.pt` | Resume from the most recently saved training state |
| `best.pt` | Evaluate the checkpoint with the lowest observed validation loss |
| `metrics.jsonl` | Training and validation measurements over time |
| `summary.json` | Run summary and measurements |

Resume an interrupted baseline from its last saved checkpoint; saved model,
training settings, and data selection are inherited:

```bash
python scripts/train.py --resume runs/baseline/latest.pt
```

`latest.pt` is saved every 500 successful updates and at completion. Work after
the last saved checkpoint must be repeated after an interruption.

## Evaluate a checkpoint

After training, measure validation loss and perplexity and generate three story
continuations using the best checkpoint:

```bash
python scripts/evaluate.py --checkpoint runs/baseline/best.pt
```

Lower loss and perplexity indicate better next-token prediction when evaluation
conditions match. By default, evaluation uses the checkpoint's saved validation
selection, device, precision, and batch size. Add `--full-validation` to evaluate
the entire validation split, or `--skip-generation` for metrics only.

Replace the default story openings with your own prompt:

```bash
python scripts/evaluate.py --checkpoint runs/baseline/best.pt --prompt "Lily found a lost kitten."
```

To evaluate a GPU-trained checkpoint on a CPU-only machine, explicitly select
CPU and FP32; batch size 1 reduces memory use:

```bash
python scripts/evaluate.py --checkpoint runs/baseline/best.pt --device cpu --precision fp32 --batch-size 1
```

Results print to the terminal and are saved under
`runs/baseline/evaluations/<UTC-id>/`: `report.json` contains metrics and settings;
`samples.txt` contains readable story continuations. Use `--output-dir` to choose
a different empty destination. Evaluation does not modify the checkpoint.

## Measure speed and memory

After baseline training, generate a report and validation-loss plots from its
saved measurements. This command reads existing results and starts no training:

```bash
python scripts/report_measurements.py --run-dir runs/baseline
```

Open `runs/baseline/measurements-report/report.md` for the report and plot links.
Use a fresh `--output-dir` when generating another report.

The following GPU diagnostics start fresh default models and perform real
training updates. They do not require a checkpoint or assess trained-model
quality. Use a fresh output directory each time.

Benchmark training speed and peak memory with five repeated trials, each with
10 warm-up attempts followed by 50 measured attempts:

```bash
python scripts/benchmark_training.py --repeats 5 --warmup-attempts 10 --measured-attempts 50 --output-dir runs/benchmark
```

Capture a PyTorch operator profile and execution timeline to investigate where
training time is spent:

```bash
python scripts/profile_training.py --backend pytorch --output-dir runs/profile-timeline
```

Capture a detailed memory profile to investigate allocations and tensor storage:

```bash
python scripts/profile_training.py --backend memory --output-dir runs/profile-memory
```

Each diagnostic writes `report.md`, `report.json`, and supporting artifacts into
its output directory. Profiles default to 10 warm-up attempts and 5 captured
attempts. Use benchmarks for speed comparisons; profiling affects execution time.
Keep model, data, precision, batch settings, and hardware consistent when comparing.

See the [measurement guide](docs/measurements.md) for metric definitions,
checkpoint-based diagnostics, comparison options, and NVIDIA Nsight commands.
Checkpoint-based diagnostics need enough remaining training steps for warm-up
and measurement; a checkpoint at the end of its training schedule cannot supply them.

## Tests and further help

Run the offline correctness tests, which use synthetic data rather than
TinyStories downloads. GPU-specific checks skip when CUDA is unavailable:

```bash
python -m unittest discover -s tests -v
```

List available options, or inspect diagnostic settings without running updates:

```bash
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/benchmark_training.py --help
python scripts/profile_training.py --help
python scripts/benchmark_training.py --dry-run
python scripts/profile_training.py --dry-run --backend all
```

Add `--offline` to training, evaluation, or diagnostics to use cached data and
tokenizer files only; missing cache entries cause an error. Detailed measurement
instructions are in [docs/measurements.md](docs/measurements.md).
