# TinyStories commands

Run these PowerShell commands from the repository root using Python 3.14.

```powershell
# Create the environment if needed, then install the package in editable mode.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# Prepare a seeded sample and inspect two batches from each official split.
.\.venv\Scripts\python.exe scripts\load_tinystories.py --max-stories 1000 --sequence-length 256 --batch-size 8

# Prepare all stories. --batches limits inspection, not preprocessing.
.\.venv\Scripts\python.exe scripts\load_tinystories.py --sequence-length 256 --batch-size 8 --batches 2

# Reuse cached source data and tokenizer with no Hub requests.
.\.venv\Scripts\python.exe scripts\load_tinystories.py --offline --max-stories 1000

# Check Windows worker-process loading; uses the same cached token windows.
.\.venv\Scripts\python.exe scripts\load_tinystories.py --offline --max-stories 1000 --num-workers 2

# Offline correctness tests, including next-token targets and loss masking.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Existing raw-text EDA (does not use the new tokenisation pipeline).
.\.venv\Scripts\python.exe scripts\explore_tinystories.py
```

The loader command writes `reports/tinystories/dataloader.json`; use
`--output reports/tinystories/my-run.json` to retain a separate record. Use
`--revision <dataset-sha>` and `--tokenizer-revision <tokenizer-sha>` to select
explicit commits. Defaults are pinned. Online branch/tag overrides are resolved
to commits and recorded. Offline mode requires a populated cache and immutable
revisions; it cannot download missing artifacts.

`--max-stories` samples original rows independently in each split, without
replacement, using seeds `seed` and `seed + 1`. Blank stories are then skipped,
so this option limits selected source rows rather than usable stories or
resulting windows. Omit it to process the full splits. Downloads still load the
full source dataset on the first run. Token windows are cached in Arrow format;
the loader does not assemble the whole tokenised corpus as a Python list.

PyTorch is pinned in `requirements.txt`. The standard installation used here is
CPU-only on Windows; a compatible GPU-enabled PyTorch build is a separate setup
choice. `--pin-memory` requests pinned CPU batches for later GPU transfer; it
does not move tensors to a GPU or measure performance.

## Training command

`train.py` is the thin entrypoint for the reusable decoder and trainer. The
[training section in the root README](../README.md#train-the-reference-decoder)
documents architecture, Linux/CUDA installation, leakage filtering, metrics,
precision choices and interruption recovery.

```powershell
# Safe inspection only: no downloads, GPU allocation or optimization.
.\.venv\Scripts\python.exe scripts\train.py --dry-run
.\.venv\Scripts\python.exe scripts\train.py --help

# The following commands START training when explicitly run on a configured GPU.
.\.venv\Scripts\python.exe scripts\train.py --output-dir runs\tinystories-fp32
.\.venv\Scripts\python.exe scripts\train.py --resume runs\tinystories-fp32\latest.pt
```

The default is one cleaned TinyStories epoch using a 33.7M-parameter model,
512-token context, FP32, microbatch size 8 and 8 accumulation steps. Training
always audits both full source splits and excludes exact validation matches from
training before optional `--max-stories` sampling. Source loading, audit and
tokenisation are outside measured training time. Nothing starts merely by
importing the package; `--dry-run` also bypasses data loading entirely.
