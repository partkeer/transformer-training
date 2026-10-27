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
