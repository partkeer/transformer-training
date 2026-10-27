# transformer-training

Build a small decoder-only transformer training system in PyTorch using the
TinyStories dataset, then systematically improve its training speed and memory
efficiency.

The focus is on understanding and measuring the effects of optimisation through
profiling, controlled experiments, and one carefully chosen custom GPU kernel.

## Project scope

- **Model:** a small decoder-only transformer trained with a causal language
  modelling objective: predict the next token using only preceding tokens.
- **Framework:** PyTorch.
- **Dataset:** TinyStories.
- **Optimisation goals:** faster training and lower GPU memory usage while
  preserving training correctness and comparable model quality.
- **Custom kernel:** implement one GPU kernel for a bottleneck identified by
  profiling, and compare it against the corresponding PyTorch implementation.

## Planned workflow

1. Establish a correct, reproducible baseline for data preparation, training, and
   validation.
2. Profile representative training steps to identify compute and memory
   bottlenecks.
3. Run controlled experiments, changing one factor at a time wherever practical.
4. Select and implement one custom GPU kernel where profiling evidence supports
   the choice.
5. Compare results against the baseline and document gains, limitations, and
   trade-offs.

## Measurement and experiment discipline

Track training throughput (tokens per second), training step time, peak GPU memory
usage, and validation loss. Record the model configuration, tokenisation and data
splits, sequence length, effective batch size, precision, hardware, software
versions, and random seed for each experiment.

Use warm-up steps and appropriate GPU synchronisation for timing. Keep benchmark
conditions consistent, separate training-step measurements from end-to-end data
loading costs, and repeat measurements to assess variability. If an optimisation
requires changing the workload or numerical precision, record that explicitly
and explain how it affects the comparison.

Check numerical correctness, including gradients where applicable, before
interpreting a speedup. Report memory savings and speed improvements alongside
any effect on validation loss or training stability.

## Current status

TinyStories exploratory analysis and next-token PyTorch dataloaders are
implemented. The model, training loop, profiling experiments, and custom GPU
kernel remain planned work; no training or performance results are claimed yet.

## Load TinyStories and explore the data

Use Python 3.14 and PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\explore_tinystories.py
```

The script downloads and loads the official
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
train and validation splits using Hugging Face `datasets`. Allow several GB of
disk space for downloads and the prepared Arrow cache. Downloads are reused on
subsequent runs; caches live under `.cache/` and are ignored by Git, as is `.venv/`.
The original splits remain separate and unchanged.

By default, EDA uses a uniform random sample of 10,000 stories **per split**,
without replacement (or the entire split if smaller). The train seed is 42 and
the validation seed is 43. This avoids treating a potentially ordered prefix as
representative. The entire dataset is still downloaded and loaded; sampling
reduces analysis work, not the initial download.

Outputs in [`reports/tinystories/`](reports/tinystories/):

- `report.md`: split sizes, schema, word-length summaries, frequent words, example
  excerpts, missing/empty text, duplicates, overlap, and interpretation.
- `length_distributions.png`: character and word-length histograms by split.
- `summary.json`: detailed descriptive statistics, selected original row indices,
  resolved Hub commit, dataset fingerprints, seed, and environment versions.

Word counts use lowercase English-letter regex matches with internal apostrophes
retained, including stop words. These are **not tokenizer lengths**. Empty strings
contribute zero lengths; missing/null and non-string values are counted separately
and excluded from length statistics. Duplicate checks compare unchanged UTF-8
text via SHA-256, within and across the selected samples. They cannot establish
that the full splits are free of leakage or near duplicates.

Use `--sample-size 0` to analyse every row (more time and memory), or change the
sample size, seed, revision, and output directory explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\explore_tinystories.py --sample-size 20000 --seed 42 --revision <commit-sha> --output-dir reports\tinystories-20k
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For an exact dataset rerun, use the commit SHA in `summary.json` as `--revision`.
The default `main` is resolved to an immutable commit before loading. Package
versions used for analysis are also recorded there; the direct dependencies are
pinned in `requirements.txt`. The dataloader below uses GPT-2 tokenisation;
context-length and packing choices still need tokenizer-specific analysis and
controlled experiments before training.

### Initial EDA findings

The completed run used dataset commit
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, with 10,000 sampled stories per split:

| Measure | Training | Validation |
|---|---:|---:|
| Full split rows | 2,119,719 | 21,990 |
| Mean words per sampled story | 175.7 | 171.5 |
| Median words | 153 | 151 |
| 95th percentile words | 352 | 331 |
| Maximum sampled words | 886 | 840 |
| Empty/whitespace sampled stories | 4 | 0 |
| Exact duplicate sampled rows beyond the first | 14 | 0 |

There were **10 distinct exact texts shared between the two samples**. This is
observed overlap, not a full-corpus estimate. Before training, filter empty
stories and audit full-split duplicates; keep validation fixed and remove matching
training stories if cleaning overlap. Record the cleaning policy and new counts.
No cleaning is performed by this exploratory script.

The sampled distributions suggest broadly similar story lengths, with a long
right tail. About 11.7% of sampled training stories and 9.7% of sampled validation
stories exceed 256 regex words. These percentages do not predict how many stories
would exceed a 256-token context. See the [generated report](reports/tinystories/report.md)
for plots, frequent words, excerpts, definitions, and limitations.

## Next-token PyTorch dataloaders

Install the source package and inspect a small, reproducible sample:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\load_tinystories.py --max-stories 1000 --sequence-length 256 --batch-size 8
```

[`scripts/README.md`](scripts/README.md) contains full-data, offline, worker-process,
EDA, and test commands. The reusable implementation is
[`src/transformer_training/data.py`](src/transformer_training/data.py):

```python
from transformer_training.data import load_tinystories

loaders = load_tinystories(
    sequence_length=256, batch_size=8, seed=42, max_stories=1000,
)
batch = next(iter(loaders.train))
validation_batch = next(iter(loaders.validation))
```

The CLI keeps Hugging Face caches in `.cache/huggingface/`. When using the Python
API directly, set `HF_HOME` before importing the module to use the same cache;
otherwise Hugging Face uses its configured/default cache location. The CLI's
`--offline` flag sets offline mode before importing Hugging Face libraries.

The pretrained GPT-2 byte-level BPE tokenizer is fixed at commit
`607a30d783dfa663caf39e06633721c8d4cfcd7e`; the dataset defaults to the EDA commit
above. No tokenizer is trained or adapted on TinyStories. GPT-2 has 50,257 token
IDs: use `loaders.metadata['vocab_size']` for the future model's vocabulary size.

Each nonblank story is encoded independently, with EOS appended. Long stories
become windows of at most `sequence_length + 1` tokens with one-token overlap.
For example, with sequence length 3, the story token stream
`[a, b, c, d, e, EOS]` produces:

```text
input_ids: [a, b, c]      labels: [b, c, d]
input_ids: [d, e, EOS]    labels: [e, EOS, -100]
```

The last input position in the second row is padding (EOS is reused as the pad
ID). Its attention mask is `[True, True, False]`. Genuine EOS **targets** remain
supervised; padding targets alone receive `-100`. Every next-token pair within a
story is retained exactly once, including at chunk boundaries. The first story
token has no preceding context and is not a target. Context resets at chunk
boundaries; windows never cross stories and short tails are kept. This favours a
clear baseline over packing efficiency; padding and short-tail overhead should
be measured before optimising it.

Each batch contains CPU tensors of shape `[batch, sequence_length]`:

| Key | dtype | Meaning |
|---|---|---|
| `input_ids` | `torch.int64` | Token IDs, right-padded with EOS |
| `labels` | `torch.int64` | Already shifted next-token targets; padding is `-100` |
| `attention_mask` | `torch.bool` | `True` for real input positions |

The final batch may be smaller than `batch_size`; neither split drops it.
Training windows shuffle each epoch with a dedicated seeded PyTorch generator.
Validation uses fixed order and a separate generator. Recreating the loaders
with the same settings reproduces their epoch-order sequence; exact mid-epoch
checkpoint resumption is not implemented. Defaults are batch size 8, sequence
length 256, seed 42, zero worker processes, and unpinned CPU memory. These are
starting settings, not measured optimal values. Windows multiprocessing callers
must create and iterate loaders inside an `if __name__ == '__main__':` guard;
the CLI already does this.

The future model must enforce causal attention. `attention_mask` only identifies
padding and is **not** a causal mask; convert it to the convention required by
the chosen attention API. Labels are already shifted, so do not shift them again
or pass them unchanged to an API that shifts labels internally. Given logits
`[B, T, V]`, compute loss with:

```python
import torch.nn.functional as F

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    batch['labels'].reshape(-1),
    ignore_index=-100,
)
```

For validation, sum token losses and divide by the total number of nonignored
targets across all batches; averaging batch means weights short batches and
heavily padded batches incorrectly.

The inspection command records resolved revisions, package versions, settings,
source/window fingerprints, sampled original row indices, window counts and
observed batch shapes in `reports/tinystories/dataloader.json`. Each cached
window also retains its original `source_row` within its own split. Blank or
whitespace-only text is skipped without stripping other text; null/non-string
text raises an error. An empty usable split fails explicitly. The official
train/validation memberships remain separate. **No exact or near-duplicate
cleaning is performed.** The EDA overlap finding still requires a full-corpus
leakage audit before interpreting training/validation results.

The loader checks that source Arrow cache paths match the requested dataset
commit before preprocessing. This prevents Hugging Face's offline fallback from
silently loading another cached revision while recording the requested one.
If this check fails, populate the requested revision online. The check relies on
the cache layout of the pinned `datasets` version and must be revisited when
upgrading that dependency.

Offline tests cover target alignment, chunk boundary coverage, EOS/padding loss
and gradients, split membership, partial batches, seeded order, sampled row
provenance, invalid data and worker processes. These checks do not establish
model causality, training stability, GPU performance or full-corpus cleanliness.

Local verification used Python 3.14.7 and PyTorch 2.14.1 on CPU: all 11 tests
passed. A real-data check selected 1,000 stories per split (seed 42), yielding
1,231 training windows and 1,193 validation windows at sequence length 256.
Batch size 8 gave 154 training batches and 150 validation batches. Offline runs
with zero and two workers agreed on inspected batches; a separate online check
loaded eight sampled stories per split successfully. The saved
[`dataloader.json`](reports/tinystories/dataloader.json) records the zero-worker
sample run. Full-corpus tokenisation and model training were not run.
