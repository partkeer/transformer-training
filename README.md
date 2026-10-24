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

TinyStories loading and exploratory analysis are implemented. The training
system, profiling experiments, and custom GPU kernel remain planned work; no
training or performance results are claimed yet.

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
pinned in `requirements.txt`. A later tokenizer analysis should measure actual
token lengths before choosing context length, padding, packing, or truncation.

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
