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

TinyStories exploratory analysis, next-token PyTorch dataloaders, a decoder-only
transformer, and a single-GPU training command are implemented. Training includes
full-split exact-overlap exclusion, validation, mixed-precision options, metrics,
and optimizer-boundary checkpoint recovery. A standalone checkpoint-quality
command measures validation loss/perplexity and generates story continuations.
Offline correctness tests and a configuration-only dry run are available.
**TinyStories training has not been
started**, and no GPU memory, throughput, convergence or generation-quality
results are claimed. Profiling experiments and the custom kernel remain planned.

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
the training baseline uses a configurable 512-token context and keeps per-story
windows. Context-length and packing experiments remain future work.

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
IDs: the model uses `loaders.metadata['vocab_size']` as its vocabulary size.

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
with the same settings reproduces their epoch-order sequence. The training runner
replaces train shuffling with its own resumable epoch sampler. Loader defaults
are batch size 8, sequence
length 256, seed 42, zero worker processes, and unpinned CPU memory. These are
starting settings, not measured optimal values. Windows multiprocessing callers
must create and iterate loaders inside an `if __name__ == '__main__':` guard;
the CLI already does this.

The model must enforce causal attention. `attention_mask` only identifies
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
train/validation memberships remain separate. **The loader's default performs no
exact or near-duplicate cleaning.** The training command explicitly opts into the
full-corpus exact-overlap audit and training-only exclusion described below.

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

## Train the reference decoder

[`scripts/train.py`](scripts/train.py) invokes reusable configuration, model,
training, validation and checkpoint modules in `src/transformer_training/`.
Install the package with `python -m pip install -e .` before invoking scripts.

Inspect the configuration safely on a CPU machine:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --dry-run
.\.venv\Scripts\python.exe scripts\train.py --help
```

`--dry-run` builds the model on CPU and prints settings and parameter count. It
does not import the dataset loader, download data, create a run directory, check
GPU availability, or perform forward/backward passes or optimizer steps. It
prints the intended CUDA training configuration even on a CPU-only machine.

### Architecture and defaults

The model has **33,691,776 parameters**: 8 pre-LayerNorm blocks, width 384,
6 attention heads, GELU feed-forward width 1,536, learned positions for 512 tokens,
and a final LayerNorm. GPT-2's 50,257-token input embedding is tied to the output
projection. Dropout is 0.1. Linear/embedding weights use normal initialisation
with standard deviation 0.02; residual output projections use
`0.02 / sqrt(2 * layers)`. Biases start at zero and LayerNorm scales at one.

Attention is an explicit PyTorch reference implementation combining causal and
padding masks, with FP32 softmax. Inputs must have nonempty right-padded prefixes.
Outputs are `[B, T, V]` logits; padded query logits are ignored. Labels from the
loader are already shifted and are not shifted a second time. This small-model
starting point is consistent with the motivation of the
[TinyStories paper](https://arxiv.org/abs/2305.07759), but is not a measured optimum.

| Training setting | Default |
|---|---|
| Duration | One cleaned training epoch |
| Microbatch / accumulation | 8 windows / 8 microbatches |
| Nominal effective batch | 64 windows, at most 32,768 padded token positions |
| Precision | FP32; TF32 disabled |
| Optimizer | Unfused AdamW, betas `(0.9, 0.95)`, epsilon `1e-8` |
| Weight decay | 0.1 on matrices, including tied embeddings once; none on biases/LayerNorm |
| Learning rate | `3e-4` peak, `3e-5` floor |
| Schedule | 2% linear warm-up, then cosine decay |
| Gradient clipping | Global norm 1.0 |
| Seed / workers | 42 / 2 |
| Logging / validation / checkpoint | Every 10 / 2,000 / 500 successful updates |

`--max-steps` caps successful optimizer updates within the selected epochs. The
schedule horizon is the smaller of that cap and the number of accumulation
groups in the configured epochs. A one-update run uses peak LR; a run with two
or more updates ends at the minimum LR. Warm-up is rounded down with at least one
warm-up update when enabled and always leaves a final decay update.

The loss is `sum(nonignored token losses) / count(nonignored targets)` across the
whole accumulation group. Only CPU batches are buffered; GPU microbatch graphs
are processed and released sequentially. Partial final batches and incomplete
groups use their actual token counts. Validation applies the same token weighting
over the entire selected validation split, every 2,000 updates and at completion.
One epoch is an initial baseline, not a convergence claim.

### Exact-overlap policy

Training sets `exclude_validation_overlap=True` on the loader. Before any
`--max-stories` sampling, it audits **both full official splits** using SHA-256 of
unchanged UTF-8 text. It records blank counts, within-split duplicate counts,
shared distinct texts and removed training rows. Blank text is counted separately;
duplicate/overlap counts refer to nonblank stories. Invalid non-string text fails.

All training rows exactly matching any validation story are excluded. Validation
membership is preserved, and other training duplicates remain. No case, whitespace
or Unicode normalisation or near-duplicate detection is performed. The existing
blank filtering and independent per-story EOS/window policy still apply. Sampling
occurs among eligible rows using the existing split-specific seeds, and original
row IDs survive cleaning and sampling. A sample still requires a full first audit.

Audit results are cached under `$HF_HOME/training_audits/`, keyed by source Arrow
fingerprints, source sizes and cleaning-policy version. The reusable
`create_dataloaders(..., exclude_validation_overlap=True, audit_cache_dir=...)`
can also use this cache; without a cache directory it audits in memory. Existing
loader callers retain the previous uncleaned default.

### Linux cloud setup and launch commands

Target one NVIDIA GPU with approximately 16–24 GB VRAM. The current configuration
has not been memory-tested on that hardware. Use Python 3.14 and install a
CUDA-enabled **PyTorch 2.14.1** build compatible with the host driver, Python ABI
and GPU. Use the official [PyTorch installation guidance](https://pytorch.org/get-started/locally/)
to select a compatible CUDA wheel index; do not assume a provider's default image
or the Windows CPU build satisfies these requirements.

The following are setup examples, **not commands already executed on a cloud host**:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Set this to the actual official CUDA wheel index chosen for your host/release.
# Example form: https://download.pytorch.org/whl/cu<runtime-version>
export PYTORCH_CUDA_INDEX_URL='<compatible-official-CUDA-wheel-index>'
python -m pip install 'torch==2.14.1' --index-url "$PYTORCH_CUDA_INDEX_URL"
python -m pip install -e .
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"

# Replace these example paths with directories on your mounted persistent disk.
export HF_HOME=/persistent/huggingface
python scripts/train.py --dry-run

# This command STARTS training when you choose to run it.
python scripts/train.py --output-dir /persistent/runs/tinystories-fp32

# Resume settings, revisions and sample selection are inherited from the checkpoint.
python scripts/train.py --resume /persistent/runs/tinystories-fp32/latest.pt
```

The package retains its pinned direct dependencies. If no compatible CUDA wheel
exists for the chosen runtime, resolve that environment issue before launching;
do not silently replace the pinned baseline version. The trainer fails before
downloading data when CUDA is unavailable. CPU execution requires `--device cpu`
and supports FP32 only. Allow substantial disk space for source data, Arrow token
windows, audit caches and optimizer checkpoints, and enough CPU RAM for the
full-split digest counts and window permutation.

Useful variants for later cloud experiments:

```bash
# Halve microbatch memory while retaining 64 windows per nominal effective batch.
python scripts/train.py --batch-size 4 --gradient-accumulation-steps 16 --output-dir runs/fp32-b4

# Explicit mixed precision; each is a separate experiment from the FP32 baseline.
python scripts/train.py --precision bf16 --output-dir runs/bf16
python scripts/train.py --precision fp16 --output-dir runs/fp16

# Bounded experiment, with full-split cleaning before seeded sample selection.
python scripts/train.py --max-stories 1000 --max-steps 20 --output-dir runs/sample
```

BF16 requires native GPU support; FP16 uses gradient scaling. Nonfinite losses
abort. FP16 gradient-overflow updates are skipped and logged, and do not advance
the learning-rate schedule or successful-update counter. Consumed data still
advances, so an epoch with skipped updates can finish below the schedule horizon.
If gradient elements are finite but their norm overflows, training aborts before
clipping or updating parameters instead of counting a zeroed-gradient update.
Accumulation, scaling and clipping follow the
[PyTorch AMP guidance](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html).
Reducing microbatch size changes padding composition and floating-point reduction
order, even when the nominal effective batch remains 64 windows. Supervised-token
counts vary with story lengths; the script records them explicitly.

### Artifacts, recovery and measurements

Default output is `runs/<UTC-id>/`, ignored by Git. A new run refuses a nonempty
directory. Invalid output paths and incompatible resume model/training settings,
checkpoint versions, sample limits and immutable revisions are rejected before
data preparation. Symbolic revisions and dataset fingerprints are checked after
resolution/preprocessing. Outputs include `config.json`, `data.json`, `audit.json`,
`environment.json`, `metrics.jsonl`, `summary.json`, `latest.pt` and `best.pt`.
The environment record includes package versions, CUDA/cuDNN build, GPU identity,
VRAM, compute capability and driver version where available. Config/data records
retain the original run settings; resume events record operational overrides and
the environment record describes the latest invocation.

Checkpoints atomically replace their destination and contain model/optimizer,
schedule/scaler, Python/PyTorch/CUDA RNG states, successful/skipped update counts,
epoch and consumed example cursor. A deterministic epoch permutation restarts at
that cursor; worker prefetch never advances it. Checkpoints are taken after whole
accumulation groups, so abrupt interruptions can lose work since the latest saved
boundary. `best.pt` is saved on improved validation loss; `latest.pt` is saved
every 500 successful updates and at completion.

Resume inherits the checkpoint configuration and data selection. Only worker
count and logging/evaluation/checkpoint intervals may change explicitly; model,
optimization, precision, device, duration and data identity must match. Changed
fingerprints or preprocessing policy are rejected. Training options require their
full names (for example, `--batch-size`); abbreviated flags are rejected so resume
overrides cannot be silently ignored. Recovery is checked exactly on
CPU synthetic tests, including dropout and switching between zero and two workers.
Bitwise equality across hardware, library versions or CUDA kernels is not promised.
Persist both cache and run directories before terminating a rented instance.

Logs report token-weighted loss, the LR used, supervised tokens, padding
utilisation, synchronized mean update-attempt time, tokens/second, skipped updates,
and peak allocated/reserved CUDA bytes (null on CPU). Skipped updates are logged
immediately. Summary timing excludes the first 10 update attempts; small runs may
have no steady-state summary. Timing includes data delivery, transfer and update
work, while excluding preprocessing, epoch iterator setup, evaluation and
checkpoint writing. Evaluation and checkpoint durations are recorded separately.
These are training-loop measurements, not isolated GPU kernel benchmarks; future
performance comparisons still need repeated controlled measurements.

### Verification limits

Run offline tests with `python -m unittest discover -s tests -v`. They use
in-memory artificial stories and tiny synthetic models; optimizer steps occur
only inside correctness tests. Coverage includes causal/padding isolation, target
alignment/EOS gradients, token-weighted validation and accumulation, partial
groups, exact-overlap filtering/provenance, schedule boundaries, atomic checkpoint
failure handling, and interrupted-versus-uninterrupted continuation with dropout
and worker prefetch. A subprocess test proves dry-run does not import datasets or
create a run directory. CUDA precision tests skip explicitly when no GPU exists.

Original training implementation verification used Python 3.14.7 and PyTorch 2.14.1+cpu:
32 tests ran, with 31 passing and one CUDA test explicitly skipped. The default
dry run reported 33,691,776 parameters and a 64-window effective batch. Dependency
validation (`pip check`) and Git whitespace checks passed.

Implementation verification has not downloaded/tokenised the full cleaned corpus,
started TinyStories training, measured cloud GPU performance/memory, established
convergence or checked generated-story quality. SDPA/fused attention, compilation,
packing, distributed training, activation checkpointing and custom kernels remain
future profiling-led experiments.

## Evaluate a trained checkpoint

After training, use `scripts/evaluate.py` with either `best.pt` (selected by
validation loss) or `latest.pt` (most recent saved training boundary):

```powershell
# Match the saved validation selection and generate three default story samples.
.\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint runs\my-run\best.pt

# Evaluate a GPU-trained checkpoint on a CPU-only machine.
# Device and precision overrides must both be explicit when needed.
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --checkpoint runs\my-run\best.pt --device cpu --precision fp32 --batch-size 1

# Evaluate every usable official validation story, with a custom opening.
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --checkpoint runs\my-run\best.pt --full-validation `
  --prompt "Lily found a lost kitten."

# Cached-only quantitative evaluation; no story generation.
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --checkpoint runs\my-run\latest.pt --offline --skip-generation

# Replace default openings with two custom prompts and use greedy decoding.
.\.venv\Scripts\python.exe scripts\evaluate.py `
  --checkpoint runs\my-run\best.pt --prompt "Once upon a time" `
  --prompt "The puppy was lost." --temperature 0 --max-new-tokens 100
```

The command restores the checkpoint's architecture and weights without creating
an optimizer, restoring training RNG state, or modifying training artifacts.
Device, precision and microbatch size default to saved values; worker count
defaults to zero. Override these with `--device`, `--precision`, `--batch-size`
and `--num-workers`. Unsupported CUDA/precision requests fail before data
downloads, with no automatic CPU or precision fallback.

### Validation selection and metrics

Evaluation loads **only the official validation split** and the checkpoint's
pinned tokenizer. It does not load, audit, or tokenize the training split.
Prepared validation data is reused directly. On a fresh cache, only validation
files are prepared under the datasets cache's `validation-only/` directory;
the normal training cache remains complete and separate.
The checkpoint retains the overlap-exclusion policy applied during training.
Dataset/tokenizer commits, context length, vocabulary, EOS and padding rules are
fixed by the checkpoint and checked against loaded metadata.

By default the command reproduces the saved validation selection, including
`--max-stories` sampling from a small training run. It uses the saved validation
selection seed and rejects changed provenance or fingerprints. `--seed` controls
generation only. `--full-validation` removes the story limit and clearly labels
the resulting report; its score may differ because it evaluates more data.
Sampling limits source rows before blank stories are skipped, and a long story
may produce several windows. The report records selected source rows, windows,
and supervised tokens separately.

The objective is identical to training validation: append EOS to each story,
preserve per-story windows, and compare logits with the loader's already-shifted
next-token targets. Padding targets are excluded; genuine EOS targets count.
Partial final batches remain included. Loss is aggregated by valid-token count:

\[
L = \frac{\sum_b \text{summed cross-entropy}_b}
          {\sum_b \text{supervised tokens}_b},
\qquad \mathrm{perplexity} = \exp(L).
\]

Lower loss/perplexity means better next-token prediction under the same data,
tokenizer and windowing policy. For example, loss 2 corresponds to perplexity
about 7.39. Perplexity is not a percentage or an automated story-quality score.
Compare scores on matching validation selections and record precision/batch-size
changes, which can affect floating-point reductions. If exponentiation overflows,
the report keeps the finite loss and writes `perplexity: null` with
`perplexity_overflow: true`. Empty supervision and nonfinite loss fail explicitly.

The validation split used to select `best.pt` is a **model-selection set**, not an
independent final test set. These measurements alone do not establish performance
on a new domain or general language-model benchmarks.

### Generated examples and artifacts

The default prompts are "Once upon a time, a little girl", "Tom found a small box
in the garden." and "The puppy was afraid of the rain." Repeated `--prompt` flags
replace that list; `--skip-generation` disables generation entirely. Defaults are
128 new tokens, temperature 0.8, top-k 50 and seed 42. Temperature zero selects
greedy decoding; top-k zero enables full-vocabulary sampling. Positive top-k is
capped at the vocabulary size. Each prompt uses an isolated generator with
`seed + prompt_index` (zero-based); identical settings are reproducible within
the same hardware/software/precision environment, without consuming global RNG
state during generation.

Prompts are never truncated. Blank prompts and prompts filling or exceeding the
learned context are rejected. Generation stops at EOS, the new-token limit, or
context capacity, without rolling the context or resetting learned positions.
The stop reason and token count are recorded. A generated EOS counts as a token
but is omitted from decoded text; immediate EOS can therefore produce an empty
continuation. Generated examples are for human inspection of coherence,
repetition and adherence to the opening, without a numerical quality grade.

Outputs go to `<checkpoint-directory>/evaluations/<UTC-id>/` by default; use
`--output-dir <path>` for another empty directory. Nonempty destinations are
rejected. Metrics and stories print to the terminal, and two UTF-8 artifacts are
written atomically:

- `report.json`: checkpoint path/update count, architecture, effective evaluation
  settings, validation provenance, hardware/software environment, loss/perplexity,
  prompts, decoding settings, seeds and generated samples.
- `samples.txt`: readable prompts, continuation-only and complete decoded text,
  generated token counts and stopping reasons (or a skipped-generation note).

The reusable interfaces are `quality.evaluate_checkpoint()`,
`generation.generate()`, and `data.load_tinystories_validation()` in
`transformer_training`. For direct API use, configure `HF_HOME` and offline
environment variables before the first Hugging Face import. The script sets
the default cache to the repository's `.cache/huggingface`; `--offline` requires
the pinned validation data and tokenizer to have been cached already.

Evaluation correctness tests use artificial stories, a small local tokenizer,
and synthetic checkpoints, including a subprocess CLI check with local Hub
stubs. They verify loss/perplexity, validation-selection equivalence, strict
checkpoint loading, generation and report behavior. A real Parquet-builder test
uses an unreadable training file to verify that fresh evaluation preparation
does not read training data. They do not establish
TinyStories model quality or cloud GPU performance. Inference speed and memory
benchmarking remain separate work.

Local verification after adding evaluation ran 48 tests: 46 passed and two CUDA
tests were explicitly skipped on CPU-only PyTorch 2.14.1. Full and sampled
synthetic data signatures matched a pre-refactor snapshot exactly. An offline
cache smoke check opened the official 21,990-row validation split and mapped
only its validation Arrow file; it did not evaluate a trained model. Git and
new-file whitespace checks passed.
