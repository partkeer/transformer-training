# Training measurements and controlled diagnostics

These commands share the reference training update: masked next-token loss,
AdamW, accumulation, clipping, AMP policy and schedule. They introduce no model
optimisation and choose no custom kernel. Offline tests use tiny synthetic data;
GPU performance and Nsight interoperability require verification on the GPU host.

## Routine training

```bash
# Starts training when explicitly executed on a configured host.
python scripts/train.py --output-dir runs/measured-baseline \
  --validation-loss-target 1.6 --validation-loss-target 1.5
python scripts/train.py --output-dir runs/budgeted \
  --supervised-token-budget 10000000
python scripts/report_measurements.py --run-dir runs/measured-baseline
```

An attempt covers every microbatch in an accumulation group and its attempted
optimiser update. Labels equal to `-100` are ignored; genuine EOS labels count.
Partial batches and partial final groups retain actual valid-token counts.

| Metric | Numerator | Denominator |
|---|---|---|
| Warmed useful tokens/s | Valid tokens in successful measured updates | Time for all measured attempts, including skipped attempts |
| Warmed processed tokens/s | Valid tokens in all measured attempts | Same attempt time |
| Whole-run useful tokens/s | Successful-update tokens in this invocation | Full training lifecycle |
| Whole-run processed tokens/s | All processed tokens in this invocation | Same lifecycle |

Warm-up defaults to ten attempts **per invocation**, including after resume.
Training-loop timing includes delivery, transfers, forward/loss, backward,
clipping and updates. It excludes logging, iterator creation, validation and
checkpoints; lifecycle time includes those costs. The lifecycle begins when
prepared loaders are available, before trainer construction, and ends after
final validation/checkpoint writing. Preprocessing and the earlier CUDA device
preflight are outside it; preparation time is recorded separately. This checkout
has no compilation feature; the lifecycle would include compilation if later
introduced in the update path.

Empty measured windows produce `null` plus a reason. All-skipped windows have
zero useful throughput and possibly positive processed throughput. FP16 overflow
consumes data/time without advancing successful updates or the schedule. Scaler
changes and numerical failures are recorded. Nonfinite losses and finite-gradient
norm reduction overflow retain their abort-before-update behaviour.

Repeat `--validation-loss-target` for finite, nonnegative targets. Record the
first **completed validation** at/below each threshold; do not interpolate or
stop training. Unreached targets have null observations and explicit statuses.
Validation events distinguish validation tokens, cumulative training tokens,
invocation training time and lifecycle time. Plots keep invocation curves separate.

The optional budget counts processed tokens, including skipped attempts, over
the checkpoint-restored trajectory. It is an operational limit, independent of
the existing epoch/`--max-steps` LR horizon. Stop follows a complete accumulation
group; requested/actual tokens and overshoot are recorded. Resuming an already
satisfied budget performs no additional update.

## Artifacts and recovery

Existing metric fields retain their processed-token meanings. Primary useful
metrics are in `summary.json` under `measurements` and in
`invocations/<id>/measurement.json`. Each invocation has an immutable
`environment.json`; the root environment describes the latest invocation.
Identity includes source revision/content hash/dirty status and available GPU
UUID, driver, capability, software and precision controls.

The initial invocation record is `running`; completion, orderly interruption or
caught failure writes the corresponding final status. Abrupt process death may
leave partial attempt logs and a running record: this is incomplete coverage,
not a zero-length completed run. Execution metrics include replayed work;
checkpoint counters represent the recovered training trajectory. Reports do not
merge replay branches into one trajectory. Downtime is excluded. Cross-resume
trajectory active time is a **checkpoint-boundary lower bound**: serialization
after its saved timestamp and unsaved work can be missing. Completed-invocation
lifecycle times remain independently available.

Version-1 checkpoints remain readable. Missing historical applied-token/time
metadata stays unknown; historical logs are not reconstructed. Invocation-local
measurements are still valid after legacy resume. Targets, token budget and
timing warm-up are operational overrides; existing model/data/optimisation
compatibility checks remain enforced.

## Repeated speed benchmarks

```bash
# No dataset access, CUDA initialization, updates or output directories.
python scripts/benchmark_training.py --dry-run
python scripts/profile_training.py --dry-run --backend all

# Five fresh processes: ten warm-up attempts, then fifty measured attempts each.
python scripts/benchmark_training.py --checkpoint runs/measured-baseline/latest.pt \
  --offline --output-dir runs/benchmark-baseline

# Cheaper pilot and explicitly different input mode.
python scripts/benchmark_training.py --checkpoint runs/measured-baseline/latest.pt \
  --repeats 3 --warmup-attempts 5 --measured-attempts 20 \
  --input-mode fixed --output-dir runs/benchmark-fixed
```

Without a checkpoint, accept the training model/training/data flags. With one,
inherit its model, optimiser, scaler, scheduler, RNG and data cursor; reject
conflicting workload/data overrides. Original checkpoints are never overwritten.
Use a fresh destination. Warm-up executes real updates, followed by measurement
without resetting the model. Every trial begins from the same state/data order.

`loader` includes real batch delivery and transfers. `fixed` repeats one CPU
accumulation group obtained before measurement; transfers/updates still count.
Keep those workloads separate. Insufficient remaining data or schedule horizon
fails rather than silently cycling or shortening the trial.

Throughput is aggregate valid tokens / elapsed measurement-window time, including
inter-attempt bookkeeping, not the mean of step throughputs. Save each trial/log,
latency percentiles, allocated/reserved peaks, useful/processed counts and skips.
Report median, mean, sample standard deviation and range; one trial has no sample
standard deviation. Profiles never enter benchmark statistics. Fresh processes
do not clear OS or persistent compilation caches.

`--compare <baseline-report.json>` checks fingerprints before giving a speedup.
Changed conditions suppress automatic speedup and list differing groups. Source
changes are also differences: inspect the two distributions after confirming
that only the intended implementation factor changed. Hold hardware/software,
model, data, precision, batching, schedule and token workload fixed. Timing
repetitions are not independent quality-training seeds.

## Separate detailed profiles

```bash
python scripts/profile_training.py --checkpoint runs/measured-baseline/latest.pt \
  --backend pytorch --output-dir runs/profile-timeline
python scripts/profile_training.py --checkpoint runs/measured-baseline/latest.pt \
  --backend memory --output-dir runs/profile-memory

# Four independent captures: PyTorch, memory, Systems, Compute.
python scripts/profile_training.py --checkpoint runs/measured-baseline/latest.pt \
  --backend all --output-dir runs/profile-all

# Compute alone needs a previous ranked kernel timeline.
python scripts/profile_training.py --checkpoint runs/measured-baseline/latest.pt \
  --backend ncu --timeline runs/profile-timeline/pytorch/trial.json \
  --output-dir runs/profile-counters
```

Defaults: ten unprofiled warm-up attempts, five captured attempts. Phase ranges
cover acquisition, CPU token preparation, transfers, forward/loss, backward,
zeroing, unscale/clipping, scaler handling and optimiser work. Main-process
preparation excludes preprocessing inside loader workers; exposed delivery wait
appears in acquisition. CPU host durations and correlated GPU activity durations
are separate. No additional phase-boundary synchronization is inserted.

Union kernel intervals across streams to obtain busy time. Separately report
copy/compute overlap, exposed copies and gaps inside complete-update windows.
Kernel sums can exceed elapsed time. GPU-phase attribution requires CUDA launch
correlations; missing attribution remains unknown. Gaps overlapping loader work
or synchronization get candidate annotations, not causal claims. CPU profiles
are supported; they do not substitute for requested CUDA measurements.

## Memory

Routine allocator peaks cover the lifecycle. Benchmark/profile peaks cover
measured attempts after warm-up resets. Allocated/reserved bytes are exact
allocator high-water counters. A persistent, CUDA-UUID-selected `nvidia-smi`
process requests device-wide memory every 200 ms. Its maximum is **sampled**,
may miss spikes and includes other processes. Timestamps/count/interval are saved.
Sampling begins at fit/trial execution, so preceding setup is not covered.
Missing UUID/tool/valid samples produces an unavailable status.

The memory pass exports `memory_timeline.json` and a simultaneous categorized
tensor peak: parameters, gradients, optimiser state, inferred activations,
temporaries, inputs, autograd details and unknown allocations. It does not sum
independent category maxima. Separate autograd hooks track saved nonparameter
storage lifetimes, deduplicate aliases/views and exclude parameter storage. That
saved-storage peak may include inputs; it must not be added to category totals.

CUDA allocator history/snapshots retain allocation stacks and reconstruct blocks
alive at the active-memory peak, including pending asynchronous frees. The ring
history is bounded at 200,000 events and may be incomplete. Active bytes,
categorized tensor bytes and `max_memory_allocated` are distinct scopes. Open
`allocator_snapshot.pickle` in [PyTorch memory_viz](https://pytorch.org/memory_viz).
Non-PyTorch CUDA allocations are outside its allocator snapshots. Reserved minus
allocated memory alone does not establish fragmentation.

Private APIs are isolated in the pinned-version memory adapter. The deprecated
`export_memory_timeline` API exports plain JSON to avoid Windows gzip temporary
file locking. Unsupported future APIs are reported explicitly and require an
adapter update; they do not produce invented replacement values.

## NVIDIA capture/import

Install compatible external `nsys` and `ncu` tools on the GPU host. They are
optional, and counter permissions may be restricted. The commands do not install
drivers, change permissions or elevate privileges. Systems captures CUDA/NVTX
after warm-up and exports SQLite. Compute queries supported metrics/sections,
then profiles up to three launches for each of the top five kernel names, with
a fresh process for each. Supported memory/roofline sections are included.
`--followup-counters` adds supported occupancy/stall diagnostics.
`--timeout-seconds` bounds each worker/capture; Windows jobs or POSIX groups
clean up descendants on timeout or completion.

Retain kernel duration, DRAM bytes, bandwidth (bytes / seconds), supported FP32
scalar instruction FLOP/s and supported dense tensor operations by precision.
FP32 FMA counts as two operations, add/multiply as one. Generic tensor instruction
counts are never converted using an invented operations multiplier. Unsupported
arithmetic paths remain unavailable; raw roofline exports retain further detail.

Triage is explicitly heuristic: measured traffic/time plus at least 80% sustained
DRAM and under 60% SM throughput suggests a bandwidth candidate; the reverse
suggests compute only with operation counts; both high suggests mixed. Otherwise
the conclusion is inconclusive. Inspect roofline and execution evidence before
selecting an optimisation. Utilization, occupancy and stalls alone are not proof.

```bash
# Imports execute no training or dataset loading.
python scripts/profile_training.py --import-kind chrome --import-path trace.json \
  --output-dir runs/import-timeline
python scripts/profile_training.py --import-kind nsys --import-path systems.sqlite \
  --output-dir runs/import-systems
python scripts/profile_training.py --import-kind ncu --import-path compute.csv \
  --output-dir runs/import-compute
```

Missing tools, `ERR_NVGPUCTRPERM`, unsupported schemas, malformed data and failed
captures have explicit statuses/reasons. Successful backends survive optional
NVIDIA failures. Counter replay timings never become benchmark measurements.

## Verification and bounded GPU acceptance

Run `python -m unittest discover -s tests -v`. Offline tests use artificial
stories/models, real fresh child processes, CPU profiler export and external-tool
fixtures. They perform no TinyStories training/download. CUDA checks explicitly
skip without a GPU.

Local verification on 2026-10-29 used Windows, Python 3.14.7 and PyTorch
2.14.1+cpu: 81 tests ran, 78 passed and three CUDA checks skipped. Dependency,
compilation and whitespace checks passed. Synthetic report generation and a
rendered validation-loss plot were checked. CUDA memory/timing, live Nsight
capture, hardware-counter permissions and full-training convergence were not
verified by this implementation work.

On the target host, before a full comparison:

1. Verify GPU UUID, pinned CUDA-enabled PyTorch, source/data identity and persistent
   storage. Hold the intended comparison conditions fixed.
2. Start with shorter explicit benchmark counts, inspect valid tokens, skipped
   updates and memory, then use the full five-trial protocol. Avoid changing a
   checkpoint's `--max-steps` merely to shorten diagnostic execution.
3. Capture PyTorch and memory separately; inspect traces, simultaneous categories
   and live allocator blocks/stacks.
4. Capture Systems and Compute independently; verify correlations, units, kernel
   selection and permissions. Unavailable measurements are not passed checks.
5. Run matched-token full quality experiments with identical validation/checkpoint
   cadence and targets. Compare loss versus time/tokens and observed time-to-target
   alongside throughput, memory and numerical failures.

Primary references: [PyTorch profiler](https://docs.pytorch.org/docs/2.14/profiler.html),
[CUDA memory](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html),
[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/), and
[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
