# Fused cross-entropy on existing logits

`--loss-backend triton` replaces cross-entropy using existing logits. The default
`--loss-backend pytorch` retains summed `F.cross_entropy` with `ignore_index=-100`.
The output projection and model architecture are unchanged.

## Installation and supported inputs

Use Python 3.12 (or 3.13), CUDA-enabled PyTorch 2.6.0, Triton 3.2.0, and Linux or
WSL2 on x86-64. NVIDIA compute capability >=7.5 is accepted, including the RTX
2070 (Turing/SM75). Turing requires the Triton 3.2 series and supports FP32/FP16;
BF16 logits require SM80 or newer. Native Windows is outside this backend's
support contract, including installations using `triton-windows`.

In a fresh Python 3.12 environment, install the CUDA wheel before the project:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[triton]'
```

This pins the matching Triton version; see the
[PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v260),
[Triton 3.2 platform support](https://github.com/triton-lang/triton/blob/v3.2.0/README.md#compatibility)
and [Turing version constraint](https://github.com/triton-lang/triton-windows#1-gpu).
Check the installation:

```bash
python -m pip check
python -m pip show torch triton
python -c "import torch, triton; print(torch.__version__, triton.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability())"
```

To inspect the installed CUDA wheel's declared Triton dependency:

```bash
python -c "import importlib.metadata as m; print('\n'.join(r for r in m.requires('torch') or [] if r.lower().startswith('triton')))"
```

`transformer_training.loss.cross_entropy_sum(logits, targets, backend="triton")`
supports contiguous finite CUDA `[N,V]` logits in FP32/FP16 (also BF16 on SM80+), `N >= 0`, and
`1 <= V <= 65,536`, including 50,257. Targets are contiguous int64 `[N]` on the
same device, each `-100` or in `[0,V)`. EOS remains supervised normally.

Metadata and target indices are validated before kernel accesses. The target
check synchronises CUDA through one boolean readback; public benchmarks include
this overhead. Finite logits are a precondition, without an additional full-input
scan. Losses beyond the FP32 representable range can overflow as in the reference;
training rejects nonfinite loss.

The shared `next_token_loss(..., backend="triton")` flattens contiguous leading
batch/sequence dimensions. Labels must match those dimensions and are already
shifted. Unsupported layouts are rejected rather than copied.

Triton returns an FP32 scalar sum; logits gradients use the input dtype. Compare
mixed precision against FP32 cross-entropy on identical quantised logits, matching
CUDA autocast. Native half-precision cross-entropy outside autocast has a different
reduction dtype. Only first-order autograd is supported; `create_graph=True`
raises explicitly. Whole-loss compilation, public-loss CUDA-graph capture,
distributed execution and nonfinite inputs are outside v1. The repository's
compiled model remains separate from the loss and has a CUDA acceptance test.

Unsupported requests and compilation/launch failures raise actionable errors.
Select `--loss-backend pytorch` explicitly for fallback. Dry runs resolve settings
without importing Triton, checking CUDA, accessing data or creating outputs.

## Computation

Forward uses one program per row and computes in FP32:

```text
m = max(logits[i, :])
s = log(sum(exp(logits[i, :] - m)))
row_loss = (m - logits[i, target[i]]) + s
```

Subtract before adding `s` to avoid cancellation for large common offsets such
as `1e20`. Save `m` and `s` separately for backward. Ignored rows write zero.
PyTorch sums the small FP32 row-loss buffer. Vocabulary padding to the next power
of two is masked with negative infinity, following the
[Triton fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
Initial forward warp counts are 4 through block size 2,048, 8 through 8,192,
16 through 32,768 and 32 above that. Vocabulary 50,257 uses block size 65,536.
These are explicit starting choices, not a measured optimum.

Backward uses 1,024-column tiles and four warps:

```text
gradient[i, j] = upstream * (exp((logits[i, j] - m[i]) - s[i]) - (j == target[i]))
```

Ignored gradients are exactly zero. `upstream` includes caller token normalisation
and GradScaler scaling; the kernel never normalises again. No log-softmax,
one-hot, stored softmax or full FP32 logits copy is created. Autograd retains
original inputs and two FP32 scalars per row; backward writes a fresh gradient.
Inputs/statistics are immutable, including retained graphs. Empty inputs return
differentiable zero. Entirely unsupervised accumulation groups and validation
loaders retain their existing error; individual ignored microbatches are allowed.

## Commands and compatibility

Run from the repository root in the activated Linux CUDA environment:

```bash
python scripts/train.py --loss-backend triton --precision bf16 --dry-run
python scripts/benchmark_loss.py --dry-run
python -m unittest discover -s tests -p test_loss.py -v
python scripts/train.py --loss-backend triton --precision bf16 --output-dir runs/triton-training
python scripts/evaluate.py --checkpoint runs/triton-training/best.pt --loss-backend triton --skip-generation --output-dir runs/triton-evaluation
# Explicit CPU/reference evaluation of a saved Triton run:
python scripts/evaluate.py --checkpoint runs/triton-training/best.pt --device cpu --precision fp32 --loss-backend pytorch --skip-generation --output-dir runs/reference-evaluation
# Loss-only measurements, then complete optimiser updates:
python scripts/benchmark_loss.py --output-dir runs/loss-benchmark
python scripts/run_experiments.py --plan experiments/fused-cross-entropy.json --output-dir runs/loss-updates
```

For an RTX 2070, run the acceptance tests above, then use FP16 or FP32 instead of
BF16. The recorded RTX 2070 SUPER run has unresolved FP16 optimiser-update
acceptance failures; see [measured results](#rtx-2070-super-measured-results).
A smaller microbatch is a starting point for 8 GB VRAM, not a verified fit:

```bash
# Effective batch remains 1 * 64 = 64 examples.
python scripts/train.py --loss-backend triton --precision fp16 --batch-size 1 --gradient-accumulation-steps 64 --output-dir runs/2070-triton
python scripts/benchmark_loss.py --rows 512 257 --precisions fp32 fp16 --output-dir runs/2070-loss
```

The supplied update plan includes BF16 variants. For Turing, use a copy with those
variants removed; keep matched PyTorch/Triton pairs and identical batch settings.
The GPU tests omit BF16 cases on devices without native BF16 support, but still
run FP32/FP16 shape, gradient, extreme-value and model-integration checks.

Use fresh output directories. Reports are `report.json` and `report.md`. The loss
benchmark defaults to five fresh-process repeats, ten warm-ups, fifty measurements,
`N=4096` and `N=257`, `V=50,257`, FP32/FP16/BF16, and ignored fractions 0 and 0.5.
Here `4096 = 8 sequences * 512 positions`, not sequence length. Unavailable or
incomplete microbenchmarks exit with code 2 and retain reasons.

The update plan changes only `implementation.loss_backend` within each precision
pair and uses fixed accumulation groups with the existing `optimizer_attempt`.
For paired quality studies, add the existing explicit quality protocol to a copy
of the plan; the quality worker uses common FP32 eager PyTorch validation.
Speed alone does not establish convergence.

Backend selection is checkpointed. Older checkpoints default to PyTorch. Normal
resume inherits and enforces the saved backend; diagnostics and standalone
evaluation allow explicit overrides. Reports record the installed Triton version
without importing it and label launch settings as configuration, not observed
dispatch. See [measurement scopes](measurements.md#cross-entropy-microbenchmarks).

## Correctness and verification limits

Loss tolerance is `rtol=1e-5`, `atol=1e-5 * max(supervised_rows, 1)`. Gradient
`(rtol,atol)` is `(1e-5,1e-6)` for FP32, `(2e-3,2e-5)` for FP16 and `(2e-2,2e-4)`
for BF16; absolute tolerance scales with upstream magnitude. Ignored gradients
must be exactly zero. Tests cover irregular/boundary shapes, partial batches,
EOS, ignored inputs, extreme finite logits, large offsets and upstream gradients
0, 1, -2.5 and 128. Integration tests cover unequal-token accumulation, actual
GradScaler updates, shared validation and compiled models. Missing prerequisites
skip GPU acceptance; compilation or numerical failures on supported GPUs fail.

Initial implementation checks used Windows with PyTorch `2.14.1+cpu`, no
CUDA devices and no Triton. Compatibility checks used WSL2, Python 3.12.3,
PyTorch `2.6.0+cpu` and Triton 3.2.0. Offline compilation assembled 40 production
kernel variants into SM75 cubins: forward/backward in FP32/FP16 at vocabulary
sizes 1, 33, 2,048, 2,049, 8,192, 8,193, 32,768, 32,769, 50,257 and 65,536.
This checks compiler compatibility without loading or executing kernels on a GPU.

## RTX 2070 SUPER measured results

The supplied GPU artifacts are in `runs/fused-cross-entropy-remote/`: the
[loss report](../runs/fused-cross-entropy-remote/2070-loss/report.md), detailed
[JSON results](../runs/fused-cross-entropy-remote/2070-loss/report.json), and
[verification logs](../runs/fused-cross-entropy-remote/verification-logs/).
They establish GPU execution and loss-operation speed/memory improvements, with
unresolved FP16 acceptance failures. The recorded environment is Linux x86-64,
Python 3.12.16, PyTorch `2.6.0+cu118`, Triton `3.2.0`, and an RTX 2070 SUPER
with 8 GB VRAM and compute capability 7.5. Git source identity was unavailable
in the report, so the artifacts do not identify the exact tested commit.

### Loss-operation speed and memory

The measured protocol uses `V=50,257`, `N=512` and `N=257`, FP32/FP16, ignored
fractions 0 and 0.5, seed 42, and upstream gradient 1. Each configuration has
five fresh-process repeats, ten warm-ups and fifty measurements per operation.
Rows are next-token prediction positions, not sequence length: for example,
`512 = 1 sequence * 512 positions`. Timings synchronise CUDA and include public
loss validation; setup, resets and compilation are excluded. Memory is measured
in separate warmed passes. Reported times are medians of repeat medians.

| Rows | Precision | Ignored targets | PyTorch F+B ms | Triton F+B ms | Speedup | Peak allocated MiB, PyTorch / Triton |
|---:|---|---:|---:|---:|---:|---:|
| 512 | FP32 | 0% | 2.3922 | 1.1892 | 2.012x | 392.64 / 196.33 |
| 512 | FP32 | 50% | 2.3918 | 0.9715 | 2.462x | 392.64 / 196.33 |
| 512 | FP16 | 0% | 2.0559 | 0.8108 | 2.536x | 296.32 / 100.01 |
| 512 | FP16 | 50% | 2.0558 | 0.7619 | 2.698x | 296.32 / 100.01 |
| 257 | FP32 | 0% | 1.2214 | 0.7898 | 1.547x | 200.00 / 100.01 |
| 257 | FP32 | 50% | 1.2205 | 0.7427 | 1.643x | 200.00 / 100.01 |

Completed comparisons show 1.55--2.70x faster loss forward/backward. At 512 rows,
peak allocated memory falls by 196.31 MiB: approximately 50% in FP32 and 66.2%
in FP16. These peaks include input tensors and describe the isolated loss
workload, not whole-model or device-wide memory. Avoiding vocabulary-sized
intermediate tensors is consistent with the savings. Ignored rows reduce Triton
time because their vocabulary calculations are skipped, although a full gradient
tensor is still allocated and written.

At 257 FP16 rows, Triton completes at 0.5805 ms (0% ignored) and 0.5568 ms
(50% ignored), with 49.28 MiB peak allocation. The matching PyTorch cases fail
the correctness gate before timing, so no speedup is available for those pairs.

### Correctness and remaining acceptance failures

All 40 Triton benchmark case/repeat checks pass against FP32 PyTorch on identical
quantised logits. Maximum recorded summed-loss error is 0.000977 in both
precisions; maximum logits-gradient error is `9.31e-10` in FP32 and `2.38e-7`
in FP16. Repeats use the same seed and inputs. The separate GPU tests also pass
shape/boundary, extreme-value, ignored-target, retained-graph and upstream-scale
checks. BF16 execution is not tested on this GPU.

The benchmark is nevertheless **incomplete**: all ten PyTorch FP16 repeats at
257 rows fail the loss tolerance against the FP32 reference. Absolute summed-loss
differences are 0.03589 without ignored targets and 0.04236 with 50% ignored;
relative differences are approximately `1.25e-5` and `2.93e-5`. This is a
reference/tolerance discrepancy requiring investigation, not a Triton failure
or an accepted timing comparison.

The [acceptance log](../runs/fused-cross-entropy-remote/verification-logs/test-loss.txt)
records 24 tests with two failing FP16 integration subcases, eager and compiled.
Both fail when comparing model parameters after optimiser updates: the reported
48-element tensor has 1 mismatch with maximum absolute error 0.000203 in eager
mode, and 2 mismatches with maximum error 0.000245 in compiled mode. The absolute
tolerance is 0.0002 (`rtol=0.01`). FP32 integration has no reported failures.
These small absolute discrepancies remain unresolved acceptance failures;
isolated loss/gradient agreement does not establish equivalent FP16 updates.
The cause is not established by these artifacts.

### Training smoke and interpretation limits

The [training smoke log](../runs/fused-cross-entropy-remote/verification-logs/training-smoke.txt)
records five successful Triton updates, no skipped updates or reported numerical
failures, completed validation and checkpoint writes, and about 366 MiB peak
allocated memory for that run's configuration. Training batch loss moves from
10.849 to 10.391 across different batches; validation loss is 10.404 over 57,374
supervised tokens. This is an operational smoke check, not convergence evidence.

The smoke run has no matched PyTorch baseline and no post-warm-up measured
attempts: all five updates fall within its ten-update warm-up. Complete-update
speedup, whole-training memory savings, sustained stability and paired validation
quality remain unmeasured. Resolve the FP16 integration failures and reference
discrepancy, then use matched complete-update benchmarks and a longer paired
quality comparison before claiming training-level gains. Earlier profiles
motivate this target; the measurements above establish loss-only gains.
