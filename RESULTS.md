# RunPod results

Summary of the downloaded artifacts in `runs/runpod-download`

## Main findings

- **Highest warmed throughput:** `compile-no-cudagraphs` reached **60,219 useful
  tokens/s**, 41.1% above the FP32 reference. `compile-default` was close at
  **59,772 tokens/s**; the 0.7% gap is too small to establish a clear winner from
  these five repeats. First forward/backward attempts took about **870 s** and
  **233 s**, respectively, versus **67 s** for FP32, so startup cost matters.
- **Lowest peak allocated memory:** `combined-example` used **2,560 MiB**, 36.2%
  below FP32, while delivering **57,806 tokens/s** (+35.5%). It combines BF16,
  Flash attention, compilation, fused AdamW and activation checkpointing.
- **Attention helped:** FP32 `sdpa-auto` selected the efficient backend and
  improved throughput by **23.7%**, with **11.1%** less allocated memory.
- **Reduced precision alone did not improve speed:** BF16/FP16 saved about
  **24%** memory but reduced throughput by about **11.5%**. Flash/efficient
  attention improved throughput by **22.6%/24.4%** over the BF16 reference.
- **Other trade-offs:** foreach/fused AdamW improved throughput by **3.4%/4.7%**.
  Activation checkpointing alone saved **16.9%** memory but reduced throughput
  by **25.7%**.

## Completed benchmarks

Values are the suite's reported medians across five separate-process trials.
Memory is PyTorch peak **allocated** GPU memory, not total device usage.

| Variant | Useful tokens/s ↑ | Update time (ms) ↓ | Peak memory (MiB) ↓ |
|---|---:|---:|---:|
| `fp32` (reference) | 42,674 | 319.7 | 4,013 |
| `bf16` | 37,772 | 361.2 | 3,050 |
| `fp16` | 37,745 | 363.5 | 3,049 |
| `sdpa-math` | 42,463 | 323.6 | 3,998 |
| `sdpa-auto` | 52,798 | 260.9 | 3,567 |
| `bf16-flash` | 46,325 | 298.7 | 2,695 |
| `bf16-efficient` | 46,995 | 291.6 | 2,695 |
| `compile-default` | 59,772 | 230.1 | 3,999 |
| `compile-no-cudagraphs` | 60,219 | 226.7 | 3,999 |
| `foreach` | 44,106 | 310.6 | 4,013 |
| `fused` | 44,659 | 306.5 | 4,013 |
| `checkpointing` | 31,700 | 433.6 | 3,336 |
| `combined-example` | 57,806 | 237.3 | 2,560 |

`compile-no-cudagraphs` uses `max-autotune-no-cudagraphs`. Unless stated
otherwise, percentage comparisons above use `fp32` as the reference.

## Measurement conditions and limits

- **Hardware/software:** NVIDIA RTX 4090; Python 3.14.10; PyTorch 2.14.1+cu126
  (CUDA 12.6), as recorded in the downloaded environment.
- **Workload:** TinyStories with GPT-2 tokenisation; 9,979,040 parameters;
  6 layers, width 160, 4 heads, feed-forward width 640; sequence length 512;
  batch size 8 × 8 accumulation steps (effective batch 64); seed 42.
- **Timing:** each trial used 10 warm-up and 50 measured update attempts.
  Useful tokens/s counts non-padding tokens in successfully applied updates.
  Timings include loader delivery and transfers, but exclude startup, validation
  and checkpointing; these are training-update benchmarks, not end-to-end rates.
- **Stability/quality:** all completed variants recorded zero skipped updates
  and zero numerical failures. Quality runs were disabled: **validation loss,
  convergence and numerical equivalence are not established by these results**.
  Small differences should be interpreted alongside trial variation.

## Profiling and incomplete results

- PyTorch profiles are available for **10 of 16 variants**, plus the separate
  reference timeline. FP32 log-softmax forward/backward consumed about **214 ms
  aggregate GPU time** across five captured updates, making the vocabulary/loss
  path a visible optimisation target. The reference trace also contains large
  gaps without recorded GPU activity; it does not establish their exact cause.
  Profiling perturbs execution and is excluded from the benchmark ranking.
- **Benchmark failures:** `bf16-cudnn` reported no available kernel, with a
  cuDNN dropout-compatibility warning. `compile-reduce-overhead` and
  `compile-max-autotune` failed with CUDA-graph gradient-overwrite errors during
  accumulation. No performance values are available for these variants.
- **Other missing captures:** BF16, FP16 and checkpointing profiles timed out;
  the other three profile failures correspond to the failed benchmark variants.
  The separate reference memory capture also timed out. Nsight Compute failed
  on an invalid metrics-query option; Nsight Systems export failed because its
  `.nsys-rep` was missing (a raw `.qdstrm` remains).

Sources: [full benchmark report](runs/runpod-download/results/implementations/report.md),
[variant profiles](runs/runpod-download/results/profiles-v1/),
[reference timeline](runs/runpod-download/results/reference-timeline/report.md).
The separate [FP32 pilot](runs/runpod-download/results/pilot-fp32/report.md)
recorded 41,878 tokens/s in one trial; it is not pooled into the comparison above.
