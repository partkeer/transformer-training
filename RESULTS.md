# Experiment results

Summary of results from GPU experiments.
- **Throughput** - supervised (non-padding) tokens per second during complete optimiser updates.
- **Memory** - peak PyTorch allocated GPU memory (reported per GPU).

Benchmark tables show medians across five fresh-process repeats, each with 10 warm-ups and 50 measurements.

## RTX 4090: training optimisations

- **GPU / software:** one RTX 4090; PyTorch 2.14.1+cu126.
- **Independent variables:** precision, attention backend, compilation,
  optimiser implementation and activation checkpointing.
- **Dependent variables:** throughput, update time and peak memory.
- **Controls:** TinyStories dataset, GPT-2 tokenisation, 9.98M-parameter model
  (6 layers, width 160, 4 heads, feed-forward width 640), sequence length 512,
  microbatch 8, accumulation 8, effective batch 64 and seed 42.

| Configuration | Tokens/s | Update (ms) | Memory (MiB) | Speedup | Relative Memory |
|---|---:|---:|---:|---:|---:|
| FP32 reference | 42,674 | 319.7 | 4,013 | 1.000x | 100.0% |
| BF16 | 37,772 | 361.2 | 3,050 | 0.885x | 76.0% |
| FP16 | 37,745 | 363.5 | 3,049 | 0.885x | 76.0% |
| SDPA: math | 42,463 | 323.6 | 3,998 | 0.995x | 99.6% |
| SDPA: auto (efficient) | 52,798 | 260.9 | 3,567 | 1.237x | 88.9% |
| BF16 + Flash attention | 46,325 | 298.7 | 2,695 | 1.086x | 67.2% |
| BF16 + efficient attention | 46,995 | 291.6 | 2,695 | 1.101x | 67.2% |
| Compile: default | 59,772 | 230.1 | 3,999 | 1.401x | 99.6% |
| Compile: max-autotune-no-cudagraphs | 60,219 | 226.7 | 3,999 | **1.411x** | 99.6% |
| Foreach AdamW | 44,106 | 310.6 | 4,013 | 1.034x | 100.0% |
| Fused AdamW | 44,659 | 306.5 | 4,013 | 1.047x | 100.0% |
| Activation checkpointing | 31,700 | 433.6 | 3,336 | 0.743x | 83.1% |
| Combined optimisations | 57,806 | 237.3 | 2,560 | 1.355x | **63.8%** |

Speedup is throughput divided by FP32 throughput. Relative Memory is peak
memory as a percentage of FP32 memory.

*Combined optimisations* uses BF16, Flash attention, compilation,
fused AdamW and checkpointing

> **Compile: max-autotune-no-cudagraphs** leads to the highest measured throughput.
>
> The **combined optimisation** leads to the lowest peak memory allocated.

### Profiling of FP32

The FP32 reference was profiled for **five complete updates after 10 warm-ups**,
with eight microbatches per update. The main operations are grouped below by
purpose. GPU times are summed operation durations across the capture; the
per-update column divides those totals by five.

| Operation | Aggregate GPU time (ms) | Average GPU time per update (ms) |
|---|---:|---:|
| Log-softmax: forward | 75.94 | 15.19 |
| Log-softmax: backward | 137.61 | 27.52 |
| Vocabulary projection: forward | 63.59 | 12.72 |
| Vocabulary projection: backward | 135.12 | 27.02 |
| Other linear layers (attention projections and feed-forward) | 86.17 | 17.23 |
| Attention matrix multiplications | 58.92 | 11.78 |
| Tensor initialisation / fills | 34.50 | 6.90 |
| Tensor copies | 28.92 | 5.78 |
| Dropout: forward + backward | 28.39 | 5.68 |
| Attention softmax: forward + backward | 26.30 | 5.26 |
| Layer normalisation: forward + backward | 15.51 | 3.10 |
| Causal attention masking | 11.50 | 2.30 |
| GELU activation: forward + backward | 7.43 | 1.49 |

> Log-softmax forward/backward used **213.55 ms** in total
(about **42.71 ms per update**). Vocabulary projection added **198.71 ms**.
>
> Vocabulary/loss path is a substantial GPU workload, which motivated the fused cross-entropy kernel.

Sources: [benchmark report](runs/remote/implementation_profiling/implementations/report.md),
[FP32 operation timings](runs/remote/implementation_profiling/profiles-v1/fp32/pytorch/report.json),
[profiles](runs/remote/implementation_profiling/profiles-v1/).

## RTX 2070 SUPER: fused cross-entropy

- **GPU / software:** one RTX 2070 SUPER (8 GB); PyTorch 2.6.0+cu118,
  Triton 3.2.0.
- **Independent variables:** PyTorch versus Triton loss, FP32/FP16, prediction
  rows (257 or 512) and ignored-target fraction (0% or 50%).
- **Dependent variables:** loss forward/backward time, speedup and peak memory.
- **Controls:** synthetic logits, vocabulary 50,257, seed 42 and upstream
  gradient 1. Both backends compute cross-entropy on existing logits.

| Rows | Precision | Ignored | PyTorch (ms) | Triton (ms) | Speedup | Memory MiB: PyTorch / Triton | Relative Memory |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | FP32 | 0% | 2.3922 | 1.1892 | 2.012x | 392.64 / 196.33 | 2.000 |
| 512 | FP32 | 50% | 2.3918 | 0.9715 | 2.462x | 392.64 / 196.33 | 2.000 |
| 512 | FP16 | 0% | 2.0559 | 0.8108 | 2.536x | 296.32 / 100.01 | 2.963 |
| 512 | FP16 | 50% | 2.0558 | 0.7619 | 2.698x | 296.32 / 100.01 | 2.963 |
| 257 | FP32 | 0% | 1.2214 | 0.7898 | 1.547x | 200.00 / 100.01 | 2.000 |
| 257 | FP32 | 50% | 1.2205 | 0.7427 | 1.643x | 200.00 / 100.01 | 2.000 |

Relative Memory is PyTorch peak memory divided by Triton peak memory.

> Triton made the loss forward/backward operation
**1.55–2.70x faster**.
>
> At 512 rows it saved **196.31 MiB**, reducing peak memory
by **50% in FP32 and 66.2% in FP16**.

Source: [loss benchmark report](runs/remote/fused-cross-entropy/2070-loss/report.md).

## RTX 4090: GPU, loss-backend and workload scaling

Shared setup for the four experiments below:

- **GPUs / software:** one or two RTX 4090s on one Linux host, connected through
  PCIe and a host bridge; PyTorch 2.6.0+cu124, Triton 3.2.0; two GPUs use
  DistributedDataParallel (DDP).
- **Controls:** TinyStories dataset (10,000 selected stories per split), GPT-2
  tokenisation, vocabulary 50,257, sequence length 512, effective batch 64,
  BF16, Flash attention, standard AdamW, dropout 0.1 and seed 42. Execution is
  eager with checkpointing disabled.
- **Default model / batch:** 6 layers, width 160, 4 heads, feed-forward width
  640 and microbatch 8. One GPU accumulates 8 microbatches; two GPUs each
  accumulate 4. The model and batch sweeps change the settings named below.
- **Measurements:** throughput, update time, peak memory and validation loss
  after 60 updates (10 warm-up + 50 measured), using 32 validation batches.
  Validation loss is in nats per supervised token. Scaling efficiency is
  two-GPU throughput divided by twice one-GPU throughput.

### 1. One versus two GPUs

**Independent variable:** GPU count.

**Controls:** shared setup with PyTorch loss;
initial weights, global example groups and effective batch match.

| Metric | One GPU | Two GPUs |
|---|---:|---:|
| Tokens/s | 130,630 | 237,914 |
| Update time (ms) | 105.27 | 57.71 |
| Memory per GPU (GiB) | 2.632 | 2.669 |
| Validation loss | 8.10169 | 8.09909 |

> Two GPUs delivered **1.821x speedup** and **91.1% scaling
efficiency**, with similar per-GPU memory and validation loss.

Source: [BF16 scaling report](runs/remote/4090-scaling/4090-scaling-bf16/report.md).

### 2. PyTorch versus Triton loss

**Independent variables:** loss backend and GPU count.

**Controls:** shared setup;
Triton replaces cross-entropy on existing logits.

| Loss | GPUs | Tokens/s | Update (ms) | Memory per GPU (GiB) | Validation loss |
|---|---:|---:|---:|---:|---:|
| PyTorch | 1 | 130,384 | 105.41 | 2.632 | 8.10167 |
| PyTorch | 2 | 238,200 | 57.77 | 2.670 | 8.09904 |
| Triton | 1 | 213,676 | 64.34 | 1.114 | 8.10160 |
| Triton | 2 | 367,356 | 37.41 | 1.152 | 8.09935 |

> Triton increased complete-update throughput by **63.9% on
one GPU and 54.2% on two**, while cutting memory by **57.7% and 56.9%**.
Two GPUs with Triton achieved the highest baseline-model throughput:
**367,356 tokens/s**. Two-GPU scaling efficiency was 91.3% with PyTorch and
86.0% with Triton; validation losses differed by at most 0.00031 nats/token.

Sources: [PyTorch report](runs/remote/4090-scaling/loss-backend-scaling-20261107T212741Z/pytorch/report.md),
[Triton report](runs/remote/4090-scaling/loss-backend-scaling-20261107T212741Z/triton/report.md).

### 3. Microbatch size

**Independent variables:** microbatch per GPU and GPU count.

**Dependent variables:** throughput, scaling efficiency and memory.

**Controls:** shared
setup with PyTorch loss; accumulation adjusts to keep effective batch 64.

| Microbatch | Accumulation: 1 / 2 GPUs | Tokens/s: 1 GPU | Tokens/s: 2 GPUs | Efficiency | Memory GiB: 1 / 2 GPUs |
|---:|---:|---:|---:|---:|---:|
| 4 | 16 / 8 | 100,189 | 179,424 | 89.5% | 1.407 / 1.445 |
| 8 | 8 / 4 | 130,325 | 239,748 | 92.0% | 2.632 / 2.670 |
| 16 | 4 / 2 | 158,610 | 266,649 | 84.1% | 5.075 / 5.113 |
| 32 | 2 / 1 | 167,821 | 286,225 | 85.3% | 9.971 / 9.972 |

> Increasing microbatch 8 to 32 raised throughput by
**28.8% on one GPU and 19.4% on two**, using about **3.8x the memory**.
Microbatch 8 had the best scaling efficiency; microbatch 32 was fastest.

Source: [microbatch reports](runs/remote/4090-scaling/microbatch-scaling-20261107T213235Z/).

### 4. Model size

**Independent variables:** model depth, width and GPU count.

**Controls:** shared
setup with PyTorch loss, microbatch 8, four heads, feed-forward width four times
model width and shared validation membership.

| Layers x width | Tokens/s: 1 GPU | Tokens/s: 2 GPUs | Efficiency | Memory GiB: 1 / 2 GPUs | Validation loss: 1 / 2 GPUs |
|---|---:|---:|---:|---:|---:|
| 3 x 80 | 159,344 | 310,161 | 97.3% | 2.434 / 2.450 | 9.3904 / 9.3910 |
| 6 x 80 | 138,129 | 264,082 | 95.6% | 2.476 / 2.494 | 9.3702 / 9.3721 |
| 6 x 160 | 130,497 | 238,712 | 91.5% | 2.632 / 2.669 | 8.1017 / 8.0992 |
| 6 x 320 | 107,209 | 182,816 | 85.3% | 3.016 / 3.103 | 6.0598 / 6.0668 |
| 12 x 320 | 83,806 | 140,769 | 84.0% | 3.423 / 3.540 | 6.3692 / 6.2825 |

> Larger models used more memory, processed fewer tokens per
second and scaled less efficiently. The **6-layer, width-320 model** had the
lowest validation loss after the matched 60-update budget; doubling its depth
to 12 layers did not improve that result.

Decreasing scaling efficiency with model size is possibly due to increased communication overhead when synchronising gradients between GPUs.

Source: [model-size reports](runs/remote/4090-scaling/model-size-scaling-20261107T182056Z/).
