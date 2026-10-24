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

This repository currently defines the project scope. The training system,
profiling experiments, and custom GPU kernel are planned work; no performance
results are claimed yet.
