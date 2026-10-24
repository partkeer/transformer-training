# Repository guidance

## Project purpose

Build a small transformer training system in PyTorch, then systematically improve
its speed and memory efficiency using profiling, controlled experiments, and one
carefully chosen custom GPU kernel.

- Use TinyStories as the dataset.
- Use a decoder-only transformer with causal next-token prediction.
- Establish a correct baseline before introducing performance optimisations.
- Keep the implementation small and understandable, with experiments that make
  the effect of each optimisation clear.

## Implementation and optimisation

- Preserve causal masking, next-token target alignment, and separation between
  training and validation data.
- Make experiment settings explicit and reproducible, including model size,
  tokenisation, data splits, sequence length, effective batch size, precision,
  random seeds, and hardware/software environment.
- Profile before selecting optimisation targets. Use measured bottlenecks to
  choose the project's single custom GPU kernel; do not choose it in advance
  without evidence.
- Retain a PyTorch reference implementation for the custom kernel. Check forward
  results and gradients where applicable, using tolerances appropriate to the
  numerical precision and covering supported shapes and boundary cases.
- Document supported kernel inputs and constraints, and provide a clear fallback
  or error for unsupported cases.
- Avoid unrelated infrastructure or features that obscure the training and
  optimisation work.

## Verification and experiments

- Run checks appropriate to each change. For training logic, verify causal
  behaviour, tensor shapes, loss computation, and gradient flow as relevant.
- Compare optimisations against a recorded baseline under consistent conditions.
  Change one factor at a time wherever practical and explain unavoidable coupled
  changes.
- Warm up GPU workloads, synchronise appropriately for timing, and repeat
  measurements. Distinguish training-step benchmarks from end-to-end throughput.
- Record tokens per second, step time, peak GPU memory usage, and validation loss.
  Explain changes in workload, precision, or effective batch size that affect
  comparisons.
- Report correctness and training-stability checks alongside performance results.
  Distinguish source inspection, local tests, GPU measurements, and full training
  runs; never imply that an unperformed check has passed.
- Update the README when implemented capabilities, execution instructions, or
  measured results change. Label planned work clearly and do not invent results.

## Communication

- Always give detailed, in-depth answers unless the user explicitly requests
  otherwise.
- Include examples, edge cases, and practical implications where relevant.
- When appropriate, show formulas, short derivations, or pseudocode.

## Git commit messages

Write clear, atomic commit messages that explain the intent of a change and make project history easy to scan.

### General rules

- Make each commit represent one logical change. Do not combine unrelated bug fixes, formatting, dependency updates, or refactors.
- Use the imperative mood in the subject line: `Add`, `Fix`, `Refactor`, `Remove`. Treat it as completing “If applied, this commit will …”.
- Keep the subject concise: aim for 50 characters; use at most 72.
- Capitalise the first word of the subject and do not end it with a full stop.
- Separate the subject from the body with one blank line.
- Wrap body lines at approximately 72 characters.
- Use the body to explain why the change was made, relevant trade-offs, alternatives considered, and any behavioural or user-visible impact. Do not merely restate the diff.
- Reference relevant issues, tickets, or discussions when useful. Prefer a full URL if the hosting platform does not reliably auto-link short references.
- Do not commit knowingly broken code, failing tests, or an incomplete change unless the repository convention explicitly supports work-in-progress commits.

### Commit format

Use Conventional Commits:

```text
<type>(optional scope): short imperative description

Optional body explaining motivation, context, and impact.

Optional footer(s), such as issue references or breaking-change notes.
```

Use these commit types:

| Type | Use for |
|---|---|
| `feat` | New user-facing or API-facing functionality |
| `fix` | Bug fix |
| `refactor` | Code change that does not change behaviour |
| `docs` | Documentation-only changes |
| `test` | Test-only changes |
| `chore` | Tooling, dependencies, CI, maintenance, and repository housekeeping |
| `perf` | Performance improvement |
| `build` | Build system or dependency changes |
| `ci` | CI configuration changes |