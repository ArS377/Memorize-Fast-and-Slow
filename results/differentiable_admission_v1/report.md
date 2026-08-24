# Differentiable Admission Benchmark

All learned models score only candidates that pass the immutable hard gate.
Hard-rejected updates are forced to `reject` and audited separately.
The oracle uses gold soft labels and is an upper bound, not a deployable baseline.
All methods reach the ceiling because this v1 dataset repeats deterministic feature templates across history-disjoint splits; the result verifies execution, not a learned-model advantage.

| Model | Dev accuracy | Test accuracy | Hard-gate violations |
|---|---:|---:|---:|
| logistic_regression | 1.0000 | 1.0000 | 0 |
| mlp | 1.0000 | 1.0000 | 0 |
| differentiable_scallop | 1.0000 | 1.0000 | 0 |
| fixed_symbolic_scallop | 1.0000 | 1.0000 | 0 |
| oracle_soft_upper_bound | 1.0000 | 1.0000 | 0 |

Scallop version: `0.2.4`.
PyTorch version: `2.5.1+cpu`.
