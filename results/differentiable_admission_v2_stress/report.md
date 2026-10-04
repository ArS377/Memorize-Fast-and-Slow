# Differentiable Admission Benchmark

All learned models score only candidates that pass the immutable hard gate.
Hard-rejected updates are forced to `reject` and audited separately.
The oracle uses gold soft labels and is an upper bound, not a deployable baseline.
The clean split remains a ceiling result because histories repeat deterministic feature templates; clean accuracy verifies execution rather than model advantage.
Confidence-noise scenarios add deterministic Gaussian noise only to observed candidate and active-conflict confidence scores. Gold decisions and hard-gate truth remain fixed to measure robustness to extraction uncertainty. Noise draws are paired across standard deviations, and stress headlines use eligible-only accuracy so invariant hard rejects cannot dilute errors.

| Model | Dev accuracy | Clean test accuracy | Noise 0.05 eligible mean / min | Noise 0.15 eligible mean / min | Noise 0.3 eligible mean / min | Max hard-gate violations |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 1.0000 | 1.0000 | 0.9804 / 0.9767 | 0.9558 / 0.9444 | 0.9382 / 0.9233 | 0 |
| mlp | 1.0000 | 1.0000 | 1.0000 / 1.0000 | 0.9878 / 0.9811 | 0.9596 / 0.9533 | 0 |
| differentiable_scallop | 1.0000 | 1.0000 | 1.0000 / 1.0000 | 0.9804 / 0.9744 | 0.9536 / 0.9389 | 0 |
| fixed_symbolic_scallop | 1.0000 | 1.0000 | 0.9216 / 0.9067 | 0.9273 / 0.9144 | 0.9260 / 0.9133 | 0 |
| oracle_soft_upper_bound | 1.0000 | 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0 |

## Limitations

- The noise model is synthetic and does not substitute for natural-language extraction evaluation.
- The oracle upper bound is invariant by construction and is not deployable.
- History identities are held out, but event and candidate-family templates repeat across splits.

Scallop version: `0.2.4`.
PyTorch version: `2.5.1+cpu`.
