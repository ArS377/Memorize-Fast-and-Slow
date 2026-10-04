# Qwen3.5-0.8B Interleaved-Memory Generation Pilot

Status: completed preliminary pilot

This run evaluates direct local Qwen3.5-0.8B generation from sliding-context prompts sampled from the interleaved delayed-retrieval benchmark.
It does not evaluate generation from structured memory.

| Prompt cap | Checkpoints | Exact match | 95% clustered CI | Hit 64-token generation cap | Oracle answer available in context |
|---|---:|---:|---:|---:|---:|
| 64K | 64 | 3.13% | [0.00%, 7.81%] | 100.00% | 18.75% |
| 128K | 64 | 1.56% | [0.00%, 4.69%] | 98.44% | 50.00% |

## Interpretation

- The model exhausted the 64-token generation budget on 127 of 128 outputs, but 124 outputs still contained a uniquely extractable label; cap saturation alone does not identify the cause of the low accuracy.
- Oracle answer availability is reported separately and is not model accuracy.
- The sample contains 128 checkpoints clustered over 32 histories.
- These results do not compare sliding context against the structured-memory method.

Machine-readable evidence is in [`metrics.json`](metrics.json), with raw generations and predictions preserved beside it.
