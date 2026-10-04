# Continual Memory Benchmark

This is continual memory under interleaved tasks without online parameter learning, not generic long-context question answering.

All baselines use deterministic structured resolution after memory selection; this is an oracle reasoning control, not a language-model reasoning score.

UNKNOWN predictions receive answer credit only when all required event and fact evidence groups are retrieved.

Cross-task interference counts only failures absent from the same method, history, and checkpoint under the matched zero-distractor tier, divided by treated comparisons whose zero-distractor control was correct.

Model parameter tiers are metadata only and are not evaluated.
No online model weight updates occur.

## Baselines

| Method | Answer accuracy | Exact evidence hit | Evidence recall | Stale intrusion | Retraction compliance | Causal cross-task interference |
|---|---:|---:|---:|---:|---:|---:|
| full_structured_memory | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| recency | 0.4722 | 0.4722 | 0.4861 | 0.0000 | 0.5000 | 0.7037 |
| bm25 | 0.7204 | 0.7204 | 0.7514 | 0.1333 | 1.0000 | 0.0463 |
| sliding_context:window_4k | 0.9722 | 0.9722 | 0.9861 | 0.0000 | 1.0000 | 0.0370 |
| sliding_context:window_16k | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| sliding_context:window_64k | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| sliding_context:window_128k | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| dense | 0.9481 | 0.8407 | 0.8944 | 0.0491 | 1.0000 | 0.0404 |

## Retention By Interference

| Tier | Method | Answer accuracy | Exact evidence hit |
|---|---|---:|---:|
| no_interference | full_structured_memory | 1.0000 | 1.0000 |
| no_interference | recency | 1.0000 | 1.0000 |
| no_interference | bm25 | 0.6667 | 0.6667 |
| no_interference | sliding_context:window_4k | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| no_interference | dense | 0.9778 | 0.8704 |
| light_interference | full_structured_memory | 1.0000 | 1.0000 |
| light_interference | recency | 0.8889 | 0.8889 |
| light_interference | bm25 | 0.5778 | 0.5778 |
| light_interference | sliding_context:window_4k | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| light_interference | dense | 0.9667 | 0.8593 |
| heavy_interference | full_structured_memory | 1.0000 | 1.0000 |
| heavy_interference | recency | 0.0000 | 0.0000 |
| heavy_interference | bm25 | 0.7593 | 0.7593 |
| heavy_interference | sliding_context:window_4k | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| heavy_interference | dense | 0.9481 | 0.8407 |
| extreme_interference | full_structured_memory | 1.0000 | 1.0000 |
| extreme_interference | recency | 0.0000 | 0.0000 |
| extreme_interference | bm25 | 0.8778 | 0.8778 |
| extreme_interference | sliding_context:window_4k | 0.8889 | 0.8889 |
| extreme_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| extreme_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| extreme_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| extreme_interference | dense | 0.9000 | 0.7926 |

## Memory Growth

- Checkpoints: 1080
- Mean stored turns: 80.50
- Maximum stored turns: 429
- Mean stored tokens: 4747.69
- Maximum stored tokens: 25488
