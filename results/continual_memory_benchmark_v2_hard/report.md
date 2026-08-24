# Continual Memory Benchmark

This is continual memory under interleaved tasks without online parameter learning, not generic long-context question answering.

All baselines use deterministic structured resolution after memory selection; this is an oracle reasoning control, not a language-model reasoning score.

Grounded answer accuracy requires every declared event and fact evidence group for both known and UNKNOWN answers.

Cross-task interference counts only failures absent from the same method, history, and checkpoint under the matched zero-distractor tier, divided by treated comparisons whose zero-distractor control was correct.

Model parameter tiers are metadata only and are not evaluated.
No online model weight updates occur.

## Baselines

| Method | Grounded accuracy | Raw answer accuracy | Exact evidence hit | Evidence recall | Stale intrusion | Retraction compliance | Causal cross-task interference |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_structured_memory | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| recency | 0.3864 | 0.6023 | 0.3864 | 0.4493 | 0.0000 | 0.2500 | 0.7037 |
| bm25 | 0.5439 | 0.6826 | 0.5439 | 0.6713 | 0.0992 | 0.4375 | 0.2968 |
| sliding_context:window_4k | 0.9545 | 1.0000 | 0.9545 | 0.9939 | 0.0000 | 0.8750 | 0.0606 |
| sliding_context:window_16k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| sliding_context:window_64k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| sliding_context:window_128k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| dense | 0.5742 | 0.7159 | 0.5742 | 0.6685 | 0.0909 | 0.5000 | 0.1302 |

## Retention By Interference

| Tier | Method | Grounded accuracy | Exact evidence hit |
|---|---|---:|---:|
| no_interference | full_structured_memory | 1.0000 | 1.0000 |
| no_interference | recency | 0.8182 | 0.8182 |
| no_interference | bm25 | 0.6364 | 0.6364 |
| no_interference | sliding_context:window_4k | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| no_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| no_interference | dense | 0.6364 | 0.6364 |
| light_interference | full_structured_memory | 1.0000 | 1.0000 |
| light_interference | recency | 0.7273 | 0.7273 |
| light_interference | bm25 | 0.4242 | 0.4242 |
| light_interference | sliding_context:window_4k | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| light_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| light_interference | dense | 0.6030 | 0.6030 |
| heavy_interference | full_structured_memory | 1.0000 | 1.0000 |
| heavy_interference | recency | 0.0000 | 0.0000 |
| heavy_interference | bm25 | 0.4545 | 0.4545 |
| heavy_interference | sliding_context:window_4k | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| heavy_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| heavy_interference | dense | 0.5485 | 0.5485 |
| extreme_interference | full_structured_memory | 1.0000 | 1.0000 |
| extreme_interference | recency | 0.0000 | 0.0000 |
| extreme_interference | bm25 | 0.6606 | 0.6606 |
| extreme_interference | sliding_context:window_4k | 0.8182 | 0.8182 |
| extreme_interference | sliding_context:window_16k | 1.0000 | 1.0000 |
| extreme_interference | sliding_context:window_64k | 1.0000 | 1.0000 |
| extreme_interference | sliding_context:window_128k | 1.0000 | 1.0000 |
| extreme_interference | dense | 0.5091 | 0.5091 |

## Scallop Recursive-Reasoning Ablation

| Method | Overall accuracy | Direct retraction | Unretracted lineage | Two-hop lineage retraction |
|---|---:|---:|---:|---:|
| python_gold_resolver_replay | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| scallop_recursive | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| scallop_one_hop | 0.6667 | 1.0000 | 1.0000 | 0.0000 |

## Memory Growth

- Checkpoints: 1320
- Mean stored turns: 108.73
- Maximum stored turns: 693
- Mean stored tokens: 2315.87
- Maximum stored tokens: 15102
