# Multi-task Context Distractor Benchmark

This benchmark isolates retrieval and context-availability failures under labeled multi-task distraction; deterministic answer resolution is an oracle reasoning control.

| Method | Answer accuracy | Gold evidence recall | Target-thread precision |
|---|---:|---:|---:|
| recency | 0.0833 | 0.0833 | 0.0833 |
| random | 0.0583 | 0.0583 | 0.0667 |
| bm25 | 0.2500 | 0.2500 | 0.3000 |
| oracle_thread_filter | 1.0000 | 1.0000 | 1.0000 |
| full_structured_history | 1.0000 | 1.0000 | 0.0807 |

## Realized context tiers

| Context tier | Model-input tokens | Requested target thread | Realized target thread | Max position error | Hard lexical distractors |
|---|---:|---:|---:|---:|---:|
| compact_4k | 2,832-3,106 | 10.000% | 9.815% | 1.547 pp | 1 |
| standard_16k | 14,055-15,420 | 2.000% | 1.978% | 0.268 pp | 3 |
| long_64k | 56,150-61,603 | 0.500% | 0.495% | 0.026 pp | 5 |
| extended_128k | 112,276-123,180 | 0.250% | 0.248% | 0.042 pp | 7 |

## Head-window truncation

| Window tier | Answer accuracy | Gold evidence recall |
|---|---:|---:|
| window_4k | 0.3333 | 0.3333 |
| window_16k | 0.6667 | 0.6667 |
| window_64k | 0.9167 | 0.9167 |
| window_128k | 1.0000 | 1.0000 |

## Limitations

- No language model is evaluated in this first context-layer run.
- Parameter-count tiers are reporting metadata and are independent of context-window tiers.
- BM25 and full-history scores use deterministic structured answer resolution after selection.

Tokenizer: `{'requested_name': 'Qwen/Qwen3-4B', 'requested_revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'resolved_revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'implementation': 'Qwen2TokenizerFast', 'transformers_version': '4.57.6', 'add_special_tokens': False}`.
