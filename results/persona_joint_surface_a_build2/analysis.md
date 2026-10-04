# Graphiti External Baseline Analysis

Analysis version `persona_graphiti_analysis.v1`.

## Pre-registered predictions

Graphiti was predicted to do well on plain preference change and plausibly on backdated correction, and to fail attributably on stale re-assertion (newest-wins invalidates the current correct fact), scope exceptions, and lineage retraction, while getting authority conflicts right by coincidence. The per-family and per-phase tables below are where those predictions are checked.

## Exact match by arm and query family

| Arm | Query family | Exact match | F1 | n |
|---|---|---:|---:|---:|
| `graphiti_memory_16384` | preference_change | 56.67% | 56.67% | 60 |
| `graphiti_memory_16384` | preference_incongruity | 76.67% | 81.67% | 60 |
| `graphiti_memory_4096` | preference_change | 48.33% | 52.83% | 60 |
| `graphiti_memory_4096` | preference_incongruity | 48.33% | 55.83% | 60 |
| `hybrid_kg_memory_16384` | preference_change | 51.67% | 51.67% | 60 |
| `hybrid_kg_memory_16384` | preference_incongruity | 100.00% | 100.00% | 60 |
| `hybrid_kg_memory_4096` | preference_change | 56.67% | 59.17% | 60 |
| `hybrid_kg_memory_4096` | preference_incongruity | 100.00% | 100.00% | 60 |
| `sliding_context_16384` | preference_change | 50.00% | 50.00% | 60 |
| `sliding_context_16384` | preference_incongruity | 70.00% | 79.17% | 60 |
| `sliding_context_4096` | preference_change | 36.67% | 38.17% | 60 |
| `sliding_context_4096` | preference_incongruity | 33.33% | 36.67% | 60 |
| `structured_memory_16384` | preference_change | 50.00% | 50.83% | 60 |
| `structured_memory_16384` | preference_incongruity | 91.67% | 94.17% | 60 |
| `structured_memory_4096` | preference_change | 53.33% | 54.83% | 60 |
| `structured_memory_4096` | preference_incongruity | 85.00% | 86.67% | 60 |

## Exact match by arm and phase or token distance

| Arm | Condition | Exact match | F1 | n |
|---|---|---:|---:|---:|
| `graphiti_memory_16384` | delayed_probe | 25.00% | 29.17% | 24 |
| `graphiti_memory_16384` | post_update | 87.50% | 89.58% | 24 |
| `graphiti_memory_16384` | pre_update | 87.50% | 87.50% | 24 |
| `graphiti_memory_16384` | token_distance_4096 | 70.83% | 72.92% | 24 |
| `graphiti_memory_16384` | token_distance_8192 | 62.50% | 66.67% | 24 |
| `graphiti_memory_4096` | delayed_probe | 25.00% | 32.92% | 24 |
| `graphiti_memory_4096` | post_update | 87.50% | 89.58% | 24 |
| `graphiti_memory_4096` | pre_update | 91.67% | 91.67% | 24 |
| `graphiti_memory_4096` | token_distance_4096 | 25.00% | 35.00% | 24 |
| `graphiti_memory_4096` | token_distance_8192 | 12.50% | 22.50% | 24 |
| `hybrid_kg_memory_16384` | delayed_probe | 54.17% | 54.17% | 24 |
| `hybrid_kg_memory_16384` | post_update | 100.00% | 100.00% | 24 |
| `hybrid_kg_memory_16384` | pre_update | 79.17% | 79.17% | 24 |
| `hybrid_kg_memory_16384` | token_distance_4096 | 75.00% | 75.00% | 24 |
| `hybrid_kg_memory_16384` | token_distance_8192 | 70.83% | 70.83% | 24 |
| `hybrid_kg_memory_4096` | delayed_probe | 70.83% | 70.83% | 24 |
| `hybrid_kg_memory_4096` | post_update | 100.00% | 100.00% | 24 |
| `hybrid_kg_memory_4096` | pre_update | 83.33% | 83.33% | 24 |
| `hybrid_kg_memory_4096` | token_distance_4096 | 70.83% | 75.00% | 24 |
| `hybrid_kg_memory_4096` | token_distance_8192 | 66.67% | 68.75% | 24 |
| `sliding_context_16384` | delayed_probe | 12.50% | 20.83% | 24 |
| `sliding_context_16384` | post_update | 87.50% | 91.67% | 24 |
| `sliding_context_16384` | pre_update | 79.17% | 81.25% | 24 |
| `sliding_context_16384` | token_distance_4096 | 66.67% | 70.83% | 24 |
| `sliding_context_16384` | token_distance_8192 | 54.17% | 58.33% | 24 |
| `sliding_context_4096` | delayed_probe | 0.00% | 2.08% | 24 |
| `sliding_context_4096` | post_update | 83.33% | 87.50% | 24 |
| `sliding_context_4096` | pre_update | 83.33% | 83.33% | 24 |
| `sliding_context_4096` | token_distance_4096 | 8.33% | 12.08% | 24 |
| `sliding_context_4096` | token_distance_8192 | 0.00% | 2.08% | 24 |
| `structured_memory_16384` | delayed_probe | 50.00% | 52.08% | 24 |
| `structured_memory_16384` | post_update | 87.50% | 91.67% | 24 |
| `structured_memory_16384` | pre_update | 79.17% | 81.25% | 24 |
| `structured_memory_16384` | token_distance_4096 | 75.00% | 75.00% | 24 |
| `structured_memory_16384` | token_distance_8192 | 62.50% | 62.50% | 24 |
| `structured_memory_4096` | delayed_probe | 87.50% | 87.50% | 24 |
| `structured_memory_4096` | post_update | 83.33% | 87.50% | 24 |
| `structured_memory_4096` | pre_update | 83.33% | 83.33% | 24 |
| `structured_memory_4096` | token_distance_4096 | 41.67% | 45.42% | 24 |
| `structured_memory_4096` | token_distance_8192 | 50.00% | 50.00% | 24 |

## Matched effect sizes within each query family

| Query family | Comparison | Delta (points) | 95% CI | histories |
|---|---|---:|---|---:|
| preference_change | `hybrid_kg_memory_4096` - `sliding_context_4096` | +20.00 | [8.33, 31.67] | 12 |
| preference_change | `hybrid_kg_memory_16384` - `sliding_context_16384` | +1.67 | [-10.00, 11.67] | 12 |
| preference_change | `graphiti_memory_4096` - `sliding_context_4096` | +11.67 | [1.67, 21.67] | 12 |
| preference_change | `graphiti_memory_16384` - `sliding_context_16384` | +6.67 | [-1.67, 15.00] | 12 |
| preference_change | `hybrid_kg_memory_4096` - `graphiti_memory_4096` | +8.33 | [-8.33, 23.33] | 12 |
| preference_change | `hybrid_kg_memory_16384` - `graphiti_memory_16384` | -5.00 | [-18.33, 6.67] | 12 |
| preference_change | `hybrid_kg_memory_4096` - `structured_memory_4096` | +3.33 | [-11.67, 16.67] | 12 |
| preference_change | `hybrid_kg_memory_16384` - `structured_memory_16384` | +1.67 | [-10.00, 11.67] | 12 |
| preference_incongruity | `hybrid_kg_memory_4096` - `sliding_context_4096` | +66.67 | [60.00, 73.33] | 12 |
| preference_incongruity | `hybrid_kg_memory_16384` - `sliding_context_16384` | +30.00 | [18.33, 45.00] | 12 |
| preference_incongruity | `graphiti_memory_4096` - `sliding_context_4096` | +15.00 | [5.00, 26.67] | 12 |
| preference_incongruity | `graphiti_memory_16384` - `sliding_context_16384` | +6.67 | [1.67, 11.67] | 12 |
| preference_incongruity | `hybrid_kg_memory_4096` - `graphiti_memory_4096` | +51.67 | [38.33, 63.33] | 12 |
| preference_incongruity | `hybrid_kg_memory_16384` - `graphiti_memory_16384` | +23.33 | [11.67, 38.33] | 12 |
| preference_incongruity | `hybrid_kg_memory_4096` - `structured_memory_4096` | +15.00 | [6.67, 25.00] | 12 |
| preference_incongruity | `hybrid_kg_memory_16384` - `structured_memory_16384` | +8.33 | [3.33, 13.33] | 12 |

## Error taxonomy by arm and query family

### `graphiti_memory_16384`

| Query family | correct | stale_intrusion | cross_account_intrusion | abstained | other |
|---|---:|---:|---:|---:|---:|
| preference_change | 34 | 13 | 1 | 9 | 3 |
| preference_incongruity | 46 | 5 | 1 | 8 | 0 |

### `graphiti_memory_4096`

| Query family | correct | stale_intrusion | cross_account_intrusion | abstained | other |
|---|---:|---:|---:|---:|---:|
| preference_change | 29 | 15 | 1 | 11 | 4 |
| preference_incongruity | 29 | 2 | 8 | 19 | 2 |

### `hybrid_kg_memory_16384`

| Query family | correct | stale_intrusion | cross_account_intrusion | abstained | other |
|---|---:|---:|---:|---:|---:|
| preference_change | 31 | 27 | 0 | 2 | 0 |
| preference_incongruity | 60 | 0 | 0 | 0 | 0 |

### `hybrid_kg_memory_4096`

| Query family | correct | stale_intrusion | cross_account_intrusion | abstained | other |
|---|---:|---:|---:|---:|---:|
| preference_change | 34 | 23 | 0 | 3 | 0 |
| preference_incongruity | 60 | 0 | 0 | 0 | 0 |

### `sliding_context_16384`

| Query family | correct | stale_intrusion | cross_account_intrusion | abstained | other |
|---|---:|---:|---:|---:|---:|
| preference_change | 30 | 17 | 0 | 11 | 2 |
| preference_incongruity | 42 | 8 | 3 | 7 | 0 |

### `sliding_context_4096`

| Query family | correct | stale_intrusion | cross_account_intrusion | abstained | other |
|---|---:|---:|---:|---:|---:|
| preference_change | 22 | 7 | 0 | 31 | 0 |
| preference_incongruity | 20 | 4 | 2 | 34 | 0 |

### `structured_memory_16384`

| Query family | correct | stale_intrusion | cross_account_intrusion | abstained | other |
|---|---:|---:|---:|---:|---:|
| preference_change | 30 | 19 | 0 | 10 | 1 |
| preference_incongruity | 55 | 2 | 1 | 2 | 0 |

### `structured_memory_4096`

| Query family | correct | stale_intrusion | cross_account_intrusion | abstained | other |
|---|---:|---:|---:|---:|---:|
| preference_change | 32 | 9 | 0 | 19 | 0 |
| preference_incongruity | 51 | 2 | 0 | 7 | 0 |

## Ingestion provenance

- graphiti-core version: `0.29.3`
- ingestion mode: `incremental_per_account`
- episode variant: `e2e`
- retrieval state: `point_in_time`
- episodes ingested: 312
- episode failures: 0
- valid facts committed: 3290
- timestamp mapping: {"base": "2024-01-01T00:00:00+00:00", "in_text_dates": "untouched", "indexing": "global_stream_turn_index", "step_seconds": 60}
- store state hash: `2c610f09adabdfd9d4104be396358f259f1f4ea91e272e89e062950a1a9eeb77`
