# Qwen Persona Graphiti Ingestion With Benchmark-Matched Retrieval

This run completed 960 deterministic generations across 120 authenticated conditions and 8 matched arms.

**Surface coverage:** this run used the historical deterministic corpus `persona_conflict_conversations_surface_a_graphiti_build2` only.
It did not use either member of the later pair-conditioned Kimi A/B corpus, and there is no matched Surface-B result for this historical corpus.
The [later matched Kimi A/B comparison](../persona_joint_kimi_ab_a100_build1/README.md) evaluates two wordings of a separate corpus; it does not establish wording robustness for this historical run.

## Memoryless floor

- distinct gold values: 37 ('Friday delivery', 'Saturday delivery', 'Sunday delivery', 'Thursday delivery', 'UNKNOWN', 'afternoon delivery', 'apple spice tea', 'blackcurrant tea', 'blueberry tea', 'breakfast delivery', 'chrysanthemum tea', 'coconut tea', 'courier delivery', 'doorstep delivery', 'elderflower tea', 'fortnightly delivery', 'ginger turmeric tea', 'holiday delivery', 'honeybush tea', 'magnolia tea', 'midday delivery', 'monthly delivery', 'next-day delivery', 'night delivery', 'office delivery', 'osmanthus tea', 'porch delivery', 'priority delivery', 'raspberry leaf tea', 'reception delivery', 'same-day delivery', 'scheduled delivery', 'standard delivery', 'sunrise delivery', 'sunset delivery', 'twilight delivery', 'verbena tea')
- best single constant answer: 10.00% EM
- best per-query-family constant answer: 13.33% EM

Read every arm below against that floor, not against zero.

| Arm | Exact match | F1 | n | histories |
|---|---:|---:|---:|---:|
| `graphiti_memory_16384` | 66.67% | 69.17% | 120 | 12 |
| `graphiti_memory_4096` | 48.33% | 54.33% | 120 | 12 |
| `hybrid_kg_memory_16384` | 75.83% | 75.83% | 120 | 12 |
| `hybrid_kg_memory_4096` | 78.33% | 79.58% | 120 | 12 |
| `sliding_context_16384` | 60.00% | 64.58% | 120 | 12 |
| `sliding_context_4096` | 35.00% | 37.42% | 120 | 12 |
| `structured_memory_16384` | 70.83% | 72.50% | 120 | 12 |
| `structured_memory_4096` | 69.17% | 70.75% | 120 | 12 |

## Matched effect sizes

- `hybrid_kg_memory_4096` minus `sliding_context_4096`: +43.33 exact-match points, history-clustered 95% bootstrap interval [35.83, 50.00] over 12 history clusters.
- `hybrid_kg_memory_16384` minus `sliding_context_16384`: +15.83 exact-match points, history-clustered 95% bootstrap interval [6.67, 25.83] over 12 history clusters.
- `graphiti_memory_4096` minus `sliding_context_4096`: +13.33 exact-match points, history-clustered 95% bootstrap interval [7.50, 19.17] over 12 history clusters.
- `graphiti_memory_16384` minus `sliding_context_16384`: +6.67 exact-match points, history-clustered 95% bootstrap interval [2.50, 10.83] over 12 history clusters.
- `hybrid_kg_memory_4096` minus `graphiti_memory_4096`: +30.00 exact-match points, history-clustered 95% bootstrap interval [20.00, 37.50] over 12 history clusters.
- `hybrid_kg_memory_16384` minus `graphiti_memory_16384`: +9.17 exact-match points, history-clustered 95% bootstrap interval [-1.67, 20.00] over 12 history clusters.
- `hybrid_kg_memory_4096` minus `structured_memory_4096`: +9.17 exact-match points, history-clustered 95% bootstrap interval [0.83, 17.50] over 12 history clusters.
- `hybrid_kg_memory_16384` minus `structured_memory_16384`: +5.00 exact-match points, history-clustered 95% bootstrap interval [-0.83, 10.00] over 12 history clusters.

## Error taxonomy

Stale intrusion means the answer names a value the queried account genuinely held at some point but not at the queried date. It is the failure a newest-wins revision policy is predicted to make. The label is coarse: it does not verify predicate, scope, or validity interval, so treat it as an upper bound on temporal stale-reads rather than a proven count.

| Arm | Correct | Stale intrusion | Cross-account | Abstained | Other |
|---|---:|---:|---:|---:|---:|
| `graphiti_memory_16384` | 66.67% | 15.00% | 1.67% | 14.17% | 2.50% |
| `graphiti_memory_4096` | 48.33% | 14.17% | 7.50% | 25.00% | 5.00% |
| `hybrid_kg_memory_16384` | 75.83% | 22.50% | 0.00% | 1.67% | 0.00% |
| `hybrid_kg_memory_4096` | 78.33% | 19.17% | 0.00% | 2.50% | 0.00% |
| `sliding_context_16384` | 60.00% | 20.83% | 2.50% | 15.00% | 1.67% |
| `sliding_context_4096` | 35.00% | 9.17% | 1.67% | 54.17% | 0.00% |
| `structured_memory_16384` | 70.83% | 17.50% | 0.83% | 10.00% | 0.83% |
| `structured_memory_4096` | 69.17% | 9.17% | 0.00% | 21.67% | 0.00% |

## Mechanism

The Graphiti arms used `graphiti-core 0.29.3` in `incremental_per_account` mode with episode variant `e2e` and retrieval state `point_in_time`, ingesting 312 episodes across 12 accounts with 0 episode failures.

Extraction used `Qwen/Qwen3-4B` and embeddings used `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The synthetic ingestion clock was {"base": "2024-01-01T00:00:00+00:00", "in_text_dates": "untouched", "indexing": "global_stream_turn_index", "step_seconds": 60}.

This arm is Graphiti ingestion with benchmark-matched retrieval, not off-the-shelf Graphiti. The deviations from stock defaults are recorded in `manifest.graphiti_memory.deviations_from_default`.

The model was `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` under Transformers 5.15.1 and Torch 2.13.0+cu130. Generation was greedy, batch size one, on the configured device.

## What this artifact does not establish

- Wording robustness is not established for this historical corpus, which was evaluated on deterministic Surface A only.
- State fidelity is not reported. `graphiti_store_states.json` exports the valid-fact sets, but the alignment from Graphiti's extracted vocabulary to the corpus's canonical fact space is unimplemented, so only downstream generation is measured here.
- Only the two delayed query families are probed directly. Scope exceptions, lineage retraction, and backdated correction are present in the stream as interference, so their effects are observable but not attributable.
- No audit cost is measured: no latency, storage, or ledger-overhead comparison is made between arms.
- The manifest hashes the configured Neo4j endpoint and database but does not authenticate a Neo4j server-instance identifier or server version.

## Headline: hybrid Scallop KG vs the Graphiti external baseline

Both memory systems ran in **one invocation on one schedule** with an
**identical, neutral evidence header**, so this is a paired history-clustered
comparison with no framing asymmetry and no cross-run confound.

| Comparison | Delta (EM points) | 95% CI | Resolvable? |
|---|---:|---|---|
| Hybrid KG − Graphiti @ 4K | **+30.00** | [+20.00, +37.50] | **yes** |
| Hybrid KG − Graphiti @ 16K | +9.17 | [−1.67, +20.00] | no |
| Hybrid KG − sliding @ 4K | +43.33 | [+35.83, +50.00] | yes |
| Hybrid KG − sliding @ 16K | +15.83 | [+6.67, +25.83] | yes |
| Graphiti − sliding @ 4K | +13.33 | [+7.50, +19.17] | yes |
| Graphiti − sliding @ 16K | +6.67 | [+2.50, +10.83] | yes |
| Hybrid KG − structured @ 4K | +9.17 | [+0.83, +17.50] | yes |
| Hybrid KG − structured @ 16K | +5.00 | [−0.83, +10.00] | no |

The 4K result is the claim this benchmark supports: under a budget too small to
carry the raw history, declared write-time revision beats inferred write-time
revision by 30 points. At 16K the gap remains positive but is not resolvable at
120 conditions, and it must not be reported as demonstrated.

## Between-build extraction variance is not measured here

The protocol asks for three independent Graphiti ingestion builds; this
artifact is one. The bootstrap intervals therefore resample histories within a
single extraction realization and do **not** include Graphiti's run-to-run
extraction variance.

An earlier build on this same corpus, not published here because it used a
biased prompt header, put the Graphiti arm at 48.33% versus 54.17% at 4K. That
is only an informal indication that the variance is non-trivial relative to the
smaller effects in the table above; it is not a measurement, and it is a further
reason to treat the 16K hybrid-versus-Graphiti gap as unresolved rather than
null. Run builds 2 and 3 to report this properly.

## Where the advantage comes from

| Arm | Gold reached the prompt | Accuracy given it did |
|---|---:|---:|
| `hybrid_kg_memory_4096` | **100.0%** | 79.6% |
| `hybrid_kg_memory_16384` | **100.0%** | 77.8% |
| `graphiti_memory_4096` | 67.6% | 54.8% |
| `graphiti_memory_16384` | 67.6% | 68.5% |

The Scallop-gated store put the gold value in front of the model on **every**
condition; Graphiti's inferred extraction and revision managed 67.6%. Where
both delivered the evidence, the hybrid prompt was still used more accurately.
The second column is the part that speaks to revision policy rather than
extraction coverage.

## Validity checks on this corpus

- memoryless per-family constant rule: **13.33%**
- recency oracle with perfect event access: **35.00%** (`scripts/shortcut_oracle_probe.py`)
- first-value oracle: **0.00%**
- foreign-account gold occurrences in prompts: **0**
- gold appearing in its own query text: **0 / 120**
- temporal boundary violations in retrieved rows: **0** (build 1 had 1)
- empty Graphiti retrievals: **0 / 120**

`sliding_context_4096` scored exactly 35.00%, matching the recency oracle,
which independently confirms the corpus discriminates against newest-wins.

## Limits

Single stochastic Graphiti ingestion build; two delayed query families;
downstream utility only, with no state-fidelity scorer; account scoping is an
acknowledged oracle for both memory arms; `full_qwen_context` excluded because
its prompts exceed 24 GB of KV cache during generation.
