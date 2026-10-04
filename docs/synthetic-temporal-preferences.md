# Synthetic Temporal Preferences MVP

This document records the current synthetic-memory MVP and the meaning of its
baselines.

## Purpose

The MVP is a deterministic testbed for temporal preference memory.
It is not yet a publishable benchmark.
It exists to separate memory-state correctness from LLM generation and
retrieval behavior before introducing natural-language extraction noise.

The generated artifacts are written by
`experiments/synthetic_temporal_preferences.py`:

- `events.jsonl`: ordered event-sourced user-memory updates.
- `facts.jsonl`: accepted fact records in the existing NeuroSym fact schema.
- `candidate_updates.jsonl`: clean and corrupted candidate writes with gold
  decisions.
- `queries.jsonl`: structured point-in-time state queries and gold answers.
- `examples.jsonl`: short-answer natural-language examples.

## Current Event Semantics

Each generated history contains these operations:

- `add`
- `supersede`
- `temporary_exception`
- `hard_constraint`
- `ambiguous_conflict`
- `retract`
- `backdated_correction`
- `duplicate_delivery`
- `direct_user_correction`

The `anti_shortcut_stream_v2` profile additionally introduces duplicate-lineage,
same-entity hard-negative, and alias-bridge event families.
Lineage uses `add`, `duplicate_delivery`, and `retract`; aliases and hard
negatives use the profile-specific `context_note` operation.

`materialize_event_states()` applies explicit retractions and emits a snapshot
after every event.
`resolve_preference()` derives a preference from subject, query date, and
scope.
The live Scallop policy resolves the same default-versus-exact-scope rule via
`/resolve_preference` on the Python 3.10 Scallop service.

The current gold examples establish:

| Query | Gold answer |
| --- | --- |
| Alice, 2025-08-01, default scope | `coffee` |
| Alice, 2025-08-05, `travel_japan` scope | `sushi` |
| Recommend peanuts to Alice | `INFEASIBLE` |
| Recall Alice's retracted old address | `UNKNOWN` |
| Resolve Alice's explicit spicy-food ambiguity | `UNKNOWN` |

## Baseline Definitions

The names below are deliberately distinct.

| Baseline | Input at query time | What it tests |
| --- | --- | --- |
| `no_history` | Question only, no event history | Parametric-model floor. It is not a transcript-memory baseline. |
| Raw full transcript | All event text concatenated and supplied directly to an LLM | Needle-in-a-haystack / full-context baseline. No fact externalization, retrieval index, or symbolic policy. |
| `recency` | Event history | Latest observed assertion, intentionally ignoring temporal validity, scope, and retraction. |
| `full_history` | Event history | Deterministic temporal and scope resolver over all events. This is a semantic control, not an independent learning baseline. |
| Scallop policy | Materialized active event state | Symbolic implementation of declared temporal/scope policy. It must match the deterministic semantic control before being compared with weaker methods. |

## Anti-Shortcut Profile

`anti_shortcut_stream_v2` removes structured role, task, thread, event, and operation labels from all model-visible retrieval text.
Retrievers receive the natural-language query only.
The profile adds aliases, same-entity lexical hard negatives, duplicate-memory lineage, positive lineage controls, and descendant-retraction checkpoints.
Its train/development and test splits hold out behavioral compositions of alias direction, lineage direction, and negative-text family rather than only history identities.
Known answers and `UNKNOWN` answers both require their complete declared evidence sets for grounded credit.

The hardened continual-memory bundle is `results/continual_memory_benchmark_v3_stream/`.
It contains 150 held-out episodes and 1,950 checkpoints drawn from histories 851 through 880.
The deepest tier reaches 408,761 Qwen3-4B tokens and places 128 long, multi-statement cross-history archives after every target event.
Sliding-context methods retain and evaluate those distractor events; they do not discard them using hidden role labels.

| Method | Answer accuracy | Grounded answer accuracy | Evidence recall |
| --- | ---: | ---: | ---: |
| Full structured-memory control | 100.00% | 100.00% | 100.00% |
| Recency | 45.38% | 26.15% | 30.41% |
| BM25 | 74.05% | 61.18% | 71.38% |
| Dense BGE retrieval | 60.15% | 46.87% | 56.48% |
| Sliding 4K context | 63.08% | 50.77% | 53.91% |
| Sliding 16K context | 80.00% | 70.77% | 74.30% |
| Sliding 64K context | 95.38% | 90.77% | 94.30% |
| Sliding 128K context | 96.92% | 95.38% | 96.73% |

These are deterministic retrieval and state-resolution results, not language-model generation results.
The Python result is a replay of the benchmark gold resolver and is not an independent oracle.

The deep-context treatment delays both an early preference transition and a later source-authority incongruity beyond 128K tokens.
For both checkpoint families and all four context windows, raw grounded accuracy is 0% and actual Scallop source injection is 100%.
The same-history rule-factorial controls isolate the symbolic relation: change-only scores 100% on transitions and 0% on authority incongruities, while incongruity-only scores 0% and 100%, respectively.
A reverse-ranked same-stream capsule control scores 0% in every deep treatment cell, testing whether arbitrary Scallop-derived source material is sufficient.
Scallop receives every target and distractor event in the causal structured stream and returns source event IDs for all changed or incongruous preference pairs.
Query-time BM25 reserves one capsule slot per enabled relation and ranks within each relation using only visible query text; the injected context copies the selected original source statements verbatim and receives neither the hidden gold answer nor evidence contract.
Answer scoring still uses the disclosed deterministic oracle resolver, so this establishes source selection and context preservation rather than language-model reasoning quality.

The bundle also contains a matched Scallop causal ablation over 90 private-memory checkpoints.
Actual `scallopy==0.2.4` recursive transitive closure scores 100%, while the one-hop program scores 0% on descendant-retraction treatment checkpoints and 100% on both direct-retraction and unretracted controls.
The interaction isolates recursive duplicate-lineage reasoning as the changed symbolic capability.

## Interleaved Long-Horizon Profile

`anti_shortcut_interleaved_v3` replaces focal-history distractor blocks with a
single peer-task stream built by a custom deterministic Python scheduler.
No external synthetic-data or scheduling library is used.
Each task advances in causal order through 4-8 event segments, then yields to
other tasks before resuming.
Rolling admission distributes preference transitions, authority conflicts,
backdated corrections, retractions, explicit contradiction openings, and
rectifications across every stream third at horizons of at least 32 accounts.

The bundle `results/interleaved_memory_benchmark_v2/` measures prompt-free
transcript-suffix complete-provenance loss across account horizons from 8 through 1,024 and
context windows from 4K through 1M tokens. It also evaluates four online causal
checkpoints per history using matched full structured memory and finite suffixes.

| Evolving tasks | Stream tokens | 4K loss | 128K loss | 256K loss | 1M loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 21,712 | 78.13% | 0.00% | 0.00% | 0.00% |
| 64 | 172,781 | 97.27% | 28.52% | 0.00% | 0.00% |
| 256 | 689,768 | 99.32% | 82.23% | 64.45% | 0.00% |
| 512 | 1,371,249 | 99.66% | 90.87% | 82.13% | 23.93% |
| 1,024 | 2,733,929 | 100.00% | 95.41% | 90.70% | 61.89% |

The final sweep allocates each declared window to transcript turns and does not
reserve query-prompt tokens. The online checkpoint windows do include their
exact query prompt. Both are context-availability measurements, not
LLM-generation scores.
It demonstrates that any fixed suffix can work at a short enough horizon but
eventually loses old preference changes and rectifications as peer tasks keep
evolving.
The generated stream is 2.607 times the declared 1,048,576-token context limit;
the run fails loudly if it does not meet the requested multiplier.
Actual `scallopy==0.2.4` derives one canonical opening pair per history: all
1,024 pairs are resolved and no unexpected unresolved pair remains.

| Online method | Oracle answer sufficiency | Complete provenance | Grounded answer |
| --- | ---: | ---: | ---: |
| Full structured memory | 100.00% | 100.00% | 100.00% |
| Sliding 4K | 41.43% | 13.11% | 13.11% |
| Sliding 16K | 92.65% | 81.49% | 81.49% |
| Sliding 128K | 100.00% | 100.00% | 100.00% |

Online answer sufficiency uses the same disclosed deterministic resolver as
source-gold generation. It measures whether selected structured context is
answer-sufficient, not independent LLM reasoning accuracy. Complete provenance
requires every grounded evidence group and is intentionally stricter.

The raw full-transcript baseline is the correct answer to the question:
"Does explicit fact externalization and symbolic policy help beyond supplying
the model every prior user statement?"

## Current Smoke Results

These results use only five synthetic queries and are plumbing checks, not
paper results.

| Method | Accuracy |
| --- | ---: |
| Qwen3-4B with no event history | 0/2, preference queries only |
| Raw full transcript supplied to Qwen3-4B | 1/2, preference queries only |
| Recency-only deterministic memory | 0/5 |
| Full-history temporal resolver | 5/5 |

The no-history Qwen run returned `unknown` for one query and hallucinated a
preference for the other.
The raw full-transcript run selected the scoped Japan preference for both
queries, which is wrong outside that scope.
This demonstrates why scope-aware state resolution is necessary, but the
sample is too small for any comparative claim.

## Multi-History Admission Results

The first deterministic run used 100 opaque independent histories and the
original two candidate families. It established the controlled write-policy
result below.

| Candidate-write policy | Query accuracy |
| --- | ---: |
| Accept every candidate | 80% |
| Actual Scallop admission control | 100% |

This is a controlled write-policy result only.
The 20-point gap comes from accept-all retaining the stale preference after its
valid successor, while Scallop rejects that corrupted write.
It does not establish an improvement over raw transcript, BM25, dense
retrieval, Mem0, Letta, or natural-language extraction.

The expanded admission dataset now contains 1,000 history-disjoint traces and
9,000 candidates split as 6,300 train, 1,350 development, and 1,350 test rows.
Three candidate families are rejected by the immutable hard gate: hard
constraint violations, tombstone resurrection, and direct-user conflicts that
lack explicit supersession. Eligible candidates are scored as `accept`,
`replace`, or `reject`.

The first differentiable run is in
`results/differentiable_admission_v1/`:

| Hard-gated method | Dev accuracy | Test accuracy | Hard-gate violations |
| --- | ---: | ---: | ---: |
| Logistic regression | 100% | 100% | 0 |
| Direct MLP | 100% | 100% | 0 |
| Differentiable Scallop | 100% | 100% | 0 |
| Fixed symbolic Scallop | 100% | 100% | 0 |
| Oracle soft-label upper bound | 100% | 100% | 0 |

This is a ceiling result, not evidence that one model is better. The v1 split
holds out history identities, but every history repeats the same deterministic
feature templates. The result verifies end-to-end differentiation, Scallop
rule execution, hard-gate isolation, artifact generation, and held-out
evaluation.

The corrected v2 stress bundle is
`results/differentiable_admission_v2_stress/`. It evaluates five deterministic
Gaussian confidence-noise repeats at each configured level and reports only
hard-gate-eligible candidates in the stress headline:

| Method | Noise 0.05 mean | Noise 0.15 mean | Noise 0.30 mean | Hard-gate violations |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 98.04% | 95.58% | 93.82% | 0 |
| Direct MLP | 100.00% | 98.78% | 95.96% | 0 |
| Differentiable Scallop | 100.00% | 98.04% | 95.36% | 0 |
| Fixed symbolic Scallop | 92.16% | 92.73% | 92.60% | 0 |
| Oracle soft-label upper bound | 100.00% | 100.00% | 100.00% | 0 |

Noise changes only observed candidate/conflict confidence and recomputes the
observed equality feature. Gold decisions and hard-gate truth remain fixed.
The bundle stores all 20,250 stressed candidate/repeat records for independent
recalculation.

## Multi-Task Context Distractors

This is a custom benchmark layer built on the synthetic temporal histories,
not an imported inconsistency benchmark.
The original v1 bundle used explicit model-visible role labels.
The v2 hard bundle removes self-disqualifying phrases such as `near-miss`,
`does not state`, and `separate task` from distractor text.

The first bundle is `results/multitask_context_benchmark_v1/`. It uses 30
held-out histories, four context/density tiers, and evidence positions near
10%, 50%, and 90%:

| Tier | Model-input tokens | Requested target thread | Realized target thread | Hard lexical near-misses |
| --- | ---: | ---: | ---: | ---: |
| Compact 4K | 2,833-3,107 | 10.000% | 9.813% | 1 |
| Standard 16K | 14,057-15,423 | 2.000% | 1.977% | 3 |
| Long 64K | 56,154-61,607 | 0.500% | 0.495% | 5 |
| Extended 128K | 112,281-123,187 | 0.250% | 0.248% | 7 |

| Context method | Answer accuracy | Gold evidence recall |
| --- | ---: | ---: |
| Recency | 8.33% | 8.33% |
| Deterministic random | 4.17% | 4.17% |
| BM25 | 25.00% | 25.00% |
| Oracle thread filter | 100.00% | 100.00% |
| Full structured-history control | 100.00% | 100.00% |

The harder text run in `results/multitask_context_benchmark_v2_hard/` reports
8.33% recency, 5.83% deterministic random, 25.00% BM25, and 100% for both
structured controls.

Head-window evidence availability is 33.33% at 4K, 66.67% at 16K, 91.67%
at 64K, and 100% at 128K. These are retrieval and context-availability
controls with deterministic structured answer resolution. No language model
has yet been evaluated on this context matrix. Parameter-count tiers (1-4B,
greater than 4-14B, greater than 14-72B, and greater than 72B) are a separate
reporting axis and do not imply a context capacity.

## Verified Scallop Behavior

The stale candidate "Alice prefers tea after July 2025" conflicts with the
active coffee preference and was rejected by the actual Scallop HTTP service
as `functional_conflict`.
The live scope policy returned `sushi` for Alice in `travel_japan` on
2025-08-05 and `coffee` for the default scope on the same date.

## Known Limits

- The admission benchmark remains structurally templated and exhibits a ceiling
  effect at low noise.
- The continual benchmark now holds out behavioral compositions, but its text
  is still generated from deterministic templates rather than human dialogue.
- Admission features come from structured synthetic records rather than a
  noisy natural-language extractor.
- Hard constraints, ambiguity, and retraction are applied by the admission
  gate but are not yet fully integrated into every query-time policy.
- The current Scallop service must be actual `scallopy`, never the Python
  fallback, for a symbolic result claim.
- The LongBench reproduction produced zero extracted facts for all 50 examples.
  Its KG scores are invalid for evaluating retrieval or Scallop.

## Required Gates Before External Baselines

1. Add natural-language extraction and verify that admission features never
   contain gold-derived fields.
2. Evaluate actual models from each parameter tier on the same frozen context
   cases and report tokenizer-specific effective context limits.
3. Add dense retrieval to the multi-task context matrix using the same cases.
4. Add Mem0 or Letta as external systems only after these controls are frozen.
