# Persona Kimi-Scallop Benchmark Pipeline

## Status and scope

This document describes the current-branch pipeline from deterministic latent benchmark construction through pair-conditioned fixed-assignment Kimi A/B realization, Scallop source selection, five-arm Qwen evaluation, and fixed-assignment bootstrap analysis (`experiments/synthetic_temporal_preferences.py:580-933`; `experiments/persona_surface_derivation.py:1750-1995`; `experiments/persona_end_to_end_benchmark.py:1183-1418`; `experiments/persona_fixed_assignment_analysis.py:1020-1119`).

The new pair-conditioned Kimi A/B run is **not claimed complete here**.
The only committed A/B corpora and pooled A/B results currently under `results/` are the older deterministic surface derivations and their Qwen evaluations; they remain historical evidence and become superseded only after both pair-conditioned Kimi replacements, their pair gate, both replacement Qwen runs, and replacement pooled analysis complete and authenticate successfully (`results/persona_conflict_conversations_surface_a/generation_manifest.json:24-59`; `results/persona_conflict_conversations_surface_b/generation_manifest.json:24-59`; `results/persona_surface_generalization_v1/analysis.md:141-173`; `experiments/persona_surface_derivation.py:1980-1995`).

## 1. Deterministic latent benchmark

`generate_dataset` creates the benchmark semantics before any Kimi call: facts carry stable IDs, subjects, predicates, objects, validity intervals, scopes, authority, evidence, and provenance; events add causal relations such as `transitions_from`, `corrects`, `supersedes`, `resolves`, `duplicate_of`, `retracts`, `retracts_lineage`, and `conflicts_with` (`experiments/synthetic_temporal_preferences.py:601-659`; `experiments/synthetic_temporal_preferences.py:793-811`).

The `anti_shortcut_interleaved_v3` profile adds same-entity hard negatives, alias bridges, duplicate private-note lineage, lineage retraction, open contradictions, explicit conflict resolution, and delayed probes (`experiments/synthetic_temporal_preferences.py:662-811`).
Each delayed query binds an ordered set of required event/fact groups and a trigger event in `checkpoint_contract`; its gold is resolved only from the causal prefix ending at that trigger (`experiments/synthetic_temporal_preferences.py:840-865`; `experiments/synthetic_temporal_preferences.py:914-933`).
The original generation configuration fixes 16 histories split 2 train, 2 validation, and 12 test, seed 47, lexical condition, and the `anti_shortcut_interleaved_v3` profile (`configs/persona_conflict_generation.json:7-19`).

Kimi realizes model-visible conversation text but does not choose latent facts, causal links, query contracts, or gold answers.
Its prompt must preserve supplied event order and meaning, direct versus inferred speaker authority, required values, conflict/correction/resolution semantics, and event IDs as non-visible metadata (`experiments/persona_conversation_generator.py:887-935`; `experiments/persona_conversation_generator.py:1115-1179`).

## 2. Original Kimi corpus

The committed parent corpus is `results/persona_conflict_conversations_v1/`.
Its manifest records a completed `kimi-k3` generation over 16 histories and cryptographically binds all nine JSONL artifacts (`results/persona_conflict_conversations_v1/generation_manifest.json:13-25`; `results/persona_conflict_conversations_v1/generation_manifest.json:41-42`; `results/persona_conflict_conversations_v1/generation_manifest.json:493-505`).

### Direct versus derived artifacts

The exact distinction is:

| Artifact | Origin and role |
|---|---|
| `requests.jsonl` | Direct request provenance: the exact messages, seed, parameters, and prompt hash sent for each Kimi call (`results/persona_conflict_conversations_v1/generation_manifest.json:9-10,36-40`; `experiments/persona_conversation_generator.py:1173-1179`). |
| `raw_responses.jsonl` | Direct endpoint response provenance: returned content, returned model name, finish reason, provider-reported usage, and response hash (`results/persona_conflict_conversations_v1/generation_manifest.json:9-10,281-493`). |
| `dialogue.jsonl` | Derived materialization parsed from accepted Kimi response content; it is the model-input conversation artifact, not a raw provider response (`results/persona_conflict_conversations_v1/generation_manifest.json:2-11`; `experiments/persona_conversation_generator.py:882-884,1030-1068`). |
| `events.jsonl` | Mixed artifact: deterministic latent event/fact contracts with Kimi-authored `model_text` attached after validation (`results/persona_conflict_conversations_v1/generation_manifest.json:2-11`; `experiments/persona_conversation_generator.py:1108-1167`). |
| `facts.jsonl`, `queries.jsonl`, `examples.jsonl`, `candidate_updates.jsonl`, `source_documents.jsonl` | Deterministically derived benchmark, supervision, or source-evidence artifacts; they are not direct provider outputs (`results/persona_conflict_conversations_v1/generation_manifest.json:2-11`; `experiments/synthetic_temporal_preferences.py:889-1079`). |

Thus “Kimi-authored corpus” means that Kimi authored the accepted visible dialogue surfaces.
It does not mean that every corpus row came directly from the provider or that Kimi authored latent semantics (`results/persona_conflict_conversations_v1/generation_manifest.json:2-11,35-41`; `experiments/persona_conversation_generator.py:1030-1089`).

### Verified corpus size

The table counts the committed files as raw bytes and complete JSONL rows.
“Qwen tokens” means encoding each complete UTF-8 file with `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with `add_special_tokens=False`; that exact tokenizer revision and tokenizer-file hashes are bound by the committed parent benchmark manifest (`results/persona_end_to_end_qwen35_4b_v1/manifest.json:76-97`).

| Committed parent artifact | Rows | Bytes | Qwen tokens |
|---|---:|---:|---:|
| `candidate_updates.jsonl` (`results/persona_conflict_conversations_v1/candidate_updates.jsonl`) | 144 | 280,448 | 84,512 |
| `dialogue.jsonl` (`results/persona_conflict_conversations_v1/dialogue.jsonl`) | 416 | 199,343 | 51,934 |
| `events.jsonl` (`results/persona_conflict_conversations_v1/events.jsonl`) | 416 | 844,221 | 250,842 |
| `examples.jsonl` (`results/persona_conflict_conversations_v1/examples.jsonl`) | 112 | 45,840 | 15,440 |
| `facts.jsonl` (`results/persona_conflict_conversations_v1/facts.jsonl`) | 368 | 443,086 | 138,075 |
| `queries.jsonl` (`results/persona_conflict_conversations_v1/queries.jsonl`) | 208 | 129,540 | 41,394 |
| `raw_responses.jsonl` (`results/persona_conflict_conversations_v1/raw_responses.jsonl`) | 210 | 293,194 | 86,421 |
| `requests.jsonl` (`results/persona_conflict_conversations_v1/requests.jsonl`) | 210 | 846,394 | 196,334 |
| `source_documents.jsonl` (`results/persona_conflict_conversations_v1/source_documents.jsonl`) | 480 | 85,932 | 27,945 |
| **Total** (the nine files above) | **2,564** | **3,167,998** | **892,897** |

These file-token counts are separate from provider billing telemetry.
The 210 raw responses report 193,917 prompt tokens, 380,726 completion tokens, 339,721 reasoning tokens, and 574,643 total prompt-plus-completion tokens; the manifest therefore reports 41,005 visible completion tokens and records that the requested non-thinking mode was not honored (`results/persona_conflict_conversations_v1/raw_responses.jsonl`; `results/persona_conflict_conversations_v1/generation_manifest.json:29-34,493-505`).

Provider provenance is bounded: exact artifact and prompt/response hashes authenticate the bytes, but the model identity is only what the endpoint self-reported.
The manifests do not provide hostile provider-origin attestation (`results/persona_conflict_conversations_v1/generation_manifest.json:35-41,255`; `results/persona_surface_generalization_v1/analysis.md:166-173`).

## 3. Pair-conditioned fixed-assignment Kimi A/B replacement

The replacement entry point is `experiments/persona_surface_derivation.py`, configured by `configs/persona_surface_pair_generation.json`.
Runtime paths and credentials come from `PERSONA_SURFACE_PARENT_DIR`, `PERSONA_SURFACE_A_OUTPUT_DIR`, `PERSONA_SURFACE_B_OUTPUT_DIR`, `PERSONA_SURFACE_PAIR_GATE_PATH`, `KIMI_BASE_URL`, `KIMI_API_KEY`, and `KIMI_MODEL` (`configs/persona_surface_pair_generation.json:2-19`; `experiments/persona_surface_derivation.py:2006-2074`).

The stage order is deliberate:

1. Authenticate the completed parent manifest and every declared parent artifact by SHA-256 (`experiments/persona_surface_derivation.py:435-465`).
2. Generate the complete fixed A mapping first through fresh Kimi calls. Then generate fixed B through separate fresh Kimi calls; B receives only a flat normalized list of A target phrases to avoid, never A history IDs, categories, source phrases, or mapping associations. A receives no B information (`experiments/persona_surface_derivation.py:571-629`; `experiments/persona_surface_derivation.py:898-978`).
3. Validate exact mapping row coverage/order, category shape and naturalness, global uniqueness, no parent-phrase reuse, no latent IDs, and equality or bidirectional token-sequence containment against prior same-assignment and A-to-B forbidden targets. The final pair validator remains a defense-in-depth check (`experiments/persona_surface_derivation.py:322-432`).
4. For each accepted mapping, issue fresh Kimi dialogue calls over all 416 parent events, validate every response against the existing semantic dialogue contract, and rebuild only visible event text, dialogue, visible query text, and visible gold (`experiments/persona_surface_derivation.py:1015-1129`).
5. Preserve byte-identical parent artifacts outside the explicitly rebuilt set; write fresh request/response journals and a completed child manifest only after all checks pass (`experiments/persona_surface_derivation.py:62-70`; `experiments/persona_surface_derivation.py:1117-1169`).
6. Validate both completed children together, reject parent provenance reuse or A/B accepted-response overlap, then atomically write the pair gate (`experiments/persona_surface_derivation.py:1276-1496`; `experiments/persona_surface_derivation.py:1563-1590`; `experiments/persona_surface_derivation.py:1980-1995`).

Generation is resumable, but a running or partial request/response journal is not completion evidence.
Cached controls, parent identity, request metadata, response hashes, and accepted/superseded mapping state must match exactly; otherwise resume fails loudly (`experiments/persona_surface_derivation.py:713-895`; `experiments/persona_surface_derivation.py:1658-1747`).

## 4. Pair gate and parent checkpoints

The completed pair gate binds one parent plus A and B paths, child manifest hashes, generation-control hashes, mapping hashes, artifact hashes, and a bounded assurance statement (`experiments/persona_surface_derivation.py:132-149`; `experiments/persona_surface_derivation.py:1563-1590`).
Authentication re-hashes the gate, parent, both children, every artifact, fresh provenance, mapping disjointness, and dataset-to-assignment path before scheduling or model loading (`experiments/persona_surface_derivation.py:1499-1560`; `experiments/persona_end_to_end_benchmark.py:1171-1186`).

The child retains the parent's latent event ordering, relations, query gold, and checkpoint contracts exactly; only `model_text`, `surface_object`, `query_text`, `surface_query_text`, and `surface_gold` may change (`experiments/persona_surface_derivation.py:62-70`; `experiments/persona_surface_derivation.py:513-543`).
The pair gate does **not** claim that changed dialogue has identical token lengths.
Exact checkpoint indices are deferred to the scheduler, which rebuilds parent and child interleavings with the configured Qwen tokenizer, requires identical turn/event identity, chooses token-distance indices from the authenticated parent schedule, and applies those indices to each child (`experiments/persona_surface_derivation.py:132-149`; `experiments/persona_interference_schedule.py:452-489`; `experiments/persona_interference_schedule.py:497-593`).
This preserves the parent's causal checkpoint design instead of allowing longer or shorter Kimi wording to move an assignment's evaluation boundary (`experiments/persona_interference_schedule.py:461-489,501-580`).

## 5. Query-blind Scallop and source-ID injection

For each `(history_id, checkpoint)` the evaluator sends Scallop only compact structured events from that same-history causal prefix.
The payload contains IDs, supersession links, and fact subject/predicate/object/temporal/qualifier fields, but no evaluation query, gold answer, prediction, or model-visible text (`experiments/persona_end_to_end_benchmark.py:1011-1056`; `experiments/preference_stream_injection.py:175-203`).

Scallop rule version `preference_stream.v1` derives two relations: a preference change requires same subject/scope, a supersession edge, changed value, changed start time, and an alias event; preference incongruity instead requires changed value and changed authority at the same start time (`experiments/preference_stream_injection.py:12-15`; `experiments/preference_stream_injection.py:110-160`).
The service may return only `kind` and exactly three causal `source_event_ids`; answer, gold, prediction, object, and text fields are forbidden, and every returned ID must belong to the supplied prefix (`experiments/preference_stream_injection.py:59-107`).

Before evaluation, a semantic canary must produce the exact expected source triples for both rules, and every delayed probe must have its expected relation (`experiments/persona_end_to_end_benchmark.py:452-528`; `experiments/persona_end_to_end_benchmark.py:1059-1099`).
The evaluator resolves returned IDs locally to original natural turns, rejects missing or cross-history sources, strips all IDs from the prompt, and packs those source turns plus the maximal recent suffix under the matched token cap (`experiments/persona_end_to_end_benchmark.py:390-449`).
If Scallop returns no source IDs, the structured arm calls the sliding-arm builder directly, making the same-cap prompts identical by construction (`experiments/persona_end_to_end_benchmark.py:405-414`).

## 6. Five Qwen arms

Both assignment configs require exactly these arms: `sliding_context_4096`, `sliding_context_16384`, `structured_memory_4096`, `structured_memory_16384`, and untruncated `full_qwen_context` (`configs/persona_end_to_end_benchmark_surface_a.json:26-51`; `configs/persona_end_to_end_benchmark_surface_b.json:26-51`; `experiments/persona_end_to_end_benchmark.py:175-190`).
All arms use the same scheduled condition, question, short-answer instruction, deterministic decoding, and local Qwen model; the two structured arms differ from matched sliding arms only by Scallop-selected source-turn injection and resulting suffix packing (`configs/persona_end_to_end_benchmark_surface_a.json:53-73`; `experiments/persona_end_to_end_benchmark.py:558-647`; `experiments/persona_end_to_end_benchmark.py:1131-1149`).
Full context must fit with the answer allowance or the run fails rather than truncating (`experiments/persona_end_to_end_benchmark.py:650-658`; `experiments/persona_end_to_end_benchmark.py:1226-1238`).

Each assignment uses 12 test histories, two delayed query families, and five phases/distances per query, producing exactly 120 conditions and 600 generations when complete (`configs/persona_end_to_end_benchmark_surface_a.json:12-25`; `experiments/persona_end_to_end_benchmark.py:964-1008`; `experiments/persona_end_to_end_benchmark.py:1141-1143`).
Outputs live in the env-selected benchmark directories and comprise `generation_manifest.json`, `generations.jsonl`, `predictions.jsonl`, `metrics.json`, and final `manifest.json` (`experiments/persona_end_to_end_benchmark.py:1272-1277`; `experiments/persona_end_to_end_benchmark.py:1362-1410`).

## 7. Scoring and bootstrap

The first non-empty generated line is normalized as the short answer and scored with exact match and token F1; exact match is primary and token F1 secondary (`experiments/persona_end_to_end_benchmark.py:661-669`; `experiments/persona_fixed_assignment_analysis.py:1034-1041`).
Per-arm summaries report row count, history-cluster count, mean exact match, and mean F1 (`experiments/persona_end_to_end_benchmark.py:680-696`).
Arm contrasts are paired on identical evaluation inputs and use a nonparametric percentile bootstrap that resamples whole base-history clusters, preserving all rows within each sampled history (`experiments/persona_end_to_end_benchmark.py:699-748`).

The final A/B analysis re-hashes corpus and benchmark artifacts from source, recomputes canonical scores, requires all five arms and complete counts, requires distinct A/B datasets and predictions under one parent and pair gate, and validates assignment golds against the authenticated parent oracle (`experiments/persona_fixed_assignment_analysis.py:70-134`; `experiments/persona_fixed_assignment_analysis.py:580-685`; `experiments/persona_fixed_assignment_analysis.py:1063-1091`).
Pooled A+B uncertainty uses 12 `base_history_id` clusters and keeps each history's A and B replicas together (`experiments/persona_fixed_assignment_analysis.py:1020-1050`).
The bootstrap unit is the base history, not the fixed surface assignment.
A and B share one parent and B is explicitly conditioned on A for lexical disjointness, so they are not statistically independent mapping replicates; assignment-level differences and difference-in-paired-differences remain descriptive for these two fixed assignments.

## 8. Superseded-on-replacement deterministic results

The current committed pooled report is from deterministic category-preserving A/B derivations, not the new pair-conditioned Kimi A/B pipeline (`results/persona_conflict_conversations_surface_a/generation_manifest.json:24-36`; `results/persona_conflict_conversations_surface_b/generation_manifest.json:24-36`; `results/persona_surface_generalization_v1/analysis.md:141-173`).
Its pooled exact-match scores are 0.350 for sliding 4K, 0.617 for sliding 16K, 0.679 for structured 4K, 0.713 for structured 16K, and 0.721 for full context (`results/persona_surface_generalization_v1/analysis.md:7-15`).
Its pooled structured-minus-sliding exact-match deltas are 0.329 at 4K and 0.096 at 16K, with the report's history-clustered 95% intervals (`results/persona_surface_generalization_v1/analysis.md:37-48`).

Those numbers are **old deterministic results, pending supersession**.
They must not be presented as evidence from statistically independent Kimi surfaces, because the committed report explicitly says A and B are deterministic derivations of the same visible parent material and limits difference-in-paired-differences claims to descriptive analysis (`results/persona_surface_generalization_v1/analysis.md:141-173`).
They become superseded only when completed pair-conditioned Kimi child manifests, the authenticated pair gate, completed five-arm A and B benchmark manifests, and replacement pooled analysis artifacts all exist and pass the code gates above (`experiments/persona_surface_derivation.py:1190-1590`; `experiments/persona_end_to_end_benchmark.py:1390-1418`; `experiments/persona_fixed_assignment_analysis.py:630-689,1020-1095`).

## Operational map

| Stage | Configuration / code | Artifacts and completion gate |
|---|---|---|
| Parent corpus | `configs/persona_conflict_generation.json`; `experiments/persona_conversation_generator.py` | `results/persona_conflict_conversations_v1/`; completed Kimi manifest plus every declared SHA-256 (`results/persona_conflict_conversations_v1/generation_manifest.json:13-25,493-505`). |
| Pair-conditioned fixed A/B | `configs/persona_surface_pair_generation.json`; `experiments/persona_surface_derivation.py` | Env-selected A and B directories; each requires a completed `kimi-k3` child manifest, fresh requests/responses, complete mapping/dialogue coverage, and artifact hashes (`experiments/persona_surface_derivation.py:1133-1169,1190-1422`). |
| Pair gate | `PERSONA_SURFACE_PAIR_GATE_PATH` and `PERSONA_SURFACE_PAIR_GATE_SHA256` | Env-selected gate JSON; must bind and re-prove parent, A, B, controls, mappings, artifacts, and paths (`experiments/persona_surface_derivation.py:1499-1590`). |
| Scheduling / Qwen A | `configs/persona_end_to_end_benchmark_surface_a.json` | `PERSONA_SURFACE_A_BENCHMARK_OUTPUT_DIR`; gate authentication occurs before tokenizer/model load (`configs/persona_end_to_end_benchmark_surface_a.json:2-25`; `experiments/persona_end_to_end_benchmark.py:1171-1205`). |
| Scheduling / Qwen B | `configs/persona_end_to_end_benchmark_surface_b.json` | `PERSONA_SURFACE_B_BENCHMARK_OUTPUT_DIR`; same gate, assignment B, and source manifest hash from env (`configs/persona_end_to_end_benchmark_surface_b.json:2-25`). |
| Scallop | `PERSONA_SCALLOP_ENDPOINT`; `experiments/preference_stream_injection.py` | Hard-timeout semantic canary and source-ID-only relation map (`configs/persona_end_to_end_benchmark_surface_a.json:54-57`; `experiments/persona_end_to_end_benchmark.py:1201-1214`). |
| Scoring | `experiments/persona_end_to_end_benchmark.py` | Final benchmark `manifest.json` with completed status and hashes for generation manifest, generations, predictions, and metrics (`experiments/persona_end_to_end_benchmark.py:1362-1410`). |
| Pooled analysis | `experiments/persona_fixed_assignment_analysis.py` | Analyst-selected output directory containing authenticated `analysis.json`, `analysis.md`, manifest, and checksums; the committed deterministic predecessor is `results/persona_surface_generalization_v1/` (`results/persona_surface_generalization_v1/manifest.json:1-74`). |

No operational stage may infer completion from directory existence, partial JSONL rows, or a running manifest.
Completion requires the stage-specific `status: completed`, complete coverage/count invariants, and successful hash and provenance revalidation (`experiments/persona_surface_derivation.py:1190-1496`; `experiments/persona_end_to_end_benchmark.py:1272-1320,1390-1418`; `experiments/persona_fixed_assignment_analysis.py:580-689`).
