# Data Creation and Benchmark Methodology

## Objective and Scope

This document describes the current synthetic data pipeline and the regenerated `interleaved_continual_memory.v2` benchmark artifacts in NeuroSym.
It is grounded in the current generator, scheduler, evaluator, tests, and artifacts under `results/synthetic_temporal_preferences_1200_v5_interleaved/` and `results/interleaved_memory_benchmark_v2/`.
The source dataset identifies itself as `synthetic_temporal_preferences.v2`, uses the `anti_shortcut_interleaved_v3` hardness profile, and contains 1,200 independent account histories.
The regenerated dataset uses the generator's default `lexical` rendering condition.
The benchmark consumes only the 1,024 test histories and schedules them as peer tasks in one causal conversation.

The benchmark asks a deliberately narrow question.
It measures whether a bounded complete-turn transcript suffix or an account-local structured store retains enough of an evolving synthetic record to recover the deterministic gold state and its declared evidence.
It also measures final-sweep retention of four long-running relations as the number of peer tasks grows.
It does not measure natural-language extraction, model generation, learned reasoning, human conversational quality, or end-to-end user utility.
The online scorer is the same disclosed deterministic resolver used to derive source gold, so online answer scores are context-availability controls rather than independent reasoning accuracy.
Crucially, v2 uses all peer turns as positional token pressure when it selects a suffix, then filters the retained suffix by hidden history identity before deterministic resolution.
It therefore measures positional retention under the generated stream, not whether semantic distractors are harder than length-matched random filler and not whether a retriever can distinguish lexical negatives or traverse alias bridges.
Same-entity lexical negatives and alias bridges are generated data features and provenance-contract requirements whose intended retrieval challenge remains unmeasured in v2.

The current source of truth is the code and regenerated artifacts, not older numeric summaries in prose.
All counts and results below were recomputed from the current files after alias regeneration.

## Versioned Inputs and Outputs

The primary source and test pointers are:

| Role | Path |
| --- | --- |
| Dataset generator and semantic resolver | `experiments/synthetic_temporal_preferences.py` |
| Interleaving scheduler and suffix accounting | `experiments/interleaved_conversation.py` |
| v2 evaluation and artifact manifest | `experiments/interleaved_memory_benchmark.py` |
| Tokenizer and serialization contracts | `experiments/_continual_memory_config.py` and `experiments/_continual_memory_episodes.py` |
| Contradiction lifecycle rules | `experiments/contradiction_ledger.py` |
| Duplicate-lineage gold resolver | `experiments/private_lineage_reasoning.py` |
| Fail-closed Scallop HTTP boundary | `services/scallop_validator_service.py` |
| Dataset determinism, split isolation, aliases, compositions, candidate labels, and gold | `tests/test_synthetic_temporal_preferences.py` |
| Segment bounds, causal order, resumptions, lifecycle distribution, and suffix boundaries | `tests/test_interleaved_conversation.py` |
| Matched causal views, evidence contracts, scoring separation, and horizon checks | `tests/test_interleaved_memory_benchmark.py` |
| Temporal overlap, scope separation, authority guards, and Python and Scallop agreement | `tests/test_contradiction_ledger.py` |

The generated dataset directory contains five complementary artifacts.
`events.jsonl` is the authoritative ordered event trace consumed by the interleaved benchmark.
`queries.jsonl` contains structured query metadata, model-visible query text, deterministic gold, and checkpoint evidence contracts where applicable.
`candidate_updates.jsonl` contains proposed writes and admission labels.
`examples.jsonl` contains compact natural-language tasks with gold operations.
`facts.jsonl` is a selected fact mirror for compatibility with the broader NeuroSym schema.
`facts.jsonl` is not an exhaustive enumeration of facts embedded in events or candidates and must not be treated as one.

The v2 output directory contains `schedule.jsonl`, `online_checkpoint_predictions.jsonl`, `contradiction_ledger.jsonl`, `suffix_loss_witnesses.jsonl`, `metrics.json`, `report.md`, and `manifest.json`.
The manifest hashes the two dataset inputs directly consumed by the benchmark and every benchmark output written before the manifest itself.

## Exact Dataset Inventory and Splits

The 1,200 histories are assigned contiguously and deterministically to 88 train histories, 88 development histories, and 1,024 test histories.
Every row belonging to one history carries the same split across all five artifacts.
No history occurs in more than one split.

| Artifact | Rows per history | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `examples.jsonl` | 7 | 616 | 616 | 7,168 | 8,400 |
| `facts.jsonl` | 16 | 1,408 | 1,408 | 16,384 | 19,200 |
| `candidate_updates.jsonl` | 9 | 792 | 792 | 9,216 | 10,800 |
| `events.jsonl` | 26 | 2,288 | 2,288 | 26,624 | 31,200 |
| `queries.jsonl` | 13 | 1,144 | 1,144 | 13,312 | 15,600 |

Each history contributes seven compact examples.
Their operations are `accept_transition`, `scoped_preference`, `backdated_correction`, `idempotent_duplicate`, `replace_overlapping_candidate`, `direct_user_authority`, and `prevent_scope_leakage`.

Each history contributes 13 queries.
Eight are preference queries, two are private-lineage queries, and one each is a recommendation, private-recall, and ambiguity query.
The eight preference queries concern current preference, scoped preference, a backdated correction, duplicate delivery, authority conflict, scope leakage, delayed preference change, and delayed authority incongruity.
Four queries per history carry the causal checkpoint contracts used by interleaved v2: positive lineage, post-retraction lineage, delayed preference change, and delayed authority incongruity.
The dataset therefore contains 4,800 contracted queries, of which 4,096 belong to the test split evaluated by v2.

Each history contributes nine candidate updates.
Across the full dataset, gold decisions comprise 2,400 `accept`, 2,400 `replace`, and 6,000 `reject` rows.
The immutable hard-gate eligibility field is true for 7,200 rows and false for 3,600 rows.
Candidate rows support admission-policy experiments and document generated semantic stressors.
Candidate inventory must not be confused with scheduled event inventory because not every candidate is inserted into the 26-event transcript.

The 26 event rows per history have the following exact family distribution.

| Event family | Per history | Full dataset | v2 test stream |
| --- | ---: | ---: | ---: |
| `non_overlap_transition` | 2 | 2,400 | 2,048 |
| `scope_exception` | 1 | 1,200 | 1,024 |
| `hard_constraint` | 1 | 1,200 | 1,024 |
| `ambiguity` | 1 | 1,200 | 1,024 |
| `retraction` | 2 | 2,400 | 2,048 |
| `backdated_correction` | 1 | 1,200 | 1,024 |
| `duplicate_delivery` | 1 | 1,200 | 1,024 |
| `overlapping_replacement` | 1 | 1,200 | 1,024 |
| `source_authority_conflict` | 2 | 2,400 | 2,048 |
| `scope_leakage` | 1 | 1,200 | 1,024 |
| `alias_bridge` | 1 | 1,200 | 1,024 |
| `same_entity_hard_negative` | 3 | 3,600 | 3,072 |
| `private_lineage` | 4 | 4,800 | 4,096 |
| `delayed_preference_probe` | 2 | 2,400 | 2,048 |
| `contradiction_opening` | 2 | 2,400 | 2,048 |
| `contradiction_rectification` | 1 | 1,200 | 1,024 |

The exact operation totals are 8,400 `add`, 7,200 `context_note`, 3,600 `duplicate_delivery`, 2,400 each of `temporary_exception`, `retract`, and `direct_user_correction`, and 1,200 each of `supersede`, `hard_constraint`, `ambiguous_conflict`, and `backdated_correction`.

`facts.jsonl` contains 16 selected fact families per history.
Those families are initial, current, scoped, backdated, replaceable, indirect source, direct correction, scope leakage, three private-lineage facts, two delayed probes, and three explicit contradiction-lifecycle facts.
Seven additional fact identities occur in each event trace but not in `facts.jsonl`: ambiguity, hard constraint, private value, alias bridge, and three lexical lineage negatives.
Candidate-only facts add further identities, including stale, accepted, replacement, hard-constraint violation, resurrection, direct conflict, and ambiguity-resolution cases.
Complete event reconstruction therefore requires `events.jsonl`, and complete admission analysis additionally requires `candidate_updates.jsonl`.

## Schemas and Semantic Families

### Facts

Every generated fact follows the NeuroSym fact shape with a stable fact ID, history and example IDs, subject, `PREFERS` predicate, object, temporal interval, qualifiers, provenance, support text, confidence metadata, and run metadata.
Synthetic provenance spans cover the generated support statement and identify `synthetic_generator.v1` as the extractor.
The default confidence score is 0.95.
Deliberately weak conflicting or replaceable facts use 0.10, and a high-evidence replacement control uses 0.99.
Facts are structured latent records even when the model-visible stream uses natural aliases and natural-language text.

### Events

An event wraps a fact with an event ID, history ID, session ID, turn index, semantic family, operation, split, hardness profile, and relation-specific pointers such as `supersedes`, `retracts`, `duplicate_of`, or `resolves`.
Per-history event order defines causal order.
The scheduler interleaves histories but never reorders events within a history.
Every interleaved-profile event has `model_text`.
The scheduler appends three fixed generic sentences and one of 16 deterministic closing variants without adding a second latent fact or changing event semantics.
This expansion exists to impose controlled token pressure and a repeated resumption texture.
It is not substantive dialogue and must not be interpreted as realistic conversation.

### Queries

Query text is generated without the derived gold answer.
The structured query retains fields needed by the deterministic resolver, including kind, subject, date, scope, candidate, or fact ID as applicable.
The prompt uses the natural-language `query_text`.
Gold is produced by replaying the relevant causal prefix through `resolve_query()`.
For contracted queries, that prefix ends at the declared trigger event so future events cannot leak into source gold.

### Candidate Updates

Candidate updates receive gold admission decisions, reason codes, hard-gate eligibility, and observable candidate features derived from the event history and candidate record.
The nine families are uncontested acceptance, low-evidence conflict rejection, high-evidence replacement, equal-evidence conflict rejection, hard-constraint rejection, tombstone-resurrection rejection, direct-user replacement with explicit supersession, direct-user conflict rejection without supersession, and direct-user ambiguity resolution.
Tests assert that candidate feature names contain no gold-derived fields.
Tests also assert that matched high-confidence and low-confidence conflicts differ in the intended evidence signal rather than in unrelated structural features.

## Temporal, Scope, Authority, and Lineage Semantics

The benchmark distinguishes valid time from observation time.
`valid_from` and optional `valid_to` determine when a preference applies, while `observed_at` records when an assertion enters the stream.
A backdated correction can therefore arrive later while governing an earlier interval.
The contradiction ledger treats intervals as closed at both ends and represents an open end as continuing through the comparison horizon.

Preference resolution filters to supported operations, subject identity, and the requested date.
It first selects an active exact scope and otherwise falls back to active `default` facts.
A temporary preference in a different scope cannot override the default preference.
The scope-leakage case deliberately places a lexically relevant but differently scoped value beside the correct default value.

When multiple active facts remain in the selected scope, the resolver ranks later `valid_from` first, then source authority, then confidence.
The current authority order is `inferred < direct_user`.
The generated authority conflict has a direct-user correction that explicitly supersedes the inferred source.
The contradiction ledger independently requires explicit resolution links and rejects a lower-authority assertion that attempts to resolve a higher-authority claim.

Retraction is event-sourced rather than destructive.
`materialize_event_states()` removes the named fact from the active snapshot but leaves the event trace intact.
Private-lineage gold treats `duplicate_of` edges as an undirected connected component and propagates a tombstone across that component.
The positive checkpoint returns the retained value before withdrawal.
The post-retraction checkpoint returns `UNKNOWN` after withdrawal of either the terminal copy or the root, depending on the held-out chain composition.

Ambiguity is represented by one pre-labeled `ambiguous_conflict` event whose generated text states that the account's statements conflict.
For a query with `kind=ambiguity`, `resolve_query()` returns `UNKNOWN` unconditionally rather than deriving ambiguity from two competing assertions.
This is a labeled abstention fixture, not ambiguity discovery.
A separate candidate family permits a direct-user correction that explicitly supersedes the ambiguity.

## Held-Out Compositions

History-disjoint splitting alone would not prevent generator templates from repeating the same behavioral composition across train and test.
The interleaved source therefore assigns each history a three-bit composition over alias direction, lineage retraction direction, and hard-negative text family.

Train and development use the four even-parity compositions.

| Composition ID | Alias direction | Chain variant | Negative family |
| --- | ---: | ---: | ---: |
| `alias0-chain0-negative0` | 0 | 0 | 0 |
| `alias0-chain1-negative1` | 0 | 1 | 1 |
| `alias1-chain0-negative1` | 1 | 0 | 1 |
| `alias1-chain1-negative0` | 1 | 1 | 0 |

Test uses the four disjoint odd-parity compositions.

| Composition ID | Alias direction | Chain variant | Negative family |
| --- | ---: | ---: | ---: |
| `alias0-chain0-negative1` | 0 | 0 | 1 |
| `alias0-chain1-negative0` | 0 | 1 | 0 |
| `alias1-chain0-negative0` | 1 | 0 | 0 |
| `alias1-chain1-negative1` | 1 | 1 | 1 |

Every primitive axis value appears outside test, but no three-way test composition appears in train or development.
These even/odd composition assignments are dataset properties intended for future learned or tuned systems.
Interleaved v2 consumes only the test split and trains or tunes nothing, so the held-out compositions do not affect its current percentages and do not demonstrate generalization.

## Alias Uniqueness and Canonical-ID Hiding

The regenerated dataset uses two deterministic natural aliases per history from a 32-by-32 prefix and suffix namespace.
A numeric cycle suffix is added after the first 1,024 base names, and the second alias appends `Guild` to the first.
At 1,200 histories, this produces exactly 2,400 distinct aliases.
Train, development, and test contain 176, 176, and 2,048 aliases respectively, with zero pairwise intersection.
This alias regeneration fixes the prior risk of collisions and cross-split alias reuse at full interleaved scale.
Alias disjointness is a dataset property for future learned or tuned systems.
Because v2 consumes only test and uses hidden history identity after suffix selection, split-disjoint aliases do not affect current v2 percentages or prove cross-alias or cross-split generalization.

Alias direction is one axis of the held-out composition.
One direction introduces the short alias in events and asks with the account alias, while the other direction reverses those roles.
Both lineage evidence contracts require an explicit alias-bridge event.
A retrieved value without the identity bridge can therefore be answer-sufficient but not provenance-complete.

Canonical account, history, example, event-field, and operation-field identifiers remain in hidden structured records because the resolver and audit trail require stable joins.
They are absent from model-visible event and query text.
The corpus is nevertheless not fully naturalized because opaque object labels such as `value-177-b` remain visible in many non-lineage statements and questions.

## Distractor and Stressor Taxonomy

The generated stream contains peer histories and within-history features designed to be semantically related to the target evidence, but v2 does not measure their semantic difficulty.
All peer turns consume positional suffix capacity before hidden-history filtering, while same-entity lexical negatives and alias bridges matter only as generated records and declared provenance groups.
There is no random-filler condition, no length-matched semantic-distractor condition, and no retriever ablation in v2.
The intended challenge for future retrievers must therefore be kept separate from the measured v2 effect of positional evidence loss.

### Peer Tasks

The dominant long-context interference comes from other complete account histories.
Every test account is a peer task with the same schema, relation vocabulary, temporal patterns, aliases, corrections, and query-relevant concepts.
No scheduled account is marked as the focal task.
A bounded suffix spends capacity on these peer turns before the evaluator filters retained turns to the queried history.
V2 does not compare this positional pressure with equal-length unrelated filler, so it does not show that peer-task semantics make retention harder.

### Same-Entity Lexical Hard Negatives

Each history contains three `same_entity_hard_negative` notes.
They mention the same account alias and quote the exact private-lineage value in contexts such as discussion, replacement consideration, travel labels, planning cards, meeting notes, reminders, or account settings.
They share entity and value tokens with gold lineage evidence but do not assert duplicate lineage and do not create `duplicate_of` edges.
They are intended to challenge exact-string and lexical-overlap retrieval in a future retrieval evaluation.
V2 performs no learned, sparse, or dense retrieval over them, so their presence does not demonstrate lexical disambiguation capability.

### Alias Bridge

The alias bridge states that two natural names identify the same account.
It is required evidence for a cross-alias query rather than decorative context.
Both private-lineage contracts score it as a separate event group.
This requirement measures whether the bridge event remains in the selected view, not whether v2 discovers or traverses the alias relation from text.

### Stale Transitions

Each history contains an initial preference followed by a non-overlapping current preference.
A candidate proposes the initial value again after the transition with low confidence, and the gold decision rejects that stale write.
Online evaluation separately records a stale-memory intrusion when an incorrect prediction equals a historically observed value for the queried subject and relevant scope.

### Scope Leakage

A temporary value in a different limited scope overlaps the default query date and uses the same subject and relation.
The correct default answer remains the current default preference.
Scope leakage is a source stressor but is not one of the four online checkpoint families scored by interleaved v2.

### Backdated Corrections

A correction observed in August assigns a different preference to March through June, and its query asks about May.
This separates arrival order from valid time and prevents recency from substituting for temporal semantics.
The final retention contract requires both the initial assertion and the later backdated correction.

### Duplicate Lineage

A private value is asserted once and delivered twice through an explicit two-edge lineage.
The positive query is triggered after the second copy.
The negative query is triggered after withdrawal of either the root or terminal copy according to the held-out chain variant.
Correct post-withdrawal state requires transitive connected-component semantics rather than recognition of repeated text alone.

### Authority Conflict

An inferred October preference is followed by a direct-user correction for the same subject, scope, relation, and validity start.
The direct assertion explicitly supersedes the inferred assertion.
The delayed authority contract requires both event and fact groups so the conflict and justified winner remain auditable.

### Ambiguity

One pre-labeled ambiguity event has text that explicitly says the account's statements conflict.
The base query returns `UNKNOWN` because the ambiguity branch of `resolve_query()` is unconditional.
The fixture tests labeled abstention bookkeeping, not inference from two competing assertions.
The candidate set separately tests whether a higher-authority update with explicit supersession can resolve the ambiguity.

### Explicit Contradiction Lifecycle

Each interleaved history contains two overlapping project-workspace preferences with different values and one later direct-user rectification that names both source fact IDs in `resolves`.
The two opening assertions and the rectification occupy distinct account segments, and the rectification cannot be stranded in the final account segment.
Other peer tasks intervene between lifecycle stages at scale.
Lifecycle family labels, fact IDs, and `resolves` IDs are supplied by the generator.
When explicit lifecycle labels are present, the ledger filters the history to these three generated events.

The contradiction key includes history, subject, predicate, domain, and scope.
Different objects form a potential contradiction only when their validity intervals overlap.
Different scopes are scope-separated rather than contradictory.
A rectification resolves a pair only through explicit links and an authority-compatible winner.

### Candidate-Only Admission Stressors

Several stressors occur only in `candidate_updates.jsonl` and are not scheduled transcript turns.
They include equal-evidence conflict, hard-constraint violation, tombstone resurrection, direct-user conflict without supersession, high-evidence replacement, and explicit ambiguity resolution.
These cases support admission experiments, but no interleaved v2 retention result is attributed to their presence in the stream.

## Distractor Rationale and Anti-Shortcut Controls

The data was designed with targeted interference that shares account aliases, relation terms, quoted values, temporal language, scopes, correction vocabulary, and lifecycle structure with relevant evidence.
Peer histories come from the same generator and occupy the same semantic family as the queried history.
Within a history, lexical negatives repeat the answer string without asserting the relation, scoped exceptions present plausible but inapplicable values, stale assertions reuse formerly correct values, duplicate deliveries repeat a true fact without adding state, and authority or contradiction controls introduce multiple plausible claims that require justification.
These are provenance contracts and generated data features, not demonstrated retrieval capabilities.
Because v2 has neither a random-filler baseline nor a length-matched semantic-distractor ablation, it cannot attribute any loss difference to semantic interference.

The generator and benchmark include the following explicit anti-shortcut controls.

| Intended shortcut risk | Generated feature or contract |
| --- | --- |
| Memorizing entity names across splits | History-disjoint and alias-disjoint train, development, and test splits |
| Memorizing complete behavioral templates | Even-parity train and development compositions and odd-parity test compositions |
| Routing by synthetic account IDs | Canonical IDs remain latent and are omitted from model-visible text |
| Returning a repeated answer string | Same-entity negatives quote the exact value without lineage semantics |
| Treating aliases as unrelated entities | Lineage contracts require the alias bridge |
| Equating arrival order with valid time | Backdated correction separates `observed_at` from validity |
| Letting another scope overwrite default state | Scope-specific resolution and scope-leakage cases |
| Treating duplicate delivery as a new state | Explicit `duplicate_of` lineage and idempotent duplicate controls |
| Ignoring retraction propagation | Root-versus-terminal held-out chain variants |
| Choosing any conflicting value | Explicit authority order, supersession links, and contradiction lifecycle |
| Receiving future evidence | Gold and method views stop at the same causal trigger |
| Receiving the gold during selection | Query prompts omit gold, and scoring fields are applied after selection |

These features can make contract failures more interpretable, but v2 does not establish that a retriever resists the listed shortcuts.
A lexical hit without the alias bridge indicates identity-evidence loss.
A correct answer without all declared groups indicates answer sufficiency without complete provenance.
A prior value returned after transition indicates stale intrusion.
A limited-scope value returned for default scope indicates scope leakage.
An unresolved or unjustifiably resolved contradiction indicates lifecycle failure.

## Interleaving Schedule and Resume-Gap Design

The scheduler is custom deterministic Python and uses no external scheduling library.
It groups events by history, partitions every 26-event history into segments of four through eight events, and preserves account-local event order.
Candidate segmentations are ranked by SHA-256 over seed 47, the history ID, and proposed lengths rather than process-randomized hashing.

The final schedule uses rolling admission with at most eight active accounts.
The scheduler avoids selecting the same account in adjacent segments.
While undispatched accounts remain, an active account can be forced after a bounded wait.
When one account finishes, a pending account is admitted.
During drain, histories with more remaining segments and longer waits are prioritized before the deterministic hash rank.

The final schedule has 5,246 segments, 4,222 resumptions, and 26,624 turns.
It has zero adjacent same-account segments and zero causal-order violations.
Every resumption has at least one intervening account, and the maximum observed number of distinct intervening accounts is 12.

The exact schedule measurements across evaluated horizons are:

| Accounts | Stream tokens | Segments | Turns | Resumptions | Minimum intervening accounts | Maximum intervening accounts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 21,712 | 43 | 208 | 35 | 7 | 7 |
| 16 | 43,396 | 84 | 416 | 68 | 1 | 11 |
| 32 | 86,840 | 167 | 832 | 135 | 1 | 11 |
| 64 | 173,873 | 328 | 1,664 | 264 | 1 | 11 |
| 128 | 349,260 | 655 | 3,328 | 527 | 1 | 12 |
| 256 | 694,696 | 1,306 | 6,656 | 1,050 | 1 | 12 |
| 512 | 1,376,177 | 2,611 | 13,312 | 2,099 | 1 | 11 |
| 1,024 | 2,738,857 | 5,246 | 26,624 | 4,222 | 1 | 12 |

At the 1,024-account horizon, all six state-changing lifecycle families occur in every stream third.
The first, middle, and final thirds contain 690, 678, and 680 non-overlap transition events; 343, 341, and 340 backdated corrections; 682, 686, and 680 source-authority conflict events; 688, 680, and 680 retraction events; 685, 683, and 680 contradiction openings; and 339, 344, and 341 contradiction rectifications.
The benchmark fails if any of these families misses a stream third at horizons of 32 accounts or more.

Each horizon is rebuilt from the lexicographically first test history IDs for that horizon rather than truncated from the final schedule.
Horizon membership is nested, but the resulting schedule is not guaranteed to be an identical transcript prefix of the larger horizon.

## Contradiction Lifecycle

The contradiction ledger replays assertion and resolution controls in causal order.
It opens a pair only for different objects under the same semantic key with overlapping validity intervals.
It records a stable pair ID, source fact IDs, open index, status, winning fact ID, and resolution index.
Higher authority alone does not silently resolve a pair.
Resolution requires an explicit `resolves` or qualifying `supersedes` relationship, and lower authority cannot defeat higher authority.

The final artifact contains one canonical explicit contradiction pair per test history.
All 1,024 pairs are resolved, all are marked `resolved_authority`, and there are zero unexpected unresolved pair IDs.
Each result comes from a deliberately constructed three-event symbolic fixture whose lifecycle labels, fact IDs, and `resolves` links are generator-provided and whose ledger input is filtered to those events.
The result validates rule execution and Python/Scallop parity on that fixture.
It does not validate contradiction discovery or extraction from unstructured text, and because every fixture supplies a resolving event, it does not evaluate naturally unresolved handling.

## Causal Gold and Evidence Contracts

Gold answers are derived rather than handwritten after evaluation.
For each contracted query, the generator locates its trigger event and invokes the resolver on exactly the account prefix ending at that event.
The interleaved evaluator locates the same trigger in the global schedule and limits every method to events observed through that turn.
No method receives future turns.

An online evidence contract contains alternative-ID groups rather than one flat list.
Complete provenance requires at least one selected member from every required event group and every required fact group.
Alternative IDs can represent semantically equivalent deliveries for one role.

| Online checkpoint | Trigger | Required event groups | Required fact groups |
| --- | --- | --- | --- |
| Positive private lineage | Second duplicate | Alias bridge, root add, copy 1, and copy 2 | Root, copy 1, and copy 2 |
| Post-retraction private lineage | Retraction | Positive groups plus retraction | Root, copy 1, and copy 2 |
| Delayed preference change | Later account-review probe | Initial add and transition | Initial and current facts |
| Delayed authority incongruity | Later account-review probe | Inferred source and direct correction | Inferred and direct facts |

The final transcript-suffix sweep uses four account-level retention contracts per history.
Those contracts are preference change, authority incongruity, contradiction rectification, and backdated correction.
They require event groups only because the final sweep asks whether source transcript events needed to reconstruct each relation remain in the suffix.

This distinction is essential.
Final retention is transcript-only event-contract retention at the end of a horizon stream.
Online checkpoint scoring is prompt-inclusive and tracks both event and fact provenance at query-specific causal triggers.
The two result families measure different contracts and must not be presented as interchangeable.

## Answer Sufficiency, Provenance, and Grounded Credit

Answer sufficiency is true when `resolve_query()` over selected events equals stored gold.
Answer sufficiency can occur when declared evidence is incomplete.
For example, an empty or incomplete state can produce `UNKNOWN`, and a current fact can produce the correct value even if an older transition source is absent.

Complete provenance requires every declared event and fact group to be represented in the selected view.
Evidence recall is the fraction of those groups hit.
Grounded answer credit requires both answer correctness and complete provenance.
This separation prevents a lucky answer, unsupported `UNKNOWN`, or semantically sufficient but incompletely documented answer from receiving fully grounded credit.

The evidence contract represents declared causal support rather than a minimal logical proof.
Some groups preserve identity, transition, conflict, leakage, or lineage context needed to audit why an answer is justified even when a smaller event subset could reproduce the scalar answer under the resolver.
Complete provenance is therefore deliberately more conservative than answer sufficiency.

## Token Accounting

All current counts use the locally cached `Qwen/Qwen3-4B` tokenizer at requested and resolved revision `1cfa9a7208912126459214e8b04321603b3df60c`.
`metrics.json` records `Qwen2TokenizerFast` from `transformers==4.57.6`.
Tokenization is local-only and uses `add_special_tokens=false`.

Turns are serialized with one newline separator.
For each turn, `token_count` is the standalone encoding length of turn text.
`serialized_token_count` is the exact increase when the newline and current turn are appended to the preceding transcript boundary.
Summing serialized increments reproduces the complete transcript count.
In the final schedule, turn counts range from 88 to 123 tokens, average 102.871732 tokens, and sum to 2,738,857 tokens.

A fresh audit with the same pinned tokenizer gives 547,273 standalone tokens when each turn's unexpanded base `model_text` is encoded independently.
Subtracting that base total from the 2,738,857-token stream leaves 2,191,584 tokens, or approximately 80.02% of the stream, attributable to the deterministic expansion scaffold.
The expansion provides controlled token pressure and repeated resumption texture rather than realistic or substantive dialogue.
V2 includes no random-filler or semantic-distractor ablation that isolates the scaffold's effect from distractor meaning.

The final retention sweep evaluates complete-turn transcript suffixes at 4,096, 16,384, 65,536, 131,072, 262,144, and 1,048,576 tokens.
It charges the first retained turn at its standalone token count and later turns at their serialized increments.
It does not reserve tokens for a query prompt because no query is appended to this final sweep.

Online windows use the same six budgets but include the exact prompt `\nQuestion: {query_text}\nAnswer:`.
The evaluator computes prompt tokens and the boundary increment after the final retained turn, then binary-searches for the earliest complete turn whose suffix plus prompt fits the budget.
It never slices through a turn.
If a prompt alone exceeds the window, evaluation fails instead of silently clipping it.

The final stream is exactly 2,738,857 tokens, or 2.6119775772094727 times the declared 1,048,576-token context limit.
The run requires a multiplier of at least 2.0 and fails if the stream is shorter than 2,097,152 tokens.

## Matched Fairness Controls

The v2 comparison is matched at every causal checkpoint.
Full structured memory receives all observed events for the queried history and no future event.
A sliding context receives the globally visible complete-turn suffix that fits its token budget, including peer-task turns.
After suffix selection, the resolver receives only retained turns belonging to the queried history.
Both views end at the same trigger turn and are scored with the same gold-free structured query and deterministic resolver.
This ordering means peer turns create positional token pressure, but hidden-history filtering prevents v2 from testing semantic distractor rejection during resolution.

No method sees gold during selection.
The serialized prompt contains only natural query text.
Hidden IDs are used after suffix selection for provenance scoring and account mapping rather than as model-visible routing labels.
This post-selection account filtering is an oracle evaluation convenience, so v2 does not measure end-to-end entity resolution or retrieval from raw text.

Full structured memory is a forced-perfect oracle storage control.
The benchmark raises an exception if this view is not grounded-correct at every checkpoint.
Its 100% result is therefore a source-gold and storage-scope sanity check by construction.
It is not a realistic memory system and does not demonstrate perfect ingestion, identity resolution, storage, retrieval, or reasoning in deployment.

Sliding context is also scored by the deterministic oracle resolver after its suffix is selected.
Its answer percentages are not language-model generation accuracy.
The fair v2 claim concerns whether structured evidence remains available inside a finite transcript suffix under the disclosed timing, serializer, and contracts.
It does not concern semantic distractor hardness, lexical retrieval, alias resolution, or train/test generalization.

## Scallop's Role

Scallop does not produce the interleaved v2 online answers.
The Python deterministic resolver creates source gold and scores full-memory and sliding-context views.
The v2 online percentages are therefore not Scallop answer-generation or retrieval results.

With `--scallop`, the benchmark sends each final account history to the fail-closed `/derive_contradiction_ledger` service.
For this dataset, the service filters each history to the generator-labeled two opening events and one rectification event, then invokes actual `scallopy` rules over generator-provided symbolic fields and links.
The client validates `engine=scallopy`, ledger version `contradiction_ledger.v1`, the pair list, and the unresolved-pair list.
The regenerated v2 artifact records `scallopy==0.2.4` for all 1,024 ledgers.
This validates rule execution and parity on constructed fixtures, not contradiction discovery, extraction, or naturally unresolved cases.

Other repository experiments use Scallop for recursive private-lineage closure and query-blind preference-source injection.
Those continual-v3 and causal-ablation results are separate experiments.
They are not outputs of `interleaved_continual_memory.v2` and are not evidence for the v2 percentages reported here.

## Reproducibility

Run the following commands from the repository root with the pinned tokenizer available in the local Hugging Face cache.

Generate the source dataset with the exact split counts and hardness profile:

```bash
python -m experiments.synthetic_temporal_preferences \
    --output-dir results/synthetic_temporal_preferences_1200_v5_interleaved \
    --history-count 1200 \
    --split-counts 88,88,1024 \
    --hardness-profile anti_shortcut_interleaved_v3
```

Start the fail-closed service from an environment with working `scallopy==0.2.4`:

```bash
python -m services.scallop_validator_service \
    --host 127.0.0.1 \
    --port 8765
```

Regenerate the interleaved v2 artifacts:

```bash
SCALLOP_VALIDATOR_URL=http://127.0.0.1:8765 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python -m experiments.interleaved_memory_benchmark \
    --dataset results/synthetic_temporal_preferences_1200_v5_interleaved \
    --output-dir results/interleaved_memory_benchmark_v2 \
    --scallop \
    --declared-context-limit 1048576 \
    --context-multiplier 2
```

Run the relevant tests:

```bash
python -m pytest -q \
    tests/test_synthetic_temporal_preferences.py \
    tests/test_interleaved_conversation.py \
    tests/test_interleaved_memory_benchmark.py \
    tests/test_contradiction_ledger.py
```

Verify current artifact hashes:

```bash
sha256sum \
    results/synthetic_temporal_preferences_1200_v5_interleaved/{examples,facts,candidate_updates,events,queries}.jsonl \
    results/interleaved_memory_benchmark_v2/{schedule,contradiction_ledger,suffix_loss_witnesses,online_checkpoint_predictions}.jsonl \
    results/interleaved_memory_benchmark_v2/{metrics.json,report.md,manifest.json}
```

### Dataset Hashes

| File | SHA-256 |
| --- | --- |
| `examples.jsonl` | `e61e66f58997d7c7415e13b83ae9046673d51f0d03359ccf97791f13ebab0941` |
| `facts.jsonl` | `5fabccd117ac9711348eda82d5a768e42a04020066fe79f954ba0e40f877128f` |
| `candidate_updates.jsonl` | `611f7fc6535343eab6554327093719b95a3cdf295491886245462f96cf7007b9` |
| `events.jsonl` | `eabda37fa9f16daa761257351fba31ab18d3091e79943cd800d6d874c0603e17` |
| `queries.jsonl` | `2fa1d96e6f501e925018ce6b2f2be5d923f93c50272d0c85656bf836be5cb16f` |

### Interleaved v2 Hashes

| File | SHA-256 |
| --- | --- |
| `schedule.jsonl` | `1e6b65ce1cd09a4270591f3f9eca5bd8d2016b15c887bc01ae34e7c40611e7f6` |
| `contradiction_ledger.jsonl` | `73470544920ba277843aa0e1393d168317ff6e40f4bdbdf0779753107aee82df` |
| `suffix_loss_witnesses.jsonl` | `b4aaf77126536261d34110a0aab786c2c59e697d45bdc9d631b0e3294cccb7ff` |
| `online_checkpoint_predictions.jsonl` | `6983d00930a43c2e299164979557e0bbded7f8c6f68a22da9527483424d37724` |
| `metrics.json` | `15078f3080312fed98571212b32725a4fd2773d9d0bfe6e98eea8fea715dbf0a` |
| `report.md` | `504ae0aea0407a9c79ae2f35cd46f5a33ad3b23de430b8b64141482ec793b238` |
| `manifest.json` | `16c7ce3bb5ede83bac9726403e6e65cd084a1d5a1c42f59277672a965d44aeb9` |

The manifest itself is not self-hashed because it is written after its listed artifacts.

## Exact Current v2 Results

### Final Transcript-Only Event-Contract Retention

The following table reports the percentage of four event-only retention contracts per account that are incomplete in a prompt-free transcript suffix taken at the end of each horizon stream.

| Accounts | 4K loss | 16K loss | 64K loss | 128K loss | 256K loss | 1M loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 100.00% | 71.88% | 0.00% | 0.00% | 0.00% | 0.00% |
| 16 | 100.00% | 79.69% | 0.00% | 0.00% | 0.00% | 0.00% |
| 32 | 100.00% | 87.50% | 33.59% | 0.00% | 0.00% | 0.00% |
| 64 | 100.00% | 94.53% | 64.84% | 29.69% | 0.00% | 0.00% |
| 128 | 100.00% | 97.07% | 83.79% | 64.45% | 26.95% | 0.00% |
| 256 | 100.00% | 98.44% | 91.60% | 82.32% | 62.99% | 0.00% |
| 512 | 100.00% | 99.27% | 95.51% | 90.92% | 81.54% | 24.22% |
| 1,024 | 100.00% | 99.63% | 97.90% | 95.43% | 90.70% | 61.89% |

At 1,024 accounts, the exact incomplete-contract counts are 4,096 of 4,096 at 4K, 4,081 of 4,096 at 16K, 4,010 of 4,096 at 64K, 3,909 of 4,096 at 128K, 3,715 of 4,096 at 256K, and 2,535 of 4,096 at 1M.
These measurements support the narrow conclusion that a fixed transcript suffix eventually loses old causal evidence as peer histories continue to evolve.
They do not show that semantic distractors are harder than length-matched filler because v2 includes no such comparison and removes non-target histories before resolution.
Same-entity lexical negatives and alias bridges remain generated features and provenance requirements in these results, not evidence of lexical retrieval or alias traversal.

### Online Prompt-Inclusive Event-and-Fact Provenance

Each method is evaluated at 4,096 immediate-trigger checkpoints.

| Method | Oracle answer sufficiency | Complete provenance | Grounded answer | Mean evidence recall | Stale intrusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full structured memory oracle control | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| Sliding 4K | 41.43% | 13.11% | 13.11% | 35.57% | 5.27% |
| Sliding 16K | 92.63% | 81.40% | 81.40% | 85.00% | 0.00% |
| Sliding 64K | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| Sliding 128K | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| Sliding 256K | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| Sliding 1M | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |

The exact 4K counts are 1,697 answer-sufficient checkpoints, 537 provenance-complete and grounded checkpoints, and 216 stale intrusions.
The exact 16K counts are 3,794 answer-sufficient checkpoints and 3,334 provenance-complete and grounded checkpoints.

### Immediate-Trigger Evidence-Age Limitation

Online checkpoints occur immediately after each query's trigger event.
Across the 4,096 full-memory checkpoint records, required evidence ages range from 11 to 402 global turns and from 1,110 to 41,255 tokenizer tokens.
The maximum evidence age is therefore below 65,536 tokens.

The maximum required-evidence ages by checkpoint family are 41,255 tokens for delayed preference change, 33,540 for delayed authority incongruity, 24,692 for post-retraction lineage, and 24,586 for positive lineage.
The perfect online 64K result does not demonstrate saturation of a 64K context window.
It shows only that every required evidence group at these immediate-trigger checkpoints falls within that budget under the current schedule.
The same limitation applies more strongly to 128K, 256K, and 1M online windows.
Using those 100% scores as evidence of useful behavior near their nominal capacity would be misleading.

The final global suffix sweep does expose substantial loss from 64K through 1M because it evaluates old relations at the end of the complete stream.
That sweep is a transcript-only retention-availability test without a query prompt, not an online query-answering result.

## Threats to Validity and Non-Claims

### Synthetic Template Bias

The data is deterministic synthetic text generated from fixed templates and deterministic expansions.
Base `model_text` accounts for 547,273 standalone tokens, while the full stream contains 2,738,857 tokens, leaving 2,191,584 tokens, or approximately 80.02%, attributable to repeated expansion scaffold.
The expansion is controlled token pressure and resumption texture, not realistic or substantive dialogue.
It does not reproduce human topic shifts, incomplete statements, coreference noise, extraction errors, corrections without explicit markers, or naturally varying discourse length.
Systems may exploit regular lexical or structural patterns that do not generalize to human conversations.

### Unablated Distractor Semantics

Suffix selection charges all peer turns positionally and hidden-history filtering then removes non-target histories before deterministic resolution.
Same-entity negatives and alias bridges are provenance-contract features rather than evaluated retrieval operations.
Without random-filler, length-matched semantic-distractor, or retriever ablations, v2 cannot show a semantic-distractor penalty or lexical and alias retrieval capability.

### Unused Training Splits

Even-parity train and development compositions, odd-parity test compositions, and split-disjoint aliases prepare the dataset for future learned or tuned systems.
V2 trains nothing and reads only test, so these split properties neither affect current percentages nor establish generalization.

### Oracle Latent Structure

The evaluator has perfect latent history identity, event boundaries, fact structure, operation labels, and checkpoint contracts.
Visible text hides canonical IDs, but evaluation maps retained turns to the queried history after suffix selection.
The benchmark isolates storage and context availability rather than end-to-end text ingestion, entity resolution, retrieval, and reasoning.

### Forced-Perfect Structured Control

Full structured memory is a forced-perfect oracle storage control, not a realistic system.
The run aborts if this control is not grounded-correct at every checkpoint.
Its 100% score validates source gold, account-local storage scope, and resolver consistency under ideal retention.
It provides no evidence for production ingestion, bounded storage, imperfect retrieval, entity linking, or learned reasoning performance.

### Shared Gold and Answer Resolver

The same Python semantic resolver creates gold and scores selected events.
This controls semantic drift but does not establish independent reasoning accuracy.
The benchmark can reveal missing context under its declared semantics, not whether another implementation would interpret the natural language correctly.

### Conservative Provenance Contracts

Complete provenance requires declared event and fact groups, some of which exceed the minimal subset needed to reproduce a scalar answer.
This choice favors auditability but makes grounded accuracy partly a contract-coverage metric.
Semantically valid evidence outside declared alternatives receives no credit.

### Immediate-Trigger Ceiling

The current online evidence-age maximum is 41,255 tokens, or approximately 41K.
The design does not make 64K or larger online windows binding, so 64K saturation is not a meaningful claim for v2.

### Final-Sweep and Online Metric Mismatch

The final sweep omits a query prompt and evaluates event-contract retention at stream end.
Online windows include the query prompt and evaluate both event and fact provenance immediately after four trigger families.
The result families are complementary but not directly comparable.

### Single Tokenizer and Serializer

All token results are specific to one pinned Qwen3-4B tokenizer revision, no special tokens, newline turn separation, and complete-turn suffixes.
Other tokenizers, chat templates, special-token policies, or truncation rules can change effective windows and loss curves.

### Deterministic Selection and Scheduling

Horizon experiments use the lexicographically first test histories and one scheduler seed.
There is no sampling-variance estimate, alternate-seed interval, or confidence interval.
Determinism strengthens exact reproduction but does not establish robustness across data draws or schedules.

### Limited Semantic Ontology

The authority lattice contains only `inferred` and `direct_user`.
Temporal values are ISO dates with closed intervals, and preference resolution uses a fixed latest-start, authority, and confidence ranking.
These semantics are transparent but narrower than real-world uncertainty, source trust, and event-time models.

### Scallop Claim Boundary

Scallop executes contradiction-ledger rules on one deliberately constructed three-event fixture per test history.
The generator supplies lifecycle labels, symbolic fact IDs, and resolution links, and the ledger filters to those labeled events.
The result validates rule execution and parity, not contradiction discovery, extraction, or naturally unresolved handling.
It does not produce the online answer percentages.
Recursive lineage and query-blind injection results elsewhere in the repository belong to separate experiments and are not evidence for interleaved v2.

### Artifact Completeness

`facts.jsonl` is intentionally selective and is not exhaustive.
The event trace and candidate artifact are required for their respective semantic analyses.

### No External-System Comparison

Interleaved v2 contains no measured LLM, Mem0, Letta, vector database, BM25, dense retriever, or deployed memory system.
It supports no claim that NeuroSym outperforms those systems.
It also contains no latency, throughput, storage-cost, or energy comparison.

### No Human, Privacy, or Safety Claim

The benchmark does not measure user satisfaction, privacy compliance, real deletion guarantees, policy safety, or harm from stale personalization.
Synthetic private values and constraints are semantic test cases rather than a privacy or safety evaluation.

## Future Work: Planned v3

The following items are planned future work and are not current v2 results or capabilities.

Planned v3 will add delayed global query sweeps after substantial additional peer-task activity so required online evidence ages deliberately cross 64K, 128K, and larger thresholds.
Queries will be issued after their underlying transition, conflict, retraction, or rectification has aged in the global stream rather than immediately after a nearby probe.
The evaluation should report realized age distributions and fail loudly when a claimed window has no checkpoint whose required evidence exceeds that capacity.

Planned v3 will also add imperfect query-blind structured retrieval.
Ingestion and stored representations will be created without access to the future query or hidden evidence contract.
At query time, retrieval may use only visible query text and previously materialized structured memory, with bounded capacity and explicit misses.
Full structured memory should remain only as the oracle upper control, while imperfect retrieval becomes the realistic structured method.
Random-filler and length-matched semantic-distractor conditions should isolate positional token pressure from semantic retrieval difficulty.
Those retrieval evaluations should separately test same-entity lexical negatives and alias-bridge traversal instead of inferring capability from provenance contracts.

Delayed global sweeps and imperfect query-blind retrieval address different limitations.
Delayed sweeps make large windows binding.
Imperfect retrieval tests whether a practical structured system can locate account-local evidence without oracle storage selection.
Neither capability exists in the current interleaved v2 result, and neither is implied by its forced-perfect control.

## Interpretation

The defensible v2 conclusion is methodological rather than universal.
The generator creates deterministic, history-disjoint, alias-disjoint, composition-held-out peer histories with lexical-negative and alias-bridge features, but these are dataset properties for future retrievers rather than demonstrated v2 capabilities or generalization results.
The scheduler creates a 2,738,857-token causal stream while preserving account-local order and distributing lifecycle events through the stream, with 2,191,584 tokens, or approximately 80.02%, attributable to deterministic expansion scaffold.
At the end of that stream, fixed transcript suffixes lose an increasing share of old causal evidence, including 61.89% of declared event contracts at 1,048,576 tokens.
Because suffix selection uses peer turns as positional pressure and hidden-history filtering precedes resolution, v2 does not show that semantic distractors are harder than length-matched filler or that lexical negatives and alias bridges are successfully retrieved.
At immediate-trigger checkpoints, 4K and 16K suffixes lose evidence, while 64K and larger windows are non-binding because the oldest required evidence is only 41,255 tokens old.
Full structured memory is a forced-perfect oracle storage control, and Scallop's current v2 role is rule execution and parity checking on generator-labeled three-event contradiction fixtures rather than contradiction discovery or online answer generation.
These boundaries are part of the result.
