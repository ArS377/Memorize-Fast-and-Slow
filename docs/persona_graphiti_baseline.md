# Graphiti external baseline for the persona end-to-end benchmark

Graphiti (getzep/graphiti; the Zep paper, arXiv 2501.13956) is the write-time
inferred-revision rival to our declared-at-write symbolic admission: it is the
same construct - a bi-temporal KG with edge invalidation - minus the declared
lifecycle-link replay and the Scallop gate. Its LLM decides at ingestion which
existing edges a new episode contradicts, and on a temporally overlapping
contradiction the older edge is invalidated (newest-wins). Retrieval is
semantic + BM25 fused with RRF and uses no LLM at query time (verified against
graphiti-core v0.29.3: the default `search()` recipe is `EDGE_HYBRID_SEARCH_RRF`
and the cross-encoder is never invoked by it; our adapter additionally installs
a cross-encoder stub that raises if any code path requests reranking).

The design space this completes: deferred-to-read (Mem0) vs inferred-at-write
(Graphiti) vs declared-at-write (ours), on identical conditions with matched
budgets.

## Role in the paper

Graphiti serves two sections, and it is worth being precise about which claim
it carries in each, because they are different strengths of claim:

- **Related work (capability matrix).** The matrix asserts that no existing
  agent-memory system supports typed, auditable revision over the four
  operations. Graphiti is the strongest counter-candidate - it is the one
  system with genuine write-time invalidation - so a *measured* row for it is
  much stronger than a row read off documentation, and it forecloses the
  obvious reviewer question ("did you actually try the closest system?").
- **Results ladder (external anchor, not the rung itself).** The ladder's
  third rung is "LLM-inferred revision". Graphiti should NOT be that rung:
  it differs from our system in extraction, entity resolution, graph schema,
  and retrieval, not only in the admission rule, so a rung-3-to-rung-4 gap
  measured against Graphiti is confounded four ways and cannot isolate the
  mechanism. The rung itself should be an internal ablation - our pipeline
  with `validate()` swapped for an LLM judge over the same candidate facts,
  one variable changed. Graphiti then does the job an internal ablation
  cannot: it shows the internal LLM-judge rung is not a strawman. If a real,
  engineered, well-funded system lands near our own LLM-judge ablation, the
  "you built a weak baseline" objection is dead.

Read the (A)-(B) variant delta below the same way: it prices extraction, which
is the confound that separates the honest off-the-shelf number from the
mechanism comparison.

## Pre-registered predictions (recorded before any run)

Graphiti should do WELL on plain preference change, and plausibly on backdated
corrections (in-text dates resolve into edge `valid_at`). It should FAIL,
attributably:

1. **Stale re-assertion** - a delayed restatement of an old value is "new
   information", so newest-wins invalidates the CURRENT correct fact. The
   corpus plants exactly this (`duplicate_delivery`, `contradiction_*`
   families).
2. **Scope exceptions** - a scoped value is semantically related and
   temporally overlapping with the default, so the extraction LLM may
   invalidate the default edge, baking scope leakage into the store
   (`scope_exception`, `scope_leakage` families).
3. **Lineage retraction** - "please forget X" extracts no contradicting edge;
   nothing propagates across duplicate deliveries (`retraction`,
   `private_lineage` families).
4. **Authority** - Graphiti may get authority conflicts right *by
   coincidence*: the direct correction arrives later in our streams, so
   newest-wins agrees with authority without modeling it
   (`source_authority_conflict` family; the pre/post-update phases separate
   the mechanisms).

Report per query family and per phase, not just overall, and count
"stale-intrusion" errors (answer equals a superseded value) for comparison
with the v3 stale-intrusion framing.

## Variants

- **`e2e`** (run first): ingest each turn's rendered dialogue exactly as the
  other arms see it (prefixed with the same account-holder label). Honest
  off-the-shelf number; confounds extraction with revision.
- **`matched`**: ingest each event's surface-mapped fact fields as structured
  JSON (subject, predicate, value, dates, scope, authority). This is
  **structured-input extraction, not an extraction bypass**: `add_episode`
  still runs graphiti's node and edge extractors over the JSON body, so the
  extractor remains in the loop and can still mis-resolve entities or dates.
  What it removes is the burden of parsing free dialogue, which makes the
  input clean and the revision judgments easier to attribute - it does not
  isolate revision from extraction outright. The e2e-minus-matched delta prices
  extraction. Latent scope qualifiers (e.g. `scope-001`) are mapped to stable
  opaque tokens because our admission operator also treats scope as an opaque
  symbol; lifecycle link fields (supersedes/corrects/resolves/duplicate_of/
  transitions_from/retracts targets) are NEVER included - inferring them is
  precisely what is under test.

## Metrics

**State fidelity (primary).** `extract_valid_fact_set` reads the store rather
than a generation: an edge counts as currently valid when neither its
world-time invalidation (`invalid_at`) nor its expiry (`expired_at`) is set.
Note that in graphiti these are not independent: it stamps `expired_at = now`
on any edge that acquires an `invalid_at`, so `expired_at` is a derived flag
rather than a separate transaction-time signal. That is harmless for a
*current-state* set, but it is why point-in-time retrieval must filter on
world time only - see the retrieval-state section. Episodic `MENTIONS` edges are not facts and are
excluded. The per-pair valid-fact set is written to
`graphiti_store_states.json` in the output directory, and the manifest carries
its hash and the total fact count.

One caveat that must be settled before the primary number is reportable:
these rows are in **Graphiti's vocabulary**, with LLM-chosen entity names and
relation names and a free-text `fact` string. Computing F1 against the
corpus's canonical fact space requires an alignment procedure, and that choice
is itself a result-bearing decision. Three options, in increasing order of
strength and cost:

1. Surface-form matching of `subject`/`object` against the corpus's surface
   vocabulary (cheap, brittle on paraphrase, will understate Graphiti).
2. Embedding-nearest-match with a reported threshold and a manually audited
   sample (defensible; report the audit).
3. Constrain extraction with graphiti's `entity_types` / `edge_types` /
   `edge_type_map` on `add_episode` so Graphiti writes into our schema
   directly (strongest alignment, and defensible for the `matched` variant on
   the same logic that justifies it - same target schema, isolate the
   revision policy). Not wired up here because it cannot be verified without
   a live extraction LLM; do this first when the GPU environment is up.

Do not report a state-fidelity number for Graphiti until one of these is
chosen and disclosed. Understating a rival through a lazy string match is the
failure mode to avoid.

**Downstream utility (secondary).** The generation arms below, scored EM/F1
with the paired history-clustered bootstrap, unchanged.

## What this arm is, precisely

Call it **"Graphiti ingestion with benchmark-matched retrieval"**, not
"off-the-shelf Graphiti". Four deliberate deviations from stock defaults, all
recorded in `manifest.graphiti_memory.deviations_from_default`:

1. **Embedder** replaced with the benchmark's pinned bge revision, so the dense
   branch matches every other arm. This is the point of a matched comparison.
2. **Cross-encoder** replaced with a stub that raises. The default RRF search
   recipe never invokes it, so this changes no result - it only stops the
   constructor's default `OpenAIRerankerClient` from making a remote call
   during an otherwise local run.
3. **Search applies an explicit bi-temporal filter** (below). Graphiti's
   default applies none.
4. **Neo4j driver subclassed** to route correctly (below).
5. **Extraction output clamped** at the client boundary. graphiti's extractors
   pass their own `max_tokens`, and the base client resolves the budget as
   `max_tokens or self.max_tokens`, so `LLMConfig.max_tokens` alone never
   takes effect; without the clamp the manifest would record a limit that did
   not run, and a 16K-context endpoint would reject every edge-extraction call.
6. **Query embeddings are encoded as documents.** graphiti's `EmbedderClient`
   interface has one `create` path and cannot distinguish a query from a
   document, so queries do not receive the retrieval instruction and query
   encoding our own dense branch applies. The embedding *model and revision*
   match; the query-side encoding does not.

Only (2) is a no-op. (1) is what makes the arm comparable, (3) is what makes
the question answerable, and (4) is a correctness fix. If you want a genuinely
stock number, add a separate arm that keeps graphiti's defaults end to end and
report it alongside - do not relabel this one.

## Neo4j database routing (upstream defect, worked around)

graphiti-core 0.29.3's `Neo4jDriver.execute_query` puts `database_` inside the
Cypher **parameters** map instead of passing it as the neo4j driver's control
argument. The consequence is a split:

- writes (`node.save`), search, and our snapshot go through `execute_query`
  and land on the **connection's default database**;
- `clear_data`, which teardown uses, goes through `session(database=...)` and
  hits the **configured database**.

Unpatched, that means namespaces are never actually cleared, stale edges from
a previous build survive into the next one, and the "Graphiti gets its own
database" isolation claim is simply false - the data may be sitting in the
same default database the hybrid KG arm uses. Every number would be quietly
contaminated.

`_routed_neo4j_driver` subclasses the driver to pass `database_` through
kwargs, which the base implementation forwards to the neo4j client as the real
control argument. On top of that, `_verify_database_routing` runs once per
session before any episode: it writes a marker through the `execute_query`
path and reads it back through the `session` path, and aborts the run if the
two disagree. That canary is version-independent, so if a future graphiti
release changes this behavior again the run fails loudly instead of silently
writing to the wrong database.

## Retrieval state: a confound worth knowing about

Verified against graphiti-core 0.29.3: `SearchFilters` defaults every temporal
field to `None`, so a **default `search()` returns invalidated edges alongside
current ones**. (This qualifies the common description that invalidated edges
are "excluded from current-state queries" - exclusion is opt-in, not the
default.) Left unfiltered, the generation arm would measure "Graphiti's store
plus a Qwen adjudicating validity at read time", which confounds precisely the
write-time claim the arm exists to test.

The obvious fix - keep only edges that were never invalidated - is **wrong for
this benchmark**, and wrong in the direction that would manufacture a finding.
The delayed probes are point-in-time questions ("what standing preference
applied to AsterArc **on 2025-08-01**?"), and the `preference_change` gold is
a *closed-interval* fact: mint tea, valid 2025-07-01 through 2025-09-30. A
Graphiti that extracts that interval perfectly sets `invalid_at = 2025-09-30`,
so a "not invalidated" filter deletes the correct answer from the candidate
set. The arm would have scored near zero on its strongest family, and the
failure would have looked like a result about write-time revision instead of a
bug in our filter.

So the filter separates world time from transaction time.
`graphiti.retrieval_state` selects:

- `point_in_time` (default): world time
  `(valid_at IS NULL OR valid_at <= D) AND (invalid_at IS NULL OR invalid_at > D)`
  where `D` is the date parsed from the visible query text, plus transaction
  time `expired_at IS NULL` for "only what the store still believes". Edges
  whose dates Graphiti failed to resolve are **kept**, so extraction gaps
  surface as wrong answers rather than silently shrinking the rival's
  candidate set.
- `current_only`: `invalid_at IS NULL AND expired_at IS NULL`. Correct only
  for present-tense questions; it will discard closed-interval gold on this
  corpus. Retained for a current-state variant, not for the delayed probes.
- `unfiltered`: graphiti's out-of-the-box behavior, where the generator
  adjudicates validity from the rendered windows.

The as-of date comes from `query_text`, which the memory system is allowed to
see, so nothing leaks. A query without exactly one date raises rather than
falling back to a current-state filter, because that fallback is precisely the
silent failure described above. The chosen state and the resolved `as_of` are
recorded in the manifest and on every retrieval row.

## Protocol invariants (enforced in code, not by convention)

- Causality: only turns with stream index < `checkpoint_turn_index` are
  ingested (hard assertion in `_causal_account_prefix`).
- Account scoping: only the queried account's turns are ingested, matching
  the hybrid arm's acknowledged oracle filter.
- Query blindness: ingestion sees no query text, gold, family, or evidence
  contract; retrieval sees `query_text` only.
- Matched budget: the arm reuses the hybrid two-phase prompt fitting
  (facts-first, then a maximal recent raw-conversation suffix) under the same
  4096/16384 caps, same instruction, same decoding.
- Isolation: one `group_id` per account (`{build_id}-{history_id}`), cleared
  before ingestion and torn down afterwards; Graphiti gets its OWN Neo4j
  database - config loading refuses a graphiti section that reuses the hybrid
  arm's (uri, database) pair.
- Reproducibility: graphiti-core is version-pinned and re-checked at runtime;
  the manifest records the timestamp mapping, per-pair canonical
  SHA-256 of the store (nodes + edges incl. created/expired/valid/invalid
  timestamps; embedding vectors excluded), retrieval map hash, store-state
  hash, retrieval state, and episode failure count. Run 3 builds (different `PERSONA_GRAPHITI_BUILD_ID`, separate
  output dirs) and report per-arm EM variance - ingestion is
  LLM-nondeterministic.
- Never compare against Graphiti's published LoCoMo numbers; only in-house
  matched arms are comparable.

## Known gaps (open decisions, not defects)

**Only two query families are probed.** The schedule selects
`-preference-change-delayed` and `-preference-incongruity-delayed`, so the
predictions above about scope exceptions, lineage retraction, and backdated
correction are tested only *indirectly*: those event families are present in
the stream as interference, and a scope-leakage or failed-retraction error
corrupts the store that the two delayed probes read. That is enough to make
the failures show up, but **not** enough to attribute them - a wrong
`preference_change` answer cannot be pinned on scope leakage specifically.

Direct attribution needs additional query suffixes (the corpus carries
`-current`, `-scope`, and related probes). The cost is not in this arm: the
condition count and every matched arm's published numbers assume these two
families, so adding families means re-running sliding, structured, and hybrid
too. That is a benchmark-design decision, not a Graphiti change, and it should
be made deliberately rather than folded into this baseline. Until then, state
the per-family predictions as motivated expectations and report only the two
families actually measured.

**State fidelity is not yet computable.** `graphiti_store_states.json` exports
the valid-fact sets, but the alignment from Graphiti's vocabulary to the
corpus's canonical fact space is unimplemented and unchosen, so `metrics.json`
carries generation EM/F1 only. See the metrics section: pick an alignment,
disclose it, then report. Do not let the absence of the primary metric turn
the secondary one into the headline.

## Ingestion shape and cost

An account's 10 checkpoints are **nested prefixes** of its stream (verified on
the corpus: strictly monotone). Rebuilding a fresh store per (history,
checkpoint) pair therefore re-ingests shared early turns ten times over. The
baseline instead ingests each account stream **once**, in stream order, pausing
at each checkpoint to snapshot the store and run that checkpoint's queries.
Ingestion stops short of the checkpoint index, which is the causality invariant
expressed as the loop condition rather than as a separate assertion.

Measured on the v1 corpus (120 conditions, 12 accounts, 10 checkpoints each):

| | episodes |
|---|---|
| fresh rebuild per (history, checkpoint) | 1,638 |
| incremental per account | **312** |

a 5.25x reduction, with all 120 conditions covered and every checkpoint seeing
a byte-identical causal turn set to what a fresh rebuild would have ingested
(both properties are asserted in the test suite). At roughly ten seconds per
episode on one GPU this is about an hour per build rather than four and a half.

Accounts are independent namespaces, so `graphiti.max_concurrent_histories`
runs several concurrently; this does not change any account's content, only how
well it saturates the extraction endpoint's batch. The failure budget is shared
across accounts and trips globally, so one account exhausting it unwinds its
siblings promptly instead of letting them run for hours - every namespace is
still torn down on the way out.

**The tradeoff to disclose.** Checkpoints of one account are no longer
independently sampled: a bad extraction at turn 4 now contaminates all ten of
that account's checkpoints rather than being re-rolled ten times. This is more
faithful to a deployed system, which has exactly one store accumulating its own
mistakes, and it makes the three-build variance report more informative because
each build resamples whole accounts rather than averaging away per-checkpoint
noise. But it does change what the error bars mean, and it belongs in the
writeup. `manifest.graphiti_memory.ingestion_mode` records it as
`incremental_per_account`.

## Timestamp mapping

The corpus has no wall clock, but the dialogue text carries real dates. Stream
order maps onto a synthetic ingestion clock: turn `i` (global stream index)
gets `reference_time = 2024-01-01T00:00:00Z + i * 60s`. In-text dates are left
untouched; Graphiti resolves them into edge validity windows itself. The
mapping is recorded in the manifest.

**Episode-time policy, and what it costs.** Two properties are in tension.
Placing T0 after the corpus window makes every reference-time fallback
ineligible for historical queries (the original failure below). Placing it
before makes every fallback *always* eligible - including a present-tense 2027
assertion that carries no explicit date, which can then satisfy a 2025 query.
We take the second: an over-permissive candidate set surfaces wrong facts for
the generator to reject, whereas the first silently empties the store and
manufactures a near-zero score. The consequence is that undated extractions
are not temporally discriminated at all, so this arm cannot support a
fine-grained temporal-precision claim; it supports downstream utility on the
probed families.

**T0 must precede the corpus's own dates.** The first serrano smoke test used
T0 = 2026-01-01 and 7 of 10 conditions retrieved zero facts. The extractor did
not resolve the in-text 2025 dates and fell back to `reference_time`, so every
edge carried a 2026 `valid_at` while the probes ask as-of 2025-08-01 and
2025-10-15 — the point-in-time predicate `valid_at <= as_of` then excluded
essentially the whole store. Placing the clock before the corpus window makes
that fallback benign: reference-time-derived facts satisfy `valid_at <= as_of`,
while an edge Graphiti actually invalidates receives an `invalid_at` from the
superseding edge and is still correctly excluded. The filter then measures
revision behavior rather than the arbitrary placement of our clock.

## Running

```
pip install -r requirements-graphiti-baseline.txt   # pins graphiti-core==0.29.3

export GRAPHITI_TELEMETRY_ENABLED=false   # disable graphiti's posthog telemetry
export PERSONA_DATASET_DIR=results/persona_conflict_conversations_v1
export PERSONA_BENCHMARK_OUTPUT_DIR=results/persona_end_to_end_graphiti_b1
export PERSONA_QWEN_MODEL_PATH=... PERSONA_QWEN_MODEL_ID=... PERSONA_QWEN_DEVICE=cuda
export PERSONA_SCALLOP_ENDPOINT=...
# hybrid arm (system Neo4j + BGE index), as for the neurosym config
export PERSONA_RETRIEVAL_INDEX_ROOT=... PERSONA_EMBEDDING_MODEL_ID=BAAI/bge-small-en-v1.5
export PERSONA_EMBEDDING_MODEL_PATH=... PERSONA_EMBEDDING_REVISION=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
export PERSONA_EMBEDDING_DEVICE=... PERSONA_NEO4J_URI=... PERSONA_NEO4J_USER=... \
       PERSONA_NEO4J_PASSWORD=... PERSONA_NEO4J_DATABASE=...
# graphiti arm: its OWN database, an OpenAI-compatible extraction endpoint
export PERSONA_GRAPHITI_BUILD_ID=build1
export PERSONA_GRAPHITI_NEO4J_URI=... PERSONA_GRAPHITI_NEO4J_USER=... \
       PERSONA_GRAPHITI_NEO4J_PASSWORD=... PERSONA_GRAPHITI_NEO4J_DATABASE=graphiti
export PERSONA_GRAPHITI_LLM_BASE_URL=http://localhost:8000/v1   # vLLM w/ guided decoding
export PERSONA_GRAPHITI_LLM_API_KEY=local PERSONA_GRAPHITI_LLM_MODEL=Qwen3.5-4B \
       PERSONA_GRAPHITI_LLM_SMALL_MODEL=Qwen3.5-4B

python -m experiments.persona_end_to_end_benchmark \
    --config configs/persona_end_to_end_joint_surface_a.json
```

## Smoke-test the extractor before any full build

Every ingestion phase asks the LLM for JSON matching a pydantic model and
constructs it strictly (`ExtractedEdges(**llm_response).edges` - no tolerant
`.get()` anywhere), and the edge schema demands `valid_at` / `invalid_at` as
resolved ISO strings. A 4B extractor can fail three different ways, and they
are NOT equally visible:

1. **Malformed JSON** - raises `JSONDecodeError`, which graphiti-core *does*
   retry (4 attempts, exponential backoff). Mostly self-healing, at 4x latency
   on those calls.
2. **Valid JSON, wrong shape** - missing key, wrong type. Raises
   `pydantic.ValidationError`, which is NOT in graphiti's retry predicate, so
   it propagates out of `add_episode`. This is the loud one, and the one
   `max_episode_failures` (0 by default) catches.
3. **Valid JSON, right shape, garbage content** - a well-formed `edges: []`, a
   plausible-but-wrong `valid_at`, an entity named "the user". **No error at
   all.** The build completes, the manifest reports zero failures, and the
   store is quietly empty or wrong.

Case 3 is the one that threatens the experiment and the failure budget is
blind to it. So the smoke test must **inspect the store**, not just check that
nothing raised: run one account, then read
`graphiti_store_states.json` and confirm the valid-fact counts and `valid_at`
values are sane. A silently under-extracting Graphiti would look like a
devastating baseline result and be an artifact of the extractor. If the local
model is unusable, fall back to a pinned API model and disclose it - the
extractor is `PERSONA_GRAPHITI_LLM_MODEL`, recorded at
`manifest.graphiti_memory.llm_model`.

Run externals on the v1 parent corpus first (the pair-conditioned surfaces
exist only on main); the schedule section of
`configs/persona_end_to_end_joint_surface_a.json` matches the neurosym config, so
conditions are identical across arms and the paired history-clustered
bootstrap applies unchanged.
