# Graphiti External Baseline: Experimental Protocol

Step-by-step run procedure for the Graphiti arms, in the same shape as the
persona NeuroSym preflight and end-to-end benchmark. Design rationale,
pre-registered predictions, and the mechanism discussion live in
`persona_graphiti_baseline.md`; this document is the operational runbook.

Every step is gated: do not advance until the stated check passes.

---

## Step 0 — Environment prerequisites

| Requirement | Why | Check |
|---|---|---|
| Neo4j instance **separate from the hybrid arm's** | graphiti writes and clears its own namespaces; sharing invalidates isolation | `SHOW DATABASES` lists the configured database, or a second instance answers on its own port |
| OpenAI-compatible extraction endpoint | graphiti's extraction, dedup, and contradiction calls | `curl $PERSONA_GRAPHITI_LLM_BASE_URL/models` |
| `BAAI/bge-small-en-v1.5` at the pinned revision, local | matches every other arm's dense branch | present in the HF cache |
| Qwen generation model + tokenizer, local | schedule reconstruction and generation | present in the HF cache |
| `graphiti-core==0.29.3` | pinned; the adapter refuses a mismatch at runtime | `pip show graphiti-core` |

**Neo4j edition matters.** Community edition supports exactly one user
database, so a single Community instance *cannot* give Graphiti its own
database alongside the hybrid arm. Either run a second Neo4j instance on its
own port, or run the Graphiti arms on a host where the hybrid stack is absent.
Config loading refuses a graphiti section whose `(uri, database)` matches the
hybrid arm's, and `_verify_database_routing` aborts at runtime if writes and
teardown resolve to different databases.

```bash
export GRAPHITI_TELEMETRY_ENABLED=false
export PERSONA_DATASET_DIR=results/persona_conflict_conversations_v1
export PERSONA_QWEN_MODEL_PATH=... PERSONA_QWEN_MODEL_ID=... PERSONA_QWEN_DEVICE=cuda
export PERSONA_SCALLOP_ENDPOINT=http://127.0.0.1:8081   # REQUIRED for Steps 3-5, see below
export PERSONA_EMBEDDING_MODEL_ID=BAAI/bge-small-en-v1.5
export PERSONA_EMBEDDING_MODEL_PATH=...
export PERSONA_EMBEDDING_REVISION=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
export PERSONA_EMBEDDING_DEVICE=cuda
export PERSONA_GRAPHITI_NEO4J_URI=... PERSONA_GRAPHITI_NEO4J_USER=... \
       PERSONA_GRAPHITI_NEO4J_PASSWORD=... PERSONA_GRAPHITI_NEO4J_DATABASE=graphiti
export PERSONA_GRAPHITI_LLM_BASE_URL=http://localhost:8000/v1
export PERSONA_GRAPHITI_LLM_API_KEY=local
export PERSONA_GRAPHITI_LLM_MODEL=... PERSONA_GRAPHITI_LLM_SMALL_MODEL=...
```

**The Scallop validator is required from Step 3 onward.** The preflight
(Steps 1-2) never calls it, but `run_benchmark` runs the Scallop semantic
canary and derives per-condition injections *unconditionally*, before any arm
is built, and the `structured_memory` arms consume those injections. A fake
endpoint is fine for the preflight and will fail the benchmark.

Launch it from an environment that has `scallopy` (the repo's Python 3.10-ish
Scallop env, not the graphiti venv):

```bash
tmux new-session -d -s scallop \
  "<scallop-python> -m services.scallop_validator_service --port 8081"
curl -s http://127.0.0.1:8081/ | head -c 200   # expects engine=scallopy + version
```

Running all seven arms in one benchmark invocation is preferable to comparing
against the previously published NeuroSym numbers, because the schedule's
token-distance checkpoints depend on the tokenizer: a run using a different
Qwen revision than the published one produces different conditions and is not
directly comparable.

---

## Unattended driver

`scripts/run_graphiti_baseline.sh` runs Steps 1-6 with the gates below
enforced between stages, so a bad extractor or a missing service fails in
minutes instead of after hours of ingestion. It checks all three services
first (extraction endpoint, graphiti Neo4j, Scallop validator), and if Scallop
is down it still completes both preflight stages and stops cleanly before the
scored builds.

```bash
source graphiti_env.sh
tmux new-session -d -s graphiti \
  "scripts/run_graphiti_baseline.sh 2>&1 | tee /tmp/graphiti_run.log"
```

The gate reads `preflight_manifest.json` rather than trusting exit codes,
and rejects a run where more than half the conditions retrieved nothing —
which is exactly how the first serrano attempt failed.

The stages below document what the script does; run them by hand when
debugging a single stage.

---

## Step 1 — One-account extraction smoke test

**Purpose.** Establish that the extractor produces usable structured output
before committing GPU-days. This is the step that catches the failure mode
nothing else catches.

```bash
export PERSONA_GRAPHITI_BUILD_ID=smoke1
export PERSONA_BENCHMARK_OUTPUT_DIR=results/persona_graphiti_preflight_smoke
python -m experiments.persona_graphiti_preflight \
    --config configs/persona_graphiti_surface_a.json \
    --histories 1
```

**Gate — read the store, do not just check the exit code.** Three extractor
failure modes exist and only two raise:

1. malformed JSON — retried by graphiti-core (4 attempts, backoff), usually
   self-heals at 4x latency;
2. valid JSON of the wrong shape — raises `pydantic.ValidationError`, caught by
   `max_episode_failures` (0 by default, so the run aborts);
3. **well-formed but empty or wrong extraction — raises nothing at all.**

`preflight_manifest.json` carries `extraction_summary` and
`extraction_health` for exactly case 3. Require:

- `extraction_health.status == "passed"` (no account committed zero facts);
- `extraction_summary.valid_at_resolved_fraction` at or above ~0.8 — the gold
  values are date-scoped, so unresolved dates degrade point-in-time retrieval
  toward unfiltered;
- `extraction_summary.empty_retrieval_count` near zero;
- spot-check `store_states.jsonl` by eye: do the extracted subjects, values,
  and validity windows correspond to the dialogue?

If the local extractor cannot clear this, fall back to a pinned API model and
record it — the extractor identity is `PERSONA_GRAPHITI_LLM_MODEL` and appears
at `manifest.graphiti_memory.llm_model`.

---

## Step 2 — Full ingestion preflight (all 12 accounts, no generation)

```bash
export PERSONA_BENCHMARK_OUTPUT_DIR=results/persona_graphiti_preflight_v1
python -m experiments.persona_graphiti_preflight \
    --config configs/persona_graphiti_surface_a.json
```

**Gate.** `condition_count == 120`, `graphiti_prompt_count == 240`,
`graphiti_memory.episode_count == 312` (one pass per account, not 1,638 — see
the ingestion-shape section of the baseline document), and
`extraction_health.status == "passed"`.

Expected wall clock: roughly one hour on a single GPU at ~10 s/episode, less
with `max_concurrent_histories` above 1.

---

## Step 3 — Build 1 end-to-end

```bash
export PERSONA_GRAPHITI_BUILD_ID=build1
export PERSONA_BENCHMARK_OUTPUT_DIR=results/persona_graphiti_e2e_build1
python -m experiments.persona_end_to_end_benchmark \
    --config configs/persona_graphiti_surface_a.json
```

Produces `generations.jsonl`, `predictions.jsonl`, `metrics.json`,
`generation_manifest.json`, `manifest.json`, and `graphiti_store_states.json`.
The run is resumable: `generation_manifest.json` pins every immutable field and
refuses a resume whose provenance differs.

**Gate.** `manifest.status == "completed"`, and
`manifest.graphiti_memory.episode_failure_count == 0`.

---

## Step 4 — Builds 2 and 3 (ingestion variance)

Repeat Step 3 with a **new `PERSONA_GRAPHITI_BUILD_ID` and a new output
directory** each time. The build id namespaces graphiti's group ids, so reusing
it across concurrent runs would collide.

```bash
for build in build2 build3; do
  export PERSONA_GRAPHITI_BUILD_ID=$build
  export PERSONA_BENCHMARK_OUTPUT_DIR=results/persona_graphiti_e2e_$build
  python -m experiments.persona_end_to_end_benchmark \
      --config configs/persona_graphiti_surface_a.json
done
```

Graphiti's ingestion is LLM-nondeterministic, so the three builds bound
extraction variance. Report per-arm exact-match spread across builds alongside
the headline number; a spread comparable to the effect size means the effect is
not resolvable at this sample size.

Note that ingestion is now incremental per account, so a build resamples whole
accounts rather than individual checkpoints — the three builds are the unit of
variance, and an account's ten checkpoints are correlated within a build.

---

## Step 5 — Variant B (schema-matched episodes)

Set `graphiti.episode_variant` to `matched` in a copy of the config and repeat
Steps 2–4 into their own directories. Variant A ingests rendered dialogue
(off-the-shelf); variant B ingests surface-mapped fact fields as structured
JSON, bypassing extraction while leaving graphiti's dedup and contradiction
judgments intact. The A−B delta prices extraction.

Run variant A first. If budget allows only one, A is the honest
"external system on our task" number.

---

## Step 6 — Analysis

```bash
python -m experiments.persona_graphiti_analysis \
    --run-dir results/persona_graphiti_e2e_build1 \
    --corpus-dir results/persona_conflict_conversations_v1
```

Writes `README.md`, `analysis.md`, and `error_taxonomy.json` into the run
directory:

- **README.md** — per-arm exact match and F1, matched effect sizes with
  history-clustered 95% bootstrap intervals, mechanism and provenance, and an
  explicit statement of what the artifact does not establish.
- **analysis.md** — exact match by arm and query family, by arm and
  phase/token-distance, and the error taxonomy per family. This is where the
  pre-registered predictions are checked.
- **error_taxonomy.json** — per-arm counts of correct, **stale intrusion** (a
  value the queried account held at some point but not at the queried date),
  cross-account intrusion, abstention, and other. Stale intrusion is the
  predicted newest-wins failure, so it is reported rather than folded into
  accuracy.

---

## Step 7 — What must not be reported

- **No state-fidelity number** until the alignment from Graphiti's extracted
  vocabulary to the corpus's canonical fact space is chosen and disclosed.
  `graphiti_store_states.json` holds the inputs; the scorer does not exist.
- **No per-operation claim** beyond the two probed query families. Scope
  exceptions, lineage retraction, and backdated correction are present as
  stream interference, so their effects are observable but not attributable.
- **No comparison to Graphiti's published LoCoMo numbers.** Only the in-house
  matched arms are comparable.
- **Not "off-the-shelf Graphiti."** Label it "Graphiti ingestion with
  benchmark-matched retrieval"; the four deviations are enumerated in
  `manifest.graphiti_memory.deviations_from_default`.

---

## Smoke-test result, 2026-08-21 (serrano, blocking)

First one-account smoke test ran end to end: 26 episodes, 219 extraction
calls, zero HTTP failures, 311 committed facts across 10 checkpoints, 100%
carrying a resolved `valid_at`. Mechanically the pipeline works. Scientifically
it does not yet, for two reasons the store inspection exposed and the exit code
would not have:

**1. Extraction resolves the synthetic clock, not the in-text dates.** Every
committed edge carries `valid_at` from the episode's `reference_time`
(2026-01-01 + i minutes), not from the dialogue's own dates ("2025-07-01
through 2025-09-30"). The Zep design resolves in-text dates against
`reference_time`; this extractor is simply not doing so.

**2. That interacts fatally with point-in-time retrieval.** The delayed probes
ask as-of 2025-08-01 and 2025-10-15, but every edge is stamped 2026, so
`valid_at <= as_of` excludes essentially everything: 7 of 10 conditions
retrieved zero facts. The arm would score near zero for a reason that is an
artifact of the timestamp mapping, not a finding about write-time revision.

**3. Entity resolution is anchored on dialogue roles.** Extracted subjects are
`Assistant` and `User` rather than the account holder, so even retrieved edges
often do not name the entity the query asks about.

Decisions needed before any scored build:

- **Timestamp base.** T0 = 2026-01-01 sits *after* the corpus's 2025 in-text
  dates, so any reference-time fallback lands in the future relative to every
  query. Moving T0 before the corpus window removes the artifact, at the cost
  of a synthetic clock that no longer reads as "now".
- **Extraction guidance.** `add_episode` accepts `entity_types`, `edge_types`,
  and `custom_extraction_instructions`. Constraining these would anchor
  subjects on the account holder and push date resolution toward the in-text
  values. It is a further deviation from stock defaults and must be recorded.
- **Variant B first.** The `matched` variant feeds surface-mapped fact fields
  with explicit `valid_from`/`valid_to`, bypassing extraction entirely. Running
  B before A would separate "Graphiti's revision policy" from "this extractor's
  competence", which is the distinction the paper actually needs.

Do not report a number from variant A until at least the timestamp base is
settled; the current configuration measures extraction failure, not revision
policy.
