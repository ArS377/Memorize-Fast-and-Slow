# Source and method notes

Companion to `README.md`. This records what was run, on what, why each
non-default choice was made, and what a reader must not conclude.

## Source inventory

| Source | Role | Integrity check |
|---|---|---|
| `results/persona_conflict_conversations_surface_a_graphiti_build2/` | Historical corpus used by this run. Account-unique surface values | `dataset.generation_manifest_sha256` in `manifest.json` (`c64c5c4268d9…`) |
| `generations.jsonl` | One row per (condition, arm), raw decode | `manifest.artifact_sha256` |
| `predictions.jsonl` | Generations plus EM/F1 scoring | `manifest.artifact_sha256` |
| `metrics.json` | Per-arm aggregates and paired bootstrap deltas | `manifest.artifact_sha256` |
| `graphiti_retrievals.json` | Exact ranked rows fitted into each Graphiti prompt | `manifest.artifact_sha256` |
| `graphiti_store_states.json` | Per-checkpoint Graphiti store snapshot | `manifest.artifact_sha256` |
| `hybrid_retrievals.json` | Exact rows fitted into each hybrid prompt | `manifest.artifact_sha256` |
| `hybrid_condition_facts.json` | Scallop-admitted facts committed to Neo4j | `manifest.artifact_sha256` |
| `hybrid_admission_ledgers.json` | Per-event admission decisions with reason and rule version | `manifest.artifact_sha256` |
| `diagnostics.json` | Memoryless floor and evidence-availability decomposition | regenerate with the analysis command below |
| `generation_manifest.json` | Immutable provenance checked on every resume | `manifest.generation_manifest_sha256` |

Retrieval maps are archived because the live stores do not survive the run:
Graphiti's group namespaces are torn down after ingestion, and the hybrid
graph session is mutated by any later run. Without these files the evidence
each arm was scored on could not be re-derived, and extraction is stochastic
so it could not be regenerated either.

## Environment

- Host `serrano`, 5x RTX 3090. Extraction on GPU 0, answer generation on GPU 2.
- Answer model `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
  greedy, batch size 1, 32 new tokens, bf16, SDPA.
- Graphiti extraction served by vLLM 0.27.1 on `Qwen/Qwen3-4B`,
  `--max-model-len 32768`. 32K rather than 16K because graphiti's extractors
  request a large output budget; at 16K every edge-extraction call returned 400.
- `graphiti-core==0.29.3`, pinned and re-verified at runtime against the config.
- Embeddings `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`,
  the same revision as the hybrid arm's dense branch.
- `scallopy 0.2.4` from the project's cp310 manylinux wheel, in a Python 3.10
  environment provisioned with `uv`; it is not on PyPI. Served over the repo's
  HTTP validator boundary on port 8081.
- Three Neo4j 5.26.0 Community instances on separate ports: the pre-existing one
  (17687, untouched), one for the hybrid arm (17693), one for Graphiti (17690).
  Community allows a single database per instance, so isolation requires
  separate instances rather than separate databases.

## Why each deviation from stock Graphiti

The arm is **Graphiti ingestion with benchmark-matched retrieval**, not
off-the-shelf Graphiti. Deviations are enumerated in
`manifest.graphiti_memory.deviations_from_default`; the reasoning:

1. **Embedder replaced** with the benchmark's pinned BGE revision. Required for
   a matched comparison; otherwise the two arms differ in embedding model as
   well as revision policy.
2. **Cross-encoder disabled.** The default RRF search recipe never invokes it,
   so this changes no result; it only prevents the constructor's default
   `OpenAIRerankerClient` from making a remote call during an offline run.
3. **Explicit bi-temporal search filter.** graphiti applies none by default, so
   an unfiltered search returns invalidated edges and the answer model, not the
   memory system, ends up adjudicating validity. That would confound the
   write-time claim under test.
4. **World-time filtering only.** graphiti sets `expired_at` on any edge that
   acquires an `invalid_at`, bounded facts included, so requiring
   `expired_at IS NULL` would delete every closed-interval fact, which is
   exactly the `preference_change` gold.
5. **Local re-application of the temporal predicate.** Build 1 returned one edge
   whose `invalid_at` equalled the as-of date under a strict `>` filter, so the
   server filter is not treated as authoritative. Drops are counted in
   `graphiti_memory.session.server_filter_violations_dropped` (0 in this build).
6. **Output tokens clamped at the client.** graphiti's extractors pass their own
   `max_tokens` and the base client resolves `max_tokens or self.max_tokens`, so
   `LLMConfig.max_tokens` never takes effect; without the clamp the manifest
   would record a limit that did not run.
7. **Neo4j driver subclassed.** graphiti 0.29.3 places `database_` in the Cypher
   parameter map rather than the driver's control argument, so writes and search
   land on the connection's default database while `clear_data` targets the
   configured one. Unpatched, namespaces are never actually cleared.
8. **Query embeddings use the document path.** graphiti's embedder interface
   cannot distinguish a query from a document, so queries do not receive the
   retrieval instruction our own dense branch applies. Model and revision match;
   query-side encoding does not.

## Method choices worth stating

- **Ingestion is incremental per account.** An account's ten checkpoints are
  nested prefixes, so rebuilding per checkpoint re-ingests shared history ten
  times. Ingesting once and snapshotting at each boundary gives 312 episodes
  instead of 1,638, and every checkpoint was verified to see a turn set
  identical to a fresh rebuild.
- **Synthetic ingestion clock at `2024-01-01 + i minutes`.** The corpus has no
  wall clock. T0 must precede the corpus's 2025 dates: an earlier attempt used
  2026 and, because the extractor often falls back to `reference_time` instead
  of resolving in-text dates, every edge was stamped in the future and
  point-in-time retrieval emptied the store. The cost of this choice is that
  undated extractions are not temporally discriminated at all.
- **`full_qwen_context` excluded.** Its prompts reach ~51k tokens on this
  corpus; the KV cache exceeds 24 GB during generation even though the model's
  context window admits them.

## Comparison caveats

1. **Single stochastic ingestion build.** The protocol asks for three, so
   between-build extraction variance is not measured. An earlier unpublished
   build differed by several points on the Graphiti arm at 4K, which suggests
   the variance is not negligible against the smaller reported effects.
2. **The 16K hybrid-vs-Graphiti gap is not resolvable** at 120 conditions; its
   interval includes zero. Only the 4K gap is a claim.
3. **Earlier builds are not published and must not be pooled with this one.**
   They used a hybrid prompt header reading "Scallop-validated retrieved memory
   facts", a framing advantage inside the comparison, and one contained the
   temporal boundary violation described above. They were removed rather than
   archived, since their numbers are not usable.
4. **Two delayed query families only.** Scope exceptions, lineage retraction,
   and backdated correction appear in the stream as interference, so their
   effects are observable but not attributable.
5. **Account scoping is an acknowledged oracle** for both memory arms: each
   ingests only the queried account's turns. It is symmetric, but it is not a
   capability either system demonstrated.
6. **No state-fidelity number.** Aligning Graphiti's extracted vocabulary to the
   corpus's canonical fact space is unimplemented, so only downstream utility is
   measured.

## Reproduction

```bash
# Analysis artifacts (README.md, analysis.md, diagnostics.json, error_taxonomy.json)
python -m experiments.persona_graphiti_analysis \
    --run-dir results/persona_joint_surface_a_build2 \
    --corpus-dir results/persona_conflict_conversations_surface_a_graphiti_build2

# Memoryless and recency oracle floors quoted in the README
python scripts/shortcut_oracle_probe.py
```

The benchmark itself is re-entrant: with
`PERSONA_GRAPHITI_RETRIEVAL_CACHE` and `PERSONA_HYBRID_RETRIEVAL_CACHE` set, a
re-run replays the scored memory and appends to `generations.jsonl` rather than
rebuilding. Both keys include the relevant implementation hash, so a code change
invalidates the cache instead of silently replaying stale retrievals.

## Operational notes

The host terminated the benchmark with SIGTERM roughly hourly, six times across
these runs. The sender was not identified: `earlyoom` is active but its dual
memory-and-swap threshold was not met, no kernel OOM record exists, sibling
processes survived, and the exit code was 143 rather than a Python error.
`scripts/supervise_joint_run.sh` re-enters until the manifest reports
completion; build 2 finished across two attempts with no manual intervention.

Caches live under `~/graphiti_caches/` rather than `/tmp`, which is reaped on
this host.

## Remote state

Left running on serrano after this build: vLLM on GPU 0, the Scallop validator
on 8081, and the Graphiti (17690) and hybrid (17693) Neo4j instances. The
pre-existing Neo4j on 17687 was not modified. Graphiti group namespaces are
cleared by the run itself; the hybrid graph session is left populated, which is
why its retrieval map is archived here rather than regenerated.
