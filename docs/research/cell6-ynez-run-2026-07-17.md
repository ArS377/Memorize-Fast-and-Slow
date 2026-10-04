---
title: "Cell 6 Ynez run: validated KG retrieval analysis"
date: 2026-07-17
tags: [neurosymbolic, scallop, knowledge-graph, longbench, rlm, reproducibility]
status: research-note
---

# Cell 6 on Ynez — 17 July 2026

## Executive result

The completed Cell 6 (`rlm_kg_scallop`) run achieved **26/50 correct (52.0%)**
on the 50-example evaluation slice. This is the best observed Cell 6 result in
the current work history, above the prior Cell 6 run (**19/50, 38.0%**) and the
team's reported historical range of roughly 17–19/50.

This is encouraging, but it is not yet evidence that Scallop validation caused
the accuracy gain. KG retrieval coverage was very low: 47/50 examples retrieved
no triples.

## Artifacts

- Results: `results/cell6_ynez_20260717_160307/cell6_rlm_kg_scallop/results.jsonl`
- Summary: `results/cell6_ynez_20260717_160307/summary.csv`
- Run log: `results/cell6_ynez_20260717_160307/cell6.log`
- Validated facts: `results/kg_builds/pilot_scallop_facts.jsonl`
- Setup log: `results/services/cell6_setup.log`
- Import log: `results/services/cell6_import_clean.log`
- Prior Cell 6 result: `results/cell6_rlm_kg_scallop/results.jsonl`

## Runtime configuration

- Server: Ynez.
- Model: `Qwen/Qwen3-4B`.
- Serving: vLLM `0.8.5`, OpenAI-compatible endpoint
  `http://127.0.0.1:8000/v1`.
- Tool serving: `--enable-auto-tool-choice --tool-call-parser hermes`.
- Engine workaround: `VLLM_USE_V1=0`, required because the V1 engine hit a
  FlashInfer/Torch native-extension ABI error.
- RLM: OpenAI-compatible backend; max depth 2; 10 iterations; 64,000 total
  token budget.
- Retrieval: `pilot_scallop`, 2 hops, up to 50 triples, 4,000 context chars,
  example scope.
- Graph: Neo4j at `bolt://127.0.0.1:7687`.
- Validation: actual `scallopy` in Python 3.10 `kg-env`.
- Process management: vLLM and Cell 6 were kept in separate tmux sessions,
  `cell6-vllm` and `cell6`.

## KG and validation state

The setup processed all 50 examples and produced 87 accepted facts.

| Measure | Value |
|---|---:|
| Accepted facts | 87 |
| Examples with at least one accepted fact | 23/50 |
| Examples with no accepted fact | 27/50 |
| Scallop rejections during clean import | 0 |
| Clean `pilot_scallop` graph count | 87 relationships |

The session initially contained 22 stale relationships. It was reset and only
the validated 87-fact corpus was re-imported, preventing old graph state from
leaking into the final run. Other Neo4j sessions were not intentionally cleared.

Scallop was active, but it accepted every fact in the final corpus. This run
therefore does not isolate an accuracy effect caused by Scallop rejecting facts.

## Current versus prior Cell 6

| Measure | 17 Jul 2026 run | Prior run |
|---|---:|---:|
| Correct | 26/50 | 19/50 |
| Accuracy | 52.0% | 38.0% |
| Errors | 0 | 0 |
| Mean latency | 102.2 s | 87.1 s |
| Zero-triple examples | 47 | 47 |
| Nonzero-triple examples | 3 | 3 |
| Mean triples/example | 0.08 | 0.06 |

The paired comparison had 12 wrong-to-right transitions and 5 right-to-wrong
transitions, for a net gain of seven correct answers. The three current-run
examples that received graph context were all answered correctly. This is
promising but too small a sample for a KG or validation claim.

## Root cause of low retrieval coverage

The Ynez implementation used sparse, entity-seeded graph retrieval:

1. Extract title-cased or quoted entities from the question.
2. Require an exact `Neo4j Entity.name` match.
3. Traverse up to two hops.
4. Restrict relationships to the current example ID.

This is brittle for the observed facts. Many accepted facts are keyed by
formulas or lower-case phrases, e.g. `L(1,5)` and `uncertain variable`, while
question seed extraction often emits no matching entity. Further, a successful
Neo4j connection suppresses JSONL fallback even if the graph query returns zero
rows. The facts file is used only when Neo4j connection fails.

As a result, 47 examples received only the fallback context `No relevant facts.`
(18 characters). The run was largely an RLM-without-KG evaluation despite a
populated, Scallop-validated graph.

## Interpretation

The result supports: **the repaired Ynez runtime can achieve 26/50**, and graph
context appears useful in the small subset where it is retrieved.

It does not support: **Scallop validation caused the 14-point gain**. Both runs
had the same 3/50 retrieval coverage; the current validator accepted all 87
facts; and the RLM is stochastic. The higher mean latency and repaired runtime
may have changed reasoning trajectories. The new run had 12 gains but also five
regressions, which is consistent with run-to-run model variability.

With only 50 examples, this should be replicated before being presented as a
robust improvement.

## Commit provenance

Local `main` was at `034199c` (`docs: align Cell 6 setup with persistent
Scallop updates`). Important recent local commits include:

- `d82647c`: Qwen working-memory feedback tools.
- `f6d665f`: fail-closed Scallop validation service.
- `966b9c1`: isolated six-cell runs, manifests, compliance checks, and audit
  metadata.
- `034199c`: persistent Cell 6 setup using Neo4j plus the validator service.

These are significant for methodological validity: actual-Scallop enforcement,
provenance, reproducibility, and future native Qwen KG/working-memory behavior.

However, the successful Ynez run used a separate checkout at commit `76dd2d0`
with substantial uncommitted modifications. That checkout did not contain local
commit `034199c`. Do not attribute 26/50 to the newest local commits without
deploying them and rerunning.

The Ynez history included re-enabling Qwen3 thinking with a 2048-token budget,
raising the RLM token budget to 64,000, and fixing answer-letter extraction.
The run also used the repaired vLLM 0.8.5 environment. These are plausible
contributors, but the prior run lacks sufficient runtime metadata for a causal
attribution.

## Next experiments for the paper

1. Deploy a clean committed version of the current local pipeline to Ynez.
2. Record commit hash, model revision, dependency versions, vLLM version and
   engine mode, command line, random seed, graph session, and data hash.
3. Run a matched Cell 5 versus Cell 6 ablation: same data, model, runtime,
   prompting, graph candidates, and retrieval; only the Scallop constraint
   differs.
4. Repeat each condition at least three times to estimate variance.
5. Report retrieval coverage alongside accuracy: fact coverage, percentage with
   at least one triple, mean triples, context length, and accuracy conditional
   on nonempty retrieval.
6. Improve fact coverage: 27/50 examples currently have no accepted facts.
7. Replace sparse exact-match-only retrieval with hybrid retrieval: dense
   semantic ranking over fact text/provenance followed by sparse graph expansion
   and Scallop-safe validation/commit checks.
8. Separate validity and accuracy claims. Scallop can make graph updates
   auditable and contradiction-safe even when it does not raise QA accuracy.

## Paper-ready claim boundary

Defensible now: "A Scallop-validated 87-fact Neo4j corpus supported a Cell 6
RLM run that achieved 26/50 accuracy on a 50-example slice. Sparse retrieval
returned graph context for only three examples; all three were correct, but the
limited coverage prevents attributing overall accuracy to validation."

Not defensible yet: "Scallop validation improved accuracy by 14 percentage
points." That requires a matched, repeated Cell 5/Cell 6 ablation.
