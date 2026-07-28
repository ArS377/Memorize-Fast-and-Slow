# Dense-seeded PPR retrieval — overnight sweep analysis

**Run:** `hippo_overnight_20260725_080810`
**Driver:** started 2026-07-25 08:08, finished 2026-07-26 22:48 (`ALL DONE`), ~38 h wall clock
**Code provenance:** git `e4530f3` ("Add opt-in dense-PPR retrieval experiment")
**Design reference:** Research proposal — *Dense-seeded PPR retrieval* (4th retrieval mode inspired by HippoRAG's Personalized PageRank)
**Analyst note:** written by Gregory as part of the HippoRAG / dense_ppr test track.

---

## 1. What was run

Two retrieval modes × three seeds, 50 examples each (100 retrieval queries per sub-run: cells 5 + 6):

| | Modes | Seeds | Examples | Cells |
|---|---|---|---|---|
| Sweep | `hybrid` (baseline), `dense_ppr` (experimental) | 0, 1, 2 | 50 | 5 = rlm_kg_noscallop, 6 = rlm_kg_scallop |

All six sub-runs completed with full `summary.csv`, `compliance_report`, `retrieval_report`, `run_metadata`, and `manifest`. Results are **deterministic across seeds** (identical scope/propagation statistics per seed), which is the reproducibility property the proposal requires.

---

## 2. Headline result

**dense_ppr executed exactly to spec, but the example-scoped graph is too small for PPR to do anything.** On this frozen slice it degenerates to plain dense retrieval in 93 % of queries, and where it does propagate it slightly *hurts* accuracy. This is not a bug — it is precisely the failure mode the proposal predicted:

> *"The default example scope may contain too little graph structure to demonstrate meaningful PPR propagation. Evaluation must also include corpus-scale session or controlled session_set retrieval."*

So this run is **not a verdict on dense_ppr**. It is a clean, reproducible demonstration that the experiment must be re-run at session / corpus scope before dense_ppr can be judged.

---

## 3. Answer accuracy (mean across 3 seeds)

| Mode | Cell 5 (noscallop) | Cell 6 (scallop) |
|---|---|---|
| **hybrid** (baseline) | **0.507** (.46 / .50 / .56) | **0.473** (.51 / .45 / .46) |
| **dense_ppr** | 0.453 (.46 / .42 / .48) | 0.433 (.42 / .46 / .42) |

hybrid beats dense_ppr by ~5 points on both cells. The seed-to-seed spread (~±0.06) is wider than the mode gap, so treat the gap as *suggestive, not decisive* — especially given the orchestration caveat in §6.

---

## 4. Why PPR didn't help — the propagation evidence

Per-query graph scope and propagation, from `retrieval_eval.jsonl` (identical across all three dense_ppr seeds):

| Metric | Value |
|---|---|
| Mean facts in per-example PPR scope | **9.78** (max 28, min 0) |
| Mean edges in scope | 19.5 |
| PPR converged | **100/100** (mean 28 iterations, final delta ~7e-9) |
| Queries where PPR surfaced a fact **beyond the dense seeds** | **7 / 100** |
| Empty-result rate | 2 / 100 (2 %) |

In **93 of 100 queries the PPR output equals the dense seed set** — there is simply no graph to walk. The dense retriever already returns ~10 facts, and the per-example scope *is* those ~10 facts plus a handful of shared entities. PPR ranks the seeds it was given and finds nothing new.

Corroborating aggregate (fusion diagnostics, per sub-run over all 100 queries):
- **dense_ppr:** 510 unique facts, all 510 returned by PPR, 498 in both branches → PPR branch ≈ dense branch.
- **hybrid:** 510 unique facts, only 84 overlapping between sparse and dense → the two branches genuinely see different facts (RRF has something to fuse).

---

## 5. Proposal scorecard (sparse / dense / **hybrid** / **dense_ppr**)

Only `hybrid` and `dense_ppr` were in this sweep. Mapping to the proposal's required metrics:

| Metric | hybrid | dense_ppr | Notes |
|---|---|---|---|
| Queries with ≥1 result | 98 % | 98 % | 2 % empty both |
| **Recall@5 / @10** | **n/a** | **n/a** | `relevance_source = "unlabeled"` for all 100 queries — no independent labels exist, so recall cannot be computed |
| Multi-hop evidence coverage | n/a | n/a | blocked by same missing labels |
| Answer accuracy | 0.507 / 0.473 | 0.453 / 0.433 | see §3; provisional (see §6) |
| Empty-result rate | 2 % | 2 % | |
| Online retrieval latency | sub-second | dense ~0.022 s + ppr ~0.001 s | negligible vs. ~200 s/example LLM time |
| Indexing (one-time) cost | shared | shared | noscallop KG 8349 s (~2.3 h), scallop KG 8304 s (~2.3 h) |
| Provenance completeness | intact | intact | each fact keeps `(session_id, fact_id)`; `dense_index_identity` + `ppr_index_identity` carry model/revision/snapshot SHA-256 + ordered-fact digest |
| Scope-isolation failures | 0 | 0 | 0 degraded rows; scope applied **before** PPR (scope counts reflect the pre-filtered per-example graph) |

**The two metrics that would actually decide the experiment — Recall@k and multi-hop coverage — could not be computed because the query slice is unlabeled.** The proposal is explicit that Qwen citations cannot substitute for independent relevance labels.

**KG build:** noscallop 498 facts (128 rejected), scallop 480 facts (158 rejected). Note this is **not** the frozen 520-fact snapshot the roadmap says to reset to before an official benchmark.

---

## 6. Critical caveats

1. **Orchestration ran as `fixed_context`, not recursive RLM.** All six sub-runs FAIL the `rlm_pair_orchestration` compliance check (`observed modes: ['fixed_context']`). This is the known "retrieves-once / blank-answer" RLM controller issue currently being fixed, not a defect in dense_ppr. Consequence: **accuracy here reflects fixed-context orchestration.** Retrieval-quality comparisons (§4) are unaffected because retrieval sits upstream of the controller; accuracy comparisons (§3) are provisional and should be regenerated once the RLM fix lands.
2. **Unlabeled slice.** No Recall@k / multi-hop coverage possible (§5). Independent fact/sentence relevance labels are a prerequisite for the real comparison.
3. **Wrong graph snapshot.** 498/480 facts, not the frozen 520-fact snapshot.
4. **Scope too small** for PPR to demonstrate its value (§4) — the central limitation.

---

## 7. Recommendations

Aligned with the proposal's own gating ("do not merge unless it improves multi-hop retrieval and preserves provenance"):

1. **Do not change the default.** `hybrid` remains the baseline; nothing here justifies promoting dense_ppr. Nothing here condemns it either.
2. **Re-run at session / corpus (session_set) scope**, not example scope, so PPR has a graph to propagate over. This is the single change most likely to make the experiment meaningful.
3. **Add independent relevance labels** to the frozen slice so Recall@5/@10 and multi-hop coverage become computable — these, not answer accuracy, are the retrieval-level signal the design cares about.
4. **Reset to the frozen 520-fact snapshot** and re-run after the RLM `fixed_context` fix so accuracy reflects true recursive orchestration.
5. What *did* pass cleanly and should be preserved as-is: deterministic ordering, 100 % PPR convergence, scope-before-propagation isolation, and full provenance/index-identity recording.

---

## 8. Where the data lives

```
dense_ppr_sweep/               # run_id hippo_overnight_20260725_080810
├── logs/driver.log            # timing + ALL DONE marker, KG build costs, git SHA
├── kg_builds/                 # noscallop/scallop facts + rejections (shared snapshot)
├── {hybrid,dense_ppr}/seed{0,1,2}/
│   ├── summary.csv            # per-cell accuracy / latency / errors
│   ├── compliance_report.md   # orchestration + scope + scallop checks
│   ├── retrieval_report.md    # fusion diagnostics, candidate counts, recall (n/a)
│   ├── retrieval_eval.jsonl   # per-query scope, seeds, PPR scores, provenance  [gitignored, bulky]
│   ├── run_metadata.json / manifest.json
│   └── cell{5,6}_*/rlm_logs/  # full reasoning traces  [gitignored, bulky]
```
