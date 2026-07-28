# hippo_overnight — HippoRAG / dense-seeded PPR results

This directory holds two **distinct** experiments. They were originally interleaved
in one folder; they are now separated.

```
hippo_overnight/
├── dense_ppr_sweep/     ← the main experiment: hybrid vs dense_ppr, 2 modes × 3 seeds
│   ├── ANALYSIS.md          full write-up of the sweep (read this first)
│   ├── hybrid/seed{0,1,2}/  RRF baseline retrieval
│   ├── dense_ppr/seed{0,1,2}/  experimental PPR-over-dense-seeds retrieval
│   ├── kg_builds/           shared noscallop/scallop fact snapshot for the sweep
│   └── logs/                driver.log (timing, ALL DONE marker), per-run logs
└── rlm_baseline/        ← a single earlier RLM run (internal run_id "hippo_cell56")
    ├── cell5_rlm_kg_noscallop/
    ├── cell6_rlm_kg_scallop/
    └── summary.csv, compliance_report.*, retrieval_report.*, run_metadata.json, ...
```

## dense_ppr_sweep — the main result

Run `hippo_overnight_20260725_080810` (2026-07-25 → 07-26). Compares the `hybrid`
baseline against the experimental `dense_ppr` mode across seeds 0/1/2, 50 examples
each. **Start with [ANALYSIS.md](dense_ppr_sweep/ANALYSIS.md).**

Headline: dense_ppr executes to spec but the per-example graph is too small for PPR
to matter (it propagates beyond the dense seeds in only 7/100 queries), so it trails
hybrid and must be re-run at session/corpus scope. All sub-runs ran orchestration as
`fixed_context` (the known retrieves-once RLM issue), so accuracy is provisional.

## rlm_baseline — kept for reference

An earlier single run (`run_id = hippo_cell56`). Unlike the sweep, this one passed the
RLM orchestration check (`qwen_native_tools_inside_rlm`) — it is the only run here that
actually engaged recursive RLM — but 21/22 of 50 examples hit execution errors, so only
~21–22 were answered. Retained because it carries the diagnostic-accuracy data and is the
one real-RLM data point for comparison once the RLM fix lands.

## What is / isn't tracked in git

Tracked: `summary.csv`, `compliance_report.md`, `retrieval_report.md`,
`run_metadata.json`, dense-index `manifest.json` (records the index digest), figures,
logs, and the analysis/README.

Not tracked (bulky or regenerable, see repo `.gitignore`): `*.jsonl` (pilot input,
per-query `retrieval_eval`, `rlm_logs`, index `facts.jsonl`), `tool_traces/`, and
`vectors.npy` (regenerable from `facts.jsonl` + the index manifest).
