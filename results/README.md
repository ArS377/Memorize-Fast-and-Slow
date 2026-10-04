# results/

Start with the [consolidated August results index](../docs/recent-results.md).
It separates current evidence from historical, smoke-only, and incomplete runs and links each conclusion to its supporting artifacts.

Result directories may contain scored runs, analyses, smoke tests, or incomplete execution provenance.
`synthetic_temporal_preferences_*` and `persona_*conversations*` are input corpora, `persona_neurosym_preflight_*` records retrieval preflights, and `dense_indexes/` contains retrieval-cache metadata rather than benchmark outcomes.

Two naming conventions are in use, from two different eras of the harness:

**Harness convention** (`experiments/run_all.py`, `experiments/aggregate.py`):
`cell{N}_{label}/results.jsonl` — one dir per ablation cell, e.g. `cell1_flat_raw/`,
`cell6_rlm_kg_scallop/`. `run_metadata.json` and `summary.csv` at the top level
are written by the aggregator and point at the current canonical run (see the
`canonical_run_metadata` field in `run_metadata.json`).

**Ad-hoc run convention** (manually orchestrated multi-shard runs, sweeps, and
ablations): `{label}_{scale}/` or `{experiment_name}_{date}/`, e.g.
`flat_bm25_500/`, `longbench_cell6_selection_ablation_20260807/`,
`gitmain_verification_2026-08-05/`. Each such directory is self-contained:
`results.jsonl` (merged), `shard{N}/results.jsonl` (per-shard), and
`logs/shard{N}.log` (per-shard stdout/stderr) where the run was GPU-sharded.

Directory names are treated as stable identifiers once a run is committed —
orchestration scripts and docs reference them by exact path, so existing
directories are not renamed retroactively even when a more consistent name
would read better.

`figures/` holds aggregator-rendered charts. `cell6_ynez_run_20260717/` is a
raw archive (summary + per-example JSONL + full RLM console trace) for the
run analyzed in `docs/research/cell6-ynez-run-2026-07-17.md`.
