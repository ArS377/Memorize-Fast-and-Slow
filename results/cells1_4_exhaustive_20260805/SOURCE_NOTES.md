# Source and figure notes

## Source inventory

| Source | Role | Integrity check |
|---|---|---|
| `raw/cell1_flat_raw/results.jsonl` | Cell 1 result rows | SHA-256 in `run_metadata.json` |
| `raw/cell2_flat_kg_noscallop/results.jsonl` | Cell 2 result rows | SHA-256 in `run_metadata.json` |
| `raw/cell3_flat_kg_scallop/results.jsonl` | Cell 3 result rows | SHA-256 in `run_metadata.json` |
| `raw/cell4_rlm_raw/results.jsonl` | Cell 4 result rows | SHA-256 in `run_metadata.json` |
| `logs/cells23-assemble-5c1e0dd.log` | Candidate, accepted, and rejected KG counts | Counts asserted to reconcile in `analyze_results.py` |
| Frozen `pilot_input.jsonl` | Example order and gold-answer audit | SHA-256 `82b7f112eaca68f2456283745f72e495c351f85f2a37921f7a9871f73ffa4b53` |

The analysis script asserts 50 rows per cell, identical ordered example IDs and gold labels, expected run IDs and labels, no runtime errors, and no Cell 2/3 retrieval degradation before writing summaries. The frozen input hash was recorded during the run; the large source contexts are not duplicated in this results package. Supplying that exact input with `--input` enables a fresh hash and source-to-result audit.

## Figure map

| Figure | Source fields | Transformation | Why included |
|---|---|---|---|
| `figures/accuracy_by_cell.png` | `correct`, 50-row denominator | Accuracy plus 95% Wilson interval | Primary outcome across the four cells |
| `figures/kg_fact_filtering.png` | 9,681 candidates, 7,902 accepted, 1,779 rejected | Accepted/rejected share of candidate corpus | Shows the concrete Scallop intervention |

Both static PNGs were rendered from generated CSV/JSON files and visually inspected for readable labels, unclipped annotations, zero-based axes, and accurate direct labels. A latency comparison was intentionally omitted because KG construction time is absent from Cell 2/3 result rows and answer-phase latency alone would be easy to misread as total runtime. Cross-cell context-size and triple-count charts were also omitted because those fields have different meanings by orchestration mode.

## Report packaging limitation

The preferred portable HTML validation workflow could not run because this workstation has no Node runtime. No runtime was installed solely for reporting, and no unvalidated HTML was published. The committed report therefore uses GitHub-native Markdown plus static, reproducibly generated figures.

## Remote cleanup audit

After local archival, run-scoped services, tmux sessions, graph data, isolated worktrees, and the isolated environment were removed from Ynez. A final read-only audit found all four GPUs at 4 MiB and 0% utilization, no tmux server, no run process, and none of the three isolated run paths. Shared model caches and the shared Neo4j installation were not created by this run and were left intact.
