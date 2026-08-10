# Source and figure notes

## Source inventory

| Source | Role | Integrity check |
|---|---|---|
| `raw/gitmain/cell5_hybrid/results.jsonl` | No-local-changes, cell 5 (no Scallop), hybrid | `kg_snapshot_sha256` in `run_metadata.json` |
| `raw/gitmain/cell6_hybrid/results.jsonl` | No-local-changes, cell 6, hybrid | `kg_snapshot_sha256` in `run_metadata.json` |
| `raw/gitmain/cell6_dense_ppr/results.jsonl` | No-local-changes, cell 6, dense_ppr | `kg_snapshot_sha256` in `run_metadata.json` |
| `raw/semantic20/{hybrid,dense_ppr}/results.jsonl` | Local changes, flat blended-query baseline | same |
| `raw/peropt/{hybrid,dense_ppr}/results.jsonl` | Local changes, + per-option query fan-out | same |
| `raw/relfloor/{hybrid,dense_ppr}/results.jsonl` | Local changes, + relevance-floor (0.85) | same |
| `raw/mmr/{hybrid,dense_ppr}/results.jsonl` | Local changes, + MMR (lambda=0.7) | same |
| `raw/combined/{hybrid,dense_ppr}/results.jsonl` | Local changes, all three combined | same |
| `logs/eval_*.log` | Per-variant/mode eval run logs (stdout of `run_longbench_cell6.py`) | manual spot-check during the runs |

`analyze_results.py` reads `dense_index_identity[0].fact_count` and `.source_snapshot_sha256` directly from each row rather than recomputing a separate hash -- these are written by the harness at eval time from the actual KG snapshot each run queried, so a mismatch there would indicate a stale dense-index cache rather than a bad report.

## Figure map

| Figure | Source fields | Transformation | Why included |
|---|---|---|---|
| `figures/accuracy_by_condition.png` | `correct`, 50-row denominator, per run | Accuracy plus 95% Wilson interval, grey=no-local-changes / blue=local-changes | Single view of all 13 runs against the 25% chance line |

A latency-vs-accuracy chart and a fact-count-vs-accuracy scatter were considered and omitted: latency is answer-phase only (KG build time lives in the source session logs, not the result rows) and would be easy to misread as total cost, and fact count vs. accuracy is already the paper's main finding in prose (more facts alone made things worse; selective facts made them better) -- a scatter of 13 points doesn't add information a reader can't get faster from the table.

## Remote cleanup audit

All build- and eval-time vLLM tmux sessions and their processes were killed after each phase completed; `nvidia-smi` was checked after each teardown throughout this investigation and confirmed all serrano GPUs at 4 MiB / 0% utilization before the next phase started. No isolated Neo4j sessions or dense-index caches were removed as part of this report -- they remain on serrano under `~/NeuroSym/results/` for anyone who wants to re-query them without rebuilding.
