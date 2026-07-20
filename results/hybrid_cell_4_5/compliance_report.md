# Run compliance: hybrid_cell5

Overall: **FAIL**

- PASS `manifest` — run and Git identity are recorded
- PASS `dataset_hash` — input SHA-256 is recorded
- PASS `typed_scope` — memory scope is explicit
- PASS `tool_contract` — tool schema version is recorded
- PASS `rule_version` — symbolic rule version is recorded
- PASS `cell_4_output` — results/runs/hybrid_cell5/cell4_rlm_raw/results.jsonl contains 50 rows
- PASS `cell_4_run_identity` — every row belongs to this run
- FAIL `cell_4_execution` — 2 execution errors
- PASS `cell_5_output` — results/runs/hybrid_cell5/cell5_rlm_kg_noscallop/results.jsonl contains 50 rows
- PASS `cell_5_run_identity` — every row belongs to this run
- PASS `cell_5_execution` — 0 execution errors
- PASS `identical_example_population` — requested cells use the same ordered example IDs
