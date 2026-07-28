# Run compliance: hippo_overnight_20260725_080810_hybrid_seed2

Overall: **FAIL**

- PASS `manifest` — run and Git identity are recorded
- PASS `dataset_hash` — input SHA-256 is recorded
- PASS `typed_scope` — memory scope is explicit
- PASS `tool_contract` — tool schema version is recorded
- PASS `rule_version` — symbolic rule version is recorded
- PASS `cell_5_output` — results/runs/hippo_overnight_20260725_080810/hybrid/seed2/cell5_rlm_kg_noscallop/results.jsonl contains 50 rows
- PASS `cell_5_run_identity` — every row belongs to this run
- PASS `cell_5_execution` — 0 execution errors
- PASS `cell_5_retrieval_configuration` — expected 'hybrid'; configured modes: ['hybrid']
- PASS `cell_5_retrieval_execution` — effective modes: ['hybrid']; degraded rows: 0; dense identities present: True
- PASS `cell_6_output` — results/runs/hippo_overnight_20260725_080810/hybrid/seed2/cell6_rlm_kg_scallop/results.jsonl contains 50 rows
- PASS `cell_6_run_identity` — every row belongs to this run
- PASS `cell_6_execution` — 0 execution errors
- PASS `cell_6_retrieval_configuration` — expected 'hybrid'; configured modes: ['hybrid']
- PASS `cell_6_retrieval_execution` — effective modes: ['hybrid']; degraded rows: 0; dense identities present: True
- PASS `identical_example_population` — requested cells use the same ordered example IDs
- FAIL `rlm_pair_orchestration` — observed modes: ['fixed_context']
- PASS `cell_6_actual_scallop` — observed backends: ['scallop']
