# Run compliance: hippo_cell56

Overall: **FAIL**

- PASS `manifest` — run and Git identity are recorded
- PASS `dataset_hash` — input SHA-256 is recorded
- PASS `typed_scope` — memory scope is explicit
- PASS `tool_contract` — tool schema version is recorded
- PASS `rule_version` — symbolic rule version is recorded
- PASS `cell_5_output` — results/runs/hippo_cell56/cell5_rlm_kg_noscallop/results.jsonl contains 50 rows
- PASS `cell_5_run_identity` — every row belongs to this run
- FAIL `cell_5_execution` — 21 execution errors
- PASS `cell_5_retrieval_configuration` — expected 'dense_ppr'; configured modes: ['dense_ppr']
- PASS `cell_5_retrieval_execution` — effective modes: ['dense_ppr']; degraded rows: 0; dense identities present: True
- PASS `cell_6_output` — results/runs/hippo_cell56/cell6_rlm_kg_scallop/results.jsonl contains 50 rows
- PASS `cell_6_run_identity` — every row belongs to this run
- FAIL `cell_6_execution` — 22 execution errors
- PASS `cell_6_retrieval_configuration` — expected 'dense_ppr'; configured modes: ['dense_ppr']
- PASS `cell_6_retrieval_execution` — effective modes: ['dense_ppr']; degraded rows: 0; dense identities present: True
- PASS `identical_example_population` — requested cells use the same ordered example IDs
- PASS `rlm_pair_orchestration` — observed modes: ['qwen_native_tools_inside_rlm']
- PASS `cell_6_actual_scallop` — observed backends: ['scallop']
