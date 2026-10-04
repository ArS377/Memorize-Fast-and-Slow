# Run compliance: e2e-smoke-final

Overall: **PASS**

- PASS `manifest` — run and Git identity are recorded
- PASS `dataset_hash` — input SHA-256 is recorded
- PASS `typed_scope` — memory scope is explicit
- PASS `tool_contract` — tool schema version is recorded
- PASS `rule_version` — symbolic rule version is recorded
- PASS `cell_1_output` — /home/kylezheng/NeuroSym/results/e2e_smoke/final_run_all_hybrid/cell1_flat_raw/results.jsonl contains 1 rows
- PASS `cell_1_run_identity` — every row belongs to this run
- PASS `cell_1_execution` — 0 execution errors
- PASS `cell_2_output` — /home/kylezheng/NeuroSym/results/e2e_smoke/final_run_all_hybrid/cell2_flat_kg_noscallop/results.jsonl contains 1 rows
- PASS `cell_2_run_identity` — every row belongs to this run
- PASS `cell_2_execution` — 0 execution errors
- PASS `cell_2_retrieval_configuration` — expected 'hybrid'; configured modes: ['hybrid']
- PASS `cell_2_retrieval_execution` — effective modes: ['hybrid']; degraded rows: 0; dense identities present: True
- PASS `identical_example_population` — requested cells use the same ordered example IDs
