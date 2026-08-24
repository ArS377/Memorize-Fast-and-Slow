# Run compliance: reproduction-20260802

Overall: **PASS**

- PASS `manifest` — run and Git identity are recorded
- PASS `dataset_hash` — input SHA-256 is recorded
- PASS `typed_scope` — memory scope is explicit
- PASS `tool_contract` — tool schema version is recorded
- PASS `rule_version` — symbolic rule version is recorded
- PASS `cell_1_output` — results/reproduction_20260802/cell1_flat_raw/results.jsonl contains 50 rows
- PASS `cell_1_run_identity` — every row belongs to this run
- PASS `cell_1_execution` — 0 execution errors
- PASS `cell_2_output` — results/reproduction_20260802/cell2_flat_kg_noscallop/results.jsonl contains 50 rows
- PASS `cell_2_run_identity` — every row belongs to this run
- PASS `cell_2_execution` — 0 execution errors
- PASS `cell_2_retrieval_configuration` — expected 'hybrid'; configured modes: ['hybrid']
- PASS `cell_2_retrieval_execution` — effective modes: ['hybrid']; degraded rows: 0; dense identities present: True
- PASS `cell_3_output` — results/reproduction_20260802/cell3_flat_kg_scallop/results.jsonl contains 50 rows
- PASS `cell_3_run_identity` — every row belongs to this run
- PASS `cell_3_execution` — 0 execution errors
- PASS `cell_3_retrieval_configuration` — expected 'hybrid'; configured modes: ['hybrid']
- PASS `cell_3_retrieval_execution` — effective modes: ['hybrid']; degraded rows: 0; dense identities present: True
- PASS `identical_example_population` — requested cells use the same ordered example IDs
- PASS `frozen_candidate_corpus` — both KG variants derive from the recorded frozen candidates
