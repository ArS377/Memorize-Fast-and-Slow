# Qwen3.5-4B Persona NeuroSym Benchmark v1

This run completed 840 deterministic generations across 120 authenticated conditions and seven matched arms.

| Arm | Exact match | F1 |
|---|---:|---:|
| Sliding context 4K | 32.50% | 34.17% |
| Structured memory 4K | 63.33% | 65.83% |
| Hybrid KG memory 4K | 75.83% | 76.25% |
| Sliding context 16K | 56.67% | 60.00% |
| Structured memory 16K | 65.00% | 68.33% |
| Hybrid KG memory 16K | 80.83% | 80.83% |
| Full Qwen context | 52.50% | 53.75% |

Hybrid KG memory improved exact match over matched sliding context by 43.33 percentage points at 4K, with a history-clustered 95% bootstrap interval of [34.17, 52.50], and by 24.17 points at 16K, with an interval of [15.00, 33.33].

Hybrid KG memory improved exact match over matched structured memory by 12.50 points at 4K, with an interval of [5.00, 20.83], and by 15.83 points at 16K, with an interval of [10.00, 22.50].

The hybrid arms used actual Scallopy admission, 946 facts committed to live Neo4j after the source fact and event lineage round-trip invariant passed, condition-isolated two-hop traversal, BGE dense retrieval, and reciprocal-rank fusion with `rrf_k=60`.
All 120 retrieval conditions executed both branches without degradation.

The model was `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` under Transformers 5.15.0 and Torch 2.6.0+cu124.
Generation was greedy, BF16, batch size one, and SDPA on the configured CUDA device.
No output reached the 32-token generation cap.

`manifest.json` authenticates clean source commit `e452b46c97aa7e39aabb281958d24d86a170f061`, every immutable prompt specification, the model files, Scallop identities, the hashed Neo4j endpoint and database configuration, the retrieval map, and all result hashes.
It does not authenticate a Neo4j server-instance identifier, server version, or GPU model.
