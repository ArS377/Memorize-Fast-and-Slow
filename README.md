# Memorize Fast and Slow

## Requirements

Use Linux x86_64, Bash, Python 3.12.9 and 3.10 with `venv`, Docker, curl, and NVIDIA GPUs with a CUDA 13-compatible driver. Run commands from the repository root. The examples assign extraction to GPU 0 and answer generation to GPU 1; adjust these device selections for your host.

```bash
python3.12 -m venv .venv-persona
.venv-persona/bin/python -m pip install 'torch==2.13.0+cu130' --index-url https://download.pytorch.org/whl/cu130
.venv-persona/bin/python -m pip install -r requirements-generation.txt
python3.12 -m venv .venv-extraction
.venv-extraction/bin/python -m pip install -r requirements-extraction.txt
python3.10 -m venv .venv-scallop
.venv-scallop/bin/python -m pip install -r requirements-scallop.txt
python3.12 -m venv .venv-support
.venv-support/bin/python -m pip install -r requirements-dense.txt
```

## Evaluation

### Table 1 — Surface A, Surface B, and Combined

The primary workflow is `kimi_ab`: both pair-conditioned Kimi surfaces, with four methods at 4K and 16K. Each surface has 120 conditions and 960 answers; together they have 240 conditions and **1,920 answers over 12 base-history clusters**, not 24 independent histories. Combined EM/F1 use unrounded pooled scores.

The complete matched A/B result artifacts and F1 values for the updated paper table have not been imported into this checkout. Step 6 generates all three table panels from validated results; it does not infer F1 from EM or silently merge older five-arm runs with newer partial runs. The historical single-surface scores are kept separately below.

**1. Download the pinned models.** The benchmark inputs are already bundled under `results/`. Saved answers and Kimi API access are not needed. These downloads also populate the cache used by the supporting experiments.

```bash
export PERSONA_QWEN_MODEL_PATH EXTRACTION_MODEL_PATH PERSONA_EMBEDDING_MODEL_PATH
PERSONA_QWEN_MODEL_PATH="$(.venv-persona/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))')"
EXTRACTION_MODEL_PATH="$(.venv-persona/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c"))')"
PERSONA_EMBEDDING_MODEL_PATH="$(.venv-persona/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"))')"
```

**2. Start two fresh Neo4j Community instances.** Choose a database password of at least eight characters. The container names and ports below must be unused.

```bash
read -rsp 'Neo4j password: ' DB_PASSWORD; printf '\n'
export NEO4J_AUTH="neo4j/$DB_PASSWORD"
docker run -d --name mfs-hybrid -p 127.0.0.1:17693:7687 -e NEO4J_AUTH neo4j:5.26.0
docker run -d --name mfs-graphiti -p 127.0.0.1:17690:7687 -e NEO4J_AUTH neo4j:5.26.0
docker logs mfs-hybrid
docker logs mfs-graphiti
```

Wait until both containers report `Started.` before continuing. Keep them running through Table 1 generation.

**3. Start Scallop and the extraction server.** These run in the background; keep this shell open. Wait for both services to finish starting, then check their endpoints.

```bash
export SCALLOP_VALIDATOR_URL=http://127.0.0.1:8081
export PERSONA_SCALLOP_ENDPOINT="$SCALLOP_VALIDATOR_URL"
export PERSONA_GRAPHITI_LLM_BASE_URL=http://127.0.0.1:8000/v1
export PERSONA_GRAPHITI_LLM_API_KEY
PERSONA_GRAPHITI_LLM_API_KEY="$(.venv-persona/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
.venv-scallop/bin/python -m services.scallop_validator_service --host 127.0.0.1 --port 8081 &
CUDA_VISIBLE_DEVICES=0 .venv-extraction/bin/vllm serve "$EXTRACTION_MODEL_PATH" --served-model-name Qwen/Qwen3-4B --max-model-len 32768 --host 127.0.0.1 --port 8000 --api-key "$PERSONA_GRAPHITI_LLM_API_KEY" &
```

After startup, run these checks; repeat them if a service is still loading:

```bash
curl -fsS "$SCALLOP_VALIDATOR_URL/health"
curl -fsS -H "Authorization: Bearer $PERSONA_GRAPHITI_LLM_API_KEY" "$PERSONA_GRAPHITI_LLM_BASE_URL/models"
```

Scallop must report `scallop_available: true`; the model listing must include `Qwen/Qwen3-4B`. Do not continue with a failed service check.

**4. Configure the benchmark connections.** Model paths came from step 1; the execution command creates fresh output, cache, index, and Graphiti build identities automatically.

```bash
export PERSONA_QWEN_MODEL_ID=Qwen/Qwen3.5-4B PERSONA_QWEN_DEVICE=cuda:1
export PERSONA_EMBEDDING_MODEL_ID=BAAI/bge-small-en-v1.5 PERSONA_EMBEDDING_DEVICE=cpu
export PERSONA_EMBEDDING_REVISION=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
export PERSONA_NEO4J_URI=bolt://127.0.0.1:17693 PERSONA_NEO4J_USER=neo4j PERSONA_NEO4J_DATABASE=neo4j
export PERSONA_NEO4J_PASSWORD="$DB_PASSWORD"
export PERSONA_GRAPHITI_NEO4J_URI=bolt://127.0.0.1:17690 PERSONA_GRAPHITI_NEO4J_USER=neo4j PERSONA_GRAPHITI_NEO4J_DATABASE=neo4j
export PERSONA_GRAPHITI_NEO4J_PASSWORD="$DB_PASSWORD"
export PERSONA_GRAPHITI_LLM_MODEL=Qwen/Qwen3-4B PERSONA_GRAPHITI_LLM_SMALL_MODEL=Qwen/Qwen3-4B
```

**5. Run both surfaces.** This rebuilds the memory stores, performs extraction and retrieval, generates 960 answers per surface, and computes each surface's metrics. A and B get separate outputs, caches, indexes, and Graphiti build IDs. Use a new run ID for each fresh experiment; do not run `--prepare` first with the same ID.

```bash
.venv-persona/bin/python -m experiments.paper_persona --mode kimi_ab --evidence-root . --output-root outputs/runsets --run-id table1 --execute
```

Outputs are in `outputs/runsets/kimi_ab/table1/surface_a/` and `surface_b/`, with generated configurations in `configs/` under the same runset. Both manifests must report `status: completed` and 960 generations. The legacy default remains `historical_a`, so specify `--mode kimi_ab` explicitly for the current paper.

**6. Verify, pool, and print Table 1.** This checks input hashes, complete eight-arm coverage, saved EM/F1 scores, and A/B model/source/configuration/checkpoint compatibility before publishing combined results. The output directory must be new and outside the source, input, and run directories.

```bash
.venv-persona/bin/python -m experiments.paper_persona_analysis --mode kimi_ab \
  --run-a outputs/runsets/kimi_ab/table1/surface_a \
  --config-a outputs/runsets/kimi_ab/table1/configs/surface_a.json \
  --run-b outputs/runsets/kimi_ab/table1/surface_b \
  --config-b outputs/runsets/kimi_ab/table1/configs/surface_b.json \
  --evidence-root . --output-dir ../mfs-table1-ab-report --table-format markdown
```

The command prints the Surface A / Surface B / Combined table and saves full metrics and paired confidence intervals in `../mfs-table1-ab-report/analysis.json`. Replace `--table-format markdown` with `--table-format latex` for the grouped Overleaf table (requires `booktabs`, `graphicx`, and `float`). Use a fresh report directory for each invocation. Combined bootstrap comparisons keep A and B together within the same 12 base-history clusters.

Do not combine historical deterministic A with modern B. Shared corpus names or similar percentages do not establish matched runs. If the new artifacts contain only hybrid/Graphiti arms, supply compatible complete eight-arm results before using this report; missing baseline rows or F1 values are not filled automatically.

#### Historical single-surface reference

These scores use the older deterministic `historical_a` corpus, not the pair-conditioned A/B corpora. They are retained for reference and are not the current three-panel Table 1.

| Method | 4K EM | 4K F1 | 16K EM | 16K F1 |
|---|---:|---:|---:|---:|
| Sliding context | 35.00 | 37.42 | 60.00 | 64.58 |
| Structured memory | 69.17 | 70.75 | 70.83 | 72.50 |
| Graphiti | 48.33 | 54.33 | 66.67 | 69.17 |
| Hybrid KG memory | **78.33** | **79.58** | **75.83** | **75.83** |

`historical_a` and `kimi_a` remain supported for explicitly requested single-surface work. The legacy `scripts/verify_paper_results.py --table1` command verifies saved historical-A results only; it does not produce the current A/B/Combined table.

### Continual Memory and Lineage (§4.2)

| Lineage method | Overall accuracy (%) | Propagated-retraction accuracy (%) |
|---|---:|---:|
| Recursive Scallop | 100.00 | 100.00 |
| One-hop rule | 66.67 | 0.00 |

| Deep-context delayed queries | Grounded accuracy (%) |
|---|---:|
| Raw sliding windows | 0.00 |
| Scallop source injection | 100.00 |

The second comparison covers both delayed query families in `deep_context_rot`, at 4K, 16K, 64K, and 128K windows. Maximum stream length is 408,761 tokens.

**1. Obtain the supporting input datasets.** Set `DATA_ROOT` to their root directory containing `results/`. These inputs remain external; a download link is not yet configured. This benchmark requires `results/synthetic_temporal_preferences_1000_v4_stream/`, not saved predictions.

```bash
export DATA_ROOT=/absolute/path/to/supporting-inputs
.venv-support/bin/python scripts/artifact_resources.py verify synthetic-continual --root "$DATA_ROOT"
```

**2. Reuse the Qwen3 tokenizer/BGE cache from Table 1 step 1 and the Scallop service from step 3.** Neo4j and the answer/extraction servers are not used by this experiment. Run with the supplied configuration unchanged and a fresh output directory:

```bash
.venv-support/bin/python -m experiments.continual_memory_benchmark --dataset "$DATA_ROOT/results/synthetic_temporal_preferences_1000_v4_stream" --config configs/continual_memory_benchmark.json --output-dir outputs/continual
```

**3. Read `outputs/continual/report.md`.** The lineage and stream-injection sections contain the comparisons above. Detailed values are in `metrics.json` under `scallop_reasoning_ablation`, `scallop_stream_injection_ablation.by_window`, and `memory_growth.stored_tokens_max`.

### Interleaved Long-Distance Memory (§4.3)

| Method | 64K availability (%) | 128K availability (%) | 64K stale intrusion (%) | 128K stale intrusion (%) |
|---|---:|---:|---:|---:|
| Sliding context | 20.86 | 47.78 | 0.00 | 0.00 |
| Bounded structured memory | 58.18 | 72.35 | 29.07 | 19.35 |

**1. Obtain `results/synthetic_temporal_preferences_1200_v5_interleaved/` under the same `DATA_ROOT` and verify it.**

```bash
.venv-support/bin/python scripts/artifact_resources.py verify synthetic-interleaved --root "$DATA_ROOT"
```

**2. Reuse the Qwen3 tokenizer cache and Scallop service.** Run the full interleaved benchmark with its default 64K/128K windows and unchanged scheduling/capacity settings, writing to a fresh directory:

```bash
.venv-support/bin/python -m experiments.interleaved_memory_benchmark_v3 --dataset "$DATA_ROOT/results/synthetic_temporal_preferences_1200_v5_interleaved" --output-dir outputs/interleaved --scallop-endpoint "$SCALLOP_VALIDATOR_URL"
```

**3. Read `outputs/interleaved/report.md`.** It reports evidence ages, availability, stale intrusion, and paired comparisons. Full metrics and the single excluded query are recorded in `metrics.json`.
