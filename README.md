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

## Anonymous review package

Build the reviewer ZIP from this checkout using Python's standard library:

```bash
python3 -B scripts/package_submission.py --output ../supplementary_code.zip
```

Use a new output filename each time. The command refuses to overwrite files, verifies the 41 bundled input files, and packages source, tests, and those inputs under `supplementary_code/`. It excludes Git metadata, local environments, caches, and run outputs. ZIP comments are empty, timestamps are fixed, and no owner metadata is copied. GitHub's **Download ZIP** may attach a commit ID in the archive comment; use this packaging command for the review upload instead.

The command checks for personal home-directory paths, unapproved GitHub repository links, and stored Git commit IDs. These checks are not a guarantee of anonymity: review names, affiliations, document contents, and any separately supplied artifacts before uploading. Third-party model revisions, dependency versions, and original bundled-data checksums are retained.

`tests/fixtures/historical_persona_manifest.json` is an explicitly anonymized test fixture: its dataset paths are relative to the project root and its historical Git identity is removed. Its test checksum identifies this sanitized copy, not the original execution record. The configuration test supplies a synthetic source identity in memory; production provenance checks are unchanged. No bundled dataset or original external manifest was rewritten.

### Running from the extracted ZIP

Extract the ZIP into any directory, enter `supplementary_code/`, and configure archive verification in the shell used for the experiments:

```bash
export NEUROSYM_SOURCE_MANIFEST=source_manifest.json
export NEUROSYM_SOURCE_MANIFEST_SHA256="$(cat source_manifest.sha256)"
python3 -B -m neurosym.application.source_provenance --verify .
python3 -B scripts/artifact_resources.py verify bundled-persona --root .
```

`source_manifest.json` records source-file checksums; `package_manifest.json` additionally inventories tests and bundled inputs. The adjacent digest detects changes relative to the packaged manifest, not the identity of its author. Keep the archive SHA-256 printed by the packaging command separately when distributing it. Reviewers do not need the original Git checkout. Leave both environment variables set when following the evaluation commands below.

This ZIP contains the persona benchmark inputs, not the external saved A/B runs or the supporting continual/interleaved datasets. Anonymous access to those missing artifacts must be supplied separately; packaging alone does not make every reported experiment reproducible.

## Evaluation

### Table 1 — Surface A, Surface B, and Combined

The primary workflow is `kimi_ab`: both pair-conditioned Kimi surfaces, with four methods at 4K and 16K. Each surface has 120 conditions and 960 answers; together they have 240 conditions and **1,920 answers over 12 base-history clusters**, not 24 independent histories. Combined EM/F1 use unrounded pooled scores.

These are the matched eight-arm A/B results reported in the paper. Full run artifacts remain external to this package; an anonymous download is not yet configured. There is one stochastic Graphiti ingestion realization per surface; fresh runs may produce different scores. The historical single-surface reference is separate below.

**Exact match (%)**

| Method | Surface A 4K | Surface A 16K | Surface B 4K | Surface B 16K | Combined 4K | Combined 16K |
|---|---:|---:|---:|---:|---:|---:|
| Sliding context | 32.50 | 50.00 | 28.33 | 55.83 | 30.42 | 52.92 |
| Structured memory | 62.50 | 60.83 | 59.17 | 63.33 | 60.83 | 62.08 |
| Graphiti | 40.83 | 52.50 | 40.00 | 58.33 | 40.42 | 55.42 |
| Hybrid KG memory | **69.17** | **71.67** | **65.83** | **66.67** | **67.50** | **69.17** |

**Token F1 (%)**

| Method | Surface A 4K | Surface A 16K | Surface B 4K | Surface B 16K | Combined 4K | Combined 16K |
|---|---:|---:|---:|---:|---:|---:|
| Sliding context | 35.83 | 55.64 | 31.44 | 59.19 | 33.64 | 57.42 |
| Structured memory | 64.50 | 64.00 | 61.03 | 65.19 | 62.76 | 64.60 |
| Graphiti | 47.28 | 57.92 | 48.15 | 61.64 | 47.71 | 59.78 |
| Hybrid KG memory | **71.19** | **72.57** | **68.86** | **67.92** | **70.03** | **70.24** |

**Combined paired exact-match effects (percentage points)**

| Comparison | Budget | EM gain | 95% CI |
|---|---|---:|---|
| Hybrid - sliding | 4K | +37.08 | [30.42, 44.58] |
| Hybrid - structured | 4K | +6.67 | [-0.42, 15.00] |
| Hybrid - Graphiti | 4K | +27.08 | [19.17, 35.42] |
| Hybrid - sliding | 16K | +16.25 | [7.08, 25.00] |
| Hybrid - structured | 16K | +7.08 | [1.25, 13.33] |
| Hybrid - Graphiti | 16K | +13.75 | [5.42, 21.25] |

Intervals use 2,000 bootstrap resamples of the 12 shared histories (seed 73), keeping A and B together. The 4K hybrid-versus-structured EM interval includes zero; all other listed EM intervals exclude zero.

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

Use the matched eight-arm A/B outputs for every method. Do not substitute older five-arm baseline results or combine historical deterministic A with modern B; shared corpus names or similar percentages do not establish matched runs.

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
