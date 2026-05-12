# External Integrations

**Analysis Date:** 2026-05-11

## APIs & External Services

**LLM Inference (OpenAI-compatible HTTP):**
- **vLLM server** (primary, local) — talks the OpenAI Chat Completions API. Default base URL `http://localhost:8000/v1`. Started per `README.md:93-95`: `vllm serve Qwen/Qwen3-4B --host 0.0.0.0 --port 8000`.
  - SDK/Client: `openai>=1.0.0,<2.0.0` via `from openai import OpenAI` in `longbench_kg_pipeline.py:48`, `evaluate.py:30`, `experiments/_cli.py:122`, `experiments/flat_answerer.py`.
  - Auth: `--api-key` flag, defaults to literal string `"EMPTY"` (vLLM does not validate). Env var: `VLLM_API_KEY` (used as default in `longbench_kg_pipeline.py:600` and `experiments/_cli.py:37`).
  - Endpoint used: `/v1/chat/completions` (with optional `response_format={"type": "json_object"}` JSON mode in `longbench_kg_pipeline.py:243` — graceful fallback if model doesn't support it).
- **Ollama** (alternative, local) — same OpenAI-compatible surface at `http://localhost:11434/v1`. Used by `rlm_baseline.py:75` and `rlm_graph_baseline.py:173` as the default `--base-url`.
- **vllm-metal / MLX** (Apple Silicon path) — `http://localhost:8001/v1`, documented in `rlm_baseline.py:14-19` with model `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
- **OpenAI / Anthropic / Azure / Gemini / LiteLLM** — exposed as `--backend` choices in `rlm_baseline.py:73` (`["openai", "vllm", "litellm", "anthropic", "azure_openai", "gemini"]`). These are passed straight through to `rlm.core.rlm.RLM(backend=...)` and are not directly wired in this repo (no calls bypass `openai.OpenAI` / `rlm.RLM`).
- **Note on `--backend vllm` vs `--backend openai`**: `experiments/_cli.py:62-66` documents that with `rlms 0.1.x`, `backend="vllm"` tries to spawn vLLM via the python `vllm` package and ignores `base_url`; to talk to an already-running vLLM HTTP server, ablation cells default `--backend openai` (vLLM is OpenAI-API compatible).

**Direct vLLM Python (non-HTTP) call:**
- `test.py:10` imports `from vllm import LLM, SamplingParams` and constructs an in-process `LLM(model="Qwen/Qwen2.5-1.5B-Instruct")`. This bypasses the HTTP server and requires the `vllm` Python package installed locally; it also munges `sys.path` to prefer a local `vllm/` repo dir if present (`test.py:1-5`).

## Data Storage

**Databases:**
- **Neo4j 5.x** (Bolt) — the dynamic KG store. All access via `neo4j_graph.Neo4jGraph` (`neo4j_graph.py:76-473`).
  - Connection: `bolt://localhost:7687` (HTTP browser at `http://localhost:7474`). Env vars: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `NEO4J_SESSION_ID` (defaults read in `longbench_kg_pipeline.py:609-613` and `experiments/_cli.py:51-53` / `experiments/run_all.py:103-105`).
  - Client: official `neo4j` Python driver (`from neo4j import GraphDatabase` in `neo4j_graph.py:39`).
  - Schema (`neo4j_graph.py:142-153`): unique constraint on `Entity.name`; indexes on relationship properties `fact_id`, `example_id`, `session_id`; fulltext index `entity_name_fulltext` on `Entity.name`.
  - Session scoping: every relationship has `session_id` (default `"default"`, see `DEFAULT_SESSION_ID = "default"` at `neo4j_graph.py:46`). The orchestrator uses two fixed sessions: `pilot_noscallop` (cells 2 & 5) and `pilot_scallop` (cells 3 & 6) — `experiments/run_all.py:28-29`.
  - Stateless mode: `Neo4jGraph(stateless=True, session_id=...)` wipes the session subgraph on `close()` (`neo4j_graph.py:122-132`).
  - Hard-coded password in committed demo: `_demo_neo4j.py:6` uses literal `"<REDACTED>"`; `rlm_graph_baseline.py:9` shows it as an example CLI value. Treat as throwaway dev credential.

**File Storage:**
- Local filesystem only. Pipeline writes JSONL outputs (e.g. `verified_facts.jsonl`, `rlm_baseline_output.jsonl`, `results/cell{N}_*/results.jsonl`, `results/kg_builds/<session>_facts.jsonl`, `results/summary.csv`, `results/figures/accuracy_grid.png`, `results/run_metadata.json`).
- `download_longbench.py:55-73` uses an atomic temp-file + rename pattern.

**Caching:**
- None — no Redis, no on-disk cache. The KG itself acts as a persistent extracted-fact cache: `experiments/build_kg.py:106-115` skips re-extraction when the target Neo4j session already has facts unless `--rebuild` is passed.

## Authentication & Identity

**Auth Provider:**
- None for the application itself (single-user research code).
- Neo4j Bolt auth: `auth=(user, password)` tuple passed to `GraphDatabase.driver(...)` at `neo4j_graph.py:108`. User/password come from CLI flags (`--neo4j-user`, `--neo4j-password`) or env vars (`NEO4J_USER`, `NEO4J_PASSWORD`).
- vLLM/Ollama: bearer token; defaults to literal `"EMPTY"`.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, no Rollbar, no OTel SDK).

**Logs:**
- `RLMLogger` from `rlm.logger.rlm_logger` — RLM-internal trace logging. Log dirs: `./rlm_logs` (`rlm_baseline.py:84`), `./rlm_logs_graph` (`rlm_graph_baseline.py:180`), `rlm_logs_ablation` (`experiments/_cli.py:73`).
- Otherwise plain `print(..., file=sys.stderr)` throughout. No `logging` module configuration.

## CI/CD & Deployment

**Hosting:**
- None — local-only research code. No `.github/workflows/`, no `Dockerfile`, no Procfile, no Terraform.

**CI Pipeline:**
- None. Tests are run manually: `python3 tests/test_pipeline_smoke.py` etc. (README:78-83).

## Environment Configuration

**Required env vars:**
| Variable | Where read | Default | Notes |
|---|---|---|---|
| `VLLM_API_KEY` | `longbench_kg_pipeline.py:600`, `experiments/_cli.py:37`, `experiments/build_kg.py:203`, `experiments/run_all.py:102` | `"EMPTY"` | vLLM ignores; the literal `EMPTY` string works locally. |
| `NEO4J_URI` | `longbench_kg_pipeline.py:609`, `experiments/_cli.py:51`, `experiments/build_kg.py:204`, `experiments/run_all.py:103` | `bolt://localhost:7687` (in run_all / _cli) | Required when KG cells are run. |
| `NEO4J_USER` | `longbench_kg_pipeline.py:610`, `experiments/_cli.py:52`, `experiments/build_kg.py:205`, `experiments/run_all.py:104` | `"neo4j"` (in run_all / _cli) | — |
| `NEO4J_PASSWORD` | `longbench_kg_pipeline.py:611`, `experiments/_cli.py:53`, `experiments/build_kg.py:206`, `experiments/run_all.py:105` | none | Hard requirement for KG cells; `experiments/build_kg.main()` errors out via `parser.error(...)` if missing (`experiments/build_kg.py:218-219`). |
| `NEO4J_DATABASE` | `longbench_kg_pipeline.py:612` | `None` | Optional Neo4j 4+ multi-db name. |
| `NEO4J_SESSION_ID` | `longbench_kg_pipeline.py:613` (CLI flag `--session-id`) | `None` → resolves to `"default"` inside `Neo4jGraph` (`neo4j_graph.py:110`) | Tag for writes/queries; required when `--stateless` is set (`neo4j_graph.py:105-106`). |
| `VLLM_LOGGING_LEVEL` | `test.py:8` | `"ERROR"` | Only used by the scratch `test.py` to mute vLLM INFO logs. |

No HuggingFace token (`HF_TOKEN`) is consumed — `download_longbench.py` uses `datasets.load_dataset(...)` against the public `zai-org/LongBench-v2` dataset without auth. No `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` is consumed by any script (the backends are only routed through `rlms` and bypassed in practice by the local vLLM/Ollama path).

**Secrets location:**
- Not centralized. Plaintext defaults (`"EMPTY"`, `"<REDACTED>"`) appear inline in source / CLI examples. No `.env`, no secret manager, no `keyring` usage.

## Webhooks & Callbacks

**Incoming:** None — no HTTP server is exposed by this repo.

**Outgoing:** None — all "external" calls go to localhost endpoints (vLLM, Ollama, Neo4j Bolt, HuggingFace dataset download).

## Datasets Sourced

- **LongBench-v2** — `zai-org/LongBench-v2`, `split="train"`, loaded in `download_longbench.py:47` via `datasets.load_dataset(...)`. Output is one JSON object per line into `data.jsonl`. Expected fields per row (`download_longbench.py:33-37`): `_id, domain, sub_domain, difficulty, length, question, choice_A..choice_D, answer, context`.
  - Validation pass at `download_longbench.py:85-131` reports per-field missing counts.
  - Only LongBench-v2 is wired in this repo. HotpotQA / MuSiQue / 2WikiMultihopQA are **not** integrated — though `flatten_context_to_sentence_records` in `longbench_kg_pipeline.py:299-358` does explicitly parse a HotpotQA-like `[[title, [sentences...]], ...]` context shape as a forward-compat hook.

## Model Checkpoints Referenced

| Checkpoint | Where | Role |
|---|---|---|
| `Qwen/Qwen3-4B` | `README.md:93,102,115`, `longbench_kg_pipeline.py:16,22,29`, `experiments/_cli.py:35`, `experiments/build_kg.py:201`, `experiments/run_all.py:100` | Default extractor / verifier / answerer for the main pipeline and the 6-cell ablation. Cells use Qwen3 thinking mode (`experiments/flat_answerer.py:40`: `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`) with a 2048-token budget. |
| `Qwen/Qwen2.5-1.5B-Instruct` | `evaluate.py` example flag value, `test.py:13` | Lightweight evaluation / scratch model. |
| `qwen2.5:1.5b` | `rlm_baseline.py:77`, `rlm_graph_baseline.py:174` | Ollama tag default for the RLM baselines. |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | `rlm_baseline.py:18` (docstring example) | Apple Silicon MLX/vLLM-metal path. |
| `gpt-3.5-turbo` | `TESTING.md:111,121` | Mentioned only in TESTING.md examples — no live call site in current code. |

## Local Model Servers / vLLM Endpoints

| Endpoint | Default base URL | Where set |
|---|---|---|
| vLLM (Qwen3 path) | `http://localhost:8000/v1` | `longbench_kg_pipeline.py:599`, `evaluate.py:220`, `experiments/_cli.py:36`, `experiments/build_kg.py:202`, `experiments/run_all.py:101` |
| Ollama | `http://localhost:11434/v1` | `rlm_baseline.py:75`, `rlm_graph_baseline.py:173` |
| vllm-metal / MLX | `http://localhost:8001/v1` | `rlm_baseline.py:17` (docstring) |
| Neo4j Bolt | `bolt://localhost:7687` | `README.md:88`, `experiments/_cli.py:51`, `experiments/run_all.py:103`; alt form `neo4j://127.0.0.1:7687` in `_demo_neo4j.py:5`, `rlm_graph_baseline.py:6` |
| Neo4j Browser HTTP | `http://localhost:7474` | `README.md:88` |

## Auth Tokens Expected

- No `HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AZURE_OPENAI_API_KEY`, or `LITELLM_API_KEY` is consumed by any script in this repo. The backend list in `rlm_baseline.py:73` is forwarded to `rlms` but every documented invocation uses the local OpenAI-compatible path with `api_key="EMPTY"`.
- Neo4j credentials are the only real secret; passed via `--neo4j-password` or `NEO4J_PASSWORD`. Demo files (`_demo_neo4j.py`, `rlm_graph_baseline.py` usage example) hard-code `"<REDACTED>"`.

---

*Integration audit: 2026-05-11*
