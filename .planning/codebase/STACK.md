# Technology Stack

**Analysis Date:** 2026-05-11

## Languages

**Primary:**
- Python 3.10 — main language for the pipeline, baselines, and ablation cells. README pins **Python 3.9** for the scallopy wheel, but `requirements.txt` notes that the pinned `scallopy 0.2.4` wheel is `cp310` and `rlms>=0.1.1` requires Python `>=3.11`; the working compromise (per the requirements.txt comment block) is Python 3.10 with `pip install --ignore-requires-python rlms==0.1.1`, or two split envs.

**Secondary:**
- Scallop (`.scl` DSL) — declarative logic rules used by `scallop_validator.py`. Example: `test.scl` defines a Fibonacci relation; production rules are embedded as strings inside `scallop_validator.py` (`contradiction`, `circular_containment`, `alive_dead_conflict`).
- Cypher — Neo4j query language; queries are inline Python f-strings throughout `neo4j_graph.py` (MERGE, MATCH, CREATE CONSTRAINT/INDEX).

## Runtime

**Environment:**
- CPython 3.10 (x86_64 build required on Apple Silicon — `CONDA_SUBDIR=osx-64`, run under Rosetta 2). The README's installation flow uses `miniforge` + `conda` to create the `scallop-env-x86` env.
- No `.python-version`, `pyproject.toml`, `setup.py`, or `Cargo.toml`. No `.nvmrc`.

**Package Manager:**
- `pip3` (per README). No `uv.lock`, no `poetry.lock`, no `requirements.lock`.
- Lockfile: missing — `requirements.txt` pins exact versions for most deps but is not a true lockfile (no hashes).

## Frameworks

**Core:**
- `openai` `>=1.0.0,<2.0.0` — used as an OpenAI-compatible client pointing at a local vLLM server. Entry: `from openai import OpenAI` in `longbench_kg_pipeline.py:48`, `evaluate.py:30`, `experiments/_cli.py:122`, `experiments/flat_answerer.py`.
- `neo4j` `>=5.0.0,<6.0.0` — Bolt driver. Entry: `from neo4j import GraphDatabase` in `neo4j_graph.py:39` (lazy import — `Neo4jGraph` constructor raises if missing).
- `scallopy` `0.2.4` (cp310 manylinux wheel from `https://github.com/scallop-lang/scallop/releases/download/0.2.4/scallopy-0.2.4-cp310-cp310-manylinux_2_27_x86_64.whl`) — Scallop context API. Entry: `import scallopy` in `scallop_validator.py:1`.
- `rlms` `>=0.1.1` (the `rlm` import name) — recursive language model. Entry: `from rlm.core.rlm import RLM` and `from rlm.logger.rlm_logger import RLMLogger` in `rlm_baseline.py:30-31`, `rlm_graph_baseline.py:36-37`, `experiments/rlm_answerer.py:31-32`.
- `pydantic` `2.13.0` / `pydantic_core` `2.46.0` — declared in `requirements.txt` (transitive to openai/rlms; not directly imported by repo code).

**Testing:**
- Plain `unittest.TestCase` (stdlib). Test files under `tests/` (`test_pipeline_smoke.py`, `test_validation.py`, `test_neo4j_format.py`, `test_neo4j_graph.py`) are invoked directly via `python3 tests/<name>.py`. No `pytest`, no `jest.config`, no `vitest.config`.

**Build/Dev:**
- None — no Dockerfile in the repo (Neo4j is started from the official `neo4j:5` image per README). No CI config (no `.github/`, no `.gitlab-ci.yml`).
- `matplotlib` (and transitively `numpy`) — used only by `experiments/aggregate.py:107` for the accuracy-grid figure. Not listed in `requirements.txt` (must be installed separately for figure rendering).
- `datasets` (HuggingFace) — required for `download_longbench.py:42` but **not listed** in `requirements.txt`; README instructs `pip install datasets`.

## Key Dependencies

**Critical:**
- `openai>=1.0.0,<2.0.0` — every LLM call (fact extraction, verification, flat answerer, evaluation) goes through `OpenAI(base_url=..., api_key=...)` against the vLLM/Ollama OpenAI-compatible endpoint. `requirements.txt` lines 11-14 contain a fix-note explaining the original `openai==2.31.0` pin was invalid.
- `neo4j>=5.0.0,<6.0.0` — backbone of the knowledge graph. All Cypher lives in `neo4j_graph.py`. `requirements.txt` lines 22-23 note the original `neo4j==6.1.0` pin was invalid.
- `scallopy 0.2.4` — symbolic validator core. Without this wheel the Scallop validation path in `scallop_validator.py` cannot run.
- `rlms>=0.1.1` (imports as `rlm`) — required for cells 4/5/6 and `rlm_baseline.py` / `rlm_graph_baseline.py`. `requirements.txt` lines 24-31 document the Python-version conflict between scallopy (cp310-only) and rlms (Python ≥3.11).

**Infrastructure:**
- `httpx 0.28.1` / `httpcore 1.0.9` / `h11 0.16.0` / `anyio 4.13.0` / `sniffio 1.3.1` — HTTP stack under `openai`.
- `tqdm 4.67.3` — progress bars (declared in `requirements.txt`, used indirectly via `datasets`).
- `pytz 2026.1.post1`, `certifi 2026.2.25`, `idna 3.11`, `colorama 0.4.6`, `jiter 0.14.0`, `distro 1.9.0`, `annotated-types 0.7.0`, `typing-inspection 0.4.2`, `typing_extensions 4.15.0` — transitive support libraries (per `requirements.txt`).
- `scli` — local Scallop CLI binary (arm64, 6.5 MB, committed at repo root). Runs `.scl` files directly. README:62-66 documents the one-time `xattr -c`, `codesign -s -`, `chmod +x` setup.

## Configuration

**Environment:**
- No `.env`, `.env.example`, or `config/` directory in the repo (verified via `ls`). Environment variables are read inline via `os.getenv(...)` at CLI parse time (see Required env vars in INTEGRATIONS.md).
- No YAML/TOML config files.

**Build:**
- `requirements.txt` — only build/config file. The `scallopy` line uses a direct wheel URL.
- `.gitignore` excludes `data.jsonl`, `data_small.jsonl`, all `*.jsonl` (except `requirements.txt` and `tests/fixtures/*.jsonl`), `__pycache__/`, `.venv/`, `venv/`, `vllm/`, `smoke_output.jsonl`, `verified_facts_test.jsonl`.

## Platform Requirements

**Development:**
- macOS Apple Silicon supported via Rosetta 2 (`CONDA_SUBDIR=osx-64`); Linux x86_64 supported by the manylinux scallopy wheel.
- Neo4j 5.x server reachable on `bolt://localhost:7687` (Docker command in `README.md:87-89`: `docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/yourpassword neo4j:5`).
- vLLM (or any OpenAI-compatible inference server, e.g. Ollama, MLX) reachable on `http://localhost:8000/v1` (vLLM default) or `http://localhost:11434/v1` (Ollama default, used in `rlm_baseline.py:75` and `rlm_graph_baseline.py:173`) or `http://localhost:8001/v1` (vllm-metal/MLX, per `rlm_baseline.py:17`).

**Production:**
- Not deployed — this is a research codebase. README is explicit: experiments are run locally with a developer-launched vLLM and Neo4j.

## Entry-Point Scripts

| Script | Purpose | Example command |
|---|---|---|
| `download_longbench.py` | Pull `zai-org/LongBench-v2` from HuggingFace and write a validated `data.jsonl`. | `python3 download_longbench.py --out data.jsonl --limit 50` |
| `longbench_kg_pipeline.py` | Main extraction pipeline: LongBench → LLM extract → LLM verify → Neo4j commit. | `python3 longbench_kg_pipeline.py --input data.jsonl --output verified_facts.jsonl --model Qwen/Qwen3-4B --vllm-base-url http://localhost:8000/v1 --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password yourpassword --limit 10` |
| `evaluate.py` | Multiple-choice eval on LongBench-v2, mode = `kg` (facts) or `raw` (full context). | `python3 evaluate.py --data data.jsonl --facts verified_facts.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --vllm-base-url http://localhost:8000/v1 --mode kg --limit 50` |
| `rlm_baseline.py` | Recursive Language Model baseline over raw context (no KG). | `python rlm_baseline.py --input data.jsonl --output rlm_baseline_output.jsonl --backend vllm --base-url http://localhost:11434/v1 --model qwen2.5:1.5b --limit 5` |
| `rlm_graph_baseline.py` | RLM baseline with Neo4j (or `--facts-file` fallback) KG context. | `python rlm_graph_baseline.py --input data.jsonl --output rlm_graph_output.jsonl --neo4j-uri neo4j://127.0.0.1:7687 --neo4j-user neo4j --neo4j-password <REDACTED> --backend vllm --base-url http://localhost:11434/v1 --model qwen2.5:1.5b --limit 3` |
| `_demo_neo4j.py` | One-shot probe against a populated Neo4j (find_conflicts + 1/2-hop query). | `python _demo_neo4j.py` |
| `scallop_validator.py` | When run as `__main__`, prints validator decisions on 7 hand-crafted test cases. | `python3 scallop_validator.py` |
| `test.py` | Scratch vLLM smoke test (`Qwen/Qwen2.5-1.5B-Instruct` via the `vllm` python package, not the HTTP API). | `python test.py` |
| `experiments/run_all.py` | Orchestrator for the 6-cell ablation grid (flat/RLM × raw/KG/KG+Scallop). | `python -m experiments.run_all --cells all --limit 50 --model Qwen/Qwen3-4B --vllm-base-url http://localhost:8000/v1 --neo4j-password $NEO4J_PASSWORD` |
| `experiments/build_kg.py` | Build a named Neo4j session (`pilot_noscallop` or `pilot_scallop`) and mirror committed facts to `results/kg_builds/<session>_facts.jsonl`. | `python -m experiments.build_kg --session pilot_scallop --validate --limit 50 --neo4j-password $NEO4J_PASSWORD` |
| `experiments/aggregate.py` | Roll the 6 cells' `results.jsonl` into `results/summary.csv` + `results/figures/accuracy_grid.png`. | `python -m experiments.aggregate --results-dir results` |
| `experiments/cells/cell{1..6}_<label>.py` | Individual ablation cells (`flat_raw`, `flat_kg_noscallop`, `flat_kg_scallop`, `rlm_raw`, `rlm_kg_noscallop`, `rlm_kg_scallop`). | `python -m experiments.cells.cell6_rlm_kg_scallop --limit 50` |
| `./scli` | Scallop CLI binary (committed). Run `.scl` files directly. | `./scli test.scl` |

---

*Stack analysis: 2026-05-11*
