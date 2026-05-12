<!-- refreshed: 2026-05-11 -->
# Architecture

**Analysis Date:** 2026-05-11

## System Overview

NeuroSym is a research pipeline that benchmarks **knowledge-graph-augmented long-context reasoning** on LongBench-v2 multiple-choice QA. The end-to-end shape is a batch evaluation harness over a 2x3 ablation grid: `{Flat LLM, Recursive LM} x {raw context, KG context, KG + Scallop-validated context}`.

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                    Data layer                                              │
│  `download_longbench.py`  →  `data.jsonl`  (LongBench-v2 examples)        │
└─────────────────────────┬─────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    KG construction layer                                   │
│  `longbench_kg_pipeline.py`  (LongBenchKGPipeline)                        │
│     · chunk context → sentence records                                     │
│     · LLM fact extraction  (vLLM, OpenAI-compatible)                       │
│     · LLM self-reflection verification (6-question prompt)                 │
│     · supported facts → `verified_facts.jsonl`                             │
│     · supported facts → Neo4j via `Neo4jGraph.insert_facts(validate=…)`    │
│                                                                            │
│  `scallop_validator.py`  (validate_update)  Scallop rules + Python pre-   │
│     checks: contradiction / circular / alive-dead / self-ref / generic    │
│                                                                            │
│  `neo4j_graph.py`  (Neo4jGraph)                                            │
│     propose_facts → validate_update → commit_facts (MERGE)                 │
│     query_context (n-hop) + format_context_for_llm                         │
└─────────────────────────┬─────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    Retrieval layer                                         │
│  KG path:  Neo4j n-hop subgraph seeded by entities from question          │
│              `experiments/graph_context.py` (`GraphSource`)               │
│              fallback: `results/kg_builds/<session>_facts.jsonl`          │
│  Raw path: first `--raw-max-chars` of raw `context` string                 │
└─────────────────────────┬─────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    Answerer layer                                          │
│  Flat answerer (`experiments/flat_answerer.py`)                            │
│      single OpenAI chat.completion call w/ Qwen3 thinking budget          │
│  RLM answerer  (`experiments/rlm_answerer.py`)                             │
│      `rlms` package recursive completion over context, max-depth/iter      │
└─────────────────────────┬─────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    Scoring + metrics layer                                 │
│  `experiments/common.py:extract_letter`  →  A/B/C/D                       │
│  per-cell `results/cell{N}_{label}/results.jsonl`                         │
│  `experiments/aggregate.py`  →  `results/summary.csv` + bar-chart PNG     │
└───────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Dataset loader | Download + validate LongBench-v2 from HuggingFace → `data.jsonl` (atomic write, ASCII-safe) | `download_longbench.py` |
| Pilot slice helper | Stable sort-by-`_id` + slice so every cell scores the same examples | `experiments/common.py` (`iter_pilot_examples`) |
| KG extraction pipeline | Chunk → LLM extract → LLM verify → emit supported facts | `longbench_kg_pipeline.py` (`LongBenchKGPipeline`) |
| KG store + retrieval | Neo4j MERGE writes, n-hop subgraph queries, deterministic LLM-format block | `neo4j_graph.py` (`Neo4jGraph`) |
| Symbolic validator | Scallop rules + Python guards on each proposed fact: `accept` / `reject` / `replace` | `scallop_validator.py` (`validate_update`) |
| KG builder (per session) | Builds `pilot_noscallop` and `pilot_scallop` sessions, mirrors to JSONL fallback | `experiments/build_kg.py` (`build_kg`) |
| Graph source abstraction | Unified Neo4j-or-JSONL-fallback context provider for KG cells | `experiments/graph_context.py` (`GraphSource`, `open_graph_source`) |
| Flat answerer | Single OpenAI chat.completion call, Qwen3 `<think>` enabled w/ 2048-tok budget | `experiments/flat_answerer.py` (`flat_answer`) |
| RLM answerer | Wraps `rlms.RLM.completion`; isolates error string from letter extractor | `experiments/rlm_answerer.py` (`make_rlm`, `rlm_answer`) |
| Cell runner (shared CLI) | Argparse + per-example loop + result-row writer for all 6 cells | `experiments/_cli.py` (`run_cell`) |
| 6 cell scripts | One-liners delegating to `run_cell` with `kind` x `retrieval` x `session_id` | `experiments/cells/cell{1..6}_*.py` |
| Orchestrator | Build KG sessions, fan out 6 cells via subprocess, aggregate at end | `experiments/run_all.py` |
| Aggregator | Collect per-cell JSONLs → `summary.csv` + 2x3 matplotlib bar chart | `experiments/aggregate.py` |
| Standalone flat evaluator | Legacy single-mode (kg/raw) evaluator pre-dating the cell grid | `evaluate.py` |
| Standalone RLM evaluator (raw) | Legacy raw-context RLM baseline | `rlm_baseline.py` |
| Standalone RLM evaluator (KG) | Legacy KG-context RLM baseline; source of the GraphSource pattern | `rlm_graph_baseline.py` |
| Scallop CLI binary | Run `.scl` files directly for ad-hoc rule debugging | `scli`, `test.scl` |
| Letter extractor | Pull A/B/C/D from Qwen3 output (handles `<think>` strip, last-letter pref) | `experiments/common.py` (`extract_letter`) |

## Pattern Overview

**Overall:** Modular batch-evaluation pipeline with a 2x3 ablation grid.

**Key Characteristics:**
- Sequential per-example loop (no GPU-side batching) — vLLM server handles concurrency.
- Two retrieval back-ends (Neo4j live, JSONL fallback) hidden behind one `GraphSource` interface.
- Symbolic validator (`validate_update`) is invoked *inside* `Neo4jGraph.insert_facts` so the same call site can flip Scallop on/off via the `validate` kwarg — this is what cleanly separates cell 3/6 from cell 2/5.
- Schema is enforced by a single source of truth (`CELLS` table + `CELL_SCHEMA` in `experiments/common.py`); aggregator never re-derives it.
- Aggregator tolerates missing cells (renders blank rows / muted bars) so partial runs still produce a figure.
- Lazy imports (`neo4j`, `rlm`, `openai`, `scallopy`, `matplotlib`) — `experiments.common` and `experiments.aggregate` stay importable without heavy deps.

## Layers

**Data layer (`download_longbench.py`, `data.jsonl`):**
- Purpose: One-shot HuggingFace dataset pull with atomic temp-file rename and per-line field-coverage validation.
- Location: repo root.
- Depends on: `datasets` (HuggingFace).
- Used by: every entry point via `iter_pilot_examples` or `load_examples`.

**KG-construction layer (`longbench_kg_pipeline.py`, `neo4j_graph.py`, `scallop_validator.py`, `experiments/build_kg.py`):**
- Purpose: Turn raw context strings into a verified, optionally Scallop-gated, KG keyed by `(subject, sanitized_predicate, object)` triples.
- Sentence/chunk splitting is dependency-light regex in `longbench_kg_pipeline.split_text_to_sentence_like_units`.
- Two LLM passes (extract → verify) via the same `chat_json` helper.
- `Neo4jGraph` holds **all** Cypher — no other file writes Cypher.

**Retrieval layer (`experiments/graph_context.py`):**
- Purpose: Given an example, produce a textual context block + triple count.
- KG mode: seed entities via capitalized-token heuristic (`extract_seed_entities`) → Neo4j n-hop query (cap `MAX_HOPS=4`) → `format_context_for_llm`.
- Raw mode: `truncate_context(raw, raw_max_chars)` (default 32000).

**Answerer layer (`experiments/flat_answerer.py`, `experiments/rlm_answerer.py`):**
- Purpose: Take `(context, question)` → letter prediction.
- Flat: one OpenAI call, Qwen3 thinking ON, 2048 max-tokens, `enable_thinking=True` passed via `extra_body={"chat_template_kwargs": ...}`.
- RLM: stateless per-example `RLM(...).completion(prompt=context, root_prompt=question)`; default `max_tokens=64000`, `max_depth=2`, `max_iterations=10`.

**Scoring/metrics layer (`experiments/common.py`, `experiments/aggregate.py`):**
- Purpose: Per-example record → per-cell JSONL → summary CSV + accuracy bar chart.
- `extract_letter` strips `<think>` blocks and prefers the *last* standalone A/B/C/D — see "Anti-Patterns" below for why.

## Data Flow

### Primary Request Path — One Cell Through One Example

1. `experiments/run_all.py` reads CLI, writes `results/run_metadata.json`, optionally calls `build_kg(...)` for the two KG sessions.
2. Orchestrator shells out to a cell module, e.g. `python -m experiments.cells.cell6_rlm_kg_scallop ...` (`experiments/run_all.py:181`).
3. Cell delegates to `run_cell(kind, retrieval, session_id, ...)` (`experiments/cells/cell6_rlm_kg_scallop.py:13` → `experiments/_cli.py:97`).
4. `iter_pilot_examples(args.input, args.limit)` loads + sorts examples (`experiments/common.py:56`).
5. For each example:
   - **KG branch:** `GraphSource.context_for` → `Neo4jGraph.query_context` seeded with `extract_seed_entities(ex)` → `format_context_for_llm` (`experiments/graph_context.py:93`).
   - **Raw branch:** `truncate_context(ex["context"], args.raw_max_chars)` (`experiments/common.py:125`).
6. **Flat branch:** `flat_answer(client, model, context, question)` → one chat completion → `extract_letter` (`experiments/flat_answerer.py:13`).
   **RLM branch:** `make_rlm(...)` then `rlm_answer(rlm, context, question)` → `rlm.completion(prompt=context, root_prompt=question)` → `extract_letter` on `result.response` (`experiments/rlm_answerer.py:48`).
7. `write_result_row(...)` writes one schema-conformant JSON line to `results/cell{N}_{label}/results.jsonl` (`experiments/common.py:119`).
8. After all cells finish, `experiments/aggregate.py:main` collects every `results.jsonl`, computes per-cell accuracy/latency/triple counts, writes `results/summary.csv` and `results/figures/accuracy_grid.png`.

### KG-Build Path (cells 2/3/5/6 only)

1. `run_all.py:154` calls `experiments.build_kg.build_kg(session_id, validate, …)` once per session.
2. `build_kg` constructs a `Neo4jGraph(session_id=…)` and a "facts-only" `LongBenchKGPipeline` (output_path is `/dev/null`).
3. For each pilot example: `pipeline.extract_facts` (LLM JSON) → `pipeline.verify_facts` (LLM JSON) → keep `status == "supported"`.
4. `graph.insert_facts(supported, session_id=…, validate=validate)`:
   - `validate=False` → `commit_facts` directly (cells 2 and 5: session `pilot_noscallop`).
   - `validate=True` → for each `new` fact, `scallop_validator.validate_update(existing, new)` → `accept` | `reject` | `replace` (cells 3 and 6: session `pilot_scallop`).
5. After the run, `_dump_session_facts` mirrors the live graph to `results/kg_builds/<session>_facts.jsonl` so cells can fall back to JSONL if Neo4j is down.

### KG-vs-Raw Branch — Where They Diverge

| Stage | Raw path (cells 1, 4) | KG path (cells 2, 3, 5, 6) |
|-------|-----------------------|----------------------------|
| Pre-step | none | one-time `build_kg(session_id, validate)` per `(scallop_on/off)` pair |
| Context source | first N chars of `ex["context"]` | n-hop Neo4j subgraph seeded by entities mined from the question |
| Context shape | raw natural-language string | deterministic `[F1] subj -PRED-> obj` block with `support_text` + `sent_id` evidence |
| Symbolic gating | none | optional `validate_update` (Scallop) at insert time |
| Fallback | none needed | `--facts-file results/kg_builds/<session>_facts.jsonl` if Neo4j unreachable |

**State Management:**
- Stateless per example: each `RLM` is reconstructed; the Neo4j graph is the only persistent state, scoped by `session_id` property on every relationship.
- `stateless=True` on `Neo4jGraph` wipes that session on `close()`; `clear_session` and `clear_all` are admin-only.
- Per-batch: `Neo4jGraph.insert_facts` maintains a local `existing_fact_dicts` cache so that within one batch the validator sees facts proposed earlier in the same call.

## Key Abstractions

**`LongBenchKGPipeline` (`longbench_kg_pipeline.py:80`):**
- Purpose: One-shot extract+verify pipeline. Owns the OpenAI client and (optionally) a `Neo4jGraph`.
- Pattern: configured by `PipelineConfig` dataclass (`longbench_kg_pipeline.py:57`); two public methods `extract_facts` and `verify_facts` are reused by `experiments.build_kg`.
- Notable: a `neo4j_driver` property exists purely for back-compat with legacy tests that poked the raw driver.

**`Neo4jGraph` (`neo4j_graph.py:76`):**
- Purpose: Single source of truth for Cypher and graph semantics. `propose_facts` is the read-only diff; `commit_facts` is the idempotent MERGE write; `insert_facts` is the wrapper that routes through the validator.
- Pattern: context-manager (`__enter__`/`__exit__`), lazy schema bootstrap (`ensure_schema`), session-scoped reads/writes.
- Cypher templates: `commit_facts` builds the relationship type from `sanitize_predicate(pred)` (UPPER_SNAKE_CASE, illegal chars → `_`).

**`GraphSource` (`experiments/graph_context.py:80`):**
- Purpose: Abstracts "live Neo4j vs JSONL fallback" behind one `context_for(ex, hops, limit_triples, max_chars)` method.
- Factory: `open_graph_source(...)` prefers Neo4j; falls back to `--facts-file`; raises `RuntimeError` if neither works.

**`validate_update` (`scallop_validator.py:35`):**
- Purpose: Pure function `(existing_facts, new_fact) -> (decision, reason, replace_fact_id)`.
- Pattern: Python guards first (empty/self-ref/generic/redundant), then `scallopy.ScallopContext` is rebuilt per call with three rules (contradiction over `functional_pred`, circular `PART_OF`, alive/dead).
- Contradiction has confidence-based resolution: higher `confidence_score` replaces the existing fact instead of rejecting the new one.

**`RLM` wrapper (consumed, not defined here):**
- `experiments/rlm_answerer.make_rlm` builds a fresh `rlm.core.rlm.RLM` per example with `backend="openai"` (the `vllm` backend in `rlms 0.1.x` tries to spawn its own server; for an already-running vLLM use `openai` because vLLM is OpenAI-API compatible).

**Cell tables — single source of truth (`experiments/common.py:31`):**
- `CELLS` list-of-dicts: `{cell_id, label, retrieval, validator, recursion}` for all six cells. Both `run_all.py` and `aggregate.py` iterate over this list rather than hard-coding cell metadata.

## Entry Points

**`python -m experiments.run_all` (`experiments/run_all.py:95`):**
- Triggers: end-to-end ablation grid.
- Responsibilities: write metadata, build KG sessions if requested, fan out 6 cells via `subprocess.call`, run aggregator once at end.

**`python -m experiments.cells.cell{N}_*` (`experiments/cells/*.py`):**
- Triggers: a single ablation cell standalone.
- Each is a 20-line shim that calls `run_cell(...)` with cell-specific `(kind, retrieval, session_id)`.

**`python -m experiments.build_kg --session ... [--validate]` (`experiments/build_kg.py:214`):**
- Triggers: KG construction for one session.
- Idempotent: skips extraction if session already populated unless `--rebuild`.

**`python -m experiments.aggregate` (`experiments/aggregate.py:167`):**
- Triggers: rebuild `summary.csv` + `accuracy_grid.png` from existing `results/cell*/results.jsonl`.

**Legacy / standalone entry points (pre-cell-grid, kept working):**
- `python longbench_kg_pipeline.py ...` — full pipeline incl. Neo4j writes (`longbench_kg_pipeline.py:640`).
- `python evaluate.py --mode {kg,raw}` — single-mode flat evaluator (`evaluate.py:215`).
- `python rlm_baseline.py ...` — raw-context RLM baseline (`rlm_baseline.py:68`).
- `python rlm_graph_baseline.py ...` — KG-context RLM baseline with Neo4j + JSONL fallback (`rlm_graph_baseline.py:154`).
- `python download_longbench.py` — dataset fetch + validate (`download_longbench.py:134`).
- `python scallop_validator.py` — runs an in-file unit-test harness (`scallop_validator.py:140`).
- `./scli test.scl` — Scallop CLI for ad-hoc `.scl` debugging.

## Architectural Constraints

- **Threading:** Single-threaded per-example loop. Concurrency lives entirely on the vLLM server and inside `rlm.core.rlm.RLM`.
- **Global state:** None at module level. The only persistent state is the Neo4j graph (scoped by `session_id` property) and JSONL output files.
- **Python version split:** `scallopy 0.2.4` ships cp310 only; `rlms >= 0.1.1` requires py3.11+. The repo's `requirements.txt` documents two coexistence options (single env with `--ignore-requires-python`, or split envs); `experiments.common` is intentionally dep-light so the aggregator works in either env.
- **Neo4j version:** Schema statements assume Neo4j 5.x relationship-property indexes; older versions degrade gracefully via try/except in `ensure_schema` (`neo4j_graph.py:155`).
- **Hop limit:** `MAX_HOPS = 4` in `neo4j_graph.py:47` caps the n-hop query depth regardless of caller input.
- **Result-row schema is frozen:** every cell writes the exact keys in `CELL_SCHEMA` (`experiments/common.py:16`); missing keys default to `None`. The aggregator depends on this.
- **`extra_body` Qwen3 thinking flag:** flat answerer requires a vLLM server that forwards `extra_body.chat_template_kwargs.enable_thinking`. Pure OpenAI cloud will silently drop it.

## Anti-Patterns

### `extract_letter` returning the FIRST A/B/C/D character

**What happens:** Older versions of `extract_letter` (still present in `rlm_baseline.py:58`) iterate forward over `text.upper()` and return the first A/B/C/D character they see.
**Why it's wrong:** With Qwen3 thinking mode enabled, the model emits `<think>Analyze the question. Answer the following...</think>` before its actual answer. The first letter token in that prefix is virtually always `A` (from "Analyze" / "Answer" / "About"), so every example silently collapses to `pred='A'`.
**Do this instead:** Use `experiments/common.py:84` `extract_letter` — it (1) regex-matches `Answer: X` first, (2) strips `<think>...</think>`, (3) prefers the *last* standalone A/B/C/D token, (4) only as a final fallback walks the string in reverse. See commit `777d195`.

### Running `extract_letter` on a Python exception string

**What happens:** `rlm_baseline.py:146` calls `extract_letter(error_str)` on a caught exception's `str(e)` so it can record a "partial" prediction.
**Why it's wrong:** Error messages routinely contain stray A/B/C/D characters ("ConnectError: 400 Bad Request", "Anthropic", "MaxIterError", etc.), so a failed call gets a fabricated letter prediction. This is the bug that made cell 4 collapse to `pred='D'`.
**Do this instead:** `experiments/rlm_answerer.py:65` returns `("", msg, msg)` on exception — empty letter, raw error in `raw_answer`, error string set so the aggregator excludes the row from `n_answered`.

### Returning `str(result)` from an `RLMChatCompletion`

**What happens:** Casting the `rlms 0.1.x` `RLMChatCompletion` object via `str(...)` emits the class repr (no A/B/C/D anywhere), and the legacy baselines do exactly this (`rlm_baseline.py:140`).
**Why it's wrong:** Even when the RLM completes successfully, the prediction is empty because the letter extractor never sees the actual response string.
**Do this instead:** `experiments/rlm_answerer.py:63` uses `getattr(result, "response", None) or str(result)` so the response field is preferred and `str(result)` is just the last-ditch fallback.

### RLM `--max-tokens` set to 32000

**What happens:** `experiments/run_all.py:116` originally defaulted to 32000.
**Why it's wrong:** Qwen3 thinking-mode trajectories with 6 RLM iterations routinely overshoot 33k summed tokens, so the run errors out partway.
**Do this instead:** Match `rlm_baseline.py`'s default of 64000 (`experiments/_cli.py:72` and commit `b06fa93` / `352ab2c`).

### Hand-rolling Cypher anywhere other than `Neo4jGraph`

**What happens:** Tempting to embed a small `MATCH ... RETURN` in a script that needs to peek at the graph.
**Why it's wrong:** Predicate sanitization, session_id scoping, schema bootstrap, and the relationship-type quoting are all baked into `Neo4jGraph` methods. Hand-rolled Cypher will skip predicate sanitization and produce un-mergeable relationships.
**Do this instead:** Add a method to `neo4j_graph.py:Neo4jGraph` (see `_facts_count` and `_dump_session_facts` in `experiments/build_kg.py:29`, which use `graph._session()` but call standard Cypher patterns).

## Error Handling

**Strategy:** Defensive per-example try/except; errors are recorded as a row, never raised out of the loop.

**Patterns:**
- `experiments/_cli.py:185` wraps the answerer in try/except, stores `error = str(e)`, sets `predicted = ""`, and the row is still written so partial runs produce usable CSVs.
- `LongBenchKGPipeline.chat_json` falls back from JSON mode to plain mode automatically when the first call raises (`longbench_kg_pipeline.py:245`).
- `Neo4jGraph.ensure_schema` swallows per-statement failures (older Neo4j may lack relationship property indexes).
- `validate_update` rejects rather than throws on malformed inputs (`scallop_validator.py:51`).
- `parse_json_object` (`longbench_kg_pipeline.py:566`) strips ```json``` fences, then falls back to the largest `{...}` substring.
- Aggregator catches per-cell read failures, logs them, and leaves the row blank (`experiments/aggregate.py:82`).

## Cross-Cutting Concerns

**Logging:** Plain `print(..., file=sys.stderr)` everywhere; no `logging` framework. `RLMLogger` is delegated for RLM-internal traces (writes to `./rlm_logs` or `./rlm_logs_graph` or `./rlm_logs_ablation`).

**Validation (data-level):** `download_longbench.py:85` does a full read-back pass and per-field-coverage report. Pipeline-internal validation is `scallop_validator.validate_update` plus `clean_fact` (`longbench_kg_pipeline.py:503`).

**Authentication:** None inside the codebase. Neo4j password is taken from `--neo4j-password` CLI or `NEO4J_PASSWORD` env. vLLM API key defaults to `EMPTY` (local).

---

*Architecture analysis: 2026-05-11*
