# Testing Patterns

**Analysis Date:** 2026-05-11

## Honest Summary

This is a research codebase. There is **no test runner configured, no CI, no
coverage gate, and no formal unit-test layer**. The `tests/` directory contains
four files that mix three different styles: one is a true `pytest`-discoverable
unit suite (`test_neo4j_graph.py`), the other three are imperative
`assert`-and-print scripts you launch by hand. The repo root also carries a
legacy smoke script (`test.py`) that is unrelated to the KG pipeline and only
checks that vLLM can be imported.

The real "test" of every change is an **eval harness on LongBench-v2**: you run
one or all six ablation cells on a small `--limit` slice, eyeball the per-cell
accuracy printed to stderr, and look at `results/summary.csv`. Regression is
defined empirically — by accuracy dropping, by `predicted` collapsing to a
single letter, or by the `error` column filling up in `results/cellN_*/results.jsonl`.

## Test Framework

- **Runner:** None. There is no `pytest.ini`, `pyproject.toml`, `tox.ini`, or `conftest.py` anywhere in the repo. `pytest` is **not** in `requirements.txt:1-33`.
- **Discovery:** Two of the four files in `tests/` define their own `main()` and call `sys.exit(...)`; you run them as plain Python scripts. `tests/test_neo4j_graph.py` happens to be pytest-compatible (functions named `test_*`, plain `assert`s) but only by convention — no command-line invocation is documented.
- **Assertion library:** Stdlib `assert` only. No `unittest.TestCase` subclasses; no `pytest.fixture`.
- **Mocking:** `unittest.mock.MagicMock` / `unittest.mock.patch` are imported directly. See `tests/test_neo4j_graph.py:14`, `tests/test_pipeline_smoke.py:19`.

## Test File Organization

**Locations and shapes:**

| File | Lines | Style | How to run |
|------|------|-------|-----------|
| `tests/test_neo4j_graph.py` | 390 | pytest-style: top-level `test_*` functions, `assert`s, hand-rolled `MockDriver`/`MockSession`/`MockResult` (`tests/test_neo4j_graph.py:29-79`) | `pytest tests/test_neo4j_graph.py` — but pytest is not in `requirements.txt`; in practice this file is read, not executed |
| `tests/test_pipeline_smoke.py` | 172 | Imperative script: patches `sys.modules["openai"]`, instantiates `LongBenchKGPipeline`, runs it on 2 rows from `data.jsonl` with a fake OpenAI client, then asserts on the output JSONL | `python tests/test_pipeline_smoke.py` — see TESTING.md (project) line 133 |
| `tests/test_validation.py` | 221 | Imperative script: each test is `def test_*()` returning `bool`; `main()` iterates them and `sys.exit(0/1)`. No pytest. | `python tests/test_validation.py` |
| `tests/test_neo4j_format.py` | 239 | Same style as `test_validation.py`: `bool`-returning function tests + a manual runner | `python tests/test_neo4j_format.py` |
| `tests/fixtures/verified_facts.sample.jsonl` | — | Static fixture: ≥8 facts in the standard schema, loaded by `tests/test_neo4j_graph.py:99-108` | (data only) |
| `test.py` (repo root) | 20 | **Not a test of NeuroSym at all.** Imports `vllm` from a sibling `vllm/` checkout, generates one completion. Used to verify the vLLM env is healthy. | `python test.py` |

There is also `TESTING.md` (repo root) — a hand-written status report describing what was tested historically; it is documentation, not a runnable suite.

## Test Style

**Three different conventions coexist:**

1. **Hand-rolled mock driver** (`tests/test_neo4j_graph.py`). A `MockDriver` exposes a `queries: List[tuple]` log of every `(cypher, kwargs)` pair; assertions are made on the captured query text. Real `Neo4jGraph` is constructed by patching out `ng.GraphDatabase` with `unittest.mock.patch.object` and then swapping a fresh `MockDriver` onto `graph._driver` (`tests/test_neo4j_graph.py:82-96`). Example assertion:

```python
# tests/test_neo4j_graph.py:114-120
def test_ensure_schema_emits_constraints_and_indexes() -> None:
    graph = make_graph()
    graph.ensure_schema()
    queries = [q for q, _ in graph._driver.queries]
    joined = "\n".join(queries)
    assert "CREATE CONSTRAINT entity_name_unique" in joined
    assert "FOR (e:Entity) REQUIRE e.name IS UNIQUE" in joined
```

2. **Smoke-script with `sys.modules` injection** (`tests/test_pipeline_smoke.py`). Before importing the pipeline, the script stuffs a fake `openai` module into `sys.modules` so the import succeeds even without the real package (`tests/test_pipeline_smoke.py:76-79`). It then loads two real rows from `data.jsonl`, builds a `PipelineConfig` with `neo4j_uri=None` (no Neo4j writes), and feeds canned JSON responses through a `MagicMock` client:

```python
# tests/test_pipeline_smoke.py:55-69 (mock client builder)
def _make_mock_client(responses):
    client = mock.MagicMock()
    call_iter = iter(responses)
    def fake_create(**kwargs):
        text = next(call_iter)
        choice = mock.MagicMock()
        choice.message.content = text
        resp = mock.MagicMock()
        resp.choices = [choice]
        return resp
    client.chat.completions.create.side_effect = fake_create
    return client
```

   Assertions live at module level (`tests/test_pipeline_smoke.py:140-167`) and include a pinned regression check on a previously-fixed bug ("FIX 3" — the `question` field must not be empty on Neo4j-bound facts).

3. **Procedural assert-and-return** (`tests/test_validation.py`, `tests/test_neo4j_format.py`). Each test function does its own work, prints `ERROR:` on failure, and returns `True/False`. The `main()` collects pass counts and exits 0/1. No isolation between tests, no setup/teardown.

## Mocking

- All mocking uses stdlib `unittest.mock`. No `pytest-mock`, no `responses` / `vcrpy`, no `freezegun`.
- **What gets mocked:** the OpenAI client, the Neo4j driver, and the `openai` module itself (via `sys.modules` injection so the pipeline can be imported in an env that lacks the dep).
- **What is NOT mocked:** Scallop. `scallopy` is invoked directly inside `scallop_validator.py:68-94`; there is no scallop unit test. Validation logic is exercised end-to-end through the `__main__` block at `scallop_validator.py:140-198`, which prints results for seven hand-crafted scenarios (contradiction, redundancy, circular containment, self-referential, alive/dead, valid). You run this by hand: `python scallop_validator.py`.
- The smoke script's mock client cycles through a fixed list of responses (one extraction + one verification per example): `tests/test_pipeline_smoke.py:126-132`.

## Fixtures and Test Data

- **Static fixture:** `tests/fixtures/verified_facts.sample.jsonl` — exercised by `load_fixture()` in `tests/test_neo4j_graph.py:99-108`. Asserts at least 8 facts present.
- **Live data dependency:** `tests/test_pipeline_smoke.py:89-95` requires `data.jsonl` at the repo root (the LongBench-v2 dump produced by `python download_longbench.py`). If absent, the script asserts and exits.
- **Inline data:** `tests/test_validation.py` and `tests/test_neo4j_format.py` build their facts/examples as Python literals inside each test function. There is no shared `conftest.py` and no factory module.

## Coverage

- **Not measured.** No `coverage`, `pytest-cov`, or `.coveragerc`. The repo's `.gitignore` and `requirements.txt` make no mention of either.
- Effective coverage (by hand-read):
  - `neo4j_graph.py` (472 lines) is fairly well covered by `tests/test_neo4j_graph.py` for schema/insert/conflict paths.
  - `longbench_kg_pipeline.py` (650 lines) has unit coverage on `build_extraction_prompt`, `build_verification_prompt`, `normalize_status`, `sanitize_predicate`, plus end-to-end via the smoke script.
  - **No coverage** for `experiments/_cli.py` (the per-cell driver), `experiments/run_all.py` (the orchestrator), `experiments/aggregate.py`, `experiments/build_kg.py`, `experiments/graph_context.py`, `experiments/flat_answerer.py`, `experiments/rlm_answerer.py`. These are exercised only by running real ablations.
  - `scallop_validator.py` has no automated test — only the demo at its `__main__`.
  - `evaluate.py`, `rlm_baseline.py`, `rlm_graph_baseline.py`, `download_longbench.py` have no tests.

## Eval Harnesses

These are the **real** quality gates:

- **LongBench-v2 multiple-choice ablation grid** (`experiments/run_all.py`). Runs up to six cells (`flat_raw`, `flat_kg_noscallop`, `flat_kg_scallop`, `rlm_raw`, `rlm_kg_noscallop`, `rlm_kg_scallop` — see `CELLS` in `experiments/common.py:31-38`) on the first `--limit` examples sorted by `_id` (`experiments/common.py:56-69` guarantees identical IDs across cells). Each cell writes `results/cell{N}_{LABEL}/results.jsonl`; the aggregator produces `results/summary.csv` and `results/figures/accuracy_grid.png`.
- **LongBench-v2 KG vs raw single-mode** (`evaluate.py`). Older harness. `--mode kg` uses the extracted-facts JSONL as context, `--mode raw` uses the truncated raw document. Prints overall accuracy and per-domain breakdown to stdout.
- **RLM-only baselines:** `rlm_baseline.py` (RLM over raw context) and `rlm_graph_baseline.py` (RLM over KG facts) — single-cell variants of cells 4 and 5/6.
- **KG build** (`experiments/build_kg.py`, also `longbench_kg_pipeline.py`) — the upstream stage. Not graded directly but its output (`results/kg_builds/<session>_facts.jsonl`) is what cells 2/3/5/6 score over.
- **HotpotQA harness:** none. Despite a HotPotQA-shaped context branch in `flatten_context_to_sentence_records` (`longbench_kg_pipeline.py:322-348`), there is no HotpotQA loader, no HotpotQA fixture, no HotpotQA scoring path.

## Running a "Test"

There is no `make test` target. The actual loop is:

```bash
# 1. Smoke the KG pipeline with a fake LLM (no servers needed)
python tests/test_pipeline_smoke.py

# 2. Run the validation-logic / format scripts
python tests/test_validation.py
python tests/test_neo4j_format.py

# 3. Pytest-style unit tests for neo4j_graph (if pytest is installed)
pytest tests/test_neo4j_graph.py

# 4. Manually verify the symbolic validator
python scallop_validator.py    # prints 7 case results to stdout

# 5. The real regression check: a tiny ablation slice
python -m experiments.run_all --cells all --limit 5 \
    --vllm-base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B
# Then inspect results/summary.csv and results/cellN_*/results.jsonl
```

Every cell CLI accepts `--limit N` (`experiments/_cli.py:33`, default 50) so a smoke-sized eval typically uses `--limit 5` or `--limit 10`.

## What Counts as a Regression

This repo does not pin numeric accuracy thresholds anywhere — there is no
`assert accuracy > 0.4`. Regressions are caught by reading the per-cell logs.
The specific failure modes the code itself defends against (each one was a real
past bug, judging by the comments):

1. **`predicted` collapses to one letter for an entire cell.** Cell 4 (RLM over raw) once returned `pred='D'` on every example because `extract_letter` was being called on an error message that happened to contain a `D`. Fix recorded at `experiments/rlm_answerer.py:65-70` — exception strings are now returned as-is, never letter-extracted. **Regression signal:** look at `predicted` in `results/cell4_rlm_raw/results.jsonl`; if all rows share one letter and `error` is also non-empty, this bug is back.

2. **Phantom `pred='A'` on Qwen3 with `<think>...` blocks.** The original `extract_letter` returned the *first* A/B/C/D in the string; the word `Answer` (or `Analyze`, `About`) inside the chain-of-thought would yield a phantom A. Fixed in `experiments/common.py:84-116` — explicit `Answer:` pattern is checked first, then `<think>` is stripped before further parsing, then the *last* standalone letter is preferred. **Regression signal:** `pred='A'` rate jumping while accuracy drops.

3. **Empty `question` field on Neo4j-bound facts** (the "FIX 3" bug). `longbench_kg_pipeline.py:146-149` propagates `example.get("question")` onto every supported fact, and `tests/test_pipeline_smoke.py:155-162` pins this with `assert fact["question"] != ""`.

4. **Prompt-echo / JSON-mode failure.** If a model does not honour `response_format={"type": "json_object"}`, `LongBenchKGPipeline.chat_json` (`longbench_kg_pipeline.py:245-253`) retries once without it. If both calls produce non-JSON, `parse_json_object` (`longbench_kg_pipeline.py:566-586`) returns `{}` and the example silently yields zero facts. **Regression signal:** `n_triples=0` for most examples in `results/cellN_kg_*/results.jsonl`, or `mean_triples` near zero in `results/summary.csv`.

5. **Status normalization.** `tests/test_validation.py:99-126` pins 14 input-output pairs for `normalize_status`. The "not_supported → supported" bug documented in `TESTING.md:51-55` is the canonical example.

6. **Predicate sanitization.** `tests/test_validation.py:128-150` pins 9 inputs to ensure UPPER_SNAKE_CASE relation types reach Neo4j.

7. **JSONL decode failure on download.** `download_longbench.py:85-131` validates the downloaded file in a second pass, reporting per-field missing counts. The fix for a UTF-8-split-across-buffer bug is in `download_longbench.py:63-66` (`ensure_ascii=True`).

## Manual Verification Loop

For non-trivial changes the practical loop is:

1. **Re-run** `tests/test_pipeline_smoke.py` (fast, no servers) to ensure pipeline plumbing still works.
2. **Re-run** `python scallop_validator.py` if KG/validator code changed, eyeball that all seven case outputs match the comments at `scallop_validator.py:156-197`.
3. **Re-run a small ablation slice:**
   `python -m experiments.run_all --limit 5 --cells <changed-cell-id>` against a live vLLM server. Inspect `results/cellN_*/results.jsonl` for per-example sanity (predictions present, errors absent, `n_triples` non-zero on KG cells).
4. **Diff `results/summary.csv`** against the previous run to spot accuracy drops or latency regressions. The aggregator robustly handles missing cells by emitting blank rows (`experiments/aggregate.py:62-81`) so partial reruns are safe.
5. **Inspect `rlm_logs_ablation/`** if an RLM cell looks broken — each example produces a logged trajectory there (set up by `experiments/rlm_answerer.py:34-36`).

## Async / Concurrency Testing

Not applicable. Every pipeline is single-threaded and synchronous. There are no `async def` functions, no `asyncio` imports, no thread pools. Concurrency between cells is achieved by `experiments/run_all.py:181-184` shelling out via `subprocess.call` to each cell module sequentially.

---

*Testing analysis: 2026-05-11*
