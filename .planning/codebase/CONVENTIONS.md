# Coding Conventions

**Analysis Date:** 2026-05-11

This is research code (KG + long-context LLM ablations), not a product. Conventions
below describe what the code actually does, not what it should do. There is no
linter, formatter, type-checker, or `pyproject.toml` in the repo — style is
enforced by author discipline and reading the existing modules.

## Naming Patterns

**Files:**
- Top-level scripts use `snake_case.py` and live at the repo root: `longbench_kg_pipeline.py`, `rlm_baseline.py`, `rlm_graph_baseline.py`, `neo4j_graph.py`, `scallop_validator.py`, `evaluate.py`, `download_longbench.py`.
- The ablation harness lives under `experiments/` as a real package with `__init__.py`. Each ablation cell is a numbered+labelled module: `experiments/cells/cell{N}_{label}.py` (e.g. `experiments/cells/cell3_flat_kg_scallop.py`). The label string matches the `CELLS` table in `experiments/common.py:31`.
- Test files use the `test_*.py` prefix, but they are intermixed: legacy ones at repo root (`test.py`, `tests/test_pipeline_smoke.py`) and proper unit tests under `tests/` (`tests/test_neo4j_graph.py`, `tests/test_neo4j_format.py`, `tests/test_validation.py`).
- Underscore-prefixed files signal "internal/helper, not a CLI entry point": `experiments/_cli.py` (shared CLI plumbing), `_demo_neo4j.py` (manual demo).

**Functions:**
- `snake_case` throughout. Private helpers are prefixed with a single `_` (e.g. `_print_summary`, `_maybe_aggregate`, `_common_cell_args`, `_fact_to_params`).
- Builder helpers follow `build_<thing>_prompt` / `build_<thing>_block` (`build_extraction_prompt`, `build_verification_prompt`, `build_question_block` in `longbench_kg_pipeline.py:406-500`).
- Parsing helpers follow `parse_<thing>` / `extract_<thing>` / `normalize_<thing>` / `sanitize_<thing>` (`parse_json_object`, `extract_letter`, `normalize_status`, `sanitize_predicate`).

**Variables:**
- `snake_case`. Module-level constants in `UPPER_SNAKE`: `CELL_SCHEMA`, `CELLS` in `experiments/common.py:16-38`; `FUNCTIONAL_PREDICATES`, `GENERIC_OBJECTS` in `scallop_validator.py:4-14`; `DEFAULT_SESSION_ID`, `MAX_HOPS` in `neo4j_graph.py:46-47`; `EXPECTED_FIELDS` in `download_longbench.py:33`.
- Predicate strings on the KG side are also `UPPER_SNAKE_CASE` (e.g. `"SPOKEN_IN"`, `"PART_OF"`); `sanitize_predicate` in `longbench_kg_pipeline.py:532` enforces this on extraction output.

**Types:**
- Module-level type aliases live near the top of the file using assignment, not `TypeAlias`: `Fact = Dict[str, Any]`, `SentenceRecord = Dict[str, Any]` (`longbench_kg_pipeline.py:53-54`, `neo4j_graph.py:44`).
- Classes are `PascalCase` (`LongBenchKGPipeline`, `PipelineConfig`, `Neo4jGraph`, `GraphSource`).

## Code Style

**Formatting:**
- No `black` / `ruff format` / `isort` config in the repo. Indent is 4 spaces, lines are loosely capped at ~100 chars (some `argparse` blocks go longer).
- All non-trivial modules open with `from __future__ import annotations` so type hints can use postponed evaluation: see `longbench_kg_pipeline.py:35`, `experiments/common.py:8`, `experiments/_cli.py:8`. Two exceptions: `evaluate.py:19` keeps it, but `scallop_validator.py` and the legacy `test.py` do not.

**Linting:**
- No `.flake8`, `.eslintrc`, `pyproject.toml`, `setup.cfg`, `tox.ini`, or `pytest.ini`. No CI config detected. All quality gates are manual.

## Type Hints

- **Used pervasively but unevenly.** New code (everything under `experiments/`, plus `neo4j_graph.py`, `longbench_kg_pipeline.py`, `download_longbench.py`) annotates every public function with `Path`, `Optional[...]`, `List[...]`, `Dict[str, Any]`, `Tuple[...]`. Older scripts (`scallop_validator.py`, `evaluate.py`, `test_validation.py`) use untyped `def` signatures.
- `Optional[X]` and `X | None` are both in use — no project-wide preference. `rlm_baseline.py:34` uses `list[dict]` / `int | None` (PEP 604), while `experiments/common.py:13` imports the old-style `Optional` from `typing`.
- No `mypy` config, no `py.typed`. Type hints are documentation, not enforced.

## Dataclasses / Pydantic

- Exactly **one** `@dataclass`: `PipelineConfig` in `longbench_kg_pipeline.py:57-77` (frozen-by-convention config bag passed into `LongBenchKGPipeline`). All other "structured data" flows as `Dict[str, Any]` with a stable string-key schema (see `CELL_SCHEMA` in `experiments/common.py:16-27`).
- `pydantic>=2.13.0` is pinned in `requirements.txt:15` but is a transitive dep of `openai`; **no `BaseModel` subclass exists in this repo.** Validation is hand-rolled: `clean_fact`, `normalize_status`, `sanitize_predicate` in `longbench_kg_pipeline.py:503-547`.

## Function Design

- Most functions are short (10-40 lines) and single-purpose. The longest are `run_cell` (`experiments/_cli.py:97-223`, ~125 lines, drives the per-example loop) and `LongBenchKGPipeline.run` (`longbench_kg_pipeline.py:115-161`).
- Keyword-only arguments via `*,` are used when a function has many params, to force call-site readability:

```python
# experiments/_cli.py:97-105
def run_cell(
    *,
    cell_id: int,
    label: str,
    kind: str,           # "flat" | "rlm"
    retrieval: str,      # "raw" | "kg"
    session_id: Optional[str],
    argv: Optional[List[str]] = None,
) -> Path:
```

- Lazy imports inside function bodies are deliberate: `experiments/_cli.py:122` does `from openai import OpenAI` only when `kind == "flat"`, and `from rlm.core.rlm import RLM` inside `experiments/rlm_answerer.py:31`. Reason given in `experiments/common.py:3-6`: the aggregator must run without `neo4j` / `rlm` / `openai` / `scallopy` installed.

## Docstrings

- **Module-level docstrings are universal** and substantive: most start with a one-line summary, then a usage block (CLI example) or design rationale. See `longbench_kg_pipeline.py:2-33` (full CLI example), `neo4j_graph.py:1-29` (public API listing + live-data backfill note), `experiments/run_all.py:1-12` (pipeline ordering).
- **Function docstrings are short and informal** — triple-quoted English sentences, no Google / NumPy / Sphinx style sections. Example:

```python
# experiments/common.py:84-99 (extract_letter docstring head)
"""Pull the answer letter (A/B/C/D) out of the model output.

Order of preference, strongest signal first:
  1. Explicit pattern like ``Answer: X`` / ``Final answer: X``...
"""
```

- No type info in docstrings (types live in annotations). Args/Returns blocks appear only in the `Neo4jGraph.__init__` constructor (`neo4j_graph.py:79-88`).

## Import Organization

Consistent order across new modules (`experiments/_cli.py:8-25` is the canonical example):
1. `from __future__ import annotations`
2. Stdlib imports, alphabetical (`argparse`, `json`, `os`, `sys`, `time`).
3. `from pathlib import Path` + `from typing import ...`.
4. Blank line, then first-party imports (`from experiments.common import ...`).
5. Third-party heavy deps (`neo4j`, `openai`, `rlm`, `scallopy`) are **lazy-imported inside functions**, not at the top of files. See `experiments/rlm_answerer.py:31`, `experiments/graph_context.py:143`.

`neo4j_graph.py:38-41` wraps the optional import in `try / except ImportError` and stashes `None` so the module remains importable when the driver is absent. This is the standard pattern for optional research deps in this repo.

## Error Handling

This is the area where research-code character is most visible. Three different patterns coexist:

1. **Per-example loops swallow exceptions and record them in the JSONL row** (the dominant pattern). The loop never aborts on a bad example; the failure is logged to the row's `error` field so aggregation downstream can count it:

```python
# experiments/_cli.py:185-191
except Exception as e:
    error = str(e)
    predicted = ""
    print(
        f"  [{i}/{len(examples)}] EXC {example_id}: {error[:200]}",
        file=sys.stderr,
    )
```

   Same pattern in `rlm_baseline.py:143-148`, `rlm_graph_baseline.py`, and `evaluate.py:149-152` (which just `continue`s past API errors). `experiments/rlm_answerer.py:65-70` adds an important guard: it explicitly does **not** run `extract_letter` on the error string, because exception messages often contain stray A/B/C/D characters that would silently fabricate a prediction.

2. **Fail-loud at startup / on config errors.** Constructor checks raise `ValueError` / `RuntimeError` immediately: `Neo4jGraph.__init__` (`neo4j_graph.py:99-106`) rejects missing creds or invalid stateless flag; `experiments/graph_context.py:161-167` raises `RuntimeError` with a remediation hint when no graph source is reachable. `experiments/_cli.py:138-140` catches that and calls `sys.exit(2)`.

3. **JSON-mode fallback / retry once on the inner LLM call.** `LongBenchKGPipeline.chat_json` (`longbench_kg_pipeline.py:245-253`) tries OpenAI JSON mode, and on the first exception retries without `response_format` rather than failing. Anything else is re-raised.

Aggregation is also fail-soft: `_maybe_aggregate` in `experiments/_cli.py:87-94` catches every exception from the aggregate step and prints a `(non-fatal)` note, so a broken matplotlib import never ruins a long ablation run.

## Logging vs print

- **No use of stdlib `logging`.** `grep "import logging"` returns zero hits in the repo. All progress output uses `print(..., file=sys.stderr)`; `stdout` is reserved for the final summary tables in `evaluate.py` and `run_all.py`.
- Per-example progress lines follow a consistent shape:

```python
# experiments/_cli.py:211-216
status = "OK " if correct else "x  "
print(
    f"  [{i}/{len(examples)}] {status}{example_id} "
    f"pred={predicted!r} gold={gold!r} ({elapsed:.1f}s)",
    file=sys.stderr,
)
```

  Older scripts use Unicode tick/cross (`✓` / `✗`) — see `rlm_baseline.py:166`, `evaluate.py:178`. `experiments/_cli.py` deliberately uses plain ASCII (`OK ` / `x  `) for the same status — likely a portability fix.

- RLM internals get their own log directory via `RLMLogger(log_dir=...)` (`experiments/rlm_answerer.py:35`, `rlm_baseline.py:91`). Default dirs: `rlm_logs/` for the baseline, `rlm_logs_ablation/` for the ablation cells.

## Seeds / Determinism

Determinism is achieved through **two coarse mechanisms**, not through seeded RNGs:

1. **`temperature=0.0` on every LLM call.** `longbench_kg_pipeline.py:601` (extraction default), `experiments/flat_answerer.py:19` (flat answerer default), `evaluate.py:146`, `experiments/build_kg.py:136`. The only `temperature=0.7` in the repo is the demo `test.py:15`.
2. **Stable pilot slicing by `_id`.** `iter_pilot_examples` in `experiments/common.py:56-69` sorts all examples by `_id` and takes the first `limit` — explicitly so that all six cells score on identical example IDs without coordination. The `--seed` CLI flag (`experiments/_cli.py:34`) is plumbed through `experiments/run_all.py:62` but is **currently unused** (kept "for future randomization", per the comment on `experiments/common.py:59`).

No `random.seed`, `numpy.random.seed`, or `torch.manual_seed` is called anywhere.

## Prompt Templating

Prompts are **plain triple-quoted f-strings**, not Jinja, not a template class:

- `build_extraction_prompt` / `build_verification_prompt` in `longbench_kg_pipeline.py:406-483` — each ends with `.strip()` and embeds a literal JSON shape spec the model is asked to echo. The verification prompt enumerates six self-reflection questions inline (`longbench_kg_pipeline.py:460-466`); `tests/test_validation.py:40-47` asserts on those exact strings.
- `format_question` in `experiments/common.py:72-81` and `rlm_baseline.py:46-55` (duplicated, with identical body) builds the multiple-choice prompt by line-concatenation.
- `evaluate.py:62-83` builds its own `build_prompt` directly. There is no shared prompt module.

JSON-shape outputs are parsed defensively by `parse_json_object` (`longbench_kg_pipeline.py:566-586`), which first strips markdown fences and then falls back to slicing between the first `{` and last `}`.

## CLI Argument Parsing

- **`argparse` only.** No `click`, `typer`, or `fire` anywhere in the repo.
- Every script that has a `main()` defines its own parser locally (`longbench_kg_pipeline.py:594-637`, `rlm_baseline.py:68-86`, `download_longbench.py:134-147`, `evaluate.py:215-226`, `experiments/run_all.py:96-117`).
- The ablation cells share a single `build_arg_parser` in `experiments/_cli.py:28-76` and call `run_cell(...)` — this is the only abstraction over `argparse` in the repo.
- Environment-variable fallbacks are used for secrets so flags can be omitted: `os.getenv("NEO4J_URI", ...)`, `os.getenv("NEO4J_PASSWORD")`, `os.getenv("VLLM_API_KEY", "EMPTY")` (`experiments/_cli.py:37,51-53`, `experiments/run_all.py:102-105`).
- Path-typed args use `type=Path`; numeric defaults match across scripts (`--temperature 0.0`, `--max-tokens 64000` for RLM, `--limit 50` for pilot ablations).

## Output Conventions

- **JSONL is the universal output format.** Every per-example pipeline writes one JSON object per line. Files use UTF-8 with `ensure_ascii=False` for human-readable text (`experiments/_cli.py:144`, `longbench_kg_pipeline.py:150`); the one exception is `download_longbench.py:66` which uses `ensure_ascii=True` deliberately to dodge a UTF-8-split-across-buffer bug (commented at `download_longbench.py:63-66`).
- **Per-cell output directory layout** is fixed by `cell_output_path` in `experiments/common.py:41-42`:
  `results/cell{N}_{LABEL}/results.jsonl` (e.g. `results/cell3_flat_kg_scallop/results.jsonl`).
- **Schema-conformant rows** are written via `write_result_row` (`experiments/common.py:119-122`), which projects the kwargs through `CELL_SCHEMA` so missing keys default to `None`. The keys are `cell_id, label, example_id, predicate, gold, correct, n_context_chars, n_triples, elapsed_seconds, error`.
- Run metadata (`results/run_metadata.json`) is written once per orchestrated run (`experiments/run_all.py:122-141`) and includes git SHA, model, all hyperparameters.
- Aggregated outputs: `results/summary.csv` and `results/figures/accuracy_grid.png` (produced by `experiments/aggregate.py`).
- Intermediate KG fact dumps live under `results/kg_builds/<session>_facts.jsonl` (e.g. `pilot_noscallop_facts.jsonl`, `pilot_scallop_facts.jsonl`) — see `experiments/run_all.py:76`. These act as the offline fallback when Neo4j is unreachable (`experiments/graph_context.py:156-159`).
- Fact IDs are deterministic 24-char SHA-256 prefixes over a sorted JSON of (`example_id, subject, predicate, object, support_text, provenance`) — `make_fact_id` in `longbench_kg_pipeline.py:550-563`. This is the *only* hashing in the repo and is load-bearing for idempotent Neo4j MERGEs.

## Module Design

- **Flat package, one level deep.** `experiments/` has `__init__.py` (1 line, empty), `experiments/cells/` has `__init__.py` (1 line, empty). No barrel `__init__.py` re-exports anywhere — callers import the full path (`from experiments.common import iter_pilot_examples`).
- Top-level scripts are intentionally not packaged: `longbench_kg_pipeline.py`, `rlm_baseline.py`, `rlm_graph_baseline.py`, `neo4j_graph.py`, `scallop_validator.py`, `evaluate.py`, `download_longbench.py` all expose `if __name__ == "__main__": main()`.
- Sibling modules import each other by bare name (e.g. `neo4j_graph.py:32` does `from scallop_validator import validate_update`), so the repo root must be on `sys.path`. Tests do this explicitly: `sys.path.insert(0, str(Path(__file__).parent.parent))` (`tests/test_neo4j_graph.py:17`, `tests/test_pipeline_smoke.py:82`).

## Comments

- **Comments document non-obvious decisions**, not what the code does. They cluster around fix points and design choices, often with explicit bug references:
  - `experiments/_cli.py:62-65` explains why `backend="openai"` is the right default for an already-running vLLM server.
  - `experiments/rlm_answerer.py:67-70` explains why `extract_letter` must NOT be called on error strings.
  - `requirements.txt:11-32` contains 22 lines of comments explaining why specific versions were pinned (`openai==2.31.0` does not exist; `rlms` requires Python ≥3.11; `scallopy` only ships cp310).
  - `longbench_kg_pipeline.py:146-149` flags "FIX 3: propagate the question from the source example" — a bug-tracking comment style that recurs (`tests/test_pipeline_smoke.py:155-162` asserts the FIX 3 bug stays fixed).
- TODO/FIXME/HACK markers are **rare** — only a handful across the whole repo. There is no `# type: ignore` comment outside the optional-import dance in `neo4j_graph.py:41`.

---

*Convention analysis: 2026-05-11*
