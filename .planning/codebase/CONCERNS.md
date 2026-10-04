# Codebase Concerns

**Analysis Date:** 2026-05-11

This audit is biased toward concreteness: every concern cites a file path or
commit hash so a future agent can act on it. Severity tags (HIGH / MEDIUM / LOW)
exist so `/gsd-plan-phase` can prioritize.

The repo is a ~2.3k-LOC NeuroSym ablation harness around LongBench-v2 +
Neo4j + Scallop + RLM. Recent fix commits (`dc340f7`, `777d195`, `352ab2c`,
`720d97b`, `7a7876f`, `17e3e63`) all point at small, sharp regressions in
the answer-extraction / RLM-config layer — and several of those fixes were
only applied in *one* copy of the duplicated code.

---

## Tech Debt

### **HIGH — Three divergent `extract_letter` implementations (and one is the un-hardened version)**

Commit `777d195` hardened `extract_letter` to use the *last* A/B/C/D after
stripping `<think>...</think>`, because Qwen3 thinking-mode put words like
"Answer" and "Analyze" at the *start* of the response, producing phantom
`A` predictions. That fix was only applied to one of three copies.

| File:line | State |
|---|---|
| `experiments/common.py:84-116` | Hardened (last-letter, post-`<think>` strip, "Answer: X" preferred). Used by `experiments/_cli.py` and the six ablation cells. |
| `rlm_baseline.py:58-65` | **STILL THE BUGGY FIRST-LETTER VERSION** — `for ch in text.upper(): if ch in "ABCD": return ch`. No regex, no `<think>` strip. Will return phantom `A` on every Qwen3-thinking trajectory. |
| `rlm_graph_baseline.py:76-85` | Half-fixed — has the `(?:final\s+answer|answer)\s*[:\-]\s*([ABCD])` regex (note: requires a separator, so "Answer X" without colon falls through) but then falls back to *first* letter, same bug as `rlm_baseline.py`. |
| `evaluate.py:86-94` | Fourth, *different* implementation (`re.search(r'\b([ABCD])\b', text)`). Truncates raw context to 4000 chars (`evaluate.py:137`) vs canonical 32k. |

**Fix approach:** Delete the three local copies; import `extract_letter`
from `experiments.common`. Same for `format_question` (`rlm_baseline.py:46-55`,
`rlm_graph_baseline.py:65-73`, `experiments/common.py:72-81`).

### **HIGH — `str(result)` instead of `result.response` in two RLM scripts**

Commit `720d97b` documented that `rlms` 0.1.x `completion()` returns an
`RLMChatCompletion` whose `.response` attribute holds the answer text;
`str(result)` returns the repr (class name + dict), which contains no
A/B/C/D and silently produces `pred=""`. `experiments/rlm_answerer.py:60-63`
applies the fix. The two top-level baselines do not:

- `rlm_baseline.py:140` — `raw_answer = str(result) if result else ""`
- `rlm_graph_baseline.py:288` — `raw_answer = str(result) if result else ""`

**Fix approach:** Replace with `raw = getattr(result, "response", None) or str(result)`.

### **HIGH — Inconsistent RLM `--max-tokens` defaults across entrypoints**

Commit `352ab2c` bumped `rlm_baseline.py` default from 32000 → 64000 because
Qwen3 thinking blocks routinely overshot 33k tokens. The bump was *not*
propagated to the orchestrator:

| File:line | Default | Comment |
|---|---|---|
| `rlm_baseline.py:83` | `64000` | Fixed |
| `experiments/_cli.py:72` | `64000` | Fixed (per inline note quoting `rlm_baseline.py`) |
| `experiments/run_all.py:116` | **`32000`** | Still pre-fix. Will silently truncate cells 4/5/6 trajectories. |
| `rlm_graph_baseline.py:179` | **`32000`** | Still pre-fix. |

**Fix approach:** Single named constant in `experiments/common.py`
(`DEFAULT_RLM_MAX_TOKENS = 64000`) and import everywhere.

### **MEDIUM — Inconsistent Qwen3 `enable_thinking` policy**

Commit `dc340f7` re-enabled Qwen3 thinking with a 2048-token budget for
accuracy. Commit `777d195` had just disabled it for the flat answerer.
Today:

- `experiments/flat_answerer.py:20` defaults `enable_thinking=True` with
  `max_tokens=2048`, and the docstring (`:26-29`) explicitly relies on
  `extract_letter` stripping `<think>` blocks. This is *only* safe if every
  caller uses the hardened `extract_letter`. The two top-level baselines do
  not (see HIGH-1).
- `experiments/_cli.py:170` calls `flat_answer(client, args.model, context, question)`
  with no `enable_thinking` argument, accepting the True default. There is
  no CLI flag to disable it for ablation.

**Fix approach:** Add `--enable-thinking/--no-enable-thinking` flag wired
through `_cli.py` and `flat_answerer.py`; record the choice in
`results/run_metadata.json` so old vs new runs are distinguishable.

### **MEDIUM — Duplicated KG-context formatting code**

`format_facts_from_jsonl` exists verbatim in two places:
- `rlm_graph_baseline.py:110-149`
- `experiments/graph_context.py:45-77`

Same for `extract_seed_entities`:
- `rlm_graph_baseline.py:88-107`
- `experiments/graph_context.py:19-32`

And `Neo4jGraph.format_context_for_llm` (`neo4j_graph.py:399-451`) is a
*third* near-duplicate with subtle differences (sort-by-fact_id,
"truncated, N more facts" wording vs "N more facts truncated").

**Fix approach:** Move the canonical implementations into
`experiments/graph_context.py` (or a new `kg_format.py`) and have
`rlm_graph_baseline.py` + `neo4j_graph.py` import them. Pick one truncation
message and one sort order.

### **MEDIUM — Magic numbers scattered across modules**

| File:line | Magic value | Meaning |
|---|---|---|
| `neo4j_graph.py:47` | `MAX_HOPS = 4` | Module constant — good |
| `rlm_graph_baseline.py:107` | `result[:10]` | Cap seed entities — undocumented |
| `experiments/graph_context.py:32` | `out[:10]` | Same cap, undocumented |
| `experiments/_cli.py:46` | `default=32000` | `raw_max_chars` — magic |
| `evaluate.py:137`, `evaluate.py:46` | `4000` | Two different "truncate to 4000" |
| `longbench_kg_pipeline.py:361` | `max_unit_chars=1200` | Sentence chunk |
| `longbench_kg_pipeline.py:603` | `chunk_chars=12000` | LLM chunk |
| `experiments/build_kg.py:79,209` | `chunk_chars=12000` | Same constant, separately declared |
| `experiments/build_kg.py:80,210` | `verify_batch_size=20` | Same |
| `experiments/rlm_answerer.py` / `_cli.py:67` | `max_iterations=10` | RLM cap |
| `rlm_baseline.py:158` | `raw_answer[:200]` | Truncate logged answer |
| `rlm_graph_baseline.py:309` | `raw_answer[:300]` | Same, *different* limit |

**Fix approach:** Centralize in `experiments/common.py` as named constants
with comments explaining what they govern.

### **LOW — Unfinished scaffolds / dead-ish files**

- `_demo_neo4j.py` (25 lines): one-shot interactive probe with a hardcoded
  password (see SECURITY). Not imported anywhere. Either move to
  `scripts/demo_neo4j.py` and parameterize, or delete.
- `test.py` (20 lines): vLLM smoke ping that mutates `sys.path` to load a
  sibling `vllm/` checkout (`test.py:4`). Not imported, not in `tests/`.
- `test.scl` (10 lines): Scallop fibonacci demo. Unused by the validator
  (`scallop_validator.py`). Either move to `examples/` or delete.
- `experiments/common.py:59` — `seed: int = 0  # reserved for future
  randomization; deterministic for now`. Argument is plumbed through
  `_cli.py:34` and `run_all.py:99` but never used; `iter_pilot_examples`
  always returns the same prefix-by-`_id` slice. Either implement seeded
  shuffling or remove the parameter.

---

## Known Bugs / Foot-guns

### **HIGH — `extract_letter` regression in `rlm_baseline.py` and `rlm_graph_baseline.py`**

See Tech Debt HIGH-1. Running either of these two scripts with Qwen3 in
thinking mode will return `A` on most examples because `<think>` blocks
start with words like "Analyze" / "Answer". Accuracy numbers from those
two scripts are *not* comparable to numbers from
`experiments/_cli.py`-driven cell runs.

### **HIGH — `str(result)` collapses RLM accuracy to 0 in two scripts**

See Tech Debt HIGH-2. Same two files.

### **MEDIUM — `extract_letter(error_str)` fabricates predictions from exception text**

`experiments/rlm_answerer.py:65-70` explicitly warns against this and
returns `""` on exception. The older code path still exists in
`rlm_baseline.py:144-147` and `rlm_graph_baseline.py:291-298`:

```python
except Exception as e:
    error_str = str(e)
    raw_answer = error_str
    predicted = extract_letter(error_str)  # ← will pick stray A/B/C/D
```

Any error containing "Bad Request" or "Connection" silently produces a
prediction. Effect: errored examples are counted as answered (`correct=False`
but `predicted!=""`) and skew the accuracy denominator.

**Fix approach:** Mirror `experiments/rlm_answerer.py:70` — never parse a
letter out of an exception string.

### **MEDIUM — `extract_seed_entities` regex misses lowercase/non-ASCII entities and proper nouns split on hyphens**

`rlm_graph_baseline.py:96` / `experiments/graph_context.py:24`:
```python
re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
```

This is the only seed for the n-hop Neo4j query. It will fail on
"COVID-19", "WW2", lowercase-language names ("kalamang"), all-caps
("NASA"), and non-Latin scripts. Effect: queries return zero rows, cells
fall back to "No relevant facts found in knowledge graph."
(`rlm_graph_baseline.py:261`) and the KG ablation degenerates to
"flat over an empty context."

**Fix approach:** Either NER (spacy/stanza) or add explicit fallbacks
(choice-text entities, quoted strings, hyphenated tokens, all-caps tokens).
The TODO is already named in `rlm_graph_baseline.py:92` ("Week 3 can
replace with NER").

### **MEDIUM — `_facts_count` uses undirected match; double-counts each rel**

`experiments/build_kg.py:30-36`:
```python
"MATCH ()-[r]-() WHERE r.session_id = $sid RETURN count(r) AS c"
```

The undirected pattern `()-[r]-()` traverses every relationship twice in
Neo4j, doubling the count. The `--rebuild` gate (`build_kg.py:107`) is
just `existing > 0`, so functionally this doesn't break the gate — but
the "session already has N facts" log message reports 2x the true count.

**Fix approach:** Use `MATCH ()-[r]->()` (directed) or
`MATCH (s)-[r]-(o) WHERE id(s) < id(o)`.

### **LOW — Predicate sanitization disagreement across modules**

`longbench_kg_pipeline.sanitize_predicate` (`:532-536`) and
`neo4j_graph.sanitize_predicate` (`:50-55`) are byte-identical but defined
separately. If one drifts, fact_ids will silently collide.

### **LOW — `Neo4jGraph.query_context` Cypher injects `hops_int` and `limit_int` via f-string**

`neo4j_graph.py:365,374` — `hops_int` and `limit_int` are bounded
(`MAX_HOPS=4`, `>=1`) before f-string interpolation, so this is currently
safe, but it's a sharp edge: future refactors that take the bounds off
would create a Cypher injection sink. Same pattern in
`neo4j_graph.py:187` (`rel_type` is sanitized first).

**Fix approach:** Use parameterized variable-length paths where Neo4j
supports them, or move bounds-checks immediately above the f-string and
add a comment marking them as injection-prevention.

---

## Security & Privacy

### **HIGH — Hardcoded Neo4j password `<REDACTED>` committed to repo (3 places)**

- `rlm_graph_baseline.py:9` — in docstring example
- `_demo_neo4j.py:7` — hardcoded as literal argument to `Neo4jGraph()`
- `.planning/codebase/STACK.md:83` — copied from the docstring during the
  earlier `tech` mapping pass

This is a real credential. Even if the Neo4j instance is local-only, it
should not live in git history. Rotate the password and replace with
`os.getenv("NEO4J_PASSWORD")` everywhere.

**Fix approach:**
1. Rotate the Neo4j password on the live instance.
2. Replace literal in `_demo_neo4j.py:7` with
   `os.environ["NEO4J_PASSWORD"]` or argparse.
3. Replace the password in the `rlm_graph_baseline.py:9` docstring with
   `<your-password>` or `$NEO4J_PASSWORD`.
4. Scrub `STACK.md:83` to the same placeholder.
5. Consider `git filter-repo` to purge history if the instance is exposed.

### **MEDIUM — `scli` 6.5 MB Mach-O arm64 binary committed**

`scli` is the Scallop CLI executable (arm64-only Mac binary). Living in
git means:
- Bloats every clone by ~6.5 MB.
- Non-Mac contributors get a non-executable file.
- No supply-chain provenance — there's no checksum, no source link.

**Fix approach:** Add `scli` to `.gitignore`; document the download URL
in `requirements.txt` (the scallopy wheel URL on line 19 already pins the
upstream version, so use the matching `scli` release from
github.com/scallop-lang/scallop).

### **LOW — Dataset licensing not documented**

`download_longbench.py` pulls LongBench-v2 (HuggingFace dataset, Apache
2.0 with attribution). The README and `download_longbench.py` do not
record the dataset license or attribution requirements. Low risk but
worth a one-line note in README.

### **LOW — `.gitignore` allows `tests/fixtures/*.jsonl`**

`.gitignore:6` explicitly whitelists `tests/fixtures/*.jsonl`. The
fixture `tests/fixtures/verified_facts.sample.jsonl` (3.9 KB) is mostly
synthetic, but anyone copying real extraction output into that path will
silently commit it. Consider a stricter whitelist (`!tests/fixtures/verified_facts.sample.jsonl`).

---

## Performance Hotspots

### **HIGH — KG-build cost is linear in (examples × chunks × 2 LLM calls)**

`longbench_kg_pipeline.LongBenchKGPipeline.run` (`:115-161`) does
sequential extraction + verification: for each example, for each
~12k-char chunk, one extraction LLM call (`:136`) then one verification
LLM call per batch of ≤20 facts (`:188-194`). On a 50-example pilot with
LongBench long-context inputs (32k-200k chars), this is hundreds of LLM
calls in serial.

- Sleep between calls defaults to 0 (`longbench_kg_pipeline.py:606`), so
  it's CPU-bound on the local vLLM server, but...
- ...there is no concurrency. A 50-example pilot can take hours.
- `experiments/build_kg.py` re-runs this for *each* validator setting
  (`pilot_noscallop`, `pilot_scallop`), but the underlying extraction is
  identical — Scallop only enters during `insert_facts`. Effect: 2x the
  LLM cost for no extraction-quality reason.

**Fix approach:**
1. Extract once into a `pilot_extracted.jsonl`, then run `commit_facts`
   twice (once with `validate=False` → `pilot_noscallop`, once with
   `validate=True` → `pilot_scallop`). Saves 50% of LLM calls.
2. Add async/`ThreadPoolExecutor`-based concurrency for the extraction
   loop. The OpenAI-compatible vLLM server handles parallel requests.

### **MEDIUM — Long-context decode budget too small**

`experiments/_cli.py:46` `--raw-max-chars` defaults to 32000 — sane for
flat cells. But `experiments/_cli.py:58` `--context-max-chars` defaults
to 4000 for KG cells, which truncates the n-hop neighborhood
aggressively. With `--limit-triples=50` (`:57`) and ~80-character
per-triple block, 50 triples need ~4000 chars exactly — any verbose
support_text pushes triples off the end.

**Fix approach:** Either drop `support_text` from the LLM-facing block
(it's already in the underlying KG) or raise `--context-max-chars` to
8000 by default.

### **MEDIUM — Neo4j `query_context` re-runs schema + driver session per call**

`neo4j_graph.Neo4jGraph._session()` (`:135-138`) opens a fresh session on
every read. Across a 50-example × 6-cell run, that's 300 session opens.
The bolt driver pools connections so this is cheap, but the schema-ready
check (`:160-169`) is correctly cached (`self._schema_ready`). No action
needed unless we move to remote Aura.

### **LOW — `subprocess.call` in `experiments/run_all.py:182` spawns a fresh Python per cell**

Each cell is run as `subprocess.call([sys.executable, "-m", mod, ...])`
(`run_all.py:181`). This re-imports the world (vllm OpenAI client, neo4j,
rlms, scallopy) six times. ~2-5s per cold start × 6 = 15-30s overhead.
Acceptable for ablation isolation, but worth noting.

---

## Reliability / Tests / Determinism

### **HIGH — Tests are ad-hoc scripts, not pytest**

`tests/test_pipeline_smoke.py:8` and `tests/test_neo4j_format.py` are
top-level scripts: they `assert` directly at module scope and exit on
import. There is no `pytest.ini`, no `conftest.py`, no `__init__.py` in
`tests/`. `tests/test_neo4j_format.py:225` and
`tests/test_neo4j_graph.py:378-381` use bare `except` to print pass/fail
counts manually.

Effects:
- `pytest tests/` will mis-collect: `test_pipeline_smoke.py` runs the
  full pipeline at import time (line 134: `pipeline.run()`).
- CI cannot get a clean exit code.
- New tests can't share fixtures.

**Fix approach:**
1. Wrap each script in `def test_*` functions.
2. Add `pytest` to `requirements.txt`.
3. Add `conftest.py` for the `verified_facts.sample.jsonl` fixture path.
4. Add a GitHub Action that runs `pytest -x` on PRs.

### **HIGH — No determinism guarantees beyond `temperature=0`**

- vLLM at `temperature=0` is *mostly* deterministic but not guaranteed
  across batch sizes or vLLM versions.
- No `torch.manual_seed`, `random.seed`, or `np.random.seed` calls
  anywhere (`grep` confirms zero hits).
- `experiments/common.iter_pilot_examples` (`:60-69`) sorts by `_id` —
  this *is* deterministic — but the `seed` parameter (`:59`) is unused.
- No `temperature` exposed on `experiments/_cli.py` flat-answerer path
  (`flat_answerer.py:19` is hardcoded to `0.0`, no CLI flag).

Effect: re-running the same cell on the same git SHA can produce slightly
different numbers and there is no way to tell whether the diff is signal
or noise.

**Fix approach:**
1. Log the vLLM server version + the cell's seed/temperature into each
   `results.jsonl` row.
2. Implement the `seed` parameter (numpy shuffle before slicing in
   `iter_pilot_examples`), or remove the flag.
3. Document the determinism guarantee (or lack of it) in README.

### **MEDIUM — `summary.csv` committed with all-zero rows is stale and misleading**

`results/summary.csv` is committed and shows `n_examples=0, accuracy=` for
all 6 cells. Anyone cloning the repo will see "zero results" and assume
the experiment has not been run. Either delete it (add `results/*.csv`
to `.gitignore`) or commit a real reference run with the git SHA in the
filename.

### **MEDIUM — Ad-hoc retry / fallback chains hide failures**

- `longbench_kg_pipeline.py:245-253` — if `response_format={"type": "json_object"}`
  errors, retry without it. No log of which path was taken — silent
  degradation.
- `rlm_graph_baseline.py:198-200` — if Neo4j connect fails, fall back to
  `--facts-file`. Silent unless the facts-file is also missing.
- `experiments/graph_context.py:152-154` — same pattern; only the first
  exception is printed, never re-raised.
- `experiments/_cli.py:185-187` — wraps the per-example loop in
  `except Exception` and continues. Errors are captured per-row in
  `results.jsonl` (`error` field), which is correct, but the orchestrator
  (`run_all.py:182`) prints "cell X exited with code N" only for non-zero
  exits — a cell that errored on every example still exits 0 (since the
  loop catches), and only the aggregate accuracy reveals it.

**Fix approach:** Single structured log line per fallback (`json.dumps`
with `event=fallback, reason=...`) emitted to stderr, and a final
"errors: K / total: N" line at cell exit. The structure already exists
in `rlm_baseline.py:174,180`; standardize it.

### **LOW — Tests have hardcoded vllm `sys.path` mangling**

`test.py:4` mutates `sys.path` to insert a local `./vllm/` directory.
`tests/test_pipeline_smoke.py:82` does
`sys.path.insert(0, str(Path(__file__).parent.parent))`. The first only
works on the author's machine (the `vllm/` repo is in `.gitignore`); the
second works but is the kind of thing that breaks under pytest's
rootdir detection.

---

## Fragile Areas

### **MEDIUM — `experiments/cells/*.py` thin wrappers must stay in sync**

Each of `experiments/cells/cell{1..6}_*.py` (19-20 lines) wraps
`experiments._cli.run_cell(...)` with a hardcoded `cell_id`, `label`,
`kind`, `retrieval`, `session_id`. Drift between `experiments/common.CELLS`
and these wrappers (e.g. renaming `flat_kg_noscallop` to `flat_kg`) will
produce a silent mismatch — the orchestrator builds the module path from
`CELLS` (`experiments/run_all.py:53-55`), so a renamed label in `CELLS`
without a matching file rename causes ImportError.

**Fix approach:** Generate the six cell files at runtime via
`importlib.util.module_from_spec`, or move all six to a single dispatcher
that consults `CELLS`.

### **MEDIUM — `Neo4jGraph._driver` accessed via property setter used only by tests**

`longbench_kg_pipeline.LongBenchKGPipeline.neo4j_driver` (`:98-108`)
exists only so `tests/test_pipeline_smoke.py:132` can do
`pipeline.client = _make_mock_client(...)` style monkeypatching on the
driver. The `@neo4j_driver.setter` mutates `self.graph._driver` — a
private attribute on the wrapped class. Brittle: any refactor of
`Neo4jGraph` to use composition or a connection pool breaks the test.

**Fix approach:** Use `unittest.mock.patch.object(ng, "GraphDatabase")`
exactly as `tests/test_neo4j_graph.py:86` already does. The
`pipeline.neo4j_driver` getter/setter can then be deleted.

### **LOW — `scallopy` is Linux-only**

`requirements.txt:19` pins
`scallopy-0.2.4-cp310-cp310-manylinux_2_27_x86_64.whl`. On macOS arm64 or
Linux aarch64 this wheel is unusable; `scallop_validator.py:1`
`import scallopy` will fail at import time, which transitively breaks
`neo4j_graph.py:32` (`from scallop_validator import validate_update`).
The `validate=False` path through `Neo4jGraph.insert_facts` still
imports scallop because the import is at module scope.

**Fix approach:** Lazy-import `scallopy` inside `validate_update`
(`scallop_validator.py:35`), and lazy-import `validate_update` inside
`Neo4jGraph.insert_facts` (`neo4j_graph.py:286`). Then macOS contributors
can at least run cells 1, 2, 4, 5.

### **LOW — `_facts_count` and `_dump_session_facts` use private `_session` accessor**

`experiments/build_kg.py:34,50` call `graph._session()` — a private
method on `Neo4jGraph`. If the wrapping changes (e.g. to async sessions),
build_kg breaks.

---

## Scaling Limits

- **MAX_HOPS = 4** (`neo4j_graph.py:47`): variable-length Cypher path
  expansion at hop=4 over a 50k-node graph will hit memory ceilings.
  Document the empirical limit in `experiments/_cli.py:56`
  (`--hops` default = 2).
- **In-memory all-examples load** in `experiments/common.iter_pilot_examples`
  (`:60-69`): reads the entire `data.jsonl` before slicing. LongBench-v2
  is ~500 examples × ~100k chars each = ~50 MB raw, fine today but a
  hard wall for any dataset > ~10 GB. Use a generator + reservoir
  sampling if scaling.
- **`results/kg_builds/<session>_facts.jsonl` is single-writer**: two
  concurrent `build_kg.py --session pilot_noscallop` invocations will
  silently corrupt the JSONL mirror.

---

## Dependencies at Risk

### **MEDIUM — Python-version split: `rlms>=0.1.1` needs Python 3.11; `scallopy==0.2.4` ships cp310 only**

Documented in `requirements.txt:14-25`. Already a known foot-gun; the
README recommends either `--ignore-requires-python` or two separate
envs. No single env satisfies both, so the orchestrator
(`experiments/run_all.py`) cannot in principle be run in one env if it
truly needs both `scallopy` (for `insert_facts(validate=True)` in
cells 3/6 KG-build) and `rlms` (for cells 4/5/6 answerers).

**Fix approach:** Split the pipeline into two binaries — `build_kg.py`
(py3.10 + scallopy) and `run_cells.py` (py3.11 + rlms). Already most of
the way there; the orchestrator just needs to invoke a different python
for the build phase.

### **LOW — `openai>=1.0.0,<2.0.0` upper bound shifts the API**

`requirements.txt:13` allows up to but not including 2.x. OpenAI's
`response_format={"type":"json_object"}` (`longbench_kg_pipeline.py:243`)
and the chat-completions interface are 1.x idioms. When OpenAI 2.x
ships, the codebase will need migration. The `try/except` at
`longbench_kg_pipeline.py:247` already handles servers that don't
support `response_format`, so the surface area for breakage is small.

---

## Missing Critical Features

### **MEDIUM — No CI / no automated regression on the six-cell grid**

There is no `.github/workflows/`, no `Makefile`, no `pyproject.toml`.
Every "fix" commit has to be hand-verified by running the pipeline
locally. Given how many of the recent fixes (`777d195`, `720d97b`,
`352ab2c`) were one-line regressions that broke accuracy silently, a
1-example smoke test per cell on PRs would be high-value.

### **LOW — No persistence of `extract_letter` test cases**

Commits `777d195` and `dc340f7` both fixed `extract_letter`. There is
*no* test asserting the regression cases ("Answer A is the bird" should
return... `A`, or actually the *last* `A`? — ambiguous!). Next regression
will pass tests.

**Fix approach:** Add `tests/test_extract_letter.py` with the
specific Qwen3 traces that motivated the fixes. Pin the behaviour.

---

## Test Coverage Gaps

| Area | What's not tested | Files | Risk |
|---|---|---|---|
| `extract_letter` regression cases | Qwen3 thinking-block edge cases, "Answer: X" vs "X. Answer" | `experiments/common.py:84` | **HIGH** — this is the single most fixed function in the repo |
| `rlm_baseline.py` / `rlm_graph_baseline.py` | No tests for these scripts at all | both files | **HIGH** — they have the buggy duplicates |
| `run_all.py` orchestrator | `_common_cell_args`, `_parse_cells` | `experiments/run_all.py:42-92` | MEDIUM |
| `aggregate.py` | `_summarize_rows` with empty / partial inputs; `render_figure` | `experiments/aggregate.py:39-57,102` | MEDIUM |
| `evaluate.py` | Not tested at all | `evaluate.py` | MEDIUM — duplicate `extract_answer` impl |
| `download_longbench.py` | Not tested | `download_longbench.py` | LOW |
| Neo4j live integration | All Neo4j tests mock the driver (`tests/test_neo4j_graph.py:86`) | n/a | LOW — by design, but means no smoke on real Cypher |
| `scallop_validator.validate_update` | Only the `__main__` block prints test results (`scallop_validator.py:140-197`); not pytest-collected | `scallop_validator.py` | MEDIUM |

---

## Areas to Refactor Next

Ranked by ROI for `/gsd-plan-phase`:

1. **(HIGH) Consolidate `extract_letter`, `format_question`, `extract_seed_entities`, `format_facts_*` into shared modules** and delete the duplicates in `rlm_baseline.py` / `rlm_graph_baseline.py` / `evaluate.py`. Single biggest source of silent regressions.

2. **(HIGH) Add `tests/test_extract_letter.py`** with the Qwen3 traces from commits `777d195` and `dc340f7` as fixtures. Same for `tests/test_rlm_answerer.py` covering `RLMChatCompletion.response` extraction (commit `720d97b`).

3. **(HIGH) Rotate the `<REDACTED>` Neo4j password and purge from history.** Then move `_demo_neo4j.py` under `scripts/` with env-var auth.

4. **(MEDIUM) Convert `tests/*.py` to pytest** with proper `test_*` functions and a `conftest.py`. Add a CI workflow that runs them.

5. **(MEDIUM) Centralize magic numbers** (`raw_max_chars`, `context_max_chars`, `chunk_chars`, `verify_batch_size`, `max_tokens`, `max_iterations`) in `experiments/common.py` and drop hardcoded duplicates in `run_all.py`, `_cli.py`, `build_kg.py`, `rlm_baseline.py`, `rlm_graph_baseline.py`.

6. **(MEDIUM) Replace `subprocess.call` per-cell** with in-process dispatch in `experiments/run_all.py:182` to save the 15-30s cold-start overhead, and to let `KeyboardInterrupt` cleanly stop the run.

7. **(MEDIUM) Extract-once / commit-twice** in `experiments/build_kg.py` so the no-scallop and scallop sessions share LLM extraction. Halves KG-build cost.

8. **(LOW) Drop `scli` from git** in favour of a download-on-first-run helper.

9. **(LOW) Implement or delete the `seed` argument** in `experiments/common.iter_pilot_examples:59`.

10. **(LOW) Replace `extract_seed_entities` regex** with a small NER pass (commit `rlm_graph_baseline.py:92` already TODOs this).

---

*Concerns audit: 2026-05-11*
