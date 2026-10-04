# Qwen sparse-tool smoke test

Phase 3 keeps the existing lexical/entity retrieval behind
`search_knowledge_graph`; it measures the native tool-call orchestration, not
a new retrieval algorithm. The committed cases are in
`tests/fixtures/qwen_sparse_smoke_cases.json` and are deliberately independent
of `data.jsonl`.

Run deterministic adapter, native-loop, live-command, and metric checks:

```bash
python -m pytest \
  tests/test_sparse_tool_adapter_smoke.py \
  tests/test_rlm_retrieval.py \
  tests/test_qwen_tool_smoke.py \
  tests/test_sparse_baseline.py
```

A live Qwen/vLLM run requires the Phase 1 tool adapter and Phase 2 Qwen-first
runner. Start vLLM with native Hermes tool parsing:

```bash
vllm serve Qwen/Qwen3-4B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 --port 8000
```

Then run the Phase 2 smoke command against the fixture file, writing one JSON
record per case to a temporary path (never at import time):

```bash
python -m experiments.qwen_tool_smoke \
  --fixtures tests/fixtures/qwen_sparse_smoke_cases.json \
  --vllm-base-url http://localhost:8000/v1 \
  --output /tmp/qwen_sparse_trace.jsonl

python -m experiments.sparse_baseline \
  --input /tmp/qwen_sparse_trace.jsonl \
  --output results/sparse_baseline/metrics.json
```

Each trace record must include `tool_call_count`, `retrieved_fact_count`,
`no_hit`, `termination_reason`, `cited_fact_ids`, `latency_seconds`, and
`error`, plus the full call/result/retry trace. The `scope_isolation` case
must never return `scope-foreign`. `paraphrase_miss` and `valid_zero` are
valid no-hit outcomes; malformed arguments, repeated calls, and timeouts must
be reported as explicit termination/error states, never treated as evidence.

The deterministic tests inject malformed, repeated, and timed-out native calls.
The live command leaves tool selection to Qwen and fails if the complete run
does not contain at least one native tool call or violates fixture scope.

The baseline reports hit rate, no-hit rate, Recall@k for labelled facts,
answer accuracy over labelled answer cases, mean tool calls, mean latency, and
error rate. Tool failures are excluded from the legitimate no-hit count. Save these
only below `results/sparse_baseline/`; future dense/hybrid runs should use
their own directories.
