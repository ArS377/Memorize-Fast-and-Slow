# Qwen Knowledge-Graph Search Tool

`search_knowledge_graph` is the stable boundary between an LLM agent and the
repository's knowledge-graph retrieval code. The tool currently exposes the
existing sparse entity and predicate search without importing Qwen, OpenAI,
vLLM, RLM, or Neo4j-specific APIs into the tool contract.

The executable contract lives in
[`experiments/kg_search_tool.py`](../experiments/kg_search_tool.py). Agent code
should import the schema and executor from that module rather than copying the
schema into a prompt.

## Quick start

The adapter works without an LLM:

```python
import json

from experiments.graph_context import GraphSource
from experiments.kg_search_tool import execute_search_knowledge_graph

facts = [{
    "example_id": "ex1",
    "subject": "Kalamang",
    "predicate": "SPOKEN_IN",
    "object": "East Indonesia",
    "fact_id": "fact-kalamang",
    "support_text": "Kalamang is spoken in East Indonesia.",
    "provenance": [{"document_id": "doc-language", "sent_id": 7}],
}]
source = GraphSource(
    fallback_facts=facts,
    session_id="session-1",
    memory_scope="example",
)

response = execute_search_knowledge_graph(
    {"query": "Where is Kalamang spoken?"},
    source,
    example_id="ex1",
)
print(json.dumps(response, ensure_ascii=False, indent=2))
```

The response has `status="ok"`, contains the matching fact in `results`, and
can be passed directly to `json.dumps` for a native tool-result message.

## Native Qwen registration

Use the exported OpenAI-compatible tool definition:

```python
from experiments.kg_search_tool import SEARCH_KNOWLEDGE_GRAPH_TOOL

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=[SEARCH_KNOWLEDGE_GRAPH_TOOL],
)
```

For Qwen3, serve vLLM with Hermes-style tool-call parsing:

```bash
vllm serve Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

See the [Qwen function-calling documentation](https://qwen.readthedocs.io/en/stable/framework/function_call.html)
for the native assistant tool-call and tool-result message format.

## When Qwen should call the tool

Call `search_knowledge_graph` when the question depends on stored facts rather
than general model knowledge, especially when it asks about an entity,
relationship, prior document, or multi-hop chain. Send the question to Qwen
before retrieval so the model chooses the retrieval intent.

Qwen may answer without a tool call only when the surrounding application
allows an unsupported answer. For benchmark questions that must be grounded in
the graph, the agent should require retrieval before accepting a final answer.

Do not treat `results=[]` as permission to guess. Reformulate the query, try a
more precise seed, or return an evidence-insufficient result.

## Query reformulation

Use this order after a valid empty result:

1. Remove explanatory wording and keep the exact entity or quoted phrase.
2. Put likely graph entity names in `seed_entities`.
3. Remove an overly narrow `predicates` filter.
4. Use an intermediate entity returned by a previous call as the next seed.
5. Stop after the agent's configured call limit and report insufficient
   evidence.

The current sparse heuristic extracts capitalized phrases and quoted text from
`query`. Lowercase paraphrases with no explicit `seed_entities` can therefore
produce `empty_reason="no_seed_entities"`. This behavior is intentional for
the sparse baseline and must not be described as semantic retrieval.

## Multi-hop retrieval

For a question such as "What is the capital of the country where Kalamang is
spoken?", use iterative calls:

```text
search(seed_entities=["Kalamang"], predicates=["SPOKEN_IN"])
  -> East Indonesia

search(seed_entities=["East Indonesia"], predicates=["PART_OF"])
  -> Indonesia

search(seed_entities=["Indonesia"], predicates=["CAPITAL_IS"])
  -> Jakarta
```

`hops` controls bounded traversal for live Neo4j retrieval. The JSONL fallback
keeps its existing lexical row-matching behavior and does not reconstruct a
true multi-edge path. Each result's `graph_path` therefore describes the
single stored relationship represented by that result. Agent traces should
preserve the sequence of tool calls to represent a multi-hop reasoning path.

## Function reference

```python
execute_search_knowledge_graph(
    arguments,
    graph_source,
    example_id,
) -> SearchToolResponse
```

- `arguments` is the untrusted JSON object produced by the model.
- `graph_source` implements `GraphSource.rows_for()` and owns the trusted
  session and memory scope.
- `example_id` is trusted application context. It must come from the current
  benchmark example, never from model arguments.

### Arguments

| Field | Type | Default | Limits | Meaning |
|---|---|---:|---|---|
| `query` | string | required | 1–2,000 characters | Natural-language retrieval query. |
| `seed_entities` | string array | `[]` | 10 items, 256 characters each | Entity names. Live Neo4j uses exact names; JSONL fallback uses case-insensitive substring matching. Empty invokes the current capitalized-phrase heuristic. |
| `predicates` | string array | `[]` | 8 items, 128 characters each | Optional relationship filters; `GraphSource` normalizes them to `UPPER_SNAKE_CASE`. |
| `retrieval_mode` | string | `"sparse"` | only `"sparse"` | Stable mode selector reserved for later dense and hybrid implementations. |
| `top_k` | integer | `10` | 1–50 | Maximum number of fact records requested from the backend. |
| `hops` | integer | `2` | 1–4 | Maximum live-graph traversal depth. |

`additionalProperties` is false. In particular, `example_id`, `session_id`,
and `memory_scope` are not valid model arguments. Attempts to provide them
return `invalid_arguments` without querying the backend.

### Resource limits

The executor applies these fixed limits after validation:

- 2,000 characters per query;
- 10 seed entities;
- 8 predicate filters;
- 50 returned records at most;
- 4 live-graph hops at most;
- 2,000 support-text characters per result;
- 10 provenance entries per result;
- 32,000 serialized characters across the `results` array.

When a field or record is shortened by an output limit, `truncated` is true.
Each result also exposes `support_text_truncated` and
`provenance_truncated`. The agent may issue a narrower follow-up query rather
than requesting a larger response.

## Response reference

Every response contains the same top-level keys:

| Field | Meaning |
|---|---|
| `status` | `"ok"` for a completed search, including zero matches; `"error"` when the call did not complete. |
| `tool` | Always `"search_knowledge_graph"`. |
| `request` | Validated arguments with defaults and derived seeds, or `null` when argument validation failed. |
| `scope` | Trusted `example_id`, `session_id`, and `memory_scope`. |
| `results` | Structured fact records. Always an array. |
| `result_count` | Number of records actually returned. |
| `truncated` | Whether backend or response limits omitted records. |
| `empty_reason` | `"no_seed_entities"`, `"no_matches"`, or `null`. |
| `error` | Structured error object, or `null` for completed searches. |

Each result record contains:

- `rank`, `fact_id`, `subject`, `predicate`, and `object`;
- `support_text`, `support_text_truncated`, `document_id`, sentence/span
  `provenance`, and `provenance_truncated`;
- `retrieval_mode` and `score`;
- a single-relationship `graph_path`;
- `scallop.decision`, `validator`, `reason`, and `rule_version`;
- trusted example/session `scope`.

The sparse `score` is the repository's existing evidence-quality context rank,
which combines confidence and provenance quality. It is not a lexical or
semantic similarity score.

Scallop fields are nullable. Older JSONL artifacts and graph relationships did
not persist every decision field or rule version. The adapter reports `null`
when metadata is unavailable instead of inferring that Scallop accepted a
fact.

## Empty results and errors

A legitimate empty search is successful:

```json
{
  "status": "ok",
  "results": [],
  "result_count": 0,
  "empty_reason": "no_matches",
  "error": null
}
```

An execution failure is not an empty search:

```json
{
  "status": "error",
  "results": [],
  "result_count": 0,
  "empty_reason": null,
  "error": {
    "code": "backend_timeout",
    "message": "Knowledge graph retrieval timed out.",
    "retryable": true
  }
}
```

### Error codes

| Code | Retryable | Meaning | Agent behavior |
|---|---:|---|---|
| `invalid_arguments` | yes | Schema, type, range, or unknown-field validation failed. | Correct the arguments before retrying. |
| `backend_timeout` | yes | The backend raised `TimeoutError`. | Retry within the application's call budget or stop cleanly. |
| `backend_failure` | no | Another backend exception occurred. | Do not treat it as no evidence; record the failure and stop or use an application-controlled fallback. |

Backend exception messages are not returned because they may contain Neo4j
credentials or internal hostnames. The response includes only the exception
type for diagnostics.

## Fact citations

The final answer must cite every fact it relies on using the exact returned
`fact_id`. A recommended machine-readable final-answer shape is:

```json
{
  "answer": "A",
  "cited_fact_ids": ["fact-kalamang"],
  "evidence_insufficient": false
}
```

The agent must not cite a fact from an earlier question, a rejected tool call,
or a result outside the current trace. Person 2's orchestration layer should
verify that every cited ID appears in a successful tool response from the
current example before scoring the answer.

## Fixtures and tests

Persons 2 and 3 can use these stable fixtures:

- [`tests/fixtures/kg_search_tool_request.json`](../tests/fixtures/kg_search_tool_request.json)
- [`tests/fixtures/kg_search_tool_response.json`](../tests/fixtures/kg_search_tool_response.json)

Focused behavior tests live in
[`tests/test_kg_search_tool.py`](../tests/test_kg_search_tool.py). Run them with:

```bash
python3 -m pytest -q tests/test_kg_search_tool.py
```

The tests cover schema serialization, sparse retrieval, trusted scope,
legitimate empty results, invalid arguments, timeouts, backend failures, and
credential-safe error reporting.

## Design boundary

```text
Qwen / agent loop (Person 2)
        |
        | native tool arguments
        v
experiments/kg_search_tool.py (Person 1)
        |
        | validated trusted call
        v
GraphSource.rows_for()
        |
        +-- Neo4j n-hop retrieval
        +-- JSONL sparse fallback
```

The adapter owns validation and response normalization. `GraphSource` owns
sparse retrieval and scope enforcement. The agent loop owns call limits,
conversation messages, retries, duplicate-call detection, final citations,
and trace persistence. Keeping these boundaries separate lets dense, hybrid,
or HippoRAG implementations adopt the same contract later without coupling
retrieval code to a specific model runtime.
