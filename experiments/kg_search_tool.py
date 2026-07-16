"""Stable, model-independent tool contract for sparse KG retrieval.

The module intentionally has no OpenAI, Qwen, vLLM, Neo4j, or RLM imports.
Agent code supplies a ``GraphSource``-compatible object and receives a plain
JSON-serializable response that can be returned as a native tool message.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, TypedDict

from compiled_memory import retrieval_score_components
from experiments.graph_context import GraphSource, extract_seed_entities


TOOL_NAME = "search_knowledge_graph"
DEFAULT_RETRIEVAL_MODE = "sparse"
DEFAULT_TOP_K = 10
DEFAULT_HOPS = 2

MAX_QUERY_CHARS = 2_000
MAX_SEED_ENTITIES = 10
MAX_SEED_ENTITY_CHARS = 256
MAX_PREDICATES = 8
MAX_PREDICATE_CHARS = 128
MAX_TOP_K = 50
MAX_HOPS = 4
MAX_SUPPORT_TEXT_CHARS = 2_000
MAX_PROVENANCE_ENTRIES = 10
MAX_RESULTS_JSON_CHARS = 32_000


SEARCH_KNOWLEDGE_GRAPH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Search the validated knowledge graph for facts relevant to a question. "
            "Use precise entity names when possible. Follow entities returned by one "
            "call with another call when the question requires multi-hop reasoning."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_CHARS,
                    "description": "Natural-language retrieval query.",
                },
                "seed_entities": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_SEED_ENTITY_CHARS,
                    },
                    "maxItems": MAX_SEED_ENTITIES,
                    "default": [],
                    "description": (
                        "Optional entity names. Live Neo4j matches exact names; the "
                        "JSONL fallback keeps its current case-insensitive substring "
                        "matching. When omitted or empty, "
                        "the current sparse capitalized-phrase heuristic derives seeds "
                        "from query."
                    ),
                },
                "predicates": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_PREDICATE_CHARS,
                    },
                    "maxItems": MAX_PREDICATES,
                    "default": [],
                    "description": "Optional UPPER_SNAKE_CASE relationship filters.",
                },
                "retrieval_mode": {
                    "type": "string",
                    "enum": [DEFAULT_RETRIEVAL_MODE],
                    "default": DEFAULT_RETRIEVAL_MODE,
                    "description": "Retrieval implementation. Only sparse is available.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TOP_K,
                    "default": DEFAULT_TOP_K,
                    "description": "Maximum number of fact records to return.",
                },
                "hops": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_HOPS,
                    "default": DEFAULT_HOPS,
                    "description": (
                        "Maximum graph traversal depth. JSONL fallback retrieval keeps "
                        "its existing one-step lexical behavior."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


class SearchToolError(TypedDict, total=False):
    code: str
    message: str
    retryable: bool
    details: List[str]


class SearchToolResponse(TypedDict):
    status: Literal["ok", "error"]
    tool: str
    request: Optional[Dict[str, Any]]
    scope: Dict[str, Any]
    results: List[Dict[str, Any]]
    working_memory: Optional[Dict[str, Any]]
    result_count: int
    truncated: bool
    empty_reason: Optional[str]
    error: Optional[SearchToolError]


_ALLOWED_ARGUMENTS = {
    "query",
    "seed_entities",
    "predicates",
    "retrieval_mode",
    "top_k",
    "hops",
}


def _scope(graph_source: GraphSource, example_id: str) -> Dict[str, Any]:
    return {
        "example_id": str(example_id),
        "session_id": (
            str(graph_source.session_id)
            if getattr(graph_source, "session_id", None) is not None
            else None
        ),
        "memory_scope": str(getattr(graph_source, "memory_scope", "example")),
    }


def _error_response(
    *,
    graph_source: GraphSource,
    example_id: str,
    code: str,
    message: str,
    retryable: bool,
    details: Optional[List[str]] = None,
    request: Optional[Dict[str, Any]] = None,
) -> SearchToolResponse:
    error: SearchToolError = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if details:
        error["details"] = details
    return {
        "status": "error",
        "tool": TOOL_NAME,
        "request": request,
        "scope": _scope(graph_source, example_id),
        "results": [],
        "working_memory": None,
        "result_count": 0,
        "truncated": False,
        "empty_reason": None,
        "error": error,
    }


def _validated_string_list(
    value: Any,
    *,
    field: str,
    max_items: int,
    max_chars: int,
    errors: List[str],
) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{field} must be an array of strings")
        return []
    if len(value) > max_items:
        errors.append(f"{field} may contain at most {max_items} items")

    normalized: List[str] = []
    seen = set()
    for index, item in enumerate(value[:max_items]):
        if not isinstance(item, str):
            errors.append(f"{field}[{index}] must be a string")
            continue
        text = item.strip()
        if not text:
            errors.append(f"{field}[{index}] must not be empty")
            continue
        if len(text) > max_chars:
            errors.append(f"{field}[{index}] exceeds {max_chars} characters")
            continue
        if text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def _validate_arguments(arguments: Any) -> tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(arguments, Mapping):
        return None, ["arguments must be a JSON object"]

    unknown = sorted(str(key) for key in arguments.keys() if key not in _ALLOWED_ARGUMENTS)
    if unknown:
        errors.append(f"unsupported argument(s): {', '.join(unknown)}")

    query_value = arguments.get("query")
    if not isinstance(query_value, str):
        errors.append("query is required and must be a string")
        query = ""
    else:
        query = query_value.strip()
        if not query:
            errors.append("query must not be empty")
        elif len(query) > MAX_QUERY_CHARS:
            errors.append(f"query exceeds {MAX_QUERY_CHARS} characters")

    seed_entities = _validated_string_list(
        arguments.get("seed_entities", []),
        field="seed_entities",
        max_items=MAX_SEED_ENTITIES,
        max_chars=MAX_SEED_ENTITY_CHARS,
        errors=errors,
    )
    predicates = _validated_string_list(
        arguments.get("predicates", []),
        field="predicates",
        max_items=MAX_PREDICATES,
        max_chars=MAX_PREDICATE_CHARS,
        errors=errors,
    )

    retrieval_mode = arguments.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE)
    if retrieval_mode != DEFAULT_RETRIEVAL_MODE:
        errors.append("retrieval_mode must be 'sparse'")

    top_k = arguments.get("top_k", DEFAULT_TOP_K)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        errors.append("top_k must be an integer")
    elif not 1 <= top_k <= MAX_TOP_K:
        errors.append(f"top_k must be between 1 and {MAX_TOP_K}")

    hops = arguments.get("hops", DEFAULT_HOPS)
    if isinstance(hops, bool) or not isinstance(hops, int):
        errors.append("hops must be an integer")
    elif not 1 <= hops <= MAX_HOPS:
        errors.append(f"hops must be between 1 and {MAX_HOPS}")

    if errors:
        return None, errors

    return {
        "query": query,
        "seed_entities": seed_entities,
        "predicates": predicates,
        "retrieval_mode": DEFAULT_RETRIEVAL_MODE,
        "top_k": top_k,
        "hops": hops,
    }, []


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _first_nonempty(*values: Any) -> Optional[str]:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None


def _decision_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    compiled = row.get("compiled_memory")
    compiled = compiled if isinstance(compiled, dict) else {}
    decision = row.get("decision")
    if not isinstance(decision, dict):
        decision = compiled.get("decision")
    decision = decision if isinstance(decision, dict) else {}

    return {
        "decision": _first_nonempty(
            row.get("decision_status"),
            decision.get("status"),
        ),
        "validator": _first_nonempty(
            row.get("decision_validator"),
            decision.get("validator"),
        ),
        "reason": _first_nonempty(
            row.get("decision_reason"),
            decision.get("reason"),
        ),
        "rule_version": _first_nonempty(
            row.get("rule_params_version"),
            decision.get("rule_params_version"),
            compiled.get("rule_params_version"),
        ),
    }


def _provenance(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = row.get("provenance", [])
    if not isinstance(value, list):
        return []
    return [
        _json_safe(entry)
        for entry in value[:MAX_PROVENANCE_ENTRIES]
        if isinstance(entry, Mapping)
    ]


def _result_record(
    row: Dict[str, Any],
    *,
    rank: int,
    scope: Dict[str, Any],
    query: str,
    seed_entities: List[str],
) -> Dict[str, Any]:
    subject = str(row.get("subject", ""))
    predicate = str(row.get("predicate", ""))
    object_value = str(row.get("object", ""))
    fact_id = str(row.get("fact_id") or row.get("memory_id") or "")
    provenance_value = row.get("provenance", [])
    provenance_count = len(provenance_value) if isinstance(provenance_value, list) else 0
    provenance = _provenance(row)
    provenance_document = next(
        (
            entry.get("document_id") or entry.get("title")
            for entry in provenance
            if entry.get("document_id") or entry.get("title")
        ),
        None,
    )
    full_support_text = str(row.get("support_text", ""))
    support_text = full_support_text[:MAX_SUPPORT_TEXT_CHARS]
    score_components = retrieval_score_components(
        row,
        query=query,
        seed_entities=seed_entities,
    )

    return {
        "rank": rank,
        "fact_id": fact_id,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "support_text": support_text,
        "support_text_truncated": len(full_support_text) > MAX_SUPPORT_TEXT_CHARS,
        "document_id": _first_nonempty(
            row.get("document_id"),
            provenance_document,
        ),
        "provenance": provenance,
        "provenance_truncated": provenance_count > MAX_PROVENANCE_ENTRIES,
        "valid_from": _first_nonempty(row.get("valid_from")),
        "valid_to": _first_nonempty(row.get("valid_to")),
        "retrieval_mode": DEFAULT_RETRIEVAL_MODE,
        "score": score_components["total"],
        "score_components": score_components,
        "graph_path": {
            "nodes": [subject, object_value],
            "edges": [
                {
                    "fact_id": fact_id,
                    "predicate": predicate,
                    "source": subject,
                    "target": object_value,
                }
            ],
        },
        "scallop": _decision_metadata(row),
        "scope": {
            "example_id": _first_nonempty(row.get("example_id"), scope["example_id"]),
            "session_id": _first_nonempty(row.get("session_id"), scope["session_id"]),
            "memory_scope": scope["memory_scope"],
        },
    }


def _shape_working_memory(
    results: List[Dict[str, Any]],
    *,
    request: Dict[str, Any],
    scope: Dict[str, Any],
    truncated: bool,
    result_json_chars: int,
) -> Dict[str, Any]:
    missing: List[str] = []
    citations: List[Dict[str, Any]] = []
    temporal_intervals: List[Dict[str, Any]] = []
    constraint_trace: List[Dict[str, Any]] = []
    for result in results:
        fact_id = result["fact_id"]
        provenance = result.get("provenance", [])
        if not provenance:
            missing.append(fact_id)
        for entry in provenance:
            citations.append({"fact_id": fact_id, **entry})
        temporal = {
            "fact_id": fact_id,
            "valid_from": result.get("valid_from"),
            "valid_to": result.get("valid_to"),
        }
        if temporal["valid_from"] or temporal["valid_to"]:
            temporal_intervals.append(temporal)
        constraint_trace.append({"fact_id": fact_id, **result["scallop"]})
    scores = [float(result.get("score", 0.0)) for result in results]
    return {
        "entity": request["seed_entities"][0] if request["seed_entities"] else None,
        "relevant_fact_ids": [result["fact_id"] for result in results],
        "excluded_claims": [],
        "temporal_scope": {"intervals": temporal_intervals},
        "confidence": (
            "supported" if scores and sum(scores) / len(scores) >= 0.5 else "uncertain"
        ),
        "citations": citations,
        "constraint_trace": constraint_trace,
        "scope": scope,
        "provenance_complete": not missing,
        "missing_provenance_fact_ids": missing,
        "budget": {
            "top_k": request["top_k"],
            "hops": request["hops"],
            "result_json_chars": result_json_chars,
            "truncated": truncated,
        },
    }


def execute_search_knowledge_graph(
    arguments: Any,
    graph_source: GraphSource,
    example_id: str,
) -> SearchToolResponse:
    """Validate and execute one sparse knowledge-graph search tool call.

    ``example_id`` and the source's ``session_id`` are trusted application
    context, not model-controlled arguments. The returned object is safe to
    pass to ``json.dumps`` and then send back as a native tool result.
    """
    request, validation_errors = _validate_arguments(arguments)
    if request is None:
        return _error_response(
            graph_source=graph_source,
            example_id=example_id,
            code="invalid_arguments",
            message="The search_knowledge_graph arguments are invalid.",
            retryable=True,
            details=validation_errors,
        )

    seeds = list(request["seed_entities"])
    if not seeds:
        seeds = extract_seed_entities({"question": request["query"]})[:MAX_SEED_ENTITIES]
    request["seed_entities"] = seeds
    scope = _scope(graph_source, example_id)

    if not seeds:
        return {
            "status": "ok",
            "tool": TOOL_NAME,
            "request": request,
            "scope": scope,
            "results": [],
            "working_memory": _shape_working_memory(
                [], request=request, scope=scope, truncated=False, result_json_chars=2
            ),
            "result_count": 0,
            "truncated": False,
            "empty_reason": "no_seed_entities",
            "error": None,
        }

    try:
        rows = graph_source.rows_for(
            seed_entities=seeds,
            example_id=str(example_id),
            hops=request["hops"],
            limit_triples=min(MAX_TOP_K, request["top_k"] * 3),
            predicates=request["predicates"],
        )
    except Exception as exc:
        is_timeout = isinstance(exc, TimeoutError) or any(
            "timeout" in cls.__name__.lower()
            for cls in type(exc).__mro__
        )
        if is_timeout:
            return _error_response(
                graph_source=graph_source,
                example_id=example_id,
                code="backend_timeout",
                message="Knowledge graph retrieval timed out.",
                retryable=True,
                request=request,
            )
        return _error_response(
            graph_source=graph_source,
            example_id=example_id,
            code="backend_failure",
            message="Knowledge graph retrieval failed.",
            retryable=False,
            details=[f"exception_type={type(exc).__name__}"],
            request=request,
        )

    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        return _error_response(
            graph_source=graph_source,
            example_id=example_id,
            code="backend_failure",
            message="Knowledge graph retrieval returned an invalid response.",
            retryable=False,
            details=[f"response_type={type(rows).__name__}"],
            request=request,
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            -retrieval_score_components(
                dict(row), query=request["query"], seed_entities=seeds
            )["total"],
            str(row.get("fact_id", "")),
        ),
    )
    results: List[Dict[str, Any]] = []
    results_json_chars = 2
    truncated = False
    for row in ranked_rows[: request["top_k"]]:
        record = _result_record(
            dict(row),
            rank=len(results) + 1,
            scope=scope,
            query=request["query"],
            seed_entities=seeds,
        )
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        added_chars = len(encoded) + (1 if results else 0)
        if results_json_chars + added_chars > MAX_RESULTS_JSON_CHARS:
            truncated = True
            break
        results.append(record)
        results_json_chars += added_chars
        if record["support_text_truncated"] or record["provenance_truncated"]:
            truncated = True

    if len(ranked_rows) > len(results):
        truncated = True

    response: SearchToolResponse = {
        "status": "ok",
        "tool": TOOL_NAME,
        "request": request,
        "scope": scope,
        "results": results,
        "working_memory": _shape_working_memory(
            results,
            request=request,
            scope=scope,
            truncated=truncated,
            result_json_chars=results_json_chars,
        ),
        "result_count": len(results),
        "truncated": truncated,
        "empty_reason": "no_matches" if not results else None,
        "error": None,
    }
    # Keep JSON serializability as an enforced contract, not a best-effort hope.
    json.dumps(response, ensure_ascii=False, allow_nan=False)
    return response
