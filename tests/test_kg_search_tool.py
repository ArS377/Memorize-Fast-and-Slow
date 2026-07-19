from __future__ import annotations

import json
from pathlib import Path

from experiments.graph_context import GraphSource
from experiments.kg_search_tool import (
    DEFAULT_HOPS,
    DEFAULT_TOP_K,
    SEARCH_KNOWLEDGE_GRAPH_TOOL,
    execute_search_knowledge_graph,
)


FACTS = [
    {
        "example_id": "ex1",
        "session_id": "session-1",
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "East Indonesia",
        "fact_id": "fact-kalamang",
        "support_text": "Kalamang is spoken in East Indonesia.",
        "document_id": "doc-language",
        "confidence": "supported",
        "confidence_score": 0.91,
        "provenance": [
            {
                "title": "Language notes",
                "document_id": "doc-language",
                "sent_id": 7,
                "sentence_id": "doc-language:7",
                "source_span_start": 0,
                "source_span_end": 40,
            }
        ],
        "decision": {
            "status": "accept",
            "validator": "scallop",
            "reason": "No contradiction found.",
            "rule_params_version": "rules.v1",
        },
    },
    {
        "example_id": "ex2",
        "session_id": "session-1",
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "Wrong Example",
        "fact_id": "fact-wrong-example",
        "support_text": "This row must not cross the example boundary.",
        "provenance": [{"document_id": "doc-wrong", "sent_id": 1}],
    },
]


def test_tool_schema_is_json_serializable_and_scope_free() -> None:
    encoded = json.dumps(SEARCH_KNOWLEDGE_GRAPH_TOOL)

    assert "search_knowledge_graph" in encoded
    properties = SEARCH_KNOWLEDGE_GRAPH_TOOL["function"]["parameters"]["properties"]
    assert "example_id" not in properties
    assert "session_id" not in properties
    assert "retrieval_mode" not in properties
    assert SEARCH_KNOWLEDGE_GRAPH_TOOL["function"]["parameters"]["additionalProperties"] is False


def test_request_and_response_fixtures_are_valid_json() -> None:
    fixtures = Path(__file__).parent / "fixtures"
    request = json.loads((fixtures / "kg_search_tool_request.json").read_text())
    response = json.loads((fixtures / "kg_search_tool_response.json").read_text())

    assert request["query"] == "Where is Kalamang spoken?"
    assert response["status"] == "ok"
    assert response["request"] == request
    assert response["results"][0]["fact_id"] == "fact-kalamang"

    source = GraphSource(
        fallback_facts=FACTS[:1],
        session_id="session-1",
        memory_scope="example",
    )
    actual = execute_search_knowledge_graph(request, source, "ex1")
    actual["retrieval"]["branch_latency_seconds"] = {"sparse": 0.0, "dense": 0.0}
    assert actual == response


def test_adapter_returns_structured_sparse_result_without_an_llm() -> None:
    source = GraphSource(
        fallback_facts=FACTS,
        session_id="session-1",
        memory_scope="example",
    )

    response = execute_search_knowledge_graph(
        {"query": "Where is Kalamang spoken?"},
        source,
        "ex1",
    )

    assert response["status"] == "ok"
    assert response["error"] is None
    assert response["request"]["seed_entities"] == ["Where", "Kalamang"]
    assert response["request"]["top_k"] == DEFAULT_TOP_K
    assert response["request"]["hops"] == DEFAULT_HOPS
    assert response["scope"] == {
        "example_id": "ex1",
        "session_id": "session-1",
        "memory_scope": "example",
    }
    assert response["result_count"] == 1

    result = response["results"][0]
    assert result["fact_id"] == "fact-kalamang"
    assert result["subject"] == "Kalamang"
    assert result["predicate"] == "SPOKEN_IN"
    assert result["object"] == "East Indonesia"
    assert result["support_text"] == "Kalamang is spoken in East Indonesia."
    assert result["support_text_truncated"] is False
    assert result["document_id"] == "doc-language"
    assert result["provenance"][0]["sentence_id"] == "doc-language:7"
    assert result["provenance_truncated"] is False
    assert result["retrieval_mode"] == "sparse"
    assert isinstance(result["score"], float)
    assert result["score"] == result["score_components"]["total"]
    assert result["graph_path"]["edges"][0]["fact_id"] == "fact-kalamang"
    assert result["scallop"] == {
        "decision": "accept",
        "validator": "scallop",
        "reason": "No contradiction found.",
        "rule_version": "rules.v1",
    }
    assert result["scope"]["example_id"] == "ex1"
    assert response["working_memory"]["relevant_fact_ids"] == [result["fact_id"]]
    assert "provenance_complete" in response["working_memory"]
    assert "Wrong Example" not in json.dumps(response)
    json.dumps(response, allow_nan=False)


def test_missing_document_provenance_remains_explicit() -> None:
    fact = {
        "example_id": "ex-no-document",
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "East Indonesia",
        "fact_id": "fact-no-document",
        "support_text": "Kalamang is spoken in East Indonesia.",
        "provenance": [{"sent_id": 3}],
    }
    source = GraphSource(fallback_facts=[fact], session_id="session-1")

    response = execute_search_knowledge_graph(
        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
        source,
        "ex-no-document",
    )

    assert response["results"][0]["document_id"] is None
    assert response["results"][0]["scope"]["example_id"] == "ex-no-document"


def test_explicit_seed_preserves_current_case_insensitive_substring_matching() -> None:
    source = GraphSource(fallback_facts=FACTS, session_id="session-1")

    response = execute_search_knowledge_graph(
        {
            "query": "where is this language used?",
            "seed_entities": ["kalam"],
            "predicates": ["spoken in"],
            "top_k": 1,
            "hops": 1,
        },
        source,
        "ex1",
    )

    assert response["status"] == "ok"
    assert response["result_count"] == 1
    assert response["results"][0]["fact_id"] == "fact-kalamang"


def test_legitimate_empty_result_is_ok() -> None:
    source = GraphSource(fallback_facts=FACTS, session_id="session-1")

    response = execute_search_knowledge_graph(
        {"query": "lowercase terms produce no heuristic entity seed"},
        source,
        "ex1",
    )

    assert response["status"] == "ok"
    assert response["results"] == []
    assert response["result_count"] == 0
    assert response["empty_reason"] == "no_seed_entities"
    assert response["error"] is None


def test_model_cannot_override_trusted_scope() -> None:
    source = GraphSource(fallback_facts=FACTS, session_id="trusted-session")

    response = execute_search_knowledge_graph(
        {
            "query": "Where is Kalamang spoken?",
            "example_id": "ex2",
            "session_id": "attacker-session",
        },
        source,
        "ex1",
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "invalid_arguments"
    assert response["scope"]["example_id"] == "ex1"
    assert response["scope"]["session_id"] == "trusted-session"
    assert response["results"] == []


def test_invalid_arguments_are_reported_without_calling_backend() -> None:
    class Source:
        session_id = "session-1"
        memory_scope = "example"

        def rows_for(self, **kwargs):
            raise AssertionError("backend must not be called")

    response = execute_search_knowledge_graph(
        {"query": "Kalamang", "top_k": 0, "hops": 99, "retrieval_mode": "dense"},
        Source(),
        "ex1",
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "invalid_arguments"
    assert response["error"]["retryable"] is True
    assert len(response["error"]["details"]) == 3


def test_timeout_is_distinct_from_valid_empty_results() -> None:
    class TimeoutSource:
        session_id = "session-1"
        memory_scope = "example"

        def rows_for(self, **kwargs):
            raise TimeoutError("driver timed out")

    response = execute_search_knowledge_graph(
        {"query": "Kalamang"},
        TimeoutSource(),
        "ex1",
    )

    assert response["status"] == "error"
    assert response["error"] == {
        "code": "backend_timeout",
        "message": "Knowledge graph retrieval timed out.",
        "retryable": True,
    }
    assert response["empty_reason"] is None


def test_backend_failure_does_not_expose_exception_message() -> None:
    class BrokenSource:
        session_id = "session-1"
        memory_scope = "session"

        def rows_for(self, **kwargs):
            raise RuntimeError("bolt://user:secret@private-host")

    response = execute_search_knowledge_graph(
        {"query": "Kalamang"},
        BrokenSource(),
        "ex1",
    )

    encoded = json.dumps(response)
    assert response["status"] == "error"
    assert response["error"]["code"] == "backend_failure"
    assert response["error"]["retryable"] is False
    assert response["error"]["details"] == ["exception_type=RuntimeError"]
    assert "private-host" not in encoded
    assert "secret" not in encoded


def test_invalid_backend_response_is_a_structured_failure() -> None:
    class InvalidSource:
        session_id = "session-1"
        memory_scope = "example"

        def rows_for(self, **kwargs):
            return None

    response = execute_search_knowledge_graph(
        {"query": "Kalamang"},
        InvalidSource(),
        "ex1",
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "backend_failure"
    assert response["error"]["details"] == ["response_type=NoneType"]
