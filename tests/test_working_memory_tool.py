from __future__ import annotations

from neurosym.adapters.graph_source import GraphSource
from neurosym.adapters.working_memory_tool import (
    UPDATE_WORKING_MEMORY_TOOL,
    execute_update_working_memory,
)


FACT = {
    "fact_id": "f1",
    "subject": "Alice",
    "predicate": "WORKS_AT",
    "object": "CompanyX",
    "provenance": [{"document_id": "doc", "sentence_id": "doc:1"}],
}


def test_tool_schema_uses_hermes_compatible_string_types() -> None:
    def visit(value):
        if isinstance(value, dict):
            if "type" in value:
                assert isinstance(value["type"], str)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(UPDATE_WORKING_MEMORY_TOOL)


def test_update_compiles_only_returned_fact_ids() -> None:
    source = GraphSource(
        fallback_facts=[FACT], session_id="run", memory_scope="example"
    )
    response = execute_update_working_memory(
        {"entity": "Alice", "selected_fact_ids": ["f1"]},
        source,
        "ex1",
        [FACT],
        validate=False,
    )
    assert response["status"] == "ok"
    assert response["artifact"]["selected_fact_ids"] == ["f1"]
    assert response["transition"]["validator"] == "none"
    assert response["persisted"] is False


def test_update_rejects_fabricated_and_scope_arguments() -> None:
    source = GraphSource(
        fallback_facts=[FACT], session_id="run", memory_scope="example"
    )
    fabricated = execute_update_working_memory(
        {"entity": "Alice", "selected_fact_ids": ["fake"]},
        source,
        "ex1",
        [FACT],
        validate=True,
    )
    override = execute_update_working_memory(
        {"entity": "Alice", "selected_fact_ids": ["f1"], "session_id": "other"},
        source,
        "ex1",
        [FACT],
        validate=True,
    )
    assert fabricated["error"]["code"] == "invalid_memory_update"
    assert override["error"]["code"] == "invalid_memory_update"


def test_validated_update_fails_closed_without_live_graph() -> None:
    source = GraphSource(
        fallback_facts=[FACT], session_id="run", memory_scope="example"
    )
    response = execute_update_working_memory(
        {"entity": "Alice", "selected_fact_ids": ["f1"]},
        source,
        "ex1",
        [FACT],
        validate=True,
    )
    assert response["status"] == "error"
    assert response["error"]["code"] == "validator_unavailable"
