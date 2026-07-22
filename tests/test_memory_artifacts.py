from __future__ import annotations

import json

import pytest

from memory_artifacts import MemoryScope, compile_working_memory, transition_for


FACTS = [
    {
        "fact_id": "f1",
        "subject": "Alice",
        "predicate": "WORKS_AT",
        "object": "CompanyX",
        "provenance": [{"document_id": "doc", "sentence_id": "doc:1"}],
    },
    {
        "fact_id": "f2",
        "subject": "CompanyX",
        "predicate": "LOCATED_IN",
        "object": "Paris",
        "provenance": [],
    },
]


def test_scope_requires_trusted_boundaries() -> None:
    scope = MemoryScope(mode="example", session_ids=("run",), example_id="ex1")
    assert scope.to_dict() == {
        "mode": "example",
        "session_ids": ["run"],
        "example_id": "ex1",
    }
    with pytest.raises(ValueError, match="at least two"):
        MemoryScope(mode="session_set", session_ids=("run",))
    with pytest.raises(ValueError, match="exactly one"):
        MemoryScope(mode="session", session_ids=("a", "b"))


def test_compiler_preserves_provenance_and_marks_missing_entries() -> None:
    scope = MemoryScope(mode="example", session_ids=("run",), example_id="ex1")
    artifact = compile_working_memory(
        {
            "entity": "Alice",
            "selected_fact_ids": ["f1", "f2"],
            "confidence": "supported",
            "derived_facts": [{
                "subject": "Alice",
                "predicate": "WORKS_IN_CITY",
                "object": "Paris",
                "support_fact_ids": ["f1", "f2"],
            }],
        },
        FACTS,
        scope,
    )
    assert artifact.selected_fact_ids == ["f1", "f2"]
    assert artifact.missing_provenance_fact_ids == ["f2"]
    assert artifact.provenance_complete is False
    assert artifact.derived_facts[0]["provenance"][0]["document_id"] == "doc"
    json.dumps(artifact.to_dict())


def test_compiler_rejects_scope_override_and_unknown_citations() -> None:
    scope = MemoryScope(mode="example", session_ids=("run",), example_id="ex1")
    with pytest.raises(ValueError, match="trusted scope"):
        compile_working_memory(
            {"entity": "Alice", "selected_fact_ids": ["f1"], "session_id": "other"},
            FACTS,
            scope,
        )
    with pytest.raises(ValueError, match="unknown or out-of-scope"):
        compile_working_memory(
            {"entity": "Alice", "selected_fact_ids": ["fabricated"]},
            FACTS,
            scope,
        )


def test_artifact_revisions_and_transition_lineage_are_deterministic() -> None:
    scope = MemoryScope(mode="example", session_ids=("run",), example_id="ex1")
    first = compile_working_memory(
        {"entity": "Alice", "selected_fact_ids": ["f1"]}, FACTS, scope
    )
    second = compile_working_memory(
        {"entity": "Alice", "selected_fact_ids": ["f1"]}, FACTS, scope, first
    )
    transition = transition_for(
        second,
        decision="accept",
        reason="Valid",
        validator="scallop",
        rule_version="rules.v1",
    )
    assert second.revision == 2
    assert second.prior_artifact_id == first.artifact_id
    assert transition.before_artifact_id == first.artifact_id
    assert transition.after_artifact_id == second.artifact_id
    assert transition.committed is True


def test_redundant_derived_selected_fact_is_an_audited_noop() -> None:
    scope = MemoryScope(mode="example", session_ids=("run",), example_id="ex1")
    without_derived = compile_working_memory(
        {"entity": "Alice", "selected_fact_ids": ["f1"]}, FACTS, scope
    )
    redundant = compile_working_memory(
        {
            "entity": "Alice",
            "selected_fact_ids": ["f1"],
            "derived_facts": [
                {
                    "subject": " Alice ",
                    "predicate": "works_at",
                    "object": "CompanyX",
                    "support_fact_ids": ["f1"],
                }
            ],
        },
        FACTS,
        scope,
    )

    assert redundant.derived_facts == []
    assert redundant.artifact_id == without_derived.artifact_id
    novelty = next(
        entry
        for entry in redundant.constraint_trace
        if entry["check"] == "derived_fact_novelty"
    )
    assert novelty["status"] == "omitted_redundant"
    assert novelty["omitted_count"] == 1


def test_novel_derived_content_changes_artifact_identity() -> None:
    scope = MemoryScope(mode="example", session_ids=("run",), example_id="ex1")
    first = compile_working_memory(
        {
            "entity": "Alice",
            "selected_fact_ids": ["f1"],
            "derived_facts": [
                {
                    "subject": "Alice",
                    "predicate": "HAS_EMPLOYER",
                    "object": "CompanyX",
                    "support_fact_ids": ["f1"],
                }
            ],
        },
        FACTS,
        scope,
    )
    second = compile_working_memory(
        {
            "entity": "Alice",
            "selected_fact_ids": ["f1"],
            "derived_facts": [
                {
                    "subject": "Alice",
                    "predicate": "IS_EMPLOYED_BY",
                    "object": "CompanyX",
                    "support_fact_ids": ["f1"],
                }
            ],
        },
        FACTS,
        scope,
    )

    assert first.artifact_id != second.artifact_id
