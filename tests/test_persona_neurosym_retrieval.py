from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import experiments.persona_neurosym_retrieval as persona_retrieval
from experiments.persona_neurosym_retrieval import (
    HybridMemoryConfig,
    build_validated_condition_facts,
    open_hybrid_graph_source,
    retrieve_condition_facts,
)
from neurosym.adapters.graph_source import GraphSource
from neurosym.domain.retrieval_config import RetrievalConfig


@dataclass(frozen=True)
class Decision:
    decision: str
    reason: str = "fixture"
    replace_fact_id: str | None = None
    rejection_label: Any = None
    rule_params_version: str = "fixture.v1"


class Validator:
    """Accept fixture facts and replace the explicitly marked predecessor."""

    def validate(
        self,
        existing_facts: list[dict[str, Any]],
        new_fact: Mapping[str, Any],
    ) -> Decision:
        del existing_facts
        replace_fact_id = new_fact.get("replace_fact_id")
        if replace_fact_id:
            return Decision("replace", replace_fact_id=str(replace_fact_id))
        return Decision("accept")


class DenseIndex:
    """Return scoped fixture facts with deterministic dense similarities."""

    def __init__(self, facts: list[dict[str, Any]]) -> None:
        self.facts = facts
        self.manifest = SimpleNamespace(identity="fixture-dense-index")

    def search(self, query: str, *, top_k: int, scope: Any, predicates: Any) -> list[dict]:
        del query, predicates
        rows = [
            {**fact, "_dense_similarity": 0.9 - index * 0.1}
            for index, fact in enumerate(self.facts)
            if fact["example_id"] == scope.example_id
        ]
        return rows[:top_k]


class RecordingGraph:
    """Return live scoped rows and record requested graph depth."""

    def __init__(self, facts: list[dict[str, Any]]) -> None:
        self.facts = facts
        self.calls: list[dict[str, Any]] = []

    def query_context(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(dict(kwargs))
        return [
            fact
            for fact in self.facts
            if fact["example_id"] == kwargs["example_id"]
        ][: kwargs["limit"]]

    def close(self) -> None:
        return None


def _event(
    index: int,
    value: str,
    *,
    operation: str = "add",
    replace_fact_id: str | None = None,
    retracts: str | None = None,
    supersedes: str | None = None,
    duplicate_of: str | None = None,
    retracts_lineage: bool = False,
    transitions_from: str | None = None,
    valid_to: str | None = None,
) -> dict[str, Any]:
    event = {
        "event_id": f"history-001-event-{index}",
        "history_id": "history-001",
        "operation": operation,
        "surface_subject": "AsterArc",
        "surface_object": value,
        "model_text": f"User: I prefer {value}.\nAssistant: Understood.",
        "fact": {
            "fact_id": f"history-001-fact-{index}",
            "subject": "subject-001",
            "predicate": "PREFERS",
            "object": f"value-001-{index}",
            "qualifiers": {"scope": "default", "source_authority": "direct_user"},
            "temporal": {"valid_from": f"2025-0{index}-01", "valid_to": valid_to},
            "confidence": "supported",
            "confidence_score": 0.95,
        },
    }
    if replace_fact_id is not None:
        event["fact"]["replace_fact_id"] = replace_fact_id
    if retracts is not None:
        event["retracts"] = retracts
    if supersedes is not None:
        event["supersedes"] = supersedes
    if duplicate_of is not None:
        event["duplicate_of"] = duplicate_of
    if retracts_lineage:
        event["retracts_lineage"] = True
    if transitions_from is not None:
        event["transitions_from"] = transitions_from
    return event


def _turn(index: int, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": f"turn-{index}",
        "history_id": event["history_id"],
        "text": event["model_text"],
        "stream_event": event,
    }


def test_condition_facts_are_causal_surface_only_and_scallop_gated() -> None:
    first = _event(1, "cedar tea")
    second = _event(2, "mint tea", replace_fact_id=first["fact"]["fact_id"])
    retraction = _event(3, "mint tea", operation="retract", retracts=second["fact"]["fact_id"])
    future = _event(4, "oolong tea")
    turns = [_turn(index, event) for index, event in enumerate((first, second, retraction, future), 1)]
    conditions = [
        {
            "evaluation_input_id": "condition-before-retraction",
            "history_id": "history-001",
            "checkpoint_turn_index": 2,
            "query_text": "What does AsterArc prefer?",
        },
        {
            "evaluation_input_id": "condition-after-retraction",
            "history_id": "history-001",
            "checkpoint_turn_index": 3,
            "query_text": "What does AsterArc prefer?",
        },
    ]

    facts, ledger = build_validated_condition_facts(
        conditions,
        turns,
        Validator(),
        session_id="persona-session",
    )

    before = [row for row in facts if row["example_id"] == "condition-before-retraction"]
    after = [row for row in facts if row["example_id"] == "condition-after-retraction"]
    assert [row["object"] for row in before] == ["mint tea"]
    assert after == []
    assert all(row["subject"] == "AsterArc" for row in facts)
    assert all("value-" not in str(row) and "subject-" not in str(row) for row in facts)
    assert ledger["condition-before-retraction"][-1]["decision"] == "replace"
    assert ledger["condition-after-retraction"][-1]["decision"] == "retract"


def test_event_supersession_and_lineage_retraction_update_the_snapshot() -> None:
    original = _event(1, "cedar tea")
    correction = _event(
        2,
        "mint tea",
        supersedes=original["fact"]["fact_id"],
    )
    duplicate = _event(
        3,
        "mint tea",
        duplicate_of=correction["fact"]["fact_id"],
    )
    retraction = _event(
        4,
        "mint tea",
        operation="retract",
        retracts=duplicate["fact"]["fact_id"],
        retracts_lineage=True,
    )
    turns = [
        _turn(index, event)
        for index, event in enumerate((original, correction, duplicate, retraction), 1)
    ]
    conditions = [
        {
            "evaluation_input_id": "after-correction",
            "history_id": "history-001",
            "checkpoint_turn_index": 2,
            "query_text": "What does AsterArc prefer?",
        },
        {
            "evaluation_input_id": "after-lineage-retraction",
            "history_id": "history-001",
            "checkpoint_turn_index": 4,
            "query_text": "What does AsterArc prefer?",
        },
    ]

    facts, ledger = build_validated_condition_facts(
        conditions,
        turns,
        Validator(),
        session_id="persona-session",
    )

    corrected = [row for row in facts if row["example_id"] == "after-correction"]
    retracted = [
        row for row in facts if row["example_id"] == "after-lineage-retraction"
    ]
    assert [row["object"] for row in corrected] == ["mint tea"]
    assert retracted == []
    assert ledger["after-correction"][-1]["event_replay_removed_fact_ids"] == [
        original["fact"]["fact_id"]
    ]
    assert set(ledger["after-lineage-retraction"][-1]["removed_fact_ids"]) == {
        correction["fact"]["fact_id"],
        duplicate["fact"]["fact_id"],
    }
    assert len({row["fact_id"] for row in facts}) == len(facts)


def test_non_overlapping_transition_preserves_temporal_history() -> None:
    original = _event(1, "cedar tea", valid_to="2025-01-31")
    transition = _event(
        2,
        "mint tea",
        transitions_from=original["fact"]["fact_id"],
    )
    turns = [_turn(1, original), _turn(2, transition)]
    condition = {
        "evaluation_input_id": "after-transition",
        "history_id": "history-001",
        "checkpoint_turn_index": 2,
        "query_text": "What did AsterArc prefer?",
    }

    facts, ledger = build_validated_condition_facts(
        [condition], turns, Validator(), session_id="persona-session"
    )

    assert [row["object"] for row in facts] == ["cedar tea", "mint tea"]
    assert ledger["after-transition"][-1][
        "event_replay_preserved_transition_fact_ids"
    ] == [original["fact"]["fact_id"]]


def test_retrieval_uses_real_hybrid_rrf_telemetry() -> None:
    facts = [
        {
            "fact_id": "fact-1",
            "source_fact_id": "source-fact-1",
            "source_event_id": "source-event-1",
            "session_id": "persona-session",
            "example_id": "condition-1",
            "subject": "AsterArc",
            "predicate": "PREFERS",
            "object": "cedar tea",
            "support_text": "AsterArc directly chose cedar tea.",
            "confidence": "supported",
            "confidence_score": 0.95,
            "provenance": [],
        },
        {
            "fact_id": "fact-2",
            "session_id": "persona-session",
            "example_id": "condition-1",
            "subject": "AsterArc",
            "predicate": "PREFERS",
            "object": "mint tea",
            "support_text": "AsterArc later chose mint tea.",
            "confidence": "supported",
            "confidence_score": 0.95,
            "provenance": [],
        },
    ]
    source = GraphSource(
        graph=RecordingGraph(facts),
        fallback_facts=facts,
        session_id="persona-session",
        memory_scope="example",
        retrieval_config=RetrievalConfig(mode="hybrid", failure_policy="error"),
        dense_indexes={"persona-session": DenseIndex(facts)},
    )

    retrieved = retrieve_condition_facts(
        [
            {
                "evaluation_input_id": "condition-1",
                "query_text": "What does AsterArc prefer now?",
            }
        ],
        source,
        top_k=2,
        hops=2,
    )

    row = retrieved["condition-1"]
    assert [fact["fact_id"] for fact in row["rows"]] == ["fact-1", "fact-2"]
    assert row["metadata"]["configured_mode"] == "hybrid"
    assert row["metadata"]["effective_mode"] == "hybrid"
    assert row["metadata"]["degraded"] is False
    assert row["metadata"]["branch_counts"] == {"sparse": 2, "dense": 2}
    assert row["metadata"]["rrf"]["k"] == 60
    assert row["metadata"]["sparse_backend"] == "neo4j_n_hop"
    assert row["metadata"]["graph_traversal_applied"] is True
    assert row["metadata"]["hops_requested"] == 2
    assert len(source.graph.calls) == 1
    graph_call = source.graph.calls[0]
    assert "AsterArc" in graph_call["seed_entities"]
    assert graph_call["hops"] == 2
    assert graph_call["example_id"] == "condition-1"
    assert graph_call["session_id"] == "persona-session"
    assert graph_call["session_ids"] is None


def test_retrieval_rejects_jsonl_only_graph_source() -> None:
    source = GraphSource(
        fallback_facts=[],
        session_id="persona-session",
        memory_scope="example",
        retrieval_config=RetrievalConfig(mode="hybrid", failure_policy="error"),
        dense_indexes={"persona-session": DenseIndex([])},
    )

    with pytest.raises(ValueError, match="live Neo4j"):
        retrieve_condition_facts([], source, top_k=2, hops=2)


def test_open_hybrid_source_commits_prevalidated_facts_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    facts = [
        {
            "fact_id": "fact-1",
            "session_id": "persona-session",
            "example_id": "condition-1",
            "subject": "AsterArc",
            "predicate": "PREFERS",
            "object": "cedar tea",
            "support_text": "AsterArc chose cedar tea.",
            "provenance": [],
        }
    ]

    class FakeNeo4jGraph(RecordingGraph):
        instance: "FakeNeo4jGraph | None" = None

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(facts)
            self.kwargs = kwargs
            self.cleared: list[tuple[str, bool]] = []
            self.committed: list[list[dict[str, Any]]] = []
            FakeNeo4jGraph.instance = self

        def clear_session(self, session_id: str, delete_orphans: bool = False) -> None:
            self.cleared.append((session_id, delete_orphans))

        def commit_facts(
            self, rows: list[dict[str, Any]], session_id: str | None = None
        ) -> int:
            assert session_id is None
            self.committed.append(rows)
            return len(rows)

        def export_facts(self) -> list[dict[str, Any]]:
            return list(facts)

    dense_index = DenseIndex(facts)
    monkeypatch.setattr(persona_retrieval, "Neo4jGraph", FakeNeo4jGraph)
    monkeypatch.setattr(
        persona_retrieval,
        "ensure_dense_index",
        lambda **kwargs: dense_index,
    )
    config = HybridMemoryConfig(
        index_root=tmp_path,
        embedding_model_id="BAAI/fixture",
        embedding_model_path=tmp_path / "embedding",
        embedding_revision="revision-1",
        embedding_device="cpu",
        embedding_batch_size=2,
        top_k=2,
        hops=2,
        rrf_k=60,
        neo4j_uri="bolt://neo4j.invalid:7687",
        neo4j_user="neo4j",
        neo4j_password="fixture-password",
        neo4j_database="neo4j",
    )

    source = open_hybrid_graph_source(
        facts,
        session_id="persona-session",
        validator_url="http://scallop.invalid",
        config=config,
    )

    graph = FakeNeo4jGraph.instance
    assert graph is not None
    assert graph.kwargs == {
        "uri": "bolt://neo4j.invalid:7687",
        "user": "neo4j",
        "password": "fixture-password",
        "database": "neo4j",
        "session_id": "persona-session",
        "validator_url": "http://scallop.invalid",
        "require_scallop": True,
    }
    assert graph.cleared == [("persona-session", False)]
    assert graph.committed == [facts]
    assert source.is_live is True
    assert source.dense_indexes == {"persona-session": dense_index}
