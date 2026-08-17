from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

from experiments.persona_neurosym_retrieval import (
    build_validated_condition_facts,
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
