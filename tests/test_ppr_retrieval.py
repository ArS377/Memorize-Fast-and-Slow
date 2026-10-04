from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from neurosym.adapters.graph_source import GraphSource
from neurosym.adapters.kg_search import execute_search_knowledge_graph
from neurosym.adapters.ppr_index import PPRFactIndex
from neurosym.domain.retrieval_config import (
    PPRConfig,
    PPRRetrievalError,
    RetrievalConfig,
)


def _fact(
    fact_id: str,
    subject: str,
    object_value: str,
    *,
    example_id: str = "ex1",
    session_id: str = "session-1",
    predicate: str = "RELATED_TO",
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "example_id": example_id,
        "fact_id": fact_id,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "support_text": f"{subject} {predicate} {object_value}.",
        "confidence": "supported",
        "provenance": [
            {
                "document_id": f"doc-{fact_id}",
                "sentence_id": f"doc-{fact_id}:1",
            }
        ],
    }


class FakeDenseIndex:
    def __init__(
        self,
        session_id: str,
        facts: Sequence[Mapping[str, Any]],
        similarities: Mapping[str, float],
    ) -> None:
        self.facts = [dict(fact) for fact in facts]
        self.similarities = dict(similarities)
        self.manifest = type(
            "Manifest",
            (),
            {
                "identity": {
                    "schema_version": "dense_fact_index.v1",
                    "source_session_id": session_id,
                    "source_snapshot_sha256": f"sha-{session_id}",
                    "ordered_fact_digest": f"digest-{session_id}",
                }
            },
        )()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        scope: Any,
        predicates: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        wanted = {str(predicate).upper() for predicate in predicates or []}
        rows = []
        for fact in self.facts:
            if str(fact["session_id"]) not in set(scope.session_ids):
                continue
            if scope.mode == "example" and fact["example_id"] != scope.example_id:
                continue
            if wanted and str(fact["predicate"]).upper() not in wanted:
                continue
            value = dict(fact)
            value["_dense_similarity"] = self.similarities.get(
                str(fact["fact_id"]),
                -1.0,
            )
            rows.append(value)
        rows.sort(
            key=lambda row: (
                -float(row["_dense_similarity"]),
                str(row["fact_id"]),
            )
        )
        return rows[:top_k]


def _source(
    facts: Sequence[Dict[str, Any]],
    *,
    similarities: Mapping[str, float],
    ppr: Optional[PPRConfig] = None,
    memory_scope: str = "example",
    source_session_ids: Optional[List[str]] = None,
) -> GraphSource:
    indexes = {
        session_id: FakeDenseIndex(
            session_id,
            [fact for fact in facts if fact["session_id"] == session_id],
            similarities,
        )
        for session_id in sorted({str(fact["session_id"]) for fact in facts})
    }
    return GraphSource(
        fallback_facts=list(facts),
        session_id=str(facts[0]["session_id"]),
        memory_scope=memory_scope,
        source_session_ids=source_session_ids,
        retrieval_config=RetrievalConfig(
            mode="dense_ppr",
            ppr=ppr or PPRConfig(seed_count=1),
        ),
        dense_indexes=indexes,
    )


def test_dense_ppr_discovers_multi_hop_facts_without_scope_leakage() -> None:
    facts = [
        _fact("f1", "Alpha", "Bridge"),
        _fact("f2", "Bridge", "Middle"),
        _fact("f3", "Middle", "Answer"),
        _fact("f-disconnected", "Elsewhere", "Nowhere"),
        _fact(
            "f-private",
            "Bridge",
            "Remote",
            example_id="ex2",
        ),
        _fact("f-scope-isolated", "Remote", "Private Answer"),
    ]
    source = _source(
        facts,
        similarities={
            "f1": 0.95,
            "f2": 0.2,
            "f3": 0.1,
            "f-disconnected": 0.05,
            "f-private": 0.99,
            "f-scope-isolated": 0.04,
        },
    )

    outcome = source.retrieve(
        query="Find the answer connected to Alpha",
        seed_entities=[],
        example_id="ex1",
        top_k=10,
        hops=1,
    )

    fact_ids = [row["fact_id"] for row in outcome.rows]
    assert fact_ids[0] == "f1"
    assert {"f2", "f3"}.issubset(fact_ids)
    assert "f-disconnected" not in fact_ids
    assert "f-private" not in fact_ids
    assert "f-scope-isolated" not in fact_ids
    propagated = next(row for row in outcome.rows if row["fact_id"] == "f3")
    assert propagated["_retrieval"]["was_dense_seed"] is False
    assert propagated["_retrieval"]["matched_branches"] == ["ppr"]
    assert outcome.metadata["ppr"]["hops_applied"] is False
    assert outcome.metadata["branch_counts"]["dense"] == 1
    assert outcome.metadata["branch_counts"]["ppr"] == 3


def test_dense_ppr_returns_legitimate_empty_result_below_threshold() -> None:
    facts = [_fact("f1", "Alpha", "Bridge")]
    source = _source(
        facts,
        similarities={"f1": 0.4},
        ppr=PPRConfig(seed_count=1, similarity_threshold=0.5),
    )

    response = execute_search_knowledge_graph(
        {"query": "unrelated lowercase query"},
        source,
        "ex1",
    )

    assert response["status"] == "ok"
    assert response["results"] == []
    assert response["empty_reason"] == "no_matches"
    assert response["retrieval"]["effective_mode"] == "dense_ppr"
    assert response["retrieval"]["ppr"]["seed_fact_ids"] == []


def test_dense_ppr_tool_preserves_fact_contract_and_exposes_scores() -> None:
    facts = [
        _fact("f1", "Alpha", "Bridge"),
        _fact("f2", "Bridge", "Answer"),
    ]
    source = _source(facts, similarities={"f1": 0.9, "f2": 0.1})

    response = execute_search_knowledge_graph(
        {"query": "Find Alpha's answer", "top_k": 10, "hops": 1},
        source,
        "ex1",
    )

    assert response["status"] == "ok"
    assert response["retrieval"]["configured_mode"] == "dense_ppr"
    assert response["retrieval"]["ppr_index_identity"]["fact_count"] == 2
    result = response["results"][0]
    assert result["fact_id"] == "f1"
    assert result["retrieval_mode"] == "dense_ppr"
    assert result["ppr_rank"] == 1
    assert result["ppr_score"] == result["score"]
    assert result["was_dense_seed"] is True
    assert result["score_components"]["dense_seed_similarity"] == pytest.approx(0.9)
    assert result["provenance"][0]["sentence_id"] == "doc-f1:1"


def test_dense_ppr_connects_only_trusted_sessions() -> None:
    facts = [
        _fact("f1", "Alpha", "Shared", session_id="session-1"),
        _fact("f2", "shared", "Answer", session_id="session-2"),
        _fact("f3", "Shared", "Excluded", session_id="session-3"),
    ]
    source = _source(
        facts,
        similarities={"f1": 0.95, "f2": -0.5, "f3": 0.99},
        memory_scope="session_set",
        source_session_ids=["session-1", "session-2"],
    )

    outcome = source.retrieve(
        query="Find the cross-session answer",
        seed_entities=[],
        example_id="unused",
        top_k=10,
        hops=4,
    )

    assert [row["fact_id"] for row in outcome.rows] == ["f1", "f2"]
    assert outcome.rows[1]["_retrieval"]["was_dense_seed"] is False


def test_ppr_rejects_seed_snapshot_mismatch_and_nonconvergence() -> None:
    facts = [_fact("f1", "Alpha", "Bridge")]
    dense_index = FakeDenseIndex("session-1", facts, {"f1": 0.9})
    index = PPRFactIndex.from_dense_indexes(
        {"session-1": dense_index},
        config=PPRConfig(),
    )
    source = _source(facts, similarities={"f1": 0.9})
    mismatched_seed = {
        **_fact("unknown", "Alpha", "Bridge"),
        "_dense_similarity": 0.9,
    }

    with pytest.raises(PPRRetrievalError) as mismatch:
        index.search(
            seed_rows=[mismatched_seed],
            scope=source.trusted_scope("ex1"),
            top_k=10,
            config=PPRConfig(),
        )
    assert mismatch.value.code == "ppr_index_mismatch"

    with pytest.raises(PPRRetrievalError) as nonconvergence:
        index.search(
            seed_rows=[{**facts[0], "_dense_similarity": 0.9}],
            scope=source.trusted_scope("ex1"),
            top_k=10,
            config=PPRConfig(max_iterations=1, tolerance=1e-30),
        )
    assert nonconvergence.value.code == "ppr_nonconvergence"


def test_ppr_config_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        PPRConfig(damping=1.0)
    with pytest.raises(ValueError):
        PPRConfig(similarity_threshold=1.1)
    assert PPRConfig(damping=0.7).to_dict()["restart_probability"] == pytest.approx(
        0.3
    )


def test_dense_ppr_threshold_and_ties_are_deterministic() -> None:
    facts = [
        _fact("f1", "Alpha", "Left"),
        _fact("f2", "Alpha", "Right"),
    ]
    source = _source(
        facts,
        similarities={"f1": 0.5, "f2": 0.5},
        ppr=PPRConfig(seed_count=2, similarity_threshold=0.5),
    )

    first = source.retrieve(
        query="Alpha branches",
        seed_entities=[],
        example_id="ex1",
        top_k=10,
        hops=1,
    )
    second = source.retrieve(
        query="Alpha branches",
        seed_entities=[],
        example_id="ex1",
        top_k=10,
        hops=4,
    )

    assert [row["fact_id"] for row in first.rows] == ["f1", "f2"]
    assert [row["fact_id"] for row in second.rows] == ["f1", "f2"]
    assert first.rows[0]["_retrieval"]["ppr_score"] == pytest.approx(
        first.rows[1]["_retrieval"]["ppr_score"]
    )
    assert first.metadata["ppr"]["seed_fact_ids"] == ["f1", "f2"]


def test_dense_ppr_uses_only_facts_present_in_dense_snapshot() -> None:
    accepted = [
        _fact("f1", "Alpha", "Bridge"),
        _fact("f2", "Bridge", "Answer"),
    ]
    rejected = _fact("f-rejected", "Bridge", "Wrong")
    dense_index = FakeDenseIndex(
        "session-1",
        accepted,
        {"f1": 0.9, "f2": 0.1},
    )
    source = GraphSource(
        fallback_facts=[*accepted, rejected],
        session_id="session-1",
        retrieval_config=RetrievalConfig(
            mode="dense_ppr",
            ppr=PPRConfig(seed_count=1),
        ),
        dense_indexes={"session-1": dense_index},
    )

    outcome = source.retrieve(
        query="Alpha answer",
        seed_entities=[],
        example_id="ex1",
        top_k=10,
        hops=2,
    )

    assert [row["fact_id"] for row in outcome.rows] == ["f1", "f2"]
    assert source.ppr_index.manifest.fact_count == 2
    with pytest.raises(ValueError):
        PPRConfig(temperature=0.0)
