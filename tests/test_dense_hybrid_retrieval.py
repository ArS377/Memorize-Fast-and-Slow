from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pytest

from experiments.dense_retrieval import (
    DenseFactIndex,
    DenseRetrievalError,
    ensure_dense_index,
    fact_text_v1,
    ordered_fact_digest,
    query_text_v1,
    session_index_path,
    snapshot_sha256,
)
from experiments.graph_context import GraphSource
from experiments.kg_search_tool import execute_search_knowledge_graph
from experiments.retrieval_config import EmbeddingConfig, RetrievalConfig


class FakeProvider:
    def __init__(self, *, model: str = "fake/bge", revision: str = "commit-1", invalid: bool = False) -> None:
        self.model = model
        self.revision = revision
        self.invalid = invalid
        self.document_calls: List[List[str]] = []
        self.query_calls: List[str] = []

    def _vector(self, text: str) -> np.ndarray:
        if self.invalid:
            return np.asarray([np.nan, 0.0, 0.0], dtype=np.float32)
        lowered = text.lower()
        if "kalamang" in lowered or "semantic language" in lowered:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "indonesia" in lowered:
            return np.asarray([0.8, 0.2, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_calls.append(list(texts))
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        return self._vector(text)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model,
            "requested_revision": None,
            "resolved_revision": self.revision,
            "sentence_transformers_version": "fake-1",
            "vector_dimension": 3,
        }


FACTS = [
    {
        "session_id": "session-1",
        "example_id": "ex1",
        "fact_id": "f-kalamang",
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "East Indonesia",
        "support_text": "Kalamang is a semantic language in East Indonesia.",
        "confidence": "supported",
        "provenance": [{"document_id": "doc-1", "sentence_id": "doc-1:1"}],
    },
    {
        "session_id": "session-1",
        "example_id": "ex2",
        "fact_id": "f-private",
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "Wrong Example",
        "support_text": "This fact belongs to another example.",
        "confidence": "supported",
        "provenance": [{"document_id": "doc-2", "sentence_id": "doc-2:1"}],
    },
]


def _config(tmp_path: Path, *, mode: str = "dense", failure_policy: str = "error") -> RetrievalConfig:
    return RetrievalConfig(
        mode=mode,
        index_root=tmp_path,
        embedding=EmbeddingConfig(model="fake/bge"),
        failure_policy=failure_policy,
    )


def _index(tmp_path: Path, facts: Sequence[Dict[str, Any]] = FACTS) -> DenseFactIndex:
    return ensure_dense_index(
        index_root=tmp_path,
        session_id="session-1",
        facts=facts,
        config=EmbeddingConfig(model="fake/bge"),
        provider=FakeProvider(),
    )


def test_embedding_text_contracts_are_exact() -> None:
    assert fact_text_v1(FACTS[0]) == (
        "subject: Kalamang\n"
        "predicate: SPOKEN_IN\n"
        "object: East Indonesia\n"
        "support: Kalamang is a semantic language in East Indonesia."
    )
    assert query_text_v1("Where is it spoken?") == (
        "Represent this sentence for searching relevant passages: Where is it spoken?"
    )


def test_dense_index_build_load_and_example_scope(tmp_path: Path) -> None:
    provider = FakeProvider()
    index = ensure_dense_index(
        index_root=tmp_path,
        session_id="session-1",
        facts=FACTS,
        config=EmbeddingConfig(model="fake/bge"),
        provider=provider,
    )
    loaded = DenseFactIndex.load(
        session_index_path(tmp_path, "session-1"),
        provider,
        config=EmbeddingConfig(model="fake/bge"),
        source_session_id="session-1",
        source_snapshot_sha256=snapshot_sha256(sorted(FACTS, key=lambda row: (row["session_id"], row["example_id"], row["fact_id"]))),
    )
    source = GraphSource(
        fallback_facts=FACTS,
        session_id="session-1",
        memory_scope="example",
        retrieval_config=_config(tmp_path),
        dense_indexes={"session-1": loaded},
    )

    outcome = source.retrieve(
        query="semantic language location",
        seed_entities=[],
        example_id="ex1",
        top_k=10,
        hops=2,
    )

    assert [row["fact_id"] for row in outcome.rows] == ["f-kalamang"]
    assert outcome.rows[0]["_retrieval"]["dense_rank"] == 1
    assert outcome.metadata["effective_mode"] == "dense"
    assert index.manifest.ordered_fact_digest == ordered_fact_digest(index.facts)
    assert provider.query_calls == [query_text_v1("semantic language location")]

    tool_response = execute_search_knowledge_graph(
        {"query": "semantic language location"},
        source,
        "ex1",
    )
    assert tool_response["status"] == "ok"
    assert tool_response["results"][0]["fact_id"] == "f-kalamang"
    assert tool_response["results"][0]["retrieval_mode"] == "dense"
    assert tool_response["results"][0]["dense_rank"] == 1


def test_dense_index_reuses_an_exact_manifest(tmp_path: Path) -> None:
    _index(tmp_path)
    provider = FakeProvider()

    reused = ensure_dense_index(
        index_root=tmp_path,
        session_id="session-1",
        facts=FACTS,
        config=EmbeddingConfig(model="fake/bge"),
        provider=provider,
    )

    assert reused.manifest.fact_count == len(FACTS)
    assert provider.document_calls == []


def test_dense_index_recovers_an_interrupted_atomic_swap(tmp_path: Path) -> None:
    _index(tmp_path)
    index_path = session_index_path(tmp_path, "session-1")
    previous = index_path.with_name(f".{index_path.name}.previous")
    index_path.replace(previous)

    recovered = DenseFactIndex.load(
        index_path,
        FakeProvider(),
        config=EmbeddingConfig(model="fake/bge"),
        source_session_id="session-1",
        source_snapshot_sha256=snapshot_sha256(sorted(FACTS, key=lambda row: (row["session_id"], row["example_id"], row["fact_id"]))),
    )

    assert recovered.manifest.fact_count == len(FACTS)
    assert index_path.is_dir()
    assert not previous.exists()


def test_dense_build_rejects_nan_duplicate_ids_and_corrupt_shape(tmp_path: Path) -> None:
    with pytest.raises(DenseRetrievalError, match="NaN") as invalid:
        ensure_dense_index(
            index_root=tmp_path / "invalid",
            session_id="session-1",
            facts=FACTS,
            config=EmbeddingConfig(model="fake/bge"),
            provider=FakeProvider(invalid=True),
        )
    assert invalid.value.code == "dense_backend_failure"

    with pytest.raises(DenseRetrievalError) as duplicate:
        ensure_dense_index(
            index_root=tmp_path / "duplicate",
            session_id="session-1",
            facts=[FACTS[0], dict(FACTS[0])],
            config=EmbeddingConfig(model="fake/bge"),
            provider=FakeProvider(),
        )
    assert duplicate.value.code == "dense_index_mismatch"

    _index(tmp_path)
    with pytest.raises(DenseRetrievalError) as mismatch:
        DenseFactIndex.load(
            session_index_path(tmp_path, "session-1"),
            FakeProvider(revision="different-commit"),
            config=EmbeddingConfig(model="fake/bge"),
            source_session_id="session-1",
            source_snapshot_sha256=snapshot_sha256(sorted(FACTS, key=lambda row: (row["session_id"], row["example_id"], row["fact_id"]))),
        )
    assert mismatch.value.code == "dense_index_mismatch"

    index_path = session_index_path(tmp_path, "session-1")
    np.save(index_path / "vectors.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(DenseRetrievalError) as corrupt:
        DenseFactIndex.load(
            index_path,
            FakeProvider(),
            config=EmbeddingConfig(model="fake/bge"),
            source_session_id="session-1",
            source_snapshot_sha256=snapshot_sha256(sorted(FACTS, key=lambda row: (row["session_id"], row["example_id"], row["fact_id"]))),
        )
    assert corrupt.value.code == "dense_index_mismatch"


def test_dense_index_rebuilds_when_source_snapshot_changes(tmp_path: Path) -> None:
    first = _index(tmp_path)
    changed = [dict(row) for row in FACTS]
    changed[0]["support_text"] = "Changed semantic support."
    second = _index(tmp_path, changed)

    assert first.manifest.source_snapshot_sha256 != second.manifest.source_snapshot_sha256
    assert first.manifest.ordered_fact_digest != second.manifest.ordered_fact_digest


def test_hybrid_rrf_deduplicates_and_uses_deterministic_ranks(tmp_path: Path) -> None:
    class FakeIndex:
        manifest = type("Manifest", (), {"identity": {"source_session_id": "session-1"}})()

        def search(self, query: str, *, top_k: int, scope: Any, predicates=None):
            dense_only = {
                "session_id": "session-1",
                "example_id": "ex1",
                "fact_id": "f-dense",
                "subject": "Other",
                "predicate": "RELATED_TO",
                "object": "Place",
                "support_text": "Semantic only.",
                "confidence": "supported",
                "provenance": [{"document_id": "doc-d", "sentence_id": "doc-d:1"}],
                "_dense_similarity": 0.99,
            }
            shared = dict(FACTS[0])
            shared["_dense_similarity"] = 0.9
            return [dense_only, shared]

    source = GraphSource(
        fallback_facts=FACTS[:1],
        session_id="session-1",
        memory_scope="example",
        retrieval_config=_config(tmp_path, mode="hybrid"),
        dense_indexes={"session-1": FakeIndex()},
    )

    outcome = source.retrieve(
        query="Where is Kalamang spoken?",
        seed_entities=["Kalamang"],
        example_id="ex1",
        top_k=10,
        hops=2,
    )

    assert [row["fact_id"] for row in outcome.rows] == ["f-kalamang", "f-dense"]
    assert outcome.rows[0]["_retrieval"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert outcome.rows[0]["_retrieval"]["matched_branches"] == ["sparse", "dense"]
    assert outcome.rows[1]["_retrieval"]["rrf_score"] == pytest.approx(1 / 61)


def test_hybrid_relaxes_invented_predicates_against_known_ontology(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    source = GraphSource(
        fallback_facts=FACTS,
        session_id="session-1",
        memory_scope="example",
        retrieval_config=_config(tmp_path, mode="hybrid"),
        dense_indexes={"session-1": index},
    )

    outcome = source.retrieve(
        query="relationship between training methods and model performance",
        seed_entities=["Kalamang"],
        example_id="ex1",
        top_k=10,
        hops=2,
        predicates=["impact", "relationship"],
    )

    assert [row["fact_id"] for row in outcome.rows] == ["f-kalamang"]
    assert outcome.metadata["predicate_filter"] == {
        "requested": ["IMPACT", "RELATIONSHIP"],
        "applied": [],
        "relaxed": True,
    }


def test_dense_ties_are_broken_by_fact_id(tmp_path: Path) -> None:
    tied = [
        {
            **FACTS[0],
            "fact_id": fact_id,
            "subject": subject,
            "support_text": "Same vector.",
        }
        for fact_id, subject in [("f-b", "Kalamang B"), ("f-a", "Kalamang A")]
    ]
    index = _index(tmp_path, tied)
    source = GraphSource(
        fallback_facts=tied,
        session_id="session-1",
        retrieval_config=_config(tmp_path),
        dense_indexes={"session-1": index},
    )

    outcome = source.retrieve(
        query="Kalamang",
        seed_entities=[],
        example_id="ex1",
        top_k=10,
        hops=2,
    )

    assert [row["fact_id"] for row in outcome.rows] == ["f-a", "f-b"]


def test_dense_session_set_searches_each_trusted_index(tmp_path: Path) -> None:
    first = [dict(FACTS[0])]
    second = [{**FACTS[0], "session_id": "session-2", "fact_id": "f-second"}]
    first_index = _index(tmp_path / "first", first)
    second_index = ensure_dense_index(
        index_root=tmp_path / "second",
        session_id="session-2",
        facts=second,
        config=EmbeddingConfig(model="fake/bge"),
        provider=FakeProvider(),
    )
    source = GraphSource(
        fallback_facts=first + second,
        session_id="session-1",
        memory_scope="session_set",
        source_session_ids=["session-1", "session-2"],
        retrieval_config=_config(tmp_path),
        dense_indexes={"session-1": first_index, "session-2": second_index},
    )

    outcome = source.retrieve(
        query="Kalamang",
        seed_entities=[],
        example_id="ex1",
        top_k=10,
        hops=2,
    )

    assert {row["fact_id"] for row in outcome.rows} == {"f-kalamang", "f-second"}


def test_fixed_and_tool_paths_share_ordering(tmp_path: Path) -> None:
    index = _index(tmp_path)
    source = GraphSource(
        fallback_facts=FACTS,
        session_id="session-1",
        retrieval_config=_config(tmp_path, mode="hybrid"),
        dense_indexes={"session-1": index},
    )
    direct = source.retrieve(
        query="Where is Kalamang spoken?",
        seed_entities=["Kalamang"],
        example_id="ex1",
        top_k=10,
        hops=2,
    )
    tool = execute_search_knowledge_graph(
        {"query": "Where is Kalamang spoken?", "seed_entities": ["Kalamang"]},
        source,
        "ex1",
    )

    assert [row["fact_id"] for row in direct.rows] == [
        row["fact_id"] for row in tool["results"]
    ]


def test_dense_failures_are_strict_or_explicitly_degraded(tmp_path: Path) -> None:
    failure = DenseRetrievalError("dense_index_unavailable", "missing index")
    strict = GraphSource(
        fallback_facts=FACTS[:1],
        session_id="session-1",
        retrieval_config=_config(tmp_path, mode="hybrid"),
        dense_failure=failure,
    )
    strict_response = execute_search_knowledge_graph(
        {"query": "Where is Kalamang spoken?", "seed_entities": ["Kalamang"]},
        strict,
        "ex1",
    )
    assert strict_response["status"] == "error"
    assert strict_response["error"]["code"] == "dense_index_unavailable"

    degraded = GraphSource(
        fallback_facts=FACTS[:1],
        session_id="session-1",
        retrieval_config=_config(tmp_path, mode="hybrid", failure_policy="sparse"),
        dense_failure=failure,
    )
    degraded_response = execute_search_knowledge_graph(
        {"query": "Where is Kalamang spoken?", "seed_entities": ["Kalamang"]},
        degraded,
        "ex1",
    )
    assert degraded_response["status"] == "ok"
    assert degraded_response["retrieval"]["configured_mode"] == "hybrid"
    assert degraded_response["retrieval"]["effective_mode"] == "sparse"
    assert degraded_response["retrieval"]["degraded"] is True
    assert degraded_response["retrieval"]["warning"]["code"] == "dense_index_unavailable"


def test_sparse_mode_never_imports_sentence_transformers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imported: List[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name.startswith("sentence_transformers"):
            imported.append(name)
            raise AssertionError("sparse retrieval imported sentence-transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    source = GraphSource(
        fallback_facts=FACTS[:1],
        session_id="session-1",
        retrieval_config=RetrievalConfig(mode="sparse", index_root=tmp_path),
    )
    response = execute_search_knowledge_graph(
        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
        source,
        "ex1",
    )

    assert response["status"] == "ok"
    assert imported == []
