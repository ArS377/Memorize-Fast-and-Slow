from __future__ import annotations

from typing import Sequence

import numpy as np

from longbench_kg_pipeline import enrich_fact_provenance
from neurosym.adapters.chunk_selection import (
    ChunkSelector,
    chunk_selection_query,
    source_evidence_snapshot,
)
from neurosym.domain.retrieval_config import EmbeddingConfig


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_calls += 1
        return np.asarray(
            [
                [0.0, 1.0] if "Kalamang" in text else [1.0, 0.0]
                for text in texts
            ],
            dtype=np.float32,
        )

    def encode_query(self, _text: str) -> np.ndarray:
        self.query_calls += 1
        return np.asarray([0.0, 1.0], dtype=np.float32)

    def metadata(self):
        return {
            "model_name": "fake/bge",
            "requested_revision": "test",
            "resolved_revision": "resolved-test",
            "sentence_transformers_version": "fake-1",
            "vector_dimension": 2,
        }


def _chunks():
    return [
        [{"sent_id": 0, "text": "Apples grow in orchards."}],
        [{"sent_id": 1, "text": "Kalamang is spoken in East Indonesia."}],
        [{"sent_id": 2, "text": "A distant unrelated paragraph."}],
    ]


def _example():
    return {
        "_id": "example-1",
        "question": "Where is Kalamang spoken?",
        "choice_A": "Europe",
        "choice_B": "East Indonesia",
        "choice_C": "Canada",
        "choice_D": "Australia",
        "answer": "SECRET_GOLD_ANSWER",
    }


def test_hybrid_selection_retrieves_a_relevant_noninitial_chunk() -> None:
    embedder = FakeEmbedder()
    selector = ChunkSelector(
        mode="hybrid",
        rrf_k=60,
        embedding_config=EmbeddingConfig(
            model="fake/bge",
            requested_revision="test",
        ),
        embedder=embedder,
    )

    selected = selector.select(_example(), _chunks(), limit=1)

    assert [item.chunk_index for item in selected] == [1]
    assert selected[0].selection_method == "hybrid_rrf"
    assert selected[0].bm25_rank == 1
    assert selected[0].dense_rank == 1
    assert embedder.document_calls == 1
    assert embedder.query_calls == 1


def test_first_and_unlimited_selection_do_not_load_embeddings() -> None:
    embedder = FakeEmbedder()
    first = ChunkSelector(mode="first", embedder=embedder)
    hybrid = ChunkSelector(mode="hybrid", embedder=embedder)

    assert [item.chunk_index for item in first.select(_example(), _chunks(), limit=2)] == [0, 1]
    assert [item.chunk_index for item in hybrid.select(_example(), _chunks(), limit=None)] == [0, 1, 2]
    assert embedder.document_calls == 0
    assert embedder.query_calls == 0


def test_chunk_selection_query_excludes_the_gold_answer() -> None:
    query = chunk_selection_query(_example())

    assert "Where is Kalamang spoken?" in query
    assert "B) East Indonesia" in query
    assert "SECRET_GOLD_ANSWER" not in query


def test_selection_audit_preserves_source_chunk_identity() -> None:
    embedder = FakeEmbedder()
    selector = ChunkSelector(
        mode="hybrid",
        embedding_config=EmbeddingConfig(
            model="fake/bge",
            requested_revision="test",
        ),
        embedder=embedder,
    )
    selected = selector.select(_example(), _chunks(), limit=1)[0]

    record = selected.audit_record(
        example_id="example-1",
        session_id="session-1",
        total_chunks=3,
        query_sha256="query-hash",
        embedding=selector.embedding_metadata,
    )

    assert record["chunk_id"] == "example-1:chunk:1"
    assert record["sentence_ids"] == [1]
    assert record["selection_method"] == "hybrid_rrf"
    assert record["embedding"]["resolved_revision"] == "resolved-test"


def test_source_evidence_snapshot_includes_selected_and_unselected_sentences() -> None:
    selector = ChunkSelector(mode="hybrid", embedder=FakeEmbedder())
    selected = selector.select(_example(), _chunks(), limit=1)

    snapshot = source_evidence_snapshot(
        _example(),
        _chunks(),
        selected,
        session_id="session-1",
        selection_mode="hybrid",
    )

    assert len(snapshot) == 3
    assert [record["selected_for_extraction"] for record in snapshot] == [
        False,
        True,
        False,
    ]
    selected_sentence = snapshot[1]
    assert selected_sentence["sentence_id"] == "example-1:1"
    assert selected_sentence["document_id"] == "example-1"
    assert selected_sentence["source_chunk_id"] == "example-1:chunk:1"
    assert selected_sentence["selection_rank"] == 1
    assert selected_sentence["text"] == "Kalamang is spoken in East Indonesia."
    assert len(selected_sentence["text_sha256"]) == 64
    assert all(
        "SECRET_GOLD_ANSWER" not in str(record)
        for record in snapshot
    )

    fact = {
        "support_text": selected_sentence["text"],
        "provenance": [{"title": "example-1", "sent_id": 1}],
    }
    enrich_fact_provenance(
        fact,
        example=_example(),
        chunk=selected[0].records,
        chunk_index=selected[0].chunk_index,
        extractor_model="extractor",
        verifier_model="verifier",
        run_id="run-1",
    )
    assert fact["provenance"][0]["sentence_id"] == selected_sentence["sentence_id"]
