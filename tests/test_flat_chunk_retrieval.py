from __future__ import annotations

from typing import Sequence

import numpy as np

from neurosym.adapters.flat_chunk_retrieval import select_flat_chunks


class KeywordCountEmbedder:
    """Encodes [baseline=1, count(TOPIC_A), count(TOPIC_B)] so dense
    similarity is deterministic and hand-verifiable."""

    TOPICS = ("TOPIC_A", "TOPIC_B")

    def _vector(self, text: str) -> list:
        return [1.0] + [float(text.count(topic)) for topic in self.TOPICS]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._vector(t) for t in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self._vector(text), dtype=np.float32)

    def document_windows(self, text, *, max_tokens, overlap_tokens):
        raise NotImplementedError

    def metadata(self):
        return {"model_name": "keyword-count-fake"}


def _example(question: str) -> dict:
    return {
        "_id": "flat-chunk-1",
        "question": question,
        "context": (
            "TOPIC_A sentence one about apples. "
            "TOPIC_A sentence two about apples again. "
            "TOPIC_B sentence about the true needle answer. "
            "An unrelated filler sentence with neither keyword."
        ),
    }


def test_bm25_mode_ranks_by_keyword_overlap():
    result = select_flat_chunks(
        _example("Tell me about TOPIC_B"),
        mode="bm25",
        top_k=1,
        max_chunk_chars=40,
    )
    assert result.n_chunks == 1
    assert "TOPIC_B" in result.context
    assert "TOPIC_A" not in result.context


def test_dense_mode_ranks_by_embedding_similarity():
    result = select_flat_chunks(
        _example("Tell me about TOPIC_B"),
        mode="dense",
        top_k=1,
        max_chunk_chars=40,
        embedder=KeywordCountEmbedder(),
    )
    assert result.n_chunks == 1
    assert "TOPIC_B" in result.context


def test_dense_mode_requires_embedder():
    try:
        select_flat_chunks(
            _example("q"), mode="dense", top_k=1, max_chunk_chars=40
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_hybrid_mode_fuses_bm25_and_dense():
    result = select_flat_chunks(
        _example("Tell me about TOPIC_B"),
        mode="hybrid",
        top_k=2,
        max_chunk_chars=40,
        embedder=KeywordCountEmbedder(),
    )
    assert result.n_chunks == 2
    assert "TOPIC_B" in result.context


def test_top_k_truncates_selection():
    result = select_flat_chunks(
        _example("Tell me about TOPIC_A and TOPIC_B"),
        mode="bm25",
        top_k=1,
        max_chunk_chars=40,
    )
    assert result.n_chunks == 1
    assert len(result.selected_chunk_indices) == 1


def test_empty_context_returns_empty_result():
    result = select_flat_chunks(
        {"_id": "empty-1", "question": "q", "context": ""},
        mode="bm25",
        top_k=5,
        max_chunk_chars=40,
    )
    assert result.n_chunks == 0
    assert result.context == ""
