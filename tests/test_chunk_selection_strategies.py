from __future__ import annotations

from typing import Sequence

import numpy as np

from neurosym.adapters.chunk_selection import (
    ChunkSelector,
    chunk_selection_queries_per_option,
)
from neurosym.adapters.dense_index import EmbeddingTextWindow
from neurosym.domain.retrieval_config import EmbeddingConfig


class KeywordCountEmbedder:
    """Encodes [baseline=1, count(TOPIC_A), count(TOPIC_B), count(TOPIC_C)]
    so dense similarity is deterministic and hand-verifiable: a query's
    vector is dominated by whichever topic it mentions most, and a chunk
    matches best when its own topic dominates the query the same way."""

    TOPICS = ("TOPIC_A", "TOPIC_B", "TOPIC_C")

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def _vector(self, text: str) -> list:
        return [1.0] + [float(text.count(topic)) for topic in self.TOPICS]

    def document_windows(self, text: str, *, max_tokens: int, overlap_tokens: int):
        return [
            EmbeddingTextWindow(
                text=text, token_start=0, token_end=1, token_count=1
            )
        ]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_calls += 1
        return np.asarray([self._vector(t) for t in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return np.asarray(self._vector(text), dtype=np.float32)

    def metadata(self):
        return {
            "model_name": "keyword-count-fake",
            "requested_revision": "test",
            "resolved_revision": "test",
            "sentence_transformers_version": "fake-1",
            "vector_dimension": 4,
        }


def _needle_example():
    # choice_A/B/D all mention TOPIC_A (decoys); only choice_C is the true
    # needle (TOPIC_C). The blended query therefore skews hard toward
    # TOPIC_A, burying the chunk that actually answers option C.
    return {
        "_id": "needle-1",
        "question": "",
        "choice_A": "TOPIC_A TOPIC_A",
        "choice_B": "TOPIC_A TOPIC_B",
        "choice_C": "TOPIC_C",
        "choice_D": "TOPIC_A TOPIC_B",
    }


def _needle_chunks():
    return [
        [{"sent_id": 0, "text": "TOPIC_A content"}],
        [{"sent_id": 1, "text": "TOPIC_C content"}],
        # A second TOPIC_A chunk breaks BM25's incidental term-rarity tie
        # between TOPIC_A and TOPIC_C in a corpus this small.
        [{"sent_id": 2, "text": "TOPIC_A appears here too"}],
    ]


def test_blended_query_buries_the_needle_chunk() -> None:
    embedder = KeywordCountEmbedder()
    selector = ChunkSelector(
        mode="hybrid",
        rrf_k=60,
        embedding_config=EmbeddingConfig(model="fake/bge", requested_revision="test"),
        embedder=embedder,
        per_option_queries=False,
    )

    selected = selector.select(_needle_example(), _needle_chunks(), limit=1)

    assert [item.chunk_index for item in selected] == [0]


def test_per_option_fan_out_recovers_the_needle_chunk() -> None:
    embedder = KeywordCountEmbedder()
    selector = ChunkSelector(
        mode="hybrid",
        rrf_k=60,
        embedding_config=EmbeddingConfig(model="fake/bge", requested_revision="test"),
        embedder=embedder,
        per_option_queries=True,
    )

    selected = selector.select(_needle_example(), _needle_chunks(), limit=1)

    assert [item.chunk_index for item in selected] == [1]
    assert selected[0].matched_query_label == "C"
    assert selected[0].selection_method == "hybrid_windowed_rrf+per_option"


def test_chunk_selection_queries_per_option_includes_question_and_each_choice() -> None:
    example = {
        "question": "What happens?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
    }
    pairs = chunk_selection_queries_per_option(example)
    labels = [label for label, _ in pairs]
    assert labels == ["question", "A", "B"]
    assert pairs[0][1] == "What happens?"
    assert "Alpha" in pairs[1][1]
    assert "Beta" in pairs[2][1]


def _floor_example():
    return {"_id": "floor-1", "question": "", "choice_A": "TOPIC_A"}


def _floor_chunks():
    # one near-perfect directional match, one perfect match, one chunk that
    # only shares the embedder's constant baseline dimension with the query
    # (stands in for "unrelated" -- see KeywordCountEmbedder).
    return [
        [{"sent_id": 0, "text": "TOPIC_A TOPIC_A TOPIC_A"}],
        [{"sent_id": 1, "text": "TOPIC_A mentioned once"}],
        [{"sent_id": 2, "text": "completely unrelated filler"}],
        [{"sent_id": 3, "text": "yet more unrelated filler"}],
    ]


def test_relevance_floor_drops_low_scoring_chunks_but_keeps_minimum_count() -> None:
    embedder = KeywordCountEmbedder()
    selector = ChunkSelector(
        mode="hybrid",
        rrf_k=60,
        embedding_config=EmbeddingConfig(model="fake/bge", requested_revision="test"),
        embedder=embedder,
        relevance_floor=0.8,
        relevance_floor_min_count=1,
    )

    selected = selector.select(_floor_example(), _floor_chunks(), limit=3)

    selected_indices = [item.chunk_index for item in selected]
    assert 2 not in selected_indices
    assert 3 not in selected_indices
    assert selected[0].selection_method == "hybrid_windowed_rrf+relevance_floor"


def test_relevance_floor_min_count_tops_up_when_floor_excludes_too_many() -> None:
    embedder = KeywordCountEmbedder()
    selector = ChunkSelector(
        mode="hybrid",
        rrf_k=60,
        embedding_config=EmbeddingConfig(model="fake/bge", requested_revision="test"),
        embedder=embedder,
        relevance_floor=0.99,
        relevance_floor_min_count=3,
    )

    selected = selector.select(_floor_example(), _floor_chunks(), limit=3)

    assert len(selected) == 3


def test_mmr_rerank_prefers_diverse_candidate_over_near_duplicate() -> None:
    # Chunk 1 scores nearly as well as chunk 0 but points in almost the same
    # embedding direction (a near-duplicate passage); chunk 2 scores lower
    # but is orthogonal (a genuinely different part of the document). MMR
    # should still take the best-scoring chunk first, then prefer the
    # diverse chunk 2 over the redundant chunk 1.
    candidates = [0, 1, 2]
    scores = {0: 1.0, 1: 0.95, 2: 0.5}
    near_duplicate = np.array([0.99, 0.01])
    near_duplicate = near_duplicate / np.linalg.norm(near_duplicate)
    vectors = {
        0: np.array([1.0, 0.0]),
        1: near_duplicate,
        2: np.array([0.0, 1.0]),
    }

    reranked = ChunkSelector._mmr_rerank(candidates, scores, vectors, lam=0.5)

    assert reranked[0] == 0
    assert reranked[1] == 2


def test_mmr_lambda_is_wired_into_selection_method_and_does_not_crash() -> None:
    example = {"_id": "mmr-1", "question": "", "choice_A": "TOPIC_A", "choice_B": "TOPIC_B"}
    chunks = [
        [{"sent_id": 0, "text": "TOPIC_A TOPIC_A"}],
        [{"sent_id": 1, "text": "TOPIC_A TOPIC_A also"}],
        [{"sent_id": 2, "text": "TOPIC_B mentioned"}],
    ]
    selector = ChunkSelector(
        mode="hybrid",
        rrf_k=60,
        embedding_config=EmbeddingConfig(model="fake/bge", requested_revision="test"),
        embedder=KeywordCountEmbedder(),
        mmr_lambda=0.5,
    )

    selected = selector.select(example, chunks, limit=2)

    assert len(selected) == 2
    assert selected[0].selection_method == "hybrid_windowed_rrf+mmr"
