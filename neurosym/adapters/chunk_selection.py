from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np

from neurosym.adapters.dense_index import (
    EmbeddingTextWindow,
    SentenceTransformerEmbedder,
    query_text_v1,
)
from neurosym.domain.retrieval_config import EmbeddingConfig


CHUNK_SELECTION_MODES = {"first", "hybrid"}
SOURCE_EVIDENCE_SCHEMA_VERSION = "source_evidence_v1"


class ChunkEmbeddingProvider(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...
    def document_windows(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int,
    ) -> List[EmbeddingTextWindow]: ...
    def metadata(self) -> Mapping[str, Any]: ...


def chunk_selection_query(example: Mapping[str, Any]) -> str:
    parts = [str(example.get("question", "")).strip()]
    for label in ("A", "B", "C", "D"):
        choice = str(example.get(f"choice_{label}", "")).strip()
        if choice:
            parts.append(f"{label}) {choice}")
    return "\n".join(part for part in parts if part)


def chunk_text(records: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(
        str(record.get("text", "")).strip()
        for record in records
        if str(record.get("text", "")).strip()
    )


def _tokens(text: str) -> List[str]:
    return re.findall(r"[^\W_]+", str(text).casefold(), flags=re.UNICODE)


def _bm25_scores(query: str, documents: Sequence[str]) -> List[float]:
    tokenized = [_tokens(document) for document in documents]
    query_terms = set(_tokens(query))
    if not tokenized or not query_terms:
        return [0.0] * len(documents)
    document_count = len(tokenized)
    average_length = sum(len(tokens) for tokens in tokenized) / document_count
    if average_length <= 0:
        return [0.0] * document_count
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens).intersection(query_terms))

    k1 = 1.5
    b = 0.75
    scores: List[float] = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length_normalizer = k1 * (
            1.0 - b + b * len(tokens) / average_length
        )
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency < 1:
                continue
            frequency_in_corpus = document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1.0
                + (document_count - frequency_in_corpus + 0.5)
                / (frequency_in_corpus + 0.5)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0)
                / (frequency + length_normalizer)
            )
        scores.append(float(score))
    return scores


def _normalized(value: Any, *, rows: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("chunk embedding provider returned an invalid shape")
    if rows is not None and array.shape[0] != rows:
        raise ValueError("chunk embedding provider returned an invalid row count")
    if not np.isfinite(array).all():
        raise ValueError("chunk embedding provider returned NaN or infinity")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("chunk embedding provider returned a zero vector")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _ranks(scores: Sequence[float]) -> Dict[int, int]:
    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return {index: rank for rank, index in enumerate(ordered, start=1)}


@dataclass(frozen=True)
class SelectedChunk:
    chunk_index: int
    records: List[Dict[str, Any]]
    selection_rank: int
    selection_method: str
    score: Optional[float] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_similarity: Optional[float] = None
    dense_rank: Optional[int] = None
    dense_window_count: Optional[int] = None
    dense_winning_window_index: Optional[int] = None
    dense_winning_token_start: Optional[int] = None
    dense_winning_token_end: Optional[int] = None
    dense_winning_token_count: Optional[int] = None

    def audit_record(
        self,
        *,
        example_id: str,
        session_id: str,
        total_chunks: int,
        query_sha256: str,
        embedding: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        text = chunk_text(self.records)
        sentence_ids = [
            record.get("sent_id")
            for record in self.records
            if record.get("sent_id") is not None
        ]
        return {
            "session_id": str(session_id),
            "example_id": str(example_id),
            "chunk_id": f"{example_id}:chunk:{self.chunk_index}",
            "chunk_index": self.chunk_index,
            "total_chunks": int(total_chunks),
            "selection_rank": self.selection_rank,
            "selection_method": self.selection_method,
            "score": self.score,
            "score_components": {
                "bm25_score": self.bm25_score,
                "bm25_rank": self.bm25_rank,
                "dense_similarity": self.dense_similarity,
                "dense_rank": self.dense_rank,
                "dense_window_count": self.dense_window_count,
                "dense_winning_window_index": self.dense_winning_window_index,
                "dense_winning_token_start": self.dense_winning_token_start,
                "dense_winning_token_end": self.dense_winning_token_end,
                "dense_winning_token_count": self.dense_winning_token_count,
            },
            "sentence_ids": sentence_ids,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_chars": len(text),
            "query_sha256": query_sha256,
            "embedding": dict(embedding) if embedding is not None else None,
        }


def source_evidence_snapshot(
    example: Mapping[str, Any],
    chunks: Sequence[Sequence[Mapping[str, Any]]],
    selected: Sequence[SelectedChunk],
    *,
    session_id: str,
    selection_mode: str,
) -> List[Dict[str, Any]]:
    """Snapshot source sentences and their extraction-selection disposition."""
    example_id = str(example.get("_id", "unknown"))
    query = chunk_selection_query(example)
    query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
    selected_by_index = {item.chunk_index: item for item in selected}
    snapshot: List[Dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks):
        selected_chunk = selected_by_index.get(chunk_index)
        chunk_value = chunk_text(chunk)
        chunk_sha256 = hashlib.sha256(chunk_value.encode("utf-8")).hexdigest()
        for sentence_offset, record in enumerate(chunk):
            text = str(record.get("text", ""))
            sent_id = record.get("sent_id")
            document_id = str(record.get("document_id") or example_id)
            stable_sentence_id = (
                f"{document_id}:{sent_id}"
                if sent_id is not None
                else f"{example_id}:chunk:{chunk_index}:offset:{sentence_offset}"
            )
            snapshot.append(
                {
                    "schema_version": SOURCE_EVIDENCE_SCHEMA_VERSION,
                    "session_id": str(session_id),
                    "example_id": example_id,
                    "document_id": document_id,
                    "sentence_id": stable_sentence_id,
                    "source_chunk_id": f"{example_id}:chunk:{chunk_index}",
                    "source_chunk_index": chunk_index,
                    "source_chunk_sha256": chunk_sha256,
                    "sentence_offset_in_chunk": sentence_offset,
                    "title": record.get("title"),
                    "sent_id": sent_id,
                    "local_sent_id": record.get("local_sent_id"),
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "chunk_selection_mode": selection_mode,
                    "selected_for_extraction": selected_chunk is not None,
                    "selection_rank": (
                        selected_chunk.selection_rank
                        if selected_chunk is not None
                        else None
                    ),
                    "selection_method": (
                        selected_chunk.selection_method
                        if selected_chunk is not None
                        else None
                    ),
                    "query_sha256": query_sha256,
                }
            )
    return snapshot


class ChunkSelector:
    def __init__(
        self,
        *,
        mode: str = "first",
        rrf_k: int = 60,
        window_tokens: int = 448,
        window_overlap_tokens: int = 64,
        embedding_config: Optional[EmbeddingConfig] = None,
        embedder: Optional[ChunkEmbeddingProvider] = None,
    ) -> None:
        if mode not in CHUNK_SELECTION_MODES:
            raise ValueError(
                f"chunk selection mode must be one of {sorted(CHUNK_SELECTION_MODES)}"
            )
        if rrf_k < 1:
            raise ValueError("chunk selection RRF k must be at least 1")
        if window_tokens < 1:
            raise ValueError("chunk embedding window size must be at least 1")
        if (
            window_overlap_tokens < 0
            or window_overlap_tokens >= window_tokens
        ):
            raise ValueError(
                "chunk embedding window overlap must be non-negative and "
                "smaller than the window size"
            )
        self.mode = mode
        self.rrf_k = int(rrf_k)
        self.window_tokens = int(window_tokens)
        self.window_overlap_tokens = int(window_overlap_tokens)
        self.embedding_config = embedding_config or EmbeddingConfig()
        self._embedder = embedder
        self._embedding_metadata: Optional[Dict[str, Any]] = None

    @property
    def embedding_metadata(self) -> Optional[Dict[str, Any]]:
        return (
            dict(self._embedding_metadata)
            if self._embedding_metadata is not None
            else None
        )

    def _provider(self) -> ChunkEmbeddingProvider:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.embedding_config)
        return self._embedder

    def select(
        self,
        example: Mapping[str, Any],
        chunks: Sequence[Sequence[Mapping[str, Any]]],
        *,
        limit: Optional[int],
    ) -> List[SelectedChunk]:
        rows = [[dict(record) for record in chunk] for chunk in chunks]
        if limit is not None and limit <= 0:
            return []
        if limit is None or limit >= len(rows):
            return [
                SelectedChunk(
                    chunk_index=index,
                    records=records,
                    selection_rank=index + 1,
                    selection_method="all",
                )
                for index, records in enumerate(rows)
            ]
        if self.mode == "first":
            return [
                SelectedChunk(
                    chunk_index=index,
                    records=rows[index],
                    selection_rank=index + 1,
                    selection_method="first",
                )
                for index in range(limit)
            ]

        query = chunk_selection_query(example)
        if not query.strip():
            return [
                SelectedChunk(
                    chunk_index=index,
                    records=rows[index],
                    selection_rank=index + 1,
                    selection_method="first_empty_query",
                )
                for index in range(limit)
            ]
        documents = [chunk_text(records) for records in rows]
        bm25_scores = _bm25_scores(query, documents)
        provider = self._provider()
        windows_by_document = [
            provider.document_windows(
                document,
                max_tokens=self.window_tokens,
                overlap_tokens=self.window_overlap_tokens,
            )
            for document in documents
        ]
        if any(not windows for windows in windows_by_document):
            raise ValueError(
                "chunk embedding provider returned no windows for a document"
            )
        flattened_windows = [
            window
            for windows in windows_by_document
            for window in windows
        ]
        window_vectors = _normalized(
            provider.encode_documents(
                [window.text for window in flattened_windows]
            ),
            rows=len(flattened_windows),
        )
        query_vector = _normalized(
            provider.encode_query(query_text_v1(query)),
            rows=1,
        )[0]
        if window_vectors.shape[1] != query_vector.shape[0]:
            raise ValueError("chunk and query embedding dimensions differ")
        similarities = np.sum(
            window_vectors * query_vector[np.newaxis, :],
            axis=1,
            dtype=np.float32,
        )
        if not np.isfinite(similarities).all():
            raise ValueError(
                "chunk embedding similarity produced NaN or infinity"
            )
        window_scores = similarities.astype(float).tolist()
        dense_scores: List[float] = []
        winning_windows: List[tuple[int, EmbeddingTextWindow]] = []
        window_offset = 0
        for windows in windows_by_document:
            scores = window_scores[window_offset : window_offset + len(windows)]
            winning_index = max(
                range(len(scores)),
                key=lambda index: (scores[index], -index),
            )
            dense_scores.append(float(scores[winning_index]))
            winning_windows.append((winning_index, windows[winning_index]))
            window_offset += len(windows)
        self._embedding_metadata = dict(provider.metadata())
        self._embedding_metadata.update(
            {
                "chunk_window_tokens": self.window_tokens,
                "chunk_window_overlap_tokens": self.window_overlap_tokens,
                "chunk_window_aggregation": "max_cosine_similarity",
            }
        )

        bm25_ranks = _ranks(bm25_scores)
        dense_ranks = _ranks(dense_scores)
        fused_scores = [
            1.0 / (self.rrf_k + bm25_ranks[index])
            + 1.0 / (self.rrf_k + dense_ranks[index])
            for index in range(len(rows))
        ]
        ordered = sorted(
            range(len(rows)),
            key=lambda index: (
                -fused_scores[index],
                bm25_ranks[index],
                dense_ranks[index],
                index,
            ),
        )
        return [
            SelectedChunk(
                chunk_index=index,
                records=rows[index],
                selection_rank=selection_rank,
                selection_method="hybrid_windowed_rrf",
                score=float(fused_scores[index]),
                bm25_score=float(bm25_scores[index]),
                bm25_rank=bm25_ranks[index],
                dense_similarity=float(dense_scores[index]),
                dense_rank=dense_ranks[index],
                dense_window_count=len(windows_by_document[index]),
                dense_winning_window_index=winning_windows[index][0],
                dense_winning_token_start=winning_windows[index][1].token_start,
                dense_winning_token_end=winning_windows[index][1].token_end,
                dense_winning_token_count=winning_windows[index][1].token_count,
            )
            for selection_rank, index in enumerate(
                ordered[:limit],
                start=1,
            )
        ]
