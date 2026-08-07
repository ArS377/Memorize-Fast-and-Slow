from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

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


def chunk_selection_queries_per_option(
    example: Mapping[str, Any],
) -> List[Tuple[str, str]]:
    """One (label, query) pair per answer option, plus the bare question.

    A single blended query (question + all options concatenated) dilutes
    the signal for multi-hop questions where different options' supporting
    facts live in different parts of the document -- a chunk that's a
    strong match for option B alone can rank below a chunk that's a
    mediocre match for the whole blend. Scoring each option separately and
    combining downstream (max score per chunk) keeps each sub-question's
    signal intact.
    """
    question = str(example.get("question", "")).strip()
    pairs: List[Tuple[str, str]] = []
    if question:
        pairs.append(("question", question))
    for label in ("A", "B", "C", "D"):
        choice = str(example.get(f"choice_{label}", "")).strip()
        if choice:
            pairs.append((label, f"{question}\n{label}) {choice}" if question else choice))
    if not pairs:
        pairs.append(("question", question))
    return pairs


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
    matched_query_label: Optional[str] = None

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
            "matched_query_label": self.matched_query_label,
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
        per_option_queries: bool = False,
        relevance_floor: Optional[float] = None,
        relevance_floor_min_count: int = 3,
        mmr_lambda: Optional[float] = None,
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
        if relevance_floor is not None and not (0.0 <= relevance_floor <= 1.0):
            raise ValueError("relevance_floor must be between 0 and 1")
        if relevance_floor_min_count < 0:
            raise ValueError("relevance_floor_min_count must not be negative")
        if mmr_lambda is not None and not (0.0 <= mmr_lambda <= 1.0):
            raise ValueError("mmr_lambda must be between 0 and 1")
        self.mode = mode
        self.rrf_k = int(rrf_k)
        self.window_tokens = int(window_tokens)
        self.window_overlap_tokens = int(window_overlap_tokens)
        self.embedding_config = embedding_config or EmbeddingConfig()
        self._embedder = embedder
        self._embedding_metadata: Optional[Dict[str, Any]] = None
        self.per_option_queries = per_option_queries
        self.relevance_floor = relevance_floor
        self.relevance_floor_min_count = int(relevance_floor_min_count)
        self.mmr_lambda = mmr_lambda

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

        if self.per_option_queries:
            query_pairs = chunk_selection_queries_per_option(example)
        else:
            query_pairs = [("blended", chunk_selection_query(example))]
        query_pairs = [(label, q) for label, q in query_pairs if q.strip()]
        if not query_pairs:
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
        doc_window_offset: List[int] = []
        offset = 0
        for windows in windows_by_document:
            doc_window_offset.append(offset)
            offset += len(windows)

        self._embedding_metadata = dict(provider.metadata())
        self._embedding_metadata.update(
            {
                "chunk_window_tokens": self.window_tokens,
                "chunk_window_overlap_tokens": self.window_overlap_tokens,
                "chunk_window_aggregation": "max_cosine_similarity",
                "selection_query_count": len(query_pairs),
            }
        )

        # Track the best *raw* component score per chunk across sub-queries,
        # then fuse once at the end. Maxing already-RRF-fused per-query
        # scores instead would lose information: RRF collapses continuous
        # similarity into discrete ranks, so with a small candidate set a
        # narrow win and a perfect win on different sub-queries can produce
        # the identical fused value, erasing exactly the signal per-option
        # fan-out is meant to surface.
        n = len(rows)
        best_label: List[Optional[str]] = [None] * n
        best_bm25_score = [0.0] * n
        best_dense_score = [0.0] * n
        best_window_index = [0] * n
        best_window: List[Optional[EmbeddingTextWindow]] = [None] * n
        best_vector: List[Optional[np.ndarray]] = [None] * n
        has_score = [False] * n

        for label, query in query_pairs:
            bm25_scores = _bm25_scores(query, documents)
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

            for index in range(n):
                if not has_score[index] or dense_scores[index] > best_dense_score[index]:
                    best_dense_score[index] = dense_scores[index]
                    win_idx, win = winning_windows[index]
                    best_window_index[index] = win_idx
                    best_window[index] = win
                    best_vector[index] = window_vectors[
                        doc_window_offset[index] + win_idx
                    ]
                    best_label[index] = label
                if not has_score[index] or bm25_scores[index] > best_bm25_score[index]:
                    best_bm25_score[index] = bm25_scores[index]
                has_score[index] = True

        best_bm25_rank = _ranks(best_bm25_score)
        best_dense_rank = _ranks(best_dense_score)
        best_fused = [
            1.0 / (self.rrf_k + best_bm25_rank[index])
            + 1.0 / (self.rrf_k + best_dense_rank[index])
            for index in range(n)
        ]

        ordered = sorted(
            range(n),
            key=lambda index: (
                -best_fused[index],
                best_bm25_rank[index],
                best_dense_rank[index],
                index,
            ),
        )

        if self.relevance_floor is not None:
            # Filter on raw dense cosine similarity, not the RRF-fused score.
            # RRF is rank-based and its output range is compressed to
            # ~1/(k+1) regardless of how similar or dissimilar the true
            # matches are, so a "percent of max" floor on the fused score is
            # nearly always non-restrictive. Cosine similarity has a real
            # 0-1 scale where a relative floor is meaningful; survivors are
            # still ordered by the fused (hybrid) score below.
            top_dense = max((best_dense_score[i] for i in ordered), default=0.0)
            threshold = self.relevance_floor * top_dense
            candidate_indices = [i for i in ordered if best_dense_score[i] >= threshold]
            if len(candidate_indices) < self.relevance_floor_min_count:
                candidate_set = set(candidate_indices)
                for i in ordered:
                    if i in candidate_set:
                        continue
                    candidate_indices.append(i)
                    candidate_set.add(i)
                    if len(candidate_indices) >= self.relevance_floor_min_count:
                        break
            candidate_indices = candidate_indices[:limit]
        else:
            candidate_indices = ordered[:limit]

        if self.mmr_lambda is not None and len(candidate_indices) > 1:
            candidate_indices = self._mmr_rerank(
                candidate_indices, best_fused, best_vector, self.mmr_lambda
            )

        method_parts = ["hybrid_windowed_rrf"]
        if self.per_option_queries:
            method_parts.append("per_option")
        if self.relevance_floor is not None:
            method_parts.append("relevance_floor")
        if self.mmr_lambda is not None:
            method_parts.append("mmr")
        selection_method = "+".join(method_parts)

        return [
            SelectedChunk(
                chunk_index=index,
                records=rows[index],
                selection_rank=selection_rank,
                selection_method=selection_method,
                score=float(best_fused[index]),
                bm25_score=float(best_bm25_score[index]),
                bm25_rank=best_bm25_rank[index],
                dense_similarity=float(best_dense_score[index]),
                dense_rank=best_dense_rank[index],
                dense_window_count=len(windows_by_document[index]),
                dense_winning_window_index=best_window_index[index],
                dense_winning_token_start=best_window[index].token_start,
                dense_winning_token_end=best_window[index].token_end,
                dense_winning_token_count=best_window[index].token_count,
                matched_query_label=best_label[index],
            )
            for selection_rank, index in enumerate(candidate_indices, start=1)
        ]

    @staticmethod
    def _mmr_rerank(
        candidates: List[int],
        scores: List[float],
        vectors: List[Optional[np.ndarray]],
        lam: float,
    ) -> List[int]:
        """Maximal-marginal-relevance reorder: iteratively pick the candidate
        maximizing (relevance - similarity to what's already been picked),
        so selection spreads across distinct regions of the document instead
        of clustering around the single best-matching passage."""
        values = [scores[i] for i in candidates]
        lo, hi = min(values), max(values)
        span = hi - lo if hi > lo else 1.0
        norm_relevance = {i: (scores[i] - lo) / span for i in candidates}

        remaining = list(candidates)
        first = max(remaining, key=lambda i: scores[i])
        selected = [first]
        remaining.remove(first)

        while remaining:
            def mmr_key(i: int) -> float:
                vec_i = vectors[i]
                max_sim = max(
                    float(np.dot(vec_i, vectors[s]))
                    for s in selected
                    if vectors[s] is not None
                )
                return lam * norm_relevance[i] - (1.0 - lam) * max_sim

            nxt = max(remaining, key=mmr_key)
            selected.append(nxt)
            remaining.remove(nxt)
        return selected
