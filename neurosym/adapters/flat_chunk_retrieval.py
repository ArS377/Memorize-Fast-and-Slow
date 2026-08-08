"""Flat (non-KG) chunk retrieval for the BM25 / dense / hybrid RAG baselines.

Retrieves the top-k chunks of a single example's raw document via BM25,
dense embedding similarity, or their RRF fusion -- no fact extraction, no
graph. Reuses the same scoring primitives as chunk_selection.py (the
build-time chunk-selection step), just applied directly ahead of a
single-shot QA call instead of feeding fact extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional

import numpy as np

from neurosym.adapters.chunk_selection import ChunkEmbeddingProvider, _bm25_scores, _ranks, chunk_text

ChunkRetrievalMode = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class ChunkRAGResult:
    context: str
    n_chunks: int
    selected_chunk_indices: List[int]


def flat_chunk_query(example: Mapping[str, Any]) -> str:
    parts = [str(example.get("question", "")).strip()]
    for label in ("A", "B", "C", "D"):
        choice = str(example.get(f"choice_{label}", "")).strip()
        if choice:
            parts.append(f"{label}) {choice}")
    return "\n".join(part for part in parts if part)


def _dense_scores(
    query: str, texts: List[str], embedder: ChunkEmbeddingProvider
) -> List[float]:
    doc_vectors = np.asarray(embedder.encode_documents(texts), dtype=float)
    query_vector = np.asarray(embedder.encode_query(query), dtype=float)
    doc_norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1.0
    query_norm = float(np.linalg.norm(query_vector)) or 1.0
    similarities = (doc_vectors / doc_norms) @ (query_vector / query_norm)
    return similarities.tolist()


def select_flat_chunks(
    example: Mapping[str, Any],
    *,
    mode: ChunkRetrievalMode,
    top_k: int,
    max_chunk_chars: int,
    embedder: Optional[ChunkEmbeddingProvider] = None,
    rrf_k: int = 60,
) -> ChunkRAGResult:
    from longbench_kg_pipeline import chunk_sentence_records, flatten_context_to_sentence_records

    records = flatten_context_to_sentence_records(dict(example))
    chunks = chunk_sentence_records(records, max_chunk_chars)
    if not chunks:
        return ChunkRAGResult(context="", n_chunks=0, selected_chunk_indices=[])

    texts = [chunk_text(chunk_records) for chunk_records in chunks]
    query = flat_chunk_query(example)
    n = len(texts)

    if mode in ("dense", "hybrid") and embedder is None:
        raise ValueError(f"embedder is required for chunk retrieval mode {mode!r}")

    bm25 = _bm25_scores(query, texts) if mode in ("bm25", "hybrid") else None
    dense = _dense_scores(query, texts, embedder) if mode in ("dense", "hybrid") else None

    if mode == "bm25":
        ordered = sorted(range(n), key=lambda i: (-bm25[i], i))
    elif mode == "dense":
        ordered = sorted(range(n), key=lambda i: (-dense[i], i))
    else:
        bm25_rank = _ranks(bm25)
        dense_rank = _ranks(dense)
        fused = [
            1.0 / (rrf_k + bm25_rank[i]) + 1.0 / (rrf_k + dense_rank[i]) for i in range(n)
        ]
        ordered = sorted(range(n), key=lambda i: (-fused[i], i))

    top_indices = ordered[:top_k]
    context = "\n\n".join(texts[i] for i in top_indices)
    return ChunkRAGResult(
        context=context, n_chunks=len(top_indices), selected_chunk_indices=top_indices
    )
