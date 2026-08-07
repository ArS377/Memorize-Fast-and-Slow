from __future__ import annotations

from typing import Sequence

import numpy as np

from longbench_kg_pipeline import chunk_sentence_records_semantic


class FakeEmbedder:
    """Returns orthogonal vectors for sentences tagged "A" vs "B", so a
    similarity-based chunker sees zero similarity across the tag boundary
    and near-1.0 similarity within a tag."""

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [[0.0, 1.0] if "TOPIC_B" in text else [1.0, 0.0] for text in texts],
            dtype=np.float32,
        )


def _records(*texts: str):
    return [{"text": text} for text in texts]


def test_cuts_at_topic_shift_once_min_chars_satisfied() -> None:
    records = _records(
        "TOPIC_A sentence one is reasonably long to pass the floor. " * 3,
        "TOPIC_A sentence two also reasonably long. " * 3,
        "TOPIC_B sentence three is a different subject entirely. " * 3,
    )
    chunks = chunk_sentence_records_semantic(
        records,
        FakeEmbedder(),
        max_chars=10_000,
        min_chars=100,
        similarity_threshold=0.5,
    )
    assert len(chunks) == 2
    assert chunks[0] == records[:2]
    assert chunks[1] == records[2:]


def test_does_not_cut_before_min_chars_even_on_topic_shift() -> None:
    records = _records("TOPIC_A hi.", "TOPIC_B bye.")
    chunks = chunk_sentence_records_semantic(
        records,
        FakeEmbedder(),
        max_chars=10_000,
        min_chars=1_000,
        similarity_threshold=0.5,
    )
    assert len(chunks) == 1
    assert chunks[0] == records


def test_max_chars_is_a_hard_cap_regardless_of_similarity() -> None:
    records = _records(
        "TOPIC_A " + "x" * 50,
        "TOPIC_A " + "y" * 50,
        "TOPIC_A " + "z" * 50,
    )
    chunks = chunk_sentence_records_semantic(
        records,
        FakeEmbedder(),
        max_chars=100,
        min_chars=0,
        similarity_threshold=0.0,
    )
    assert len(chunks) == 3
    assert [chunks[i][0] for i in range(3)] == records


def test_empty_records_returns_no_chunks() -> None:
    assert chunk_sentence_records_semantic([], FakeEmbedder(), max_chars=1000) == []


def test_blank_text_records_are_dropped() -> None:
    records = _records("TOPIC_A real sentence.", "   ", "")
    chunks = chunk_sentence_records_semantic(
        records, FakeEmbedder(), max_chars=10_000, min_chars=0
    )
    flattened = [record for chunk in chunks for record in chunk]
    assert flattened == [records[0]]
