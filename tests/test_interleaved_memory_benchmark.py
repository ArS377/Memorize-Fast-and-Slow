from __future__ import annotations

import json
from pathlib import Path

from experiments.interleaved_conversation import build_interleaved_schedule
from experiments.interleaved_memory_benchmark import (
    _historical_values,
    evaluate_online_checkpoints,
    validate_context_horizon,
)
from experiments.synthetic_temporal_preferences import generate_dataset


class WhitespaceTokenizer:
    """Auditable tokenizer fixture."""

    def encode(self, text: str) -> list[str]:
        """Split fixture text on whitespace."""
        return text.split()


def _rows(path: Path) -> list[dict]:
    """Read fixture JSONL rows."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_online_checkpoints_compare_matched_causal_memory_views(tmp_path: Path) -> None:
    paths = generate_dataset(
        tmp_path / "dataset",
        history_count=24,
        split_counts=(4, 4, 16),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = [event for event in _rows(paths["events"]) if event["split"] == "test"]
    queries = [query for query in _rows(paths["queries"]) if query["split"] == "test"]
    tokenizer = WhitespaceTokenizer()
    schedule = build_interleaved_schedule(
        events,
        tokenizer=tokenizer,
        seed=67,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
    )

    rows = evaluate_online_checkpoints(
        schedule,
        queries,
        tokenizer=tokenizer,
        windows=(256, 1024),
    )

    assert len(rows) == 16 * 4 * 3
    full_memory = [row for row in rows if row["method"] == "full_structured_memory"]
    assert len(full_memory) == 16 * 4
    assert all(row["answer_correct"] for row in full_memory)
    assert all(row["complete_provenance"] for row in full_memory)
    assert all(row["grounded_answer_correct"] for row in full_memory)
    assert all(
        row["checkpoint_turn_index"] == row["selected_through_turn_index"]
        for row in rows
    )
    assert all(
        row["trigger_event_id"] in row["selected_event_ids"]
        for row in full_memory
    )
    assert all(0 < row["stream_position_percentage"] <= 100 for row in rows)
    assert all(row["turn_age"] >= 0 and row["token_age"] >= 0 for row in rows)


def test_online_metrics_separate_answer_sufficiency_from_complete_provenance(
    tmp_path: Path,
) -> None:
    paths = generate_dataset(
        tmp_path / "dataset",
        history_count=12,
        split_counts=(2, 2, 8),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = [event for event in _rows(paths["events"]) if event["split"] == "test"]
    queries = [query for query in _rows(paths["queries"]) if query["split"] == "test"]
    tokenizer = WhitespaceTokenizer()
    schedule = build_interleaved_schedule(
        events,
        tokenizer=tokenizer,
        seed=71,
        concurrent_accounts=4,
        min_segment_events=4,
        max_segment_events=8,
    )

    rows = evaluate_online_checkpoints(
        schedule,
        queries,
        tokenizer=tokenizer,
        windows=(128,),
    )
    sliding = [row for row in rows if row["method"] == "sliding_context:128"]

    assert any(row["answer_correct"] and not row["complete_provenance"] for row in sliding)
    assert all(
        row["grounded_answer_correct"]
        == (row["answer_correct"] and row["complete_provenance"])
        for row in rows
    )


def test_private_values_are_eligible_stale_intrusions() -> None:
    turns = [
        {
            "stream_event": {
                "fact": {
                    "subject": "subject-001",
                    "object": "private value",
                    "qualifiers": {"scope": "private"},
                }
            }
        }
    ]

    assert _historical_values(
        turns, {"kind": "private_lineage", "subject": "subject-001"}
    ) == {"private value"}


def test_declared_context_horizon_fails_loud_when_stream_is_too_short() -> None:
    assert validate_context_horizon(4096, declared_context_limit=1024, multiplier=4.0) == 4.0

    try:
        validate_context_horizon(4095, declared_context_limit=1024, multiplier=4.0)
    except ValueError as error:
        assert "4095" in str(error)
        assert "4096" in str(error)
    else:
        raise AssertionError("short context horizon should fail")

    try:
        validate_context_horizon(3, declared_context_limit=3, multiplier=1.0001)
    except ValueError as error:
        assert "4" in str(error)
    else:
        raise AssertionError("fractional context requirement should round up")
