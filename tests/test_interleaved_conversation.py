from __future__ import annotations

import json
from pathlib import Path

from experiments.interleaved_conversation import (
    build_interleaved_schedule,
    lifecycle_distribution,
    schedule_metrics,
    suffix_contract_availability,
    suffix_event_ids_by_window,
)
from experiments.synthetic_temporal_preferences import generate_dataset


class WhitespaceTokenizer:
    """Auditable tokenizer fixture."""

    def encode(self, text: str) -> list[str]:
        """Split fixture text on whitespace."""
        return text.split()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_scheduler_is_deterministic_causal_and_genuinely_interleaved(tmp_path: Path) -> None:
    paths = generate_dataset(
        tmp_path / "dataset",
        history_count=24,
        split_counts=(4, 4, 16),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = [event for event in _rows(paths["events"]) if event["split"] == "test"]

    first = build_interleaved_schedule(
        events,
        tokenizer=WhitespaceTokenizer(),
        seed=41,
        concurrent_accounts=6,
        min_segment_events=4,
        max_segment_events=8,
    )
    second = build_interleaved_schedule(
        events,
        tokenizer=WhitespaceTokenizer(),
        seed=41,
        concurrent_accounts=6,
        min_segment_events=4,
        max_segment_events=8,
    )

    assert first == second
    segments = first["segments"]
    assert all(4 <= segment["event_count"] <= 8 for segment in segments)
    assert all(
        left["history_id"] != right["history_id"]
        for left, right in zip(segments, segments[1:])
    )
    by_history: dict[str, list[int]] = {}
    for turn in first["turns"]:
        by_history.setdefault(turn["history_id"], []).append(turn["account_event_index"])
    assert all(indices == list(range(len(indices))) for indices in by_history.values())
    metrics = schedule_metrics(first)
    assert metrics["causal_order_violations"] == 0
    assert metrics["adjacent_same_account_segments"] == 0
    assert metrics["resume_count"] > len(by_history)
    assert metrics["minimum_intervening_accounts_per_resume"] >= 1
    distribution = lifecycle_distribution(first)
    assert distribution["contradiction_opening"]["covers_all_stream_thirds"] is True
    assert distribution["contradiction_rectification"]["covers_all_stream_thirds"] is True


def test_final_retention_sweep_exposes_finite_suffix_loss(tmp_path: Path) -> None:
    paths = generate_dataset(
        tmp_path / "dataset",
        history_count=36,
        split_counts=(4, 4, 28),
        hardness_profile="anti_shortcut_stream_v2",
    )
    events = [event for event in _rows(paths["events"]) if event["split"] == "test"]
    schedule = build_interleaved_schedule(
        events,
        tokenizer=WhitespaceTokenizer(),
        seed=43,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
    )
    first_history = sorted({event["history_id"] for event in events})[0]
    required = [
        [f"{first_history}-add"],
        [f"{first_history}-transition"],
        [f"{first_history}-indirect-source"],
        [f"{first_history}-direct-correction"],
    ]

    availability = suffix_contract_availability(
        schedule["turns"],
        required,
        windows=(64, 256, 1024),
    )

    assert availability == {64: False, 256: False, 1024: False}


def test_suffix_boundary_charges_first_retained_turn_standalone_tokens() -> None:
    turns = [
        {
            "token_count": 2,
            "serialized_token_count": 2,
            "stream_event": {"event_id": "event-a"},
        },
        {
            "token_count": 2,
            "serialized_token_count": 1,
            "stream_event": {"event_id": "event-b"},
        },
    ]

    assert suffix_contract_availability(
        turns, [["event-b"]], windows=(1, 2)
    ) == {1: False, 2: True}
    assert suffix_event_ids_by_window(turns, windows=(1, 2)) == {
        1: set(),
        2: {"event-b"},
    }


def test_larger_horizon_never_reduces_suffix_contract_loss(tmp_path: Path) -> None:
    paths = generate_dataset(
        tmp_path / "dataset",
        history_count=72,
        split_counts=(4, 4, 64),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = [event for event in _rows(paths["events"]) if event["split"] == "test"]
    history_ids = sorted({event["history_id"] for event in events})
    losses = []
    for horizon in (16, 32, 64):
        selected_ids = set(history_ids[:horizon])
        schedule = build_interleaved_schedule(
            [event for event in events if event["history_id"] in selected_ids],
            tokenizer=WhitespaceTokenizer(),
            seed=43,
            concurrent_accounts=8,
            min_segment_events=4,
            max_segment_events=8,
        )
        missing = 0
        for history_id in selected_ids:
            availability = suffix_contract_availability(
                schedule["turns"],
                [[f"{history_id}-add"], [f"{history_id}-transition"]],
                windows=(1024,),
            )
            missing += not availability[1024]
        losses.append(missing / horizon)

    assert losses == sorted(losses)


def test_scheduler_has_variable_lifetimes_and_resume_gaps(tmp_path: Path) -> None:
    paths = generate_dataset(
        tmp_path / "dataset",
        history_count=80,
        split_counts=(4, 4, 72),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = [event for event in _rows(paths["events"]) if event["split"] == "test"]
    schedule = build_interleaved_schedule(
        events,
        tokenizer=WhitespaceTokenizer(),
        seed=53,
        concurrent_accounts=12,
        min_segment_events=4,
        max_segment_events=8,
    )
    segments_by_history: dict[str, list[int]] = {}
    for index, segment in enumerate(schedule["segments"]):
        segments_by_history.setdefault(segment["history_id"], []).append(index)
    segment_counts = {len(indices) for indices in segments_by_history.values()}
    resume_gaps = {
        right - left - 1
        for indices in segments_by_history.values()
        for left, right in zip(indices, indices[1:])
    }

    assert len(segment_counts) >= 2
    assert len(resume_gaps) >= 4
    assert max(resume_gaps) >= 12


def test_contradiction_lifecycle_is_separated_by_resumptions(tmp_path: Path) -> None:
    paths = generate_dataset(
        tmp_path / "dataset",
        history_count=40,
        split_counts=(4, 4, 32),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = [event for event in _rows(paths["events"]) if event["split"] == "test"]
    schedule = build_interleaved_schedule(
        events,
        tokenizer=WhitespaceTokenizer(),
        seed=59,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
    )
    segment_by_event = {
        turn["stream_event"]["event_id"]: turn["segment_id"]
        for turn in schedule["turns"]
    }
    segment_index = {
        segment["segment_id"]: index for index, segment in enumerate(schedule["segments"])
    }
    history_id = sorted({event["history_id"] for event in events})[0]
    left = segment_index[segment_by_event[f"{history_id}-conflict-left"]]
    right = segment_index[segment_by_event[f"{history_id}-conflict-right"]]
    resolution = segment_index[segment_by_event[f"{history_id}-conflict-resolution"]]

    assert left < right < resolution
    assert right - left > 1
    assert resolution - right > 1


def test_dialogue_expansion_is_not_one_repeated_scaffold(tmp_path: Path) -> None:
    paths = generate_dataset(
        tmp_path / "dataset",
        history_count=24,
        split_counts=(4, 4, 16),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = [event for event in _rows(paths["events"]) if event["split"] == "test"]
    schedule = build_interleaved_schedule(
        events,
        tokenizer=WhitespaceTokenizer(),
        seed=61,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
    )
    trailing_sentences = {
        turn["text"].split(". ")[-1] for turn in schedule["turns"]
    }

    assert len(trailing_sentences) >= 12
