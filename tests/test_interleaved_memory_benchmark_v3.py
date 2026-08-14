from __future__ import annotations

import json
from pathlib import Path

from experiments.interleaved_conversation import build_interleaved_schedule
from experiments.interleaved_memory_benchmark_v3 import (
    _paired_grounded_comparison,
    _target_history_turns,
    build_preference_capsules,
    compact_valid_capsules_to_capacity,
    delayed_checkpoint_specs,
    evaluate_delayed_preference_checkpoints,
    rank_global_capsule,
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


def _fixture(tmp_path: Path) -> tuple[dict, list[dict], list[dict], WhitespaceTokenizer]:
    paths = generate_dataset(
        tmp_path / "dataset",
        history_count=20,
        split_counts=(2, 2, 16),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = [event for event in _rows(paths["events"]) if event["split"] == "test"]
    queries = [query for query in _rows(paths["queries"]) if query["split"] == "test"]
    tokenizer = WhitespaceTokenizer()
    schedule = build_interleaved_schedule(
        events,
        tokenizer=tokenizer,
        seed=73,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
    )
    injections = {}
    for history_id in sorted({event["history_id"] for event in events}):
        injections[history_id] = {
            "engine": "scallopy",
            "scallopy_version": "0.2.4",
            "rule_version": "preference_stream.v1",
            "injections": [
                {
                    "kind": "preference_change",
                    "source_event_ids": [
                        f"{history_id}-add",
                        f"{history_id}-transition",
                        f"{history_id}-lineage-alias",
                    ],
                },
                {
                    "kind": "preference_incongruity",
                    "source_event_ids": [
                        f"{history_id}-indirect-source",
                        f"{history_id}-direct-correction",
                        f"{history_id}-lineage-alias",
                    ],
                },
            ],
        }
    capsules = build_preference_capsules(schedule, injections)
    return schedule, queries, capsules, tokenizer


def test_delayed_sweeps_age_queries_after_eligibility(tmp_path: Path) -> None:
    schedule, queries, _, _ = _fixture(tmp_path)

    specs = delayed_checkpoint_specs(schedule, queries, sweep_interval_tokens=1024)

    assert 16 * 2 - 2 <= len(specs) < 16 * 2
    assert all(spec["checkpoint_turn_index"] > spec["trigger_turn_index"] for spec in specs)
    assert all(
        spec["sweep_kind"] == "terminal"
        or spec["token_age"] >= 0
        for spec in specs
    )
    assert all(spec["token_age"] > 0 for spec in specs)
    assert {spec["checkpoint_kind"] for spec in specs} == {
        "preference-change-delayed",
        "preference-incongruity-delayed",
    }
    assert max(spec["token_age"] for spec in specs) > 1024


def test_global_capsule_ranking_uses_only_visible_query_text(tmp_path: Path) -> None:
    _, queries, capsules, _ = _fixture(tmp_path)
    query = next(query for query in queries if query["query_id"].endswith("-change-delayed"))

    selected = rank_global_capsule(capsules, query["query_text"])
    corrupted = dict(query)
    corrupted["gold"] = "wrong"
    corrupted["history_id"] = "wrong-history"
    corrupted["checkpoint_contract"] = {"required_event_id_groups": [["wrong-event"]]}

    assert len(capsules) == 16 * 5
    assert selected == rank_global_capsule(capsules, corrupted["query_text"])
    assert selected == rank_global_capsule(list(reversed(capsules)), query["query_text"])
    assert set(selected) == {
        "history_id",
        "kind",
        "release_turn_index",
        "source_event_ids",
        "text",
    }


def test_delayed_methods_share_caps_and_do_not_force_perfection(tmp_path: Path) -> None:
    schedule, queries, capsules, tokenizer = _fixture(tmp_path)
    capsules = [dict(capsule) for capsule in capsules]
    capsules[0]["text"] = "deliberately misleading lexical match"
    capsules = compact_valid_capsules_to_capacity(capsules, 8)

    rows = evaluate_delayed_preference_checkpoints(
        schedule,
        queries,
        capsules,
        tokenizer=tokenizer,
        windows=(256, 512),
        sweep_interval_tokens=1024,
        capsule_capacity=8,
    )

    checkpoint_count = len(
        delayed_checkpoint_specs(schedule, queries, sweep_interval_tokens=1024)
    )
    assert len(rows) == checkpoint_count * 2 * 2
    assert {row["method"] for row in rows} == {
        "sliding_context:256",
        "sliding_context:512",
        "structured_capacity_top1:256",
        "structured_capacity_top1:512",
    }
    assert all(row["model_input_token_count"] <= row["max_tokens"] for row in rows)
    assert all(row["checkpoint_turn_index"] > row["trigger_turn_index"] for row in rows)
    structured = [row for row in rows if row["method"].startswith("structured_")]
    assert all(row["selected_capsule_count"] == 1 for row in structured)
    assert all(
        row["selected_capsule_release_turn_index"] <= row["checkpoint_turn_index"]
        for row in structured
    )
    assert any(not row["grounded_answer_correct"] for row in structured)


def test_compaction_is_query_blind_and_capacity_bounded() -> None:
    capsules = [
        {
            "kind": "preference_change",
            "history_id": "history-001",
            "release_turn_index": 3,
            "source_event_ids": ["a", "b", "alias"],
            "text": "old valid capsule",
        },
        {
            "kind": "hard_negative",
            "history_id": "history-001",
            "release_turn_index": 10,
            "source_event_ids": ["negative"],
            "text": "newer but invalid note",
        },
        {
            "kind": "preference_incongruity",
            "history_id": "history-001",
            "release_turn_index": 8,
            "source_event_ids": ["c", "d", "alias"],
            "text": "new valid capsule",
        },
    ]

    retained = compact_valid_capsules_to_capacity(capsules, 1)

    assert len(retained) == 1
    assert retained[0]["kind"] in {"preference_change", "preference_incongruity"}
    assert retained[0]["kind"] != "hard_negative"


def test_selected_target_events_preserve_global_causal_order() -> None:
    selected = [
        {
            "history_id": "history-001",
            "stream_event": {"event_id": "event-retract", "turn_index": 1},
        },
        {
            "history_id": "history-001",
            "stream_event": {"event_id": "event-add", "turn_index": 2},
        },
    ]

    ordered = _target_history_turns(
        selected,
        "history-001",
        {"event-add": 0, "event-retract": 1},
    )

    assert [turn["stream_event"]["event_id"] for turn in ordered] == [
        "event-add",
        "event-retract",
    ]


def test_paired_comparison_is_deterministic_and_history_clustered() -> None:
    rows = []
    outcomes = {
        "history-001": (False, True),
        "history-002": (False, True),
        "history-003": (True, True),
        "history-004": (True, False),
    }
    for history_id, (sliding, structured) in outcomes.items():
        for method, correct in (
            ("sliding_context:64", sliding),
            ("structured_capacity_top1:64", structured),
        ):
            rows.append(
                {
                    "history_id": history_id,
                    "query_id": f"{history_id}-query",
                    "method": method,
                    "max_tokens": 64,
                    "grounded_answer_correct": correct,
                }
            )

    first = _paired_grounded_comparison(rows, bootstrap_samples=200, seed=17)
    second = _paired_grounded_comparison(rows, bootstrap_samples=200, seed=17)

    assert first == second
    assert first["64"]["matched_checkpoint_count"] == 4
    assert first["64"]["history_cluster_count"] == 4
    assert first["64"]["paired_delta"] == 0.25
    assert first["64"]["confidence_interval_95"][0] <= 0.25
    assert first["64"]["confidence_interval_95"][1] >= 0.25
