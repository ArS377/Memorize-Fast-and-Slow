from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from experiments.persona_interference_schedule import (
    ScheduleConfig,
    _load_cli_config,
    build_evaluation_schedule,
    write_evaluation_schedule,
)


class WhitespaceTokenizer:
    """Deterministic tokenizer fixture with explicit provenance."""

    def encode(self, text: str) -> list[str]:
        """Tokenize fixture text on whitespace."""
        return text.split()

    def metadata(self) -> dict[str, str]:
        """Return stable fixture identity."""
        return {"name": "whitespace", "revision": "v1"}


def _dataset() -> Path:
    """Return the committed conflict corpus without relying on the process CWD."""
    return Path(__file__).parents[1] / "results" / "persona_conflict_conversations_v1"


def _config(
    *,
    token_distances: tuple[int, ...] = (100, 500),
    manifest_sha256: str | None = None,
    minimum_stream_tokens: int = 0,
) -> ScheduleConfig:
    """Return a compact production-corpus scheduling fixture."""
    return ScheduleConfig(
        source_split="test",
        source_profile="anti_shortcut_interleaved_v3",
        source_manifest_sha256=manifest_sha256
        or _sha256(_dataset() / "generation_manifest.json"),
        seed=73,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
        query_suffixes=(
            "-preference-change-delayed",
            "-preference-incongruity-delayed",
        ),
        token_distance_thresholds=token_distances,
        minimum_stream_tokens=minimum_stream_tokens,
    )


def _sha256(path: Path) -> str:
    """Return one fixture artifact digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_and_distance_schedule_is_causal_complete_and_immutable(tmp_path: Path) -> None:
    dataset = _dataset()
    source_hashes = {path.name: _sha256(path) for path in dataset.iterdir() if path.is_file()}

    result = write_evaluation_schedule(
        dataset,
        tmp_path / "scheduled",
        WhitespaceTokenizer(),
        _config(),
    )

    assert len(result["inputs"]) == 12 * 2 * 5
    by_query: dict[str, list[dict]] = {}
    for row in result["inputs"]:
        by_query.setdefault(row["query_id"], []).append(row)
        assert row["checkpoint_turn_index"] <= len(result["turns"])
        assert row["context_sha256"]
        assert row["selected_turn_ids"]
        if row["phase"] != "pre_update":
            assert row["checkpoint_turn_index"] > row["relevant_update_turn_index"]
        if row["phase"] == "token_distance":
            assert row["actual_token_distance"] >= row["requested_token_distance"]
            assert row["distractor_turn_count_since_update"] > 0
    assert all(
        {row["phase"] for row in rows}
        == {"pre_update", "post_update", "delayed_probe", "token_distance"}
        for rows in by_query.values()
    )
    assert all(
        next(row for row in rows if row["phase"] == "pre_update")["gold"]
        != next(row for row in rows if row["phase"] == "post_update")["gold"]
        for rows in by_query.values()
    )
    assert source_hashes == {
        path.name: _sha256(path) for path in dataset.iterdir() if path.is_file()
    }
    manifest = json.loads((tmp_path / "scheduled" / "schedule_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["input_count"] == len(result["inputs"])
    assert result["schedule_metrics"]["stream_token_count"] == sum(
        turn["serialized_token_count"] for turn in result["turns"]
    )
    written_turns = [
        json.loads(line)
        for line in (tmp_path / "scheduled" / "scheduled_turns.jsonl").read_text().splitlines()
    ]
    assert written_turns
    assert all("stream_event" not in turn for turn in written_turns)
    assert all("fact" not in turn for turn in written_turns)
    assert all("value-" not in turn["text"] for turn in written_turns)
    source_events_by_history: dict[str, list[dict]] = {}
    for event in (
        json.loads(line) for line in (dataset / "events.jsonl").read_text().splitlines()
    ):
        if (
            event["split"] == "test"
            and event["hardness_profile"] == "anti_shortcut_interleaved_v3"
        ):
            source_events_by_history.setdefault(event["history_id"], []).append(event)
    assert all(
        turn["text"]
        == source_events_by_history[turn["history_id"]][turn["account_event_index"]]["model_text"]
        for turn in written_turns
    )
    model_inputs = [
        json.loads(line)
        for line in (tmp_path / "scheduled" / "model_inputs.jsonl").read_text().splitlines()
    ]
    assert len(model_inputs) == len(result["inputs"])
    assert all(
        set(row) == {"evaluation_input_id", "context", "query_text"}
        for row in model_inputs
    )
    assert all(not re.search(r"\b(?:value|subject|history)-\d+", row["context"]) for row in model_inputs)
    assert manifest["artifact_sha256"]["evaluation_inputs.jsonl"] == _sha256(
        tmp_path / "scheduled" / "evaluation_inputs.jsonl"
    )


def test_schedule_rejects_tampered_authoritative_corpus(tmp_path: Path) -> None:
    dataset = tmp_path / "tampered"
    shutil.copytree(_dataset(), dataset)
    with (dataset / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="artifact hash mismatch for events.jsonl"):
        build_evaluation_schedule(dataset, WhitespaceTokenizer(), _config())


def test_schedule_rejects_resigned_corpus_and_missing_evidence(tmp_path: Path) -> None:
    dataset = tmp_path / "resigned"
    shutil.copytree(_dataset(), dataset)
    queries = [json.loads(line) for line in (dataset / "queries.jsonl").read_text().splitlines()]
    target = next(
        row
        for row in queries
        if row["split"] == "test"
        and row["query_id"].endswith("-preference-change-delayed")
    )
    target["checkpoint_contract"]["required_event_id_groups"][0] = ["missing-event"]
    (dataset / "queries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in queries),
        encoding="utf-8",
    )
    manifest_path = dataset / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"]["queries.jsonl"] = _sha256(dataset / "queries.jsonl")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="generation manifest hash mismatch"):
        build_evaluation_schedule(dataset, WhitespaceTokenizer(), _config())
    with pytest.raises(ValueError, match="evidence group is absent"):
        build_evaluation_schedule(
            dataset,
            WhitespaceTokenizer(),
            _config(manifest_sha256=_sha256(manifest_path)),
        )


def test_cli_config_rejects_string_arrays_and_boolean_strings(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.json"
    config_path.write_text(
        json.dumps(
            {
                "schedule": {
                    "query_suffixes": "-delayed",
                    "token_distance_thresholds": [100],
                    "seed": 1,
                    "concurrent_accounts": 2,
                    "min_segment_events": 1,
                    "max_segment_events": 2,
                },
                "tokenizer": {
                    "name": "fixture",
                    "revision": "v1",
                    "local_files_only": "false",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schedule.query_suffixes must be an array"):
        _load_cli_config(config_path)

    payload = json.loads(config_path.read_text())
    payload["schedule"]["query_suffixes"] = ["-delayed"]
    payload["schedule"]["token_distance_thresholds"] = [1.9]
    payload["tokenizer"]["local_files_only"] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError, match="schedule.token_distance_thresholds elements must be integers"
    ):
        _load_cli_config(config_path)


def test_checked_in_schedule_config_declares_stream_minimum() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "persona_interference_schedule.json"

    schedule, _ = _load_cli_config(config_path)

    assert schedule.minimum_stream_tokens == 0


def test_schedule_rejects_unreachable_token_distance() -> None:
    with pytest.raises(ValueError, match="cannot reach token distance 1000000000"):
        build_evaluation_schedule(
            _dataset(),
            WhitespaceTokenizer(),
            _config(token_distances=(1_000_000_000,)),
        )


def test_schedule_rejects_stream_at_or_below_required_length() -> None:
    with pytest.raises(ValueError, match="must exceed 1000000000 tokens"):
        build_evaluation_schedule(
            _dataset(),
            WhitespaceTokenizer(),
            _config(minimum_stream_tokens=1_000_000_000),
        )


def test_direct_config_rejects_non_integer_token_distances() -> None:
    values = _config().__dict__
    values["token_distance_thresholds"] = (True,)
    with pytest.raises(ValueError, match="token_distance_thresholds"):
        ScheduleConfig(**values)
