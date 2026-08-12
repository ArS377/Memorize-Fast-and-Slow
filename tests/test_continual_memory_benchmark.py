from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import pytest

from experiments import (
    _continual_memory_config,
    _continual_memory_episodes,
    _continual_memory_evaluation,
    continual_memory_benchmark,
)
from experiments.continual_memory_benchmark import (
    _default_dense_embedder,
    _prediction,
    _serialized_model_input,
    _visible_query_text,
    build_episodes,
    evaluate_episodes,
    load_benchmark_config,
    run_benchmark,
)
from experiments.synthetic_temporal_preferences import generate_dataset, resolve_query
from experiments.private_lineage_reasoning import resolve_private_lineage_reference


class WhitespaceTokenizer:
    """Deterministic tokenizer fixture with auditable token counts."""

    def encode(self, text: str) -> list[str]:
        """Return non-whitespace spans as fixture tokens."""
        return re.findall(r"\n|\S+", text)

    def metadata(self) -> dict[str, str]:
        """Return a stable test-only tokenizer identity."""
        return {"name": "whitespace-test", "revision": "fixture-v1"}


class FakeDenseEmbedder:
    """Small batched embedder that routes matching subject identifiers together."""

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        match = re.search(r"subject-(\d+)", text)
        subject = float(int(match.group(1))) if match else 0.0
        return [1.0, subject, subject * subject]

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode every turn in one observed batch."""
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode every checkpoint query in one observed batch."""
        self.query_calls += 1
        return [self._vector(text) for text in texts]

    def encode_query(self, text: str) -> list[float]:
        """Support the repository's single-query protocol as a fallback."""
        self.query_calls += 1
        return self._vector(text)

    def metadata(self) -> dict[str, Any]:
        """Return stable fake dense provenance."""
        return {"implementation": "fake-dense", "vector_dimension": 3}


class FakeLineageClient:
    """Deterministic service fixture preserving the recursive feature interaction."""

    def resolve(
        self,
        events: Sequence[dict[str, Any]],
        query: dict[str, Any],
        *,
        mode: str,
    ) -> dict[str, Any]:
        reference = resolve_private_lineage_reference(events, query, strict=True)
        answer = reference["answer"]
        if mode == "one_hop" and query["query_id"].endswith("-private-lineage"):
            root = next(
                event["fact"]
                for event in events
                if event["fact"]["fact_id"] == query["fact_id"]
            )
            answer = root["object"]
        return {
            **reference,
            "answer": answer,
            "engine": "scallopy",
            "mode": mode,
            "scallopy_version": "0.2.4",
        }


class FakePreferenceStreamClient:
    """Query-blind fixture that returns source pairs from the causal event prefix."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def derive(self, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Return only source IDs for the known transition and authority conflict."""
        event_ids = [str(event["event_id"]) for event in events]
        self.calls.append(event_ids)
        event_by_fact_id = {
            str(event["fact"]["fact_id"]): event
            for event in events
            if isinstance(event.get("fact"), dict)
        }
        injections = []
        for event in events:
            supersedes = event.get("supersedes")
            prior = event_by_fact_id.get(str(supersedes))
            if prior is None:
                continue
            prior_fact = prior["fact"]
            fact = event["fact"]
            if prior_fact["object"] == fact["object"]:
                continue
            if prior_fact["temporal"]["valid_from"] != fact["temporal"]["valid_from"]:
                kind = "preference_change"
            elif (
                prior_fact["qualifiers"].get("source_authority", "inferred")
                != fact["qualifiers"].get("source_authority", "inferred")
            ):
                kind = "preference_incongruity"
            else:
                continue
            injections.append(
                {
                    "kind": kind,
                    "source_event_ids": [
                        prior["event_id"],
                        event["event_id"],
                        next(
                            candidate["event_id"]
                            for candidate in events
                            if candidate["history_id"] == event["history_id"]
                            and candidate["event_id"].endswith("-lineage-alias")
                        ),
                    ],
                }
            )
        return {
            "engine": "scallopy",
            "scallopy_version": "0.2.4",
            "rule_version": "preference_stream.v1",
            "injections": injections,
        }


def _write_config(
    path: Path,
    *,
    dense_enabled: bool = False,
    source_profile: str = "base",
    scallop_enabled: bool = False,
    interference_tiers: list[dict[str, Any]] | None = None,
    sliding_token_windows: list[dict[str, Any]] | None = None,
    history_limit: int = 1,
) -> Path:
    payload = {
        "benchmark_version": "test.continual-memory.v1",
        "source_split": "test",
        "source_profile": source_profile,
        "history_limit": history_limit,
        "retrieval_k": 5,
        "seed": 17,
        "tokenizer": {
            "name": "fixture/tokenizer",
            "revision": "fixture-tokenizer-v1",
            "local_files_only": True,
        },
        "interference_tiers": interference_tiers or [
            {
                "name": "none",
                "distractor_turns_per_target_event": 0,
                "distractor_turn_percentage": 0.0,
            },
            {
                "name": "two",
                "distractor_turns_per_target_event": 2,
                "distractor_turn_percentage": 66.66666666666667,
            },
        ],
        "sliding_token_windows": sliding_token_windows or [
            {"name": "tiny", "max_tokens": 32},
            {"name": "full", "max_tokens": 10000},
        ],
        "model_parameter_tiers": [
            {"name": "small", "min_billions": 1.0, "max_billions": 4.0},
            {"name": "large", "min_billions": 4.0, "max_billions": None},
        ],
        "metric_bins": {
            "turn_age_upper_bounds": [2, 8],
            "token_age_upper_bounds": [32, 128],
            "stream_position_upper_bounds_percent": [25.0, 75.0],
        },
        "dense_embedding": {
            "enabled": dense_enabled,
            "model": "fixture/dense",
            "revision": "fixture-dense-v1",
            "device": "cpu",
            "batch_size": 16,
            "local_files_only": True,
        },
        "scallop_reasoner": {
            "enabled": scallop_enabled,
            "endpoint_env": "TEST_SCALLOP_URL",
            "timeout_seconds": 1.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dataset(path: Path) -> Path:
    generate_dataset(path, history_count=4, split_counts=(1, 1, 2))
    return path


def _hard_dataset(path: Path) -> Path:
    generate_dataset(
        path,
        history_count=4,
        split_counts=(1, 1, 2),
        hardness_profile="anti_shortcut_stream_v2",
    )
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load mutable JSONL fixture rows."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Rewrite JSONL fixture rows in supplied order."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_delayed_preference_probes_force_context_rot_without_breaking_memory(
    tmp_path: Path,
) -> None:
    dataset_dir = _hard_dataset(tmp_path / "dataset")
    config = load_benchmark_config(
        _write_config(
            tmp_path / "config.json",
            source_profile="anti_shortcut_stream_v2",
            interference_tiers=[
                {
                    "name": "none",
                    "distractor_turns_per_target_event": 0,
                    "distractor_turn_percentage": 0.0,
                },
                {
                    "name": "long_context_rot",
                    "distractor_turns_per_target_event": 128,
                    "distractor_turn_percentage": 99.2248062015504,
                },
            ],
            sliding_token_windows=[
                {"name": "window_4k", "max_tokens": 4096},
                {"name": "window_16k", "max_tokens": 16384},
                {"name": "window_64k", "max_tokens": 65536},
                {"name": "window_128k", "max_tokens": 131072},
            ],
        )
    )
    episodes = build_episodes(dataset_dir, config, WhitespaceTokenizer())
    long_episode = next(
        episode
        for episode in episodes
        if episode["interference_tier"] == "long_context_rot"
    )
    delayed = {
        checkpoint["checkpoint_kind"]: checkpoint
        for checkpoint in long_episode["checkpoints"]
        if checkpoint["checkpoint_kind"]
        in {"preference-change-delayed", "preference-incongruity-delayed"}
    }
    assert set(delayed) == {
        "preference-change-delayed",
        "preference-incongruity-delayed",
    }
    assert all(checkpoint["token_age"] > 131072 for checkpoint in delayed.values())

    metrics, predictions = evaluate_episodes(episodes, config)
    for checkpoint_kind in delayed:
        rows = {
            row["method"]: row
            for row in predictions
            if row["interference_tier"] == "long_context_rot"
            and row["checkpoint_kind"] == checkpoint_kind
            and row["evaluation_family"] == "retrieval"
        }
        assert rows["full_structured_memory"]["grounded_answer_correct"] is True
        assert rows["sliding_context:window_4k"]["grounded_answer_correct"] is False
        assert rows["sliding_context:window_16k"]["grounded_answer_correct"] is False
        assert rows["sliding_context:window_64k"]["grounded_answer_correct"] is False
        assert rows["sliding_context:window_128k"]["grounded_answer_correct"] is False
    assert metrics["methods"]["full_structured_memory"]["grounded_answer_accuracy"] == 1.0


def test_sliding_context_does_not_discard_distractor_events(tmp_path: Path) -> None:
    dataset_dir = _hard_dataset(tmp_path / "dataset")
    config = load_benchmark_config(
        _write_config(tmp_path / "config.json", source_profile="anti_shortcut_stream_v2")
    )
    episodes = build_episodes(dataset_dir, config, WhitespaceTokenizer())
    episode = next(item for item in episodes if item["interference_tier"] == "two")
    metrics, predictions = evaluate_episodes(episodes, config)
    del metrics
    row = next(
        prediction
        for prediction in predictions
        if prediction["episode_id"] == episode["episode_id"]
        and prediction["method"] == "sliding_context:full"
        and prediction["checkpoint_kind"] == "current"
    )
    distractor_history_ids = {
        turn["stream_event"]["history_id"]
        for turn in episode["turns"][: row["stored_turn_count"]]
        if turn["role"] == "distractor"
    }
    assert distractor_history_ids
    assert any(
        event_id.startswith(tuple(f"{history_id}-" for history_id in distractor_history_ids))
        for event_id in row["selected_event_ids"]
    )


def test_scallop_source_injection_recovers_rotted_preferences_under_same_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = _hard_dataset(tmp_path / "dataset")
    monkeypatch.setenv("TEST_SCALLOP_URL", "http://fixture.invalid")
    config = load_benchmark_config(
        _write_config(
            tmp_path / "config.json",
            source_profile="anti_shortcut_stream_v2",
            scallop_enabled=True,
            history_limit=2,
            interference_tiers=[
                {
                    "name": "none",
                    "distractor_turns_per_target_event": 0,
                    "distractor_turn_percentage": 0.0,
                },
                {
                    "name": "long_context_rot",
                    "distractor_turns_per_target_event": 128,
                    "distractor_turn_percentage": 99.2248062015504,
                },
            ],
            sliding_token_windows=[
                {"name": "window_4k", "max_tokens": 4096},
                {"name": "window_16k", "max_tokens": 16384},
                {"name": "window_64k", "max_tokens": 65536},
                {"name": "window_128k", "max_tokens": 131072},
            ],
        )
    )
    episodes = build_episodes(dataset_dir, config, WhitespaceTokenizer())
    injection_client = FakePreferenceStreamClient()

    metrics, predictions = evaluate_episodes(
        episodes,
        config,
        lineage_client=FakeLineageClient(),
        stream_injection_client=injection_client,
        tokenizer=WhitespaceTokenizer(),
    )

    assert injection_client.calls
    for window_name in ("window_4k", "window_16k", "window_64k", "window_128k"):
        for checkpoint_kind in (
            "preference-change-delayed",
            "preference-incongruity-delayed",
        ):
            rows = {
                row["method"]: row
                for row in predictions
                if row["history_id"] == "history-003"
                and row["interference_tier"] == "long_context_rot"
                and row["checkpoint_kind"] == checkpoint_kind
            }
            raw_method = f"sliding_context:{window_name}"
            injected_method = f"{raw_method}+scallop_injection"
            reverse_method = f"{raw_method}+reverse_ranked_injection"
            change_only_method = f"{raw_method}+scallop_change_only"
            incongruity_only_method = f"{raw_method}+scallop_incongruity_only"
            assert rows[raw_method]["grounded_answer_correct"] is False
            assert rows[injected_method]["grounded_answer_correct"] is True
            assert rows[reverse_method]["max_model_input_tokens"] == rows[injected_method][
                "max_model_input_tokens"
            ]
            if checkpoint_kind == "preference-change-delayed":
                assert rows[change_only_method]["grounded_answer_correct"] is True
                assert rows[incongruity_only_method]["grounded_answer_correct"] is False
            else:
                assert rows[change_only_method]["grounded_answer_correct"] is False
                assert rows[incongruity_only_method]["grounded_answer_correct"] is True
            assert rows[injected_method]["model_input_token_count"] <= rows[injected_method][
                "max_model_input_tokens"
            ]
            assert rows[injected_method]["max_model_input_tokens"] in {
                4096,
                16384,
                65536,
                131072,
            }
    ablation = metrics["scallop_stream_injection_ablation"]
    assert ablation["scallop_query_or_gold_used"] is False
    treatment = ablation["by_window"]["window_16k"]["long_context_rot"]
    assert treatment["preference-change-delayed"]["paired_delta"] > 0
    assert treatment["preference-incongruity-delayed"]["paired_delta"] > 0


def test_facade_exports_owner_module_objects_by_identity() -> None:
    assert (
        continual_memory_benchmark.BenchmarkTokenizer
        is _continual_memory_config.BenchmarkTokenizer
    )
    assert continual_memory_benchmark.TokenizerConfig is _continual_memory_config.TokenizerConfig
    assert (
        continual_memory_benchmark.InterferenceTier
        is _continual_memory_config.InterferenceTier
    )
    assert (
        continual_memory_benchmark.SlidingTokenWindow
        is _continual_memory_config.SlidingTokenWindow
    )
    assert (
        continual_memory_benchmark.ModelParameterTier
        is _continual_memory_config.ModelParameterTier
    )
    assert continual_memory_benchmark.MetricBins is _continual_memory_config.MetricBins
    assert (
        continual_memory_benchmark.DenseEmbeddingConfig
        is _continual_memory_config.DenseEmbeddingConfig
    )
    assert (
        continual_memory_benchmark.ScallopReasonerConfig
        is _continual_memory_config.ScallopReasonerConfig
    )
    assert (
        continual_memory_benchmark.ContinualMemoryConfig
        is _continual_memory_config.ContinualMemoryConfig
    )
    assert (
        continual_memory_benchmark.HuggingFaceTokenizer
        is _continual_memory_config.HuggingFaceTokenizer
    )
    assert (
        continual_memory_benchmark.load_benchmark_config
        is _continual_memory_config.load_benchmark_config
    )
    assert continual_memory_benchmark.build_episodes is _continual_memory_episodes.build_episodes
    assert (
        continual_memory_benchmark._serialized_model_input
        is _continual_memory_episodes._serialized_model_input
    )
    assert (
        continual_memory_benchmark._visible_query_text
        is _continual_memory_episodes._visible_query_text
    )
    assert (
        continual_memory_benchmark.evaluate_episodes
        is _continual_memory_evaluation.evaluate_episodes
    )
    assert continual_memory_benchmark.DenseEmbedder is _continual_memory_evaluation.DenseEmbedder
    assert (
        continual_memory_benchmark._default_dense_embedder
        is _continual_memory_evaluation._default_dense_embedder
    )
    assert continual_memory_benchmark._prediction is _continual_memory_evaluation._prediction


def test_facade_main_runs_end_to_end_cli_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config_path = _write_config(tmp_path / "config.json")
    output_dir = tmp_path / "cli-run"
    monkeypatch.setattr(
        continual_memory_benchmark,
        "HuggingFaceTokenizer",
        lambda _config: WhitespaceTokenizer(),
    )

    exit_code = continual_memory_benchmark.main(
        [
            "--dataset",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    assert "full_structured_memory: grounded_answer_accuracy=1.0000" in capsys.readouterr().out
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))[
        "status"
    ] == "completed"


def test_online_interleaving_checkpoint_mapping_and_gold_are_exact(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config = load_benchmark_config(_write_config(tmp_path / "config.json"))
    tokenizer = WhitespaceTokenizer()

    first = build_episodes(dataset_dir, config, tokenizer)
    second = build_episodes(dataset_dir, config, tokenizer)

    assert first == second
    assert len(first) == 2
    none_episode = next(episode for episode in first if episode["interference_tier"] == "none")
    two_episode = next(episode for episode in first if episode["interference_tier"] == "two")
    assert [turn["role"] for turn in none_episode["turns"]] == ["target"] * 13
    assert [turn["role"] for turn in two_episode["turns"][:6]] == [
        "target",
        "distractor",
        "distractor",
        "target",
        "distractor",
        "distractor",
    ]
    assert all("target_event" in turn for turn in two_episode["turns"] if turn["role"] == "target")
    assert all("target_event" not in turn for turn in two_episode["turns"] if turn["role"] == "distractor")
    assert all(turn["token_count"] == len(tokenizer.encode(turn["text"])) for turn in two_episode["turns"])

    expected_mapping = {
        "current": "transition",
        "scope": "scope",
        "constraint": "constraint",
        "private": "retract",
        "ambiguity": "ambiguity",
        "backdated": "backdated",
        "duplicate": "duplicate",
        "authority": "direct-correction",
        "scope-leakage": "scope-leakage",
    }
    checkpoints = {checkpoint["checkpoint_kind"]: checkpoint for checkpoint in two_episode["checkpoints"]}
    assert set(checkpoints) == set(expected_mapping)
    for kind, event_suffix in expected_mapping.items():
        checkpoint = checkpoints[kind]
        assert checkpoint["after_event_id"].endswith(f"-{event_suffix}")
        assert checkpoint["query"]["query_id"].endswith(f"-{kind}")
        assert "gold" not in checkpoint["query"]
        assert checkpoint["gold"] is not None
        assert checkpoint["turn_age"] >= 2
        assert checkpoint["token_age"] >= 0
        assert 0.0 < checkpoint["stream_position_percentage"] <= 100.0
        visible_turns = two_episode["turns"][: checkpoint["stream_turn_index"]]
        serialized_stream = "\n".join(turn["text"] for turn in visible_turns)
        assert checkpoint["stream_token_count"] == len(tokenizer.encode(serialized_stream))
        for window in config.sliding_token_windows:
            window_metadata = checkpoint["sliding_context"][window.name]
            start = window_metadata["start_turn_index"]
            model_input = _serialized_model_input(
                visible_turns[start:], _visible_query_text(checkpoint["query"])
            )
            assert len(tokenizer.encode(model_input)) == window_metadata["model_input_token_count"]
            assert window_metadata["model_input_token_count"] <= window.max_tokens
            if start > 0:
                expanded = _serialized_model_input(
                    visible_turns[start - 1 :], _visible_query_text(checkpoint["query"])
                )
                assert len(tokenizer.encode(expanded)) > window.max_tokens
    assert checkpoints["private"]["gold"] == "UNKNOWN"


def test_hard_profile_hides_oracle_labels_and_adds_lineage_checkpoints(
    tmp_path: Path,
) -> None:
    dataset_dir = _hard_dataset(tmp_path / "dataset")
    config = load_benchmark_config(
        _write_config(tmp_path / "config.json", source_profile="anti_shortcut_stream_v2")
    )
    episode = next(
        item
        for item in build_episodes(dataset_dir, config, WhitespaceTokenizer())
        if item["interference_tier"] == "none"
    )
    forbidden = (
        "role=",
        "task_id=",
        "thread_id=",
        "event_id=",
        "operation=",
        "supersedes=",
        "retracts=",
        "distractor_type=",
    )
    assert all(not any(token in turn["text"] for token in forbidden) for turn in episode["turns"])
    assert {checkpoint["checkpoint_kind"] for checkpoint in episode["checkpoints"]} >= {
        "private-lineage-positive",
        "private-lineage",
    }


def test_grounded_credit_requires_complete_non_unknown_evidence(tmp_path: Path) -> None:
    dataset_dir = _hard_dataset(tmp_path / "dataset")
    config = load_benchmark_config(
        _write_config(tmp_path / "config.json", source_profile="anti_shortcut_stream_v2")
    )
    episode = next(
        item
        for item in build_episodes(dataset_dir, config, WhitespaceTokenizer())
        if item["interference_tier"] == "none"
    )
    checkpoint = next(
        item
        for item in episode["checkpoints"]
        if item["checkpoint_kind"] == "private-lineage-positive"
    )
    seen = episode["turns"][: checkpoint["stream_turn_index"]]
    complete = _prediction(episode, checkpoint, "complete", seen)
    missing_bridge = [
        turn
        for turn in seen
        if not turn.get("target_event", {}).get("event_id", "").endswith("-lineage-alias")
    ]
    incomplete = _prediction(episode, checkpoint, "missing", missing_bridge)

    assert complete["answer_correct"] is True
    assert complete["grounded_answer_correct"] is True
    assert incomplete["answer_correct"] is True
    assert incomplete["exact_evidence_hit"] is False
    assert incomplete["grounded_answer_correct"] is False


def test_recursive_scallop_ablation_isolated_in_benchmark_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = _hard_dataset(tmp_path / "dataset")
    monkeypatch.setenv("TEST_SCALLOP_URL", "http://fixture.invalid")
    config = load_benchmark_config(
        _write_config(
            tmp_path / "config.json",
            source_profile="anti_shortcut_stream_v2",
            scallop_enabled=True,
        )
    )
    episodes = build_episodes(dataset_dir, config, WhitespaceTokenizer())

    metrics, predictions = evaluate_episodes(
        episodes,
        config,
        lineage_client=FakeLineageClient(),
    )

    ablation = metrics["scallop_reasoning_ablation"]
    assert ablation["recursive_accuracy"] == 1.0
    assert ablation["one_hop_accuracy"] == 0.0
    assert ablation["methods"]["scallop_one_hop"]["by_checkpoint_kind"] == {
        "private": {"count": 1, "accuracy": 1.0},
        "private-lineage-positive": {"count": 1, "accuracy": 1.0},
        "private-lineage": {"count": 1, "accuracy": 0.0},
    }
    assert any(
        row.get("evaluation_family") == "scallop_reasoning_ablation"
        for row in predictions
    )


def test_checkpoint_gold_uses_causal_prefix_while_source_gold_uses_full_history(
    tmp_path: Path,
) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config = load_benchmark_config(_write_config(tmp_path / "config.json"))
    events = _read_jsonl(dataset_dir / "events.jsonl")
    queries = _read_jsonl(dataset_dir / "queries.jsonl")
    history_id = "history-003"
    history_events = [event for event in events if event["history_id"] == history_id]
    direct = next(
        event for event in history_events if event["event_id"].endswith("-direct-correction")
    )
    direct["fact"]["temporal"]["valid_from"] = "2025-08-01"
    for query in queries:
        if query["history_id"] == history_id:
            query["gold"] = resolve_query(history_events, query)
    _write_jsonl(dataset_dir / "events.jsonl", events)
    _write_jsonl(dataset_dir / "queries.jsonl", queries)

    episodes = build_episodes(dataset_dir, config, WhitespaceTokenizer())

    episode = next(item for item in episodes if item["interference_tier"] == "none")
    current = next(
        checkpoint for checkpoint in episode["checkpoints"] if checkpoint["checkpoint_kind"] == "current"
    )
    source_current = next(
        query
        for query in queries
        if query["history_id"] == history_id and query["query_id"].endswith("-current")
    )
    assert current["gold"] == "value-003-b"
    assert source_current["gold"] == "value-003-k"


def test_build_rejects_source_gold_that_disagrees_with_causal_prefix(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config = load_benchmark_config(_write_config(tmp_path / "config.json"))
    queries = _read_jsonl(dataset_dir / "queries.jsonl")
    current = next(
        query
        for query in queries
        if query["history_id"] == "history-003" and query["query_id"].endswith("-current")
    )
    current["gold"] = "incorrect-source-gold"
    _write_jsonl(dataset_dir / "queries.jsonl", queries)

    with pytest.raises(ValueError, match="causal-prefix gold mismatch"):
        build_episodes(dataset_dir, config, WhitespaceTokenizer())


def test_build_rejects_forward_causal_reference(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config = load_benchmark_config(_write_config(tmp_path / "config.json"))
    events = _read_jsonl(dataset_dir / "events.jsonl")
    private_add_index = next(
        index
        for index, event in enumerate(events)
        if event["event_id"] == "history-003-private-add"
    )
    retract_index = next(
        index for index, event in enumerate(events) if event["event_id"] == "history-003-retract"
    )
    events[private_add_index], events[retract_index] = events[retract_index], events[private_add_index]
    _write_jsonl(dataset_dir / "events.jsonl", events)

    with pytest.raises(ValueError, match="forward causal reference.*history-003-private"):
        build_episodes(dataset_dir, config, WhitespaceTokenizer())


def test_config_validates_derived_tier_percentages(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "config.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["interference_tiers"][1]["distractor_turn_percentage"] = 60.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="derived distractor-turn percentage"):
        load_benchmark_config(path)


def test_config_requires_exactly_one_no_interference_tier(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "config.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["interference_tiers"][1]["distractor_turns_per_target_event"] = 0
    payload["interference_tiers"][1]["distractor_turn_percentage"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one zero-distractor"):
        load_benchmark_config(path)


def test_pinned_production_axes_match_source_tokenizer_and_required_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    continual_path = repository_root / "configs" / "continual_memory_benchmark.json"
    multitask_path = repository_root / "configs" / "multitask_context_benchmark.json"
    continual_raw = json.loads(continual_path.read_text(encoding="utf-8"))
    multitask_raw = json.loads(multitask_path.read_text(encoding="utf-8"))

    monkeypatch.setenv("SCALLOP_VALIDATOR_URL", "http://127.0.0.1:8765")
    config = load_benchmark_config(continual_path)

    assert continual_raw["tokenizer"] == multitask_raw["tokenizer"]
    assert [tier.distractor_turns_per_target_event for tier in config.interference_tiers] == [
        0,
        2,
        8,
        32,
        128,
    ]
    assert [window.max_tokens for window in config.sliding_token_windows] == [
        4096,
        16384,
        65536,
        131072,
    ]
    assert all(
        tier.distractor_turn_percentage
        == pytest.approx(
            100.0
            * tier.distractor_turns_per_target_event
            / (tier.distractor_turns_per_target_event + 1)
        )
        for tier in config.interference_tiers
    )


def test_sliding_window_degrades_and_bm25_dense_route_seen_turns(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config = load_benchmark_config(
        _write_config(tmp_path / "config.json", dense_enabled=True)
    )
    episodes = build_episodes(dataset_dir, config, WhitespaceTokenizer())
    embedder = FakeDenseEmbedder()

    metrics, predictions = evaluate_episodes(episodes, config, embedder=embedder)

    assert metrics["methods"]["full_structured_memory"]["answer_accuracy"] == 1.0
    assert (
        metrics["methods"]["sliding_context:tiny"]["answer_accuracy"]
        < metrics["methods"]["sliding_context:full"]["answer_accuracy"]
    )
    assert set(metrics["methods"]) >= {
        "full_structured_memory",
        "recency",
        "bm25",
        "dense",
        "sliding_context:tiny",
        "sliding_context:full",
    }
    assert embedder.document_calls == 1
    assert embedder.query_calls == 1
    assert any(prediction["method"] == "bm25" for prediction in predictions)
    assert any(prediction["method"] == "dense" for prediction in predictions)
    assert all(
        set(prediction["retrieval_query"]) == {"query_text"}
        for prediction in predictions
    )


@pytest.mark.parametrize("checkpoint_kind", ["private", "ambiguity"])
def test_unknown_requires_exact_evidence_for_grounded_credit(
    tmp_path: Path, checkpoint_kind: str
) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config = load_benchmark_config(_write_config(tmp_path / "config.json"))
    episode = next(
        item
        for item in build_episodes(dataset_dir, config, WhitespaceTokenizer())
        if item["interference_tier"] == "none"
    )
    checkpoint = next(
        item for item in episode["checkpoints"] if item["checkpoint_kind"] == checkpoint_kind
    )

    unsupported = _prediction(episode, checkpoint, "empty", [])
    seen_turns = episode["turns"][: checkpoint["stream_turn_index"]]
    supported = _prediction(episode, checkpoint, "full", seen_turns)

    assert unsupported["prediction"] == "UNKNOWN"
    assert unsupported["exact_evidence_hit"] is False
    assert unsupported["answer_correct"] is True
    assert unsupported["grounded_answer_correct"] is False
    assert supported["answer_correct"] is True
    assert supported["exact_evidence_hit"] is True
    if checkpoint_kind == "private":
        assert unsupported["retraction_compliant"] is False
        assert supported["retraction_compliant"] is True


def test_default_dense_embedder_passes_local_files_only_to_sentence_transformer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers_utils = pytest.importorskip("transformers.utils")
    config = load_benchmark_config(
        _write_config(tmp_path / "config.json", dense_enabled=True)
    )
    captured: dict[str, Any] = {}

    class CapturingSentenceTransformer:
        """Capture constructor arguments without loading model files."""

        def __init__(self, model: str, **kwargs: Any) -> None:
            captured["model"] = model
            captured.update(kwargs)

    monkeypatch.setattr(transformers_utils, "cached_file", lambda *_args, **_kwargs: __file__)
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = CapturingSentenceTransformer
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", sentence_transformers
    )

    embedder = _default_dense_embedder(config.dense_embedding)
    embedder._load()

    assert captured == {
        "model": "fixture/dense",
        "revision": "fixture-dense-v1",
        "device": "cpu",
        "local_files_only": True,
    }


def test_full_memory_rejects_unreachable_evidence_contract(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config = load_benchmark_config(_write_config(tmp_path / "config.json"))
    episode = deepcopy(
        next(
            item
            for item in build_episodes(dataset_dir, config, WhitespaceTokenizer())
            if item["interference_tier"] == "none"
        )
    )
    transition = next(
        turn
        for turn in episode["turns"]
        if turn.get("target_event", {}).get("event_id", "").endswith("-transition")
    )
    transition["target_event"]["fact"]["fact_id"] = "wrong-fact-id"

    with pytest.raises(RuntimeError, match="full structured memory failed evidence contract"):
        evaluate_episodes([episode], config)


def test_metrics_group_retention_interference_age_and_growth(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config = load_benchmark_config(_write_config(tmp_path / "config.json"))
    episodes = build_episodes(dataset_dir, config, WhitespaceTokenizer())

    metrics, predictions = evaluate_episodes(episodes, config)

    assert set(metrics["retention_by_interference_tier"]) == {"none", "two"}
    assert set(metrics["retention_by_checkpoint_kind"]) == {
        "current",
        "scope",
        "constraint",
        "private",
        "ambiguity",
        "backdated",
        "duplicate",
        "authority",
        "scope-leakage",
    }
    assert metrics["retention_by_turn_age_bin"]
    assert metrics["retention_by_token_age_bin"]
    assert metrics["retention_by_stream_position_bin"]
    assert metrics["memory_growth"]["checkpoint_count"] == 18
    assert metrics["methods"]["recency"]["cross_task_interference_rate"] >= 0.0
    assert "retraction_compliance" in metrics["methods"]["full_structured_memory"]
    assert metrics["interpretation"]["model_parameter_axis_evaluated"] is False
    assert metrics["interpretation"]["online_parameter_updates"] is False
    no_interference = {
        (row["history_id"], row["checkpoint_kind"], row["method"]): row
        for row in predictions
        if row["distractor_turns_per_target_event"] == 0
    }
    assert all(
        not row["cross_task_interference"]
        if row["distractor_turns_per_target_event"] == 0
        else row["cross_task_interference"]
        == (
            no_interference[(row["history_id"], row["checkpoint_kind"], row["method"])][
                "grounded_answer_correct"
            ]
            and not row["grounded_answer_correct"]
        )
        for row in predictions
    )
    for method, values in metrics["methods"].items():
        method_rows = [row for row in predictions if row["method"] == method]
        eligible = [row for row in method_rows if row["cross_task_interference_eligible"]]
        expected_rate = (
            sum(row["cross_task_interference"] for row in eligible) / len(eligible)
            if eligible
            else 0.0
        )
        assert values["cross_task_interference_rate"] == pytest.approx(expected_rate)


def test_failed_manifest_records_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config_path = _write_config(tmp_path / "config.json")
    config = load_benchmark_config(config_path)
    output_dir = tmp_path / "failed-run"

    def fail_build(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise ValueError("unanswerable checkpoint fixture")

    monkeypatch.setattr("experiments.continual_memory_benchmark.build_episodes", fail_build)
    with pytest.raises(ValueError, match="unanswerable checkpoint fixture"):
        run_benchmark(
            dataset_dir,
            output_dir,
            config_path,
            config,
            WhitespaceTokenizer(),
        )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "ValueError"
    assert "unanswerable checkpoint fixture" in manifest["error"]["message"]


def test_preflight_failure_records_failed_manifest(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.json")
    config = load_benchmark_config(config_path)
    output_dir = tmp_path / "preflight-failure"

    with pytest.raises(FileNotFoundError):
        run_benchmark(
            tmp_path / "missing-dataset",
            output_dir,
            config_path,
            config,
            WhitespaceTokenizer(),
        )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "FileNotFoundError"
    assert manifest["artifacts_present_at_failure"] == []


def test_completed_manifest_hashes_required_artifacts(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    config_path = _write_config(tmp_path / "config.json")
    config = load_benchmark_config(config_path)
    output_dir = tmp_path / "complete-run"

    metrics = run_benchmark(
        dataset_dir,
        output_dir,
        config_path,
        config,
        WhitespaceTokenizer(),
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    required = {"episodes.jsonl", "predictions.jsonl", "metrics.json", "report.md"}
    assert manifest["status"] == "completed"
    assert set(manifest["artifacts"]) == required
    assert set(manifest["artifact_sha256"]) == required
    assert all(len(digest) == 64 for digest in manifest["artifact_sha256"].values())
    expected_sources = {
        "experiments/_continual_memory_config.py",
        "experiments/_continual_memory_episodes.py",
        "experiments/_continual_memory_evaluation.py",
        "experiments/continual_memory_benchmark.py",
            "experiments/private_lineage_reasoning.py",
            "experiments/preference_stream_injection.py",
        "experiments/synthetic_temporal_baselines.py",
        "experiments/synthetic_temporal_preferences.py",
        "neurosym/adapters/dense_index.py",
        "neurosym/domain/retrieval_config.py",
        "services/scallop_validator_service.py",
    }
    repository_root = Path(__file__).resolve().parents[1]
    assert set(manifest["source_sha256"]) == expected_sources
    assert all(
        manifest["source_sha256"][source]
        == hashlib.sha256((repository_root / source).read_bytes()).hexdigest()
        for source in expected_sources
    )
    assert manifest["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert manifest["dataset"]["split"] == "test"
    assert set(manifest["dataset"]["sha256"]) == {
        "candidate_updates.jsonl",
        "events.jsonl",
        "examples.jsonl",
        "facts.jsonl",
        "queries.jsonl",
    }
    assert manifest["tokenizer"]["name"] == "whitespace-test"
    assert manifest["dense"]["configuration"]["enabled"] is False
    assert metrics["methods"]["full_structured_memory"]["answer_accuracy"] == 1.0
