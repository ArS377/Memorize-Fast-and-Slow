from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.continual_memory_benchmark import build_episodes, load_benchmark_config
from experiments.persona_conversation_generator import (
    GenerationConfig,
    LLMResponse,
    default_personas,
    generate_persona_conversations,
)
from experiments.persona_recall_benchmark import (
    ContextTier,
    RecallConfig,
    _answer_state,
    evaluate_persona_recall,
)


class WhitespaceTokenizer:
    """Deterministic complete-turn token accounting fixture."""

    def encode(self, text: str) -> list[str]:
        return re.findall(r"\n|\S+", text)

    def metadata(self) -> dict[str, str]:
        return {"name": "fixture-tokenizer", "revision": "v1"}


class SurfaceFakeClient:
    """Generate deterministic dialogue retaining each supplied surface fact."""

    def complete(self, **request: Any) -> LLMResponse:
        payload = json.loads(request["messages"][-1]["content"])
        return LLMResponse(
            content=json.dumps(
                {
                    "events": [
                        {
                            "event_id": event["event_id"],
                            "turns": [
                                {
                                    "role": "user",
                                    "content": " ".join(
                                        [
                                            event["surface_text"],
                                            *event["required_surface_values"],
                                        ]
                                    ),
                                },
                                {
                                    "role": "assistant",
                                    "content": "Understood. "
                                    + (
                                        event["semantic_markers"][0]
                                        if event["semantic_markers"]
                                        else ""
                                    )
                                    + " "
                                    + (
                                        event["authority_markers"][0]
                                        if event["authority_markers"]
                                        else ""
                                    ),
                                },
                            ],
                        }
                        for event in payload["events"]
                    ]
                },
                sort_keys=True,
            ),
            finish_reason="stop",
            model="fake-kimi",
            usage={},
        )


class ContextRecallClient:
    """Answer only when the expected natural answer remains in the raw context."""

    def __init__(self, answer_by_query: Mapping[str, str]) -> None:
        self.answer_by_query = answer_by_query
        self.requests: list[Sequence[Mapping[str, str]]] = []

    def complete(self, **request: Any) -> LLMResponse:
        messages = request["messages"]
        self.requests.append(messages)
        user = messages[-1]["content"]
        query = user.split("Question:\n", 1)[1].split("\n\nAnswer", 1)[0]
        answer = self.answer_by_query[query]
        context = user.split("Conversation:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        content = answer if answer.casefold() in context.casefold() else "I don't know"
        return LLMResponse(
            content=content,
            finish_reason="stop",
            model=request["model"],
            usage={"prompt_tokens": 8, "completion_tokens": 2},
        )


def test_answer_state_separates_wrong_answers_unknown_gold_and_refusals() -> None:
    assert _answer_state("Quiet workspace.", "shared workspace") == (
        "Quiet workspace.", False, "answered"
    )
    assert _answer_state("UNKNOWN", "UNKNOWN") == ("UNKNOWN", True, "answered")
    assert _answer_state("I don't know", "mint tea") == (None, False, "refusal")


def _generation_config() -> GenerationConfig:
    return GenerationConfig(
        endpoint="https://fixture.invalid/v1",
        api_key="fixture-secret",
        model="fixture-kimi",
        timeout_seconds=2.0,
        seed=7,
        max_tokens=4096,
        history_count=4,
        condition="lexical",
        hardness_profile="base",
        split_counts=(1, 1, 2),
        prompt_schema_version="persona-conversation.v1",
        turn_pairs_per_event=1,
        minimum_words_per_turn=1,
        events_per_request=100,
        enable_thinking=False,
        max_validation_attempts=2,
        resume_existing=True,
    )


def _benchmark_config(path: Path) -> Path:
    payload = {
        "benchmark_version": "persona-fixture.v1",
        "source_split": "test",
        "source_profile": "base",
        "history_limit": 1,
        "retrieval_k": 3,
        "seed": 11,
        "tokenizer": {"name": "fixture", "revision": "v1", "local_files_only": True},
        "interference_tiers": [
            {"name": "none", "distractor_turns_per_target_event": 0, "distractor_turn_percentage": 0.0},
            {"name": "distractors", "distractor_turns_per_target_event": 2, "distractor_turn_percentage": 66.66666666666667},
        ],
        "sliding_token_windows": [
            {"name": "tiny", "max_tokens": 32},
            {"name": "full", "max_tokens": 10000},
        ],
        "model_parameter_tiers": [{"name": "small", "min_billions": 1.0, "max_billions": 5.0}],
        "metric_bins": {
            "turn_age_upper_bounds": [2, 8],
            "token_age_upper_bounds": [32, 128],
            "stream_position_upper_bounds_percent": [25.0, 75.0],
        },
        "dense_embedding": {
            "enabled": False,
            "model": "fixture",
            "revision": "v1",
            "device": "cpu",
            "batch_size": 1,
            "local_files_only": True,
        },
        "scallop_reasoner": {
            "enabled": False,
            "endpoint_env": "UNUSED_SCALLOP_URL",
            "timeout_seconds": 1.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_matched_persona_qwen_recall_and_controls_are_isolated(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generate_persona_conversations(dataset, _generation_config(), SurfaceFakeClient())
    benchmark_config = load_benchmark_config(_benchmark_config(tmp_path / "benchmark.json"))
    queries = [json.loads(line) for line in (dataset / "queries.jsonl").read_text().splitlines()]
    answer_by_query = {row["surface_query_text"]: row["surface_gold"] for row in queries}
    client = ContextRecallClient(answer_by_query)
    recall_config = RecallConfig(
        endpoint="https://fixture.invalid/v1",
        api_key="fixture-qwen-secret",
        model="Qwen-fixture",
        timeout_seconds=2.0,
        seed=13,
        max_tokens=16,
        history_limit=1,
        max_concurrent_requests=2,
        interference_tiers=("none", "distractors"),
        query_suffixes=("-current", "-authority"),
        context_tiers=(ContextTier("tiny", 100), ContextTier("full", 10000)),
        prompt_schema_version="persona-recall.v1",
    )

    report = evaluate_persona_recall(
        dataset,
        tmp_path / "results",
        benchmark_config,
        WhitespaceTokenizer(),
        recall_config,
        client,
        personas=default_personas(),
    )

    rows = [json.loads(line) for line in (tmp_path / "results" / "qwen_predictions.jsonl").read_text().splitlines()]
    matched = {}
    for row in rows:
        key = (row["episode_id"], row["context_tier"], row["query_id"])
        matched.setdefault(key, []).append(row)
    for group in matched.values():
        assert len({row["conversation_sha256"] for row in group}) == 1
        assert len({row["query_text"] for row in group}) == 1
        assert len({row["context_start_turn"] for row in group}) == 1
        assert all(row["model_input_token_count"] <= row["context_max_tokens"] for row in group)
        assert len({row["model_input_token_count"] for row in group}) > 1
    assert any(row["persona_axis"] == "ethnicity" for row in rows)
    assert any(row["persona_axis"] == "education" for row in rows)
    assert report["controls"]["full_structured_memory"]["label"] == "oracle control"
    assert report["controls"]["sliding_context:tiny"]
    assert report["paired_persona_deltas"]
    assert report["subgroup_summaries"]
    assert report["intersection_summaries"]

    delayed = [row for row in rows if row["query_id"].endswith("-current")]
    assert any(row["context_tier"] == "tiny" and not row["correct"] for row in delayed)
    oracle_episodes = build_episodes(dataset, benchmark_config, WhitespaceTokenizer())
    assert oracle_episodes
    assert report["controls"]["full_structured_memory"]["answer_accuracy"] == 1.0


def test_protected_attribute_changes_never_change_shared_task_inputs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generate_persona_conversations(dataset, _generation_config(), SurfaceFakeClient())
    benchmark_config = load_benchmark_config(_benchmark_config(tmp_path / "benchmark.json"))
    queries = [json.loads(line) for line in (dataset / "queries.jsonl").read_text().splitlines()]
    client = ContextRecallClient(
        {row["surface_query_text"]: row["surface_gold"] for row in queries}
    )
    config = RecallConfig(
        endpoint="https://fixture.invalid/v1",
        api_key="secret",
        model="Qwen-fixture",
        timeout_seconds=2.0,
        seed=5,
        max_tokens=16,
        history_limit=1,
        max_concurrent_requests=2,
        interference_tiers=("none", "distractors"),
        query_suffixes=("-current", "-authority"),
        context_tiers=(ContextTier("full", 10000),),
        prompt_schema_version="persona-recall.v1",
    )
    evaluate_persona_recall(
        dataset,
        tmp_path / "results",
        benchmark_config,
        WhitespaceTokenizer(),
        config,
        client,
    )
    rows = [json.loads(line) for line in (tmp_path / "results" / "qwen_predictions.jsonl").read_text().splitlines()]
    ethnicity_rows = [row for row in rows if row["persona_axis"] == "ethnicity"]
    by_query: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ethnicity_rows:
        by_query.setdefault((row["episode_id"], row["query_id"]), []).append(row)
    assert by_query
    for pair in by_query.values():
        assert len(pair) == 2
        assert pair[0]["gold"] == pair[1]["gold"]
        assert pair[0]["conversation_sha256"] == pair[1]["conversation_sha256"]
        assert pair[0]["query_text"] == pair[1]["query_text"]
        assert pair[0]["persona_bio"] != pair[1]["persona_bio"]
