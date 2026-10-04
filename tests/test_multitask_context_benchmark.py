from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.multitask_context_benchmark import (
    build_context_cases,
    evaluate_context_cases,
    load_context_config,
)
from experiments.synthetic_temporal_preferences import generate_dataset


class WhitespaceTokenizer:
    """Small deterministic tokenizer fixture for context accounting tests."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def encode_with_offsets(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        matches = list(re.finditer(r"\S+", text))
        return [match.group(0) for match in matches], [match.span() for match in matches]

    def metadata(self) -> dict[str, str]:
        return {"name": "whitespace-test", "revision": "fixture-v1"}


def _config(path: Path) -> Path:
    payload = {
        "benchmark_version": "test.multitask.v1",
        "context_tiers": [
            {"name": "compact", "max_tokens": 2048, "target_thread_percent": 20.0, "hard_lexical_distractors": 1},
            {"name": "sparse", "max_tokens": 8192, "target_thread_percent": 5.0, "hard_lexical_distractors": 2},
        ],
        "context_window_tiers": [
            {"name": "window_small", "max_tokens": 256},
            {"name": "window_full", "max_tokens": 8192},
        ],
        "distractor_block_tokens": 32,
        "distractor_mix_percent_by_block": {
            "lexical_overlap": 25.0,
            "obsolete_task": 25.0,
            "same_user_other_task": 25.0,
            "unrelated_task": 25.0,
        },
        "evidence_positions_percent": [10.0, 90.0],
        "history_limit": 1,
        "model_parameter_tiers": [
            {"name": "small", "min_billions": 1.0, "max_billions": 4.0},
            {"name": "large", "min_billions": 4.0, "max_billions": None},
        ],
        "retrieval_k": 3,
        "seed": 5,
        "source_split": "test",
        "tokenizer": {
            "local_files_only": True,
            "name": "fixture/tokenizer",
            "revision": "fixture-v1",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cases_distinguish_gold_target_thread_and_distractor_blocks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=3, split_counts=(1, 1, 1))
    config = load_context_config(_config(tmp_path / "config.json"))

    cases = build_context_cases(dataset_dir, config, WhitespaceTokenizer())

    assert len(cases) == 2
    assert {case["context_metadata"]["context_token_tier"] for case in cases} == {
        "compact",
        "sparse",
    }
    for case in cases:
        roles = {item["role"] for item in case["context"]}
        assert roles == {"gold_evidence", "target_thread_context", "distractor"}
        assert sum(item["role"] == "gold_evidence" for item in case["context"]) == 2
        assert all(item["thread_id"] == case["target_history_id"] for item in case["context"] if item["role"] != "distractor")
        assert all(item["distractor_type"] for item in case["context"] if item["role"] == "distractor")
        assert all(
            item["distractor_difficulty"] in {"standard", "hard_lexical_near_miss"}
            for item in case["context"]
            if item["role"] == "distractor"
        )
        metadata = case["context_metadata"]
        assert metadata["model_input_token_count"] == len(
            WhitespaceTokenizer().encode(case["model_input_text"])
        )
        assert metadata["context_token_count"] < metadata["model_input_token_count"]
        assert (
            metadata["query_prompt_token_count"]
            == metadata["model_input_token_count"] - metadata["context_token_count"]
        )
        assert metadata["target_thread_percentage"] == 100.0 * sum(
            item["token_count"] for item in case["context"] if item["role"] != "distractor"
        ) / metadata["model_input_token_count"]
        assert metadata["gold_evidence_token_count"] > 0
        assert 0.0 <= metadata["gold_evidence_position_percent"] <= 100.0
        assert metadata["target_thread_percentage"] > 0.0
        tier = next(
            tier for tier in config.context_tiers if tier.name == metadata["context_token_tier"]
        )
        assert metadata["hard_lexical_distractor_count"] == tier.hard_lexical_distractors
        hard_lexical_texts = [
            item["text"]
            for item in case["context"]
            if item.get("distractor_difficulty") == "hard_lexical_near_miss"
        ]
        assert len(hard_lexical_texts) == tier.hard_lexical_distractors
        forbidden_hints = (
            "does not state",
            "near-miss",
            "not a personal preference",
            "should not control",
            "separate task",
        )
        assert all(
            not any(hint in text.lower() for hint in forbidden_hints)
            for text in hard_lexical_texts
        )
        total_distractors = sum(metadata["distractor_counts"].values())
        for name, requested in config.distractor_mix_percent_by_block.items():
            realized = 100.0 * metadata["distractor_counts"][name] / total_distractors
            assert abs(realized - requested) <= 10.0


def test_context_generation_is_deterministic_and_model_tiers_are_metadata_only(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=3, split_counts=(1, 1, 1))
    config = load_context_config(_config(tmp_path / "config.json"))
    tokenizer = WhitespaceTokenizer()

    first = build_context_cases(dataset_dir, config, tokenizer)
    second = build_context_cases(dataset_dir, config, tokenizer)

    assert first == second
    assert config.model_parameter_tiers[0].name == "small"
    assert all("model_parameter_tier" not in case["context_metadata"] for case in first)


def test_retrieval_baselines_report_tier_and_position_metrics(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=3, split_counts=(1, 1, 1))
    config = load_context_config(_config(tmp_path / "config.json"))
    cases = build_context_cases(dataset_dir, config, WhitespaceTokenizer())

    metrics, predictions = evaluate_context_cases(cases, config)

    assert set(metrics["methods"]) >= {
        "bm25",
        "random",
        "recency",
        "oracle_thread_filter",
        "full_structured_history",
    }
    assert metrics["methods"]["oracle_thread_filter"]["answer_accuracy"] == 1.0
    assert metrics["methods"]["full_structured_history"]["answer_accuracy"] == 1.0
    assert metrics["methods"]["oracle_thread_filter"]["target_thread_precision"] == 1.0
    assert metrics["methods"]["full_structured_history"]["target_thread_precision"] < 1.0
    assert set(metrics["window_truncation"]) == {"window_small", "window_full"}
    assert set(metrics["window_truncation_by_context_tier"]) == {"compact", "sparse"}
    assert set(metrics["window_truncation_by_evidence_position"]) == {"10", "90"}
    assert set(metrics["by_context_tier"]) == {"compact", "sparse"}
    assert set(metrics["by_evidence_position"]) == {"10", "90"}
    assert len(predictions) == len(cases)
