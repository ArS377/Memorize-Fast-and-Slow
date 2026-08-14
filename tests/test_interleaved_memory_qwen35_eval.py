from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.interleaved_memory_qwen35_eval import (
    _aggregate_generation_rows,
    _extract_synthetic_answer,
    _fit_chat_sliding,
    _model_context_limit,
    _model_file_hashes,
    _select_history_ids,
    _score_generation_rows,
    _validate_dataset_hashes,
    _validate_resume_manifest,
    _validate_resumed_row,
)


class WordChatTokenizer:
    """Small tokenizer with explicit chat-template overhead."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        """Tokenize fixture text on whitespace."""
        assert not add_special_tokens
        return text.split()

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        """Wrap one user message with three fixed model-control tokens."""
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return f"USER {messages[0]['content']} ASSISTANT READY"


class LengthGuardTokenizer(WordChatTokenizer):
    """Tokenizer that rejects probes beyond its declared model limit."""

    model_max_length = 30

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        """Reject any candidate longer than the model could accept."""
        tokens = super().encode(text, add_special_tokens=add_special_tokens)
        if len(tokens) > self.model_max_length:
            raise ValueError("overlong fitting probe")
        return tokens


def _sha256(path: Path) -> str:
    """Return the digest for one fixture file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_chat_sliding_fit_is_exact_and_maximal() -> None:
    tokenizer = WordChatTokenizer()
    turns = [
        {"turn_id": "turn-1", "text": "one two three"},
        {"turn_id": "turn-2", "text": "four five six"},
        {"turn_id": "turn-3", "text": "seven eight nine"},
    ]

    selected, prompt, token_count = _fit_chat_sliding(
        turns,
        "which value",
        13,
        tokenizer,
    )

    assert [turn["turn_id"] for turn in selected] == ["turn-2", "turn-3"]
    assert token_count == len(tokenizer.encode(prompt, add_special_tokens=False)) == 13
    longer_prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "one two three\nfour five six\nseven eight nine\nQuestion: which value\nAnswer:",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert len(tokenizer.encode(longer_prompt, add_special_tokens=False)) > 13


def test_chat_sliding_never_probes_the_full_overlong_prefix() -> None:
    tokenizer = LengthGuardTokenizer()
    turns = [
        {"turn_id": f"turn-{index}", "text": f"token-{index}"}
        for index in range(100)
    ]

    selected, _, token_count = _fit_chat_sliding(turns, "which value", 20, tokenizer)

    assert 0 < len(selected) < len(turns)
    assert token_count <= 20


def test_dataset_hash_validation_rejects_drift(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (dataset / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    declared = {
        "events.jsonl": _sha256(dataset / "events.jsonl"),
        "queries.jsonl": _sha256(dataset / "queries.jsonl"),
    }

    _validate_dataset_hashes(dataset, declared)
    (dataset / "queries.jsonl").write_text(json.dumps({"changed": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="queries.jsonl hash mismatch"):
        _validate_dataset_hashes(dataset, declared)


def test_history_sampling_is_stable_and_keeps_complete_histories() -> None:
    rows = [
        {"history_id": history_id, "window": window, "query_id": f"{history_id}-q"}
        for history_id in ("history-1", "history-2", "history-3", "history-4")
        for window in (65536, 131072)
    ]

    selected_ids = _select_history_ids(rows, history_limit=2, seed=47)
    repeated_ids = _select_history_ids(list(reversed(rows)), history_limit=2, seed=47)

    assert selected_ids == repeated_ids
    assert len(selected_ids) == 2
    assert sum(row["history_id"] in selected_ids for row in rows) == 4


def test_resume_rejects_different_model_provenance(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    existing = {
        "evaluator_version": "v1",
        "source_benchmark_version": "source-v1",
        "dataset_sha256": {"events.jsonl": "a"},
        "model": {"resolved_revision": "revision-a"},
    }
    manifest_path.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(ValueError, match="model differs"):
        _validate_resume_manifest(
            manifest_path,
            {**existing, "model": {"resolved_revision": "revision-b"}},
        )


def test_model_context_limit_accepts_text_only_and_nested_configs() -> None:
    class Config:
        max_position_embeddings = 262144

    class NestedConfig:
        text_config = Config()

    assert _model_context_limit(Config()) == 262144
    assert _model_context_limit(NestedConfig()) == 262144

    with pytest.raises(ValueError, match="max_position_embeddings"):
        _model_context_limit(object())


def test_model_file_hashes_cover_slow_tokenizer_inputs(tmp_path: Path) -> None:
    required = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model.safetensors-00001-of-00001.safetensors",
    )
    for name in required:
        (tmp_path / name).write_text(name, encoding="utf-8")

    hashes = _model_file_hashes(tmp_path)

    assert set(hashes) == set(required)


def test_generation_metrics_use_history_clustered_bootstrap() -> None:
    rows = [
        {
            "history_id": history_id,
            "window": window,
            "exact_match": exact,
            "f1": exact,
            "generation_seconds": 1.0,
            "oracle_context_answer_available": bool(exact),
        }
        for history_id, values in {
            "history-1": (1.0, 0.0),
            "history-2": (1.0, 1.0),
            "history-3": (0.0, 1.0),
        }.items()
        for window, exact in zip((65536, 131072), values)
    ]

    first = _aggregate_generation_rows(rows, bootstrap_samples=200, seed=47)
    second = _aggregate_generation_rows(rows, bootstrap_samples=200, seed=47)

    assert first == second
    assert first["65536"]["checkpoint_count"] == 3
    assert first["65536"]["history_cluster_count"] == 3
    assert first["65536"]["exact_match"] == pytest.approx(2 / 3)
    assert first["65536"]["exact_match_ci_95"][0] <= 2 / 3
    assert first["65536"]["exact_match_ci_95"][1] >= 2 / 3


def test_synthetic_answer_extraction_is_unique_and_format_bounded() -> None:
    assert _extract_synthetic_answer("The answer is **value-371-b**.") == "value-371-b"
    assert _extract_synthetic_answer("value-371-b then value-371-b") == "value-371-b"
    assert _extract_synthetic_answer("No synthetic label here") is None
    assert _extract_synthetic_answer("value-371-b or value-371-c") is None
    assert _extract_synthetic_answer("prefixvalue-371-bsuffix") is None


def test_resumed_row_must_match_rebuilt_spec() -> None:
    spec = {
        "history_id": "history-1",
        "query_id": "history-1-query",
        "window": 65536,
        "gold": "value-1-a",
        "input_token_count": 65500,
        "selected_turn_count": 10,
        "selected_start_turn_id": "turn-2",
        "selected_end_turn_id": "turn-11",
        "checkpoint_turn_index": 11,
    }
    resumed = {**spec, "answer": "value-1-a", "exact_match": 1.0}

    _validate_resumed_row(resumed, spec)
    with pytest.raises(ValueError, match="selected_start_turn_id"):
        _validate_resumed_row({**resumed, "selected_start_turn_id": "turn-3"}, spec)


def test_generation_scoring_recomputes_all_answer_fields() -> None:
    row = {
        "answer": "The answer is value-1-a.",
        "gold": "value-1-a",
        "extracted_answer": "stale",
        "exact_match": 0.0,
        "f1": 0.0,
    }

    scored = _score_generation_rows([row])

    assert scored[0]["extracted_answer"] == "value-1-a"
    assert scored[0]["exact_match"] == 1.0
    assert scored[0]["f1"] == 1.0
