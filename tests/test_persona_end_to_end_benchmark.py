from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.persona_end_to_end_benchmark import (
    _aggregate_rows,
    _assert_full_context_fits,
    _build_arm_specs,
    _derive_condition_sources,
    _fit_sliding_prompt,
    _fit_structured_prompt,
    _fit_hybrid_kg_prompt,
    _git_provenance,
    _load_resumable_jsonl,
    _load_source_and_rebuild,
    _relation_coverage_matrix,
    _resume_manifest_fields,
    _run_scallop_canary,
    _score_short_answer,
    _validate_resume_manifest,
    load_benchmark_config,
)


class WordChatTokenizer:
    """Small tokenizer with visible chat-template overhead."""

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
        """Render one deterministic non-thinking prompt."""
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return f"USER {messages[0]['content']} ASSISTANT READY"


class CanaryClient:
    """Return a valid semantic result only for the fixed canary fixture."""

    def derive(self, events: list[dict]) -> dict:
        """Derive both expected canary relations from source IDs."""
        assert not any("query" in event for event in events)
        return {
            "engine": "scallopy",
            "scallopy_version": "fixture",
            "rule_version": "preference_stream.v1",
            "injections": [
                {
                    "kind": "preference_change",
                    "source_event_ids": ["canary-old-change", "canary-new-change", "canary-alias"],
                },
                {
                    "kind": "preference_incongruity",
                    "source_event_ids": ["canary-old-conflict", "canary-new-conflict", "canary-alias"],
                },
            ],
        }


class EmptyInjectionClient:
    """Return no source relations while recording causal inputs."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def derive(self, events: list[dict]) -> dict:
        """Return an empty valid relation set."""
        self.calls.append(events)
        return {
            "engine": "scallopy",
            "scallopy_version": "fixture",
            "rule_version": "preference_stream.v1",
            "injections": [],
        }


def _turn(index: int, history_id: str = "account-a") -> dict:
    """Build one natural source turn fixture."""
    return {
        "turn_id": f"turn-{index}",
        "history_id": history_id,
        "text": f"natural conversation words {index}",
        "stream_event": {"event_id": f"event-{index}", "history_id": history_id},
    }


def test_config_declares_exact_five_qwen_arms_and_env_paths() -> None:
    root = Path(__file__).parents[1]
    config = load_benchmark_config(
        root / "configs" / "persona_end_to_end_benchmark.json",
        environ={
            "PERSONA_DATASET_DIR": "/dataset",
            "PERSONA_BENCHMARK_OUTPUT_DIR": "/output",
            "PERSONA_QWEN_MODEL_PATH": "/model",
            "PERSONA_QWEN_MODEL_ID": "Qwen/fixture",
            "PERSONA_QWEN_DEVICE": "cuda",
            "PERSONA_SCALLOP_ENDPOINT": "http://scallop.invalid",
        },
    )

    assert [(arm.name, arm.kind, arm.prompt_token_cap) for arm in config.arms] == [
        ("sliding_context_4096", "sliding_context", 4096),
        ("sliding_context_16384", "sliding_context", 16384),
        ("structured_memory_4096", "structured_memory", 4096),
        ("structured_memory_16384", "structured_memory", 16384),
        ("full_qwen_context", "full_qwen_context", None),
    ]
    assert config.dataset_dir == Path("/dataset")
    assert config.output_dir == Path("/output")
    assert config.model_path == Path("/model")
    assert config.hybrid_memory is None


def test_neurosym_config_adds_pinned_hybrid_kg_arms() -> None:
    root = Path(__file__).parents[1]
    config = load_benchmark_config(
        root / "configs" / "persona_end_to_end_neurosym.json",
        environ={
            "PERSONA_DATASET_DIR": "/dataset",
            "PERSONA_BENCHMARK_OUTPUT_DIR": "/output",
            "PERSONA_QWEN_MODEL_PATH": "/model",
            "PERSONA_QWEN_MODEL_ID": "Qwen/fixture",
            "PERSONA_QWEN_DEVICE": "cuda",
            "PERSONA_SCALLOP_ENDPOINT": "http://scallop.invalid",
            "PERSONA_RETRIEVAL_INDEX_ROOT": "/indexes",
            "PERSONA_EMBEDDING_MODEL_ID": "BAAI/fixture",
            "PERSONA_EMBEDDING_MODEL_PATH": "/embedding-model",
            "PERSONA_EMBEDDING_REVISION": "revision-1",
            "PERSONA_EMBEDDING_DEVICE": "cpu",
            "PERSONA_NEO4J_URI": "bolt://neo4j.invalid:7687",
            "PERSONA_NEO4J_USER": "neo4j",
            "PERSONA_NEO4J_PASSWORD": "fixture-password",
            "PERSONA_NEO4J_DATABASE": "neo4j",
        },
    )

    assert [arm.name for arm in config.arms[-2:]] == [
        "hybrid_kg_memory_4096",
        "hybrid_kg_memory_16384",
    ]
    assert config.hybrid_memory is not None
    assert config.hybrid_memory.embedding_model_id == "BAAI/fixture"
    assert config.hybrid_memory.embedding_model_path == Path("/embedding-model")
    assert config.hybrid_memory.neo4j_uri == "bolt://neo4j.invalid:7687"
    assert config.hybrid_memory.neo4j_user == "neo4j"
    assert config.hybrid_memory.neo4j_password == "fixture-password"
    assert config.hybrid_memory.neo4j_database == "neo4j"
    assert config.hybrid_memory.rrf_k == 60


def test_neurosym_config_requires_live_neo4j() -> None:
    root = Path(__file__).parents[1]
    environ = {
        "PERSONA_DATASET_DIR": "/dataset",
        "PERSONA_BENCHMARK_OUTPUT_DIR": "/output",
        "PERSONA_QWEN_MODEL_PATH": "/model",
        "PERSONA_QWEN_MODEL_ID": "Qwen/fixture",
        "PERSONA_QWEN_DEVICE": "cuda",
        "PERSONA_SCALLOP_ENDPOINT": "http://scallop.invalid",
        "PERSONA_RETRIEVAL_INDEX_ROOT": "/indexes",
        "PERSONA_EMBEDDING_MODEL_ID": "BAAI/fixture",
        "PERSONA_EMBEDDING_MODEL_PATH": "/embedding-model",
        "PERSONA_EMBEDDING_REVISION": "revision-1",
        "PERSONA_EMBEDDING_DEVICE": "cpu",
        "PERSONA_NEO4J_USER": "neo4j",
        "PERSONA_NEO4J_PASSWORD": "fixture-password",
        "PERSONA_NEO4J_DATABASE": "neo4j",
    }

    with pytest.raises(ValueError, match="PERSONA_NEO4J_URI"):
        load_benchmark_config(
            root / "configs" / "persona_end_to_end_neurosym.json",
            environ=environ,
        )


def test_benchmark_authenticates_and_rebuilds_all_120_conditions(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    config = load_benchmark_config(
        root / "configs" / "persona_end_to_end_benchmark.json",
        environ={
            "PERSONA_DATASET_DIR": str(root / "results" / "persona_conflict_conversations_v1"),
            "PERSONA_BENCHMARK_OUTPUT_DIR": str(tmp_path),
            "PERSONA_QWEN_MODEL_PATH": "/model",
            "PERSONA_QWEN_MODEL_ID": "Qwen/fixture",
            "PERSONA_QWEN_DEVICE": "cuda",
            "PERSONA_SCALLOP_ENDPOINT": "http://scallop.invalid",
        },
    )

    scheduled, turns, events = _load_source_and_rebuild(config, WordChatTokenizer())

    assert len(scheduled["inputs"]) == 120
    assert len(turns) == len(scheduled["turns"])
    assert events
    assert scheduled["dataset"]["generation_manifest_sha256"] == (
        config.schedule.source_manifest_sha256
    )


def test_sliding_prompt_uses_exact_chat_template_count_and_maximal_suffix() -> None:
    tokenizer = WordChatTokenizer()
    turns = [_turn(index) for index in range(1, 5)]

    selected, prompt, count = _fit_sliding_prompt(
        turns,
        history_id="account-a",
        query_text="which preference",
        prompt_instruction="Answer briefly.",
        cap=27,
        tokenizer=tokenizer,
    )

    assert count == len(tokenizer.encode(prompt, add_special_tokens=False)) <= 27
    assert selected
    if len(selected) < len(turns):
        _, _, longer_count = _fit_sliding_prompt(
            turns[-len(selected) - 1 :],
            history_id="account-a",
            query_text="which preference",
            prompt_instruction="Answer briefly.",
            cap=10_000,
            tokenizer=tokenizer,
        )
        assert longer_count > 27


def test_structured_prompt_is_causal_same_history_deduplicated_and_maximal() -> None:
    tokenizer = WordChatTokenizer()
    prefix = [_turn(1), _turn(2, "account-b"), _turn(3), _turn(4, "account-b"), _turn(5)]

    selected_suffix, prompt, count = _fit_structured_prompt(
        prefix,
        source_event_ids=["event-1", "event-3", "event-3"],
        history_id="account-a",
        query_text="which preference",
        prompt_instruction="Answer briefly.",
        cap=48,
        tokenizer=tokenizer,
    )

    assert count == len(tokenizer.encode(prompt, add_special_tokens=False)) <= 48
    assert prompt.count("natural conversation words 1") == 1
    assert prompt.count("natural conversation words 3") == 1
    assert "event-" not in prompt and "account-a" not in prompt and "turn-" not in prompt
    assert selected_suffix == prefix[-len(selected_suffix) :] if selected_suffix else True

    with pytest.raises(ValueError, match="causal prefix"):
        _fit_structured_prompt(
            prefix,
            source_event_ids=["event-99"],
            history_id="account-a",
            query_text="which preference",
            prompt_instruction="Answer briefly.",
            cap=48,
            tokenizer=tokenizer,
        )
    with pytest.raises(ValueError, match="crosses histories"):
        _fit_structured_prompt(
            prefix,
            source_event_ids=["event-2"],
            history_id="account-a",
            query_text="which preference",
            prompt_instruction="Answer briefly.",
            cap=48,
            tokenizer=tokenizer,
        )


def test_hybrid_kg_prompt_preserves_ranked_facts_without_identifiers() -> None:
    tokenizer = WordChatTokenizer()
    prefix = [_turn(1), _turn(2, "account-b"), _turn(3)]
    rows = [
        {
            "fact_id": "hidden-fact-1",
            "subject": "AsterArc",
            "predicate": "PREFERS",
            "object": "mint tea",
            "support_text": "AsterArc directly chose mint tea.",
        }
    ]

    suffix, retained, prompt, count = _fit_hybrid_kg_prompt(
        prefix,
        retrieved_rows=rows,
        history_id="account-a",
        query_text="What does AsterArc prefer?",
        prompt_instruction="Answer briefly.",
        cap=80,
        tokenizer=tokenizer,
    )

    assert count <= 80
    assert retained == rows
    assert suffix
    assert "AsterArc -PREFERS-> mint tea" in prompt
    assert "hidden-fact-1" not in prompt
    assert "account-a" not in prompt


def test_zero_source_structured_prompt_is_identical_to_sliding() -> None:
    tokenizer = WordChatTokenizer()
    prefix = [_turn(1, "history-001")]

    sliding = _fit_sliding_prompt(
        prefix,
        history_id="history-001",
        query_text="What changed?",
        prompt_instruction="Answer briefly.",
        cap=100,
        tokenizer=tokenizer,
    )
    structured = _fit_structured_prompt(
        prefix,
        source_event_ids=(),
        history_id="history-001",
        query_text="What changed?",
        prompt_instruction="Answer briefly.",
        cap=100,
        tokenizer=tokenizer,
    )

    assert structured == sliding


def test_scallop_canary_checks_semantics_not_just_service_identity() -> None:
    result = _run_scallop_canary(CanaryClient())
    assert {row["kind"] for row in result["injections"]} == {
        "preference_change",
        "preference_incongruity",
    }

    class WrongClient(CanaryClient):
        def derive(self, events: list[dict]) -> dict:
            result = super().derive(events)
            result["injections"][0]["source_event_ids"] = [
                "canary-old-conflict",
                "canary-new-conflict",
                "canary-alias",
            ]
            return result

    with pytest.raises(ValueError, match="semantic canary"):
        _run_scallop_canary(WrongClient())


def test_relation_coverage_allows_early_zero_but_requires_delayed_expected_relation() -> None:
    base = {
        "history_id": "account-a",
        "query_id": "query-preference-change-delayed",
        "requested_token_distance": None,
    }
    conditions = [
        {**base, "evaluation_input_id": "q:pre", "phase": "pre_update"},
        {**base, "evaluation_input_id": "q:post", "phase": "post_update"},
        {**base, "evaluation_input_id": "q:delayed", "phase": "delayed_probe"},
    ]
    injections = {
        "q:pre": {"relations": {}, "source_event_ids": []},
        "q:post": {"relations": {}, "source_event_ids": []},
        "q:delayed": {
            "relations": {"preference_change": ["event-1", "event-2", "event-3"]},
            "source_event_ids": ["event-1", "event-2", "event-3"],
        },
    }

    matrix = _relation_coverage_matrix(conditions, injections)

    assert matrix[0]["zero_sources_allowed"] is True
    assert matrix[1]["zero_sources_allowed"] is True
    assert matrix[2]["expected_relation_present"] is True
    injections["q:delayed"] = {"relations": {}, "source_event_ids": []}
    with pytest.raises(ValueError, match="delayed_probe.*preference_change"):
        _relation_coverage_matrix(conditions, injections)


def test_condition_source_derivation_never_receives_future_aliases() -> None:
    turns = [
        _turn(1),
        _turn(2),
        {
            **_turn(3),
            "stream_event": {"event_id": "future-alias", "history_id": "account-a"},
        },
    ]
    conditions = [
        {
            "evaluation_input_id": "q:post",
            "history_id": "account-a",
            "checkpoint_turn_index": 2,
        }
    ]
    client = EmptyInjectionClient()

    result = _derive_condition_sources(conditions, turns, client)

    assert result["q:post"]["source_event_ids"] == []
    assert {event["event_id"] for event in client.calls[0]} == {"event-1", "event-2"}
    assert "future-alias" not in {event["event_id"] for event in client.calls[0]}


def test_full_context_fails_instead_of_truncating_over_capacity() -> None:
    _assert_full_context_fits(input_tokens=90, max_new_tokens=10, model_capacity=100)
    with pytest.raises(ValueError, match="full_qwen_context.*model capacity"):
        _assert_full_context_fits(input_tokens=91, max_new_tokens=10, model_capacity=100)


def test_arm_specs_cover_every_condition_without_latent_prompt_ids() -> None:
    tokenizer = WordChatTokenizer()
    turns = [_turn(1), _turn(2)]
    condition = {
        "evaluation_input_id": "query-1:post_update",
        "history_id": "account-a",
        "query_id": "query-1-preference-change-delayed",
        "phase": "post_update",
        "checkpoint_turn_index": 2,
        "requested_token_distance": None,
        "query_text": "which preference",
        "gold": "mint tea",
    }

    specs = _build_arm_specs(
        [condition],
        turns,
        arms=(
            ("sliding_context_4096", "sliding_context", 4096),
            ("structured_memory_4096", "structured_memory", 4096),
            ("full_qwen_context", "full_qwen_context", None),
        ),
        injections={
            condition["evaluation_input_id"]: {
                "relations": {"preference_change": ["event-1"]},
                "source_event_ids": ["event-1"],
            }
        },
        prompt_instruction="Answer briefly.",
        tokenizer=tokenizer,
    )

    assert len(specs) == 3
    assert {spec["query_family"] for spec in specs} == {"preference_change"}
    assert all("event-" not in spec["prompt"] for spec in specs)
    assert all("account-a" not in spec["prompt"] for spec in specs)


def test_arm_specs_report_unrelated_gold_occurrences_without_filtering() -> None:
    tokenizer = WordChatTokenizer()
    turns = [
        {**_turn(1), "text": "The account holder selected mint tea."},
        {**_turn(2, "account-b"), "text": "Someone unrelated also mentioned mint tea twice: mint tea."},
    ]
    condition = {
        "evaluation_input_id": "query-1:post_update",
        "history_id": "account-a",
        "query_id": "query-1-preference-change-delayed",
        "phase": "post_update",
        "checkpoint_turn_index": 2,
        "requested_token_distance": None,
        "query_text": "which preference",
        "gold": "mint tea",
    }

    specs = _build_arm_specs(
        [condition],
        turns,
        arms=(("full_qwen_context", "full_qwen_context", None),),
        injections={condition["evaluation_input_id"]: {"relations": {}, "source_event_ids": []}},
        prompt_instruction="Answer briefly.",
        tokenizer=tokenizer,
    )

    assert specs[0]["unrelated_gold_occurrence_count"] == 2
    assert "Someone unrelated" in specs[0]["prompt"]


def test_resume_recovers_only_one_malformed_unterminated_tail(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    path.write_bytes(b'{"row": 1}\n{"row":')

    rows, recovery = _load_resumable_jsonl(path)

    assert rows == [{"row": 1}]
    assert recovery["recovered"] is True
    assert recovery["removed_byte_count"] == len(b'{"row":')
    assert path.read_bytes() == b'{"row": 1}\n'

    path.write_bytes(b'{"row":\n{"row": 2}\n')
    with pytest.raises(ValueError, match="interior|terminated"):
        _load_resumable_jsonl(path)


def test_git_provenance_records_head_and_dirty_diff_hash() -> None:
    provenance = _git_provenance(Path(__file__).parents[1])

    assert len(provenance["git_head"]) == 40
    assert len(provenance["dirty_diff_sha256"]) == 64
    assert isinstance(provenance["dirty"], bool)


def test_resume_provenance_covers_scallop_injections_specs_script_and_git(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    config = load_benchmark_config(
        root / "configs" / "persona_end_to_end_benchmark.json",
        environ={
            "PERSONA_DATASET_DIR": "/dataset",
            "PERSONA_BENCHMARK_OUTPUT_DIR": "/output",
            "PERSONA_QWEN_MODEL_PATH": "/model",
            "PERSONA_QWEN_MODEL_ID": "Qwen/fixture",
            "PERSONA_QWEN_DEVICE": "cuda",
            "PERSONA_SCALLOP_ENDPOINT": "http://scallop.invalid",
        },
    )
    scheduled = {
        "dataset": {"generation_manifest_sha256": "a" * 64},
        "tokenizer": {"implementation": "fixture"},
        "turns": [{"turn_id": "turn-1"}],
        "inputs": [{"evaluation_input_id": "q:post"}],
        "schedule_metrics": {"turn_count": 1},
    }
    injections = {
        "q:post": {
            "relations": {"preference_change": ["event-1"]},
            "source_event_ids": ["event-1"],
        }
    }
    specs = [{"evaluation_input_id": "q:post", "arm": "sliding", "prompt": "exact"}]
    immutable = _resume_manifest_fields(
        config,
        scheduled,
        {"model_id": "Qwen/fixture"},
        "b" * 64,
        scallop_identity={
            "engine": "scallopy",
            "version": "0.4.2",
            "rule_version": "preference_stream.v1",
        },
        injections=injections,
        specs=specs,
        relation_coverage=[{"condition": "post_update"}],
        evaluator_script_sha256="c" * 64,
        git_provenance={
            "git_head": "d" * 40,
            "dirty": True,
            "dirty_diff_sha256": "e" * 64,
            "untracked_file_count": 3,
        },
    )

    assert immutable["scallop"]["rule_version"] == "preference_stream.v1"
    assert len(immutable["injection_map_sha256"]) == 64
    assert len(immutable["ordered_spec_sha256"]) == 64
    assert immutable["evaluator_script_sha256"] == "c" * 64
    assert immutable["git"]["git_head"] == "d" * 40
    manifest_path = tmp_path / "generation_manifest.json"
    manifest_path.write_text(json.dumps(immutable), encoding="utf-8")
    _validate_resume_manifest(manifest_path, immutable)
    changed = {**immutable, "injection_map_sha256": "f" * 64}
    with pytest.raises(ValueError, match="injection_map_sha256 differs"):
        _validate_resume_manifest(manifest_path, changed)


def test_short_answer_scoring_and_history_clustered_paired_deltas() -> None:
    assert _score_short_answer("Answer: mint tea\nExtra explanation", "mint tea") == {
        "short_answer": "mint tea",
        "exact_match": 1.0,
        "f1": 1.0,
    }
    rows = []
    for history_id, sliding, structured in (
        ("h1", 0.0, 1.0),
        ("h2", 1.0, 1.0),
        ("h3", 0.0, 0.0),
    ):
        for arm, score in (("sliding", sliding), ("structured", structured)):
            rows.append(
                {
                    "evaluation_input_id": f"{history_id}:post",
                    "history_id": history_id,
                    "arm": arm,
                    "condition": "post_update",
                    "query_family": "preference_change",
                    "exact_match": score,
                    "f1": score,
                }
            )

    metrics = _aggregate_rows(
        rows,
        comparisons=(("structured", "sliding"),),
        bootstrap_samples=200,
        seed=7,
    )

    assert {row["arm"] for row in metrics["by_arm"]} == {"sliding", "structured"}
    assert metrics["by_arm_condition"][0]["condition"] == "post_update"
    assert metrics["by_arm_query_family"][0]["query_family"] == "preference_change"
    delta = metrics["paired_deltas"][0]
    assert delta["history_cluster_count"] == 3
    assert delta["exact_match_delta"] == pytest.approx(1 / 3)
    assert delta["bootstrap_samples"] == 200
    assert len(metrics["paired_deltas_by_condition"]) == 1
    assert len(metrics["paired_deltas_by_query_family"]) == 1
    assert len(metrics["paired_deltas_by_condition_query_family"]) == 1
    assert "exact_match_delta_ci_95" in metrics["paired_deltas_by_condition"][0]
    assert "f1_delta_ci_95" in metrics["paired_deltas_by_query_family"][0]
    diagnostic = metrics["unrelated_gold_occurrences_by_arm_condition"][0]
    assert diagnostic["total_occurrences"] == 0
