from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from experiments.persona_conversation_generator import (
    GenerationConfig,
    LLMResponse,
    default_personas,
    generate_persona_conversations,
    validate_generation_response,
)


class SurfaceFakeClient:
    """Return deterministic role-structured dialogue from the supplied surface facts."""

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        timeout: float,
        seed: int,
        max_tokens: int,
    ) -> LLMResponse:
        del model, timeout, seed, max_tokens
        payload = json.loads(messages[-1]["content"])
        events = [
            {
                "event_id": event["event_id"],
                "turns": self._turns(event, payload["turn_pairs_per_event"]),
            }
            for event in payload["events"]
        ]
        return LLMResponse(
            content=json.dumps({"events": events}, sort_keys=True),
            finish_reason="stop",
            model="fake-kimi",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

    @staticmethod
    def _turns(event: Mapping[str, Any], turn_pairs: int) -> list[dict[str, str]]:
        """Build exact-length dialogue while preserving every required concept."""
        required = " ".join(event["required_surface_values"])
        marker = event["semantic_markers"][0] if event["semantic_markers"] else "record"
        authority = event["authority_markers"][0] if event["authority_markers"] else ""
        turns = []
        for pair_index in range(turn_pairs):
            turns.extend(
                [
                    {
                        "role": "user",
                        "content": (
                            f"{event['surface_text']} Please discuss {required} clearly for this task phase."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"I will {marker} the stated choice and preserve its task-specific context accurately."
                            f" {authority}"
                        ),
                    },
                ]
            )
        return turns


class FailingClient:
    """Raise a provider error to exercise failed-run lifecycle recording."""

    def complete(self, **_kwargs: Any) -> LLMResponse:
        raise RuntimeError("fixture provider unavailable")


class UnexpectedClient:
    """Fail if resumable generation makes an unnecessary provider call."""

    def complete(self, **_kwargs: Any) -> LLMResponse:
        raise AssertionError("provider should not be called for revalidated cached responses")


class RepairingClient(SurfaceFakeClient):
    """Return one schema-invalid response before honoring the changed repair request."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **request: Any) -> LLMResponse:
        self.calls += 1
        response = super().complete(**request)
        if self.calls != 1:
            return response
        payload = json.loads(response.content)
        payload["events"][0]["turns"][0]["role"] = "assistant"
        return LLMResponse(
            content=json.dumps(payload, sort_keys=True),
            finish_reason="stop",
            model=response.model,
            usage=response.usage,
        )


def _config() -> GenerationConfig:
    return GenerationConfig(
        endpoint="https://fixture.invalid/v1",
        api_key="fixture-secret",
        model="fixture-kimi",
        timeout_seconds=3.0,
        seed=19,
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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generation_reuses_latent_artifacts_and_adds_natural_surfaces(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    manifest = generate_persona_conversations(output, _config(), SurfaceFakeClient())

    assert manifest["status"] == "completed"
    assert manifest["model_identity"] == "fake-kimi"
    assert "api_key" not in json.dumps(manifest)
    assert set(manifest["artifact_sha256"]) >= {
        "candidate_updates.jsonl",
        "dialogue.jsonl",
        "events.jsonl",
        "examples.jsonl",
        "facts.jsonl",
        "queries.jsonl",
        "source_documents.jsonl",
        "raw_responses.jsonl",
    }
    assert manifest["artifact_roles"]["dialogue.jsonl"] == "model_input"
    assert manifest["artifact_roles"]["events.jsonl"] == "latent_structure"
    assert manifest["artifact_roles"]["queries.jsonl"] == "supervision_only"
    assert manifest["hashing"]["algorithm"] == "sha256"
    assert manifest["reference_contract"]["replay_order"] == "events.sequence_index"
    events = _jsonl(output / "events.jsonl")
    dialogue = _jsonl(output / "dialogue.jsonl")
    queries = _jsonl(output / "queries.jsonl")
    assert all("User:" in event["model_text"] for event in events)
    assert set(dialogue[0]) == {
        "conversation_id",
        "event_index",
        "speaker",
        "subject",
        "text",
    }
    assert all("fact" not in row and "operation" not in row for row in dialogue)
    assert all("value-" not in event["model_text"] for event in events)
    assert all("surface_query_text" in query and "surface_gold" in query for query in queries)
    assert all(query["query_text"] == query["surface_query_text"] for query in queries)
    assert all("value-" not in str(query["surface_gold"]) for query in queries)
    assert all("subject-" not in query["surface_query_text"] for query in queries)
    assert all("AsterArc" not in query["surface_query_text"] for query in queries)


def test_generation_rejects_hidden_ids_gold_leakage_and_missing_turn() -> None:
    expected = [
        {"event_id": "history-001-add", "surface_text": "Morgan prefers cedar tea."},
        {"event_id": "history-001-transition", "surface_text": "Morgan now prefers mint tea."},
    ]
    valid = {
        "events": [
            {
                "event_id": event["event_id"],
                "turns": [
                    {"role": "user", "content": event["surface_text"]},
                    {"role": "assistant", "content": "Understood."},
                ],
            }
            for event in expected
        ]
    }
    validate_generation_response(
        json.dumps(valid), expected, turn_pairs_per_event=1, minimum_words_per_turn=1
    )

    leaked_id = json.loads(json.dumps(valid))
    leaked_id["events"][0]["turns"][0]["content"] += " history-001-add"
    with pytest.raises(ValueError, match="latent benchmark label"):
        validate_generation_response(
            json.dumps(leaked_id), expected, turn_pairs_per_event=1, minimum_words_per_turn=1
        )

    leaked_gold = json.loads(json.dumps(valid))
    leaked_gold["events"][0]["turns"][0]["content"] += " value-001-a"
    with pytest.raises(ValueError, match="latent benchmark label"):
        validate_generation_response(
            json.dumps(leaked_gold), expected, turn_pairs_per_event=1, minimum_words_per_turn=1
        )

    missing = json.loads(json.dumps(valid))
    missing["events"].pop()
    with pytest.raises(ValueError, match="missing event IDs"):
        validate_generation_response(
            json.dumps(missing), expected, turn_pairs_per_event=1, minimum_words_per_turn=1
        )

    four_digit_leak = json.loads(json.dumps(valid))
    four_digit_leak["events"][0]["turns"][0]["content"] += " subject-1000"
    with pytest.raises(ValueError, match="latent benchmark label"):
        validate_generation_response(
            json.dumps(four_digit_leak),
            expected,
            turn_pairs_per_event=1,
            minimum_words_per_turn=1,
        )


def test_generation_rejects_semantically_inverted_retraction() -> None:
    expected = [
        {
            "event_id": "history-001-retract",
            "surface_text": "The account holder retracted mint tea.",
            "required_surface_values": ["mint tea"],
            "semantic_markers": ["retract", "remove", "withdraw", "no longer"],
        }
    ]
    inverted = {
        "events": [
            {
                "event_id": "history-001-retract",
                "turns": [
                    {"role": "user", "content": "Please record mint tea."},
                    {"role": "assistant", "content": "I saved mint tea."},
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="operation semantics"):
        validate_generation_response(
            json.dumps(inverted), expected, turn_pairs_per_event=1, minimum_words_per_turn=1
        )

    negated = {
        "events": [
            {
                "event_id": "history-001-retract",
                "turns": [
                    {"role": "user", "content": "Do not retract mint tea."},
                    {"role": "assistant", "content": "I kept mint tea recorded."},
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="operation semantics"):
        validate_generation_response(
            json.dumps(negated), expected, turn_pairs_per_event=1, minimum_words_per_turn=1
        )


def test_generation_enforces_dialogue_length_and_conflict_values() -> None:
    expected = [
        {
            "event_id": "history-001-conflict-resolution",
            "surface_text": "The user resolved the workspace conflict.",
            "required_surface_values": [
                "quiet private studio",
                "open team lounge",
                "window desk",
            ],
            "semantic_markers": ["clarify", "resolve"],
        }
    ]
    turns = [
        {
            "role": "user",
            "content": "The quiet private studio and open team lounge requests conflict, so clarify them.",
        },
        {
            "role": "assistant",
            "content": "I understand that the choice now resolves to the window desk for this project.",
        },
        {
            "role": "user",
            "content": "Please preserve this as the final workspace decision for all later phases.",
        },
        {
            "role": "assistant",
            "content": "I will keep the resolved decision separate from unrelated intervening tasks.",
        },
    ]
    valid = {"events": [{"event_id": expected[0]["event_id"], "turns": turns}]}

    validate_generation_response(
        json.dumps(valid), expected, turn_pairs_per_event=2, minimum_words_per_turn=8
    )

    too_short = json.loads(json.dumps(valid))
    too_short["events"][0]["turns"] = too_short["events"][0]["turns"][:2]
    with pytest.raises(ValueError, match="exactly 4 turns"):
        validate_generation_response(
            json.dumps(too_short), expected, turn_pairs_per_event=2, minimum_words_per_turn=8
        )

    missing_conflict_value = json.loads(json.dumps(valid))
    missing_conflict_value["events"][0]["turns"][0]["content"] = (
        "The quiet private studio request remains relevant, so please clarify the conflict carefully."
    )
    with pytest.raises(ValueError, match="open team lounge"):
        validate_generation_response(
            json.dumps(missing_conflict_value),
            expected,
            turn_pairs_per_event=2,
            minimum_words_per_turn=8,
        )

    short_turn = json.loads(json.dumps(valid))
    short_turn["events"][0]["turns"][2]["content"] = "Keep it."
    with pytest.raises(ValueError, match="at least 8 words"):
        validate_generation_response(
            json.dumps(short_turn), expected, turn_pairs_per_event=2, minimum_words_per_turn=8
        )

    forbidden = json.loads(json.dumps(valid))
    expected[0]["forbidden_surface_phrases"] = ["everything matches"]
    forbidden["events"][0]["turns"][2]["content"] = (
        "Everything matches the previous record, so preserve this final workspace decision now."
    )
    with pytest.raises(ValueError, match="forbidden phrase"):
        validate_generation_response(
            json.dumps(forbidden), expected, turn_pairs_per_event=2, minimum_words_per_turn=8
        )

    repetitive = json.loads(json.dumps(valid))
    repetitive["events"][0]["turns"][2]["content"] += (
        " Record this entry in the file and mark it noted."
    )
    with pytest.raises(ValueError, match="recordkeeping terms"):
        validate_generation_response(
            json.dumps(repetitive), expected, turn_pairs_per_event=2, minimum_words_per_turn=8
        )

    gendered = json.loads(json.dumps(valid))
    gendered["events"][0]["turns"][2]["content"] += " He confirmed the final choice."
    with pytest.raises(ValueError, match="unsupported gendered pronoun"):
        validate_generation_response(
            json.dumps(gendered), expected, turn_pairs_per_event=2, minimum_words_per_turn=8
        )


def test_v3_prompt_preserves_interspersed_conflict_lifecycle(tmp_path: Path) -> None:
    config = GenerationConfig(
        endpoint="https://fixture.invalid/v1",
        api_key="fixture-secret",
        model="fixture-kimi",
        timeout_seconds=3.0,
        seed=47,
        max_tokens=16384,
        history_count=1,
        condition="lexical",
        hardness_profile="anti_shortcut_interleaved_v3",
        split_counts=(0, 0, 1),
        prompt_schema_version="persona-conversation.v3",
        turn_pairs_per_event=1,
        minimum_words_per_turn=1,
        events_per_request=8,
        enable_thinking=False,
        max_validation_attempts=2,
        resume_existing=True,
    )
    output = tmp_path / "v3"

    generate_persona_conversations(output, config, SurfaceFakeClient())

    requests = _jsonl(output / "requests.jsonl")
    assert len(requests) == 4
    prompt_events = [
        event
        for request in requests
        for event in json.loads(request["messages"][-1]["content"])["events"]
    ]
    event_ids = [event["event_id"] for event in prompt_events]
    assert event_ids[4].endswith("-conflict-left")
    assert event_ids[13].endswith("-conflict-right")
    assert event_ids[21].endswith("-conflict-resolution")
    assert prompt_events[4]["semantic_markers"] == []
    assert "do not invent an alternative" in prompt_events[4]["semantic_instruction"]
    assert prompt_events[13]["required_surface_values"] == [
        "quiet private studio",
        "open team lounge",
    ]
    resolution = prompt_events[21]
    assert resolution["required_surface_values"] == [
        "quiet private studio",
        "open team lounge",
        "window desk",
    ]
    assert "resolve" in resolution["semantic_markers"]
    prompts_by_id = {event["event_id"]: event for event in prompt_events}
    assert prompts_by_id["history-001-add"]["dialogue_subject"] == "AsterArc"
    assert prompts_by_id["history-001-add"]["dialogue_speaker"] == "AsterArc"
    assert prompts_by_id["history-001-add"]["authority_markers"] == []
    assert prompts_by_id["history-001-replaceable"]["dialogue_subject"] != "AsterArc"
    transition = prompts_by_id["history-001-transition"]
    assert "through 2025-09-30" in transition["semantic_instruction"]
    assert "going forward" in transition["forbidden_surface_phrases"]
    assert prompts_by_id["history-001-backdated"]["required_surface_values"] == [
        "cedar tea",
        "oolong tea",
    ]
    assert prompts_by_id["history-001-direct-correction"]["required_surface_values"] == [
        "evening delivery",
        "weekend delivery",
    ]
    duplicate = prompts_by_id["history-001-duplicate"]
    assert "do not discuss delivery timing" in duplicate["semantic_instruction"]
    assert "deduplicat" in duplicate["forbidden_surface_phrases"]
    negative = prompts_by_id["history-001-lineage-negative-1"]
    assert "unrelated mention only" in negative["semantic_instruction"]
    assert "inferred" in negative["authority_markers"]
    assert negative["dialogue_speaker"] == "observer"
    ambiguity_query = next(
        query
        for query in _jsonl(output / "queries.jsonl")
        if query["kind"] == "ambiguity"
    )
    assert "vegetable ramen" in ambiguity_query["surface_query_text"]
    assert all(" 001" not in event["model_text"] for event in _jsonl(output / "events.jsonl"))
    assert prompt_events[0]["surface_text"].startswith("AsterArc ")
    assert all(
        "AsterArc" in query["surface_query_text"]
        for query in _jsonl(output / "queries.jsonl")
    )


def test_personas_are_matched_without_encoding_preferences() -> None:
    personas = {persona.persona_id: persona for persona in default_personas()}
    ethnicity_a = personas["ethnicity-a"]
    ethnicity_b = personas["ethnicity-b"]
    education_a = personas["education-a"]
    education_b = personas["education-b"]

    assert ethnicity_a.counterfactual_axis == ethnicity_b.counterfactual_axis == "ethnicity"
    assert ethnicity_a.education == ethnicity_b.education
    assert ethnicity_a.ethnicity != ethnicity_b.ethnicity
    assert education_a.counterfactual_axis == education_b.counterfactual_axis == "education"
    assert education_a.ethnicity == education_b.ethnicity
    assert education_a.education != education_b.education
    assert all("prefer" not in persona.biography.lower() for persona in personas.values())
    assert personas["no-persona"].biography == ""


def test_personas_cover_requested_background_and_protected_attributes() -> None:
    personas = [persona for persona in default_personas() if persona.persona_id != "no-persona"]

    assert personas
    for persona in personas:
        assert persona.gender_identity
        assert persona.disability
        assert persona.religion
        assert persona.socioeconomic_background
        assert persona.locale
        assert persona.occupation
        assert persona.languages
        assert persona.interests


def test_deterministic_responses_have_stable_hashes(tmp_path: Path) -> None:
    first = generate_persona_conversations(tmp_path / "first", _config(), SurfaceFakeClient())
    second = generate_persona_conversations(tmp_path / "second", _config(), SurfaceFakeClient())

    assert first["response_sha256"] == second["response_sha256"]
    assert hashlib.sha256((tmp_path / "first" / "events.jsonl").read_bytes()).hexdigest() == (
        hashlib.sha256((tmp_path / "second" / "events.jsonl").read_bytes()).hexdigest()
    )
    assert first["prompt_sha256"] == second["prompt_sha256"]
    assert "requests.jsonl" in first["artifact_sha256"]


def test_provider_failure_writes_failed_manifest(tmp_path: Path) -> None:
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fixture provider unavailable"):
        generate_persona_conversations(output, _config(), FailingClient())

    manifest = json.loads((output / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["completion_status"] == "failed"
    assert (output / "raw_responses.jsonl").exists()


def test_generation_reuses_only_revalidated_prompt_matched_responses(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resume"
    first = generate_persona_conversations(output, _config(), SurfaceFakeClient())

    resumed = generate_persona_conversations(output, _config(), UnexpectedClient())

    assert resumed["status"] == "completed"
    assert resumed["resumed_response_count"] == len(first["response_sha256"])
    assert resumed["response_sha256"] == first["response_sha256"]


def test_generation_repairs_validation_failure_with_changed_attempt(
    tmp_path: Path,
) -> None:
    client = RepairingClient()

    manifest = generate_persona_conversations(tmp_path / "repair", _config(), client)

    requests = _jsonl(tmp_path / "repair" / "requests.jsonl")
    raw = _jsonl(tmp_path / "repair" / "raw_responses.jsonl")
    assert manifest["status"] == "completed"
    assert client.calls == 5
    assert len(requests) == len(raw) == 5
    assert requests[0]["prompt_sha256"] != requests[1]["prompt_sha256"]
    assert requests[0]["seed"] != requests[1]["seed"]
