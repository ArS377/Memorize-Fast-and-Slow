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
                "turns": [
                    {"role": "user", "content": event["surface_text"]},
                    {
                        "role": "assistant",
                        "content": "Thanks, I will keep that in mind. "
                        + (event["semantic_markers"][0] if event["semantic_markers"] else ""),
                    },
                ],
            }
            for event in payload["events"]
        ]
        return LLMResponse(
            content=json.dumps({"events": events}, sort_keys=True),
            finish_reason="stop",
            model="fake-kimi",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )


class FailingClient:
    """Raise a provider error to exercise failed-run lifecycle recording."""

    def complete(self, **_kwargs: Any) -> LLMResponse:
        raise RuntimeError("fixture provider unavailable")


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
        "events.jsonl",
        "examples.jsonl",
        "facts.jsonl",
        "queries.jsonl",
        "raw_responses.jsonl",
    }
    events = _jsonl(output / "events.jsonl")
    queries = _jsonl(output / "queries.jsonl")
    assert all("User:" in event["model_text"] for event in events)
    assert all("value-" not in event["model_text"] for event in events)
    assert all("surface_query_text" in query and "surface_gold" in query for query in queries)
    assert all("value-" not in str(query["surface_gold"]) for query in queries)
    assert all("account holder" in query["surface_query_text"].lower() for query in queries)
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
    validate_generation_response(json.dumps(valid), expected)

    leaked_id = json.loads(json.dumps(valid))
    leaked_id["events"][0]["turns"][0]["content"] += " history-001-add"
    with pytest.raises(ValueError, match="latent benchmark label"):
        validate_generation_response(json.dumps(leaked_id), expected)

    leaked_gold = json.loads(json.dumps(valid))
    leaked_gold["events"][0]["turns"][0]["content"] += " value-001-a"
    with pytest.raises(ValueError, match="latent benchmark label"):
        validate_generation_response(json.dumps(leaked_gold), expected)

    missing = json.loads(json.dumps(valid))
    missing["events"].pop()
    with pytest.raises(ValueError, match="missing event IDs"):
        validate_generation_response(json.dumps(missing), expected)


def test_generation_rejects_semantically_inverted_retraction() -> None:
    expected = [
        {
            "event_id": "history-001-retract",
            "surface_text": "The account holder retracted mint tea.",
            "required_surface_value": "mint tea",
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
        validate_generation_response(json.dumps(inverted), expected)

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
        validate_generation_response(json.dumps(negated), expected)


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
