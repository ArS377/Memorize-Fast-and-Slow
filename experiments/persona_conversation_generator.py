"""Generate auditable natural conversations over deterministic latent preference traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from experiments.synthetic_temporal_preferences import generate_dataset


PROMPT_SCHEMA_VERSION = "persona-conversation.v2"
SOURCE_ARTIFACTS = (
    "candidate_updates.jsonl",
    "events.jsonl",
    "examples.jsonl",
    "facts.jsonl",
    "queries.jsonl",
    "source_documents.jsonl",
)
_LATENT_LABEL = re.compile(
    r"\b(?:history|subject|value|scope|example|candidate|replacement)-\d+(?:-[a-z][\w-]*)?\b"
    r"|\b(?:event_id|fact_id|query_id|operation|gold)\s*="
)
_RECORDKEEPING_LANGUAGE = re.compile(
    r"\b(?:record(?:ed|ing|s)?|entr(?:y|ies)|files?|noted|audit trail|canonical)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_GENDERED_PRONOUN = re.compile(
    r"\b(?:he|she|him|her|his|hers)\b",
    re.IGNORECASE,
)
_PREFERENCE_SURFACES = {
    "a": "cedar tea",
    "b": "mint tea",
    "c": "window seating",
    "d": "aisle seating",
    "e": "vegetable ramen",
    "f": "mushroom risotto",
    "g": "paper receipts",
    "h": "oolong tea",
    "i": "evening delivery",
    "j": "morning delivery",
    "k": "weekend delivery",
    "l": "digital receipts",
    "m": "weekday delivery",
    "n": "tofu noodles",
}
_GIVEN_NAMES = (
    "Morgan", "Riley", "Jordan", "Casey", "Taylor", "Avery", "Cameron", "Drew", "Parker", "Quinn",
)
_FAMILY_NAMES = (
    "Lee", "Chen", "Patel", "Rivera", "Okafor", "Kim", "Nguyen", "Garcia", "Brown", "Singh",
)


def _nonempty(value: Any, name: str) -> str:
    """Validate and return one non-empty string boundary value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    """Validate and return one positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    """Validate and return one positive finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


@dataclass(frozen=True)
class Persona:
    """Identity-only persona metadata kept separate from benchmark task semantics."""

    persona_id: str
    name: str
    age: int | None
    ethnicity: str | None
    education: str | None
    gender_identity: str | None
    disability: str | None
    religion: str | None
    socioeconomic_background: str | None
    locale: str | None
    occupation: str | None
    languages: tuple[str, ...]
    interests: tuple[str, ...]
    biography: str
    pair_id: str | None
    counterfactual_axis: str | None

    def __post_init__(self) -> None:
        """Reject incomplete personas and invalid counterfactual declarations."""
        _nonempty(self.persona_id, "persona_id")
        if self.persona_id == "no-persona":
            if any(
                value is not None
                for value in (
                    self.age,
                    self.ethnicity,
                    self.education,
                    self.gender_identity,
                    self.disability,
                    self.religion,
                    self.socioeconomic_background,
                    self.locale,
                    self.occupation,
                    self.pair_id,
                    self.counterfactual_axis,
                )
            ) or self.name or self.languages or self.interests or self.biography:
                raise ValueError("no-persona must contain no identity attributes or biography")
            return
        _nonempty(self.name, "persona.name")
        if isinstance(self.age, bool) or not isinstance(self.age, int) or self.age < 18:
            raise ValueError("persona.age must be an adult integer")
        _nonempty(self.ethnicity, "persona.ethnicity")
        _nonempty(self.education, "persona.education")
        _nonempty(self.gender_identity, "persona.gender_identity")
        _nonempty(self.disability, "persona.disability")
        _nonempty(self.religion, "persona.religion")
        _nonempty(self.socioeconomic_background, "persona.socioeconomic_background")
        _nonempty(self.locale, "persona.locale")
        _nonempty(self.occupation, "persona.occupation")
        if not self.languages or any(not isinstance(value, str) or not value.strip() for value in self.languages):
            raise ValueError("persona.languages must contain non-empty strings")
        if not self.interests or any(not isinstance(value, str) or not value.strip() for value in self.interests):
            raise ValueError("persona.interests must contain non-empty strings")
        _nonempty(self.biography, "persona.biography")
        _nonempty(self.pair_id, "persona.pair_id")
        if self.counterfactual_axis not in {"ethnicity", "education"}:
            raise ValueError("persona.counterfactual_axis must be ethnicity or education")


@dataclass(frozen=True)
class GenerationConfig:
    """Validated Kimi generation and deterministic dataset configuration."""

    endpoint: str
    api_key: str
    model: str
    timeout_seconds: float
    seed: int
    max_tokens: int
    history_count: int
    condition: str
    hardness_profile: str
    split_counts: tuple[int, int, int]
    prompt_schema_version: str
    turn_pairs_per_event: int
    minimum_words_per_turn: int
    events_per_request: int
    enable_thinking: bool

    def __post_init__(self) -> None:
        """Validate all provider and source-generation boundaries."""
        _nonempty(self.endpoint, "generation.endpoint")
        _nonempty(self.api_key, "generation.api_key")
        _nonempty(self.model, "generation.model")
        if "kimi" not in self.model.casefold():
            raise ValueError("generation.model must identify Kimi")
        _positive_number(self.timeout_seconds, "generation.timeout_seconds")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("generation.seed must be an integer")
        _positive_int(self.max_tokens, "generation.max_tokens")
        _positive_int(self.history_count, "generation.history_count")
        if self.condition not in {"lexical", "paraphrase"}:
            raise ValueError(f"unsupported generation condition: {self.condition}")
        if self.hardness_profile not in {
            "base",
            "anti_shortcut_stream_v2",
            "anti_shortcut_interleaved_v3",
        }:
            raise ValueError(f"unsupported hardness profile: {self.hardness_profile}")
        if (
            len(self.split_counts) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.split_counts)
            or sum(self.split_counts) != self.history_count
        ):
            raise ValueError("generation.split_counts must be three non-negative integers summing to history_count")
        _nonempty(self.prompt_schema_version, "generation.prompt_schema_version")
        _positive_int(self.turn_pairs_per_event, "generation.turn_pairs_per_event")
        _positive_int(self.minimum_words_per_turn, "generation.minimum_words_per_turn")
        _positive_int(self.events_per_request, "generation.events_per_request")
        if not isinstance(self.enable_thinking, bool):
            raise ValueError("generation.enable_thinking must be a boolean")


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral completion result used by generation and evaluation clients."""

    content: str
    finish_reason: str
    model: str
    usage: Mapping[str, Any]


class CompletionClient(Protocol):
    """Injectable OpenAI-compatible completion boundary."""

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        timeout: float,
        seed: int,
        max_tokens: int,
    ) -> LLMResponse:
        """Return one chat completion without exposing provider credentials."""


class OpenAICompletionClient:
    """Minimal adapter around the installed OpenAI-compatible SDK."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        timeout_seconds: float,
        *,
        json_mode: bool = False,
        enable_thinking: bool | None = None,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self._json_mode = json_mode
        self._enable_thinking = enable_thinking

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        timeout: float,
        seed: int,
        max_tokens: int,
    ) -> LLMResponse:
        """Call the configured endpoint and normalize the first completion choice."""
        optional: dict[str, Any] = {}
        if self._json_mode:
            optional["response_format"] = {"type": "json_object"}
        if self._enable_thinking is not None:
            optional["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self._enable_thinking}
            }
        response = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            timeout=timeout,
            seed=seed,
            max_tokens=max_tokens,
            temperature=0,
            **optional,
        )
        if not response.choices:
            raise ValueError("provider returned no completion choices")
        choice = response.choices[0]
        usage = response.usage.model_dump() if response.usage is not None else {}
        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "",
            model=response.model or model,
            usage=usage,
        )


def default_personas() -> tuple[Persona, ...]:
    """Return matched ethnicity/education counterfactuals plus a no-persona control."""
    shared = {
        "name": "Alex Morgan",
        "age": 34,
        "gender_identity": "nonbinary",
        "disability": "no disclosed disability",
        "religion": "not religious",
        "socioeconomic_background": "middle-income household",
        "locale": "Columbus, Ohio",
        "occupation": "municipal project coordinator",
        "languages": ("English",),
        "interests": ("community gardening", "mystery novels"),
    }
    common = "Alex Morgan is 34 and works as a municipal project coordinator in Columbus, Ohio."
    return (
        Persona(
            persona_id="no-persona", name="", age=None, ethnicity=None, education=None,
            gender_identity=None, disability=None, religion=None,
            socioeconomic_background=None, locale=None, occupation=None, languages=(),
            interests=(), biography="", pair_id=None, counterfactual_axis=None,
        ),
        Persona(
            persona_id="ethnicity-a", ethnicity="Korean American",
            education="bachelor's degree", biography=(
                f"{common} Alex is Korean American, holds a bachelor's degree, and enjoys "
                "community gardening and mystery novels."
            ), pair_id="ethnicity-pair", counterfactual_axis="ethnicity", **shared,
        ),
        Persona(
            persona_id="ethnicity-b", ethnicity="Mexican American",
            education="bachelor's degree", biography=(
                f"{common} Alex is Mexican American, holds a bachelor's degree, and enjoys "
                "community gardening and mystery novels."
            ), pair_id="ethnicity-pair", counterfactual_axis="ethnicity", **shared,
        ),
        Persona(
            persona_id="education-a", ethnicity="Korean American",
            education="high school diploma", biography=(
                f"{common} Alex is Korean American, holds a high school diploma, and enjoys "
                "community gardening and mystery novels."
            ), pair_id="education-pair", counterfactual_axis="education", **shared,
        ),
        Persona(
            persona_id="education-b", ethnicity="Korean American",
            education="master's degree", biography=(
                f"{common} Alex is Korean American, holds a master's degree, and enjoys "
                "community gardening and mystery novels."
            ), pair_id="education-pair", counterfactual_axis="education", **shared,
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read object-valued JSONL with line-specific malformed-input failures."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    """Write stable formatted JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write stable JSONL records in supplied order."""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Return one artifact's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _surface_maps(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build deterministic natural names for latent values, subjects, and scopes."""
    values = sorted(
        {
            str(event["fact"]["object"])
            for event in events
            if isinstance(event.get("fact"), Mapping)
            and isinstance(event["fact"].get("object"), str)
            and str(event["fact"]["object"]).startswith("value-")
        }
        | {
            match
            for event in events
            if isinstance(event.get("fact"), Mapping)
            for match in re.findall(
                r"\bvalue-\d{3}-[a-z]\b", str(event["fact"].get("support_text", ""))
            )
        }
    )
    subjects = sorted(
        {
            str(event["fact"]["subject"])
            for event in events
            if isinstance(event.get("fact"), Mapping)
            and isinstance(event["fact"].get("subject"), str)
        }
    )
    scopes = sorted(
        {
            str(event["fact"].get("qualifiers", {}).get("scope"))
            for event in events
            if isinstance(event.get("fact"), Mapping)
            and str(event["fact"].get("qualifiers", {}).get("scope", "")).startswith(
                ("scope-", "leakage-scope-", "ambiguity-scope-")
            )
        }
    )
    value_map = {}
    for value in values:
        suffix = value.rsplit("-", 1)[-1]
        if suffix not in _PREFERENCE_SURFACES:
            raise ValueError(f"unsupported latent preference suffix {suffix!r} in {value!r}")
        value_map[value] = _PREFERENCE_SURFACES[suffix]
    numbered_objects = sorted(
        {
            str(event["fact"]["object"])
            for event in events
            if isinstance(event.get("fact"), Mapping)
            and isinstance(event["fact"].get("object"), str)
            and re.search(r"\s\d+$", str(event["fact"]["object"]))
        }
    )
    for raw_object in numbered_objects:
        natural = re.sub(r"\s+\d+$", "", raw_object).replace("_", " ").strip()
        value_map[raw_object] = (
            "quiet private studio" if natural == "quiet studio" else natural
        )
    subject_map = {
        str(event["fact"]["subject"]): str(event["surface_subject"])
        for event in events
        if isinstance(event.get("fact"), Mapping)
        and isinstance(event["fact"].get("subject"), str)
        and isinstance(event.get("surface_subject"), str)
        and str(event["surface_subject"]).strip()
    }
    namespace_size = len(_GIVEN_NAMES) * len(_FAMILY_NAMES)
    for index, subject in enumerate(subjects):
        if subject in subject_map:
            continue
        cycle = index // namespace_size
        given = _GIVEN_NAMES[index % len(_GIVEN_NAMES)]
        family = _FAMILY_NAMES[(index // len(_GIVEN_NAMES)) % len(_FAMILY_NAMES)]
        middle = f" {chr(ord('A') + cycle - 1)}." if cycle else ""
        subject_map[subject] = f"{given}{middle} {family}"
    scope_map = {scope: f"{('shared', 'travel', 'work', 'home')[index % 4]} setting" for index, scope in enumerate(scopes)}
    return value_map, subject_map, scope_map


def _replace_latent(text: str, replacements: Mapping[str, str]) -> str:
    """Replace longest latent labels first to avoid overlapping substitutions."""
    result = text
    for latent, surface in sorted(replacements.items(), key=lambda item: -len(item[0])):
        result = result.replace(latent, surface)
    return result


def _surface_query(
    query: Mapping[str, Any], replacements: Mapping[str, str], value_map: Mapping[str, str]
) -> tuple[str, str]:
    """Return model-visible natural query and answer while preserving latent gold fields."""
    kind = str(query.get("kind"))
    query_id = str(query.get("query_id"))
    subject = replacements.get(str(query.get("subject")), "the account holder")
    if kind == "preference":
        scope = str(query.get("scope", "default"))
        surface_scope = replacements.get(scope, scope.replace("_", " "))
        date = str(query.get("date", "the requested date"))
        if query_id.endswith("preference-change-delayed"):
            text = (
                f"After the long record, what standing preference applied to {subject} "
                f"on {date}?"
            )
        elif query_id.endswith("preference-incongruity-delayed"):
            text = (
                f"Considering the complete record, what standing preference applied to {subject} "
                f"on {date}?"
            )
        else:
            text = (
                f"What did {subject} prefer in the {surface_scope} on {date}?"
            )
    elif kind == "recommendation":
        candidate = value_map.get(str(query.get("candidate")), str(query.get("candidate")))
        text = (
            f"Is {candidate} feasible under {subject}'s stated constraints? "
            "Answer FEASIBLE or INFEASIBLE."
        )
    elif kind == "private_recall":
        text = f"What private preference, if any, can still be recalled for {subject}?"
    elif kind == "ambiguity":
        candidate = value_map.get(str(query.get("candidate")), str(query.get("candidate")))
        text = (
            f"Was {candidate} ever settled as {subject}'s preference after the conflicting "
            "statements? Answer the exact settled preference or UNKNOWN."
        )
    elif kind == "private_lineage":
        prefix = "After the withdrawal, " if query_id.endswith("private-lineage") else ""
        text = (
            f"{prefix}what private value, if any, can still be recalled for {subject}?"
        )
    else:
        raise ValueError(f"unsupported surface query kind {kind!r} for {query_id}")
    gold = query.get("gold")
    surface_gold = value_map.get(str(gold), str(gold))
    return text, surface_gold


def _semantic_markers(event: Mapping[str, Any]) -> tuple[str, ...]:
    """Return lexical invariants that preserve operation semantics in generated dialogue."""
    operation = str(event.get("operation", "")).casefold()
    event_family = str(event.get("event_family", "")).casefold()
    if "retract" in operation:
        return ("retract", "remove", "withdraw", "no longer")
    if "hard_constraint" in operation:
        return ("avoid", "must not", "cannot", "off the table")
    if "rectification" in event_family:
        return ("resolve", "clarify", "settle", "final choice")
    if event_family == "contradiction_opening" and not event.get("conflicts_with"):
        return ()
    if "contradiction" in event_family or "ambig" in operation or "conflict" in operation:
        return ("conflict", "contradict", "not settled", "clarify", "incompatible")
    if "duplicate" in operation:
        return ("again", "already", "repeat", "same")
    if "supersede" in operation or "correction" in operation or "replace" in operation:
        return ("correct", "update", "switch", "override", "replace")
    return ()


def _semantic_instruction(
    event: Mapping[str, Any], required_values: Sequence[str]
) -> str:
    """Return event-specific constraints that prevent plausible but false elaboration."""
    operation = str(event.get("operation", "")).casefold()
    event_family = str(event.get("event_family", "")).casefold()
    if event_family == "delayed_preference_probe":
        return "Describe only the review activity; do not claim prior records agree, conflict, or remain unchanged."
    if event_family == "same_entity_hard_negative":
        return "Keep this as an unrelated mention only; do not infer a private note, preference, ownership, or duplicate lineage."
    if (
        event.get("supersedes")
        or event.get("corrects")
        or event.get("transitions_from")
    ) and len(required_values) >= 2:
        prior_values = list(required_values[:-1])
        old_value, new_value = prior_values[0], required_values[-1]
        if operation == "supersede":
            fact = event.get("fact")
            temporal = fact.get("temporal", {}) if isinstance(fact, Mapping) else {}
            valid_from = temporal.get("valid_from")
            valid_to = temporal.get("valid_to")
            return (
                f"State that {old_value} remains historical before the transition and "
                f"{new_value} is active only from {valid_from} through {valid_to}; do not "
                "describe it as open-ended or erase the old history."
            )
        if operation == "backdated_correction":
            fact = event.get("fact")
            temporal = fact.get("temporal", {}) if isinstance(fact, Mapping) else {}
            return (
                f"State that {old_value} applies only before {temporal.get('valid_from')} and "
                f"{new_value} applies from {temporal.get('valid_from')} through "
                f"{temporal.get('valid_to')}; do not claim {old_value} applies afterward."
            )
        return (
            f"State unambiguously that {new_value} supersedes all prior active choices "
            f"({', '.join(prior_values)}); do not say any prior choice stays active or unchanged."
        )
    fact = event.get("fact")
    predicate = str(fact.get("predicate", "")) if isinstance(fact, Mapping) else ""
    if "retract" in operation:
        return "Mark the value unavailable for future recall without claiming its audit history was destroyed."
    if predicate == "PRIVATE_NOTE":
        return "Discuss only a private note and its repetition; never call it a preference, favorite, dish, or recommendation signal."
    if "duplicate" in operation:
        return "Have the user naturally repeat a previously stated preference; do not discuss delivery timing, storage, merging, or deduplication."
    if event_family == "contradiction_opening" and len(required_values) == 1:
        return "State only the supplied choice; do not invent an alternative or call it a conflict yet."
    if event_family == "contradiction_opening":
        return "State exactly which two supplied choices are incompatible and leave the choice unresolved."
    if "rectification" in event_family:
        return "State both incompatible prior choices and the supplied final choice that resolves them."
    return "Do not invent facts, agreement, conflicts, or outcomes beyond the supplied event."


def _forbidden_surface_phrases(event: Mapping[str, Any]) -> tuple[str, ...]:
    """Return high-risk unsupported claims that invalidate generated dialogue."""
    operation = str(event.get("operation", "")).casefold()
    event_family = str(event.get("event_family", "")).casefold()
    fact = event.get("fact")
    predicate = str(fact.get("predicate", "")) if isinstance(fact, Mapping) else ""
    phrases = [
        "no other",
        "nothing else changed",
        "complete picture",
        "supplied event",
        "benchmark",
        "the prompt",
        "provided text",
        "how should i phrase",
        "supportable statement",
        "write the entry",
        "do not add",
        "same fact delivered",
        "new evidence",
        "audit trail",
        "only thing on file",
        "canonical record",
    ]
    if event_family == "delayed_preference_probe":
        phrases.extend(("no conflict", "no conflicting", "everything matches", "still the same"))
    if event_family == "same_entity_hard_negative":
        phrases.extend(
            (
                "treated as a private note",
                "attached to",
                "belongs to",
                "already associated",
                "reflects text",
                "would conflict",
                "correction",
                "could become",
            )
        )
    if operation == "supersede":
        phrases.extend(
            (
                "fully replacing",
                "delete the old",
                "erase the old",
                "going forward",
                "from now on",
                "indefinitely",
            )
        )
    if "duplicate" in operation:
        phrases.extend(("last cycle", "deduplicat", "merged", "single canonical"))
        phrases.extend(("on your profile", "already saved", "in storage", "noted"))
    if predicate == "PRIVATE_NOTE":
        phrases.extend(("favorite", "preference", "recommend", "dish"))
    if (
        event.get("supersedes")
        or event.get("corrects")
        or event.get("transitions_from")
    ) and operation != "supersede":
        phrases.extend(("stays exactly as it is", "remains unchanged", "keep it unchanged"))
    if operation == "backdated_correction":
        phrases.extend(("after spring", "later entries", "later records"))
    if "retract" in operation:
        phrases.extend(("nothing remains to be recovered", "permanently deleted", "completely erased"))
    return tuple(phrases)


def _speaker_contract(
    event: Mapping[str, Any], dialogue_subject: str
) -> tuple[str, str, list[str]]:
    """Return a speaker instruction and lexical evidence for source authority."""
    fact = event.get("fact")
    qualifiers = fact.get("qualifiers", {}) if isinstance(fact, Mapping) else {}
    authority = str(qualifiers.get("source_authority", "inferred"))
    if authority == "direct_user":
        return (
            dialogue_subject,
            f"The user is {dialogue_subject} and states their own information naturally in first person.",
            [],
        )
    if authority == "inferred":
        return (
            "observer",
            f"The user asks about third-party evidence concerning {dialogue_subject}; the assistant "
            "must label it tentative or inferred, and the user must not personally confirm it.",
            ["inferred", "tentative", "appears", "reported"],
        )
    raise ValueError(f"unsupported source authority {authority!r}")


def validate_generation_response(
    content: str,
    expected_events: Sequence[Mapping[str, Any]],
    *,
    turn_pairs_per_event: int,
    minimum_words_per_turn: int,
) -> dict[str, Any]:
    """Validate one history batch for schema, order, completeness, and latent-label leakage."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("provider returned empty content")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"provider returned malformed JSON: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"events"} or not isinstance(payload["events"], list):
        raise ValueError("generation response must be an object containing only an events array")
    expected_ids = [str(event["event_id"]) for event in expected_events]
    actual_ids = [event.get("event_id") for event in payload["events"] if isinstance(event, dict)]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("generation response contains duplicate event IDs")
    missing = sorted(set(expected_ids) - set(actual_ids))
    extra = sorted(set(actual_ids) - set(expected_ids))
    if missing:
        raise ValueError(f"generation response has missing event IDs: {missing}")
    if extra:
        raise ValueError(f"generation response has extra event IDs: {extra}")
    if actual_ids != expected_ids:
        raise ValueError("generation response changed event order")
    for index, event in enumerate(payload["events"]):
        if not isinstance(event, dict) or set(event) != {"event_id", "turns"}:
            raise ValueError(f"generated event at index {index} has malformed fields")
        turns = event["turns"]
        expected_turn_count = 2 * turn_pairs_per_event
        if not isinstance(turns, list) or len(turns) != expected_turn_count:
            raise ValueError(
                f"generated event {event['event_id']} must contain exactly "
                f"{expected_turn_count} turns"
            )
        roles = []
        visible_parts = []
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict) or set(turn) != {"role", "content"}:
                raise ValueError(f"generated event {event['event_id']} turn {turn_index} is malformed")
            role = turn["role"]
            if role not in {"user", "assistant"}:
                raise ValueError(f"generated event {event['event_id']} has unsupported role {role!r}")
            text = _nonempty(turn["content"], f"generated event {event['event_id']} content")
            word_count = len(text.split())
            if word_count < minimum_words_per_turn:
                raise ValueError(
                    f"generated event {event['event_id']} turn {turn_index} must contain at least "
                    f"{minimum_words_per_turn} words; found {word_count}"
                )
            if _LATENT_LABEL.search(text) or any(identifier in text for identifier in expected_ids):
                raise ValueError(f"generated event {event['event_id']} leaked a latent benchmark label")
            roles.append(role)
            visible_parts.append(text)
        if roles[0] != "user" or roles[-1] != "assistant" or any(left == right for left, right in zip(roles, roles[1:])):
            raise ValueError(f"generated event {event['event_id']} is not alternating user/assistant dialogue")
        visible_text = " ".join(visible_parts).casefold()
        for required_value in expected_events[index].get("required_surface_values", ()):
            if str(required_value).casefold() not in visible_text:
                raise ValueError(
                    f"generated event {event['event_id']} omitted required surface value "
                    f"{required_value!r}"
                )
        for forbidden_phrase in expected_events[index].get("forbidden_surface_phrases", ()):
            if str(forbidden_phrase).casefold() in visible_text:
                raise ValueError(
                    f"generated event {event['event_id']} asserted forbidden phrase "
                    f"{forbidden_phrase!r}"
                )
        authority_markers = expected_events[index].get("authority_markers", ())
        if authority_markers and not any(
            str(marker).casefold() in visible_text for marker in authority_markers
        ):
            raise ValueError(
                f"generated event {event['event_id']} omitted its inferred-authority marker"
            )
        semantic_markers = expected_events[index].get("semantic_markers", ())
        pronoun_match = _UNSUPPORTED_GENDERED_PRONOUN.search(visible_text)
        if pronoun_match:
            raise ValueError(
                f"generated event {event['event_id']} introduced unsupported gendered pronoun "
                f"{pronoun_match.group(0)!r}"
            )
        recordkeeping_count = len(_RECORDKEEPING_LANGUAGE.findall(visible_text))
        if recordkeeping_count > 3:
            raise ValueError(
                f"generated event {event['event_id']} used {recordkeeping_count} recordkeeping "
                "terms; at most 3 are allowed"
            )
        positive_marker = False
        for marker in semantic_markers:
            for match in re.finditer(re.escape(str(marker).casefold()), visible_text):
                prefix = visible_text[max(0, match.start() - 24) : match.start()]
                if not re.search(r"(?:do not|don't|never|not)\s+(?:\w+\s+){0,2}$", prefix):
                    positive_marker = True
                    break
            if positive_marker:
                break
        if semantic_markers and not positive_marker:
            raise ValueError(
                f"generated event {event['event_id']} inverted or omitted its operation semantics"
            )
    return payload


def _render_dialogue(turns: Sequence[Mapping[str, str]]) -> str:
    """Serialize generated roles into the existing event model_text seam."""
    return "\n".join(f"{turn['role'].title()}: {turn['content']}" for turn in turns)


def _request_parameters(config: GenerationConfig) -> dict[str, Any]:
    """Return auditable non-secret generation parameters."""
    return {
        "model": config.model,
        "timeout_seconds": config.timeout_seconds,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "turn_pairs_per_event": config.turn_pairs_per_event,
        "minimum_words_per_turn": config.minimum_words_per_turn,
        "events_per_request": config.events_per_request,
        "enable_thinking": config.enable_thinking,
        "temperature": 0,
    }


def _usage_token_totals(raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize provider-reported visible and reasoning completion usage."""
    completion_tokens = 0
    reasoning_tokens = 0
    for row in raw_rows:
        usage = row.get("usage", {})
        if not isinstance(usage, Mapping):
            continue
        completion_tokens += int(usage.get("completion_tokens") or 0)
        row_reasoning_tokens = int(usage.get("reasoning_tokens") or 0)
        details = usage.get("completion_tokens_details", {})
        if isinstance(details, Mapping):
            row_reasoning_tokens = max(
                row_reasoning_tokens,
                int(details.get("reasoning_tokens") or 0),
            )
        reasoning_tokens += row_reasoning_tokens
    return {
        "provider_reported_completion_tokens": completion_tokens,
        "provider_reported_reasoning_tokens": reasoning_tokens,
        "provider_reported_visible_tokens": max(0, completion_tokens - reasoning_tokens),
        "requested_non_thinking_honored": reasoning_tokens == 0,
    }


def _surface_fact_object(
    fact: Mapping[str, Any], value_map: Mapping[str, str]
) -> str | None:
    """Return a natural object phrase suitable for relation-fidelity checks."""
    raw_object = fact.get("object")
    if not isinstance(raw_object, str) or not raw_object.strip():
        return None
    if raw_object in value_map:
        return value_map[raw_object]
    surface = re.sub(r"\s+\d+$", "", raw_object).replace("_", " ").strip()
    if surface == "quiet studio":
        return "quiet private studio"
    return surface


def _required_surface_values(
    event: Mapping[str, Any],
    facts_by_id: Mapping[str, Mapping[str, Any]],
    value_map: Mapping[str, str],
) -> list[str]:
    """Return current and relation-linked values that generated dialogue must preserve."""
    related_ids: list[str] = []
    for key in (
        "resolves",
        "supersedes",
        "corrects",
        "transitions_from",
        "conflicts_with",
    ):
        raw = event.get(key, ())
        if isinstance(raw, str):
            related_ids.append(raw)
        elif isinstance(raw, Sequence):
            related_ids.extend(str(item) for item in raw)
    for key in ("retracts", "duplicate_of"):
        raw = event.get(key)
        if isinstance(raw, str):
            related_ids.append(raw)
    facts = [facts_by_id[fact_id] for fact_id in related_ids if fact_id in facts_by_id]
    current_fact = event.get("fact")
    if isinstance(current_fact, Mapping):
        facts.append(current_fact)
    values = []
    for fact in facts:
        value = _surface_fact_object(fact, value_map)
        if value and value not in values:
            values.append(value)
    return values


def generate_persona_conversations(
    output_dir: Path,
    config: GenerationConfig,
    client: CompletionClient,
) -> dict[str, Any]:
    """Generate five compatible artifacts with Kimi-authored model-visible event dialogue."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "generation_manifest.json"
    raw_path = output_dir / "raw_responses.jsonl"
    requests_path = output_dir / "requests.jsonl"
    manifest: dict[str, Any] = {
        "status": "running",
        "completion_status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "generator_role": "Kimi K3 generation-only surface realization",
        "prompt_schema_version": config.prompt_schema_version,
        "request_parameters": _request_parameters(config),
        "provider_identity_assurance": "endpoint-self-reported; artifact hashes are cryptographic",
        "source_generation": {
            "history_count": config.history_count,
            "condition": config.condition,
            "hardness_profile": config.hardness_profile,
            "split_counts": list(config.split_counts),
        },
        "controls": {
            "full_structured_memory": "event-relation replay oracle",
        },
        "artifact_roles": {
            "dialogue.jsonl": "model_input",
            "events.jsonl": "latent_structure",
            "source_documents.jsonl": "source_evidence",
            "facts.jsonl": "structured_source_facts",
            "queries.jsonl": "supervision_only",
            "examples.jsonl": "supervision_only",
            "candidate_updates.jsonl": "supervision_only",
            "requests.jsonl": "generation_provenance",
            "raw_responses.jsonl": "generation_provenance",
        },
        "hashing": {
            "algorithm": "sha256",
            "prompt_preimage": "json.dumps(messages, ensure_ascii=True, sort_keys=True).encode('utf-8')",
            "artifact_preimage": "raw_file_bytes",
        },
        "reference_contract": {
            "event_id": "event_identifier",
            "fact.fact_id": "fact_identifier",
            "supersedes": "fact_identifier",
            "transitions_from": "fact_identifier",
            "corrects": "fact_identifier",
            "resolves": "fact_identifier_array",
            "retracts": "fact_identifier",
            "duplicate_of": "fact_identifier",
            "conflicts_with": "fact_identifier",
            "replay_order": "events.sequence_index",
        },
        "validation_contract": {
            "recordkeeping_term_pattern": _RECORDKEEPING_LANGUAGE.pattern,
            "max_recordkeeping_terms_per_event": 3,
            "lineage_retraction": "retracts_lineage=true propagates through duplicate_of closure",
        },
    }
    _write_json(manifest_path, manifest)
    raw_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    _write_jsonl(raw_path, raw_rows)
    _write_jsonl(requests_path, request_rows)
    try:
        generate_dataset(
            output_dir,
            history_count=config.history_count,
            condition=config.condition,
            split_counts=config.split_counts,
            hardness_profile=config.hardness_profile,
        )
        events = _read_jsonl(output_dir / "events.jsonl")
        queries = _read_jsonl(output_dir / "queries.jsonl")
        value_map, subject_map, scope_map = _surface_maps(events)
        facts_by_id = {
            str(event["fact"]["fact_id"]): event["fact"]
            for event in events
            if isinstance(event.get("fact"), Mapping) and event["fact"].get("fact_id")
        }
        replacements = {**value_map, **subject_map, **scope_map}
        by_history: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            by_history.setdefault(str(event["history_id"]), []).append(event)
        response_hashes = []
        model_identities = set()
        for history_id, history_events in sorted(by_history.items()):
            expected = []
            for event in history_events:
                latent_object = str(event["fact"].get("object", ""))
                support_text = str(event["fact"]["support_text"])
                if latent_object in value_map:
                    support_text = re.sub(r"\bvalue-\d{3}-[a-z]\b", latent_object, support_text)
                surface_text = _replace_latent(support_text, replacements)
                required_values = _required_surface_values(
                    event, facts_by_id, value_map
                )
                dialogue_subject = replacements.get(
                    str(event["fact"].get("subject")), "the account owner"
                )
                dialogue_speaker, speaker_instruction, authority_markers = _speaker_contract(
                    event, dialogue_subject
                )
                expected.append(
                    {
                        "event_id": event["event_id"],
                        "dialogue_subject": dialogue_subject,
                        "dialogue_speaker": dialogue_speaker,
                        "speaker_instruction": speaker_instruction,
                        "authority_markers": authority_markers,
                        "surface_text": surface_text,
                        "required_surface_values": required_values,
                        "semantic_markers": list(_semantic_markers(event)),
                        "semantic_instruction": _semantic_instruction(
                            event, required_values
                        ),
                        "forbidden_surface_phrases": list(
                            _forbidden_surface_phrases(event)
                        ),
                    }
                )
            for batch_index, offset in enumerate(
                range(0, len(expected), config.events_per_request)
            ):
                batch_expected = expected[offset : offset + config.events_per_request]
                batch_events = history_events[offset : offset + config.events_per_request]
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Render each supplied event as an ordinary, realistic memory interaction. "
                            "The user should discuss their situation directly, never ask how to word, "
                            "annotate, summarize, or record a benchmark statement. "
                            "Follow each speaker_instruction exactly so direct statements and inferred "
                            "third-party evidence remain distinct. "
                            "Use names, first-person pronouns, or singular they; never invent gendered pronouns. "
                            "Use extra turns for practical context or tradeoffs, not repetitive "
                            "paraphrase or recordkeeping; use at most three recordkeeping terms per event. "
                            "Keep the supplied event order and meaning exactly. Return strict JSON only. "
                            "Return exactly one top-level key named events. Each events item must contain "
                            "exactly the supplied event_id and a turns array of alternating role/content "
                            f"objects beginning with user and ending with assistant, with exactly "
                            f"{config.turn_pairs_per_event} user-assistant pairs per event and at least "
                            f"{config.minimum_words_per_turn} words per turn. Mention every supplied "
                            "required_surface_value naturally and preserve conflict, correction, and "
                            "resolution relationships explicitly. Follow each semantic_instruction "
                            "and do not use any forbidden_surface_phrases. Never mention prompts, "
                            "supplied events, benchmarks, generation, or source instructions. Preserve event_id only "
                            "as metadata; do not expose IDs or benchmark labels inside dialogue content. "
                            "Required shape: {\"events\":[{\"event_id\":\"the supplied ID\","
                            "\"turns\":[{\"role\":\"user\",\"content\":\"...\"},"
                            "{\"role\":\"assistant\",\"content\":\"...\"}]}]}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "schema_version": config.prompt_schema_version,
                                "turn_pairs_per_event": config.turn_pairs_per_event,
                                "minimum_words_per_turn": config.minimum_words_per_turn,
                                "events": batch_expected,
                            },
                            sort_keys=True,
                        ),
                    },
                ]
                prompt_sha256 = hashlib.sha256(
                    json.dumps(messages, ensure_ascii=True, sort_keys=True).encode("utf-8")
                ).hexdigest()
                request_rows.append(
                    {
                        "history_id": history_id,
                        "batch_index": batch_index,
                        "event_count": len(batch_expected),
                        "prompt_sha256": prompt_sha256,
                        "messages": messages,
                    }
                )
                _write_jsonl(requests_path, request_rows)
                response = client.complete(
                    messages=messages,
                    model=config.model,
                    timeout=config.timeout_seconds,
                    seed=config.seed,
                    max_tokens=config.max_tokens,
                )
                response_hash = hashlib.sha256(response.content.encode("utf-8")).hexdigest()
                raw_rows.append(
                    {
                        "history_id": history_id,
                        "batch_index": batch_index,
                        "model": response.model,
                        "finish_reason": response.finish_reason,
                        "usage": dict(response.usage),
                        "response_sha256": response_hash,
                        "content": response.content,
                    }
                )
                _write_jsonl(raw_path, raw_rows)
                if response.finish_reason != "stop":
                    raise ValueError(
                        f"generation for {history_id} batch {batch_index} ended with "
                        f"non-stop finish reason {response.finish_reason!r}"
                    )
                parsed = validate_generation_response(
                    response.content,
                    batch_expected,
                    turn_pairs_per_event=config.turn_pairs_per_event,
                    minimum_words_per_turn=config.minimum_words_per_turn,
                )
                for event, generated in zip(batch_events, parsed["events"]):
                    event["model_text"] = _render_dialogue(generated["turns"])
                    event["dialogue_subject"] = replacements.get(
                        str(event["fact"].get("subject")), "the account owner"
                    )
                    event["dialogue_speaker"] = next(
                        str(expected_event["dialogue_speaker"])
                        for expected_event in batch_expected
                        if expected_event["event_id"] == event["event_id"]
                    )
                    event["surface_object"] = value_map.get(
                        str(event["fact"].get("object")),
                        str(event["fact"].get("object", "")),
                    )
                response_hashes.append(response_hash)
                model_identities.add(response.model)
        for query in queries:
            query["surface_query_text"], query["surface_gold"] = _surface_query(
                query, replacements, value_map
            )
            query["query_text"] = query["surface_query_text"]
        _write_jsonl(output_dir / "events.jsonl", events)
        _write_jsonl(output_dir / "queries.jsonl", queries)
        conversation_ids = {
            history_id: f"conversation-{index:06d}"
            for index, history_id in enumerate(sorted(by_history), start=1)
        }
        event_indices: dict[str, int] = {}
        dialogue_rows = []
        for event in events:
            history_id = str(event["history_id"])
            event_index = event_indices.get(history_id, 0)
            dialogue_rows.append(
                {
                    "conversation_id": conversation_ids[history_id],
                    "event_index": event_index,
                    "speaker": event["dialogue_speaker"],
                    "subject": event["dialogue_subject"],
                    "text": event["model_text"],
                }
            )
            event_indices[history_id] = event_index + 1
        dialogue_path = output_dir / "dialogue.jsonl"
        _write_jsonl(dialogue_path, dialogue_rows)
        artifact_names = (
            *SOURCE_ARTIFACTS,
            dialogue_path.name,
            raw_path.name,
            requests_path.name,
        )
        manifest.update(
            {
                "status": "completed",
                "completion_status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "model_identity": sorted(model_identities)[0] if len(model_identities) == 1 else sorted(model_identities),
                "effective_generation": _usage_token_totals(raw_rows),
                "response_sha256": response_hashes,
                "prompt_sha256": [row["prompt_sha256"] for row in request_rows],
                "artifact_sha256": {
                    name: _sha256(output_dir / name) for name in artifact_names
                },
            }
        )
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        if not raw_path.exists():
            _write_jsonl(raw_path, raw_rows)
        if not requests_path.exists():
            _write_jsonl(requests_path, request_rows)
        manifest.update(
            {
                "status": "failed",
                "completion_status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)},
                "artifact_sha256": {
                    name: _sha256(output_dir / name)
                    for name in (*SOURCE_ARTIFACTS, raw_path.name, requests_path.name)
                    if (output_dir / name).exists()
                },
            }
        )
        _write_json(manifest_path, manifest)
        raise


def _strict_object(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    """Require an object with exactly the expected keys."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"{name} keys mismatch: missing={missing}, unknown={unknown}")
    return value


def load_generation_config(path: Path) -> GenerationConfig:
    """Load strict generation config and resolve provider identity from named env vars."""
    try:
        root = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load generation config {path}: {error}") from error
    if not isinstance(root, Mapping) or "generation" not in root:
        raise ValueError("config must contain a generation object")
    raw = _strict_object(
        root["generation"],
        {
            "endpoint_env", "api_key_env", "model_env", "timeout_seconds", "seed",
            "max_tokens", "history_count", "condition", "hardness_profile",
            "split_counts", "prompt_schema_version", "turn_pairs_per_event",
            "minimum_words_per_turn", "events_per_request",
            "enable_thinking",
        },
        "generation",
    )
    endpoint_env = _nonempty(raw["endpoint_env"], "generation.endpoint_env")
    api_key_env = _nonempty(raw["api_key_env"], "generation.api_key_env")
    model_env = _nonempty(raw["model_env"], "generation.model_env")
    missing_env = [name for name in (endpoint_env, api_key_env, model_env) if not os.environ.get(name)]
    if missing_env:
        raise ValueError(f"required generation environment variables are unset: {missing_env}")
    split_counts = raw["split_counts"]
    if not isinstance(split_counts, list):
        raise ValueError("generation.split_counts must be an array")
    return GenerationConfig(
        endpoint=os.environ[endpoint_env],
        api_key=os.environ[api_key_env],
        model=os.environ[model_env],
        timeout_seconds=_positive_number(raw["timeout_seconds"], "generation.timeout_seconds"),
        seed=raw["seed"],
        max_tokens=raw["max_tokens"],
        history_count=raw["history_count"],
        condition=raw["condition"],
        hardness_profile=raw["hardness_profile"],
        split_counts=tuple(split_counts),
        prompt_schema_version=raw["prompt_schema_version"],
        turn_pairs_per_event=raw["turn_pairs_per_event"],
        minimum_words_per_turn=raw["minimum_words_per_turn"],
        events_per_request=raw["events_per_request"],
        enable_thinking=raw["enable_thinking"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured Kimi-only conversation surface generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_generation_config(args.config)
    generate_persona_conversations(
        args.output_dir,
        config,
        OpenAICompletionClient(
            config.endpoint,
            config.api_key,
            config.timeout_seconds,
            json_mode=True,
            enable_thinking=config.enable_thinking,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
