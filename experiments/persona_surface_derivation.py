"""Generate and authenticate pair-conditioned Kimi A/B surfaces from one parent."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from experiments.persona_conversation_generator import (
    CompletionClient,
    OpenAICompletionClient,
    SOURCE_ARTIFACTS,
    _forbidden_surface_phrases,
    _render_dialogue,
    _replace_latent,
    _required_surface_values,
    _semantic_instruction,
    _semantic_markers,
    _semantic_polarity,
    _speaker_contract,
    _surface_maps,
    _usage_token_totals,
    build_generation_messages,
    validate_generation_response,
)


DERIVATION_METHOD = "kimi-pair-conditioned-surface-and-dialogue.v2"
PAIR_GATE_VERSION = "persona-surface-pair-gate.v2"
PROMPT_SCHEMA_VERSION = "persona-pair-conditioned-surface.v3"
SURFACE_ASSIGNMENT_SEEDS = {"A": 137, "B": 911}
EXPECTED_MODEL = "kimi-k3"
EXPECTED_EVENT_COUNT = 416
MAX_PHRASE_WORDS = 5
MAX_PHRASE_CHARACTERS = 64

# These are prompt-only naturalness guides, not hidden validation rules. A and
# B receive different real-world pools; only B receives A's flat avoid-list.
MAPPING_LEXICAL_DIRECTIVES = (
    (
        "Use established botanical, culinary, and traditional options: named tea or herbal "
        "blends, familiar seating positions, recognizable dishes, ordinary delivery windows, "
        "standard receipt formats, and plausible physical keepsakes."
    ),
    (
        "Use established regional, modern, and practical options: recognizable teas, useful "
        "seating positions or types, familiar dishes, real delivery methods or times, standard "
        "receipt formats, and plausible physical keepsakes."
    ),
    (
        "Use conventional cafe, household, and culinary options that a person could naturally "
        "request, prefer, receive, eat, or keep; every phrase must name a real-world value."
    ),
    (
        "Use conventional contemporary service options and recognizable regional foods or teas; "
        "prefer plain practical names that people use in ordinary conversation."
    ),
)

CATEGORY_SOURCE_PHRASES = {
    "tea": ("cedar tea", "mint tea", "oolong tea"),
    "seating": ("window seating", "aisle seating"),
    "food": ("vegetable ramen", "mushroom risotto"),
    "delivery": ("evening delivery", "weekend delivery"),
    "receipt": ("digital receipts",),
    "private_lineage": ("cedar glass",),
}

_VISIBLE_EVENT_FIELDS = {"model_text", "surface_object"}
_VISIBLE_QUERY_FIELDS = {"query_text", "surface_query_text", "surface_gold"}
_REBUILT_ARTIFACTS = {
    "events.jsonl",
    "dialogue.jsonl",
    "queries.jsonl",
    "requests.jsonl",
    "raw_responses.jsonl",
}
_LATENT_ID = re.compile(
    r"\b(?:history|subject|value|scope|event|fact|query|candidate|replacement)-\d+(?:-[a-z][\w-]*)?\b"
    r"|\b(?:event_id|fact_id|query_id)\s*=",
    re.IGNORECASE,
)
_NORMALIZED_TOKEN = re.compile(r"[a-z0-9]+")
_FOOD_NOUNS = {
    "barley",
    "chowder",
    "couscous",
    "curry",
    "dumplings",
    "enchiladas",
    "farro",
    "fritters",
    "gnocchi",
    "lasagna",
    "noodles",
    "orzo",
    "paella",
    "pasta",
    "pilaf",
    "polenta",
    "ramen",
    "ravioli",
    "rice",
    "risotto",
    "soba",
    "soup",
    "stew",
    "tacos",
    "tagine",
    "tofu",
    "udon",
}
_PRIVATE_LINEAGE_NOUNS = {
    "badge",
    "bookmark",
    "button",
    "charm",
    "compass",
    "emblem",
    "feather",
    "glass",
    "keepsake",
    "key",
    "lantern",
    "locket",
    "marker",
    "medallion",
    "pebble",
    "pendant",
    "ribbon",
    "seal",
    "token",
}
_PUBLIC_PREFERENCE_CATEGORIES = {"tea", "seating", "food", "delivery", "receipt"}
_CROSS_CATEGORY_MODIFIERS = {
    "amethyst",
    "azurite",
    "basalt",
    "cotton",
    "denim",
    "emerald",
    "flannel",
    "granite",
    "jade",
    "linen",
    "marble",
    "nylon",
    "obsidian",
    "opal",
    "organza",
    "polyester",
    "quartz",
    "ruby",
    "sapphire",
    "satin",
    "tweed",
    "turquoise",
    "velvet",
    "wool",
}
_CONVENTIONAL_CROSS_CATEGORY_PHRASES = {
    ("seating", "linen seating"),
    ("seating", "velvet seating"),
    ("tea", "cotton candy tea"),
}
_TRANSIENT_PROVIDER_ERROR_TYPES = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    TimeoutError,
    ConnectionError,
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionRefusedError,
    ConnectionResetError,
)
_TRANSIENT_PROVIDER_ERROR_NAMES = {
    error_type.__name__ for error_type in _TRANSIENT_PROVIDER_ERROR_TYPES
}
_RESPONSE_METADATA_FIELDS = (
    "stage",
    "assignment",
    "request_index",
    "mapping_request_index",
    "history_ids",
    "event_ids",
    "attempt_index",
    "seed",
    "requested_model",
    "max_tokens",
    "enable_thinking",
)


class CacheIntegrityError(ValueError):
    """Persisted request or response provenance does not match the current call."""


GATE_ASSURANCES = {
    "proved": [
        "parent_and_sibling_hash_binding",
        "latent_causal_preservation",
        "cross_assignment_surface_disjointness",
        "a_to_b_flat_target_conditioning",
        "fresh_kimi_response_provenance",
        "dialogue_semantic_validation",
    ],
    "deferred": {
        "checkpoint_identity": "scheduler_with_configured_tokenizer_before_qwen"
    },
    "provider_identity": "endpoint-self-reported",
    "artifact_integrity": "sha256-verified",
}
PRESERVATION_SCOPE = {
    "pair_gate": "latent_causal_structure",
    "scheduler": "exact_checkpoint_indices_with_configured_tokenizer",
}


@dataclass(frozen=True)
class SurfacePairConfig:
    """Validated provider and realization controls for paired generation."""

    parent_dir: Path
    output_a: Path
    output_b: Path
    gate_path: Path
    endpoint: str
    api_key: str
    model: str
    timeout_seconds: float
    mapping_max_tokens: int
    dialogue_max_tokens: int
    mapping_histories_per_request: int
    dialogue_assignment_concurrency: int
    events_per_request: int
    turn_pairs_per_event: int
    minimum_words_per_turn: int
    max_validation_attempts: int
    resume_existing: bool
    enable_thinking: bool

    def __post_init__(self) -> None:
        """Reject incomplete paths, provider identity drift, and invalid request bounds."""
        for name in ("endpoint", "api_key", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        validate_kimi_model_identity(self.model, self.model)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        for name in (
            "mapping_max_tokens",
            "dialogue_max_tokens",
            "mapping_histories_per_request",
            "events_per_request",
            "turn_pairs_per_event",
            "minimum_words_per_turn",
            "max_validation_attempts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean")
        if not isinstance(self.resume_existing, bool):
            raise ValueError("resume_existing must be a boolean")
        if (
            isinstance(self.dialogue_assignment_concurrency, bool)
            or not isinstance(self.dialogue_assignment_concurrency, int)
            or self.dialogue_assignment_concurrency not in {1, 2}
        ):
            raise ValueError("dialogue_assignment_concurrency must be 1 or 2")


def _sha256_bytes(payload: bytes) -> str:
    """Return a byte payload's SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def _stable_hash(value: Any) -> str:
    """Hash one JSON value with canonical serialization."""
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _transient_retry_instruction(provider_error: Mapping[str, str]) -> str:
    """Return deterministic retry feedback derived only from persisted error data."""
    return (
        "The prior provider call failed transiently with "
        f"{provider_error['type']}: {provider_error['message']}. "
        "Retry with the changed seed and return a concise complete response."
    )


def _provider_error_row(
    request_row: Mapping[str, Any],
    provider_error: Mapping[str, str],
    *,
    reconciled_from_failed_manifest: bool = False,
) -> dict[str, Any]:
    """Build one canonical response-side provenance record for a provider failure."""
    row = {
        field: request_row[field]
        for field in _RESPONSE_METADATA_FIELDS
        if field in request_row
    }
    row.update(
        {
            "model": None,
            "returned_model": None,
            "accepted": False,
            "provider_error": dict(provider_error),
            **(
                {"superseded": True}
                if request_row.get("stage") == "mapping"
                else {}
            ),
            **(
                {"reconciled_from_failed_manifest": True}
                if reconciled_from_failed_manifest
                else {}
            ),
        }
    )
    row["error_record_sha256"] = _stable_hash(row)
    return row


def _validate_provider_error_row(row: Mapping[str, Any]) -> None:
    """Authenticate one transient provider-error record and reject response masquerading."""
    provider_error = row.get("provider_error")
    if (
        not isinstance(provider_error, Mapping)
        or set(provider_error) != {"type", "message"}
        or not isinstance(provider_error.get("type"), str)
        or provider_error.get("type") not in _TRANSIENT_PROVIDER_ERROR_NAMES
        or not isinstance(provider_error.get("message"), str)
    ):
        raise ValueError("provider error row has an invalid transient error classification")
    if row.get("accepted") is not False:
        raise ValueError("provider error row cannot be accepted")
    if row.get("model") is not None or row.get("returned_model") is not None:
        raise ValueError("provider error row cannot claim a returned model")
    if any(
        field in row
        for field in ("content", "response_sha256", "finish_reason", "usage")
    ):
        raise ValueError("provider error row cannot contain fabricated response data")
    if row.get("stage") == "mapping" and row.get("superseded") is not True:
        raise ValueError("mapping provider error row must be superseded")
    if "reconciled_from_failed_manifest" in row and row.get(
        "reconciled_from_failed_manifest"
    ) is not True:
        raise ValueError("provider error reconciliation marker must be true")
    recorded_hash = row.get("error_record_sha256")
    canonical = {key: value for key, value in row.items() if key != "error_record_sha256"}
    if recorded_hash != _stable_hash(canonical):
        raise ValueError("provider error record hash mismatch")


def _raw_provenance_hash(row: Mapping[str, Any]) -> str:
    """Return the authenticated hash for a model response or provider-error record."""
    if "provider_error" in row:
        _validate_provider_error_row(row)
        return str(row["error_record_sha256"])
    response_hash = row.get("response_sha256")
    if not isinstance(response_hash, str):
        raise ValueError("raw response row lacks response_sha256")
    return response_hash


def _generation_controls(config: SurfacePairConfig) -> dict[str, Any]:
    """Return every effective non-secret control that defines generated artifacts."""
    return {
        "endpoint_sha256": _sha256_bytes(config.endpoint.encode("utf-8")),
        "requested_model": config.model,
        "timeout_seconds": config.timeout_seconds,
        "mapping_max_tokens": config.mapping_max_tokens,
        "dialogue_max_tokens": config.dialogue_max_tokens,
        "mapping_histories_per_request": config.mapping_histories_per_request,
        "dialogue_assignment_concurrency": config.dialogue_assignment_concurrency,
        "events_per_request": config.events_per_request,
        "turn_pairs_per_event": config.turn_pairs_per_event,
        "minimum_words_per_turn": config.minimum_words_per_turn,
        "max_validation_attempts": config.max_validation_attempts,
        "enable_thinking": config.enable_thinking,
        "temperature": 0,
        "json_mode": True,
        "surface_assignment_seeds": dict(SURFACE_ASSIGNMENT_SEEDS),
        "derivation_method": DERIVATION_METHOD,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
    }


def _request_parameters(config: SurfacePairConfig) -> dict[str, Any]:
    """Return the auditable request-shaping subset of generation controls."""
    return {
        "model": config.model,
        "timeout_seconds": config.timeout_seconds,
        "mapping_max_tokens": config.mapping_max_tokens,
        "dialogue_max_tokens": config.dialogue_max_tokens,
        "mapping_histories_per_request": config.mapping_histories_per_request,
        "dialogue_assignment_concurrency": config.dialogue_assignment_concurrency,
        "events_per_request": config.events_per_request,
        "turn_pairs_per_event": config.turn_pairs_per_event,
        "minimum_words_per_turn": config.minimum_words_per_turn,
        "max_validation_attempts": config.max_validation_attempts,
        "enable_thinking": config.enable_thinking,
        "temperature": 0,
        "json_mode": True,
    }


def _json_bytes(value: Any) -> bytes:
    """Serialize stable human-readable JSON."""
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize stable object-valued JSONL in supplied order."""
    return "".join(
        json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace one file through a same-directory fsynced temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_jsonl(payload: bytes, name: str) -> list[dict[str, Any]]:
    """Load object-valued UTF-8 JSONL with line-specific failures."""
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"artifact {name} is not UTF-8: {error}") from error
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {name} line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{name} line {line_number} must be an object")
        rows.append(row)
    return rows


def _normalize_phrase(value: str) -> str:
    """Normalize a phrase for equality and containment collision checks."""
    return " ".join(_NORMALIZED_TOKEN.findall(value.casefold()))


def validate_kimi_model_identity(requested: str, returned: str) -> None:
    """Require exact requested and returned Kimi K3 identities."""
    if requested != EXPECTED_MODEL:
        raise ValueError(f"requested model must be exactly {EXPECTED_MODEL}, got {requested!r}")
    if returned != EXPECTED_MODEL:
        raise ValueError(f"returned model must be exactly {EXPECTED_MODEL}, got {returned!r}")


def expected_surface_mapping_rows(history_ids: Sequence[str]) -> list[dict[str, str]]:
    """Return the exact ordered mapping keys required from one mapping request."""
    return [
        {
            "history_id": history_id,
            "category": category,
            "source_phrase": source_phrase,
        }
        for history_id in sorted(history_ids)
        for category, source_phrases in CATEGORY_SOURCE_PHRASES.items()
        for source_phrase in source_phrases
    ]


def _validate_category_shape(category: str, phrase: str) -> None:
    """Require a short lexical phrase that remains visibly in its source category."""
    words = phrase.split()
    if len(words) < 2 or len(words) > MAX_PHRASE_WORDS or len(phrase) > MAX_PHRASE_CHARACTERS:
        raise ValueError(f"target phrase length is outside bounds: {phrase!r}")
    normalized = _normalize_phrase(phrase)
    if len(normalized.split()) < 2:
        raise ValueError(f"target phrase has invalid {category} lexical shape: {phrase!r}")
    suffixes = {
        "tea": " tea",
        "seating": " seating",
        "delivery": " delivery",
        "receipt": " receipts",
    }
    suffix = suffixes.get(category)
    if suffix is not None and not normalized.endswith(suffix):
        raise ValueError(f"target phrase has invalid {category} category shape: {phrase!r}")
    forbidden_suffixes = tuple(suffixes.values())
    if category in {"food", "private_lineage"} and normalized.endswith(forbidden_suffixes):
        raise ValueError(f"target phrase has invalid {category} category shape: {phrase!r}")
    final_word = normalized.rsplit(" ", 1)[-1]
    if category == "food" and final_word not in _FOOD_NOUNS:
        raise ValueError(f"target phrase has invalid food category shape: {phrase!r}")
    if category == "private_lineage" and final_word not in _PRIVATE_LINEAGE_NOUNS:
        raise ValueError(f"target phrase has invalid private_lineage category shape: {phrase!r}")


def _contains_token_sequence(container: str, candidate: str) -> bool:
    """Return whether normalized candidate tokens occur contiguously in container."""
    container_tokens = container.split()
    candidate_tokens = candidate.split()
    if not candidate_tokens or len(candidate_tokens) > len(container_tokens):
        return False
    return any(
        container_tokens[index : index + len(candidate_tokens)] == candidate_tokens
        for index in range(len(container_tokens) - len(candidate_tokens) + 1)
    )


def _parent_phrase_collision(target: str, parent_phrases: set[str]) -> str | None:
    """Return the first parent phrase with equal or bidirectional token containment."""
    normalized_target = _normalize_phrase(target)
    for parent in sorted(parent_phrases):
        normalized_parent = _normalize_phrase(parent)
        if _contains_token_sequence(normalized_target, normalized_parent) or _contains_token_sequence(
            normalized_parent, normalized_target
        ):
            return parent
    return None


def _forbidden_target_collision(
    target: str, forbidden_targets: Sequence[str]
) -> str | None:
    """Return the first forbidden target with bidirectional token containment."""
    normalized_target = _normalize_phrase(target)
    for forbidden in forbidden_targets:
        normalized_forbidden = _normalize_phrase(forbidden)
        if _contains_token_sequence(
            normalized_target, normalized_forbidden
        ) or _contains_token_sequence(normalized_forbidden, normalized_target):
            return forbidden
    return None


def _validate_lexical_plausibility(category: str, phrase: str) -> None:
    """Reject finite cross-category vocabulary and robust synthetic stacking patterns."""
    tokens = _normalize_phrase(phrase).split()
    modifiers = tokens[:-1]
    if len(modifiers) != len(set(modifiers)):
        raise ValueError(f"target phrase uses repeated synthetic modifiers: {phrase!r}")
    blocked = sorted(set(tokens) & _CROSS_CATEGORY_MODIFIERS)
    if (
        category in _PUBLIC_PREFERENCE_CATEGORIES
        and blocked
        and (category, " ".join(tokens)) not in _CONVENTIONAL_CROSS_CATEGORY_PHRASES
    ):
        raise ValueError(
            f"target phrase uses cross-category fabric, mineral, or gemstone terms "
            f"{blocked}: {phrase!r}"
        )


def validate_surface_mapping_response(
    content: str,
    expected_rows: Sequence[Mapping[str, str]],
    parent_phrases: set[str],
    forbidden_targets: Sequence[str] = (),
    cross_assignment_forbidden_targets: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Validate exact mapping coverage/order and all lexical anti-leakage contracts."""
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"mapping response is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"mapping"} or not isinstance(
        payload["mapping"], list
    ):
        raise ValueError("mapping response must contain only a mapping array")
    rows = payload["mapping"]
    if len(rows) != len(expected_rows):
        raise ValueError(
            f"mapping coverage mismatch: expected {len(expected_rows)} rows, got {len(rows)}"
        )
    expected_keys = [dict(row) for row in expected_rows]
    actual_keys = []
    targets = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "history_id",
            "category",
            "source_phrase",
            "target_phrase",
        }:
            raise ValueError(f"mapping row {index} has malformed fields")
        key = {name: row[name] for name in ("history_id", "category", "source_phrase")}
        if any(not isinstance(value, str) or not value for value in key.values()):
            raise ValueError(f"mapping row {index} has incomplete keys")
        actual_keys.append(key)
        target = row["target_phrase"]
        if not isinstance(target, str) or not target.strip() or target != target.strip():
            raise ValueError(f"mapping row {index} has an invalid target phrase")
        if _LATENT_ID.search(target):
            raise ValueError(f"mapping row {index} target contains a latent ID")
        _validate_category_shape(str(row["category"]), target)
        _validate_lexical_plausibility(str(row["category"]), target)
        normalized = _normalize_phrase(target)
        parent_collision = _parent_phrase_collision(target, parent_phrases)
        if parent_collision is not None:
            raise ValueError(
                f"mapping row {index} has parent collision with {parent_collision!r}: "
                f"{target!r} has normalized token-sequence equality or containment"
            )
        prior_collision = _forbidden_target_collision(target, forbidden_targets)
        if prior_collision is not None:
            raise ValueError(
                f"mapping row {index} has prior-target collision with {prior_collision!r}: "
                f"{target!r} has normalized token-sequence equality or containment"
            )
        cross_collision = _forbidden_target_collision(
            target, cross_assignment_forbidden_targets
        )
        if cross_collision is not None:
            raise ValueError(
                f"mapping row {index} has cross-assignment collision with "
                f"{cross_collision!r}: {target!r} has normalized token-sequence "
                "equality or containment"
            )
        targets.append(normalized)
    if actual_keys != expected_keys:
        if set(map(_stable_hash, actual_keys)) == set(map(_stable_hash, expected_keys)):
            raise ValueError("mapping changed required row order")
        raise ValueError("mapping coverage keys differ from the expected corpus")
    if len(set(targets)) != len(targets):
        raise ValueError("mapping target phrases must be globally unique")
    return [{name: str(row[name]) for name in row} for row in rows]


def validate_whole_mapping_naturalness(
    mapping: Sequence[Mapping[str, str]], parent_phrases: set[str]
) -> None:
    """Recheck complete mapping lexical naturalness before any dialogue starts."""
    normalized_targets = []
    for index, row in enumerate(mapping):
        category = str(row.get("category", ""))
        target = str(row.get("target_phrase", ""))
        _validate_category_shape(category, target)
        _validate_lexical_plausibility(category, target)
        parent_collision = _parent_phrase_collision(target, parent_phrases)
        if parent_collision is not None:
            raise ValueError(
                f"whole mapping row {index} has parent collision with {parent_collision!r}: "
                f"{target!r} has normalized token-sequence equality or containment"
            )
        normalized_targets.append(_normalize_phrase(target))
    if len(set(normalized_targets)) != len(normalized_targets):
        raise ValueError("whole mapping target phrases must be globally unique")


def validate_paired_surface_mappings(
    mapping_a: Sequence[Mapping[str, str]], mapping_b: Sequence[Mapping[str, str]]
) -> None:
    """Reject normalized equality or containment anywhere across assignments."""
    a_targets = [_normalize_phrase(str(row["target_phrase"])) for row in mapping_a]
    b_targets = [_normalize_phrase(str(row["target_phrase"])) for row in mapping_b]
    for left in a_targets:
        for right in b_targets:
            if left == right or left in right or right in left:
                raise ValueError(
                    f"cross-assignment normalized equality/containment collision: {left!r}, {right!r}"
                )


def _authenticate_parent(parent_dir: Path) -> tuple[dict[str, Any], str, dict[str, bytes]]:
    """Read and authenticate every artifact declared by a completed Kimi parent."""
    manifest_path = parent_dir / "generation_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load parent generation manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError("paired generation requires a completed parent manifest")
    model = manifest.get("model_identity")
    if not isinstance(model, str) or "kimi" not in model.casefold():
        raise ValueError(f"paired generation requires a Kimi parent, got {model!r}")
    recorded = manifest.get("artifact_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("parent generation manifest lacks artifact_sha256")
    artifacts = {}
    for name, expected in sorted(recorded.items()):
        if not isinstance(name, str) or Path(name).name != name or name == "generation_manifest.json":
            raise ValueError(f"invalid parent artifact name: {name!r}")
        try:
            payload = (parent_dir / name).read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read parent artifact {name}: {error}") from error
        if _sha256_bytes(payload) != expected:
            raise ValueError(f"parent artifact hash mismatch for {name}")
        artifacts[name] = payload
    required = {*SOURCE_ARTIFACTS, "dialogue.jsonl", "requests.jsonl", "raw_responses.jsonl"}
    if not required.issubset(artifacts):
        raise ValueError(f"parent manifest omits required artifacts: {sorted(required - artifacts.keys())}")
    return manifest, _sha256_bytes(manifest_bytes), artifacts


def _parent_surface_phrases(
    events: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]]
) -> set[str]:
    """Return canonical mapping sources plus parent model-visible answer values."""
    phrases = {
        source for values in CATEGORY_SOURCE_PHRASES.values() for source in values
    }
    phrases.update(
        str(event["surface_object"])
        for event in events
        if isinstance(event.get("surface_object"), str)
    )
    phrases.update(
        str(query["surface_gold"])
        for query in queries
        if isinstance(query.get("surface_gold"), str)
    )
    return phrases


def _replacement_index(mapping: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    """Index one complete mapping by history and source phrase."""
    result: dict[str, dict[str, str]] = {}
    for row in mapping:
        history = result.setdefault(str(row["history_id"]), {})
        source = str(row["source_phrase"])
        if source in history:
            raise ValueError(f"duplicate mapping source {source!r} for {row['history_id']!r}")
        history[source] = str(row["target_phrase"])
    return result


def _replace_visible_text(text: str, replacements: Mapping[str, str]) -> str:
    """Replace canonical parent phrases longest-first without touching latent structure."""
    result = text
    for source, target in sorted(replacements.items(), key=lambda item: (-len(item[0]), item[0])):
        result = re.sub(
            rf"\b{re.escape(source)}\b",
            lambda match: target[:1].upper() + target[1:] if match.group(0)[:1].isupper() else target,
            result,
            flags=re.IGNORECASE,
        )
    return result


def validate_parent_preservation(
    parent_events: Sequence[Mapping[str, Any]],
    child_events: Sequence[Mapping[str, Any]],
    parent_queries: Sequence[Mapping[str, Any]],
    child_queries: Sequence[Mapping[str, Any]],
) -> None:
    """Prove event/query latent fields, ordering, relations, gold, and checkpoints are unchanged."""
    if len(parent_events) != len(child_events):
        raise ValueError("child event coverage differs from parent")
    if len(parent_queries) != len(child_queries):
        raise ValueError("child query coverage differs from parent")
    event_latent = [
        {key: value for key, value in row.items() if key not in _VISIBLE_EVENT_FIELDS}
        for row in parent_events
    ]
    child_event_latent = [
        {key: value for key, value in row.items() if key not in _VISIBLE_EVENT_FIELDS}
        for row in child_events
    ]
    if event_latent != child_event_latent:
        raise ValueError("child event latent structure/order differs from parent")
    query_latent = [
        {key: value for key, value in row.items() if key not in _VISIBLE_QUERY_FIELDS}
        for row in parent_queries
    ]
    child_query_latent = [
        {key: value for key, value in row.items() if key not in _VISIBLE_QUERY_FIELDS}
        for row in child_queries
    ]
    if query_latent != child_query_latent:
        raise ValueError("child query causal/checkpoint contract differs from parent")


def _validate_no_stale_parent_phrases(
    child_events: Sequence[Mapping[str, Any]],
    child_queries: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, str]],
) -> None:
    """Reject canonical parent phrases remaining in same-history model-visible fields."""
    replacements = _replacement_index(mapping)
    visible_by_history: dict[str, list[str]] = {}
    for event in child_events:
        visible_by_history.setdefault(str(event.get("history_id", "")), []).extend(
            str(event.get(field, "")) for field in _VISIBLE_EVENT_FIELDS
        )
    for query in child_queries:
        visible_by_history.setdefault(str(query.get("history_id", "")), []).extend(
            str(query.get(field, "")) for field in _VISIBLE_QUERY_FIELDS
        )
    for history_id, history_mapping in replacements.items():
        visible = "\n".join(visible_by_history.get(history_id, ()))
        for source in history_mapping:
            if re.search(rf"\b{re.escape(source)}\b", visible, re.IGNORECASE):
                raise ValueError(
                    f"history {history_id} retains stale parent phrase {source!r}"
                )


def _mapping_messages(
    expected_rows: Sequence[Mapping[str, str]],
    assignment: str,
    request_index: int,
    history_ids: Sequence[str],
    forbidden_targets: Sequence[str],
    cross_assignment_forbidden_targets: Sequence[str] | None,
) -> list[dict[str, str]]:
    """Build one fixed-assignment prompt with only the permitted conditioning data."""
    if assignment not in SURFACE_ASSIGNMENT_SEEDS:
        raise ValueError(f"unsupported surface assignment {assignment!r}")
    if assignment == "A" and cross_assignment_forbidden_targets is not None:
        raise ValueError("assignment A must not receive cross-assignment targets")
    if assignment == "B" and cross_assignment_forbidden_targets is None:
        raise ValueError("assignment B requires a flat cross-assignment target avoid-list")
    realization_id = f"surface-{assignment.lower()}-{SURFACE_ASSIGNMENT_SEEDS[assignment]}"
    lexical_directive = MAPPING_LEXICAL_DIRECTIVES[0 if assignment == "A" else 1]
    conditioning_instruction = (
        "No sibling mapping information is supplied to assignment A."
        if assignment == "A"
        else (
            "For assignment B, avoid every normalized phrase in the flat "
            "cross_assignment_forbidden_targets list. That list exposes no sibling history, "
            "category, source, or mapping association."
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "Propose a complete replacement phrase mapping for this corpus. Return strict JSON with "
                "exactly one key named mapping. Preserve every supplied row and its order, add only "
                "target_phrase, keep each target in the stated category, use two to five words and at "
                "most 64 characters. Tea targets must end with 'tea'; seating targets must end with "
                "'seating'; delivery targets must end with 'delivery'; receipt targets must end with "
                "the plural 'receipts'. Obey the supplied finite final-noun contracts for food and "
                "private_lineage. Use no latent or benchmark IDs, and do not reuse any normalized "
                "parent surface phrase by equality or token-sequence containment. Every target must "
                "be a conventional real-world option a person could naturally prefer, request, eat, "
                "receive, or keep. Forbid arbitrary adjective stacking and synthetic code-like labels. "
                "Forbid cross-category material combinations, including fabric, mineral, or gemstone "
                "terms used outside a conventional category meaning. "
                "Make every target unique across the complete assignment, "
                "including other history batches. Do not reuse any normalized phrase listed in "
                "forbidden_targets by equality or token-sequence containment. "
                f"{conditioning_instruction} Follow lexical_directive as style guidance; it does not alter "
                "the schema or category contracts. The "
                "realization_id identifies only this request; do not infer or discuss other assignments."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "schema_version": PROMPT_SCHEMA_VERSION,
                    "realization_id": realization_id,
                    "mapping_request_index": request_index,
                    "history_ids": list(history_ids),
                    "forbidden_targets": list(forbidden_targets),
                    "lexical_directive": lexical_directive,
                    "category_lexical_contracts": {
                        "food_final_nouns": sorted(_FOOD_NOUNS),
                        "private_lineage_final_nouns": sorted(_PRIVATE_LINEAGE_NOUNS),
                    },
                    "mapping": list(expected_rows),
                    **(
                        {
                            "cross_assignment_forbidden_targets": list(
                                cross_assignment_forbidden_targets
                            )
                        }
                        if cross_assignment_forbidden_targets is not None
                        else {}
                    ),
                },
                sort_keys=True,
            ),
        },
    ]


def _dialogue_messages(
    expected: Sequence[Mapping[str, Any]], turn_pairs: int, minimum_words: int
) -> list[dict[str, str]]:
    """Reuse the existing conversation response contract for one fresh realization batch."""
    return build_generation_messages(
        expected,
        prompt_schema_version=PROMPT_SCHEMA_VERSION,
        turn_pairs_per_event=turn_pairs,
        minimum_words_per_turn=minimum_words,
    )


def _expected_dialogue_events(
    parent_events: Sequence[Mapping[str, Any]], mapping: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    """Build existing-generator semantic validation inputs from parent-backed visible facts."""
    replacements = _replacement_index(mapping)
    parent_value_map, subject_map, scope_map = _surface_maps(parent_events)
    value_maps_by_history: dict[str, dict[str, str]] = {}
    for event in parent_events:
        fact = event.get("fact")
        if not isinstance(fact, Mapping):
            continue
        history_id = str(event.get("history_id", ""))
        latent_object = str(fact.get("object", ""))
        surface = parent_value_map.get(latent_object)
        if surface is not None:
            value_maps_by_history.setdefault(history_id, {})[
                latent_object
            ] = replacements.get(history_id, {}).get(surface, surface)
    facts_by_id = {
        str(event["fact"]["fact_id"]): event["fact"]
        for event in parent_events
        if isinstance(event.get("fact"), Mapping) and event["fact"].get("fact_id")
    }
    expected = []
    for event in parent_events:
        history_id = str(event.get("history_id", ""))
        history_mapping = replacements.get(history_id)
        if history_mapping is None:
            raise ValueError(f"mapping omits event history {history_id!r}")
        required_values = _required_surface_values(
            event, facts_by_id, value_maps_by_history.get(history_id, {})
        )
        fact = event.get("fact")
        if not isinstance(fact, Mapping):
            raise ValueError(f"event {event.get('event_id')} lacks a fact")
        latent_object = str(fact.get("object", ""))
        support_text = str(fact.get("support_text", ""))
        history_values = value_maps_by_history.get(history_id, {})
        if latent_object in history_values:
            support_text = re.sub(
                r"\bvalue-\d{3}-[a-z]\b", latent_object, support_text
            )
        visible_replacements = {**history_values, **subject_map, **scope_map}
        surface_text = _replace_latent(support_text, visible_replacements)
        dialogue_subject = visible_replacements.get(
            str(fact.get("subject")), "the account owner"
        )
        speaker, instruction, authority = _speaker_contract(event, dialogue_subject)
        expected.append(
            {
                "event_id": event["event_id"],
                "dialogue_subject": dialogue_subject,
                "dialogue_speaker": speaker,
                "speaker_instruction": instruction,
                "authority_markers": authority,
                "surface_text": surface_text,
                "required_surface_values": required_values,
                "semantic_markers": list(_semantic_markers(event)),
                "semantic_instruction": _semantic_instruction(event, required_values),
                "semantic_polarity": _semantic_polarity(event, required_values),
                "forbidden_surface_phrases": [
                    *_forbidden_surface_phrases(event),
                    *history_mapping.keys(),
                ],
            }
        )
    return expected


def _complete_with_validation(
    client: CompletionClient,
    *,
    output_dir: Path,
    running_manifest: dict[str, Any],
    messages: Sequence[Mapping[str, str]],
    config: SurfacePairConfig,
    seed: int,
    max_tokens: int,
    validator: Any,
    request_context: Mapping[str, Any],
    request_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    max_attempts: int | None = None,
    accept_on_validation: bool = True,
    read_only_cache: bool = False,
) -> Any:
    """Reuse exact validated responses or durably issue changed attempts."""
    attempt_limit = config.max_validation_attempts if max_attempts is None else max_attempts
    validation_error: ValueError | None = None
    transient_error: dict[str, str] | None = None
    last_failure: str | None = None
    for attempt in range(attempt_limit):
        attempt_messages = list(messages)
        if validation_error is not None:
            attempt_messages.insert(
                -1,
                {
                    "role": "system",
                    "content": f"Repair this validation failure and return the complete response: {validation_error}",
                },
            )
        elif transient_error is not None:
            attempt_messages.insert(
                -1,
                {
                    "role": "system",
                    "content": _transient_retry_instruction(transient_error),
                },
            )
        prompt_hash = _stable_hash(attempt_messages)
        attempt_seed = seed + attempt
        expected_request = {
            **dict(request_context),
            "attempt_index": attempt,
            "seed": attempt_seed,
            "requested_model": config.model,
            "max_tokens": max_tokens,
            "enable_thinking": config.enable_thinking,
            "prompt_sha256": prompt_hash,
            "messages": attempt_messages,
        }
        key = (
            str(request_context.get("stage")),
            int(request_context.get("request_index", -1)),
            attempt,
        )
        matching_requests = [
            row
            for row in request_rows
            if (
                str(row.get("stage")),
                int(row.get("request_index", -1)),
                int(row.get("attempt_index", -1)),
            )
            == key
        ]
        if len(matching_requests) > 1:
            raise CacheIntegrityError(f"duplicate cached request key {key}")
        if matching_requests:
            request_row = matching_requests[0]
            for field, expected in expected_request.items():
                if request_row.get(field) != expected:
                    raise CacheIntegrityError(
                        f"cached request mismatch for {key} on {field}"
                    )
        else:
            if read_only_cache:
                raise CacheIntegrityError(f"completed assignment lacks cached request {key}")
            request_row = {
                **expected_request,
                "accepted": False,
                **(
                    {"superseded": False}
                    if request_context.get("stage") == "mapping"
                    else {}
                ),
            }
            request_rows.append(request_row)
            _persist_generation_state(
                output_dir, running_manifest, request_rows, raw_rows
            )

        matching_responses = [
            row
            for row in raw_rows
            if (
                str(row.get("stage")),
                int(row.get("request_index", -1)),
                int(row.get("attempt_index", -1)),
            )
            == key
        ]
        if len(matching_responses) > 1:
            raise CacheIntegrityError(f"duplicate cached response key {key}")
        cached = bool(matching_responses)
        if cached:
            response_row = matching_responses[0]
            expected_response_metadata = {
                **dict(request_context),
                "attempt_index": attempt,
                "seed": attempt_seed,
                "requested_model": config.model,
                "max_tokens": max_tokens,
                "enable_thinking": config.enable_thinking,
            }
            for field, expected in expected_response_metadata.items():
                if response_row.get(field) != expected:
                    raise CacheIntegrityError(
                        f"cached response mismatch for {key} on {field}"
                    )
            if "provider_error" in response_row:
                try:
                    _validate_provider_error_row(response_row)
                except ValueError as error:
                    raise CacheIntegrityError(
                        f"cached provider error mismatch for {key}: {error}"
                    ) from error
                if request_row.get("accepted") is not False:
                    raise CacheIntegrityError(
                        f"cached provider error request is accepted for {key}"
                    )
                if request_context.get("stage") == "mapping" and request_row.get(
                    "superseded"
                ) is not True:
                    raise CacheIntegrityError(
                        f"cached provider error request is not superseded for {key}"
                    )
                transient_error = dict(response_row["provider_error"])
                validation_error = None
                last_failure = _transient_retry_instruction(transient_error)
                continue
            content = response_row.get("content")
            if not isinstance(content, str) or response_row.get(
                "response_sha256"
            ) != _sha256_bytes(content.encode("utf-8")):
                raise CacheIntegrityError(f"cached response hash mismatch for {key}")
        else:
            if read_only_cache:
                raise CacheIntegrityError(f"completed assignment lacks cached response {key}")
            try:
                response = client.complete(
                    messages=attempt_messages,
                    model=config.model,
                    timeout=config.timeout_seconds,
                    seed=attempt_seed,
                    max_tokens=max_tokens,
                )
            except Exception as error:
                if type(error) not in _TRANSIENT_PROVIDER_ERROR_TYPES:
                    raise
                transient_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                validation_error = None
                request_row["accepted"] = False
                if request_context.get("stage") == "mapping":
                    request_row["superseded"] = True
                response_row = _provider_error_row(request_row, transient_error)
                raw_rows.append(response_row)
                _persist_generation_state(
                    output_dir, running_manifest, request_rows, raw_rows
                )
                last_failure = _transient_retry_instruction(transient_error)
                continue
            response_row = {
                **dict(request_context),
                "attempt_index": attempt,
                "seed": attempt_seed,
                "requested_model": config.model,
                "max_tokens": max_tokens,
                "enable_thinking": config.enable_thinking,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "usage": dict(response.usage),
                "response_sha256": _sha256_bytes(response.content.encode("utf-8")),
                "content": response.content,
                "accepted": False,
                **(
                    {"superseded": False}
                    if request_context.get("stage") == "mapping"
                    else {}
                ),
            }
            raw_rows.append(response_row)
            _persist_generation_state(
                output_dir, running_manifest, request_rows, raw_rows
            )
        try:
            validate_kimi_model_identity(config.model, str(response_row.get("model")))
            if response_row.get("finish_reason") != "stop":
                raise ValueError(
                    f"generation ended with finish reason {response_row.get('finish_reason')!r}"
                )
            parsed = validator(str(response_row["content"]))
            if accept_on_validation:
                request_row["accepted"] = True
                response_row["accepted"] = True
            if cached and not read_only_cache:
                running_manifest["resumed_response_count"] = int(
                    running_manifest.get("resumed_response_count", 0)
                ) + 1
            if not read_only_cache:
                _persist_generation_state(
                    output_dir, running_manifest, request_rows, raw_rows
                )
            return parsed
        except ValueError as error:
            if not read_only_cache:
                request_row["accepted"] = False
                response_row["accepted"] = False
                if request_context.get("stage") == "mapping":
                    request_row["superseded"] = True
                    response_row["superseded"] = True
                _persist_generation_state(
                    output_dir, running_manifest, request_rows, raw_rows
                )
            validation_error = error
            transient_error = None
            last_failure = str(error)
    raise ValueError(
        f"{request_context.get('stage')} request failed after "
        f"{attempt_limit} attempts: {last_failure}"
    )


def _request_mapping_candidate(
    client: CompletionClient,
    *,
    output_dir: Path,
    running_manifest: dict[str, Any],
    assignment: str,
    expected_rows: Sequence[Mapping[str, str]],
    parent_phrases: set[str],
    cross_assignment_forbidden_targets: Sequence[str] | None,
    config: SurfacePairConfig,
    request_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    read_only_cache: bool = False,
) -> list[dict[str, str]]:
    """Request sequential resumable mapping batches with prior-target exclusion."""
    rows_by_history: dict[str, list[Mapping[str, str]]] = {}
    for row in expected_rows:
        rows_by_history.setdefault(str(row["history_id"]), []).append(row)
    histories = list(rows_by_history)
    histories_per_request = config.mapping_histories_per_request
    batches = [
        histories[index : index + histories_per_request]
        for index in range(0, len(histories), histories_per_request)
    ]
    mapping = []
    prior_targets: list[str] = []
    rows_per_history = sum(len(values) for values in CATEGORY_SOURCE_PHRASES.values())
    for batch_index, batch_history_ids in enumerate(batches):
        batch_rows = [
            row for history_id in batch_history_ids for row in rows_by_history[history_id]
        ]
        if len(batch_rows) != rows_per_history * len(batch_history_ids):
            raise ValueError("mapping batch has incomplete history row coverage")
        request_index = batch_index
        seed = (
            SURFACE_ASSIGNMENT_SEEDS[assignment]
            + batch_index * 1_000
        )
        messages = _mapping_messages(
            batch_rows,
            assignment,
            request_index,
            batch_history_ids,
            prior_targets,
            cross_assignment_forbidden_targets,
        )
        context = {
            "stage": "mapping",
            "assignment": assignment,
            "request_index": request_index,
            "mapping_request_index": request_index,
            "history_ids": batch_history_ids,
        }
        accepted_batch = _complete_with_validation(
            client,
            output_dir=output_dir,
            running_manifest=running_manifest,
            messages=messages,
            seed=seed,
            max_tokens=config.mapping_max_tokens,
            config=config,
            validator=lambda content, expected=batch_rows, forbidden=tuple(
                prior_targets
            ): validate_surface_mapping_response(
                content,
                expected,
                parent_phrases,
                forbidden,
                cross_assignment_forbidden_targets or (),
            ),
            request_context=context,
            request_rows=request_rows,
            raw_rows=raw_rows,
            accept_on_validation=False,
            read_only_cache=read_only_cache,
        )
        mapping.extend(accepted_batch)
        prior_targets.extend(
            _normalize_phrase(str(row["target_phrase"])) for row in accepted_batch
        )
    return validate_surface_mapping_response(
        json.dumps({"mapping": mapping}),
        expected_rows,
        parent_phrases,
        (),
        cross_assignment_forbidden_targets or (),
    )


def _persist_generation_state(
    output_dir: Path,
    running_manifest: dict[str, Any],
    request_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically persist logs and their running-manifest provenance."""
    requests_payload = _jsonl_bytes(request_rows)
    responses_payload = _jsonl_bytes(raw_rows)
    _atomic_write_bytes(output_dir / "requests.jsonl", requests_payload)
    _atomic_write_bytes(output_dir / "raw_responses.jsonl", responses_payload)
    running_manifest.update(
        {
            "status": "running",
            "completion_status": "running",
            "request_count": len(request_rows),
            "response_count": len(raw_rows),
            "prompt_sha256": [
                str(row["prompt_sha256"]) for row in request_rows
            ],
            "response_sha256": [
                _raw_provenance_hash(row) for row in raw_rows
            ],
            "provenance_artifact_sha256": {
                "requests.jsonl": _sha256_bytes(requests_payload),
                "raw_responses.jsonl": _sha256_bytes(responses_payload),
            },
        }
    )
    _atomic_write_bytes(
        output_dir / "generation_manifest.json", _json_bytes(running_manifest)
    )


def _generate_assignment(
    output_dir: Path,
    assignment: str,
    config: SurfacePairConfig,
    client: CompletionClient,
    *,
    parent_dir: Path,
    parent_manifest: Mapping[str, Any],
    parent_manifest_hash: str,
    parent_artifacts: Mapping[str, bytes],
    parent_events: Sequence[Mapping[str, Any]],
    parent_queries: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, str]],
    mapping_request_rows: Sequence[Mapping[str, Any]],
    mapping_raw_rows: Sequence[Mapping[str, Any]],
    running_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Realize dialogue and materialize one assignment from a prevalidated mapping."""
    seed = SURFACE_ASSIGNMENT_SEEDS[assignment]
    request_rows = [dict(row) for row in mapping_request_rows]
    raw_rows = [dict(row) for row in mapping_raw_rows]
    running = dict(running_manifest)
    _persist_generation_state(output_dir, running, request_rows, raw_rows)
    try:
        replacements = _replacement_index(mapping)
        expected_events = _expected_dialogue_events(parent_events, mapping)
        generated_by_id: dict[str, Mapping[str, Any]] = {}
        for request_index, offset in enumerate(
            range(0, len(parent_events), config.events_per_request), start=1
        ):
            batch = expected_events[offset : offset + config.events_per_request]
            messages = _dialogue_messages(
                batch, config.turn_pairs_per_event, config.minimum_words_per_turn
            )
            parsed = _complete_with_validation(
                client,
                output_dir=output_dir,
                running_manifest=running,
                messages=messages,
                config=config,
                seed=seed + request_index * 1000,
                max_tokens=config.dialogue_max_tokens,
                validator=lambda content, expected=batch: validate_generation_response(
                    content,
                    expected,
                    turn_pairs_per_event=config.turn_pairs_per_event,
                    minimum_words_per_turn=config.minimum_words_per_turn,
                ),
                request_context={
                    "stage": "dialogue",
                    "assignment": assignment,
                    "request_index": request_index,
                    "event_ids": [str(row["event_id"]) for row in batch],
                },
                request_rows=request_rows,
                raw_rows=raw_rows,
            )
            generated_by_id.update(
                {str(row["event_id"]): row for row in parsed["events"]}
            )
        expected_ids = [str(row["event_id"]) for row in parent_events]
        if list(generated_by_id) != expected_ids:
            raise ValueError("dialogue requests did not realize exact parent event order")

        child_events = []
        dialogue_rows = []
        parent_dialogue = _load_jsonl(parent_artifacts["dialogue.jsonl"], "parent dialogue.jsonl")
        if len(parent_dialogue) != len(parent_events):
            raise ValueError("parent dialogue/event coverage differs")
        for parent_event, parent_dialogue_row in zip(
            parent_events, parent_dialogue, strict=True
        ):
            event_id = str(parent_event["event_id"])
            generated = generated_by_id[event_id]
            history_id = str(parent_event["history_id"])
            transformed = dict(parent_event)
            transformed["model_text"] = _render_dialogue(generated["turns"])
            transformed["surface_object"] = _replace_visible_text(
                str(parent_event.get("surface_object", "")), replacements[history_id]
            )
            child_events.append(transformed)
            dialogue_row = dict(parent_dialogue_row)
            dialogue_row["text"] = transformed["model_text"]
            dialogue_rows.append(dialogue_row)

        child_queries = []
        for parent_query in parent_queries:
            history_id = str(parent_query.get("history_id", ""))
            if history_id not in replacements:
                raise ValueError(f"query references unmapped history {history_id!r}")
            child = dict(parent_query)
            for field in _VISIBLE_QUERY_FIELDS:
                value = parent_query.get(field)
                if not isinstance(value, str):
                    raise ValueError(f"query {parent_query.get('query_id')} lacks {field}")
                child[field] = _replace_visible_text(value, replacements[history_id])
            child_queries.append(child)
        validate_parent_preservation(
            parent_events, child_events, parent_queries, child_queries
        )
        _validate_no_stale_parent_phrases(child_events, child_queries, mapping)

        output_artifacts = {
            name: payload
            for name, payload in parent_artifacts.items()
            if name not in _REBUILT_ARTIFACTS
        }
        output_artifacts.update(
            {
                "events.jsonl": _jsonl_bytes(child_events),
                "dialogue.jsonl": _jsonl_bytes(dialogue_rows),
                "queries.jsonl": _jsonl_bytes(child_queries),
                "requests.jsonl": _jsonl_bytes(request_rows),
                "raw_responses.jsonl": _jsonl_bytes(raw_rows),
            }
        )
        for name, payload in output_artifacts.items():
            _atomic_write_bytes(output_dir / name, payload)
        manifest = {
            **running,
            "status": "completed",
            "completion_status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model_identity": EXPECTED_MODEL,
            "returned_model": EXPECTED_MODEL,
            "prompt_schema_version": PROMPT_SCHEMA_VERSION,
            "generator_role": "Separately Kimi-authored pair-conditioned surface mapping and dialogue realization",
            "provider_identity_assurance": "endpoint-self-reported; artifact hashes are cryptographic",
            "parent": {
                "corpus_name": parent_dir.name,
                "generation_manifest_sha256": parent_manifest_hash,
                "artifact_sha256": dict(sorted(parent_manifest["artifact_sha256"].items())),
            },
            "surface_mapping": mapping,
            "surface_mapping_sha256": _stable_hash(mapping),
            "event_count": len(child_events),
            "request_count": len(request_rows),
            "response_count": len(raw_rows),
            "prompt_sha256": [row["prompt_sha256"] for row in request_rows],
            "response_sha256": [_raw_provenance_hash(row) for row in raw_rows],
            "effective_generation": _usage_token_totals(raw_rows),
            "artifact_roles": {
                **dict(parent_manifest.get("artifact_roles", {})),
                "requests.jsonl": "fresh_generation_provenance",
                "raw_responses.jsonl": "fresh_generation_provenance",
            },
            "preservation_scope": PRESERVATION_SCOPE,
            "artifact_sha256": {
                name: _sha256_bytes(payload) for name, payload in sorted(output_artifacts.items())
            },
        }
        _atomic_write_bytes(
            output_dir / "generation_manifest.json", _json_bytes(manifest)
        )
        return manifest
    except Exception as error:
        failed = {
            **running,
            "status": "failed",
            "completion_status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        _atomic_write_bytes(
            output_dir / "requests.jsonl", _jsonl_bytes(request_rows)
        )
        _atomic_write_bytes(
            output_dir / "raw_responses.jsonl", _jsonl_bytes(raw_rows)
        )
        _atomic_write_bytes(
            output_dir / "generation_manifest.json", _json_bytes(failed)
        )
        raise


def _read_completed_variant(
    path: Path, assignment: str, parent_manifest_hash: str
) -> tuple[dict[str, Any], str, list[dict[str, str]], dict[str, bytes]]:
    """Authenticate one completed Kimi child manifest, mapping, and every artifact."""
    manifest_path = path / "generation_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {assignment} manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError(f"{assignment} manifest is not completed")
    expected = {
        "method": DERIVATION_METHOD,
        "assignment": assignment,
        "seed": SURFACE_ASSIGNMENT_SEEDS[assignment],
        "requested_model": EXPECTED_MODEL,
        "returned_model": EXPECTED_MODEL,
        "model_identity": EXPECTED_MODEL,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"{assignment} manifest has invalid {field}")
    controls = manifest.get("generation_controls")
    if not isinstance(controls, Mapping) or manifest.get(
        "generation_controls_sha256"
    ) != _stable_hash(controls):
        raise ValueError(f"{assignment} manifest generation controls hash mismatch")
    expected_request_parameters = {
        "model": controls.get("requested_model"),
        **{
            key: controls.get(key)
            for key in (
                "timeout_seconds",
                "mapping_max_tokens",
                "dialogue_max_tokens",
                "mapping_histories_per_request",
                "dialogue_assignment_concurrency",
                "events_per_request",
                "turn_pairs_per_event",
                "minimum_words_per_turn",
                "max_validation_attempts",
                "enable_thinking",
                "temperature",
                "json_mode",
            )
        },
    }
    if manifest.get("request_parameters") != expected_request_parameters:
        raise ValueError(
            f"{assignment} manifest request parameters differ from generation controls"
        )
    if (
        controls.get("surface_assignment_seeds") != SURFACE_ASSIGNMENT_SEEDS
        or controls.get("derivation_method") != DERIVATION_METHOD
        or controls.get("prompt_schema_version") != PROMPT_SCHEMA_VERSION
        or controls.get("temperature") != 0
        or controls.get("json_mode") is not True
    ):
        raise ValueError(f"{assignment} manifest has invalid fixed generation controls")
    if manifest.get("preservation_scope") != PRESERVATION_SCOPE:
        raise ValueError(f"{assignment} manifest has invalid preservation scope")
    parent = manifest.get("parent")
    if not isinstance(parent, Mapping) or parent.get(
        "generation_manifest_sha256"
    ) != parent_manifest_hash:
        raise ValueError(f"{assignment} manifest has wrong parent identity")
    mapping = manifest.get("surface_mapping")
    if not isinstance(mapping, list) or manifest.get("surface_mapping_sha256") != _stable_hash(mapping):
        raise ValueError(f"{assignment} manifest mapping hash mismatch")
    recorded = manifest.get("artifact_sha256")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"{assignment} manifest lacks artifact hashes")
    artifacts = {}
    for name, digest in recorded.items():
        if not isinstance(name, str) or Path(name).name != name or name == "generation_manifest.json":
            raise ValueError(f"{assignment} has invalid artifact name {name!r}")
        try:
            payload = (path / name).read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read {assignment} artifact {name}: {error}") from error
        if _sha256_bytes(payload) != digest:
            raise ValueError(f"{assignment} artifact hash mismatch for {name}")
        artifacts[name] = payload
    return manifest, _sha256_bytes(manifest_bytes), mapping, artifacts


def _validate_fresh_provenance(
    assignment: str,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    parent_artifacts: Mapping[str, bytes],
    expected_event_ids: Sequence[str],
) -> set[str]:
    """Authenticate fresh requests/responses and semantic coverage for one child."""
    if artifacts["requests.jsonl"] == parent_artifacts["requests.jsonl"] or artifacts[
        "raw_responses.jsonl"
    ] == parent_artifacts["raw_responses.jsonl"]:
        raise ValueError(f"{assignment} reused parent Kimi response provenance")
    requests = _load_jsonl(artifacts["requests.jsonl"], f"{assignment} requests.jsonl")
    responses = _load_jsonl(
        artifacts["raw_responses.jsonl"], f"{assignment} raw_responses.jsonl"
    )
    if len(requests) != len(responses) or not requests:
        raise ValueError(f"{assignment} request/response coverage differs")
    def provenance_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
        return (
            str(row.get("stage")),
            int(row.get("request_index", -1)),
            int(row.get("attempt_index", -1)),
        )

    responses_by_key = {provenance_key(row): row for row in responses}
    if len(responses_by_key) != len(responses):
        raise ValueError(f"{assignment} contains duplicate response provenance keys")
    request_keys = [provenance_key(row) for row in requests]
    if len(set(request_keys)) != len(request_keys) or set(request_keys) != set(
        responses_by_key
    ):
        raise ValueError(f"{assignment} request/response provenance key sets differ")
    accepted_dialogue_batches = []
    accepted_mapping_batches = []
    response_hashes = set()
    controls = manifest["generation_controls"]
    for request in requests:
        response = responses_by_key[provenance_key(request)]
        for key in (
            "stage",
            "assignment",
            "request_index",
            "attempt_index",
            "seed",
            "requested_model",
            "max_tokens",
            "enable_thinking",
            "mapping_request_index",
            "history_ids",
            "event_ids",
        ):
            if request.get(key) != response.get(key):
                raise ValueError(f"{assignment} request/response provenance differs on {key}")
        if request.get("assignment") != assignment:
            raise ValueError(f"{assignment} request carries wrong assignment")
        expected_max_tokens = controls.get(
            "mapping_max_tokens"
            if request.get("stage") == "mapping"
            else "dialogue_max_tokens"
        )
        if (
            request.get("requested_model") != controls.get("requested_model")
            or request.get("max_tokens") != expected_max_tokens
            or request.get("enable_thinking") != controls.get("enable_thinking")
        ):
            raise ValueError(f"{assignment} request differs from generation controls")
        if request.get("prompt_sha256") != _stable_hash(request.get("messages")):
            raise ValueError(f"{assignment} prompt hash mismatch")
        if request.get("accepted") != response.get("accepted") or not isinstance(
            request.get("accepted"), bool
        ):
            raise ValueError(f"{assignment} request/response accepted state differs")
        if request.get("stage") == "mapping":
            if request.get("superseded") != response.get("superseded") or not isinstance(
                request.get("superseded"), bool
            ):
                raise ValueError(
                    f"{assignment} mapping request/response superseded state differs"
                )
            if bool(request["accepted"]) == bool(request["superseded"]):
                raise ValueError(
                    f"{assignment} mapping response must be exactly accepted or superseded"
                )
        if "provider_error" in response:
            _validate_provider_error_row(response)
            if request.get("accepted") is not False:
                raise ValueError(f"{assignment} provider error request cannot be accepted")
            continue
        content = response.get("content")
        if not isinstance(content, str) or response.get("response_sha256") != _sha256_bytes(
            content.encode("utf-8")
        ):
            raise ValueError(f"{assignment} response hash mismatch")
        validate_kimi_model_identity(EXPECTED_MODEL, str(response.get("model")))
        if not request["accepted"]:
            continue
        response_hashes.add(str(response["response_sha256"]))
        if request.get("stage") == "mapping":
            try:
                mapping_payload = json.loads(content)
            except json.JSONDecodeError as error:
                raise ValueError(f"{assignment} accepted mapping response is malformed") from error
            if not isinstance(mapping_payload, dict) or set(mapping_payload) != {"mapping"}:
                raise ValueError(f"{assignment} accepted mapping response is malformed")
            accepted_mapping_batches.append(
                (int(request["mapping_request_index"]), mapping_payload["mapping"])
            )
        elif request.get("stage") == "dialogue":
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{assignment} dialogue request lacks messages")
            user_payload = json.loads(str(messages[-1]["content"]))
            expected = user_payload.get("events")
            if not isinstance(expected, list):
                raise ValueError(f"{assignment} dialogue request lacks expected events")
            validate_generation_response(
                content,
                expected,
                turn_pairs_per_event=int(user_payload["turn_pairs_per_event"]),
                minimum_words_per_turn=int(user_payload["minimum_words_per_turn"]),
            )
            accepted_dialogue_batches.append(
                (
                    int(request["request_index"]),
                    [str(row["event_id"]) for row in expected],
                )
            )
        else:
            raise ValueError(f"{assignment} accepted response has unknown stage")
    accepted_mapping = [
        row
        for _, batch in sorted(accepted_mapping_batches)
        for row in batch
    ]
    if accepted_mapping != manifest.get("surface_mapping"):
        raise ValueError(
            f"{assignment} accepted mapping batches differ from manifest mapping"
        )
    accepted_event_ids = [
        event_id
        for _, event_ids in sorted(accepted_dialogue_batches)
        for event_id in event_ids
    ]
    if accepted_event_ids != list(expected_event_ids):
        raise ValueError(f"{assignment} fresh dialogue responses do not cover parent event order")
    if manifest.get("prompt_sha256") != [row["prompt_sha256"] for row in requests]:
        raise ValueError(f"{assignment} manifest prompt hashes differ from requests")
    if manifest.get("response_sha256") != [
        _raw_provenance_hash(row) for row in responses
    ]:
        raise ValueError(f"{assignment} manifest response hashes differ from logs")
    return response_hashes


def _validate_materialized_variants(
    *,
    parent_hash: str,
    parent_artifacts: Mapping[str, bytes],
    parent_events: Sequence[Mapping[str, Any]],
    parent_queries: Sequence[Mapping[str, Any]],
    variant_paths: Mapping[str, Path],
    expected_bindings: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate completed A/B artifacts against one already-authenticated parent."""
    if len(parent_events) != EXPECTED_EVENT_COUNT:
        raise ValueError(
            f"variant validation requires exactly {EXPECTED_EVENT_COUNT} parent events, "
            f"got {len(parent_events)}"
        )
    expected_rows = expected_surface_mapping_rows(
        sorted({str(row["history_id"]) for row in parent_events})
    )
    variants = {}
    bindings = {}
    response_hashes = {}
    for assignment in ("A", "B"):
        path = variant_paths.get(assignment)
        if not isinstance(path, Path):
            raise ValueError(f"variant path is missing for {assignment}")
        manifest, manifest_hash, mapping, artifacts = _read_completed_variant(
            path, assignment, parent_hash
        )
        validate_surface_mapping_response(
            json.dumps({"mapping": mapping}),
            expected_rows,
            _parent_surface_phrases(parent_events, parent_queries),
        )
        child_events = _load_jsonl(artifacts["events.jsonl"], f"{assignment} events.jsonl")
        child_queries = _load_jsonl(artifacts["queries.jsonl"], f"{assignment} queries.jsonl")
        validate_parent_preservation(
            parent_events, child_events, parent_queries, child_queries
        )
        _validate_no_stale_parent_phrases(child_events, child_queries, mapping)
        for name, payload in parent_artifacts.items():
            if name not in _REBUILT_ARTIFACTS and artifacts.get(name) != payload:
                raise ValueError(
                    f"{assignment} non-visible artifact differs from parent: {name}"
                )
        response_hashes[assignment] = _validate_fresh_provenance(
            assignment,
            manifest,
            artifacts,
            parent_artifacts,
            [str(row["event_id"]) for row in parent_events],
        )
        binding = {
            "generation_manifest_sha256": manifest_hash,
            "generation_controls_sha256": manifest["generation_controls_sha256"],
            "surface_mapping_sha256": manifest["surface_mapping_sha256"],
            "artifact_sha256": manifest["artifact_sha256"],
        }
        if expected_bindings is not None and expected_bindings.get(assignment) != binding:
            raise ValueError(f"pair gate sibling binding differs for {assignment}")
        bindings[assignment] = binding
        variants[assignment] = {
            "manifest": manifest,
            "mapping": mapping,
            "path": path,
            "artifacts": artifacts,
        }
    if (
        variants["A"]["manifest"]["generation_controls_sha256"]
        != variants["B"]["manifest"]["generation_controls_sha256"]
    ):
        raise ValueError("A/B generation controls differ")
    validate_paired_surface_mappings(
        variants["A"]["mapping"], variants["B"]["mapping"]
    )
    expected_b_forbidden = sorted(
        {
            _normalize_phrase(str(row["target_phrase"]))
            for row in variants["A"]["mapping"]
        }
    )
    for assignment in ("A", "B"):
        requests = _load_jsonl(
            variants[assignment]["artifacts"]["requests.jsonl"],
            f"{assignment} requests.jsonl",
        )
        for request in requests:
            if request.get("stage") != "mapping":
                continue
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{assignment} mapping request lacks messages")
            payload = json.loads(str(messages[-1]["content"]))
            cross_forbidden = payload.get("cross_assignment_forbidden_targets")
            if assignment == "A" and cross_forbidden is not None:
                raise ValueError("A mapping request improperly exposes sibling targets")
            if assignment == "B" and cross_forbidden != expected_b_forbidden:
                raise ValueError("B mapping request has invalid flat A target conditioning")
    if response_hashes["A"] & response_hashes["B"]:
        raise ValueError("A/B accepted Kimi response logs overlap")
    return bindings


def authenticate_pair_gate(
    gate_path: Path,
    expected_gate_sha256: str,
    *,
    dataset_dir: Path | None = None,
    expected_assignment: str | None = None,
) -> dict[str, Any]:
    """Authenticate the gate and re-prove parent/A/B pair contracts from artifacts."""
    gate_path = Path(gate_path)
    try:
        gate_bytes = gate_path.read_bytes()
        gate = json.loads(gate_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load pair gate {gate_path}: {error}") from error
    actual_gate_hash = _sha256_bytes(gate_bytes)
    if actual_gate_hash != expected_gate_sha256:
        raise ValueError(
            f"pair gate hash mismatch: expected {expected_gate_sha256}, got {actual_gate_hash}"
        )
    if (
        not isinstance(gate, dict)
        or gate.get("status") != "completed"
        or gate.get("version") != PAIR_GATE_VERSION
        or gate.get("method") != DERIVATION_METHOD
    ):
        raise ValueError("pair gate is not a completed supported gate")
    if gate.get("assurances") != GATE_ASSURANCES:
        raise ValueError("pair gate assurance scope is invalid")
    paths = gate.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != {"parent", "A", "B"}:
        raise ValueError("pair gate path bindings are incomplete")
    resolved = {
        label: (gate_path.parent / str(relative)).resolve()
        for label, relative in paths.items()
    }
    _, parent_hash, parent_artifacts = _authenticate_parent(resolved["parent"])
    if gate.get("parent_generation_manifest_sha256") != parent_hash:
        raise ValueError("pair gate parent manifest binding differs")
    parent_events = _load_jsonl(parent_artifacts["events.jsonl"], "parent events.jsonl")
    parent_queries = _load_jsonl(parent_artifacts["queries.jsonl"], "parent queries.jsonl")
    _validate_materialized_variants(
        parent_hash=parent_hash,
        parent_artifacts=parent_artifacts,
        parent_events=parent_events,
        parent_queries=parent_queries,
        variant_paths={"A": resolved["A"], "B": resolved["B"]},
        expected_bindings=gate.get("variants"),
    )
    if dataset_dir is not None:
        if expected_assignment not in {"A", "B"}:
            raise ValueError("expected_assignment must be A or B when dataset_dir is supplied")
        if Path(dataset_dir).resolve() != resolved[expected_assignment]:
            raise ValueError("benchmark dataset is not the gate-bound assignment path")
    return {
        "pair_gate_sha256": actual_gate_hash,
        "parent_generation_manifest_sha256": parent_hash,
        "parent_path": resolved["parent"],
        "assignment_manifest_sha256": {
            assignment: gate["variants"][assignment]["generation_manifest_sha256"]
            for assignment in ("A", "B")
        },
    }


def write_pair_gate(
    parent_dir: Path,
    output_a: Path,
    output_b: Path,
    gate_path: Path,
    *,
    parent_manifest_hash: str,
    variant_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a completed gate from already-validated parent and variant bindings."""
    gate_path = Path(gate_path)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate = {
        "status": "completed",
        "version": PAIR_GATE_VERSION,
        "method": DERIVATION_METHOD,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "parent": os.path.relpath(Path(parent_dir).resolve(), gate_path.parent.resolve()),
            "A": os.path.relpath(Path(output_a).resolve(), gate_path.parent.resolve()),
            "B": os.path.relpath(Path(output_b).resolve(), gate_path.parent.resolve()),
        },
        "parent_generation_manifest_sha256": parent_manifest_hash,
        "variants": dict(variant_bindings),
        "assurances": GATE_ASSURANCES,
    }
    _atomic_write_bytes(gate_path, _json_bytes(gate))
    return gate


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    """Load an existing provenance JSONL file from authenticated bytes."""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CacheIntegrityError(f"cannot read cached provenance {path}: {error}") from error
    return _load_jsonl(payload, path.name)


def _validate_resumable_log_prefixes(
    assignment: str,
    requests: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
) -> None:
    """Validate well-formed request/response journals that may lead the manifest."""
    def key(row: Mapping[str, Any]) -> tuple[str, int, int]:
        return (
            str(row.get("stage")),
            int(row.get("request_index", -1)),
            int(row.get("attempt_index", -1)),
        )

    request_by_key = {}
    for row in requests:
        row_key = key(row)
        if row_key in request_by_key:
            raise CacheIntegrityError(f"duplicate cached request key {row_key}")
        if row.get("assignment") != assignment:
            raise CacheIntegrityError(f"cached request {row_key} has wrong assignment")
        messages = row.get("messages")
        if not isinstance(messages, list) or row.get("prompt_sha256") != _stable_hash(messages):
            raise CacheIntegrityError(f"cached request prompt hash mismatch for {row_key}")
        request_by_key[row_key] = row
    response_keys = set()
    for row in responses:
        row_key = key(row)
        if row_key in response_keys or row_key not in request_by_key:
            raise CacheIntegrityError(f"cached response key is invalid: {row_key}")
        response_keys.add(row_key)
        request = request_by_key[row_key]
        for field in (
            "assignment",
            "stage",
            "request_index",
            "attempt_index",
            "seed",
            "requested_model",
            "max_tokens",
            "enable_thinking",
            "mapping_request_index",
            "history_ids",
            "event_ids",
        ):
            if row.get(field) != request.get(field):
                raise CacheIntegrityError(
                    f"cached response {row_key} differs from request on {field}"
                )
        if "provider_error" in row:
            try:
                _validate_provider_error_row(row)
            except ValueError as error:
                raise CacheIntegrityError(
                    f"cached provider error mismatch for {row_key}: {error}"
                ) from error
            if request.get("accepted") is not False:
                raise CacheIntegrityError(
                    f"cached provider error request is accepted for {row_key}"
                )
            if request.get("stage") == "mapping" and request.get(
                "superseded"
            ) is not True:
                raise CacheIntegrityError(
                    f"cached provider error request is not superseded for {row_key}"
                )
            continue
        content = row.get("content")
        if not isinstance(content, str) or row.get("response_sha256") != _sha256_bytes(
            content.encode("utf-8")
        ):
            raise CacheIntegrityError(f"cached response hash mismatch for {row_key}")


def _load_or_initialize_assignment_state(
    *,
    output_dir: Path,
    assignment: str,
    config: SurfacePairConfig,
    parent_manifest_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Create or authenticate one resumable assignment state directory."""
    manifest_path = output_dir / "generation_manifest.json"
    requests_path = output_dir / "requests.jsonl"
    responses_path = output_dir / "raw_responses.jsonl"
    if output_dir.exists() and config.resume_existing:
        for temporary_path in output_dir.glob(".*.tmp"):
            temporary_path.unlink()
    has_state = output_dir.exists() and any(output_dir.iterdir())
    if has_state and not config.resume_existing:
        raise ValueError(f"output directory must be absent or empty: {output_dir}")
    if not has_state:
        output_dir.mkdir(parents=True, exist_ok=True)
        controls = _generation_controls(config)
        manifest = {
            "status": "running",
            "completion_status": "running",
            "method": DERIVATION_METHOD,
            "assignment": assignment,
            "seed": SURFACE_ASSIGNMENT_SEEDS[assignment],
            "requested_model": config.model,
            "generation_controls": controls,
            "generation_controls_sha256": _stable_hash(controls),
            "request_parameters": _request_parameters(config),
            "resume_existing": config.resume_existing,
            "resumed_response_count": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "parent": {
                "corpus_name": config.parent_dir.name,
                "generation_manifest_sha256": parent_manifest_hash,
            },
        }
        _persist_generation_state(output_dir, manifest, (), ())
        return manifest, [], [], False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheIntegrityError(f"cannot read cached manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise CacheIntegrityError("cached generation manifest must be an object")
    expected_identity = {
        "method": DERIVATION_METHOD,
        "assignment": assignment,
        "seed": SURFACE_ASSIGNMENT_SEEDS[assignment],
        "requested_model": config.model,
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise CacheIntegrityError(f"cached manifest mismatch on {field}")
    controls = _generation_controls(config)
    if (
        manifest.get("generation_controls") != controls
        or manifest.get("generation_controls_sha256") != _stable_hash(controls)
        or manifest.get("request_parameters") != _request_parameters(config)
    ):
        raise CacheIntegrityError("cached manifest generation controls mismatch")
    parent = manifest.get("parent")
    if not isinstance(parent, Mapping) or parent.get(
        "generation_manifest_sha256"
    ) != parent_manifest_hash:
        raise CacheIntegrityError("cached manifest parent identity mismatch")
    requests = _load_jsonl_file(requests_path)
    responses = _load_jsonl_file(responses_path)
    recorded = manifest.get("provenance_artifact_sha256")
    completed = manifest.get("status") == "completed"
    if not completed:
        def provenance_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
            return (
                str(row.get("stage")),
                int(row.get("request_index", -1)),
                int(row.get("attempt_index", -1)),
            )

        response_keys = {provenance_key(row) for row in responses}
        dangling = [row for row in requests if provenance_key(row) not in response_keys]
        if manifest.get("status") == "failed" and dangling:
            error = manifest.get("error")
            transient_type = error.get("type") if isinstance(error, Mapping) else None
            exact_trailing_dangling = (
                len(dangling) == 1
                and len(requests) == len(responses) + 1
                and dangling[0] is requests[-1]
            )
            actual_hashes = {
                "requests.jsonl": _sha256_bytes(requests_path.read_bytes()),
                "raw_responses.jsonl": _sha256_bytes(responses_path.read_bytes()),
            }
            artifact_hashes = manifest.get("artifact_sha256")
            hashes_match = (
                isinstance(recorded, Mapping)
                and all(recorded.get(name) == digest for name, digest in actual_hashes.items())
                and isinstance(artifact_hashes, Mapping)
                and all(
                    artifact_hashes.get(name) == digest
                    for name, digest in actual_hashes.items()
                )
            )
            if (
                transient_type not in _TRANSIENT_PROVIDER_ERROR_NAMES
                or not exact_trailing_dangling
                or not hashes_match
                or not isinstance(error.get("message"), str)
            ):
                raise CacheIntegrityError(
                    "failed manifest dangling request is not an exact transient trailing call"
                )
            dangling_request = dangling[0]
            dangling_request["accepted"] = False
            if dangling_request.get("stage") == "mapping":
                dangling_request["superseded"] = True
            responses.append(
                _provider_error_row(
                    dangling_request,
                    {"type": str(transient_type), "message": str(error["message"])},
                    reconciled_from_failed_manifest=True,
                )
            )
    if completed:
        if not isinstance(recorded, Mapping):
            raise CacheIntegrityError("completed manifest lacks provenance artifact hashes")
        actual_requests = _sha256_bytes(requests_path.read_bytes())
        actual_responses = _sha256_bytes(responses_path.read_bytes())
        if recorded.get("requests.jsonl") != actual_requests:
            raise CacheIntegrityError("cached request log hash mismatch")
        if recorded.get("raw_responses.jsonl") != actual_responses:
            raise CacheIntegrityError("cached response log hash mismatch")
    else:
        _validate_resumable_log_prefixes(assignment, requests, responses)
    if not completed:
        manifest["status"] = "running"
        manifest["completion_status"] = "running"
        manifest.pop("failed_at", None)
        manifest.pop("error", None)
        _persist_generation_state(output_dir, manifest, requests, responses)
    return manifest, requests, responses, completed


def generate_surface_pair(
    config: SurfacePairConfig,
    client: CompletionClient,
    *,
    dialogue_client_factory: Callable[[str], CompletionClient] | None = None,
) -> dict[str, Any]:
    """Map both assignments first, then realize dialogue and gate the completed pair."""
    if config.resume_existing and config.gate_path.parent.exists():
        for temporary_path in config.gate_path.parent.glob(
            f".{config.gate_path.name}.*.tmp"
        ):
            temporary_path.unlink()
    if config.gate_path.exists():
        if not config.resume_existing:
            raise ValueError(f"pair gate path must not already exist: {config.gate_path}")
        gate_bytes = config.gate_path.read_bytes()
        gate = json.loads(gate_bytes)
        authentication = authenticate_pair_gate(
            config.gate_path, _sha256_bytes(gate_bytes)
        )
        if Path(authentication["parent_path"]).resolve() != config.parent_dir.resolve():
            raise CacheIntegrityError("completed gate parent path differs from configuration")
        resolved_paths = {
            assignment: (config.gate_path.parent / str(gate["paths"][assignment])).resolve()
            for assignment in ("A", "B")
        }
        if resolved_paths != {
            "A": config.output_a.resolve(),
            "B": config.output_b.resolve(),
        }:
            raise CacheIntegrityError("completed gate assignment paths differ from configuration")
        manifests = {
            assignment: json.loads(
                (path / "generation_manifest.json").read_text(encoding="utf-8")
            )
            for assignment, path in resolved_paths.items()
        }
        controls_hash = _stable_hash(_generation_controls(config))
        if any(
            manifest.get("generation_controls_sha256") != controls_hash
            for manifest in manifests.values()
        ):
            raise CacheIntegrityError(
                "completed gate generation controls differ from configuration"
            )
        return {"manifests": manifests, "gate": gate}
    output_paths = {"A": config.output_a, "B": config.output_b}

    parent_manifest, parent_manifest_hash, parent_artifacts = _authenticate_parent(
        config.parent_dir
    )
    parent_events = _load_jsonl(parent_artifacts["events.jsonl"], "parent events.jsonl")
    parent_queries = _load_jsonl(parent_artifacts["queries.jsonl"], "parent queries.jsonl")
    history_ids = sorted({str(row.get("history_id", "")) for row in parent_events})
    if not history_ids or any(not value for value in history_ids):
        raise ValueError("parent events contain incomplete history IDs")
    if len(parent_events) != EXPECTED_EVENT_COUNT:
        raise ValueError(
            f"paired generation requires exactly {EXPECTED_EVENT_COUNT} parent events, "
            f"got {len(parent_events)}"
        )

    running_manifests: dict[str, dict[str, Any]] = {}
    request_rows: dict[str, list[dict[str, Any]]] = {}
    raw_rows: dict[str, list[dict[str, Any]]] = {}
    completed_assignments = {}
    for assignment, output_dir in output_paths.items():
        (
            running_manifests[assignment],
            request_rows[assignment],
            raw_rows[assignment],
            completed_assignments[assignment],
        ) = _load_or_initialize_assignment_state(
            output_dir=output_dir,
            assignment=assignment,
            config=config,
            parent_manifest_hash=parent_manifest_hash,
        )
    if config.dialogue_assignment_concurrency == 2 and dialogue_client_factory is None:
        raise ValueError(
            "dialogue_client_factory is required when dialogue_assignment_concurrency is 2"
        )

    expected_rows = expected_surface_mapping_rows(history_ids)
    parent_phrases = _parent_surface_phrases(parent_events, parent_queries)
    mappings: dict[str, list[dict[str, str]]] = {}
    try:
        mappings["A"] = _request_mapping_candidate(
            client,
            output_dir=output_paths["A"],
            running_manifest=running_manifests["A"],
            assignment="A",
            expected_rows=expected_rows,
            parent_phrases=parent_phrases,
            cross_assignment_forbidden_targets=None,
            config=config,
            request_rows=request_rows["A"],
            raw_rows=raw_rows["A"],
            read_only_cache=completed_assignments["A"],
        )
        validate_whole_mapping_naturalness(mappings["A"], parent_phrases)
        a_forbidden_targets = sorted(
            {_normalize_phrase(str(row["target_phrase"])) for row in mappings["A"]}
        )
        mappings["B"] = _request_mapping_candidate(
            client,
            output_dir=output_paths["B"],
            running_manifest=running_manifests["B"],
            assignment="B",
            expected_rows=expected_rows,
            parent_phrases=parent_phrases,
            cross_assignment_forbidden_targets=a_forbidden_targets,
            config=config,
            request_rows=request_rows["B"],
            raw_rows=raw_rows["B"],
            read_only_cache=completed_assignments["B"],
        )
        validate_whole_mapping_naturalness(mappings["B"], parent_phrases)
        validate_paired_surface_mappings(mappings["A"], mappings["B"])
    except CacheIntegrityError:
        raise
    except Exception as failure:
        for assignment, output_dir in output_paths.items():
            failed_manifest = {
                **running_manifests[assignment],
                "status": "failed",
                "completion_status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(failure).__name__, "message": str(failure)},
                "prompt_sha256": [
                    row["prompt_sha256"] for row in request_rows[assignment]
                ],
                "response_sha256": [
                    _raw_provenance_hash(row) for row in raw_rows[assignment]
                ],
                "artifact_sha256": {
                    "requests.jsonl": _sha256_bytes(
                        (output_dir / "requests.jsonl").read_bytes()
                    ),
                    "raw_responses.jsonl": _sha256_bytes(
                        (output_dir / "raw_responses.jsonl").read_bytes()
                    ),
                },
            }
            _atomic_write_bytes(
                output_dir / "generation_manifest.json", _json_bytes(failed_manifest)
            )
        raise failure

    for assignment in ("A", "B"):
        if completed_assignments[assignment]:
            continue
        for row in request_rows[assignment]:
            if row.get("stage") == "mapping":
                row["accepted"] = not bool(row.get("superseded"))
        for row in raw_rows[assignment]:
            if row.get("stage") == "mapping":
                row["accepted"] = not bool(row.get("superseded"))
        _persist_generation_state(
            output_paths[assignment],
            running_manifests[assignment],
            request_rows[assignment],
            raw_rows[assignment],
        )

    manifests: dict[str, dict[str, Any]] = {}
    try:
        incomplete_assignments = [
            assignment
            for assignment in ("A", "B")
            if not completed_assignments[assignment]
        ]
        for assignment in ("A", "B"):
            if completed_assignments[assignment]:
                completed_manifest, _, completed_mapping, _ = _read_completed_variant(
                    output_paths[assignment], assignment, parent_manifest_hash
                )
                if completed_mapping != mappings[assignment]:
                    raise CacheIntegrityError(
                        f"completed {assignment} mapping differs from resumed pair mapping"
                    )
                manifests[assignment] = completed_manifest

        assignment_clients = {
            assignment: (
                dialogue_client_factory(assignment)
                if dialogue_client_factory is not None
                else client
            )
            for assignment in incomplete_assignments
        }
        if (
            config.dialogue_assignment_concurrency == 2
            and len(assignment_clients) == 2
            and assignment_clients["A"] is assignment_clients["B"]
        ):
            raise ValueError(
                "dialogue_client_factory must return distinct assignment-local clients"
            )

        def materialize(assignment: str) -> dict[str, Any]:
            """Materialize one assignment using only its client and output journal."""
            return _generate_assignment(
                output_paths[assignment],
                assignment,
                config,
                assignment_clients[assignment],
                parent_dir=config.parent_dir,
                parent_manifest=parent_manifest,
                parent_manifest_hash=parent_manifest_hash,
                parent_artifacts=parent_artifacts,
                parent_events=parent_events,
                parent_queries=parent_queries,
                mapping=mappings[assignment],
                mapping_request_rows=request_rows[assignment],
                mapping_raw_rows=raw_rows[assignment],
                running_manifest=running_manifests[assignment],
            )

        if config.dialogue_assignment_concurrency == 1:
            for assignment in incomplete_assignments:
                manifests[assignment] = materialize(assignment)
        else:
            worker_failure: BaseException | None = None
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="persona-surface-dialogue"
            ) as executor:
                futures = {
                    assignment: executor.submit(materialize, assignment)
                    for assignment in incomplete_assignments
                }
                done, pending = wait(futures.values(), return_when=FIRST_EXCEPTION)
                failed = [
                    future
                    for future in done
                    if not future.cancelled() and future.exception() is not None
                ]
                if failed:
                    worker_failure = failed[0].exception()
                    for future in pending:
                        future.cancel()
            for assignment in incomplete_assignments:
                future = futures[assignment]
                if not future.cancelled() and future.exception() is None:
                    manifests[assignment] = future.result()
            if worker_failure is not None:
                raise worker_failure
    except Exception as error:
        for assignment, output_dir in output_paths.items():
            manifest_path = output_dir / "generation_manifest.json"
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if current.get("status") == "running":
                current.update(
                    {
                        "status": "failed",
                        "completion_status": "failed",
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "error": {
                            "type": type(error).__name__,
                            "message": "paired dialogue materialization aborted: " + str(error),
                        },
                    }
                )
                _atomic_write_bytes(manifest_path, _json_bytes(current))
        raise

    variant_bindings = _validate_materialized_variants(
        parent_hash=parent_manifest_hash,
        parent_artifacts=parent_artifacts,
        parent_events=parent_events,
        parent_queries=parent_queries,
        variant_paths={"A": config.output_a, "B": config.output_b},
    )
    gate = write_pair_gate(
        config.parent_dir,
        config.output_a,
        config.output_b,
        config.gate_path,
        parent_manifest_hash=parent_manifest_hash,
        variant_bindings=variant_bindings,
    )
    return {"manifests": manifests, "gate": gate}


def _required_env(environ: Mapping[str, str], name: str) -> str:
    """Resolve one required non-empty environment value."""
    value = environ.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"required environment variable {name} is not set")
    return value


def load_surface_pair_config(
    path: Path, *, environ: Mapping[str, str] | None = None
) -> SurfacePairConfig:
    """Load strict paired generation controls with all paths/provider values from env."""
    environ = os.environ if environ is None else environ
    try:
        root = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load paired generation config {path}: {error}") from error
    if not isinstance(root, Mapping) or not isinstance(root.get("generation"), Mapping):
        raise ValueError("paired generation config requires a generation object")
    raw = root["generation"]
    expected = {
        "parent_dir_env",
        "output_a_env",
        "output_b_env",
        "gate_path_env",
        "endpoint_env",
        "api_key_env",
        "model_env",
        "timeout_seconds",
        "mapping_max_tokens",
        "dialogue_max_tokens",
        "mapping_histories_per_request",
        "dialogue_assignment_concurrency",
        "events_per_request",
        "turn_pairs_per_event",
        "minimum_words_per_turn",
        "max_validation_attempts",
        "resume_existing",
        "enable_thinking",
    }
    if set(raw) != expected:
        raise ValueError(
            f"paired generation keys mismatch: missing={sorted(expected - raw.keys())}, "
            f"unknown={sorted(raw.keys() - expected)}"
        )
    env_names = {}
    for field in (
        "parent_dir_env",
        "output_a_env",
        "output_b_env",
        "gate_path_env",
        "endpoint_env",
        "api_key_env",
        "model_env",
    ):
        name = raw[field]
        if not isinstance(name, str) or not name:
            raise ValueError(f"generation.{field} must be a non-empty string")
        env_names[field] = name
    return SurfacePairConfig(
        parent_dir=Path(_required_env(environ, env_names["parent_dir_env"])),
        output_a=Path(_required_env(environ, env_names["output_a_env"])),
        output_b=Path(_required_env(environ, env_names["output_b_env"])),
        gate_path=Path(_required_env(environ, env_names["gate_path_env"])),
        endpoint=_required_env(environ, env_names["endpoint_env"]),
        api_key=_required_env(environ, env_names["api_key_env"]),
        model=_required_env(environ, env_names["model_env"]),
        timeout_seconds=raw["timeout_seconds"],
        mapping_max_tokens=raw["mapping_max_tokens"],
        dialogue_max_tokens=raw["dialogue_max_tokens"],
        mapping_histories_per_request=raw["mapping_histories_per_request"],
        dialogue_assignment_concurrency=raw["dialogue_assignment_concurrency"],
        events_per_request=raw["events_per_request"],
        turn_pairs_per_event=raw["turn_pairs_per_event"],
        minimum_words_per_turn=raw["minimum_words_per_turn"],
        max_validation_attempts=raw["max_validation_attempts"],
        resume_existing=raw["resume_existing"],
        enable_thinking=raw["enable_thinking"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured Kimi pair-conditioned fixed-assignment generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_surface_pair_config(args.config)
    def client_factory(_assignment: str) -> CompletionClient:
        """Create one isolated OpenAI SDK client for a generation stream."""
        return OpenAICompletionClient(
            config.endpoint,
            config.api_key,
            config.timeout_seconds,
            json_mode=True,
            enable_thinking=config.enable_thinking,
        )

    generate_surface_pair(
        config,
        client_factory("mapping"),
        dialogue_client_factory=client_factory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
