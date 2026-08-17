"""Schedule authenticated persona conflicts across causal phase and token distance."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from experiments._continual_memory_config import HuggingFaceTokenizer, TokenizerConfig
from experiments._continual_memory_episodes import TURN_SEPARATOR, _serialized_turn_increment
from experiments.interleaved_conversation import build_interleaved_schedule, schedule_metrics
from experiments.persona_conversation_generator import SOURCE_ARTIFACTS
from experiments.persona_surface_derivation import (
    DERIVATION_ALGORITHM,
    DERIVATION_VERSION,
    ROLE_VARIANT_RULES,
    build_surface_mapping,
    transform_surface_artifacts,
)
from experiments.synthetic_temporal_preferences import resolve_query


SCHEDULE_VERSION = "persona-interference-schedule.v1"


class ScheduleTokenizer(Protocol):
    """Tokenizer boundary for exact stream-distance accounting."""

    def encode(self, text: str) -> list[Any]:
        """Encode model-visible text."""

    def metadata(self) -> Mapping[str, Any]:
        """Return tokenizer provenance."""


@dataclass(frozen=True)
class ScheduleConfig:
    """Validated controls for one persona interference schedule."""

    source_split: str
    source_profile: str
    source_manifest_sha256: str
    seed: int
    concurrent_accounts: int
    min_segment_events: int
    max_segment_events: int
    query_suffixes: tuple[str, ...]
    token_distance_thresholds: tuple[int, ...]
    minimum_stream_tokens: int = 0

    def __post_init__(self) -> None:
        """Reject incomplete or silently clamped scheduling requests."""
        if (
            not isinstance(self.source_split, str)
            or not self.source_split
            or not isinstance(self.source_profile, str)
            or not self.source_profile
        ):
            raise ValueError("source_split and source_profile must be non-empty")
        if not isinstance(self.source_manifest_sha256, str) or (
            len(self.source_manifest_sha256) != 64
        ) or any(
            character not in "0123456789abcdef" for character in self.source_manifest_sha256
        ):
            raise ValueError("source_manifest_sha256 must be a lowercase SHA-256 digest")
        integer_values = {
            "seed": self.seed,
            "concurrent_accounts": self.concurrent_accounts,
            "min_segment_events": self.min_segment_events,
            "max_segment_events": self.max_segment_events,
            "minimum_stream_tokens": self.minimum_stream_tokens,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.minimum_stream_tokens < 0:
            raise ValueError("minimum_stream_tokens must be non-negative")
        if self.concurrent_accounts < 2:
            raise ValueError("concurrent_accounts must be at least two")
        if self.min_segment_events < 1 or self.max_segment_events < self.min_segment_events:
            raise ValueError("segment bounds must satisfy 1 <= minimum <= maximum")
        if not isinstance(self.query_suffixes, tuple) or not self.query_suffixes or any(
            not isinstance(suffix, str) or not suffix for suffix in self.query_suffixes
        ):
            raise ValueError("query_suffixes must contain non-empty values")
        if (
            not isinstance(self.token_distance_thresholds, tuple)
            or not self.token_distance_thresholds
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.token_distance_thresholds
            )
            or tuple(sorted(set(self.token_distance_thresholds)))
            != self.token_distance_thresholds
        ):
            raise ValueError("token_distance_thresholds must be unique increasing positives")


def _read_jsonl_bytes(payload: bytes, path: Path) -> list[dict[str, Any]]:
    """Parse authenticated object-valued JSONL bytes and name malformed rows."""
    rows = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 in {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"expected object in {path} line {line_number}")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of already-read bytes."""
    return hashlib.sha256(payload).hexdigest()


def _stable_sha256(value: Any) -> str:
    """Hash one value using canonical JSON serialization."""
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _authenticate_dataset(
    dataset_dir: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any] | None]:
    """Bind scheduling to one completed Kimi corpus or explicit derived chain."""
    manifest_path = dataset_dir / "generation_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read generation manifest {manifest_path}: {error}") from error
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "generation manifest hash mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"cannot load generation manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, Mapping) or manifest.get("status") != "completed":
        raise ValueError("persona interference scheduling requires a completed manifest")
    recorded = manifest.get("artifact_sha256")
    if not isinstance(recorded, Mapping):
        raise ValueError("generation manifest lacks artifact_sha256")
    model_identity = manifest.get("model_identity")
    derivation = manifest.get("derivation")
    parent = manifest.get("parent")
    if isinstance(model_identity, str) and "kimi" in model_identity.casefold():
        derived_provenance = None
        authenticated_parent = None
    else:
        if not isinstance(derivation, Mapping) or not isinstance(parent, Mapping):
            raise ValueError(
                f"generation manifest is neither Kimi-authored nor explicitly derived: {model_identity!r}"
            )
        if (
            derivation.get("algorithm") != DERIVATION_ALGORITHM
            or derivation.get("version") != DERIVATION_VERSION
            or isinstance(derivation.get("seed"), bool)
            or not isinstance(derivation.get("seed"), int)
        ):
            raise ValueError("derived generation manifest has an invalid derivation contract")
        parent_manifest_sha256 = parent.get("generation_manifest_sha256")
        parent_artifacts = parent.get("artifact_sha256")
        kimi_provenance = parent.get("kimi_provenance")
        parent_corpus_name = parent.get("corpus_name")
        if not isinstance(parent_manifest_sha256, str) or len(parent_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in parent_manifest_sha256
        ):
            raise ValueError("derived generation manifest has an invalid parent manifest hash")
        if not isinstance(parent_artifacts, Mapping) or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in parent_artifacts.items()
        ):
            raise ValueError("derived generation manifest has invalid parent artifact hashes")
        required_parent_artifacts = {*SOURCE_ARTIFACTS, "dialogue.jsonl"}
        if not required_parent_artifacts.issubset(parent_artifacts):
            raise ValueError("derived generation manifest has an incomplete parent artifact chain")
        if set(parent_artifacts) != set(recorded):
            raise ValueError(
                "derived generation parent and child artifact hash sets differ"
            )
        if not isinstance(kimi_provenance, Mapping) or kimi_provenance.get("status") != "completed":
            raise ValueError("derived generation manifest lacks completed Kimi parent provenance")
        parent_model = kimi_provenance.get("model_identity")
        if not isinstance(parent_model, str) or "kimi" not in parent_model.casefold():
            raise ValueError(f"derived generation parent is not Kimi-authored: {parent_model!r}")
        if (
            not isinstance(parent_corpus_name, str)
            or not parent_corpus_name
            or Path(parent_corpus_name).name != parent_corpus_name
        ):
            raise ValueError("derived generation manifest has an invalid parent corpus name")
        mapping = manifest.get("surface_mapping")
        mapping_sha256 = manifest.get("surface_mapping_sha256")
        if not isinstance(mapping, list) or _stable_sha256(mapping) != mapping_sha256:
            raise ValueError("derived generation manifest surface mapping hash mismatch")
        expected_replacement_contract = [
            {"mode": mode, "pattern": pattern, "source_phrase": source}
            for source, pattern, mode in ROLE_VARIANT_RULES
        ]
        if (
            manifest.get("visible_replacement_contract") != expected_replacement_contract
            or manifest.get("visible_replacement_contract_sha256")
            != _stable_sha256(ROLE_VARIANT_RULES)
        ):
            raise ValueError("derived generation manifest has an invalid visible replacement contract")

        resolved_dataset = dataset_dir.resolve()
        parent_dir = (resolved_dataset.parent / parent_corpus_name).resolve()
        if parent_dir.parent != resolved_dataset.parent or parent_dir == resolved_dataset:
            raise ValueError("derived generation parent must resolve to a sibling corpus")
        parent_manifest_path = parent_dir / "generation_manifest.json"
        try:
            actual_parent_manifest_bytes = parent_manifest_path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"cannot read actual parent generation manifest {parent_manifest_path}: {error}"
            ) from error
        actual_parent_manifest_sha256 = _sha256_bytes(actual_parent_manifest_bytes)
        if actual_parent_manifest_sha256 != parent_manifest_sha256:
            raise ValueError(
                "actual parent generation manifest hash mismatch: "
                f"expected {parent_manifest_sha256}, got {actual_parent_manifest_sha256}"
            )
        try:
            actual_parent_manifest = json.loads(actual_parent_manifest_bytes)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"cannot load actual parent generation manifest {parent_manifest_path}: {error}"
            ) from error
        if not isinstance(actual_parent_manifest, Mapping) or actual_parent_manifest.get(
            "status"
        ) != "completed":
            raise ValueError("actual parent generation manifest is not completed")
        actual_parent_model = actual_parent_manifest.get("model_identity")
        if not isinstance(actual_parent_model, str) or "kimi" not in actual_parent_model.casefold():
            raise ValueError(
                f"actual parent generation is not Kimi-authored: {actual_parent_model!r}"
            )
        actual_parent_recorded = actual_parent_manifest.get("artifact_sha256")
        if not isinstance(actual_parent_recorded, Mapping) or dict(actual_parent_recorded) != dict(
            parent_artifacts
        ):
            raise ValueError(
                "actual parent artifact hash manifest differs from derived provenance"
            )
        expected_kimi_provenance = {
            "generator_role": actual_parent_manifest.get("generator_role"),
            "model_identity": actual_parent_model,
            "prompt_schema_version": actual_parent_manifest.get("prompt_schema_version"),
            "provider_identity_assurance": actual_parent_manifest.get(
                "provider_identity_assurance"
            ),
            "status": actual_parent_manifest.get("status"),
        }
        if dict(kimi_provenance) != expected_kimi_provenance:
            raise ValueError("copied Kimi parent provenance differs from actual parent")
        actual_parent_artifacts = {}
        for name, expected in sorted(parent_artifacts.items()):
            if Path(name).name != name or name == "generation_manifest.json":
                raise ValueError(f"invalid actual parent artifact name: {name!r}")
            path = parent_dir / name
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise ValueError(f"cannot read actual parent artifact {path}: {error}") from error
            actual = _sha256_bytes(payload)
            if actual != expected:
                raise ValueError(f"actual parent artifact hash mismatch for {name}")
            actual_parent_artifacts[name] = payload
        authenticated_parent = {
            "path": parent_dir,
            "generation_manifest_sha256": actual_parent_manifest_sha256,
            "artifact_bytes": actual_parent_artifacts,
        }
        parent_events = _read_jsonl_bytes(
            actual_parent_artifacts["events.jsonl"], parent_dir / "events.jsonl"
        )
        history_ids = sorted({str(event.get("history_id", "")) for event in parent_events})
        if not history_ids or any(not history_id for history_id in history_ids):
            raise ValueError("authenticated parent events contain incomplete history identifiers")
        expected_mapping = build_surface_mapping(history_ids, derivation["seed"])
        if mapping != expected_mapping:
            raise ValueError("derived surface mapping does not match its declared seed")
        expected_child_artifacts = transform_surface_artifacts(
            actual_parent_artifacts, mapping
        )
        for name, expected_payload in sorted(expected_child_artifacts.items()):
            path = dataset_dir / name
            try:
                child_payload = path.read_bytes()
            except OSError as error:
                raise ValueError(f"cannot read derived corpus artifact {path}: {error}") from error
            if child_payload != expected_payload:
                if name == "queries.jsonl":
                    raise ValueError(
                        "derived query gold or causal contract differs from canonical transformation"
                    )
                raise ValueError(
                    f"derived canonical transformation mismatch for {name}"
                )
        derived_provenance = {
            "checkpoint_policy": "authenticated_parent_exact_indices",
            "derivation": dict(derivation),
            "parent_generation_manifest_sha256": parent_manifest_sha256,
            "parent_path": str(parent_dir),
        }
    hashes = {}
    artifact_bytes = {}
    for name in (*SOURCE_ARTIFACTS, "dialogue.jsonl"):
        path = dataset_dir / name
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read corpus artifact {path}: {error}") from error
        current = _sha256_bytes(payload)
        if recorded.get(name) != current:
            raise ValueError(f"artifact hash mismatch for {name}")
        hashes[name] = current
        artifact_bytes[name] = payload
    provenance = {
        "path": str(dataset_dir),
        "generation_model": model_identity,
        "generation_manifest_sha256": manifest_sha256,
        "artifact_sha256": hashes,
    }
    if derived_provenance is not None:
        provenance.update(derived_provenance)
    return provenance, artifact_bytes, authenticated_parent


def _surface_gold(
    resolved: str | None, history_events: Sequence[Mapping[str, Any]]
) -> str:
    """Map an internal causal answer to its model-visible surface value."""
    if resolved is None:
        return "UNKNOWN"
    surfaces = {
        str(event["surface_object"])
        for event in history_events
        if str(event.get("fact", {}).get("object", "")) == resolved
        and isinstance(event.get("surface_object"), str)
        and str(event["surface_object"]).strip()
    }
    if len(surfaces) != 1:
        raise ValueError(
            f"internal answer {resolved!r} has {len(surfaces)} surface mappings: {sorted(surfaces)}"
        )
    return next(iter(surfaces))


def _visible_context(turns: Sequence[Mapping[str, Any]], history_id: str) -> str:
    """Render target and unrelated turns without exposing latent event fields."""
    parts = []
    for turn in turns:
        label = (
            "Account holder conversation"
            if str(turn["history_id"]) == history_id
            else "Unrelated person's conversation, not the account holder"
        )
        parts.append(f"{label}:\n{turn['text']}")
    return TURN_SEPARATOR.join(parts)


def _visible_cumulative_tokens(
    turns: Sequence[Mapping[str, Any]], history_id: str, tokenizer: ScheduleTokenizer
) -> list[int]:
    """Count the exact labeled stream prefix visible for one target history."""
    visible_turns = [
        _visible_context([turn], history_id)
        for turn in turns
    ]
    cumulative = []
    total = 0
    for index, text in enumerate(visible_turns):
        total += _serialized_turn_increment(
            visible_turns[index - 1] if index else None,
            text,
            tokenizer,
        )
        cumulative.append(total)
    return cumulative


def _condition_row(
    *,
    query: Mapping[str, Any],
    phase: str,
    checkpoint: int,
    update_index: int,
    requested_distance: int | None,
    turns: Sequence[Mapping[str, Any]],
    cumulative: Sequence[int],
    history_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one causal, model-safe evaluation condition."""
    history_id = str(query["history_id"])
    if checkpoint < 1 or checkpoint > len(turns):
        raise ValueError(
            f"query {query.get('query_id')} has invalid checkpoint {checkpoint}"
        )
    if phase != "pre_update" and checkpoint <= update_index + 1:
        if phase != "post_update" or checkpoint != update_index + 1:
            raise ValueError(f"query {query.get('query_id')} checkpoint is not post-update")
    selected = list(turns[:checkpoint])
    target_prefix = [
        dict(turn["stream_event"])
        for turn in selected
        if str(turn["history_id"]) == history_id
    ]
    resolved = resolve_query(target_prefix, dict(query))
    gold = _surface_gold(resolved, history_events)
    update_tokens = cumulative[update_index]
    actual_distance = None if phase == "pre_update" else cumulative[checkpoint - 1] - update_tokens
    intervening = turns[update_index + 1 : checkpoint]
    context = _visible_context(selected, history_id)
    condition_suffix = phase if requested_distance is None else f"{phase}-{requested_distance}"
    return {
        "evaluation_input_id": f"{query['query_id']}:{condition_suffix}",
        "history_id": history_id,
        "query_id": str(query["query_id"]),
        "phase": phase,
        "checkpoint_turn_index": checkpoint,
        "relevant_update_turn_index": update_index,
        "relevant_update_event_id": str(turns[update_index]["stream_event"]["event_id"]),
        "requested_token_distance": requested_distance,
        "actual_token_distance": actual_distance,
        "distractor_turn_count_since_update": sum(
            str(turn["history_id"]) != history_id for turn in intervening
        ),
        "target_turn_count_since_update": sum(
            str(turn["history_id"]) == history_id for turn in intervening
        ),
        "query_text": str(query["surface_query_text"]),
        "gold": gold,
        "internal_gold": resolved,
        "selected_turn_ids": [str(turn["turn_id"]) for turn in selected],
        "selected_turn_count": len(selected),
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "_model_context": context,
    }


def build_evaluation_schedule(
    dataset_dir: Path,
    tokenizer: ScheduleTokenizer,
    config: ScheduleConfig,
) -> dict[str, Any]:
    """Build causal phase and token-distance conditions from an immutable corpus."""
    dataset_dir = Path(dataset_dir)
    provenance, artifacts, authenticated_parent = _authenticate_dataset(
        dataset_dir, config.source_manifest_sha256
    )
    events = [
        row
        for row in _read_jsonl_bytes(
            artifacts["events.jsonl"], dataset_dir / "events.jsonl"
        )
        if row.get("split") == config.source_split
        and row.get("hardness_profile") == config.source_profile
    ]
    queries = [
        row
        for row in _read_jsonl_bytes(
            artifacts["queries.jsonl"], dataset_dir / "queries.jsonl"
        )
        if row.get("split") == config.source_split
        and row.get("hardness_profile") == config.source_profile
        and str(row.get("query_id", "")).endswith(config.query_suffixes)
    ]
    if not events or not queries:
        raise ValueError("source split/profile/query filters selected no evaluation data")
    parent_events = None
    parent_queries = None
    if authenticated_parent is not None:
        parent_artifacts = authenticated_parent["artifact_bytes"]
        parent_path = Path(authenticated_parent["path"])
        parent_events = [
            row
            for row in _read_jsonl_bytes(
                parent_artifacts["events.jsonl"], parent_path / "events.jsonl"
            )
            if row.get("split") == config.source_split
            and row.get("hardness_profile") == config.source_profile
        ]
        parent_queries = [
            row
            for row in _read_jsonl_bytes(
                parent_artifacts["queries.jsonl"], parent_path / "queries.jsonl"
            )
            if row.get("split") == config.source_split
            and row.get("hardness_profile") == config.source_profile
            and str(row.get("query_id", "")).endswith(config.query_suffixes)
        ]
        latent_events = [
            {
                key: value
                for key, value in row.items()
                if key not in {"model_text", "surface_object"}
            }
            for row in events
        ]
        parent_latent_events = [
            {
                key: value
                for key, value in row.items()
                if key not in {"model_text", "surface_object"}
            }
            for row in parent_events
        ]
        if latent_events != parent_latent_events:
            raise ValueError("derived event latent structure differs from authenticated parent")
        latent_queries = [
            {
                key: value
                for key, value in row.items()
                if key not in {"query_text", "surface_query_text", "surface_gold"}
            }
            for row in queries
        ]
        parent_latent_queries = [
            {
                key: value
                for key, value in row.items()
                if key not in {"query_text", "surface_query_text", "surface_gold"}
            }
            for row in parent_queries
        ]
        if latent_queries != parent_latent_queries:
            raise ValueError(
                "derived query gold or causal contract differs from authenticated parent"
            )
    schedule = build_interleaved_schedule(
        events,
        tokenizer=tokenizer,
        seed=config.seed,
        concurrent_accounts=config.concurrent_accounts,
        min_segment_events=config.min_segment_events,
        max_segment_events=config.max_segment_events,
    )
    turns = list(schedule["turns"])
    checkpoint_turns = turns
    if parent_events is not None:
        parent_schedule = build_interleaved_schedule(
            parent_events,
            tokenizer=tokenizer,
            seed=config.seed,
            concurrent_accounts=config.concurrent_accounts,
            min_segment_events=config.min_segment_events,
            max_segment_events=config.max_segment_events,
        )
        checkpoint_turns = list(parent_schedule["turns"])
        turn_identity = [
            (
                str(turn["turn_id"]),
                str(turn["history_id"]),
                str(turn["stream_event"]["event_id"]),
            )
            for turn in turns
        ]
        parent_turn_identity = [
            (
                str(turn["turn_id"]),
                str(turn["history_id"]),
                str(turn["stream_event"]["event_id"]),
            )
            for turn in checkpoint_turns
        ]
        if turn_identity != parent_turn_identity:
            raise ValueError("derived interleaving differs from authenticated parent")
    metrics = schedule_metrics(schedule)
    if metrics["stream_token_count"] <= config.minimum_stream_tokens:
        raise ValueError(
            f"scheduled stream has {metrics['stream_token_count']} tokens; "
            f"must exceed {config.minimum_stream_tokens} tokens"
        )
    event_turn = {
        str(turn["stream_event"]["event_id"]): index for index, turn in enumerate(turns)
    }
    events_by_history: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        events_by_history.setdefault(str(event["history_id"]), []).append(event)

    inputs = []
    for query in sorted(queries, key=lambda row: str(row["query_id"])):
        query_id = str(query["query_id"])
        history_id = str(query["history_id"])
        cumulative = _visible_cumulative_tokens(turns, history_id, tokenizer)
        checkpoint_cumulative = _visible_cumulative_tokens(
            checkpoint_turns, history_id, tokenizer
        )
        contract = query.get("checkpoint_contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"query {query_id} lacks checkpoint_contract")
        groups = contract.get("required_event_id_groups")
        if not isinstance(groups, list) or len(groups) < 2:
            raise ValueError(f"query {query_id} lacks ordered event evidence groups")
        evidence_positions = []
        for group in groups:
            if not isinstance(group, list) or not group:
                raise ValueError(f"query {query_id} has an empty event evidence group")
            available = [
                event_turn[str(event_id)]
                for event_id in group
                if str(event_id) in event_turn
            ]
            if not available:
                raise ValueError(f"query {query_id} evidence group is absent: {group}")
            position = max(available)
            if str(turns[position]["history_id"]) != history_id:
                raise ValueError(f"query {query_id} evidence crosses histories: {group}")
            evidence_positions.append(position)
        if evidence_positions != sorted(evidence_positions):
            raise ValueError(f"query {query_id} evidence groups are not causally ordered")
        update_index = evidence_positions[-1]
        trigger_id = str(contract.get("trigger_event_id", ""))
        trigger_index = event_turn.get(trigger_id)
        if trigger_index is None or trigger_index <= update_index:
            raise ValueError(f"query {query_id} has invalid delayed trigger {trigger_id}")
        if str(turns[trigger_index]["history_id"]) != history_id:
            raise ValueError(f"query {query_id} delayed trigger crosses histories")
        trigger_prefix = [
            dict(turn["stream_event"])
            for turn in turns[: trigger_index + 1]
            if str(turn["history_id"]) == history_id
        ]
        authoritative_gold = resolve_query(trigger_prefix, dict(query))
        if authoritative_gold != query.get("gold"):
            raise ValueError(
                f"query {query_id} trigger-prefix gold mismatch: "
                f"expected {query.get('gold')!r}, got {authoritative_gold!r}"
            )
        history_events = events_by_history.get(history_id)
        if not history_events:
            raise ValueError(f"query {query_id} has no source events")
        authoritative_surface = _surface_gold(authoritative_gold, history_events)
        if authoritative_surface != query.get("surface_gold"):
            raise ValueError(
                f"query {query_id} surface gold mismatch: "
                f"expected {query.get('surface_gold')!r}, got {authoritative_surface!r}"
            )
        checkpoints: list[tuple[str, int, int | None]] = [
            ("pre_update", update_index, None),
            ("post_update", update_index + 1, None),
            ("delayed_probe", trigger_index + 1, None),
        ]
        for threshold in config.token_distance_thresholds:
            checkpoint = next(
                (
                    index + 1
                    for index in range(update_index + 1, len(turns))
                    if checkpoint_cumulative[index] - checkpoint_cumulative[update_index]
                    >= threshold
                    and any(
                        str(turn["history_id"]) != history_id
                        for turn in turns[update_index + 1 : index + 1]
                    )
                ),
                None,
            )
            if checkpoint is None:
                available = checkpoint_cumulative[-1] - checkpoint_cumulative[update_index]
                raise ValueError(
                    f"query {query_id} cannot reach token distance {threshold}; "
                    f"maximum is {available}"
                )
            checkpoints.append(("token_distance", checkpoint, threshold))
        for phase, checkpoint, requested_distance in checkpoints:
            inputs.append(
                _condition_row(
                    query=query,
                    phase=phase,
                    checkpoint=checkpoint,
                    update_index=update_index,
                    requested_distance=requested_distance,
                    turns=turns,
                    cumulative=cumulative,
                    history_events=history_events,
                )
            )
    if len({row["evaluation_input_id"] for row in inputs}) != len(inputs):
        raise ValueError("scheduled evaluation input IDs are not unique")
    model_inputs = [
        {
            "evaluation_input_id": row["evaluation_input_id"],
            "context": row.pop("_model_context"),
            "query_text": row["query_text"],
        }
        for row in inputs
    ]
    safe_turns = [
        {
            key: value
            for key, value in turn.items()
            if key != "stream_event"
        }
        for turn in turns
    ]
    return {
        "schedule_version": SCHEDULE_VERSION,
        "config": asdict(config),
        "dataset": provenance,
        "tokenizer": dict(tokenizer.metadata()),
        "schedule_metrics": metrics,
        "turns": safe_turns,
        "inputs": inputs,
        "model_inputs": model_inputs,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write stable object-valued JSONL."""
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_evaluation_schedule(
    dataset_dir: Path,
    output_dir: Path,
    tokenizer: ScheduleTokenizer,
    config: ScheduleConfig,
) -> dict[str, Any]:
    """Write one authenticated schedule bundle and return its in-memory form."""
    result = build_evaluation_schedule(dataset_dir, tokenizer, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    turns_path = output_dir / "scheduled_turns.jsonl"
    inputs_path = output_dir / "evaluation_inputs.jsonl"
    model_inputs_path = output_dir / "model_inputs.jsonl"
    manifest_path = output_dir / "schedule_manifest.json"
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "running",
        "started_at": started_at,
        "schedule_version": result["schedule_version"],
        "config": result["config"],
        "config_sha256": _stable_sha256(result["config"]),
        "dataset": result["dataset"],
        "tokenizer": result["tokenizer"],
        "schedule_metrics": result["schedule_metrics"],
        "turn_count": len(result["turns"]),
        "input_count": len(result["inputs"]),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(turns_path, result["turns"])
    _write_jsonl(inputs_path, result["inputs"])
    _write_jsonl(model_inputs_path, result["model_inputs"])
    manifest.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": {
                turns_path.name: _sha256(turns_path),
                inputs_path.name: _sha256(inputs_path),
                model_inputs_path.name: _sha256(model_inputs_path),
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_cli_config(path: Path) -> tuple[ScheduleConfig, TokenizerConfig]:
    """Load schedule and tokenizer controls from one JSON configuration."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load schedule config {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("schedule config must be a JSON object")
    schedule = payload.get("schedule")
    tokenizer = payload.get("tokenizer")
    if not isinstance(schedule, Mapping) or not isinstance(tokenizer, Mapping):
        raise ValueError("schedule config requires schedule and tokenizer objects")
    list_fields = ("query_suffixes", "token_distance_thresholds")
    for field in list_fields:
        if not isinstance(schedule.get(field), list):
            raise ValueError(f"schedule.{field} must be an array")
    if any(not isinstance(value, str) or not value for value in schedule["query_suffixes"]):
        raise ValueError("schedule.query_suffixes elements must be non-empty strings")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in schedule["token_distance_thresholds"]
    ):
        raise ValueError("schedule.token_distance_thresholds elements must be integers")
    integer_fields = (
        "seed",
        "concurrent_accounts",
        "min_segment_events",
        "max_segment_events",
        "minimum_stream_tokens",
    )
    for field in integer_fields:
        value = schedule.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"schedule.{field} must be an integer")
    if not isinstance(tokenizer.get("local_files_only"), bool):
        raise ValueError("tokenizer.local_files_only must be a boolean")
    for field in ("name", "revision"):
        if not isinstance(tokenizer.get(field), str) or not tokenizer[field]:
            raise ValueError(f"tokenizer.{field} must be a non-empty string")
    return (
        ScheduleConfig(
            source_split=str(schedule.get("source_split", "")),
            source_profile=str(schedule.get("source_profile", "")),
            source_manifest_sha256=str(schedule.get("source_manifest_sha256", "")),
            seed=schedule["seed"],
            concurrent_accounts=schedule["concurrent_accounts"],
            min_segment_events=schedule["min_segment_events"],
            max_segment_events=schedule["max_segment_events"],
            query_suffixes=tuple(str(value) for value in schedule.get("query_suffixes", [])),
            token_distance_thresholds=tuple(
                int(value) for value in schedule.get("token_distance_thresholds", [])
            ),
            minimum_stream_tokens=schedule.get("minimum_stream_tokens", 0),
        ),
        TokenizerConfig(
            name=str(tokenizer.get("name", "")),
            revision=str(tokenizer.get("revision", "")),
            local_files_only=tokenizer["local_files_only"],
        ),
    )


def main() -> None:
    """Run the standalone persona interference scheduling stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    schedule_config, tokenizer_config = _load_cli_config(args.config)
    tokenizer = HuggingFaceTokenizer(tokenizer_config)
    write_evaluation_schedule(args.dataset_dir, args.output_dir, tokenizer, schedule_config)


if __name__ == "__main__":
    main()
