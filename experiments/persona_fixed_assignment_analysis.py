"""Analyze fixed-assignment persona benchmark results with paired cluster bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from experiments.persona_end_to_end_benchmark import _score_short_answer
from experiments.persona_interference_schedule import (
    _authenticate_dataset,
    _read_jsonl_bytes,
    _surface_gold,
)
from experiments.persona_surface_derivation import (
    DERIVATION_METHOD,
    SURFACE_ASSIGNMENT_SEEDS,
)
from experiments.synthetic_temporal_preferences import resolve_query


ARM_NAMES = (
    "sliding_context_4096",
    "sliding_context_16384",
    "structured_memory_4096",
    "structured_memory_16384",
    "full_qwen_context",
)
METRICS = ("exact_match", "f1")
STRATA = ("overall", "delayed", "source_present", "source_absent")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _authenticate_corpus(
    path: Path,
    label: str,
    expected_assignment: str | None,
    pair_gate_path: Path | None = None,
    pair_gate_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate one explicit corpus and return its result-facing identity."""
    path = Path(path)
    manifest_path = path / "generation_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} corpus manifest cannot be loaded: {error}") from error
    if not isinstance(manifest, Mapping) or manifest.get("status") != "completed":
        raise ValueError(f"{label} corpus manifest is not completed")
    declared = manifest.get("artifact_sha256")
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError(f"{label} corpus manifest lacks artifact_sha256")
    verified: dict[str, str] = {}
    for name, expected in sorted(declared.items()):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name == "generation_manifest.json"
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError(f"{label} corpus has an invalid artifact declaration: {name!r}")
        artifact_path = path / name
        if not artifact_path.is_file():
            raise ValueError(f"{label} corpus is missing declared artifact: {name}")
        actual = _sha256(artifact_path)
        if actual != expected:
            raise ValueError(f"{label} corpus artifact hash mismatch for {name}")
        verified[name] = actual

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_assignment is None:
        model_identity = manifest.get("model_identity")
        if (
            not isinstance(model_identity, str)
            or "kimi" not in model_identity.casefold()
            or manifest.get("method") is not None
            or manifest.get("parent") is not None
        ):
            raise ValueError("v1 corpus must be completed Kimi-authored and non-derived")
    provenance, artifact_bytes, authenticated_parent = _authenticate_dataset(
        path,
        manifest_sha256,
        pair_gate_path,
        pair_gate_sha256,
        expected_assignment,
    )
    allowed_targets_by_history: dict[str, set[str]] | None = None
    surface_mapping_by_history: dict[str, dict[str, str]] | None = None
    if expected_assignment is None:
        if authenticated_parent is not None:
            raise ValueError("v1 corpus must be completed Kimi-authored and non-derived")
    else:
        if (
            manifest.get("method") != DERIVATION_METHOD
            or manifest.get("assignment") != expected_assignment
            or manifest.get("seed") != SURFACE_ASSIGNMENT_SEEDS[expected_assignment]
            or authenticated_parent is None
        ):
            raise ValueError(f"{label} corpus has an unexpected Kimi assignment")
        mapping = manifest.get("surface_mapping")
        if not isinstance(mapping, list):
            raise ValueError(f"{label} corpus lacks its canonical surface mapping")
        allowed_targets_by_history = defaultdict(set)
        surface_mapping_by_history = defaultdict(dict)
        for row in mapping:
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} corpus has a malformed surface mapping row")
            history_id = row.get("history_id")
            source_phrase = row.get("source_phrase")
            target_phrase = row.get("target_phrase")
            if (
                not isinstance(history_id, str)
                or not history_id
                or not isinstance(source_phrase, str)
                or not source_phrase
                or not isinstance(target_phrase, str)
                or not target_phrase
            ):
                raise ValueError(f"{label} corpus has an incomplete surface mapping row")
            allowed_targets_by_history[history_id].add(target_phrase)
            existing = surface_mapping_by_history[history_id].get(source_phrase)
            if existing is not None and existing != target_phrase:
                raise ValueError(
                    f"{label} corpus has an ambiguous source mapping for "
                    f"{history_id!r}, {source_phrase!r}"
                )
            surface_mapping_by_history[history_id][source_phrase] = target_phrase
    return {
        "generation_manifest_sha256": manifest_sha256,
        "result_artifact_sha256": provenance["artifact_sha256"],
        "verified_artifact_sha256": verified,
        "allowed_targets_by_history": allowed_targets_by_history,
        "surface_mapping_by_history": surface_mapping_by_history,
        "artifact_bytes": artifact_bytes,
        "path": path,
    }


def _bind_result_to_corpus(
    manifest: Mapping[str, Any], corpus: Mapping[str, Any], label: str
) -> None:
    """Require outer and nested result provenance to identify the supplied corpus."""
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError(f"{label}: result manifest lacks dataset provenance")
    if dataset.get("generation_manifest_sha256") != corpus[
        "generation_manifest_sha256"
    ] or dataset.get("artifact_sha256") != corpus["result_artifact_sha256"]:
        raise ValueError(f"{label}: result dataset does not match the supplied corpus")


def _validate_assignment_golds(
    rows: Sequence[Mapping[str, Any]], corpus: Mapping[str, Any], label: str
) -> None:
    """Bind generation-equivalent prediction golds to authenticated corpus surfaces."""
    allowed = corpus.get("allowed_targets_by_history")
    if not isinstance(allowed, Mapping):
        raise ValueError(f"{label}: corpus target mapping is unavailable")
    for row in rows:
        history_id = str(row["history_id"])
        targets = allowed.get(history_id)
        if not isinstance(targets, set) or not targets:
            raise ValueError(
                f"{label}: row history {history_id!r} is absent from the authenticated surface mapping"
            )
        gold = row.get("gold")
        if not isinstance(gold, str) or not gold:
            raise ValueError(f"{label}: row for {history_id!r} has an invalid gold")
        if gold != "UNKNOWN" and gold not in targets:
            raise ValueError(
                f"{label}: gold {gold!r} for {history_id!r} is outside its authenticated target set"
            )


_EVALUATION_SEMANTIC_FIELDS = (
    "query_id",
    "query_family",
    "history_id",
    "condition",
    "phase",
    "requested_token_distance",
)


def _query_family(query_id: str) -> str:
    """Return the supported evaluated query family from its authenticated ID."""
    for suffix, family in (
        ("-preference-change-delayed", "preference_change"),
        ("-preference-incongruity-delayed", "preference_incongruity"),
    ):
        if query_id.endswith(suffix):
            return family
    raise ValueError(f"v1 corpus oracle does not support evaluated query {query_id!r}")


def _evaluation_condition(evaluation_input_id: str, query_id: str) -> dict[str, Any]:
    """Derive arm-independent condition semantics from one evaluation input key."""
    prefix = f"{query_id}:"
    if not evaluation_input_id.startswith(prefix):
        raise ValueError(
            f"v1 evaluation input {evaluation_input_id!r} is not keyed by query {query_id!r}"
        )
    suffix = evaluation_input_id[len(prefix) :]
    if suffix in {"pre_update", "post_update", "delayed_probe"}:
        return {
            "condition": suffix,
            "phase": suffix,
            "requested_token_distance": None,
        }
    token_prefix = "token_distance-"
    if suffix.startswith(token_prefix):
        raw_distance = suffix[len(token_prefix) :]
        try:
            distance = int(raw_distance)
        except ValueError as error:
            raise ValueError(
                f"v1 evaluation input {evaluation_input_id!r} has an invalid token distance"
            ) from error
        if distance < 1 or str(distance) != raw_distance:
            raise ValueError(
                f"v1 evaluation input {evaluation_input_id!r} has an invalid token distance"
            )
        return {
            "condition": f"token_distance_{distance}",
            "phase": "token_distance",
            "requested_token_distance": distance,
        }
    raise ValueError(f"v1 evaluation input {evaluation_input_id!r} has an unknown condition")


def _v1_query_surfaces(
    query: Mapping[str, Any], history_events: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    """Resolve authenticated pre-update and stable post-update query surfaces."""
    query_id = str(query.get("query_id", ""))
    contract = query.get("checkpoint_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"v1 corpus query {query_id!r} lacks checkpoint_contract")
    event_positions: dict[str, int] = {}
    for index, event in enumerate(history_events):
        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in event_positions:
            raise ValueError(f"v1 corpus history has an invalid event ID {event_id!r}")
        event_positions[event_id] = index
    groups = contract.get("required_event_id_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"v1 corpus query {query_id!r} lacks required event groups")
    evidence_positions = []
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError(f"v1 corpus query {query_id!r} has an empty event group")
        available = [event_positions[event_id] for event_id in group if event_id in event_positions]
        if not available:
            raise ValueError(f"v1 corpus query {query_id!r} has an absent event group")
        evidence_positions.append(max(available))
    if evidence_positions != sorted(evidence_positions):
        raise ValueError(f"v1 corpus query {query_id!r} has unordered event groups")
    update_index = evidence_positions[-1]
    trigger_id = str(contract.get("trigger_event_id", ""))
    trigger_index = event_positions.get(trigger_id)
    if trigger_index is None or trigger_index <= update_index:
        raise ValueError(f"v1 corpus query {query_id!r} has an invalid trigger")

    expected_internal = query.get("gold")
    if not isinstance(expected_internal, str) or not expected_internal:
        raise ValueError(f"v1 corpus query {query_id!r} has an invalid internal gold")
    pre_internal = resolve_query(
        [dict(event) for event in history_events[:update_index]], dict(query)
    )
    post_internal = resolve_query(
        [dict(event) for event in history_events[: update_index + 1]], dict(query)
    )
    if post_internal != expected_internal:
        raise ValueError(f"v1 corpus query {query_id!r} post-update prefix is incorrect")
    for end_index in range(trigger_index, len(history_events)):
        resolved = resolve_query(
            [dict(event) for event in history_events[: end_index + 1]], dict(query)
        )
        if resolved != expected_internal:
            stage = "trigger" if end_index == trigger_index else f"later prefix {end_index}"
            raise ValueError(
                f"v1 corpus query {query_id!r} is unstable at {stage}: "
                f"expected {expected_internal!r}, got {resolved!r}"
            )
    pre_surface = _surface_gold(pre_internal, history_events)
    post_surface = _surface_gold(expected_internal, history_events)
    if post_surface != query.get("surface_gold"):
        raise ValueError(f"v1 corpus query {query_id!r} has an invalid surface_gold")
    return pre_surface, post_surface


def _build_v1_corpus_oracle(
    baseline_rows: Sequence[Mapping[str, Any]], corpus: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Build and validate exact v1 evaluation semantics from authenticated corpus bytes."""
    artifacts = corpus.get("artifact_bytes")
    corpus_path = corpus.get("path")
    if not isinstance(artifacts, Mapping) or not isinstance(corpus_path, Path):
        raise ValueError("v1 corpus oracle artifacts are unavailable")
    events = _read_jsonl_bytes(artifacts["events.jsonl"], corpus_path / "events.jsonl")
    queries = _read_jsonl_bytes(artifacts["queries.jsonl"], corpus_path / "queries.jsonl")
    events_by_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_history[str(event.get("history_id", ""))].append(event)
    queries_by_id: dict[str, dict[str, Any]] = {}
    for query in queries:
        query_id = str(query.get("query_id", ""))
        if not query_id or query_id in queries_by_id:
            raise ValueError(f"v1 corpus has an invalid query ID {query_id!r}")
        queries_by_id[query_id] = query

    evaluated_query_ids = set()
    for query_id, query in queries_by_id.items():
        if query.get("split") != "test" or not isinstance(
            query.get("checkpoint_contract"), Mapping
        ):
            continue
        try:
            _query_family(query_id)
        except ValueError:
            continue
        evaluated_query_ids.add(query_id)
    if not evaluated_query_ids:
        raise ValueError("authenticated v1 corpus has no evaluated checkpoint-contract queries")
    token_distances = set()
    for row in baseline_rows:
        if row.get("phase") != "token_distance":
            continue
        distance = row.get("requested_token_distance")
        if isinstance(distance, bool) or not isinstance(distance, int) or distance < 1:
            raise ValueError("v1 baseline has a non-positive token-distance descriptor")
        token_distances.add(distance)
    if len(token_distances) != 2:
        raise ValueError(
            "v1 baseline complete fixed design requires exactly two global token distances"
        )
    condition_suffixes = (
        "pre_update",
        "post_update",
        "delayed_probe",
        *(f"token_distance-{distance}" for distance in sorted(token_distances)),
    )
    expected_keys = {
        (f"{query_id}:{suffix}", arm)
        for query_id in evaluated_query_ids
        for suffix in condition_suffixes
        for arm in ARM_NAMES
    }
    actual_keys = {
        (str(row["evaluation_input_id"]), str(row["arm"])) for row in baseline_rows
    }
    if actual_keys != expected_keys:
        raise ValueError(
            "v1 baseline does not realize the complete fixed design for every corpus query"
        )

    surfaces_by_query: dict[str, tuple[str, str]] = {}
    oracle: dict[str, dict[str, Any]] = {}
    for row in baseline_rows:
        evaluation_input_id = str(row["evaluation_input_id"])
        query_id = str(row.get("query_id", ""))
        query = queries_by_id.get(query_id)
        if query is None:
            raise ValueError(f"v1 result references absent corpus query {query_id!r}")
        history_id = str(query.get("history_id", ""))
        history_events = events_by_history.get(history_id)
        if not history_events:
            raise ValueError(f"v1 corpus query {query_id!r} has no history events")
        if query_id not in surfaces_by_query:
            surfaces_by_query[query_id] = _v1_query_surfaces(query, history_events)
        pre_surface, post_surface = surfaces_by_query[query_id]
        condition = _evaluation_condition(evaluation_input_id, query_id)
        expected = {
            "query_id": query_id,
            "query_family": _query_family(query_id),
            "history_id": history_id,
            **condition,
            "gold": pre_surface if condition["phase"] == "pre_update" else post_surface,
        }
        existing = oracle.get(evaluation_input_id)
        if existing is not None and existing != expected:
            raise ValueError(
                f"v1 corpus oracle produced inconsistent semantics for {evaluation_input_id!r}"
            )
        oracle[evaluation_input_id] = expected
        for field, value in expected.items():
            if row.get(field) != value:
                detail = "exact gold" if field == "gold" else field
                raise ValueError(
                    f"v1 corpus oracle {detail} mismatch for {evaluation_input_id!r}: "
                    f"expected {value!r}, got {row.get(field)!r}"
                )
    return oracle


def _validate_exact_gold_semantics(
    baseline_by_input: Mapping[str, Mapping[str, Any]],
    assignment_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    corpora: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind A/B checkpoint semantics and golds to v1 through canonical mappings."""
    for label, rows in assignment_rows.items():
        assignment_inputs = {str(row["evaluation_input_id"]) for row in rows}
        if assignment_inputs != set(baseline_by_input):
            raise ValueError(f"{label} and v1 evaluation-input keys differ")
        mappings = corpora[label].get("surface_mapping_by_history")
        if not isinstance(mappings, Mapping):
            raise ValueError(f"{label}: canonical source mapping is unavailable")
        for row in rows:
            evaluation_input_id = str(row["evaluation_input_id"])
            baseline = baseline_by_input[evaluation_input_id]
            for field in _EVALUATION_SEMANTIC_FIELDS:
                if row.get(field) != baseline[field]:
                    raise ValueError(
                        f"{label} evaluation input {evaluation_input_id!r} "
                        f"differs from v1 on {field}"
                    )
            baseline_gold = baseline["gold"]
            if not isinstance(baseline_gold, str) or not baseline_gold:
                raise ValueError(f"v1 evaluation input {evaluation_input_id!r} has an invalid gold")
            if baseline_gold == "UNKNOWN":
                expected_gold = "UNKNOWN"
            else:
                history_id = str(baseline["history_id"])
                history_mapping = mappings.get(history_id)
                if not isinstance(history_mapping, Mapping):
                    raise ValueError(
                        f"{label}: v1 history {history_id!r} is absent from the canonical mapping"
                    )
                expected_gold = history_mapping.get(baseline_gold)
                if not isinstance(expected_gold, str) or not expected_gold:
                    raise ValueError(
                        f"{label}: v1 gold {baseline_gold!r} has no unambiguous canonical "
                        f"mapping for {history_id!r}"
                    )
            if row.get("gold") != expected_gold:
                raise ValueError(
                    f"{label} evaluation input {evaluation_input_id!r} has an exact gold mismatch: "
                    f"expected {expected_gold!r}, got {row.get('gold')!r}"
                )


def _load_source(
    path: Path, label: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load one result directory and verify every artifact declared by its manifest."""
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{label}: missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("artifact_sha256")
    required_artifacts = {
        "generation_manifest.json",
        "generations.jsonl",
        "predictions.jsonl",
    }
    if not isinstance(declared, dict) or not required_artifacts.issubset(declared):
        raise ValueError(f"{label}: manifest must authenticate generation and prediction artifacts")
    verified: dict[str, str] = {}
    for name, expected in sorted(declared.items()):
        artifact = path / name
        if artifact.parent != path or not artifact.is_file():
            raise ValueError(f"{label}: invalid or missing declared artifact: {name}")
        actual = _sha256(artifact)
        if actual != expected:
            raise ValueError(
                f"{label}: SHA-256 mismatch for {name}: expected {expected}, got {actual}"
            )
        verified[name] = actual

    generation_manifest_path = path / "generation_manifest.json"
    generation_manifest_sha256 = verified["generation_manifest.json"]
    if manifest.get("generation_manifest_sha256") != generation_manifest_sha256:
        raise ValueError(f"{label}: outer manifest does not authenticate generation_manifest.json")
    try:
        generation_manifest = json.loads(
            generation_manifest_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label}: invalid generation_manifest.json") from error
    if not isinstance(generation_manifest, Mapping) or generation_manifest.get(
        "status"
    ) != "completed":
        raise ValueError(f"{label}: generation manifest is not completed")
    for field, expected in manifest.items():
        if field in {"artifact_sha256", "generation_manifest_sha256"}:
            continue
        if generation_manifest.get(field) != expected:
            raise ValueError(f"{label}: generation and outer manifests differ on {field}")
    generation_artifacts = generation_manifest.get("artifact_sha256")
    if (
        not isinstance(generation_artifacts, Mapping)
        or generation_artifacts.get("generations.jsonl")
        != verified["generations.jsonl"]
    ):
        raise ValueError(f"{label}: generation manifest does not authenticate generations.jsonl")

    generation_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(
        (path / "generations.jsonl").read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            generation = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}: invalid JSON at generations line {line_number}") from error
        if not isinstance(generation, dict):
            raise ValueError(f"{label}: generations line {line_number} must be an object")
        if "evaluation_input_id" not in generation or "arm" not in generation:
            raise ValueError(f"{label}: generations line {line_number} lacks its key")
        key = (str(generation["evaluation_input_id"]), str(generation["arm"]))
        if key in generation_rows:
            raise ValueError(f"{label}: duplicate generation key: {key}")
        generation_rows[key] = generation
    if not generation_rows:
        raise ValueError(f"{label}: generations.jsonl is empty")

    rows: list[dict[str, Any]] = []
    prediction_keys: set[tuple[str, str]] = set()
    predictions_path = path / "predictions.jsonl"
    for line_number, line in enumerate(
        predictions_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}: invalid JSON at predictions line {line_number}") from error
        required = {
            "arm",
            "condition",
            "evaluation_input_id",
            "exact_match",
            "f1",
            "history_id",
            "input_token_count",
            "prompt_sha256",
            "selected_turn_count",
            "structured_source_turn_count",
        }
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{label}: predictions line {line_number} missing {sorted(missing)}")
        if row["arm"] not in ARM_NAMES:
            raise ValueError(f"{label}: unsupported arm {row['arm']!r}")
        key = (str(row["evaluation_input_id"]), str(row["arm"]))
        if key in prediction_keys:
            raise ValueError(f"{label}: duplicate prediction key: {key}")
        prediction_keys.add(key)
        generation = generation_rows.get(key)
        if generation is None:
            raise ValueError(f"{label}: prediction has no authenticated generation row: {key}")
        for field, expected in generation.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"{label}: prediction differs from authenticated generation row on {field}: {key}"
                )
        scored = _score_short_answer(
            str(generation.get("answer", "")), str(generation.get("gold", ""))
        )
        for field, expected in scored.items():
            if row.get(field) != expected:
                raise ValueError(f"{label}: prediction has invalid canonical score {field}: {key}")
        if set(row) != set(generation) | set(scored):
            raise ValueError(
                f"{label}: prediction fields differ from generation plus canonical score: {key}"
            )
        for metric in METRICS:
            score = row[metric]
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"{label}: invalid {metric} at predictions line {line_number}")
        row["assignment"] = label
        rows.append(row)
    if not rows:
        raise ValueError(f"{label}: predictions.jsonl is empty")
    if prediction_keys != set(generation_rows):
        raise ValueError(f"{label}: prediction and generation key sets differ")
    arms = manifest.get("arms")
    if (
        not isinstance(arms, list)
        or any(not isinstance(row, Mapping) for row in arms)
        or [row.get("name") for row in arms] != list(ARM_NAMES)
    ):
        raise ValueError(f"{label}: benchmark manifest does not declare all five arms")
    arms_by_input: dict[str, set[str]] = defaultdict(set)
    for evaluation_input_id, arm in prediction_keys:
        arms_by_input[evaluation_input_id].add(arm)
    if any(arms != set(ARM_NAMES) for arms in arms_by_input.values()):
        raise ValueError(f"{label}: evaluation inputs have missing arms")
    condition_count = manifest.get("condition_count")
    generation_count = manifest.get("generation_count")
    if (
        isinstance(condition_count, bool)
        or not isinstance(condition_count, int)
        or condition_count != len(arms_by_input)
        or generation_count != len(prediction_keys)
        or generation_count != condition_count * len(ARM_NAMES)
    ):
        raise ValueError(f"{label}: benchmark manifest has invalid evaluation counts")
    return (
        rows,
        {
            "manifest_sha256": _sha256(manifest_path),
            "verified_artifact_sha256": verified,
        },
        manifest,
    )


def _validate_assignments(
    assignment_paths: Mapping[str, Path],
    assignment_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    manifests: Mapping[str, Mapping[str, Any]],
    source_authentication: Mapping[str, Mapping[str, Any]],
) -> None:
    """Authenticate distinct, complete A/B derivations under shared evaluation semantics."""
    if Path(assignment_paths["A"]).resolve() == Path(assignment_paths["B"]).resolve():
        raise ValueError("A and B must be distinct directories")

    for label in ("A", "B"):
        manifest = manifests[label]
        if manifest.get("status") != "completed":
            raise ValueError(f"{label}: benchmark manifest is not completed")
        dataset = manifest.get("dataset")
        if not isinstance(dataset, Mapping):
            raise ValueError(f"{label}: manifest lacks dataset provenance")
        if (
            dataset.get("method") != DERIVATION_METHOD
            or dataset.get("assignment") != label
            or dataset.get("seed") != SURFACE_ASSIGNMENT_SEEDS[label]
        ):
            raise ValueError(f"{label}: unexpected Kimi assignment")
        if dataset.get("checkpoint_policy") != "authenticated_parent_exact_indices":
            raise ValueError(f"{label}: unexpected checkpoint policy")
        parent_hash = dataset.get("parent_generation_manifest_sha256")
        if not isinstance(parent_hash, str) or not parent_hash:
            raise ValueError(f"{label}: missing authenticated parent identity")
        if not isinstance(dataset.get("pair_gate_sha256"), str):
            raise ValueError(f"{label}: missing authenticated pair gate identity")
    if manifests["A"]["dataset"].get("parent_generation_manifest_sha256") != manifests[
        "B"
    ]["dataset"].get("parent_generation_manifest_sha256"):
        raise ValueError("A and B must share the same authenticated parent")
    if manifests["A"]["dataset"].get("pair_gate_sha256") != manifests["B"][
        "dataset"
    ].get("pair_gate_sha256"):
        raise ValueError("A and B must share the same authenticated pair gate")
    prediction_hashes = {
        source_authentication[label]["verified_artifact_sha256"]["predictions.jsonl"]
        for label in ("A", "B")
    }
    if len(prediction_hashes) != 2:
        raise ValueError("A and B prediction artifacts must be distinct")
    dataset_manifest_hashes = {
        manifests[label]["dataset"].get("generation_manifest_sha256")
        for label in ("A", "B")
    }
    if len(dataset_manifest_hashes) != 2:
        raise ValueError("A and B dataset manifest hashes must be distinct")
    dataset_content_identities = {
        _canonical_sha256(manifests[label]["dataset"].get("artifact_sha256"))
        for label in ("A", "B")
    }
    if len(dataset_content_identities) != 2:
        raise ValueError("A and B dataset content identities must be distinct")
    shared_fields = (
        "arms",
        "condition_count",
        "decoding",
        "evaluator_script_sha256",
        "evaluator_version",
        "generation_count",
        "injection_map_sha256",
        "model",
        "prompt_instruction_sha256",
        "scallop",
        "schedule_version",
        "tokenizer",
    )
    for field in shared_fields:
        if field not in manifests["A"] or field not in manifests["B"]:
            raise ValueError(f"A or B lacks evaluation semantics: {field}")
        if manifests["A"].get(field) != manifests["B"].get(field):
            raise ValueError(f"A and B differ in evaluation semantics: {field}")

    keys_by_assignment = {
        label: {(str(row["evaluation_input_id"]), str(row["arm"])) for row in rows}
        for label, rows in assignment_rows.items()
    }
    if keys_by_assignment["A"] != keys_by_assignment["B"]:
        raise ValueError("A and B evaluation-input key sets differ")
    for label, rows in assignment_rows.items():
        arms_by_input: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            arms_by_input[str(row["evaluation_input_id"])].add(str(row["arm"]))
        incomplete = {
            input_id: sorted(set(ARM_NAMES) - arms)
            for input_id, arms in arms_by_input.items()
            if arms != set(ARM_NAMES)
        }
        if incomplete:
            raise ValueError(f"{label}: evaluation inputs have missing arms: {incomplete}")
        if len(arms_by_input) != manifests[label]["condition_count"]:
            raise ValueError(f"{label}: evaluation input count differs from manifest")
        if len(keys_by_assignment[label]) != manifests[label]["generation_count"]:
            raise ValueError(f"{label}: evaluation-input key count differs from manifest")


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """Index rows by assignment, evaluation input, and arm, rejecting duplicates."""
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (row["assignment"], row["evaluation_input_id"], row["arm"])
        if key in indexed:
            raise ValueError(f"duplicate prediction key: {key}")
        indexed[key] = row
    return indexed


def _stratum_matches(name: str, structured_row: Mapping[str, Any]) -> bool:
    """Return whether a paired observation belongs to a named analysis stratum."""
    source_present = structured_row["structured_source_turn_count"] > 0
    if name == "overall":
        return True
    if name == "delayed":
        return structured_row["condition"] == "delayed_probe"
    if name == "source_present":
        return source_present
    if name == "source_absent":
        return not source_present
    raise ValueError(f"unsupported stratum: {name}")


def _paired_observations(
    rows: Sequence[Mapping[str, Any]], left_arm: str, right_arm: str, stratum: str
) -> list[dict[str, Any]]:
    """Build complete paired observations for a comparison and stratum."""
    indexed = _index_rows(rows)
    observations: list[dict[str, Any]] = []
    left_keys = sorted(key for key in indexed if key[2] == left_arm)
    for assignment, evaluation_input_id, _ in left_keys:
        left = indexed[(assignment, evaluation_input_id, left_arm)]
        right_key = (assignment, evaluation_input_id, right_arm)
        if right_key not in indexed:
            raise ValueError(f"missing paired row: {right_key}")
        right = indexed[right_key]
        if left["history_id"] != right["history_id"]:
            raise ValueError(f"paired rows disagree on history: {right_key}")
        if left_arm.startswith("structured_memory_"):
            structured = left
        elif right_arm.startswith("structured_memory_"):
            structured = right
        else:
            structured_key = (assignment, evaluation_input_id, "structured_memory_16384")
            if structured_key not in indexed:
                raise ValueError(f"missing source-presence reference row: {structured_key}")
            structured = indexed[structured_key]
        if not _stratum_matches(stratum, structured):
            continue
        is_source_absent_pair = (
            stratum == "source_absent"
            and left_arm.startswith("structured_memory_")
            and right_arm.startswith("sliding_context_")
        )
        if is_source_absent_pair:
            for row_name, row in (("structured", left), ("sliding", right)):
                if row["structured_source_turn_count"] != 0:
                    raise ValueError(
                        "source-absent structured/sliding pair has nonzero "
                        f"structured_source_turn_count on {row_name}: {right_key}"
                    )
            for field in ("prompt_sha256", "input_token_count", "selected_turn_count"):
                if left[field] != right[field]:
                    raise ValueError(
                        f"source-absent structured/sliding pair differs on {field}: {right_key}"
                    )
        observations.append(
            {
                "assignment": assignment,
                "history_id": left["history_id"],
                "exact_match": float(left["exact_match"]) - float(right["exact_match"]),
                "f1": float(left["f1"]) - float(right["f1"]),
            }
        )
    if not observations:
        raise ValueError(f"no paired observations for {left_arm} - {right_arm}, {stratum}")
    return observations


def _percentile(values: Sequence[float], probability: float) -> float:
    """Compute a linearly interpolated percentile from sorted finite values."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_summary(
    observations: Sequence[Mapping[str, Any]],
    metric: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    analysis_key: str,
) -> dict[str, Any]:
    """Estimate a mean paired difference and history-cluster bootstrap interval."""
    by_history: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        by_history[observation["history_id"]].append(float(observation[metric]))
    histories = sorted(by_history)
    seed_material = f"{bootstrap_seed}:{analysis_key}:{metric}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(derived_seed)
    draws: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(histories) for _ in histories]
        draws.append(fmean(value for history in sampled for value in by_history[history]))
    estimate = fmean(float(observation[metric]) for observation in observations)
    return {
        "estimate": estimate,
        "ci_95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
    }


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    left_arm: str,
    right_arm: str,
    stratum: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    analysis_key: str,
) -> dict[str, Any]:
    """Summarize one paired arm comparison with cluster-bootstrap intervals."""
    observations = _paired_observations(rows, left_arm, right_arm, stratum)
    is_structured_sliding = left_arm.startswith(
        "structured_memory_"
    ) and right_arm.startswith("sliding_context_")
    if is_structured_sliding and stratum == "source_absent":
        for observation in observations:
            if any(observation[metric] != 0.0 for metric in METRICS):
                raise ValueError(
                    "source-absent paired effect is nonzero; "
                    "fixed-assignment inputs are contaminated"
                )
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "stratum": stratum,
        "paired_row_count": len(observations),
        "history_cluster_count": len({row["history_id"] for row in observations}),
        "metrics": {
            metric: _bootstrap_summary(
                observations,
                metric,
                bootstrap_samples,
                bootstrap_seed,
                analysis_key,
            )
            for metric in METRICS
        },
    }


def _analyze_group(
    rows: Sequence[Mapping[str, Any]],
    label: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compute arm scores and requested paired comparisons for one assignment group."""
    arm_metrics = []
    for arm in ARM_NAMES:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            raise ValueError(f"{label}: missing arm {arm}")
        arm_metrics.append(
            {
                "arm": arm,
                "row_count": len(arm_rows),
                "history_cluster_count": len({row["history_id"] for row in arm_rows}),
                "exact_match": fmean(float(row["exact_match"]) for row in arm_rows),
                "f1": fmean(float(row["f1"]) for row in arm_rows),
            }
        )
    structured_deltas = []
    for budget in (4096, 16384):
        for stratum in STRATA:
            structured_deltas.append(
                _comparison(
                    rows,
                    f"structured_memory_{budget}",
                    f"sliding_context_{budget}",
                    stratum,
                    bootstrap_samples,
                    bootstrap_seed,
                    f"{label}:structured:{budget}:{stratum}",
                )
            )
    full_context_comparisons = []
    for right_arm in ("sliding_context_16384", "structured_memory_16384"):
        for stratum in STRATA:
            full_context_comparisons.append(
                _comparison(
                    rows,
                    "full_qwen_context",
                    right_arm,
                    stratum,
                    bootstrap_samples,
                    bootstrap_seed,
                    f"{label}:full:{right_arm}:{stratum}",
                )
            )
    return {
        "arm_metrics": arm_metrics,
        "structured_minus_sliding": structured_deltas,
        "full_context_comparisons": full_context_comparisons,
    }


def _effect_lookup(group: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Index structured-minus-sliding records by budget and stratum."""
    lookup = {}
    for record in group["structured_minus_sliding"]:
        budget = record["left_arm"].rsplit("_", 1)[1]
        lookup[(budget, record["stratum"])] = record
    return lookup


def analyze_sources(
    baseline_path: Path | None,
    assignment_paths: Mapping[str, Path],
    baseline_corpus_path: Path | None,
    assignment_corpus_paths: Mapping[str, Path],
    pair_gate_path: Path | None = None,
    pair_gate_sha256: str | None = None,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 73,
) -> dict[str, Any]:
    """Authenticate sources and build pooled, per-assignment, and descriptive v1 analyses."""
    if set(assignment_paths) != {"A", "B"}:
        raise ValueError("assignment_paths must contain exactly A and B")
    if set(assignment_corpus_paths) != {"A", "B"}:
        raise ValueError("assignment_corpus_paths must contain exactly A and B")
    if baseline_path is not None and baseline_corpus_path is None:
        raise ValueError("baseline_corpus_path is required when baseline_path is supplied")
    if baseline_path is None and baseline_corpus_path is not None:
        raise ValueError("baseline_corpus_path requires baseline_path")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if Path(assignment_paths["A"]).resolve() == Path(assignment_paths["B"]).resolve():
        raise ValueError("A and B must be distinct directories")
    if pair_gate_path is None or pair_gate_sha256 is None:
        raise ValueError("A/B analysis requires pair_gate_path and pair_gate_sha256")

    corpus_authentication = {
        label: _authenticate_corpus(
            Path(assignment_corpus_paths[label]),
            label,
            label,
            pair_gate_path,
            pair_gate_sha256,
        )
        for label in ("A", "B")
    }
    baseline_corpus_authentication = (
        _authenticate_corpus(Path(baseline_corpus_path), "v1", None)
        if baseline_corpus_path is not None
        else None
    )

    source_authentication: dict[str, Any] = {}
    assignment_rows: dict[str, list[dict[str, Any]]] = {}
    assignment_manifests: dict[str, dict[str, Any]] = {}
    for label in ("A", "B"):
        rows, authentication, manifest = _load_source(Path(assignment_paths[label]), label)
        assignment_rows[label] = rows
        source_authentication[label] = authentication
        assignment_manifests[label] = manifest
        _bind_result_to_corpus(manifest, corpus_authentication[label], label)
        _validate_assignment_golds(rows, corpus_authentication[label], label)
        authentication["corpus_generation_manifest_sha256"] = corpus_authentication[label][
            "generation_manifest_sha256"
        ]
        authentication["corpus_verified_artifact_sha256"] = corpus_authentication[label][
            "verified_artifact_sha256"
        ]
    _validate_assignments(
        assignment_paths,
        assignment_rows,
        assignment_manifests,
        source_authentication,
    )
    histories_by_assignment = {
        label: {row["history_id"] for row in rows}
        for label, rows in assignment_rows.items()
    }
    if histories_by_assignment["A"] != histories_by_assignment["B"]:
        raise ValueError("A and B must contain the same base history IDs")
    histories = sorted(histories_by_assignment["A"])
    if len(histories) != 12:
        raise ValueError(f"expected 12 base-history bootstrap clusters, got {len(histories)}")

    pooled_rows = assignment_rows["A"] + assignment_rows["B"]
    groups = {
        "pooled_A_B": _analyze_group(
            pooled_rows, "pooled_A_B", bootstrap_samples, bootstrap_seed
        ),
        "A": _analyze_group(assignment_rows["A"], "A", bootstrap_samples, bootstrap_seed),
        "B": _analyze_group(assignment_rows["B"], "B", bootstrap_samples, bootstrap_seed),
    }
    report: dict[str, Any] = {
        "schema_version": "persona-fixed-assignment-analysis.v1",
        "metric_priority": ["exact_match", "f1"],
        "bootstrap": {
            "method": "paired nonparametric base-history cluster percentile bootstrap",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
        },
        "design": {
            "bootstrap_unit": "base_history_id",
            "bootstrap_cluster_count": len(histories),
            "pooled_assignment_history_row_count": sum(
                len(values) for values in histories_by_assignment.values()
            ),
            "replicas_kept_together": True,
            "cluster_assignments": {history: ["A", "B"] for history in histories},
        },
        "source_authentication": source_authentication,
        "analyses": groups,
        "bounded_claims": [
            "Intervals quantify sampling variation across the 12 observed base histories only.",
            "Exact match is primary; token F1 is secondary.",
            "Source-absent structured-minus-sliding effects are required to be exactly zero.",
            "Hashes and invariants authenticate internal consistency and corpus-semantic binding; "
            "they do not prove execution or provide hostile artifact-origin attestation.",
        ],
    }

    if baseline_path is not None:
        baseline_rows, authentication, baseline_manifest = _load_source(
            Path(baseline_path), "v1"
        )
        if baseline_corpus_authentication is None:
            raise ValueError("baseline corpus authentication is unavailable")
        _bind_result_to_corpus(
            baseline_manifest, baseline_corpus_authentication, "v1"
        )
        authentication["corpus_generation_manifest_sha256"] = (
            baseline_corpus_authentication["generation_manifest_sha256"]
        )
        authentication["corpus_verified_artifact_sha256"] = (
            baseline_corpus_authentication["verified_artifact_sha256"]
        )
        source_authentication["v1"] = authentication
        baseline_histories = {row["history_id"] for row in baseline_rows}
        if baseline_histories != set(histories):
            raise ValueError("v1 and A/B must contain the same base history IDs")
        v1_oracle = _build_v1_corpus_oracle(
            baseline_rows, baseline_corpus_authentication
        )
        _validate_exact_gold_semantics(
            v1_oracle, assignment_rows, corpus_authentication
        )
        report["bounded_claims"].append(
            "Exact A/B gold semantics are bound to authenticated v1 corpus checkpoints "
            "through causal resolution and each assignment corpus's canonical surface mapping."
        )
        baseline_analysis = _analyze_group(
            baseline_rows, "v1", bootstrap_samples, bootstrap_seed
        )
        groups["v1"] = baseline_analysis
        pooled_lookup = _effect_lookup(groups["pooled_A_B"])
        baseline_lookup = _effect_lookup(baseline_analysis)
        differences = []
        for key in sorted(pooled_lookup):
            pooled_record = pooled_lookup[key]
            baseline_record = baseline_lookup[key]
            differences.append(
                {
                    "budget": int(key[0]),
                    "stratum": key[1],
                    "metrics": {
                        metric: {
                            "estimate": pooled_record["metrics"][metric]["estimate"]
                            - baseline_record["metrics"][metric]["estimate"]
                        }
                        for metric in METRICS
                    },
                }
            )
        warning = (
            "CONTAMINATION WARNING: A and B use separately Kimi-generated, pair-conditioned surfaces and dialogue "
            "over the same v1 histories, facts, and evaluation structure, not independent latent "
            "replications. These "
            "difference-in-paired-differences estimates are descriptive only and do not support "
            "causal, independence, or out-of-sample generalization claims."
        )
        report["descriptive_difference_in_paired_differences_vs_v1"] = {
            "warning": warning,
            "estimates": differences,
        }
        report["bounded_claims"].append(warning)

    report["authentication"] = {
        "analysis_payload_sha256": _canonical_sha256(report),
        "analysis_module_sha256": _sha256(Path(__file__)),
    }
    return report


def _fmt(value: float) -> str:
    """Format a score compactly for Markdown."""
    return f"{value:.3f}"


def _comparison_markdown(title: str, records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Render paired comparison records as a compact Markdown table."""
    lines = [
        f"## {title}",
        "",
        "| Contrast | Stratum | EM delta [95% CI] | F1 delta [95% CI] |",
        "|---|---|---:|---:|",
    ]
    for record in records:
        em = record["metrics"]["exact_match"]
        f1 = record["metrics"]["f1"]
        contrast = f"{record['left_arm']} - {record['right_arm']}"
        lines.append(
            f"| {contrast} | {record['stratum']} | {_fmt(em['estimate'])} "
            f"[{_fmt(em['ci_95'][0])}, {_fmt(em['ci_95'][1])}] | {_fmt(f1['estimate'])} "
            f"[{_fmt(f1['ci_95'][0])}, {_fmt(f1['ci_95'][1])}] |"
        )
    return lines


def render_markdown(report: Mapping[str, Any], result_json_sha256: str) -> str:
    """Render the authenticated report as concise Markdown with bounded claims."""
    lines = [
        "# Pooled Fixed-Assignment Analysis",
        "",
        "Primary metric: exact match (EM). Secondary metric: token F1.",
        "",
        "A+B uses 12 base-history bootstrap clusters; each sampled history retains both assignment replicas.",
        "",
        "## Pooled Arm Scores",
        "",
        "| Arm | EM | F1 | Rows | Histories |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in report["analyses"]["pooled_A_B"]["arm_metrics"]:
        lines.append(
            f"| {record['arm']} | {_fmt(record['exact_match'])} | {_fmt(record['f1'])} | "
            f"{record['row_count']} | {record['history_cluster_count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Assignment Arm Scores",
            "",
            "| Assignment | Arm | EM | F1 |",
            "|---|---|---:|---:|",
        ]
    )
    for label in ("A", "B", "v1"):
        if label not in report["analyses"]:
            continue
        for record in report["analyses"][label]["arm_metrics"]:
            lines.append(
                f"| {label} | {record['arm']} | {_fmt(record['exact_match'])} | "
                f"{_fmt(record['f1'])} |"
            )
    for label in ("pooled_A_B", "A", "B", "v1"):
        if label not in report["analyses"]:
            continue
        group = report["analyses"][label]
        lines.extend(
            [""]
            + _comparison_markdown(
                f"{label}: Structured Minus Sliding",
                group["structured_minus_sliding"],
            )
        )
        lines.extend(
            [""]
            + _comparison_markdown(
                f"{label}: Full-Context Comparisons",
                group["full_context_comparisons"],
            )
        )
    difference = report.get("descriptive_difference_in_paired_differences_vs_v1")
    if difference:
        lines.extend(
            [
                "",
                "## Descriptive Difference-in-Paired-Differences vs v1",
                "",
                difference["warning"],
                "",
                "| Budget | Stratum | EM | F1 |",
                "|---:|---|---:|---:|",
            ]
        )
        for record in difference["estimates"]:
            lines.append(
                f"| {record['budget']} | {record['stratum']} | "
                f"{_fmt(record['metrics']['exact_match']['estimate'])} | "
                f"{_fmt(record['metrics']['f1']['estimate'])} |"
            )
    lines.extend(
        [
            "",
            "## Authentication",
            "",
            f"- Result JSON SHA-256: `{result_json_sha256}`",
        ]
    )
    for label, authentication in sorted(report["source_authentication"].items()):
        lines.append(f"- {label} manifest SHA-256: `{authentication['manifest_sha256']}`")
        lines.append(
            f"- {label} predictions SHA-256: "
            f"`{authentication['verified_artifact_sha256']['predictions.jsonl']}`"
        )
    lines.extend(["", "## Bounded Claims", ""])
    lines.extend(f"- {claim}" for claim in report["bounded_claims"])
    return "\n".join(lines) + "\n"


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    """Write JSON, Markdown, a result manifest, and SHA-256 authentication sidecar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis.json"
    markdown_path = output_dir / "analysis.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    json_sha256 = _sha256(json_path)
    markdown_path.write_text(render_markdown(report, json_sha256), encoding="utf-8")
    artifact_hashes = {
        "analysis.json": json_sha256,
        "analysis.md": _sha256(markdown_path),
    }
    manifest = {
        "schema_version": "persona-fixed-assignment-result-manifest.v1",
        "artifact_sha256": artifact_hashes,
        "source_authentication": report["source_authentication"],
        "analysis_payload_sha256": report["authentication"]["analysis_payload_sha256"],
        "analysis_module_sha256": report["authentication"]["analysis_module_sha256"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = {**artifact_hashes, "manifest.json": _sha256(manifest_path)}
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="ascii",
    )
    return hashes


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI paths and deterministic bootstrap settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True, help="v1 result directory")
    parser.add_argument(
        "--v1-corpus", type=Path, required=True, help="authenticated v1 corpus directory"
    )
    parser.add_argument(
        "--assignment-a", type=Path, required=True, help="surface A result directory"
    )
    parser.add_argument(
        "--assignment-b", type=Path, required=True, help="surface B result directory"
    )
    parser.add_argument(
        "--assignment-a-corpus",
        type=Path,
        required=True,
        help="authenticated surface A corpus directory",
    )
    parser.add_argument(
        "--assignment-b-corpus",
        type=Path,
        required=True,
        help="authenticated surface B corpus directory",
    )
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--pair-gate", type=Path, required=True)
    parser.add_argument("--pair-gate-sha256", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=73)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed-assignment analysis CLI."""
    args = _parse_args(argv)
    report = analyze_sources(
        baseline_path=args.v1,
        assignment_paths={"A": args.assignment_a, "B": args.assignment_b},
        baseline_corpus_path=args.v1_corpus,
        assignment_corpus_paths={
            "A": args.assignment_a_corpus,
            "B": args.assignment_b_corpus,
        },
        pair_gate_path=args.pair_gate,
        pair_gate_sha256=args.pair_gate_sha256,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    hashes = write_outputs(report, args.output)
    print(json.dumps({"output": str(args.output), "sha256": hashes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
