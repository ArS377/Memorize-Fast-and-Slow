"""Generate a deterministic MVP for temporal preference-memory evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neurosym.domain.hard_gate_admission import derive_candidate_features
from neurosym.domain.validation_rules import DEFAULT_RULE_PARAMETERS, RuleParameters


DATASET_VERSION = "synthetic_temporal_preferences.v2"
DEFAULT_HISTORY_COUNT = 1
DEFAULT_RENDERING_CONDITION = "lexical"
RENDERING_CONDITIONS = ("lexical", "paraphrase")
HARDNESS_PROFILES = (
    "base",
    "anti_shortcut_stream_v2",
    "anti_shortcut_interleaved_v3",
)
ANTI_SHORTCUT_PROFILES = {
    "anti_shortcut_stream_v2",
    "anti_shortcut_interleaved_v3",
}
DEFAULT_HARDNESS_PROFILE = "base"
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "dev", "test")


_ALIAS_PREFIXES = (
    "Aster", "Birch", "Cedar", "Dahlia", "Elm", "Fern", "Grove", "Hazel",
    "Iris", "Juniper", "Kestrel", "Laurel", "Maple", "Nettle", "Olive", "Pine",
    "Quartz", "Reed", "Sage", "Thistle", "Umber", "Violet", "Willow", "Xenia",
    "Yarrow", "Zephyr", "Amber", "Bracken", "Clover", "Drift", "Ember", "Flint",
)
_ALIAS_SUFFIXES = (
    "Arc", "Bloom", "Cove", "Dawn", "Echo", "Field", "Glen", "Harbor",
    "Isle", "Jade", "Knoll", "Lake", "Moor", "North", "Oak", "Path",
    "Quill", "Ridge", "Stone", "Trail", "Vale", "West", "Yard", "Zenith",
    "Brook", "Cloud", "Dell", "Frost", "Glass", "Heath", "Light", "Voss",
)
_EVEN_COMPOSITIONS = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
_ODD_COMPOSITIONS = ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1))


def _aliases(index: int) -> tuple[str, str]:
    """Return two deterministic natural aliases without canonical benchmark IDs."""
    if index < 1:
        raise ValueError(f"alias index must be positive, got {index}")
    zero_based = index - 1
    namespace_size = len(_ALIAS_PREFIXES) * len(_ALIAS_SUFFIXES)
    cycle = zero_based // namespace_size
    first = (
        f"{_ALIAS_PREFIXES[zero_based % len(_ALIAS_PREFIXES)]}"
        f"{_ALIAS_SUFFIXES[(zero_based // len(_ALIAS_PREFIXES)) % len(_ALIAS_SUFFIXES)]}"
    )
    if cycle:
        first = f"{first}{cycle + 1}"
    second = f"{first}Guild"
    return first, second


def _composition(index: int, split: str) -> dict[str, Any]:
    """Assign held-out three-way compositions while covering every primitive axis."""
    choices = _ODD_COMPOSITIONS if split == "test" else _EVEN_COMPOSITIONS
    alias_direction, chain_variant, negative_family = choices[(index - 1) % len(choices)]
    return {
        "id": f"alias{alias_direction}-chain{chain_variant}-negative{negative_family}",
        "axes": {
            "alias_direction": alias_direction,
            "chain_variant": chain_variant,
            "negative_family": negative_family,
        },
    }


def preference_rule_parameters() -> RuleParameters:
    """Return benchmark-local rules that make overlapping preferences exclusive."""
    return RuleParameters(
        version=DATASET_VERSION,
        functional_predicates=(*DEFAULT_RULE_PARAMETERS.functional_predicates, "PREFERS"),
    )


def _history_splits(
    history_count: int,
    split_ratios: tuple[float, float, float],
    split_counts: tuple[int, int, int] | None,
) -> dict[str, str]:
    """Assign each history to one deterministic, history-disjoint data split."""
    if split_counts is not None:
        if len(split_counts) != len(SPLIT_NAMES) or any(count < 0 for count in split_counts):
            raise ValueError("split_counts must contain three non-negative counts")
        if sum(split_counts) != history_count:
            raise ValueError("split_counts must sum to history_count")
        counts = split_counts
    else:
        if len(split_ratios) != len(SPLIT_NAMES) or any(ratio < 0 for ratio in split_ratios):
            raise ValueError("split_ratios must contain three non-negative ratios")
        if abs(sum(split_ratios) - 1.0) > 1e-9:
            raise ValueError("split_ratios must sum to 1.0")
        expected_counts = [ratio * history_count for ratio in split_ratios]
        counts = [int(count) for count in expected_counts]
        for index in sorted(
            range(len(SPLIT_NAMES)),
            key=lambda item: (-(expected_counts[item] - counts[item]), item),
        )[:history_count - sum(counts)]:
            counts[index] += 1

    split_by_history: dict[str, str] = {}
    offset = 0
    for split, count in zip(SPLIT_NAMES, counts):
        for index in range(offset + 1, offset + count + 1):
            split_by_history[f"history-{index:03d}"] = split
        offset += count
    return split_by_history


def _fact(
    *,
    fact_id: str,
    example_id: str,
    subject: str,
    object_: str,
    valid_from: str,
    valid_to: str | None,
    support_text: str,
    confidence_score: float = 0.95,
    scope: str = "default",
    observed_at: str = "2025-01-01T00:00:00+00:00",
    source_authority: str = "inferred",
    predicate: str = "PREFERS",
    domain: str = "personal_preference",
) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "session_id": DATASET_VERSION,
        "example_id": example_id,
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "temporal": {
            "valid_from": valid_from,
            "valid_to": valid_to,
            "observed_at": observed_at,
        },
        "qualifiers": {
            "scope": scope,
            "domain": domain,
            "source_authority": source_authority,
        },
        "provenance": [{
            "title": fact_id,
            "document_id": fact_id,
            "sentence_id": f"{fact_id}:0",
            "sent_id": 0,
            "source_span_start": 0,
            "source_span_end": len(support_text),
            "extractor_model": "synthetic_generator.v1",
            "run_id": DATASET_VERSION,
        }],
        "support_text": support_text,
        "question_relevance": (
            "States an explicit preference and its temporal scope."
            if predicate == "PREFERS"
            else "Provides non-preference context for interference and identity reasoning."
        ),
        "confidence": "supported",
        "confidence_score": confidence_score,
        "confidence_method": "synthetic_ground_truth",
        "verification_reason": "Deterministic synthetic ground-truth assertion.",
        "normalization_notes": "",
        "run_id": DATASET_VERSION,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _support_texts(
    *,
    condition: str,
    subject: str,
    initial_value: str,
    current_value: str,
    scoped_value: str,
    forbidden_value: str,
    ambiguous_value: str,
    private_value: str,
    scope: str,
    candidate_subject: str,
    replacement_subject: str,
    accepted_value: str,
    backdated_value: str,
    replaceable_value: str,
    indirect_value: str,
    direct_value: str,
    leakage_value: str,
    replacement_value: str,
) -> dict[str, str]:
    """Render deterministic raw-event text without changing fact fields."""
    if condition == "lexical":
        return {
            "initial": f"{subject} preferred {initial_value} through June 2025.",
            "current": f"{subject} preferred {current_value} from July through September 2025.",
            "scoped": f"{subject} preferred {scoped_value} in {scope} during early August 2025.",
            "constraint": f"{subject} must avoid {forbidden_value}.",
            "ambiguity": f"{subject} gave conflicting statements about {ambiguous_value}.",
            "private": f"{subject}'s private value was recorded as {private_value}.",
            "stale": f"{subject} preferred {initial_value} from July 2025 onward.",
            "accepted": f"{candidate_subject} preferred {accepted_value}.",
            "backdated": f"A correction records that {subject} preferred {backdated_value} during spring 2025.",
            "indirect": f"An inferred record says {subject} preferred {indirect_value} from October 2025 onward.",
            "direct": f"{subject} directly corrected the record to {direct_value} from October 2025 onward.",
            "leakage": f"{subject} preferred {leakage_value} only in leakage-scope-{scope[-3:]} during mid-August 2025.",
            "replaceable": f"{replacement_subject} was tentatively recorded as preferring {replaceable_value}.",
            "replacement": f"Corroborating records establish that {replacement_subject} preferred {replacement_value}.",
        }
    if condition == "paraphrase":
        return {
            "initial": f"The standing selection for {subject} was {initial_value} until the end of June 2025.",
            "current": f"From July through September 2025, {subject}'s standing selection was {current_value}.",
            "scoped": f"For the short-lived {scope} setting, {subject} selected {scoped_value} from August 1 through August 14, 2025.",
            "constraint": f"A safety exclusion bars {subject} from {forbidden_value}.",
            "ambiguity": f"The record leaves {ambiguous_value} unresolved because {subject}'s accounts disagree.",
            "private": f"An access-restricted record associates {subject} with {private_value}.",
            "stale": f"Beginning in July 2025, the record incorrectly assigns {initial_value} to {subject} indefinitely.",
            "accepted": f"The standing selection for {candidate_subject} was {accepted_value}.",
            "backdated": f"A later audit corrects the spring 2025 record: {subject}'s selection was {backdated_value}.",
            "indirect": f"An inferred October 2025 entry associates {subject} with {indirect_value}.",
            "direct": f"In a direct October 2025 correction, {subject} selected {direct_value}.",
            "leakage": f"Within the limited leakage-scope-{scope[-3:]} context, {subject} selected {leakage_value} in mid-August 2025.",
            "replaceable": f"A tentative entry associates {replacement_subject} with {replaceable_value}.",
            "replacement": f"Multiple corroborating records associate {replacement_subject} with {replacement_value}.",
        }
    raise ValueError(f"unsupported rendering condition: {condition}")


def _query_text(query: dict[str, str], condition: str) -> str:
    """Render visible query wording without including the derived gold answer."""
    kind = query["kind"]
    subject = query["subject"]
    if condition == "lexical":
        if kind == "preference":
            return f"What item was preferred by {subject} in {query['scope']} on {query['date']}?"
        if kind == "recommendation":
            return f"Must {subject} avoid {query['candidate']}?"
        if kind == "private_recall":
            return f"What private value was recorded for {subject} under {query['fact_id']}?"
        if kind == "ambiguity":
            return f"Did {subject} give conflicting statements?"
    if condition == "paraphrase":
        if kind == "preference":
            return f"Which option governs {subject} within {query['scope']} as of {query['date']}?"
        if kind == "recommendation":
            return f"May {query['candidate']} be suggested to {subject}?"
        if kind == "private_recall":
            return f"Can the entry identified by {query['fact_id']} still be recalled for {subject}?"
        if kind == "ambiguity":
            return f"Can a determinate state be established for {subject}?"
    raise ValueError(f"unsupported query rendering: condition={condition}, kind={kind}")


def resolve_preference(
    events: list[dict[str, Any]], subject: str, date: str, scope: str = "default"
) -> str | None:
    """Return the active preference using valid time and source-authority tie breaks."""
    active: list[dict[str, Any]] = []
    for event in events:
        if event["operation"] not in {
            "add", "supersede", "temporary_exception", "backdated_correction",
            "duplicate_delivery", "direct_user_correction",
        }:
            continue
        fact = event["fact"]
        if fact.get("predicate") != "PREFERS":
            continue
        temporal = fact["temporal"]
        if fact["subject"] != subject or not (temporal["valid_from"] <= date):
            continue
        if temporal["valid_to"] is not None and date > temporal["valid_to"]:
            continue
        active.append(fact)
    scoped = [fact for fact in active if fact["qualifiers"].get("scope") == scope]
    selected = scoped or [fact for fact in active if fact["qualifiers"].get("scope") == "default"]
    if not selected:
        return None
    authority = {"inferred": 0, "direct_user": 1}
    return max(
        selected,
        key=lambda fact: (
            fact["temporal"]["valid_from"],
            authority.get(fact["qualifiers"].get("source_authority"), 0),
            fact.get("confidence_score", 0.0),
        ),
    )["object"]


def materialize_event_states(events: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    """Return append-only event snapshots with explicit retractions applied."""
    state: dict[str, dict[str, Any]] = {}
    snapshots: list[dict[str, dict[str, Any]]] = []
    for event in events:
        if event["operation"] == "retract":
            state.pop(event["retracts"], None)
        elif event["operation"] == "duplicate_delivery" and event["fact"]["fact_id"] in state:
            pass
        else:
            state[event["fact"]["fact_id"]] = {
                "operation": event["operation"], "fact": event["fact"]
            }
        snapshots.append(dict(state))
    return snapshots


def resolve_query(events: list[dict[str, Any]], query: dict[str, str]) -> str | None:
    """Resolve preference, safety, and privacy queries from an event trace."""
    kind = query.get("kind", "preference")
    if kind == "preference":
        return resolve_preference(events, query["subject"], query["date"], query["scope"])
    snapshots = materialize_event_states(events)
    state = snapshots[-1] if snapshots else {}
    if kind == "recommendation":
        forbidden = {
            item["fact"]["object"]
            for item in state.values()
            if item["operation"] == "hard_constraint" and item["fact"]["subject"] == query["subject"]
        }
        return "INFEASIBLE" if query["candidate"] in forbidden else "ALLOWED"
    if kind == "private_recall":
        return "UNKNOWN" if query["fact_id"] not in state else state[query["fact_id"]]["fact"]["object"]
    if kind == "private_lineage":
        from experiments.private_lineage_reasoning import resolve_private_lineage_reference

        return resolve_private_lineage_reference(events, query, strict=False)["answer"]
    if kind == "ambiguity":
        return "UNKNOWN"
    raise ValueError(f"unsupported query kind: {kind}")


def resolve_preference_with_scallop(
    events: list[dict[str, Any]], subject: str, date: str, scope: str = "default"
) -> str | None:
    """Resolve a scoped active preference through Scallop's rule engine."""
    try:
        import scallopy
    except ImportError as exc:
        raise RuntimeError("scallopy is required for symbolic preference resolution") from exc
    active_by_scope: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["operation"] not in {
            "add", "supersede", "temporary_exception", "backdated_correction",
            "duplicate_delivery", "direct_user_correction",
        }:
            continue
        fact = event["fact"]
        if fact.get("predicate") != "PREFERS":
            continue
        temporal = fact["temporal"]
        if fact["subject"] != subject or temporal["valid_from"] > date:
            continue
        if temporal["valid_to"] is not None and date > temporal["valid_to"]:
            continue
        fact_scope = fact["qualifiers"].get("scope", "default")
        prior = active_by_scope.get(fact_scope)
        authority = {"inferred": 0, "direct_user": 1}
        fact_key = (
            temporal["valid_from"],
            authority.get(fact["qualifiers"].get("source_authority"), 0),
            fact.get("confidence_score", 0.0),
        )
        prior_key = (
            prior["temporal"]["valid_from"],
            authority.get(prior["qualifiers"].get("source_authority"), 0),
            prior.get("confidence_score", 0.0),
        ) if prior is not None else None
        if prior_key is None or prior_key < fact_key:
            active_by_scope[fact_scope] = fact
    ctx = scallopy.ScallopContext()
    ctx.add_relation("active_pref", (str, str, str))
    ctx.add_relation("query_scope", (str,))
    ctx.add_facts(
        "active_pref",
        [(subject, fact["object"], scope_name) for scope_name, fact in active_by_scope.items()],
    )
    ctx.add_facts("query_scope", [(scope,)])
    ctx.add_rule("scoped(s) :- active_pref(s, _, scope), query_scope(scope), scope != \"default\"")
    ctx.add_rule("effective(s, v) :- active_pref(s, v, scope), query_scope(scope), scope != \"default\"")
    ctx.add_rule("effective(s, v) :- active_pref(s, v, \"default\"), query_scope(scope), not scoped(s)")
    ctx.run()
    values = sorted(value for resolved_subject, value in ctx.relation("effective") if resolved_subject == subject)
    return values[0] if len(values) == 1 else None


def generate_dataset(
    output_dir: Path,
    history_count: int = DEFAULT_HISTORY_COUNT,
    condition: str = DEFAULT_RENDERING_CONDITION,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    split_counts: tuple[int, int, int] | None = None,
    hardness_profile: str = DEFAULT_HARDNESS_PROFILE,
) -> dict[str, Path]:
    """Write deterministic histories with surface-text variants and disjoint splits."""
    if history_count < 1:
        raise ValueError(f"history_count must be positive, got {history_count}")
    if condition not in RENDERING_CONDITIONS:
        raise ValueError(f"unsupported rendering condition: {condition}")
    if hardness_profile not in HARDNESS_PROFILES:
        raise ValueError(f"unsupported hardness profile: {hardness_profile}")

    output_dir = Path(output_dir)
    split_by_history = _history_splits(history_count, split_ratios, split_counts)
    facts: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for index in range(1, history_count + 1):
        history_id = f"history-{index:03d}"
        subject = f"subject-{index:03d}"
        initial_value = f"value-{index:03d}-a"
        current_value = f"value-{index:03d}-b"
        scoped_value = f"value-{index:03d}-c"
        forbidden_value = f"value-{index:03d}-d"
        ambiguous_value = f"value-{index:03d}-e"
        private_value = f"value-{index:03d}-f"
        accepted_value = f"value-{index:03d}-g"
        backdated_value = f"value-{index:03d}-h"
        replaceable_value = f"value-{index:03d}-i"
        replacement_value = f"value-{index:03d}-j"
        direct_value = f"value-{index:03d}-k"
        leakage_value = f"value-{index:03d}-l"
        direct_conflict_value = f"value-{index:03d}-m"
        ambiguity_resolution_value = f"value-{index:03d}-n"
        alias_a, alias_b = _aliases(index)
        lineage_value = f"cedar glass {index:03d}"
        composition = _composition(index, split_by_history[history_id])
        composition_axes = composition["axes"]
        query_name = f"{alias_b} account"
        event_alias, query_alias = (
            (alias_a, query_name)
            if composition_axes["alias_direction"] == 0
            else (query_name, alias_a)
        )
        transition_example = f"example-{index:03d}-transition"
        scope_example = f"example-{index:03d}-scope"
        support_texts = _support_texts(
            condition=condition,
            subject=subject,
            initial_value=initial_value,
            current_value=current_value,
            scoped_value=scoped_value,
            forbidden_value=forbidden_value,
            ambiguous_value=ambiguous_value,
            private_value=private_value,
            scope=f"scope-{index:03d}",
            candidate_subject=f"candidate-{index:03d}",
            replacement_subject=f"replacement-{index:03d}",
            accepted_value=accepted_value,
            backdated_value=backdated_value,
            replaceable_value=replaceable_value,
            indirect_value=replaceable_value,
            direct_value=direct_value,
            leakage_value=leakage_value,
            replacement_value=replacement_value,
        )

        initial = _fact(
            fact_id=f"{history_id}-initial", example_id=transition_example, subject=subject,
            object_=initial_value, valid_from="2025-01-01", valid_to="2025-06-30",
            support_text=support_texts["initial"],
            source_authority="direct_user",
        )
        current = _fact(
            fact_id=f"{history_id}-current", example_id=transition_example, subject=subject,
            object_=current_value, valid_from="2025-07-01", valid_to="2025-09-30",
            support_text=support_texts["current"],
            source_authority="direct_user",
        )
        scoped = _fact(
            fact_id=f"{history_id}-scoped", example_id=scope_example, subject=subject,
            object_=scoped_value, valid_from="2025-08-01", valid_to="2025-08-14",
            scope=f"scope-{index:03d}",
            support_text=support_texts["scoped"],
            source_authority="direct_user",
        )
        constraint = _fact(
            fact_id=f"{history_id}-constraint", example_id=scope_example, subject=subject,
            object_=forbidden_value, valid_from="2025-01-01", valid_to=None,
            scope="hard_constraint",
            predicate="AVOIDS",
            domain="hard_constraint",
            support_text=support_texts["constraint"],
            source_authority="direct_user",
        )
        ambiguity = _fact(
            fact_id=f"{history_id}-ambiguity", example_id=scope_example, subject=subject,
            object_=ambiguous_value, valid_from="2025-09-01", valid_to=None,
            scope=f"ambiguity-scope-{index:03d}",
            predicate="AMBIGUOUS_PREFERENCE",
            domain="preference_ambiguity",
            support_text=support_texts["ambiguity"],
            source_authority="direct_user",
        )
        private = _fact(
            fact_id=f"{history_id}-private", example_id=scope_example, subject=subject,
            object_=private_value, valid_from="2025-01-01", valid_to=None, scope="private",
            support_text=support_texts["private"],
            predicate="PRIVATE_NOTE",
            domain="private_memory",
            source_authority="direct_user",
        )
        stale = _fact(
            fact_id=f"{history_id}-stale", example_id=transition_example, subject=subject,
            object_=initial_value, valid_from="2025-07-02", valid_to=None,
            support_text=support_texts["stale"],
            confidence_score=0.10,
        )
        accepted = _fact(
            fact_id=f"{history_id}-accepted", example_id=scope_example,
            subject=f"candidate-{index:03d}", object_=accepted_value,
            valid_from="2025-01-01", valid_to=None,
            support_text=support_texts["accepted"],
        )
        backdated = _fact(
            fact_id=f"{history_id}-backdated", example_id=transition_example, subject=subject,
            object_=backdated_value, valid_from="2025-03-01", valid_to="2025-06-30",
            observed_at="2025-08-20T00:00:00+00:00", support_text=support_texts["backdated"],
            source_authority="direct_user",
        )
        replaceable = _fact(
            fact_id=f"{history_id}-replaceable", example_id=scope_example,
            subject=f"replacement-{index:03d}", object_=replaceable_value,
            valid_from="2025-01-01", valid_to=None, confidence_score=0.10,
            support_text=support_texts["replaceable"],
        )
        replacement = _fact(
            fact_id=f"{history_id}-overlap-replacement", example_id=scope_example,
            subject=f"replacement-{index:03d}", object_=replacement_value,
            valid_from="2025-01-01", valid_to=None, confidence_score=0.99,
            support_text=support_texts["replacement"],
        )
        indirect = _fact(
            fact_id=f"{history_id}-indirect-source", example_id=transition_example, subject=subject,
            object_=replaceable_value, valid_from="2025-10-01", valid_to=None,
            support_text=support_texts["indirect"],
        )
        direct = _fact(
            fact_id=f"{history_id}-direct-correction", example_id=transition_example, subject=subject,
            object_=direct_value, valid_from="2025-10-01", valid_to=None,
            observed_at="2025-10-02T00:00:00+00:00", source_authority="direct_user",
            support_text=support_texts["direct"],
        )
        leakage = _fact(
            fact_id=f"{history_id}-scope-leakage", example_id=scope_example, subject=subject,
            object_=leakage_value, valid_from="2025-08-10", valid_to="2025-08-14",
            scope=f"leakage-scope-{index:03d}", support_text=support_texts["leakage"],
        )
        equal_evidence = _fact(
            fact_id=f"{history_id}-equal-evidence-conflict", example_id=transition_example,
            subject=subject, object_=initial_value, valid_from="2025-07-01", valid_to=None,
            support_text=support_texts["stale"],
        )
        hard_constraint_violation = _fact(
            fact_id=f"{history_id}-hard-constraint-violation", example_id=scope_example,
            subject=subject, object_=forbidden_value, valid_from="2025-07-01", valid_to=None,
            support_text=f"{subject} requested {forbidden_value} despite the standing constraint.",
        )
        resurrection = _fact(
            fact_id=private["fact_id"], example_id=scope_example, subject=subject,
            object_=private_value, valid_from="2025-01-01", valid_to=None, scope="private",
            support_text=support_texts["private"],
            predicate="PRIVATE_NOTE",
            domain="private_memory",
        )
        direct_conflict = _fact(
            fact_id=f"{history_id}-direct-conflict-no-supersession", example_id=transition_example,
            subject=subject, object_=direct_conflict_value, valid_from="2025-10-01", valid_to=None,
            source_authority="direct_user",
            support_text=(
                f"{subject} directly requested {direct_conflict_value} from October 2025 onward."
            ),
        )
        ambiguity_resolution = _fact(
            fact_id=f"{history_id}-ambiguity-resolution", example_id=scope_example,
            subject=subject, object_=ambiguity_resolution_value, valid_from="2025-09-01", valid_to=None,
            scope=f"ambiguity-scope-{index:03d}",
            source_authority="direct_user",
            support_text=(
                f"{subject} directly resolved the scoped ambiguity to {ambiguity_resolution_value}."
            ),
        )
        lineage_root = _fact(
            fact_id=f"{history_id}-lineage-root",
            example_id=f"{history_id}-lineage",
            subject=subject,
            object_=lineage_value,
            valid_from="2026-01-01",
            valid_to=None,
            scope="private",
            predicate="PRIVATE_NOTE",
            domain="private_memory",
            support_text=f'{event_alias} asked that "{lineage_value}" be retained as a private note.',
            source_authority="direct_user",
        )
        lineage_copy_one = _fact(
            fact_id=f"{history_id}-lineage-copy-1",
            example_id=f"{history_id}-lineage",
            subject=subject,
            object_=lineage_value,
            valid_from="2026-01-01",
            valid_to=None,
            scope="private",
            predicate="PRIVATE_NOTE",
            domain="private_memory",
            support_text=f'The private note for {query_alias} was delivered again as "{lineage_value}".',
        )
        lineage_copy_two = _fact(
            fact_id=f"{history_id}-lineage-copy-2",
            example_id=f"{history_id}-lineage",
            subject=subject,
            object_=lineage_value,
            valid_from="2026-01-01",
            valid_to=None,
            scope="private",
            predicate="PRIVATE_NOTE",
            domain="private_memory",
            support_text=f'A later delivery repeated {event_alias}\'s private note, "{lineage_value}".',
        )
        for fact in [
            initial, current, scoped, constraint, ambiguity, private, stale, accepted, backdated,
            replaceable, replacement, indirect, direct, leakage, equal_evidence,
            hard_constraint_violation, resurrection, direct_conflict, ambiguity_resolution,
        ]:
            fact["history_id"] = history_id
            fact["split"] = split_by_history[history_id]
        history_events = [
            {"event_id": f"{history_id}-add", "history_id": history_id, "session_id": f"{history_id}-session-1", "turn_index": 1, "event_family": "non_overlap_transition", "operation": "add", "fact": initial},
            {"event_id": f"{history_id}-transition", "history_id": history_id, "session_id": f"{history_id}-session-2", "turn_index": 1, "event_family": "non_overlap_transition", "operation": "supersede", "fact": current, "transitions_from": initial["fact_id"]},
            {"event_id": f"{history_id}-scope", "history_id": history_id, "session_id": f"{history_id}-session-3", "turn_index": 1, "event_family": "scope_exception", "operation": "temporary_exception", "fact": scoped},
            {"event_id": f"{history_id}-constraint", "history_id": history_id, "session_id": f"{history_id}-session-3", "turn_index": 2, "event_family": "hard_constraint", "operation": "hard_constraint", "fact": constraint},
            {"event_id": f"{history_id}-ambiguity", "history_id": history_id, "session_id": f"{history_id}-session-4", "turn_index": 1, "event_family": "ambiguity", "operation": "ambiguous_conflict", "fact": ambiguity},
            {"event_id": f"{history_id}-private-add", "history_id": history_id, "session_id": f"{history_id}-session-4", "turn_index": 2, "event_family": "retraction", "operation": "add", "fact": private},
            {"event_id": f"{history_id}-retract", "history_id": history_id, "session_id": f"{history_id}-session-5", "turn_index": 1, "event_family": "retraction", "operation": "retract", "fact": private, "retracts": private["fact_id"]},
            {"event_id": f"{history_id}-backdated", "history_id": history_id, "session_id": f"{history_id}-session-6", "turn_index": 1, "event_family": "backdated_correction", "operation": "backdated_correction", "fact": backdated, "corrects": initial["fact_id"]},
            {"event_id": f"{history_id}-duplicate", "history_id": history_id, "session_id": f"{history_id}-session-7", "turn_index": 1, "event_family": "duplicate_delivery", "operation": "duplicate_delivery", "fact": current, "duplicate_of": current["fact_id"]},
            {"event_id": f"{history_id}-replaceable", "history_id": history_id, "session_id": f"{history_id}-session-8", "turn_index": 1, "event_family": "overlapping_replacement", "operation": "add", "fact": replaceable},
            {"event_id": f"{history_id}-indirect-source", "history_id": history_id, "session_id": f"{history_id}-session-9", "turn_index": 1, "event_family": "source_authority_conflict", "operation": "add", "fact": indirect},
            {"event_id": f"{history_id}-direct-correction", "history_id": history_id, "session_id": f"{history_id}-session-9", "turn_index": 2, "event_family": "source_authority_conflict", "operation": "direct_user_correction", "fact": direct, "supersedes": indirect["fact_id"], "resolves": [indirect["fact_id"]]},
            {"event_id": f"{history_id}-scope-leakage", "history_id": history_id, "session_id": f"{history_id}-session-10", "turn_index": 1, "event_family": "scope_leakage", "operation": "temporary_exception", "fact": leakage},
        ]
        if hardness_profile in ANTI_SHORTCUT_PROFILES:
            negative_families = (
                [
                    f'{query_alias} discussed "{lineage_value}" during a conversation about private notes.',
                    f'A message to {query_alias} asked whether "{lineage_value}" should replace an existing note.',
                    f'{event_alias}\'s travel profile used "{lineage_value}" as a temporary label.',
                ],
                [
                    f'A planning card for {query_alias} quoted "{lineage_value}" beside a private-note heading.',
                    f'{event_alias} compared the labels "{lineage_value}" and "ember field" in meeting notes.',
                    f'A reminder sent to {query_alias} repeated "{lineage_value}" while discussing account settings.',
                ],
            )
            note_texts = negative_families[composition_axes["negative_family"]]
            note_facts = [
                _fact(
                    fact_id=f"{history_id}-lineage-negative-{note_index}",
                    example_id=f"{history_id}-lineage",
                    subject=subject,
                    object_=lineage_value,
                    valid_from="2026-01-01",
                    valid_to=None,
                    scope="private",
                    predicate="MENTIONS",
                    domain="conversation_context",
                    support_text=text,
                )
                for note_index, text in enumerate(note_texts, start=1)
            ]
            alias_fact = _fact(
                fact_id=f"{history_id}-lineage-alias",
                example_id=f"{history_id}-lineage",
                subject=subject,
                object_=alias_b,
                valid_from="2026-01-01",
                valid_to=None,
                scope="identity",
                predicate="SAME_ACCOUNT",
                domain="identity_resolution",
                support_text=(
                    f"{event_alias} and the {query_alias} refer to the same account. "
                    f"The standing preference history belongs to the {query_alias}."
                ),
            )
            preference_change_probe = _fact(
                fact_id=f"{history_id}-preference-change-probe",
                example_id=f"{history_id}-long-memory",
                subject=subject,
                object_="archive review",
                valid_from="2026-02-01",
                valid_to=None,
                scope="context",
                predicate="CONTEXT_NOTE",
                domain="conversation_context",
                support_text=(
                    f"Much later, {query_alias} reviewed unrelated calendar and account settings."
                ),
            )
            preference_incongruity_probe = _fact(
                fact_id=f"{history_id}-preference-incongruity-probe",
                example_id=f"{history_id}-long-memory",
                subject=subject,
                object_="archive follow-up",
                valid_from="2026-03-01",
                valid_to=None,
                scope="context",
                predicate="CONTEXT_NOTE",
                domain="conversation_context",
                support_text=(
                    f"During a later account review, {event_alias} revisited the long-running record."
                ),
            )
            conflict_left = _fact(
                fact_id=f"{history_id}-conflict-left",
                example_id=f"{history_id}-contradiction",
                subject=subject,
                object_=f"quiet studio {index:03d}",
                valid_from="2027-01-01",
                valid_to=None,
                scope="project_workspace",
                observed_at="2027-01-01T00:00:00+00:00",
                source_authority="direct_user",
                support_text=(
                    f"While planning a long project, {event_alias} preferred a quiet private studio."
                ),
            )
            conflict_right = _fact(
                fact_id=f"{history_id}-conflict-right",
                example_id=f"{history_id}-contradiction",
                subject=subject,
                object_=f"open team lounge {index:03d}",
                valid_from="2027-01-01",
                valid_to=None,
                scope="project_workspace",
                observed_at="2027-02-01T00:00:00+00:00",
                source_authority="direct_user",
                support_text=(
                    f"Weeks later, the {query_alias} instead requested an open team lounge for the same project."
                ),
            )
            conflict_resolution = _fact(
                fact_id=f"{history_id}-conflict-resolution",
                example_id=f"{history_id}-contradiction",
                subject=subject,
                object_=f"window desk {index:03d}",
                valid_from="2027-03-01",
                valid_to=None,
                scope="project_workspace",
                source_authority="direct_user",
                observed_at="2027-03-01T00:00:00+00:00",
                support_text=(
                    f"After reviewing both incompatible requests, {event_alias} directly chose a window desk for the project."
                ),
            )
            for fact in [
                alias_fact,
                *note_facts,
                lineage_root,
                lineage_copy_one,
                lineage_copy_two,
                preference_change_probe,
                preference_incongruity_probe,
                conflict_left,
                conflict_right,
                conflict_resolution,
            ]:
                fact["history_id"] = history_id
                fact["split"] = split_by_history[history_id]
            history_events.extend(
                [
                    {"event_id": f"{history_id}-lineage-alias", "history_id": history_id, "session_id": f"{history_id}-session-11", "turn_index": 1, "event_family": "alias_bridge", "operation": "context_note", "fact": alias_fact, "model_text": alias_fact["support_text"]},
                    *[
                        {"event_id": f"{history_id}-lineage-negative-{note_index}", "history_id": history_id, "session_id": f"{history_id}-session-11", "turn_index": note_index + 1, "event_family": "same_entity_hard_negative", "operation": "context_note", "fact": fact, "model_text": fact["support_text"]}
                        for note_index, fact in enumerate(note_facts, start=1)
                    ],
                    {"event_id": f"{history_id}-lineage-add", "history_id": history_id, "session_id": f"{history_id}-session-12", "turn_index": 1, "event_family": "private_lineage", "operation": "add", "fact": lineage_root, "model_text": lineage_root["support_text"]},
                    {"event_id": f"{history_id}-lineage-copy-1", "history_id": history_id, "session_id": f"{history_id}-session-13", "turn_index": 1, "event_family": "private_lineage", "operation": "duplicate_delivery", "fact": lineage_copy_one, "duplicate_of": lineage_root["fact_id"], "model_text": lineage_copy_one["support_text"]},
                    {"event_id": f"{history_id}-lineage-copy-2", "history_id": history_id, "session_id": f"{history_id}-session-14", "turn_index": 1, "event_family": "private_lineage", "operation": "duplicate_delivery", "fact": lineage_copy_two, "duplicate_of": lineage_copy_one["fact_id"], "model_text": lineage_copy_two["support_text"]},
                    {"event_id": f"{history_id}-lineage-retract", "history_id": history_id, "session_id": f"{history_id}-session-15", "turn_index": 1, "event_family": "private_lineage", "operation": "retract", "fact": (lineage_copy_two if composition_axes["chain_variant"] == 0 else lineage_root), "retracts": (lineage_copy_two["fact_id"] if composition_axes["chain_variant"] == 0 else lineage_root["fact_id"]), "retracts_lineage": True, "model_text": f'{query_alias} withdrew one delivery of the private note and requested that the linked note no longer be recalled.'},
                    {"event_id": f"{history_id}-preference-change-probe", "history_id": history_id, "session_id": f"{history_id}-session-16", "turn_index": 1, "event_family": "delayed_preference_probe", "operation": "context_note", "fact": preference_change_probe, "model_text": preference_change_probe["support_text"]},
                    {"event_id": f"{history_id}-preference-incongruity-probe", "history_id": history_id, "session_id": f"{history_id}-session-17", "turn_index": 1, "event_family": "delayed_preference_probe", "operation": "context_note", "fact": preference_incongruity_probe, "model_text": preference_incongruity_probe["support_text"]},
                ]
            )
            if hardness_profile == "anti_shortcut_interleaved_v3":
                history_events.insert(4, {"event_id": f"{history_id}-conflict-left", "history_id": history_id, "session_id": f"{history_id}-session-18", "turn_index": 1, "event_family": "contradiction_opening", "operation": "add", "fact": conflict_left, "model_text": conflict_left["support_text"]})
                history_events.insert(13, {"event_id": f"{history_id}-conflict-right", "history_id": history_id, "session_id": f"{history_id}-session-19", "turn_index": 1, "event_family": "contradiction_opening", "operation": "add", "fact": conflict_right, "conflicts_with": conflict_left["fact_id"], "model_text": conflict_right["support_text"]})
                history_events.insert(21, {"event_id": f"{history_id}-conflict-resolution", "history_id": history_id, "session_id": f"{history_id}-session-20", "turn_index": 1, "event_family": "contradiction_rectification", "operation": "direct_user_correction", "fact": conflict_resolution, "resolves": [conflict_left["fact_id"], conflict_right["fact_id"]], "model_text": conflict_resolution["support_text"]})
            natural_overrides = {
                "retract": f"{subject} withdrew the previously recorded private value.",
                "duplicate": f"The standing July preference for {subject} was delivered again.",
            }
            for event in history_events:
                if "model_text" not in event:
                    suffix = event["event_id"].removeprefix(f"{history_id}-")
                    event["model_text"] = natural_overrides.get(
                        suffix, str(event["fact"]["support_text"])
                    )
                event["model_text"] = event["model_text"].replace(subject, event_alias)
                event["model_text"] = event["model_text"].replace(
                    f"scope-{index:03d}", "shared setting"
                )
                event["model_text"] = event["model_text"].replace(
                    f"leakage-scope-{index % 1000:03d}", "limited setting"
                )
        history_queries = [
            {"query_id": f"{history_id}-current", "history_id": history_id, "kind": "preference", "subject": subject, "date": "2025-08-01", "scope": "default"},
            {"query_id": f"{history_id}-scope", "history_id": history_id, "kind": "preference", "subject": subject, "date": "2025-08-05", "scope": f"scope-{index:03d}"},
            {"query_id": f"{history_id}-constraint", "history_id": history_id, "kind": "recommendation", "subject": subject, "candidate": forbidden_value},
            {"query_id": f"{history_id}-private", "history_id": history_id, "kind": "private_recall", "subject": subject, "fact_id": private["fact_id"]},
            {"query_id": f"{history_id}-ambiguity", "history_id": history_id, "kind": "ambiguity", "subject": subject, "fact_id": ambiguity["fact_id"], "candidate": ambiguous_value},
            {"query_id": f"{history_id}-backdated", "history_id": history_id, "kind": "preference", "subject": subject, "date": "2025-05-01", "scope": "default"},
            {"query_id": f"{history_id}-duplicate", "history_id": history_id, "kind": "preference", "subject": subject, "date": "2025-08-01", "scope": "default"},
            {"query_id": f"{history_id}-authority", "history_id": history_id, "kind": "preference", "subject": subject, "date": "2025-10-15", "scope": "default"},
            {"query_id": f"{history_id}-scope-leakage", "history_id": history_id, "kind": "preference", "subject": subject, "date": "2025-08-12", "scope": "default"},
        ]
        if hardness_profile in ANTI_SHORTCUT_PROFILES:
            lineage_contract = {
                "required_event_id_groups": [
                    [f"{history_id}-lineage-alias"],
                    [f"{history_id}-lineage-add"],
                    [f"{history_id}-lineage-copy-1"],
                    [f"{history_id}-lineage-copy-2"],
                ],
                "required_fact_id_groups": [
                    [lineage_root["fact_id"]],
                    [lineage_copy_one["fact_id"]],
                    [lineage_copy_two["fact_id"]],
                ],
            }
            requested_lineage_fact = (
                lineage_root
                if composition_axes["chain_variant"] == 0
                else lineage_copy_two
            )
            history_queries.extend(
                [
                    {"query_id": f"{history_id}-private-lineage-positive", "history_id": history_id, "kind": "private_lineage", "subject": subject, "fact_id": requested_lineage_fact["fact_id"], "query_text": f"What private value, if any, can still be recalled for {query_alias}?", "checkpoint_contract": {**lineage_contract, "trigger_event_id": f"{history_id}-lineage-copy-2"}, "composition": composition},
                    {"query_id": f"{history_id}-private-lineage", "history_id": history_id, "kind": "private_lineage", "subject": subject, "fact_id": requested_lineage_fact["fact_id"], "query_text": f"After the withdrawal, what private value, if any, can still be recalled for {query_alias}?", "checkpoint_contract": {"trigger_event_id": f"{history_id}-lineage-retract", "required_event_id_groups": [*lineage_contract["required_event_id_groups"], [f"{history_id}-lineage-retract"]], "required_fact_id_groups": lineage_contract["required_fact_id_groups"]}, "composition": composition},
                    {"query_id": f"{history_id}-preference-change-delayed", "history_id": history_id, "kind": "preference", "subject": subject, "date": "2025-08-01", "scope": "default", "query_text": f"After the long record, what standing preference applies to {query_alias} on 2025-08-01?", "checkpoint_contract": {"trigger_event_id": f"{history_id}-preference-change-probe", "required_event_id_groups": [[f"{history_id}-add"], [f"{history_id}-transition"]], "required_fact_id_groups": [[initial["fact_id"]], [current["fact_id"]]]}, "composition": composition},
                    {"query_id": f"{history_id}-preference-incongruity-delayed", "history_id": history_id, "kind": "preference", "subject": subject, "date": "2025-10-15", "scope": "default", "query_text": f"Considering the complete record, what standing preference applies to {query_alias} on 2025-10-15?", "checkpoint_contract": {"trigger_event_id": f"{history_id}-preference-incongruity-probe", "required_event_id_groups": [[f"{history_id}-indirect-source"], [f"{history_id}-direct-correction"]], "required_fact_id_groups": [[indirect["fact_id"]], [direct["fact_id"]]]}, "composition": composition},
                ]
            )
        for query in history_queries:
            if "query_text" not in query:
                query["query_text"] = _query_text(query, condition)
            if hardness_profile in ANTI_SHORTCUT_PROFILES:
                query["query_text"] = query["query_text"].replace(subject, event_alias)
                query["query_text"] = query["query_text"].replace(
                    f"scope-{index:03d}", "shared setting"
                )
                if query["kind"] == "private_recall":
                    query["query_text"] = (
                        f"Can the private entry still be recalled for {event_alias}?"
                    )
            query["split"] = split_by_history[history_id]
            query["hardness_profile"] = hardness_profile
        for event in history_events:
            event["split"] = split_by_history[history_id]
            event["hardness_profile"] = hardness_profile
            if (
                hardness_profile in ANTI_SHORTCUT_PROFILES
                and event["fact"]["subject"] == subject
            ):
                event["surface_subject"] = event_alias
        facts.extend([initial, current, scoped, backdated, replaceable, indirect, direct, leakage])
        if hardness_profile in ANTI_SHORTCUT_PROFILES:
            facts.extend([
                lineage_root,
                lineage_copy_one,
                lineage_copy_two,
                preference_change_probe,
                preference_incongruity_probe,
                conflict_left,
                conflict_right,
                conflict_resolution,
            ])
        events.extend(history_events)
        example_start = len(examples)
        examples.extend([
            {"_id": transition_example, "history_id": history_id, "context": f"{initial['support_text']} {current['support_text']}", "question": f"What does {subject} prefer on 2025-08-01?", "answer": current_value, "gold_operation": "accept_transition", "gold_fact_ids": [current["fact_id"]]},
            {"_id": scope_example, "history_id": history_id, "context": f"{current['support_text']} {scoped['support_text']}", "question": f"What does {subject} prefer in scope-{index:03d} on 2025-08-05?", "answer": scoped_value, "gold_operation": "scoped_preference", "gold_fact_ids": [scoped["fact_id"]]},
            {"_id": f"{history_id}-backdated", "history_id": history_id, "context": f"{current['support_text']} {backdated['support_text']}", "question": f"What did {subject} prefer on 2025-05-01?", "answer": backdated_value, "gold_operation": "backdated_correction", "gold_fact_ids": [backdated["fact_id"]]},
            {"_id": f"{history_id}-duplicate", "history_id": history_id, "context": current["support_text"], "question": f"What does {subject} prefer on 2025-08-01 after duplicate delivery?", "answer": current_value, "gold_operation": "idempotent_duplicate", "gold_fact_ids": [current["fact_id"]]},
            {"_id": f"{history_id}-replacement", "history_id": history_id, "context": f"{replaceable['support_text']} {replacement['support_text']}", "question": f"Which update replaces replacement-{index:03d}'s tentative preference?", "answer": replacement_value, "gold_operation": "replace_overlapping_candidate", "gold_fact_ids": [replacement["fact_id"]]},
            {"_id": f"{history_id}-authority", "history_id": history_id, "context": f"{indirect['support_text']} {direct['support_text']}", "question": f"What does {subject} prefer on 2025-10-15?", "answer": direct_value, "gold_operation": "direct_user_authority", "gold_fact_ids": [direct["fact_id"]]},
            {"_id": f"{history_id}-scope-leakage", "history_id": history_id, "context": f"{current['support_text']} {leakage['support_text']}", "question": f"What does {subject} prefer by default on 2025-08-12?", "answer": current_value, "gold_operation": "prevent_scope_leakage", "gold_fact_ids": [current["fact_id"]]},
        ])
        for example in examples[example_start:]:
            example["split"] = split_by_history[history_id]
        for query in history_queries:
            query_events = history_events
            checkpoint_contract = query.get("checkpoint_contract")
            if checkpoint_contract is not None:
                trigger_event_id = checkpoint_contract["trigger_event_id"]
                trigger_index = next(
                    (
                        position
                        for position, event in enumerate(history_events)
                        if event["event_id"] == trigger_event_id
                    ),
                    None,
                )
                if trigger_index is None:
                    raise ValueError(
                        f"query {query['query_id']} references missing trigger event "
                        f"{trigger_event_id}"
                    )
                query_events = history_events[: trigger_index + 1]
            queries.append({**query, "gold": resolve_query(query_events, query)})
        history_candidates = [
            {
                "candidate_id": f"{history_id}-stale",
                "history_id": history_id,
                "event_family": "low_evidence_conflicting_reject",
                "observed_after_event_id": f"{history_id}-transition",
                "fact": stale,
                "gold_decision": "reject",
                "gold_hard_gate": True,
                "gold_soft_label": "reject",
                "gold_reason_code": "low_evidence_conflict",
            },
            {
                "candidate_id": f"{history_id}-accepted",
                "history_id": history_id,
                "event_family": "uncontested_accept",
                "observed_after_event_id": f"{history_id}-scope",
                "fact": accepted,
                "gold_decision": "accept",
                "gold_hard_gate": True,
                "gold_soft_label": "accept",
                "gold_reason_code": "no_conflict",
            },
            {
                "candidate_id": f"{history_id}-overlap-replacement",
                "history_id": history_id,
                "event_family": "high_evidence_conflicting_replace",
                "observed_after_event_id": f"{history_id}-replaceable",
                "fact": replacement,
                "gold_decision": "replace",
                "gold_replace_fact_id": replaceable["fact_id"],
                "gold_hard_gate": True,
                "gold_soft_label": "replace",
                "gold_reason_code": "higher_evidence_conflict",
            },
            {
                "candidate_id": f"{history_id}-equal-evidence-conflict",
                "history_id": history_id,
                "event_family": "equal_evidence_conflict_reject",
                "observed_after_event_id": f"{history_id}-transition",
                "fact": equal_evidence,
                "gold_decision": "reject",
                "gold_hard_gate": True,
                "gold_soft_label": "reject",
                "gold_reason_code": "equal_evidence_conflict",
            },
            {
                "candidate_id": f"{history_id}-hard-constraint-violation",
                "history_id": history_id,
                "event_family": "hard_constraint_violation_reject",
                "observed_after_event_id": f"{history_id}-constraint",
                "fact": hard_constraint_violation,
                "gold_decision": "reject",
                "gold_hard_gate": False,
                "gold_soft_label": "reject",
                "gold_reason_code": "hard_constraint_violation",
            },
            {
                "candidate_id": f"{history_id}-retraction-resurrection",
                "history_id": history_id,
                "event_family": "retraction_tombstone_resurrection_reject",
                "observed_after_event_id": f"{history_id}-retract",
                "fact": resurrection,
                "gold_decision": "reject",
                "gold_hard_gate": False,
                "gold_soft_label": "reject",
                "gold_reason_code": "retraction_tombstone",
            },
            {
                "candidate_id": f"{history_id}-direct-correction-replacement",
                "history_id": history_id,
                "event_family": "direct_user_correction_with_valid_supersedes_replacement",
                "observed_after_event_id": f"{history_id}-indirect-source",
                "fact": direct,
                "supersedes": indirect["fact_id"],
                "gold_decision": "replace",
                "gold_replace_fact_id": indirect["fact_id"],
                "gold_hard_gate": True,
                "gold_soft_label": "replace",
                "gold_reason_code": "direct_user_supersedes_conflict",
            },
            {
                "candidate_id": f"{history_id}-direct-conflict-no-supersession",
                "history_id": history_id,
                "event_family": "direct_user_conflict_lacking_supersession_reject",
                "observed_after_event_id": f"{history_id}-direct-correction",
                "fact": direct_conflict,
                "gold_decision": "reject",
                "gold_hard_gate": False,
                "gold_soft_label": "reject",
                "gold_reason_code": "direct_user_conflict_without_supersedes",
            },
            {
                "candidate_id": f"{history_id}-ambiguity-resolution",
                "history_id": history_id,
                "event_family": "ambiguity_resolving_direct_correction",
                "observed_after_event_id": f"{history_id}-ambiguity",
                "fact": ambiguity_resolution,
                "supersedes": ambiguity["fact_id"],
                "gold_decision": "accept",
                "gold_hard_gate": True,
                "gold_soft_label": "accept",
                "gold_reason_code": "direct_user_resolves_ambiguity",
            },
        ]
        for candidate in history_candidates:
            candidate["split"] = split_by_history[history_id]
            candidate["hardness_profile"] = hardness_profile
            candidate["candidate_features"] = derive_candidate_features(history_events, candidate)
        candidates.extend(history_candidates)
    for sequence_index, event in enumerate(events):
        event["sequence_index"] = sequence_index
    canonical_facts_by_id: dict[str, dict[str, Any]] = {}
    for fact in [*facts, *[event["fact"] for event in events]]:
        fact_id = str(fact["fact_id"])
        prior = canonical_facts_by_id.get(fact_id)
        if prior is not None and prior != fact:
            raise ValueError(f"conflicting canonical facts for {fact_id}")
        canonical_facts_by_id[fact_id] = fact
    facts = [canonical_facts_by_id[key] for key in sorted(canonical_facts_by_id)]
    source_documents_by_id: dict[str, dict[str, str]] = {}
    source_facts = [
        *facts,
        *[event["fact"] for event in events],
        *[candidate["fact"] for candidate in candidates],
    ]
    for fact in source_facts:
        document_id = str(fact["provenance"][0]["document_id"])
        document = {
            "document_id": document_id,
            "history_id": str(fact["history_id"]),
            "split": str(fact["split"]),
            "text": str(fact["support_text"]),
        }
        prior = source_documents_by_id.get(document_id)
        if prior is not None and prior != document:
            raise ValueError(f"conflicting source documents for {document_id}")
        source_documents_by_id[document_id] = document
    paths = {
        "examples": output_dir / "examples.jsonl",
        "facts": output_dir / "facts.jsonl",
        "candidates": output_dir / "candidate_updates.jsonl",
        "events": output_dir / "events.jsonl",
        "queries": output_dir / "queries.jsonl",
        "source_documents": output_dir / "source_documents.jsonl",
    }
    _write_jsonl(paths["examples"], examples)
    _write_jsonl(paths["facts"], facts)
    _write_jsonl(paths["candidates"], candidates)
    _write_jsonl(paths["events"], events)
    _write_jsonl(paths["queries"], queries)
    _write_jsonl(
        paths["source_documents"],
        [source_documents_by_id[key] for key in sorted(source_documents_by_id)],
    )
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-count", type=int, default=DEFAULT_HISTORY_COUNT)
    parser.add_argument("--condition", choices=RENDERING_CONDITIONS, default=DEFAULT_RENDERING_CONDITION)
    parser.add_argument(
        "--hardness-profile",
        choices=HARDNESS_PROFILES,
        default=DEFAULT_HARDNESS_PROFILE,
    )
    parser.add_argument(
        "--split-ratios",
        default=",".join(str(ratio) for ratio in DEFAULT_SPLIT_RATIOS),
        help="comma-separated train,dev,test ratios summing to 1.0",
    )
    parser.add_argument(
        "--split-counts",
        help="optional comma-separated train,dev,test counts summing to --history-count",
    )
    args = parser.parse_args(argv)
    split_ratios = tuple(float(value) for value in args.split_ratios.split(","))
    split_counts = tuple(int(value) for value in args.split_counts.split(",")) if args.split_counts else None
    for name, path in generate_dataset(
        args.output_dir,
        args.history_count,
        args.condition,
        split_ratios=split_ratios,
        split_counts=split_counts,
        hardness_profile=args.hardness_profile,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
