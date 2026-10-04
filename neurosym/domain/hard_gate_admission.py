"""Deterministic hard-gate admission for candidate memory updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AdmissionDecision:
    """A hard-gate result that downstream scoring can consume unchanged."""

    eligible: bool
    reason_code: str
    related_fact_id: str | None = None


def _fact_id(fact: Mapping[str, Any]) -> str:
    """Return a fact identifier, treating malformed facts as unidentified."""
    return str(fact.get("fact_id") or "")


def _temporal_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two fact validity intervals overlap."""
    left_temporal = left.get("temporal")
    right_temporal = right.get("temporal")
    if not isinstance(left_temporal, Mapping) or not isinstance(right_temporal, Mapping):
        return False
    left_start = str(left_temporal.get("valid_from") or "")
    left_end = str(left_temporal.get("valid_to") or "9999-12-31")
    right_start = str(right_temporal.get("valid_from") or "")
    right_end = str(right_temporal.get("valid_to") or "9999-12-31")
    return max(left_start, right_start) <= min(left_end, right_end)


def _same_claim(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two facts compete for the same temporal assertion."""
    return (
        left.get("subject") == right.get("subject")
        and left.get("predicate") == right.get("predicate")
        and _scope(left) == _scope(right)
        and left.get("object") != right.get("object")
        and _temporal_overlap(left, right)
    )


def _source_authority(fact: Mapping[str, Any]) -> str:
    """Return the source-authority qualifier, if present."""
    qualifiers = fact.get("qualifiers")
    return str(qualifiers.get("source_authority") or "") if isinstance(qualifiers, Mapping) else ""


def _scope(fact: Mapping[str, Any]) -> str:
    """Return the fact scope used to distinguish semantically separate claims."""
    qualifiers = fact.get("qualifiers")
    return str(qualifiers.get("scope") or "default") if isinstance(qualifiers, Mapping) else "default"


def _same_tombstoned_content(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether a new identifier attempts to restore retracted semantic content."""
    return (
        left.get("subject") == right.get("subject")
        and left.get("predicate") == right.get("predicate")
        and left.get("object") == right.get("object")
        and _scope(left) == _scope(right)
        and _temporal_overlap(left, right)
    )


def _active_events(history_events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Materialize current events, preserving input order and retractions."""
    active: dict[str, Mapping[str, Any]] = {}
    for event in history_events:
        operation = str(event.get("operation") or "")
        if operation == "retract":
            active.pop(str(event.get("retracts") or ""), None)
            continue
        resolved = event.get("resolves") or []
        if not isinstance(resolved, Sequence) or isinstance(resolved, (str, bytes)):
            raise ValueError(f"event {event.get('event_id')} resolves must be a sequence")
        supersedes = event.get("supersedes")
        resolved_ids = [str(identifier) for identifier in resolved]
        if supersedes is not None:
            resolved_ids.append(str(supersedes))
        for resolved_id in resolved_ids:
            active.pop(resolved_id, None)
        fact = event.get("fact")
        if not isinstance(fact, Mapping):
            continue
        fact_id = _fact_id(fact)
        if fact_id:
            active[fact_id] = event
    return list(active.values())


def _events_at_candidate(
    history_events: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Return the event prefix visible when the candidate was observed."""
    observed_after = str(candidate.get("observed_after_event_id") or "")
    if not observed_after:
        return list(history_events)
    visible: list[Mapping[str, Any]] = []
    for event in history_events:
        visible.append(event)
        if str(event.get("event_id") or "") == observed_after:
            return visible
    raise ValueError(f"candidate references unknown observed_after_event_id: {observed_after}")


def _tombstones(
    history_events: Sequence[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any] | None]]:
    """Recover retracted facts from prior events when compact retracts carry only IDs."""
    observed_facts: dict[str, Mapping[str, Any]] = {}
    tombstones: list[tuple[str, Mapping[str, Any] | None]] = []
    for event in history_events:
        operation = str(event.get("operation") or "")
        fact = event.get("fact")
        if operation == "retract":
            retracted_id = str(event.get("retracts") or "")
            embedded = fact if isinstance(fact, Mapping) else None
            tombstones.append((retracted_id, embedded or observed_facts.get(retracted_id)))
        elif isinstance(fact, Mapping):
            fact_id = _fact_id(fact)
            if fact_id:
                observed_facts[fact_id] = fact
    return tombstones


def derive_candidate_features(
    history_events: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive admission-time evidence from history and candidate content only."""
    fact = candidate.get("fact")
    if not isinstance(fact, Mapping):
        raise ValueError("candidate must contain a fact mapping")
    history_events = _events_at_candidate(history_events, candidate)
    active_events = _active_events(history_events)
    active_facts = [event["fact"] for event in active_events]
    conflicts = [active_fact for active_fact in active_facts if _same_claim(active_fact, fact)]
    confidence_scores = [float(conflict.get("confidence_score") or 0.0) for conflict in conflicts]
    supersedes = str(candidate.get("supersedes") or "")
    tombstones = _tombstones(history_events)
    violates_hard_constraint = any(
        str(event.get("operation") or "") == "hard_constraint"
        and event["fact"].get("subject") == fact.get("subject")
        and event["fact"].get("object") == fact.get("object")
        for event in active_events
    )
    targets_retracted_fact = any(
        retracted_id == _fact_id(fact)
        or retracted_fact is not None and _same_tombstoned_content(retracted_fact, fact)
        for retracted_id, retracted_fact in tombstones
    )
    ambiguities = [
        event["fact"]
        for event in active_events
        if str(event.get("operation") or "") == "ambiguous_conflict"
        and supersedes == _fact_id(event["fact"])
    ]
    resolves_ambiguity = bool(
        ambiguities
        and _source_authority(fact) == "direct_user"
        and supersedes in {_fact_id(ambiguity) for ambiguity in ambiguities}
    )
    return {
        "candidate_confidence_score": float(fact.get("confidence_score") or 0.0),
        "candidate_source_authority": _source_authority(fact),
        "candidate_has_supersedes": bool(supersedes),
        "active_conflict_count": len(conflicts),
        "max_active_conflict_confidence_score": max(confidence_scores, default=0.0),
        "equal_evidence_conflict": any(
            score == float(fact.get("confidence_score") or 0.0) for score in confidence_scores
        ),
        "violates_hard_constraint": violates_hard_constraint,
        "targets_retracted_fact": targets_retracted_fact,
        "resolves_ambiguity": resolves_ambiguity,
    }


def admit_candidate(
    history_events: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> AdmissionDecision:
    """Return a deterministic hard-gate decision without reading gold metadata.

    This oracle only rejects safety, tombstone, authority, and unresolved-ambiguity
    violations. All soft evidence conflicts remain eligible for a future scorer.
    """
    fact = candidate.get("fact")
    if not isinstance(fact, Mapping):
        raise ValueError("candidate must contain a fact mapping")

    history_events = _events_at_candidate(history_events, candidate)
    active_events = _active_events(history_events)
    active_facts = [event["fact"] for event in active_events]
    candidate_id = _fact_id(fact)
    supersedes = str(candidate.get("supersedes") or "")

    for retracted_id, retracted_fact in _tombstones(history_events):
        if (
            retracted_id == candidate_id
            or isinstance(retracted_fact, Mapping)
            and _same_tombstoned_content(retracted_fact, fact)
        ):
            return AdmissionDecision(
                False,
                "tombstone_resurrection",
                retracted_id,
            )

    for event in active_events:
        constraint = event["fact"]
        if (
            str(event.get("operation") or "") == "hard_constraint"
            and constraint.get("subject") == fact.get("subject")
            and constraint.get("object") == fact.get("object")
        ):
            return AdmissionDecision(False, "hard_constraint_violation", _fact_id(constraint))

    ambiguities = [
        event["fact"]
        for event in active_events
        if str(event.get("operation") or "") == "ambiguous_conflict"
        and (
            supersedes == _fact_id(event["fact"])
            or (
                event["fact"].get("subject") == fact.get("subject")
                and _scope(event["fact"]) == _scope(fact)
                and _temporal_overlap(event["fact"], fact)
            )
        )
    ]
    if ambiguities:
        ambiguity_ids = {_fact_id(ambiguity) for ambiguity in ambiguities}
        if _source_authority(fact) != "direct_user" or supersedes not in ambiguity_ids:
            return AdmissionDecision(False, "unresolved_ambiguity", sorted(ambiguity_ids)[0])
        return AdmissionDecision(True, "direct_correction_resolves_ambiguity", supersedes)

    is_direct_correction = _source_authority(fact) == "direct_user"
    if is_direct_correction:
        conflicts = [
            active_fact
            for active_fact in active_facts
            if _same_claim(active_fact, fact)
        ]
        conflict_ids = {_fact_id(conflict) for conflict in conflicts}
        if conflicts and supersedes not in conflict_ids:
            return AdmissionDecision(False, "authority_conflict_without_supersession", sorted(conflict_ids)[0])
        if supersedes in conflict_ids:
            return AdmissionDecision(True, "direct_correction_supersedes", supersedes)

    return AdmissionDecision(True, "no_hard_gate")
