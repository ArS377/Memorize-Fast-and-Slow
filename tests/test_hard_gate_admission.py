from __future__ import annotations

import json
from pathlib import Path

from experiments.synthetic_temporal_preferences import generate_dataset
from neurosym.domain.hard_gate_admission import admit_candidate


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_hard_gate_oracle_handles_generated_candidate_families(tmp_path: Path) -> None:
    """Hard-gate outcomes derive from event evidence, not candidate gold fields."""
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    candidates = {row["event_family"]: row for row in _rows(paths["candidates"])}

    decisions = {
        family: admit_candidate(events, candidate)
        for family, candidate in candidates.items()
    }

    assert decisions["hard_constraint_violation_reject"].eligible is False
    assert decisions["hard_constraint_violation_reject"].reason_code == "hard_constraint_violation"
    assert decisions["retraction_tombstone_resurrection_reject"].eligible is False
    assert decisions["retraction_tombstone_resurrection_reject"].reason_code == "tombstone_resurrection"
    assert decisions["direct_user_correction_with_valid_supersedes_replacement"].eligible is True
    assert decisions["direct_user_correction_with_valid_supersedes_replacement"].reason_code == "direct_correction_supersedes"
    assert decisions["direct_user_conflict_lacking_supersession_reject"].eligible is False
    assert decisions["direct_user_conflict_lacking_supersession_reject"].reason_code == "authority_conflict_without_supersession"
    assert decisions["ambiguity_resolving_direct_correction"].eligible is True
    assert decisions["ambiguity_resolving_direct_correction"].reason_code == "direct_correction_resolves_ambiguity"


def test_hard_gate_oracle_ignores_gold_labels_and_leaves_soft_conflicts_eligible(tmp_path: Path) -> None:
    """Gold fields cannot affect admission and non-hard-gated rows reach scoring."""
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    stale = next(row for row in _rows(paths["candidates"]) if row["candidate_id"] == "history-001-stale")
    altered = {
        **stale,
        "gold_hard_gate": "none",
        "gold_decision": "accept",
        "gold_reason_code": "misleading",
        "gold_soft_label": "accept",
    }

    assert admit_candidate(events, stale) == admit_candidate(events, altered)
    assert admit_candidate(events, altered).eligible is True
    assert admit_candidate(events, altered).reason_code == "no_hard_gate"


def test_direct_user_authority_cannot_be_hidden_by_candidate_metadata(tmp_path: Path) -> None:
    """The fact's authority qualifier, not candidate-controlled family text, activates the gate."""
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    direct_conflict = next(
        row
        for row in _rows(paths["candidates"])
        if row["event_family"] == "direct_user_conflict_lacking_supersession_reject"
    )
    disguised = {**direct_conflict, "event_family": "neutral", "operation": "add"}

    decision = admit_candidate(events, disguised)

    assert decision.eligible is False
    assert decision.reason_code == "authority_conflict_without_supersession"


def test_tombstone_blocks_semantic_resurrection_with_a_new_fact_id(tmp_path: Path) -> None:
    """Changing an identifier cannot restore semantically identical retracted content."""
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    resurrection = next(
        row
        for row in _rows(paths["candidates"])
        if row["event_family"] == "retraction_tombstone_resurrection_reject"
    )
    renamed = {
        **resurrection,
        "fact": {**resurrection["fact"], "fact_id": "renamed-private-fact"},
    }

    decision = admit_candidate(events, renamed)

    assert decision.eligible is False
    assert decision.reason_code == "tombstone_resurrection"


def test_factless_retract_recovers_prior_content_for_tombstone(tmp_path: Path) -> None:
    """A compact retract-by-ID must retain the original fact's semantic tombstone."""
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    for event in events:
        if event["operation"] == "retract":
            event.pop("fact")
    resurrection = next(
        row
        for row in _rows(paths["candidates"])
        if row["event_family"] == "retraction_tombstone_resurrection_reject"
    )
    renamed = {
        **resurrection,
        "fact": {**resurrection["fact"], "fact_id": "renamed-private-fact"},
    }

    decision = admit_candidate(events, renamed)

    assert decision.eligible is False
    assert decision.reason_code == "tombstone_resurrection"


def test_unresolved_ambiguity_blocks_a_different_object_candidate(tmp_path: Path) -> None:
    """Changing the proposed value cannot bypass an active claim-level ambiguity."""
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    resolution = next(
        row
        for row in _rows(paths["candidates"])
        if row["event_family"] == "ambiguity_resolving_direct_correction"
    )
    unresolved = {
        **resolution,
        "candidate_id": "different-object-ambiguity-bypass",
        "supersedes": "",
        "fact": {
            **resolution["fact"],
            "fact_id": "different-object-ambiguity-bypass",
            "object": "different-object",
            "qualifiers": {
                **resolution["fact"]["qualifiers"],
                "source_authority": "inferred",
            },
        },
    }

    decision = admit_candidate(events, unresolved)

    assert decision.eligible is False
    assert decision.reason_code == "unresolved_ambiguity"
