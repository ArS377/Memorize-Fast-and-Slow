from __future__ import annotations

import pytest

from experiments.contradiction_ledger import (
    derive_contradiction_ledger,
    derive_contradiction_ledger_with_scallop,
)


def _event(
    event_id: str,
    fact_id: str,
    value: str,
    *,
    authority: str = "inferred",
    resolves: tuple[str, ...] = (),
    valid_from: str = "2027-01-01",
    valid_to: str | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "history_id": "history-001",
        "operation": "direct_user_correction" if resolves else "add",
        "resolves": list(resolves),
        "fact": {
            "fact_id": fact_id,
            "subject": "subject-001",
            "predicate": "PREFERS",
            "object": value,
            "qualifiers": {
                "domain": "personal_preference",
                "scope": "workspace",
                "source_authority": authority,
            },
            "temporal": {"valid_from": valid_from, "valid_to": valid_to},
        },
    }


def test_ledger_records_opening_and_justified_resolution() -> None:
    events = [
        _event("event-a", "fact-a", "quiet room"),
        _event("event-b", "fact-b", "open lounge"),
        _event(
            "event-c",
            "fact-c",
            "window desk",
            authority="direct_user",
            resolves=("fact-a", "fact-b"),
        ),
    ]

    ledger = derive_contradiction_ledger(events)

    assert len(ledger["pairs"]) == 1
    pair = next(
        pair
        for pair in ledger["pairs"]
        if set(pair["fact_ids"]) == {"fact-a", "fact-b"}
    )
    assert pair["status"] == "resolved_authority"
    assert pair["winner_fact_id"] == "fact-c"
    assert pair["opened_event_index"] == 1
    assert pair["resolved_event_index"] == 2
    assert ledger["unexpected_unresolved_pair_ids"] == []


def test_higher_authority_without_explicit_resolution_stays_unresolved() -> None:
    events = [
        _event("event-a", "fact-a", "quiet room"),
        _event("event-b", "fact-b", "open lounge", authority="direct_user"),
    ]

    ledger = derive_contradiction_ledger(events)

    assert len(ledger["unresolved_pair_ids"]) == 1
    assert ledger["pairs"][0]["status"] == "unresolved"


def test_different_scopes_do_not_contradict() -> None:
    events = [
        _event("event-a", "fact-a", "quiet room"),
        _event("event-b", "fact-b", "open lounge"),
    ]
    events[1]["fact"]["qualifiers"]["scope"] = "travel"

    ledger = derive_contradiction_ledger(events)

    assert ledger["pairs"] == []
    assert ledger["scope_separated_count"] == 1


def test_non_overlapping_values_do_not_contradict() -> None:
    events = [
        _event("event-a", "fact-a", "quiet room", valid_to="2027-01-31"),
        _event("event-b", "fact-b", "open lounge", valid_from="2027-02-01"),
    ]

    reference = derive_contradiction_ledger(events)

    assert reference["pairs"] == []


def test_lower_authority_member_cannot_resolve_higher_authority_claim() -> None:
    events = [
        _event("event-a", "fact-a", "quiet room", authority="direct_user"),
        _event("event-b", "fact-b", "open lounge", resolves=("fact-a",)),
    ]

    with pytest.raises(ValueError, match="lower-authority"):
        derive_contradiction_ledger(events)


def test_higher_authority_member_resolution_matches_reference_status() -> None:
    events = [
        _event("event-a", "fact-a", "quiet room"),
        _event(
            "event-b",
            "fact-b",
            "open lounge",
            authority="direct_user",
            resolves=("fact-a",),
        ),
    ]

    reference = derive_contradiction_ledger(events)

    assert reference["pairs"][0]["status"] == "resolved_authority"


def test_actual_scallop_matches_reference_ledger() -> None:
    pytest.importorskip("scallopy")
    events = [
        _event("event-a", "fact-a", "quiet room"),
        _event("event-b", "fact-b", "open lounge"),
        _event(
            "event-c",
            "fact-c",
            "window desk",
            authority="direct_user",
            resolves=("fact-a", "fact-b"),
        ),
    ]

    actual = derive_contradiction_ledger_with_scallop(events)

    assert actual["engine"] == "scallopy"
    assert actual["ledger_version"] == "contradiction_ledger.v1"
    assert actual["unexpected_unresolved_pair_ids"] == []
    assert len(actual["pairs"]) == 1
    assert actual["pairs"][0]["status"] == "resolved_authority"


def test_actual_scallop_matches_non_overlap_and_authority_guards() -> None:
    pytest.importorskip("scallopy")
    non_overlapping = [
        _event("event-a", "fact-a", "quiet room", valid_to="2027-01-31"),
        _event("event-b", "fact-b", "open lounge", valid_from="2027-02-01"),
    ]
    lower_authority = [
        _event("event-a", "fact-a", "quiet room", authority="direct_user"),
        _event("event-b", "fact-b", "open lounge", resolves=("fact-a",)),
    ]

    assert derive_contradiction_ledger_with_scallop(non_overlapping)["pairs"] == []
    with pytest.raises(ValueError, match="lower-authority"):
        derive_contradiction_ledger_with_scallop(lower_authority)


def test_actual_scallop_matches_member_winner_authority_status() -> None:
    pytest.importorskip("scallopy")
    events = [
        _event("event-a", "fact-a", "quiet room"),
        _event(
            "event-b",
            "fact-b",
            "open lounge",
            authority="direct_user",
            resolves=("fact-a",),
        ),
    ]

    reference = derive_contradiction_ledger(events)
    actual = derive_contradiction_ledger_with_scallop(events)

    assert actual["pairs"][0]["status"] == reference["pairs"][0]["status"]
