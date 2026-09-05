from __future__ import annotations

from typing import Any

import pytest

from experiments.private_lineage_reasoning import resolve_private_lineage_reference
from experiments.private_lineage_reasoning import resolve_private_lineage_with_scallop


def _fact(fact_id: str, value: str = "cedar glass") -> dict[str, Any]:
    """Build one minimal private-memory fact."""
    return {
        "fact_id": fact_id,
        "subject": "account-1",
        "object": value,
        "qualifiers": {"scope": "private"},
    }


def _lineage_events() -> list[dict[str, Any]]:
    """Build a two-hop duplicate lineage ending in a grandchild retraction."""
    root = _fact("root")
    copy_one = _fact("copy-1")
    copy_two = _fact("copy-2")
    return [
        {"history_id": "history-a", "operation": "add", "fact": root},
        {
            "history_id": "history-a",
            "operation": "duplicate_delivery",
            "duplicate_of": "root",
            "fact": copy_one,
        },
        {
            "history_id": "history-a",
            "operation": "duplicate_delivery",
            "duplicate_of": "copy-1",
            "fact": copy_two,
        },
        {
            "history_id": "history-a",
            "operation": "retract",
            "retracts": "copy-2",
            "fact": copy_two,
        },
    ]


def test_transitive_reference_erases_entire_duplicate_lineage() -> None:
    query = {
        "query_id": "lineage-query",
        "history_id": "history-a",
        "kind": "private_lineage",
        "subject": "account-1",
        "fact_id": "root",
    }

    positive = resolve_private_lineage_reference(_lineage_events()[:-1], query)
    retracted = resolve_private_lineage_reference(_lineage_events(), query)

    assert positive["answer"] == "cedar glass"
    assert retracted["answer"] == "UNKNOWN"
    assert retracted["retracted_fact_ids"] == ["copy-1", "copy-2", "root"]


def test_lineage_validation_rejects_semantically_mismatched_duplicate() -> None:
    events = _lineage_events()
    events[1]["fact"] = _fact("copy-1", value="wrong secret")
    query = {
        "query_id": "lineage-query",
        "history_id": "history-a",
        "kind": "private_lineage",
        "subject": "account-1",
        "fact_id": "root",
    }

    with pytest.raises(ValueError, match="duplicate lineage mismatch"):
        resolve_private_lineage_reference(events, query)


def test_recursive_scallop_isolates_two_hop_feature_effect() -> None:
    pytest.importorskip("scallopy")
    query = {
        "query_id": "lineage-query",
        "history_id": "history-a",
        "kind": "private_lineage",
        "subject": "account-1",
        "fact_id": "root",
    }

    recursive_positive = resolve_private_lineage_with_scallop(
        _lineage_events()[:-1], query, mode="recursive"
    )
    one_hop_positive = resolve_private_lineage_with_scallop(
        _lineage_events()[:-1], query, mode="one_hop"
    )
    recursive_retracted = resolve_private_lineage_with_scallop(
        _lineage_events(), query, mode="recursive"
    )
    one_hop_retracted = resolve_private_lineage_with_scallop(
        _lineage_events(), query, mode="one_hop"
    )

    assert recursive_positive["answer"] == one_hop_positive["answer"] == "cedar glass"
    assert recursive_retracted["answer"] == "UNKNOWN"
    assert one_hop_retracted["answer"] == "cedar glass"
    assert recursive_retracted["scallopy_version"] == "0.2.4"
