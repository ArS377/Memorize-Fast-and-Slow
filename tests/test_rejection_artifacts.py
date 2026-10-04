#!/usr/bin/env python3
"""Tests for rejected-fact artifact records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurosym.reporting.rejections import (  # noqa: E402
    append_rejection_jsonl,
    build_rejection_record,
    infer_rule_fired,
)


def test_rejection_record_contains_auditable_decision_fields(tmp_path: Path) -> None:
    candidate = {
        "subject": "Exampleland",
        "predicate": "CAPITAL_IS",
        "object": "Wrong City",
        "fact_id": "fact_wrong",
        "example_id": "ex_1",
        "run_id": "run_test",
    }
    existing = {
        "subject": "Exampleland",
        "predicate": "CAPITAL_IS",
        "object": "Right City",
        "fact_id": "fact_right",
    }

    record = build_rejection_record(
        candidate_fact=candidate,
        reason="Contradiction: Exampleland has conflicting CAPITAL_IS",
        example_id="ex_1",
        session_id="pilot_scallop",
        stage="scallop_validation",
        existing_conflicting_fact=existing,
        validator="scallop",
    )

    assert record["rule_fired"] == "functional_conflict"
    assert record["candidate_fact"]["fact_id"] == "fact_wrong"
    assert record["existing_conflicting_fact"]["fact_id"] == "fact_right"
    assert record["example_id"] == "ex_1"
    assert record["session_id"] == "pilot_scallop"
    assert record["rule_version"]
    assert record["created_at"]

    out = tmp_path / "rejections.jsonl"
    append_rejection_jsonl(out, record)
    loaded = json.loads(out.read_text(encoding="utf-8").strip())
    assert loaded["rejection_id"] == record["rejection_id"]


def test_infer_rule_fired_handles_llm_verification_stage() -> None:
    assert (
        infer_rule_fired("not explicit in the cited support", stage="llm_verification")
        == "llm_verification_rejected"
    )
    assert (
        infer_rule_fired("uncertain support", stage="llm_verification")
        == "llm_verification_uncertain"
    )
