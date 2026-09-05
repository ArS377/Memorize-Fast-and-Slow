"""Rejected-fact artifact helpers.

Rejected facts are decision records, not KG edges. Keeping them as JSONL gives
the build pipeline an append-only audit trail while preserving the accepted KG
schema and retrieval behavior.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


Fact = Dict[str, Any]

RULE_VERSION = "neurosym_validator_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def infer_rule_fired(reason: Any, *, stage: str = "") -> str:
    text = str(reason or "").strip().lower()
    stage_text = str(stage or "").strip().lower()
    if stage_text == "llm_verification":
        if "uncertain" in text or "partial" in text or "maybe" in text:
            return "llm_verification_uncertain"
        return "llm_verification_rejected"
    if "redundancy" in text or "already exists" in text:
        return "redundant_fact"
    if "self-referential" in text:
        return "self_referential_fact"
    if "too generic" in text:
        return "generic_object"
    if "subject, predicate, or object is empty" in text:
        return "empty_field"
    if "circular containment" in text:
        return "circular_containment"
    if "both alive and dead" in text:
        return "alive_dead_conflict"
    if "contradiction" in text or "conflicting" in text:
        return "functional_conflict"
    return "validator_reject"


def rejection_id(
    *,
    candidate_fact: Fact,
    reason: Any,
    rule_fired: str,
    session_id: str,
    stage: str,
) -> str:
    payload = {
        "candidate_fact_id": candidate_fact.get("fact_id"),
        "candidate": {
            "subject": candidate_fact.get("subject"),
            "predicate": candidate_fact.get("predicate"),
            "object": candidate_fact.get("object"),
        },
        "reason": str(reason or ""),
        "rule_fired": rule_fired,
        "session_id": session_id,
        "stage": stage,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_rejection_record(
    *,
    candidate_fact: Fact,
    reason: Any,
    example_id: str,
    session_id: str,
    stage: str,
    existing_conflicting_fact: Optional[Fact] = None,
    rule_fired: Optional[str] = None,
    validator: str = "scallop",
    rule_version: str = RULE_VERSION,
) -> Dict[str, Any]:
    """Return a total JSON-serializable rejected-fact artifact record."""
    rule = rule_fired or infer_rule_fired(reason, stage=stage)
    return {
        "rejection_id": rejection_id(
            candidate_fact=candidate_fact,
            reason=reason,
            rule_fired=rule,
            session_id=session_id,
            stage=stage,
        ),
        "stage": stage,
        "validator": validator,
        "rule_version": rule_version,
        "rule_fired": rule,
        "rejection_reason": str(reason or ""),
        "example_id": str(example_id or candidate_fact.get("example_id", "")),
        "session_id": str(session_id or ""),
        "run_id": str(candidate_fact.get("run_id", "")),
        "candidate_fact": dict(candidate_fact),
        "existing_conflicting_fact": (
            dict(existing_conflicting_fact) if existing_conflicting_fact else None
        ),
        "created_at": utc_now_iso(),
    }


def append_rejection_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
        fp.write("\n")
