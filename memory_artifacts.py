"""Typed, provenance-preserving working-memory transitions.

The LLM may propose a working-memory update, but trusted application scope and
source facts are supplied by the orchestrator.  This module compiles the
proposal into a deterministic artifact without mutating Neo4j.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from compiled_memory import stable_id


Fact = Dict[str, Any]
MEMORY_SCOPE_MODES = {"example", "session", "session_set"}
CONFIDENCE_LEVELS = {"supported", "uncertain"}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _unique_strings(values: Iterable[Any]) -> Tuple[str, ...]:
    seen = set()
    output: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return tuple(output)


@dataclass(frozen=True)
class MemoryScope:
    """Trusted retrieval/mutation boundary; never accepted from model args."""

    mode: str = "example"
    session_ids: Tuple[str, ...] = field(default_factory=tuple)
    example_id: Optional[str] = None

    def __post_init__(self) -> None:
        mode = str(self.mode or "").strip().lower()
        sessions = _unique_strings(self.session_ids)
        example_id = str(self.example_id).strip() if self.example_id is not None else None
        if mode not in MEMORY_SCOPE_MODES:
            raise ValueError(f"memory scope must be one of {sorted(MEMORY_SCOPE_MODES)}")
        if not sessions:
            raise ValueError("memory scope requires at least one trusted session ID")
        if mode == "example" and not example_id:
            raise ValueError("example scope requires a trusted example ID")
        if mode != "session_set" and len(sessions) != 1:
            raise ValueError(f"{mode} scope requires exactly one session ID")
        if mode == "session_set" and len(sessions) < 2:
            raise ValueError("session_set scope requires at least two session IDs")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "session_ids", sessions)
        object.__setattr__(self, "example_id", example_id or None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "session_ids": list(self.session_ids),
            "example_id": self.example_id,
        }


@dataclass
class WorkingMemoryArtifact:
    artifact_id: str
    revision: int
    scope: MemoryScope
    entity: str
    selected_fact_ids: List[str]
    excluded_fact_ids: List[str]
    relevant_facts: List[Fact]
    excluded_claims: List[Fact]
    temporal_scope: Dict[str, Optional[str]]
    confidence: str
    citations: List[Dict[str, Any]]
    constraint_trace: List[Dict[str, Any]]
    derived_facts: List[Fact]
    provenance_complete: bool
    missing_provenance_fact_ids: List[str]
    status: str = "proposed"
    prior_artifact_id: Optional[str] = None
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["scope"] = self.scope.to_dict()
        return result


@dataclass
class MemoryTransitionRecord:
    transition_id: str
    artifact_id: str
    scope: MemoryScope
    decision: str
    reason: str
    validator: str
    rule_version: str
    before_artifact_id: Optional[str]
    after_artifact_id: Optional[str]
    committed: bool
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["scope"] = self.scope.to_dict()
        return result


def _fact_map(facts: Sequence[Mapping[str, Any]]) -> Dict[str, Fact]:
    output: Dict[str, Fact] = {}
    for raw in facts:
        fact = dict(raw)
        fact_id = str(fact.get("fact_id", "")).strip()
        if fact_id:
            output[fact_id] = fact
    return output


def _provenance_for(fact: Mapping[str, Any]) -> List[Dict[str, Any]]:
    provenance = fact.get("provenance")
    if not isinstance(provenance, list):
        return []
    return [dict(entry) for entry in provenance if isinstance(entry, dict)]


def _triple_key(fact: Mapping[str, Any]) -> Tuple[str, str, str]:
    return tuple(
        " ".join(str(fact.get(field, "")).split()).casefold()
        for field in ("subject", "predicate", "object")
    )


def _compile_derived_facts(
    raw_facts: Any,
    known: Mapping[str, Fact],
    selected: Sequence[str],
) -> List[Fact]:
    if raw_facts in (None, []):
        return []
    if not isinstance(raw_facts, list):
        raise ValueError("derived_facts must be a list")
    selected_set = set(selected)
    output: List[Fact] = []
    for index, raw in enumerate(raw_facts):
        if not isinstance(raw, dict):
            raise ValueError(f"derived_facts[{index}] must be an object")
        subject = str(raw.get("subject", "")).strip()
        predicate = str(raw.get("predicate", "")).strip()
        obj = str(raw.get("object", "")).strip()
        support_ids = list(_unique_strings(raw.get("support_fact_ids", [])))
        if not subject or not predicate or not obj:
            raise ValueError(f"derived_facts[{index}] requires subject, predicate, and object")
        if not support_ids or any(fid not in selected_set for fid in support_ids):
            raise ValueError(
                f"derived_facts[{index}] must cite only selected returned fact IDs"
            )
        provenance: List[Dict[str, Any]] = []
        for fact_id in support_ids:
            provenance.extend(_provenance_for(known[fact_id]))
        output.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "support_fact_ids": support_ids,
            "support_text": str(raw.get("support_text", "")).strip(),
            "temporal": dict(raw.get("temporal", {})) if isinstance(raw.get("temporal"), dict) else {},
            "confidence": str(raw.get("confidence", "uncertain")).strip().lower(),
            "provenance": provenance,
        })
    return output


def compile_working_memory(
    arguments: Mapping[str, Any],
    returned_facts: Sequence[Mapping[str, Any]],
    scope: MemoryScope,
    prior_artifact: Optional[WorkingMemoryArtifact] = None,
) -> WorkingMemoryArtifact:
    """Validate a model proposal against returned facts and trusted scope."""

    if not isinstance(arguments, Mapping):
        raise ValueError("working-memory arguments must be an object")
    forbidden = {"scope", "memory_scope", "session_id", "session_ids", "example_id"}
    supplied_forbidden = sorted(forbidden.intersection(arguments))
    if supplied_forbidden:
        raise ValueError(f"model may not override trusted scope: {', '.join(supplied_forbidden)}")

    known = _fact_map(returned_facts)
    selected = list(_unique_strings(arguments.get("selected_fact_ids", [])))
    excluded = list(_unique_strings(arguments.get("excluded_fact_ids", [])))
    if not selected:
        raise ValueError("selected_fact_ids must contain at least one returned fact ID")
    unknown = sorted((set(selected) | set(excluded)) - set(known))
    if unknown:
        raise ValueError(f"unknown or out-of-scope fact IDs: {', '.join(unknown)}")
    overlap = sorted(set(selected).intersection(excluded))
    if overlap:
        raise ValueError(f"facts cannot be both selected and excluded: {', '.join(overlap)}")

    entity = str(arguments.get("entity", "")).strip()
    if not entity:
        raise ValueError("entity is required")
    confidence = str(arguments.get("confidence", "uncertain")).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"confidence must be one of {sorted(CONFIDENCE_LEVELS)}")
    temporal = arguments.get("temporal_scope", {})
    if temporal is None:
        temporal = {}
    if not isinstance(temporal, dict):
        raise ValueError("temporal_scope must be an object")
    temporal_scope = {
        "valid_from": temporal.get("valid_from"),
        "valid_to": temporal.get("valid_to"),
    }

    relevant = [known[fact_id] for fact_id in selected]
    excluded_claims = [known[fact_id] for fact_id in excluded]
    citations: List[Dict[str, Any]] = []
    missing: List[str] = []
    for fact_id in selected:
        entries = _provenance_for(known[fact_id])
        if not entries:
            missing.append(fact_id)
        for entry in entries:
            citations.append({"fact_id": fact_id, **entry})
    proposed_derived = _compile_derived_facts(
        arguments.get("derived_facts", []), known, selected
    )
    selected_triples = {_triple_key(known[fact_id]) for fact_id in selected}
    redundant_derived = [
        fact for fact in proposed_derived if _triple_key(fact) in selected_triples
    ]
    derived = [
        fact for fact in proposed_derived if _triple_key(fact) not in selected_triples
    ]
    revision = (prior_artifact.revision + 1) if prior_artifact else 1
    prior_id = prior_artifact.artifact_id if prior_artifact else None
    artifact_id = stable_id(
        "working_memory",
        [
            scope.to_dict(),
            entity,
            selected,
            excluded,
            temporal_scope,
            confidence,
            derived,
            revision,
            prior_id,
        ],
        length=24,
    )
    trace = [
        {"check": "trusted_scope", "status": "passed"},
        {"check": "returned_fact_ids", "status": "passed"},
        {
            "check": "provenance_complete",
            "status": "passed" if not missing else "missing",
            "fact_ids": missing,
        },
    ]
    if proposed_derived:
        trace.append(
            {
                "check": "derived_fact_novelty",
                "status": "omitted_redundant" if redundant_derived else "passed",
                "omitted_count": len(redundant_derived),
                "omitted_triples": [
                    {
                        "subject": fact["subject"],
                        "predicate": fact["predicate"],
                        "object": fact["object"],
                    }
                    for fact in redundant_derived
                ],
            }
        )
    return WorkingMemoryArtifact(
        artifact_id=artifact_id,
        revision=revision,
        scope=scope,
        entity=entity,
        selected_fact_ids=selected,
        excluded_fact_ids=excluded,
        relevant_facts=relevant,
        excluded_claims=excluded_claims,
        temporal_scope=temporal_scope,
        confidence=confidence,
        citations=citations,
        constraint_trace=trace,
        derived_facts=derived,
        provenance_complete=not missing,
        missing_provenance_fact_ids=missing,
        prior_artifact_id=prior_id,
    )


def transition_for(
    artifact: WorkingMemoryArtifact,
    *,
    decision: str,
    reason: str,
    validator: str,
    rule_version: str,
) -> MemoryTransitionRecord:
    committed = decision in {"accept", "replace"}
    return MemoryTransitionRecord(
        transition_id=stable_id(
            "transition",
            [artifact.artifact_id, decision, validator, rule_version],
            length=24,
        ),
        artifact_id=artifact.artifact_id,
        scope=artifact.scope,
        decision=decision,
        reason=reason,
        validator=validator,
        rule_version=rule_version,
        before_artifact_id=artifact.prior_artifact_id,
        after_artifact_id=artifact.artifact_id if committed else artifact.prior_artifact_id,
        committed=committed,
    )
