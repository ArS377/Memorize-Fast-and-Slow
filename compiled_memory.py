"""Compiled memory entity/relationship format.

This module is a compatibility layer between the current extracted-fact
pipeline and the richer "compiled memory" model:

    fact dict -> CompiledRelationship -> existing fact dict / Neo4j params /
    Scallop relation facts

The current pipeline can keep storing simple triples. New code can attach
identity, temporal scope, provenance, confidence, and validation-decision
metadata without changing every caller at once.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


Fact = Dict[str, Any]

RELATIONSHIP_KIND = "relationship"
DEFAULT_ENTITY_TYPE = "entity"
DEFAULT_CONFIDENCE_LEVEL = "uncertain"


FUNCTIONAL_PREDICATES = {
    "CAPITAL_IS", "BORN_IN", "BIRTH_DATE", "DEATH_DATE",
    "DIED_IN", "FOUNDED_IN", "LOCATED_IN",
    "HAS_ISO_CODE", "HAS_GLOTTOCODE",
}


def _stable_hash(parts: List[Any], length: int = 16) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stable_id(prefix: str, parts: List[Any], length: int = 16) -> str:
    return f"{prefix}_{_stable_hash(parts, length=length)}"


def sanitize_predicate(predicate: str) -> str:
    pred = (predicate or "").strip().upper()
    pred = re.sub(r"[^A-Z0-9_]+", "_", pred)
    pred = re.sub(r"_+", "_", pred).strip("_")
    return pred or "RELATED_TO"


def normalize_confidence_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "not" in text or "reject" in text or text == "false":
        return "rejected"
    if "support" in text:
        return "supported"
    if "uncertain" in text or "partial" in text or "maybe" in text:
        return "uncertain"
    return DEFAULT_CONFIDENCE_LEVEL


def _canonical_name(value: Any) -> str:
    return str(value or "").strip()


def _unique_aliases(name: str, aliases: Any) -> List[str]:
    seen = {name.lower()} if name else set()
    out: List[str] = []
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list):
        return out
    for alias in aliases:
        text = str(alias or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class CompiledEntity:
    name: str
    entity_type: str = DEFAULT_ENTITY_TYPE
    entity_id: Optional[str] = None
    aliases: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = _canonical_name(self.name)
        self.entity_type = _canonical_name(self.entity_type).lower() or DEFAULT_ENTITY_TYPE
        self.aliases = _unique_aliases(self.name, self.aliases)
        if not self.entity_id:
            self.entity_id = stable_id("entity", [self.entity_type, self.name.lower()])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entity_id,
            "name": self.name,
            "type": self.entity_type,
            "aliases": list(self.aliases),
        }


@dataclass
class TemporalScope:
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    observed_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.observed_at:
            self.observed_at = _utc_now_iso()

    def overlaps(self, other: "TemporalScope") -> bool:
        """Return True when two validity intervals overlap.

        Missing bounds are treated as open-ended. Dates are compared
        lexicographically, which is safe for ISO-8601 date/datetime strings.
        """
        left_start = self.valid_from or ""
        right_start = other.valid_from or ""
        left_end = self.valid_to or "9999-12-31T23:59:59Z"
        right_end = other.valid_to or "9999-12-31T23:59:59Z"
        return left_start <= right_end and right_start <= left_end

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "observed_at": self.observed_at,
        }


@dataclass
class ProvenanceRecord:
    source_id: str
    title: str = ""
    sent_id: Optional[Any] = None
    support_text: str = ""

    @classmethod
    def from_fact_entry(cls, entry: Any, support_text: str = "") -> "ProvenanceRecord":
        if isinstance(entry, dict):
            title = str(entry.get("title", ""))
            sent_id = entry.get("sent_id")
            source_id = str(entry.get("source_id") or stable_id("source", [title, sent_id]))
            return cls(
                source_id=source_id,
                title=title,
                sent_id=sent_id,
                support_text=str(entry.get("support_text") or support_text or ""),
            )
        text = str(entry or "")
        return cls(source_id=stable_id("source", [text]), title=text, support_text=support_text)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceRecord:
    level: str = DEFAULT_CONFIDENCE_LEVEL
    score: Optional[float] = None
    method: str = "llm_self_reflection"

    def __post_init__(self) -> None:
        self.level = normalize_confidence_level(self.level)
        if self.score is None:
            self.score = {"supported": 0.9, "uncertain": 0.5, "rejected": 0.1}.get(
                self.level, 0.5
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintMetadata:
    functional: bool = False
    scope: str = "atemporal"
    replace_policy: str = "higher_confidence_wins"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationDecision:
    status: str = "proposed"
    validator: str = "scallop"
    reason: Optional[str] = None
    replaces: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompiledRelationship:
    subject: CompiledEntity
    predicate: str
    object: CompiledEntity
    memory_id: Optional[str] = None
    kind: str = RELATIONSHIP_KIND
    temporal: TemporalScope = field(default_factory=TemporalScope)
    provenance: List[ProvenanceRecord] = field(default_factory=list)
    confidence: ConfidenceRecord = field(default_factory=ConfidenceRecord)
    constraints: ConstraintMetadata = field(default_factory=ConstraintMetadata)
    decision: ValidationDecision = field(default_factory=ValidationDecision)
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    example_id: str = ""
    question: str = ""
    question_relevance: str = ""
    normalization_notes: str = ""
    verification_reason: str = ""

    def __post_init__(self) -> None:
        self.predicate = sanitize_predicate(self.predicate)
        if self.predicate in FUNCTIONAL_PREDICATES:
            self.constraints.functional = True
        if self.temporal.valid_from or self.temporal.valid_to:
            self.constraints.scope = "temporal"
        if not self.memory_id:
            self.memory_id = stable_id(
                "mem",
                [
                    self.subject.entity_id,
                    self.predicate,
                    self.object.entity_id,
                    self.temporal.valid_from,
                    self.temporal.valid_to,
                    self.example_id,
                ],
            )

    @property
    def support_text(self) -> str:
        for record in self.provenance:
            if record.support_text:
                return record.support_text
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "kind": self.kind,
            "subject": self.subject.to_dict(),
            "predicate": self.predicate,
            "object": self.object.to_dict(),
            "temporal": self.temporal.to_dict(),
            "provenance": [p.to_dict() for p in self.provenance],
            "confidence": self.confidence.to_dict(),
            "constraints": self.constraints.to_dict(),
            "decision": self.decision.to_dict(),
            "qualifiers": dict(self.qualifiers),
            "example_id": self.example_id,
            "question": self.question,
            "question_relevance": self.question_relevance,
            "normalization_notes": self.normalization_notes,
            "verification_reason": self.verification_reason,
        }

    def to_fact(self) -> Fact:
        """Return the existing simple fact shape with compiled memory attached."""
        fact = {
            "subject": self.subject.name,
            "predicate": self.predicate,
            "object": self.object.name,
            "qualifiers": dict(self.qualifiers),
            "provenance": [
                {
                    "source_id": p.source_id,
                    "title": p.title,
                    "sent_id": p.sent_id,
                    "support_text": p.support_text,
                }
                for p in self.provenance
            ],
            "support_text": self.support_text,
            "question_relevance": self.question_relevance,
            "confidence": self.confidence.level,
            "normalization_notes": self.normalization_notes,
            "example_id": self.example_id,
            "fact_id": self.memory_id,
            "question": self.question,
            "verification_reason": self.verification_reason,
            "compiled_memory": self.to_dict(),
        }
        return fact


def fact_to_compiled_memory(fact: Fact) -> CompiledRelationship:
    """Compile the current fact dict into the richer memory format."""
    compiled = fact.get("compiled_memory")
    if isinstance(compiled, dict):
        fact = dict(fact)
        subject_raw = compiled.get("subject", {})
        object_raw = compiled.get("object", {})
        confidence_raw = compiled.get("confidence", {})
        if isinstance(subject_raw, dict):
            fact.setdefault("subject", subject_raw.get("name", ""))
            fact.setdefault("subject_type", subject_raw.get("type", DEFAULT_ENTITY_TYPE))
            fact.setdefault("subject_id", subject_raw.get("id"))
            fact.setdefault("subject_aliases", subject_raw.get("aliases", []))
        if isinstance(object_raw, dict):
            fact.setdefault("object", object_raw.get("name", ""))
            fact.setdefault("object_type", object_raw.get("type", DEFAULT_ENTITY_TYPE))
            fact.setdefault("object_id", object_raw.get("id"))
            fact.setdefault("object_aliases", object_raw.get("aliases", []))
        fact.setdefault("predicate", compiled.get("predicate", ""))
        fact.setdefault("memory_id", compiled.get("memory_id"))
        fact.setdefault("temporal", compiled.get("temporal", {}))
        fact.setdefault("provenance", compiled.get("provenance", []))
        fact.setdefault("constraints", compiled.get("constraints", {}))
        fact.setdefault("decision", compiled.get("decision", {}))
        fact.setdefault("qualifiers", compiled.get("qualifiers", {}))
        fact.setdefault("example_id", compiled.get("example_id", ""))
        fact.setdefault("question", compiled.get("question", ""))
        fact.setdefault("question_relevance", compiled.get("question_relevance", ""))
        fact.setdefault("normalization_notes", compiled.get("normalization_notes", ""))
        fact.setdefault("verification_reason", compiled.get("verification_reason", ""))
        if isinstance(confidence_raw, dict):
            fact.setdefault("confidence", confidence_raw.get("level", DEFAULT_CONFIDENCE_LEVEL))
            fact.setdefault("confidence_score", confidence_raw.get("score"))
            fact.setdefault("confidence_method", confidence_raw.get("method", "llm_self_reflection"))

    subject = CompiledEntity(
        name=fact.get("subject", ""),
        entity_type=fact.get("subject_type", DEFAULT_ENTITY_TYPE),
        entity_id=fact.get("subject_id"),
        aliases=fact.get("subject_aliases", []),
    )
    obj = CompiledEntity(
        name=fact.get("object", ""),
        entity_type=fact.get("object_type", DEFAULT_ENTITY_TYPE),
        entity_id=fact.get("object_id"),
        aliases=fact.get("object_aliases", []),
    )
    temporal_raw = fact.get("temporal", {})
    if not isinstance(temporal_raw, dict):
        temporal_raw = {}
    temporal = TemporalScope(
        valid_from=temporal_raw.get("valid_from") or fact.get("valid_from"),
        valid_to=temporal_raw.get("valid_to") or fact.get("valid_to"),
        observed_at=temporal_raw.get("observed_at") or fact.get("observed_at"),
    )
    confidence_raw = fact.get("confidence", fact.get("status", DEFAULT_CONFIDENCE_LEVEL))
    confidence = ConfidenceRecord(
        level=confidence_raw,
        score=fact.get("confidence_score"),
        method=fact.get("confidence_method", "llm_self_reflection"),
    )
    provenance_raw = fact.get("provenance", [])
    if not isinstance(provenance_raw, list):
        provenance_raw = []
    support_text = str(fact.get("support_text", ""))
    provenance = [
        ProvenanceRecord.from_fact_entry(entry, support_text=support_text)
        for entry in provenance_raw
    ]
    if support_text and not provenance:
        provenance = [
            ProvenanceRecord(
                source_id=stable_id("source", [fact.get("example_id", ""), support_text]),
                title=str(fact.get("example_id", "")),
                sent_id=None,
                support_text=support_text,
            )
        ]

    predicate = sanitize_predicate(str(fact.get("predicate", "")))
    constraints_raw = fact.get("constraints", {})
    if not isinstance(constraints_raw, dict):
        constraints_raw = {}
    constraints = ConstraintMetadata(
        functional=bool(
            constraints_raw.get("functional", predicate in FUNCTIONAL_PREDICATES)
        ),
        scope=str(
            constraints_raw.get(
                "scope",
                "temporal" if temporal.valid_from or temporal.valid_to else "atemporal",
            )
        ),
        replace_policy=str(
            constraints_raw.get("replace_policy", "higher_confidence_wins")
        ),
    )

    decision_raw = fact.get("decision", {})
    if not isinstance(decision_raw, dict):
        decision_raw = {}
    decision = ValidationDecision(
        status=str(decision_raw.get("status", "proposed")),
        validator=str(decision_raw.get("validator", "scallop")),
        reason=decision_raw.get("reason"),
        replaces=decision_raw.get("replaces"),
    )

    qualifiers = fact.get("qualifiers", {})
    if not isinstance(qualifiers, dict):
        qualifiers = {}

    return CompiledRelationship(
        subject=subject,
        predicate=predicate,
        object=obj,
        memory_id=fact.get("memory_id") or fact.get("fact_id"),
        temporal=temporal,
        provenance=provenance,
        confidence=confidence,
        constraints=constraints,
        decision=decision,
        qualifiers=qualifiers,
        example_id=str(fact.get("example_id", "")),
        question=str(fact.get("question", "")),
        question_relevance=str(fact.get("question_relevance", "")),
        normalization_notes=str(fact.get("normalization_notes", "")),
        verification_reason=str(fact.get("verification_reason", "")),
    )


def compiled_memory_to_fact(memory: CompiledRelationship) -> Fact:
    return memory.to_fact()


def fact_to_compiled_fact(fact: Fact) -> Fact:
    return compiled_memory_to_fact(fact_to_compiled_memory(fact))


def compiled_memory_to_scallop_facts(memory: CompiledRelationship) -> Dict[str, List[tuple]]:
    """Project compiled memory into relation facts Scallop can consume.

    This does not run Scallop. It returns relation names and tuples so the
    validator can add only the relations it needs.
    """
    return {
        "entity": [
            (memory.subject.entity_id, memory.subject.name, memory.subject.entity_type),
            (memory.object.entity_id, memory.object.name, memory.object.entity_type),
        ],
        "alias": [
            (memory.subject.entity_id, alias) for alias in memory.subject.aliases
        ] + [
            (memory.object.entity_id, alias) for alias in memory.object.aliases
        ],
        "memory_relationship": [
            (
                memory.memory_id,
                memory.subject.entity_id,
                memory.predicate,
                memory.object.entity_id,
            )
        ],
        "validity": [
            (
                memory.memory_id,
                memory.temporal.valid_from or "",
                memory.temporal.valid_to or "",
            )
        ],
        "confidence": [
            (
                memory.memory_id,
                memory.confidence.level,
                float(memory.confidence.score or 0.0),
            )
        ],
        "decision": [
            (
                memory.memory_id,
                memory.decision.status,
                memory.decision.validator,
                memory.decision.replaces or "",
            )
        ],
        "constraint": [
            (
                memory.memory_id,
                str(bool(memory.constraints.functional)).lower(),
                memory.constraints.scope,
                memory.constraints.replace_policy,
            )
        ],
        "provenance": [
            (
                memory.memory_id,
                p.source_id,
                p.title,
                str(p.sent_id if p.sent_id is not None else ""),
            )
            for p in memory.provenance
        ],
    }


def compiled_memory_to_neo4j_properties(memory: CompiledRelationship) -> Dict[str, Any]:
    """Return relationship properties for a future richer Neo4j write path."""
    return {
        "memory_id": memory.memory_id,
        "fact_id": memory.memory_id,
        "kind": memory.kind,
        "subject_id": memory.subject.entity_id,
        "subject_type": memory.subject.entity_type,
        "subject_aliases_json": json.dumps(memory.subject.aliases, ensure_ascii=False),
        "object_id": memory.object.entity_id,
        "object_type": memory.object.entity_type,
        "object_aliases_json": json.dumps(memory.object.aliases, ensure_ascii=False),
        "valid_from": memory.temporal.valid_from or "",
        "valid_to": memory.temporal.valid_to or "",
        "observed_at": memory.temporal.observed_at or "",
        "confidence_level": memory.confidence.level,
        "confidence_score": float(memory.confidence.score or 0.0),
        "confidence_method": memory.confidence.method,
        "decision_status": memory.decision.status,
        "decision_validator": memory.decision.validator,
        "decision_reason": memory.decision.reason or "",
        "replaces_memory_id": memory.decision.replaces or "",
        "constraints_json": json.dumps(memory.constraints.to_dict(), ensure_ascii=False),
        "compiled_memory_json": json.dumps(memory.to_dict(), ensure_ascii=False),
    }
