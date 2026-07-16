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


def _as_float(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _get_nested(mapping: Dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _support_text_for_score(fact: Fact) -> str:
    support = str(fact.get("support_text", "")).strip()
    if support:
        return support
    compiled = fact.get("compiled_memory")
    if isinstance(compiled, dict):
        for entry in _as_list(compiled.get("provenance")):
            if isinstance(entry, dict) and entry.get("support_text"):
                return str(entry.get("support_text", "")).strip()
    return ""


def _provenance_entries(fact: Fact) -> List[Dict[str, Any]]:
    provenance = _as_list(fact.get("provenance"))
    if not provenance:
        compiled = fact.get("compiled_memory")
        if isinstance(compiled, dict):
            provenance = _as_list(compiled.get("provenance"))
    return [p for p in provenance if isinstance(p, dict)]


def _span_bounds(entry: Dict[str, Any]) -> tuple[Optional[Any], Optional[Any]]:
    start = entry.get("source_span_start")
    end = entry.get("source_span_end")
    span = entry.get("source_span")
    if isinstance(span, dict):
        start = span.get("start_char", span.get("start", start))
        end = span.get("end_char", span.get("end", end))
    return start, end


def provenance_quality_score(fact: Fact) -> float:
    """Return a 0..1 auditability score for a fact's evidence trail."""
    entries = _provenance_entries(fact)
    support = _support_text_for_score(fact)
    if not entries:
        stored_quality = _as_float(fact.get("provenance_quality"))
        if stored_quality is not None:
            return stored_quality
        return 0.12 if support else 0.0

    per_entry: List[float] = []
    for entry in entries:
        score = 0.0
        if entry.get("source_id"):
            score += 0.10
        if entry.get("document_id") or entry.get("doc_id") or entry.get("title"):
            score += 0.15
        if entry.get("sentence_id") or entry.get("sent_id") is not None:
            score += 0.20
        start, end = _span_bounds(entry)
        if start is not None and end is not None:
            score += 0.25
        if entry.get("support_text") or support:
            score += 0.15
        if entry.get("extractor_model") or fact.get("extractor_model"):
            score += 0.05
        if entry.get("verifier_model") or fact.get("verifier_model"):
            score += 0.05
        if entry.get("run_id") or fact.get("run_id"):
            score += 0.05
        per_entry.append(min(1.0, score))

    multi_source_bonus = min(0.10, max(0, len(entries) - 1) * 0.03)
    return min(1.0, (sum(per_entry) / len(per_entry)) + multi_source_bonus)


def evidence_strength_score(fact: Fact) -> float:
    """Weighted confidence score used for conflict resolution and ranking.

    The score combines verifier/extractor confidence with evidence quality and
    utility signals. It keeps the existing supported/uncertain/rejected labels
    as a prior instead of letting them collapse all supported facts to one
    value.
    """
    compiled_confidence = _get_nested(fact, "compiled_memory", "confidence")
    raw_score = fact.get("confidence_score")
    if raw_score is None and isinstance(compiled_confidence, dict):
        raw_score = compiled_confidence.get("score")

    explicit = _as_float(raw_score)
    confidence_method = fact.get("confidence_method")
    if confidence_method is None and isinstance(compiled_confidence, dict):
        confidence_method = compiled_confidence.get("method")
    if explicit is not None and confidence_method == "weighted_evidence_v1":
        return round(explicit, 4)

    raw_level = fact.get("confidence", fact.get("status", fact.get("confidence_level")))
    if raw_level is None and isinstance(compiled_confidence, dict):
        raw_level = compiled_confidence.get("level")
    level = normalize_confidence_level(raw_level)
    level_prior = {"supported": 0.78, "uncertain": 0.42, "rejected": 0.06}.get(level, 0.42)
    base = explicit if explicit is not None else level_prior

    provenance = provenance_quality_score(fact)
    support = _support_text_for_score(fact)
    support_bonus = 0.04 if support else 0.0
    if len(support) >= 80:
        support_bonus += 0.03
    if str(fact.get("verification_reason", "")).strip():
        support_bonus += 0.03
    if str(fact.get("question_relevance", "")).strip():
        support_bonus += 0.04

    score = (0.72 * base) + (0.20 * provenance) + support_bonus
    if fact.get("verifier_model") or fact.get("extractor_model"):
        score += 0.02
    if fact.get("run_id"):
        score += 0.01
    if level == "rejected":
        score = min(score, 0.20)
    return round(max(0.0, min(1.0, score)), 4)


def context_rank_score(row: Fact) -> float:
    """Score a retrieved fact row for context ordering."""
    confidence = evidence_strength_score(row)
    provenance = provenance_quality_score(row)
    utility = 0.0
    if str(row.get("question_relevance", "")).strip():
        utility += 0.04
    if str(row.get("support_text", "")).strip():
        utility += 0.03
    if row.get("retrieval_utility") is not None:
        utility += 0.05 * (_as_float(row.get("retrieval_utility")) or 0.0)
    return round(min(1.0, 0.78 * confidence + 0.17 * provenance + utility), 4)


def sparse_relevance_score(
    row: Fact,
    query: str,
    seed_entities: Optional[List[str]] = None,
) -> float:
    """Lexical relevance for the current query, separate from evidence quality."""
    query_tokens = set(re.findall(r"[a-z0-9]+", str(query).lower()))
    seed_tokens = set()
    for seed in seed_entities or []:
        seed_tokens.update(re.findall(r"[a-z0-9]+", str(seed).lower()))
    wanted = query_tokens | seed_tokens
    if not wanted:
        return 0.0
    subject = str(row.get("subject", "")).lower()
    predicate = str(row.get("predicate", "")).lower().replace("_", " ")
    obj = str(row.get("object", "")).lower()
    support = str(row.get("support_text", "")).lower()
    field_tokens = set(re.findall(r"[a-z0-9]+", " ".join([subject, predicate, obj, support])))
    overlap = len(wanted.intersection(field_tokens)) / max(1, len(wanted))
    exact_entity_bonus = 0.0
    for seed in seed_entities or []:
        normalized = str(seed).strip().lower()
        if normalized and normalized in {subject.strip(), obj.strip()}:
            exact_entity_bonus = 0.25
            break
    return round(min(1.0, overlap + exact_entity_bonus), 4)


def retrieval_score_components(
    row: Fact,
    *,
    query: str,
    seed_entities: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Return auditable score components for sparse retrieval ordering."""
    relevance = sparse_relevance_score(row, query, seed_entities)
    evidence = evidence_strength_score(row)
    provenance = provenance_quality_score(row)
    utility = 0.0
    if str(row.get("question_relevance", "")).strip():
        utility += 0.6
    if str(row.get("support_text", "")).strip():
        utility += 0.25
    explicit_utility = _as_float(row.get("retrieval_utility"))
    if explicit_utility is not None:
        utility += 0.15 * explicit_utility
    utility = min(1.0, utility)
    total = (0.50 * relevance) + (0.25 * evidence) + (0.15 * provenance) + (0.10 * utility)
    return {
        "query_relevance": round(relevance, 4),
        "evidence_strength": round(evidence, 4),
        "provenance_quality": round(provenance, 4),
        "utility": round(utility, 4),
        "total": round(min(1.0, total), 4),
    }


def context_sort_key(row: Fact) -> tuple:
    return (
        -context_rank_score(row),
        str(row.get("example_id", "")),
        str(row.get("fact_id", "")),
        str(row.get("subject", "")),
        str(row.get("predicate", "")),
        str(row.get("object", "")),
    )


def temporal_scope_from_fact(fact: Fact) -> "TemporalScope":
    temporal = fact.get("temporal", {})
    if not isinstance(temporal, dict):
        temporal = {}
    compiled_temporal = _get_nested(fact, "compiled_memory", "temporal")
    if isinstance(compiled_temporal, dict):
        temporal = {**compiled_temporal, **temporal}
    return TemporalScope(
        valid_from=temporal.get("valid_from") or fact.get("valid_from"),
        valid_to=temporal.get("valid_to") or fact.get("valid_to"),
        observed_at=temporal.get("observed_at") or fact.get("observed_at") or "",
    )


def facts_temporally_overlap(left: Fact, right: Fact) -> bool:
    return temporal_scope_from_fact(left).overlaps(temporal_scope_from_fact(right))


def _metadata_for_context(row: Fact) -> str:
    example_id = str(row.get("example_id", ""))
    provenance = _provenance_entries(row)
    sent_ids = [
        str(p.get("sent_id"))
        for p in provenance
        if p.get("sent_id") is not None
    ]
    sent_part = f"sent_id={','.join(sent_ids)}" if sent_ids else "sent_id=?"
    source = f"({example_id}, {sent_part})"

    annotations = [
        f"confidence={evidence_strength_score(row):.2f}",
        f"provenance={provenance_quality_score(row):.2f}",
    ]
    valid_from = str(row.get("valid_from") or _get_nested(row, "temporal", "valid_from") or "")
    valid_to = str(row.get("valid_to") or _get_nested(row, "temporal", "valid_to") or "")
    if valid_from or valid_to:
        annotations.append(f"valid={valid_from or '..'}..{valid_to or '..'}")
    if provenance:
        spans = []
        docs = []
        for entry in provenance[:2]:
            doc = entry.get("document_id") or entry.get("doc_id") or entry.get("title")
            if doc:
                docs.append(str(doc))
            start, end = _span_bounds(entry)
            if start is not None and end is not None:
                spans.append(f"{start}:{end}")
        if docs:
            annotations.append(f"doc={','.join(dict.fromkeys(docs))}")
        if spans:
            annotations.append(f"span={','.join(spans)}")
    return f"{source} [{', '.join(annotations)}]"


def format_fact_rows_for_llm(rows: List[Fact], max_chars: int = 4000) -> str:
    """Rank and render fact rows for LLM context."""
    if not rows:
        return ""
    sorted_rows = sorted(rows, key=context_sort_key)
    lines: List[str] = []
    used = 0
    rendered = 0
    for i, row in enumerate(sorted_rows, start=1):
        subject = str(row.get("subject", ""))
        predicate = str(row.get("predicate", ""))
        obj = str(row.get("object", ""))
        support = str(row.get("support_text", "")).strip()
        head = f"[F{i}] {subject} -{predicate}-> {obj}"
        metadata = _metadata_for_context(row)
        evidence = (
            f"     evidence: \"{support}\" {metadata}"
            if support
            else f"     evidence: {metadata}"
        )
        block = head + "\n" + evidence
        block_len = len(block) + 1
        if used + block_len > max_chars:
            remaining = len(sorted_rows) - rendered
            if remaining > 0:
                lines.append(f"... [truncated, {remaining} more facts]")
            break
        lines.append(block)
        used += block_len
        rendered += 1
    return "\n".join(lines)


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
    document_id: str = ""
    sentence_id: str = ""
    source_span_start: Optional[int] = None
    source_span_end: Optional[int] = None
    local_sent_id: Optional[Any] = None
    chunk_index: Optional[int] = None
    extractor_model: str = ""
    verifier_model: str = ""
    run_id: str = ""

    @classmethod
    def from_fact_entry(
        cls,
        entry: Any,
        support_text: str = "",
        fact: Optional[Fact] = None,
    ) -> "ProvenanceRecord":
        fact = fact or {}
        if isinstance(entry, dict):
            title = str(entry.get("title", ""))
            sent_id = entry.get("sent_id")
            document_id = str(
                entry.get("document_id")
                or entry.get("doc_id")
                or fact.get("document_id")
                or fact.get("example_id")
                or title
            )
            sentence_id = str(
                entry.get("sentence_id")
                or (f"{document_id}:{sent_id}" if document_id and sent_id is not None else "")
            )
            start, end = _span_bounds(entry)
            source_id = str(
                entry.get("source_id")
                or stable_id("source", [document_id, title, sent_id, start, end])
            )
            return cls(
                source_id=source_id,
                title=title,
                sent_id=sent_id,
                support_text=str(entry.get("support_text") or support_text or ""),
                document_id=document_id,
                sentence_id=sentence_id,
                source_span_start=_as_int(start),
                source_span_end=_as_int(end),
                local_sent_id=entry.get("local_sent_id"),
                chunk_index=entry.get("chunk_index", fact.get("chunk_index")),
                extractor_model=str(entry.get("extractor_model") or fact.get("extractor_model", "")),
                verifier_model=str(entry.get("verifier_model") or fact.get("verifier_model", "")),
                run_id=str(entry.get("run_id") or fact.get("run_id", "")),
            )
        text = str(entry or "")
        return cls(
            source_id=stable_id("source", [text]),
            title=text,
            support_text=support_text,
            document_id=str(fact.get("document_id") or fact.get("example_id") or ""),
            extractor_model=str(fact.get("extractor_model", "")),
            verifier_model=str(fact.get("verifier_model", "")),
            run_id=str(fact.get("run_id", "")),
        )

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
        else:
            self.score = _as_float(self.score)
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
    document_id: str = ""
    extractor_model: str = ""
    verifier_model: str = ""
    run_id: str = ""

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
            "document_id": self.document_id,
            "extractor_model": self.extractor_model,
            "verifier_model": self.verifier_model,
            "run_id": self.run_id,
        }

    def to_fact(self) -> Fact:
        """Return the existing simple fact shape with compiled memory attached."""
        fact = {
            "subject": self.subject.name,
            "predicate": self.predicate,
            "object": self.object.name,
            "qualifiers": dict(self.qualifiers),
            "temporal": self.temporal.to_dict(),
            "provenance": [
                {
                    "source_id": p.source_id,
                    "title": p.title,
                    "sent_id": p.sent_id,
                    "support_text": p.support_text,
                    "document_id": p.document_id,
                    "sentence_id": p.sentence_id,
                    "source_span_start": p.source_span_start,
                    "source_span_end": p.source_span_end,
                    "local_sent_id": p.local_sent_id,
                    "chunk_index": p.chunk_index,
                    "extractor_model": p.extractor_model,
                    "verifier_model": p.verifier_model,
                    "run_id": p.run_id,
                }
                for p in self.provenance
            ],
            "support_text": self.support_text,
            "question_relevance": self.question_relevance,
            "confidence": self.confidence.level,
            "confidence_score": self.confidence.score,
            "confidence_method": self.confidence.method,
            "normalization_notes": self.normalization_notes,
            "example_id": self.example_id,
            "document_id": self.document_id,
            "fact_id": self.memory_id,
            "question": self.question,
            "verification_reason": self.verification_reason,
            "extractor_model": self.extractor_model,
            "verifier_model": self.verifier_model,
            "run_id": self.run_id,
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
        fact.setdefault("document_id", compiled.get("document_id", ""))
        fact.setdefault("extractor_model", compiled.get("extractor_model", ""))
        fact.setdefault("verifier_model", compiled.get("verifier_model", ""))
        fact.setdefault("run_id", compiled.get("run_id", ""))
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
    provenance_raw = fact.get("provenance", [])
    if not isinstance(provenance_raw, list):
        provenance_raw = []
    support_text = str(fact.get("support_text", ""))
    provenance = [
        ProvenanceRecord.from_fact_entry(entry, support_text=support_text, fact=fact)
        for entry in provenance_raw
    ]
    if support_text and not provenance:
        provenance = [
            ProvenanceRecord(
                source_id=stable_id("source", [fact.get("example_id", ""), support_text]),
                title=str(fact.get("example_id", "")),
                sent_id=None,
                support_text=support_text,
                document_id=str(fact.get("document_id") or fact.get("example_id", "")),
                chunk_index=fact.get("chunk_index"),
                extractor_model=str(fact.get("extractor_model", "")),
                verifier_model=str(fact.get("verifier_model", "")),
                run_id=str(fact.get("run_id", "")),
            )
        ]
    score_fact = dict(fact)
    score_fact["provenance"] = [p.to_dict() for p in provenance]
    score_fact["document_id"] = str(fact.get("document_id") or fact.get("example_id", ""))
    confidence_raw = fact.get("confidence", fact.get("status", DEFAULT_CONFIDENCE_LEVEL))
    confidence_score = fact.get("confidence_score")
    confidence = ConfidenceRecord(
        level=confidence_raw,
        score=confidence_score if confidence_score is not None else evidence_strength_score(score_fact),
        method=fact.get(
            "confidence_method",
            "explicit_confidence_score" if confidence_score is not None else "weighted_evidence_v1",
        ),
    )

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
        document_id=str(fact.get("document_id") or fact.get("example_id", "")),
        extractor_model=str(fact.get("extractor_model", "")),
        verifier_model=str(fact.get("verifier_model", "")),
        run_id=str(fact.get("run_id", "")),
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
        "provenance_quality": provenance_quality_score(memory.to_fact()),
        "document_id": memory.document_id,
        "extractor_model": memory.extractor_model,
        "verifier_model": memory.verifier_model,
        "run_id": memory.run_id,
        "decision_status": memory.decision.status,
        "decision_validator": memory.decision.validator,
        "decision_reason": memory.decision.reason or "",
        "replaces_memory_id": memory.decision.replaces or "",
        "constraints_json": json.dumps(memory.constraints.to_dict(), ensure_ascii=False),
        "compiled_memory_json": json.dumps(memory.to_dict(), ensure_ascii=False),
    }
