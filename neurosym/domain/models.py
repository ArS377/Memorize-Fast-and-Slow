from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple, Type, TypeVar


T = TypeVar("T", bound="MappingDTO")


@dataclass
class MappingDTO:
    extension_fields: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    _provided_fields: set[str] = field(default_factory=set, init=False, repr=False)
    aliases: ClassVar[Dict[str, str]] = {}

    @classmethod
    def from_mapping(cls: Type[T], value: Mapping[str, Any]) -> T:
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} requires a mapping")
        names = {
            item.name
            for item in fields(cls)
            if item.name not in {"extension_fields", "_provided_fields"}
        }
        reverse_aliases = {external: internal for internal, external in cls.aliases.items()}
        known: Dict[str, Any] = {}
        extensions: Dict[str, Any] = {}
        for key, item in value.items():
            target = reverse_aliases.get(str(key), str(key))
            if target in names:
                known[target] = item
            else:
                extensions[str(key)] = item
        known["extension_fields"] = extensions
        instance = cls(**known)
        instance._provided_fields = set(known) - {"extension_fields"}
        return instance

    def to_mapping(self) -> Dict[str, Any]:
        result = dict(self.extension_fields)
        for item in fields(self):
            if item.name in {"extension_fields", "_provided_fields"}:
                continue
            if self._provided_fields and item.name not in self._provided_fields:
                continue
            key = self.aliases.get(item.name, item.name)
            value = getattr(self, item.name)
            result[key] = _mapping_value(value)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return self.to_mapping()


def _mapping_value(value: Any) -> Any:
    if isinstance(value, MappingDTO):
        return value.to_mapping()
    if isinstance(value, tuple):
        return [_mapping_value(item) for item in value]
    if isinstance(value, list):
        return [_mapping_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _mapping_value(item) for key, item in value.items()}
    return value


@dataclass
class Provenance(MappingDTO):
    source_id: str = ""
    title: str = ""
    sent_id: Any = None
    support_text: str = ""
    document_id: str = ""
    sentence_id: str = ""
    source_span_start: Optional[int] = None
    source_span_end: Optional[int] = None
    local_sent_id: Any = None
    chunk_index: Optional[int] = None
    extractor_model: str = ""
    verifier_model: str = ""
    run_id: str = ""


@dataclass
class Fact(MappingDTO):
    fact_id: str = ""
    session_id: str = ""
    example_id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    support_text: str = ""
    question: str = ""
    question_relevance: str = ""
    confidence: str = ""
    confidence_score: Optional[float] = None
    status: str = ""
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    decision: Dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> Tuple[str, str]:
        return self.fact_id, self.session_id


@dataclass
class Scope(MappingDTO):
    mode: str = "example"
    session_ids: Tuple[str, ...] = field(default_factory=tuple)
    example_id: Optional[str] = None

    aliases: ClassVar[Dict[str, str]] = {"mode": "memory_scope"}

    def __post_init__(self) -> None:
        self.mode = str(self.mode or "example").strip().lower()
        self.session_ids = tuple(dict.fromkeys(str(item) for item in self.session_ids if str(item)))
        if self.mode not in {"example", "session", "session_set"}:
            raise ValueError("scope mode must be example, session, or session_set")
        if self.mode == "example" and not self.example_id:
            raise ValueError("example scope requires example_id")
        if self.mode in {"example", "session"} and len(self.session_ids) != 1:
            raise ValueError(f"{self.mode} scope requires exactly one session_id")
        if self.mode == "session_set" and len(self.session_ids) < 2:
            raise ValueError("session_set scope requires at least two session_ids")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Scope":
        normalized = dict(value)
        if "session_ids" not in normalized and normalized.get("session_id") is not None:
            normalized["session_ids"] = [normalized["session_id"]]
        normalized.pop("session_id", None)
        return super().from_mapping(normalized)


@dataclass
class ValidationDecision(MappingDTO):
    decision: str = "reject"
    reason: str = ""
    validator: str = ""
    rule_version: str = ""
    replace_fact_id: Optional[str] = None
    rejection_label: Optional[Dict[str, Any]] = None
    committed: bool = False


@dataclass
class RetrievalRequest(MappingDTO):
    query: str = ""
    scope: Optional[Scope] = None
    seed_entities: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    top_k: int = 10
    hops: int = 2

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RetrievalRequest":
        normalized = dict(value)
        if isinstance(normalized.get("scope"), Mapping):
            normalized["scope"] = Scope.from_mapping(normalized["scope"])
        return super().from_mapping(normalized)


@dataclass
class RetrievalTelemetry(MappingDTO):
    configured_mode: str = "sparse"
    effective_mode: str = "sparse"
    degraded: bool = False
    branch_counts: Dict[str, int] = field(default_factory=dict)
    branch_fact_ids: Dict[str, List[str]] = field(default_factory=dict)
    branch_latency_seconds: Dict[str, float] = field(default_factory=dict)
    result_fact_ids: List[str] = field(default_factory=list)
    dense_index_identity: List[str] = field(default_factory=list)
    fusion: Optional[Dict[str, Any]] = None
    warning: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult(MappingDTO):
    rows: List[Dict[str, Any]] = field(default_factory=list)
    telemetry: RetrievalTelemetry = field(default_factory=RetrievalTelemetry)

    aliases: ClassVar[Dict[str, str]] = {"telemetry": "metadata"}

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.telemetry.to_mapping()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RetrievalResult":
        normalized = dict(value)
        telemetry = normalized.get("telemetry", normalized.get("metadata", {}))
        if isinstance(telemetry, Mapping):
            normalized["telemetry"] = RetrievalTelemetry.from_mapping(telemetry)
        normalized.pop("metadata", None)
        return super().from_mapping(normalized)


@dataclass
class WorkingMemoryTransition(MappingDTO):
    transition_id: str = ""
    artifact_id: str = ""
    scope: Optional[Scope] = None
    decision: str = ""
    reason: str = ""
    validator: str = ""
    rule_version: str = ""
    before_artifact_id: Optional[str] = None
    after_artifact_id: Optional[str] = None
    committed: bool = False
    created_at: str = ""


@dataclass
class ExperimentResult(MappingDTO):
    cell_id: Optional[int] = None
    label: str = ""
    example_id: str = ""
    predicted: str = ""
    gold: str = ""
    correct: bool = False
    n_context_chars: int = 0
    n_triples: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    memory_scope: Optional[str] = None
    orchestration_mode: Optional[str] = None
    validator_backend: Optional[str] = None
    configured_retrieval_mode: Optional[str] = None
    effective_retrieval_mode: Optional[str] = None
    retrieval_degraded: Optional[bool] = None
