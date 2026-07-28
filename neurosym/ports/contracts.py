from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

from neurosym.domain import RetrievalRequest, RetrievalResult, Scope, ValidationDecision


@runtime_checkable
class FactReader(Protocol):
    def read_facts(self, scope: Scope) -> List[Dict[str, Any]]:
        ...


@runtime_checkable
class FactWriter(Protocol):
    def write_facts(self, facts: Sequence[Mapping[str, Any]], scope: Scope) -> int:
        ...


@runtime_checkable
class RetrievalPort(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        ...


@runtime_checkable
class ValidationPort(Protocol):
    def validate(
        self,
        existing_facts: Sequence[Mapping[str, Any]],
        candidate: Mapping[str, Any],
        rule_params: Optional[Any] = None,
    ) -> ValidationDecision:
        ...


@runtime_checkable
class WorkingMemoryPersistence(Protocol):
    def persist_working_memory(self, artifact: Any, transition: Any) -> Mapping[str, Any]:
        ...

    def working_memory_history(self, *, session_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ...


@runtime_checkable
class LLMExecutor(Protocol):
    def complete(self, messages: Sequence[Mapping[str, Any]], **options: Any) -> Any:
        ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        ...

    def monotonic(self) -> float:
        ...


@runtime_checkable
class ArtifactSink(Protocol):
    def write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        ...

    def append_jsonl(self, path: Path, value: Mapping[str, Any]) -> None:
        ...


@runtime_checkable
class DenseIndex(Protocol):
    @property
    def manifest(self) -> Any:
        ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        scope: Scope,
        predicates: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        ...
