from __future__ import annotations

from typing import Any, Callable, Dict, Protocol

from neurosym.domain import RetrievalRequest, RetrievalResult


class RetrievalStrategy(Protocol):
    def __call__(self, request: RetrievalRequest) -> RetrievalResult:
        ...


class RetrievalService:
    def __init__(self, mode: str, strategies: Dict[str, RetrievalStrategy]) -> None:
        self._mode = str(mode)
        self._strategies = dict(strategies)
        self._validate_registry()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(self._strategies)

    def register(self, mode: str, strategy: RetrievalStrategy) -> None:
        name = str(mode)
        if name not in {"sparse", "dense", "hybrid", "dense_ppr"}:
            raise ValueError(f"unsupported retrieval mode: {name}")
        self._strategies[name] = strategy

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        strategy = self._strategies.get(self._mode)
        if strategy is None:
            raise ValueError(f"retrieval strategy is not registered: {self._mode}")
        outcome = strategy(request)
        if isinstance(outcome, RetrievalResult):
            return outcome
        if hasattr(outcome, "rows") and hasattr(outcome, "metadata"):
            return RetrievalResult.from_mapping(
                {"rows": list(outcome.rows), "metadata": dict(outcome.metadata)}
            )
        raise TypeError("retrieval strategy returned an invalid result")

    def _validate_registry(self) -> None:
        allowed = {"sparse", "dense", "hybrid", "dense_ppr"}
        invalid = set(self._strategies) - allowed
        if invalid:
            raise ValueError(f"unsupported retrieval strategies: {', '.join(sorted(invalid))}")
        if self._mode not in allowed:
            raise ValueError(f"unsupported retrieval mode: {self._mode}")


class CallableRetrievalStrategy:
    def __init__(self, callback: Callable[[RetrievalRequest], Any]) -> None:
        self._callback = callback

    def __call__(self, request: RetrievalRequest) -> RetrievalResult:
        outcome = self._callback(request)
        if isinstance(outcome, RetrievalResult):
            return outcome
        return RetrievalResult.from_mapping(
            {"rows": list(outcome.rows), "metadata": dict(outcome.metadata)}
        )
