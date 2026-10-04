from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from neurosym.domain import Scope


def canonical_predicate(value: Any) -> str:
    return re.sub(
        r"_+",
        "_",
        re.sub(r"[^A-Z0-9_]+", "_", str(value).strip().upper()),
    ).strip("_")


class JsonlFactRepository:
    def __init__(
        self,
        facts: Sequence[Mapping[str, Any]],
        *,
        sort_key: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    ) -> None:
        self._facts = [dict(fact) for fact in facts]
        self._sort_key = sort_key or self._default_sort_key

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        sort_key: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    ) -> "JsonlFactRepository":
        facts: List[Dict[str, Any]] = []
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid fact at line {line_number}: expected an object")
                facts.append(value)
        return cls(facts, sort_key=sort_key)

    @staticmethod
    def _default_sort_key(fact: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            str(fact.get("example_id", "")),
            str(fact.get("session_id", "")),
            str(fact.get("fact_id", "")),
            str(fact.get("subject", "")),
            canonical_predicate(fact.get("predicate", "")),
            str(fact.get("object", "")),
        )

    @property
    def facts(self) -> List[Dict[str, Any]]:
        return [dict(fact) for fact in self._facts]

    def read_facts(self, scope: Scope) -> List[Dict[str, Any]]:
        sessions = set(scope.session_ids)
        rows: List[Dict[str, Any]] = []
        for fact in self._facts:
            session_id = str(fact.get("session_id", ""))
            if session_id and session_id not in sessions:
                continue
            if scope.mode == "example" and str(fact.get("example_id", "")) != str(scope.example_id):
                continue
            rows.append(dict(fact))
        return sorted(rows, key=self._sort_key)

    def search(
        self,
        scope: Scope,
        *,
        seed_entities: Sequence[str],
        predicates: Optional[Sequence[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        seeds = [str(seed).strip().casefold() for seed in seed_entities if str(seed).strip()]
        if not seeds:
            return []
        wanted = {
            canonical_predicate(predicate)
            for predicate in predicates or []
            if canonical_predicate(predicate)
        }
        rows: List[Dict[str, Any]] = []
        for fact in self.read_facts(scope):
            predicate = canonical_predicate(fact.get("predicate", ""))
            if wanted and predicate not in wanted:
                continue
            subject = str(fact.get("subject", "")).casefold()
            object_value = str(fact.get("object", "")).casefold()
            if any(seed in subject or seed in object_value for seed in seeds):
                row = dict(fact)
                row["predicate"] = predicate
                rows.append(row)
        return sorted(rows, key=self._sort_key)[: max(0, int(limit))]
