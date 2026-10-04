"""Resolve retractions across duplicate private-memory lineages."""

from __future__ import annotations

import importlib.metadata
import json
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


UNKNOWN = "UNKNOWN"
LINEAGE_MODES = ("recursive", "one_hop")


def _lineage_inputs(
    events: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
    *,
    strict: bool,
) -> tuple[str, str, dict[str, Mapping[str, Any]], list[tuple[str, str]], set[str]]:
    """Validate and normalize one history-local private-lineage problem."""
    history_id = str(query.get("history_id") or "").strip()
    subject = str(query.get("subject") or "").strip()
    requested_fact_id = str(query.get("fact_id") or "").strip()
    if not history_id or not subject or not requested_fact_id:
        raise ValueError("private-lineage query requires history_id, subject, and fact_id")

    memories: dict[str, Mapping[str, Any]] = {}
    duplicate_edges: list[tuple[str, str]] = []
    retracted: set[str] = set()
    pending_duplicates: list[tuple[str, str]] = []
    for event in events:
        if str(event.get("history_id")) != history_id:
            continue
        fact = event.get("fact")
        if not isinstance(fact, Mapping):
            if strict:
                raise ValueError("private-lineage event fact must be an object")
            continue
        fact_id = str(fact.get("fact_id") or "").strip()
        if not fact_id:
            if strict:
                raise ValueError("private-lineage event fact_id must not be empty")
            continue
        qualifiers = fact.get("qualifiers")
        if (
            fact.get("subject") == subject
            and isinstance(qualifiers, Mapping)
            and qualifiers.get("scope") == "private"
        ):
            memories[fact_id] = fact

        operation = event.get("operation")
        if operation == "duplicate_delivery":
            if fact_id not in memories:
                continue
            parent = str(event.get("duplicate_of") or "").strip()
            if not parent:
                if strict:
                    raise ValueError(f"duplicate event for {fact_id} lacks duplicate_of")
                continue
            pending_duplicates.append((fact_id, parent))
        elif operation == "retract":
            target = str(event.get("retracts") or "").strip()
            if target:
                retracted.add(target)

    for child, parent in pending_duplicates:
        child_fact = memories.get(child)
        parent_fact = memories.get(parent)
        if child_fact is None or parent_fact is None:
            if strict:
                raise ValueError(
                    f"duplicate lineage reference {child!r}->{parent!r} is not fully observed"
                )
            continue
        comparable_fields = (
            child_fact.get("subject") == parent_fact.get("subject"),
            child_fact.get("object") == parent_fact.get("object"),
            child_fact.get("qualifiers", {}).get("scope")
            == parent_fact.get("qualifiers", {}).get("scope"),
        )
        if not all(comparable_fields):
            raise ValueError(f"duplicate lineage mismatch for {child!r}->{parent!r}")
        duplicate_edges.append((child, parent))
    return history_id, requested_fact_id, memories, duplicate_edges, retracted


def _connected_retractions(
    memories: Mapping[str, Mapping[str, Any]],
    duplicate_edges: Sequence[tuple[str, str]],
    directly_retracted: set[str],
) -> set[str]:
    """Propagate tombstones through each undirected duplicate component."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    for child, parent in duplicate_edges:
        neighbors[child].add(parent)
        neighbors[parent].add(child)
    retracted = set(directly_retracted)
    queue = deque(directly_retracted)
    while queue:
        fact_id = queue.popleft()
        for neighbor in neighbors[fact_id]:
            if neighbor not in retracted:
                retracted.add(neighbor)
                queue.append(neighbor)
    return retracted & memories.keys()


def resolve_private_lineage_reference(
    events: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Replay the benchmark's Python gold resolver for one private query."""
    _, requested, memories, edges, directly_retracted = _lineage_inputs(
        events, query, strict=strict
    )
    retracted = _connected_retractions(memories, edges, directly_retracted)
    requested_fact = memories.get(requested)
    answer = (
        UNKNOWN
        if requested_fact is None or requested in retracted
        else str(requested_fact["object"])
    )
    return {
        "answer": answer,
        "retracted_fact_ids": sorted(retracted),
        "same_memory_edges": sorted([list(edge) for edge in edges]),
        "engine": "python_gold_resolver_replay",
    }


def resolve_private_lineage_with_scallop(
    events: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Resolve one lineage through recursive or one-hop Scallop Datalog."""
    if mode not in LINEAGE_MODES:
        raise ValueError(f"unsupported private-lineage mode: {mode}")
    try:
        import scallopy
    except ImportError as error:
        raise RuntimeError("scallopy is required for private-lineage reasoning") from error

    history_id, requested, memories, edges, directly_retracted = _lineage_inputs(
        events, query, strict=True
    )
    context = scallopy.ScallopContext(provenance="unit")
    context.add_relation("memory", (str, str, str, str, str))
    context.add_relation("alias", (str, str, str))
    context.add_relation("retract", (str, str))
    context.add_relation("private_query", (str, str, str, str))
    context.add_facts(
        "memory",
        [
            (
                history_id,
                fact_id,
                str(fact["subject"]),
                str(fact["object"]),
                "private",
            )
            for fact_id, fact in memories.items()
        ],
    )
    context.add_facts("alias", [(history_id, child, parent) for child, parent in edges])
    context.add_facts(
        "retract", [(history_id, fact_id) for fact_id in directly_retracted]
    )
    context.add_facts(
        "private_query",
        [
            (
                str(query.get("query_id") or "private-lineage-query"),
                history_id,
                str(query["subject"]),
                requested,
            )
        ],
    )
    context.add_rule("linked(h, a, b) = alias(h, a, b)")
    context.add_rule("linked(h, a, b) = alias(h, b, a)")
    context.add_rule("same_memory(h, a, b) = linked(h, a, b)")
    if mode == "recursive":
        context.add_rule(
            "same_memory(h, a, c) = linked(h, a, b) and same_memory(h, b, c)"
        )
    context.add_rule("retracted(h, fact) = retract(h, fact)")
    context.add_rule(
        "retracted(h, fact) = retract(h, target) and same_memory(h, target, fact)"
    )
    context.add_rule(
        'private_answer(q, value) = private_query(q, h, subject, fact) and '
        'memory(h, fact, subject, value, "private") and not retracted(h, fact)'
    )
    context.run()
    answers = sorted({str(row[1]) for row in context.relation("private_answer")})
    if len(answers) > 1:
        raise RuntimeError(f"Scallop private-lineage query produced multiple answers: {answers}")
    retracted_rows = sorted(
        str(row[1])
        for row in context.relation("retracted")
        if str(row[0]) == history_id
    )
    same_memory_rows = sorted(
        [str(row[1]), str(row[2])]
        for row in context.relation("same_memory")
        if str(row[0]) == history_id
    )
    return {
        "answer": answers[0] if answers else UNKNOWN,
        "retracted_fact_ids": retracted_rows,
        "same_memory_edges": same_memory_rows,
        "engine": "scallopy",
        "mode": mode,
        "scallopy_version": importlib.metadata.version("scallopy"),
    }


class ScallopLineageClient:
    """Hard-timeout HTTP client for the Python 3.10 Scallop service."""

    def __init__(self, endpoint: str, timeout_seconds: float) -> None:
        endpoint = endpoint.rstrip("/")
        if not endpoint:
            raise ValueError("Scallop lineage endpoint must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("Scallop lineage timeout must be positive")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def resolve(
        self,
        events: Sequence[Mapping[str, Any]],
        query: Mapping[str, Any],
        *,
        mode: str,
    ) -> dict[str, Any]:
        """Resolve one query and validate the service response contract."""
        payload = json.dumps(
            {"events": list(events), "query": dict(query), "mode": mode}
        ).encode("utf-8")
        request = Request(
            f"{self.endpoint}/resolve_private_lineage",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Scallop private-lineage request failed with HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Scallop private-lineage request failed: {error}") from error
        if not isinstance(result, dict):
            raise RuntimeError("Scallop private-lineage response must be an object")
        if result.get("engine") != "scallopy" or result.get("mode") != mode:
            raise RuntimeError(f"invalid Scallop private-lineage response: {result}")
        if result.get("answer") is None or not isinstance(
            result.get("retracted_fact_ids"), list
        ):
            raise RuntimeError(f"incomplete Scallop private-lineage response: {result}")
        return result
