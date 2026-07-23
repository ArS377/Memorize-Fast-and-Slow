from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from compiled_memory import evidence_strength_score, retrieval_score_components
from experiments.retrieval_config import DenseRetrievalError


@dataclass(frozen=True)
class RetrievalOutcome:
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def _canonical_predicate(value: Any) -> str:
    return re.sub(
        r"_+",
        "_",
        re.sub(r"[^A-Z0-9_]+", "_", str(value).strip().upper()),
    ).strip("_")


def _resolve_predicate_filter(
    source: Any,
    predicates: Optional[Sequence[str]],
) -> tuple[Optional[List[str]], Dict[str, Any]]:
    requested = list(
        dict.fromkeys(
            canonical
            for value in predicates or []
            if (canonical := _canonical_predicate(value))
        )
    )
    known = {
        _canonical_predicate(fact.get("predicate", ""))
        for index in (getattr(source, "dense_indexes", {}) or {}).values()
        for fact in getattr(index, "facts", [])
    }
    known.update(
        _canonical_predicate(fact.get("predicate", ""))
        for fact in (getattr(source, "fallback_facts", None) or [])
    )
    known.discard("")
    if not requested or not known:
        applied = requested
        relaxed = False
    else:
        applied = [predicate for predicate in requested if predicate in known]
        relaxed = applied != requested
    return (applied or None), {
        "requested": requested,
        "applied": applied,
        "relaxed": relaxed,
    }


def _sparse_rows(
    source: Any,
    *,
    query: str,
    seed_entities: List[str],
    example_id: str,
    hops: int,
    top_k: int,
    predicates: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if not seed_entities:
        return []
    rows = source.rows_for(
        seed_entities=seed_entities,
        example_id=example_id,
        hops=hops,
        limit_triples=top_k,
        predicates=predicates,
    )
    if not isinstance(rows, list):
        raise TypeError("sparse retrieval returned a non-list response")
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -retrieval_score_components(row, query=query, seed_entities=seed_entities)["total"],
            str(row.get("fact_id", "")),
        ),
    )
    return ranked[:top_k]


def _dense_rows(
    source: Any,
    *,
    query: str,
    example_id: str,
    top_k: int,
    predicates: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    failure = getattr(source, "dense_failure", None)
    if failure is not None:
        raise failure
    indexes = getattr(source, "dense_indexes", {}) or {}
    scope = source.trusted_scope(example_id)
    missing = [session for session in scope.session_ids if session not in indexes]
    if missing:
        raise DenseRetrievalError(
            "dense_index_unavailable",
            f"dense index unavailable for trusted session(s): {', '.join(missing)}",
        )
    rows: List[Dict[str, Any]] = []
    for session_id in scope.session_ids:
        rows.extend(
            indexes[session_id].search(
                query,
                top_k=top_k,
                scope=scope,
                predicates=predicates,
            )
        )
    fact_ids = [str(row.get("fact_id", "")) for row in rows]
    if len(fact_ids) != len(set(fact_ids)):
        raise DenseRetrievalError(
            "dense_index_mismatch",
            "trusted dense indexes contain duplicate fact IDs",
        )
    rows.sort(key=lambda row: (-float(row.get("_dense_similarity", 0.0)), str(row.get("fact_id", ""))))
    return rows[:top_k]


def _metadata(
    source: Any,
    *,
    configured_mode: str,
    effective_mode: str,
    degraded: bool,
    sparse_count: int,
    dense_count: int,
    sparse_latency: float,
    dense_latency: float,
    predicate_filter: Dict[str, Any],
    failure: Optional[DenseRetrievalError] = None,
) -> Dict[str, Any]:
    config = source.retrieval_config
    indexes = getattr(source, "dense_indexes", {}) or {}
    identities = [indexes[key].manifest.identity for key in sorted(indexes)]
    return {
        "configured_mode": configured_mode,
        "effective_mode": effective_mode,
        "degraded": degraded,
        "branch_counts": {"sparse": sparse_count, "dense": dense_count},
        "branch_latency_seconds": {
            "sparse": round(sparse_latency, 3),
            "dense": round(dense_latency, 3),
        },
        "rrf": {
            "k": config.rrf_k,
            "branch_candidate_multiplier": config.branch_candidate_multiplier,
            "branch_candidate_cap": config.branch_candidate_cap,
        },
        "dense_index_identity": identities,
        "predicate_filter": predicate_filter,
        "warning": (
            {"code": failure.code, "message": str(failure)}
            if failure is not None and degraded
            else None
        ),
    }


def _annotate_sparse(rows: List[Dict[str, Any]], query: str, seeds: List[str]) -> List[Dict[str, Any]]:
    output = []
    for rank, row in enumerate(rows, start=1):
        value = dict(row)
        components = retrieval_score_components(value, query=query, seed_entities=seeds)
        value["_retrieval"] = {
            "score": components["total"],
            "score_components": components,
            "sparse_rank": rank,
            "sparse_score": components["total"],
            "dense_rank": None,
            "dense_similarity": None,
            "rrf_score": None,
            "matched_branches": ["sparse"],
        }
        output.append(value)
    return output


def _annotate_dense(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for rank, row in enumerate(rows, start=1):
        value = dict(row)
        similarity = float(value.pop("_dense_similarity"))
        evidence = evidence_strength_score(value)
        components = {
            "dense_similarity": similarity,
            "evidence_strength": evidence,
            "total": similarity,
        }
        value["_retrieval"] = {
            "score": similarity,
            "score_components": components,
            "sparse_rank": None,
            "sparse_score": None,
            "dense_rank": rank,
            "dense_similarity": similarity,
            "rrf_score": None,
            "matched_branches": ["dense"],
        }
        output.append(value)
    return output


def _rrf_rows(
    sparse_rows: List[Dict[str, Any]],
    dense_rows: List[Dict[str, Any]],
    *,
    query: str,
    seeds: List[str],
    rrf_k: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    combined: Dict[str, Dict[str, Any]] = {}
    sparse_info: Dict[str, tuple[int, float]] = {}
    dense_info: Dict[str, tuple[int, float]] = {}
    for rank, row in enumerate(sparse_rows, start=1):
        fact_id = str(row.get("fact_id", ""))
        components = retrieval_score_components(row, query=query, seed_entities=seeds)
        combined[fact_id] = dict(row)
        sparse_info[fact_id] = (rank, components["total"])
    for rank, row in enumerate(dense_rows, start=1):
        fact_id = str(row.get("fact_id", ""))
        combined.setdefault(fact_id, dict(row))
        dense_info[fact_id] = (rank, float(row.get("_dense_similarity", 0.0)))
    output: List[Dict[str, Any]] = []
    for fact_id, row in combined.items():
        sparse_rank, sparse_score = sparse_info.get(fact_id, (None, None))
        dense_rank, dense_similarity = dense_info.get(fact_id, (None, None))
        score = 0.0
        branches = []
        if sparse_rank is not None:
            score += 1.0 / (rrf_k + sparse_rank)
            branches.append("sparse")
        if dense_rank is not None:
            score += 1.0 / (rrf_k + dense_rank)
            branches.append("dense")
        value = dict(row)
        value.pop("_dense_similarity", None)
        components = {
            "rrf": score,
            "evidence_strength": evidence_strength_score(value),
            "sparse_score": sparse_score,
            "dense_similarity": dense_similarity,
            "total": score,
        }
        value["_retrieval"] = {
            "score": score,
            "score_components": components,
            "sparse_rank": sparse_rank,
            "sparse_score": sparse_score,
            "dense_rank": dense_rank,
            "dense_similarity": dense_similarity,
            "rrf_score": score,
            "matched_branches": branches,
        }
        output.append(value)
    output.sort(
        key=lambda row: (
            -float(row["_retrieval"]["rrf_score"]),
            -evidence_strength_score(row),
            str(row.get("fact_id", "")),
        )
    )
    return output[:top_k]


def retrieve(
    source: Any,
    *,
    query: str,
    seed_entities: List[str],
    example_id: str,
    hops: int,
    top_k: int,
    predicates: Optional[List[str]] = None,
) -> RetrievalOutcome:
    config = source.retrieval_config
    mode = config.mode
    depth = config.branch_depth(top_k)
    sparse: List[Dict[str, Any]] = []
    dense: List[Dict[str, Any]] = []
    sparse_latency = 0.0
    dense_latency = 0.0
    effective_predicates, predicate_filter = _resolve_predicate_filter(
        source, predicates
    )

    if mode in {"sparse", "hybrid"}:
        started = time.perf_counter()
        sparse = _sparse_rows(
            source,
            query=query,
            seed_entities=seed_entities,
            example_id=example_id,
            hops=hops,
            top_k=depth if mode == "hybrid" else top_k,
            predicates=effective_predicates,
        )
        sparse_latency = time.perf_counter() - started
    if mode in {"dense", "hybrid"}:
        started = time.perf_counter()
        try:
            dense = _dense_rows(
                source,
                query=query,
                example_id=example_id,
                top_k=depth if mode == "hybrid" else top_k,
                predicates=effective_predicates,
            )
        except DenseRetrievalError as exc:
            dense_latency = time.perf_counter() - started
            if mode == "hybrid" and config.failure_policy == "sparse":
                rows = _annotate_sparse(sparse[:top_k], query, seed_entities)
                metadata = _metadata(
                    source,
                    configured_mode=mode,
                    effective_mode="sparse",
                    degraded=True,
                    sparse_count=len(sparse),
                    dense_count=0,
                    sparse_latency=sparse_latency,
                    dense_latency=dense_latency,
                    predicate_filter=predicate_filter,
                    failure=exc,
                )
                metadata["branch_fact_ids"] = {
                    "sparse": [str(row.get("fact_id", "")) for row in sparse],
                    "dense": [],
                }
                metadata["result_fact_ids"] = [
                    str(row.get("fact_id", "")) for row in rows
                ]
                source._record_retrieval_metadata(metadata)
                return RetrievalOutcome(rows=rows, metadata=metadata)
            metadata = _metadata(
                source,
                configured_mode=mode,
                effective_mode=mode,
                degraded=False,
                sparse_count=len(sparse),
                dense_count=0,
                sparse_latency=sparse_latency,
                dense_latency=dense_latency,
                predicate_filter=predicate_filter,
            )
            metadata["error"] = {"code": exc.code, "message": str(exc)}
            metadata["branch_fact_ids"] = {
                "sparse": [str(row.get("fact_id", "")) for row in sparse],
                "dense": [],
            }
            metadata["result_fact_ids"] = []
            source._record_retrieval_metadata(metadata)
            raise
        dense_latency = time.perf_counter() - started

    if mode == "sparse":
        rows = _annotate_sparse(sparse[:top_k], query, seed_entities)
    elif mode == "dense":
        rows = _annotate_dense(dense[:top_k])
    else:
        rows = _rrf_rows(
            sparse,
            dense,
            query=query,
            seeds=seed_entities,
            rrf_k=config.rrf_k,
            top_k=top_k,
        )
    metadata = _metadata(
        source,
        configured_mode=mode,
        effective_mode=mode,
        degraded=False,
        sparse_count=len(sparse),
        dense_count=len(dense),
        sparse_latency=sparse_latency,
        dense_latency=dense_latency,
        predicate_filter=predicate_filter,
    )
    metadata["branch_fact_ids"] = {
        "sparse": [str(row.get("fact_id", "")) for row in sparse],
        "dense": [str(row.get("fact_id", "")) for row in dense],
    }
    metadata["result_fact_ids"] = [str(row.get("fact_id", "")) for row in rows]
    source._record_retrieval_metadata(metadata)
    return RetrievalOutcome(rows=rows, metadata=metadata)
