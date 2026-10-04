from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neurosym.domain.retrieval_config import PPRConfig, PPRRetrievalError


PPR_INDEX_SCHEMA_VERSION = "ppr_fact_index.v1"
FactKey = Tuple[str, str]


def _fact_key(fact: Mapping[str, Any]) -> FactKey:
    return str(fact.get("session_id", "")), str(fact.get("fact_id", ""))


def _entity_key(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split()).casefold()


def _canonical_predicate(value: Any) -> str:
    return re.sub(
        r"_+",
        "_",
        re.sub(r"[^A-Z0-9_]+", "_", str(value).strip().upper()),
    ).strip("_")


def _graph_digest(facts: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for fact in facts:
        payload = {
            "session_id": str(fact.get("session_id", "")),
            "fact_id": str(fact.get("fact_id", "")),
            "subject": _entity_key(fact.get("subject", "")),
            "object": _entity_key(fact.get("object", "")),
        }
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class PPRIndexManifest:
    schema_version: str
    graph_template_version: str
    source_dense_index_identities: List[Dict[str, Any]]
    graph_digest: str
    fact_count: int
    entity_count: int
    edge_count: int
    built_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def identity(self) -> Dict[str, Any]:
        value = self.to_dict()
        value.pop("built_at_utc", None)
        return value


@dataclass(frozen=True)
class PPRSearchResult:
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class PPRFactIndex:
    """Deterministic entity/fact projection of validated dense-index snapshots."""

    def __init__(
        self,
        *,
        facts: Sequence[Mapping[str, Any]],
        fact_endpoints: Mapping[FactKey, Sequence[str]],
        manifest: PPRIndexManifest,
        build_seconds: float,
    ) -> None:
        self.facts = [dict(fact) for fact in facts]
        self._fact_endpoints = {
            key: tuple(endpoints) for key, endpoints in fact_endpoints.items()
        }
        self.manifest = manifest
        self.build_seconds = float(build_seconds)
        self._facts_by_key = {_fact_key(fact): fact for fact in self.facts}

    @classmethod
    def from_dense_indexes(
        cls,
        dense_indexes: Mapping[str, Any],
        *,
        config: PPRConfig,
    ) -> "PPRFactIndex":
        started = time.perf_counter()
        facts: List[Dict[str, Any]] = []
        identities: List[Dict[str, Any]] = []
        for session_id in sorted(dense_indexes):
            dense_index = dense_indexes[session_id]
            identity = getattr(getattr(dense_index, "manifest", None), "identity", None)
            if not isinstance(identity, dict):
                raise PPRRetrievalError(
                    "ppr_index_mismatch",
                    "dense index identity is unavailable for PPR projection",
                )
            identity_session = identity.get("source_session_id")
            if (
                identity_session is not None
                and str(identity_session) != str(session_id)
            ):
                raise PPRRetrievalError(
                    "ppr_index_mismatch",
                    "dense index identity differs from its PPR source session",
                )
            identities.append(dict(identity))
            for raw_fact in getattr(dense_index, "facts", []):
                fact = dict(raw_fact)
                fact_session = str(fact.get("session_id", ""))
                if fact_session != str(session_id):
                    raise PPRRetrievalError(
                        "ppr_index_mismatch",
                        "dense fact session differs from its PPR source session",
                    )
                facts.append(fact)

        facts.sort(
            key=lambda fact: (
                str(fact.get("session_id", "")),
                str(fact.get("example_id", "")),
                str(fact.get("fact_id", "")),
            )
        )
        keys = [_fact_key(fact) for fact in facts]
        if any(not session or not fact_id for session, fact_id in keys):
            raise PPRRetrievalError(
                "ppr_index_mismatch",
                "PPR facts require non-empty session and fact IDs",
            )
        if len(keys) != len(set(keys)):
            raise PPRRetrievalError(
                "ppr_index_mismatch",
                "PPR facts require unique session/fact identities",
            )

        fact_endpoints = {
            _fact_key(fact): tuple(
                sorted(
                    {
                        entity
                        for entity in (
                            _entity_key(fact.get("subject", "")),
                            _entity_key(fact.get("object", "")),
                        )
                        if entity
                    }
                )
            )
            for fact in facts
        }
        entities = {
            entity
            for endpoints in fact_endpoints.values()
            for entity in endpoints
        }
        edge_count = sum(len(endpoints) for endpoints in fact_endpoints.values())
        manifest = PPRIndexManifest(
            schema_version=PPR_INDEX_SCHEMA_VERSION,
            graph_template_version=config.graph_template_version,
            source_dense_index_identities=identities,
            graph_digest=_graph_digest(facts),
            fact_count=len(facts),
            entity_count=len(entities),
            edge_count=edge_count,
            built_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )
        return cls(
            facts=facts,
            fact_endpoints=fact_endpoints,
            manifest=manifest,
            build_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _in_scope(fact: Mapping[str, Any], scope: Any) -> bool:
        if str(fact.get("session_id", "")) not in {
            str(value) for value in scope.session_ids
        }:
            return False
        return not (
            scope.mode == "example"
            and str(fact.get("example_id", "")) != str(scope.example_id)
        )

    @staticmethod
    def _restart_weights(
        seed_rows: Sequence[Mapping[str, Any]],
        *,
        config: PPRConfig,
    ) -> List[Tuple[FactKey, int, float, float]]:
        eligible: List[Tuple[FactKey, int, float]] = []
        for rank, row in enumerate(seed_rows[: config.seed_count], start=1):
            similarity = float(row.get("_dense_similarity", float("-inf")))
            if math.isfinite(similarity) and similarity >= config.similarity_threshold:
                eligible.append((_fact_key(row), rank, similarity))
        if not eligible:
            return []

        similarities = np.asarray(
            [similarity for _key, _rank, similarity in eligible],
            dtype=np.float64,
        )
        logits = (similarities - float(np.max(similarities))) / config.temperature
        weights = np.exp(logits)
        weights /= float(np.sum(weights))
        return [
            (key, rank, similarity, float(weight))
            for (key, rank, similarity), weight in zip(eligible, weights.tolist())
        ]

    @staticmethod
    def _run_ppr(
        adjacency: Sequence[Sequence[int]],
        restart: np.ndarray,
        *,
        config: PPRConfig,
    ) -> Tuple[np.ndarray, int, float]:
        rank = np.asarray(restart, dtype=np.float64).copy()
        degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.int64)
        final_delta = float("inf")
        for iteration in range(1, config.max_iterations + 1):
            updated = (1.0 - config.damping) * restart
            dangling_mass = float(np.sum(rank[degrees == 0]))
            if dangling_mass:
                updated += config.damping * dangling_mass * restart
            for node, neighbors in enumerate(adjacency):
                if not neighbors:
                    continue
                contribution = config.damping * rank[node] / len(neighbors)
                for neighbor in neighbors:
                    updated[neighbor] += contribution
            final_delta = float(np.sum(np.abs(updated - rank)))
            rank = updated
            if final_delta <= config.tolerance:
                return rank, iteration, final_delta
        raise PPRRetrievalError(
            "ppr_nonconvergence",
            (
                "PPR did not converge within the configured iteration limit "
                f"(delta={final_delta:.6g})"
            ),
        )

    def search(
        self,
        *,
        seed_rows: Sequence[Mapping[str, Any]],
        scope: Any,
        top_k: int,
        config: PPRConfig,
        predicates: Optional[Sequence[str]] = None,
    ) -> PPRSearchResult:
        scoped_facts = [
            fact for fact in self.facts if self._in_scope(fact, scope)
        ]
        scoped_keys = {_fact_key(fact) for fact in scoped_facts}
        for row in seed_rows:
            key = _fact_key(row)
            if key not in self._facts_by_key or key not in scoped_keys:
                raise PPRRetrievalError(
                    "ppr_index_mismatch",
                    "dense seed is absent from the trusted PPR projection",
                )

        weighted_seeds = self._restart_weights(seed_rows, config=config)
        empty_metadata = {
            "settings": config.to_dict(),
            "seed_fact_ids": [],
            "seed_weights": [],
            "iterations": 0,
            "converged": True,
            "final_delta": 0.0,
            "scope_fact_count": len(scoped_facts),
            "scope_entity_count": len(
                {
                    entity
                    for fact in scoped_facts
                    for entity in self._fact_endpoints[_fact_key(fact)]
                }
            ),
            "scope_edge_count": sum(
                len(self._fact_endpoints[_fact_key(fact)])
                for fact in scoped_facts
            ),
            "positive_fact_count": 0,
            "hops_applied": False,
        }
        if not weighted_seeds or top_k < 1:
            return PPRSearchResult(rows=[], metadata=empty_metadata)

        fact_keys = sorted(scoped_keys)
        fact_node = {key: index for index, key in enumerate(fact_keys)}
        entities = sorted(
            {
                entity
                for fact in scoped_facts
                for entity in self._fact_endpoints[_fact_key(fact)]
            }
        )
        entity_node = {
            entity: len(fact_keys) + index for index, entity in enumerate(entities)
        }
        adjacency: List[List[int]] = [
            [] for _ in range(len(fact_keys) + len(entities))
        ]
        edge_count = 0
        for fact in scoped_facts:
            key = _fact_key(fact)
            fact_index = fact_node[key]
            for entity in self._fact_endpoints[key]:
                entity_index = entity_node[entity]
                adjacency[fact_index].append(entity_index)
                adjacency[entity_index].append(fact_index)
                edge_count += 1

        restart = np.zeros(len(adjacency), dtype=np.float64)
        seed_info: Dict[FactKey, Tuple[int, float, float]] = {}
        for key, rank, similarity, weight in weighted_seeds:
            restart[fact_node[key]] = weight
            seed_info[key] = (rank, similarity, weight)
        scores, iterations, final_delta = self._run_ppr(
            adjacency,
            restart,
            config=config,
        )

        wanted = {
            _canonical_predicate(predicate)
            for predicate in predicates or []
            if _canonical_predicate(predicate)
        }
        ranked: List[Tuple[float, FactKey, Dict[str, Any]]] = []
        for key in fact_keys:
            fact = self._facts_by_key[key]
            if wanted and _canonical_predicate(fact.get("predicate", "")) not in wanted:
                continue
            score = float(scores[fact_node[key]])
            if score <= 0.0:
                continue
            value = dict(fact)
            dense = seed_info.get(key)
            value["_ppr_score"] = score
            value["_ppr_rank"] = None
            value["_dense_seed_rank"] = dense[0] if dense else None
            value["_dense_seed_similarity"] = dense[1] if dense else None
            value["_dense_seed_weight"] = dense[2] if dense else None
            value["_was_dense_seed"] = dense is not None
            ranked.append((score, key, value))
        ranked.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))

        rows: List[Dict[str, Any]] = []
        for rank, (_score, _key, row) in enumerate(ranked[:top_k], start=1):
            row["_ppr_rank"] = rank
            rows.append(row)
        metadata = {
            "settings": config.to_dict(),
            "seed_fact_ids": [key[1] for key, _rank, _similarity, _weight in weighted_seeds],
            "seed_weights": [
                {
                    "session_id": key[0],
                    "fact_id": key[1],
                    "dense_rank": rank,
                    "dense_similarity": similarity,
                    "restart_weight": weight,
                }
                for key, rank, similarity, weight in weighted_seeds
            ],
            "iterations": iterations,
            "converged": True,
            "final_delta": final_delta,
            "scope_fact_count": len(scoped_facts),
            "scope_entity_count": len(entities),
            "scope_edge_count": edge_count,
            "positive_fact_count": len(ranked),
            "hops_applied": False,
        }
        return PPRSearchResult(rows=rows, metadata=metadata)
