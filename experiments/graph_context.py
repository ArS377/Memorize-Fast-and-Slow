"""Graph-context retrieval helpers for KG cells (2, 3, 5, 6).

Encapsulates the "Neo4j primary, facts-file fallback" pattern from
``rlm_graph_baseline.py`` so that cell scripts stay short. Imports
``Neo4jGraph`` lazily so this module can be imported without ``neo4j``
installed (e.g. by the aggregator).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import re

from compiled_memory import context_sort_key, format_fact_rows_for_llm
from experiments.retrieval_config import RetrievalConfig
from memory_artifacts import MemoryScope


def extract_seed_entities(ex: Dict[str, Any]) -> List[str]:
    """Heuristic seed extraction from a question (mirrors
    ``rlm_graph_baseline.extract_seed_entities`` to avoid an ``rlm`` import).
    """
    question = ex.get("question", "")
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    entities += re.findall(r'"([^"]+)"', question)
    seen, out = set(), []
    for e in entities:
        e = e.strip()
        if len(e) > 2 and e not in seen:
            seen.add(e)
            out.append(e)
    return out[:10]


def load_facts_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                facts.append(json.loads(line))
    return facts


def format_facts_from_jsonl(
    facts: List[Dict[str, Any]],
    example_id: str,
    memory_scope: str = "example",
    seed_entities: Optional[List[str]] = None,
    max_chars: int = 4000,
) -> str:
    """Mirrors ``rlm_graph_baseline.format_facts_from_jsonl``."""
    scope = normalize_memory_scope(memory_scope)
    if scope == "session":
        relevant = list(facts)
        seeds = {str(s).strip().lower() for s in (seed_entities or []) if str(s).strip()}
        if seeds:
            relevant = [
                f for f in relevant
                if str(f.get("subject", "")).strip().lower() in seeds
                or str(f.get("object", "")).strip().lower() in seeds
            ]
    else:
        relevant = [f for f in facts if f.get("example_id") == example_id]
    return format_fact_rows_for_llm(relevant, max_chars=max_chars)


def _row_key(row: Dict[str, Any]) -> str:
    fact_id = str(row.get("fact_id", ""))
    if fact_id:
        return fact_id
    return "|".join(
        str(row.get(k, ""))
        for k in ("example_id", "subject", "predicate", "object", "support_text")
    )


def format_fact_rows(
    rows: List[Dict[str, Any]],
    max_chars: int = 4000,
) -> str:
    """Format already-selected fact rows for LLM ingestion."""
    if not rows:
        return ""
    return format_fact_rows_for_llm(rows, max_chars=max_chars)


def filter_rows_by_predicates(
    rows: List[Dict[str, Any]],
    predicates: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if not predicates:
        return rows
    wanted = {
        re.sub(r"_+", "_", re.sub(r"[^A-Z0-9_]+", "_", str(p).strip().upper())).strip("_")
        for p in predicates
        if str(p).strip()
    }
    if not wanted:
        return rows
    return [r for r in rows if str(r.get("predicate", "")).upper() in wanted]


def normalize_memory_scope(memory_scope: str) -> str:
    scope = str(memory_scope or "example").strip().lower()
    if scope not in {"example", "session", "session_set"}:
        raise ValueError("memory_scope must be 'example', 'session', or 'session_set'")
    return scope


class GraphSource:
    """Either a live Neo4jGraph or a JSONL fact-file fallback."""

    def __init__(self, graph=None, fallback_facts: Optional[List[Dict[str, Any]]] = None,
                 session_id: Optional[str] = None, memory_scope: str = "example",
                 source_session_ids: Optional[List[str]] = None,
                 retrieval_config: Optional[RetrievalConfig] = None,
                 dense_indexes: Optional[Dict[str, Any]] = None,
                 dense_failure: Optional[Exception] = None):
        self.graph = graph
        self.fallback_facts = fallback_facts
        self.session_id = session_id
        self.memory_scope = normalize_memory_scope(memory_scope)
        self.retrieval_config = retrieval_config or RetrievalConfig(mode="sparse")
        self.dense_indexes = dict(dense_indexes or {})
        self.dense_failure = dense_failure
        self._retrieval_history: List[Dict[str, Any]] = []
        requested_sessions = [str(s) for s in (source_session_ids or []) if str(s).strip()]
        self.source_session_ids = list(dict.fromkeys(requested_sessions or ([session_id] if session_id else [])))
        if self.memory_scope == "session_set" and len(self.source_session_ids) < 2:
            raise ValueError("session_set memory scope requires at least two trusted source sessions")
        if self.memory_scope != "session_set" and len(self.source_session_ids) > 1:
            raise ValueError(f"{self.memory_scope} memory scope accepts exactly one source session")

    def trusted_scope(self, example_id: str) -> MemoryScope:
        return MemoryScope(
            mode=self.memory_scope,
            session_ids=tuple(self.source_session_ids),
            example_id=example_id if self.memory_scope == "example" else None,
        )

    @property
    def is_live(self) -> bool:
        return self.graph is not None

    def _record_retrieval_metadata(self, metadata: Dict[str, Any]) -> None:
        self._retrieval_history.append(dict(metadata))

    def reset_retrieval_history(self) -> None:
        self._retrieval_history.clear()

    def retrieval_summary(self) -> Dict[str, Any]:
        configured = self.retrieval_config.mode
        if not self._retrieval_history:
            return {
                "configured_mode": configured,
                # Configuration is not execution evidence.  Keep this unset until
                # retrieve() records an actual branch run so compliance cannot
                # mistake an initialized dense index for a completed retrieval.
                "effective_mode": None,
                "degraded": False,
                "dense_index_identity": [
                    self.dense_indexes[key].manifest.identity
                    for key in sorted(self.dense_indexes)
                ],
                "branch_counts": {"sparse": 0, "dense": 0},
                "branch_fact_ids": {"sparse": [], "dense": []},
                "result_fact_ids": [],
                "branch_latency_seconds": {"sparse": 0.0, "dense": 0.0},
                "rrf": {
                    "k": self.retrieval_config.rrf_k,
                    "branch_candidate_multiplier": self.retrieval_config.branch_candidate_multiplier,
                    "branch_candidate_cap": self.retrieval_config.branch_candidate_cap,
                },
                "warning": None,
            }
        effective = {str(value.get("effective_mode", configured)) for value in self._retrieval_history}
        latency = {
            branch: round(
                sum(float(value.get("branch_latency_seconds", {}).get(branch, 0.0)) for value in self._retrieval_history),
                6,
            )
            for branch in ("sparse", "dense")
        }
        counts = {
            branch: sum(
                int(value.get("branch_counts", {}).get(branch, 0))
                for value in self._retrieval_history
            )
            for branch in ("sparse", "dense")
        }
        branch_fact_ids = {
            branch: list(
                dict.fromkeys(
                    str(fact_id)
                    for value in self._retrieval_history
                    for fact_id in value.get("branch_fact_ids", {}).get(branch, [])
                )
            )
            for branch in ("sparse", "dense")
        }
        result_fact_ids = list(
            dict.fromkeys(
                str(fact_id)
                for value in self._retrieval_history
                for fact_id in value.get("result_fact_ids", [])
            )
        )
        latest = dict(self._retrieval_history[-1])
        latest["effective_mode"] = next(iter(effective)) if len(effective) == 1 else "mixed"
        latest["degraded"] = any(bool(value.get("degraded")) for value in self._retrieval_history)
        latest["branch_counts"] = counts
        latest["branch_fact_ids"] = branch_fact_ids
        latest["result_fact_ids"] = result_fact_ids
        latest["branch_latency_seconds"] = latency
        return latest

    def retrieve(
        self,
        *,
        query: str,
        seed_entities: List[str],
        example_id: str,
        hops: int = 2,
        top_k: int = 10,
        predicates: Optional[List[str]] = None,
    ):
        from experiments.hybrid_retrieval import retrieve

        return retrieve(
            self,
            query=query,
            seed_entities=seed_entities,
            example_id=example_id,
            hops=hops,
            top_k=top_k,
            predicates=predicates,
        )

    def context_for(
        self,
        ex: Dict[str, Any],
        hops: int = 2,
        limit_triples: int = 50,
        max_chars: int = 4000,
    ) -> Tuple[str, int]:
        """Return ``(formatted_context, n_triples)`` for one example."""
        example_id = str(ex.get("_id", ""))
        query = str(ex.get("question", ""))
        seeds = extract_seed_entities(ex)
        # JSONL fallback uses the same scoped row path as the tool adapter.
        outcome = self.retrieve(
            query=query,
            seed_entities=seeds,
            example_id=example_id,
            hops=hops,
            top_k=limit_triples,
        )
        return format_fact_rows_for_llm(outcome.rows, max_chars=max_chars), len(outcome.rows)

    def rows_for(
        self,
        *,
        seed_entities: List[str],
        example_id: str,
        hops: int = 2,
        limit_triples: int = 50,
        predicates: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return raw fact rows for explicit retrieval seeds.

        This is the lower-level hook used by RLM-controlled retrieval. It keeps
        the same example/session isolation as ``context_for`` while allowing a
        planner to choose seeds and optional predicate filters.
        """
        if not seed_entities:
            return []
        query_example_id = example_id if self.memory_scope == "example" else None
        if self.graph is not None:
            rows = self.graph.query_context(
                seed_entities=seed_entities,
                hops=hops,
                limit=limit_triples,
                example_id=query_example_id,
                session_id=self.session_id if self.memory_scope != "session_set" else None,
                session_ids=self.source_session_ids if self.memory_scope == "session_set" else None,
            )
            return filter_rows_by_predicates(rows, predicates)

        facts = self.fallback_facts or []
        seeds_lower = [s.lower() for s in seed_entities if str(s).strip()]
        rows: List[Dict[str, Any]] = []
        for fact in facts:
            fact_session = str(fact.get("session_id", ""))
            if (
                self.memory_scope == "session_set"
                and fact_session
                and fact_session not in self.source_session_ids
            ):
                continue
            if query_example_id is not None and str(fact.get("example_id", "")) != query_example_id:
                continue
            subject = str(fact.get("subject", "")).lower()
            obj = str(fact.get("object", "")).lower()
            if any(seed in subject or seed in obj for seed in seeds_lower):
                rows.append(fact)
        rows = filter_rows_by_predicates(rows, predicates)
        return sorted(rows, key=context_sort_key)[:limit_triples]

    def format_rows(
        self,
        rows: List[Dict[str, Any]],
        max_chars: int = 4000,
    ) -> str:
        return format_fact_rows(rows, max_chars=max_chars)

    def close(self) -> None:
        if self.graph is not None:
            try:
                self.graph.close()
            except Exception:
                pass


def open_graph_source(
    *,
    neo4j_uri: Optional[str],
    neo4j_user: Optional[str],
    neo4j_password: Optional[str],
    session_id: str,
    facts_file: Optional[Path],
    memory_scope: str = "example",
    source_session_ids: Optional[List[str]] = None,
    validator_url: Optional[str] = None,
    require_scallop: bool = False,
    retrieval_config: Optional[RetrievalConfig] = None,
) -> GraphSource:
    """Open a graph source preferring Neo4j; fall back to facts-file.

    Raises ``RuntimeError`` if neither is reachable.
    """
    config = retrieval_config or RetrievalConfig()
    graph = None
    graph_error: Optional[Exception] = None
    if neo4j_uri and neo4j_user and neo4j_password:
        try:
            from neo4j_graph import Neo4jGraph  # local import: optional dep
            graph = Neo4jGraph(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                session_id=session_id,
                validator_url=validator_url,
                require_scallop=require_scallop,
            )
            print(f"Connected to Neo4j at {neo4j_uri} (session={session_id})", file=sys.stderr)
        except Exception as e:
            graph_error = e
            if require_scallop:
                raise RuntimeError(f"required Scallop validator unavailable: {e}") from e
            print(f"Neo4j connection failed: {e}; trying --facts-file", file=sys.stderr)
            graph = None

    facts: Optional[List[Dict[str, Any]]] = None
    dense_source_failure: Optional[Exception] = None
    if facts_file and Path(facts_file).exists():
        facts = load_facts_from_jsonl(Path(facts_file))
        for fact in facts:
            if not fact.get("session_id"):
                fact["session_id"] = session_id
        print(f"Loaded {len(facts)} facts from {facts_file}", file=sys.stderr)
    elif graph is not None and config.mode != "sparse":
        trusted = [str(value) for value in (source_session_ids or []) if str(value).strip()]
        try:
            facts = graph.export_facts(
                session_ids=trusted if memory_scope == "session_set" else None,
                session_id=None if memory_scope == "session_set" else session_id,
            )
        except Exception as exc:
            from experiments.retrieval_config import DenseRetrievalError

            dense_source_failure = DenseRetrievalError(
                "dense_index_unavailable",
                "authoritative fact snapshot could not be exported",
            )
            graph_error = exc

    if graph is None and facts is None:
        raise RuntimeError(
            "No graph source available. Provide either reachable --neo4j-uri/--neo4j-password, "
            f"or --facts-file (e.g. results/kg_builds/{session_id}_facts.jsonl). "
            f"Tip: build the KG first with: "
            f"python -m experiments.build_kg --session {session_id} "
            f"{'--validate' if 'scallop' in session_id else ''} --limit 50"
        ) from graph_error

    trusted_sessions = [str(value) for value in (source_session_ids or []) if str(value).strip()]
    trusted_sessions = trusted_sessions or [session_id]
    dense_indexes: Dict[str, Any] = {}
    dense_failure: Optional[Exception] = None
    if config.mode != "sparse":
        try:
            if dense_source_failure is not None:
                raise dense_source_failure
            from experiments.dense_retrieval import (
                ensure_dense_index,
                ensure_dense_index_from_snapshot,
            )

            available_facts = facts or []
            for trusted_session in trusted_sessions:
                if facts_file and len(trusted_sessions) == 1:
                    dense_indexes[trusted_session] = ensure_dense_index_from_snapshot(
                        Path(facts_file),
                        index_root=config.index_root,
                        session_id=trusted_session,
                        config=config.embedding,
                    )
                    continue
                session_facts = [
                    fact for fact in available_facts
                    if str(fact.get("session_id", trusted_session)) == trusted_session
                ]
                dense_indexes[trusted_session] = ensure_dense_index(
                    index_root=config.index_root,
                    session_id=trusted_session,
                    facts=session_facts,
                    config=config.embedding,
                )
        except Exception as exc:
            from experiments.retrieval_config import DenseRetrievalError

            dense_failure = (
                exc
                if isinstance(exc, DenseRetrievalError)
                else DenseRetrievalError(
                    "dense_backend_failure",
                    "dense retrieval initialization failed",
                )
            )

    return GraphSource(
        graph=graph,
        fallback_facts=facts,
        session_id=session_id,
        memory_scope=memory_scope,
        source_session_ids=source_session_ids,
        retrieval_config=config,
        dense_indexes=dense_indexes,
        dense_failure=dense_failure,
    )
