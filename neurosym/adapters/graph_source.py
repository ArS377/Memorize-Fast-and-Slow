"""Graph-context retrieval helpers for KG cells (2, 3, 5, 6).

Encapsulates the "Neo4j primary, facts-file fallback" pattern from
``rlm_graph_baseline.py`` so that cell scripts stay short. Imports
``Neo4jGraph`` lazily so this module can be imported without ``neo4j``
installed (e.g. by the aggregator).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re

from neurosym.domain.compiled_memory import context_sort_key, format_fact_rows_for_llm
from neurosym.domain.retrieval_config import RetrievalConfig
from neurosym.domain.memory_artifacts import MemoryScope
from neurosym.adapters import JsonlFactRepository, canonical_predicate
from neurosym.application.retrieval import CallableRetrievalStrategy, RetrievalService
from neurosym.domain import RetrievalRequest, Scope


_RELEVANCE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_RELEVANCE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "which", "with", "what", "who", "when", "where",
    "why", "how", "does", "do", "did", "can", "could", "would",
}


def _content_tokens(value: Any) -> set[str]:
    return {
        token.lower()
        for token in _RELEVANCE_TOKEN_RE.findall(str(value or ""))
        if len(token) > 1 and token.lower() not in _RELEVANCE_STOPWORDS
    }


def _fact_text(row: Dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key, ""))
        for key in ("subject", "predicate", "object", "support_text")
    )


def _token_overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens.intersection(_content_tokens(text))) / len(query_tokens)


@dataclass(frozen=True)
class ContextRowScore:
    question_overlap: float
    option_margin: float
    retrieval_score: float


def select_context_rows(
    rows: List[Dict[str, Any]],
    *,
    question: str,
    choices: Optional[Dict[str, str]] = None,
    fact_cap: Optional[int] = None,
    min_question_overlap: float = 0.0,
) -> List[Dict[str, Any]]:
    """Gate and rank facts before they are shown as fixed RLM context.

    Scallop validates whether a fact is internally safe to store.  This layer
    answers the different retrieval question: whether a safe fact is useful for
    *this* question.  In MCQ runs, facts that discriminate one option from the
    others are preferred, but the gate itself is based only on the question so
    that an option's wording cannot manufacture relevance.
    """
    if fact_cap is not None and fact_cap < 1:
        raise ValueError("fact_cap must be positive when set")
    if not 0.0 <= min_question_overlap <= 1.0:
        raise ValueError("min_question_overlap must be between 0 and 1")

    question_tokens = _content_tokens(question)
    choice_tokens = {
        label: _content_tokens(text)
        for label, text in (choices or {}).items()
        if str(text).strip()
    }
    scored: List[tuple[ContextRowScore, Dict[str, Any]]] = []
    for row in rows:
        value = dict(row)
        text = _fact_text(value)
        question_overlap = _token_overlap_score(question_tokens, text)
        if question_overlap < min_question_overlap:
            continue
        option_scores = sorted(
            (_token_overlap_score(tokens, text) for tokens in choice_tokens.values()),
            reverse=True,
        )
        option_margin = (
            option_scores[0] - option_scores[1]
            if len(option_scores) > 1
            else (option_scores[0] if option_scores else 0.0)
        )
        retrieval = value.get("_retrieval") or {}
        score = ContextRowScore(
            question_overlap=question_overlap,
            option_margin=option_margin,
            retrieval_score=float(retrieval.get("score", 0.0) or 0.0),
        )
        value["_context_selection"] = {
            "question_overlap": round(question_overlap, 6),
            "option_margin": round(option_margin, 6),
        }
        scored.append((score, value))

    # Preserve the established rank when no new selection policy is requested.
    if not choice_tokens and min_question_overlap == 0.0 and fact_cap is None:
        return [row for _score, row in scored]
    scored.sort(
        key=lambda item: (
            -item[0].question_overlap,
            -item[0].option_margin,
            -item[0].retrieval_score,
            context_sort_key(item[1]),
        )
    )
    selected = [row for _score, row in scored]
    return selected[:fact_cap] if fact_cap is not None else selected


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
    return JsonlFactRepository.from_path(Path(path), sort_key=context_sort_key).facts


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
        canonical_predicate(predicate)
        for predicate in predicates
        if str(predicate).strip()
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
                 dense_failure: Optional[Exception] = None,
                 ppr_index: Optional[Any] = None,
                 ppr_failure: Optional[Exception] = None):
        self.graph = graph
        self.fallback_facts = fallback_facts
        self.session_id = session_id
        self.memory_scope = normalize_memory_scope(memory_scope)
        self.retrieval_config = retrieval_config or RetrievalConfig(mode="sparse")
        self.dense_indexes = dict(dense_indexes or {})
        self.dense_failure = dense_failure
        self.ppr_index = ppr_index
        self.ppr_failure = ppr_failure
        self._retrieval_history: List[Dict[str, Any]] = []
        self._fact_repository = JsonlFactRepository(
            fallback_facts or [], sort_key=context_sort_key
        )
        self._retrieval_service = RetrievalService(
            self.retrieval_config.mode,
            {
                mode: CallableRetrievalStrategy(self._retrieve_with_registered_strategy)
                for mode in ("sparse", "dense", "hybrid", "dense_ppr")
            },
        )
        requested_sessions = [str(s) for s in (source_session_ids or []) if str(s).strip()]
        self.source_session_ids = list(dict.fromkeys(requested_sessions or ([session_id] if session_id else [])))
        if self.memory_scope == "session_set" and len(self.source_session_ids) < 2:
            raise ValueError("session_set memory scope requires at least two trusted source sessions")
        if self.memory_scope != "session_set" and len(self.source_session_ids) > 1:
            raise ValueError(f"{self.memory_scope} memory scope accepts exactly one source session")
        if (
            self.retrieval_config.mode == "dense_ppr"
            and self.ppr_index is None
            and self.ppr_failure is None
            and self.dense_failure is None
        ):
            try:
                from neurosym.adapters.ppr_index import PPRFactIndex

                self.ppr_index = PPRFactIndex.from_dense_indexes(
                    self.dense_indexes,
                    config=self.retrieval_config.ppr,
                )
            except Exception as exc:
                from neurosym.domain.retrieval_config import PPRRetrievalError

                self.ppr_failure = (
                    exc
                    if isinstance(exc, PPRRetrievalError)
                    else PPRRetrievalError(
                        "ppr_index_unavailable",
                        "PPR sidecar initialization failed",
                    )
                )

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
        branches = (
            ("dense", "ppr")
            if configured == "dense_ppr"
            else ("sparse", "dense")
        )
        ppr_identity = (
            self.ppr_index.manifest.identity
            if self.ppr_index is not None
            else None
        )
        if not self._retrieval_history:
            summary = {
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
                "branch_counts": {branch: 0 for branch in branches},
                "branch_fact_ids": {branch: [] for branch in branches},
                "result_fact_ids": [],
                "branch_latency_seconds": {branch: 0.0 for branch in branches},
                "rrf": {
                    "k": self.retrieval_config.rrf_k,
                    "branch_candidate_multiplier": self.retrieval_config.branch_candidate_multiplier,
                    "branch_candidate_cap": self.retrieval_config.branch_candidate_cap,
                },
                "warning": None,
            }
            if configured == "dense_ppr":
                summary.update(
                    {
                        "ppr_index_identity": ppr_identity,
                        "ppr_index_build_seconds": (
                            self.ppr_index.build_seconds
                            if self.ppr_index is not None
                            else None
                        ),
                        "ppr": {"settings": self.retrieval_config.ppr.to_dict()},
                    }
                )
            return summary
        effective = {str(value.get("effective_mode", configured)) for value in self._retrieval_history}
        latency = {
            branch: round(
                sum(float(value.get("branch_latency_seconds", {}).get(branch, 0.0)) for value in self._retrieval_history),
                6,
            )
            for branch in branches
        }
        counts = {
            branch: sum(
                int(value.get("branch_counts", {}).get(branch, 0))
                for value in self._retrieval_history
            )
            for branch in branches
        }
        branch_fact_ids = {
            branch: list(
                dict.fromkeys(
                    str(fact_id)
                    for value in self._retrieval_history
                    for fact_id in value.get("branch_fact_ids", {}).get(branch, [])
                )
            )
            for branch in branches
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

    def _retrieve_with_registered_strategy(self, request: RetrievalRequest):
        from neurosym.application.retrieval_strategies import retrieve

        scope = request.scope
        if scope is None or not scope.example_id:
            example_id = ""
        else:
            example_id = str(scope.example_id)
        return retrieve(
            self,
            query=request.query,
            seed_entities=request.seed_entities,
            example_id=example_id,
            hops=request.hops,
            top_k=request.top_k,
            predicates=request.predicates,
        )

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
        scope = Scope(
            mode=self.memory_scope,
            session_ids=tuple(self.source_session_ids),
            example_id=example_id if self.memory_scope == "example" else None,
        )
        return self._retrieval_service.retrieve(
            RetrievalRequest(
                query=query,
                scope=scope,
                seed_entities=list(seed_entities),
                predicates=list(predicates or []),
                top_k=top_k,
                hops=hops,
            )
        )

    def context_for(
        self,
        ex: Dict[str, Any],
        hops: int = 2,
        limit_triples: int = 50,
        max_chars: int = 4000,
        choices: Optional[Dict[str, str]] = None,
        context_fact_cap: Optional[int] = None,
        min_question_overlap: float = 0.0,
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
        rows = select_context_rows(
            outcome.rows,
            question=query,
            choices=choices,
            fact_cap=context_fact_cap,
            min_question_overlap=min_question_overlap,
        )
        return format_fact_rows_for_llm(rows, max_chars=max_chars), len(rows)

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

        scope = Scope(
            mode=self.memory_scope,
            session_ids=tuple(self.source_session_ids),
            example_id=query_example_id,
        )
        return self._fact_repository.search(
            scope,
            seed_entities=seed_entities,
            predicates=predicates,
            limit=limit_triples,
        )

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
            from neurosym.adapters.neo4j_graph import Neo4jGraph  # local import: optional dep
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
            from neurosym.domain.retrieval_config import DenseRetrievalError

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
            from neurosym.adapters.dense_index import (
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
            from neurosym.domain.retrieval_config import DenseRetrievalError

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
