"""Graphiti write-time inferred-revision baseline for the persona benchmark.

Graphiti (getzep/graphiti; Zep, arXiv 2501.13956) is the external rival that
shares our construct - a bi-temporal KG with edge invalidation - but infers
revision neurally at write time instead of resolving declared lifecycle links
symbolically. This module builds one Graphiti memory store per causal
(history_id, checkpoint_turn_index) pair, queries it with the visible query
text only, and returns rendered retrieval rows for the matched-budget arms.

Protocol invariants enforced here:
- causality: only turns with stream index < checkpoint are ever ingested;
- account scoping: only the queried account's turns are ingested;
- query blindness: ingestion never sees query text, gold, or the family;
- isolation: every pair uses a fresh group namespace, torn down afterwards.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid
from typing import Any, Mapping, Protocol, Sequence

from experiments.persona_neurosym_retrieval import _LATENT_LABEL


EPISODE_VARIANTS = ("e2e", "matched")
RETRIEVAL_STATES = ("point_in_time", "current_only", "unfiltered")
_QUERY_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_GROUP_SAFE = re.compile(r"^[A-Za-z0-9_-]+$")
_SEARCH_RECIPE = "EDGE_HYBRID_SEARCH_RRF (graphiti default; no LLM at query time)"
_ENTITY_EDGE_TYPE = "RELATES_TO"


@dataclass(frozen=True)
class GraphitiMemoryConfig:
    """Pinned controls for the Graphiti external memory baseline."""

    graphiti_core_version: str
    episode_variant: str
    retrieval_state: str
    timestamp_base: str
    timestamp_step_seconds: int
    num_results: int
    llm_max_tokens: int
    max_episode_failures: int
    max_concurrent_histories: int
    embedding_batch_size: int
    build_id: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_small_model: str
    embedding_model_id: str
    embedding_model_path: Path
    embedding_revision: str
    embedding_device: str

    def __post_init__(self) -> None:
        """Reject incomplete or unpinned Graphiti controls."""
        if self.episode_variant not in EPISODE_VARIANTS:
            raise ValueError(
                f"episode_variant must be one of {EPISODE_VARIANTS}, "
                f"got {self.episode_variant!r}"
            )
        if self.retrieval_state not in RETRIEVAL_STATES:
            raise ValueError(
                f"retrieval_state must be one of {RETRIEVAL_STATES}, "
                f"got {self.retrieval_state!r}"
            )
        for name, value in (
            ("graphiti_core_version", self.graphiti_core_version),
            ("build_id", self.build_id),
            ("neo4j_uri", self.neo4j_uri),
            ("neo4j_user", self.neo4j_user),
            ("neo4j_password", self.neo4j_password),
            ("neo4j_database", self.neo4j_database),
            ("llm_base_url", self.llm_base_url),
            ("llm_api_key", self.llm_api_key),
            ("llm_model", self.llm_model),
            ("llm_small_model", self.llm_small_model),
            ("embedding_model_id", self.embedding_model_id),
            ("embedding_revision", self.embedding_revision),
            ("embedding_device", self.embedding_device),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"graphiti {name} must be a non-empty string")
        if not str(self.embedding_model_path).strip():
            raise ValueError("graphiti embedding_model_path must be non-empty")
        if not _GROUP_SAFE.match(self.build_id):
            raise ValueError(
                "graphiti build_id must contain only letters, digits, '_', or '-'"
            )
        for name, value in (
            ("timestamp_step_seconds", self.timestamp_step_seconds),
            ("num_results", self.num_results),
            ("llm_max_tokens", self.llm_max_tokens),
            ("max_concurrent_histories", self.max_concurrent_histories),
            ("embedding_batch_size", self.embedding_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"graphiti {name} must be a positive integer")
        if (
            isinstance(self.max_episode_failures, bool)
            or not isinstance(self.max_episode_failures, int)
            or self.max_episode_failures < 0
        ):
            raise ValueError("graphiti max_episode_failures must be a non-negative integer")
        base = _parse_timestamp_base(self.timestamp_base)
        if base.tzinfo is None:
            raise ValueError("graphiti timestamp_base must carry an explicit timezone")


class GraphitiSession(Protocol):
    """Backend boundary so preparation logic is testable without graphiti-core."""

    async def add_episode(
        self,
        *,
        name: str,
        body: str,
        source: str,
        source_description: str,
        reference_time: datetime,
        group_id: str,
    ) -> None:
        """Ingest one episode into one group namespace."""

    async def search(
        self,
        *,
        query: str,
        group_id: str,
        num_results: int,
        retrieval_state: str,
        as_of: datetime | None,
    ) -> list[dict[str, Any]]:
        """Run the default RRF hybrid search and return normalized edge rows."""

    async def snapshot(self, *, group_id: str) -> dict[str, Any]:
        """Return a canonical serialization of the group's nodes and edges."""

    async def clear_group(self, *, group_id: str) -> None:
        """Delete every node and edge in one group namespace."""

    async def close(self) -> None:
        """Release backend resources."""

    def identity(self) -> dict[str, Any]:
        """Return pinned backend, extractor, and embedder identity."""


def query_as_of_date(query_text: str) -> datetime:
    """Parse the as-of date the visible query asks about.

    The delayed probes are point-in-time questions ("what standing preference
    applied to AsterArc on 2025-08-01?"), and the date is part of the text the
    memory system is allowed to see, so reading it here leaks nothing. Missing
    or ambiguous dates raise rather than silently degrading to a current-state
    filter, which would drop closed-interval facts that are the correct answer.
    """
    matches = _QUERY_DATE.findall(str(query_text))
    if len(set(matches)) != 1:
        raise ValueError(
            f"point-in-time retrieval needs exactly one date in the query, "
            f"found {sorted(set(matches))} in {query_text!r}"
        )
    return datetime.fromisoformat(matches[0]).replace(tzinfo=timezone.utc)


def _parse_timestamp_base(value: str) -> datetime:
    """Parse the configured synthetic-clock origin."""
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"graphiti timestamp_base is not ISO-8601: {value!r}") from error


def reference_time_for_turn(turn_index: int, config: GraphitiMemoryConfig) -> datetime:
    """Map one global stream index onto the synthetic ingestion clock.

    The corpus has no wall clock; in-text dates stay untouched inside episode
    bodies while the synthetic clock preserves stream order for Graphiti's
    bi-temporal bookkeeping.
    """
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError(f"turn_index must be a non-negative integer, got {turn_index!r}")
    return _parse_timestamp_base(config.timestamp_base) + timedelta(
        seconds=turn_index * config.timestamp_step_seconds
    )


def _scope_token(value: str) -> str:
    """Map one latent scope qualifier to a stable, surface-safe opaque token.

    Our admission operator treats scope as an opaque symbol; the matched
    variant grants Graphiti the same information content without leaking
    latent benchmark labels into episode text.
    """
    scope = str(value).strip()
    if not scope:
        return ""
    if not _LATENT_LABEL.search(scope):
        return scope
    digest = hashlib.sha256(f"persona-graphiti-scope:{scope}".encode("utf-8")).hexdigest()
    return f"setting_{digest[:8]}"


def render_episode(
    turn: Mapping[str, Any], *, turn_index: int, variant: str
) -> dict[str, str]:
    """Render one ingestion episode for the configured variant.

    Variant "e2e" ingests the natural dialogue exactly as the other arms see
    it (honest off-the-shelf number; confounds extraction with revision).
    Variant "matched" ingests the latent event's surface-mapped fact fields as
    structured JSON, bypassing extraction so Graphiti's inferred invalidation
    is isolated. Neither variant ever includes lifecycle link identifiers
    (supersedes/corrects/resolves/duplicate_of/transitions_from/retracts) -
    those declared links are exactly what our system consumes and Graphiti's
    write-time LLM must infer.
    """
    if variant not in EPISODE_VARIANTS:
        raise ValueError(f"unsupported episode variant {variant!r}")
    name = f"episode-{turn_index:05d}"
    if variant == "e2e":
        body = "Account holder conversation:\n" + str(turn["text"])
        rendered = {
            "name": name,
            "body": body,
            "source": "message",
            "source_description": "account holder conversation turn",
        }
    else:
        event = turn.get("stream_event")
        if not isinstance(event, Mapping):
            raise ValueError(f"turn {turn.get('turn_id')} lacks a stream event")
        fact = event.get("fact")
        if not isinstance(fact, Mapping):
            raise ValueError(f"event {event.get('event_id')} lacks a fact mapping")
        subject = str(
            event.get("surface_subject")
            or event.get("dialogue_subject")
            or ""
        ).strip()
        value = str(event.get("surface_object") or "").strip()
        if not subject or not value:
            raise ValueError(
                f"event {event.get('event_id')} lacks surface subject or object"
            )
        temporal = fact.get("temporal")
        temporal = temporal if isinstance(temporal, Mapping) else {}
        qualifiers = fact.get("qualifiers")
        qualifiers = qualifiers if isinstance(qualifiers, Mapping) else {}
        record: dict[str, Any] = {
            "subject": subject,
            "predicate": str(fact.get("predicate", "")).strip(),
            "value": value,
            "valid_from": temporal.get("valid_from"),
            "valid_to": temporal.get("valid_to"),
            "scope": _scope_token(str(qualifiers.get("scope", ""))),
            "source_authority": str(qualifiers.get("source_authority", "")).strip(),
        }
        if not record["predicate"]:
            raise ValueError(f"event {event.get('event_id')} lacks a predicate")
        if str(event.get("operation", "")) == "retract":
            record["record"] = "memory_retraction"
            record["statement"] = (
                "The subject withdraws this remembered value and asks that it "
                "be unavailable for future recall."
            )
        else:
            record["record"] = "memory_assertion"
        body = json.dumps(record, sort_keys=True)
        rendered = {
            "name": name,
            "body": body,
            "source": "json",
            "source_description": "structured account holder memory record",
        }
    leak = _LATENT_LABEL.search(rendered["body"])
    if leak:
        raise ValueError(
            f"episode for turn {turn.get('turn_id')} exposes latent label {leak.group(0)!r}"
        )
    return rendered


def render_graphiti_facts(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render ranked Graphiti edges with validity windows, without identifiers."""
    blocks = []
    for index, row in enumerate(rows, start=1):
        fact_text = str(row.get("support_text", "")).strip()
        leak = _LATENT_LABEL.search(fact_text)
        if leak:
            raise ValueError(
                f"retrieved Graphiti fact exposes latent label {leak.group(0)!r}"
            )
        block = f"[Memory {index}] {fact_text}"
        valid_at = row.get("valid_at")
        invalid_at = row.get("invalid_at")
        window = []
        if valid_at:
            window.append(f"valid from {valid_at}")
        if invalid_at:
            window.append(f"invalid from {invalid_at}")
        if window:
            block += "\nValidity: " + "; ".join(window)
        blocks.append(block)
    return "\n".join(blocks)


def extract_valid_fact_set(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the store's currently-valid entity edges as state-fidelity rows.

    The paper's primary metric is state fidelity - the F1 of the valid-fact set
    against ground truth, read off the store rather than off a generation. An
    edge counts as currently valid when neither its world-time invalidation
    (``invalid_at``) nor its transaction-time expiry (``expired_at``) is set;
    those are exactly the edges Graphiti's write-time contradiction handling
    left standing. Episodic ``MENTIONS`` edges are not facts and are excluded.

    Rows are emitted in Graphiti's own vocabulary (LLM-chosen entity names and
    relation names). Aligning that vocabulary to the corpus's canonical fact
    space is the scorer's job, not this module's - see
    docs/persona_graphiti_baseline.md.
    """
    node_names: dict[str, str] = {}
    for node in snapshot.get("nodes", []):
        properties = node.get("properties", {})
        uuid = str(properties.get("uuid", ""))
        if uuid:
            node_names[uuid] = str(properties.get("name", ""))
    rows = []
    for edge in snapshot.get("edges", []):
        if str(edge.get("type", "")) != _ENTITY_EDGE_TYPE:
            continue
        properties = edge.get("properties", {})
        if properties.get("invalid_at") is not None:
            continue
        if properties.get("expired_at") is not None:
            continue
        rows.append(
            {
                "subject": node_names.get(str(edge.get("source_uuid", "")), ""),
                "predicate": str(properties.get("name", "")),
                "object": node_names.get(str(edge.get("target_uuid", "")), ""),
                "fact_text": str(properties.get("fact", "")),
                "valid_at": properties.get("valid_at"),
                "created_at": properties.get("created_at"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["subject"],
            row["predicate"],
            row["object"],
            row["fact_text"],
        ),
    )


def _canonical_sha256(value: Any) -> str:
    """Hash one canonical JSON serialization."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()


def _group_identifier(config: GraphitiMemoryConfig, history_id: str) -> str:
    """Return one namespace-safe group identifier for a whole account stream."""
    group_id = f"{config.build_id}-{history_id}"
    if not _GROUP_SAFE.match(group_id):
        raise ValueError(f"group identifier {group_id!r} contains unsafe characters")
    return group_id


def _history_checkpoints(
    conditions: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, list[Mapping[str, Any]]]]:
    """Group validated conditions by account, then by causal checkpoint."""
    histories: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    for condition in conditions:
        evaluation_input_id = str(condition.get("evaluation_input_id", ""))
        history_id = str(condition.get("history_id", ""))
        checkpoint = int(condition.get("checkpoint_turn_index", 0))
        query_text = str(condition.get("query_text", ""))
        if not evaluation_input_id or not history_id or not query_text:
            raise ValueError(f"invalid graphiti condition {evaluation_input_id!r}")
        if checkpoint < 1 or checkpoint > len(turns):
            raise ValueError(
                f"condition {evaluation_input_id} has invalid checkpoint {checkpoint}"
            )
        histories.setdefault(history_id, {}).setdefault(checkpoint, []).append(condition)
    return histories


def _account_stream(
    turns: Sequence[Mapping[str, Any]], *, history_id: str
) -> list[tuple[int, Mapping[str, Any]]]:
    """Select one account's turns in stream order, carrying global stream indices.

    An account's checkpoints are nested prefixes of this stream, so ingesting it
    once and pausing at each checkpoint reproduces every causal prefix exactly.
    The global index is retained because it drives both the episode name and the
    synthetic reference-time clock.
    """
    selected = [
        (index, turn)
        for index, turn in enumerate(turns)
        if str(turn.get("history_id", "")) == history_id
    ]
    indices = [index for index, _ in selected]
    if indices != sorted(indices):
        raise ValueError("graphiti ingestion must preserve stream order")
    if len(set(indices)) != len(indices):
        raise ValueError("graphiti ingestion selected one turn twice")
    return selected


class _FailureBudget:
    """Shared episode-failure accounting across concurrently ingested accounts.

    Mutation happens between awaits on a single event loop, so the append and the
    limit check are atomic with respect to the other account coroutines.
    """

    def __init__(self, limit: int) -> None:
        self.records: list[dict[str, Any]] = []
        self.exhausted = False
        self._limit = limit

    def record(self, *, group_id: str, turn_id: str, error: Exception) -> None:
        """Record one failed episode and trip the budget when it is exceeded."""
        self.records.append(
            {
                "group_id": group_id,
                "turn_id": turn_id,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        if len(self.records) > self._limit:
            self.exhausted = True
            raise ValueError(
                f"graphiti ingestion exceeded {self._limit} allowed episode "
                f"failures: {self.records}"
            ) from error

    def check(self) -> None:
        """Abort this account early once another account exhausted the budget."""
        if self.exhausted:
            raise _IngestionAborted("graphiti ingestion aborted by failure budget")


class _IngestionAborted(Exception):
    """Raised to unwind sibling accounts after the failure budget is exhausted."""


async def _ingest_history(
    history_id: str,
    checkpoints: Mapping[int, Sequence[Mapping[str, Any]]],
    turns: Sequence[Mapping[str, Any]],
    config: GraphitiMemoryConfig,
    session: GraphitiSession,
    budget: _FailureBudget,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], int]:
    """Ingest one account stream once, snapshotting and querying at each checkpoint.

    The account's checkpoints are nested prefixes, so a single ordered pass that
    pauses at each boundary reproduces every causal prefix without re-ingesting
    shared history. Ingestion stops short of each checkpoint index, which is the
    causality invariant expressed as the loop condition.
    """
    retrievals: dict[str, dict[str, Any]] = {}
    store_states: dict[str, dict[str, Any]] = {}
    group_id = _group_identifier(config, history_id)
    stream = _account_stream(turns, history_id=history_id)
    await session.clear_group(group_id=group_id)
    ingested = 0
    cursor = 0
    try:
        for checkpoint in sorted(checkpoints):
            while cursor < len(stream) and stream[cursor][0] < checkpoint:
                turn_index, turn = stream[cursor]
                budget.check()
                episode = render_episode(
                    turn, turn_index=turn_index, variant=config.episode_variant
                )
                try:
                    await session.add_episode(
                        name=episode["name"],
                        body=episode["body"],
                        source=episode["source"],
                        source_description=episode["source_description"],
                        reference_time=reference_time_for_turn(turn_index, config),
                        group_id=group_id,
                    )
                except Exception as error:
                    budget.record(
                        group_id=group_id,
                        turn_id=str(turn.get("turn_id", "")),
                        error=error,
                    )
                else:
                    ingested += 1
                cursor += 1
            snapshot = await session.snapshot(group_id=group_id)
            store_sha256 = _canonical_sha256(snapshot)
            valid_facts = extract_valid_fact_set(snapshot)
            store_states[f"{history_id}:{checkpoint}"] = {
                "history_id": history_id,
                "checkpoint_turn_index": checkpoint,
                "group_id": group_id,
                "memory_store_sha256": store_sha256,
                "episode_count": ingested,
                "valid_fact_count": len(valid_facts),
                "valid_facts": valid_facts,
            }
            for condition in checkpoints[checkpoint]:
                query_text = str(condition["query_text"])
                as_of = (
                    query_as_of_date(query_text)
                    if config.retrieval_state == "point_in_time"
                    else None
                )
                rows = await session.search(
                    query=query_text,
                    group_id=group_id,
                    num_results=config.num_results,
                    retrieval_state=config.retrieval_state,
                    as_of=as_of,
                )
                retrievals[str(condition["evaluation_input_id"])] = {
                    "seed_entities": [],
                    "rows": [dict(row) for row in rows],
                    "metadata": {
                        "backend": "graphiti_core",
                        "group_id": group_id,
                        "checkpoint_turn_index": checkpoint,
                        "memory_store_sha256": store_sha256,
                        "episode_count": ingested,
                        "episode_failure_count": len(
                            [
                                row
                                for row in budget.records
                                if row["group_id"] == group_id
                            ]
                        ),
                        "search_recipe": _SEARCH_RECIPE,
                        "retrieval_state": config.retrieval_state,
                        "as_of": as_of.isoformat() if as_of is not None else None,
                        "num_results": config.num_results,
                    },
                }
    finally:
        await session.clear_group(group_id=group_id)
    return retrievals, store_states, ingested


async def _prepare_pairs(
    histories: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    turns: Sequence[Mapping[str, Any]],
    config: GraphitiMemoryConfig,
    session: GraphitiSession,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    """Ingest every account stream once and collect checkpoint retrievals."""
    budget = _FailureBudget(config.max_episode_failures)
    semaphore = asyncio.Semaphore(config.max_concurrent_histories)

    async def _run_history(history_id: str) -> tuple[
        dict[str, dict[str, Any]], dict[str, dict[str, Any]], int
    ]:
        async with semaphore:
            return await _ingest_history(
                history_id, histories[history_id], turns, config, session, budget
            )

    results = await asyncio.gather(
        *(_run_history(history_id) for history_id in sorted(histories)),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException) and not isinstance(
            result, _IngestionAborted
        ):
            raise result
    retrievals: dict[str, dict[str, Any]] = {}
    store_states: dict[str, dict[str, Any]] = {}
    store_hashes: dict[str, str] = {}
    episode_total = 0
    for result in results:
        if isinstance(result, BaseException):
            continue
        history_retrievals, history_states, ingested = result
        retrievals.update(history_retrievals)
        store_states.update(history_states)
        episode_total += ingested
    for key, state in store_states.items():
        store_hashes[key] = str(state["memory_store_sha256"])
    failure_records = budget.records
    identity = {
        "backend": "graphiti_core",
        "arm_label": "Graphiti ingestion with benchmark-matched retrieval",
        "deviations_from_default": [
            "embedder replaced with the benchmark's pinned bge revision so the "
            "dense branch matches every other arm",
            "cross-encoder replaced with a stub that raises; the default RRF "
            "search recipe never invokes it, so this only prevents an "
            "accidental call to a remote reranker",
            "search applies an explicit bi-temporal filter; graphiti's default "
            "applies none and would let the generator adjudicate validity",
            "neo4j driver subclassed to pass database_ as the driver control "
            "argument, which graphiti 0.29.3 places in Cypher parameters",
            "llm max_tokens clamped at the client boundary; graphiti's "
            "callers pass their own budget and 'max_tokens or self.max_tokens' "
            "lets it win, so LLMConfig.max_tokens alone is never enforced",
            "query embeddings are encoded with the document path; graphiti's "
            "embedder interface cannot distinguish query from document, so the "
            "query-side instruction our dense branch applies is not used",
            "point-in-time retrieval filters world time only; graphiti derives "
            "expired_at from invalid_at, so requiring expired_at IS NULL would "
            "exclude naturally bounded facts including the gold",
        ],
        "graphiti_core_version": config.graphiti_core_version,
        "episode_variant": config.episode_variant,
        "retrieval_state": config.retrieval_state,
        "ingestion_mode": "incremental_per_account",
        "ingestion_note": (
            "each account stream is ingested once in stream order and snapshotted "
            "at every checkpoint; checkpoints are nested prefixes, so each sees the "
            "same causal content as an isolated rebuild while sharing one store"
        ),
        "history_count": len(histories),
        "max_concurrent_histories": config.max_concurrent_histories,
        "build_id": config.build_id,
        "timestamp_mapping": {
            "base": config.timestamp_base,
            "step_seconds": config.timestamp_step_seconds,
            "indexing": "global_stream_turn_index",
            "in_text_dates": "untouched",
        },
        "search_recipe": _SEARCH_RECIPE,
        "num_results": config.num_results,
        "llm_max_tokens": config.llm_max_tokens,
        "graph_endpoint_database_sha256": _canonical_sha256(
            {"uri": config.neo4j_uri, "database": config.neo4j_database}
        ),
        "database": config.neo4j_database,
        "llm_model": config.llm_model,
        "llm_small_model": config.llm_small_model,
        "llm_base_url_sha256": hashlib.sha256(
            config.llm_base_url.encode("utf-8")
        ).hexdigest(),
        "embedding_model_id": config.embedding_model_id,
        "embedding_revision": config.embedding_revision,
        "embedding_device": config.embedding_device,
        "embedding_batch_size": config.embedding_batch_size,
        "episode_count": episode_total,
        "episode_failure_count": len(failure_records),
        "episode_failures": failure_records,
        "memory_store_sha256": store_hashes,
        "memory_store_combined_sha256": _canonical_sha256(store_hashes),
        "retrieval_map_sha256": _canonical_sha256(retrievals),
        "store_states_sha256": _canonical_sha256(store_states),
        "valid_fact_total": sum(
            int(state["valid_fact_count"]) for state in store_states.values()
        ),
        "session": session.identity(),
    }
    return retrievals, identity, store_states


def prepare_graphiti_memory(
    conditions: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
    config: GraphitiMemoryConfig,
    *,
    session_factory: Any = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    """Build every causal Graphiti store and return retrievals and store state.

    Runs before model load, mirrors ``_prepare_hybrid_memory``: results are
    cached per (history_id, checkpoint_turn_index) pair, keyed back to every
    evaluation_input_id, with store hashes and backend identity for the
    manifest. The third return value is the per-pair valid-fact set that the
    state-fidelity metric scores.
    """
    pairs = _history_checkpoints(conditions, turns)

    async def _run() -> tuple[
        dict[str, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]
    ]:
        factory = session_factory or _open_graphiti_session
        session = factory(config)
        if asyncio.iscoroutine(session):
            session = await session
        try:
            return await _prepare_pairs(pairs, turns, config, session)
        finally:
            await session.close()

    return asyncio.run(_run())


def _open_graphiti_session(config: GraphitiMemoryConfig) -> GraphitiSession:
    """Open one pinned graphiti-core session against its own Neo4j database."""
    return _GraphitiCoreSession(config)


def _routed_neo4j_driver(base_class: Any) -> Any:
    """Build a Neo4jDriver subclass that actually routes to its database.

    graphiti-core 0.29.3 sets ``database_`` inside the Cypher *parameters* map
    rather than passing it as the neo4j driver's control argument, so
    ``execute_query`` (writes, search, our snapshot) lands on the connection's
    default database while ``session``-based work (notably ``clear_data``) uses
    the configured one. Left alone that splits reads/writes from teardown:
    namespaces are never actually cleared, stale edges survive into the next
    build, and the "graphiti gets its own database" isolation claim is false.
    Passing ``database_`` through kwargs restores routing, because the base
    implementation forwards unknown kwargs straight to the neo4j client.
    """

    class _RoutedNeo4jDriver(base_class):
        """Neo4j driver that pins every query to the configured database."""

        async def execute_query(self, cypher_query_: Any, **kwargs: Any) -> Any:
            """Route one query explicitly instead of relying on the default."""
            kwargs.setdefault("database_", self._database)
            return await super().execute_query(cypher_query_, **kwargs)

    return _RoutedNeo4jDriver


class _GraphitiCoreSession:
    """GraphitiSession adapter over pinned graphiti-core with local embeddings."""

    def __init__(self, config: GraphitiMemoryConfig) -> None:
        import importlib.metadata

        resolved_version = importlib.metadata.version("graphiti-core")
        if resolved_version != config.graphiti_core_version:
            raise ValueError(
                f"graphiti-core {resolved_version} is installed but the config "
                f"pins {config.graphiti_core_version}"
            )
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.client import CrossEncoderClient
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        from graphiti_core.embedder.client import EmbedderClient
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.nodes import EpisodeType

        from neurosym.adapters.dense_index import SentenceTransformerEmbedder
        from neurosym.domain.retrieval_config import EmbeddingConfig

        embedder_backend = SentenceTransformerEmbedder(
            EmbeddingConfig(
                model=str(config.embedding_model_path),
                requested_revision=config.embedding_revision,
                device=config.embedding_device,
                batch_size=config.embedding_batch_size,
            ),
            local_files_only=True,
        )

        class _PinnedEmbedder(EmbedderClient):
            """Serve graphiti embeddings from the benchmark's pinned encoder."""

            async def create(self, input_data: Any) -> list[float]:
                if isinstance(input_data, str):
                    texts = [input_data]
                else:
                    texts = [str(item) for item in input_data]
                return embedder_backend.encode_documents(texts)[0].tolist()

            async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
                matrix = embedder_backend.encode_documents(
                    [str(item) for item in input_data_list]
                )
                return [row.tolist() for row in matrix]

        class _ClampedGenericClient(OpenAIGenericClient):
            """Enforce the configured output cap that graphiti otherwise ignores.

            graphiti's extractors call ``generate_response(max_tokens=...)``
            with their own default, and the base client resolves the budget as
            ``max_tokens or self.max_tokens``, so a caller-supplied value always
            wins and ``LLMConfig.max_tokens`` never takes effect. Clamping here
            makes the manifest's recorded limit the limit that actually ran, and
            keeps edge extraction inside a smaller served context window.
            """

            def __init__(self, *args: Any, cap: int, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._output_cap = cap

            async def generate_response(  # type: ignore[override]
                self, *args: Any, max_tokens: Any = None, **kwargs: Any
            ) -> Any:
                requested = max_tokens or self._output_cap
                return await super().generate_response(
                    *args, max_tokens=min(int(requested), self._output_cap), **kwargs
                )

        class _DisabledCrossEncoder(CrossEncoderClient):
            """Refuse reranking so retrieval stays the default RRF recipe."""

            async def rank(
                self, query: str, passages: list[str]
            ) -> list[tuple[str, float]]:
                """Fail loudly if any search path requests cross-encoder reranking."""
                raise RuntimeError(
                    "cross-encoder reranking is disabled for the graphiti baseline; "
                    "only the default RRF search recipe is allowed"
                )

        self._episode_type = EpisodeType
        self._config = config
        self._resolved_version = resolved_version
        self._init_lock = asyncio.Lock()
        self._graphiti = Graphiti(
            llm_client=_ClampedGenericClient(
                cap=config.llm_max_tokens,
                config=LLMConfig(
                    api_key=config.llm_api_key,
                    model=config.llm_model,
                    small_model=config.llm_small_model,
                    base_url=config.llm_base_url,
                    # graphiti defaults max_tokens to 16384, which it sends as the
                    # OUTPUT budget; against a 16384-token served context that
                    # leaves nothing for the prompt and every call 400s.
                    max_tokens=config.llm_max_tokens,
                )
            ),
            embedder=_PinnedEmbedder(),
            cross_encoder=_DisabledCrossEncoder(),
            graph_driver=_routed_neo4j_driver(Neo4jDriver)(
                config.neo4j_uri,
                config.neo4j_user,
                config.neo4j_password,
                database=config.neo4j_database,
            ),
            store_raw_episode_content=True,
        )
        self._indices_ready = False
        self._post_filter_drops = 0

    async def _ensure_indices(self) -> None:
        """Create graph indices and constraints exactly once per session.

        Accounts are ingested concurrently against one session, so the check
        and the initialization must be atomic: without the lock several
        coroutines enter together, and the routing canaries race over a shared
        marker, one deleting another's row and reporting a false mismatch.
        """
        if self._indices_ready:
            return
        async with self._init_lock:
            if self._indices_ready:
                return
            await self._verify_database_routing()
            await self._graphiti.build_indices_and_constraints()
            self._indices_ready = True

    async def _verify_database_routing(self) -> None:
        """Fail loudly unless query and session paths hit the same database.

        Writes a marker through the ``execute_query`` path that carries every
        write and search, then reads it back through the ``session`` path that
        teardown uses. If the two disagree the run would silently write to one
        database and clear another, so abort before any episode is ingested.
        """
        driver = self._graphiti.driver
        # Unique per invocation so concurrent callers can never delete one
        # another's canary row and misread the result as a routing failure.
        marker = f"persona-graphiti-routing-{self._config.build_id}-{uuid.uuid4().hex}"
        await driver.execute_query(
            "CREATE (n:PersonaRoutingCanary {marker: $marker})", marker=marker
        )
        try:
            async with driver.session(database=self._config.neo4j_database) as session:
                result = await session.run(
                    "MATCH (n:PersonaRoutingCanary {marker: $marker}) "
                    "RETURN count(n) AS visible",
                    marker=marker,
                )
                record = await result.single()
                visible = int(record["visible"]) if record is not None else 0
            if visible < 1:
                raise ValueError(
                    "graphiti Neo4j routing mismatch: writes are not landing in "
                    f"database {self._config.neo4j_database!r}, so namespace "
                    "teardown would clear a different database than the one "
                    "being written; refusing to ingest"
                )
        finally:
            await driver.execute_query(
                "MATCH (n:PersonaRoutingCanary {marker: $marker}) DELETE n",
                marker=marker,
            )

    async def add_episode(
        self,
        *,
        name: str,
        body: str,
        source: str,
        source_description: str,
        reference_time: datetime,
        group_id: str,
    ) -> None:
        """Ingest one episode through graphiti-core's extraction pipeline."""
        await self._ensure_indices()
        await self._graphiti.add_episode(
            name=name,
            episode_body=body,
            source_description=source_description,
            reference_time=reference_time,
            source=self._episode_type[source],
            group_id=group_id,
        )

    async def search(
        self,
        *,
        query: str,
        group_id: str,
        num_results: int,
        retrieval_state: str,
        as_of: datetime | None,
    ) -> list[dict[str, Any]]:
        """Run the default RRF hybrid edge search scoped to one group.

        graphiti-core 0.29.3 applies no temporal filter by default, so an
        unfiltered search returns invalidated edges alongside current ones and
        the generator ends up adjudicating validity. The benchmark's delayed
        probes are point-in-time questions, and their gold values include
        closed-interval facts, so filtering to "not invalidated" would discard
        the correct answer. ``point_in_time`` therefore separates world time
        from transaction time: keep edges whose validity window covers the
        as-of date, and independently keep only edges the store still believes.
        """
        await self._ensure_indices()
        from graphiti_core.search.search_filters import (
            ComparisonOperator,
            DateFilter,
            SearchFilters,
        )

        is_null = DateFilter(comparison_operator=ComparisonOperator.is_null)
        search_filter = None
        if retrieval_state == "point_in_time":
            if as_of is None:
                raise ValueError("point-in-time retrieval requires an as-of date")
            search_filter = SearchFilters(
                # World time: the window covers the as-of date. An edge whose
                # date Graphiti could not resolve is kept rather than hidden,
                # so extraction gaps surface as wrong answers instead of
                # silently shrinking the rival's candidate set.
                valid_at=[
                    [is_null],
                    [
                        DateFilter(
                            date=as_of,
                            comparison_operator=ComparisonOperator.less_than_equal,
                        )
                    ],
                ],
                invalid_at=[
                    [is_null],
                    [
                        DateFilter(
                            date=as_of,
                            comparison_operator=ComparisonOperator.greater_than,
                        )
                    ],
                ],
                # No expired_at clause. graphiti sets expired_at = now for ANY
                # edge carrying an invalid_at (edge_operations: "if
                # resolved_edge.invalid_at and not resolved_edge.expired_at"),
                # including naturally bounded facts. Requiring expired_at IS
                # NULL would therefore delete every closed-interval fact -
                # which is exactly the preference_change gold, valid July
                # through September. expired_at carries no independent
                # transaction-time signal here, so world time alone decides.
            )
        elif retrieval_state == "current_only":
            # invalid_at IS NULL already implies expired_at IS NULL for edges
            # graphiti invalidated, so the world-time clause alone is enough.
            search_filter = SearchFilters(invalid_at=[[is_null]])
        edges = await self._graphiti.search(
            query,
            group_ids=[group_id],
            num_results=num_results,
            search_filter=search_filter,
        )
        rows = []
        dropped = 0
        for edge in edges:
            # Do not trust the server-side filter. A committed build returned an
            # edge whose invalid_at equalled the as-of date under a strict
            # "invalid_at > as_of" filter, so the predicate is re-applied here
            # against the values actually returned.
            if retrieval_state == "point_in_time" and as_of is not None:
                valid_at = _as_utc(edge.valid_at)
                invalid_at = _as_utc(edge.invalid_at)
                if valid_at is not None and valid_at > as_of:
                    dropped += 1
                    continue
                if invalid_at is not None and invalid_at <= as_of:
                    dropped += 1
                    continue
            elif retrieval_state == "current_only" and edge.invalid_at is not None:
                dropped += 1
                continue
            rows.append(
                {
                    "fact_id": str(edge.uuid),
                    "predicate": str(edge.name),
                    "object": "",
                    "support_text": str(edge.fact),
                    "valid_at": _isoformat_or_none(edge.valid_at),
                    "invalid_at": _isoformat_or_none(edge.invalid_at),
                    "expired_at": _isoformat_or_none(edge.expired_at),
                    "created_at": _isoformat_or_none(edge.created_at),
                }
            )
        if dropped:
            self._post_filter_drops += dropped
        return rows

    async def snapshot(self, *, group_id: str) -> dict[str, Any]:
        """Serialize the group's nodes and edges with all four timestamps."""
        driver = self._graphiti.driver
        node_result = await driver.execute_query(
            "MATCH (n {group_id: $group_id}) "
            "RETURN labels(n) AS labels, properties(n) AS properties "
            "ORDER BY n.uuid",
            group_id=group_id,
        )
        edge_result = await driver.execute_query(
            "MATCH (a {group_id: $group_id})-[r {group_id: $group_id}]->"
            "(b {group_id: $group_id}) "
            "RETURN type(r) AS type, properties(r) AS properties, "
            "a.uuid AS source_uuid, b.uuid AS target_uuid "
            "ORDER BY r.uuid",
            group_id=group_id,
        )
        return {
            "nodes": [
                {
                    "labels": sorted(record["labels"]),
                    "properties": _jsonable_properties(record["properties"]),
                }
                for record in node_result.records
            ],
            "edges": [
                {
                    "type": str(record["type"]),
                    "source_uuid": str(record["source_uuid"]),
                    "target_uuid": str(record["target_uuid"]),
                    "properties": _jsonable_properties(record["properties"]),
                }
                for record in edge_result.records
            ],
        }

    async def clear_group(self, *, group_id: str) -> None:
        """Delete one group namespace from graphiti's dedicated database."""
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        await clear_data(self._graphiti.driver, [group_id])

    async def close(self) -> None:
        """Close the graphiti driver."""
        await self._graphiti.close()

    def identity(self) -> dict[str, Any]:
        """Return the resolved graphiti-core identity."""
        return {
            "graphiti_core_version": self._resolved_version,
            "server_filter_violations_dropped": self._post_filter_drops,
            "llm_client": "OpenAIGenericClient",
            "embedder": "SentenceTransformerEmbedder(pinned bge revision)",
            "cross_encoder": "disabled",
            "driver": "Neo4jDriver",
        }


def _as_utc(value: Any) -> datetime | None:
    """Coerce a driver datetime to an aware UTC datetime for comparison."""
    if value is None:
        return None
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value))
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _isoformat_or_none(value: Any) -> str | None:
    """Serialize one optional datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _jsonable_properties(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Return sorted JSON-safe properties, excluding embedding vectors."""
    rendered = {}
    for key in sorted(properties):
        if key.endswith("_embedding"):
            continue
        value = properties[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            rendered[key] = value
        elif isinstance(value, (list, tuple)):
            rendered[key] = [str(item) for item in value]
        else:
            rendered[key] = str(value)
    return rendered
