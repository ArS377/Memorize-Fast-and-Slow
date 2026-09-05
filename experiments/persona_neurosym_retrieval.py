"""Bridge persona memory conditions to Scallop-gated hybrid KG retrieval."""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from neurosym.adapters.dense_index import ensure_dense_index
from neurosym.adapters.graph_source import GraphSource, extract_seed_entities
from neurosym.adapters.neo4j_graph import Neo4jGraph
from neurosym.domain.retrieval_config import EmbeddingConfig, RetrievalConfig


_LATENT_LABEL = re.compile(r"\b(?:history|subject|value|scope|example)-\d+", re.IGNORECASE)


class Validator(Protocol):
    """Scallop-compatible fact admission boundary."""

    def validate(
        self,
        existing_facts: list[dict[str, Any]],
        new_fact: Mapping[str, Any],
    ) -> Any:
        """Return an accept, reject, or replace decision."""


@dataclass(frozen=True)
class HybridMemoryConfig:
    """Pinned controls for persona hybrid fact retrieval."""

    index_root: Path
    embedding_model_id: str
    embedding_model_path: Path
    embedding_revision: str
    embedding_device: str
    embedding_batch_size: int
    top_k: int
    hops: int
    rrf_k: int
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str

    def __post_init__(self) -> None:
        """Reject incomplete retrieval controls."""
        if not all(
            str(value).strip()
            for value in (
                self.embedding_model_id,
                self.embedding_model_path,
                self.embedding_revision,
                self.embedding_device,
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                self.neo4j_database,
            )
        ):
            raise ValueError("hybrid embedding identity and device must be non-empty")
        for name, value in (
            ("embedding_batch_size", self.embedding_batch_size),
            ("top_k", self.top_k),
            ("hops", self.hops),
            ("rrf_k", self.rrf_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hops < 2:
            raise ValueError("hops must be at least 2 for live n-hop retrieval")


def _surface_fact(
    event: Mapping[str, Any], *, session_id: str, example_id: str
) -> dict[str, Any]:
    """Convert one latent event into a model-safe KG fact row."""
    fact = event.get("fact")
    if not isinstance(fact, Mapping):
        raise ValueError(f"event {event.get('event_id')} lacks a fact mapping")
    subject = str(
        event.get("surface_subject")
        or event.get("dialogue_subject")
        or fact.get("subject", "")
    ).strip()
    object_value = str(event.get("surface_object") or fact.get("object", "")).strip()
    support_text = str(
        event.get("retrieval_text")
        or event.get("model_text")
        or fact.get("support_text", "")
    ).strip()
    visible = "\n".join((subject, object_value, support_text))
    if _LATENT_LABEL.search(visible):
        raise ValueError(f"event {event.get('event_id')} exposes a latent label in KG context")
    temporal = fact.get("temporal")
    temporal = temporal if isinstance(temporal, Mapping) else {}
    row = {
        "fact_id": str(fact.get("fact_id", "")),
        "session_id": session_id,
        "example_id": example_id,
        "subject": subject,
        "predicate": str(fact.get("predicate", "")),
        "object": object_value,
        "qualifiers": dict(fact.get("qualifiers", {})),
        "provenance": [],
        "support_text": support_text,
        "confidence": str(fact.get("confidence", "")),
        "confidence_score": fact.get("confidence_score"),
        "valid_from": temporal.get("valid_from"),
        "valid_to": temporal.get("valid_to"),
        "source_event_id": str(event.get("event_id", "")),
    }
    if fact.get("replace_fact_id") is not None:
        row["replace_fact_id"] = str(fact["replace_fact_id"])
    if not row["fact_id"] or not subject or not row["predicate"] or not object_value:
        raise ValueError(f"event {event.get('event_id')} has incomplete KG fact fields")
    return row


def _decision_row(event: Mapping[str, Any], decision: Any) -> dict[str, Any]:
    """Serialize one validator decision without depending on its concrete class."""
    label = getattr(decision, "rejection_label", None)
    if label is not None and hasattr(label, "__dict__"):
        label = dict(label.__dict__)
    return {
        "event_id": str(event.get("event_id", "")),
        "fact_id": str(event.get("fact", {}).get("fact_id", "")),
        "decision": str(getattr(decision, "decision", "")),
        "reason": str(getattr(decision, "reason", "")),
        "replace_fact_id": getattr(decision, "replace_fact_id", None),
        "rejection_label": label,
        "rule_params_version": str(
            getattr(decision, "rule_params_version", "")
        ),
    }


def _validated_prefix(
    events: Sequence[Mapping[str, Any]],
    validator: Validator,
    *,
    session_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay one causal event prefix through Scallop admission."""
    accepted: dict[str, dict[str, Any]] = {}
    lineage_links: dict[str, set[str]] = {}
    ledger = []
    for event in events:
        fact = event.get("fact")
        fact_id = str(fact.get("fact_id", "")) if isinstance(fact, Mapping) else ""
        duplicate_of = str(event.get("duplicate_of", ""))
        if fact_id and duplicate_of:
            lineage_links.setdefault(fact_id, set()).add(duplicate_of)
            lineage_links.setdefault(duplicate_of, set()).add(fact_id)
        if str(event.get("operation", "")) == "retract":
            target = str(event.get("retracts", ""))
            if not target:
                raise ValueError(f"retraction event {event.get('event_id')} lacks a target")
            targets = {target}
            if event.get("retracts_lineage"):
                frontier = [target]
                while frontier:
                    current = frontier.pop()
                    for linked in lineage_links.get(current, set()):
                        if linked not in targets:
                            targets.add(linked)
                            frontier.append(linked)
            removed = [identifier for identifier in targets if accepted.pop(identifier, None)]
            ledger.append(
                {
                    "event_id": str(event.get("event_id", "")),
                    "fact_id": target,
                    "decision": "retract",
                    "reason": "explicit event retraction",
                    "replace_fact_id": None,
                    "rejection_label": None,
                    "rule_params_version": "event-replay.v1",
                    "removed_fact_ids": sorted(removed),
                }
            )
            continue
        candidate = _surface_fact(event, session_id=session_id, example_id="snapshot")
        decision = validator.validate(list(accepted.values()), candidate)
        decision_record = _decision_row(event, decision)
        action = str(getattr(decision, "decision", ""))
        if action == "replace":
            replace_fact_id = str(getattr(decision, "replace_fact_id", ""))
            if not replace_fact_id or replace_fact_id not in accepted:
                raise ValueError(
                    f"validator replacement for {candidate['fact_id']} names an absent fact"
                )
            accepted.pop(replace_fact_id)
            accepted[candidate["fact_id"]] = candidate
        elif action == "accept":
            resolves = event.get("resolves")
            if resolves is None:
                resolved_ids: Sequence[Any] = ()
            elif isinstance(resolves, str):
                resolved_ids = (resolves,)
            elif isinstance(resolves, Sequence):
                resolved_ids = resolves
            else:
                raise ValueError(f"event {event.get('event_id')} has malformed resolves")
            explicit_removals = {
                str(value)
                for value in (
                    event.get("supersedes"),
                    event.get("corrects"),
                    *resolved_ids,
                )
                if value
            }
            transition_from = str(event.get("transitions_from", ""))
            if transition_from:
                predecessor = accepted.get(transition_from)
                if predecessor is None:
                    raise ValueError(
                        f"transition event {event.get('event_id')} names an absent predecessor"
                    )
                predecessor_end = str(predecessor.get("valid_to") or "")
                candidate_start = str(candidate.get("valid_from") or "")
                if not predecessor_end or not candidate_start or predecessor_end >= candidate_start:
                    raise ValueError(
                        f"transition event {event.get('event_id')} must follow a closed "
                        "non-overlapping predecessor interval"
                    )
                decision_record["event_replay_preserved_transition_fact_ids"] = [
                    transition_from
                ]
            removed = [
                identifier
                for identifier in sorted(explicit_removals)
                if accepted.pop(identifier, None) is not None
            ]
            decision_record["event_replay_removed_fact_ids"] = removed
            accepted[candidate["fact_id"]] = candidate
        elif action != "reject":
            raise ValueError(f"validator returned unsupported decision {action!r}")
        ledger.append(decision_record)
    return list(accepted.values()), ledger


def build_validated_condition_facts(
    conditions: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
    validator: Validator,
    *,
    session_id: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Build causal, condition-scoped KG snapshots through Scallop admission."""
    cache: dict[tuple[str, int], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    facts = []
    ledgers = {}
    for condition in conditions:
        evaluation_input_id = str(condition.get("evaluation_input_id", ""))
        history_id = str(condition.get("history_id", ""))
        checkpoint = int(condition.get("checkpoint_turn_index", 0))
        if not evaluation_input_id or not history_id or checkpoint < 1 or checkpoint > len(turns):
            raise ValueError(f"invalid retrieval condition {evaluation_input_id!r}")
        key = (history_id, checkpoint)
        if key not in cache:
            prefix_events = [
                {**dict(turn["stream_event"]), "retrieval_text": str(turn["text"])}
                for turn in turns[:checkpoint]
                if str(turn.get("history_id", "")) == history_id
            ]
            cache[key] = _validated_prefix(
                prefix_events,
                validator,
                session_id=session_id,
            )
        snapshot, ledger = cache[key]
        for fact in snapshot:
            source_fact_id = str(fact["fact_id"])
            digest = hashlib.sha256(
                f"{evaluation_input_id}:{source_fact_id}".encode("utf-8")
            ).hexdigest()
            facts.append(
                {
                    **fact,
                    "fact_id": f"persona-condition-{digest}",
                    "source_fact_id": source_fact_id,
                    "example_id": evaluation_input_id,
                }
            )
        ledgers[evaluation_input_id] = [dict(row) for row in ledger]
    return facts, ledgers


def retrieve_condition_facts(
    conditions: Sequence[Mapping[str, Any]],
    source: GraphSource,
    *,
    top_k: int,
    hops: int,
) -> dict[str, dict[str, Any]]:
    """Run the configured retrieval stack for every visible condition query."""
    if not source.is_live:
        raise ValueError("persona NeuroSym retrieval requires live Neo4j")
    if source.retrieval_config.mode != "hybrid":
        raise ValueError("persona NeuroSym retrieval requires hybrid mode")
    retrieved = {}
    for condition in conditions:
        evaluation_input_id = str(condition.get("evaluation_input_id", ""))
        query = str(condition.get("query_text", ""))
        if not evaluation_input_id or not query:
            raise ValueError("retrieval conditions require IDs and visible query text")
        seeds = extract_seed_entities({"question": query})
        outcome = source.retrieve(
            query=query,
            seed_entities=seeds,
            example_id=evaluation_input_id,
            hops=hops,
            top_k=top_k,
        )
        metadata = dict(outcome.metadata)
        metadata.pop("branch_latency_seconds", None)
        if metadata.get("effective_mode") != "hybrid" or metadata.get("degraded"):
            raise ValueError(
                f"condition {evaluation_input_id} did not execute non-degraded hybrid retrieval"
            )
        branch_counts = metadata.get("branch_counts", {})
        if branch_counts.get("sparse", 0) < 1 or branch_counts.get("dense", 0) < 1:
            raise ValueError(
                f"condition {evaluation_input_id} did not execute both hybrid branches"
            )
        metadata.update(
            {
                "sparse_backend": "neo4j_n_hop",
                "graph_traversal_applied": True,
                "hops_requested": hops,
            }
        )
        retrieved[evaluation_input_id] = {
            "seed_entities": seeds,
            "rows": [dict(row) for row in outcome.rows],
            "metadata": metadata,
        }
    return retrieved


def open_hybrid_graph_source(
    facts: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    validator_url: str,
    config: HybridMemoryConfig,
) -> GraphSource:
    """Persist one authenticated snapshot and build live hybrid retrieval."""
    rows = [dict(fact) for fact in facts]
    fact_ids = [str(row.get("fact_id", "")) for row in rows]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("hybrid condition snapshot fact IDs must be globally unique")
    embedding = EmbeddingConfig(
        model=str(config.embedding_model_path),
        requested_revision=config.embedding_revision,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
    )
    retrieval = RetrievalConfig(
        mode="hybrid",
        rrf_k=config.rrf_k,
        index_root=config.index_root,
        embedding=embedding,
        failure_policy="error",
    )
    graph = Neo4jGraph(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        database=config.neo4j_database,
        session_id=session_id,
        validator_url=validator_url,
        require_scallop=True,
    )
    try:
        graph.clear_session(session_id)
        committed = graph.commit_facts(rows)
        if committed != len(rows):
            raise ValueError(
                f"Neo4j committed {committed} of {len(rows)} authenticated facts"
            )
        persisted = graph.export_facts()
        persisted_ids = {str(row.get("fact_id", "")) for row in persisted}
        if persisted_ids != set(fact_ids):
            raise ValueError("Neo4j persisted fact IDs differ from the authenticated snapshot")
        expected_lineage = {
            str(row["fact_id"]): (
                str(row.get("source_fact_id", "")),
                str(row.get("source_event_id", "")),
            )
            for row in rows
        }
        persisted_lineage = {
            str(row["fact_id"]): (
                str(row.get("source_fact_id", "")),
                str(row.get("source_event_id", "")),
            )
            for row in persisted
        }
        if persisted_lineage != expected_lineage:
            raise ValueError("Neo4j persisted lineage differs from the authenticated snapshot")
        dense_index = ensure_dense_index(
            index_root=config.index_root,
            session_id=session_id,
            facts=rows,
            config=embedding,
        )
        return GraphSource(
            graph=graph,
            fallback_facts=rows,
            session_id=session_id,
            memory_scope="example",
            retrieval_config=retrieval,
            dense_indexes={session_id: dense_index},
        )
    except Exception as error:
        try:
            graph.clear_session(session_id)
        except Exception as cleanup_error:
            error.add_note(f"Neo4j partial-session cleanup failed: {cleanup_error}")
        graph.close()
        raise


def verify_live_n_hop(graph: Neo4jGraph, *, benchmark_session_id: str) -> dict[str, Any]:
    """Prove that the live database reaches a scoped second-hop edge."""
    canary_session = f"{benchmark_session_id}-n-hop-canary"
    example_id = "persona-n-hop-canary"
    digest = hashlib.sha256(benchmark_session_id.encode("utf-8")).hexdigest()[:16]
    start = f"PersonaCanaryStart{digest}"
    middle = f"PersonaCanaryMiddle{digest}"
    end = f"PersonaCanaryEnd{digest}"
    first_id = f"persona-canary-{digest}-first"
    second_id = f"persona-canary-{digest}-second"
    isolation_session = f"{canary_session}-isolation"
    isolation_start = f"PersonaCanaryIsolationStart{digest}"
    isolation_middle = f"PersonaCanaryIsolationMiddle{digest}"
    isolation_end = f"PersonaCanaryIsolationEnd{digest}"
    facts = [
        {
            "fact_id": first_id,
            "example_id": example_id,
            "subject": start,
            "predicate": "CANARY_LINK",
            "object": middle,
            "support_text": "first live traversal edge",
            "provenance": [],
        },
        {
            "fact_id": second_id,
            "example_id": example_id,
            "subject": middle,
            "predicate": "CANARY_LINK",
            "object": end,
            "support_text": "second live traversal edge",
            "provenance": [],
        },
    ]
    graph.clear_session(canary_session)
    graph.clear_session(isolation_session)
    try:
        if graph.commit_facts(facts, session_id=canary_session) != len(facts):
            raise ValueError("Neo4j n-hop canary did not commit both edges")
        one_hop = graph.query_context(
            [start],
            hops=1,
            limit=10,
            example_id=example_id,
            session_id=canary_session,
        )
        two_hop = graph.query_context(
            [start],
            hops=2,
            limit=10,
            example_id=example_id,
            session_id=canary_session,
        )
        one_hop_ids = {str(row.get("fact_id", "")) for row in one_hop}
        two_hop_ids = {str(row.get("fact_id", "")) for row in two_hop}
        if second_id in one_hop_ids or two_hop_ids != {first_id, second_id}:
            raise ValueError("live Neo4j did not demonstrate isolated two-hop traversal")
        isolation_facts = [
            {
                "fact_id": f"persona-canary-{digest}-foreign-bridge",
                "example_id": "foreign-condition",
                "subject": isolation_start,
                "predicate": "CANARY_LINK",
                "object": isolation_middle,
                "support_text": "foreign bridge must not establish reachability",
                "provenance": [],
            },
            {
                "fact_id": f"persona-canary-{digest}-scoped-hidden",
                "example_id": example_id,
                "subject": isolation_middle,
                "predicate": "CANARY_LINK",
                "object": isolation_end,
                "support_text": "scoped edge behind a foreign bridge",
                "provenance": [],
            },
        ]
        if graph.commit_facts(
            isolation_facts, session_id=isolation_session
        ) != len(isolation_facts):
            raise ValueError("Neo4j isolation canary did not commit both edges")
        leaked = graph.query_context(
            [isolation_start],
            hops=2,
            limit=10,
            example_id=example_id,
            session_id=isolation_session,
        )
        if leaked:
            raise ValueError("live Neo4j n-hop traversal crossed condition scope")
        return {
            "status": "passed",
            "one_hop_fact_count": len(one_hop_ids),
            "two_hop_fact_count": len(two_hop_ids),
            "cross_condition_path_fact_count": 0,
        }
    finally:
        graph.clear_session(canary_session)
        graph.clear_session(isolation_session)


def render_retrieved_facts(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render RRF-ranked facts without exposing benchmark identifiers."""
    blocks = []
    for index, row in enumerate(rows, start=1):
        subject = str(row.get("subject", "")).strip()
        predicate = str(row.get("predicate", "")).strip()
        object_value = str(row.get("object", "")).strip()
        support = str(row.get("support_text", "")).strip()
        visible = "\n".join((subject, object_value, support))
        if _LATENT_LABEL.search(visible):
            raise ValueError("retrieved KG context exposes a latent benchmark label")
        block = f"[Memory {index}] {subject} -{predicate}-> {object_value}"
        if support:
            block += f'\nEvidence: "{support}"'
        blocks.append(block)
    return "\n".join(blocks)
