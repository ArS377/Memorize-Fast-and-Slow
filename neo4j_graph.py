#!/usr/bin/env python3
"""
neo4j_graph.py - Dynamic knowledge graph helper for the NeuroSym pipeline.

Adapts the minimal-schema patterns from neo4j-labs/llm-graph-builder to our
verified-fact triples. Designed to be the single source of truth for Cypher
operations and to slot Scallop validation between propose/commit later.

Public API on Neo4jGraph:
    insert_facts(facts)                 # convenience: propose + commit
    propose_facts(facts)                # read-only diff
    commit_facts(facts)                 # idempotent MERGE write
    query_context(seed_entities, ...)   # entity-seeded n-hop neighborhood
    format_context_for_llm(rows, ...)   # deterministic textual block
    find_conflicts(subject, predicate, object_, session_id=None)
    ensure_schema()
    clear_session(session_id, delete_orphans=False)
    clear_all()

Stateless vs stateful:
    Pass session_id to scope writes/queries to a single session. Construct with
    stateless=True to have close() wipe that session_id automatically.

Notes for live data:
    Legacy relationships inserted before the session_id column existed have
    no session_id property. To backfill them so they participate in scoped
    queries, run once:
        MATCH ()-[r]-() WHERE r.session_id IS NULL SET r.session_id = "default"
"""

from __future__ import annotations
from compiled_memory import (
    compiled_memory_to_neo4j_properties,
    fact_to_compiled_memory,
)
from scallop_validator import validate_update

import json
import re
from typing import Any, Dict, List, Optional

try:
    from neo4j import GraphDatabase
except ImportError:  # neo4j is optional until Neo4jGraph() is constructed.
    GraphDatabase = None  # type: ignore[assignment]


Fact = Dict[str, Any]

DEFAULT_SESSION_ID = "default"
MAX_HOPS = 4


def sanitize_predicate(predicate: str) -> str:
    """Make an arbitrary string safe to use as a Neo4j relationship type."""
    pred = (predicate or "").strip().upper()
    pred = re.sub(r"[^A-Z0-9_]+", "_", pred)
    pred = re.sub(r"_+", "_", pred).strip("_")
    return pred or "RELATED_TO"


def _fact_to_params(fact: Fact, session_id: str) -> Dict[str, Any]:
    memory = fact_to_compiled_memory(fact)
    compiled_props = compiled_memory_to_neo4j_properties(memory)
    params = {
        "subject": str(fact["subject"]),
        "object": str(fact["object"]),
        "fact_id": str(fact["fact_id"]),
        "example_id": str(fact.get("example_id", "")),
        "session_id": session_id,
        "question": str(fact.get("question", "")),
        "support_text": str(fact.get("support_text", "")),
        "provenance_json": json.dumps(fact.get("provenance", []), ensure_ascii=False),
        "qualifiers_json": json.dumps(fact.get("qualifiers", {}), ensure_ascii=False),
        "question_relevance": str(fact.get("question_relevance", "")),
        "confidence": str(fact.get("confidence", fact.get("status", "supported"))),
        "normalization_notes": str(fact.get("normalization_notes", "")),
        "verification_reason": str(fact.get("verification_reason", "")),
    }
    params.update(compiled_props)
    # Preserve the public fact_id contract when callers pass an existing ID.
    params["fact_id"] = str(fact["fact_id"])
    return params


class Neo4jGraph:
    """Thin wrapper around the official neo4j driver for our fact schema.

    Args:
        uri: bolt URI, e.g. bolt://localhost:7687
        user: Neo4j user
        password: Neo4j password
        database: optional database name (Neo4j 4+ multi-db / Aura)
        session_id: tag applied to all writes/queries when not overridden.
            None resolves to "default".
        stateless: when True and session_id is set, close() wipes the
            session's relationships before closing the driver.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: Optional[str] = None,
        session_id: Optional[str] = None,
        stateless: bool = False,
    ) -> None:
        if GraphDatabase is None:
            raise RuntimeError("neo4j package not installed. Run: pip install neo4j")
        if not uri:
            raise ValueError("Neo4j URI is required.")
        if not user or not password:
            raise ValueError("Neo4j user and password are required.")
        if stateless and not session_id:
            raise ValueError("stateless=True requires an explicit session_id.")

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self.session_id = session_id or DEFAULT_SESSION_ID
        self.stateless = stateless
        self._schema_ready = False

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> "Neo4jGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._driver is None:
            return
        try:
            if self.stateless:
                try:
                    self.clear_session(self.session_id)
                except Exception:
                    pass
        finally:
            self._driver.close()
            self._driver = None  # type: ignore[assignment]

    def _session(self):
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    # ------------------------------------------------------------------ schema

    SCHEMA_STATEMENTS = [
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE INDEX rel_fact_id IF NOT EXISTS "
        "FOR ()-[r]-() ON (r.fact_id)",
        "CREATE INDEX rel_example_id IF NOT EXISTS "
        "FOR ()-[r]-() ON (r.example_id)",
        "CREATE INDEX rel_session_id IF NOT EXISTS "
        "FOR ()-[r]-() ON (r.session_id)",
        "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS "
        "FOR (e:Entity) ON EACH [e.name]",
    ]

    def ensure_schema(self) -> None:
        """Create constraints + indexes idempotently. Each statement is
        guarded so older Neo4j versions degrade gracefully (e.g., relationship
        property indexes require Neo4j 5.x).
        """
        if self._schema_ready:
            return
        with self._session() as session:
            for stmt in self.SCHEMA_STATEMENTS:
                try:
                    session.run(stmt)
                except Exception:
                    # Logged-skip: best-effort across versions.
                    pass
        self._schema_ready = True

    # ------------------------------------------------------------------ writes

    def commit_facts(self, facts: List[Fact], session_id: Optional[str] = None) -> int:
        """Idempotent MERGE write. Returns the number of facts processed."""
        if not facts:
            return 0
        self.ensure_schema()
        sid = session_id or self.session_id
        count = 0
        with self._session() as session:
            for fact in facts:
                rel_type = sanitize_predicate(str(fact.get("predicate", "")))
                params = _fact_to_params(fact, sid)
                query = (
                    "MERGE (s:Entity {name: $subject})\n"
                    "MERGE (o:Entity {name: $object})\n"
                    f"MERGE (s)-[r:`{rel_type}` "
                    "{fact_id: $fact_id, session_id: $session_id}]->(o)\n"
                    "SET r.example_id = $example_id,\n"
                    "    r.session_id = $session_id,\n"
                    "    r.question = $question,\n"
                    "    r.support_text = $support_text,\n"
                    "    r.provenance_json = $provenance_json,\n"
                    "    r.qualifiers_json = $qualifiers_json,\n"
                    "    r.question_relevance = $question_relevance,\n"
                    "    r.confidence = $confidence,\n"
                    "    r.normalization_notes = $normalization_notes,\n"
                    "    r.verification_reason = $verification_reason,\n"
                    "    r.memory_id = $memory_id,\n"
                    "    r.kind = $kind,\n"
                    "    r.subject_id = $subject_id,\n"
                    "    r.subject_type = $subject_type,\n"
                    "    r.subject_aliases_json = $subject_aliases_json,\n"
                    "    r.object_id = $object_id,\n"
                    "    r.object_type = $object_type,\n"
                    "    r.object_aliases_json = $object_aliases_json,\n"
                    "    r.valid_from = $valid_from,\n"
                    "    r.valid_to = $valid_to,\n"
                    "    r.observed_at = $observed_at,\n"
                    "    r.confidence_level = $confidence_level,\n"
                    "    r.confidence_score = $confidence_score,\n"
                    "    r.confidence_method = $confidence_method,\n"
                    "    r.decision_status = $decision_status,\n"
                    "    r.decision_validator = $decision_validator,\n"
                    "    r.decision_reason = $decision_reason,\n"
                    "    r.replaces_memory_id = $replaces_memory_id,\n"
                    "    r.constraints_json = $constraints_json,\n"
                    "    r.compiled_memory_json = $compiled_memory_json"
                )
                session.run(query, **params)
                count += 1
        return count

    def find_conflicts(
        self,
        subject: str,
        predicate: str,
        object_: str,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return existing facts with same (subject, predicate) but a different
        object. Used by propose_facts() and the contradiction probe.
        """
        rel_type = sanitize_predicate(predicate)
        sid = session_id  # may be None to mean "across all sessions"
        query = (
            "MATCH (s:Entity {name: $subject})-"
            f"[r:`{rel_type}`]->(o:Entity)\n"
            "WHERE ($session_id IS NULL OR r.session_id = $session_id)\n"
            "  AND o.name <> $object\n"
            "RETURN s.name AS subject, type(r) AS predicate, o.name AS object,\n"
            "       r.fact_id AS fact_id, r.session_id AS session_id,\n"
            "       r.example_id AS example_id, r.support_text AS support_text"
        )
        with self._session() as session:
            result = session.run(
                query,
                subject=str(subject),
                object=str(object_),
                session_id=sid,
            )
            return [dict(record) for record in result]

    def _delete_fact(self, fact_id: str, session_id: Optional[str] = None) -> None:
        """Remove a relationship by its composite (fact_id, session_id) identity.

        Used when a higher-confidence fact replaces it. Scoped to one session so
        a replace in one session never deletes the same fact_id in another (the
        same triple in two sessions is two distinct relationships).
        """
        sid = session_id or self.session_id
        query = (
            "MATCH ()-[r]-() WHERE r.fact_id = $fact_id "
            "AND r.session_id = $session_id DELETE r"
        )
        with self._session() as session:
            session.run(query, fact_id=str(fact_id), session_id=sid)

    def _fact_id_exists(self, fact_id: str, session_id: Optional[str] = None) -> bool:
        """True iff this (fact_id, session_id) relationship already exists.

        Scoped to one session so the same fact present in another session is not
        mistaken for an idempotent re-insert (which would drop it from this one).
        """
        sid = session_id or self.session_id
        query = (
            "MATCH ()-[r]-() WHERE r.fact_id = $fact_id "
            "AND r.session_id = $session_id RETURN count(r) AS c"
        )
        with self._session() as session:
            result = session.run(query, fact_id=str(fact_id), session_id=sid)
            record = result.single()
            return bool(record and record.get("c", 0) > 0)

    def propose_facts(
        self,
        facts: List[Fact],
        session_id: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        """Read-only diff. Returns {new, existing, conflicts}.

        - new: facts whose fact_id is not yet in the graph.
        - existing: facts whose fact_id is already present (idempotent re-insert).
        - conflicts: list of {candidate, existing} pairs where the same
          (subject, predicate) has a different object in the graph.
        """
        sid = session_id or self.session_id
        new_facts: List[Fact] = []
        existing_facts: List[Fact] = []
        conflicts: List[Dict[str, Any]] = []
        for fact in facts:
            fact_id = str(fact.get("fact_id", ""))
            if fact_id and self._fact_id_exists(fact_id, session_id=sid):
                existing_facts.append(fact)
            else:
                new_facts.append(fact)
            collisions = self.find_conflicts(
                subject=str(fact.get("subject", "")),
                predicate=str(fact.get("predicate", "")),
                object_=str(fact.get("object", "")),
                session_id=sid,
            )
            for existing in collisions:
                conflicts.append({"candidate": fact, "existing": existing})
        return {"new": new_facts, "existing": existing_facts, "conflicts": conflicts}

    def insert_facts(
        self,
        facts: List[Fact],
        session_id: Optional[str] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Convenience: propose + commit. When validate=True (default), each
        new fact is run through the Scallop validator; when False, the
        validator is skipped and all proposed-new facts are committed directly.
        Returns {"committed", "conflicts"}.
        """
        proposal = self.propose_facts(facts, session_id=session_id)

        if not validate:
            committed = self.commit_facts(proposal["new"], session_id=session_id)
            return {
                "committed": committed,
                "conflicts": proposal["conflicts"],
                "replaced": [],
            }

        # Seed the validator with facts already committed in this batch
        # plus facts whose fact_id already exists in the graph.
        existing_fact_dicts = list(proposal["existing"])

        valid_facts = []
        replaced = []
        for fact in proposal["new"]:
            decision, reason, replace_id = validate_update(existing_fact_dicts, fact)

            if decision == "accept":
                valid_facts.append(fact)
                existing_fact_dicts.append(fact)

            elif decision == "replace":
                # Delete the lower-confidence existing fact, then commit the new one.
                if replace_id:
                    self._delete_fact(replace_id, session_id=session_id)
                    replaced.append({"removed_fact_id": replace_id, "reason": reason})
                    # Remove from local cache so subsequent facts see the updated state.
                    existing_fact_dicts = [
                        e for e in existing_fact_dicts
                        if e.get("fact_id") != replace_id
                    ]
                valid_facts.append(fact)
                existing_fact_dicts.append(fact)

            else:  # reject
                proposal["conflicts"].append({"candidate": fact, "reason": reason})

        committed = self.commit_facts(valid_facts, session_id=session_id)
        return {"committed": committed, "conflicts": proposal["conflicts"], "replaced": replaced}

    # ------------------------------------------------------------------ reads

    def query_context(
        self,
        seed_entities: List[str],
        hops: int = 1,
        limit: int = 50,
        example_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the n-hop neighborhood of the seed entities as triples."""
        if not seed_entities:
            return []
        try:
            hops_int = int(hops)
        except (TypeError, ValueError):
            hops_int = 1
        if hops_int < 1:
            hops_int = 1
        if hops_int > MAX_HOPS:
            hops_int = MAX_HOPS

        try:
            limit_int = int(limit)
        except (TypeError, ValueError):
            limit_int = 50
        if limit_int < 1:
            limit_int = 1

        query = (
            "MATCH (start:Entity) WHERE start.name IN $seed_entities\n"
            f"MATCH path = (start)-[rels*1..{hops_int}]-(neighbor:Entity)\n"
            "UNWIND rels AS r\n"
            "WITH DISTINCT r, startNode(r) AS s, endNode(r) AS o\n"
            "WHERE ($example_id IS NULL OR r.example_id = $example_id)\n"
            "  AND ($session_id IS NULL OR r.session_id = $session_id)\n"
            "RETURN s.name AS subject, type(r) AS predicate, o.name AS object,\n"
            "       r.fact_id AS fact_id, r.example_id AS example_id,\n"
            "       r.session_id AS session_id, r.support_text AS support_text,\n"
            "       r.provenance_json AS provenance_json, r.confidence AS confidence\n"
            f"LIMIT {limit_int}"
        )
        with self._session() as session:
            result = session.run(
                query,
                seed_entities=[str(s) for s in seed_entities],
                example_id=example_id,
                session_id=session_id,
            )
            rows: List[Dict[str, Any]] = []
            for record in result:
                row = dict(record)
                prov_raw = row.get("provenance_json")
                if isinstance(prov_raw, str) and prov_raw:
                    try:
                        row["provenance"] = json.loads(prov_raw)
                    except json.JSONDecodeError:
                        row["provenance"] = []
                else:
                    row["provenance"] = []
                rows.append(row)
            return rows

    # ------------------------------------------------------------------ format

    @staticmethod
    def format_context_for_llm(
        rows: List[Dict[str, Any]],
        max_chars: int = 4000,
    ) -> str:
        """Deterministic textual block of fact rows for LLM ingestion.

        Layout:
            [F1] subject -PREDICATE-> object
                 evidence: "support_text" (example_id, sent_id=...)
        """
        if not rows:
            return ""

        def sort_key(row: Dict[str, Any]):
            return (str(row.get("example_id", "")), str(row.get("fact_id", "")))

        sorted_rows = sorted(rows, key=sort_key)
        lines: List[str] = []
        used = 0
        rendered = 0
        for i, row in enumerate(sorted_rows, start=1):
            subject = str(row.get("subject", ""))
            predicate = str(row.get("predicate", ""))
            obj = str(row.get("object", ""))
            support = str(row.get("support_text", "")).strip()
            example_id = str(row.get("example_id", ""))
            prov = row.get("provenance") or []
            sent_ids = []
            if isinstance(prov, list):
                for p in prov:
                    if isinstance(p, dict) and "sent_id" in p:
                        sent_ids.append(str(p.get("sent_id")))
            sent_part = (
                f"sent_id={','.join(sent_ids)}" if sent_ids else "sent_id=?"
            )
            head = f"[F{i}] {subject} -{predicate}-> {obj}"
            evidence = (
                f"     evidence: \"{support}\" ({example_id}, {sent_part})"
                if support
                else f"     evidence: ({example_id}, {sent_part})"
            )
            block = head + "\n" + evidence
            block_len = len(block) + 1  # for trailing newline
            if used + block_len > max_chars:
                remaining = len(sorted_rows) - rendered
                if remaining > 0:
                    lines.append(f"... [truncated, {remaining} more facts]")
                break
            lines.append(block)
            used += block_len
            rendered += 1
        return "\n".join(lines)

    # ------------------------------------------------------------------ admin

    def clear_session(self, session_id: str, delete_orphans: bool = False) -> None:
        """Delete all relationships tagged with session_id. Optionally remove
        :Entity nodes that become orphaned.
        """
        with self._session() as session:
            session.run(
                "MATCH ()-[r]-() WHERE r.session_id = $session_id DELETE r",
                session_id=str(session_id),
            )
            if delete_orphans:
                session.run(
                    "MATCH (e:Entity) WHERE NOT (e)--() DETACH DELETE e"
                )

    def clear_all(self) -> None:
        """Test helper: nuke everything. Never auto-called."""
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")
