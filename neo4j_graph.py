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
    facts_temporally_overlap,
    format_fact_rows_for_llm,
)
from scallop_validator import DEFAULT_RULE_PARAMETERS, RuleParameters
from memory_artifacts import MemoryTransitionRecord, WorkingMemoryArtifact
from validator_backend import make_validator_backend

import json
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional

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
    compiled_fact = memory.to_fact()
    compiled_props = compiled_memory_to_neo4j_properties(memory)
    params = {
        "subject": str(compiled_fact["subject"]),
        "object": str(compiled_fact["object"]),
        "fact_id": str(fact["fact_id"]),
        "example_id": str(fact.get("example_id", "")),
        "session_id": session_id,
        "question": str(fact.get("question", "")),
        "support_text": str(compiled_fact.get("support_text", "")),
        "provenance_json": json.dumps(compiled_fact.get("provenance", []), ensure_ascii=False),
        "qualifiers_json": json.dumps(compiled_fact.get("qualifiers", {}), ensure_ascii=False),
        "question_relevance": str(compiled_fact.get("question_relevance", "")),
        "confidence": str(compiled_fact.get("confidence", fact.get("status", "supported"))),
        "normalization_notes": str(compiled_fact.get("normalization_notes", "")),
        "verification_reason": str(compiled_fact.get("verification_reason", "")),
    }
    params.update(compiled_props)
    # Preserve the public fact_id contract when callers pass an existing ID.
    params["fact_id"] = str(fact["fact_id"])
    params["rule_params_version"] = str(fact.get("rule_params_version") or "")
    return params


def _fact_with_decision(
    fact: Fact,
    *,
    status: str,
    validator: str,
    reason: str,
    replaces: Optional[str],
    rule_params_version: Optional[str],
) -> Fact:
    committed = dict(fact)
    decision = dict(committed.get("decision", {})) if isinstance(
        committed.get("decision"), dict
    ) else {}
    decision.update(
        {
            "status": status,
            "validator": validator,
            "reason": reason,
            "replaces": replaces,
        }
    )
    committed["decision"] = decision
    committed["rule_params_version"] = rule_params_version or ""
    return committed


def _json_property(value: Any, fallback: Any) -> Any:
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


def _first_conflicting_fact(existing_facts: List[Fact], candidate: Fact) -> Optional[Fact]:
    subj = str(candidate.get("subject", ""))
    pred = sanitize_predicate(str(candidate.get("predicate", "")))
    obj = str(candidate.get("object", ""))
    for fact in existing_facts:
        fact_subj = str(fact.get("subject", ""))
        fact_pred = sanitize_predicate(str(fact.get("predicate", "")))
        fact_obj = str(fact.get("object", ""))
        if not facts_temporally_overlap(fact, candidate):
            continue
        if fact_subj == subj and fact_pred == pred and fact_obj == obj:
            return fact
        if fact_subj == subj and fact_pred == pred and fact_obj != obj:
            return fact
        if pred == "PART_OF" and fact_pred == "PART_OF" and fact_subj == obj and fact_obj == subj:
            return fact
        if (
            pred == "IS_ALIVE"
            and fact_pred == "IS_ALIVE"
            and fact_subj == subj
            and {fact_obj.lower(), obj.lower()} == {"true", "false"}
        ):
            return fact
    return None


def _record_to_fact(record: Dict[str, Any]) -> Fact:
    """Hydrate a Neo4j relationship row back into the fact dict shape."""
    fact: Fact = {
        "subject": str(record.get("subject", "")),
        "predicate": str(record.get("predicate", "")),
        "object": str(record.get("object", "")),
        "fact_id": str(record.get("fact_id", "")),
        "example_id": str(record.get("example_id", "")),
        "session_id": str(record.get("session_id", "")),
        "question": str(record.get("question", "")),
        "support_text": str(record.get("support_text", "")),
        "provenance": _json_property(record.get("provenance_json"), []),
        "qualifiers": _json_property(record.get("qualifiers_json"), {}),
        "question_relevance": str(record.get("question_relevance", "")),
        "confidence": str(
            record.get("confidence")
            or record.get("confidence_level")
            or record.get("status")
            or "supported"
        ),
        "confidence_score": record.get("confidence_score"),
        "confidence_method": str(record.get("confidence_method", "")),
        "provenance_quality": record.get("provenance_quality"),
        "normalization_notes": str(record.get("normalization_notes", "")),
        "verification_reason": str(record.get("verification_reason", "")),
        "valid_from": str(record.get("valid_from", "")),
        "valid_to": str(record.get("valid_to", "")),
        "observed_at": str(record.get("observed_at", "")),
        "document_id": str(record.get("document_id", "")),
        "extractor_model": str(record.get("extractor_model", "")),
        "verifier_model": str(record.get("verifier_model", "")),
        "run_id": str(record.get("run_id", "")),
        "decision_status": str(record.get("decision_status", "")),
        "decision_validator": str(record.get("decision_validator", "")),
        "decision_reason": str(record.get("decision_reason", "")),
        "rule_params_version": str(record.get("rule_params_version", "")),
    }
    compiled = _json_property(record.get("compiled_memory_json"), None)
    if isinstance(compiled, dict):
        fact["compiled_memory"] = compiled
    return fact


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
        validator_backend=None,
        validator_url: Optional[str] = None,
        require_scallop: bool = False,
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
        self.validator_backend = validator_backend or make_validator_backend(
            endpoint=validator_url,
            require_scallop=require_scallop,
        )
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
        "CREATE CONSTRAINT decision_ledger_id_unique IF NOT EXISTS "
        "FOR (d:DecisionLedger) REQUIRE d.decision_id IS UNIQUE",
        "CREATE INDEX decision_ledger_session_id IF NOT EXISTS "
        "FOR (d:DecisionLedger) ON (d.session_id)",
        "CREATE INDEX decision_ledger_label_code IF NOT EXISTS "
        "FOR (d:DecisionLedger) ON (d.rejection_label_code)",
        "CREATE CONSTRAINT derived_session_id_unique IF NOT EXISTS "
        "FOR (d:DerivedSession) REQUIRE d.session_id IS UNIQUE",
        "CREATE CONSTRAINT working_memory_artifact_id_unique IF NOT EXISTS "
        "FOR (a:WorkingMemoryArtifact) REQUIRE a.artifact_id IS UNIQUE",
        "CREATE CONSTRAINT memory_transition_id_unique IF NOT EXISTS "
        "FOR (t:MemoryTransition) REQUIRE t.transition_id IS UNIQUE",
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
                    "    r.provenance_quality = $provenance_quality,\n"
                    "    r.document_id = $document_id,\n"
                    "    r.extractor_model = $extractor_model,\n"
                    "    r.verifier_model = $verifier_model,\n"
                    "    r.run_id = $run_id,\n"
                    "    r.decision_status = $decision_status,\n"
                    "    r.decision_validator = $decision_validator,\n"
                    "    r.decision_reason = $decision_reason,\n"
                    "    r.rule_params_version = $rule_params_version,\n"
                    "    r.replaces_memory_id = $replaces_memory_id,\n"
                    "    r.constraints_json = $constraints_json,\n"
                    "    r.compiled_memory_json = $compiled_memory_json"
                )
                session.run(query, **params)
                count += 1
        return count

    def reconcile_fact_decisions(self, session_id: Optional[str] = None) -> int:
        """Refresh relationship decision fields from committed audit-ledger rows."""
        sid = session_id or self.session_id
        query = (
            "MATCH (d:DecisionLedger)\n"
            "WHERE d.session_id = $session_id AND d.committed = true\n"
            "  AND d.decision IN ['accept', 'replace']\n"
            "WITH d ORDER BY d.created_at DESC\n"
            "WITH d.candidate_fact_id AS fact_id, collect(d)[0] AS latest\n"
            "MATCH ()-[r]->()\n"
            "WHERE r.session_id = $session_id AND r.fact_id = fact_id\n"
            "SET r.decision_status = latest.decision,\n"
            "    r.decision_validator = $validator,\n"
            "    r.decision_reason = latest.reason,\n"
            "    r.rule_params_version = latest.rule_params_version\n"
            "RETURN count(r) AS reconciled"
        )
        with self._session() as session:
            record = session.run(
                query,
                session_id=sid,
                validator=self.validator_backend.info.name,
            ).single()
        return int(record.get("reconciled", 0) if record else 0)

    def _record_decision_ledger(
        self,
        *,
        candidate: Fact,
        session_id: str,
        decision: str,
        reason: str,
        rule_params: RuleParameters,
        rejection_label: Optional[Dict[str, str]] = None,
        replace_fact_id: Optional[str] = None,
        committed: bool = False,
        derived_from_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist one validation decision, including rejected candidates.

        Accepted facts already live as relationships. Rejected facts do not, so
        this ledger is the auditable source for why an update was denied and
        the exact rule-parameter version that denied it.
        """
        self.ensure_schema()
        label = rejection_label or {}
        entry = {
            "decision_id": str(uuid.uuid4()),
            "session_id": session_id,
            "derived_from_session_id": derived_from_session_id or "",
            "candidate_fact_id": str(candidate.get("fact_id", "")),
            "decision": decision,
            "reason": reason,
            "rejection_label": rejection_label,
            "rejection_label_code": str(label.get("code", "")),
            "rejection_label_category": str(label.get("category", "")),
            "rejection_label_rule_id": str(label.get("rule_id", "")),
            "replace_fact_id": str(replace_fact_id or ""),
            "committed": bool(committed),
            "rule_params_version": rule_params.version,
            "rule_params": rule_params.to_dict(),
            "candidate": dict(candidate),
        }
        params = {
            **entry,
            "rejection_label_json": json.dumps(rejection_label, ensure_ascii=False),
            "rule_params_json": json.dumps(rule_params.to_dict(), ensure_ascii=False),
            "candidate_json": json.dumps(candidate, ensure_ascii=False),
        }
        query = (
            "CREATE (d:DecisionLedger {decision_id: $decision_id})\n"
            "SET d.session_id = $session_id,\n"
            "    d.derived_from_session_id = $derived_from_session_id,\n"
            "    d.candidate_fact_id = $candidate_fact_id,\n"
            "    d.decision = $decision,\n"
            "    d.reason = $reason,\n"
            "    d.rejection_label_code = $rejection_label_code,\n"
            "    d.rejection_label_category = $rejection_label_category,\n"
            "    d.rejection_label_rule_id = $rejection_label_rule_id,\n"
            "    d.rejection_label_json = $rejection_label_json,\n"
            "    d.replace_fact_id = $replace_fact_id,\n"
            "    d.committed = $committed,\n"
            "    d.rule_params_version = $rule_params_version,\n"
            "    d.rule_params_json = $rule_params_json,\n"
            "    d.candidate_json = $candidate_json"
        )
        with self._session() as session:
            session.run(query, **params)
        return entry

    def decision_ledger(
        self,
        session_id: Optional[str] = None,
        decisions: Optional[Iterable[str]] = None,
        rejection_label_codes: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Read ledger entries for approvals, replacements, and rejections."""
        decision_values = list(decisions or [])
        label_values = list(rejection_label_codes or [])
        query = (
            "MATCH (d:DecisionLedger)\n"
            "WHERE ($session_id IS NULL OR d.session_id = $session_id)\n"
            "  AND (size($decisions) = 0 OR d.decision IN $decisions)\n"
            "  AND (size($label_codes) = 0 "
            "OR d.rejection_label_code IN $label_codes)\n"
            "RETURN d.decision_id AS decision_id,\n"
            "       d.session_id AS session_id,\n"
            "       d.derived_from_session_id AS derived_from_session_id,\n"
            "       d.candidate_fact_id AS candidate_fact_id,\n"
            "       d.decision AS decision,\n"
            "       d.reason AS reason,\n"
            "       d.rejection_label_json AS rejection_label_json,\n"
            "       d.replace_fact_id AS replace_fact_id,\n"
            "       d.committed AS committed,\n"
            "       d.rule_params_version AS rule_params_version,\n"
            "       d.rule_params_json AS rule_params_json,\n"
            "       d.candidate_json AS candidate_json"
        )
        sid = session_id if session_id is not None else None
        with self._session() as session:
            result = session.run(
                query,
                session_id=sid,
                decisions=decision_values,
                label_codes=label_values,
            )
            entries: List[Dict[str, Any]] = []
            for record in result:
                row = dict(record)
                row["rejection_label"] = _json_property(
                    row.pop("rejection_label_json", None), None
                )
                row["rule_params"] = _json_property(
                    row.pop("rule_params_json", None), {}
                )
                row["candidate"] = _json_property(row.pop("candidate_json", None), {})
                entries.append(row)
            return entries

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
            "       r.example_id AS example_id, r.support_text AS support_text,\n"
            "       r.provenance_json AS provenance_json,\n"
            "       r.qualifiers_json AS qualifiers_json,\n"
            "       r.question_relevance AS question_relevance,\n"
            "       r.confidence AS confidence,\n"
            "       r.confidence_level AS confidence_level,\n"
            "       r.confidence_score AS confidence_score,\n"
            "       r.confidence_method AS confidence_method,\n"
            "       r.provenance_quality AS provenance_quality,\n"
            "       r.valid_from AS valid_from, r.valid_to AS valid_to,\n"
            "       r.observed_at AS observed_at, r.document_id AS document_id,\n"
            "       r.extractor_model AS extractor_model,\n"
            "       r.verifier_model AS verifier_model, r.run_id AS run_id,\n"
            "       r.compiled_memory_json AS compiled_memory_json"
        )
        with self._session() as session:
            result = session.run(
                query,
                subject=str(subject),
                object=str(object_),
                session_id=sid,
            )
            return [_record_to_fact(dict(record)) for record in result]

    def validation_context_for_facts(
        self,
        facts: List[Fact],
        session_id: Optional[str] = None,
    ) -> List[Fact]:
        """Hydrate committed graph facts relevant to validating candidates.

        Scallop validation needs the persisted session state, not just facts
        accepted earlier in the same batch. This pulls the smallest useful
        state slice for the current rule set:

        - same subject + predicate facts, including exact duplicates and
          functional-predicate rivals
        - inverse PART_OF edges that can create circular containment
        - same-subject IS_ALIVE facts
        """
        if not facts:
            return []

        sid = session_id or self.session_id
        subject_pred_pairs = []
        part_of_subjects = set()
        part_of_objects = set()
        alive_subjects = set()

        for fact in facts:
            subj = str(fact.get("subject", ""))
            pred = sanitize_predicate(str(fact.get("predicate", "")))
            obj = str(fact.get("object", ""))
            if subj and pred:
                subject_pred_pairs.append({"subject": subj, "predicate": pred})
            if pred == "PART_OF" and subj and obj:
                part_of_subjects.add(subj)
                part_of_objects.add(obj)
            if pred == "IS_ALIVE" and subj:
                alive_subjects.add(subj)

        query = (
            "MATCH (s:Entity)-[r]->(o:Entity)\n"
            "WHERE r.session_id = $session_id\n"
            "  AND (\n"
            "    any(pair IN $subject_pred_pairs "
            "WHERE s.name = pair.subject AND type(r) = pair.predicate)\n"
            "    OR (type(r) = 'PART_OF' "
            "AND s.name IN $part_of_objects AND o.name IN $part_of_subjects)\n"
            "    OR (type(r) = 'IS_ALIVE' AND s.name IN $alive_subjects)\n"
            "  )\n"
            "RETURN s.name AS subject, type(r) AS predicate, o.name AS object,\n"
            "       r.fact_id AS fact_id, r.session_id AS session_id,\n"
            "       r.example_id AS example_id, r.question AS question,\n"
            "       r.support_text AS support_text,\n"
            "       r.provenance_json AS provenance_json,\n"
            "       r.qualifiers_json AS qualifiers_json,\n"
            "       r.question_relevance AS question_relevance,\n"
            "       r.confidence AS confidence,\n"
            "       r.confidence_level AS confidence_level,\n"
            "       r.confidence_score AS confidence_score,\n"
            "       r.confidence_method AS confidence_method,\n"
            "       r.provenance_quality AS provenance_quality,\n"
            "       r.normalization_notes AS normalization_notes,\n"
            "       r.verification_reason AS verification_reason,\n"
            "       r.valid_from AS valid_from, r.valid_to AS valid_to,\n"
            "       r.observed_at AS observed_at, r.document_id AS document_id,\n"
            "       r.extractor_model AS extractor_model,\n"
            "       r.verifier_model AS verifier_model, r.run_id AS run_id,\n"
            "       r.compiled_memory_json AS compiled_memory_json"
        )
        with self._session() as session:
            result = session.run(
                query,
                session_id=sid,
                subject_pred_pairs=subject_pred_pairs,
                part_of_subjects=list(part_of_subjects),
                part_of_objects=list(part_of_objects),
                alive_subjects=list(alive_subjects),
            )
            hydrated: List[Fact] = []
            seen = set()
            for record in result:
                fact = _record_to_fact(dict(record))
                key = (
                    fact.get("fact_id"),
                    fact.get("session_id"),
                    fact.get("subject"),
                    fact.get("predicate"),
                    fact.get("object"),
                )
                if key not in seen:
                    seen.add(key)
                    hydrated.append(fact)
            return hydrated

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
        rule_params: Optional[RuleParameters] = None,
        derived_from_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience: propose + commit. When validate=True (default), each
        new fact is run through the Scallop validator; when False, the
        validator is skipped and all proposed-new facts are committed directly.
        Returns {"committed", "conflicts", "replaced", "ledger", "rejected"}.
        """
        params = rule_params or DEFAULT_RULE_PARAMETERS
        sid = session_id or self.session_id
        proposal = self.propose_facts(facts, session_id=session_id)

        if not validate:
            accepted_facts = [
                _fact_with_decision(
                    fact,
                    status="accept",
                    validator="none",
                    reason="Accepted without validation",
                    replaces=None,
                    rule_params_version=None,
                )
                for fact in proposal["new"]
            ]
            committed = self.commit_facts(accepted_facts, session_id=session_id)
            ledger = [
                self._record_decision_ledger(
                    candidate=fact,
                    session_id=sid,
                    decision="accept",
                    reason="Accepted without validation",
                    rule_params=params,
                    committed=True,
                    derived_from_session_id=derived_from_session_id,
                )
                for fact in proposal["new"]
            ]
            return {
                "committed": committed,
                "conflicts": proposal["conflicts"],
                "replaced": [],
                "ledger": ledger,
                "rejected": [],
            }

        # Seed the validator with relevant committed session facts, plus facts
        # accepted earlier in this batch. This makes Scallop gate against the
        # persistent graph state instead of only the current insert batch.
        existing_fact_dicts = self.validation_context_for_facts(
            proposal["new"], session_id=session_id
        )
        known_keys = {
            (
                str(f.get("fact_id", "")),
                str(f.get("session_id", session_id or self.session_id)),
            )
            for f in existing_fact_dicts
        }
        for fact in proposal["existing"]:
            key = (
                str(fact.get("fact_id", "")),
                str(fact.get("session_id", session_id or self.session_id)),
            )
            if key not in known_keys:
                existing_fact_dicts.append(fact)
                known_keys.add(key)

        valid_facts = []
        replaced = []
        ledger = []
        rejected = []
        for fact in proposal["new"]:
            validation = self.validator_backend.validate(
                existing_fact_dicts, fact, rule_params=params
            )
            decision = validation.decision
            reason = validation.reason
            replace_id = validation.replace_fact_id

            if decision == "accept":
                committed_fact = _fact_with_decision(
                    fact,
                    status=decision,
                    validator=self.validator_backend.info.name,
                    reason=reason,
                    replaces=replace_id,
                    rule_params_version=validation.rule_params_version,
                )
                valid_facts.append(committed_fact)
                existing_fact_dicts.append(committed_fact)
                ledger.append(
                    {
                        "candidate": fact,
                        "decision": decision,
                        "reason": reason,
                        "replace_fact_id": replace_id,
                        "rejection_label": None,
                    }
                )

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
                committed_fact = _fact_with_decision(
                    fact,
                    status=decision,
                    validator=self.validator_backend.info.name,
                    reason=reason,
                    replaces=replace_id,
                    rule_params_version=validation.rule_params_version,
                )
                valid_facts.append(committed_fact)
                existing_fact_dicts.append(committed_fact)
                ledger.append(
                    {
                        "candidate": fact,
                        "decision": decision,
                        "reason": reason,
                        "replace_fact_id": replace_id,
                        "rejection_label": None,
                    }
                )

            else:  # reject
                label = (
                    validation.rejection_label.to_dict()
                    if validation.rejection_label
                    else None
                )
                proposal["conflicts"].append({
                    "candidate": fact,
                    "reason": reason,
                    "rejection_label": label,
                    "rule_params_version": validation.rule_params_version,
                })
                existing_conflicting_fact = _first_conflicting_fact(
                    existing_fact_dicts, fact
                )
                rejected.append(
                    {
                        "candidate": fact,
                        "reason": reason,
                        "rule_fired": (label or {}).get("code", "validator_reject"),
                        "rejection_label": label,
                        "rule_params_version": validation.rule_params_version,
                        "existing_conflicting_fact": existing_conflicting_fact,
                    }
                )
                ledger.append(
                    {
                        "candidate": fact,
                        "decision": decision,
                        "reason": reason,
                        "replace_fact_id": replace_id,
                        "rejection_label": label,
                    }
                )

        committed = self.commit_facts(valid_facts, session_id=session_id)
        persisted_ledger = [
            self._record_decision_ledger(
                candidate=entry["candidate"],
                session_id=sid,
                decision=entry["decision"],
                reason=entry["reason"],
                rule_params=params,
                rejection_label=entry["rejection_label"],
                replace_fact_id=entry["replace_fact_id"],
                committed=entry["decision"] in {"accept", "replace"},
                derived_from_session_id=derived_from_session_id,
            )
            for entry in ledger
        ]
        return {
            "committed": committed,
            "conflicts": proposal["conflicts"],
            "replaced": replaced,
            "ledger": persisted_ledger,
            "rejected": rejected,
        }

    def export_facts(
        self,
        session_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
    ) -> List[Fact]:
        requested = [str(value) for value in (session_ids or []) if str(value).strip()]
        if session_id is not None and requested:
            raise ValueError("provide session_id or session_ids, not both")
        trusted_sessions = requested or [str(session_id or self.session_id)]
        where_clause = (
            "WHERE r.session_id IN $session_ids\n"
            if requested
            else "WHERE r.session_id = $session_id\n"
        )
        query = (
            "MATCH (s:Entity)-[r]->(o:Entity)\n"
            + where_clause
            +
            "RETURN s.name AS subject, type(r) AS predicate, o.name AS object,\n"
            "       r.fact_id AS fact_id, r.session_id AS session_id,\n"
            "       r.example_id AS example_id, r.question AS question,\n"
            "       r.support_text AS support_text,\n"
            "       r.provenance_json AS provenance_json,\n"
            "       r.qualifiers_json AS qualifiers_json,\n"
            "       r.question_relevance AS question_relevance,\n"
            "       r.confidence AS confidence,\n"
            "       r.confidence_level AS confidence_level,\n"
            "       r.confidence_score AS confidence_score,\n"
            "       r.confidence_method AS confidence_method,\n"
            "       r.provenance_quality AS provenance_quality,\n"
            "       r.normalization_notes AS normalization_notes,\n"
            "       r.verification_reason AS verification_reason,\n"
            "       r.valid_from AS valid_from, r.valid_to AS valid_to,\n"
            "       r.observed_at AS observed_at, r.document_id AS document_id,\n"
            "       r.extractor_model AS extractor_model,\n"
            "       r.verifier_model AS verifier_model, r.run_id AS run_id,\n"
            "       r.decision_status AS decision_status,\n"
            "       r.decision_validator AS decision_validator,\n"
            "       r.decision_reason AS decision_reason,\n"
            "       r.rule_params_version AS rule_params_version,\n"
            "       r.compiled_memory_json AS compiled_memory_json\n"
            "ORDER BY coalesce(r.session_id, ''), coalesce(r.example_id, ''), "
            "coalesce(r.fact_id, '')"
        )
        with self._session() as session:
            result = session.run(
                query,
                session_ids=trusted_sessions,
                session_id=trusted_sessions[0],
            )
            facts = [_record_to_fact(dict(record)) for record in result]
        return sorted(
            facts,
            key=lambda fact: (
                str(fact.get("session_id", "")),
                str(fact.get("example_id", "")),
                str(fact.get("fact_id", "")),
            ),
        )

    def session_facts(self, session_id: Optional[str] = None) -> List[Fact]:
        """Return all committed facts in one session."""
        return self.export_facts(session_id=session_id)

    def create_derived_session(
        self,
        *,
        source_session_id: str,
        derived_session_id: str,
        rule_params: Optional[RuleParameters] = None,
        include_rejection_labels: Optional[Iterable[str]] = None,
        exclude_rejection_labels: Optional[Iterable[str]] = None,
        copy_accepted: bool = True,
        replay_rejections: bool = True,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Create a controlled learning run from a prior session.

        The derived session can copy prior approvals, then selectively replay
        rejected candidate updates through a chosen rule-parameter version. This
        makes rule evolution auditable: the original rejection stays in the
        source ledger while the derived session records its own new decisions.
        """
        params = rule_params or DEFAULT_RULE_PARAMETERS
        self.ensure_schema()
        include = set(include_rejection_labels or [])
        exclude = set(exclude_rejection_labels or [])

        with self._session() as session:
            session.run(
                "MERGE (d:DerivedSession {session_id: $derived_session_id})\n"
                "SET d.source_session_id = $source_session_id,\n"
                "    d.rule_params_version = $rule_params_version,\n"
                "    d.rule_params_json = $rule_params_json,\n"
                "    d.copy_accepted = $copy_accepted,\n"
                "    d.replay_rejections = $replay_rejections,\n"
                "    d.include_rejection_labels_json = $include_json,\n"
                "    d.exclude_rejection_labels_json = $exclude_json",
                derived_session_id=str(derived_session_id),
                source_session_id=str(source_session_id),
                rule_params_version=params.version,
                rule_params_json=json.dumps(params.to_dict(), ensure_ascii=False),
                copy_accepted=bool(copy_accepted),
                replay_rejections=bool(replay_rejections),
                include_json=json.dumps(sorted(include), ensure_ascii=False),
                exclude_json=json.dumps(sorted(exclude), ensure_ascii=False),
            )

        copied = 0
        if copy_accepted:
            source_facts = self.session_facts(source_session_id)
            copied = self.commit_facts(source_facts, session_id=derived_session_id)

        replay_candidates: List[Fact] = []
        if replay_rejections:
            rejected_entries = self.decision_ledger(
                session_id=source_session_id,
                decisions=["reject"],
            )
            for entry in rejected_entries:
                label = entry.get("rejection_label") or {}
                code = str(label.get("code", ""))
                if include and code not in include:
                    continue
                if exclude and code in exclude:
                    continue
                candidate = entry.get("candidate")
                if isinstance(candidate, dict) and candidate:
                    replay_candidates.append(candidate)

        replay_result = {
            "committed": 0,
            "conflicts": [],
            "replaced": [],
            "ledger": [],
            "rejected": [],
        }
        if replay_candidates:
            replay_result = self.insert_facts(
                replay_candidates,
                session_id=derived_session_id,
                validate=validate,
                rule_params=params,
                derived_from_session_id=source_session_id,
            )

        return {
            "source_session_id": source_session_id,
            "derived_session_id": derived_session_id,
            "rule_params_version": params.version,
            "copied_approvals": copied,
            "replayed_rejections": len(replay_candidates),
            "replay_result": replay_result,
        }

    def persist_working_memory(
        self,
        artifact: WorkingMemoryArtifact,
        transition: MemoryTransitionRecord,
    ) -> Dict[str, Any]:
        """Persist one proposed artifact and its auditable transition.

        Rejected proposals remain as history nodes but never become current
        working memory and never create accepted fact relationships.
        """
        self.ensure_schema()
        artifact_payload = artifact.to_dict()
        transition_payload = transition.to_dict()
        params = {
            "artifact_id": artifact.artifact_id,
            "revision": artifact.revision,
            "status": transition.decision,
            "entity": artifact.entity,
            "scope_mode": artifact.scope.mode,
            "session_ids": list(artifact.scope.session_ids),
            "session_ids_json": json.dumps(list(artifact.scope.session_ids)),
            "example_id": artifact.scope.example_id or "",
            "prior_artifact_id": artifact.prior_artifact_id or "",
            "artifact_json": json.dumps(artifact_payload, ensure_ascii=False),
            "transition_id": transition.transition_id,
            "decision": transition.decision,
            "reason": transition.reason,
            "validator": transition.validator,
            "rule_version": transition.rule_version,
            "committed": transition.committed,
            "before_artifact_id": transition.before_artifact_id or "",
            "after_artifact_id": transition.after_artifact_id or "",
            "transition_json": json.dumps(transition_payload, ensure_ascii=False),
        }
        query = (
            "MERGE (a:WorkingMemoryArtifact {artifact_id: $artifact_id})\n"
            "SET a.revision = $revision, a.status = $status, a.entity = $entity,\n"
            "    a.scope_mode = $scope_mode, a.session_ids = $session_ids,\n"
            "    a.session_ids_json = $session_ids_json,\n"
            "    a.example_id = $example_id, a.prior_artifact_id = $prior_artifact_id,\n"
            "    a.artifact_json = $artifact_json\n"
            "MERGE (t:MemoryTransition {transition_id: $transition_id})\n"
            "SET t.decision = $decision, t.reason = $reason, t.validator = $validator,\n"
            "    t.rule_version = $rule_version, t.committed = $committed,\n"
            "    t.before_artifact_id = $before_artifact_id,\n"
            "    t.after_artifact_id = $after_artifact_id,\n"
            "    t.transition_json = $transition_json\n"
            "MERGE (t)-[:PROPOSED]->(a)\n"
            "FOREACH (_ IN CASE WHEN $committed THEN [1] ELSE [] END |\n"
            "  MERGE (t)-[:COMMITTED]->(a))"
        )
        with self._session() as session:
            session.run(query, **params)
        return {"artifact": artifact_payload, "transition": transition_payload}

    def working_memory_history(
        self,
        *,
        session_ids: Iterable[str],
        example_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sessions = [str(value) for value in session_ids if str(value).strip()]
        query = (
            "MATCH (t:MemoryTransition)-[:PROPOSED]->(a:WorkingMemoryArtifact)\n"
            "WHERE any(sid IN $session_ids WHERE sid IN coalesce(a.session_ids, []))\n"
            "  AND ($example_id IS NULL OR a.example_id = $example_id)\n"
            "RETURN a.artifact_json AS artifact_json, "
            "t.transition_json AS transition_json\n"
            "ORDER BY a.revision, t.transition_id"
        )
        with self._session() as session:
            result = session.run(query, session_ids=sessions, example_id=example_id)
            return [
                {
                    "artifact": _json_property(record.get("artifact_json"), {}),
                    "transition": _json_property(record.get("transition_json"), {}),
                }
                for record in result
            ]

    # ------------------------------------------------------------------ reads

    def query_context(
        self,
        seed_entities: List[str],
        hops: int = 1,
        limit: int = 50,
        example_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
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
            "  AND (size($session_ids) = 0 OR r.session_id IN $session_ids)\n"
            "RETURN s.name AS subject, type(r) AS predicate, o.name AS object,\n"
            "       r.fact_id AS fact_id, r.example_id AS example_id,\n"
            "       r.session_id AS session_id, r.support_text AS support_text,\n"
            "       r.provenance_json AS provenance_json, r.confidence AS confidence,\n"
            "       r.confidence_level AS confidence_level,\n"
            "       r.confidence_score AS confidence_score,\n"
            "       r.confidence_method AS confidence_method,\n"
            "       r.provenance_quality AS provenance_quality,\n"
            "       r.question_relevance AS question_relevance,\n"
            "       r.valid_from AS valid_from, r.valid_to AS valid_to,\n"
            "       r.observed_at AS observed_at, r.document_id AS document_id,\n"
            "       r.extractor_model AS extractor_model,\n"
            "       r.verifier_model AS verifier_model, r.run_id AS run_id,\n"
            "       r.decision_status AS decision_status,\n"
            "       r.decision_validator AS decision_validator,\n"
            "       r.decision_reason AS decision_reason,\n"
            "       r.rule_params_version AS rule_params_version,\n"
            "       r.compiled_memory_json AS compiled_memory_json\n"
            "ORDER BY coalesce(r.confidence_score, 0.0) DESC,\n"
            "         coalesce(r.provenance_quality, 0.0) DESC,\n"
            "         coalesce(r.question_relevance, '') DESC,\n"
            "         coalesce(r.example_id, ''), coalesce(r.fact_id, '')\n"
            f"LIMIT {limit_int}"
        )
        with self._session() as session:
            result = session.run(
                query,
                seed_entities=[str(s) for s in seed_entities],
                example_id=example_id,
                session_id=session_id,
                session_ids=[str(value) for value in (session_ids or [])],
            )
            return [_record_to_fact(dict(record)) for record in result]

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

        return format_fact_rows_for_llm(rows, max_chars=max_chars)

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
