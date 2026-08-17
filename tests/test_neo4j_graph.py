#!/usr/bin/env python3
"""
Tests for neo4j_graph.Neo4jGraph using a mock Neo4j driver.

No live database is required. The mock driver captures every Cypher
statement executed and lets the tests inject canned RETURN rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

# Make the repo root importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

import neurosym.adapters.neo4j_graph as ng
from neurosym.adapters.neo4j_graph import Neo4jGraph
from neurosym.adapters.scallop import validate_update_detailed
from neurosym.domain.validation_rules import RuleParameters
from neurosym.domain.memory_artifacts import MemoryScope, compile_working_memory, transition_for


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "verified_facts.sample.jsonl"


# --------------------------------------------------------------------- mocks


class MockRecord(dict):
    """Behaves like a neo4j.Record for our purposes (dict + get)."""


class MockResult:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows = list(rows or [])

    def __iter__(self):
        return iter(MockRecord(r) for r in self.rows)

    def single(self) -> Optional[MockRecord]:
        if self.rows:
            return MockRecord(self.rows[0])
        return None


class MockSession:
    def __init__(self, driver: "MockDriver") -> None:
        self.driver = driver

    def __enter__(self) -> "MockSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def run(self, query: str, **kwargs):
        self.driver.queries.append((query, kwargs))
        if self.driver.result_provider is not None:
            return self.driver.result_provider(query, kwargs)
        if self.driver.canned_results:
            return self.driver.canned_results.pop(0)
        return MockResult([])

    def close(self) -> None:
        return None


class MockDriver:
    def __init__(self) -> None:
        self.queries: List[tuple] = []
        self.canned_results: List[MockResult] = []
        self.result_provider = None
        self.closed = False

    def session(self, **kwargs):
        return MockSession(self)

    def close(self) -> None:
        self.closed = True


def make_graph(session_id: Optional[str] = None, stateless: bool = False) -> Neo4jGraph:
    """Construct a Neo4jGraph with the real driver factory patched out so
    no socket is opened, then swap a MockDriver into _driver.
    """
    with mock.patch.object(ng, "GraphDatabase") as fake:
        fake.driver.return_value = MockDriver()  # placeholder, immediately replaced
        graph = Neo4jGraph(
            uri="bolt://mock",
            user="u",
            password="p",
            session_id=session_id,
            stateless=stateless,
        )
    graph._driver = MockDriver()
    return graph


def load_fixture() -> List[Dict[str, Any]]:
    assert FIXTURE_PATH.exists(), f"Fixture missing: {FIXTURE_PATH}"
    facts: List[Dict[str, Any]] = []
    for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        facts.append(json.loads(line))
    assert len(facts) >= 8, "Expected at least 8 fixture facts"
    return facts


# --------------------------------------------------------------------- tests


def test_ensure_schema_emits_constraints_and_indexes() -> None:
    graph = make_graph()
    graph.ensure_schema()
    queries = [q for q, _ in graph._driver.queries]
    joined = "\n".join(queries)
    assert "CREATE CONSTRAINT entity_name_unique" in joined
    assert "FOR (e:Entity) REQUIRE e.name IS UNIQUE" in joined
    assert "CREATE INDEX rel_fact_id" in joined
    assert "CREATE INDEX rel_example_id" in joined
    assert "CREATE INDEX rel_session_id" in joined
    assert "CREATE FULLTEXT INDEX entity_name_fulltext" in joined
    # Idempotency: second call should be a no-op.
    before = len(graph._driver.queries)
    graph.ensure_schema()
    assert len(graph._driver.queries) == before
    print("PASS test_ensure_schema_emits_constraints_and_indexes")


def test_commit_facts_emits_expected_cypher_with_session_tag() -> None:
    graph = make_graph(session_id="sess_test")
    facts = load_fixture()
    n = graph.commit_facts(facts)
    assert n == len(facts)
    # Strip out schema queries; only inspect MERGE writes.
    write_queries = [
        (q, p) for q, p in graph._driver.queries if "MERGE (s:Entity" in q
    ]
    assert len(write_queries) == len(facts)
    for (query, params), fact in zip(write_queries, facts):
        rel_type = ng.sanitize_predicate(fact["predicate"])
        assert "MERGE (s:Entity {name: $subject})" in query
        assert "MERGE (o:Entity {name: $object})" in query
        # Composite (fact_id, session_id) identity prevents one session from
        # overwriting another's relationship (session-isolation fix).
        assert f"[r:`{rel_type}` {{fact_id: $fact_id, session_id: $session_id}}]" in query
        assert "r.session_id = $session_id" in query
        assert "r.example_id = $example_id" in query
        assert "r.support_text = $support_text" in query
        assert "r.provenance_json = $provenance_json" in query
        assert "r.confidence_score = $confidence_score" in query
        assert "r.provenance_quality = $provenance_quality" in query
        assert "r.document_id = $document_id" in query
        assert "r.extractor_model = $extractor_model" in query
        assert "r.verifier_model = $verifier_model" in query
        assert "r.run_id = $run_id" in query
        assert "r.rule_params_version = $rule_params_version" in query
        assert params["session_id"] == "sess_test"
        assert params["fact_id"] == fact["fact_id"]
        assert params["subject"] == fact["subject"]
        assert params["object"] == fact["object"]
        provenance = json.loads(params["provenance_json"])
        assert provenance[0]["title"] == fact["provenance"][0]["title"]
        assert provenance[0]["sent_id"] == fact["provenance"][0]["sent_id"]
        assert "source_id" in provenance[0]
        assert isinstance(params["confidence_score"], float)
        assert isinstance(params["provenance_quality"], float)
    print("PASS test_commit_facts_emits_expected_cypher_with_session_tag")


def test_propose_facts_classifies_new_existing_conflict() -> None:
    graph = make_graph(session_id="sess_test")
    facts = load_fixture()
    # Pick three representative facts:
    new_fact = facts[0]                     # Kalamang SPOKEN_IN East Indonesia
    existing_fact = facts[1]                # Kalamang HAS_ISO_CODE kgv
    candidate_conflict = facts[6]           # Indonesia CAPITAL_IS Jakarta
    rival = facts[7]                        # Indonesia CAPITAL_IS Bandung (the "existing" one)

    candidates = [new_fact, existing_fact, candidate_conflict]

    def provider(query: str, params: Dict[str, Any]):
        # _fact_id_exists -> count(r) result
        if "RETURN count(r)" in query:
            fid = params.get("fact_id")
            return MockResult([{"c": 1 if fid == existing_fact["fact_id"] else 0}])
        # find_conflicts -> rows where Indonesia/CAPITAL_IS already maps to Bandung
        if "AND o.name <> $object" in query:
            if (
                params.get("subject") == candidate_conflict["subject"]
                and params.get("object") == candidate_conflict["object"]
            ):
                return MockResult([
                    {
                        "subject": rival["subject"],
                        "predicate": ng.sanitize_predicate(rival["predicate"]),
                        "object": rival["object"],
                        "fact_id": rival["fact_id"],
                        "session_id": "sess_test",
                        "example_id": rival["example_id"],
                        "support_text": rival["support_text"],
                    }
                ])
            return MockResult([])
        return MockResult([])

    graph._driver.result_provider = provider
    proposal = graph.propose_facts(candidates)

    assert [f["fact_id"] for f in proposal["new"]] == [
        new_fact["fact_id"],
        candidate_conflict["fact_id"],
    ]
    assert [f["fact_id"] for f in proposal["existing"]] == [existing_fact["fact_id"]]
    assert len(proposal["conflicts"]) == 1
    conflict = proposal["conflicts"][0]
    assert conflict["candidate"]["fact_id"] == candidate_conflict["fact_id"]
    assert conflict["existing"]["object"] == rival["object"]
    print("PASS test_propose_facts_classifies_new_existing_conflict")


def test_find_conflicts_filters_relationship_type_as_data() -> None:
    graph = make_graph(session_id="sess_test")

    assert graph.find_conflicts("A", "new relation", "B") == []

    query, params = graph._driver.queries[-1]
    assert "MATCH (s:Entity {name: $subject})-[r]->(o:Entity)" in query
    assert "type(r) = $predicate" in query
    assert "[r:`NEW_RELATION`]" not in query
    assert params["predicate"] == "NEW_RELATION"


def test_insert_facts_idempotent_on_repeated_call() -> None:
    graph = make_graph(session_id="sess_test")
    facts = load_fixture()[:3]

    # propose_facts will issue lookups; treat all facts as new with no conflicts.
    def provider(query: str, params: Dict[str, Any]):
        if "RETURN count(r)" in query:
            return MockResult([{"c": 0}])
        if "AND o.name <> $object" in query:
            return MockResult([])
        return MockResult([])

    graph._driver.result_provider = provider
    first = graph.insert_facts(facts)
    second = graph.insert_facts(facts)
    assert first["committed"] == len(facts)
    assert second["committed"] == len(facts)

    # Both calls should emit the same MERGE-with-fact_id pattern per fact.
    write_queries = [(q, p) for q, p in graph._driver.queries if "MERGE (s:Entity" in q]
    assert len(write_queries) == 2 * len(facts)
    half = len(facts)
    for i in range(half):
        q1, p1 = write_queries[i]
        q2, p2 = write_queries[i + half]
        assert q1 == q2
        assert p1["fact_id"] == p2["fact_id"]
        assert p1["session_id"] == p2["session_id"] == "sess_test"
    print("PASS test_insert_facts_idempotent_on_repeated_call")


def test_same_fact_id_coexists_across_sessions() -> None:
    graph = make_graph(session_id="session_a")
    fact = load_fixture()[0]
    stored = set()

    def provider(query: str, params: Dict[str, Any]):
        key = (params.get("fact_id"), params.get("session_id"))
        if "RETURN count(r)" in query:
            return MockResult([{"c": int(key in stored)}])
        if "MERGE (s:Entity" in query:
            stored.add(key)
        elif "DELETE r" in query:
            stored.discard(key)
        return MockResult([])

    graph._driver.result_provider = provider

    first = graph.insert_facts([fact], session_id="session_a", validate=False)
    second = graph.insert_facts([fact], session_id="session_b", validate=False)

    assert first["committed"] == 1
    assert second["committed"] == 1
    assert (fact["fact_id"], "session_a") in stored
    assert (fact["fact_id"], "session_b") in stored

    graph._delete_fact(fact["fact_id"], session_id="session_a")
    assert (fact["fact_id"], "session_a") not in stored
    assert (fact["fact_id"], "session_b") in stored
    print("PASS test_same_fact_id_coexists_across_sessions")


def _stored_row(fact: Dict[str, Any], session_id: str = "sess_test") -> Dict[str, Any]:
    return {
        "subject": fact["subject"],
        "predicate": ng.sanitize_predicate(fact["predicate"]),
        "object": fact["object"],
        "fact_id": fact["fact_id"],
        "session_id": session_id,
        "example_id": fact.get("example_id", ""),
        "question": fact.get("question", ""),
        "support_text": fact.get("support_text", ""),
        "provenance_json": json.dumps(fact.get("provenance", [])),
        "qualifiers_json": json.dumps(fact.get("qualifiers", {})),
        "question_relevance": fact.get("question_relevance", ""),
        "confidence": fact.get("confidence", "supported"),
        "normalization_notes": fact.get("normalization_notes", ""),
        "verification_reason": fact.get("verification_reason", ""),
        "compiled_memory_json": json.dumps(fact.get("compiled_memory", {})),
    }


def _validation_context_provider(stored_rows: List[Dict[str, Any]]):
    def provider(query: str, params: Dict[str, Any]):
        if "RETURN count(r)" in query:
            return MockResult([{"c": 0}])
        if "AND o.name <> $object" in query:
            return MockResult([])
        if "subject_pred_pairs" in query:
            sid = params.get("session_id")
            subject_pred_pairs = {
                (p.get("subject"), p.get("predicate"))
                for p in params.get("subject_pred_pairs", [])
            }
            part_of_subjects = set(params.get("part_of_subjects", []))
            part_of_objects = set(params.get("part_of_objects", []))
            alive_subjects = set(params.get("alive_subjects", []))
            rows = []
            for row in stored_rows:
                if row.get("session_id") != sid:
                    continue
                key = (row.get("subject"), row.get("predicate"))
                inverse_part_of = (
                    row.get("predicate") == "PART_OF"
                    and row.get("subject") in part_of_objects
                    and row.get("object") in part_of_subjects
                )
                alive_match = (
                    row.get("predicate") == "IS_ALIVE"
                    and row.get("subject") in alive_subjects
                )
                if key in subject_pred_pairs or inverse_part_of or alive_match:
                    rows.append(row)
            return MockResult(rows)
        return MockResult([])

    return provider


def test_insert_facts_rejects_conflict_from_persistent_graph_state() -> None:
    graph = make_graph(session_id="sess_test")
    facts = load_fixture()
    existing = dict(facts[7])  # Indonesia CAPITAL_IS Bandung
    candidate = dict(facts[6])  # Indonesia CAPITAL_IS Jakarta
    candidate["confidence"] = "uncertain"

    graph._driver.result_provider = _validation_context_provider([
        _stored_row(existing, session_id="sess_test")
    ])

    result = graph.insert_facts([candidate], validate=True)

    assert result["committed"] == 0
    assert result["conflicts"], "persistent graph contradiction must be reported"
    assert "Contradiction" in result["conflicts"][-1]["reason"]
    assert result["conflicts"][-1]["rejection_label"]["code"] == "functional_conflict"
    assert result["conflicts"][-1]["rule_params_version"] == "rules.v1"
    ledger_writes = [
        p for q, p in graph._driver.queries if "CREATE (d:DecisionLedger" in q
    ]
    assert ledger_writes, "rejected updates must be persisted to the ledger"
    assert ledger_writes[-1]["decision"] == "reject"
    assert ledger_writes[-1]["rejection_label_code"] == "functional_conflict"
    assert ledger_writes[-1]["rule_params_version"] == "rules.v1"
    write_queries = [q for q, _ in graph._driver.queries if "MERGE (s:Entity" in q]
    assert not write_queries
    print("PASS test_insert_facts_rejects_conflict_from_persistent_graph_state")


def test_insert_facts_replaces_lower_confidence_persistent_fact() -> None:
    graph = make_graph(session_id="sess_test")
    facts = load_fixture()
    existing = dict(facts[7])  # Indonesia CAPITAL_IS Bandung
    existing["confidence"] = "uncertain"
    existing["provenance"] = []
    candidate = dict(facts[6])  # Indonesia CAPITAL_IS Jakarta
    candidate["confidence"] = "supported"
    candidate["provenance"] = [{"title": "ex_geo", "sent_id": 4}]

    graph._driver.result_provider = _validation_context_provider([
        _stored_row(existing, session_id="sess_test")
    ])

    result = graph.insert_facts([candidate], validate=True)

    assert result["committed"] == 1
    assert result["replaced"]
    assert result["replaced"][0]["removed_fact_id"] == existing["fact_id"]
    delete_queries = [p for q, p in graph._driver.queries if "DELETE r" in q]
    assert delete_queries and delete_queries[0]["fact_id"] == existing["fact_id"]
    write_queries = [p for q, p in graph._driver.queries if "MERGE (s:Entity" in q]
    assert write_queries and write_queries[0]["fact_id"] == candidate["fact_id"]
    print("PASS test_insert_facts_replaces_lower_confidence_persistent_fact")


def test_insert_facts_rejects_circular_containment_from_persistent_state() -> None:
    graph = make_graph(session_id="sess_test")
    facts = load_fixture()
    existing = dict(facts[4])  # East Indonesia PART_OF Indonesia
    candidate = {
        **existing,
        "subject": "Indonesia",
        "predicate": "PART_OF",
        "object": "East Indonesia",
        "fact_id": "fact_geo_circular",
        "support_text": "Bad circular containment candidate.",
    }

    graph._driver.result_provider = _validation_context_provider([
        _stored_row(existing, session_id="sess_test")
    ])

    result = graph.insert_facts([candidate], validate=True)

    assert result["committed"] == 0
    assert "Circular containment" in result["conflicts"][-1]["reason"]
    print("PASS test_insert_facts_rejects_circular_containment_from_persistent_state")


def test_insert_facts_rejects_alive_dead_conflict_from_persistent_state() -> None:
    graph = make_graph(session_id="sess_test")
    existing = {
        "subject": "John",
        "predicate": "IS_ALIVE",
        "object": "true",
        "fact_id": "john_alive",
        "example_id": "ex_people",
        "confidence": "supported",
        "provenance": [{"title": "ex_people", "sent_id": 1}],
    }
    candidate = {
        "subject": "John",
        "predicate": "IS_ALIVE",
        "object": "false",
        "fact_id": "john_dead",
        "example_id": "ex_people",
        "confidence": "supported",
        "provenance": [{"title": "ex_people", "sent_id": 2}],
    }

    graph._driver.result_provider = _validation_context_provider([
        _stored_row(existing, session_id="sess_test")
    ])

    result = graph.insert_facts([candidate], validate=True)

    assert result["committed"] == 0
    assert "both alive and dead" in result["conflicts"][-1]["reason"]
    print("PASS test_insert_facts_rejects_alive_dead_conflict_from_persistent_state")


def test_persistent_validation_context_is_session_scoped() -> None:
    graph = make_graph(session_id="session_a")
    facts = load_fixture()
    other_session_conflict = dict(facts[7])  # Indonesia CAPITAL_IS Bandung
    candidate = dict(facts[6])  # Indonesia CAPITAL_IS Jakarta
    candidate["confidence"] = "uncertain"

    graph._driver.result_provider = _validation_context_provider([
        _stored_row(other_session_conflict, session_id="session_b")
    ])

    result = graph.insert_facts([candidate], session_id="session_a", validate=True)

    assert result["committed"] == 1
    assert not any("reason" in c for c in result["conflicts"])
    write_queries = [p for q, p in graph._driver.queries if "MERGE (s:Entity" in q]
    assert write_queries and write_queries[0]["session_id"] == "session_a"
    assert write_queries[0]["decision_status"] == "accept"
    assert write_queries[0]["decision_validator"] == graph.validator_backend.info.name
    assert write_queries[0]["rule_params_version"] == "rules.v1"
    print("PASS test_persistent_validation_context_is_session_scoped")


def test_insert_facts_does_not_reconcile_decisions_implicitly() -> None:
    graph = make_graph(session_id="sess_test")
    facts = load_fixture()[:2]

    graph.insert_facts(facts, validate=True)

    reconcile_queries = [
        q for q, _ in graph._driver.queries
        if "r.decision_status = latest.decision" in q
    ]
    assert not reconcile_queries, (
        "insert_facts must not trigger ledger reconciliation; callers such as "
        "experiments.build_kg reconcile explicitly once per session"
    )


def test_reconcile_fact_decisions_refreshes_relationship_metadata() -> None:
    graph = make_graph(session_id="sess_test")
    graph._driver.canned_results = [MockResult([{"reconciled": 3}])]

    reconciled = graph.reconcile_fact_decisions()

    assert reconciled == 3
    query, params = graph._driver.queries[-1]
    assert "d.committed = true" in query
    assert "r.decision_status = latest.decision" in query
    assert params["session_id"] == "sess_test"
    assert params["validator"] == graph.validator_backend.info.name


def test_validate_update_detailed_exposes_rejection_label_and_rule_version() -> None:
    candidate = {
        "subject": "Kalamang",
        "predicate": "LOCATED_IN",
        "object": "unknown",
        "fact_id": "generic_object_candidate",
        "confidence": "supported",
        "provenance": [],
    }
    decision = validate_update_detailed([], candidate)

    assert decision.decision == "reject"
    assert decision.rejection_label is not None
    assert decision.rejection_label.code == "generic_object"
    assert decision.rejection_label.rule_id == "object_not_generic"
    assert decision.rule_params_version == "rules.v1"
    # Legacy tuple API remains intact for existing callers.
    assert decision.as_legacy_tuple()[0] == "reject"
    print("PASS test_validate_update_detailed_exposes_rejection_label_and_rule_version")


def test_create_derived_session_replays_prior_rejections_with_new_rules() -> None:
    graph = make_graph(session_id="source_sess")
    approved = {
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "East Indonesia",
        "fact_id": "approved_fact",
        "example_id": "ex_lang",
        "confidence": "supported",
        "provenance": [{"title": "ex_lang", "sent_id": 3}],
    }
    rejected_candidate = {
        "subject": "Kalamang",
        "predicate": "LOCATED_IN",
        "object": "unknown",
        "fact_id": "rejected_fact",
        "example_id": "ex_lang",
        "confidence": "supported",
        "provenance": [{"title": "ex_lang", "sent_id": 4}],
    }

    def provider(query: str, params: Dict[str, Any]):
        if "RETURN count(r)" in query:
            return MockResult([{"c": 0}])
        if "AND o.name <> $object" in query:
            return MockResult([])
        if "subject_pred_pairs" in query:
            return MockResult([])
        if "RETURN d.decision_id AS decision_id" in query:
            return MockResult([
                {
                    "decision_id": "ledger_reject_1",
                    "session_id": "source_sess",
                    "derived_from_session_id": "",
                    "candidate_fact_id": rejected_candidate["fact_id"],
                    "decision": "reject",
                    "reason": "Rejected: object 'unknown' is too generic to be useful",
                    "rejection_label_json": json.dumps({
                        "code": "generic_object",
                        "category": "quality",
                        "rule_id": "object_not_generic",
                        "severity": "reject",
                    }),
                    "replace_fact_id": "",
                    "committed": False,
                    "rule_params_version": "rules.v1",
                    "rule_params_json": json.dumps({"version": "rules.v1"}),
                    "candidate_json": json.dumps(rejected_candidate),
                }
            ])
        if (
            "MATCH (s:Entity)-[r]->(o:Entity)" in query
            and "WHERE r.session_id = $session_id" in query
            and "subject_pred_pairs" not in query
        ):
            return MockResult([_stored_row(approved, session_id="source_sess")])
        return MockResult([])

    graph._driver.result_provider = provider
    rules_v2 = RuleParameters(version="rules.v2", generic_objects=())

    result = graph.create_derived_session(
        source_session_id="source_sess",
        derived_session_id="derived_sess",
        rule_params=rules_v2,
        include_rejection_labels=["generic_object"],
    )

    assert result["copied_approvals"] == 1
    assert result["replayed_rejections"] == 1
    assert result["replay_result"]["committed"] == 1
    ledger_writes = [
        p for q, p in graph._driver.queries if "CREATE (d:DecisionLedger" in q
    ]
    assert ledger_writes
    assert ledger_writes[-1]["session_id"] == "derived_sess"
    assert ledger_writes[-1]["derived_from_session_id"] == "source_sess"
    assert ledger_writes[-1]["rule_params_version"] == "rules.v2"
    assert ledger_writes[-1]["decision"] == "accept"
    print("PASS test_create_derived_session_replays_prior_rejections_with_new_rules")


def test_export_facts_is_complete_and_deterministic() -> None:
    graph = make_graph(session_id="sess_test")
    graph._driver.canned_results = [MockResult([
        {
            "subject": "B",
            "predicate": "RELATED_TO",
            "object": "C",
            "fact_id": "f2",
            "example_id": "ex2",
            "session_id": "sess_test",
            "support_text": "support two",
            "provenance_json": json.dumps([{"document_id": "doc2"}]),
            "qualifiers_json": json.dumps({"kind": "secondary"}),
            "normalization_notes": "normalized",
            "verification_reason": "verified",
            "decision_status": "accept",
            "decision_validator": "scallop",
            "decision_reason": "valid",
            "rule_params_version": "rules.v1",
        },
        {
            "subject": "A",
            "predicate": "RELATED_TO",
            "object": "B",
            "fact_id": "f1",
            "example_id": "ex1",
            "session_id": "sess_test",
            "support_text": "support one",
            "provenance_json": "[]",
            "qualifiers_json": "{}",
        },
    ])]

    rows = graph.export_facts(session_id="sess_test")

    assert [row["fact_id"] for row in rows] == ["f1", "f2"]
    assert rows[1]["qualifiers"] == {"kind": "secondary"}
    assert rows[1]["normalization_notes"] == "normalized"
    assert rows[1]["verification_reason"] == "verified"
    assert rows[1]["decision_status"] == "accept"
    query, params = graph._driver.queries[-1]
    assert "r.qualifiers_json AS qualifiers_json" in query
    assert "r.normalization_notes AS normalization_notes" in query
    assert "r.decision_status AS decision_status" in query
    assert "ORDER BY" in query
    assert params["session_id"] == "sess_test"



def test_query_context_one_hop_filters_by_example_and_session() -> None:
    graph = make_graph(session_id="sess_test")
    canned = MockResult([
        {
            "subject": "Kalamang",
            "predicate": "SPOKEN_IN",
            "object": "East Indonesia",
            "fact_id": "fact_lang_001",
            "source_fact_id": "source-fact-001",
            "source_event_id": "source-event-001",
            "example_id": "ex_lang",
            "session_id": "sess_test",
            "support_text": "It is spoken by around 130 people in East Indonesia.",
            "provenance_json": json.dumps([{"title": "ex_lang", "sent_id": 3}]),
            "confidence": "supported",
            "decision_status": "accept",
            "decision_validator": "scallop",
            "decision_reason": "No contradiction found",
            "rule_params_version": "rules.v1",
        }
    ])
    graph._driver.canned_results = [canned]
    rows = graph.query_context(
        seed_entities=["Kalamang"],
        hops=1,
        limit=50,
        example_id="ex_lang",
        session_id="sess_test",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["subject"] == "Kalamang"
    assert row["object"] == "East Indonesia"
    assert row["source_fact_id"] == "source-fact-001"
    assert row["source_event_id"] == "source-event-001"
    assert row["provenance"] == [{"title": "ex_lang", "sent_id": 3}]
    assert row["decision_status"] == "accept"
    assert row["decision_validator"] == "scallop"
    assert row["decision_reason"] == "No contradiction found"
    assert row["rule_params_version"] == "rules.v1"

    query, params = graph._driver.queries[-1]
    assert "MATCH (start:Entity) WHERE start.name IN $seed_entities" in query
    assert "[rels*1..1]" in query
    assert "($example_id IS NULL OR r.example_id = $example_id)" in query
    assert "($session_id IS NULL OR r.session_id = $session_id)" in query
    assert "r.compiled_memory_json AS compiled_memory_json" in query
    assert "r.source_fact_id AS source_fact_id" in query
    assert "LIMIT 50" in query
    assert params["seed_entities"] == ["Kalamang"]
    assert params["example_id"] == "ex_lang"
    assert params["session_id"] == "sess_test"
    print("PASS test_query_context_one_hop_filters_by_example_and_session")


def test_query_context_multi_hop_uses_bounded_path() -> None:
    graph = make_graph()
    graph._driver.canned_results = [MockResult([])]
    graph.query_context(seed_entities=["Indonesia"], hops=3, limit=10)
    query, _ = graph._driver.queries[-1]
    assert "[rels*1..3]" in query
    assert "all(r IN rels WHERE" in query
    assert query.index("all(r IN rels WHERE") < query.index("UNWIND rels AS r")
    assert "LIMIT 10" in query

    # Out-of-range hops should clamp to MAX_HOPS (4) and at least 1.
    graph._driver.canned_results = [MockResult([])]
    graph.query_context(seed_entities=["Indonesia"], hops=99)
    query2, _ = graph._driver.queries[-1]
    assert f"[rels*1..{ng.MAX_HOPS}]" in query2

    graph._driver.canned_results = [MockResult([])]
    graph.query_context(seed_entities=["Indonesia"], hops=0)
    query3, _ = graph._driver.queries[-1]
    assert "[rels*1..1]" in query3
    print("PASS test_query_context_multi_hop_uses_bounded_path")


def test_query_context_supports_trusted_session_set() -> None:
    graph = make_graph(session_id="derived")
    graph.query_context(
        seed_entities=["Indonesia"],
        session_id=None,
        session_ids=["source-a", "source-b"],
    )
    query, params = graph._driver.queries[-1]
    assert "r.session_id IN $session_ids" in query
    assert params["session_ids"] == ["source-a", "source-b"]


def test_working_memory_persists_proposal_and_transition_without_fact_edge() -> None:
    graph = make_graph(session_id="sess_test")
    source_fact = {
        "fact_id": "f1",
        "subject": "Alice",
        "predicate": "WORKS_AT",
        "object": "CompanyX",
        "provenance": [{"document_id": "doc", "sentence_id": "doc:1"}],
    }
    scope = MemoryScope(mode="example", session_ids=("sess_test",), example_id="ex1")
    artifact = compile_working_memory(
        {"entity": "Alice", "selected_fact_ids": ["f1"]},
        [source_fact],
        scope,
    )
    transition = transition_for(
        artifact,
        decision="reject",
        reason="conflict",
        validator="scallop",
        rule_version="rules.v1",
    )
    result = graph.persist_working_memory(artifact, transition)
    query, params = graph._driver.queries[-1]
    assert "WorkingMemoryArtifact" in query
    assert "MemoryTransition" in query
    assert "MERGE (s:Entity" not in query
    assert params["committed"] is False
    assert result["transition"]["after_artifact_id"] is None


def test_format_context_for_llm_is_deterministic_and_truncates() -> None:
    rows = [
        {
            "subject": "Kalamang",
            "predicate": "SPOKEN_IN",
            "object": "East Indonesia",
            "fact_id": "fact_lang_001",
            "example_id": "ex_lang",
            "support_text": "It is spoken by around 130 people in East Indonesia.",
            "provenance": [{"title": "ex_lang", "sent_id": 3}],
        },
        {
            "subject": "Indonesia",
            "predicate": "CAPITAL_IS",
            "object": "Jakarta",
            "fact_id": "fact_geo_003",
            "example_id": "ex_geo",
            "support_text": "The capital of Indonesia is Jakarta.",
            "provenance": [{"title": "ex_geo", "sent_id": 4}],
        },
        {
            "subject": "Kalamang",
            "predicate": "HAS_ISO_CODE",
            "object": "kgv",
            "fact_id": "fact_lang_002",
            "example_id": "ex_lang",
            "support_text": "Kalamang (ISO 639-3 code kgv).",
            "provenance": [{"title": "ex_lang", "sent_id": 5}],
        },
    ]
    out1 = Neo4jGraph.format_context_for_llm(rows)
    out2 = Neo4jGraph.format_context_for_llm(list(reversed(rows)))
    assert out1 == out2, "format_context_for_llm must be deterministic"
    # Output starts with ex_geo (sorts before ex_lang).
    assert out1.splitlines()[0].startswith("[F1] Indonesia -CAPITAL_IS-> Jakarta")
    # All three facts rendered.
    assert "[F2]" in out1 and "[F3]" in out1
    # sent_id annotation is present.
    assert "sent_id=4" in out1

    # Truncation: tiny budget -> truncation marker emitted.
    truncated = Neo4jGraph.format_context_for_llm(rows, max_chars=80)
    assert "[truncated" in truncated
    print("PASS test_format_context_for_llm_is_deterministic_and_truncates")


def test_stateless_mode_clears_session_on_close() -> None:
    graph = make_graph(session_id="ephemeral_sess", stateless=True)
    driver_ref = graph._driver  # close() nulls the attribute, hold a ref
    graph.close()
    assert driver_ref.closed, "stateless close() must close the driver"
    delete_queries = [
        (q, p) for q, p in driver_ref.queries
        if "DELETE r" in q and p.get("session_id") == "ephemeral_sess"
    ]
    assert delete_queries, "stateless close() must run a session-scoped DELETE"
    print("PASS test_stateless_mode_clears_session_on_close")


def main() -> int:
    tests = [
        test_ensure_schema_emits_constraints_and_indexes,
        test_commit_facts_emits_expected_cypher_with_session_tag,
        test_propose_facts_classifies_new_existing_conflict,
        test_insert_facts_idempotent_on_repeated_call,
        test_same_fact_id_coexists_across_sessions,
        test_insert_facts_rejects_conflict_from_persistent_graph_state,
        test_insert_facts_replaces_lower_confidence_persistent_fact,
        test_insert_facts_rejects_circular_containment_from_persistent_state,
        test_insert_facts_rejects_alive_dead_conflict_from_persistent_state,
        test_persistent_validation_context_is_session_scoped,
        test_validate_update_detailed_exposes_rejection_label_and_rule_version,
        test_create_derived_session_replays_prior_rejections_with_new_rules,
        test_export_facts_is_complete_and_deterministic,
        test_query_context_one_hop_filters_by_example_and_session,
        test_query_context_multi_hop_uses_bounded_path,
        test_format_context_for_llm_is_deterministic_and_truncates,
        test_stateless_mode_clears_session_on_close,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    total = len(tests)
    print(f"\nResults: {total - failed}/{total} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
