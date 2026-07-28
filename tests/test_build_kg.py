from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import neurosym.adapters.neo4j_graph as neo4j_graph

from experiments.build_kg import _facts_count, build_kg


class _Result:
    def single(self):
        return {"c": 0}


def test_facts_count_counts_each_directed_relationship_once() -> None:
    class RecordingSession:
        def __init__(self) -> None:
            self.query = ""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def run(self, query, **_kwargs):
            self.query = query
            return SimpleNamespace(single=lambda: {"c": 38})

    session = RecordingSession()
    graph = SimpleNamespace(_session=lambda: session)

    assert _facts_count(graph, "audit") == 38
    assert "MATCH ()-[r]->()" in session.query


class _FakeGraph:
    def __init__(self) -> None:
        self.rows = []
        self.reconciled = []
        self.closed = False
        self.validator_backend = SimpleNamespace(
            info=SimpleNamespace(name="scallop")
        )

    def _session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def run(self, *_args, **_kwargs):
        return _Result()

    def insert_facts(self, facts, *, session_id, validate):
        self.rows = [{**fact, "session_id": session_id} for fact in facts]
        return {"committed": len(self.rows), "rejected": []}

    def reconcile_fact_decisions(self, session_id):
        self.reconciled.append(session_id)
        return len(self.rows)

    def export_facts(self, *, session_id):
        return list(self.rows)

    def close(self):
        self.closed = True


class _RejectingFakeGraph(_FakeGraph):
    def insert_facts(self, facts, *, session_id, validate):
        self.rows = []
        return {
            "committed": 0,
            "rejected": [
                {
                    "candidate": dict(facts[0]),
                    "reason": "Redundancy: fact already exists",
                    "rejection_label": {"code": "duplicate_fact"},
                }
            ],
        }


def test_frozen_scallop_candidate_build_reconciles_before_mirroring(
    tmp_path: Path, monkeypatch
) -> None:
    graph = _FakeGraph()
    monkeypatch.setattr(neo4j_graph, "Neo4jGraph", lambda **_kwargs: graph)
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        json.dumps(
            {
                "fact_id": "f1",
                "example_id": "ex1",
                "subject": "A",
                "predicate": "RELATED_TO",
                "object": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = build_kg(
        session_id="audit_scallop",
        validate=True,
        input_path=tmp_path / "unused.jsonl",
        model="unused",
        vllm_base_url="http://unused/v1",
        api_key="unused",
        neo4j_uri="bolt://unused",
        neo4j_user="unused",
        neo4j_password="unused",
        facts_out_dir=tmp_path / "facts",
        candidate_facts_path=candidate_path,
    )

    assert graph.reconciled == ["audit_scallop"]
    assert graph.closed is True
    mirrored = [json.loads(line) for line in output.read_text().splitlines()]
    assert mirrored == [{**graph.rows[0]}]


def test_frozen_candidate_build_reports_scallop_rejections(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    graph = _RejectingFakeGraph()
    monkeypatch.setattr(neo4j_graph, "Neo4jGraph", lambda **_kwargs: graph)
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        json.dumps(
            {
                "fact_id": "duplicate",
                "example_id": "ex1",
                "subject": "A",
                "predicate": "RELATED_TO",
                "object": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    build_kg(
        session_id="audit_scallop",
        validate=True,
        input_path=tmp_path / "unused.jsonl",
        model="unused",
        vllm_base_url="http://unused/v1",
        api_key="unused",
        neo4j_uri="bolt://unused",
        neo4j_user="unused",
        neo4j_password="unused",
        facts_out_dir=tmp_path / "facts",
        candidate_facts_path=candidate_path,
    )

    assert (
        "[build_kg] rejection summary: llm_rejected=0 scallop_rejected=1"
        in capsys.readouterr().err
    )
