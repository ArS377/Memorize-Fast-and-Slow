#!/usr/bin/env python3
"""Import the validated Cell 6 facts mirror into a Neo4j session."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from neurosym.adapters.neo4j_graph import Neo4jGraph
from neurosym.adapters.scallop import scallopy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facts-file",
        type=Path,
        default=Path("results/kg_builds/pilot_scallop_facts.jsonl"),
    )
    parser.add_argument("--session", default="pilot_scallop")
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    )
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument(
        "--replace-session",
        action="store_true",
        help="Clear only the target session before importing the facts mirror.",
    )
    return parser.parse_args()


def load_facts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Facts file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def session_fact_count(graph: Neo4jGraph, session_id: str) -> int:
    session = graph._session()
    try:
        record = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.session_id = $session_id
            RETURN count(r) AS count
            """,
            session_id=session_id,
        ).single()
        return int(record["count"] if record else 0)
    finally:
        session.close()


def clear_session_state(graph: Neo4jGraph, session_id: str) -> None:
    """Clear one session's facts and audit entries, preserving other sessions."""
    session = graph._session()
    try:
        session.run(
            "MATCH ()-[r]-() WHERE r.session_id = $session_id DELETE r",
            session_id=session_id,
        ).consume()
        session.run(
            "MATCH (d:DecisionLedger) WHERE d.session_id = $session_id "
            "DETACH DELETE d",
            session_id=session_id,
        ).consume()
        session.run(
            "MATCH (e:Entity) WHERE NOT (e)--() DETACH DELETE e"
        ).consume()
    finally:
        session.close()


def main() -> None:
    args = parse_args()
    if not args.neo4j_password:
        raise SystemExit(
            "NEO4J_PASSWORD is not set. Export it or pass --neo4j-password."
        )
    if scallopy is None:
        raise SystemExit(
            "Actual scallopy is unavailable. Run this script with the kg-env Python."
        )

    facts = load_facts(args.facts_file)
    graph = Neo4jGraph(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        session_id=args.session,
    )
    try:
        before = session_fact_count(graph, args.session)
        if args.replace_session:
            clear_session_state(graph, args.session)
        result = graph.insert_facts(
            facts,
            session_id=args.session,
            validate=True,
        )
        reconciled = graph.reconcile_fact_decisions(args.session)
        after = session_fact_count(graph, args.session)
    finally:
        graph.close()

    print(
        json.dumps(
            {
                "input_facts": len(facts),
                "graph_count_before": before,
                "replaced_session": bool(args.replace_session),
                "committed_now": int(result.get("committed", 0)),
                "rejected_now": len(result.get("rejected", [])),
                "reconciled": reconciled,
                "graph_count_after": after,
                "session": args.session,
                "validator": "scallopy",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
