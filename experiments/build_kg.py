"""Build a Neo4j knowledge graph for the pilot ablation slice.

Two sessions are used across the grid:

* ``pilot_noscallop`` -- ``Neo4jGraph.insert_facts(..., validate=False)``
  (no Scallop gating). Shared by cells 2 and 5.
* ``pilot_scallop`` -- ``Neo4jGraph.insert_facts(..., validate=True)``
  (Scallop gating). Shared by cells 3 and 6.

The build is idempotent: if the target session already has facts and
``--rebuild`` is not passed, we skip the LLM extraction and reuse what is
there. Committed facts are mirrored to
``results/kg_builds/<session>_facts.jsonl`` so cells can fall back to the
JSONL when Neo4j is unreachable, mirroring ``rlm_graph_baseline.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.common import iter_pilot_examples


def _facts_count(graph, session_id: str) -> int:
    query = (
        "MATCH ()-[r]-() WHERE r.session_id = $sid "
        "RETURN count(r) AS c"
    )
    with graph._session() as s:
        rec = s.run(query, sid=session_id).single()
        return int(rec.get("c", 0)) if rec else 0


def _dump_session_facts(graph, session_id: str, out_path: Path) -> int:
    """Mirror committed facts to JSONL for the offline fallback."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    query = (
        "MATCH (s:Entity)-[r]->(o:Entity) WHERE r.session_id = $sid "
        "RETURN s.name AS subject, type(r) AS predicate, o.name AS object, "
        "       r.fact_id AS fact_id, r.example_id AS example_id, "
        "       r.support_text AS support_text, r.provenance_json AS provenance_json, "
        "       r.confidence AS confidence, r.question AS question"
    )
    n = 0
    with graph._session() as s, out_path.open("w", encoding="utf-8") as fp:
        for record in s.run(query, sid=session_id):
            row = dict(record)
            prov_raw = row.pop("provenance_json", None)
            if isinstance(prov_raw, str) and prov_raw:
                try:
                    row["provenance"] = json.loads(prov_raw)
                except json.JSONDecodeError:
                    row["provenance"] = []
            else:
                row["provenance"] = []
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_kg(
    *,
    session_id: str,
    validate: bool,
    input_path: Path,
    model: str,
    vllm_base_url: str,
    api_key: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    limit: Optional[int] = None,
    rebuild: bool = False,
    chunk_chars: int = 12000,
    verify_batch_size: int = 20,
    facts_out_dir: Path = Path("results/kg_builds"),
) -> Path:
    """Extract+verify facts on the pilot slice and write to Neo4j.

    Returns the JSONL mirror path (``results/kg_builds/<session>_facts.jsonl``).
    """
    from neo4j_graph import Neo4jGraph
    from longbench_kg_pipeline import (
        LongBenchKGPipeline,
        PipelineConfig,
        chunk_sentence_records,
        flatten_context_to_sentence_records,
        make_fact_id,
        normalize_status,
    )

    out_path = facts_out_dir / f"{session_id}_facts.jsonl"

    graph = Neo4jGraph(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        session_id=session_id,
    )
    try:
        existing = _facts_count(graph, session_id)
        if existing > 0 and not rebuild:
            print(
                f"[build_kg] session={session_id} already has {existing} facts; "
                "skipping extraction (use --rebuild to force).",
                file=sys.stderr,
            )
            n = _dump_session_facts(graph, session_id, out_path)
            print(f"[build_kg] mirrored {n} facts -> {out_path}", file=sys.stderr)
            return out_path

        if rebuild and existing > 0:
            print(f"[build_kg] --rebuild: clearing session {session_id}", file=sys.stderr)
            graph.clear_session(session_id)

        examples = iter_pilot_examples(input_path, limit)
        print(
            f"[build_kg] session={session_id} validate={validate} "
            f"examples={len(examples)} model={model}",
            file=sys.stderr,
        )

        # Build a pipeline purely for extract_facts/verify_facts; we route
        # writes ourselves through Neo4jGraph.insert_facts(..., validate=...).
        cfg = PipelineConfig(
            input_path=Path(input_path),
            output_path=Path("/dev/null") if os.name != "nt" else Path("NUL"),
            model=model,
            vllm_base_url=vllm_base_url,
            api_key=api_key,
            temperature=0.0,
            max_tokens=2048,
            chunk_chars=chunk_chars,
            max_chunks_per_example=None,
            limit=None,
            sleep_seconds=0.0,
            use_json_mode=False,
            neo4j_uri=None,
            neo4j_user=None,
            neo4j_password=None,
            verify_batch_size=verify_batch_size,
        )
        pipeline = LongBenchKGPipeline(cfg)

        total_committed = 0
        for i, example in enumerate(examples, start=1):
            example_id = str(example.get("_id", f"example_{i}"))
            print(f"[build_kg] ({i}/{len(examples)}) extracting {example_id}", file=sys.stderr)

            sentence_records = flatten_context_to_sentence_records(example)
            chunks = chunk_sentence_records(sentence_records, chunk_chars)

            extracted: List[Dict[str, Any]] = []
            for ci, chunk in enumerate(chunks):
                extracted.extend(pipeline.extract_facts(example, chunk, ci))

            verified = pipeline.verify_facts(example, extracted)
            supported = [
                f for f in verified
                if normalize_status(f.get("status") or f.get("confidence")) == "supported"
            ]
            for fact in supported:
                fact["example_id"] = example_id
                fact["fact_id"] = make_fact_id(example_id, fact)
                fact["question"] = str(example.get("question", ""))

            if supported:
                result = graph.insert_facts(
                    supported, session_id=session_id, validate=validate
                )
                total_committed += int(result.get("committed", 0))
            print(
                f"           extracted={len(extracted)} supported={len(supported)} "
                f"committed_total={total_committed}",
                file=sys.stderr,
            )

        n = _dump_session_facts(graph, session_id, out_path)
        print(
            f"[build_kg] done. session={session_id} committed={total_committed} "
            f"mirrored={n} -> {out_path}",
            file=sys.stderr,
        )
        return out_path
    finally:
        graph.close()


def _add_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", required=True,
                        help="Neo4j session_id, e.g. pilot_noscallop or pilot_scallop")
    parser.add_argument("--validate", action="store_true",
                        help="Run Scallop validator on each fact (cells 3/6 setting)")
    parser.add_argument("--input", type=Path, default=Path("data.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe the target session before rebuilding")
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument("--verify-batch-size", type=int, default=20)
    parser.add_argument("--facts-out-dir", type=Path, default=Path("results/kg_builds"))


def main(argv: Optional[List[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_cli(parser)
    args = parser.parse_args(argv)
    if not args.neo4j_password:
        parser.error("--neo4j-password (or NEO4J_PASSWORD env) is required")
    return build_kg(
        session_id=args.session,
        validate=args.validate,
        input_path=args.input,
        model=args.model,
        vllm_base_url=args.vllm_base_url,
        api_key=args.api_key,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        limit=args.limit,
        rebuild=args.rebuild,
        chunk_chars=args.chunk_chars,
        verify_batch_size=args.verify_batch_size,
        facts_out_dir=args.facts_out_dir,
    )


if __name__ == "__main__":
    main()
