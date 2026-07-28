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

from neurosym.application.experiment_io import iter_pilot_examples


def _facts_count(graph, session_id: str) -> int:
    query = (
        "MATCH ()-[r]->() WHERE r.session_id = $sid "
        "RETURN count(r) AS c"
    )
    with graph._session() as s:
        rec = s.run(query, sid=session_id).single()
        return int(rec.get("c", 0)) if rec else 0


def _dump_session_facts(graph, session_id: str, out_path: Path) -> int:
    """Mirror committed facts to JSONL for the offline fallback."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = graph.export_facts(session_id=session_id)
    temporary = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as fp:
            for row in rows:
                fp.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary, out_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(rows)


def _finalize_session(
    graph,
    *,
    session_id: str,
    out_path: Path,
    rejections_path: Path,
    validate: bool,
    committed: int,
) -> Path:
    """Reconcile audit metadata and mirror one completed KG session."""
    reconciled = graph.reconcile_fact_decisions(session_id) if validate else 0
    mirrored = _dump_session_facts(graph, session_id, out_path)
    print(
        f"[build_kg] done. session={session_id} committed={committed} "
        f"reconciled={reconciled} mirrored={mirrored} -> {out_path}; "
        f"rejections -> {rejections_path}",
        file=sys.stderr,
    )
    return out_path


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
    max_tokens: int = 2048,
    max_chunks_per_example: Optional[int] = None,
    verify_batch_size: int = 20,
    facts_out_dir: Path = Path("results/kg_builds"),
    scallop_validator_url: Optional[str] = None,
    candidate_facts_path: Optional[Path] = None,
) -> Path:
    """Extract+verify facts on the pilot slice and write to Neo4j.

    Returns the JSONL mirror path (``results/kg_builds/<session>_facts.jsonl``).
    """
    from neurosym.adapters.neo4j_graph import Neo4jGraph
    from neurosym.domain.compiled_memory import fact_to_compiled_fact
    from neurosym.reporting.rejections import append_rejection_jsonl, build_rejection_record
    from longbench_kg_pipeline import (
        LongBenchKGPipeline,
        PipelineConfig,
        chunk_sentence_records,
        flatten_context_to_sentence_records,
        make_fact_id,
        normalize_status,
    )

    out_path = facts_out_dir / f"{session_id}_facts.jsonl"
    rejections_path = facts_out_dir / f"{session_id}_rejections.jsonl"

    graph = Neo4jGraph(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        session_id=session_id,
        validator_url=scallop_validator_url,
        require_scallop=validate,
    )
    try:
        existing = _facts_count(graph, session_id)
        if existing > 0 and not rebuild:
            print(
                f"[build_kg] session={session_id} already has {existing} facts; "
                "skipping extraction (use --rebuild to force).",
                file=sys.stderr,
            )
            rejections_path.parent.mkdir(parents=True, exist_ok=True)
            rejections_path.touch(exist_ok=True)
            return _finalize_session(
                graph,
                session_id=session_id,
                out_path=out_path,
                rejections_path=rejections_path,
                validate=validate,
                committed=0,
            )

        if rebuild and existing > 0:
            print(f"[build_kg] --rebuild: clearing session {session_id}", file=sys.stderr)
            graph.clear_session(session_id)

        rejections_path.parent.mkdir(parents=True, exist_ok=True)
        rejections_path.write_text("", encoding="utf-8")

        if candidate_facts_path is not None:
            candidates = []
            with Path(candidate_facts_path).open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        candidates.append(json.loads(line))
            print(
                f"[build_kg] loading frozen candidate corpus: {candidate_facts_path} "
                f"({len(candidates)} facts)",
                file=sys.stderr,
            )
            result = graph.insert_facts(
                candidates,
                session_id=session_id,
                validate=validate,
            )
            scallop_rejected = result.get("rejected", [])
            if not isinstance(scallop_rejected, list):
                scallop_rejected = []
            for rejected in scallop_rejected:
                candidate = rejected.get("candidate", {})
                append_rejection_jsonl(
                    rejections_path,
                    build_rejection_record(
                        candidate_fact=candidate,
                        reason=rejected.get("reason", ""),
                        example_id=str(candidate.get("example_id", "")),
                        session_id=session_id,
                        stage="scallop_validation",
                        existing_conflicting_fact=None,
                        rule_fired=(rejected.get("rejection_label") or {}).get("code", "validator_reject"),
                        validator=graph.validator_backend.info.name,
                    ),
                )
            print(
                f"[build_kg] rejection summary: llm_rejected=0 "
                f"scallop_rejected={len(scallop_rejected)}",
                file=sys.stderr,
            )
            return _finalize_session(
                graph,
                session_id=session_id,
                out_path=out_path,
                rejections_path=rejections_path,
                validate=validate,
                committed=int(result.get("committed", 0)),
            )

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
            max_tokens=max_tokens,
            chunk_chars=chunk_chars,
            max_chunks_per_example=max_chunks_per_example,
            limit=None,
            sleep_seconds=0.0,
            use_json_mode=False,
            neo4j_uri=None,
            neo4j_user=None,
            neo4j_password=None,
            verify_batch_size=verify_batch_size,
            run_id=f"build_{session_id}",
        )
        pipeline = LongBenchKGPipeline(cfg)

        total_committed = 0
        total_llm_rejected = 0
        total_scallop_rejected = 0
        for i, example in enumerate(examples, start=1):
            example_id = str(example.get("_id", f"example_{i}"))
            print(f"[build_kg] ({i}/{len(examples)}) extracting {example_id}", file=sys.stderr)

            sentence_records = flatten_context_to_sentence_records(example)
            chunks = chunk_sentence_records(sentence_records, chunk_chars)
            if max_chunks_per_example is not None:
                chunks = chunks[:max_chunks_per_example]

            extracted: List[Dict[str, Any]] = []
            for ci, chunk in enumerate(chunks):
                text_chars = sum(len(str(record.get("text", ""))) for record in chunk)
                print(
                    f"           chunk {ci + 1}/{len(chunks)} "
                    f"sentences={len(chunk)} text_chars={text_chars}",
                    file=sys.stderr,
                )
                extracted.extend(pipeline.extract_facts(example, chunk, ci))

            verified = pipeline.verify_facts(example, extracted)
            for fact in verified:
                fact["example_id"] = example_id
                fact["fact_id"] = make_fact_id(example_id, fact)
                fact["question"] = str(example.get("question", ""))

            supported = []
            verifier_rejected = []
            for fact in verified:
                status = normalize_status(fact.get("status") or fact.get("confidence"))
                if status == "supported":
                    supported.append(fact)
                else:
                    verifier_rejected.append(fact)
                    append_rejection_jsonl(
                        rejections_path,
                        build_rejection_record(
                            candidate_fact=fact,
                            reason=fact.get("verification_reason") or status,
                            example_id=example_id,
                            session_id=session_id,
                            stage="llm_verification",
                            existing_conflicting_fact=None,
                            validator="llm_self_reflection",
                        ),
                    )
            total_llm_rejected += len(verifier_rejected)

            for fact in supported:
                fact.update(fact_to_compiled_fact(fact))

            scallop_rejected = []
            if supported:
                result = graph.insert_facts(
                    supported, session_id=session_id, validate=validate
                )
                total_committed += int(result.get("committed", 0))
                result_rejected = result.get("rejected", [])
                if isinstance(result_rejected, list):
                    scallop_rejected = result_rejected
                for rejected in scallop_rejected:
                    candidate = rejected.get("candidate", {})
                    if not isinstance(candidate, dict):
                        continue
                    append_rejection_jsonl(
                        rejections_path,
                        build_rejection_record(
                            candidate_fact=candidate,
                            reason=rejected.get("reason", ""),
                            example_id=str(candidate.get("example_id") or example_id),
                            session_id=session_id,
                            stage="scallop_validation",
                            existing_conflicting_fact=rejected.get("existing_conflicting_fact"),
                            rule_fired=rejected.get("rule_fired"),
                            validator="scallop",
                        ),
                    )
            total_scallop_rejected += len(scallop_rejected)
            print(
                f"           extracted={len(extracted)} supported={len(supported)} "
                f"llm_rejected={len(verifier_rejected)} "
                f"scallop_rejected={len(scallop_rejected)} "
                f"committed_total={total_committed}",
                file=sys.stderr,
            )

        print(
            f"[build_kg] rejection summary: llm_rejected={total_llm_rejected} "
            f"scallop_rejected={total_scallop_rejected}",
            file=sys.stderr,
        )
        return _finalize_session(
            graph,
            session_id=session_id,
            out_path=out_path,
            rejections_path=rejections_path,
            validate=validate,
            committed=total_committed,
        )
    finally:
        graph.close()


def _add_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", required=True,
                        help="Neo4j session_id, e.g. pilot_noscallop or pilot_scallop")
    parser.add_argument("--validate", action="store_true",
                        help="Run Scallop validator on each fact (cells 3/6 setting)")
    parser.add_argument("--input", type=Path, default=Path("data.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--scallop-validator-url", default=os.getenv("SCALLOP_VALIDATOR_URL"))
    parser.add_argument("--candidate-facts", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe the target session before rebuilding")
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-chunks-per-example", type=int, default=None)
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
        max_tokens=args.max_tokens,
        max_chunks_per_example=args.max_chunks_per_example,
        verify_batch_size=args.verify_batch_size,
        facts_out_dir=args.facts_out_dir,
        scallop_validator_url=args.scallop_validator_url,
        candidate_facts_path=args.candidate_facts,
    )


if __name__ == "__main__":
    main()
