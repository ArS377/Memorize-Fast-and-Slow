#!/usr/bin/env python3
"""
Cumulative KG evaluation: answer LongBench questions using the FULL knowledge
graph (all examples' facts combined) rather than per-example isolation.

Unlike the standard ablation cells which filter by example_id at query time,
this script passes example_id=None so seed-entity queries can retrieve facts
extracted from ANY document in the session.

Usage (Neo4j):
    python evaluate_cumulative_kg.py \
        --input data.jsonl \
        --output results/cumulative_kg/results.jsonl \
        --model Qwen/Qwen3-4B \
        --vllm-base-url http://localhost:8000/v1 \
        --neo4j-uri bolt://localhost:7687 \
        --neo4j-user neo4j \
        --neo4j-password <password> \
        --session-id pilot_scallop \
        --limit 50

Usage (JSONL fallback, no Neo4j):
    python evaluate_cumulative_kg.py \
        --input data.jsonl \
        --output results/cumulative_kg/results.jsonl \
        --model Qwen/Qwen3-4B \
        --vllm-base-url http://localhost:8000/v1 \
        --facts-file results/kg_builds/pilot_scallop_facts.jsonl \
        --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from experiments.common import (
    extract_letter,
    format_question,
    iter_pilot_examples,
    write_result_row,
)
from experiments.flat_answerer import flat_answer
from experiments.graph_context import (
    extract_seed_entities,
    load_facts_from_jsonl,
)
from scallop_validator import validate_update, confidence_score


# ---------------------------------------------------------------------------
# Scallop consistency filter for cross-example fact sets
# ---------------------------------------------------------------------------


def scallop_filter_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter a set of retrieved facts for mutual consistency using Scallop.

    Since cumulative retrieval pulls facts from multiple documents, there may
    be contradictions (e.g., two documents giving different capitals for the
    same country). This incrementally validates each fact against the accepted
    set, keeping only consistent facts (highest-confidence wins on conflicts).
    """
    if not facts:
        return []

    # Sort by confidence descending so higher-confidence facts get in first
    sorted_facts = sorted(facts, key=confidence_score, reverse=True)

    accepted: List[Dict[str, Any]] = []
    for fact in sorted_facts:
        decision, reason, replace_id = validate_update(accepted, fact)
        if decision == "accept":
            accepted.append(fact)
        elif decision == "replace" and replace_id:
            accepted = [f for f in accepted if f.get("fact_id") != replace_id]
            accepted.append(fact)
        # "reject" → skip silently

    return accepted


# ---------------------------------------------------------------------------
# Cumulative context retrieval (no example_id filter)
# ---------------------------------------------------------------------------


def cumulative_context_neo4j(
    graph,
    ex: Dict[str, Any],
    session_id: str,
    hops: int = 2,
    limit_triples: int = 50,
    max_chars: int = 4000,
    validate: bool = True,
) -> tuple[str, int]:
    """Query the full KG session without example_id filtering."""
    seeds = extract_seed_entities(ex)
    if not seeds:
        return "", 0
    rows = graph.query_context(
        seed_entities=seeds,
        hops=hops,
        limit=limit_triples,
        example_id=None,  # <-- KEY: no per-example filter
        session_id=session_id,
    )
    if validate and rows:
        rows = scallop_filter_facts(rows)
    from neo4j_graph import Neo4jGraph
    context = Neo4jGraph.format_context_for_llm(rows, max_chars=max_chars)
    return context, len(rows)


def cumulative_context_jsonl(
    facts: List[Dict[str, Any]],
    ex: Dict[str, Any],
    max_chars: int = 4000,
    validate: bool = True,
) -> tuple[str, int]:
    """JSONL fallback: match seed entities against ALL facts (no example_id filter)."""
    seeds = extract_seed_entities(ex)
    if not seeds:
        return "", 0
    seeds_lower = {s.lower() for s in seeds}

    # Score facts by seed-entity overlap
    relevant: List[Dict[str, Any]] = []
    for fact in facts:
        subj = str(fact.get("subject", "")).lower()
        obj = str(fact.get("object", "")).lower()
        if any(s in subj or s in obj for s in seeds_lower):
            relevant.append(fact)

    if not relevant:
        # Fallback: take first N facts (better than nothing)
        relevant = facts[:50]

    # Scallop validation: filter for cross-document consistency
    if validate and relevant:
        relevant = scallop_filter_facts(relevant)

    # Format
    lines: List[str] = []
    used = 0
    for i, fact in enumerate(relevant, start=1):
        subject = str(fact.get("subject", ""))
        predicate = str(fact.get("predicate", ""))
        obj = str(fact.get("object", ""))
        support = str(fact.get("support_text", "")).strip()
        eid = str(fact.get("example_id", ""))
        prov = fact.get("provenance", []) or []
        sent_ids = [str(p["sent_id"]) for p in prov if isinstance(p, dict) and "sent_id" in p]
        sent_part = f"sent_id={','.join(sent_ids)}" if sent_ids else "sent_id=?"
        head = f"[F{i}] {subject} -{predicate}-> {obj}"
        evidence = (
            f"     evidence: \"{support}\" ({eid}, {sent_part})"
            if support else
            f"     evidence: ({eid}, {sent_part})"
        )
        block = head + "\n" + evidence
        if used + len(block) > max_chars:
            lines.append(f"... [{len(relevant) - i + 1} more facts truncated]")
            break
        lines.append(block)
        used += len(block) + 1

    context = "\n".join(lines)
    n_triples = context.count("[F") if context else 0
    return context, n_triples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LongBench using cumulative (cross-example) KG context."
    )
    parser.add_argument("--input", type=Path, default=Path("data.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/cumulative_kg/results.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))

    # KG source
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--facts-file", type=Path, default=None,
                        help="JSONL fallback when Neo4j is unreachable")
    parser.add_argument("--session-id", default="pilot_scallop",
                        help="KG session to query (default: pilot_scallop)")

    # Retrieval tuning
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--limit-triples", type=int, default=50)
    parser.add_argument("--context-max-chars", type=int, default=4000)

    # Scallop validation at retrieval time
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Apply Scallop consistency filter on retrieved facts (default: on)")
    parser.add_argument("--no-validate", dest="validate", action="store_false",
                        help="Disable Scallop filter (raw cumulative retrieval)")

    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load examples
    examples = iter_pilot_examples(args.input, args.limit, seed=args.seed)
    validate_label = "Scallop ON" if args.validate else "Scallop OFF"
    print(f"Cumulative KG eval ({validate_label}): {len(examples)} examples; output -> {args.output}",
          file=sys.stderr)

    # Open KG source
    graph = None
    fallback_facts: Optional[List[Dict[str, Any]]] = None

    if args.neo4j_uri and args.neo4j_user and args.neo4j_password:
        try:
            from neo4j_graph import Neo4jGraph
            graph = Neo4jGraph(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
                session_id=args.session_id,
            )
            print(f"Connected to Neo4j ({args.session_id}), querying FULL graph (no example_id filter)",
                  file=sys.stderr)
        except Exception as e:
            print(f"Neo4j connection failed: {e}; trying --facts-file", file=sys.stderr)
            graph = None

    if graph is None:
        if args.facts_file and args.facts_file.exists():
            fallback_facts = load_facts_from_jsonl(args.facts_file)
            print(f"Loaded {len(fallback_facts)} facts from {args.facts_file} (cumulative mode)",
                  file=sys.stderr)
        else:
            print("ERROR: No graph source. Provide --neo4j-password or --facts-file.", file=sys.stderr)
            sys.exit(2)

    # LLM client
    client = OpenAI(base_url=args.vllm_base_url, api_key=args.api_key)

    # Evaluate
    cell_label = "flat_cumulative_kg_scallop" if args.validate else "flat_cumulative_kg_noscallop"
    n_correct = 0
    n_total = 0

    try:
        with args.output.open("w", encoding="utf-8") as out:
            for i, ex in enumerate(examples, start=1):
                example_id = str(ex.get("_id", f"ex_{i}"))
                gold = str(ex.get("answer", "")).strip().upper()
                question = format_question(ex)

                # Retrieve cumulative context
                if graph is not None:
                    context, n_triples = cumulative_context_neo4j(
                        graph, ex, args.session_id,
                        hops=args.hops,
                        limit_triples=args.limit_triples,
                        max_chars=args.context_max_chars,
                        validate=args.validate,
                    )
                else:
                    context, n_triples = cumulative_context_jsonl(
                        fallback_facts, ex,
                        max_chars=args.context_max_chars,
                        validate=args.validate,
                    )

                if not context:
                    context = "No relevant facts found in the knowledge graph."

                # Answer
                t0 = time.time()
                error: Optional[str] = None
                predicted = ""
                try:
                    predicted, _raw = flat_answer(client, args.model, context, question)
                except Exception as e:
                    error = str(e)
                    print(f"  [{i}/{len(examples)}] EXC {example_id}: {error[:200]}", file=sys.stderr)

                elapsed = round(time.time() - t0, 2)
                correct = bool(predicted) and predicted == gold and not error
                if correct:
                    n_correct += 1
                n_total += 1

                write_result_row(
                    out,
                    cell_id=7,
                    label=cell_label,
                    example_id=example_id,
                    predicted=predicted,
                    gold=gold,
                    correct=correct,
                    n_context_chars=len(context),
                    n_triples=n_triples,
                    elapsed_seconds=elapsed,
                    error=error,
                )
                status = "OK " if correct else "x  "
                print(
                    f"  [{i}/{len(examples)}] {status}{example_id} "
                    f"pred={predicted!r} gold={gold!r} triples={n_triples} ({elapsed:.1f}s)",
                    file=sys.stderr,
                )
    finally:
        if graph is not None:
            try:
                graph.close()
            except Exception:
                pass

    # Summary
    pct = (100.0 * n_correct / n_total) if n_total else 0.0
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Cumulative KG ({cell_label}): {n_correct}/{n_total} correct ({pct:.1f}%)", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)


if __name__ == "__main__":
    main()
