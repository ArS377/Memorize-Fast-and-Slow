#!/usr/bin/env python3
"""
Usage:
    python rlm_graph_baseline.py \
        --input data.jsonl \
        --output rlm_graph_output.jsonl \
        --neo4j-uri neo4j://127.0.0.1:7687 \
        --neo4j-user neo4j \
        --neo4j-password acmaisf2026 \
        --backend vllm \
        --base-url http://localhost:11434/v1 \
        --model qwen2.5:1.5b \
        --limit 3

Fallback (no Neo4j — load triples from verified_facts.jsonl):
    python rlm_graph_baseline.py \
        --input data.jsonl \
        --output rlm_graph_output.jsonl \
        --facts-file smoke_output.jsonl \
        --backend vllm \
        --base-url http://localhost:11434/v1 \
        --model qwen2.5:1.5b \
        --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rlm.core.rlm import RLM
from rlm.logger.rlm_logger import RLMLogger

from experiments.common import format_question, load_examples
from experiments.rlm_answerer import rlm_answer
from experiments.graph_context import (
    extract_seed_entities,
    format_facts_from_jsonl,
    load_facts_from_jsonl,
)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description="RLM over graph context (Week 2).")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    # Neo4j connection (primary)
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-session-id", default="demo_run")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--limit-triples", type=int, default=50)

    # Fallback: load from JSONL
    parser.add_argument("--facts-file", type=Path, default=None,
                        help="Fallback: load triples from verified_facts.jsonl instead of Neo4j")

    # RLM / model
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--log-dir", default="./rlm_logs_graph")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ── graph source ──────────────────────────────────────────────────────
    graph = None
    fallback_facts = None

    if args.neo4j_uri and args.neo4j_password:
        try:
            from neo4j_graph import Neo4jGraph
            graph = Neo4jGraph(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
                session_id=args.neo4j_session_id,
            )
            print(f"Connected to Neo4j at {args.neo4j_uri}", file=sys.stderr)
        except Exception as e:
            print(f"Neo4j connection failed: {e}", file=sys.stderr)
            print("Falling back to --facts-file if provided.", file=sys.stderr)

    if graph is None:
        if args.facts_file and args.facts_file.exists():
            fallback_facts = load_facts_from_jsonl(args.facts_file)
            print(f"Loaded {len(fallback_facts)} facts from {args.facts_file}", file=sys.stderr)
        else:
            print("ERROR: provide --neo4j-uri + --neo4j-password, or --facts-file", file=sys.stderr)
            sys.exit(1)

    # ── RLM setup ─────────────────────────────────────────────────────────
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = RLMLogger(log_dir=str(log_dir))

    backend_kwargs = {
        "model_name": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
    }

    print(f"Backend: {args.backend} @ {args.base_url}", file=sys.stderr)
    print(f"Model:   {args.model}", file=sys.stderr)
    print(f"Context: {'Neo4j graph' if graph else args.facts_file}", file=sys.stderr)

    examples = load_examples(args.input)
    if args.limit is not None:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} examples", file=sys.stderr)

    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with args.output.open("w") as out:
            for i, ex in enumerate(examples, start=1):
                example_id = str(ex.get("_id", f"ex_{i}"))
                question = format_question(ex)
                gold = str(ex.get("answer", "")).strip().upper()

                # ── build graph context ───────────────────────────────────
                if graph is not None:
                    seed_entities = extract_seed_entities(ex)
                    print(f"\n[{i}/{len(examples)}] {example_id}", file=sys.stderr)
                    print(f"  seeds: {seed_entities}", file=sys.stderr)
                    rows = graph.query_context(
                        seed_entities=seed_entities,
                        hops=args.hops,
                        limit=args.limit_triples,
                        example_id=example_id,
                        session_id=args.neo4j_session_id,
                    )
                    context = graph.format_context_for_llm(rows)
                    n_triples = len(rows)
                else:
                    print(f"\n[{i}/{len(examples)}] {example_id}", file=sys.stderr)
                    context = format_facts_from_jsonl(
                        fallback_facts, example_id, max_chars=4000
                    )
                    n_triples = context.count("[F")

                if not context:
                    print(f"  WARNING: no triples found for {example_id}", file=sys.stderr)
                    context = "No relevant facts found in knowledge graph."

                print(f"  triples: {n_triples} | gold: {gold}", file=sys.stderr)
                print(f"  context preview: {context[:120].replace(chr(10), ' ')}", file=sys.stderr)

                # ── RLM completion over graph context ─────────────────────
                rlm = RLM(
                    backend=args.backend,
                    backend_kwargs=backend_kwargs,
                    environment="local",
                    max_depth=args.max_depth,
                    max_iterations=args.max_iterations,
                    max_tokens=args.max_tokens,
                    logger=logger,
                    verbose=args.verbose,
                )

                t0 = time.time()
                predicted, raw_answer, error = rlm_answer(rlm, context, question)
                if error:
                    print(f"  ERROR: {error}", file=sys.stderr)

                elapsed = time.time() - t0
                correct = (predicted == gold) if predicted and gold else False

                record = {
                    "example_id": example_id,
                    "predicted": predicted,
                    "gold": gold,
                    "correct": correct,
                    "raw_answer": raw_answer[:300],
                    "n_triples": n_triples,
                    "elapsed_seconds": round(elapsed, 2),
                    "error": error,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)

                status = "✓" if correct else "✗"
                print(f"  {status} predicted={predicted!r} gold={gold!r} ({elapsed:.1f}s)",
                      file=sys.stderr)

    finally:
        if graph is not None:
            graph.close()

    # ── summary ───────────────────────────────────────────────────────────
    answered = [r for r in results if r["predicted"]]
    correct_count = sum(r["correct"] for r in answered)
    errors = sum(1 for r in results if r["error"])
    avg_triples = (
        sum(r["n_triples"] for r in results) / len(results) if results else 0
    )

    print(f"\n{'='*50}", file=sys.stderr)
    if answered:
        print(f"Results: {correct_count}/{len(answered)} correct "
              f"({correct_count/len(answered)*100:.1f}%)", file=sys.stderr)
    else:
        print("Results: 0 answered", file=sys.stderr)
    print(f"Avg triples per example: {avg_triples:.1f}", file=sys.stderr)
    print(f"Errors: {errors}/{len(results)}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Logs:   {args.log_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
