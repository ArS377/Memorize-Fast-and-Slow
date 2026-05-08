"""Shared CLI plumbing for cell runners.

Every cell exposes the same flags so the orchestrator can shell out
uniformly. Each cell calls ``run_cell(...)`` with the appropriate
``kind`` (``flat`` / ``rlm``) and ``retrieval`` (``raw`` / ``kg``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.common import (
    cell_output_path,
    extract_letter,
    format_question,
    iter_pilot_examples,
    truncate_context,
    write_result_row,
)


def build_arg_parser(*, cell_id: int, label: str, kind: str, retrieval: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Cell {cell_id} ({label}): {kind} answerer over {retrieval} context."
    )
    p.add_argument("--input", type=Path, default=Path("data.jsonl"))
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--output", type=Path, default=None,
                   help="Per-cell results.jsonl path (default: results/cell{N}_{LABEL}/results.jsonl)")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--no-aggregate", action="store_true",
                   help="Skip auto-aggregation at end (used by orchestrator)")

    # Raw-context cells
    if retrieval == "raw":
        p.add_argument("--raw-max-chars", type=int, default=32000,
                       help="Truncate raw context to this many chars")

    # KG cells
    if retrieval == "kg":
        p.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        p.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
        p.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
        p.add_argument("--facts-file", type=Path, default=None,
                       help="Fallback when Neo4j is unreachable")
        p.add_argument("--hops", type=int, default=2)
        p.add_argument("--limit-triples", type=int, default=50)
        p.add_argument("--context-max-chars", type=int, default=4000)

    # RLM cells
    if kind == "rlm":
        p.add_argument("--backend", default="vllm")
        p.add_argument("--max-depth", type=int, default=2)
        p.add_argument("--max-iterations", type=int, default=10)
        p.add_argument("--max-tokens", type=int, default=32000)
        p.add_argument("--log-dir", type=Path, default=Path("rlm_logs_ablation"))
        p.add_argument("--verbose", action="store_true")

    return p


def _print_summary(cell_id: int, label: str, n_correct: int, n_total: int, out_path: Path) -> None:
    pct = (100.0 * n_correct / n_total) if n_total else 0.0
    print(
        f"Cell {cell_id} ({label}): {n_correct}/{n_total} correct ({pct:.1f}%) -> {out_path}",
        file=sys.stderr,
    )


def _maybe_aggregate(args) -> None:
    if getattr(args, "no_aggregate", False):
        return
    try:
        from experiments.aggregate import main as agg_main
        agg_main(["--results-dir", str(args.results_dir)])
    except Exception as e:
        print(f"[cell] aggregate step failed (non-fatal): {e}", file=sys.stderr)


def run_cell(
    *,
    cell_id: int,
    label: str,
    kind: str,           # "flat" | "rlm"
    retrieval: str,      # "raw" | "kg"
    session_id: Optional[str],  # KG cells only
    argv: Optional[List[str]] = None,
) -> Path:
    parser = build_arg_parser(cell_id=cell_id, label=label, kind=kind, retrieval=retrieval)
    args = parser.parse_args(argv)

    out_path = args.output or cell_output_path(cell_id, label, args.results_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = iter_pilot_examples(args.input, args.limit, seed=args.seed)
    print(
        f"Cell {cell_id} ({label}): {len(examples)} examples; output -> {out_path}",
        file=sys.stderr,
    )

    # Lazy-imported answerer + memory builder
    client = None
    rlm_log_dir = None
    if kind == "flat":
        from openai import OpenAI
        client = OpenAI(base_url=args.vllm_base_url, api_key=args.api_key)
    else:
        rlm_log_dir = args.log_dir

    graph_source = None
    if retrieval == "kg":
        from experiments.graph_context import open_graph_source
        try:
            graph_source = open_graph_source(
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=args.neo4j_password,
                session_id=session_id,
                facts_file=args.facts_file,
            )
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)

    n_correct = 0
    try:
        with out_path.open("w", encoding="utf-8") as out:
            for i, ex in enumerate(examples, start=1):
                example_id = str(ex.get("_id", f"ex_{i}"))
                gold = str(ex.get("answer", "")).strip().upper()
                question = format_question(ex)

                # Build memory/context.
                if retrieval == "raw":
                    context = truncate_context(ex.get("context", ""), args.raw_max_chars)
                    n_triples = 0
                else:
                    context, n_triples = graph_source.context_for(
                        ex,
                        hops=args.hops,
                        limit_triples=args.limit_triples,
                        max_chars=args.context_max_chars,
                    )
                    if not context:
                        context = "No relevant facts."

                t0 = time.time()
                error: Optional[str] = None
                predicted = ""
                try:
                    if kind == "flat":
                        from experiments.flat_answerer import flat_answer
                        predicted, _raw = flat_answer(client, args.model, context, question)
                    else:
                        from experiments.rlm_answerer import make_rlm, rlm_answer
                        rlm = make_rlm(
                            backend=args.backend,
                            model=args.model,
                            base_url=args.vllm_base_url,
                            api_key=args.api_key,
                            max_depth=args.max_depth,
                            max_iterations=args.max_iterations,
                            max_tokens=args.max_tokens,
                            log_dir=rlm_log_dir,
                            verbose=args.verbose,
                        )
                        predicted, _raw, error = rlm_answer(rlm, context, question)
                except Exception as e:
                    error = str(e)
                    predicted = extract_letter(error)

                elapsed = round(time.time() - t0, 2)
                correct = bool(predicted) and predicted == gold and not error
                if correct:
                    n_correct += 1

                write_result_row(
                    out,
                    cell_id=cell_id,
                    label=label,
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
                    f"pred={predicted!r} gold={gold!r} ({elapsed:.1f}s)",
                    file=sys.stderr,
                )
    finally:
        if graph_source is not None:
            graph_source.close()

    _print_summary(cell_id, label, n_correct, len(examples), out_path)
    _maybe_aggregate(args)
    return out_path
