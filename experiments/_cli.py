"""Shared CLI plumbing for cell runners.

Every cell exposes the same flags so the orchestrator can shell out
uniformly. Each cell calls ``run_cell(...)`` with the appropriate
``kind`` (``flat`` / ``rlm``) and ``retrieval`` (``raw`` / ``kg``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.common import (
    cell_output_path,
    format_question,
    iter_pilot_examples,
    truncate_context,
    write_result_row,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _write_tool_trace(
    trace_dir: Path,
    *,
    example_id: str,
    index: int,
    trace: Dict[str, Any],
) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", example_id).strip("._") or "example"
    path = trace_dir / f"{index:04d}_{safe_id}.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_arg_parser(*, cell_id: int, label: str, kind: str, retrieval: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Cell {cell_id} ({label}): {kind} answerer over {retrieval} context."
    )
    p.add_argument("--input", type=Path, default=Path("data.jsonl"))
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
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

    if kind == "flat":
        p.add_argument("--max-completion-tokens", type=int, default=2048,
                       help="Maximum tokens for the flat LLM answer call")
        p.add_argument("--disable-thinking", action="store_true",
                       help="Disable Qwen3 thinking mode for faster flat-cell runs")

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
        p.add_argument(
            "--memory-scope",
            choices=["example", "session", "session_set"],
            default="example",
            help=(
                "example=only retrieve facts for the current example; "
                "session=retrieve accumulated facts across one KG session; "
                "session_set=retrieve across an application-owned source-session allowlist"
            ),
        )
        p.add_argument(
            "--source-session",
            action="append",
            default=[],
            help="Trusted source session; repeat at least twice with --memory-scope session_set",
        )

    # RLM cells
    if kind == "rlm":
        # rlms 0.1.x: backend="vllm" tries to spawn vLLM via the python `vllm`
        # package (ignores base_url). To talk to an already-running vLLM HTTP
        # server use backend="openai" (vLLM is OpenAI-API compatible).
        p.add_argument("--backend", default="openai")
        p.add_argument("--max-depth", type=int, default=2)
        p.add_argument("--max-iterations", type=int, default=10)
        # RLM total token budget (summed across all sub-LM calls per example).
        # 32000 was too tight for Qwen3 with <think> reasoning blocks: a
        # 6-iteration trajectory routinely overshoots ~33k tokens. Match
        # rlm_baseline.py default of 64000.
        p.add_argument("--max-tokens", type=int, default=64000)
        p.add_argument("--log-dir", type=Path, default=Path("rlm_logs_ablation"))
        p.add_argument("--verbose", action="store_true")
        if retrieval == "kg":
            p.add_argument(
                "--qwen-tool-retrieval",
                action="store_true",
                help=(
                    "Deprecated compatibility flag; native KG tools are the default "
                    "for both cells 5 and 6"
                ),
            )
            p.add_argument(
                "--fixed-kg-retrieval",
                action="store_true",
                help="Run the legacy pre-retrieved KG baseline instead of native tools",
            )
            p.add_argument(
                "--max-tool-calls",
                type=_positive_int,
                default=3,
                help="Maximum native knowledge-graph tool calls per example",
            )
            p.add_argument(
                "--tool-choice",
                choices=["auto", "required"],
                default="required",
                help="OpenAI-compatible tool selection mode before the call limit",
            )
            p.add_argument(
                "--tool-timeout",
                type=_positive_float,
                default=30.0,
                help="Maximum seconds allowed for one graph-tool execution",
            )
            p.add_argument(
                "--tool-trace-dir",
                type=Path,
                default=None,
                help="Trace directory (default: beside the cell output)",
            )
            p.add_argument(
                "--tool-max-tokens",
                type=_positive_int,
                default=2048,
                help="Maximum completion tokens for each Qwen tool-loop turn",
            )

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

    qwen_tool_mode = bool(
        kind == "rlm"
        and retrieval == "kg"
        and not getattr(args, "fixed_kg_retrieval", False)
    )
    # Lazy-imported answerer + memory builder
    client = None
    rlm_log_dir = None
    if kind == "flat":
        from openai import OpenAI
        client = OpenAI(base_url=args.vllm_base_url, api_key=args.api_key)
    if kind == "rlm":
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
                memory_scope=args.memory_scope,
                source_session_ids=args.source_session,
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
                    n_context_chars = len(context)
                elif qwen_tool_mode:
                    # The root RLM sees the question before native retrieval.
                    context = ""
                    n_triples = 0
                    n_context_chars = 0
                else:
                    context, n_triples = graph_source.context_for(
                        ex,
                        hops=args.hops,
                        limit_triples=args.limit_triples,
                        max_chars=args.context_max_chars,
                    )
                    # Preserve the pre-existing fixed-retrieval baseline only.
                    if not context:
                        context = "No relevant facts."
                    n_context_chars = len(context)

                t0 = time.time()
                error: Optional[str] = None
                predicted = ""
                try:
                    if qwen_tool_mode:
                        from experiments.rlm_retrieval import qwen_rlm_tool_answer

                        outcome = qwen_rlm_tool_answer(
                            backend=args.backend,
                            model=args.model,
                            base_url=args.vllm_base_url,
                            api_key=args.api_key,
                            max_depth=args.max_depth,
                            max_iterations=args.max_iterations,
                            max_tokens=args.max_tokens,
                            log_dir=Path(rlm_log_dir) / "qwen_tool",
                            verbose=args.verbose,
                            graph_source=graph_source,
                            example=ex,
                            max_tool_calls=args.max_tool_calls,
                            tool_choice=args.tool_choice,
                            tool_timeout=args.tool_timeout,
                            max_completion_tokens=args.tool_max_tokens,
                            validate_memory_updates=(cell_id == 6),
                            require_memory_update=True,
                        )
                        trace_dir = args.tool_trace_dir or (out_path.parent / "tool_traces")
                        trace_path = _write_tool_trace(
                            trace_dir,
                            example_id=example_id,
                            index=i,
                            trace=outcome.trace,
                        )
                        predicted = outcome.predicted
                        error = outcome.error
                        n_triples = outcome.retrieved_fact_count
                        n_context_chars = outcome.tool_result_chars
                        print(
                            f"  [{i}/{len(examples)}] tool_trace={trace_path} "
                            f"termination={outcome.termination_reason}",
                            file=sys.stderr,
                        )
                    elif kind == "flat":
                        from experiments.flat_answerer import flat_answer
                        predicted, _raw = flat_answer(
                            client,
                            args.model,
                            context,
                            question,
                            max_tokens=args.max_completion_tokens,
                            enable_thinking=not args.disable_thinking,
                        )
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
                    predicted = ""
                    print(
                        f"  [{i}/{len(examples)}] EXC {example_id}: {error[:200]}",
                        file=sys.stderr,
                    )

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
                    n_context_chars=n_context_chars,
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
