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
from experiments.retrieval_config import EmbeddingConfig, PPRConfig, RetrievalConfig


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


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
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
    p.add_argument("--run-id", default=None, help="Run identifier persisted in every result row")
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
        p.add_argument("--session-id", default=None, help="Override the cell's default KG session")
        p.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        p.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
        p.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
        p.add_argument(
            "--scallop-validator-url",
            default=os.getenv("SCALLOP_VALIDATOR_URL"),
            help="Actual-scallopy validator service used by Cell 6 (for example http://localhost:8765)",
        )
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
        p.add_argument(
            "--retrieval-mode",
            choices=["sparse", "dense", "hybrid", "dense_ppr"],
            default="hybrid",
        )
        p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
        p.add_argument("--embedding-revision", default=None)
        p.add_argument("--embedding-device", default="cpu")
        p.add_argument("--embedding-batch-size", type=_positive_int, default=32)
        p.add_argument("--dense-index-root", type=Path, default=Path("results/dense_indexes"))
        p.add_argument("--dense-failure-policy", choices=["error", "sparse"], default="error")
        p.add_argument("--rrf-k", type=_positive_int, default=60)
        p.add_argument("--ppr-seed-count", type=_positive_int, default=20)
        p.add_argument("--ppr-similarity-threshold", type=float, default=0.0)
        p.add_argument("--ppr-temperature", type=_positive_float, default=0.1)
        p.add_argument("--ppr-damping", type=_positive_float, default=0.5)
        p.add_argument("--ppr-tolerance", type=_positive_float, default=1e-8)
        p.add_argument("--ppr-max-iterations", type=_positive_int, default=100)

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
            p.add_argument(
                "--model-timeout",
                type=_positive_float,
                default=90.0,
                help="Maximum seconds for one native Qwen model completion",
            )
            p.add_argument(
                "--model-max-retries",
                type=int,
                choices=range(0, 4),
                default=0,
                metavar="{0,1,2,3}",
                help="Retries for one native Qwen model completion (default: 0)",
            )
            p.add_argument(
                "--termination-mode",
                choices=["order_gap", "external_budget"],
                default="order_gap",
                help="State-derived stopping or the legacy external RLM budgets",
            )
            p.add_argument(
                "--order-gap-epsilon",
                type=_nonnegative_float,
                default=0.025,
                help="Maximum windowed order-gap considered settled",
            )
            p.add_argument(
                "--order-gap-window",
                type=_positive_int,
                default=2,
                help="Number of root RLM completions in the stopping window",
            )
            p.add_argument(
                "--order-gap-min-iterations",
                type=_positive_int,
                default=2,
                help="Minimum root RLM completions before state-based stopping",
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
    if retrieval == "kg" and args.session_id:
        session_id = args.session_id

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
    validator_backend_label = "n/a"
    retrieval_config = None
    if retrieval == "kg":
        retrieval_config = RetrievalConfig(
            mode=args.retrieval_mode,
            rrf_k=args.rrf_k,
            index_root=args.dense_index_root,
            embedding=EmbeddingConfig(
                model=args.embedding_model,
                requested_revision=args.embedding_revision,
                device=args.embedding_device,
                batch_size=args.embedding_batch_size,
            ),
            ppr=PPRConfig(
                seed_count=args.ppr_seed_count,
                similarity_threshold=args.ppr_similarity_threshold,
                temperature=args.ppr_temperature,
                damping=args.ppr_damping,
                tolerance=args.ppr_tolerance,
                max_iterations=args.ppr_max_iterations,
            ),
            failure_policy=args.dense_failure_policy,
        )
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
                # Cells 2/3/5 only read their already-built sessions. Only
                # Cell 6 performs validated runtime memory updates.
                validator_url=args.scallop_validator_url if cell_id == 6 else None,
                require_scallop=(cell_id == 6),
                retrieval_config=retrieval_config,
            )
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        if cell_id in {2, 5}:
            validator_backend_label = "none"
        elif cell_id == 3:
            validator_backend_label = "scallop"
        else:
            backend = getattr(getattr(graph_source, "graph", None), "validator_backend", None)
            backend_info = getattr(backend, "info", None)
            validator_backend_label = (
                backend_info.name if backend_info is not None else "fallback_file"
            )

    n_correct = 0
    retrieval_eval_path = out_path.parent / "retrieval_eval.jsonl"
    if graph_source is not None:
        retrieval_eval_path.write_text("", encoding="utf-8")
    try:
        with out_path.open("w", encoding="utf-8") as out:
            for i, ex in enumerate(examples, start=1):
                example_id = str(ex.get("_id", f"ex_{i}"))
                gold = str(ex.get("answer", "")).strip().upper()
                question = format_question(ex)
                if graph_source is not None:
                    graph_source.reset_retrieval_history()
                # Per-example latency covers retrieval/context preparation and
                # generation so the six cells remain directly comparable.
                t0 = time.time()

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

                error: Optional[str] = None
                predicted = ""
                relevant_fact_ids: List[str] = []
                relevance_source = "unlabeled"
                termination_mode: Optional[str] = None
                termination_reason: Optional[str] = None
                order_gap_final: Optional[float] = None
                order_gap_window_mean: Optional[float] = None
                rlm_completion_count: Optional[int] = None
                diagnostic_predicted = ""
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
                            model_timeout=args.model_timeout,
                            model_max_retries=args.model_max_retries,
                            termination_mode=args.termination_mode,
                            order_gap_epsilon=args.order_gap_epsilon,
                            order_gap_window=args.order_gap_window,
                            order_gap_min_iterations=args.order_gap_min_iterations,
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
                        relevant_fact_ids = list(getattr(outcome, "cited_fact_ids", []))
                        relevance_source = "cited_fact_ids" if relevant_fact_ids else "unlabeled"
                        termination_mode = getattr(outcome, "termination_mode", None)
                        termination_reason = getattr(outcome, "termination_reason", None)
                        order_gap_final = getattr(outcome, "order_gap_final", None)
                        order_gap_window_mean = getattr(
                            outcome, "order_gap_window_mean", None
                        )
                        rlm_completion_count = getattr(
                            outcome, "rlm_completion_count", None
                        )
                        diagnostic_predicted = str(
                            getattr(outcome, "diagnostic_predicted", "") or ""
                        )
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
                diagnostic_correct = bool(diagnostic_predicted) and diagnostic_predicted == gold
                if correct:
                    n_correct += 1

                retrieval_summary = (
                    graph_source.retrieval_summary() if graph_source is not None else {}
                )
                if graph_source is not None:
                    branch_fact_ids = retrieval_summary.get("branch_fact_ids", {})
                    retrieval_eval_row = {
                        "cell_id": cell_id,
                        "example_id": example_id,
                        "mode": retrieval_summary.get("effective_mode"),
                        "configured_mode": retrieval_summary.get("configured_mode"),
                        "degraded": retrieval_summary.get("degraded", False),
                        "retrieved_fact_ids": retrieval_summary.get("result_fact_ids", []),
                        "relevant_fact_ids": relevant_fact_ids,
                        "relevance_source": relevance_source,
                        "sparse_fact_ids": branch_fact_ids.get("sparse", []),
                        "dense_fact_ids": branch_fact_ids.get("dense", []),
                        "pre_fusion_fact_ids": (
                            list(branch_fact_ids.get("sparse", []))
                            + list(branch_fact_ids.get("dense", []))
                        ),
                        "branch_counts": retrieval_summary.get("branch_counts", {}),
                        "branch_latency_seconds": retrieval_summary.get(
                            "branch_latency_seconds", {}
                        ),
                        "dense_index_identity": retrieval_summary.get(
                            "dense_index_identity", []
                        ),
                    }
                    if retrieval_summary.get("configured_mode") == "dense_ppr":
                        retrieval_eval_row.update(
                            {
                                "ppr_fact_ids": branch_fact_ids.get("ppr", []),
                                "ppr_index_identity": retrieval_summary.get(
                                    "ppr_index_identity"
                                ),
                                "ppr_index_build_seconds": retrieval_summary.get(
                                    "ppr_index_build_seconds"
                                ),
                                "ppr": retrieval_summary.get("ppr"),
                            }
                        )
                    with retrieval_eval_path.open("a", encoding="utf-8") as retrieval_eval:
                        retrieval_eval.write(
                            json.dumps(
                                retrieval_eval_row,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
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
                    run_id=args.run_id,
                    session_id=session_id,
                    memory_scope=getattr(args, "memory_scope", None),
                    orchestration_mode=(
                        "qwen_native_tools_inside_rlm" if qwen_tool_mode
                        else "fixed_context"
                    ),
                    validator_backend=validator_backend_label,
                    configured_retrieval_mode=(
                        retrieval_config.mode if retrieval_config is not None else None
                    ),
                    effective_retrieval_mode=retrieval_summary.get("effective_mode"),
                    retrieval_degraded=retrieval_summary.get("degraded"),
                    embedding_model=(
                        retrieval_config.embedding.model if retrieval_config is not None else None
                    ),
                    embedding_revision=(
                        retrieval_config.embedding.requested_revision
                        if retrieval_config is not None else None
                    ),
                    embedding_device=(
                        retrieval_config.embedding.device if retrieval_config is not None else None
                    ),
                    embedding_batch_size=(
                        retrieval_config.embedding.batch_size if retrieval_config is not None else None
                    ),
                    dense_index_root=(
                        str(retrieval_config.index_root) if retrieval_config is not None else None
                    ),
                    dense_failure_policy=(
                        retrieval_config.failure_policy if retrieval_config is not None else None
                    ),
                    rrf_k=(retrieval_config.rrf_k if retrieval_config is not None else None),
                    dense_index_identity=retrieval_summary.get("dense_index_identity"),
                    retrieval_branch_counts=retrieval_summary.get("branch_counts"),
                    retrieval_branch_latency_seconds=retrieval_summary.get(
                        "branch_latency_seconds"
                    ),
                    retrieval_rrf_settings=retrieval_summary.get("rrf"),
                    termination_mode=termination_mode,
                    termination_reason=termination_reason,
                    order_gap_final=order_gap_final,
                    order_gap_window_mean=order_gap_window_mean,
                    rlm_completion_count=rlm_completion_count,
                    diagnostic_predicted=diagnostic_predicted or None,
                    diagnostic_correct=diagnostic_correct if diagnostic_predicted else None,
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
