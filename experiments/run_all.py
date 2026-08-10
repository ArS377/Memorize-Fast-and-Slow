"""Orchestrator: run all six ablation cells end-to-end.

Pipeline:

1. Resolve config and create an isolated ``results/runs/<run_id>`` manifest.
2. Extract one frozen candidate corpus, then build run-specific unconstrained
   and Scallop-filtered KG sessions from it.
3. Run each requested cell as a standalone module call (passing
   ``--no-aggregate`` so we aggregate just once at the end).
4. Aggregate results and write the Summer 6/20 compliance report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from neurosym.application.experiment_io import CELLS, iter_pilot_examples
from neurosym.adapters.kg_search import TOOL_VERSION as SEARCH_TOOL_VERSION
from neurosym.adapters.rlm import (
    FULL_CONTEXT_BATCH_CHARS,
    FULL_CONTEXT_COMPACTION_THRESHOLD_PCT,
    FULL_CONTEXT_RESPONSE_CHARS,
    FULL_CONTEXT_SUBQUERY_CHARS,
    FULL_CONTEXT_TOTAL_TOKEN_BUDGET,
)
from neurosym.domain.retrieval_config import EmbeddingConfig, PPRConfig, RetrievalConfig
from neurosym.adapters.working_memory_tool import TOOL_VERSION as MEMORY_TOOL_VERSION
from neurosym.domain.validation_rules import DEFAULT_RULE_PARAMETERS

KG_CELL_IDS = {2, 3, 5, 6}
KG_SESSIONS = {2: "pilot_noscallop", 3: "pilot_scallop",
               5: "pilot_noscallop", 6: "pilot_scallop"}


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_run_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("run_id must contain at least one safe character")
    return cleaned


def _write_manifest(results_dir: Path, metadata: dict) -> None:
    payload = json.dumps(metadata, indent=2) + "\n"
    (results_dir / "manifest.json").write_text(payload, encoding="utf-8")
    # Retain the original filename for scripts written before run-local manifests.
    (results_dir / "run_metadata.json").write_text(payload, encoding="utf-8")


def _materialize_pilot_input(
    source_path: Path,
    output_path: Path,
    *,
    limit: int,
    seed: int,
) -> dict:
    """Freeze the shared pilot slice once so every stage avoids rescanning the corpus."""
    rows = iter_pilot_examples(source_path, limit, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "example_count": len(rows),
        "example_ids": [str(row.get("_id", "")) for row in rows],
    }


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


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _parse_cells(spec: str) -> List[int]:
    if not spec or spec.lower() == "all":
        return [c["cell_id"] for c in CELLS]
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _module_for(cell_id: int) -> str:
    label = next(c["label"] for c in CELLS if c["cell_id"] == cell_id)
    return f"experiments.cells.cell{cell_id}_{label}"


def _common_cell_args(args, cell_id: int) -> List[str]:
    base = [
        "--input", str(getattr(args, "pilot_input", args.input)),
        "--limit", str(args.limit),
        "--seed", str(args.seed),
        "--model", args.model,
        "--vllm-base-url", args.vllm_base_url,
        "--results-dir", str(args.results_dir),
        "--run-id", getattr(args, "run_id", "adhoc"),
        "--no-aggregate",
    ]
    if cell_id in KG_CELL_IDS:
        retrieval_mode = (
            getattr(args, "cell6_retrieval_mode", "dense_ppr")
            if cell_id == 6
            else getattr(args, "retrieval_mode", "hybrid")
        )
        if args.neo4j_uri:
            base += ["--neo4j-uri", args.neo4j_uri]
        if args.neo4j_user:
            base += ["--neo4j-user", args.neo4j_user]
        session_id = getattr(args, "kg_sessions", KG_SESSIONS)[cell_id]
        base += ["--session-id", session_id]
        facts_file = args.results_dir / "kg_builds" / f"{session_id}_facts.jsonl"
        if facts_file.exists():
            base += ["--facts-file", str(facts_file)]
        if args.hops is not None:
            base += ["--hops", str(args.hops)]
        if args.limit_triples is not None:
            base += ["--limit-triples", str(args.limit_triples)]
        base += [
            "--memory-scope", args.memory_scope,
            "--retrieval-mode", retrieval_mode,
            "--embedding-model", getattr(args, "embedding_model", "BAAI/bge-small-en-v1.5"),
            "--embedding-device", getattr(args, "embedding_device", "cpu"),
            "--embedding-batch-size", str(getattr(args, "embedding_batch_size", 32)),
            "--dense-index-root", str(getattr(args, "dense_index_root", args.results_dir / "dense_indexes")),
            "--dense-failure-policy", getattr(args, "dense_failure_policy", "error"),
            "--rrf-k", str(getattr(args, "rrf_k", 60)),
        ]
        if retrieval_mode == "dense_ppr":
            base += [
                "--ppr-seed-count", str(getattr(args, "ppr_seed_count", 20)),
                "--ppr-similarity-threshold", str(
                    getattr(args, "ppr_similarity_threshold", 0.0)
                ),
                "--ppr-temperature", str(getattr(args, "ppr_temperature", 0.1)),
                "--ppr-damping", str(getattr(args, "ppr_damping", 0.5)),
                "--ppr-tolerance", str(getattr(args, "ppr_tolerance", 1e-8)),
                "--ppr-max-iterations", str(
                    getattr(args, "ppr_max_iterations", 100)
                ),
            ]
        if getattr(args, "embedding_revision", None):
            base += ["--embedding-revision", args.embedding_revision]
        for source_session in getattr(args, "source_session", []):
            base += ["--source-session", source_session]
        if cell_id == 6 and getattr(args, "scallop_validator_url", None):
            base += ["--scallop-validator-url", args.scallop_validator_url]
    else:
        raw_max_chars = (
            getattr(args, "cell4_raw_max_chars", None)
            if cell_id == 4
            else args.raw_max_chars
        )
        if raw_max_chars is not None:
            base += ["--raw-max-chars", str(raw_max_chars)]
    if cell_id in (4, 5, 6):
        max_tokens = (
            getattr(args, "cell4_max_tokens", FULL_CONTEXT_TOTAL_TOKEN_BUDGET)
            if cell_id == 4
            else args.max_tokens
        )
        base += [
            "--max-depth", str(args.max_depth),
            "--max-iterations", str(args.max_iterations),
            "--max-tokens", str(max_tokens),
            "--log-dir", str(args.results_dir / f"cell{cell_id}_{next(c['label'] for c in CELLS if c['cell_id'] == cell_id)}" / "rlm_logs"),
        ]
        if cell_id in (5, 6):
            if not getattr(args, "fixed_kg_retrieval", False):
                base += [
                    "--qwen-tool-retrieval",
                    "--max-tool-calls", str(
                        2 if cell_id == 6 else args.max_tool_calls
                    ),
                    "--tool-choice", args.tool_choice,
                    "--tool-timeout", str(args.tool_timeout),
                    "--tool-max-tokens", str(args.tool_max_tokens),
                    "--model-timeout", str(args.model_timeout),
                    "--model-max-retries", str(args.model_max_retries),
                    "--termination-mode", getattr(args, "termination_mode", "order_gap"),
                    "--order-gap-epsilon", str(getattr(args, "order_gap_epsilon", 0.025)),
                    "--order-gap-window", str(getattr(args, "order_gap_window", 2)),
                    "--order-gap-min-iterations", str(
                        getattr(args, "order_gap_min_iterations", 2)
                    ),
                ]
                if args.tool_trace_dir is not None:
                    base += ["--tool-trace-dir", str(args.tool_trace_dir / f"cell{cell_id}")]
            else:
                base += ["--fixed-kg-retrieval"]
    return base


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--rebuild-kg", action="store_true")
    parser.add_argument("--skip-kg-build", action="store_true")
    parser.add_argument("--kg-session-noscallop", default=None)
    parser.add_argument("--kg-session-scallop", default=None)
    parser.add_argument("--cells", default="all",
                        help="Comma-separated cell ids, e.g. '1,3,5'. Default: all six.")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--raw-max-chars",
        type=_positive_int,
        default=32000,
        help="Cell 1 raw-context cap in characters (default: 32000).",
    )
    parser.add_argument(
        "--cell4-raw-max-chars",
        type=_positive_int,
        default=None,
        help=(
            "Optional Cell 4 external-context cap. By default Cell 4 exposes the "
            "full document to the RLM REPL."
        ),
    )
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--limit-triples", type=int, default=50)
    parser.add_argument("--memory-scope", choices=["example", "session", "session_set"], default="example")
    parser.add_argument("--source-session", action="append", default=[])
    parser.add_argument("--scallop-validator-url", default=os.getenv("SCALLOP_VALIDATOR_URL"))
    parser.add_argument(
        "--retrieval-mode",
        choices=["sparse", "dense", "hybrid", "dense_ppr"],
        default="hybrid",
    )
    parser.add_argument(
        "--cell6-retrieval-mode",
        choices=["sparse", "dense", "hybrid", "dense_ppr"],
        default="dense_ppr",
        help="Cell 6 retrieval mode; defaults to HippoRAG-style dense PPR.",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=_positive_int, default=32)
    parser.add_argument("--dense-index-root", type=Path, default=None)
    parser.add_argument("--dense-failure-policy", choices=["error", "sparse"], default="error")
    parser.add_argument("--rrf-k", type=_positive_int, default=60)
    parser.add_argument("--ppr-seed-count", type=_positive_int, default=20)
    parser.add_argument("--ppr-similarity-threshold", type=float, default=0.0)
    parser.add_argument("--ppr-temperature", type=_positive_float, default=0.1)
    parser.add_argument("--ppr-damping", type=_positive_float, default=0.5)
    parser.add_argument("--ppr-tolerance", type=_positive_float, default=1e-8)
    parser.add_argument("--ppr-max-iterations", type=_positive_int, default=100)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument(
        "--cell4-max-tokens",
        type=_positive_int,
        default=FULL_CONTEXT_TOTAL_TOKEN_BUDGET,
        help="Cell 4 aggregate RLM execution budget after bounded sub-calls.",
    )
    parser.add_argument(
        "--qwen-tool-retrieval",
        action="store_true",
        help="Deprecated compatibility flag; native Qwen tools are now the cells 5/6 default",
    )
    parser.add_argument(
        "--fixed-kg-retrieval",
        action="store_true",
        help="Use legacy fixed pre-retrieval for both cells 5 and 6",
    )
    parser.add_argument("--max-tool-calls", type=_positive_int, default=3)
    parser.add_argument("--tool-choice", choices=["auto", "required"], default="required")
    parser.add_argument("--tool-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--tool-trace-dir", type=Path, default=None)
    parser.add_argument("--tool-max-tokens", type=_positive_int, default=2048)
    parser.add_argument("--model-timeout", type=_positive_float, default=90.0)
    parser.add_argument("--model-max-retries", type=int, choices=range(0, 4), default=0)
    parser.add_argument(
        "--termination-mode",
        choices=["order_gap", "external_budget"],
        default="order_gap",
    )
    parser.add_argument("--order-gap-epsilon", type=_nonnegative_float, default=0.025)
    parser.add_argument("--order-gap-window", type=_positive_int, default=2)
    parser.add_argument("--order-gap-min-iterations", type=_positive_int, default=2)
    args = parser.parse_args(argv)

    sha = _git_sha()
    default_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + (sha or "nogit")[:7]
    args.run_id = _safe_run_id(args.run_id or default_run_id)
    args.results_dir = args.results_dir or (Path("results") / "runs" / args.run_id)
    args.dense_index_root = args.dense_index_root or (args.results_dir / "dense_indexes")
    if args.dense_failure_policy != "error":
        parser.error(
            "canonical run_all requires --dense-failure-policy error; "
            "use a direct cell entry point for an explicitly degraded run"
        )
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
    args.kg_sessions = {
        2: args.kg_session_noscallop or f"{args.run_id}_noscallop",
        3: args.kg_session_scallop or f"{args.run_id}_scallop",
        5: args.kg_session_noscallop or f"{args.run_id}_noscallop",
        6: args.kg_session_scallop or f"{args.run_id}_scallop",
    }

    cells = _parse_cells(args.cells)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.pilot_input = args.results_dir / "pilot_input.jsonl"
    pilot_input = _materialize_pilot_input(
        args.input,
        args.pilot_input,
        limit=args.limit,
        seed=args.seed,
    )
    retrieval_eval_path = args.results_dir / "retrieval_eval.jsonl"
    if retrieval_eval_path.exists():
        retrieval_eval_path.unlink()
    orchestration_mode = "fixed" if args.fixed_kg_retrieval else "qwen_native_tools_inside_rlm"

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "git_sha": sha,
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "pilot_input": pilot_input,
        "limit": args.limit,
        "seed": args.seed,
        "model": args.model,
        "vllm_base_url": args.vllm_base_url,
        "neo4j_uri": args.neo4j_uri,
        "cells": cells,
        "raw_max_chars": args.raw_max_chars,
        "cell4_raw_max_chars": args.cell4_raw_max_chars,
        "cell4_subquery_max_chars": FULL_CONTEXT_SUBQUERY_CHARS,
        "cell4_batch_max_chars": FULL_CONTEXT_BATCH_CHARS,
        "cell4_response_max_chars": FULL_CONTEXT_RESPONSE_CHARS,
        "cell4_max_tokens": args.cell4_max_tokens,
        "cell4_compaction_threshold_pct": FULL_CONTEXT_COMPACTION_THRESHOLD_PCT,
        "hops": args.hops,
        "limit_triples": args.limit_triples,
        "memory_scope": args.memory_scope,
        "retrieval_config": retrieval_config.to_dict(),
        "retrieval_mode": args.retrieval_mode,
        "cell6_retrieval_mode": args.cell6_retrieval_mode,
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "dense_index_root": str(args.dense_index_root),
        "dense_failure_policy": args.dense_failure_policy,
        "rrf_k": args.rrf_k,
        "max_depth": args.max_depth,
        "max_iterations": args.max_iterations,
        "max_tokens": args.max_tokens,
        "orchestration_mode": orchestration_mode,
        "qwen_tool_retrieval": args.qwen_tool_retrieval,
        "fixed_kg_retrieval": args.fixed_kg_retrieval,
        "max_tool_calls": args.max_tool_calls,
        "tool_choice": args.tool_choice,
        "tool_timeout": args.tool_timeout,
        "tool_trace_dir": str(args.tool_trace_dir) if args.tool_trace_dir else None,
        "tool_max_tokens": args.tool_max_tokens,
        "termination_mode": args.termination_mode,
        "order_gap_epsilon": args.order_gap_epsilon,
        "order_gap_window": args.order_gap_window,
        "order_gap_min_iterations": args.order_gap_min_iterations,
        "kg_sessions": args.kg_sessions,
        "scallop_validator_url": args.scallop_validator_url,
        "tool_contract_version": f"{SEARCH_TOOL_VERSION}+{MEMORY_TOOL_VERSION}",
        "rule_version": DEFAULT_RULE_PARAMETERS.version,
        "cell_status": {str(cell): "pending" for cell in cells},
        "cell_retrieval": {},
        "skip_kg_build": args.skip_kg_build,
        "kg_artifacts": {},
    }
    if retrieval_config.mode == "dense_ppr":
        metadata["ppr_config"] = retrieval_config.ppr.to_dict()
    _write_manifest(args.results_dir, metadata)
    print(f"[run_all] metadata -> {args.results_dir / 'run_metadata.json'}", file=sys.stderr)

    # KG build phase
    selected_kg = [c for c in cells if c in KG_CELL_IDS]
    if selected_kg and not args.skip_kg_build:
        if not args.neo4j_password:
            print(
                "[run_all] WARNING: KG cells selected but --neo4j-password missing; "
                "skipping KG build. Cells will need --facts-file or will fail.",
                file=sys.stderr,
            )
        else:
            from experiments.build_kg import build_kg
            sessions = [(args.kg_sessions[2], False)]
            if any(c in (3, 6) for c in selected_kg):
                sessions.append((args.kg_sessions[3], True))
            candidate_path = None
            for session, validate in sessions:
                built_path = build_kg(
                    session_id=session,
                    validate=validate,
                    input_path=args.pilot_input,
                    model=args.model,
                    vllm_base_url=args.vllm_base_url,
                    api_key=args.api_key,
                    neo4j_uri=args.neo4j_uri,
                    neo4j_user=args.neo4j_user,
                    neo4j_password=args.neo4j_password,
                    limit=args.limit,
                    rebuild=args.rebuild_kg,
                    facts_out_dir=args.results_dir / "kg_builds",
                    scallop_validator_url=args.scallop_validator_url,
                    candidate_facts_path=candidate_path if validate else None,
                )
                if not validate:
                    candidate_path = built_path
                    metadata["kg_artifacts"]["candidate_path"] = str(built_path)
                    metadata["kg_artifacts"]["candidate_sha256"] = _sha256(built_path)
                else:
                    metadata["kg_artifacts"]["scallop_path"] = str(built_path)
                    metadata["kg_artifacts"]["scallop_sha256"] = _sha256(built_path)
                if retrieval_config.mode != "sparse":
                    from neurosym.adapters.dense_index import ensure_dense_index_from_snapshot

                    dense_index = ensure_dense_index_from_snapshot(
                        built_path,
                        index_root=retrieval_config.index_root,
                        session_id=session,
                        config=retrieval_config.embedding,
                    )
                    metadata["kg_artifacts"].setdefault("dense_indexes", {})[session] = (
                        dense_index.manifest.identity
                    )
            _write_manifest(args.results_dir, metadata)

    # Per-cell runs
    for cid in cells:
        mod = _module_for(cid)
        cell_args = _common_cell_args(args, cid)
        print(f"[run_all] -> {mod} {' '.join(cell_args)}", file=sys.stderr)
        cmd = [sys.executable, "-m", mod] + cell_args
        cell_env = os.environ.copy()
        cell_env["VLLM_API_KEY"] = args.api_key
        if args.neo4j_password:
            cell_env["NEO4J_PASSWORD"] = args.neo4j_password
        rc = subprocess.call(cmd, env=cell_env)
        metadata["cell_status"][str(cid)] = "complete" if rc == 0 else f"failed:{rc}"
        cell = next(cell for cell in CELLS if cell["cell_id"] == cid)
        output = args.results_dir / f"cell{cid}_{cell['label']}" / "results.jsonl"
        metadata.setdefault("cell_artifacts", {})[str(cid)] = {
            "results_path": str(output),
            "results_sha256": _sha256(output),
        }
        if cid in KG_CELL_IDS and output.exists():
            runtime_rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            cell_retrieval_eval = output.parent / "retrieval_eval.jsonl"
            if cell_retrieval_eval.exists():
                run_retrieval_eval = args.results_dir / "retrieval_eval.jsonl"
                with run_retrieval_eval.open("a", encoding="utf-8") as combined:
                    combined.write(cell_retrieval_eval.read_text(encoding="utf-8"))
            metadata["cell_retrieval"][str(cid)] = [
                {
                    "example_id": row.get("example_id"),
                    "configured_mode": row.get("configured_retrieval_mode"),
                    "effective_mode": row.get("effective_retrieval_mode"),
                    "degraded": row.get("retrieval_degraded"),
                    "dense_index_identity": row.get("dense_index_identity"),
                    "branch_counts": row.get("retrieval_branch_counts"),
                    "branch_latency_seconds": row.get(
                        "retrieval_branch_latency_seconds"
                    ),
                    "rrf": row.get("retrieval_rrf_settings"),
                }
                for row in runtime_rows
            ]
        _write_manifest(args.results_dir, metadata)
        if rc != 0:
            print(f"[run_all] cell {cid} exited with code {rc}", file=sys.stderr)

    # Aggregate once at the end
    from experiments.aggregate import main as agg_main
    agg_main(["--results-dir", str(args.results_dir)])
    from experiments.compliance import main as compliance_main
    compliance_main(["--results-dir", str(args.results_dir)])
    retrieval_eval = args.results_dir / "retrieval_eval.jsonl"
    if retrieval_eval.exists():
        from experiments.retrieval_report import main as retrieval_report_main

        retrieval_report_main(
            ["--input", str(retrieval_eval), "--output-dir", str(args.results_dir)]
        )
        metadata["retrieval_report"] = {
            "json": str(args.results_dir / "retrieval_report.json"),
            "markdown": str(args.results_dir / "retrieval_report.md"),
        }
        _write_manifest(args.results_dir, metadata)


if __name__ == "__main__":
    main()
