"""Orchestrator: run all six ablation cells end-to-end.

Pipeline:

1. Resolve config and write ``results/run_metadata.json``.
2. Optionally build the two shared KG sessions (``pilot_noscallop`` /
   ``pilot_scallop``) once; cells 2/5 share the first, cells 3/6 share
   the second.
3. Run each requested cell as a standalone module call (passing
   ``--no-aggregate`` so we aggregate just once at the end).
4. Aggregate -> ``results/summary.csv`` + ``results/figures/accuracy_grid.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from experiments.common import CELLS

KG_CELL_IDS = {2, 3, 5, 6}
KG_SESSIONS = {2: "pilot_noscallop", 3: "pilot_scallop",
               5: "pilot_noscallop", 6: "pilot_scallop"}


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
        "--input", str(args.input),
        "--limit", str(args.limit),
        "--seed", str(args.seed),
        "--model", args.model,
        "--vllm-base-url", args.vllm_base_url,
        "--api-key", args.api_key,
        "--results-dir", str(args.results_dir),
        "--no-aggregate",
    ]
    if cell_id in KG_CELL_IDS:
        if args.neo4j_uri:
            base += ["--neo4j-uri", args.neo4j_uri]
        if args.neo4j_user:
            base += ["--neo4j-user", args.neo4j_user]
        if args.neo4j_password:
            base += ["--neo4j-password", args.neo4j_password]
        facts_file = args.results_dir / "kg_builds" / f"{KG_SESSIONS[cell_id]}_facts.jsonl"
        if facts_file.exists():
            base += ["--facts-file", str(facts_file)]
        if args.hops is not None:
            base += ["--hops", str(args.hops)]
        if args.limit_triples is not None:
            base += ["--limit-triples", str(args.limit_triples)]
        base += ["--memory-scope", args.memory_scope]
    else:
        if args.raw_max_chars is not None:
            base += ["--raw-max-chars", str(args.raw_max_chars)]
    if cell_id in (4, 5, 6):
        base += [
            "--max-depth", str(args.max_depth),
            "--max-iterations", str(args.max_iterations),
            "--max-tokens", str(args.max_tokens),
        ]
        if cell_id in (5, 6):
            if not getattr(args, "fixed_kg_retrieval", False):
                base += [
                    "--qwen-tool-retrieval",
                    "--max-tool-calls", str(args.max_tool_calls),
                    "--tool-choice", args.tool_choice,
                    "--tool-timeout", str(args.tool_timeout),
                    "--tool-max-tokens", str(args.tool_max_tokens),
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
    parser.add_argument("--cells", default="all",
                        help="Comma-separated cell ids, e.g. '1,3,5'. Default: all six.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--raw-max-chars", type=int, default=32000)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--limit-triples", type=int, default=50)
    parser.add_argument("--memory-scope", choices=["example", "session"], default="example")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64000)
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
    args = parser.parse_args(argv)

    cells = _parse_cells(args.cells)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    orchestration_mode = "fixed" if args.fixed_kg_retrieval else "qwen_native_tools_inside_rlm"

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "input": str(args.input),
        "limit": args.limit,
        "seed": args.seed,
        "model": args.model,
        "vllm_base_url": args.vllm_base_url,
        "neo4j_uri": args.neo4j_uri,
        "cells": cells,
        "raw_max_chars": args.raw_max_chars,
        "hops": args.hops,
        "limit_triples": args.limit_triples,
        "memory_scope": args.memory_scope,
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
    }
    (args.results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
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
            sessions = []
            if any(c in (2, 5) for c in selected_kg):
                sessions.append(("pilot_noscallop", False))
            if any(c in (3, 6) for c in selected_kg):
                sessions.append(("pilot_scallop", True))
            for session, validate in sessions:
                build_kg(
                    session_id=session,
                    validate=validate,
                    input_path=args.input,
                    model=args.model,
                    vllm_base_url=args.vllm_base_url,
                    api_key=args.api_key,
                    neo4j_uri=args.neo4j_uri,
                    neo4j_user=args.neo4j_user,
                    neo4j_password=args.neo4j_password,
                    limit=args.limit,
                    rebuild=args.rebuild_kg,
                    facts_out_dir=args.results_dir / "kg_builds",
                )

    # Per-cell runs
    for cid in cells:
        mod = _module_for(cid)
        cell_args = _common_cell_args(args, cid)
        print(f"[run_all] -> {mod} {' '.join(cell_args)}", file=sys.stderr)
        cmd = [sys.executable, "-m", mod] + cell_args
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[run_all] cell {cid} exited with code {rc}", file=sys.stderr)

    # Aggregate once at the end
    from experiments.aggregate import main as agg_main
    agg_main(["--results-dir", str(args.results_dir)])


if __name__ == "__main__":
    main()
