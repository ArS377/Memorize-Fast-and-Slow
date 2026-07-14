"""Run the committed sparse fixtures through the live Qwen-first RLM path."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.graph_context import GraphSource
from experiments.rlm_retrieval import QwenRLMToolOutcome, qwen_rlm_tool_answer


class _TimeoutGraphSource(GraphSource):
    def rows_for(self, **kwargs: Any) -> List[Dict[str, Any]]:
        raise TimeoutError("simulated sparse-smoke timeout")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(case, dict) for case in value):
        raise ValueError("fixtures must contain a JSON array of objects")
    return value


def _example(case: Dict[str, Any]) -> Dict[str, Any]:
    choices = case.get("choices", {})
    choices = choices if isinstance(choices, dict) else {}
    return {
        "_id": str(case.get("id", "")),
        "question": str(case.get("question", "")),
        "choice_A": str(choices.get("A", "[not provided]")),
        "choice_B": str(choices.get("B", "[not provided]")),
        "choice_C": str(choices.get("C", "[not provided]")),
        "choice_D": str(choices.get("D", "[not provided]")),
    }


def _source(case: Dict[str, Any]) -> GraphSource:
    source_type = _TimeoutGraphSource if case.get("simulate_timeout") else GraphSource
    return source_type(
        fallback_facts=list(case.get("facts", [])),
        session_id="sparse-smoke",
        memory_scope="example",
    )


def _trace_states(trace: Dict[str, Any]) -> tuple[List[str], int]:
    error_codes: List[str] = []
    valid_empty_count = 0
    for event in trace.get("events", []):
        if not isinstance(event, dict) or event.get("event") != "tool_result":
            continue
        response = event.get("response")
        if not isinstance(response, dict):
            continue
        if response.get("status") == "ok" and response.get("results") == []:
            valid_empty_count += 1
        error = response.get("error")
        if isinstance(error, dict) and error.get("code"):
            error_codes.append(str(error["code"]))
    return error_codes, valid_empty_count


def smoke_record(
    case: Dict[str, Any],
    outcome: QwenRLMToolOutcome,
    *,
    latency_seconds: float,
) -> Dict[str, Any]:
    tool_error_codes, valid_empty_count = _trace_states(outcome.trace)
    expected = [str(value) for value in case.get("expected_fact_ids", [])]
    forbidden = {str(value) for value in case.get("forbidden_fact_ids", [])}
    leaked = sorted(forbidden.intersection(outcome.retrieved_fact_ids))
    gold = str(case.get("gold", "")).strip().upper()
    labelled = gold in {"A", "B", "C", "D"}
    error = outcome.error
    if leaked:
        error = f"scope violation; retrieved forbidden fact IDs: {', '.join(leaked)}"

    return {
        "example_id": str(case.get("id", "")),
        "status": outcome.status,
        "predicted": outcome.predicted,
        "gold": gold,
        "correct": (
            outcome.predicted == gold and not error
            if labelled
            else None
        ),
        "expected_fact_ids": expected,
        "forbidden_fact_ids": sorted(forbidden),
        "retrieved_fact_ids": list(outcome.retrieved_fact_ids),
        "cited_fact_ids": list(outcome.cited_fact_ids),
        "retrieved_fact_count": outcome.retrieved_fact_count,
        "tool_call_count": outcome.tool_call_count,
        "no_hit": (
            outcome.retrieved_fact_count == 0
            and valid_empty_count > 0
            and not tool_error_codes
        ),
        "valid_empty_result_count": valid_empty_count,
        "tool_error_codes": tool_error_codes,
        "termination_reason": outcome.termination_reason,
        "latency_seconds": round(latency_seconds, 6),
        "error": error,
        "trace": outcome.trace,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--tool-choice", choices=["auto", "required"], default="auto")
    parser.add_argument("--tool-timeout", type=float, default=30.0)
    parser.add_argument("--tool-max-tokens", type=int, default=2048)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    cases = load_cases(args.fixtures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_root = args.log_dir or args.output.parent / "qwen_sparse_rlm_logs"
    records: List[Dict[str, Any]] = []

    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            example_id = str(case.get("id", ""))
            safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", example_id) or "example"
            source = _source(case)
            started = time.perf_counter()
            try:
                outcome = qwen_rlm_tool_answer(
                    backend="openai",
                    model=args.model,
                    base_url=args.vllm_base_url,
                    api_key=args.api_key,
                    max_depth=args.max_depth,
                    max_iterations=args.max_iterations,
                    max_tokens=args.max_tokens,
                    log_dir=log_root / safe_id,
                    verbose=args.verbose,
                    graph_source=source,
                    example=_example(case),
                    max_tool_calls=args.max_tool_calls,
                    tool_choice=args.tool_choice,
                    tool_timeout=args.tool_timeout,
                    max_completion_tokens=args.tool_max_tokens,
                )
                record = smoke_record(
                    case,
                    outcome,
                    latency_seconds=time.perf_counter() - started,
                )
            finally:
                source.close()
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if records and not any(record["tool_call_count"] > 0 for record in records):
        raise RuntimeError("live sparse smoke completed without a native tool call")
    violations = [
        record["example_id"]
        for record in records
        if record["error"] and "scope violation" in record["error"]
    ]
    if violations:
        raise RuntimeError(f"scope isolation failed for: {', '.join(violations)}")
    return args.output


if __name__ == "__main__":
    main()
