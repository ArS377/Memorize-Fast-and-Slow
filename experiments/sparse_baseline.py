"""Metrics for the Phase 3 sparse-tool smoke baseline.

This module deliberately reads persisted run records instead of reaching into
the retriever.  Dense and hybrid experiments can therefore use the same
evaluator while keeping their metrics in a different output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


BASELINE_METRIC_FIELDS = (
    "examples",
    "hit_rate",
    "no_hit_rate",
    "recall_at_k",
    "answer_accuracy",
    "mean_tool_calls",
    "mean_latency_seconds",
    "error_rate",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load non-empty JSONL records from a persisted smoke trace file."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fact_ids(record: Dict[str, Any]) -> set[str]:
    """Read retrieved IDs from either the summary or individual trace events."""
    ids = {str(value) for value in record.get("retrieved_fact_ids", []) if value}
    for event in record.get("trace", []):
        if not isinstance(event, dict):
            continue
        ids.update(str(value) for value in event.get("fact_ids", []) if value)
        result = event.get("result")
        if isinstance(result, dict):
            ids.update(
                str(item.get("fact_id"))
                for item in result.get("results", [])
                if isinstance(item, dict) and item.get("fact_id")
            )
    return ids


def compute_sparse_baseline(records: Iterable[Dict[str, Any]]) -> Dict[str, float | int]:
    """Compute the fixed Phase 3 metrics from persisted per-example records.

    ``expected_fact_ids`` is optional: examples without labels are excluded
    from Recall@k rather than being counted as failures.  A valid no-hit must
    be represented by ``no_hit=True`` (or zero retrieved facts); errors remain
    visible separately in ``error_rate``.
    """
    rows = list(records)
    total = len(rows)
    if not total:
        return {field: 0 for field in BASELINE_METRIC_FIELDS}

    hits = 0
    no_hits = 0
    correct = 0
    errors = 0
    calls: List[float] = []
    latencies: List[float] = []
    recall_scores: List[float] = []

    for row in rows:
        fact_ids = _fact_ids(row)
        retrieved_count = int(row.get("retrieved_fact_count", len(fact_ids)) or 0)
        is_no_hit = bool(row.get("no_hit", retrieved_count == 0))
        hits += int(retrieved_count > 0)
        no_hits += int(is_no_hit)
        correct += int(bool(row.get("correct", row.get("predicted") == row.get("gold"))))
        errors += int(bool(row.get("error")))
        calls.append(float(row.get("tool_call_count", 0) or 0))
        latencies.append(float(row.get("latency_seconds", row.get("elapsed_seconds", 0)) or 0))

        expected = {str(value) for value in row.get("expected_fact_ids", []) if value}
        if expected:
            recall_scores.append(len(fact_ids & expected) / len(expected))

    return {
        "examples": total,
        "hit_rate": round(hits / total, 4),
        "no_hit_rate": round(no_hits / total, 4),
        "recall_at_k": round(mean(recall_scores), 4) if recall_scores else 0.0,
        "answer_accuracy": round(correct / total, 4),
        "mean_tool_calls": round(mean(calls), 4),
        "mean_latency_seconds": round(mean(latencies), 4),
        "error_rate": round(errors / total, 4),
    }


def main(argv: List[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Evaluate a sparse tool-call smoke trace.")
    parser.add_argument("--input", required=True, type=Path, help="JSONL trace from a sparse smoke run")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sparse_baseline/metrics.json"),
        help="Baseline metrics output; keep separate from dense/hybrid results",
    )
    args = parser.parse_args(argv)
    metrics = compute_sparse_baseline(load_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return args.output


if __name__ == "__main__":
    main()
