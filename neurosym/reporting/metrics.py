from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_experiment_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = list(rows)
    answered = [row for row in values if row.get("predicted") and not row.get("error")]
    diagnostic_answered = [row for row in values if row.get("diagnostic_predicted")]
    correct = sum(1 for row in answered if row.get("correct"))
    diagnostic_correct = sum(
        1 for row in diagnostic_answered if row.get("diagnostic_correct")
    )
    latencies: List[float] = [
        float(row["elapsed_seconds"])
        for row in values
        if isinstance(row.get("elapsed_seconds"), (int, float))
    ]
    triples: List[float] = [
        float(row["n_triples"])
        for row in values
        if isinstance(row.get("n_triples"), (int, float))
    ]
    return {
        "n_examples": len(values),
        "n_answered": len(answered),
        "accuracy": correct / len(answered) if answered else None,
        "mean_latency_s": _mean(latencies),
        "mean_triples": _mean(triples),
        "n_errors": sum(1 for row in values if row.get("error")),
        "n_diagnostic_answered": len(diagnostic_answered),
        "diagnostic_accuracy": (
            diagnostic_correct / len(diagnostic_answered)
            if diagnostic_answered
            else None
        ),
    }
