from __future__ import annotations

from experiments.aggregate import _summarize_rows


def test_diagnostic_accuracy_is_separate_from_official_accuracy() -> None:
    summary = _summarize_rows(
        [
            {
                "predicted": None,
                "correct": False,
                "error": "evidence_insufficient",
                "diagnostic_predicted": "C",
                "diagnostic_correct": True,
            },
            {
                "predicted": "A",
                "correct": False,
                "error": None,
                "diagnostic_predicted": "A",
                "diagnostic_correct": False,
            },
        ]
    )

    assert summary["n_answered"] == 1
    assert summary["accuracy"] == 0.0
    assert summary["n_diagnostic_answered"] == 2
    assert summary["diagnostic_accuracy"] == 0.5


def test_cell6_reports_overall_grounded_and_fallback_accuracy() -> None:
    summary = _summarize_rows(
        [
            {
                "predicted": "A",
                "correct": True,
                "error": None,
                "outcome_status": "supported",
            },
            {
                "predicted": "B",
                "correct": True,
                "error": None,
                "outcome_status": "unsupported_fallback",
            },
            {
                "predicted": "C",
                "correct": False,
                "error": None,
                "outcome_status": "unsupported_fallback",
            },
            {
                "predicted": None,
                "correct": False,
                "error": "fallback failed",
                "outcome_status": "fallback_failed",
            },
        ]
    )

    assert summary["accuracy"] == 0.5
    assert summary["n_grounded"] == 1
    assert summary["grounded_accuracy"] == 1.0
    assert summary["grounded_coverage"] == 0.25
    assert summary["n_fallback"] == 2
    assert summary["fallback_accuracy"] == 0.5
