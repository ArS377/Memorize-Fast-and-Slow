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
