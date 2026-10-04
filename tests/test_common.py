from __future__ import annotations

import json

from neurosym.application.experiment_io import iter_pilot_examples


def test_limited_pilot_selection_is_sorted_and_stable(tmp_path) -> None:
    path = tmp_path / "examples.jsonl"
    rows = [
        {"_id": "c", "value": 1},
        {"_id": "a", "value": 2},
        {"_id": "b", "value": 3},
        {"_id": "a", "value": 4},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = iter_pilot_examples(path, limit=3)

    assert [(row["_id"], row["value"]) for row in selected] == [
        ("a", 2),
        ("a", 4),
        ("b", 3),
    ]


def test_zero_limit_does_not_read_the_dataset(tmp_path) -> None:
    missing = tmp_path / "missing.jsonl"

    assert iter_pilot_examples(missing, limit=0) == []
