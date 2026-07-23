from __future__ import annotations

import json
from pathlib import Path

from experiments.run_all import _materialize_pilot_input


def test_materialize_pilot_input_freezes_one_sorted_shared_slice(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {"_id": "c", "context": "third"},
        {"_id": "a", "context": "first"},
        {"_id": "b", "context": "second"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    frozen = tmp_path / "run" / "pilot_input.jsonl"

    metadata = _materialize_pilot_input(source, frozen, limit=2, seed=0)

    selected = [json.loads(line) for line in frozen.read_text().splitlines()]
    assert [row["_id"] for row in selected] == ["a", "b"]
    assert metadata["example_count"] == 2
    assert metadata["example_ids"] == ["a", "b"]
    assert metadata["path"] == str(frozen)
    assert len(metadata["sha256"]) == 64
