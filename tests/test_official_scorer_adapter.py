#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.official_scorer_adapter import to_official_scorer_row, write_official_scorer_input


def main() -> int:
    row = {
        "prediction": "B",
        "answers": ["B"],
        "raw_example": {
            "choice_A": "Paris",
            "choice_B": "London",
            "choice_C": "Tokyo",
            "choice_D": "Berlin",
            "length": 4096,
        },
    }
    converted = to_official_scorer_row(row)
    assert converted["pred"] == "B", "pred mapping failed"
    assert converted["answers"] == ["B"], "answers mapping failed"
    assert len(converted["all_classes"]) == 4, "class extraction failed"
    assert converted["length"] == 4096, "length extraction failed"

    out_path = Path("official_adapter_smoke.jsonl")
    write_official_scorer_input(out_path, [row])
    lines = [x for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1, "expected one output line"
    parsed = json.loads(lines[0])
    assert parsed["pred"] == "B", "written file pred mismatch"
    print("Official scorer adapter smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

