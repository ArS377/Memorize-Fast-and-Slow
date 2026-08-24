from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _extract_all_classes(row: Dict[str, Any]) -> List[str]:
    raw_example = row.get("raw_example", {})
    if not isinstance(raw_example, dict):
        return []
    keys = ("choice_A", "choice_B", "choice_C", "choice_D")
    classes = [str(raw_example.get(k, "")).strip() for k in keys]
    return [x for x in classes if x]


def to_official_scorer_row(row: Dict[str, Any]) -> Dict[str, Any]:
    prediction = str(row.get("prediction", "")).strip()
    answers = row.get("answers", [])
    if not isinstance(answers, list):
        answers = [str(answers)]
    answers = [str(x).strip() for x in answers if str(x).strip()]
    if not prediction:
        raise ValueError("Missing prediction for scorer row.")
    if not answers:
        raise ValueError("Missing answers for scorer row.")

    scorer_row: Dict[str, Any] = {
        "pred": prediction,
        "answers": answers,
        "all_classes": _extract_all_classes(row),
    }
    raw_example = row.get("raw_example", {})
    if isinstance(raw_example, dict) and raw_example.get("length") is not None:
        scorer_row["length"] = raw_example.get("length")
    return scorer_row


def write_official_scorer_input(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            scorer_row = to_official_scorer_row(row)
            f.write(json.dumps(scorer_row, ensure_ascii=False) + "\n")
    return path

