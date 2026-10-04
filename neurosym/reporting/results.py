from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: Path, *, missing_ok: bool = False) -> Dict[str, Any]:
    source = Path(path)
    if missing_ok and not source.exists():
        return {}
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def read_jsonl(path: Path, *, missing_ok: bool = False) -> List[Dict[str, Any]]:
    source = Path(path)
    if missing_ok and not source.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number} in {source}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Invalid JSONL object at line {line_number} in {source}")
            rows.append(value)
    return rows


def read_result_rows(
    paths: Iterable[Path],
    *,
    missing_ok: bool = False,
) -> List[Dict[str, Any]]:
    return [
        row
        for path in paths
        for row in read_jsonl(Path(path), missing_ok=missing_ok)
    ]
