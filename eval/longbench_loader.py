from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _as_answer_list(example: Dict[str, Any]) -> List[str]:
    raw = example.get("answers")
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is not None and str(raw).strip():
        return [str(raw).strip()]

    answer = example.get("answer")
    if answer is not None and str(answer).strip():
        return [str(answer).strip()]
    return []


def normalize_example(example: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    return {
        "id": str(example.get("_id", example.get("id", ""))).strip(),
        "dataset_name": dataset_name,
        "question": str(example.get("question", "")).strip(),
        "context": example.get("context", ""),
        "answers": _as_answer_list(example),
        "raw_example": example,
    }


def _load_from_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
        return rows

    data = json.loads(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "examples", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [data]
    raise ValueError("Input must be JSON object/list or JSONL file.")


def _load_from_hf(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  
    except ImportError as exc:
        raise RuntimeError("datasets package not installed. Run: pip install datasets") from exc
    ds = load_dataset(dataset_name, split=split)
    return [dict(x) for x in ds]


def load_examples(
    dataset_path: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not dataset_path and not dataset_name:
        raise ValueError("Provide dataset_path or dataset_name.")

    if dataset_path:
        raw = _load_from_json_or_jsonl(dataset_path)
        source_name = dataset_name or dataset_path.stem
    else:
        assert dataset_name is not None
        raw = _load_from_hf(dataset_name, split)
        source_name = dataset_name

    out = [normalize_example(x, source_name) for x in raw]
    if max_examples is not None:
        out = out[:max_examples]
    return out

