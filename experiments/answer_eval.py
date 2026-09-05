"""Free-form answer scoring for HotpotQA / 2WikiMultihopQA.

LongBench-v2 is multiple choice, so the pipeline scores answers by letter
equality (``predicted == gold``). HotpotQA and 2Wiki are free-form span
answers, whose standard metric is SQuAD-style Exact Match + token-level F1 over
a normalized string (lowercase, strip articles/punctuation/extra whitespace).
This module provides exactly that so a produced answer can be scored the way the
datasets' leaderboards do.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, drop punctuation/articles, squash spaces."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        # If either is empty, F1 is 1 only when both are empty (e.g. yes/no edge).
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(
    predictions: Mapping[str, str],
    golds: Mapping[str, str],
) -> Dict[str, float]:
    """Score predictions against gold answers keyed by example id."""
    ems: List[float] = []
    f1s: List[float] = []
    for example_id, gold in golds.items():
        pred = predictions.get(example_id, "")
        ems.append(exact_match(pred, gold))
        f1s.append(f1_score(pred, gold))
    n = len(golds)
    return {
        "exact_match": round(mean(ems), 6) if ems else 0.0,
        "f1": round(mean(f1s), 6) if f1s else 0.0,
        "n": n,
        "answered": sum(1 for i in golds if predictions.get(i)),
    }


def _load_golds(dataset_path: Path) -> Dict[str, str]:
    golds: Dict[str, str] = {}
    with Path(dataset_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                golds[str(row.get("_id"))] = str(row.get("answer") or "")
    return golds


def _load_predictions(pred_path: Path) -> Dict[str, str]:
    """Accept either {id: answer} JSON or JSONL rows with _id/example_id + predicted."""
    text = Path(pred_path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return {str(k): str(v) for k, v in json.loads(text).items()}
    preds: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        example_id = str(row.get("_id") or row.get("example_id") or "")
        preds[example_id] = str(row.get("predicted") or row.get("answer") or "")
    return preds


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score free-form answers (EM / token-F1).")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Normalized dataset jsonl (gold answers)")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    golds = _load_golds(args.dataset)
    preds = _load_predictions(args.predictions)
    result = evaluate_predictions(preds, golds)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
