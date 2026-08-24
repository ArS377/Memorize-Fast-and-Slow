from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def _exact_match(prediction: str, answer: str) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(answer) else 0.0


def _substring_match(prediction: str, answer: str) -> float:
    pred = _normalize_text(prediction)
    gold = _normalize_text(answer)
    if not pred or not gold:
        return 0.0
    return 1.0 if gold in pred or pred in gold else 0.0


def _token_f1(prediction: str, answer: str) -> float:
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(answer)
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_example(prediction: str, answers: List[str]) -> Dict[str, Any]:
    """
    Score one prediction against one or more gold answers.
    Uses max-over-answers for each metric.
    """
    valid_answers = [str(a) for a in answers if str(a).strip()]
    if not valid_answers:
        return {
            "exact_match": 0.0,
            "substring_match": 0.0,
            "token_f1": 0.0,
            "best_answer": "",
            "num_answers": 0,
        }

    best_em = 0.0
    best_sub = 0.0
    best_f1 = 0.0
    best_answer = valid_answers[0]
    best_total = -1.0

    for answer in valid_answers:
        em = _exact_match(prediction, answer)
        sub = _substring_match(prediction, answer)
        f1 = _token_f1(prediction, answer)
        total = em + sub + f1

        best_em = max(best_em, em)
        best_sub = max(best_sub, sub)
        best_f1 = max(best_f1, f1)

        if total > best_total:
            best_total = total
            best_answer = answer

    return {
        "exact_match": best_em,
        "substring_match": best_sub,
        "token_f1": best_f1,
        "best_answer": best_answer,
        "num_answers": len(valid_answers),
    }

