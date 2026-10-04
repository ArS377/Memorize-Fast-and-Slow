"""Tests for free-form answer scoring (EM / token-F1)."""
from __future__ import annotations

from experiments.answer_eval import (
    evaluate_predictions,
    exact_match,
    f1_score,
    normalize_answer,
)


def test_normalize_strips_articles_punctuation_case():
    assert normalize_answer("The Anurag Kashyap.") == "anurag kashyap"
    assert normalize_answer("YES") == "yes"


def test_exact_match_is_normalization_insensitive():
    assert exact_match("Paris, France", "Paris") == 0.0
    assert exact_match("Anurag Kashyap", "the anurag  kashyap") == 1.0


def test_f1_partial_overlap():
    # 2 of 3 predicted tokens overlap 2 of 2 gold -> P=2/3, R=1 -> F1=0.8
    assert round(f1_score("Anurag Kashyap Sr", "Anurag Kashyap"), 3) == 0.8
    assert f1_score("completely wrong", "anurag kashyap") == 0.0


def test_evaluate_predictions_aggregate():
    golds = {"a": "yes", "b": "Anurag Kashyap", "c": "Paris"}
    preds = {"a": "yes", "b": "Anurag Kashyap", "c": ""}  # c unanswered
    result = evaluate_predictions(preds, golds)
    assert result["n"] == 3
    assert result["answered"] == 2
    assert round(result["exact_match"], 4) == round(2 / 3, 4)
