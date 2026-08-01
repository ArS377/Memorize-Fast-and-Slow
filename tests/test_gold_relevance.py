"""Tests for gold-relevance labeling + PPR-relevant scoring."""
from __future__ import annotations

from experiments.gold_relevance import (
    FactIndex,
    label_rows,
    multi_hop_covered,
    recall_at_k,
    relevant_facts_for_example,
)


def _fact(fact_id, example_id, title, local_sent_id, subject="", obj=""):
    return {
        "fact_id": fact_id,
        "example_id": example_id,
        "subject": subject,
        "object": obj,
        "provenance": [{"title": title, "local_sent_id": local_sent_id}],
    }


# Two gold hops: doc "A" sent 0, doc "B" sent 1. f1 supports A, f2 supports B,
# f3 is a distractor from doc "C".
FACTS = [
    _fact("f1", "ex1", "A", 0, subject="kingdom of dreams", obj="vikramaditya motwane"),
    _fact("f2", "ex1", "B", 1, subject="vikramaditya motwane", obj="anurag kashyap"),
    _fact("f3", "ex1", "C", 0, subject="distractor", obj="noise"),
]

GOLD = {
    "ex1": {
        "supporting": [("a", 0), ("b", 1)],
        "triples": [("kingdom of dreams", "director", "vikramaditya motwane")],
        "titles": {"a", "b"},
    }
}


def _index():
    idx = FactIndex()
    for f in FACTS:
        idx.add_fact(f)
    return idx


def test_recall_at_k_basic():
    assert recall_at_k(["f1", "f3"], ["f1", "f2"], 5) == 0.5
    assert recall_at_k(["f1", "f2"], ["f1", "f2"], 10) == 1.0
    assert recall_at_k(["f3"], ["f1", "f2"], 5) == 0.0
    assert recall_at_k(["f1", "f2"], [], 5) == 0.0


def test_relevant_facts_match_support_and_triple():
    facts = _index().facts_for("ex1")
    fact_to_titles, source = relevant_facts_for_example(facts, GOLD["ex1"])
    assert set(fact_to_titles) == {"f1", "f2"}   # f3 distractor excluded
    assert fact_to_titles["f1"] == {"a"}
    assert fact_to_titles["f2"] == {"b"}
    assert source == "gold_supporting_facts+evidence_triples"


def test_multi_hop_coverage_requires_all_hops():
    facts = _index().facts_for("ex1")
    fact_to_titles, _ = relevant_facts_for_example(facts, GOLD["ex1"])
    gold_titles = {"a", "b"}
    # Only the first hop retrieved -> not covered.
    assert multi_hop_covered(["f1", "f3"], fact_to_titles, gold_titles, 5) is False
    # Both hops retrieved -> covered.
    assert multi_hop_covered(["f1", "f2"], fact_to_titles, gold_titles, 5) is True
    # Second hop below the k cutoff -> not covered.
    assert multi_hop_covered(["f1", "f3", "f2"], fact_to_titles, gold_titles, 2) is False


def test_label_rows_scores_modes():
    rows = [
        {"example_id": "ex1", "mode": "dense", "retrieved_fact_ids": ["f1", "f3"]},
        {"example_id": "ex1", "mode": "dense_ppr", "retrieved_fact_ids": ["f1", "f2"]},
    ]
    labeled, report = label_rows(rows, _index(), GOLD, k_values=(1, 5))
    dense = next(r for r in labeled if r["mode"] == "dense")
    ppr = next(r for r in labeled if r["mode"] == "dense_ppr")

    assert dense["relevant_fact_ids"] == ["f1", "f2"]
    assert dense["relevance_source"] == "gold_supporting_facts+evidence_triples"
    assert dense["recall_at_5"] == 0.5
    assert dense["multi_hop_covered_at_5"] is False
    # PPR reaches the bridge fact -> full recall and hop coverage.
    assert ppr["recall_at_5"] == 1.0
    assert ppr["multi_hop_covered_at_5"] is True

    assert report["modes"]["dense_ppr"]["recall_at_5"] == 1.0
    assert report["modes"]["dense"]["multi_hop_covered_at_5"] == 0.0


def test_unlabeled_when_no_gold_for_example():
    rows = [{"example_id": "unknown", "mode": "dense", "retrieved_fact_ids": ["f1"]}]
    labeled, _ = label_rows(rows, _index(), GOLD)
    assert labeled[0]["relevance_source"] == "unlabeled"
    assert labeled[0]["relevant_fact_ids"] == []
