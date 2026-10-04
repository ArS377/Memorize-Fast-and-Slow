"""Attach gold relevance labels to a retrieval eval and score PPR-relevant metrics.

This is the piece the dense_ppr sweep was missing: independent relevance labels.
``retrieval_eval.jsonl`` records, per query, which fact IDs each retrieval mode
returned (``retrieved_fact_ids``) but leaves ``relevance_source = "unlabeled"``
so Recall@k is uncomputable. For HotpotQA / 2WikiMultihopQA we DO have gold
labels, so this module maps them onto the extracted KG facts and rewrites each
row with ``relevant_fact_ids`` + a real ``relevance_source``.

Two mappings (a fact is *relevant* to an example if either fires):

1. ``gold_supporting_facts`` — a fact whose provenance ``title`` matches a gold
   supporting document AND whose ``local_sent_id`` matches the gold sentence
   index (title-only fallback when the sentence index is unavailable). Works for
   both datasets.
2. ``gold_evidence_triples`` — a fact whose ``(subject, object)`` entity pair
   matches a gold ``(subject, relation, object)`` evidence triple after
   normalization. 2Wiki only. Relation strings are compared loosely because the
   extractor's predicate vocabulary differs from 2Wiki's.

Beyond Recall@k it reports **multi-hop coverage**: the fraction of queries whose
top-k retrieval surfaces a relevant fact from *every* gold supporting document
(i.e. all hops present) — the signal PPR is supposed to improve by walking to
bridge-document facts that dense scoring alone misses.

CLI
---
    python -m experiments.gold_relevance \
        --retrieval-eval results/<run>/retrieval_eval.jsonl \
        --facts results/kg_builds/<session>_facts.jsonl \
        --dataset data_2wiki.jsonl \
        --out results/<run>/retrieval_eval.labeled.jsonl \
        --report results/<run>/gold_relevance_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _norm(value: Any) -> str:
    text = _PUNCT.sub(" ", str(value or "").lower())
    return _WS.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Fact index
# --------------------------------------------------------------------------- #
class FactIndex:
    """Per-example view of extracted facts for gold-label matching."""

    def __init__(self) -> None:
        # example_id -> fact_id -> {"titles": {(title, local_sent_id)}, "entities": (subj, obj)}
        self._by_example: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def add_fact(self, fact: Mapping[str, Any]) -> None:
        example_id = str(fact.get("example_id") or "")
        fact_id = str(fact.get("fact_id") or "")
        if not fact_id:
            return
        title_sents: Set[Tuple[str, Any]] = set()
        title_only: Set[str] = set()
        for entry in fact.get("provenance", []) or []:
            if not isinstance(entry, Mapping):
                continue
            title = _norm(entry.get("title"))
            if not title:
                continue
            title_only.add(title)
            local = entry.get("local_sent_id")
            if local is None:
                local = entry.get("sent_id")
            title_sents.add((title, local))
        self._by_example[example_id][fact_id] = {
            "title_sents": title_sents,
            "title_only": title_only,
            "subject": _norm(fact.get("subject")),
            "object": _norm(fact.get("object")),
        }

    def facts_for(self, example_id: str) -> Dict[str, Dict[str, Any]]:
        return self._by_example.get(str(example_id), {})

    @classmethod
    def from_jsonl(cls, paths: Iterable[Path]) -> "FactIndex":
        index = cls()
        for path in paths:
            with Path(path).open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        index.add_fact(json.loads(line))
        return index


# --------------------------------------------------------------------------- #
# Gold labels
# --------------------------------------------------------------------------- #
def load_gold(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    """example_id -> {supporting: [(title, sent_id)], triples: [(subj, rel, obj)], titles: {title}}"""
    gold: Dict[str, Dict[str, Any]] = {}
    with Path(dataset_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example_id = str(row.get("_id") or "")
            supporting = [
                (_norm(pair[0]), pair[1] if len(pair) > 1 else None)
                for pair in row.get("gold_supporting_facts", [])
                if isinstance(pair, (list, tuple)) and pair
            ]
            triples = [
                (_norm(t[0]), _norm(t[1]), _norm(t[2]))
                for t in row.get("gold_evidence_triples", [])
                if isinstance(t, (list, tuple)) and len(t) >= 3
            ]
            gold[example_id] = {
                "supporting": supporting,
                "triples": triples,
                "titles": {title for title, _ in supporting},
            }
    return gold


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def relevant_facts_for_example(
    facts: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, Any],
) -> Tuple[Dict[str, Set[str]], str]:
    """Return (fact_id -> set(gold titles it supports), relevance_source)."""
    supporting = gold.get("supporting", [])
    triples = gold.get("triples", [])
    support_pairs = set(supporting)
    support_titles = {title for title, _ in supporting}

    fact_to_titles: Dict[str, Set[str]] = defaultdict(set)
    matched_via_triple = False

    for fact_id, meta in facts.items():
        # 1. Supporting-fact match (title + sentence, then title-only fallback).
        for title, local in meta["title_sents"]:
            if (title, local) in support_pairs or (title, None) in support_pairs:
                fact_to_titles[fact_id].add(title)
        if fact_id not in fact_to_titles:
            for title in meta["title_only"]:
                if title in support_titles:
                    fact_to_titles[fact_id].add(title)

        # 2. Evidence-triple match (2Wiki): subject+object entity pair. A triple
        # match marks the fact relevant even if it carries no provenance title
        # (it just contributes to Recall@k, not to hop coverage).
        if triples:
            subj, obj = meta["subject"], meta["object"]
            for t_subj, _t_rel, t_obj in triples:
                if subj and obj and {subj, obj} == {t_subj, t_obj}:
                    fact_to_titles.setdefault(fact_id, set())
                    matched_via_triple = True
                    break

    if triples and matched_via_triple and support_pairs:
        source = "gold_supporting_facts+evidence_triples"
    elif triples and matched_via_triple:
        source = "gold_evidence_triples"
    else:
        source = "gold_supporting_facts"
    return fact_to_titles, source


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = {str(r) for r in relevant if str(r)}
    if not relevant_set:
        return 0.0
    top = {str(r) for r in list(retrieved)[:k]}
    return len(relevant_set & top) / len(relevant_set)


def multi_hop_covered(
    retrieved: Sequence[str],
    fact_to_titles: Mapping[str, Set[str]],
    gold_titles: Set[str],
    k: int,
) -> bool:
    """True iff top-k surfaces a relevant fact from every gold supporting doc."""
    if not gold_titles:
        return False
    covered: Set[str] = set()
    for fact_id in list(retrieved)[:k]:
        covered |= fact_to_titles.get(str(fact_id), set())
    return gold_titles.issubset(covered)


def label_rows(
    rows: Iterable[Mapping[str, Any]],
    fact_index: FactIndex,
    gold: Mapping[str, Dict[str, Any]],
    k_values: Sequence[int] = (1, 5, 10),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Augment each retrieval-eval row with gold labels; return (rows, report)."""
    labeled: List[Dict[str, Any]] = []
    per_mode: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    # Cache per-example matching (independent of retrieval mode).
    cache: Dict[str, Tuple[Dict[str, Set[str]], Set[str], str]] = {}

    for row in rows:
        row = dict(row)
        example_id = str(row.get("example_id") or "")
        mode = str(row.get("mode") or row.get("retrieval_mode") or "unknown")
        if example_id not in cache:
            facts = fact_index.facts_for(example_id)
            gold_ex = gold.get(example_id, {"supporting": [], "triples": [], "titles": set()})
            fact_to_titles, source = relevant_facts_for_example(facts, gold_ex)
            cache[example_id] = (fact_to_titles, set(gold_ex["titles"]), source)
        fact_to_titles, gold_titles, source = cache[example_id]

        relevant_ids = sorted(fact_to_titles.keys())
        retrieved = [str(x) for x in row.get("retrieved_fact_ids", [])]
        row["relevant_fact_ids"] = relevant_ids
        row["relevance_source"] = source if relevant_ids else "unlabeled"
        row["gold_supporting_titles"] = sorted(gold_titles)

        if relevant_ids:
            for k in k_values:
                r = recall_at_k(retrieved, relevant_ids, k)
                row[f"recall_at_{k}"] = round(r, 6)
                per_mode[mode][f"recall_at_{k}"].append(r)
            for k in k_values:
                covered = multi_hop_covered(retrieved, fact_to_titles, gold_titles, k)
                row[f"multi_hop_covered_at_{k}"] = covered
                per_mode[mode][f"multi_hop_covered_at_{k}"].append(1.0 if covered else 0.0)
        labeled.append(row)

    report = {
        "k_values": list(k_values),
        "modes": {
            mode: {
                "labeled_queries": len(next(iter(metrics.values()), [])),
                **{name: round(mean(values), 6) for name, values in metrics.items() if values},
            }
            for mode, metrics in sorted(per_mode.items())
        },
    }
    return labeled, report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Attach gold relevance labels and score PPR metrics.")
    parser.add_argument("--retrieval-eval", type=Path, required=True)
    parser.add_argument("--facts", type=Path, nargs="+", required=True,
                        help="One or more <session>_facts.jsonl files")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Normalized dataset jsonl with gold_* fields")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the labeled retrieval_eval (default: alongside input)")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args(argv)

    fact_index = FactIndex.from_jsonl(args.facts)
    gold = load_gold(args.dataset)
    rows = _read_jsonl(args.retrieval_eval)
    labeled, report = label_rows(rows, fact_index, gold, k_values=tuple(args.k))

    out_path = args.out or args.retrieval_eval.with_suffix(".labeled.jsonl")
    with out_path.open("w", encoding="utf-8") as handle:
        for row in labeled:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(labeled)} labeled rows → {out_path}")

    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report → {args.report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
