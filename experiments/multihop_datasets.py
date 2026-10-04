"""Normalize multi-hop QA datasets (HotpotQA, 2WikiMultihopQA) into the shared
NeuroSym example schema so the existing KG build + retrieval eval can consume
them and so gold evidence maps back to extracted fact IDs.

Why this exists
---------------
LongBench-v2 is single-passage multiple-choice with no evidence labels, so
Recall@k and multi-hop coverage are uncomputable and PPR has no cross-document
graph to walk (see ``results/hippo_overnight/dense_ppr_sweep/ANALYSIS.md``).
HotpotQA and 2WikiMultihopQA are multi-document, free-form-answer, and ship
gold supporting facts — HotpotQA at the sentence level, 2Wiki additionally as
``(subject, relation, object)`` evidence triples that line up with the KG fact
schema. That gives us the independent relevance labels the dense_ppr experiment
needs.

Normalized schema (one JSON object per line)
--------------------------------------------
``_id``                   stable example id (str)
``dataset``               "hotpotqa" | "2wikimultihopqa"
``question``              str
``answer``                free-form gold answer string
``context``               ``[[title, [sentence, ...]], ...]`` — kept STRUCTURED so
                          ``longbench_kg_pipeline.flatten_context_to_sentence_records``
                          preserves each document ``title`` and ``local_sent_id``
                          in fact provenance.
``gold_supporting_facts`` ``[[title, sent_id], ...]`` — ``sent_id`` is the local
                          sentence index within that title's sentence list.
``gold_evidence_triples`` ``[[subject, relation, object], ...]`` — 2Wiki only;
                          ``[]`` for HotpotQA.
``type``                  question type when present (e.g. "bridge"/"comparison")
``level``                 difficulty when present (HotpotQA)

The ``choice_A..D`` fields LongBench uses are intentionally absent; downstream
code reads them with ``.get()`` and tolerates their absence.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Field set every normalized row must carry (used by the validators/reporters).
NORMALIZED_FIELDS: List[str] = [
    "_id",
    "dataset",
    "question",
    "answer",
    "context",
    "gold_supporting_facts",
    "gold_evidence_triples",
]


# --------------------------------------------------------------------------- #
# Encoding helpers: HuggingFace usually stores the parallel columns as a dict of
# lists ({"title": [...], "sentences": [[...]]}); the original release JSON uses
# a list of pairs ([[title, [sentences]], ...]). Accept both.
# --------------------------------------------------------------------------- #
def _strip_wrapping_quotes(text: str) -> str:
    """Some HF mirrors (e.g. ``voidful/2WikiMultihopQA``) store titles as
    literal ``"Title"`` strings — the quote characters are part of the Python
    string, not JSON syntax. Strip one layer if present."""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def _normalize_context(context: Any) -> List[List[Any]]:
    """Return ``[[title, [sentence, ...]], ...]`` from either encoding."""
    blocks: List[List[Any]] = []

    # HF parallel-list dict form: {"title": [...], "sentences": [[...], ...]}
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences")
        if sentences is None:
            sentences = context.get("content") or context.get("text") or []
        for title, sents in zip(titles, sentences):
            title_text = _strip_wrapping_quotes(str(title))
            blocks.append([title_text, _as_sentence_list(sents)])
        return blocks

    # List-of-pairs form: [[title, [sentences]], ...]. Some mirrors store the
    # sentence list as a JSON-encoded STRING instead of an actual list — an
    # extra encoding layer baked into the export, not a nested-list schema.
    if isinstance(context, list):
        for block in context:
            if isinstance(block, (list, tuple)) and len(block) == 2:
                title_text = _strip_wrapping_quotes(str(block[0]))
                blocks.append([title_text, _as_sentence_list(block[1])])
            elif isinstance(block, dict):
                title = _strip_wrapping_quotes(str(block.get("title", "")))
                sents = block.get("sentences", block.get("text", []))
                blocks.append([title, _as_sentence_list(sents)])
        return blocks

    return blocks


def _as_sentence_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(s) for s in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                return [str(s) for s in decoded]
        return [value] if value else []
    if value is None:
        return []
    return [str(value)]


def _normalize_supporting_facts(supporting: Any) -> List[List[Any]]:
    """Return ``[[title, sent_id], ...]`` from either encoding."""
    pairs: List[List[Any]] = []

    # HF parallel-list dict form: {"title": [...], "sent_id": [...]}
    if isinstance(supporting, dict):
        titles = supporting.get("title") or supporting.get("titles") or []
        sent_ids = supporting.get("sent_id")
        if sent_ids is None:
            sent_ids = supporting.get("sent_ids") or supporting.get("sentence_id") or []
        for title, sent_id in zip(titles, sent_ids):
            pairs.append([_strip_wrapping_quotes(str(title)), _maybe_int(sent_id)])
        return pairs

    # List-of-pairs form: [[title, sent_id], ...]
    if isinstance(supporting, list):
        for item in supporting:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append([_strip_wrapping_quotes(str(item[0])), _maybe_int(item[1])])
        return pairs

    return pairs


def _normalize_evidence_triples(evidences: Any) -> List[List[str]]:
    """Return ``[[subject, relation, object], ...]`` (2Wiki ``evidences``)."""
    triples: List[List[str]] = []
    if not isinstance(evidences, list):
        return triples
    for item in evidences:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            triples.append([str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()])
        elif isinstance(item, dict):
            subj = item.get("subject") or item.get("head")
            rel = item.get("relation") or item.get("predicate")
            obj = item.get("object") or item.get("tail")
            if subj is not None and obj is not None:
                triples.append([str(subj).strip(), str(rel or "").strip(), str(obj).strip()])
    return triples


def _maybe_int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


# --------------------------------------------------------------------------- #
# Row normalizers
# --------------------------------------------------------------------------- #
def normalize_hotpotqa_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(row.get("_id") or row.get("id") or ""),
        "dataset": "hotpotqa",
        "question": str(row.get("question") or ""),
        "answer": str(row.get("answer") or ""),
        "context": _normalize_context(row.get("context")),
        "gold_supporting_facts": _normalize_supporting_facts(row.get("supporting_facts")),
        "gold_evidence_triples": [],  # HotpotQA has no triple-form evidence
        "type": row.get("type"),
        "level": row.get("level"),
    }


def normalize_2wiki_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(row.get("_id") or row.get("id") or ""),
        "dataset": "2wikimultihopqa",
        "question": str(row.get("question") or ""),
        "answer": str(row.get("answer") or ""),
        "context": _normalize_context(row.get("context")),
        "gold_supporting_facts": _normalize_supporting_facts(row.get("supporting_facts")),
        "gold_evidence_triples": _normalize_evidence_triples(row.get("evidences")),
        "type": row.get("type"),
        "level": row.get("level"),
    }


NORMALIZERS = {
    "hotpotqa": normalize_hotpotqa_row,
    "2wikimultihopqa": normalize_2wiki_row,
}


# --------------------------------------------------------------------------- #
# Validation (mirrors download_longbench.validate: round-trip + coverage)
# --------------------------------------------------------------------------- #
def validate_row(row: Dict[str, Any]) -> List[str]:
    """Return a list of problems with a normalized row (empty == valid)."""
    problems: List[str] = []
    if not row.get("_id"):
        problems.append("missing _id")
    if not row.get("question"):
        problems.append("missing question")
    if not row.get("answer"):
        problems.append("missing answer")
    context = row.get("context")
    if not isinstance(context, list) or not context:
        problems.append("empty/invalid context")
    if not row.get("gold_supporting_facts"):
        problems.append("no gold_supporting_facts")
    # Every supporting-fact title must exist among the context documents so the
    # label can be mapped to a fact later.
    titles = {str(block[0]) for block in (context or []) if isinstance(block, (list, tuple))}
    for pair in row.get("gold_supporting_facts", []):
        if isinstance(pair, (list, tuple)) and pair and str(pair[0]) not in titles:
            problems.append(f"supporting title not in context: {pair[0]!r}")
            break
    return problems
