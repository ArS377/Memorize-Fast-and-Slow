"""Tests for HotpotQA / 2WikiMultihopQA normalization into the shared schema."""
from __future__ import annotations

from experiments.multihop_datasets import (
    normalize_2wiki_row,
    normalize_hotpotqa_row,
    validate_row,
)


HOTPOT_HF_ROW = {  # HuggingFace parallel-list encoding
    "id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
    "context": {
        "title": ["Scott Derrickson", "Ed Wood", "Distractor"],
        "sentences": [
            ["Scott Derrickson is an American director.", "He was born in 1966."],
            ["Edward Wood was an American filmmaker."],
            ["Unrelated sentence."],
        ],
    },
}

TWO_WIKI_JSON_ROW = {  # official release list-of-pairs encoding
    "_id": "228546780bdd11eba7f7acde48001122",
    "type": "compositional",
    "question": "Who is the father of the director of film Kingdom Of Dreams?",
    "answer": "Anurag Kashyap",
    "supporting_facts": [["Kingdom of Dreams", 0], ["Vikramaditya Motwane", 1]],
    "context": [
        ["Kingdom of Dreams", ["It is a film directed by Vikramaditya Motwane.", "Released 2010."]],
        ["Vikramaditya Motwane", ["He is an Indian director.", "His father is Anurag Kashyap."]],
        ["Distractor Doc", ["Nothing relevant here."]],
    ],
    "evidences": [
        ["Kingdom of Dreams", "director", "Vikramaditya Motwane"],
        ["Vikramaditya Motwane", "father", "Anurag Kashyap"],
    ],
}


def test_hotpotqa_normalizes_parallel_lists():
    row = normalize_hotpotqa_row(HOTPOT_HF_ROW)
    assert row["_id"] == "5a8b57f25542995d1e6f1371"
    assert row["dataset"] == "hotpotqa"
    assert row["answer"] == "yes"
    assert row["gold_evidence_triples"] == []
    assert row["gold_supporting_facts"] == [["Scott Derrickson", 0], ["Ed Wood", 0]]
    # Context kept structured so titles survive into fact provenance.
    assert row["context"][0][0] == "Scott Derrickson"
    assert row["context"][0][1][0].startswith("Scott Derrickson is an American")
    assert validate_row(row) == []


def test_2wiki_normalizes_list_of_pairs_and_triples():
    row = normalize_2wiki_row(TWO_WIKI_JSON_ROW)
    assert row["dataset"] == "2wikimultihopqa"
    assert row["answer"] == "Anurag Kashyap"
    assert row["gold_supporting_facts"] == [["Kingdom of Dreams", 0], ["Vikramaditya Motwane", 1]]
    assert ["Vikramaditya Motwane", "father", "Anurag Kashyap"] in row["gold_evidence_triples"]
    assert row["context"][1][0] == "Vikramaditya Motwane"
    assert validate_row(row) == []


def test_validate_flags_supporting_title_absent_from_context():
    row = normalize_2wiki_row(TWO_WIKI_JSON_ROW)
    row["gold_supporting_facts"].append(["Ghost Document", 3])
    problems = validate_row(row)
    assert any("supporting title not in context" in p for p in problems)


def test_validate_flags_missing_answer():
    row = normalize_hotpotqa_row({**HOTPOT_HF_ROW, "answer": ""})
    assert "missing answer" in validate_row(row)


# voidful/2WikiMultihopQA on HuggingFace: titles are literal-quote-wrapped
# strings, and each document's sentence list is a JSON-encoded STRING rather
# than an actual list (see experiments/multihop_datasets._as_sentence_list).
VOIDFUL_2WIKI_ROW = {
    "_id": "abc123",
    "type": "compositional",
    "question": "Who is the mother of the director of Polish-Russian War?",
    "answer": "Małgorzata Braunek",
    "supporting_facts": [["\"Polish-Russian War (film)\"", "1"], ["\"Xawery Żuławski\"", "2"]],
    "context": [
        ["\"Polish-Russian War (film)\"", "[\"Polish-Russian War is a 2009 film.\",\"It is directed by Xawery \\u017bu\\u0142awski.\"]"],
        ["\"Xawery Żuławski\"", "[\"He is a Polish director.\",\"His mother is Małgorzata Braunek.\"]"],
        ["\"Distractor\"", "[\"Nothing relevant.\"]"],
    ],
    "evidences": [
        ["Polish-Russian War", "director", "Xawery Żuławski"],
        ["Xawery Żuławski", "mother", "Małgorzata Braunek"],
    ],
}


def test_2wiki_unwraps_voidful_mirror_quote_and_string_encoding():
    row = normalize_2wiki_row(VOIDFUL_2WIKI_ROW)
    # Titles have their literal wrapping quotes stripped.
    assert row["context"][0][0] == "Polish-Russian War (film)"
    assert row["context"][1][0] == "Xawery Żuławski"
    # The JSON-encoded-string sentence list is decoded into real sentences.
    assert row["context"][0][1] == [
        "Polish-Russian War is a 2009 film.",
        "It is directed by Xawery Żuławski.",
    ]
    # supporting_facts titles are unwrapped the same way, matching context titles.
    assert row["gold_supporting_facts"] == [
        ["Polish-Russian War (film)", 1],
        ["Xawery Żuławski", 2],
    ]
    assert validate_row(row) == []
