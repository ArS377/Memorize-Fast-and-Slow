from __future__ import annotations

import json

from experiments.graph_context import GraphSource
from experiments.rlm_retrieval import parse_retrieval_action, rlm_guided_context


FACTS = [
    {
        "example_id": "ex1",
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "East Indonesia",
        "fact_id": "f1",
        "support_text": "Kalamang is spoken in East Indonesia.",
        "provenance": [{"sent_id": 1}],
    },
    {
        "example_id": "ex1",
        "subject": "East Indonesia",
        "predicate": "PART_OF",
        "object": "Indonesia",
        "fact_id": "f2",
        "support_text": "East Indonesia is part of Indonesia.",
        "provenance": [{"sent_id": 2}],
    },
    {
        "example_id": "ex1",
        "subject": "Indonesia",
        "predicate": "CAPITAL_IS",
        "object": "Jakarta",
        "fact_id": "f3",
        "support_text": "The capital of Indonesia is Jakarta.",
        "provenance": [{"sent_id": 3}],
    },
    {
        "example_id": "ex2",
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "Wrong Example",
        "fact_id": "f4",
        "support_text": "This fact belongs to another example.",
        "provenance": [{"sent_id": 4}],
    },
]


def test_parse_retrieval_action_accepts_embedded_json() -> None:
    parsed = parse_retrieval_action(
        'Plan:\n```json\n{"action": "retrieve", "seed_entities": ["Kalamang"]}\n```'
    )

    assert parsed["action"] == "retrieve"
    assert parsed["seed_entities"] == ["Kalamang"]


def test_rlm_guided_context_follows_multi_hop_plan() -> None:
    source = GraphSource(fallback_facts=FACTS, session_id="pilot_scallop")
    example = {
        "_id": "ex1",
        "question": "What is the capital of the country where Kalamang is spoken?",
        "choice_A": "Jakarta",
        "choice_B": "Bandung",
        "choice_C": "Manila",
        "choice_D": "Bangkok",
    }
    actions = iter([
        {"action": "retrieve", "seed_entities": ["Kalamang"], "predicates": ["SPOKEN_IN"]},
        {"action": "retrieve", "seed_entities": ["East Indonesia"], "predicates": ["PART_OF"]},
        {"action": "retrieve", "seed_entities": ["Indonesia"], "predicates": ["CAPITAL_IS"]},
    ])

    def complete_action(prompt: str, root_prompt: str) -> str:
        return json.dumps(next(actions))

    context, n_rows, trace = rlm_guided_context(
        graph_source=source,
        example=example,
        complete_action=complete_action,
        max_steps=3,
    )

    assert n_rows == 3
    assert "Kalamang -SPOKEN_IN-> East Indonesia" in context
    assert "East Indonesia -PART_OF-> Indonesia" in context
    assert "Indonesia -CAPITAL_IS-> Jakarta" in context
    assert "Wrong Example" not in context
    assert [step["n_new_rows"] for step in trace] == [1, 1, 1]
