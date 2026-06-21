import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.graph_context import GraphSource, format_facts_from_jsonl


FACTS = [
    {
        "example_id": "example-a",
        "subject": "Secret A",
        "predicate": "ANSWER_IS",
        "object": "A",
        "support_text": "The answer for example A is A.",
        "provenance": [{"sent_id": 1}],
    }
]


def test_jsonl_context_does_not_leak_facts_between_examples() -> None:
    context = format_facts_from_jsonl(FACTS, "example-b")

    assert context == ""


def test_graph_source_reports_zero_triples_for_missing_example() -> None:
    source = GraphSource(fallback_facts=FACTS, session_id="pilot_scallop")

    context, n_triples = source.context_for({"_id": "example-b"})

    assert context == ""
    assert n_triples == 0
