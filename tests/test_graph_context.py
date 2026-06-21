import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.graph_context import GraphSource, format_facts_from_jsonl


FACTS = [
    {
        "example_id": "example-a",
        "subject": "Secret Alpha",
        "predicate": "ANSWER_IS",
        "object": "A",
        "support_text": "The answer for Secret Alpha is A.",
        "provenance": [{"sent_id": 1}],
    }
]


def test_jsonl_context_does_not_leak_facts_between_examples() -> None:
    context = format_facts_from_jsonl(FACTS, "example-b")

    assert context == ""


def test_jsonl_session_scope_retrieves_accumulated_seeded_facts() -> None:
    context = format_facts_from_jsonl(
        FACTS,
        "example-b",
        memory_scope="session",
        seed_entities=["Secret Alpha"],
    )

    assert "Secret Alpha -ANSWER_IS-> A" in context
    assert "(example-a, sent_id=1)" in context


def test_graph_source_reports_zero_triples_for_missing_example() -> None:
    source = GraphSource(fallback_facts=FACTS, session_id="pilot_scallop")

    context, n_triples = source.context_for({"_id": "example-b"})

    assert context == ""
    assert n_triples == 0


def test_graph_source_session_scope_uses_fallback_across_examples() -> None:
    source = GraphSource(
        fallback_facts=FACTS,
        session_id="pilot_scallop",
        memory_scope="session",
    )

    context, n_triples = source.context_for(
        {"_id": "example-b", "question": "What is Secret Alpha?"}
    )

    assert "Secret Alpha -ANSWER_IS-> A" in context
    assert n_triples == 1


class MockGraph:
    def __init__(self):
        self.calls = []

    def query_context(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "example_id": "example-a",
                "fact_id": "fact-a",
                "subject": "Secret Alpha",
                "predicate": "ANSWER_IS",
                "object": "A",
                "support_text": "The answer for Secret Alpha is A.",
                "provenance": [{"sent_id": 1}],
            }
        ]

    def format_context_for_llm(self, rows, max_chars=4000):
        return "\n".join(
            f"[F{i}] {r['subject']} -{r['predicate']}-> {r['object']}"
            for i, r in enumerate(rows, start=1)
        )

    def close(self):
        pass


def test_live_graph_example_scope_filters_by_example_id() -> None:
    graph = MockGraph()
    source = GraphSource(graph=graph, session_id="pilot_scallop")

    context, n_triples = source.context_for(
        {"_id": "example-b", "question": "What is Secret Alpha?"}
    )

    assert "Secret Alpha -ANSWER_IS-> A" in context
    assert n_triples == 1
    assert graph.calls[0]["example_id"] == "example-b"
    assert graph.calls[0]["session_id"] == "pilot_scallop"


def test_live_graph_session_scope_drops_example_filter() -> None:
    graph = MockGraph()
    source = GraphSource(
        graph=graph,
        session_id="pilot_scallop",
        memory_scope="session",
    )

    context, n_triples = source.context_for(
        {"_id": "example-b", "question": "What is Secret Alpha?"}
    )

    assert "Secret Alpha -ANSWER_IS-> A" in context
    assert n_triples == 1
    assert graph.calls[0]["example_id"] is None
    assert graph.calls[0]["session_id"] == "pilot_scallop"


def main() -> bool:
    tests = [
        test_jsonl_context_does_not_leak_facts_between_examples,
        test_jsonl_session_scope_retrieves_accumulated_seeded_facts,
        test_graph_source_reports_zero_triples_for_missing_example,
        test_graph_source_session_scope_uses_fallback_across_examples,
        test_live_graph_example_scope_filters_by_example_id,
        test_live_graph_session_scope_drops_example_filter,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nResults: {len(tests)}/{len(tests)} tests passed")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
