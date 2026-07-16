"""Phase 3 deterministic sparse smoke checks against Person 1's adapter.

These tests cover the retrieval half of the future native Qwen flow.  Person
2's runner will reuse the same fixture and feed the returned responses back as
OpenAI-compatible tool messages.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.graph_context import GraphSource
from experiments.kg_search_tool import execute_search_knowledge_graph


CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "qwen_sparse_smoke_cases.json").read_text(
        encoding="utf-8"
    )
)


def _case(case_id: str):
    return next(case for case in CASES if case["id"] == case_id)


def _source(case_id: str) -> tuple[dict, GraphSource]:
    case = _case(case_id)
    return case, GraphSource(
        fallback_facts=case["facts"], session_id="sparse-smoke", memory_scope="example"
    )


def test_sparse_adapter_handles_exact_and_lowercase_alias_queries() -> None:
    for case_id, arguments in (
        ("exact_entity", {"query": "Where is Kalamang spoken?"}),
        ("lowercase_alias", {"query": "where is this language used?", "seed_entities": ["kalamang"]}),
    ):
        case, source = _source(case_id)
        response = execute_search_knowledge_graph(arguments, source, case["id"])
        assert response["status"] == "ok"
        assert [result["fact_id"] for result in response["results"]] == case["expected_fact_ids"]


def test_sparse_adapter_preserves_paraphrase_miss_and_valid_zero_results() -> None:
    for case_id in ("paraphrase_miss", "valid_zero"):
        case, source = _source(case_id)
        response = execute_search_knowledge_graph({"query": case["question"]}, source, case["id"])
        assert response["status"] == "ok"
        assert response["results"] == []
        assert response["empty_reason"] in {"no_seed_entities", "no_matches"}


def test_sparse_adapter_supports_one_hop_and_explicit_multi_hop_calls() -> None:
    one_hop, source = _source("one_hop")
    one = execute_search_knowledge_graph({"query": one_hop["question"]}, source, one_hop["id"])
    assert [result["fact_id"] for result in one["results"]] == ["one-1"]

    multi_hop, source = _source("multi_hop")
    calls = [
        {"query": multi_hop["question"], "seed_entities": ["Kalamang"], "predicates": ["SPOKEN_IN"]},
        {"query": multi_hop["question"], "seed_entities": ["East Indonesia"], "predicates": ["PART_OF"]},
        {"query": multi_hop["question"], "seed_entities": ["Indonesia"], "predicates": ["CAPITAL_IS"]},
    ]
    fact_ids = [
        response["results"][0]["fact_id"]
        for arguments in calls
        for response in [execute_search_knowledge_graph(arguments, source, multi_hop["id"])]
    ]
    assert fact_ids == multi_hop["expected_fact_ids"]


def test_sparse_adapter_cannot_leak_another_example() -> None:
    case, source = _source("scope_isolation")
    response = execute_search_knowledge_graph({"query": case["question"]}, source, case["id"])
    returned = {result["fact_id"] for result in response["results"]}
    assert returned == set(case["expected_fact_ids"])
    assert returned.isdisjoint(case["forbidden_fact_ids"])


def test_sparse_adapter_reports_malformed_repeated_and_timeout_states() -> None:
    malformed, source = _source("malformed_arguments")
    bad = execute_search_knowledge_graph(malformed["tool_call"], source, malformed["id"])
    assert bad["status"] == "error"
    assert bad["error"]["code"] == "invalid_arguments"

    repeated, source = _source("repeated_call")
    arguments = {"query": repeated["question"]}
    assert execute_search_knowledge_graph(arguments, source, repeated["id"]) == execute_search_knowledge_graph(
        arguments, source, repeated["id"]
    )

    class TimeoutSource:
        session_id = "sparse-smoke"
        memory_scope = "example"

        def rows_for(self, **kwargs):
            raise TimeoutError("fixture timeout")

    timeout = _case("retrieval_timeout")
    failed = execute_search_knowledge_graph({"query": timeout["question"]}, TimeoutSource(), timeout["id"])
    assert failed["status"] == "error"
    assert failed["error"]["code"] == "backend_timeout"
