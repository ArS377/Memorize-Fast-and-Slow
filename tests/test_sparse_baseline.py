from __future__ import annotations

import json
from pathlib import Path

from experiments.sparse_baseline import compute_sparse_baseline, main


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "qwen_sparse_smoke_cases.json"


def test_sparse_smoke_fixture_covers_every_required_scenario() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert {case["id"] for case in cases} == {
        "exact_entity", "lowercase_alias", "paraphrase_miss", "one_hop",
        "multi_hop", "valid_zero", "scope_isolation", "malformed_arguments",
        "repeated_call", "retrieval_timeout",
    }
    scope_case = next(case for case in cases if case["id"] == "scope_isolation")
    assert scope_case["expected_fact_ids"] == ["scope-own"]
    assert scope_case["forbidden_fact_ids"] == ["scope-foreign"]


def test_compute_sparse_baseline_records_phase_three_metrics() -> None:
    metrics = compute_sparse_baseline([
        {"retrieved_fact_count": 1, "expected_fact_ids": ["f1"], "retrieved_fact_ids": ["f1"], "correct": True, "tool_call_count": 1, "latency_seconds": 0.2},
        {"retrieved_fact_count": 0, "no_hit": True, "expected_fact_ids": [], "correct": False, "tool_call_count": 2, "latency_seconds": 0.4},
        {"retrieved_fact_count": 0, "no_hit": True, "expected_fact_ids": [], "correct": False, "tool_call_count": 1, "latency_seconds": 0.6, "tool_error_codes": ["backend_timeout"]},
    ])
    assert metrics == {
        "examples": 3, "hit_rate": 0.3333, "no_hit_rate": 0.3333,
        "recall_at_k": 1.0, "answer_accuracy": 0.3333, "mean_tool_calls": 1.3333,
        "mean_latency_seconds": 0.4, "error_rate": 0.3333,
    }


def test_sparse_baseline_reads_native_trace_and_excludes_unlabelled_accuracy() -> None:
    metrics = compute_sparse_baseline([
        {
            "trace": {
                "events": [
                    {
                        "event": "tool_result",
                        "response": {"status": "ok", "results": [{"fact_id": "f1"}]},
                    }
                ]
            },
            "expected_fact_ids": ["f1"],
        }
    ])

    assert metrics["recall_at_k"] == 1.0
    assert metrics["answer_accuracy"] == 0.0


def test_sparse_baseline_cli_writes_separate_metrics_file(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"retrieved_fact_count": 0, "no_hit": True}) + "\n", encoding="utf-8")
    output = tmp_path / "sparse" / "metrics.json"
    assert main(["--input", str(trace), "--output", str(output)]) == output
    assert json.loads(output.read_text(encoding="utf-8"))["no_hit_rate"] == 1.0
