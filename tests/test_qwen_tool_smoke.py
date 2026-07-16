from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from experiments.qwen_tool_smoke import main, smoke_record
from experiments.rlm_retrieval import QwenRLMToolOutcome


def _outcome(*, response: dict, status: str = "supported") -> QwenRLMToolOutcome:
    results = response.get("results", [])
    retrieved = [str(result["fact_id"]) for result in results]
    return QwenRLMToolOutcome(
        status=status,
        predicted="A" if status == "supported" else "",
        raw_answer="FINAL(FINAL_ANSWER: A\nCITED_FACT_IDS: f1)",
        error=None,
        termination_reason=(
            "supported_final_answer" if status == "supported" else "evidence_insufficient"
        ),
        cited_fact_ids=retrieved,
        retrieved_fact_ids=retrieved,
        tool_call_count=1,
        retrieved_fact_count=len(retrieved),
        tool_result_chars=100,
        trace={
            "events": [
                {
                    "event": "tool_result",
                    "response": response,
                    "elapsed_seconds": 0.01,
                }
            ]
        },
        working_memory_artifact_ids=[],
    )


def test_smoke_record_distinguishes_empty_results_from_tool_errors() -> None:
    empty = _outcome(
        response={"status": "ok", "results": [], "error": None},
        status="evidence_insufficient",
    )
    failed = _outcome(
        response={
            "status": "error",
            "results": [],
            "error": {"code": "backend_timeout"},
        },
        status="evidence_insufficient",
    )

    empty_record = smoke_record({"id": "empty", "gold": ""}, empty, latency_seconds=0.1)
    failed_record = smoke_record({"id": "failed", "gold": ""}, failed, latency_seconds=0.1)

    assert empty_record["no_hit"] is True
    assert empty_record["correct"] is None
    assert failed_record["no_hit"] is False
    assert failed_record["tool_error_codes"] == ["backend_timeout"]


def test_live_smoke_command_writes_evaluator_ready_jsonl(tmp_path: Path) -> None:
    fixtures = tmp_path / "cases.json"
    output = tmp_path / "trace.jsonl"
    fixtures.write_text(
        json.dumps(
            [
                {
                    "id": "exact_entity",
                    "question": "Where is Kalamang spoken?",
                    "choices": {"A": "East Indonesia", "B": "Jakarta"},
                    "facts": [],
                    "expected_fact_ids": ["f1"],
                    "gold": "A",
                }
            ]
        ),
        encoding="utf-8",
    )
    outcome = _outcome(
        response={
            "status": "ok",
            "results": [{"fact_id": "f1"}],
            "error": None,
        }
    )

    with patch(
        "experiments.qwen_tool_smoke.qwen_rlm_tool_answer",
        return_value=outcome,
    ) as answer:
        assert main(["--fixtures", str(fixtures), "--output", str(output)]) == output

    record = json.loads(output.read_text(encoding="utf-8"))
    assert answer.call_args.kwargs["example"]["question"] == "Where is Kalamang spoken?"
    assert record["tool_call_count"] == 1
    assert record["retrieved_fact_ids"] == ["f1"]
    assert record["correct"] is True
    assert record["termination_reason"] == "supported_final_answer"
