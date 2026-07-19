from __future__ import annotations

import json
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from experiments._cli import build_arg_parser, run_cell
from experiments.graph_context import GraphSource
from experiments.kg_search_tool import (
    SEARCH_KNOWLEDGE_GRAPH_TOOL,
    execute_search_knowledge_graph,
)
from experiments.rlm_retrieval import (
    NativeToolSession,
    qwen_rlm_tool_answer,
)
from experiments.run_all import _common_cell_args
from experiments.working_memory_tool import UPDATE_WORKING_MEMORY_TOOL


FACTS = [
    {
        "example_id": "ex1",
        "session_id": "pilot_scallop",
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "East Indonesia",
        "fact_id": "f1",
        "support_text": "Kalamang is spoken in East Indonesia.",
        "document_id": "doc-1",
        "provenance": [{"document_id": "doc-1", "sent_id": 1}],
    },
    {
        "example_id": "ex1",
        "session_id": "pilot_scallop",
        "subject": "East Indonesia",
        "predicate": "PART_OF",
        "object": "Indonesia",
        "fact_id": "f2",
        "support_text": "East Indonesia is part of Indonesia.",
        "document_id": "doc-1",
        "provenance": [{"document_id": "doc-1", "sent_id": 2}],
    },
    {
        "example_id": "ex2",
        "session_id": "pilot_scallop",
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "Wrong Example",
        "fact_id": "f-wrong",
        "support_text": "This fact belongs to another example.",
        "provenance": [{"sent_id": 3}],
    },
]

EXAMPLE = {
    "_id": "ex1",
    "question": "Where is Kalamang spoken?",
    "choice_A": "East Indonesia",
    "choice_B": "West Indonesia",
    "choice_C": "Malaysia",
    "choice_D": "Thailand",
}


def _tool_call(call_id: str, arguments: Any, name: str = "search_knowledge_graph") -> Any:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(
    *,
    content: Optional[str] = None,
    tool_calls: Optional[List[Any]] = None,
) -> Any:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or [])
    finish_reason = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


class FakeCompletions:
    def __init__(self, responses: List[Any], events: Optional[List[str]] = None):
        self.responses = iter(responses)
        self.requests: List[Dict[str, Any]] = []
        self.events = events

    def create(self, **kwargs: Any) -> Any:
        if self.events is not None:
            self.events.append("model")
        self.requests.append(kwargs)
        return next(self.responses)


class FakeRLMClient:
    """Enough of rlms' OpenAIClient interface for an actual RLM trajectory."""

    def __init__(
        self,
        responses: List[Any],
        *,
        events: Optional[List[str]] = None,
        subcall_response: str = "Subcall confirms choice A using fact f1.",
    ) -> None:
        self.model_name = "Qwen/Qwen3-4B"
        self.timeout = 30.0
        self.completions = FakeCompletions(responses, events=events)
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )
        self.subcall_response = subcall_response
        self.subcall_prompts: List[Any] = []
        self._model_usage = SimpleNamespace(
            total_calls=0,
            total_input_tokens=0,
            total_output_tokens=0,
        )
        self._usage = SimpleNamespace(
            model_usage_summaries={self.model_name: self._model_usage}
        )

    def completion(self, prompt: Any, model: Optional[str] = None) -> str:
        self.subcall_prompts.append(prompt)
        return self.subcall_response

    async def acompletion(self, prompt: Any, model: Optional[str] = None) -> str:
        return self.completion(prompt, model=model)

    def _track_cost(self, response: Any, model: str) -> None:
        self._model_usage.total_calls += 1

    def get_usage_summary(self) -> Any:
        return self._usage

    def get_last_usage(self) -> Any:
        return self._model_usage


def _source(*, events: Optional[List[str]] = None) -> GraphSource:
    if events is None:
        return GraphSource(
            fallback_facts=FACTS,
            session_id="pilot_scallop",
            memory_scope="example",
        )

    class TrackingSource(GraphSource):
        def rows_for(self, **kwargs):
            events.append("retrieval")
            return super().rows_for(**kwargs)

    return TrackingSource(
        fallback_facts=FACTS,
        session_id="pilot_scallop",
        memory_scope="example",
    )


def _session(client: FakeRLMClient, **overrides: Any) -> NativeToolSession:
    values = {
        "model": "Qwen/Qwen3-4B",
        "graph_source": _source(),
        "example_id": "ex1",
        "max_tool_calls": 3,
        "tool_choice": "auto",
        "tool_timeout": 30.0,
        "max_completion_tokens": 2048,
        "tool_schema": SEARCH_KNOWLEDGE_GRAPH_TOOL,
        "execute_tool": execute_search_knowledge_graph,
    }
    values.update(overrides)
    return NativeToolSession(**values)


def _messages() -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "RLM system"},
        {"role": "user", "content": "Where is Kalamang spoken?"},
    ]


def _tool_messages(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [message for message in request["messages"] if message["role"] == "tool"]


def test_native_tool_protocol_starts_with_question_and_returns_structured_results() -> None:
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call-1",
                        {
                            "query": "Where is Kalamang spoken?",
                            "seed_entities": ["Kalamang"],
                            "predicates": ["SPOKEN_IN"],
                        },
                    )
                ]
            ),
            _response(content="Continue RLM reasoning with fact f1."),
        ]
    )
    session = _session(client)

    content = session.complete(client, _messages())

    first_request = client.completions.requests[0]
    assert "Where is Kalamang spoken?" in first_request["messages"][-1]["content"]
    assert not _tool_messages(first_request)
    assert first_request["tools"][0]["function"]["name"] == "search_knowledge_graph"
    assert first_request["tool_choice"] == "auto"
    assert first_request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    second_messages = client.completions.requests[1]["messages"]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    result = json.loads(second_messages[-1]["content"])
    assert result["status"] == "ok"
    assert set(result) == {
        "status", "tool", "request", "scope", "results", "result_count",
        "working_memory", "truncated", "empty_reason", "retrieval", "error",
    }
    assert result["results"][0]["fact_id"] == "f1"
    assert result["results"][0]["document_id"] == "doc-1"
    assert result["results"][0]["retrieval_mode"] == "sparse"
    assert all(row["fact_id"] != "f-wrong" for row in result["results"])
    assert content == "Continue RLM reasoning with fact f1."
    assert session.retrieved_fact_ids == {"f1"}
    tool_result = next(
        event for event in session.trace["events"] if event.get("event") == "tool_result"
    )
    assert tool_result["elapsed_seconds"] >= 0
    assert "elapsed_seconds" not in tool_result["response"]
    json.dumps(session.trace)


def test_empty_result_is_explicit_and_can_be_reformulated() -> None:
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call-empty",
                        {"query": "missing alias", "seed_entities": ["Missing Alias"]},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "call-reformulated",
                        {"query": "Kalamang location", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(content="Use f1 in the RLM trajectory."),
        ]
    )
    session = _session(client, max_tool_calls=2)

    session.complete(client, _messages())

    first_result = json.loads(_tool_messages(client.completions.requests[1])[0]["content"])
    assert first_result["status"] == "ok"
    assert first_result["results"] == []
    assert session.tool_call_count == 2
    assert session.retrieved_fact_ids == {"f1"}
    assert any(
        event.get("event") == "retry" and event.get("reason") == "valid_empty_result"
        for event in session.trace["events"]
    )


def test_search_then_working_memory_update_uses_only_returned_facts() -> None:
    client = FakeRLMClient(
        [
            _response(tool_calls=[_tool_call("search", {"query": "Kalamang", "seed_entities": ["Kalamang"]})]),
            _response(tool_calls=[_tool_call(
                "update",
                {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                name="update_working_memory",
            )]),
            _response(content="FINAL(FINAL_ANSWER: A\nCITED_FACT_IDS: f1)"),
        ]
    )
    session = _session(
        client,
        max_tool_calls=3,
        require_memory_update=True,
        validate_memory_updates=False,
    )
    content = session.complete(client, _messages())
    assert "FINAL_ANSWER: A" in content
    assert session.working_memory_fact_ids == {"f1"}
    assert len(session.working_memory_artifact_ids) == 1
    assert [tool["function"]["name"] for tool in client.completions.requests[0]["tools"]] == [
        "search_knowledge_graph", "update_working_memory"
    ]


def test_malformed_and_duplicate_calls_return_tool_errors() -> None:
    executions: List[Dict[str, Any]] = []

    def execute(arguments, graph_source, example_id):
        executions.append(dict(arguments))
        return execute_search_knowledge_graph(arguments, graph_source, example_id)

    repeated = {"query": "Kalamang", "seed_entities": ["Kalamang"]}
    client = FakeRLMClient(
        [
            _response(tool_calls=[_tool_call("bad", "{not-json")]),
            _response(tool_calls=[_tool_call("good", repeated)]),
            _response(tool_calls=[_tool_call("duplicate", repeated)]),
            _response(content="Continue after explicit errors."),
        ]
    )
    session = _session(client, max_tool_calls=3, execute_tool=execute)

    session.complete(client, _messages())

    malformed = json.loads(_tool_messages(client.completions.requests[1])[-1]["content"])
    duplicate = json.loads(_tool_messages(client.completions.requests[3])[-1]["content"])
    assert malformed["error"]["code"] == "malformed_arguments"
    assert duplicate["error"]["code"] == "duplicate_tool_call"
    assert malformed["tool"] == "search_knowledge_graph"
    assert duplicate["scope"]["example_id"] == "ex1"
    assert malformed["result_count"] == duplicate["result_count"] == 0
    assert len(executions) == 1


def test_tool_timeout_is_an_error_not_an_empty_result() -> None:
    def slow_execute(arguments, graph_source, example_id):
        time.sleep(0.05)
        return {"status": "ok", "results": [], "error": None}

    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call("slow", {"query": "Kalamang", "seed_entities": ["Kalamang"]})
                ]
            ),
            _response(content="FINAL(EVIDENCE_INSUFFICIENT: retrieval timed out)"),
        ]
    )
    session = _session(
        client,
        max_tool_calls=1,
        tool_timeout=0.001,
        execute_tool=slow_execute,
    )

    session.complete(client, _messages())

    timeout = json.loads(_tool_messages(client.completions.requests[1])[0]["content"])
    assert timeout["status"] == "error"
    assert timeout["error"]["code"] == "backend_timeout"


def test_scope_arguments_are_rejected_and_valid_empty_is_not_an_error() -> None:
    source = _source()
    rejected = execute_search_knowledge_graph(
        {"query": "Kalamang", "seed_entities": ["Kalamang"], "example_id": "ex2"},
        source,
        "ex1",
    )
    empty = execute_search_knowledge_graph(
        {"query": "Unknown", "seed_entities": ["Unknown"]},
        source,
        "ex1",
    )

    assert rejected["status"] == "error"
    assert rejected["error"]["code"] == "invalid_arguments"
    assert empty["status"] == "ok"
    assert empty["results"] == []
    assert empty["scope"]["example_id"] == "ex1"


def test_native_tool_call_runs_inside_real_rlm_repl_trajectory(tmp_path: Path) -> None:
    if importlib.util.find_spec("rlm") is None:
        import pytest
        pytest.skip("rlms runtime is not installed")
    events: List[str] = []
    source = _source(events=events)
    repl_action = """```repl
analysis = llm_query("Check whether returned fact f1 supports a choice")
print(analysis)
```"""
    base_client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call-1",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(content=repl_action),
            _response(content="FINAL(FINAL_ANSWER: A\nCITED_FACT_IDS: f1)"),
        ],
        events=events,
    )

    with patch("rlm.core.rlm.get_client", return_value=base_client):
        outcome = qwen_rlm_tool_answer(
            backend="openai",
            model="Qwen/Qwen3-4B",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            max_depth=2,
            max_iterations=3,
            max_tokens=64000,
            log_dir=tmp_path / "rlm_logs",
            verbose=False,
            graph_source=source,
            example=EXAMPLE,
        )

    assert events[:2] == ["model", "retrieval"], outcome.error
    assert base_client.subcall_prompts == ["Check whether returned fact f1 supports a choice"]
    assert "Subcall confirms choice A" in json.dumps(
        base_client.completions.requests[2]["messages"]
    )
    assert outcome.status == "supported"
    assert outcome.predicted == "A"
    assert outcome.cited_fact_ids == ["f1"]
    assert outcome.trace["orchestration"] == "qwen_native_tool_inside_rlm"
    assert outcome.trace["rlm_model_completion_count"] == 2


def test_uncited_or_fabricated_final_answer_is_not_accepted(tmp_path: Path) -> None:
    if importlib.util.find_spec("rlm") is None:
        import pytest
        pytest.skip("rlms runtime is not installed")
    base_client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call-1",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(content="FINAL(FINAL_ANSWER: A\nCITED_FACT_IDS: fabricated)"),
        ]
    )

    with patch("rlm.core.rlm.get_client", return_value=base_client):
        outcome = qwen_rlm_tool_answer(
            backend="openai",
            model="Qwen/Qwen3-4B",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            max_depth=1,
            max_iterations=2,
            max_tokens=64000,
            log_dir=tmp_path / "rlm_logs",
            verbose=False,
            graph_source=_source(),
            example=EXAMPLE,
        )

    assert outcome.status == "error"
    assert outcome.predicted == ""
    assert outcome.termination_reason == "invalid_final_answer", outcome.error
    assert outcome.error == "unknown cited fact IDs: fabricated"


def test_cli_exposes_one_integrated_retrieval_flag() -> None:
    parser = build_arg_parser(
        cell_id=5,
        label="rlm_kg_noscallop",
        kind="rlm",
        retrieval="kg",
    )
    enabled = parser.parse_args(
        [
            "--qwen-tool-retrieval",
            "--max-tool-calls",
            "4",
            "--tool-choice",
            "required",
            "--tool-timeout",
            "2.5",
            "--tool-trace-dir",
            "traces",
        ]
    )
    assert enabled.qwen_tool_retrieval is True
    assert enabled.max_tool_calls == 4
    assert enabled.tool_choice == "required"
    assert enabled.tool_timeout == 2.5
    assert enabled.tool_trace_dir == Path("traces")
    assert "--rlm-retrieval" not in parser.format_help()
    assert "--rlm-retrieval-steps" not in parser.format_help()


def test_official_schema_does_not_expose_scope_controls() -> None:
    properties = SEARCH_KNOWLEDGE_GRAPH_TOOL["function"]["parameters"]["properties"]

    assert "example_id" not in properties
    assert "session_id" not in properties
    assert SEARCH_KNOWLEDGE_GRAPH_TOOL["function"]["name"] == "search_knowledge_graph"
    update_properties = UPDATE_WORKING_MEMORY_TOOL["function"]["parameters"]["properties"]
    assert "session_id" not in update_properties
    assert "memory_scope" not in update_properties


def test_run_all_propagates_one_integrated_mode(tmp_path: Path) -> None:
    common = {
        "input": Path("input.jsonl"),
        "limit": 1,
        "seed": 0,
        "model": "Qwen/Qwen3-4B",
        "vllm_base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "results_dir": tmp_path,
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": None,
        "hops": 2,
        "limit_triples": 50,
        "memory_scope": "example",
        "raw_max_chars": 32000,
        "max_depth": 2,
        "max_iterations": 6,
        "max_tokens": 64000,
        "max_tool_calls": 3,
        "tool_choice": "auto",
        "tool_timeout": 30.0,
        "tool_max_tokens": 2048,
        "tool_trace_dir": None,
    }
    enabled = _common_cell_args(SimpleNamespace(**common, qwen_tool_retrieval=True), 5)
    fixed = _common_cell_args(
        SimpleNamespace(**common, qwen_tool_retrieval=False, fixed_kg_retrieval=True), 5
    )

    assert "--qwen-tool-retrieval" in enabled
    assert "--max-tool-calls" in enabled
    assert "--rlm-retrieval" not in enabled
    assert enabled[enabled.index("--retrieval-mode") + 1] == "hybrid"
    assert enabled[enabled.index("--dense-failure-policy") + 1] == "error"
    assert "--fixed-kg-retrieval" in fixed


def test_cell_runner_does_not_pre_retrieve_and_persists_integrated_trace(
    tmp_path: Path,
) -> None:
    class NoPreRetrievalSource(GraphSource):
        def context_for(self, *args, **kwargs):
            raise AssertionError("integrated mode must not pre-retrieve context")

    source = NoPreRetrievalSource(
        fallback_facts=FACTS,
        session_id="pilot_noscallop",
        memory_scope="example",
    )
    trace = {
        "orchestration": "qwen_native_tool_inside_rlm",
        "termination_reason": "supported_final_answer",
    }
    outcome = SimpleNamespace(
        predicted="A",
        error=None,
        retrieved_fact_count=1,
        tool_result_chars=300,
        trace=trace,
        termination_reason="supported_final_answer",
    )
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "cell5" / "results.jsonl"
    input_path.write_text(json.dumps({**EXAMPLE, "answer": "A"}) + "\n", encoding="utf-8")

    with patch(
        "experiments.graph_context.open_graph_source", return_value=source
    ), patch(
        "experiments.rlm_retrieval.qwen_rlm_tool_answer", return_value=outcome
    ) as answer:
        run_cell(
            cell_id=5,
            label="rlm_kg_noscallop",
            kind="rlm",
            retrieval="kg",
            session_id="pilot_noscallop",
            argv=[
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--qwen-tool-retrieval",
                "--no-aggregate",
            ],
        )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    trace_path = output_path.parent / "tool_traces" / "0001_ex1.json"
    persisted = json.loads(trace_path.read_text(encoding="utf-8"))
    assert answer.call_count == 1
    assert result["predicted"] == "A"
    assert result["n_triples"] == 1
    assert persisted["orchestration"] == "qwen_native_tool_inside_rlm"


def test_cell5_and_cell6_enable_identical_integrated_retrieval_defaults(tmp_path: Path) -> None:
    class NoPreRetrievalSource(GraphSource):
        def context_for(self, *args, **kwargs):
            raise AssertionError("cell 6 must use integrated retrieval by default")

    source = NoPreRetrievalSource(
        fallback_facts=FACTS,
        session_id="pilot_scallop",
        memory_scope="example",
    )
    outcome = SimpleNamespace(
        predicted="A",
        error=None,
        retrieved_fact_count=1,
        tool_result_chars=300,
        trace={
            "orchestration": "qwen_native_tool_inside_rlm",
            "termination_reason": "supported_final_answer",
        },
        termination_reason="supported_final_answer",
    )
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "cell6" / "results.jsonl"
    input_path.write_text(json.dumps({**EXAMPLE, "answer": "A"}) + "\n", encoding="utf-8")

    with patch("experiments.graph_context.open_graph_source", return_value=source), patch(
        "experiments.rlm_retrieval.qwen_rlm_tool_answer", return_value=outcome
    ) as answer:
        for cell_id, label, session in [
            (5, "rlm_kg_noscallop", "pilot_noscallop"),
            (6, "rlm_kg_scallop", "pilot_scallop"),
        ]:
            run_cell(
                cell_id=cell_id,
                label=label,
                kind="rlm",
                retrieval="kg",
                session_id=session,
                argv=["--input", str(input_path), "--output", str(output_path), "--no-aggregate"],
            )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert answer.call_count == 2
    assert result["predicted"] == "A"
    assert result["n_triples"] == 1
