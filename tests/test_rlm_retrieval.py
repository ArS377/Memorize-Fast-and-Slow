from __future__ import annotations

import json
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from experiments._cli import build_arg_parser, run_cell
from neurosym.adapters.graph_source import GraphSource
from neurosym.adapters.kg_search import (
    SEARCH_KNOWLEDGE_GRAPH_TOOL,
    execute_search_knowledge_graph,
)
from neurosym.adapters.qwen_rlm import (
    NativeToolSession,
    _choice_supports_fact,
    _native_tool_instructions,
    _parse_candidate_answer,
    _parse_final_response,
    _question_message,
    make_qwen_tool_rlm,
    qwen_rlm_tool_answer,
)
from experiments._cli import normalize_short_answer, short_answer_scores
from experiments.run_all import _common_cell_args
from neurosym.adapters.working_memory_tool import UPDATE_WORKING_MEMORY_TOOL


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


class FakeHybridIndex:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.facts = [
            {**fact, "session_id": session_id}
            for fact in FACTS
            if fact["example_id"] == "ex1"
        ]
        self.manifest = SimpleNamespace(
            identity={"source_session_id": session_id, "schema_version": "test.v1"}
        )

    def search(self, query: str, *, top_k: int, scope: Any, predicates=None):
        row = dict(FACTS[0])
        row["session_id"] = self.session_id
        row["_dense_similarity"] = 0.95
        return [row]


def _hybrid_graph_source(open_kwargs: Dict[str, Any], source_class=GraphSource) -> GraphSource:
    session_id = str(open_kwargs["session_id"])
    facts = [{**fact, "session_id": session_id} for fact in FACTS]
    source = source_class(
        fallback_facts=facts,
        session_id=session_id,
        memory_scope=open_kwargs["memory_scope"],
        source_session_ids=open_kwargs["source_session_ids"],
        retrieval_config=open_kwargs["retrieval_config"],
        dense_indexes={session_id: FakeHybridIndex(session_id)},
    )
    assert source.retrieval_config.mode == open_kwargs["retrieval_config"].mode
    return source


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
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeModelUsage(SimpleNamespace):
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


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
        self._model_usage = FakeModelUsage(
            total_calls=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=None,
        )
        self._usage = SimpleNamespace(
            model_usage_summaries={self.model_name: self._model_usage}
        )
        self._usage.to_dict = lambda: {
            "model_usage_summaries": {
                self.model_name: {
                    "total_calls": self._model_usage.total_calls,
                    "total_input_tokens": self._model_usage.total_input_tokens,
                    "total_output_tokens": self._model_usage.total_output_tokens,
                }
            }
        }

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
        "question": EXAMPLE["question"],
        "choices": {
            letter: EXAMPLE[f"choice_{letter}"] for letter in "ABCD"
        },
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


def test_native_tool_instructions_use_the_rlm_completion_protocol() -> None:
    instructions = _native_tool_instructions(3)

    assert 'answer["ready"] = True' in instructions
    assert "FINAL_ANSWER:" in instructions
    assert "do not answer in prose" in instructions
    assert "before emitting any" in instructions
    assert "call `search_knowledge_graph`" in instructions




def test_cell6_instructions_require_reasoned_fallback_after_one_search() -> None:
    instructions = _native_tool_instructions(2, allow_unsupported_fallback=True)

    assert "at most one knowledge-graph search" in instructions
    assert "do not emit EVIDENCE_INSUFFICIENT" in instructions
    assert "return FINAL_ANSWER without CITED_FACT_IDS" in instructions


def test_native_tool_rlm_disables_automatic_model_retries(tmp_path: Path) -> None:
    if importlib.util.find_spec("rlm") is None:
        import pytest
        pytest.skip("rlms runtime is not installed")

    client = FakeRLMClient([])
    rlm = make_qwen_tool_rlm(
        backend="openai",
        model="Qwen/Qwen3-4B",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        max_depth=1,
        max_iterations=1,
        max_tokens=100,
        log_dir=tmp_path / "rlm_logs",
        verbose=False,
        tool_session=_session(client),
        model_timeout=45.0,
        model_max_retries=0,
    )

    assert rlm.backend_kwargs["timeout"] == 45.0
    assert rlm.backend_kwargs["max_retries"] == 0


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


def test_order_gap_stops_repeated_no_evidence_prose() -> None:
    prose = "The graph returned no relevant facts. Therefore, the correct answer is C."
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call-empty",
                        {"query": "missing", "seed_entities": ["Missing"]},
                    )
                ]
            ),
            _response(content=prose),
            _response(content=prose),
        ]
    )
    session = _session(client, max_tool_calls=1, require_memory_update=True)

    assert session.complete(client, _messages()) == prose
    completion = session.complete(client, _messages())

    assert 'answer["ready"] = True' in completion
    assert "FINAL_ANSWER: C" in completion
    assert session.diagnostic_predicted == "C"
    assert (
        session.controller_termination_reason
        == "order_gap_evidence_insufficient_fallback"
    )
    assert session.state_tracker.stable is True
    assert any(
        event.get("event") == "order_gap_stop" for event in session.trace["events"]
    )


def test_order_gap_converts_stable_supported_prose_to_rlm_completion() -> None:
    structured = "FINAL_ANSWER: A\nCITED_FACT_IDS: f1"
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "update",
                        {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                        name="update_working_memory",
                    )
                ]
            ),
            _response(content=structured),
            _response(content=structured),
            _response(content=structured),
        ]
    )
    session = _session(
        client,
        max_tool_calls=2,
        require_memory_update=True,
        validate_memory_updates=False,
    )

    assert session.complete(client, _messages()) == structured
    assert session.complete(client, _messages()) == structured
    completion = session.complete(client, _messages())

    assert 'answer["ready"] = True' in completion
    assert "FINAL_ANSWER: A" in completion
    assert "CITED_FACT_IDS: f1" in completion
    assert session.controller_termination_reason == "order_gap_supported_answer"
    assert session.working_memory_fact_ids == {"f1"}


def test_order_gap_does_not_attach_committed_facts_to_uncited_prose() -> None:
    prose = "The returned fact supports the option. The correct answer is A."
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "update",
                        {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                        name="update_working_memory",
                    )
                ]
            ),
            _response(content=prose),
            _response(content=prose),
        ]
    )
    session = _session(
        client,
        max_tool_calls=2,
        require_memory_update=True,
        validate_memory_updates=False,
    )

    assert session.complete(client, _messages()) == prose
    completion = session.complete(client, _messages())

    assert "FINAL_ANSWER: A" in completion
    assert "CITED_FACT_IDS" not in completion
    assert session.diagnostic_predicted == "A"
    assert (
        session.controller_termination_reason
        == "order_gap_evidence_insufficient_fallback"
    )


def test_cell6_uncited_answer_becomes_first_class_fallback() -> None:
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "update",
                        {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                        name="update_working_memory",
                    )
                ]
            ),
            _response(content="The graph is inconclusive, so the correct answer is C."),
        ]
    )
    session = _session(
        client,
        max_tool_calls=2,
        require_memory_update=True,
        validate_memory_updates=False,
        allow_unsupported_fallback=True,
    )

    completion = session.complete(client, _messages())

    assert 'answer["ready"] = True' in completion
    assert "FINAL_ANSWER: C" in completion
    assert "EVIDENCE_INSUFFICIENT" not in completion
    assert session.search_call_count == 1
    assert session.controller_termination_reason == (
        "unsupported_fallback_after_single_retrieval"
    )
    assert "tools" not in client.completions.requests[-1]


def test_cell6_forces_one_answer_turn_after_evidence_insufficient() -> None:
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search-empty",
                        {"query": "missing", "seed_entities": ["Missing"]},
                    )
                ]
            ),
            _response(content="EVIDENCE_INSUFFICIENT: no graph support"),
            _response(content="FINAL_ANSWER: B"),
        ]
    )
    session = _session(
        client,
        max_tool_calls=2,
        require_memory_update=True,
        allow_unsupported_fallback=True,
    )

    completion = session.complete(client, _messages())

    assert "FINAL_ANSWER: B" in completion
    assert session.fallback_correction_count == 1
    assert session.controller_termination_reason == (
        "unsupported_fallback_after_single_retrieval"
    )
    assert "tools" not in client.completions.requests[-1]
    assert any(
        event.get("event") == "fallback_answer_forced"
        for event in session.trace["events"]
    )


def test_cell6_forces_answer_when_controller_detects_insufficient_state() -> None:
    reasoning = "```repl\nprint(SHOW_VARS())\n```"
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "update",
                        {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                        name="update_working_memory",
                    )
                ]
            ),
            _response(content=reasoning),
            _response(content=reasoning),
            _response(content="FINAL_ANSWER: B"),
        ]
    )
    session = _session(
        client,
        max_tool_calls=2,
        require_memory_update=True,
        validate_memory_updates=False,
        allow_unsupported_fallback=True,
    )

    assert session.complete(client, _messages()) == reasoning
    completion = session.complete(client, _messages())

    assert "FINAL_ANSWER: B" in completion
    assert session.controller_termination_reason == (
        "unsupported_fallback_after_single_retrieval"
    )
    forced = [
        event
        for event in session.trace["events"]
        if event.get("event") == "fallback_answer_forced"
    ]
    assert len(forced) == 1
    assert forced[0]["reason"] == (
        "controller_evidence_insufficient_after_single_retrieval"
    )


def test_cell6_downgrades_inadequate_citations_to_fallback() -> None:
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "update",
                        {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                        name="update_working_memory",
                    )
                ]
            ),
            _response(
                content='''```repl
answer["content"] = "FINAL_ANSWER: B\\nCITED_FACT_IDS: f1"
answer["ready"] = True
```'''
            ),
        ]
    )
    session = _session(
        client,
        max_tool_calls=2,
        require_memory_update=True,
        validate_memory_updates=False,
        allow_unsupported_fallback=True,
    )

    completion = session.complete(client, _messages())

    assert "FINAL_ANSWER: B" in completion
    assert "CITED_FACT_IDS:" not in completion
    assert session.controller_termination_reason == (
        "unsupported_fallback_after_single_retrieval"
    )
    fallback = next(
        event
        for event in session.trace["events"]
        if event.get("event") == "unsupported_fallback_selected"
    )
    assert fallback["discarded_cited_fact_ids"] == ["f1"]
    assert fallback["grounding_errors"] == ["cited facts do not support option B"]


def test_cell6_records_failed_fallback_without_looping() -> None:
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search-empty",
                        {"query": "missing", "seed_entities": ["Missing"]},
                    )
                ]
            ),
            _response(content="EVIDENCE_INSUFFICIENT: no graph support"),
            _response(content="EVIDENCE_INSUFFICIENT: still cannot decide"),
        ]
    )
    session = _session(
        client,
        max_tool_calls=2,
        require_memory_update=True,
        allow_unsupported_fallback=True,
    )

    completion = session.complete(client, _messages())

    assert "EVIDENCE_INSUFFICIENT:" in completion
    assert session.fallback_correction_count == 1
    assert session.controller_termination_reason == (
        "fallback_failed_after_forced_answer"
    )
    assert len(client.completions.requests) == 3


def test_cell6_tool_outcome_preserves_fallback_prediction() -> None:
    def fake_rlm(**kwargs: Any) -> Any:
        kwargs["tool_session"].search_call_count = 1
        return SimpleNamespace(
            completion=lambda **_kwargs: SimpleNamespace(
                response="FINAL_ANSWER: D"
            ),
            close=lambda: None,
        )

    with patch(
        "neurosym.adapters.qwen_rlm.make_qwen_tool_rlm",
        side_effect=fake_rlm,
    ):
        outcome = qwen_rlm_tool_answer(
            backend="openai",
            model="Qwen/Qwen3-4B",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            max_depth=2,
            max_iterations=10,
            max_tokens=64000,
            log_dir=Path("rlm_logs"),
            verbose=False,
            graph_source=_source(),
            example=EXAMPLE,
            max_tool_calls=2,
            allow_unsupported_fallback=True,
        )

    assert outcome.status == "unsupported_fallback"
    assert outcome.predicted == "D"
    assert outcome.error is None
    assert outcome.cited_fact_ids == []


def test_required_tool_choice_applies_only_until_the_first_search() -> None:
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(content="Continue reasoning with f1."),
        ]
    )
    session = _session(client, tool_choice="required")

    session.complete(client, _messages())

    assert client.completions.requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_knowledge_graph"},
    }
    assert client.completions.requests[1]["tool_choice"] == "auto"
    assert session.tool_call_count == 1


def test_required_tool_choice_retries_vllm_repl_parser_rejection_in_auto_mode() -> None:
    parser_error = RuntimeError(
        "Error code: 400 - {'error': {'message': \"Invalid JSON: expected value "
        "[type=json_invalid, input_value='```repl\\nSHOW_VARS()\\n```']}}"
    )
    client = FakeRLMClient(
        [
            parser_error,
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(content="Continue reasoning with f1."),
        ]
    )
    session = _session(client, tool_choice="required")

    session.complete(client, _messages())

    assert client.completions.requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_knowledge_graph"},
    }
    assert client.completions.requests[1]["tool_choice"] == "auto"
    assert any(
        event["event"] == "required_tool_parser_fallback"
        for event in session.trace["events"]
    )
    assert session.tool_call_count == 1


def test_external_budget_mode_does_not_synthesize_completion() -> None:
    prose = "The graph returned no facts. The correct answer is B."
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call("empty", {"query": "missing", "seed_entities": ["Missing"]})
                ]
            ),
            _response(content=prose),
            _response(content=prose),
        ]
    )
    session = _session(
        client,
        require_memory_update=True,
        termination_mode="external_budget",
    )

    assert session.complete(client, _messages()) == prose
    assert session.complete(client, _messages()) == prose
    assert session.controller_termination_reason is None


def test_search_then_working_memory_update_uses_only_returned_facts() -> None:
    client = FakeRLMClient(
        [
            _response(tool_calls=[_tool_call("search", {"query": "Kalamang", "seed_entities": ["Kalamang"]})]),
            _response(tool_calls=[_tool_call(
                "update",
                {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                name="update_working_memory",
            )]),
            _response(
                content='''```repl
answer["content"] = "FINAL_ANSWER: A\\nCITED_FACT_IDS: f1"
answer["ready"] = True
```'''
            ),
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


def test_inadequate_answer_can_search_again_after_memory_commit() -> None:
    inadequate_ready = '''```repl
answer["content"] = "FINAL_ANSWER: B\\nCITED_FACT_IDS: f1"
answer["ready"] = True
```'''
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search-east",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "commit-east",
                        {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                        name="update_working_memory",
                    )
                ]
            ),
            _response(content=inadequate_ready),
            _response(
                tool_calls=[
                    _tool_call(
                        "search-west",
                        {"query": "West Indonesia", "seed_entities": ["West Indonesia"]},
                    )
                ]
            ),
            _response(content="Continue reasoning after the discriminative search."),
        ]
    )
    session = _session(
        client,
        max_tool_calls=4,
        require_memory_update=True,
        validate_memory_updates=False,
    )

    content = session.complete(client, _messages())

    assert content == "Continue reasoning after the discriminative search."
    assert session.working_memory_fact_ids == {"f1"}
    assert session.tool_call_count == 3
    follow_up = client.completions.requests[3]
    assert follow_up["tool_choice"] == "auto"
    assert "cited facts do not support option B" in follow_up["messages"][-1]["content"]
    assert any(
        event.get("event") == "invalid_ready_rejected"
        and "cited facts do not support option B" in event.get("errors", [])
        for event in session.trace["events"]
    )


def test_native_turn_stops_before_exceeding_aggregate_rlm_token_limit() -> None:
    client = FakeRLMClient([])
    client._model_usage.total_input_tokens = 63_900
    session = _session(client, aggregate_token_limit=64_000)

    completion = session.complete(client, _messages())

    assert "EVIDENCE_INSUFFICIENT:" in completion
    assert session.controller_termination_reason == "aggregate_token_budget_guard"
    assert client.completions.requests == []
    guard = next(
        event
        for event in session.trace["events"]
        if event.get("event") == "token_budget_guard"
    )
    assert guard["aggregate_token_limit"] == 64_000


def test_qwen_text_tool_wrapper_uses_the_validated_tool_path() -> None:
    text_update = (
        "<tool_call>\n"
        + json.dumps(
            {
                "name": "update_working_memory",
                "arguments": {
                    "entity": "Kalamang",
                    "selected_fact_ids": ["f1"],
                },
            }
        )
        + "\n</tool_call>"
    )
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(content=text_update),
            _response(
                content='''```repl
answer["content"] = "FINAL_ANSWER: A\\nCITED_FACT_IDS: f1"
answer["ready"] = True
```'''
            ),
        ]
    )
    session = _session(
        client,
        require_memory_update=True,
        validate_memory_updates=False,
    )

    content = session.complete(client, _messages())

    assert "FINAL_ANSWER: A" in content
    assert session.tool_call_count == 2
    assert session.working_memory_fact_ids == {"f1"}
    assert any(
        event.get("event") == "text_tool_call_adapted"
        for event in session.trace["events"]
    )
    adapted_message = client.completions.requests[2]["messages"][-2]
    assert adapted_message["content"] is None
    assert adapted_message["tool_calls"][0]["function"]["name"] == "update_working_memory"


def test_order_gap_rejects_early_ready_and_owns_the_final_stop() -> None:
    bare_ready = '''```python
answer["content"] = "A) East Indonesia"
answer["ready"] = True
```'''
    supported_ready = '''```repl
answer["content"] = "FINAL_ANSWER: A) East Indonesia"
answer["ready"] = True
```'''
    text_update = (
        "<tool_call>\n"
        + json.dumps(
            {
                "name": "update_working_memory",
                "arguments": {
                    "entity": "Kalamang",
                    "selected_fact_ids": ["f1"],
                },
            }
        )
        + "\n</tool_call>"
    )
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(content=bare_ready),
            _response(content=text_update),
            _response(content=supported_ready),
            _response(content=supported_ready),
            _response(content=supported_ready),
        ]
    )
    session = _session(
        client,
        tool_choice="required",
        require_memory_update=True,
        validate_memory_updates=False,
    )

    proposed = "FINAL_ANSWER: A\nCITED_FACT_IDS: f1"
    assert session.complete(client, _messages()) == proposed
    assert session.complete(client, _messages()) == proposed
    completion = session.complete(client, _messages())

    assert 'answer["ready"] = True' in completion
    assert session.controller_termination_reason == "order_gap_supported_answer"
    assert session.working_memory_fact_ids == {"f1"}
    assert any(
        event.get("event") == "invalid_ready_rejected"
        for event in session.trace["events"]
    )
    assert any(
        event.get("event") == "model_ready_deferred"
        for event in session.trace["events"]
    )
    assert any(
        event.get("event") == "citations_grounded" and event.get("synthesized")
        for event in session.trace["events"]
    )
    memory_request = client.completions.requests[2]
    assert memory_request["tool_choice"] == {
        "type": "function",
        "function": {"name": "update_working_memory"},
    }
    assert [tool["function"]["name"] for tool in memory_request["tools"]] == [
        "update_working_memory"
    ]
    assert "tools" not in client.completions.requests[-1]


def test_order_gap_normalizes_single_choice_ready_after_grounding() -> None:
    supported_ready = '''```repl
answer["content"] = "FINAL_ANSWER: A\\nCITED_FACT_IDS: f1"
answer["ready"] = True
```'''
    short_ready = '''```repl
answer["content"] = "A"
answer["ready"] = True
```'''
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search",
                        {"query": "Kalamang", "seed_entities": ["Kalamang"]},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "update",
                        {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                        name="update_working_memory",
                    )
                ]
            ),
            _response(content=supported_ready),
            _response(content=supported_ready),
            _response(content=short_ready),
        ]
    )
    session = _session(
        client,
        tool_choice="required",
        require_memory_update=True,
        validate_memory_updates=False,
    )

    assert session.complete(client, _messages()) == "FINAL_ANSWER: A\nCITED_FACT_IDS: f1"
    assert session.complete(client, _messages()) == "FINAL_ANSWER: A\nCITED_FACT_IDS: f1"
    completion = session.complete(client, _messages())

    assert "FINAL_ANSWER: A" in completion
    assert "CITED_FACT_IDS: f1" in completion
    assert session.controller_termination_reason == "order_gap_supported_answer"
    assert any(
        event.get("event") == "citations_grounded" and event.get("synthesized")
        for event in session.trace["events"]
    )


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
            _response(
                content='''```repl
answer["content"] = "FINAL_ANSWER: A\\nCITED_FACT_IDS: f1"
answer["ready"] = True
```'''
            ),
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
            termination_mode="external_budget",
        )

    assert events[:2] == ["model", "retrieval"], outcome.error
    assert base_client.subcall_prompts == ["Check whether returned fact f1 supports a choice"]
    assert len(base_client.completions.requests) == 3
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
            _response(
                content='''```repl
answer["content"] = "FINAL_ANSWER: A\\nCITED_FACT_IDS: fabricated"
answer["ready"] = True
```'''
            ),
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
            termination_mode="external_budget",
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
            "--model-timeout",
            "45",
            "--model-max-retries",
            "0",
            "--tool-trace-dir",
            "traces",
        ]
    )
    assert enabled.qwen_tool_retrieval is True
    assert enabled.max_tool_calls == 4
    assert enabled.tool_choice == "required"
    assert enabled.tool_timeout == 2.5
    assert enabled.model_timeout == 45.0
    assert enabled.model_max_retries == 0
    assert enabled.termination_mode == "order_gap"
    assert enabled.order_gap_epsilon == 0.025
    assert enabled.order_gap_window == 2
    assert enabled.order_gap_min_iterations == 2
    assert enabled.tool_trace_dir == Path("traces")
    assert "--rlm-retrieval" not in parser.format_help()
    assert "--rlm-retrieval-steps" not in parser.format_help()


def test_direct_cell6_cli_defaults_to_dense_ppr() -> None:
    parser = build_arg_parser(
        cell_id=6,
        label="rlm_kg_scallop",
        kind="rlm",
        retrieval="kg",
    )

    assert parser.parse_args([]).retrieval_mode == "dense_ppr"


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
        "pilot_input": Path("pilot_input.jsonl"),
        "limit": 1,
        "seed": 0,
        "model": "Qwen/Qwen3-4B",
        "vllm_base_url": "http://localhost:8000/v1",
        "api_key": "private-api-key",
        "results_dir": tmp_path,
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "private-neo4j-password",
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
        "model_timeout": 90.0,
        "model_max_retries": 0,
        "tool_trace_dir": None,
        "scallop_validator_url": "http://validator:8765",
    }
    enabled = _common_cell_args(SimpleNamespace(**common, qwen_tool_retrieval=True), 5)
    cell6 = _common_cell_args(SimpleNamespace(**common, qwen_tool_retrieval=True), 6)
    fixed = _common_cell_args(
        SimpleNamespace(**common, qwen_tool_retrieval=False, fixed_kg_retrieval=True), 5
    )

    assert "--qwen-tool-retrieval" in enabled
    assert enabled[enabled.index("--input") + 1] == "pilot_input.jsonl"
    assert "private-api-key" not in enabled
    assert "private-neo4j-password" not in enabled
    assert "--api-key" not in enabled
    assert "--neo4j-password" not in enabled
    assert "--scallop-validator-url" not in enabled
    assert cell6[cell6.index("--scallop-validator-url") + 1] == "http://validator:8765"
    assert "--max-tool-calls" in enabled
    assert enabled[enabled.index("--max-tool-calls") + 1] == "3"
    assert cell6[cell6.index("--max-tool-calls") + 1] == "3"
    assert "--no-allow-unsupported-fallback" in enabled
    assert "--no-allow-unsupported-fallback" in cell6
    assert "--rlm-retrieval" not in enabled
    assert enabled[enabled.index("--retrieval-mode") + 1] == "hybrid"
    assert cell6[cell6.index("--retrieval-mode") + 1] == "hybrid"
    assert "--ppr-seed-count" not in cell6
    assert enabled[enabled.index("--dense-failure-policy") + 1] == "error"
    assert enabled[enabled.index("--termination-mode") + 1] == "order_gap"
    assert enabled[enabled.index("--order-gap-epsilon") + 1] == "0.025"
    assert enabled[enabled.index("--order-gap-window") + 1] == "2"
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
        "neurosym.adapters.graph_source.open_graph_source", return_value=source
    ), patch(
        "neurosym.adapters.qwen_rlm.qwen_rlm_tool_answer", return_value=outcome
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


def test_cell2_and_cell3_execute_hybrid_retrieval_before_flat_qwen(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({**EXAMPLE, "answer": "A"}) + "\n", encoding="utf-8")

    open_calls = []

    def open_source(**kwargs):
        open_calls.append(kwargs)
        return _hybrid_graph_source(kwargs)

    with patch("openai.OpenAI", return_value=object()), patch(
        "neurosym.adapters.graph_source.open_graph_source",
        side_effect=open_source,
    ), patch("experiments.flat_answerer.flat_answer", return_value=("A", "A")):
        for cell_id, label, session in [
            (2, "flat_kg_noscallop", "pilot_noscallop"),
            (3, "flat_kg_scallop", "pilot_scallop"),
        ]:
            output_path = tmp_path / f"cell{cell_id}" / "results.jsonl"
            run_cell(
                cell_id=cell_id,
                label=label,
                kind="flat",
                retrieval="kg",
                session_id=session,
                argv=[
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--scallop-validator-url",
                    "http://validator:8765",
                    "--no-aggregate",
                ],
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))
            assert result["configured_retrieval_mode"] == "hybrid"
            assert result["effective_retrieval_mode"] == "hybrid"
            assert result["retrieval_degraded"] is False
            assert result["retrieval_branch_counts"] == {"sparse": 1, "dense": 1}
            assert result["dense_index_identity"]
            assert result["validator_backend"] == ("none" if cell_id == 2 else "scallop")

    assert [call["validator_url"] for call in open_calls] == [None, None]


def test_cell5_and_cell6_enable_integrated_mode_specific_defaults(tmp_path: Path) -> None:
    class NoPreRetrievalSource(GraphSource):
        def context_for(self, *args, **kwargs):
            raise AssertionError("cell 6 must use integrated retrieval by default")

    def hybrid_answer(**kwargs):
        graph_source = kwargs["graph_source"]
        retrieval = graph_source.retrieve(
            query=EXAMPLE["question"],
            seed_entities=["Kalamang"],
            example_id=EXAMPLE["_id"],
            hops=2,
            top_k=10,
        )
        return SimpleNamespace(
            predicted="A",
            error=None,
            retrieved_fact_count=len(retrieval.rows),
            tool_result_chars=300,
            cited_fact_ids=[row["fact_id"] for row in retrieval.rows],
            trace={
                "orchestration": "qwen_native_tool_inside_rlm",
                "termination_reason": "supported_final_answer",
            },
            termination_reason="supported_final_answer",
    )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({**EXAMPLE, "answer": "A"}) + "\n", encoding="utf-8")

    open_calls = []

    def open_source(**kwargs):
        open_calls.append(kwargs)
        return _hybrid_graph_source(kwargs, NoPreRetrievalSource)

    with patch(
        "neurosym.adapters.graph_source.open_graph_source",
        side_effect=open_source,
    ), patch(
        "neurosym.adapters.qwen_rlm.qwen_rlm_tool_answer", side_effect=hybrid_answer
    ) as answer:
        for cell_id, label, session in [
            (5, "rlm_kg_noscallop", "pilot_noscallop"),
            (6, "rlm_kg_scallop", "pilot_scallop"),
        ]:
            output_path = tmp_path / f"cell{cell_id}" / "results.jsonl"
            run_cell(
                cell_id=cell_id,
                label=label,
                kind="rlm",
                retrieval="kg",
                session_id=session,
                argv=[
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--scallop-validator-url",
                    "http://validator:8765",
                    "--no-aggregate",
                ],
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))
            assert result["predicted"] == "A"
            assert result["n_triples"] > 0
            expected_mode = "hybrid" if cell_id == 5 else "dense_ppr"
            assert result["configured_retrieval_mode"] == expected_mode
            assert result["effective_retrieval_mode"] == expected_mode
            assert result["retrieval_degraded"] is False
            if cell_id == 5:
                assert result["retrieval_branch_counts"] == {
                    "sparse": 1,
                    "dense": 1,
                }
            else:
                assert result["retrieval_branch_counts"]["dense"] == 1
                assert result["retrieval_branch_counts"]["ppr"] >= 1
            assert result["dense_index_identity"]
            if cell_id == 5:
                assert result["validator_backend"] == "none"

    assert answer.call_count == 2
    assert [call["validator_url"] for call in open_calls] == [None, "http://validator:8765"]


# ---------------------------------------------------------------------------
# answer_format="short" -- HotpotQA / 2WikiMultihopQA free-text answers
# ---------------------------------------------------------------------------


def test_short_format_question_message_omits_mcq_choices() -> None:
    message = _question_message(EXAMPLE, answer_format="short")
    assert message == "Question: Where is Kalamang spoken?"
    assert "A)" not in message and "choice" not in message.lower()


def test_short_format_instructions_drop_mcq_letters() -> None:
    instructions = _native_tool_instructions(
        2, allow_unsupported_fallback=True, answer_format="short"
    )
    assert "<A|B|C|D>" not in instructions
    assert "give your best short answer" in instructions.lower()
    assert 'FINAL_ANSWER: <brief answer' in instructions


def test_short_format_parses_free_text_final_answer() -> None:
    predicted, citations, insufficient = _parse_final_response(
        'FINAL_ANSWER: East Indonesia\nCITED_FACT_IDS: f1, f2',
        answer_format="short",
    )
    assert predicted == "East Indonesia"
    assert citations == ["f1", "f2"]
    assert insufficient is False


def test_short_format_candidate_answer_requires_structured_label() -> None:
    assert (
        _parse_candidate_answer(
            'FINAL_ANSWER: East Indonesia', answer_format="short"
        )
        == "East Indonesia"
    )
    # Unlike MCQ mode, short mode never infers an answer from bare prose --
    # there is no fixed letter set to pattern-match against.
    assert _parse_candidate_answer("East Indonesia.", answer_format="short") == ""


def test_short_format_evidence_grounds_against_predicted_text_not_a_choice_key() -> None:
    client = FakeRLMClient([])
    session = _session(
        client,
        answer_format="short",
        choices={},
        question="Where is Kalamang spoken?",
    )
    session.retrieved_fact_ids = {"f1", "f2"}
    session.retrieved_facts = {row["fact_id"]: row for row in FACTS if "fact_id" in row}
    session.working_memory_fact_ids = {"f1"}

    grounded = session._grounded_fact_ids("East Indonesia")
    assert grounded == ["f1"]

    assessment = session._evidence_assessment("East Indonesia", ["f1"])
    assert assessment["adequate"] is True
    assert assessment["missing_evidence"] == []

    unsupported = session._evidence_assessment("Antarctica", ["f1"])
    assert unsupported["adequate"] is False
    assert "cited facts do not support the answer" in unsupported["missing_evidence"]

    uncited = session._evidence_assessment("East Indonesia", [])
    assert uncited["adequate"] is False
    assert "no cited fact IDs" in uncited["missing_evidence"]


def test_short_format_full_trajectory_search_commit_then_free_text_answer() -> None:
    client = FakeRLMClient(
        [
            _response(tool_calls=[_tool_call(
                "search", {"query": "Kalamang", "seed_entities": ["Kalamang"]}
            )]),
            _response(tool_calls=[_tool_call(
                "update",
                {"entity": "Kalamang", "selected_fact_ids": ["f1"]},
                name="update_working_memory",
            )]),
            _response(
                content='''```repl
answer["content"] = "FINAL_ANSWER: East Indonesia\\nCITED_FACT_IDS: f1"
answer["ready"] = True
```'''
            ),
        ]
    )
    session = _session(
        client,
        max_tool_calls=3,
        require_memory_update=True,
        validate_memory_updates=False,
        answer_format="short",
        choices={},
    )

    content = session.complete(client, _messages())

    assert "FINAL_ANSWER: East Indonesia" in content
    assert session.working_memory_fact_ids == {"f1"}
    assert session.diagnostic_predicted == "East Indonesia"
    # The system prompt sent to the model should never mention A/B/C/D.
    first_system_content = client.completions.requests[0]["messages"][0]["content"]
    assert "<A|B|C|D>" not in first_system_content


def test_short_answer_scores_normalizes_and_computes_f1() -> None:
    assert normalize_short_answer("The Galați City!") == "galați city"
    exact, f1 = short_answer_scores("Galati City", "the Galați city")
    assert exact is False  # diacritic mismatch, standard HotpotQA-style behavior
    assert f1 == 0.5  # only "city" token overlaps
    exact, f1 = short_answer_scores("the city of Galati", "Galati")
    assert exact is False
    assert f1 == 0.5
    exact, f1 = short_answer_scores("A galati", "the Galati")
    assert exact is True
    assert f1 == 1.0
    exact, f1 = short_answer_scores("Bucharest", "Galati")
    assert exact is False
    assert f1 == 0.0


# ---------------------------------------------------------------------------
# Evidence-adequacy tightening + per-option query guidance
# ---------------------------------------------------------------------------


def test_choice_supports_fact_numeric_option_requires_matching_object() -> None:
    # A source table often lists several related numbers in the same
    # sentence (e.g. GPU counts across parallelism configs). A numeric MCQ
    # option must match the fact's actually-asserted value (object), not
    # merely appear somewhere in the fact's text.
    fact = {
        "subject": "Nemotron-4-340B-Base",
        "predicate": "USED_DURING_PRE_TRAINING",
        "object": "1536",
        "support_text": (
            "Configurations of 768, 1536, 3072, and 6144 GPUs were "
            "evaluated; 1536 was used during pre-training."
        ),
    }
    assert _choice_supports_fact("1536", fact) is True
    assert _choice_supports_fact("3072", fact) is False
    assert _choice_supports_fact("6144", fact) is False
    assert _choice_supports_fact("768", fact) is False


def test_option_evidence_flags_citations_that_also_support_a_conflicting_option() -> None:
    client = FakeRLMClient([])
    session = _session(
        client,
        choices={
            "A": "Micro Adaptive Interface Component",
            "B": "Multi-Agent Incentive Communication",
            "C": "Measure, Analyze, Improve, Control",
            "D": "Massive AI-powered Courses",
        },
    )
    facts = {
        "f-b": {
            "fact_id": "f-b",
            "subject": "MAIC",
            "predicate": "FULL_FORM",
            "object": "Multi-Agent Incentive Communication",
            "support_text": "MAIC stands for Multi-Agent Incentive Communication.",
        },
        "f-d": {
            "fact_id": "f-d",
            "subject": "MAIC",
            "predicate": "FULL_NAME",
            "object": "Massive AI-powered Courses",
            "support_text": "MAIC is short for Massive AI-powered Courses.",
        },
    }
    session.retrieved_fact_ids = {"f-b", "f-d"}
    session.retrieved_facts = facts
    session.working_memory_fact_ids = {"f-b", "f-d"}

    # Citing only the fact that actually supports the predicted option: fine.
    clean = session._option_evidence("B", ["f-b"])
    assert clean["adequate"] is True
    assert clean["missing_evidence"] == []

    # Citing both -- one supports B, the other supports the conflicting
    # option D -- should no longer be treated as adequate.
    conflicted = session._option_evidence("B", ["f-b", "f-d"])
    assert conflicted["adequate"] is False
    assert any(
        "conflicting option(s) D" in reason
        for reason in conflicted["missing_evidence"]
    )


def test_native_turn_budget_exhaustion_after_rejection_falls_back_instead_of_erroring() -> None:
    # A mismatched citation gets rejected by the (now-tightened)
    # evidence-adequacy check every time the model repeats it. Once native
    # turns run out, the trajectory should fall back to an uncited answer
    # instead of raising "native tool turn limit reached without a text
    # response" and losing the candidate entirely.
    mismatched_ready = '''```repl
answer["content"] = "FINAL_ANSWER: C\\nCITED_FACT_IDS: f1"
answer["ready"] = True
```'''
    client = FakeRLMClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "search", {"query": "Kalamang", "seed_entities": ["Kalamang"]}
                    )
                ]
            ),
            _response(content=mismatched_ready),
            _response(content=mismatched_ready),
        ]
    )
    session = _session(client, max_tool_calls=1, require_memory_update=False)

    completion = session.complete(client, _messages())

    assert 'answer["ready"] = True' in completion
    assert "FINAL_ANSWER: C" in completion
    assert "CITED_FACT_IDS" not in completion
    assert (
        session.controller_termination_reason
        == "native_turn_budget_exhausted_after_rejection"
    )
    assert any(
        event.get("event") == "ready_rejection_accepted_out_of_turns"
        for event in session.trace["events"]
    )


