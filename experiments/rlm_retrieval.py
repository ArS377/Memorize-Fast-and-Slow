"""Native Qwen knowledge-graph tools integrated with the RLM loop.

The root Qwen model sees the question before retrieval and can issue native
OpenAI-compatible tool calls. The surrounding ``rlms`` RLM still owns its
persistent REPL, iterative reasoning, and recursive model calls. This replaces
the former prompt-parsed retrieval planner; it is not a separate answerer.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from experiments.graph_context import GraphSource
from experiments.kg_search_tool import (
    SEARCH_KNOWLEDGE_GRAPH_TOOL,
    TOOL_NAME as SEARCH_TOOL_NAME,
    execute_search_knowledge_graph,
)


TRACE_SCHEMA_VERSION = "qwen_rlm_tool_trace.v1"

SearchToolExecutor = Callable[
    [Mapping[str, Any], GraphSource, str],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class QwenRLMToolOutcome:
    """Terminal result from one integrated Qwen/tool/RLM trajectory."""

    status: str
    predicted: str
    raw_answer: str
    error: Optional[str]
    termination_reason: str
    cited_fact_ids: List[str]
    retrieved_fact_ids: List[str]
    tool_call_count: int
    retrieved_fact_count: int
    tool_result_chars: int
    trace: Dict[str, Any]


def _tool_error(
    code: str,
    message: str,
    *,
    retryable: bool,
    graph_source: GraphSource,
    example_id: str,
    request: Optional[Dict[str, Any]] = None,
    details: Optional[List[str]] = None,
) -> Dict[str, Any]:
    error: Dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if details:
        error["details"] = details
    return {
        "status": "error",
        "tool": SEARCH_TOOL_NAME,
        "request": request,
        "scope": {
            "example_id": str(example_id),
            "session_id": (
                str(graph_source.session_id)
                if getattr(graph_source, "session_id", None) is not None
                else None
            ),
            "memory_scope": str(getattr(graph_source, "memory_scope", "example")),
        },
        "results": [],
        "result_count": 0,
        "truncated": False,
        "empty_reason": None,
        "error": error,
    }


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _tool_call_dict(tool_call: Any, *, fallback_id: str) -> Dict[str, Any]:
    function = _get(tool_call, "function", {})
    return {
        "id": str(_get(tool_call, "id", None) or fallback_id),
        "type": str(_get(tool_call, "type", None) or "function"),
        "function": {
            "name": str(_get(function, "name", "")),
            "arguments": _get(function, "arguments", ""),
        },
    }


def _assistant_message_dict(message: Any, *, turn: int) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        dumped = message.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            dumped["role"] = "assistant"
            return dumped

    calls = [
        _tool_call_dict(call, fallback_id=f"tool-call-{turn}-{index}")
        for index, call in enumerate(_get(message, "tool_calls", None) or [], start=1)
    ]
    output: Dict[str, Any] = {
        "role": "assistant",
        "content": _get(message, "content", None),
    }
    if calls:
        output["tool_calls"] = calls
    return output


def _parse_arguments(raw_arguments: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if isinstance(raw_arguments, Mapping):
        return dict(raw_arguments), None
    if not isinstance(raw_arguments, str):
        return None, "tool arguments must be a JSON object"
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return None, f"malformed JSON arguments: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, "tool arguments must decode to a JSON object"
    return parsed, None


def _normalise_tool_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, Mapping):
        output = dict(response)
    elif is_dataclass(response):
        output = asdict(response)
    elif hasattr(response, "model_dump"):
        output = response.model_dump()
    elif hasattr(response, "to_dict"):
        output = response.to_dict()
    else:
        raise TypeError("tool executor must return a mapping or serializable response object")
    return json.loads(json.dumps(output, ensure_ascii=False))


def _execute_with_timeout(
    execute_tool: SearchToolExecutor,
    arguments: Mapping[str, Any],
    graph_source: GraphSource,
    example_id: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kg-search-tool")
    future = pool.submit(execute_tool, arguments, graph_source, example_id)
    try:
        return _normalise_tool_response(future.result(timeout=timeout_seconds))
    except FutureTimeoutError:
        future.cancel()
        return _tool_error(
            "backend_timeout",
            f"Knowledge graph retrieval exceeded {timeout_seconds:g} seconds.",
            retryable=True,
            graph_source=graph_source,
            example_id=example_id,
            request=dict(arguments),
        )
    except Exception as exc:
        return _tool_error(
            "backend_failure",
            "Knowledge graph retrieval failed.",
            retryable=False,
            graph_source=graph_source,
            example_id=example_id,
            request=dict(arguments),
            details=[f"exception_type={type(exc).__name__}"],
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _question_message(example: Mapping[str, Any]) -> str:
    return (
        f"Question: {example.get('question', '')}\n"
        f"A) {example.get('choice_A', '')}\n"
        f"B) {example.get('choice_B', '')}\n"
        f"C) {example.get('choice_C', '')}\n"
        f"D) {example.get('choice_D', '')}"
    )


def _native_tool_instructions(max_tool_calls: int) -> str:
    return f"""
The root model also has the native `{SEARCH_TOOL_NAME}` function. The REPL
contains no pre-retrieved facts. Call the native function when graph evidence
is needed; do not print or hand-parse a JSON retrieval action. Tool results are
retained across RLM iterations.

Use precise seed entities for sparse matching. After status="ok" with an empty
results list, reformulate with an alias or follow an intermediate entity. A
status="error" response is a failed call, not a no-hit. You may make at most
{max_tool_calls} tool calls in the entire RLM trajectory.

You may use the RLM REPL, `llm_query`, and `rlm_query` to reason over returned
facts. The terminal RLM answer must be one of:

FINAL(FINAL_ANSWER: <A|B|C|D>\nCITED_FACT_IDS: <returned fact IDs>)
FINAL(EVIDENCE_INSUFFICIENT: <brief reason>)

Never accept an answer supported by a fact ID that was not returned by the
native tool.
""".strip()


_FINAL_ANSWER_RE = re.compile(r"^\s*FINAL_ANSWER\s*:\s*([ABCD])\s*$", re.MULTILINE)
_CITATIONS_RE = re.compile(r"^\s*CITED_FACT_IDS\s*:\s*(.+?)\s*$", re.MULTILINE)
_INSUFFICIENT_RE = re.compile(r"^\s*EVIDENCE_INSUFFICIENT\s*:", re.MULTILINE)


def _parse_final_response(content: str) -> Tuple[str, List[str], bool]:
    text = content or ""
    if _INSUFFICIENT_RE.search(text):
        return "", [], True
    answer_match = _FINAL_ANSWER_RE.search(text)
    citations_match = _CITATIONS_RE.search(text)
    if not answer_match:
        return "", [], False
    citations: List[str] = []
    if citations_match:
        raw = citations_match.group(1).strip().strip("[]")
        seen: Set[str] = set()
        for item in raw.split(","):
            fact_id = item.strip().strip("`\"'")
            if fact_id and fact_id not in seen:
                citations.append(fact_id)
                seen.add(fact_id)
    return answer_match.group(1), citations, False


class NativeToolSession:
    """State shared by native tool calls across one root RLM completion."""

    def __init__(
        self,
        *,
        model: str,
        graph_source: GraphSource,
        example_id: str,
        max_tool_calls: int,
        tool_choice: str,
        tool_timeout: float,
        max_completion_tokens: int,
        tool_schema: Mapping[str, Any],
        execute_tool: SearchToolExecutor,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        if tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")
        if max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be at least 1")
        if tool_choice not in {"auto", "required"}:
            raise ValueError("tool_choice must be 'auto' or 'required'")

        self.model = model
        self.graph_source = graph_source
        self.example_id = example_id
        self.max_tool_calls = max_tool_calls
        self.tool_choice = tool_choice
        self.tool_timeout = tool_timeout
        self.max_completion_tokens = max_completion_tokens
        self.tool_schema = dict(tool_schema)
        self.execute_tool = execute_tool
        self.seen_calls: Set[str] = set()
        self.retrieved_fact_ids: Set[str] = set()
        self.tool_call_count = 0
        self.tool_result_chars = 0
        self.evidence_responses: List[Dict[str, Any]] = []
        self.model_completion_count = 0
        self.trace: Dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "example_id": example_id,
            "session_id": graph_source.session_id,
            "memory_scope": graph_source.memory_scope,
            "model": model,
            "tool_name": SEARCH_TOOL_NAME,
            "tool_choice": tool_choice,
            "max_tool_calls": max_tool_calls,
            "orchestration": "qwen_native_tool_inside_rlm",
            "events": [],
        }

    def _messages_with_state(self, prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages = [dict(message) for message in prompt]
        instructions = _native_tool_instructions(self.max_tool_calls)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{instructions}"
        else:
            messages.insert(0, {"role": "system", "content": instructions})

        if self.evidence_responses:
            evidence = json.dumps(self.evidence_responses, ensure_ascii=False, separators=(",", ":"))
            reminder = (
                "\n\nNative knowledge-graph responses retained from earlier RLM iterations:\n"
                f"{evidence}"
            )
            if messages and messages[-1].get("role") in {"user", "assistant"}:
                messages[-1]["content"] = f"{messages[-1].get('content') or ''}{reminder}"
            else:
                messages.append({"role": "user", "content": reminder.strip()})
        return messages

    def _track_response(self, base_client: Any, response: Any) -> None:
        track_cost = getattr(base_client, "_track_cost", None)
        if callable(track_cost):
            track_cost(response, self.model)

    def complete(self, base_client: Any, prompt: List[Dict[str, Any]]) -> str:
        """Run native tool subturns, returning text for the surrounding RLM turn."""
        messages = self._messages_with_state(prompt)
        self.model_completion_count += 1
        completion_index = self.model_completion_count
        if "initial_model_messages" not in self.trace:
            self.trace["initial_model_messages"] = [dict(message) for message in messages]

        max_native_turns = self.max_tool_calls + 2
        for native_turn in range(1, max_native_turns + 1):
            request_tool_choice = (
                "none" if self.tool_call_count >= self.max_tool_calls else self.tool_choice
            )
            self.trace["events"].append(
                {
                    "event": "model_request",
                    "rlm_completion": completion_index,
                    "native_turn": native_turn,
                    "tool_choice": request_tool_choice,
                    "message_count": len(messages),
                }
            )
            try:
                response = base_client.client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    tools=[self.tool_schema],
                    tool_choice=request_tool_choice,
                    temperature=0.0,
                    max_tokens=self.max_completion_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                self._track_response(base_client, response)
            except Exception as exc:
                self.trace["events"].append(
                    {
                        "event": "model_error",
                        "rlm_completion": completion_index,
                        "native_turn": native_turn,
                        "error": str(exc),
                    }
                )
                raise RuntimeError(f"Qwen native-tool completion failed: {exc}") from exc

            choices = _get(response, "choices", []) or []
            if not choices:
                raise RuntimeError("Qwen native-tool completion returned no choices")
            choice = choices[0]
            assistant_message = _assistant_message_dict(
                _get(choice, "message"), turn=native_turn
            )
            messages.append(assistant_message)
            self.trace["events"].append(
                {
                    "event": "assistant_message",
                    "rlm_completion": completion_index,
                    "native_turn": native_turn,
                    "finish_reason": _get(choice, "finish_reason"),
                    "message": assistant_message,
                }
            )

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                content = str(assistant_message.get("content") or "")
                if not content.strip():
                    raise RuntimeError("Qwen returned neither tool calls nor text")
                return content

            for index, raw_call in enumerate(tool_calls, start=1):
                call = _tool_call_dict(
                    raw_call, fallback_id=f"tool-call-{completion_index}-{native_turn}-{index}"
                )
                call_id = call["id"]
                name = call["function"]["name"]
                raw_arguments = call["function"]["arguments"]
                arguments: Optional[Dict[str, Any]] = None
                signature: Optional[str] = None
                elapsed_seconds = 0.0

                if self.tool_call_count >= self.max_tool_calls:
                    tool_response = _tool_error(
                        "call_limit_exceeded",
                        f"maximum of {self.max_tool_calls} tool calls already reached",
                        retryable=False,
                        graph_source=self.graph_source,
                        example_id=self.example_id,
                    )
                else:
                    self.tool_call_count += 1
                    arguments, parse_error = _parse_arguments(raw_arguments)
                    if parse_error:
                        tool_response = _tool_error(
                            "malformed_arguments",
                            parse_error,
                            retryable=True,
                            graph_source=self.graph_source,
                            example_id=self.example_id,
                        )
                    elif name != SEARCH_TOOL_NAME:
                        tool_response = _tool_error(
                            "unknown_tool",
                            f"unsupported tool: {name}",
                            retryable=True,
                            graph_source=self.graph_source,
                            example_id=self.example_id,
                            request=arguments,
                        )
                    else:
                        assert arguments is not None
                        signature = json.dumps(
                            {"name": name, "arguments": arguments},
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if signature in self.seen_calls:
                            tool_response = _tool_error(
                                "duplicate_tool_call",
                                "an identical tool call was already attempted",
                                retryable=True,
                                graph_source=self.graph_source,
                                example_id=self.example_id,
                                request=arguments,
                            )
                        else:
                            self.seen_calls.add(signature)
                            started = time.perf_counter()
                            tool_response = _execute_with_timeout(
                                self.execute_tool,
                                arguments,
                                self.graph_source,
                                self.example_id,
                                self.tool_timeout,
                            )
                            elapsed_seconds = round(time.perf_counter() - started, 6)

                self.trace["events"].append(
                    {
                        "event": "tool_call",
                        "rlm_completion": completion_index,
                        "native_turn": native_turn,
                        "call_id": call_id,
                        "name": name,
                        "raw_arguments": raw_arguments,
                        "arguments": arguments,
                        "signature": signature,
                    }
                )

                results = tool_response.get("results")
                if tool_response.get("status") == "ok" and isinstance(results, list):
                    for result in results:
                        if isinstance(result, Mapping) and result.get("fact_id"):
                            self.retrieved_fact_ids.add(str(result["fact_id"]))

                tool_content = json.dumps(
                    tool_response, ensure_ascii=False, separators=(",", ":")
                )
                self.tool_result_chars += len(tool_content)
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": tool_content}
                )
                self.evidence_responses.append(tool_response)
                self.trace["events"].append(
                    {
                        "event": "tool_result",
                        "rlm_completion": completion_index,
                        "native_turn": native_turn,
                        "call_id": call_id,
                        "response": tool_response,
                        "elapsed_seconds": elapsed_seconds,
                    }
                )

                error = tool_response.get("error")
                if tool_response.get("status") == "ok" and not results:
                    self.trace["events"].append(
                        {
                            "event": "retry",
                            "rlm_completion": completion_index,
                            "reason": "valid_empty_result",
                        }
                    )
                elif error:
                    self.trace["events"].append(
                        {
                            "event": "retry",
                            "rlm_completion": completion_index,
                            "reason": _get(error, "code", "tool_error"),
                        }
                    )

        raise RuntimeError("native tool turn limit reached without a text response")

    def finish(
        self,
        *,
        status: str,
        predicted: str,
        raw_answer: str,
        error: Optional[str],
        termination_reason: str,
        cited_fact_ids: List[str],
    ) -> QwenRLMToolOutcome:
        ordered_retrieved = sorted(self.retrieved_fact_ids)
        self.trace.update(
            {
                "status": status,
                "termination_reason": termination_reason,
                "cited_fact_ids": list(cited_fact_ids),
                "retrieved_fact_ids": ordered_retrieved,
                "tool_call_count": self.tool_call_count,
                "retrieved_fact_count": len(ordered_retrieved),
                "tool_result_chars": self.tool_result_chars,
                "rlm_model_completion_count": self.model_completion_count,
                "error": error,
            }
        )
        self.trace["events"].append(
            {
                "event": "termination",
                "status": status,
                "reason": termination_reason,
                "error": error,
            }
        )
        return QwenRLMToolOutcome(
            status=status,
            predicted=predicted,
            raw_answer=raw_answer,
            error=error,
            termination_reason=termination_reason,
            cited_fact_ids=list(cited_fact_ids),
            retrieved_fact_ids=ordered_retrieved,
            tool_call_count=self.tool_call_count,
            retrieved_fact_count=len(ordered_retrieved),
            tool_result_chars=self.tool_result_chars,
            trace=self.trace,
        )


class _NativeToolClient:
    """Wrap an RLM OpenAI client, adding tools only to root RLM turns."""

    def __init__(self, base_client: Any, tool_session: NativeToolSession) -> None:
        if not hasattr(base_client, "client"):
            raise TypeError("native Qwen tools require an OpenAI-compatible RLM client")
        self.base_client = base_client
        self.tool_session = tool_session
        self.model_name = base_client.model_name
        self.timeout = getattr(base_client, "timeout", None)

    def completion(self, prompt: Any, model: Optional[str] = None) -> str:
        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            return self.tool_session.complete(self.base_client, prompt)
        return self.base_client.completion(prompt, model=model)

    async def acompletion(self, prompt: Any, model: Optional[str] = None) -> str:
        return await self.base_client.acompletion(prompt, model=model)

    def get_usage_summary(self) -> Any:
        return self.base_client.get_usage_summary()

    def get_last_usage(self) -> Any:
        return self.base_client.get_last_usage()


class _NativeToolRLMMixin:
    """Replace the root handler's client without modifying the rlms package."""

    @contextmanager
    def _spawn_completion_context(self, prompt: Any):
        with super()._spawn_completion_context(prompt) as (lm_handler, environment):
            base_client = lm_handler.default_client
            wrapped = _NativeToolClient(base_client, self._native_tool_session)
            lm_handler.default_client = wrapped
            for model_name, registered in list(lm_handler.clients.items()):
                if registered is base_client:
                    lm_handler.clients[model_name] = wrapped
            yield lm_handler, environment


def make_qwen_tool_rlm(
    *,
    backend: str,
    model: str,
    base_url: str,
    api_key: str,
    max_depth: int,
    max_iterations: int,
    max_tokens: int,
    log_dir: Path,
    verbose: bool,
    tool_session: NativeToolSession,
) -> Any:
    """Create an RLM whose root Qwen calls the KG tool natively."""
    if backend != "openai":
        raise ValueError(
            "native Qwen/vLLM tool calls require --backend openai with the vLLM base URL"
        )

    from rlm.core.rlm import RLM
    from rlm.logger.rlm_logger import RLMLogger

    class NativeToolRLM(_NativeToolRLMMixin, RLM):
        pass

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    rlm = NativeToolRLM(
        backend=backend,
        backend_kwargs={"model_name": model, "base_url": base_url, "api_key": api_key},
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        logger=RLMLogger(log_dir=str(log_dir)),
        verbose=verbose,
    )
    rlm._native_tool_session = tool_session
    return rlm


def qwen_rlm_tool_answer(
    *,
    backend: str,
    model: str,
    base_url: str,
    api_key: str,
    max_depth: int,
    max_iterations: int,
    max_tokens: int,
    log_dir: Path,
    verbose: bool,
    graph_source: GraphSource,
    example: Mapping[str, Any],
    max_tool_calls: int = 3,
    tool_choice: str = "auto",
    tool_timeout: float = 30.0,
    max_completion_tokens: int = 2048,
    tool_schema: Optional[Mapping[str, Any]] = None,
    execute_tool: SearchToolExecutor = execute_search_knowledge_graph,
) -> QwenRLMToolOutcome:
    """Run one Qwen-first native-tool trajectory inside the RLM structure."""
    example_id = str(example.get("_id", ""))
    session = NativeToolSession(
        model=model,
        graph_source=graph_source,
        example_id=example_id,
        max_tool_calls=max_tool_calls,
        tool_choice=tool_choice,
        tool_timeout=tool_timeout,
        max_completion_tokens=max_completion_tokens,
        tool_schema=tool_schema or SEARCH_KNOWLEDGE_GRAPH_TOOL,
        execute_tool=execute_tool,
    )
    rlm = make_qwen_tool_rlm(
        backend=backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        log_dir=log_dir,
        verbose=verbose,
        tool_session=session,
    )

    root_prompt = _question_message(example)
    empty_context = {
        "retrieval_state": (
            "No facts are preloaded. The root Qwen model must use the native "
            f"{SEARCH_TOOL_NAME} function when evidence is needed."
        )
    }
    try:
        result = rlm.completion(prompt=empty_context, root_prompt=root_prompt)
        raw_answer = "" if result is None else (getattr(result, "response", None) or str(result))
    except Exception as exc:
        return session.finish(
            status="error",
            predicted="",
            raw_answer="",
            error=str(exc),
            termination_reason="rlm_error",
            cited_fact_ids=[],
        )
    finally:
        close = getattr(rlm, "close", None)
        if callable(close):
            close()

    predicted, citations, evidence_insufficient = _parse_final_response(raw_answer)
    if evidence_insufficient:
        return session.finish(
            status="evidence_insufficient",
            predicted="",
            raw_answer=raw_answer,
            error=None,
            termination_reason="evidence_insufficient",
            cited_fact_ids=[],
        )

    unknown_citations = sorted(set(citations) - session.retrieved_fact_ids)
    if predicted and citations and not unknown_citations:
        return session.finish(
            status="supported",
            predicted=predicted,
            raw_answer=raw_answer,
            error=None,
            termination_reason="supported_final_answer",
            cited_fact_ids=citations,
        )

    if unknown_citations:
        error = f"unknown cited fact IDs: {', '.join(unknown_citations)}"
    elif predicted and not citations:
        error = "supported answer omitted CITED_FACT_IDS"
    else:
        error = "RLM did not return the required supported-answer or evidence-insufficient format"
    return session.finish(
        status="error",
        predicted="",
        raw_answer=raw_answer,
        error=error,
        termination_reason="invalid_final_answer",
        cited_fact_ids=[],
    )
