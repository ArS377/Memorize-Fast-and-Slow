"""Native Qwen knowledge-graph tools integrated with the RLM loop.

The root Qwen model sees the question before retrieval and can issue native
OpenAI-compatible tool calls. The surrounding ``rlms`` RLM still owns its
persistent REPL, iterative reasoning, and recursive model calls. This replaces
the former prompt-parsed retrieval planner; it is not a separate answerer.
"""

from __future__ import annotations

import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from neurosym.domain.epistemic_state import EpistemicStateTracker, StateTransition
from neurosym.adapters.graph_source import GraphSource
from neurosym.adapters.kg_search import (
    SEARCH_KNOWLEDGE_GRAPH_TOOL,
    TOOL_NAME as SEARCH_TOOL_NAME,
    execute_search_knowledge_graph,
)
from neurosym.adapters.working_memory_tool import (
    TOOL_NAME as UPDATE_TOOL_NAME,
    UPDATE_WORKING_MEMORY_TOOL,
    execute_update_working_memory,
)


TRACE_SCHEMA_VERSION = "qwen_rlm_tool_trace.v4"

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
    working_memory_artifact_ids: List[str]
    diagnostic_predicted: str = ""
    termination_mode: str = "external_budget"
    order_gap_final: Optional[float] = None
    order_gap_window_mean: Optional[float] = None
    rlm_completion_count: int = 0


def _tool_error(
    code: str,
    message: str,
    *,
    retryable: bool,
    graph_source: GraphSource,
    example_id: str,
    request: Optional[Dict[str, Any]] = None,
    details: Optional[List[str]] = None,
    tool_name: str = SEARCH_TOOL_NAME,
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
        "tool": tool_name,
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
        "working_memory": None,
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


_TEXT_TOOL_CALL_RE = re.compile(
    r"\A\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*\Z",
    re.DOTALL,
)


def _parse_text_tool_call(content: str, *, fallback_id: str) -> Optional[Dict[str, Any]]:
    """Adapt Qwen's documented text wrapper when vLLM does not parse it.

    Some vLLM configurations return Qwen's native ``<tool_call>`` wrapper as
    assistant text under ``tool_choice=auto``. Accept only a full-message,
    single JSON wrapper; all normal argument and tool-name validation still
    happens in the regular execution path.
    """
    match = _TEXT_TOOL_CALL_RE.fullmatch(str(content or ""))
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    name = str(payload.get("name") or "").strip()
    arguments = payload.get("arguments", {})
    if not name or not isinstance(arguments, (str, Mapping)):
        return None
    if isinstance(arguments, Mapping):
        arguments = json.dumps(
            dict(arguments), ensure_ascii=False, separators=(",", ":")
        )
    return {
        "id": fallback_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


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
The root model has native `{SEARCH_TOOL_NAME}` and `{UPDATE_TOOL_NAME}` functions. The REPL
contains no pre-retrieved facts. Call the native function when graph evidence
is needed; do not print or hand-parse a JSON retrieval action. Tool results are
retained across RLM iterations.

Use precise seed entities so the configured retriever can use its entity branch.
On the first search, omit `predicates`. Only use exact UPPER_SNAKE_CASE predicate
names copied from an earlier tool result; never invent generic predicate filters.
After status="ok" with an empty results list, reformulate with an alias or follow
an intermediate entity. A
status="error" response is a failed call, not a no-hit. You may make at most
{max_tool_calls} tool calls in the entire RLM trajectory.

After selecting evidence, call `{UPDATE_TOOL_NAME}` with the returned fact IDs
to compile shaped working memory. Never invent an ID or pass scope/session
arguments. Derived facts must cite selected returned fact IDs. The application,
not the model, owns memory scope and decides whether Scallop gates the update.

You may use the RLM REPL, `llm_query`, and `rlm_query` to reason over returned
facts. To finish, do not answer in prose. Emit one `repl` code block that sets
the RLM answer and marks it ready. The fence label must be `repl`, not `python`,
for example:

```repl
answer["content"] = "FINAL_ANSWER: <A|B|C|D>\\nCITED_FACT_IDS: <returned fact IDs>"
answer["ready"] = True
```

For insufficient evidence, set `answer["content"]` to
`EVIDENCE_INSUFFICIENT: <brief reason>` and set `answer["ready"] = True`.

Never accept an answer supported by a fact ID that was not returned by the
native tool.
""".strip()


_FINAL_ANSWER_RE = re.compile(
    r"^\s*FINAL_ANSWER\s*:\s*([ABCD])(?=\s|\)|$).*$",
    re.MULTILINE,
)
_CITATIONS_RE = re.compile(r"^\s*CITED_FACT_IDS\s*:\s*(.+?)\s*$", re.MULTILINE)
_INSUFFICIENT_RE = re.compile(r"^\s*EVIDENCE_INSUFFICIENT\s*:", re.MULTILINE)
_PROSE_ANSWER_RE = re.compile(
    r"\b(?:the\s+)?(?:final\s+answer|correct\s+answer|answer)\s*"
    r"(?:is|:|-)\s*\(?([ABCD])\)?\b",
    re.IGNORECASE,
)
_GROUNDING_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}


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


def _parse_candidate_answer(content: str) -> str:
    """Extract only an explicitly labelled multiple-choice answer from prose."""
    text = str(content or "")
    structured = re.search(r"\bFINAL_ANSWER\s*:\s*([ABCD])\b", text)
    if structured:
        return structured.group(1).upper()
    leading = re.match(
        r"^\s*([ABCD])(?:\s*[).:\-]|\s+|$)", text, re.IGNORECASE
    )
    if leading:
        return leading.group(1).upper()
    matches = list(_PROSE_ANSWER_RE.finditer(text))
    return matches[-1].group(1).upper() if matches else ""


def _grounding_text(value: Any) -> str:
    return " ".join(
        re.findall(r"[^\W_]+", str(value or "").casefold(), flags=re.UNICODE)
    )


def _grounding_score(choice: str, fact: Mapping[str, Any]) -> float:
    choice_text = _grounding_text(choice)
    fact_text = _grounding_text(
        " ".join(
            str(fact.get(field) or "")
            for field in ("subject", "predicate", "object", "support_text")
        )
    )
    if not choice_text or not fact_text:
        return 0.0
    if len(choice_text) >= 4 and choice_text in fact_text:
        return 1.0
    choice_tokens = {
        token
        for token in choice_text.split()
        if token not in _GROUNDING_STOPWORDS
    }
    if not choice_tokens:
        return 0.0
    fact_tokens = set(fact_text.split())
    return len(choice_tokens.intersection(fact_tokens)) / len(choice_tokens)


def _rlm_ready_block(content: str) -> str:
    encoded = json.dumps(str(content), ensure_ascii=False)
    return (
        "```repl\n"
        f'answer["content"] = {encoded}\n'
        'answer["ready"] = True\n'
        "```"
    )


_REPL_BLOCK_RE = re.compile(
    r"```(?:repl|python)\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _answer_assignment_key(target: ast.expr) -> Optional[str]:
    if not isinstance(target, ast.Subscript):
        return None
    if not isinstance(target.value, ast.Name) or target.value.id != "answer":
        return None
    try:
        key = ast.literal_eval(target.slice)
    except (ValueError, TypeError):
        return None
    return str(key) if key in {"content", "ready"} else None


def _ready_answer_payload(response: str) -> Optional[str]:
    """Read a constant RLM answer assignment without executing model code."""
    for code in _REPL_BLOCK_RE.findall(str(response or "")):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        assignments: List[Tuple[int, str, Any]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                key = _answer_assignment_key(target)
                if key is not None:
                    assignments.append((node.lineno, key, value))
        values: Dict[str, Any] = {}
        for _line, key, value in sorted(assignments):
            values[key] = value
        if values.get("ready") is True and isinstance(values.get("content"), str):
            return values["content"]
    return None


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
        update_tool_schema: Mapping[str, Any] = UPDATE_WORKING_MEMORY_TOOL,
        execute_update_tool: Callable[..., Mapping[str, Any]] = execute_update_working_memory,
        validate_memory_updates: bool = True,
        require_memory_update: bool = False,
        question: str = "",
        choices: Optional[Mapping[str, Any]] = None,
        termination_mode: str = "order_gap",
        order_gap_epsilon: float = 0.025,
        order_gap_window: int = 2,
        order_gap_min_iterations: int = 2,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        if tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")
        if max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be at least 1")
        if tool_choice not in {"auto", "required"}:
            raise ValueError("tool_choice must be 'auto' or 'required'")
        if termination_mode not in {"order_gap", "external_budget"}:
            raise ValueError("termination_mode must be 'order_gap' or 'external_budget'")

        self.model = model
        self.graph_source = graph_source
        self.example_id = example_id
        self.max_tool_calls = max_tool_calls
        self.tool_choice = tool_choice
        self.tool_timeout = tool_timeout
        self.max_completion_tokens = max_completion_tokens
        self.tool_schema = dict(tool_schema)
        self.execute_tool = execute_tool
        self.update_tool_schema = dict(update_tool_schema)
        self.execute_update_tool = execute_update_tool
        self.validate_memory_updates = validate_memory_updates
        self.require_memory_update = require_memory_update
        self.termination_mode = termination_mode
        self.state_tracker = EpistemicStateTracker(
            question=question,
            choices=choices,
            epsilon=order_gap_epsilon,
            window=order_gap_window,
            min_iterations=order_gap_min_iterations,
        )
        self.seen_calls: Set[str] = set()
        self.retrieved_fact_ids: Set[str] = set()
        self.retrieved_facts: Dict[str, Dict[str, Any]] = {}
        self.working_memory_artifact_ids: List[str] = []
        self.working_memory_fact_ids: Set[str] = set()
        self.tool_call_count = 0
        self.tool_result_chars = 0
        self.model_completion_count = 0
        self.diagnostic_predicted = ""
        self.controller_termination_reason: Optional[str] = None
        self.trace: Dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "example_id": example_id,
            "session_id": graph_source.session_id,
            "memory_scope": graph_source.memory_scope,
            "configured_retrieval_mode": graph_source.retrieval_config.mode,
            "effective_retrieval_mode": None,
            "dense_index_identity": [
                graph_source.dense_indexes[key].manifest.identity
                for key in sorted(graph_source.dense_indexes)
            ],
            "model": model,
            "tool_names": [SEARCH_TOOL_NAME, UPDATE_TOOL_NAME],
            "tool_choice": tool_choice,
            "max_tool_calls": max_tool_calls,
            "termination_mode": termination_mode,
            "order_gap_config": {
                "epsilon": order_gap_epsilon,
                "window": order_gap_window,
                "min_iterations": order_gap_min_iterations,
            },
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

        state_view = json.dumps(
            self.state_tracker.state.prompt_view(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        reminder = (
            "\n\nController-owned epistemic state from prior RLM iterations:\n"
            f"{state_view}"
        )
        if messages and messages[-1].get("role") in {"user", "assistant"}:
            messages[-1]["content"] = f"{messages[-1].get('content') or ''}{reminder}"
        else:
            messages.append({"role": "user", "content": reminder.strip()})
        return messages

    def _observe_epistemic(
        self,
        evidence: Mapping[str, Any],
        *,
        completion_index: int,
        native_turn: int,
        completion_boundary: bool,
    ) -> StateTransition:
        transition = self.state_tracker.observe(
            evidence, completion_boundary=completion_boundary
        )
        self.trace["events"].append(
            {
                "event": "epistemic_transition",
                "rlm_completion": completion_index,
                "native_turn": native_turn,
                "evidence_kind": evidence.get("kind"),
                "completion_boundary": completion_boundary,
                "order_gap": transition.order_gap,
                "window_mean": self.state_tracker.window_mean,
                "before_digest": transition.before_digest,
                "after_digest": transition.after_digest,
                "state_summary": transition.state_summary,
            }
        )
        return transition

    def _state_based_completion(
        self,
        *,
        predicted: str,
        citations: List[str],
    ) -> Optional[str]:
        if self.termination_mode != "order_gap" or not self.state_tracker.stable:
            return None
        state = self.state_tracker.state
        successful_search = state.successful_searches > 0
        cited_ids = list(dict.fromkeys(citations))
        cited_set = set(cited_ids)
        citations_retrieved = bool(cited_ids) and cited_set <= self.retrieved_fact_ids
        citations_committed = (
            not self.require_memory_update or cited_set <= self.working_memory_fact_ids
        )
        supported = bool(
            predicted and citations_retrieved and citations_committed
        )
        coverage = {
            "successful_search": successful_search,
            "candidate": self.diagnostic_predicted or None,
            "cited_fact_ids": cited_ids,
            "citations_retrieved": citations_retrieved,
            "citations_committed": citations_committed,
            "require_memory_update": self.require_memory_update,
            "passed": successful_search and supported,
        }
        self.trace["events"].append(
            {
                "event": "order_gap_coverage",
                "rlm_completion": self.model_completion_count,
                "window_mean": self.state_tracker.window_mean,
                **coverage,
            }
        )
        if not successful_search:
            return None
        if supported:
            self.controller_termination_reason = "order_gap_supported_answer"
            final_content = (
                f"FINAL_ANSWER: {predicted}\n"
                f"CITED_FACT_IDS: {', '.join(cited_ids)}"
            )
        else:
            self.controller_termination_reason = "order_gap_evidence_insufficient"
            final_content = (
                "EVIDENCE_INSUFFICIENT: the epistemic state settled without "
                "a citable committed answer"
            )
        self.trace["events"].append(
            {
                "event": "order_gap_stop",
                "rlm_completion": self.model_completion_count,
                "reason": self.controller_termination_reason,
                "diagnostic_predicted": self.diagnostic_predicted,
                "cited_fact_ids": cited_ids,
                "window_mean": self.state_tracker.window_mean,
            }
        )
        return _rlm_ready_block(final_content)

    def _request_tool_choice(self) -> str:
        if (
            self.tool_call_count >= self.max_tool_calls
            or bool(self.working_memory_fact_ids)
            or self.state_tracker.state.current_answer_id is not None
        ):
            return "none"
        if (
            self.tool_choice == "required"
            and (
                self.state_tracker.state.successful_searches == 0
                or (
                    self.require_memory_update
                    and bool(self.retrieved_fact_ids)
                    and not self.working_memory_fact_ids
                )
            )
        ):
            return "required"
        return "auto"

    def _request_tools(self, request_tool_choice: str) -> List[Dict[str, Any]]:
        if (
            request_tool_choice == "required"
            and self.state_tracker.state.successful_searches > 0
            and self.require_memory_update
            and not self.working_memory_fact_ids
        ):
            return [self.update_tool_schema]
        return [self.tool_schema, self.update_tool_schema]

    def _grounded_fact_ids(self, predicted: str) -> List[str]:
        choice = str(self.state_tracker.state.choices.get(predicted) or "")
        ranked = sorted(
            (
                (_grounding_score(choice, self.retrieved_facts[fact_id]), fact_id)
                for fact_id in self.working_memory_fact_ids
                if fact_id in self.retrieved_facts
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [fact_id for score, fact_id in ranked if score >= 0.2]

    def _resolve_citations(
        self,
        predicted: str,
        citations: List[str],
    ) -> Tuple[List[str], bool]:
        if not predicted:
            return list(dict.fromkeys(citations)), False
        grounded = self._grounded_fact_ids(predicted)
        grounded_set = set(grounded)
        provided = list(dict.fromkeys(citations))
        if provided:
            return [fact_id for fact_id in provided if fact_id in grounded_set], False
        return grounded[:3], bool(grounded)

    def _completion_errors(self, predicted: str, citations: List[str]) -> List[str]:
        errors: List[str] = []
        cited = set(citations)
        if not predicted:
            errors.append("use the exact FINAL_ANSWER and CITED_FACT_IDS format")
        if not citations:
            errors.append("cite at least one returned fact ID")
        unknown = sorted(cited - self.retrieved_fact_ids)
        if unknown:
            errors.append("unknown fact IDs: " + ", ".join(unknown))
        missing = sorted(cited - self.working_memory_fact_ids)
        if self.require_memory_update and missing:
            errors.append(
                "fact IDs not committed to working memory: " + ", ".join(missing)
            )
        return errors

    def _completion_correction(self, errors: List[str]) -> str:
        retrieved = ", ".join(sorted(self.retrieved_fact_ids)) or "none"
        committed = ", ".join(sorted(self.working_memory_fact_ids)) or "none"
        return (
            "The controller rejected that ready signal: "
            + "; ".join(errors)
            + ". Do not set answer[\"ready\"] yet. If evidence supports an option, "
            f"first call {UPDATE_TOOL_NAME} as needed, then use the exact final format. "
            f"Returned fact IDs: {retrieved}. Committed fact IDs: {committed}."
        )

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
            request_tool_choice = self._request_tool_choice()
            self.trace["events"].append(
                {
                    "event": "model_request",
                    "rlm_completion": completion_index,
                    "native_turn": native_turn,
                    "tool_choice": request_tool_choice,
                    "message_count": len(messages),
                }
            )
            request: Dict[str, Any] = {
                "model": self.model,
                "messages": list(messages),
                "temperature": 0.0,
                "max_tokens": self.max_completion_tokens,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            if request_tool_choice != "none":
                request.update(
                    {
                        "tools": self._request_tools(request_tool_choice),
                        "tool_choice": request_tool_choice,
                    }
                )
            try:
                response = base_client.client.chat.completions.create(**request)
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
            raw_text_tool_call: Optional[str] = None
            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                content = str(assistant_message.get("content") or "")
                adapted = _parse_text_tool_call(
                    content,
                    fallback_id=f"text-tool-call-{completion_index}-{native_turn}",
                )
                if adapted is not None:
                    raw_text_tool_call = content
                    tool_calls = [adapted]
                    assistant_message["content"] = None
                    assistant_message["tool_calls"] = tool_calls
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
            if raw_text_tool_call is not None:
                self.trace["events"].append(
                    {
                        "event": "text_tool_call_adapted",
                        "rlm_completion": completion_index,
                        "native_turn": native_turn,
                        "raw_content": raw_text_tool_call,
                        "tool_call": tool_calls[0],
                    }
                )

            if not tool_calls:
                content = str(assistant_message.get("content") or "")
                if not content.strip():
                    raise RuntimeError("Qwen returned neither tool calls nor text")
                ready_payload = _ready_answer_payload(content)
                completion_content = ready_payload if ready_payload is not None else content
                candidate = _parse_candidate_answer(completion_content)
                predicted, model_citations, evidence_insufficient = _parse_final_response(
                    completion_content
                )
                if ready_payload is not None and not predicted:
                    # Qwen sometimes compresses an otherwise valid stabilized answer to
                    # ``answer["content"] = "B"``. Treat that explicit ready payload as
                    # the candidate, then require the normal controller-side grounding
                    # and working-memory checks below before it can terminate the run.
                    predicted = candidate
                citations, citations_synthesized = self._resolve_citations(
                    predicted, model_citations
                )
                if citations_synthesized:
                    completion_content = (
                        f"FINAL_ANSWER: {predicted}\n"
                        f"CITED_FACT_IDS: {', '.join(citations)}"
                    )
                if predicted:
                    self.trace["events"].append(
                        {
                            "event": "citations_grounded",
                            "rlm_completion": completion_index,
                            "native_turn": native_turn,
                            "predicted": predicted,
                            "model_cited_fact_ids": model_citations,
                            "grounded_fact_ids": citations,
                            "synthesized": citations_synthesized,
                        }
                    )
                candidate = predicted or candidate
                if candidate:
                    self.diagnostic_predicted = candidate
                completion_errors = self._completion_errors(predicted, citations)
                if (
                    ready_payload is not None
                    and self.termination_mode == "order_gap"
                    and not evidence_insufficient
                    and completion_errors
                ):
                    self.trace["events"].append(
                        {
                            "event": "invalid_ready_rejected",
                            "rlm_completion": completion_index,
                            "native_turn": native_turn,
                            "errors": completion_errors,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": self._completion_correction(completion_errors),
                        }
                    )
                    continue
                self._observe_epistemic(
                    {
                        "kind": "model",
                        "candidate": candidate,
                        "cited_fact_ids": citations,
                        "content": completion_content,
                    },
                    completion_index=completion_index,
                    native_turn=native_turn,
                    completion_boundary=True,
                )
                if self.termination_mode == "order_gap":
                    state_completion = self._state_based_completion(
                        predicted=predicted,
                        citations=citations,
                    )
                    if state_completion is not None:
                        return state_completion
                    if ready_payload is not None:
                        self.trace["events"].append(
                            {
                                "event": "model_ready_deferred",
                                "rlm_completion": completion_index,
                                "native_turn": native_turn,
                                "window_mean": self.state_tracker.window_mean,
                            }
                        )
                        return completion_content
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
                    elif name not in {SEARCH_TOOL_NAME, UPDATE_TOOL_NAME}:
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
                            if name == SEARCH_TOOL_NAME:
                                tool_response = _execute_with_timeout(
                                    self.execute_tool,
                                    arguments,
                                    self.graph_source,
                                    self.example_id,
                                    self.tool_timeout,
                                )
                            else:
                                pool = ThreadPoolExecutor(
                                    max_workers=1, thread_name_prefix="working-memory-tool"
                                )
                                future = pool.submit(
                                    self.execute_update_tool,
                                    arguments,
                                    self.graph_source,
                                    self.example_id,
                                    list(self.retrieved_facts.values()),
                                    validate=self.validate_memory_updates,
                                )
                                try:
                                    tool_response = _normalise_tool_response(
                                        future.result(timeout=self.tool_timeout)
                                    )
                                except FutureTimeoutError:
                                    future.cancel()
                                    tool_response = _tool_error(
                                        "backend_timeout",
                                        "Working-memory update timed out.",
                                        retryable=True,
                                        graph_source=self.graph_source,
                                        example_id=self.example_id,
                                        request=arguments,
                                        tool_name=UPDATE_TOOL_NAME,
                                    )
                                finally:
                                    pool.shutdown(wait=False, cancel_futures=True)
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
                            fact_id = str(result["fact_id"])
                            self.retrieved_fact_ids.add(fact_id)
                            self.retrieved_facts[fact_id] = dict(result)
                if name == UPDATE_TOOL_NAME and tool_response.get("status") == "ok":
                    artifact = tool_response.get("artifact")
                    if isinstance(artifact, Mapping) and artifact.get("artifact_id"):
                        artifact_id = str(artifact["artifact_id"])
                        self.working_memory_artifact_ids.append(artifact_id)
                        self.working_memory_fact_ids.update(
                            str(value)
                            for value in artifact.get("selected_fact_ids", [])
                            if value
                        )

                tool_content = json.dumps(
                    tool_response, ensure_ascii=False, separators=(",", ":")
                )
                self.tool_result_chars += len(tool_content)
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": tool_content}
                )
                if name in {SEARCH_TOOL_NAME, UPDATE_TOOL_NAME}:
                    self._observe_epistemic(
                        {
                            "kind": "search" if name == SEARCH_TOOL_NAME else "memory",
                            "response": tool_response,
                        },
                        completion_index=completion_index,
                        native_turn=native_turn,
                        completion_boundary=False,
                    )
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
                if (
                    name == SEARCH_TOOL_NAME
                    and tool_response.get("status") == "ok"
                    and not results
                ):
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
        retrieval_summary = self.graph_source.retrieval_summary()
        self.trace.update(
            {
                "status": status,
                "configured_retrieval_mode": retrieval_summary.get("configured_mode"),
                "effective_retrieval_mode": retrieval_summary.get("effective_mode"),
                "retrieval_degraded": retrieval_summary.get("degraded"),
                "dense_index_identity": retrieval_summary.get("dense_index_identity", []),
                "retrieval_branch_counts": retrieval_summary.get("branch_counts", {}),
                "retrieval_rrf_settings": retrieval_summary.get("rrf", {}),
                "retrieval_branch_latency_seconds": retrieval_summary.get(
                    "branch_latency_seconds", {}
                ),
                "termination_reason": termination_reason,
                "cited_fact_ids": list(cited_fact_ids),
                "retrieved_fact_ids": ordered_retrieved,
                "tool_call_count": self.tool_call_count,
                "retrieved_fact_count": len(ordered_retrieved),
                "working_memory_artifact_ids": list(self.working_memory_artifact_ids),
                "working_memory_fact_ids": sorted(self.working_memory_fact_ids),
                "tool_result_chars": self.tool_result_chars,
                "rlm_model_completion_count": self.model_completion_count,
                "diagnostic_predicted": self.diagnostic_predicted,
                "termination_mode": self.termination_mode,
                "order_gap_final": (
                    self.state_tracker.completion_gaps[-1]
                    if self.state_tracker.completion_gaps
                    else None
                ),
                "order_gap_window_mean": self.state_tracker.window_mean,
                "epistemic_state": self.state_tracker.to_dict(),
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
            working_memory_artifact_ids=list(self.working_memory_artifact_ids),
            diagnostic_predicted=self.diagnostic_predicted,
            termination_mode=self.termination_mode,
            order_gap_final=(
                self.state_tracker.completion_gaps[-1]
                if self.state_tracker.completion_gaps
                else None
            ),
            order_gap_window_mean=self.state_tracker.window_mean,
            rlm_completion_count=self.model_completion_count,
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
    model_timeout: float = 90.0,
    model_max_retries: int = 0,
) -> Any:
    """Create an RLM whose root Qwen calls the KG tool natively."""
    if backend != "openai":
        raise ValueError(
            "native Qwen/vLLM tool calls require --backend openai with the vLLM base URL"
        )
    if model_timeout <= 0:
        raise ValueError("model_timeout must be positive")
    if model_max_retries < 0:
        raise ValueError("model_max_retries must not be negative")

    from rlm.core.rlm import RLM
    from rlm.logger.rlm_logger import RLMLogger

    class NativeToolRLM(_NativeToolRLMMixin, RLM):
        pass

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    rlm = NativeToolRLM(
        backend=backend,
        backend_kwargs={
            "model_name": model,
            "base_url": base_url,
            "api_key": api_key,
            # A stalled guided-decoding request must not consume the OpenAI
            # client's 300s timeout plus two automatic retries per example.
            "timeout": model_timeout,
            "max_retries": model_max_retries,
        },
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
    model_timeout: float = 90.0,
    model_max_retries: int = 0,
    termination_mode: str = "order_gap",
    order_gap_epsilon: float = 0.025,
    order_gap_window: int = 2,
    order_gap_min_iterations: int = 2,
    tool_schema: Optional[Mapping[str, Any]] = None,
    execute_tool: SearchToolExecutor = execute_search_knowledge_graph,
    update_tool_schema: Mapping[str, Any] = UPDATE_WORKING_MEMORY_TOOL,
    execute_update_tool: Callable[..., Mapping[str, Any]] = execute_update_working_memory,
    validate_memory_updates: bool = True,
    require_memory_update: bool = False,
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
        update_tool_schema=update_tool_schema,
        execute_update_tool=execute_update_tool,
        validate_memory_updates=validate_memory_updates,
        require_memory_update=require_memory_update,
        question=str(example.get("question") or ""),
        choices={
            letter: str(example.get(f"choice_{letter}") or "")
            for letter in ("A", "B", "C", "D")
        },
        termination_mode=termination_mode,
        order_gap_epsilon=order_gap_epsilon,
        order_gap_window=order_gap_window,
        order_gap_min_iterations=order_gap_min_iterations,
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
        model_timeout=model_timeout,
        model_max_retries=model_max_retries,
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
            termination_reason=(
                session.controller_termination_reason or "evidence_insufficient"
            ),
            cited_fact_ids=[],
        )

    unknown_citations = sorted(set(citations) - session.retrieved_fact_ids)
    missing_from_memory = sorted(set(citations) - session.working_memory_fact_ids)
    if (
        predicted
        and citations
        and not unknown_citations
        and (not require_memory_update or not missing_from_memory)
    ):
        return session.finish(
            status="supported",
            predicted=predicted,
            raw_answer=raw_answer,
            error=None,
            termination_reason=(
                session.controller_termination_reason or "supported_final_answer"
            ),
            cited_fact_ids=citations,
        )

    if unknown_citations:
        error = f"unknown cited fact IDs: {', '.join(unknown_citations)}"
    elif require_memory_update and missing_from_memory:
        error = (
            "cited fact IDs were not committed to working memory: "
            + ", ".join(missing_from_memory)
        )
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
