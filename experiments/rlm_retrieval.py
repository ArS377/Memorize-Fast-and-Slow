"""RLM-controlled retrieval for KG-backed RLM cells.

The existing KG cells perform one fixed retrieval before answering. This module
adds an optional planner loop where an RLM chooses retrieval seeds/predicates,
inspects returned facts, and may ask for follow-up retrieval before the final
answering RLM sees the selected context.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from experiments.common import format_question
from experiments.graph_context import (
    GraphSource,
    _row_key,
    extract_seed_entities,
)


ActionCompleter = Callable[[str, str], str]


def parse_retrieval_action(text: str) -> Dict[str, Any]:
    """Parse a planner JSON action from an RLM response.

    Accepts raw JSON or JSON embedded in surrounding text/code fences. Invalid
    output returns an empty dict so callers can fall back deterministically.
    """
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _as_str_list(value: Any, *, limit: int) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
        if len(out) >= limit:
            break
    return out


def _planner_prompt(
    *,
    example: Dict[str, Any],
    selected_rows: List[Dict[str, Any]],
    attempted: List[Dict[str, Any]],
    initial_seeds: List[str],
    step: int,
) -> Tuple[str, str]:
    question = format_question(example)
    known = [
        {
            "subject": row.get("subject"),
            "predicate": row.get("predicate"),
            "object": row.get("object"),
            "fact_id": row.get("fact_id"),
        }
        for row in selected_rows[-20:]
    ]
    state = {
        "question": question,
        "initial_seed_entities": initial_seeds,
        "known_facts": known,
        "attempted_retrievals": attempted[-10:],
        "step": step,
    }
    root = """
You are controlling knowledge-graph retrieval for a multiple-choice question.
Return only one JSON object, with no markdown.

Choose one action:
1. Retrieve more facts:
{
  "action": "retrieve",
  "seed_entities": ["entity name"],
  "predicates": ["OPTIONAL_PREDICATE"],
  "reason": "short reason"
}

2. Stop retrieval when known_facts are sufficient:
{
  "action": "stop",
  "reason": "short reason"
}

Prefer precise entity names and UPPER_SNAKE_CASE predicates. Follow
intermediate entities discovered in known_facts when the question needs
multi-hop reasoning.
""".strip()
    return json.dumps(state, ensure_ascii=False, indent=2), root


def _fallback_action(initial_seeds: List[str]) -> Dict[str, Any]:
    return {
        "action": "retrieve",
        "seed_entities": initial_seeds,
        "predicates": [],
        "reason": "fallback to initial question entities",
    }


def rlm_guided_context(
    *,
    graph_source: GraphSource,
    example: Dict[str, Any],
    complete_action: ActionCompleter,
    hops: int = 2,
    limit_triples: int = 50,
    max_chars: int = 4000,
    max_steps: int = 3,
    max_seed_entities: int = 6,
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """Run an RLM retrieval loop and return ``(context, n_rows, trace)``.

    ``complete_action`` is called as ``complete_action(prompt, root_prompt)`` and
    should return the planner's raw text. Production passes an RLM completion;
    tests pass a deterministic stub.
    """
    example_id = str(example.get("_id", ""))
    initial_seeds = extract_seed_entities(example)[:max_seed_entities]
    selected: List[Dict[str, Any]] = []
    seen_rows = set()
    trace: List[Dict[str, Any]] = []
    attempted: List[Dict[str, Any]] = []

    if not initial_seeds:
        return "", 0, trace

    for step in range(1, max(1, max_steps) + 1):
        prompt, root = _planner_prompt(
            example=example,
            selected_rows=selected,
            attempted=attempted,
            initial_seeds=initial_seeds,
            step=step,
        )
        raw = complete_action(prompt, root)
        action = parse_retrieval_action(raw)
        if not action and step == 1:
            action = _fallback_action(initial_seeds)

        action_name = str(action.get("action", "")).strip().lower()
        if action_name in {"stop", "answer", "final"}:
            trace.append({"step": step, "action": "stop", "reason": action.get("reason", "")})
            break
        if action_name != "retrieve":
            trace.append({"step": step, "action": "invalid", "raw": raw})
            break

        seeds = _as_str_list(action.get("seed_entities"), limit=max_seed_entities)
        if not seeds and step == 1:
            seeds = initial_seeds
        predicates = _as_str_list(action.get("predicates"), limit=8)
        attempted_row = {
            "seed_entities": seeds,
            "predicates": predicates,
            "reason": str(action.get("reason", "")),
        }
        attempted.append(attempted_row)

        rows = graph_source.rows_for(
            seed_entities=seeds,
            example_id=example_id,
            hops=hops,
            limit_triples=limit_triples,
            predicates=predicates,
        )
        new_rows = []
        for row in rows:
            key = _row_key(row)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            selected.append(row)
            new_rows.append(row)

        trace.append({
            "step": step,
            "action": "retrieve",
            "seed_entities": seeds,
            "predicates": predicates,
            "n_rows": len(rows),
            "n_new_rows": len(new_rows),
            "reason": action.get("reason", ""),
        })

        if not new_rows and selected:
            break

    context = graph_source.format_rows(selected, max_chars=max_chars)
    return context, len(selected), trace


def make_rlm_action_completer(rlm) -> ActionCompleter:
    """Wrap an RLM instance as an action-completion callable."""
    def complete(prompt: str, root_prompt: str) -> str:
        result = rlm.completion(prompt=prompt, root_prompt=root_prompt)
        if result is None:
            return ""
        return getattr(result, "response", None) or str(result)

    return complete
