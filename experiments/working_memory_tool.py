"""Native tool adapter for controlled working-memory updates."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from compiled_memory import stable_id
from memory_artifacts import compile_working_memory, transition_for
from scallop_validator import DEFAULT_RULE_PARAMETERS


TOOL_NAME = "update_working_memory"

UPDATE_WORKING_MEMORY_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Compile selected returned facts into a provenance-aware working-memory "
            "artifact. Call this after knowledge-graph search and before the final answer."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "entity": {"type": "string", "minLength": 1, "maxLength": 256},
                "selected_fact_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": 50,
                },
                "excluded_fact_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 50,
                    "default": [],
                },
                "temporal_scope": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "valid_from": {"type": ["string", "null"]},
                        "valid_to": {"type": ["string", "null"]},
                    },
                    "default": {},
                },
                "confidence": {
                    "type": "string",
                    "enum": ["supported", "uncertain"],
                    "default": "uncertain",
                },
                "derived_facts": {
                    "type": "array",
                    "maxItems": 10,
                    "default": [],
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "subject": {"type": "string", "minLength": 1},
                            "predicate": {"type": "string", "minLength": 1},
                            "object": {"type": "string", "minLength": 1},
                            "support_fact_ids": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                                "minItems": 1,
                            },
                            "support_text": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["supported", "uncertain"],
                            },
                            "temporal": {"type": "object"},
                        },
                        "required": ["subject", "predicate", "object", "support_fact_ids"],
                    },
                },
            },
            "required": ["entity", "selected_fact_ids"],
        },
    },
}


def _error(graph_source, example_id: str, code: str, message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "tool": TOOL_NAME,
        "scope": graph_source.trusted_scope(example_id).to_dict(),
        "artifact": None,
        "transition": None,
        "derived_update_result": None,
        "persisted": False,
        "error": {"code": code, "message": message, "retryable": True},
    }


def execute_update_working_memory(
    arguments: Mapping[str, Any],
    graph_source,
    example_id: str,
    returned_facts: List[Mapping[str, Any]],
    *,
    validate: bool,
    prior_artifact=None,
) -> Dict[str, Any]:
    """Compile, validate, and persist a model-proposed working-memory update."""
    try:
        artifact = compile_working_memory(
            arguments,
            returned_facts,
            graph_source.trusted_scope(example_id),
            prior_artifact,
        )
    except (TypeError, ValueError) as exc:
        return _error(graph_source, example_id, "invalid_memory_update", str(exc))

    graph = getattr(graph_source, "graph", None)
    derived_result: Optional[Dict[str, Any]] = None
    decision = "accept"
    reason = "Working-memory references and provenance are valid."
    validator = "scallop" if validate else "none"
    rule_version = DEFAULT_RULE_PARAMETERS.version if validate else "unconstrained.v1"
    derived = list(artifact.derived_facts)
    if derived and graph is not None:
        for fact in derived:
            fact["example_id"] = example_id
            fact["fact_id"] = fact.get("fact_id") or stable_id(
                "derived_fact",
                [
                    artifact.artifact_id,
                    fact["subject"],
                    fact["predicate"],
                    fact["object"],
                    fact["support_fact_ids"],
                ],
                length=24,
            )
            fact["question"] = "RLM-derived working-memory update"
        derived_result = graph.insert_facts(
            derived,
            session_id=graph_source.session_id,
            validate=validate,
        )
        if derived_result.get("rejected"):
            decision = "reject"
            reason = "One or more derived facts failed transition validation."
        elif derived_result.get("replaced"):
            decision = "replace"
            reason = "A stronger derived fact replaced prior state."

    transition = transition_for(
        artifact,
        decision=decision,
        reason=reason,
        validator=validator,
        rule_version=rule_version,
    )
    artifact.status = decision
    persisted = False
    if graph is not None:
        graph.persist_working_memory(artifact, transition)
        persisted = True
    response = {
        "status": "ok" if decision in {"accept", "replace"} else "rejected",
        "tool": TOOL_NAME,
        "scope": artifact.scope.to_dict(),
        "artifact": artifact.to_dict(),
        "transition": transition.to_dict(),
        "derived_update_result": derived_result,
        "persisted": persisted,
        "error": None,
    }
    json.dumps(response, ensure_ascii=False, allow_nan=False)
    return response
