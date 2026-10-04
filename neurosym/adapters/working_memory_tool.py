"""Native tool adapter for controlled working-memory updates."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from neurosym.application.working_memory import WorkingMemoryService


TOOL_NAME = "update_working_memory"
TOOL_VERSION = "update_working_memory.v1"

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
                        # Hermes/vLLM requires JSON Schema ``type`` to be a
                        # string. These fields are optional, so omission is the
                        # portable representation of an unset bound.
                        "valid_from": {"type": "string"},
                        "valid_to": {"type": "string"},
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
    return WorkingMemoryService().update(
        arguments,
        graph_source,
        example_id,
        returned_facts,
        validate=validate,
        prior_artifact=prior_artifact,
        tool_name=TOOL_NAME,
    )
