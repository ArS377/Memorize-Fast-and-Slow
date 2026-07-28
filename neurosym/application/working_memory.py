from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from neurosym.domain.compiled_memory import stable_id
from neurosym.domain.memory_artifacts import compile_working_memory, transition_for
from neurosym.domain.validation_rules import DEFAULT_RULE_PARAMETERS


def _error(
    graph_source: Any,
    example_id: str,
    code: str,
    message: str,
    *,
    tool_name: str,
) -> Dict[str, Any]:
    return {
        "status": "error",
        "tool": tool_name,
        "scope": graph_source.trusted_scope(example_id).to_dict(),
        "artifact": None,
        "transition": None,
        "derived_update_result": None,
        "persisted": False,
        "error": {"code": code, "message": message, "retryable": True},
    }


class WorkingMemoryService:
    def update(
        self,
        arguments: Mapping[str, Any],
        graph_source: Any,
        example_id: str,
        returned_facts: List[Mapping[str, Any]],
        *,
        validate: bool,
        prior_artifact: Optional[Any] = None,
        tool_name: str = "update_working_memory",
    ) -> Dict[str, Any]:
        try:
            artifact = compile_working_memory(
                arguments,
                returned_facts,
                graph_source.trusted_scope(example_id),
                prior_artifact,
            )
        except (TypeError, ValueError) as exc:
            return _error(
                graph_source,
                example_id,
                "invalid_memory_update",
                str(exc),
                tool_name=tool_name,
            )

        graph = getattr(graph_source, "graph", None)
        if validate and graph is None:
            return _error(
                graph_source,
                example_id,
                "validator_unavailable",
                "validated working-memory updates require Neo4j and an actual Scallop backend",
                tool_name=tool_name,
            )
        derived_result: Optional[Dict[str, Any]] = None
        decision = "accept"
        reason = "Working-memory references and provenance are valid."
        backend = getattr(graph, "validator_backend", None) if graph is not None else None
        validator = backend.info.name if validate and backend is not None else "none"
        rule_version = DEFAULT_RULE_PARAMETERS.version if validate else "unconstrained.v1"
        if validate:
            if backend is None or not backend.info.scallop_available:
                return _error(
                    graph_source,
                    example_id,
                    "validator_unavailable",
                    "Cell 6 refuses a working-memory commit without actual scallopy",
                    tool_name=tool_name,
                )
            selected_ids = set(artifact.selected_fact_ids)
            selected = [
                dict(fact)
                for fact in returned_facts
                if str(fact.get("fact_id", "")) in selected_ids
            ]
            accepted_selected: List[Dict[str, Any]] = []
            for fact in selected:
                validation = backend.validate(
                    accepted_selected,
                    fact,
                    rule_params=DEFAULT_RULE_PARAMETERS,
                )
                if validation.decision == "reject":
                    decision = "reject"
                    reason = f"Selected fact failed Scallop validation: {validation.reason}"
                    break
                accepted_selected.append(fact)
        derived = list(artifact.derived_facts)
        if decision != "reject" and derived and graph is not None:
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
            "tool": tool_name,
            "scope": artifact.scope.to_dict(),
            "artifact": artifact.to_dict(),
            "transition": transition.to_dict(),
            "derived_update_result": derived_result,
            "persisted": persisted,
            "error": None,
        }
        json.dumps(response, ensure_ascii=False, allow_nan=False)
        return response
