"""Explicit validator backends for fail-closed Scallop experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

from neurosym.adapters import scallop as scallop_validator
from neurosym.adapters.scallop import (
    RejectionLabel,
    ValidationDecision,
    validate_update_detailed,
)
from neurosym.domain.validation_rules import DEFAULT_RULE_PARAMETERS, RuleParameters


@dataclass(frozen=True)
class ValidatorInfo:
    name: str
    engine: str
    rule_version: str
    scallop_available: bool
    endpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "engine": self.engine,
            "rule_version": self.rule_version,
            "scallop_available": self.scallop_available,
            "endpoint": self.endpoint,
        }


class LocalValidatorBackend:
    def __init__(self, *, require_scallop: bool = False) -> None:
        available = scallop_validator.scallopy is not None
        if require_scallop and not available:
            raise RuntimeError(
                "actual Scallop is required but scallopy is unavailable; "
                "configure --scallop-validator-url or use the Python 3.10 Scallop environment"
            )
        self.info = ValidatorInfo(
            name="scallop" if available else "python_symbolic_fallback",
            engine="scallopy" if available else "python",
            rule_version=DEFAULT_RULE_PARAMETERS.version,
            scallop_available=available,
        )

    def validate(
        self,
        existing_facts,
        new_fact,
        rule_params: Optional[RuleParameters] = None,
    ) -> ValidationDecision:
        return validate_update_detailed(existing_facts, new_fact, rule_params=rule_params)


class HttpScallopValidatorBackend:
    def __init__(self, endpoint: str, *, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        health = self._request("GET", "/health", None)
        if not health.get("scallop_available") or health.get("engine") != "scallopy":
            raise RuntimeError("validator service is not backed by actual scallopy")
        self.info = ValidatorInfo(
            name="scallop",
            engine="scallopy-http",
            rule_version=str(health.get("rule_version") or DEFAULT_RULE_PARAMETERS.version),
            scallop_available=True,
            endpoint=self.endpoint,
        )

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError("validator service returned a non-object response")
        return result

    def validate(
        self,
        existing_facts,
        new_fact,
        rule_params: Optional[RuleParameters] = None,
    ) -> ValidationDecision:
        params = rule_params or DEFAULT_RULE_PARAMETERS
        response = self._request(
            "POST",
            "/validate",
            {
                "existing_facts": existing_facts,
                "new_fact": new_fact,
                "rule_parameters": params.to_dict(),
            },
        )
        label = response.get("rejection_label")
        rejection_label = RejectionLabel(**label) if isinstance(label, dict) else None
        return ValidationDecision(
            decision=str(response["decision"]),
            reason=str(response["reason"]),
            replace_fact_id=response.get("replace_fact_id"),
            rejection_label=rejection_label,
            rule_params_version=str(response.get("rule_params_version") or params.version),
        )


def make_validator_backend(
    *,
    endpoint: Optional[str] = None,
    require_scallop: bool = False,
    timeout: float = 30.0,
):
    if endpoint:
        return HttpScallopValidatorBackend(endpoint, timeout=timeout)
    return LocalValidatorBackend(require_scallop=require_scallop)
