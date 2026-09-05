"""Derive query-blind, source-grounded preference capsules with Scallop."""

from __future__ import annotations

import importlib.metadata
import json
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RULE_VERSION = "preference_stream.v1"
INJECTION_KINDS = ("preference_change", "preference_incongruity")
FORBIDDEN_INJECTION_KEYS = {"answer", "gold", "prediction", "object", "text"}


def _preference_rows(
    events: Sequence[Mapping[str, Any]],
) -> tuple[
    list[tuple[str, str, str, str, str, str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """Return canonical preference, supersession, and identity relations."""
    preference_rows = []
    supersession_rows = []
    identity_rows = []
    for event in events:
        event_id = str(event.get("event_id", ""))
        fact = event.get("fact")
        if not event_id or not isinstance(fact, Mapping):
            raise ValueError("every preference-stream event needs an event_id and fact")
        if fact.get("predicate") == "SAME_ACCOUNT":
            identity_rows.append((event_id, str(fact.get("subject", ""))))
            continue
        if fact.get("predicate") != "PREFERS":
            continue
        qualifiers = fact.get("qualifiers")
        temporal = fact.get("temporal")
        if not isinstance(qualifiers, Mapping) or not isinstance(temporal, Mapping):
            raise ValueError(f"event {event_id} has malformed preference metadata")
        preference_rows.append(
            (
                event_id,
                str(fact.get("fact_id", "")),
                str(fact.get("subject", "")),
                str(qualifiers.get("scope", "default")),
                str(fact.get("object", "")),
                str(temporal.get("valid_from", "")),
                str(qualifiers.get("source_authority", "inferred")),
            )
        )
        supersedes = event.get("supersedes") or event.get("transitions_from")
        if supersedes is not None:
            supersession_rows.append((event_id, str(supersedes)))
    return sorted(preference_rows), sorted(supersession_rows), sorted(identity_rows)


def validate_preference_injection_result(
    result: Mapping[str, Any], *, causal_event_ids: set[str]
) -> dict[str, Any]:
    """Validate a source-ID-only Scallop response against one causal prefix."""
    if result.get("engine") != "scallopy" or result.get("rule_version") != RULE_VERSION:
        raise ValueError(f"invalid Scallop preference-stream identity: {dict(result)}")
    version = result.get("scallopy_version")
    injections = result.get("injections")
    if not isinstance(version, str) or not version or not isinstance(injections, list):
        raise ValueError("Scallop preference-stream response is incomplete")
    normalized = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for injection in injections:
        if not isinstance(injection, Mapping):
            raise ValueError("preference-stream injections must be objects")
        forbidden = sorted(FORBIDDEN_INJECTION_KEYS & set(injection))
        if forbidden:
            raise ValueError(f"forbidden response keys in preference injection: {forbidden}")
        if set(injection) != {"kind", "source_event_ids"}:
            raise ValueError(
                f"unexpected preference-injection keys: {sorted(set(injection))}"
            )
        kind = str(injection.get("kind", ""))
        source_ids = injection.get("source_event_ids")
        if kind not in INJECTION_KINDS:
            raise ValueError(f"unsupported preference-injection kind: {kind!r}")
        if (
            not isinstance(source_ids, list)
            or len(source_ids) != 3
            or not all(isinstance(identifier, str) and identifier for identifier in source_ids)
        ):
            raise ValueError(f"{kind} must cite two preference events and one alias event")
        missing = sorted(set(source_ids) - causal_event_ids)
        if missing:
            raise ValueError(
                f"injected source event is not in the causal prefix: {missing}"
            )
        key = (kind, tuple(source_ids))
        if key in seen:
            raise ValueError(f"duplicate preference injection: {key}")
        seen.add(key)
        normalized.append({"kind": kind, "source_event_ids": list(source_ids)})
    normalized.sort(key=lambda item: INJECTION_KINDS.index(item["kind"]))
    return {
        "engine": "scallopy",
        "scallopy_version": version,
        "rule_version": RULE_VERSION,
        "injections": normalized,
    }


def derive_preference_injections_with_scallop(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Use actual Scallop rules to derive changed and incongruous source pairs."""
    try:
        import scallopy
    except ImportError as error:
        raise RuntimeError("scallopy is required for preference-stream injection") from error
    preference_rows, supersession_rows, identity_rows = _preference_rows(events)
    context = scallopy.ScallopContext(provenance="unit")
    context.add_relation("preference_event", (str, str, str, str, str, str, str))
    context.add_relation("supersedes_event", (str, str))
    context.add_relation("identity_event", (str, str))
    context.add_facts("preference_event", preference_rows)
    context.add_facts("supersedes_event", supersession_rows)
    context.add_facts("identity_event", identity_rows)
    context.add_rule(
        "preference_change(old_event, new_event, alias_event) = "
        "preference_event(old_event, old_fact, subject, scope, old_value, old_from, _) "
        "and preference_event(new_event, _, subject, scope, new_value, new_from, _) "
        "and identity_event(alias_event, subject) "
        "and supersedes_event(new_event, old_fact) "
        "and old_value != new_value and old_from != new_from"
    )
    context.add_rule(
        "preference_incongruity(old_event, new_event, alias_event) = "
        "preference_event(old_event, old_fact, subject, scope, old_value, valid_from, old_authority) "
        "and preference_event(new_event, _, subject, scope, new_value, valid_from, new_authority) "
        "and identity_event(alias_event, subject) "
        "and supersedes_event(new_event, old_fact) "
        "and old_value != new_value and old_authority != new_authority"
    )
    context.run()
    result = {
        "engine": "scallopy",
        "scallopy_version": importlib.metadata.version("scallopy"),
        "rule_version": RULE_VERSION,
        "injections": [
            {
                "kind": kind,
                "source_event_ids": source_event_ids,
            }
            for kind in INJECTION_KINDS
            for old_event, new_event, alias_event in sorted(context.relation(kind))
            for source_event_ids in [[str(old_event), str(new_event), str(alias_event)]]
        ],
    }
    return validate_preference_injection_result(
        result,
        causal_event_ids={str(event.get("event_id")) for event in events},
    )


class PreferenceStreamInjectionClient:
    """Hard-timeout client for query-blind Scallop preference derivation."""

    def __init__(self, endpoint: str, timeout_seconds: float) -> None:
        endpoint = endpoint.rstrip("/")
        if not endpoint:
            raise ValueError("Scallop preference-stream endpoint must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("Scallop preference-stream timeout must be positive")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def derive(self, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Derive and validate source-only injections from a causal event prefix."""
        compact_events = []
        for event in events:
            fact = event.get("fact")
            if not isinstance(fact, Mapping):
                raise ValueError("preference-stream event fact must be an object")
            compact_events.append(
                {
                    "event_id": event.get("event_id"),
                    "supersedes": event.get("supersedes"),
                    "transitions_from": event.get("transitions_from"),
                    "fact": {
                        "fact_id": fact.get("fact_id"),
                        "subject": fact.get("subject"),
                        "predicate": fact.get("predicate"),
                        "object": fact.get("object"),
                        "temporal": fact.get("temporal"),
                        "qualifiers": fact.get("qualifiers"),
                    },
                }
            )
        payload = json.dumps({"events": compact_events}).encode("utf-8")
        request = Request(
            f"{self.endpoint}/derive_preference_injections",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Scallop preference-stream request failed with HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Scallop preference-stream request failed: {error}") from error
        if not isinstance(result, Mapping):
            raise RuntimeError("Scallop preference-stream response must be an object")
        return validate_preference_injection_result(
            result,
            causal_event_ids={str(event.get("event_id")) for event in events},
        )
