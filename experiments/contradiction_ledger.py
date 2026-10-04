"""Append-only contradiction tracking for synthetic preference histories."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any, Mapping, Sequence


AUTHORITY_RANK = {"inferred": 0, "direct_user": 1}


def _pair_id(left: str, right: str) -> str:
    """Return a stable contradiction pair identifier."""
    material = ":".join(sorted((left, right)))
    return f"contradiction-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:20]}"


def _claim_key(event: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """Return the semantic key whose overlapping values may contradict."""
    fact = event["fact"]
    qualifiers = fact["qualifiers"]
    return (
        str(event["history_id"]),
        str(fact["subject"]),
        str(fact["predicate"]),
        str(qualifiers.get("domain", "")),
        str(qualifiers.get("scope", "default")),
    )


def _overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two closed validity intervals overlap."""
    left_start = _date_rank(left["valid_from"])
    right_start = _date_rank(right["valid_from"])
    left_end = _date_rank(left.get("valid_to"), open_end=True)
    right_end = _date_rank(right.get("valid_to"), open_end=True)
    return left_start <= right_end and right_start <= left_end


def _date_rank(value: str | None, *, open_end: bool = False) -> int:
    """Encode an ISO calendar date for Scallop's integer comparisons."""
    if value is None and open_end:
        return 99991231
    text = str(value or "")
    digits = text.replace("-", "")
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"validity date must use YYYY-MM-DD: {value!r}")
    return int(digits)


def derive_contradiction_ledger(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replay assertions and explicit resolution controls into a pair ledger."""
    explicit_lifecycle = any(
        str(event.get("event_family", "")).startswith("contradiction_")
        for event in events
    )
    ledger_events = [
        event
        for event in events
        if not explicit_lifecycle
        or str(event.get("event_family", ""))
        in {"contradiction_opening", "contradiction_rectification"}
    ]
    assertions: list[tuple[int, Mapping[str, Any]]] = []
    pairs: dict[str, dict[str, Any]] = {}
    scope_separated = 0
    for event_index, event in enumerate(ledger_events):
        fact = event.get("fact")
        if not isinstance(fact, Mapping) or fact.get("predicate") != "PREFERS":
            continue
        authority = str(fact["qualifiers"].get("source_authority", "inferred"))
        if authority not in AUTHORITY_RANK:
            raise ValueError(f"unsupported source authority: {authority!r}")
        resolves = event.get("resolves") or []
        if event.get("supersedes") is not None:
            resolves = [*resolves, str(event["supersedes"])]
        if not isinstance(resolves, Sequence) or isinstance(resolves, (str, bytes)):
            raise ValueError(f"event {event.get('event_id')} resolves must be a sequence")
        resolved_ids = {str(identifier) for identifier in resolves}
        opens_pair = len(resolved_ids) < 2
        for _, prior_event in assertions:
            prior_fact = prior_event["fact"]
            same_base = _claim_key(prior_event)[:4] == _claim_key(event)[:4]
            if (
                same_base
                and _claim_key(prior_event)[4] != _claim_key(event)[4]
                and prior_fact["object"] != fact["object"]
                and _overlaps(prior_fact["temporal"], fact["temporal"])
            ):
                scope_separated += 1
                continue
            if opens_pair and (
                _claim_key(prior_event) != _claim_key(event)
                or prior_fact["object"] == fact["object"]
                or not _overlaps(prior_fact["temporal"], fact["temporal"])
            ):
                continue
            if not opens_pair:
                continue
            left_id = str(prior_fact["fact_id"])
            right_id = str(fact["fact_id"])
            pair_id = _pair_id(left_id, right_id)
            pairs[pair_id] = {
                "pair_id": pair_id,
                "fact_ids": sorted((left_id, right_id)),
                "status": "unresolved",
                "winner_fact_id": None,
                "opened_event_index": event_index,
                "resolved_event_index": None,
            }
        if resolved_ids:
            for pair in pairs.values():
                if pair["status"] != "unresolved":
                    continue
                if set(pair["fact_ids"]).issubset(resolved_ids):
                    winner_authority = AUTHORITY_RANK[authority]
                    loser_authorities = [
                        AUTHORITY_RANK[
                            str(prior["fact"]["qualifiers"].get("source_authority", "inferred"))
                        ]
                        for _, prior in assertions
                        if str(prior["fact"]["fact_id"]) in pair["fact_ids"]
                    ]
                    if winner_authority < max(loser_authorities, default=0):
                        raise ValueError("lower-authority assertion cannot resolve contradiction")
                    pair["status"] = (
                        "resolved_authority"
                        if winner_authority > max(loser_authorities, default=0)
                        else "resolved_supersession"
                    )
                    pair["winner_fact_id"] = str(fact["fact_id"])
                    pair["resolved_event_index"] = event_index
                elif str(fact["fact_id"]) in pair["fact_ids"]:
                    other_ids = set(pair["fact_ids"]) - {str(fact["fact_id"])}
                    if other_ids & resolved_ids:
                        loser_authority = max(
                            AUTHORITY_RANK[
                                str(prior["fact"]["qualifiers"].get("source_authority", "inferred"))
                            ]
                            for _, prior in assertions
                            if str(prior["fact"]["fact_id"]) in other_ids
                        )
                        winner_authority = AUTHORITY_RANK[authority]
                        if winner_authority < loser_authority:
                            raise ValueError(
                                "lower-authority assertion cannot resolve contradiction"
                            )
                        pair["status"] = (
                            "resolved_authority"
                            if winner_authority > loser_authority
                            else "resolved_supersession"
                        )
                        pair["winner_fact_id"] = str(fact["fact_id"])
                        pair["resolved_event_index"] = event_index
        assertions.append((event_index, event))
    ordered_pairs = sorted(pairs.values(), key=lambda pair: pair["pair_id"])
    unresolved = [pair["pair_id"] for pair in ordered_pairs if pair["status"] == "unresolved"]
    return {
        "ledger_version": "contradiction_ledger.v1",
        "pairs": ordered_pairs,
        "unresolved_pair_ids": unresolved,
        "scope_separated_count": scope_separated,
        "unexpected_unresolved_pair_ids": unresolved,
    }


def derive_contradiction_ledger_with_scallop(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive explicit contradiction lifecycles with actual Scallop rules."""
    try:
        import scallopy
    except ImportError as error:
        raise RuntimeError("scallopy is required for contradiction-ledger derivation") from error
    lifecycle = [
        event
        for event in events
        if str(event.get("event_family", ""))
        in {"contradiction_opening", "contradiction_rectification"}
        or not any(
            str(candidate.get("event_family", "")).startswith("contradiction_")
            for candidate in events
        )
    ]
    claims = []
    resolutions = []
    event_index_by_fact = {}
    event_id_by_fact = {}
    authority_by_fact = {}
    for event_index, event in enumerate(lifecycle):
        fact = event.get("fact")
        if not isinstance(fact, Mapping) or fact.get("predicate") != "PREFERS":
            continue
        qualifiers = fact["qualifiers"]
        temporal = fact["temporal"]
        authority = str(qualifiers.get("source_authority", "inferred"))
        if authority not in AUTHORITY_RANK:
            raise ValueError(f"unsupported source authority: {authority!r}")
        fact_id = str(fact["fact_id"])
        claims.append(
            (
                fact_id,
                str(event["history_id"]),
                str(fact["subject"]),
                str(fact["predicate"]),
                str(qualifiers.get("domain", "")),
                str(qualifiers.get("scope", "default")),
                str(fact["object"]),
                _date_rank(temporal["valid_from"]),
                _date_rank(temporal.get("valid_to"), open_end=True),
                event_index,
                AUTHORITY_RANK[authority],
            )
        )
        event_index_by_fact[fact_id] = event_index
        event_id_by_fact[fact_id] = str(event["event_id"])
        authority_by_fact[fact_id] = AUTHORITY_RANK[authority]
        resolves = event.get("resolves") or []
        if event.get("supersedes") is not None:
            resolves = [*resolves, str(event["supersedes"])]
        if not isinstance(resolves, Sequence) or isinstance(resolves, (str, bytes)):
            raise ValueError(f"event {event.get('event_id')} resolves must be a sequence")
        for target in resolves:
            target_id = str(target)
            target_authority = authority_by_fact.get(target_id)
            if target_authority is None:
                raise ValueError(
                    f"event {event.get('event_id')} resolves unknown fact {target_id!r}"
                )
            if AUTHORITY_RANK[authority] < target_authority:
                raise ValueError("lower-authority assertion cannot resolve contradiction")
        for target in resolves:
            resolutions.append((fact_id, str(target)))
        claims[-1] = (*claims[-1], int(len(resolves) < 2))
    context = scallopy.ScallopContext(provenance="unit")
    context.add_relation(
        "claim", (str, str, str, str, str, str, str, int, int, int, int, int)
    )
    context.add_relation("resolves", (str, str))
    context.add_facts("claim", sorted(claims))
    context.add_facts("resolves", sorted(resolutions))
    context.add_rule(
        "potential(a, b) = claim(a, h, s, p, d, scope, va, afrom, ato, ai, _, 1) "
        "and claim(b, h, s, p, d, scope, vb, bfrom, bto, bi, _, 1) "
        "and va != vb and ai < bi and afrom <= bto and bfrom <= ato"
    )
    context.add_rule(
        "defeats(winner, loser) = resolves(winner, loser)"
    )
    context.add_rule(
        "resolved(a, b, winner) = potential(a, b) and defeats(winner, a) "
        "and defeats(winner, b)"
    )
    context.add_rule(
        "resolved(a, b, a) = potential(a, b) and defeats(a, b)"
    )
    context.add_rule(
        "resolved(a, b, b) = potential(a, b) and defeats(b, a)"
    )
    context.add_rule(
        "resolved_authority(a, b, winner) = resolved(a, b, winner) "
        "and claim(a, _, _, _, _, _, _, _, _, _, aa, _) "
        "and claim(b, _, _, _, _, _, _, _, _, _, ba, _) "
        "and claim(winner, _, _, _, _, _, _, _, _, _, wa, _) and wa > aa and wa > ba"
    )
    context.add_rule(
        "unresolved(a, b) = potential(a, b) and not resolved(a, b, _)"
    )
    context.run()
    resolved_authority = {
        (str(left), str(right)): str(winner)
        for left, right, winner in context.relation("resolved_authority")
    }
    resolved = {
        (str(left), str(right)): str(winner)
        for left, right, winner in context.relation("resolved")
    }
    unresolved = {
        (str(left), str(right)) for left, right in context.relation("unresolved")
    }
    pairs = []
    for left, right in sorted(
        (str(left), str(right)) for left, right in context.relation("potential")
    ):
        winner = resolved.get((left, right))
        if winner is None:
            status = "unresolved"
        else:
            loser_authorities = [
                authority_by_fact[fact_id]
                for fact_id in (left, right)
                if fact_id != winner
            ]
            status = (
                "resolved_authority"
                if authority_by_fact[winner] > max(loser_authorities, default=-1)
                else "resolved_supersession"
            )
            if (left, right) in resolved_authority:
                status = "resolved_authority"
        pairs.append(
            {
                "pair_id": _pair_id(left, right),
                "fact_ids": sorted((left, right)),
                "status": status,
                "winner_fact_id": winner,
                "opened_event_index": max(
                    event_index_by_fact[left], event_index_by_fact[right]
                ),
                "resolved_event_index": (
                    event_index_by_fact[winner] if winner is not None else None
                ),
                "source_event_ids": [event_id_by_fact[left], event_id_by_fact[right]],
            }
        )
    unresolved_ids = [
        _pair_id(left, right) for left, right in sorted(unresolved)
    ]
    return {
        "ledger_version": "contradiction_ledger.v1",
        "engine": "scallopy",
        "scallopy_version": importlib.metadata.version("scallopy"),
        "pairs": pairs,
        "unresolved_pair_ids": unresolved_ids,
        "scope_separated_count": 0,
        "unexpected_unresolved_pair_ids": unresolved_ids,
    }


class ContradictionLedgerClient:
    """Hard-timeout client for the actual Scallop ledger service."""

    def __init__(self, endpoint: str, timeout_seconds: float = 10.0) -> None:
        endpoint = endpoint.rstrip("/")
        if not endpoint or timeout_seconds <= 0:
            raise ValueError("Scallop ledger endpoint and timeout must be valid")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def derive(self, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Derive and validate one account-local contradiction ledger."""
        request = Request(
            f"{self.endpoint}/derive_contradiction_ledger",
            data=json.dumps({"events": list(events)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Scallop contradiction-ledger request failed with HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Scallop contradiction-ledger request failed: {error}") from error
        if (
            not isinstance(result, dict)
            or result.get("engine") != "scallopy"
            or result.get("ledger_version") != "contradiction_ledger.v1"
            or not isinstance(result.get("pairs"), list)
            or not isinstance(result.get("unexpected_unresolved_pair_ids"), list)
        ):
            raise RuntimeError(f"invalid Scallop contradiction-ledger response: {result}")
        return result
