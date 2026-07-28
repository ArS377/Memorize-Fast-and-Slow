from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import scallopy
except ImportError:  # Keep non-Scallop tests/imports usable.
    scallopy = None

from neurosym.domain.compiled_memory import (
    evidence_strength_score,
    facts_temporally_overlap,
    sanitize_predicate,
)
from neurosym.domain.validation_rules import DEFAULT_RULE_PARAMETERS, RuleParameters


@dataclass(frozen=True)
class RejectionLabel:
    code: str
    category: str
    rule_id: str
    severity: str = "reject"

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "category": self.category,
            "rule_id": self.rule_id,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ValidationDecision:
    decision: str
    reason: str
    replace_fact_id: Optional[str] = None
    rejection_label: Optional[RejectionLabel] = None
    rule_params_version: str = "rules.v1"

    def as_legacy_tuple(self):
        return (self.decision, self.reason, self.replace_fact_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "replace_fact_id": self.replace_fact_id,
            "rejection_label": (
                self.rejection_label.to_dict() if self.rejection_label else None
            ),
            "rule_params_version": self.rule_params_version,
        }


# Backward-compatible aliases for code/tests that import the old constants.
FUNCTIONAL_PREDICATES = set(DEFAULT_RULE_PARAMETERS.functional_predicates)
GENERIC_OBJECTS = set(DEFAULT_RULE_PARAMETERS.generic_objects)


def _reject(
    reason: str,
    code: str,
    category: str,
    rule_id: str,
    rule_params: RuleParameters,
) -> ValidationDecision:
    return ValidationDecision(
        decision="reject",
        reason=reason,
        rejection_label=RejectionLabel(
            code=code,
            category=category,
            rule_id=rule_id,
        ),
        rule_params_version=rule_params.version,
    )


def _accept(rule_params: RuleParameters) -> ValidationDecision:
    return ValidationDecision(
        decision="accept",
        reason="Valid",
        rule_params_version=rule_params.version,
    )


def _replace(
    reason: str,
    replace_fact_id: Optional[str],
    rule_params: RuleParameters,
) -> ValidationDecision:
    return ValidationDecision(
        decision="replace",
        reason=reason,
        replace_fact_id=replace_fact_id,
        rule_params_version=rule_params.version,
    )


def confidence_score(fact, rule_params: Optional[RuleParameters] = None):
    """
    Numeric score for a fact dict. Higher = more trustworthy.
    Uses weighted evidence strength rather than only the coarse confidence
    label. Returned on a 0..100 scale for readable validator messages.
    """
    return round(evidence_strength_score(fact) * 100, 2)


def _to_triple(fact):
    """Extract (subject, predicate, object) tuple from a fact dict."""
    return (
        str(fact["subject"]),
        sanitize_predicate(str(fact["predicate"])),
        str(fact["object"]),
    )


def _python_symbolic_checks(
    existing_facts,
    new_fact,
    existing_triples,
    new_triple,
    rule_params: RuleParameters,
):
    """Fallback implementation of the current Scallop rules.

    This keeps the pipeline importable in environments that do not have the
    Scallop Python wheel installed. It intentionally mirrors only the rules
    below, not arbitrary Scallop programs.
    """
    all_triples = existing_triples + [new_triple]
    subj, pred, obj = new_triple

    if pred in set(rule_params.functional_predicates):
        for s, p, o in all_triples:
            if s == subj and p == pred and o != obj:
                conflicting = next(
                    (
                        e for e in existing_facts
                        if str(e["subject"]) == subj
                        and sanitize_predicate(str(e["predicate"])) == pred
                        and str(e["object"]) != obj
                        and facts_temporally_overlap(e, new_fact)
                    ),
                    None,
                )
                if conflicting is not None:
                    new_score = confidence_score(new_fact, rule_params)
                    old_score = confidence_score(conflicting, rule_params)
                    if new_score > old_score:
                        return _replace(
                            f"Replace: '{subj}' {pred} '{conflicting['object']}' "
                            f"(score {old_score}) → '{obj}' (score {new_score})",
                            conflicting.get("fact_id"),
                            rule_params,
                        )
                return _reject(
                    f"Contradiction: '{subj}' has conflicting '{pred}': '{o}' vs '{obj}'",
                    code="functional_conflict",
                    category="contradiction",
                    rule_id="functional_predicate_unique_object",
                    rule_params=rule_params,
                )

    triple_set = set(all_triples)
    for a, p, b in all_triples:
        if p == "PART_OF" and (b, "PART_OF", a) in triple_set:
            return _reject(
                f"Circular containment: '{a}' PART_OF '{b}' and '{b}' PART_OF '{a}'",
                code="circular_containment",
                category="graph_consistency",
                rule_id="part_of_acyclic_pair",
                rule_params=rule_params,
            )

    alive = set()
    dead = set()
    for s, p, o in all_triples:
        if p == "IS_ALIVE" and o == "true":
            alive.add(s)
        if p == "IS_ALIVE" and o == "false":
            dead.add(s)
    conflict = alive & dead
    if conflict:
        name = sorted(conflict)[0]
        return _reject(
            f"Conflict: '{name}' is both alive and dead",
            code="alive_dead_conflict",
            category="contradiction",
            rule_id="is_alive_single_truth_value",
            rule_params=rule_params,
        )

    return _accept(rule_params)


def validate_update_detailed(
    existing_facts: List[Dict[str, Any]],
    new_fact: Dict[str, Any],
    rule_params: Optional[RuleParameters] = None,
) -> ValidationDecision:
    """
    existing_facts: list of fact dicts already in the KG
    new_fact: fact dict being proposed
    returns: (decision, reason, replace_fact_id)
        decision        — "accept", "reject", or "replace"
        reason          — human-readable explanation
        replace_fact_id — fact_id of the existing fact to remove when
                          decision == "replace", otherwise None
    """
    params = rule_params or DEFAULT_RULE_PARAMETERS
    subj = str(new_fact["subject"])
    pred = sanitize_predicate(str(new_fact["predicate"]))
    obj = str(new_fact["object"])

    # --- Python-side checks (fast, no Scallop needed) ---

    if not subj or not obj or not pred:
        return _reject(
            "Rejected: subject, predicate, or object is empty",
            code="empty_required_field",
            category="shape",
            rule_id="required_triple_fields_non_empty",
            rule_params=params,
        )

    if subj.strip().lower() == obj.strip().lower():
        return _reject(
            f"Rejected: self-referential fact ({subj} -> {obj})",
            code="self_referential_fact",
            category="shape",
            rule_id="subject_object_must_differ",
            rule_params=params,
        )

    if obj.strip().lower() in set(params.generic_objects):
        return _reject(
            f"Rejected: object '{obj}' is too generic to be useful",
            code="generic_object",
            category="quality",
            rule_id="object_not_generic",
            rule_params=params,
        )

    new_triple = _to_triple(new_fact)
    if any(
        _to_triple(e) == new_triple and facts_temporally_overlap(e, new_fact)
        for e in existing_facts
    ):
        return _reject(
            f"Redundancy: {new_triple} already exists",
            code="duplicate_fact",
            category="deduplication",
            rule_id="triple_must_be_new",
            rule_params=params,
        )

    # --- Scallop-side checks (symbolic reasoning) ---

    # Symbolic contradictions apply only across overlapping temporal scopes.
    # Atemporal facts keep the old behavior because unbounded scopes overlap
    # everything.
    temporally_relevant_existing = [
        e for e in existing_facts
        if facts_temporally_overlap(e, new_fact)
    ]
    existing_triples = [_to_triple(e) for e in temporally_relevant_existing]

    if scallopy is None:
        return _python_symbolic_checks(
            existing_facts=existing_facts,
            new_fact=new_fact,
            existing_triples=existing_triples,
            new_triple=new_triple,
            rule_params=params,
        )

    ctx = scallopy.ScallopContext()
    ctx.add_relation("triple", (str, str, str))
    ctx.add_relation("functional_pred", (str,))

    ctx.add_facts("triple", existing_triples + [new_triple])
    ctx.add_facts("functional_pred", [(p,) for p in params.functional_predicates])

    # Contradiction: same subject + functional predicate, different object
    ctx.add_rule(
        "contradiction(s, p, o1, o2) :- "
        "triple(s, p, o1), triple(s, p, o2), "
        "functional_pred(p), o1 != o2"
    )

    # Circular containment: A PART_OF B and B PART_OF A
    ctx.add_rule(
        "circular_containment(a, b) :- "
        "triple(a, \"PART_OF\", b), triple(b, \"PART_OF\", a)"
    )

    # Existence contradiction: alive and dead at the same time
    ctx.add_rule(
        "alive_dead_conflict(s) :- "
        "triple(s, \"IS_ALIVE\", \"true\"), triple(s, \"IS_ALIVE\", \"false\")"
    )

    ctx.run()

    # --- Contradiction: confidence-based resolution ---
    contradictions = list(ctx.relation("contradiction"))
    if contradictions:
        s, p, o1, o2 = contradictions[0]
        # Find the existing fact that conflicts with the new one
        conflicting = next(
            (e for e in existing_facts
             if str(e["subject"]) == s
             and sanitize_predicate(str(e["predicate"])) == p
             and str(e["object"]) != obj
             and facts_temporally_overlap(e, new_fact)),
            None,
        )
        if conflicting is not None:
            new_score = confidence_score(new_fact, params)
            old_score = confidence_score(conflicting, params)
            if new_score > old_score:
                return _replace(
                    f"Replace: '{s}' {p} '{conflicting['object']}' "
                    f"(score {old_score}) → '{obj}' (score {new_score})",
                    conflicting.get("fact_id"),
                    params,
                )
        return _reject(
            f"Contradiction: '{s}' has conflicting '{p}': '{o1}' vs '{o2}'",
            code="functional_conflict",
            category="contradiction",
            rule_id="functional_predicate_unique_object",
            rule_params=params,
        )

    circular = list(ctx.relation("circular_containment"))
    if circular:
        a, b = circular[0]
        return _reject(
            f"Circular containment: '{a}' PART_OF '{b}' and '{b}' PART_OF '{a}'",
            code="circular_containment",
            category="graph_consistency",
            rule_id="part_of_acyclic_pair",
            rule_params=params,
        )

    alive_dead = list(ctx.relation("alive_dead_conflict"))
    if alive_dead:
        return _reject(
            f"Conflict: '{alive_dead[0][0]}' is both alive and dead",
            code="alive_dead_conflict",
            category="contradiction",
            rule_id="is_alive_single_truth_value",
            rule_params=params,
        )

    return _accept(params)


def validate_update(existing_facts, new_fact, rule_params: Optional[RuleParameters] = None):
    """
    Backward-compatible tuple API.

    Use validate_update_detailed(...) when callers need rejection labels or
    rule-parameter version metadata.
    """
    return validate_update_detailed(existing_facts, new_fact, rule_params).as_legacy_tuple()


if __name__ == "__main__":
    existing = [
        {"subject": "Indonesia",      "predicate": "CAPITAL_IS", "object": "Jakarta",
         "confidence": "uncertain", "provenance": [{"title": "doc1", "sent_id": 0}],
         "fact_id": "f001"},
        {"subject": "East Indonesia", "predicate": "PART_OF",    "object": "Indonesia",
         "confidence": "supported", "provenance": [{"title": "doc1", "sent_id": 1}],
         "fact_id": "f002"},
        {"subject": "Kalamang",       "predicate": "HAS_ISO_CODE", "object": "kgv",
         "confidence": "supported", "provenance": [{"title": "doc1", "sent_id": 2}],
         "fact_id": "f003"},
        {"subject": "John",           "predicate": "IS_ALIVE",   "object": "true",
         "confidence": "supported", "provenance": [],
         "fact_id": "f004"},
    ]

    # Test 1: contradiction — new fact has HIGHER confidence → replace
    new = {"subject": "Indonesia", "predicate": "CAPITAL_IS", "object": "Bandung",
           "confidence": "supported",
           "provenance": [{"title": "doc2", "sent_id": 0}, {"title": "doc3", "sent_id": 1}],
           "fact_id": "f005"}
    print("Test 1 (replace — new wins):", validate_update(existing, new))

    # Test 2: contradiction — new fact has LOWER confidence → reject
    new = {"subject": "Indonesia", "predicate": "CAPITAL_IS", "object": "Bandung",
           "confidence": "rejected", "provenance": [],
           "fact_id": "f006"}
    print("Test 2 (reject — old wins):", validate_update(existing, new))

    # Test 3: redundant
    new = {"subject": "Kalamang", "predicate": "HAS_ISO_CODE", "object": "kgv",
           "confidence": "supported", "provenance": [],
           "fact_id": "f007"}
    print("Test 3 (redundant):", validate_update(existing, new))

    # Test 4: circular containment
    new = {"subject": "Indonesia", "predicate": "PART_OF", "object": "East Indonesia",
           "confidence": "supported", "provenance": [],
           "fact_id": "f008"}
    print("Test 4 (circular containment):", validate_update(existing, new))

    # Test 5: self-referential
    new = {"subject": "Jakarta", "predicate": "LOCATED_IN", "object": "Jakarta",
           "confidence": "supported", "provenance": [],
           "fact_id": "f009"}
    print("Test 5 (self-referential):", validate_update(existing, new))

    # Test 6: alive/dead conflict
    new = {"subject": "John", "predicate": "IS_ALIVE", "object": "false",
           "confidence": "supported", "provenance": [],
           "fact_id": "f010"}
    print("Test 6 (alive/dead):", validate_update(existing, new))

    # Test 7: valid new fact
    new = {"subject": "Indonesia", "predicate": "LOCATED_IN", "object": "Southeast Asia",
           "confidence": "supported", "provenance": [{"title": "doc1", "sent_id": 3}],
           "fact_id": "f011"}
    print("Test 7 (valid):", validate_update(existing, new))
