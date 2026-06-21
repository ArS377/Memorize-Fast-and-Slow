try:
    import scallopy
except ImportError:  # Keep non-Scallop tests/imports usable.
    scallopy = None

# Predicates where a subject can only have one value.
FUNCTIONAL_PREDICATES = {
    "CAPITAL_IS", "BORN_IN", "BIRTH_DATE", "DEATH_DATE",
    "DIED_IN", "FOUNDED_IN", "LOCATED_IN",
    "HAS_ISO_CODE", "HAS_GLOTTOCODE",
}

# Objects that are too vague to be worth storing.
GENERIC_OBJECTS = {
    "unknown", "n/a", "none", "various", "multiple",
    "several", "many", "some", "thing", "entity",
}


def confidence_score(fact):
    """
    Numeric score for a fact dict. Higher = more trustworthy.
    Primary: confidence level (supported > uncertain > rejected).
    Tiebreaker: provenance count (more source citations = more reliable).
    """
    level = {"supported": 3, "uncertain": 2, "rejected": 1}.get(
        str(fact.get("confidence", "")).lower(), 0
    )
    provenance_count = len(fact.get("provenance", []))
    return level * 10 + provenance_count


def _to_triple(fact):
    """Extract (subject, predicate, object) tuple from a fact dict."""
    return (str(fact["subject"]), str(fact["predicate"]), str(fact["object"]))


def _python_symbolic_checks(existing_facts, new_fact, existing_triples, new_triple):
    """Fallback implementation of the current Scallop rules.

    This keeps the pipeline importable in environments that do not have the
    Scallop Python wheel installed. It intentionally mirrors only the rules
    below, not arbitrary Scallop programs.
    """
    all_triples = existing_triples + [new_triple]
    subj, pred, obj = new_triple

    if pred in FUNCTIONAL_PREDICATES:
        for s, p, o in all_triples:
            if s == subj and p == pred and o != obj:
                conflicting = next(
                    (
                        e for e in existing_facts
                        if str(e["subject"]) == subj
                        and str(e["predicate"]) == pred
                        and str(e["object"]) != obj
                    ),
                    None,
                )
                if conflicting is not None:
                    new_score = confidence_score(new_fact)
                    old_score = confidence_score(conflicting)
                    if new_score > old_score:
                        return (
                            "replace",
                            f"Replace: '{subj}' {pred} '{conflicting['object']}' "
                            f"(score {old_score}) → '{obj}' (score {new_score})",
                            conflicting.get("fact_id"),
                        )
                return (
                    "reject",
                    f"Contradiction: '{subj}' has conflicting '{pred}': '{o}' vs '{obj}'",
                    None,
                )

    triple_set = set(all_triples)
    for a, p, b in all_triples:
        if p == "PART_OF" and (b, "PART_OF", a) in triple_set:
            return (
                "reject",
                f"Circular containment: '{a}' PART_OF '{b}' and '{b}' PART_OF '{a}'",
                None,
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
        return ("reject", f"Conflict: '{name}' is both alive and dead", None)

    return ("accept", "Valid", None)


def validate_update(existing_facts, new_fact):
    """
    existing_facts: list of fact dicts already in the KG
    new_fact: fact dict being proposed
    returns: (decision, reason, replace_fact_id)
        decision        — "accept", "reject", or "replace"
        reason          — human-readable explanation
        replace_fact_id — fact_id of the existing fact to remove when
                          decision == "replace", otherwise None
    """
    subj = str(new_fact["subject"])
    pred = str(new_fact["predicate"])
    obj  = str(new_fact["object"])

    # --- Python-side checks (fast, no Scallop needed) ---

    if not subj or not obj or not pred:
        return ("reject", "Rejected: subject, predicate, or object is empty", None)

    if subj.strip().lower() == obj.strip().lower():
        return ("reject", f"Rejected: self-referential fact ({subj} -> {obj})", None)

    if obj.strip().lower() in GENERIC_OBJECTS:
        return ("reject", f"Rejected: object '{obj}' is too generic to be useful", None)

    new_triple = _to_triple(new_fact)
    if any(_to_triple(e) == new_triple for e in existing_facts):
        return ("reject", f"Redundancy: {new_triple} already exists", None)

    # --- Scallop-side checks (symbolic reasoning) ---

    existing_triples = [_to_triple(e) for e in existing_facts]

    if scallopy is None:
        return _python_symbolic_checks(
            existing_facts=existing_facts,
            new_fact=new_fact,
            existing_triples=existing_triples,
            new_triple=new_triple,
        )

    ctx = scallopy.ScallopContext()
    ctx.add_relation("triple", (str, str, str))
    ctx.add_relation("functional_pred", (str,))

    ctx.add_facts("triple", existing_triples + [new_triple])
    ctx.add_facts("functional_pred", [(p,) for p in FUNCTIONAL_PREDICATES])

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
             and str(e["predicate"]) == p
             and str(e["object"]) != obj),
            None,
        )
        if conflicting is not None:
            new_score = confidence_score(new_fact)
            old_score = confidence_score(conflicting)
            if new_score > old_score:
                return (
                    "replace",
                    f"Replace: '{s}' {p} '{conflicting['object']}' "
                    f"(score {old_score}) → '{obj}' (score {new_score})",
                    conflicting.get("fact_id"),
                )
        return (
            "reject",
            f"Contradiction: '{s}' has conflicting '{p}': '{o1}' vs '{o2}'",
            None,
        )

    circular = list(ctx.relation("circular_containment"))
    if circular:
        a, b = circular[0]
        return (
            "reject",
            f"Circular containment: '{a}' PART_OF '{b}' and '{b}' PART_OF '{a}'",
            None,
        )

    alive_dead = list(ctx.relation("alive_dead_conflict"))
    if alive_dead:
        return ("reject", f"Conflict: '{alive_dead[0][0]}' is both alive and dead", None)

    return ("accept", "Valid", None)


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
