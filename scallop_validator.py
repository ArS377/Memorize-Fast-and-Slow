import scallopy

# Predicates where a subject can only have one value.
# e.g. a country has one capital, a person has one birthplace.
FUNCTIONAL_PREDICATES = {
    "CAPITAL_IS", "BORN_IN", "BIRTH_DATE", "DEATH_DATE",
    "DIED_IN", "FOUNDED_IN", "LOCATED_IN", "SPOKEN_IN",
    "HAS_ISO_CODE", "HAS_GLOTTOCODE",
}

# Objects that are too vague to be worth storing.
GENERIC_OBJECTS = {
    "unknown", "n/a", "none", "various", "multiple",
    "several", "many", "some", "thing", "entity",
}


def validate_update(existing_facts, new_fact):
    """
    existing_facts: list of (subject, predicate, object) tuples already in the KG
    new_fact: (subject, predicate, object) tuple being proposed
    returns: (bool, str) — (is_valid, reason)
    """
    subj, pred, obj = new_fact

    # --- Python-side checks (fast, no Scallop needed) ---

    if not subj or not obj or not pred:
        return (False, "Rejected: subject, predicate, or object is empty")

    if subj.strip().lower() == obj.strip().lower():
        return (False, f"Rejected: self-referential fact ({subj} -> {obj})")

    if obj.strip().lower() in GENERIC_OBJECTS:
        return (False, f"Rejected: object '{obj}' is too generic to be useful")

    if new_fact in existing_facts:
        return (False, f"Redundancy: {new_fact} already exists")

    # --- Scallop-side checks (symbolic reasoning) ---

    ctx = scallopy.ScallopContext()
    ctx.add_relation("triple", (str, str, str))
    ctx.add_relation("functional_pred", (str,))

    all_facts = existing_facts + [new_fact]
    ctx.add_facts("triple", all_facts)
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

    contradictions = list(ctx.relation("contradiction"))
    if contradictions:
        s, p, o1, o2 = contradictions[0]
        return (False, f"Contradiction: '{s}' has conflicting '{p}': '{o1}' vs '{o2}'")

    circular = list(ctx.relation("circular_containment"))
    if circular:
        a, b = circular[0]
        return (False, f"Circular containment: '{a}' PART_OF '{b}' and '{b}' PART_OF '{a}'")

    alive_dead = list(ctx.relation("alive_dead_conflict"))
    if alive_dead:
        return (False, f"Conflict: '{alive_dead[0][0]}' is both alive and dead")

    return (True, "Valid")


if __name__ == "__main__":
    existing = [
        ("Indonesia", "CAPITAL_IS", "Jakarta"),
        ("East Indonesia", "PART_OF", "Indonesia"),
        ("Kalamang", "SPOKEN_IN", "East Indonesia"),
        ("Kalamang", "HAS_ISO_CODE", "kgv"),
        ("John", "IS_ALIVE", "true"),
    ]

    # Test 1: functional predicate contradiction (False)
    result = validate_update(existing, ("Indonesia", "CAPITAL_IS", "Bandung"))
    print("Test 1 (functional contradiction):", result)

    # Test 2: redundant fact (False)
    result = validate_update(existing, ("Kalamang", "HAS_ISO_CODE", "kgv"))
    print("Test 2 (redundant):", result)

    # Test 3: circular containment (False)
    result = validate_update(existing, ("Indonesia", "PART_OF", "East Indonesia"))
    print("Test 3 (circular containment):", result)

    # Test 4: self-referential (False)
    result = validate_update(existing, ("Jakarta", "LOCATED_IN", "Jakarta"))
    print("Test 4 (self-referential):", result)

    # Test 5: generic object (False)
    result = validate_update(existing, ("Kalamang", "SPOKEN_IN", "various"))
    print("Test 5 (generic object):", result)

    # Test 6: alive/dead conflict (False)
    result = validate_update(existing, ("John", "IS_ALIVE", "false"))
    print("Test 6 (alive/dead conflict):", result)

    # Test 7: valid new fact (True)
    result = validate_update(existing, ("Indonesia", "LOCATED_IN", "Southeast Asia"))
    print("Test 7 (valid):", result)

    # Test 8: non-functional predicate with multiple values (True)
    result = validate_update(existing, ("Kalamang", "SPOKEN_IN", "West Papua"))
    print("Test 8 (non-functional multi-value):", result)
