import scallopy

def validate_update(existing_facts, new_fact):
    ctx = scallopy.ScallopContext()

    ctx.add_relation("triple", (str, str, str))

    all_facts = existing_facts + [new_fact]
    ctx.add_facts("triple", all_facts)

    ctx.add_rule("contradiction(s, p, o1, o2) :- triple(s, p, o1), triple(s, p, o2), o1 != o2")
    ctx.run()

    contradictions = list(ctx.relation("contradiction"))
    if contradictions:
        return (False, f"Contradiction: {contradictions}")
    elif new_fact in existing_facts:
        return (False, f"Redundancy: {new_fact} already exists")
    return (True, "No contradictions or redundancies.")


if __name__ == "__main__":
    existing = [
        ("I", "born_in", "California"),
        ("I", "occupation", "Student"),
        ("I", "love", "computer science")
    ]

    #Test 1: Contradiction (False)
    result = validate_update(existing, ("I", "born_in", "Georgia"))
    print("Test 1:", result)

    #Test 2: Redundant (False)
    result = validate_update(existing, ("I", "occupation", "Student"))
    print("Test 2:", result)

    #Test 3: Contradiction (False)
    result = validate_update(existing, ("I", "occupation", "Doctor"))
    print("Test 3:", result)

    #Test 4: Valid (True)
    result = validate_update(existing, ("I", "love", "food"))
    print("Test 4:", result)