"""One-shot demo: probe the Neo4j graph populated by the fixture insert."""
from neo4j_graph import Neo4jGraph

with Neo4jGraph(
    "neo4j://127.0.0.1:7687",
    "neo4j",
    "acmaisf2026",
    session_id="demo_run",
) as g:
    print("--- find_conflicts(Indonesia, CAPITAL_IS, Jakarta) ---")
    for c in g.find_conflicts("Indonesia", "CAPITAL_IS", "Jakarta", session_id="demo_run"):
        print(f"  CONFLICT: existing object={c['object']!r} fact_id={c['fact_id']}")

    print("\n--- query_context: 1-hop neighborhood of Kalamang ---")
    rows = g.query_context(["Kalamang"], hops=1, limit=20, session_id="demo_run")
    for r in rows:
        print(f"  {r['subject']} --{r['predicate']}--> {r['object']}")

    print("\n--- query_context: 2-hop from Kalamang (reaches Indonesia) ---")
    rows2 = g.query_context(["Kalamang"], hops=2, limit=30, session_id="demo_run")
    for r in rows2:
        print(f"  {r['subject']} --{r['predicate']}--> {r['object']}")

    print("\n--- format_context_for_llm output (what the LLM would receive) ---")
    print(g.format_context_for_llm(rows2))
