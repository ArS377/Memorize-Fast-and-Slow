# Persona NeuroSym Neo4j Preflight v2

This artifact verifies Scallop fact admission, live graph retrieval, hybrid fusion, and prompt construction for the persona hybrid-memory arms.

The run replayed all 120 conditions through actual Scallopy admission, committed 946 condition-scoped facts to Neo4j after the source fact and event lineage round-trip invariant passed, executed scoped two-hop sparse traversal plus BGE dense retrieval, fused both branches with reciprocal-rank fusion, and fitted 240 matched 4K/16K prompts.

All 120 retrievals were non-degraded and contained at least one sparse and one dense candidate.

The live traversal canary returned one fact at one hop and two facts at two hops.
Its adversarial cross-condition path returned zero facts.

The preflight used `scallopy 0.2.4`, `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, and the Qwen3.5 tokenizer revision `2fc06364715b967f1860aea9cf38778875588b17`.

`preflight_manifest.json` authenticates clean source commit `e452b46c97aa7e39aabb281958d24d86a170f061`, the source corpus, configuration, graph identity, Scallop identities, traversal canary, retrieval map, prompt specifications, and artifact hashes.

The separate Scallop relation-injection gate is complete for all delayed probes.
Token-distance checkpoints can precede the alias event needed to derive a preference-change or preference-incongruity relation, so relation-source coverage at those checkpoints is recorded but is not required to be nonzero.

The manifest hashes the configured Neo4j endpoint and database but does not authenticate a Neo4j server-instance identifier or server version.

This artifact verifies retrieval execution and prompt construction, not Qwen answer quality.
