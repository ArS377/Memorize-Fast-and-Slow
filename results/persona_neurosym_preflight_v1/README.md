# Persona NeuroSym Retrieval Preflight v1

This artifact verifies the non-generative layers of the matched persona benchmark on all 120 authenticated conditions.

The run used the current HTTP Scallop service for candidate admission and preference-source derivation, built one condition-scoped 946-fact dense snapshot, executed sparse+dense reciprocal-rank fusion for every condition, and fitted 240 matched 4K/16K hybrid-memory prompts.

All 120 retrieval rows report non-degraded hybrid execution with at least one sparse and one dense candidate.

The dense encoder was `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.

The Qwen tokenizer was `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`.

This preflight uses the supported condition-scoped JSONL fact repository.
It does not exercise Neo4j n-hop traversal, Qwen generation, or answer-quality improvement.

`preflight_manifest.json` authenticates the source corpus, configuration, clean git checkpoint, Scallop identities, retrieval map, prompt specifications, and artifact hashes.
