# NeuroSymbolic LLM Context with Dynamic Knowledge Graphs

> **Thesis**: Replace the LLM's unstructured context buffer with a symbolically constrained dynamic knowledge graph, where logical rules govern what facts persist, ensuring contradiction-free long-horizon reasoning.

## Current Implementation: Foundation Pipeline + Dynamic KG

This repository implements the **baseline extraction pipeline** for converting LongBench-v2 contexts into structured triples *and* the **dynamic knowledge graph layer** (`neo4j_graph.py`) that stores them, queries them, and exposes the propose/commit seam that Scallop will plug into next.

### What's Implemented

**LLM-Powered Fact Extraction** — Uses Qwen 3.5 (4B) via vLLM to extract atomic facts from long-context documents

**Self-Reflection Validation** — 6-question verification ensures each fact is explicitly supported, atomic, and relevant

**Provenance & Citations** — Each fact links to source sentences with exact text evidence

**Dynamic Knowledge Graph (`neo4j_graph.py`)** — `Neo4jGraph` class wrapping Neo4j with idempotent `MERGE`-based inserts, `session_id`-scoped state for stateful **and** stateless modes, n-hop entity-seeded retrieval, and a deterministic LLM-context serializer

**Scallop-ready propose/commit API** — `propose_facts()` returns `{new, existing, conflicts}` so a Scallop validator can sit between proposal and write without API changes

**Contradiction probe primitive** — `find_conflicts(subject, predicate, object_)` surfaces `(s, p, ≠o)` collisions; the building block for the contradiction benchmark

### Output Format

Each extracted fact is a verified triple:
```json
{
  "subject": "Kalamang",
  "predicate": "SPOKEN_IN", 
  "object": "East Indonesia",
  "provenance": [{"title": "doc_0", "sent_id": 3}],
  "support_text": "It is spoken by around 130 people in East Indonesia.",
  "status": "supported"
}
```

## Prerequisites

- **Python 3.8+**
- **vLLM** with Qwen 3.5 4B model (or compatible LLM)
- **Neo4j 5.x** (optional, only needed for live graph inserts/queries)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download LongBench-v2 data
pip install datasets
python download_longbench.py

# Start vLLM server (separate terminal)
vllm serve Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8000

# Run extraction pipeline (extraction + validation only)
python longbench_kg_pipeline.py \
    --input data.jsonl \
    --output verified_facts.jsonl \
    --model Qwen/Qwen3.5-4B
```

### Stateful run (persistent graph across sessions)

```bash
python longbench_kg_pipeline.py \
    --input data.jsonl --output verified_facts.jsonl --model Qwen/Qwen3.5-4B \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j --neo4j-password yourpass
```

### Stateless run (per-session subgraph, wiped on close)

```bash
python longbench_kg_pipeline.py \
    --input data.jsonl --output verified_facts.jsonl --model Qwen/Qwen3.5-4B \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j --neo4j-password yourpass \
    --session-id run_2026_04_27 --stateless
```

See detailed options with `python longbench_kg_pipeline.py --help`

### Using `Neo4jGraph` directly

```python
from neo4j_graph import Neo4jGraph

with Neo4jGraph("bolt://localhost:7687", "neo4j", "yourpass",
                session_id="demo") as g:
    g.insert_facts(facts)                                  # propose + commit
    rows = g.query_context(["Kalamang"], hops=2, limit=20) # n-hop retrieval
    print(g.format_context_for_llm(rows))                  # LLM-ready text
    conflicts = g.find_conflicts("Indonesia", "CAPITAL_IS", "Bandung")
```

## Testing

```bash
python tests/test_pipeline_smoke.py  # End-to-end smoke test (mocked LLM)
python tests/test_validation.py      # Extraction/verification logic
python tests/test_neo4j_format.py    # Pipeline -> Neo4j formatting
python tests/test_neo4j_graph.py     # Neo4jGraph: insert/query/propose/format (mocked driver)
```

No live Neo4j is required — all tests use a mock driver and the checked-in fixture `tests/fixtures/verified_facts.sample.jsonl`.

## Architecture Roadmap

### Current: Baseline Extraction + Dynamic KG (✓ Implemented)
- LLM-powered fact extraction from LongBench-v2
- Self-reflection validation (6-question verification)
- Neo4j-ready triple output with provenance
- **`neo4j_graph.py`** — `Neo4jGraph`: idempotent `MERGE` inserts, `session_id`-scoped state (stateful **and** stateless), n-hop `query_context`, deterministic `format_context_for_llm`, `propose`/`commit` seam for Scallop, `find_conflicts` primitive for the contradiction probe

### Next: Scallop Symbolic Constraints
- **Logical rule enforcement** — Scallop programs validate fact updates between `propose_facts` and `commit_facts`
- **Contradiction detection** — Symbolic constraints veto conflicting writes (today `find_conflicts` surfaces them; next they get rejected)
- **Recursive context updates** — Treat context as constrained symbolic state transitions

### Future: Recursive LLM Integration
- **RLM-style retrieval** — Recursive language model queries the graph via `query_context` instead of reading raw long contexts
- **Benchmarks** — LongBench-v2, HotpotQA / MuSiQue (multi-hop), and the contradiction probe

## Positioning vs. Related Work

| System | Memory Type | Symbolic Constraints | Recursive |
|--------|-------------|---------------------|-----------|
| Vanilla RAG | Vector store | ✗ | ✗ |
| GEPA / Dynamic Cheatsheet | Unstructured text | ✗ | ✗ |
| RLMs | Raw context chunks | ✗ | ✓ |
| KG-RAG | Static KG | Schema-only | ✗ |
| **Ours (planned)** | **Dynamic KG** | **Scallop logic** | **✓** |

**Key Innovation**: No existing system treats context update as a constrained symbolic state transition. Current approaches use either unconstrained text memory or static knowledge graphs.
