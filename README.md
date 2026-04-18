# NeuroSymbolic LLM Context with Dynamic Knowledge Graphs

> **Thesis**: Replace the LLM's unstructured context buffer with a symbolically constrained dynamic knowledge graph, where logical rules govern what facts persist, ensuring contradiction-free long-horizon reasoning.

## Current Implementation: Foundation Pipeline

This repository currently implements the **baseline extraction pipeline** for converting LongBench-v2 contexts into structured knowledge graph triples. This forms the foundation for future integration with Scallop-enforced symbolic constraints.

### What's Implemented

**LLM-Powered Fact Extraction** — Uses Qwen 3.5 (4B) via vLLM to extract atomic facts from long-context documents

**Self-Reflection Validation** — 6-question verification ensures each fact is explicitly supported, atomic, and relevant

**Neo4j-Ready Output** — Structured triples with provenance tracking, ready for graph database insertion

**Provenance & Citations** — Each fact links to source sentences with exact text evidence

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
- **Neo4j** (optional, for knowledge graph insertion)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download LongBench-v2 data
pip install datasets
python download_longbench.py

# Start vLLM server (separate terminal)
vllm serve Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8000

# Run extraction pipeline
python longbench_kg_pipeline.py \
    --input data.jsonl \
    --output verified_facts.jsonl \
    --model Qwen/Qwen3.5-4B
```

See detailed options with `python longbench_kg_pipeline.py --help`

## Testing

```bash
python tests/test_pipeline_smoke.py  # End-to-end smoke test
python tests/test_validation.py      # Validation logic
python tests/test_neo4j_format.py    # Neo4j formatting
```

## Architecture Roadmap

### Current: Baseline Extraction (✓ Implemented)
- LLM-powered fact extraction from LongBench-v2
- Self-reflection validation (6-question verification)
- Neo4j-ready triple output with provenance

### Next: Dynamic Knowledge Graph
- **Stateful Neo4j integration** — Persistent graph across sessions
- **Stateless variant** — Per-session graph construction
- **Graph-to-context serialization** — Extract structured context for LLM ingestion

### Future: Scallop Symbolic Constraints
- **Logical rule enforcement** — Scallop programs validate fact updates
- **Contradiction detection** — Symbolic constraints prevent conflicting facts
- **Recursive context updates** — Treat context as constrained symbolic state transitions

## Positioning vs. Related Work

| System | Memory Type | Symbolic Constraints | Recursive |
|--------|-------------|---------------------|-----------|
| Vanilla RAG | Vector store | ✗ | ✗ |
| GEPA / Dynamic Cheatsheet | Unstructured text | ✗ | ✗ |
| RLMs | Raw context chunks | ✗ | ✓ |
| KG-RAG | Static KG | Schema-only | ✗ |
| **Ours (planned)** | **Dynamic KG** | **Scallop logic** | **✓** |

**Key Innovation**: No existing system treats context update as a constrained symbolic state transition. Current approaches use either unconstrained text memory or static knowledge graphs.
