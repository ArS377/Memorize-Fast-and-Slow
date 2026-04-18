#!/usr/bin/env python3
"""
Smoke-test for longbench_kg_pipeline.py.

Runs the full pipeline against the first 2 rows of data.jsonl using a
*mock* LLM — no vLLM server or API key needed.

Usage:
    python test_pipeline_smoke.py

Pass: prints "ALL CHECKS PASSED" and writes smoke_output.jsonl.
Fail: prints the assertion that failed.
"""
from __future__ import annotations

import json
import sys
import types
import unittest.mock as mock
from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  Patch openai.OpenAI before importing the pipeline so that the module
#     loads cleanly even if the real openai package is not installed yet.
# ---------------------------------------------------------------------------
FAKE_EXTRACTION_RESPONSE = json.dumps({
    "facts": [
        {
            "subject": "Kalamang",
            "predicate": "SPOKEN_IN",
            "object": "East Indonesia",
            "qualifiers": {},
            "provenance": [{"title": "example_0", "sent_id": 0}],
            "support_text": "It is spoken by around 130 people in East Indonesia.",
            "question_relevance": "Relevant to locate the language geographically.",
            "confidence": "supported",
            "normalization_notes": "",
        }
    ]
})

FAKE_VERIFICATION_RESPONSE = json.dumps({
    "verified_facts": [
        {
            "verification_id": "f0",
            "status": "supported",
            "verification_reason": "Directly stated in support_text.",
            "subject": "Kalamang",
            "predicate": "SPOKEN_IN",
            "object": "East Indonesia",
        }
    ]
})

def _make_mock_client(responses):
    """Return a mock OpenAI client that cycles through `responses`."""
    client = mock.MagicMock()
    call_iter = iter(responses)

    def fake_create(**kwargs):
        text = next(call_iter)
        choice = mock.MagicMock()
        choice.message.content = text
        resp = mock.MagicMock()
        resp.choices = [choice]
        return resp

    client.chat.completions.create.side_effect = fake_create
    return client


# ---------------------------------------------------------------------------
# 2.  Build a fake `openai` module so the import inside the pipeline succeeds
#     even without the real package.
# ---------------------------------------------------------------------------
if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = mock.MagicMock  # will be replaced per-instance below
    sys.modules["openai"] = fake_openai

# Now import the pipeline (it will see the fake openai module).
sys.path.insert(0, str(Path(__file__).parent.parent))
import longbench_kg_pipeline as pipe  # noqa: E402


# ---------------------------------------------------------------------------
# 3.  Load two real examples from data.jsonl.
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "data.jsonl"
if not DATA_PATH.exists():
    # Try current directory (in case script is run from a different cwd)
    DATA_PATH = Path("data.jsonl")

examples = pipe.load_longbench_examples(DATA_PATH)[:2]
assert len(examples) >= 1, "data.jsonl must have at least 1 row"
print(f"Loaded {len(examples)} example(s) from {DATA_PATH}")


# ---------------------------------------------------------------------------
# 4.  Run the pipeline with the mock client.
# ---------------------------------------------------------------------------
OUTPUT_PATH = Path("smoke_output.jsonl")

config = pipe.PipelineConfig(
    input_path=DATA_PATH,
    output_path=OUTPUT_PATH,
    model="mock-model",
    vllm_base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.0,
    max_tokens=512,
    chunk_chars=50_000,       # large enough to fit example in one chunk
    max_chunks_per_example=1, # only one chunk per example for speed
    limit=2,
    sleep_seconds=0.0,
    use_json_mode=False,
    neo4j_uri=None,
    neo4j_user=None,
    neo4j_password=None,
)

pipeline = pipe.LongBenchKGPipeline(config)

# Each example uses 2 LLM calls: 1 extraction + 1 verification.
# We have 2 examples → 4 calls total; alternate extraction/verification.
mock_responses = [
    FAKE_EXTRACTION_RESPONSE,
    FAKE_VERIFICATION_RESPONSE,
    FAKE_EXTRACTION_RESPONSE,
    FAKE_VERIFICATION_RESPONSE,
]
pipeline.client = _make_mock_client(mock_responses)

pipeline.run()


# ---------------------------------------------------------------------------
# 5.  Assertions
# ---------------------------------------------------------------------------
assert OUTPUT_PATH.exists(), "Output JSONL was not created."

output_lines = [l for l in OUTPUT_PATH.read_text().splitlines() if l.strip()]
assert len(output_lines) > 0, "No facts were written to output."

fact = json.loads(output_lines[0])

# Core structure checks
assert "subject" in fact,           "Missing 'subject' field"
assert "predicate" in fact,         "Missing 'predicate' field"
assert "object" in fact,            "Missing 'object' field"
assert "provenance" in fact,        "Missing 'provenance' field"
assert "support_text" in fact,      "Missing 'support_text' field"
assert "question_relevance" in fact,"Missing 'question_relevance' field"
assert "example_id" in fact,        "Missing 'example_id' field"
assert "fact_id" in fact,           "Missing 'fact_id' field"
assert "question" in fact,          "Missing 'question' field (Bug #3 fix check)"

# Status check
assert fact.get("status") == "supported", f"Expected status=supported, got {fact.get('status')}"

# FIX 3 validation: question must not be empty (the key bug we fixed)
assert fact["question"] != "", "question field is empty — Bug #3 may not be fixed"

# Predicate must be UPPER_SNAKE_CASE
import re
assert re.match(r"^[A-Z0-9_]+$", fact["predicate"]), \
    f"Predicate not in UPPER_SNAKE_CASE: {fact['predicate']}"

print("\n--- Sample output fact ---")
print(json.dumps(fact, indent=2, ensure_ascii=False))
print(f"\nTotal facts written: {len(output_lines)}")
print("\n✅  ALL CHECKS PASSED")
