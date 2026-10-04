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
import re
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


def test_pipeline_smoke_uses_fixture_and_tmp_path(tmp_path: Path) -> None:
    import longbench_kg_pipeline as pipe

    data_path = Path(__file__).parent / "fixtures" / "pipeline_smoke_input.jsonl"
    output_path = tmp_path / "smoke_output.jsonl"
    examples = pipe.load_longbench_examples(data_path)
    assert len(examples) == 2

    config = pipe.PipelineConfig(
        input_path=data_path,
        output_path=output_path,
        model="mock-model",
        vllm_base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=512,
        chunk_chars=50_000,
        max_chunks_per_example=1,
        limit=2,
        sleep_seconds=0.0,
        use_json_mode=False,
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
    )
    pipeline = pipe.LongBenchKGPipeline(config)
    pipeline.client = _make_mock_client(
        [
            FAKE_EXTRACTION_RESPONSE,
            FAKE_VERIFICATION_RESPONSE,
            FAKE_EXTRACTION_RESPONSE,
            FAKE_VERIFICATION_RESPONSE,
        ]
    )

    try:
        pipeline.run()
    finally:
        pipeline.close()

    assert output_path.exists()
    output_lines = [line for line in output_path.read_text().splitlines() if line.strip()]
    assert len(output_lines) == 2
    fact = json.loads(output_lines[0])
    required = {
        "subject",
        "predicate",
        "object",
        "provenance",
        "support_text",
        "question_relevance",
        "example_id",
        "fact_id",
        "question",
    }
    assert required.issubset(fact)
    assert fact["status"] == "supported"
    assert fact["question"]
    assert fact["provenance"]
    assert re.fullmatch(r"[A-Z0-9_]+", fact["predicate"])
