#!/usr/bin/env python3
"""Regression tests for accepted/rejected JSONL separation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from longbench_kg_pipeline import (  # noqa: E402
    LongBenchKGPipeline,
    PipelineConfig,
    enrich_fact_provenance,
)


def make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "accepted.jsonl",
        model="mock-model",
        vllm_base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=128,
        chunk_chars=1000,
        max_chunks_per_example=1,
        limit=None,
        sleep_seconds=0.0,
        use_json_mode=False,
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
        rejections_output_path=tmp_path / "rejections.jsonl",
    )


class RejectingGraph:
    session_id = "rejecting_graph"

    def insert_facts(self, facts):
        return {
            "committed": 0,
            "conflicts": [],
            "replaced": [],
            "rejected": [
                {
                    "candidate": facts[0],
                    "reason": "Rejected: object 'unknown' is too generic to be useful",
                    "rule_fired": "generic_object",
                    "existing_conflicting_fact": None,
                }
            ],
        }

    def close(self):
        return None


def test_graph_rejected_fact_is_not_written_to_supported_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "accepted.jsonl"
    rejections_path = tmp_path / "rejections.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "_id": "ex_1",
                "question": "What is A related to?",
                "context": "A is related to unknown.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = make_config(tmp_path)
    config.input_path = input_path
    config.output_path = output_path
    config.rejections_output_path = rejections_path
    pipeline = LongBenchKGPipeline(config)
    pipeline.graph = RejectingGraph()
    pipeline.extract_facts = lambda example, chunk, chunk_index: [
        {
            "subject": "A",
            "predicate": "RELATED_TO",
            "object": "unknown",
            "qualifiers": {},
            "provenance": [{"title": "ex_1", "sent_id": 0}],
            "support_text": "A is related to unknown.",
            "question_relevance": "Candidate relation.",
            "confidence": "supported",
            "status": "supported",
            "normalization_notes": "",
            "verification_reason": "Directly stated.",
        }
    ]
    pipeline.verify_facts = lambda example, facts: facts

    try:
        pipeline.run()
    finally:
        pipeline.close()

    assert output_path.read_text(encoding="utf-8") == ""
    rejection_lines = [
        line for line in rejections_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rejection_lines) == 1
    record = json.loads(rejection_lines[0])
    assert record["rule_fired"] == "generic_object"
    assert record["candidate_fact"]["fact_id"]
    assert record["candidate_fact"]["object"] == "unknown"


def test_missing_verifier_judgment_becomes_rejected_fact(tmp_path: Path) -> None:
    pipeline = LongBenchKGPipeline(make_config(tmp_path))
    pipeline.chat_json = lambda prompt: {"verified_facts": []}

    facts = [{
        "subject": "A",
        "predicate": "RELATED_TO",
        "object": "B",
        "support_text": "A is related to B.",
        "provenance": [{"title": "ex", "sent_id": 0}],
        "confidence": "supported",
    }]

    verified = pipeline.verify_facts({"_id": "ex"}, facts)

    assert len(verified) == 1
    assert verified[0]["status"] == "rejected"
    assert verified[0]["verification_reason"] == "No verifier judgment returned."


def test_provenance_is_recovered_from_support_text_when_extractor_omits_it() -> None:
    fact = {
        "subject": "A",
        "predicate": "RELATED_TO",
        "object": "B",
        "provenance": [],
        "support_text": "A is directly related to B.",
    }
    chunk = [{
        "title": "doc",
        "sent_id": 7,
        "local_sent_id": 1,
        "text": "Context prefix. A is directly related to B. Context suffix.",
    }]

    enrich_fact_provenance(
        fact,
        example={"_id": "ex_1"},
        chunk=chunk,
        chunk_index=2,
        extractor_model="extractor",
        verifier_model="",
        run_id="run_test",
    )

    assert len(fact["provenance"]) == 1
    provenance = fact["provenance"][0]
    assert provenance["title"] == "doc"
    assert provenance["sent_id"] == 7
    assert provenance["sentence_id"] == "ex_1:7"
    assert provenance["source_span_start"] > 0
    assert provenance["source_span_end"] > provenance["source_span_start"]
    assert provenance["extractor_model"] == "extractor"
    assert provenance["run_id"] == "run_test"
