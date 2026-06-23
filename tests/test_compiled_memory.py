#!/usr/bin/env python3
"""Tests for the compiled memory compatibility layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from compiled_memory import (  # noqa: E402
    TemporalScope,
    compiled_memory_to_fact,
    compiled_memory_to_neo4j_properties,
    compiled_memory_to_scallop_facts,
    evidence_strength_score,
    format_fact_rows_for_llm,
    fact_to_compiled_fact,
    fact_to_compiled_memory,
    provenance_quality_score,
)


def sample_fact():
    return {
        "subject": "Alice Chen",
        "predicate": "works at",
        "object": "Acme Robotics",
        "subject_type": "person",
        "object_type": "organization",
        "subject_aliases": ["Alice"],
        "object_aliases": ["Acme"],
        "temporal": {
            "valid_from": "2025-01-01",
            "valid_to": None,
            "observed_at": "2026-06-21T00:00:00Z",
        },
        "qualifiers": {"role": "engineer"},
        "provenance": [{
            "title": "doc_1",
            "sent_id": 4,
            "document_id": "doc_1",
            "sentence_id": "doc_1:4",
            "source_span_start": 0,
            "source_span_end": 43,
            "extractor_model": "extractor-x",
            "verifier_model": "verifier-x",
            "run_id": "run_test",
        }],
        "support_text": "Alice Chen joined Acme Robotics in 2025.",
        "question_relevance": "Identifies Alice Chen's employer.",
        "confidence": "supported",
        "confidence_score": 0.91,
        "normalization_notes": "Alice resolved to Alice Chen.",
        "example_id": "ex_1",
        "fact_id": "fact_1",
        "question": "Where does Alice Chen work?",
        "verification_reason": "Directly supported.",
        "document_id": "doc_1",
        "extractor_model": "extractor-x",
        "verifier_model": "verifier-x",
        "run_id": "run_test",
    }


def test_fact_to_compiled_memory_preserves_entities():
    memory = fact_to_compiled_memory(sample_fact())
    assert memory.memory_id == "fact_1"
    assert memory.subject.name == "Alice Chen"
    assert memory.subject.entity_type == "person"
    assert memory.subject.aliases == ["Alice"]
    assert memory.predicate == "WORKS_AT"
    assert memory.object.name == "Acme Robotics"
    assert memory.object.entity_type == "organization"
    assert memory.object.aliases == ["Acme"]
    assert memory.temporal.valid_from == "2025-01-01"
    assert memory.constraints.scope == "temporal"
    assert memory.confidence.level == "supported"
    assert memory.confidence.score == 0.91
    assert memory.provenance[0].document_id == "doc_1"
    assert memory.provenance[0].sentence_id == "doc_1:4"
    assert memory.provenance[0].source_span_start == 0
    assert memory.extractor_model == "extractor-x"
    return True


def test_compiled_memory_round_trips_to_fact_contract():
    compiled_fact = fact_to_compiled_fact(sample_fact())
    assert compiled_fact["subject"] == "Alice Chen"
    assert compiled_fact["predicate"] == "WORKS_AT"
    assert compiled_fact["object"] == "Acme Robotics"
    assert compiled_fact["fact_id"] == "fact_1"
    assert compiled_fact["confidence"] == "supported"
    assert compiled_fact["temporal"]["valid_from"] == "2025-01-01"
    assert "compiled_memory" in compiled_fact
    assert compiled_fact["compiled_memory"]["subject"]["id"].startswith("entity_")
    assert compiled_fact["compiled_memory"]["object"]["id"].startswith("entity_")

    round_trip = compiled_memory_to_fact(fact_to_compiled_memory(compiled_fact))
    assert round_trip["subject"] == compiled_fact["subject"]
    assert round_trip["predicate"] == compiled_fact["predicate"]
    assert round_trip["fact_id"] == compiled_fact["fact_id"]
    assert (
        round_trip["compiled_memory"]["temporal"]["observed_at"]
        == compiled_fact["compiled_memory"]["temporal"]["observed_at"]
    )
    return True


def test_temporal_scope_overlap_semantics():
    old_job = TemporalScope(valid_from="2020-01-01", valid_to="2024-12-31")
    new_job = TemporalScope(valid_from="2025-01-01", valid_to=None)
    overlapping_job = TemporalScope(valid_from="2024-06-01", valid_to="2025-06-01")

    assert not old_job.overlaps(new_job)
    assert old_job.overlaps(overlapping_job)
    assert new_job.overlaps(overlapping_job)
    return True


def test_scallop_projection_contains_memory_relations():
    memory = fact_to_compiled_memory(sample_fact())
    relations = compiled_memory_to_scallop_facts(memory)

    assert "entity" in relations
    assert "alias" in relations
    assert "memory_relationship" in relations
    assert "validity" in relations
    assert "confidence" in relations
    assert "decision" in relations
    assert "constraint" in relations
    assert "provenance" in relations

    relationship = relations["memory_relationship"][0]
    assert relationship[0] == "fact_1"
    assert relationship[2] == "WORKS_AT"
    assert relations["validity"][0] == ("fact_1", "2025-01-01", "")
    assert relations["confidence"][0] == ("fact_1", "supported", 0.91)
    return True


def test_neo4j_properties_include_compiled_memory_json():
    memory = fact_to_compiled_memory(sample_fact())
    props = compiled_memory_to_neo4j_properties(memory)

    assert props["memory_id"] == "fact_1"
    assert props["subject_type"] == "person"
    assert props["object_type"] == "organization"
    assert props["valid_from"] == "2025-01-01"
    assert props["confidence_level"] == "supported"
    assert props["decision_status"] == "proposed"
    assert json.loads(props["subject_aliases_json"]) == ["Alice"]
    assert json.loads(props["object_aliases_json"]) == ["Acme"]
    compiled = json.loads(props["compiled_memory_json"])
    assert compiled["predicate"] == "WORKS_AT"
    assert compiled["temporal"]["valid_from"] == "2025-01-01"
    assert props["document_id"] == "doc_1"
    assert props["extractor_model"] == "extractor-x"
    assert props["verifier_model"] == "verifier-x"
    assert props["run_id"] == "run_test"
    assert props["provenance_quality"] > 0.8
    return True


def test_weighted_confidence_and_context_ranking_use_evidence_quality():
    weak = {
        "subject": "A",
        "predicate": "RELATED_TO",
        "object": "B",
        "confidence": "supported",
        "fact_id": "weak",
        "example_id": "ex",
    }
    strong = {
        **weak,
        "object": "C",
        "fact_id": "strong",
        "support_text": "A is directly related to C in the cited sentence.",
        "question_relevance": "Directly answers the question.",
        "verification_reason": "Explicitly stated.",
        "provenance": [{
            "title": "doc",
            "sent_id": 2,
            "document_id": "doc",
            "sentence_id": "doc:2",
            "source_span_start": 0,
            "source_span_end": 49,
            "extractor_model": "extractor-x",
            "verifier_model": "verifier-x",
            "run_id": "run_test",
        }],
    }

    assert provenance_quality_score(strong) > provenance_quality_score(weak)
    assert evidence_strength_score(strong) > evidence_strength_score(weak)
    formatted = format_fact_rows_for_llm([weak, strong])
    assert formatted.splitlines()[0] == "[F1] A -RELATED_TO-> C"
    assert "confidence=" in formatted
    assert "provenance=" in formatted
    return True


def test_explicit_confidence_scores_are_normalized_to_unit_interval():
    fact = {
        "subject": "A",
        "predicate": "RELATED_TO",
        "object": "B",
        "confidence": "supported",
        "confidence_score": 91,
        "fact_id": "fact_score",
    }
    memory = fact_to_compiled_memory(fact)
    assert memory.confidence.score == 0.91
    props = compiled_memory_to_neo4j_properties(memory)
    assert props["confidence_score"] == 0.91
    return True


def test_weighted_confidence_score_is_not_scored_twice():
    fact = {
        "subject": "A",
        "predicate": "RELATED_TO",
        "object": "B",
        "confidence": "supported",
        "support_text": "A is directly related to B.",
        "provenance": [{
            "title": "doc",
            "sent_id": 0,
            "document_id": "doc",
            "sentence_id": "doc:0",
            "source_span_start": 0,
            "source_span_end": 27,
        }],
        "fact_id": "fact_weighted",
    }
    compiled = fact_to_compiled_fact(fact)

    assert compiled["confidence_method"] == "weighted_evidence_v1"
    assert evidence_strength_score(compiled) == compiled["confidence_score"]
    return True


def main():
    tests = [
        ("fact -> compiled memory", test_fact_to_compiled_memory_preserves_entities),
        ("compiled memory round trip", test_compiled_memory_round_trips_to_fact_contract),
        ("temporal overlap semantics", test_temporal_scope_overlap_semantics),
        ("Scallop relation projection", test_scallop_projection_contains_memory_relations),
        ("Neo4j compiled properties", test_neo4j_properties_include_compiled_memory_json),
        ("weighted confidence ranking", test_weighted_confidence_and_context_ranking_use_evidence_quality),
        ("confidence score normalization", test_explicit_confidence_scores_are_normalized_to_unit_interval),
        ("weighted confidence idempotence", test_weighted_confidence_score_is_not_scored_twice),
    ]
    passed = 0
    for name, fn in tests:
        print(f"Running {name}...")
        try:
            if fn():
                passed += 1
                print("  PASSED\n")
            else:
                print("  FAILED\n")
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
    print(f"Results: {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
