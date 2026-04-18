#!/usr/bin/env python3
"""
Test Neo4j knowledge graph formatting and query generation.
"""
import json
import sys
from pathlib import Path

# Add parent directory to path to import pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
import longbench_kg_pipeline as pipe

def test_neo4j_query_generation():
    """Test that Neo4j queries are generated correctly."""
    
    # Create sample facts
    facts = [
        {
            "subject": "Kalamang",
            "predicate": "SPOKEN_IN",
            "object": "East Indonesia",
            "qualifiers": {"population": "130"},
            "provenance": [{"title": "test_doc", "sent_id": 0}],
            "support_text": "It is spoken by around 130 people in East Indonesia.",
            "question_relevance": "Geographic location information",
            "confidence": "supported",
            "normalization_notes": "Entity name normalized",
            "example_id": "test_1",
            "fact_id": "fact_123",
            "question": "Where is Kalamang spoken?",
            "verification_reason": "Explicitly stated in text"
        },
        {
            "subject": "Kalamang",
            "predicate": "HAS_ISO_CODE",
            "object": "kgv",
            "qualifiers": {},
            "provenance": [{"title": "test_doc", "sent_id": 1}],
            "support_text": "Kalamang (ISO 639-3 code kgv, glottocode kara1499)",
            "question_relevance": "Language identification",
            "confidence": "supported",
            "normalization_notes": "",
            "example_id": "test_1",
            "fact_id": "fact_124",
            "question": "Where is Kalamang spoken?",
            "verification_reason": "Direct citation"
        }
    ]
    
    # Mock Neo4j driver to capture queries
    queries_executed = []
    
    class MockSession:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            return None
        
        def run(self, query, **kwargs):
            queries_executed.append((query, kwargs))
            return None
    
    class MockDriver:
        def session(self):
            return MockSession()
    
    # Create pipeline config with Neo4j
    config = pipe.PipelineConfig(
        input_path=Path("test_input.jsonl"),
        output_path=Path("test_output.jsonl"),
        model="test-model",
        vllm_base_url="http://localhost:8000/v1",
        api_key="test",
        temperature=0.0,
        max_tokens=512,
        chunk_chars=8000,
        max_chunks_per_example=None,
        limit=None,
        sleep_seconds=0.0,
        use_json_mode=False,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )
    
    pipeline = pipe.LongBenchKGPipeline(config)
    pipeline.neo4j_driver = MockDriver()
    
    # Test insertion
    pipeline.insert_facts_neo4j(facts)
    
    # Validate queries
    assert len(queries_executed) == len(facts), f"Expected {len(facts)} queries, got {len(queries_executed)}"
    
    for i, (query, kwargs) in enumerate(queries_executed):
        fact = facts[i]
        
        # Check query structure
        assert "MERGE (s:Entity {name: $subject})" in query
        assert "MERGE (o:Entity {name: $object})" in query
        assert "MERGE (s)-[r:" in query
        assert "SET r.example_id = $example_id" in query
        assert "r.question = $question" in query
        assert "r.support_text = $support_text" in query
        assert "r.provenance_json = $provenance_json" in query
        assert "r.qualifiers_json = $qualifiers_json" in query
        assert "r.question_relevance = $question_relevance" in query
        assert "r.confidence = $confidence" in query
        assert "r.normalization_notes = $normalization_notes" in query
        assert "r.verification_reason = $verification_reason" in query
        
        # Check parameters
        assert kwargs["subject"] == fact["subject"]
        assert kwargs["object"] == fact["object"]
        assert kwargs["fact_id"] == fact["fact_id"]
        assert kwargs["example_id"] == fact["example_id"]
        assert kwargs["question"] == fact["question"]
        assert kwargs["support_text"] == fact["support_text"]
        assert kwargs["question_relevance"] == fact["question_relevance"]
        assert kwargs["confidence"] == fact["confidence"]
        assert kwargs["normalization_notes"] == fact["normalization_notes"]
        assert kwargs["verification_reason"] == fact["verification_reason"]
        
        # Check JSON fields
        provenance_json = json.loads(kwargs["provenance_json"])
        assert provenance_json == fact["provenance"]
        
        qualifiers_json = json.loads(kwargs["qualifiers_json"])
        assert qualifiers_json == fact["qualifiers"]
        
        # Check relationship type (should be sanitized predicate)
        expected_rel_type = pipe.sanitize_predicate(fact["predicate"])
        assert f"[r:`{expected_rel_type}`" in query
    
    print("Neo4j query generation test passed.")
    return True

def test_predicate_sanitization_in_neo4j():
    """Test that predicates are properly sanitized for Neo4j relationship types."""
    
    test_cases = [
        ("spoken in", "SPOKEN_IN"),
        ("has-part", "HAS_PART"),
        ("IS A", "IS_A"),
        ("related to", "RELATED_TO"),
        ("has@property", "HAS_PROPERTY"),
        ("multiple___underscores", "MULTIPLE_UNDERSCORES"),
        ("123 test", "123_TEST"),
        ("", "RELATED_TO")  # Empty predicate defaults to RELATED_TO
    ]
    
    for input_pred, expected_rel in test_cases:
        sanitized = pipe.sanitize_predicate(input_pred)
        assert sanitized == expected_rel, f"Predicate '{input_pred}' sanitized to '{sanitized}', expected '{expected_rel}'"
        
        # Test that this would be valid in Neo4j query
        query_part = f"[r:`{sanitized}` {{fact_id: $fact_id}}]"
        assert "`" in query_part and sanitized in query_part
    
    print("Predicate sanitization for Neo4j test passed.")
    return True

def test_fact_id_generation():
    """Test that fact IDs are generated consistently."""
    
    fact = {
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "East Indonesia",
        "support_text": "It is spoken by around 130 people in East Indonesia.",
        "provenance": [{"title": "test", "sent_id": 0}]
    }
    
    example_id = "test_1"
    
    # Generate ID twice, should be same
    id1 = pipe.make_fact_id(example_id, fact)
    id2 = pipe.make_fact_id(example_id, fact)
    
    assert id1 == id2, "Fact ID generation should be deterministic"
    assert len(id1) == 24, "Fact ID should be 24 characters long"
    assert all(c in "0123456789abcdef" for c in id1), "Fact ID should be hexadecimal"
    
    # Different facts should have different IDs
    fact2 = fact.copy()
    fact2["object"] = "West Indonesia"
    id3 = pipe.make_fact_id(example_id, fact2)
    assert id1 != id3, "Different facts should have different IDs"
    
    print("Fact ID generation test passed.")
    return True

def main():
    """Run all Neo4j formatting tests."""
    print("Testing Neo4j knowledge graph formatting...\n")
    
    tests = [
        ("Neo4j query generation", test_neo4j_query_generation),
        ("Predicate sanitization in Neo4j", test_predicate_sanitization_in_neo4j),
        ("Fact ID generation", test_fact_id_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"  PASSED\n")
            else:
                print(f"  FAILED\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All Neo4j formatting tests PASSED!")
        return True
    else:
        print("Some tests FAILED.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
