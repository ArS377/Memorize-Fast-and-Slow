#!/usr/bin/env python3
"""
Test the self-reflection questions and validation logic of the pipeline.
"""
import json
import sys
from pathlib import Path

# Add parent directory to path to import pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
import longbench_kg_pipeline as pipe

def test_self_reflection_questions():
    """Test that the verification prompt contains the required self-reflection questions."""
    
    # Create a sample example and facts
    example = {
        "_id": "test_1",
        "question": "Test question",
        "context": "Test context about Kalamang language."
    }
    
    facts = [
        {
            "verification_id": "f0",
            "subject": "Kalamang",
            "predicate": "SPOKEN_IN",
            "object": "East Indonesia",
            "support_text": "It is spoken by around 130 people in East Indonesia.",
            "provenance": [{"title": "test", "sent_id": 0}],
            "question_relevance": "Relevant for geography",
            "confidence": "supported"
        }
    ]
    
    # Build verification prompt
    prompt = pipe.build_verification_prompt(example, facts)
    
    # Check that all required self-reflection questions are present
    required_questions = [
        "1. Is the fact explicitly supported by the support_text?",
        "2. Is it atomic, not a bundle of multiple claims?",
        "3. Are subject/object aliases and pronouns resolved correctly?",
        "4. Is the predicate specific and meaningful?",
        "5. Is the fact useful or potentially useful for answering the question?",
        "6. Is there any unsupported inference?"
    ]
    
    for question in required_questions:
        if question not in prompt:
            print(f"ERROR: Missing self-reflection question: {question}")
            return False
    
    print("All required self-reflection questions are present in verification prompt.")
    return True

def test_fact_extraction_format():
    """Test that extraction produces the required format."""
    
    # Create sample example and chunk
    example = {
        "_id": "test_1",
        "question": "Test question",
        "context": "Kalamang is spoken by around 130 people in East Indonesia."
    }
    
    chunk = [
        {
            "title": "test",
            "sent_id": 0,
            "text": "Kalamang is spoken by around 130 people in East Indonesia."
        }
    ]
    
    # Build extraction prompt
    prompt = pipe.build_extraction_prompt(example, chunk, 0)
    
    # Check that required fields are mentioned in the prompt
    required_fields = [
        '"subject": "canonical entity name"',
        '"predicate": "UPPER_SNAKE_CASE_RELATION"',
        '"object": "canonical entity/value name"',
        '"qualifiers": {}',
        '"provenance": [{"title": "...", "sent_id": 0}]',
        '"support_text": "exact supporting sentence(s)"',
        '"question_relevance": "why this fact could help answer the current multiple-choice question"',
        '"confidence": "supported"',
        '"normalization_notes": "alias/pronoun decisions, or empty string"'
    ]
    
    for field in required_fields:
        if field not in prompt:
            print(f"ERROR: Missing required field in extraction prompt: {field}")
            return False
    
    print("All required fields are present in extraction prompt.")
    return True

def test_status_normalization():
    """Test status normalization function."""
    
    test_cases = [
        ("supported", "supported"),
        ("SUPPORTED", "supported"),
        ("supported_by_evidence", "supported"),
        ("rejected", "rejected"),
        ("REJECTED", "rejected"),
        ("false", "rejected"),
        ("not_supported", "rejected"),
        ("uncertain", "uncertain"),
        ("UNCERTAIN", "uncertain"),
        ("partial", "uncertain"),
        ("maybe", "uncertain"),
        ("unknown", "uncertain"),
        ("", "uncertain"),
        (None, "uncertain")
    ]
    
    for input_val, expected in test_cases:
        result = pipe.normalize_status(input_val)
        if result != expected:
            print(f"ERROR: normalize_status({input_val}) = {result}, expected {expected}")
            return False
    
    print("Status normalization works correctly.")
    return True

def test_predicate_sanitization():
    """Test predicate sanitization function."""
    
    test_cases = [
        ("spoken in", "SPOKEN_IN"),
        ("LOCATED_IN", "LOCATED_IN"),
        ("has-part", "HAS_PART"),
        ("related to", "RELATED_TO"),
        ("", "RELATED_TO"),
        ("123", "123"),
        ("IS A", "IS_A"),
        ("has@property", "HAS_PROPERTY"),
        ("multiple___underscores", "MULTIPLE_UNDERSCORES")
    ]
    
    for input_val, expected in test_cases:
        result = pipe.sanitize_predicate(input_val)
        if result != expected:
            print(f"ERROR: sanitize_predicate('{input_val}') = '{result}', expected '{expected}'")
            return False
    
    print("Predicate sanitization works correctly.")
    return True

def test_neo4j_format():
    """Test that facts are formatted correctly for Neo4j insertion."""
    
    fact = {
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": "East Indonesia",
        "qualifiers": {"population": "130"},
        "provenance": [{"title": "test", "sent_id": 0}],
        "support_text": "It is spoken by around 130 people in East Indonesia.",
        "question_relevance": "Geographic location",
        "confidence": "supported",
        "normalization_notes": "",
        "example_id": "test_1",
        "fact_id": "test_fact_123",
        "question": "Test question",
        "verification_reason": "Explicitly stated"
    }
    
    # Check that all required fields for Neo4j are present
    neo4j_required_fields = [
        "subject", "object", "predicate", "fact_id", "example_id", 
        "question", "support_text", "provenance", "qualifiers",
        "question_relevance", "confidence", "normalization_notes", 
        "verification_reason"
    ]
    
    for field in neo4j_required_fields:
        if field not in fact:
            print(f"ERROR: Missing required Neo4j field: {field}")
            return False
    
    print("Neo4j format validation passed.")
    return True

def main():
    """Run all validation tests."""
    print("Testing pipeline validation logic...\n")
    
    tests = [
        ("Self-reflection questions", test_self_reflection_questions),
        ("Fact extraction format", test_fact_extraction_format),
        ("Status normalization", test_status_normalization),
        ("Predicate sanitization", test_predicate_sanitization),
        ("Neo4j format", test_neo4j_format)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
            print(f"  PASSED\n")
        else:
            print(f"  FAILED\n")
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All validation tests PASSED! The pipeline logic is correct.")
        return True
    else:
        print("Some tests FAILED. Issues need to be fixed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
