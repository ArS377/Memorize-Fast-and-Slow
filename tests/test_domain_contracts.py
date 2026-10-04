from neurosym.domain import Fact, RetrievalRequest, RetrievalResult, Scope


def test_fact_mapping_preserves_unknown_extension_fields_losslessly() -> None:
    raw = {
        "fact_id": "fact-1",
        "session_id": "session-1",
        "subject": "A",
        "predicate": "RELATED_TO",
        "object": "B",
        "vendor_extension": {"nested": [1, 2, 3]},
    }

    assert Fact.from_mapping(raw).to_mapping() == raw


def test_nested_retrieval_contract_preserves_metadata_extensions() -> None:
    raw = {
        "rows": [{"fact_id": "fact-1", "extension": True}],
        "metadata": {
            "configured_mode": "hybrid",
            "effective_mode": "sparse",
            "degraded": True,
            "fusion_diagnostic": {"rrf": 60},
        },
    }

    result = RetrievalResult.from_mapping(raw)

    assert result.to_mapping() == raw
    assert result.metadata["fusion_diagnostic"] == {"rrf": 60}


def test_retrieval_request_hydrates_trusted_scope() -> None:
    request = RetrievalRequest.from_mapping(
        {
            "query": "A",
            "scope": {
                "memory_scope": "example",
                "session_ids": ["session-1"],
                "example_id": "example-1",
            },
        }
    )

    assert isinstance(request.scope, Scope)
    assert request.scope.session_ids == ("session-1",)
