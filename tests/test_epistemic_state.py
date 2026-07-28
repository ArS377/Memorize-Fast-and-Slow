from __future__ import annotations

from neurosym.domain.epistemic_state import (
    EpistemicState,
    EpistemicStateTracker,
    consolidate,
    expand,
    state_distance,
)


def _search_response(*results):
    return {
        "kind": "search",
        "response": {
            "status": "ok",
            "request": {"query": "Where is Kalamang spoken?"},
            "results": list(results),
        },
    }


def _fact(fact_id: str, object_value: str, score: float = 0.8):
    return {
        "fact_id": fact_id,
        "subject": "Kalamang",
        "predicate": "SPOKEN_IN",
        "object": object_value,
        "support_text": f"Kalamang is spoken in {object_value}.",
        "score": score,
        "provenance": [{"document_id": "doc-1", "sent_id": 1}],
    }


def test_epistemic_state_expands_claims_and_is_deterministic() -> None:
    initial = EpistemicState(
        question="Where is Kalamang spoken?",
        choices={"A": "East Indonesia"},
    )
    evidence = _search_response(_fact("f1", "East Indonesia"))

    first = consolidate(expand(initial.clone(), evidence))
    second = consolidate(expand(initial.clone(), evidence))

    assert first.digest() == second.digest()
    assert first.nodes["claim:f1"].attributes["fact_id"] == "f1"
    assert first.nodes["claim:f1"].attributes["provenance"]
    assert first.successful_searches == 1
    assert state_distance(first, second) == 0.0


def test_consolidation_marks_conflicts_without_repeated_downweighting() -> None:
    state = EpistemicState(question="Where is Kalamang spoken?")
    state = expand(
        state,
        _search_response(
            _fact("f1", "East Indonesia", 0.9),
            _fact("f2", "West Indonesia", 0.6),
        ),
    )

    once = consolidate(state)
    confidence = once.nodes["claim:f2"].confidence
    twice = consolidate(once.clone())

    assert any(edge.edge_type == "contradicts" for edge in once.edges.values())
    assert confidence == 0.3
    assert twice.nodes["claim:f2"].confidence == confidence
    assert twice.digest() == once.digest()


def test_windowed_order_gap_stops_after_two_stable_completions() -> None:
    tracker = EpistemicStateTracker(
        question="Where is Kalamang spoken?",
        choices={"A": "East Indonesia"},
        epsilon=0.025,
        window=2,
        min_iterations=2,
    )
    tracker.observe(_search_response(), completion_boundary=False)

    first = tracker.observe(
        {"kind": "model", "candidate": "A"}, completion_boundary=True
    )
    second = tracker.observe(
        {"kind": "model", "candidate": "A"}, completion_boundary=True
    )

    assert first.order_gap == 0.0
    assert second.order_gap == 0.0
    assert tracker.window_mean == 0.0
    assert tracker.stable is True


def test_supported_answer_requires_consolidation_before_gap_settles() -> None:
    tracker = EpistemicStateTracker(
        question="Where is Kalamang spoken?",
        choices={"A": "East Indonesia"},
    )
    tracker.observe(
        _search_response(_fact("f1", "East Indonesia")), completion_boundary=False
    )
    tracker.observe(
        {
            "kind": "memory",
            "response": {
                "status": "ok",
                "artifact": {"selected_fact_ids": ["f1"], "excluded_fact_ids": []},
            },
        },
        completion_boundary=False,
    )

    first = tracker.observe(
        {"kind": "model", "candidate": "A", "cited_fact_ids": ["f1"]},
        completion_boundary=True,
    )
    tracker.observe(
        {"kind": "model", "candidate": "A", "cited_fact_ids": ["f1"]},
        completion_boundary=True,
    )
    tracker.observe(
        {"kind": "model", "candidate": "A", "cited_fact_ids": ["f1"]},
        completion_boundary=True,
    )

    assert first.order_gap > tracker.epsilon
    assert tracker.stable is True
    assert tracker.state.summary()["root_status"] == "resolved"
    assert tracker.state.committed_fact_ids == ["f1"]


def test_committed_fact_is_not_automatically_support_for_an_uncited_answer() -> None:
    tracker = EpistemicStateTracker(
        question="Where is Kalamang spoken?",
        choices={"A": "East Indonesia"},
    )
    tracker.observe(
        _search_response(_fact("f1", "East Indonesia")), completion_boundary=False
    )
    tracker.observe(
        {
            "kind": "memory",
            "response": {
                "status": "ok",
                "artifact": {"selected_fact_ids": ["f1"], "excluded_fact_ids": []},
            },
        },
        completion_boundary=False,
    )

    tracker.observe(
        {"kind": "model", "candidate": "A", "cited_fact_ids": []},
        completion_boundary=True,
    )

    assert tracker.state.summary()["root_status"] == "open"
    assert not any(
        edge.target == "answer:A" and edge.edge_type == "supports"
        for edge in tracker.state.edges.values()
    )


def test_prompt_view_prioritises_selected_claims_and_bounds_support_text() -> None:
    state = EpistemicState(question="Which fact matters?")
    state = expand(
        state,
        _search_response(
            *(
                {
                    **_fact(f"f{index:02d}", f"place-{index}", 0.5),
                    "support_text": "x" * 1_000,
                }
                for index in range(30)
            )
        ),
    )
    state.nodes["claim:f29"].attributes["selected"] = True

    view = state.prompt_view()

    assert len(view["claims"]) == 25
    assert view["claims"][0]["fact_id"] == "f29"
    assert len(view["claims"][0]["support_text"]) == 400
