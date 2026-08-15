from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import experiments.synthetic_temporal_baselines as temporal_baselines
from experiments.synthetic_temporal_preferences import (
    _aliases,
    generate_dataset,
    main as generate_main,
    materialize_event_states,
    preference_rule_parameters,
    resolve_preference,
    resolve_query,
)
from neurosym.adapters.scallop import validate_update_detailed
from experiments.synthetic_temporal_baselines import (
    answer_bm25_raw_events,
    answer_dense_raw_events,
    candidate_decision_accept_all,
    evaluate,
    evaluate_bm25_raw_events,
    evaluate_bm25_replayed_candidates,
    evaluate_dense_raw_events,
    evaluate_dense_replayed_candidates,
    evaluate_replayed_candidates,
    replay_candidates,
    render_full_transcript,
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FakeDenseEmbedder:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [[1.0, 0.0] if "value-001-b" in text else [0.0, 1.0] for text in texts]

    def encode_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0]

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "fake/bge",
            "requested_revision": "test-revision",
            "resolved_revision": "test-commit",
            "sentence_transformers_version": "fake-1",
            "vector_dimension": 2,
        }


class BatchFakeDenseEmbedder(FakeDenseEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.query_batch_calls: list[list[str]] = []

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.query_batch_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class StaleCandidateFirstDenseEmbedder(FakeDenseEmbedder):
    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [
            [2.0, 0.0] if "stale-replay-marker" in text
            else [1.0, 0.0] if "current-replay-marker" in text
            else [0.0, 1.0]
            for text in texts
        ]


def test_generator_is_deterministic_and_emits_existing_fact_shape(tmp_path: Path) -> None:
    first = generate_dataset(tmp_path / "first", history_count=3)
    second = generate_dataset(tmp_path / "second", history_count=3)

    assert first["facts"].read_bytes() == second["facts"].read_bytes()
    assert first["examples"].read_bytes() == second["examples"].read_bytes()
    assert first["candidates"].read_bytes() == second["candidates"].read_bytes()
    assert first["events"].read_bytes() == second["events"].read_bytes()
    assert first["queries"].read_bytes() == second["queries"].read_bytes()
    facts = _rows(first["facts"])
    assert facts
    assert all(
        fact["fact_id"]
        and fact["subject"]
        and fact["predicate"]
        and fact["object"]
        and fact["support_text"]
        and fact["provenance"]
        and fact["temporal"]["valid_from"]
        for fact in facts
    )
    assert {fact["predicate"] for fact in facts} >= {
        "PREFERS",
        "AVOIDS",
        "AMBIGUOUS_PREFERENCE",
        "PRIVATE_NOTE",
    }
    assert {fact["subject"] for fact in facts} >= {
        "subject-001", "subject-002", "subject-003"
    }


def test_anti_shortcut_profile_is_deterministic_and_composition_held_out(
    tmp_path: Path,
) -> None:
    first = generate_dataset(
        tmp_path / "first",
        history_count=12,
        split_counts=(4, 4, 4),
        hardness_profile="anti_shortcut_stream_v2",
    )
    second = generate_dataset(
        tmp_path / "second",
        history_count=12,
        split_counts=(4, 4, 4),
        hardness_profile="anti_shortcut_stream_v2",
    )
    assert first["events"].read_bytes() == second["events"].read_bytes()
    assert first["queries"].read_bytes() == second["queries"].read_bytes()
    queries = [
        query for query in _rows(first["queries"])
        if query["kind"] == "private_lineage"
    ]
    train_compositions = {
        query["composition"]["id"] for query in queries if query["split"] != "test"
    }
    test_compositions = {
        query["composition"]["id"] for query in queries if query["split"] == "test"
    }
    assert train_compositions
    assert test_compositions
    assert train_compositions.isdisjoint(test_compositions)
    aliases_by_history = {
        query["history_id"]: query["query_text"]
        for query in queries
        if query["query_id"].endswith("-positive")
    }
    assert len(set(aliases_by_history.values())) == len(aliases_by_history)
    events_by_history = {
        history_id: [
            event
            for event in _rows(first["events"])
            if event["history_id"] == history_id
        ]
        for history_id in {query["history_id"] for query in queries}
    }
    for query in queries:
        events = events_by_history[query["history_id"]]
        axes = query["composition"]["axes"]
        retract = next(
            event for event in events if event["event_id"].endswith("-lineage-retract")
        )
        expected_suffix = "-lineage-copy-2" if axes["chain_variant"] == 0 else "-lineage-root"
        assert retract["retracts"].endswith(expected_suffix)
        negative_text = " ".join(
            event["model_text"]
            for event in events
            if event["event_family"] == "same_entity_hard_negative"
        )
        expected_marker = "discussed" if axes["negative_family"] == 0 else "planning card"
        assert expected_marker in negative_text.lower()


def test_interleaved_v3_preserves_typed_context_and_conflict_relations(
    tmp_path: Path,
) -> None:
    paths = generate_dataset(
        tmp_path,
        history_count=1,
        split_counts=(0, 0, 1),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = _rows(paths["events"])
    candidates = _rows(paths["candidates"])
    facts = _rows(paths["facts"])
    source_documents = {
        document["document_id"]: document["text"]
        for document in _rows(paths["source_documents"])
    }
    assert [event["sequence_index"] for event in events] == list(range(len(events)))
    event_fact_ids = {event["fact"]["fact_id"] for event in events}
    assert event_fact_ids <= {fact["fact_id"] for fact in facts}
    by_suffix = {
        event["event_id"].removeprefix("history-001-"): event for event in events
    }

    context_families = {
        "alias_bridge",
        "same_entity_hard_negative",
        "delayed_preference_probe",
    }
    assert all(
        event["fact"]["predicate"] != "PREFERS"
        for event in events
        if event["event_family"] in context_families
    )
    assert by_suffix["constraint"]["fact"]["predicate"] == "AVOIDS"
    assert by_suffix["ambiguity"]["fact"]["predicate"] == "AMBIGUOUS_PREFERENCE"
    assert by_suffix["private-add"]["fact"]["predicate"] == "PRIVATE_NOTE"
    assert all(
        event.get("surface_subject") == "AsterArc"
        for event in events
        if event["fact"]["subject"] == "subject-001"
    )
    assert set(by_suffix["direct-correction"]["resolves"]) == {
        "history-001-indirect-source",
    }
    assert by_suffix["transition"]["fact"]["temporal"]["valid_to"] == "2025-09-30"
    assert by_suffix["transition"]["transitions_from"] == "history-001-initial"
    assert "supersedes" not in by_suffix["transition"]
    assert by_suffix["transition"]["fact"]["qualifiers"]["source_authority"] == (
        "direct_user"
    )
    assert by_suffix["conflict-right"]["conflicts_with"] == (
        "history-001-conflict-left"
    )
    assert by_suffix["lineage-retract"]["retracts_lineage"] is True
    assert set(by_suffix["conflict-resolution"]["resolves"]) == {
        "history-001-conflict-left",
        "history-001-conflict-right",
    }
    assert by_suffix["conflict-left"]["fact"]["temporal"]["observed_at"].startswith(
        "2027-01-01"
    )
    assert by_suffix["conflict-right"]["fact"]["temporal"]["observed_at"].startswith(
        "2027-02-01"
    )
    assert by_suffix["conflict-resolution"]["fact"]["temporal"]["valid_from"] == (
        "2027-03-01"
    )
    checked_candidates = {
        "history-001-overlap-replacement",
        "history-001-direct-conflict-no-supersession",
        "history-001-ambiguity-resolution",
    }
    for candidate in candidates:
        if candidate["candidate_id"] in checked_candidates:
            assert candidate["fact"]["object"] in candidate["fact"]["support_text"]
    constraint_violation = next(
        candidate
        for candidate in candidates
        if candidate["candidate_id"] == "history-001-hard-constraint-violation"
    )
    assert "requested" in constraint_violation["fact"]["support_text"]
    resurrection = next(
        candidate
        for candidate in candidates
        if candidate["candidate_id"] == "history-001-retraction-resurrection"
    )
    assert resurrection["fact"]["predicate"] == "PRIVATE_NOTE"
    direct_conflict = next(
        candidate
        for candidate in candidates
        if candidate["candidate_id"] == "history-001-direct-conflict-no-supersession"
    )
    assert direct_conflict["candidate_features"]["active_conflict_count"] == 1
    all_facts = [
        *[event["fact"] for event in events],
        *[candidate["fact"] for candidate in candidates],
    ]
    for fact in all_facts:
        source = fact["provenance"][0]
        text = source_documents[source["document_id"]]
        assert text[source["source_span_start"] : source["source_span_end"]] == (
            fact["support_text"]
        )


def test_aliases_remain_disjoint_at_full_interleaved_scale() -> None:
    split_counts = (88, 88, 1024)
    boundaries = (split_counts[0], split_counts[0] + split_counts[1])
    all_aliases = [alias for index in range(1, 1201) for alias in _aliases(index)]
    aliases = [_aliases(index)[0] for index in range(1, sum(split_counts) + 1)]
    train = set(aliases[: boundaries[0]])
    dev = set(aliases[boundaries[0] : boundaries[1]])
    test = set(aliases[boundaries[1] :])

    assert len(set(all_aliases)) == len(all_aliases)
    assert train.isdisjoint(dev)
    assert train.isdisjoint(test)
    assert dev.isdisjoint(test)


def test_anti_shortcut_private_lineage_has_natural_text_and_transitive_gold(
    tmp_path: Path,
) -> None:
    paths = generate_dataset(
        tmp_path,
        history_count=3,
        split_counts=(1, 1, 1),
        hardness_profile="anti_shortcut_stream_v2",
    )
    events = _rows(paths["events"])
    queries = _rows(paths["queries"])
    history_events = [event for event in events if event["history_id"] == "history-003"]
    lineage_queries = [
        query
        for query in queries
        if query["history_id"] == "history-003" and query["kind"] == "private_lineage"
    ]

    assert len(history_events) == 23
    assert len(lineage_queries) == 2
    positive = next(
        query for query in lineage_queries if query["query_id"].endswith("-positive")
    )
    final_query = next(
        query for query in lineage_queries if not query["query_id"].endswith("-positive")
    )
    assert positive["gold"] == "cedar glass 003"
    assert final_query["gold"] == "UNKNOWN"
    copy_two_index = next(
        index
        for index, event in enumerate(history_events)
        if event["event_id"].endswith("-lineage-copy-2")
    )
    assert resolve_query(history_events[: copy_two_index + 1], positive) == "cedar glass 003"
    forbidden = ("role=", "task_id=", "thread_id=", "event_id=", "operation=")
    assert all(event.get("model_text") for event in history_events)
    assert all(not any(token in event["model_text"] for token in forbidden) for event in history_events)
    assert all("subject-003" not in event["model_text"] for event in history_events)
    assert all("subject-003" not in query["query_text"] for query in queries if query["history_id"] == "history-003")
    assert all("history-003" not in query["query_text"] for query in queries if query["history_id"] == "history-003")
    assert "subject-003" not in final_query["query_text"]
    assert "history-003" not in final_query["query_text"]
    assert resolve_query(history_events, final_query) == "UNKNOWN"


def test_paraphrase_condition_preserves_latent_state_and_gold_answers(tmp_path: Path) -> None:
    lexical_paths = generate_dataset(tmp_path / "lexical", history_count=2, condition="lexical")
    paraphrase_paths = generate_dataset(tmp_path / "paraphrase", history_count=2, condition="paraphrase")
    repeat_paths = generate_dataset(tmp_path / "paraphrase-repeat", history_count=2, condition="paraphrase")
    lexical_events = _rows(lexical_paths["events"])
    paraphrase_events = _rows(paraphrase_paths["events"])
    lexical_queries = _rows(lexical_paths["queries"])
    paraphrase_queries = _rows(paraphrase_paths["queries"])
    lexical_candidates = _rows(lexical_paths["candidates"])
    paraphrase_candidates = _rows(paraphrase_paths["candidates"])

    def latent_event(event: dict[str, Any]) -> dict[str, Any]:
        fact = event["fact"]
        return {
            key: value for key, value in event.items() if key != "fact"
        } | {
            "fact": {
                key: fact[key]
                for key in (
                    "fact_id", "session_id", "example_id", "subject", "predicate", "object",
                    "temporal", "qualifiers", "confidence_score", "scope", "history_id",
                )
                if key in fact
            }
            }

    def latent_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        fact = candidate["fact"]
        return {
            key: value for key, value in candidate.items() if key != "fact"
        } | {
            "fact": {
                key: fact[key]
                for key in (
                    "fact_id", "session_id", "example_id", "subject", "predicate", "object",
                    "temporal", "qualifiers", "confidence_score", "scope", "history_id", "split",
                )
                if key in fact
            }
        }

    assert [latent_event(event) for event in lexical_events] == [
        latent_event(event) for event in paraphrase_events
    ]
    assert [query["gold"] for query in lexical_queries] == [query["gold"] for query in paraphrase_queries]
    assert [
        {key: value for key, value in query.items() if key != "query_text"}
        for query in lexical_queries
    ] == [
        {key: value for key, value in query.items() if key != "query_text"}
        for query in paraphrase_queries
    ]
    assert [event["fact"]["support_text"] for event in lexical_events] != [
        event["fact"]["support_text"] for event in paraphrase_events
    ]
    assert [latent_candidate(candidate) for candidate in lexical_candidates] == [
        latent_candidate(candidate) for candidate in paraphrase_candidates
    ]
    assert [candidate["fact"]["support_text"] for candidate in lexical_candidates] != [
        candidate["fact"]["support_text"] for candidate in paraphrase_candidates
    ]
    assert [query["query_text"] for query in lexical_queries] != [
        query["query_text"] for query in paraphrase_queries
    ]
    assert all(query["gold"] not in query["query_text"] for query in paraphrase_queries)
    assert paraphrase_paths["events"].read_bytes() == repeat_paths["events"].read_bytes()
    assert paraphrase_paths["queries"].read_bytes() == repeat_paths["queries"].read_bytes()


def test_paraphrase_query_and_support_use_distinct_natural_language(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, condition="paraphrase")
    events = _rows(paths["events"])
    queries = _rows(paths["queries"])
    current = next(event for event in events if event["event_id"] == "history-001-transition")
    query = next(item for item in queries if item["query_id"] == "history-001-current")

    support_tokens = set(temporal_baselines._tokenize(current["fact"]["support_text"]))
    query_tokens = set(temporal_baselines._tokenize(query["query_text"]))

    assert {"standing", "selection", "became", "indefinitely"}.isdisjoint(query_tokens)
    assert {"option", "governs"}.isdisjoint(support_tokens)
    assert query["gold"] not in query["query_text"]


def test_generator_cli_accepts_paraphrase_condition(tmp_path: Path) -> None:
    generate_main(["--output-dir", str(tmp_path), "--condition", "paraphrase"])

    query = _rows(tmp_path / "queries.jsonl")[0]
    assert query["query_text"].startswith("Which option governs")


def test_generator_produces_independent_histories_and_all_event_families(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, history_count=3)
    rows_by_file = {name: _rows(path) for name, path in paths.items()}

    expected_history_ids = {"history-001", "history-002", "history-003"}
    for name, rows in rows_by_file.items():
        assert {row["history_id"] for row in rows} == expected_history_ids, name
    assert len({event["event_id"] for event in rows_by_file["events"]}) == 39
    assert {event["event_family"] for event in rows_by_file["events"]} == {
        "non_overlap_transition",
        "scope_exception",
        "hard_constraint",
        "ambiguity",
        "retraction",
        "backdated_correction",
        "duplicate_delivery",
        "overlapping_replacement",
        "source_authority_conflict",
        "scope_leakage",
    }
    assert {
        candidate["event_family"]
        for candidate in rows_by_file["candidates"]
    } == {
        "low_evidence_conflicting_reject",
        "uncontested_accept",
        "high_evidence_conflicting_replace",
        "equal_evidence_conflict_reject",
        "hard_constraint_violation_reject",
        "retraction_tombstone_resurrection_reject",
        "direct_user_correction_with_valid_supersedes_replacement",
        "direct_user_conflict_lacking_supersession_reject",
        "ambiguity_resolving_direct_correction",
    }


def test_candidate_families_include_observable_features_and_admission_labels(tmp_path: Path) -> None:
    candidates = _rows(generate_dataset(tmp_path)["candidates"])
    by_family = {candidate["event_family"]: candidate for candidate in candidates}

    expected_labels = {
        "low_evidence_conflicting_reject": (True, "reject", "low_evidence_conflict"),
        "high_evidence_conflicting_replace": (True, "replace", "higher_evidence_conflict"),
        "equal_evidence_conflict_reject": (True, "reject", "equal_evidence_conflict"),
        "hard_constraint_violation_reject": (False, "reject", "hard_constraint_violation"),
        "retraction_tombstone_resurrection_reject": (False, "reject", "retraction_tombstone"),
        "direct_user_correction_with_valid_supersedes_replacement": (
            True, "replace", "direct_user_supersedes_conflict",
        ),
        "direct_user_conflict_lacking_supersession_reject": (
            False, "reject", "direct_user_conflict_without_supersedes",
        ),
        "ambiguity_resolving_direct_correction": (True, "accept", "direct_user_resolves_ambiguity"),
    }
    assert set(expected_labels).issubset(by_family)
    for family, (hard_gate, soft_label, reason_code) in expected_labels.items():
        candidate = by_family[family]
        assert candidate["gold_hard_gate"] is hard_gate
        assert candidate["gold_soft_label"] == candidate["gold_decision"] == soft_label
        assert candidate["gold_reason_code"] == reason_code
        assert all("gold" not in key.lower() for key in candidate["candidate_features"])
    assert by_family["equal_evidence_conflict_reject"]["candidate_features"]["equal_evidence_conflict"]
    high_evidence = by_family["high_evidence_conflicting_replace"]
    assert high_evidence["fact"]["qualifiers"]["source_authority"] == "inferred"
    assert high_evidence["candidate_features"]["candidate_has_supersedes"] is False
    assert (
        high_evidence["candidate_features"]["candidate_confidence_score"]
        > high_evidence["candidate_features"]["max_active_conflict_confidence_score"]
    )
    low_evidence = by_family["low_evidence_conflicting_reject"]
    non_confidence_keys = {
        "candidate_source_authority",
        "candidate_has_supersedes",
        "active_conflict_count",
        "equal_evidence_conflict",
        "violates_hard_constraint",
        "targets_retracted_fact",
        "resolves_ambiguity",
    }
    assert {
        key: high_evidence["candidate_features"][key] for key in non_confidence_keys
    } == {
        key: low_evidence["candidate_features"][key] for key in non_confidence_keys
    }
    assert (
        low_evidence["candidate_features"]["candidate_confidence_score"]
        < low_evidence["candidate_features"]["max_active_conflict_confidence_score"]
    )
    assert by_family["hard_constraint_violation_reject"]["candidate_features"]["violates_hard_constraint"]
    assert by_family["retraction_tombstone_resurrection_reject"]["candidate_features"]["targets_retracted_fact"]
    assert by_family["ambiguity_resolving_direct_correction"]["candidate_features"]["resolves_ambiguity"]


def test_history_splits_are_deterministic_and_disjoint_across_artifacts(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, history_count=6, split_counts=(2, 2, 2))
    rows_by_file = {name: _rows(path) for name, path in paths.items()}
    split_by_history: dict[str, str] = {}

    for name, rows in rows_by_file.items():
        for row in rows:
            prior = split_by_history.setdefault(row["history_id"], row["split"])
            assert prior == row["split"], name

    assert set(split_by_history.values()) == {"train", "dev", "test"}
    assert {split: list(split_by_history.values()).count(split) for split in set(split_by_history.values())} == {
        "train": 2,
        "dev": 2,
        "test": 2,
    }


def test_preference_revisions_and_corruptions_exercise_validator(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path)
    facts = _rows(paths["facts"])
    candidates = _rows(paths["candidates"])
    initial = next(fact for fact in facts if fact["fact_id"] == "history-001-initial")
    current = next(fact for fact in facts if fact["fact_id"] == "history-001-current")
    stale = next(candidate for candidate in candidates if candidate["candidate_id"] == "history-001-stale")
    valid = next(candidate for candidate in candidates if candidate["gold_decision"] == "accept")

    revision = validate_update_detailed([initial], current, preference_rule_parameters())
    corruption = validate_update_detailed([current], stale["fact"], preference_rule_parameters())
    unrelated = validate_update_detailed([initial, current], valid["fact"], preference_rule_parameters())

    assert revision.decision == "accept"
    assert corruption.decision == "reject"
    assert corruption.rejection_label is not None
    assert corruption.rejection_label.code == "functional_conflict"
    assert unrelated.decision == "accept"


def test_examples_include_gold_operations_and_short_answer_queries(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, history_count=2)
    examples = _rows(paths["examples"])
    candidates = _rows(paths["candidates"])
    events = _rows(paths["events"])

    assert {example["gold_operation"] for example in examples} == {
        "accept_transition", "scoped_preference", "backdated_correction",
        "idempotent_duplicate", "replace_overlapping_candidate",
        "direct_user_authority", "prevent_scope_leakage",
    }
    assert all(example["_id"] and example["context"] and example["question"] and example["answer"] for example in examples)
    assert {candidate["gold_decision"] for candidate in candidates} == {"reject", "accept", "replace"}
    assert {event["operation"] for event in events} == {
        "add", "supersede", "temporary_exception", "hard_constraint", "ambiguous_conflict", "retract",
        "backdated_correction", "duplicate_delivery", "direct_user_correction",
    }
    assert all(event["session_id"] and isinstance(event["turn_index"], int) for event in events)


def test_query_gold_is_derived_from_each_history_event_trace(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, history_count=2)
    events = _rows(paths["events"])
    queries = _rows(paths["queries"])

    for query in queries:
        history_events = [event for event in events if event["history_id"] == query["history_id"]]
        assert resolve_query(history_events, query) == query["gold"]

    first_history_events = [event for event in events if event["history_id"] == "history-001"]
    assert resolve_preference(first_history_events, "subject-001", "2025-08-01") == "value-001-b"
    assert resolve_preference(first_history_events, "subject-001", "2025-08-05", "scope-001") == "value-001-c"


def test_new_event_families_have_expected_gold_state_and_operations(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    queries = {query["query_id"]: query for query in _rows(paths["queries"])}
    candidates = {candidate["candidate_id"]: candidate for candidate in _rows(paths["candidates"])}
    examples = {example["gold_operation"]: example for example in _rows(paths["examples"])}
    history_events = [event for event in events if event["history_id"] == "history-001"]

    assert resolve_query(history_events, queries["history-001-backdated"]) == "value-001-h"
    assert resolve_query(history_events, queries["history-001-duplicate"]) == "value-001-b"
    assert resolve_query(history_events, queries["history-001-authority"]) == "value-001-k"
    assert resolve_query(history_events, queries["history-001-scope-leakage"]) == "value-001-b"
    backdated_index = next(
        index for index, event in enumerate(history_events)
        if event["event_id"] == "history-001-backdated"
    )
    transition_index = next(
        index for index, event in enumerate(history_events)
        if event["event_id"] == "history-001-transition"
    )
    duplicate_index = next(
        index for index, event in enumerate(history_events)
        if event["event_id"] == "history-001-duplicate"
    )
    snapshots = materialize_event_states(history_events)
    assert backdated_index > transition_index
    assert history_events[backdated_index]["fact"]["temporal"] == {
        "valid_from": "2025-03-01",
        "valid_to": "2025-06-30",
        "observed_at": "2025-08-20T00:00:00+00:00",
    }
    assert snapshots[duplicate_index] == snapshots[duplicate_index - 1]
    assert candidates["history-001-overlap-replacement"]["gold_decision"] == "replace"
    assert candidates["history-001-overlap-replacement"]["gold_replace_fact_id"] == "history-001-replaceable"
    gold_fact_ids = {operation: example["gold_fact_ids"] for operation, example in examples.items()}
    for operation, fact_ids in {
        "backdated_correction": ["history-001-backdated"],
        "idempotent_duplicate": ["history-001-current"],
        "replace_overlapping_candidate": ["history-001-overlap-replacement"],
        "direct_user_authority": ["history-001-direct-correction"],
        "prevent_scope_leakage": ["history-001-current"],
    }.items():
        assert gold_fact_ids[operation] == fact_ids


def test_overlapping_high_confidence_candidate_returns_replace(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    candidate = next(
        item for item in _rows(paths["candidates"])
        if item["candidate_id"] == "history-001-overlap-replacement"
    )

    decision = validate_update_detailed(
        [event["fact"] for event in events if event["history_id"] == "history-001"],
        candidate["fact"],
        preference_rule_parameters(),
    )

    assert decision.decision == "replace"
    assert decision.replace_fact_id == candidate["gold_replace_fact_id"]


def test_event_snapshots_apply_retraction_and_preserve_constraints(tmp_path: Path) -> None:
    events = _rows(generate_dataset(tmp_path)["events"])
    snapshots = materialize_event_states(events)

    assert "history-001-private" not in snapshots[-1]
    assert snapshots[-1]["history-001-constraint"]["operation"] == "hard_constraint"


def test_local_baselines_share_structured_queries(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path)
    scores = evaluate(_rows(paths["events"]), _rows(paths["queries"]))

    assert scores["no_history"] == 0.0
    assert scores["full_history"] == 1.0
    assert scores["recency"] == 0.0
    assert "subject-001 preferred value-001-b" in render_full_transcript(_rows(paths["events"]))
    stale = next(candidate for candidate in _rows(paths["candidates"]) if candidate["candidate_id"] == "history-001-stale")
    assert candidate_decision_accept_all(stale) == "accept"
    assert stale["gold_decision"] == "reject"
    replayed = replay_candidates(_rows(paths["events"]), _rows(paths["candidates"]), use_scallop=False)
    assert resolve_preference(replayed, "subject-001", "2025-08-01") == "value-001-a"


def test_bm25_raw_event_baseline_uses_visible_query_fields_only(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, history_count=3)
    events = _rows(paths["events"])
    queries = _rows(paths["queries"])

    answers = [answer_bm25_raw_events(events, query, k=5) for query in queries]
    altered_gold_answers = [
        answer_bm25_raw_events(events, {**query, "gold": "unrelated-answer"}, k=5)
        for query in queries
    ]

    assert answers == altered_gold_answers
    assert answers != [query["gold"] for query in queries]
    assert evaluate_bm25_raw_events(events, queries, k=5)["bm25_raw_event"] < 1.0


def test_dense_raw_event_baseline_uses_visible_fields_and_only_resolves_retrieved_events(
    tmp_path: Path, monkeypatch
) -> None:
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    query = next(item for item in _rows(paths["queries"]) if item["query_id"] == "history-001-current")
    embedder = FakeDenseEmbedder()
    resolved_events: list[list[dict]] = []

    def capture_resolution(retrieved: list[dict], _query: dict) -> str:
        resolved_events.append(retrieved)
        return "captured"

    monkeypatch.setattr(temporal_baselines, "resolve_query", capture_resolution)

    assert answer_dense_raw_events(events, {**query, "gold": "unrelated-answer"}, k=1, embedder=embedder) == "captured"
    assert resolved_events == [[events[1]]]
    assert embedder.document_calls == [[event["fact"]["support_text"] for event in events]]
    assert embedder.query_calls == [
        "Represent this sentence for searching relevant passages: "
        "What item was preferred by subject-001 in default on 2025-08-01?"
    ]


def test_dense_raw_event_evaluator_reports_model_metadata_without_loading_a_model(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    query = next(item for item in _rows(paths["queries"]) if item["query_id"] == "history-001-current")
    embedder = FakeDenseEmbedder()

    result = evaluate_dense_raw_events(
        events,
        [query, query],
        k=1,
        embedder=embedder,
        model="fake/bge",
        revision="test-revision",
        batch_size=7,
    )

    assert result["dense_raw_event"] == 1.0
    assert embedder.document_calls == [[event["fact"]["support_text"] for event in events]]
    assert result["metadata"] == {
        "embedding": {
            "model_name": "fake/bge",
            "requested_revision": "test-revision",
            "resolved_revision": "test-commit",
            "sentence_transformers_version": "fake-1",
            "vector_dimension": 2,
        },
        "embedding_device": "cpu",
        "embedding_batch_size": 7,
        "embedding_requested_revision": "test-revision",
        "embedding_requested_model": "fake/bge",
            "query_fields": ["query_text"],
        "query_template": "bge_query.v1",
        "retrieval_k": 1,
        "retrieval_unit": "raw_event_support_text",
    }


def test_dense_raw_event_evaluator_batches_query_encoding(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path)
    events = _rows(paths["events"])
    query = next(item for item in _rows(paths["queries"]) if item["query_id"] == "history-001-current")
    queries = [query, query]
    embedder = BatchFakeDenseEmbedder()

    result = evaluate_dense_raw_events(events, queries, k=1, embedder=embedder)

    assert result["dense_raw_event"] == 1.0
    assert embedder.document_calls == [[event["fact"]["support_text"] for event in events]]
    assert embedder.query_batch_calls == [
        [
            "Represent this sentence for searching relevant passages: "
            + temporal_baselines._visible_query_text(query)
            for query in queries
        ]
    ]
    assert embedder.query_calls == []


def test_replayed_bm25_and_dense_evaluators_keep_histories_local_and_hide_gold(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, history_count=2)
    events = _rows(paths["events"])
    candidates = [
        candidate for candidate in _rows(paths["candidates"])
        if candidate["candidate_id"].endswith(("-stale", "-accepted"))
    ]
    current_queries = [query for query in _rows(paths["queries"]) if query["query_id"].endswith("-current")]
    query_by_history = {query["history_id"]: query for query in current_queries}
    for event in events:
        if event["event_id"].endswith("-transition"):
            event["fact"]["support_text"] = (
                f"current-replay-marker {query_by_history[event['history_id']]['query_text']}"
            )
    for candidate in candidates:
        if candidate["gold_decision"] == "reject":
            candidate["fact"]["support_text"] = (
                "stale-replay-marker "
                f"{query_by_history[candidate['history_id']]['query_text']} "
                f"{query_by_history[candidate['history_id']]['query_text']}"
            )

    bm25_accept_all = evaluate_bm25_replayed_candidates(
        events, candidates, current_queries, k=1, use_scallop=False
    )
    bm25_scallop = evaluate_bm25_replayed_candidates(
        events, candidates, current_queries, k=1, use_scallop=True
    )
    dense_accept_all = evaluate_dense_replayed_candidates(
        events,
        candidates,
        current_queries,
        k=1,
        use_scallop=False,
        embedder=StaleCandidateFirstDenseEmbedder(),
    )
    dense_scallop = evaluate_dense_replayed_candidates(
        events,
        candidates,
        current_queries,
        k=1,
        use_scallop=True,
        embedder=StaleCandidateFirstDenseEmbedder(),
    )

    for accept_all, scallop in [
        (bm25_accept_all, bm25_scallop),
        (dense_accept_all, dense_scallop),
    ]:
        assert accept_all["candidate_decision_accuracy"] == 0.5
        assert scallop["candidate_decision_accuracy"] == 1.0
        assert accept_all["candidate_accept_count"] == 4.0
        assert scallop["candidate_accept_count"] == 2.0
        assert accept_all["query_accuracy"] == 0.0
        assert scallop["query_accuracy"] == 1.0

    captured_queries: list[dict[str, Any]] = []
    evaluate_replayed_candidates(
        events,
        candidates,
        current_queries,
        use_scallop=True,
        answer_query=lambda _events, query: captured_queries.append(query) or "value-001-b",
    )
    assert all("gold" not in query for query in captured_queries)


def test_dense_replayed_evaluator_reuses_document_and_batched_query_embeddings(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, history_count=2)
    events = _rows(paths["events"])
    candidates = [
        candidate for candidate in _rows(paths["candidates"])
        if candidate["candidate_id"].endswith(("-stale", "-accepted", "-overlap-replacement"))
    ]
    queries = _rows(paths["queries"])
    embedder = BatchFakeDenseEmbedder()

    result = evaluate_dense_replayed_candidates(
        events,
        candidates,
        queries,
        k=1,
        use_scallop=True,
        embedder=embedder,
    )

    replayed = replay_candidates(events, candidates, use_scallop=True)
    assert result["candidate_decision_accuracy"] == 1.0
    assert len(embedder.document_calls) == 1
    assert sorted(embedder.document_calls[0]) == sorted(
        event["fact"]["support_text"] for event in replayed
    )
    assert len(embedder.query_batch_calls) == 1
    assert embedder.query_batch_calls[0] == [
        "Represent this sentence for searching relevant passages: "
        + temporal_baselines._visible_query_text({key: value for key, value in query.items() if key != "gold"})
        for query in queries
    ]
    assert embedder.query_calls == []
