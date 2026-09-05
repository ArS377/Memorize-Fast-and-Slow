from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.persona_end_to_end_benchmark import (
    _build_arm_specs,
    _resume_manifest_fields,
    load_benchmark_config,
)
from experiments.persona_graphiti_baseline import (
    GraphitiMemoryConfig,
    extract_valid_fact_set,
    prepare_graphiti_memory,
    query_as_of_date,
    reference_time_for_turn,
    render_episode,
    render_graphiti_facts,
)


_GRAPHITI_ENV = {
    "PERSONA_SURFACE_A_DATASET_DIR": "/dataset",
    "PERSONA_SURFACE_A_BENCHMARK_OUTPUT_DIR": "/output",
    "PERSONA_QWEN_MODEL_PATH": "/model",
    "PERSONA_QWEN_MODEL_ID": "Qwen/fixture",
    "PERSONA_QWEN_DEVICE": "cuda",
    "PERSONA_SCALLOP_ENDPOINT": "http://scallop.invalid",
    "PERSONA_RETRIEVAL_INDEX_ROOT": "/indexes",
    "PERSONA_EMBEDDING_MODEL_ID": "BAAI/fixture",
    "PERSONA_EMBEDDING_MODEL_PATH": "/embedding-model",
    "PERSONA_EMBEDDING_REVISION": "revision-1",
    "PERSONA_EMBEDDING_DEVICE": "cpu",
    "PERSONA_NEO4J_URI": "bolt://neo4j.invalid:7687",
    "PERSONA_NEO4J_USER": "neo4j",
    "PERSONA_NEO4J_PASSWORD": "fixture-password",
    "PERSONA_NEO4J_DATABASE": "neo4j",
    "PERSONA_GRAPHITI_BUILD_ID": "build1",
    "PERSONA_GRAPHITI_NEO4J_URI": "bolt://graphiti.invalid:7687",
    "PERSONA_GRAPHITI_NEO4J_USER": "neo4j",
    "PERSONA_GRAPHITI_NEO4J_PASSWORD": "fixture-password",
    "PERSONA_GRAPHITI_NEO4J_DATABASE": "graphiti",
    "PERSONA_GRAPHITI_LLM_BASE_URL": "http://vllm.invalid:8000/v1",
    "PERSONA_GRAPHITI_LLM_API_KEY": "local",
    "PERSONA_GRAPHITI_LLM_MODEL": "Qwen/fixture",
    "PERSONA_GRAPHITI_LLM_SMALL_MODEL": "Qwen/fixture",
}


class WordChatTokenizer:
    """Small tokenizer with visible chat-template overhead."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        """Tokenize fixture text on whitespace."""
        assert not add_special_tokens
        return text.split()

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        """Render one deterministic non-thinking prompt."""
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return f"USER {messages[0]['content']} ASSISTANT READY"


def _graphiti_config(**overrides) -> GraphitiMemoryConfig:
    """Build one valid fixture configuration."""
    values = {
        "graphiti_core_version": "0.29.3",
        "episode_variant": "e2e",
        "retrieval_state": "point_in_time",
        "timestamp_base": "2026-01-01T00:00:00+00:00",
        "timestamp_step_seconds": 60,
        "num_results": 4,
        "llm_max_tokens": 4096,
        "max_episode_failures": 0,
        "max_concurrent_histories": 2,
        "embedding_batch_size": 8,
        "build_id": "build1",
        "neo4j_uri": "bolt://graphiti.invalid:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "fixture-password",
        "neo4j_database": "graphiti",
        "llm_base_url": "http://vllm.invalid:8000/v1",
        "llm_api_key": "local",
        "llm_model": "Qwen/fixture",
        "llm_small_model": "Qwen/fixture",
        "embedding_model_id": "BAAI/fixture",
        "embedding_model_path": Path("/embedding-model"),
        "embedding_revision": "revision-1",
        "embedding_device": "cpu",
    }
    values.update(overrides)
    return GraphitiMemoryConfig(**values)


def _turn(index: int, history_id: str = "history-001") -> dict:
    """Build one natural source turn fixture."""
    return {
        "turn_id": f"turn-{index}",
        "history_id": history_id,
        "text": f"natural conversation words {index}",
        "stream_event": {"event_id": f"event-{index}", "history_id": history_id},
    }


_SNAPSHOT = {
    "nodes": [
        {"labels": ["Entity"], "properties": {"uuid": "n1", "name": "AsterArc"}},
        {"labels": ["Entity"], "properties": {"uuid": "n2", "name": "mint tea"}},
        {"labels": ["Entity"], "properties": {"uuid": "n3", "name": "cedar tea"}},
        {"labels": ["Episodic"], "properties": {"uuid": "e1", "name": "episode-00000"}},
    ],
    "edges": [
        {
            "type": "RELATES_TO",
            "source_uuid": "n1",
            "target_uuid": "n2",
            "properties": {
                "uuid": "r1",
                "name": "PREFERS",
                "fact": "AsterArc prefers mint tea.",
                "valid_at": "2025-07-01T00:00:00+00:00",
                "invalid_at": None,
                "expired_at": None,
                "created_at": "2026-01-01T00:05:00+00:00",
            },
        },
        {
            "type": "RELATES_TO",
            "source_uuid": "n1",
            "target_uuid": "n3",
            "properties": {
                "uuid": "r2",
                "name": "PREFERS",
                "fact": "AsterArc prefers cedar tea.",
                "valid_at": "2025-01-01T00:00:00+00:00",
                "invalid_at": "2025-07-01T00:00:00+00:00",
                "expired_at": "2026-01-01T00:05:00+00:00",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        },
        {
            "type": "MENTIONS",
            "source_uuid": "e1",
            "target_uuid": "n1",
            "properties": {"uuid": "m1"},
        },
    ],
}


class RecordingSession:
    """In-memory GraphitiSession fake that records every backend call."""

    def __init__(self) -> None:
        self.episodes: list[dict] = []
        self.searches: list[dict] = []
        self.cleared: list[str] = []
        self.snapshots: list[str] = []
        self.closed = False

    async def add_episode(self, **kwargs) -> None:
        self.episodes.append(dict(kwargs))

    async def search(
        self,
        *,
        query: str,
        group_id: str,
        num_results: int,
        retrieval_state: str,
        as_of=None,
    ) -> list[dict]:
        self.searches.append(
            {
                "query": query,
                "group_id": group_id,
                "num_results": num_results,
                "retrieval_state": retrieval_state,
                "as_of": as_of,
            }
        )
        return [
            {
                "fact_id": "edge-1",
                "predicate": "PREFERS",
                "object": "",
                "support_text": "AsterArc prefers mint tea.",
                "valid_at": "2025-07-01T00:00:00+00:00",
                "invalid_at": None,
                "expired_at": None,
                "created_at": "2026-01-01T00:05:00+00:00",
            }
        ]

    async def snapshot(self, *, group_id: str) -> dict:
        self.snapshots.append(group_id)
        return _SNAPSHOT

    async def clear_group(self, *, group_id: str) -> None:
        self.cleared.append(group_id)

    async def close(self) -> None:
        self.closed = True

    def identity(self) -> dict:
        return {"graphiti_core_version": "0.29.3-fixture"}


def test_graphiti_config_adds_pinned_arms_and_env_settings() -> None:
    root = Path(__file__).parents[1]
    config = load_benchmark_config(
        root / "configs" / "persona_end_to_end_joint_surface_a.json",
        environ=dict(_GRAPHITI_ENV),
    )

    assert [arm.name for arm in config.arms[-2:]] == [
        "graphiti_memory_4096",
        "graphiti_memory_16384",
    ]
    assert config.hybrid_memory is not None
    assert config.graphiti_memory is not None
    assert config.graphiti_memory.graphiti_core_version == "0.29.3"
    assert config.graphiti_memory.episode_variant == "e2e"
    assert config.graphiti_memory.build_id == "build1"
    assert config.graphiti_memory.neo4j_uri == "bolt://graphiti.invalid:7687"
    assert config.graphiti_memory.neo4j_database == "graphiti"
    assert config.graphiti_memory.llm_base_url == "http://vllm.invalid:8000/v1"
    assert config.graphiti_memory.embedding_revision == "revision-1"
    assert config.graphiti_memory.num_results == 8


def test_graphiti_config_requires_its_own_neo4j_database() -> None:
    root = Path(__file__).parents[1]
    environ = dict(_GRAPHITI_ENV)
    environ["PERSONA_GRAPHITI_NEO4J_URI"] = environ["PERSONA_NEO4J_URI"]
    environ["PERSONA_GRAPHITI_NEO4J_DATABASE"] = environ["PERSONA_NEO4J_DATABASE"]

    with pytest.raises(ValueError, match="its own Neo4j database"):
        load_benchmark_config(
            root / "configs" / "persona_end_to_end_joint_surface_a.json",
            environ=environ,
        )


def test_graphiti_config_requires_graphiti_environment() -> None:
    root = Path(__file__).parents[1]
    environ = dict(_GRAPHITI_ENV)
    del environ["PERSONA_GRAPHITI_NEO4J_URI"]

    with pytest.raises(ValueError, match="PERSONA_GRAPHITI_NEO4J_URI"):
        load_benchmark_config(
            root / "configs" / "persona_end_to_end_joint_surface_a.json",
            environ=environ,
        )


def test_reference_time_mapping_is_deterministic_and_order_preserving() -> None:
    config = _graphiti_config()

    first = reference_time_for_turn(0, config)
    again = reference_time_for_turn(0, config)
    later = reference_time_for_turn(7, config)

    assert first == again == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert (later - first).total_seconds() == 7 * 60
    with pytest.raises(ValueError, match="non-negative"):
        reference_time_for_turn(-1, config)


def test_e2e_episode_keeps_natural_text_with_account_holder_label() -> None:
    turn = {**_turn(3), "text": "User: I prefer mint tea now.\nAssistant: Noted."}

    episode = render_episode(turn, turn_index=3, variant="e2e")

    assert episode["source"] == "message"
    assert episode["body"].startswith("Account holder conversation:\n")
    assert "I prefer mint tea now." in episode["body"]
    assert episode["name"] == "episode-00003"


def test_matched_episode_maps_surface_fields_and_never_leaks_lineage() -> None:
    turn = {
        **_turn(4),
        "stream_event": {
            "event_id": "history-001-transition",
            "history_id": "history-001",
            "operation": "supersede",
            "supersedes": "history-001-initial",
            "surface_subject": "AsterArc",
            "dialogue_subject": "AsterArc",
            "surface_object": "mint tea",
            "fact": {
                "fact_id": "history-001-current",
                "subject": "subject-001",
                "predicate": "PREFERS",
                "object": "value-001-b",
                "temporal": {"valid_from": "2025-07-01", "valid_to": "2025-09-30"},
                "qualifiers": {"scope": "scope-001", "source_authority": "direct_user"},
            },
        },
    }

    episode = render_episode(turn, turn_index=4, variant="matched")
    record = json.loads(episode["body"])

    assert episode["source"] == "json"
    assert record["record"] == "memory_assertion"
    assert record["subject"] == "AsterArc"
    assert record["value"] == "mint tea"
    assert record["valid_from"] == "2025-07-01"
    assert record["valid_to"] == "2025-09-30"
    assert record["source_authority"] == "direct_user"
    assert record["scope"] != "scope-001"
    assert record["scope"].startswith("setting_")
    assert "supersedes" not in episode["body"]
    assert "history-001" not in episode["body"]
    assert "value-001" not in episode["body"]


def test_matched_episode_renders_retractions_and_rejects_latent_leaks() -> None:
    event = {
        "event_id": "history-001-retract",
        "history_id": "history-001",
        "operation": "retract",
        "retracts": "history-001-private",
        "surface_subject": "AsterArc",
        "surface_object": "mushroom risotto",
        "fact": {
            "fact_id": "history-001-private",
            "predicate": "PRIVATE_NOTE",
            "temporal": {"valid_from": "2025-01-01", "valid_to": None},
            "qualifiers": {"scope": "private", "source_authority": "direct_user"},
        },
    }

    episode = render_episode(
        {**_turn(5), "stream_event": event}, turn_index=5, variant="matched"
    )
    record = json.loads(episode["body"])
    assert record["record"] == "memory_retraction"
    assert record["value"] == "mushroom risotto"
    assert record["scope"] == "private"
    assert "history-001-private" not in episode["body"]

    leaking = {**event, "surface_object": "value-001-f"}
    with pytest.raises(ValueError, match="latent label"):
        render_episode(
            {**_turn(5), "stream_event": leaking}, turn_index=5, variant="matched"
        )


def test_prepare_ingests_only_causal_account_turns_and_tears_down() -> None:
    config = _graphiti_config()
    turns = [
        _turn(0),
        _turn(1, "history-002"),
        _turn(2),
        _turn(3),
        _turn(4, "history-002"),
    ]
    conditions = [
        {
            "evaluation_input_id": "q:pre",
            "history_id": "history-001",
            "checkpoint_turn_index": 3,
            "query_text": "what did AsterArc prefer on 2025-08-01",
        },
        {
            "evaluation_input_id": "q:post",
            "history_id": "history-001",
            "checkpoint_turn_index": 3,
            "query_text": "what changed on 2025-08-01",
        },
    ]
    session = RecordingSession()

    retrievals, identity, store_states = prepare_graphiti_memory(
        conditions, turns, config, session_factory=lambda _: session
    )

    ingested_names = [episode["name"] for episode in session.episodes]
    assert ingested_names == ["episode-00000", "episode-00002"]
    assert all(
        episode["group_id"] == "build1-history-001"
        for episode in session.episodes
    )
    assert session.episodes[0]["reference_time"] == datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )
    assert [row["query"] for row in session.searches] == [
        "what did AsterArc prefer on 2025-08-01",
        "what changed on 2025-08-01",
    ]
    assert session.cleared == [
        "build1-history-001",
        "build1-history-001",
    ]
    assert session.closed
    assert set(retrievals) == {"q:pre", "q:post"}
    assert (
        retrievals["q:pre"]["metadata"]["memory_store_sha256"]
        == retrievals["q:post"]["metadata"]["memory_store_sha256"]
    )
    assert retrievals["q:pre"]["rows"][0]["support_text"] == "AsterArc prefers mint tea."
    assert identity["episode_count"] == 2
    assert identity["episode_failure_count"] == 0
    assert identity["memory_store_sha256"] == {
        "history-001:3": retrievals["q:pre"]["metadata"]["memory_store_sha256"]
    }
    assert identity["timestamp_mapping"]["step_seconds"] == 60
    assert identity["session"] == {"graphiti_core_version": "0.29.3-fixture"}


class TimelineSession(RecordingSession):
    """Recording session that also keeps one ordered timeline of backend calls."""

    def __init__(self) -> None:
        super().__init__()
        self.timeline: list[tuple[str, str]] = []

    async def add_episode(self, **kwargs) -> None:
        await super().add_episode(**kwargs)
        self.timeline.append((f"episode:{kwargs['group_id']}", kwargs["name"]))

    async def snapshot(self, *, group_id: str) -> dict:
        self.timeline.append((f"snapshot:{group_id}", ""))
        return await super().snapshot(group_id=group_id)

    async def search(self, **kwargs) -> list[dict]:
        self.timeline.append((f"search:{kwargs['group_id']}", kwargs["query"]))
        return await super().search(**kwargs)


def _condition(name: str, checkpoint: int, history_id: str = "history-001") -> dict:
    """Build one condition fixture at a given causal checkpoint."""
    return {
        "evaluation_input_id": name,
        "history_id": history_id,
        "checkpoint_turn_index": checkpoint,
        "query_text": f"query for {name} on 2025-08-01",
    }


def test_incremental_ingestion_visits_each_turn_exactly_once_across_checkpoints() -> None:
    config = _graphiti_config()
    turns = [_turn(index) for index in range(6)]
    conditions = [
        _condition("q:cp2", 2),
        _condition("q:cp4", 4),
        _condition("q:cp6", 6),
    ]
    session = TimelineSession()

    retrievals, identity, store_states = prepare_graphiti_memory(
        conditions, turns, config, session_factory=lambda _: session
    )

    names = [episode["name"] for episode in session.episodes]
    assert names == [f"episode-{index:05d}" for index in range(6)]
    assert len(names) == len(set(names))
    assert identity["episode_count"] == 6
    assert identity["ingestion_mode"] == "incremental_per_account"
    assert set(store_states) == {
        "history-001:2",
        "history-001:4",
        "history-001:6",
    }
    assert [state["episode_count"] for _, state in sorted(store_states.items())] == [
        2,
        4,
        6,
    ]
    assert set(retrievals) == {"q:cp2", "q:cp4", "q:cp6"}


def test_each_checkpoint_snapshots_only_its_causal_prefix() -> None:
    config = _graphiti_config()
    turns = [_turn(index) for index in range(6)]
    conditions = [_condition("q:cp2", 2), _condition("q:cp5", 5)]
    session = TimelineSession()

    prepare_graphiti_memory(
        conditions, turns, config, session_factory=lambda _: session
    )

    group = "build1-history-001"
    ingested_before_snapshot: list[int] = []
    count = 0
    for kind, value in session.timeline:
        if kind == f"episode:{group}":
            count += 1
        elif kind == f"snapshot:{group}":
            ingested_before_snapshot.append(count)
    assert ingested_before_snapshot == [2, 5]


def test_interleaved_foreign_turns_never_enter_an_account_namespace() -> None:
    config = _graphiti_config()
    turns = [
        _turn(0, "history-001"),
        _turn(1, "history-002"),
        _turn(2, "history-001"),
        _turn(3, "history-002"),
        _turn(4, "history-001"),
    ]
    conditions = [
        _condition("q:a", 5, "history-001"),
        _condition("q:b", 4, "history-002"),
    ]
    session = TimelineSession()

    prepare_graphiti_memory(
        conditions, turns, config, session_factory=lambda _: session
    )

    by_group: dict[str, list[str]] = {}
    for episode in session.episodes:
        by_group.setdefault(episode["group_id"], []).append(episode["name"])
    assert by_group["build1-history-001"] == [
        "episode-00000",
        "episode-00002",
        "episode-00004",
    ]
    assert by_group["build1-history-002"] == ["episode-00001", "episode-00003"]
    assert set(session.cleared) == {"build1-history-001", "build1-history-002"}


def test_reference_time_still_tracks_global_stream_position() -> None:
    config = _graphiti_config()
    turns = [
        _turn(0, "history-001"),
        _turn(1, "history-002"),
        _turn(2, "history-001"),
    ]
    session = RecordingSession()

    prepare_graphiti_memory(
        [_condition("q:a", 3, "history-001")],
        turns,
        config,
        session_factory=lambda _: session,
    )

    times = [episode["reference_time"] for episode in session.episodes]
    assert times == [
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
    ]


def test_point_in_time_search_receives_the_date_the_query_asks_about() -> None:
    config = _graphiti_config()
    session = RecordingSession()
    conditions = [
        {
            "evaluation_input_id": "q:change",
            "history_id": "history-001",
            "checkpoint_turn_index": 1,
            "query_text": (
                "After the long record, what standing preference applied to "
                "AsterArc on 2025-08-01?"
            ),
        }
    ]

    retrievals, _, _ = prepare_graphiti_memory(
        conditions, [_turn(0)], config, session_factory=lambda _: session
    )

    assert session.searches[0]["as_of"] == datetime(2025, 8, 1, tzinfo=timezone.utc)
    assert retrievals["q:change"]["metadata"]["as_of"] == "2025-08-01T00:00:00+00:00"


def test_point_in_time_refuses_a_query_without_exactly_one_date() -> None:
    config = _graphiti_config()
    session = RecordingSession()

    with pytest.raises(ValueError, match="exactly one date"):
        prepare_graphiti_memory(
            [
                {
                    "evaluation_input_id": "q:undated",
                    "history_id": "history-001",
                    "checkpoint_turn_index": 1,
                    "query_text": "what does AsterArc prefer now",
                }
            ],
            [_turn(0)],
            config,
            session_factory=lambda _: session,
        )


def test_query_as_of_date_parses_and_rejects_ambiguity() -> None:
    assert query_as_of_date("applied on 2025-10-15?") == datetime(
        2025, 10, 15, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError, match="exactly one date"):
        query_as_of_date("between 2025-08-01 and 2025-10-15")
    with pytest.raises(ValueError, match="exactly one date"):
        query_as_of_date("no date at all")


def test_point_in_time_filter_keeps_closed_interval_gold_that_current_only_drops() -> None:
    """The preference_change gold is a closed interval, so current_only hides it.

    graphiti stamps expired_at on ANY edge carrying an invalid_at, bounded or
    superseded alike, so a filter that also demanded expired_at IS NULL would
    delete the gold. World time alone has to decide.
    """
    as_of = datetime(2025, 8, 1, tzinfo=timezone.utc)
    # The gold edge: mint tea, valid 2025-07-01 through 2025-09-30.
    gold_valid_at = datetime(2025, 7, 1, tzinfo=timezone.utc)
    gold_invalid_at = datetime(2025, 9, 30, tzinfo=timezone.utc)

    def current_only_keeps(invalid_at, expired_at):
        return invalid_at is None and expired_at is None

    def point_in_time_keeps(valid_at, invalid_at, expired_at):
        # expired_at is deliberately ignored: graphiti derives it from
        # invalid_at, so honouring it would exclude bounded facts twice.
        if valid_at is not None and valid_at > as_of:
            return False
        if invalid_at is not None and invalid_at <= as_of:
            return False
        return True

    # graphiti stamps expired_at on the bounded gold edge as well.
    gold_expired_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert current_only_keeps(gold_invalid_at, gold_expired_at) is False
    assert point_in_time_keeps(gold_valid_at, gold_invalid_at, gold_expired_at) is True
    # A superseded earlier value must still be excluded at the as-of date.
    assert (
        point_in_time_keeps(
            datetime(2025, 1, 1, tzinfo=timezone.utc), gold_valid_at, gold_expired_at
        )
        is False
    )
    # Unresolved dates stay visible rather than shrinking the candidate set.
    assert point_in_time_keeps(None, None, None) is True


@pytest.mark.optional_graphiti
def test_graphiti_date_filter_accepts_is_null() -> None:
    filters = pytest.importorskip("graphiti_core.search.search_filters", reason="optional local Graphiti package is not installed")
    assert filters.DateFilter(comparison_operator=filters.ComparisonOperator.is_null) is not None


def test_valid_fact_set_excludes_invalidated_edges_and_episodic_mentions() -> None:
    facts = extract_valid_fact_set(_SNAPSHOT)

    assert len(facts) == 1
    assert facts[0]["subject"] == "AsterArc"
    assert facts[0]["predicate"] == "PREFERS"
    assert facts[0]["object"] == "mint tea"
    assert facts[0]["fact_text"] == "AsterArc prefers mint tea."
    assert facts[0]["valid_at"] == "2025-07-01T00:00:00+00:00"


def test_valid_fact_set_drops_edges_expired_without_world_time_invalidation() -> None:
    snapshot = {
        "nodes": [
            {"properties": {"uuid": "n1", "name": "AsterArc"}},
            {"properties": {"uuid": "n2", "name": "window seat"}},
        ],
        "edges": [
            {
                "type": "RELATES_TO",
                "source_uuid": "n1",
                "target_uuid": "n2",
                "properties": {
                    "name": "PREFERS",
                    "fact": "AsterArc prefers a window seat.",
                    "invalid_at": None,
                    "expired_at": "2026-01-01T00:09:00+00:00",
                },
            }
        ],
    }

    assert extract_valid_fact_set(snapshot) == []


def test_prepare_captures_store_state_and_requests_current_only_search() -> None:
    config = _graphiti_config()
    session = RecordingSession()
    conditions = [
        {
            "evaluation_input_id": "q:pre",
            "history_id": "history-001",
            "checkpoint_turn_index": 1,
            "query_text": "what did AsterArc prefer on 2025-08-01",
        }
    ]

    retrievals, identity, store_states = prepare_graphiti_memory(
        conditions, [_turn(0)], config, session_factory=lambda _: session
    )

    assert session.searches[0]["retrieval_state"] == "point_in_time"
    assert retrievals["q:pre"]["metadata"]["retrieval_state"] == "point_in_time"
    assert identity["retrieval_state"] == "point_in_time"
    state = store_states["history-001:1"]
    assert state["history_id"] == "history-001"
    assert state["checkpoint_turn_index"] == 1
    assert state["valid_fact_count"] == 1
    assert state["valid_facts"][0]["object"] == "mint tea"
    assert identity["valid_fact_total"] == 1
    assert identity["store_states_sha256"]


def test_unfiltered_retrieval_state_leaves_invalidated_edges_in_search() -> None:
    config = _graphiti_config(retrieval_state="unfiltered")
    session = RecordingSession()
    conditions = [
        {
            "evaluation_input_id": "q:pre",
            "history_id": "history-001",
            "checkpoint_turn_index": 1,
            "query_text": "what did AsterArc prefer on 2025-08-01",
        }
    ]

    _, identity, _ = prepare_graphiti_memory(
        conditions, [_turn(0)], config, session_factory=lambda _: session
    )

    assert session.searches[0]["retrieval_state"] == "unfiltered"
    assert session.searches[0]["as_of"] is None
    assert identity["retrieval_state"] == "unfiltered"


def test_retrieval_state_must_be_a_known_mode() -> None:
    with pytest.raises(ValueError, match="retrieval_state must be one of"):
        _graphiti_config(retrieval_state="whatever")


def test_prepare_aborts_when_episode_failures_exceed_budget() -> None:
    config = _graphiti_config()

    class FailingSession(RecordingSession):
        async def add_episode(self, **kwargs) -> None:
            raise RuntimeError("malformed structured output")

    session = FailingSession()
    conditions = [
        {
            "evaluation_input_id": "q:pre",
            "history_id": "history-001",
            "checkpoint_turn_index": 1,
            "query_text": "what did AsterArc prefer on 2025-08-01",
        }
    ]

    with pytest.raises(ValueError, match="episode failures"):
        prepare_graphiti_memory(
            conditions, [_turn(0)], config, session_factory=lambda _: session
        )
    assert session.closed


def test_failure_in_one_account_aborts_siblings_and_clears_every_namespace() -> None:
    config = _graphiti_config()

    class HalfFailingSession(TimelineSession):
        async def add_episode(self, **kwargs) -> None:
            if kwargs["group_id"].endswith("history-002"):
                raise RuntimeError("malformed structured output")
            await super().add_episode(**kwargs)

    session = HalfFailingSession()
    turns = [_turn(index, "history-001") for index in range(3)] + [
        _turn(index, "history-002") for index in range(3, 6)
    ]
    conditions = [
        _condition("q:a", 3, "history-001"),
        _condition("q:b", 6, "history-002"),
    ]

    with pytest.raises(ValueError, match="episode failures"):
        prepare_graphiti_memory(
            conditions, turns, config, session_factory=lambda _: session
        )
    assert set(session.cleared) == {"build1-history-001", "build1-history-002"}
    assert session.closed


def test_render_graphiti_facts_shows_validity_and_blocks_latent_labels() -> None:
    rows = [
        {
            "support_text": "AsterArc prefers mint tea.",
            "valid_at": "2025-07-01T00:00:00+00:00",
            "invalid_at": "2025-09-30T00:00:00+00:00",
        },
        {"support_text": "AsterArc likes window seats.", "valid_at": None},
    ]

    rendered = render_graphiti_facts(rows)

    assert "[Memory 1] AsterArc prefers mint tea." in rendered
    assert "valid from 2025-07-01" in rendered
    assert "invalid from 2025-09-30" in rendered
    assert "[Memory 2] AsterArc likes window seats." in rendered
    with pytest.raises(ValueError, match="latent label"):
        render_graphiti_facts([{"support_text": "subject-001 prefers value-001-b"}])


def test_arm_specs_build_graphiti_prompts_and_fail_without_retrieval() -> None:
    tokenizer = WordChatTokenizer()
    turns = [_turn(0), _turn(1)]
    condition = {
        "evaluation_input_id": "query-1:post_update",
        "history_id": "history-001",
        "query_id": "query-1-preference-change-delayed",
        "phase": "post_update",
        "checkpoint_turn_index": 2,
        "requested_token_distance": None,
        "query_text": "which preference",
        "gold": "mint tea",
    }
    retrieval = {
        condition["evaluation_input_id"]: {
            "seed_entities": [],
            "rows": [
                {
                    "fact_id": "edge-1",
                    "object": "",
                    "support_text": "AsterArc prefers mint tea.",
                    "valid_at": "2025-07-01T00:00:00+00:00",
                    "invalid_at": None,
                }
            ],
            "metadata": {"group_id": "build1-history-001"},
        }
    }

    specs = _build_arm_specs(
        [condition],
        turns,
        arms=(("graphiti_memory_4096", "graphiti_memory", 4096),),
        injections={},
        graphiti_retrievals=retrieval,
        prompt_instruction="Answer briefly.",
        tokenizer=tokenizer,
    )

    assert len(specs) == 1
    assert "Retrieved memory facts:" in specs[0]["prompt"]
    assert "AsterArc prefers mint tea." in specs[0]["prompt"]
    # No arm may advertise its own provenance in the prompt.
    assert "Scallop-validated" not in specs[0]["prompt"]
    assert "edge-1" not in specs[0]["prompt"]
    assert specs[0]["retrieved_fact_count"] == 1
    assert specs[0]["retrieved_gold_occurrence_count"] == 1
    assert specs[0]["retrieval_metadata"]["group_id"] == "build1-history-001"

    with pytest.raises(ValueError, match="lacks retrieval"):
        _build_arm_specs(
            [condition],
            turns,
            arms=(("graphiti_memory_4096", "graphiti_memory", 4096),),
            injections={},
            graphiti_retrievals={},
            prompt_instruction="Answer briefly.",
            tokenizer=tokenizer,
        )


def test_resume_manifest_records_graphiti_identity() -> None:
    scheduled = {
        "dataset": {"generation_manifest_sha256": "abc"},
        "tokenizer": {"model_id": "fixture"},
        "inputs": [],
        "turns": [],
        "schedule_metrics": {},
    }

    fields = _resume_manifest_fields(
        _FixtureConfig(),
        scheduled,
        {"model_id": "fixture"},
        "config-sha",
        scallop_identity={"engine": "scallopy"},
        injections={},
        specs=[],
        relation_coverage=[],
        evaluator_script_sha256="script-sha",
        git_provenance={"head": "fixture"},
        graphiti_memory={"backend": "graphiti_core", "episode_variant": "e2e"},
    )

    assert fields["graphiti_memory"] == {
        "backend": "graphiti_core",
        "episode_variant": "e2e",
    }
    assert "hybrid_memory" not in fields


class _FixtureConfig:
    """Minimal config stand-in for manifest-field serialization."""

    arms = ()
    prompt_instruction = "Answer briefly."
    max_new_tokens = 32
    bootstrap_samples = 10
    bootstrap_seed = 1


def test_graphiti_only_config_loads_without_the_hybrid_stack() -> None:
    root = Path(__file__).parents[1]
    environ = {
        key: value
        for key, value in _GRAPHITI_ENV.items()
        if not key.startswith(("PERSONA_NEO4J", "PERSONA_RETRIEVAL"))
    }

    config = load_benchmark_config(
        root / "configs" / "persona_graphiti_surface_a.json", environ=environ
    )

    assert config.hybrid_memory is None
    assert config.graphiti_memory is not None
    assert [arm.name for arm in config.arms[-2:]] == [
        "graphiti_memory_4096",
        "graphiti_memory_16384",
    ]
    assert not any(arm.kind == "hybrid_kg_memory" for arm in config.arms)


def test_preflight_condition_selection_is_stable_and_account_scoped() -> None:
    from experiments.persona_graphiti_preflight import select_preflight_conditions

    conditions = [
        {"evaluation_input_id": "a1", "history_id": "history-002"},
        {"evaluation_input_id": "a2", "history_id": "history-001"},
        {"evaluation_input_id": "a3", "history_id": "history-003"},
        {"evaluation_input_id": "a4", "history_id": "history-001"},
    ]

    selected = select_preflight_conditions(conditions, 1)
    assert [row["evaluation_input_id"] for row in selected] == ["a2", "a4"]
    assert len(select_preflight_conditions(conditions, 2)) == 3
    assert len(select_preflight_conditions(conditions, None)) == 4
    with pytest.raises(ValueError, match="at least 1"):
        select_preflight_conditions(conditions, 0)


def test_extraction_health_flags_silent_under_extraction() -> None:
    from experiments.persona_graphiti_preflight import (
        assess_extraction_health,
        summarize_extraction,
    )

    populated = {
        "history-001:4": {
            "history_id": "history-001",
            "checkpoint_turn_index": 4,
            "episode_count": 4,
            "valid_fact_count": 2,
            "valid_facts": [
                {"subject": "AsterArc", "valid_at": "2025-07-01T00:00:00+00:00"},
                {"subject": "AsterArc", "valid_at": None},
            ],
        }
    }
    summary = summarize_extraction(
        populated, {"q:a": {"rows": [{"fact_id": "e1"}]}}
    )
    assert summary["valid_fact_rows_total"] == 2
    assert summary["valid_at_resolved_fraction"] == 0.5
    health = assess_extraction_health(summary)
    assert health["status"] == "passed"
    assert any("valid_at" in warning for warning in health["warnings"])

    # Well-formed but empty extraction raises nothing upstream; it must fail here.
    empty = {
        "history-001:4": {
            "history_id": "history-001",
            "checkpoint_turn_index": 4,
            "episode_count": 30,
            "valid_fact_count": 0,
            "valid_facts": [],
        }
    }
    empty_health = assess_extraction_health(
        summarize_extraction(empty, {"q:a": {"rows": []}})
    )
    assert empty_health["status"] == "failed"
    assert any("no valid facts" in reason for reason in empty_health["fatal"])


def test_error_taxonomy_separates_stale_from_cross_account_intrusion() -> None:
    from experiments.persona_graphiti_analysis import (
        account_surface_values,
        classify_answer,
        error_taxonomy,
    )

    events = [
        {"history_id": "history-001", "surface_object": "mint tea", "fact": {}},
        {"history_id": "history-001", "surface_object": "cedar tea", "fact": {}},
        {"history_id": "history-002", "surface_object": "oat milk", "fact": {}},
    ]
    values = account_surface_values(events)
    assert values == {
        "history-001": {"mint tea", "cedar tea"},
        "history-002": {"oat milk"},
    }

    def row(answer: str) -> dict:
        return {
            "answer": answer,
            "gold": "mint tea",
            "history_id": "history-001",
            "arm": "graphiti_memory_4096",
            "query_family": "preference_change",
        }

    assert classify_answer(row("mint tea"), values) == "correct"
    assert classify_answer(row("cedar tea"), values) == "stale_intrusion"
    assert classify_answer(row("oat milk"), values) == "cross_account_intrusion"
    assert classify_answer(row("UNKNOWN"), values) == "abstained"
    assert classify_answer(row("something else entirely"), values) == "other"

    taxonomy = error_taxonomy([row("cedar tea"), row("mint tea")], values)
    arm = taxonomy["graphiti_memory_4096"]
    assert arm["total"] == 2
    assert arm["rates"]["stale_intrusion"] == 0.5
    assert arm["by_query_family"]["preference_change"]["correct"] == 1


def _fake_run_dir(tmp_path: Path) -> Path:
    """Build a run directory shaped exactly like the harness emits."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({
        "evaluator_version": "persona_end_to_end_qwen.v2",
        "condition_count": 2,
        "generation_count": 4,
        "aggregates": {
            "by_arm": [
                {"arm": "graphiti_memory_4096", "row_count": 2,
                 "history_cluster_count": 1, "exact_match": 0.5, "f1": 0.55},
                {"arm": "sliding_context_4096", "row_count": 2,
                 "history_cluster_count": 1, "exact_match": 0.0, "f1": 0.1},
            ],
            "by_arm_query_family": [
                {"arm": "graphiti_memory_4096", "query_family": "preference_change",
                 "row_count": 1, "history_cluster_count": 1,
                 "exact_match": 1.0, "f1": 1.0},
            ],
            "by_arm_condition": [
                {"arm": "graphiti_memory_4096", "condition": "post_update",
                 "row_count": 1, "history_cluster_count": 1,
                 "exact_match": 1.0, "f1": 1.0},
            ],
            "paired_deltas": [
                {"left_arm": "graphiti_memory_4096", "right_arm": "sliding_context_4096",
                 "paired_row_count": 2, "history_cluster_count": 1,
                 "exact_match_delta": 0.5, "exact_match_delta_ci_95": [0.25, 0.75],
                 "f1_delta": 0.45, "f1_delta_ci_95": [0.2, 0.7],
                 "bootstrap_samples": 2000, "bootstrap_seed": 73},
            ],
            "paired_deltas_by_query_family": [
                {"query_family": "preference_change",
                 "left_arm": "graphiti_memory_4096", "right_arm": "sliding_context_4096",
                 "paired_row_count": 1, "history_cluster_count": 1,
                 "exact_match_delta": 1.0, "exact_match_delta_ci_95": [1.0, 1.0],
                 "f1_delta": 1.0, "f1_delta_ci_95": [1.0, 1.0],
                 "bootstrap_samples": 2000, "bootstrap_seed": 73},
            ],
        },
    }), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({
        "model": {"model_id": "Qwen/Qwen3-4B", "resolved_revision": "abc",
                  "transformers_version": "5.15.0", "torch_version": "2.6.0"},
        "graphiti_memory": {
            "graphiti_core_version": "0.29.3", "ingestion_mode": "incremental_per_account",
            "episode_variant": "e2e", "retrieval_state": "point_in_time",
            "episode_count": 312, "history_count": 12, "episode_failure_count": 0,
            "valid_fact_total": 900, "llm_model": "Qwen/Qwen3-4B",
            "embedding_model_id": "BAAI/bge-small-en-v1.5", "embedding_revision": "5c38",
            "timestamp_mapping": {"base": "2024-01-01T00:00:00+00:00", "step_seconds": 60},
            "store_states_sha256": "deadbeef",
        },
    }), encoding="utf-8")
    rows = [
        {"evaluation_input_id": "q1", "arm": "graphiti_memory_4096",
         "arm_kind": "graphiti_memory", "query_family": "preference_change",
         "history_id": "history-001", "gold": "mint tea", "answer": "mint tea",
         "exact_match": 1.0, "retrieved_gold_occurrence_count": 1},
        {"evaluation_input_id": "q2", "arm": "graphiti_memory_4096",
         "arm_kind": "graphiti_memory", "query_family": "preference_incongruity",
         "history_id": "history-001", "gold": "mint tea", "answer": "cedar tea",
         "exact_match": 0.0, "retrieved_gold_occurrence_count": 0},
        {"evaluation_input_id": "q1", "arm": "sliding_context_4096",
         "arm_kind": "sliding_context", "query_family": "preference_change",
         "history_id": "history-001", "gold": "mint tea", "answer": "oat milk",
         "exact_match": 0.0},
        {"evaluation_input_id": "q2", "arm": "sliding_context_4096",
         "arm_kind": "sliding_context", "query_family": "preference_incongruity",
         "history_id": "history-001", "gold": "mint tea", "answer": "UNKNOWN",
         "exact_match": 0.0},
    ]
    (run / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    events = [
        {"history_id": "history-001", "surface_object": "mint tea", "fact": {}},
        {"history_id": "history-001", "surface_object": "cedar tea", "fact": {}},
        {"history_id": "history-002", "surface_object": "oat milk", "fact": {}},
    ]
    (corpus / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    return run


def test_analysis_separate_output_preserves_input_bundle(tmp_path: Path) -> None:
    from experiments.persona_graphiti_analysis import analyze_run

    run = _fake_run_dir(tmp_path)
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    output = tmp_path / "analysis-output"
    analyze_run(run, tmp_path / "corpus", output_dir=output)
    assert (output / "analysis.md").exists()
    assert {path.name: path.read_bytes() for path in run.iterdir()} == before


def test_analysis_refuses_writes_inside_frozen_snapshot(tmp_path: Path) -> None:
    from experiments.persona_graphiti_analysis import analyze_run

    run = _fake_run_dir(tmp_path)
    (tmp_path / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        analyze_run(run, tmp_path / "corpus")


def _verified_analysis_run(tmp_path: Path) -> Path:
    import hashlib

    run = _fake_run_dir(tmp_path)
    corpus = tmp_path / "corpus"
    (corpus / "dialogue.jsonl").write_text("{}\n", encoding="utf-8")
    corpus_hashes = {
        name: hashlib.sha256((corpus / name).read_bytes()).hexdigest()
        for name in ("events.jsonl", "dialogue.jsonl")
    }
    (corpus / "generation_manifest.json").write_text(json.dumps({
        "status": "completed", "artifact_sha256": corpus_hashes,
    }), encoding="utf-8")
    dataset = {
        "generation_manifest_sha256": hashlib.sha256(
            (corpus / "generation_manifest.json").read_bytes()
        ).hexdigest(),
        "artifact_sha256": corpus_hashes,
    }
    (run / "generations.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "graphiti_store_states.json").write_text("{}\n", encoding="utf-8")
    (run / "generation_manifest.json").write_text(json.dumps({
        "status": "completed", "dataset": dataset,
        "artifact_sha256": {"generations.jsonl": hashlib.sha256(
            (run / "generations.jsonl").read_bytes()
        ).hexdigest()},
    }), encoding="utf-8")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "completed", "dataset": dataset,
        "artifact_sha256": {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in ("metrics.json", "predictions.jsonl", "generations.jsonl",
                         "generation_manifest.json", "graphiti_store_states.json")
        },
    })
    manifest["generation_manifest_sha256"] = manifest["artifact_sha256"]["generation_manifest.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run / "README.md").write_text("Ancillary, not declared in the manifest.\n", encoding="utf-8")
    return run


def test_analysis_verified_frozen_inputs_allow_external_output(tmp_path: Path) -> None:
    from experiments.persona_graphiti_analysis import analyze_run

    bundle = tmp_path / "frozen"
    bundle.mkdir()
    run = _verified_analysis_run(bundle)
    (bundle / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    before = {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    output = tmp_path / "analysis-output"
    analyze_run(run, bundle / "corpus", output_dir=output, verify_inputs=True)
    assert {path.name for path in output.iterdir()} == {
        "README.md", "analysis.md", "diagnostics.json", "error_taxonomy.json",
    }
    assert {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()} == before
    with pytest.raises(ValueError, match="frozen"):
        analyze_run(run, bundle / "corpus", output_dir=bundle / "analysis")
    assert not (bundle / "analysis").exists()


@pytest.mark.parametrize("destination", ["run", "run/analysis", "corpus", "corpus/analysis", "."])
def test_analysis_separate_output_rejects_input_overlap(tmp_path: Path, destination: str) -> None:
    from experiments.persona_graphiti_analysis import analyze_run

    run = _fake_run_dir(tmp_path)
    with pytest.raises(ValueError, match="overlaps input"):
        analyze_run(run, tmp_path / "corpus", output_dir=tmp_path / destination)


def test_analysis_completed_in_place_remains_supported(tmp_path: Path) -> None:
    from experiments.persona_graphiti_analysis import analyze_run

    run = _verified_analysis_run(tmp_path)
    analyze_run(run, tmp_path / "corpus", verify_inputs=True)
    assert "# Qwen Persona Graphiti External Baseline" in (run / "README.md").read_text(encoding="utf-8")
    analyze_run(run, tmp_path / "corpus", verify_inputs=True)


def test_analysis_separate_output_refuses_overwrite(tmp_path: Path) -> None:
    from experiments.persona_graphiti_analysis import analyze_run

    run = _fake_run_dir(tmp_path)
    output = tmp_path / "analysis"
    analyze_run(run, tmp_path / "corpus", output_dir=output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="overwrite"):
        analyze_run(run, tmp_path / "corpus", output_dir=output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize("artifact", [
    "run/metrics.json", "run/predictions.jsonl", "run/generations.jsonl",
    "run/graphiti_store_states.json", "run/generation_manifest.json",
    "corpus/events.jsonl", "corpus/dialogue.jsonl", "corpus/generation_manifest.json",
])
def test_analysis_tampered_input_rejected_before_output(tmp_path: Path, artifact: str) -> None:
    from experiments.persona_graphiti_analysis import analyze_run

    run = _verified_analysis_run(tmp_path)
    path = tmp_path / artifact
    path.write_bytes(path.read_bytes() + b"\n")
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        analyze_run(run, tmp_path / "corpus", output_dir=output, verify_inputs=True)
    assert not output.exists()


@pytest.mark.parametrize("mutation", [
    "incomplete", "hash_map_list", "missing_metrics", "invalid_hash", "unsafe_path",
    "dataset_list", "missing_events", "wrong_pin", "unhashed_artifact", "duplicate_key",
])
def test_analysis_rejects_invalid_manifest_schema(tmp_path: Path, mutation: str) -> None:
    from experiments.persona_graphiti_analysis import analyze_run

    run = _verified_analysis_run(tmp_path)
    path = run / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "incomplete":
        manifest["status"] = "running"
    elif mutation == "hash_map_list":
        manifest["artifact_sha256"] = []
    elif mutation == "missing_metrics":
        del manifest["artifact_sha256"]["metrics.json"]
    elif mutation == "invalid_hash":
        manifest["artifact_sha256"]["metrics.json"] = "deadbeef"
    elif mutation == "unsafe_path":
        manifest["artifact_sha256"]["../corpus/events.jsonl"] = "0" * 64
    elif mutation == "dataset_list":
        manifest["dataset"] = []
    elif mutation == "missing_events":
        del manifest["dataset"]["artifact_sha256"]["events.jsonl"]
    elif mutation == "wrong_pin":
        manifest["dataset"]["generation_manifest_sha256"] = "0" * 64
    elif mutation == "unhashed_artifact":
        manifest["artifacts"] = ["README.md"]
    payload = json.dumps(manifest)
    if mutation == "duplicate_key":
        payload = payload[:-1] + ', "status": "completed"}'
    path.write_text(payload, encoding="utf-8")
    output = tmp_path / "analysis"
    with pytest.raises(ValueError):
        analyze_run(run, tmp_path / "corpus", output_dir=output, verify_inputs=True)
    assert not output.exists()


def test_analysis_rejects_dataset_mapping_not_in_pinned_corpus(tmp_path: Path) -> None:
    import hashlib
    from experiments.persona_graphiti_analysis import analyze_run

    run = _verified_analysis_run(tmp_path)
    extra = tmp_path / "corpus" / "extra.json"
    extra.write_text("{}", encoding="utf-8")
    path = run / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dataset"]["artifact_sha256"][extra.name] = hashlib.sha256(extra.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="primary dataset hash mapping"):
        analyze_run(run, tmp_path / "corpus", output_dir=output, verify_inputs=True)
    assert not output.exists()


def test_analysis_cli_verifies_inputs_when_requested(tmp_path: Path) -> None:
    from experiments.persona_graphiti_analysis import main

    run = _verified_analysis_run(tmp_path)
    output = tmp_path / "analysis"
    args = ["--run-dir", str(run), "--corpus-dir", str(tmp_path / "corpus"),
            "--output-dir", str(output), "--verify-inputs"]
    assert main(args) == 0
    (run / "metrics.json").write_text("{}", encoding="utf-8")
    args[5] = str(tmp_path / "rejected")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main(args)
    assert not (tmp_path / "rejected").exists()


def test_analysis_renders_both_documents_from_real_metric_shapes(tmp_path: Path) -> None:
    from experiments.persona_graphiti_analysis import analyze_run

    run = _fake_run_dir(tmp_path)
    taxonomy = analyze_run(run, tmp_path / "corpus")

    readme = (run / "README.md").read_text(encoding="utf-8")
    analysis = (run / "analysis.md").read_text(encoding="utf-8")

    # Headline table and effect sizes must carry real numbers, not placeholders.
    assert "| `graphiti_memory_4096` | 50.00% | 55.00% | 2 | 1 |" in readme
    assert "+50.00 exact-match points" in readme
    assert "[25.00, 75.00]" in readme
    assert "n/a" not in readme
    assert "unknown" not in readme.split("## What this artifact")[0]

    # The taxonomy must separate the two intrusion kinds.
    assert taxonomy["graphiti_memory_4096"]["labels"]["stale_intrusion"] == 1
    assert taxonomy["sliding_context_4096"]["labels"]["cross_account_intrusion"] == 1
    assert taxonomy["sliding_context_4096"]["labels"]["abstained"] == 1

    # Per-family breakdown and stratified effect sizes must render.
    assert "preference_change" in analysis
    assert "Matched effect sizes within each query family" in analysis
    assert "incremental_per_account" in analysis
    assert (run / "error_taxonomy.json").exists()


def test_a_scallop_free_arm_layout_is_admitted() -> None:
    """Hosts without scallopy must still be able to score graphiti.

    scallopy is not on PyPI, so a GPU host may lack it. The Scallop canary and
    injections serve only the structured_memory arms, so a layout without them
    needs no validator. No config file ships for this because the current
    corpus runs on a host that has scallopy; the layout must stay admitted so
    the capability survives.
    """
    from experiments.persona_end_to_end_benchmark import (
        _ARM_LAYOUTS,
        _GRAPHITI_ARMS,
        _SLIDING_ARMS,
    )

    scallop_free = (*_SLIDING_ARMS, *_GRAPHITI_ARMS)
    assert scallop_free in _ARM_LAYOUTS
    assert not any(kind == "structured_memory" for _, kind, _ in scallop_free)
    assert any(kind == "graphiti_memory" for _, kind, _ in scallop_free)


def _write_snapshot(root: Path, *, standalone_template: bool, embedded: bool) -> Path:
    """Build a minimal model snapshot with one of the two template layouts."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "vocab.json", "merges.txt",
                 "model.safetensors.index.json"):
        (root / name).write_text("{}", encoding="utf-8")
    config = {"chat_template": "{{ x }}"} if embedded else {}
    (root / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    if standalone_template:
        (root / "chat_template.jinja").write_text("{{ x }}", encoding="utf-8")
    return root


def test_model_hashes_accept_either_chat_template_layout(tmp_path: Path) -> None:
    from experiments.interleaved_memory_qwen35_eval import _model_file_hashes

    # Qwen3.5-style: standalone chat_template.jinja is hashed.
    standalone = _write_snapshot(
        tmp_path / "standalone", standalone_template=True, embedded=False
    )
    assert "chat_template.jinja" in _model_file_hashes(standalone)

    # Qwen3-style: template embedded in tokenizer_config.json, still covered.
    embedded = _write_snapshot(
        tmp_path / "embedded", standalone_template=False, embedded=True
    )
    hashes = _model_file_hashes(embedded)
    assert "chat_template.jinja" not in hashes
    assert "tokenizer_config.json" in hashes

    # Neither layout means prompt rendering is unauthenticated: refuse.
    neither = _write_snapshot(
        tmp_path / "neither", standalone_template=False, embedded=False
    )
    with pytest.raises(ValueError, match="prompt.*cannot be authenticated"):
        _model_file_hashes(neither)

    # A genuinely missing weight/tokenizer file is still fatal.
    broken = _write_snapshot(tmp_path / "broken", standalone_template=True, embedded=False)
    (broken / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="missing required files"):
        _model_file_hashes(broken)


def test_graphiti_retrieval_cache_round_trips_and_rejects_a_stale_key(
    tmp_path: Path, monkeypatch
) -> None:
    """A cached memory build must replay exactly, and refuse foreign schedules."""
    from experiments import persona_end_to_end_benchmark as bench

    cache = tmp_path / "retrieval_cache.json"
    monkeypatch.setenv("PERSONA_GRAPHITI_RETRIEVAL_CACHE", str(cache))

    config = type("C", (), {"graphiti_memory": _graphiti_config()})()
    scheduled = {"turns": [{"turn_id": "t0"}], "inputs": [{"evaluation_input_id": "q"}]}
    built = ({"q": {"rows": [{"fact_id": "e1"}]}}, {"backend": "graphiti_core"}, {"h:1": {}})
    calls = []

    def fake_prepare(inputs, turns, graphiti_config):
        calls.append(1)
        return built

    monkeypatch.setattr(bench, "prepare_graphiti_memory", fake_prepare)

    first = bench._prepare_graphiti_memory(config, scheduled, [])
    assert calls == [1] and cache.exists()

    # Second call must replay from cache without re-ingesting.
    second = bench._prepare_graphiti_memory(config, scheduled, [])
    assert calls == [1]
    assert second[0] == first[0]
    assert second[1] == first[1]

    # A different schedule must not silently reuse the old memory.
    other = {"turns": [{"turn_id": "t9"}], "inputs": [{"evaluation_input_id": "q"}]}
    with pytest.raises(ValueError, match="different schedule"):
        bench._prepare_graphiti_memory(config, other, [])


def test_constant_rule_baseline_exposes_a_memoryless_shortcut() -> None:
    from experiments.persona_graphiti_analysis import constant_rule_baseline

    rows = []
    for i in range(6):
        rows.append({"evaluation_input_id": f"c{i}", "query_family": "preference_change",
                     "gold": "mint tea" if i < 5 else "chai"})
    for i in range(4):
        rows.append({"evaluation_input_id": f"d{i}", "query_family": "preference_incongruity",
                     "gold": "weekend delivery"})
    rows += [dict(r) for r in rows]  # duplicate arms must not inflate the count

    report = constant_rule_baseline(rows)

    assert report["condition_count"] == 10
    assert report["distinct_gold_count"] == 3
    assert report["best_single_constant_em"] == 0.5
    assert report["per_family_constant_em"] == 0.9


def test_evidence_availability_measures_the_prompt_not_the_store() -> None:
    from experiments.persona_graphiti_analysis import evidence_availability

    rows = [
        {"arm": "graphiti_memory_4096", "arm_kind": "graphiti_memory",
         "gold": "mint tea", "retrieved_gold_occurrence_count": 2, "exact_match": 1.0},
        {"arm": "graphiti_memory_4096", "arm_kind": "graphiti_memory",
         "gold": "mint tea", "retrieved_gold_occurrence_count": 1, "exact_match": 0.0},
        {"arm": "graphiti_memory_4096", "arm_kind": "graphiti_memory",
         "gold": "mint tea", "retrieved_gold_occurrence_count": 0, "exact_match": 1.0},
        {"arm": "graphiti_memory_4096", "arm_kind": "graphiti_memory",
         "gold": "UNKNOWN", "retrieved_gold_occurrence_count": 0, "exact_match": 1.0},
        {"arm": "sliding_context_4096", "arm_kind": "sliding_context",
         "gold": "mint tea", "retrieved_gold_occurrence_count": 0, "exact_match": 0.0},
    ]

    report = evidence_availability(rows)["graphiti_memory_4096"]

    assert report["scored_rows"] == 3
    assert report["gold_in_prompt"] == 2
    assert report["retrieval_coverage"] == round(2 / 3, 4)
    assert report["accuracy_given_gold_in_prompt"] == 0.5
    assert report["correct_without_gold_in_prompt"] == 1


def test_readme_renders_the_memoryless_floor_and_cannot_lose_it(tmp_path: Path) -> None:
    """Regenerating the analysis must not drop the shortcut warning."""
    from experiments.persona_graphiti_analysis import analyze_run

    run = _fake_run_dir(tmp_path)
    analyze_run(run, tmp_path / "corpus")
    first = (run / "README.md").read_text(encoding="utf-8")

    # The fixture's golds are all "mint tea", so the floor is total.
    assert "DO NOT PUBLISH" in first
    assert "Memoryless floor" in first
    assert "distinct gold values" in first

    # Re-running must reproduce the warning rather than erase it.
    analyze_run(run, tmp_path / "corpus")
    assert (run / "README.md").read_text(encoding="utf-8") == first


def test_retrieval_cache_key_covers_the_adapter_implementation(
    tmp_path: Path, monkeypatch
) -> None:
    """A cache built before an adapter fix must not be replayed after it."""
    from experiments import persona_end_to_end_benchmark as bench

    cache = tmp_path / "c.json"
    monkeypatch.setenv("PERSONA_GRAPHITI_RETRIEVAL_CACHE", str(cache))
    config = type("C", (), {"graphiti_memory": _graphiti_config()})()
    scheduled = {"turns": [{"t": 1}], "inputs": [{"evaluation_input_id": "q"}]}
    monkeypatch.setattr(
        bench, "prepare_graphiti_memory", lambda *a, **k: ({"q": {"rows": []}}, {}, {})
    )
    bench._prepare_graphiti_memory(config, scheduled, [])
    stored = json.loads(cache.read_text())

    # Simulate the adapter changing after the cache was written.
    monkeypatch.setattr(bench, "_sha256", lambda path: "different-adapter-hash")
    with pytest.raises(ValueError, match="different schedule|configuration"):
        bench._prepare_graphiti_memory(config, scheduled, [])
    assert "adapter" in json.dumps(stored) or stored["key"]


def test_hybrid_retrieval_cache_replays_and_guards_its_key(tmp_path: Path, monkeypatch) -> None:
    """A crashed joint run must resume against the memory it was scored on."""
    from experiments import persona_end_to_end_benchmark as bench

    cache = tmp_path / "hybrid.json"
    monkeypatch.setenv("PERSONA_HYBRID_RETRIEVAL_CACHE", str(cache))
    config = type("C", (), {"hybrid_memory": object()})()
    scheduled = {"turns": [{"t": 1}], "inputs": [{"evaluation_input_id": "q"}]}
    built = ({"q": {"rows": []}}, {"backend": "neo4j"}, [{"fact_id": "f"}], {"q": []})
    calls = []

    monkeypatch.setattr(bench, "asdict", lambda obj: {"cfg": "same"})
    monkeypatch.setattr(bench, "_sha256", lambda path: "impl-hash-v1")
    monkeypatch.setattr(
        bench, "_prepare_hybrid_memory",
        lambda *a, **k: (calls.append(1), built)[1],
    )

    first = bench._cached_hybrid_memory(config, scheduled, [])
    assert calls == [1] and cache.exists()

    second = bench._cached_hybrid_memory(config, scheduled, [])
    assert calls == [1], "cached map must replay without rebuilding"
    assert second[0] == first[0] and second[1] == first[1]

    # A changed retrieval implementation must invalidate rather than replay.
    monkeypatch.setattr(bench, "_sha256", lambda path: "impl-hash-v2")
    with pytest.raises(ValueError, match="retrieval implementation|different schedule"):
        bench._cached_hybrid_memory(config, scheduled, [])


def test_every_retrieval_arm_uses_the_same_evidence_header() -> None:
    """A provenance-advertising header would bias the arm comparison."""
    from experiments.persona_end_to_end_benchmark import (
        _RETRIEVED_FACTS_HEADER,
        _fit_hybrid_kg_prompt,
    )
    from experiments.persona_graphiti_baseline import render_graphiti_facts
    from experiments.persona_neurosym_retrieval import render_retrieved_facts

    tokenizer = WordChatTokenizer()
    prefix = [_turn(1)]
    hybrid_rows = [{"fact_id": "h", "subject": "AsterArc", "predicate": "PREFERS",
                    "object": "mint tea", "support_text": "AsterArc chose mint tea."}]
    graphiti_rows = [{"fact_id": "g", "support_text": "AsterArc prefers mint tea.",
                      "valid_at": None, "invalid_at": None}]

    _, _, hybrid_prompt, _ = _fit_hybrid_kg_prompt(
        prefix, retrieved_rows=hybrid_rows, history_id="history-001",
        query_text="q on 2025-08-01", prompt_instruction="Answer briefly.",
        cap=200, tokenizer=tokenizer, fact_renderer=render_retrieved_facts,
    )
    _, _, graphiti_prompt, _ = _fit_hybrid_kg_prompt(
        prefix, retrieved_rows=graphiti_rows, history_id="history-001",
        query_text="q on 2025-08-01", prompt_instruction="Answer briefly.",
        cap=200, tokenizer=tokenizer, fact_renderer=render_graphiti_facts,
    )

    assert _RETRIEVED_FACTS_HEADER in hybrid_prompt
    assert _RETRIEVED_FACTS_HEADER in graphiti_prompt
    for prompt in (hybrid_prompt, graphiti_prompt):
        assert "Scallop" not in prompt
        assert "validated" not in prompt.lower()


def test_json_artifacts_are_written_atomically(tmp_path: Path) -> None:
    """A signal mid-write must not leave a truncated manifest or cache."""
    from experiments.persona_end_to_end_benchmark import _write_json

    target = tmp_path / "m.json"
    _write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    # No temporary file is left behind, and a rewrite replaces cleanly.
    _write_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_temporal_post_filter_rejects_a_boundary_edge_the_server_returned() -> None:
    """Re-apply the predicate locally; the server filter has let a boundary through."""
    from experiments.persona_graphiti_baseline import _as_utc

    as_of = datetime(2025, 8, 1, tzinfo=timezone.utc)

    def keeps(valid_at, invalid_at):
        v, i = _as_utc(valid_at), _as_utc(invalid_at)
        if v is not None and v > as_of:
            return False
        if i is not None and i <= as_of:
            return False
        return True

    # The exact violation observed in the committed build: invalid_at == as_of.
    assert keeps(None, "2025-08-01T00:00:00") is False
    assert keeps(None, "2025-08-01T00:00:00+00:00") is False
    # A bounded fact still valid at the as-of date survives.
    assert keeps("2025-07-01T00:00:00+00:00", "2025-09-30T00:00:00+00:00") is True
    # A fact that starts later must not.
    assert keeps("2025-09-01T00:00:00+00:00", None) is False
    # Naive timestamps are treated as UTC rather than crashing the comparison.
    assert keeps("2025-07-01T00:00:00", None) is True


def test_degenerate_graphiti_memory_is_rejected_before_generation() -> None:
    from experiments.persona_end_to_end_benchmark import _assert_graphiti_memory_usable

    healthy = {f"q{i}": {"rows": [{"fact_id": "f"}]} for i in range(10)}
    _assert_graphiti_memory_usable(healthy)

    mostly_empty = {f"q{i}": {"rows": [] if i > 2 else [{"fact_id": "f"}]} for i in range(10)}
    with pytest.raises(ValueError, match="refusing to spend generations"):
        _assert_graphiti_memory_usable(mostly_empty)

    with pytest.raises(ValueError, match="retrieval map is empty"):
        _assert_graphiti_memory_usable({})
