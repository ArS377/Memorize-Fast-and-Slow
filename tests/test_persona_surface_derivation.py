from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from experiments.persona_end_to_end_benchmark import (
    _load_source_and_rebuild,
    load_benchmark_config,
)
from experiments.persona_interference_schedule import ScheduleConfig, build_evaluation_schedule
from experiments.persona_surface_derivation import (
    CATEGORY_VOCABULARIES,
    DERIVATION_VERSION,
    SEATING_ROLE_VOCABULARIES,
    _replace_visible_text,
    _stale_role_variants,
    _validate_visible_transform,
    derive_surface_assignments,
    derive_surface_corpus,
    transform_surface_artifacts,
)


class WhitespaceTokenizer:
    """Deterministic tokenizer fixture with explicit provenance."""

    def encode(self, text: str) -> list[str]:
        """Tokenize fixture text on whitespace."""
        return text.split()

    def metadata(self) -> dict[str, str]:
        """Return stable fixture identity."""
        return {"name": "whitespace", "revision": "v1"}


class WordChatTokenizer:
    """Small chat tokenizer for loading the real benchmark schedules without inference."""

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


def _root() -> Path:
    """Return the repository root independently of the process CWD."""
    return Path(__file__).parents[1]


def _parent() -> Path:
    """Return the authenticated parent corpus."""
    return _root() / "results" / "persona_conflict_conversations_v1"


def _jsonl(path: Path) -> list[dict]:
    """Load object-valued JSONL test data."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    """Hash one fixture artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(manifest: dict) -> dict[str, dict[str, str]]:
    """Index the complete manifest mapping by history and source phrase."""
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for row in manifest["surface_mapping"]:
        result[row["history_id"]][row["source_phrase"]] = row["target_phrase"]
    return dict(result)


def _replace(text: str, replacements: dict[str, str]) -> str:
    """Apply the manifest's simultaneous phrase replacement contract."""
    for source in sorted(replacements, key=lambda value: (-len(value), value)):
        text = text.replace(source, replacements[source])
    return text


@pytest.fixture(scope="module")
def derived(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Derive both requested assignments once for the focused contract tests."""
    root = tmp_path_factory.mktemp("persona-surfaces")
    parent = root / _parent().name
    shutil.copytree(_parent(), parent)
    surface_a = root / "surface-a"
    surface_b = root / "surface-b"
    derive_surface_assignments(parent, surface_a, surface_b)
    return surface_a, surface_b


def test_derivation_is_reproducible_and_authenticates_parent(
    tmp_path: Path, derived: tuple[Path, Path]
) -> None:
    surface_a, _ = derived
    repeated = tmp_path / "repeated"
    derive_surface_corpus(_parent(), repeated, seed=137)

    assert {path.name: path.read_bytes() for path in surface_a.iterdir()} == {
        path.name: path.read_bytes() for path in repeated.iterdir()
    }
    manifest = json.loads((surface_a / "generation_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["derivation"] == {
        "algorithm": "category_block_permutation",
        "version": DERIVATION_VERSION,
        "seed": 137,
    }
    assert manifest["parent"]["generation_manifest_sha256"] == _sha256(
        _parent() / "generation_manifest.json"
    )
    assert manifest["parent"]["artifact_sha256"] == json.loads(
        (_parent() / "generation_manifest.json").read_text()
    )["artifact_sha256"]
    assert manifest["surface_mapping_sha256"] == hashlib.sha256(
        json.dumps(
            manifest["surface_mapping"],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["artifact_sha256"] == {
        path.name: _sha256(path)
        for path in surface_a.iterdir()
        if path.name != "generation_manifest.json"
    }

    tampered_parent = tmp_path / "tampered-parent"
    shutil.copytree(_parent(), tampered_parent)
    with (tampered_parent / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="parent artifact hash mismatch for events.jsonl"):
        derive_surface_corpus(tampered_parent, tmp_path / "rejected", seed=137)


def test_scheduler_rejects_a_resigned_invalid_parent_chain(
    tmp_path: Path, derived: tuple[Path, Path]
) -> None:
    dataset = tmp_path / "invalid-chain"
    shutil.copytree(derived[0], dataset)
    shutil.copytree(_parent(), tmp_path / _parent().name)
    manifest_path = dataset / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parent"]["kimi_provenance"]["model_identity"] = "unrelated-generator"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    config = ScheduleConfig(
        source_split="test",
        source_profile="anti_shortcut_interleaved_v3",
        source_manifest_sha256=_sha256(manifest_path),
        seed=73,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
        query_suffixes=("-preference-change-delayed",),
        token_distance_thresholds=(100,),
    )

    with pytest.raises(ValueError, match="parent is not Kimi-authored"):
        build_evaluation_schedule(dataset, WhitespaceTokenizer(), config)


def test_scheduler_cryptographically_verifies_the_actual_parent(
    tmp_path: Path, derived: tuple[Path, Path]
) -> None:
    dataset = tmp_path / "derived"
    parent = tmp_path / _parent().name
    shutil.copytree(derived[0], dataset)
    shutil.copytree(_parent(), parent)
    with (parent / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    config = ScheduleConfig(
        source_split="test",
        source_profile="anti_shortcut_interleaved_v3",
        source_manifest_sha256=_sha256(dataset / "generation_manifest.json"),
        seed=73,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
        query_suffixes=("-preference-change-delayed",),
        token_distance_thresholds=(100,),
    )

    with pytest.raises(ValueError, match="actual parent artifact hash mismatch for events.jsonl"):
        build_evaluation_schedule(dataset, WhitespaceTokenizer(), config)


def test_scheduler_rejects_resigned_modified_child_surface(
    tmp_path: Path, derived: tuple[Path, Path]
) -> None:
    dataset = tmp_path / "modified-child"
    shutil.copytree(derived[0], dataset)
    shutil.copytree(_parent(), tmp_path / _parent().name)
    events_path = dataset / "events.jsonl"
    events = _jsonl(events_path)
    events[0]["model_text"] += " Modified visible surface."
    events_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
        encoding="utf-8",
    )
    manifest_path = dataset / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"]["events.jsonl"] = _sha256(events_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    config = ScheduleConfig(
        source_split="test",
        source_profile="anti_shortcut_interleaved_v3",
        source_manifest_sha256=_sha256(manifest_path),
        seed=73,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
        query_suffixes=("-preference-change-delayed",),
        token_distance_thresholds=(100,),
    )

    with pytest.raises(ValueError, match="canonical transformation.*events.jsonl"):
        build_evaluation_schedule(dataset, WhitespaceTokenizer(), config)


def test_scheduler_rejects_resigned_noncanonical_surface_mapping(
    tmp_path: Path, derived: tuple[Path, Path]
) -> None:
    dataset = tmp_path / "modified-mapping"
    parent = tmp_path / _parent().name
    shutil.copytree(derived[0], dataset)
    shutil.copytree(_parent(), parent)
    manifest_path = dataset / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mapping = manifest["surface_mapping"]
    matching = [row for row in mapping if row["source_phrase"] == "cedar tea"][:2]
    matching[0]["target_phrase"], matching[1]["target_phrase"] = (
        matching[1]["target_phrase"],
        matching[0]["target_phrase"],
    )
    parent_manifest = json.loads((parent / "generation_manifest.json").read_text())
    parent_artifacts = {
        name: (parent / name).read_bytes()
        for name in parent_manifest["artifact_sha256"]
    }
    child_artifacts = transform_surface_artifacts(parent_artifacts, mapping)
    for name, payload in child_artifacts.items():
        (dataset / name).write_bytes(payload)
    manifest["surface_mapping_sha256"] = hashlib.sha256(
        json.dumps(
            mapping, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    manifest["artifact_sha256"] = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in child_artifacts.items()
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    config = ScheduleConfig(
        source_split="test",
        source_profile="anti_shortcut_interleaved_v3",
        source_manifest_sha256=_sha256(manifest_path),
        seed=73,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
        query_suffixes=("-preference-change-delayed",),
        token_distance_thresholds=(100,),
    )

    with pytest.raises(ValueError, match="surface mapping.*declared seed"):
        build_evaluation_schedule(dataset, WhitespaceTokenizer(), config)


def test_case_insensitive_role_variants_preserve_capitalization_and_coherence() -> None:
    replacements = {
        "cedar tea": "assam tea",
        "mint tea": "sencha tea",
        "oolong tea": "jasmine tea",
        "window seating": "balcony seating",
        "aisle seating": "courtyard seating",
        "evening delivery": "dawn delivery",
        "weekend delivery": "sunset delivery",
        "cedar glass": "amber compass",
    }
    source = (
        "Cedar tea gave way to MINT TEA, while the Oolong preference stayed historical. "
        "A Window spot remained preferable to an AISLE SEAT. The earlier evening option "
        "was replaced by WEEKEND DELIVERY. CEDAR-GLASS was the private note."
    )

    transformed = _replace_visible_text(source, replacements)

    assert transformed == (
        "Assam tea gave way to SENCHA TEA, while the Jasmine tea preference stayed historical. "
        "A Balcony spot remained preferable to a COURTYARD SEAT. The earlier dawn option "
        "was replaced by SUNSET DELIVERY. AMBER COMPASS was the private note."
    )
    assert _stale_role_variants(transformed, replacements) == []


def test_every_seating_term_is_role_and_context_compatible() -> None:
    contexts = ("airplane", "train", "home", "work", "venue")
    role_sources = {
        "positive": "window seating",
        "avoided": "aisle seating",
    }
    for role, vocabulary in SEATING_ROLE_VOCABULARIES.items():
        assert len(vocabulary) == len(set(vocabulary)) == 16
        source = role_sources[role]
        source_descriptor = source.removesuffix(" seating")
        for target in vocabulary:
            assert target.endswith(" seating")
            descriptor = target.removesuffix(" seating")
            for context in contexts:
                replacements = {source: target}
                seating = _replace_visible_text(
                    f"In the {context}, {source} was the relevant option.",
                    replacements,
                )
                seat = _replace_visible_text(
                    f"In the {context}, I selected a {source_descriptor} seat.",
                    replacements,
                )
                spot = _replace_visible_text(
                    f"In the {context}, I selected a {source_descriptor} spot.",
                    replacements,
                )
                article = "an" if descriptor[0].casefold() in "aeiou" else "a"
                assert target in seating
                assert f"{article} {descriptor} seat" in seat
                assert f"{article} {descriptor} spot" in spot
                assert _stale_role_variants(seating + seat + spot, replacements) == []


def test_materialized_seating_assignments_use_each_role_pool_exactly_once() -> None:
    for assignment in ("a", "b"):
        manifest = json.loads(
            (
                _root()
                / "results"
                / f"persona_conflict_conversations_surface_{assignment}"
                / "generation_manifest.json"
            ).read_text()
        )
        by_source: dict[str, set[str]] = defaultdict(set)
        for row in manifest["surface_mapping"]:
            if row["category"] == "seating":
                by_source[row["source_phrase"]].add(row["target_phrase"])
        assert by_source["window seating"] == set(
            SEATING_ROLE_VOCABULARIES["positive"]
        )
        assert by_source["aisle seating"] == set(
            SEATING_ROLE_VOCABULARIES["avoided"]
        )


def test_surface_contradictions_fail_loud() -> None:
    with pytest.raises(ValueError, match="surface contradiction.*mint tea"):
        _validate_visible_transform(
            "The preference was mint tea.",
            "The preference was sencha tea, but the mint phase remains.",
            {"mint tea": "sencha tea"},
            "fixture",
        )


@pytest.mark.parametrize("mutation", ("gold", "causal_contract"))
def test_scheduler_rejects_derived_gold_or_causal_contract_drift(
    tmp_path: Path, derived: tuple[Path, Path], mutation: str
) -> None:
    dataset = tmp_path / mutation
    shutil.copytree(derived[0], dataset)
    shutil.copytree(_parent(), tmp_path / _parent().name)
    queries_path = dataset / "queries.jsonl"
    queries = _jsonl(queries_path)
    target = next(
        row
        for row in queries
        if row["split"] == "test"
        and row["query_id"].endswith("-preference-change-delayed")
    )
    if mutation == "gold":
        target["gold"] = "wrong-internal-gold"
    else:
        target["checkpoint_contract"]["trigger_event_id"] = "wrong-trigger"
    queries_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in queries),
        encoding="utf-8",
    )
    manifest_path = dataset / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"]["queries.jsonl"] = _sha256(queries_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    config = ScheduleConfig(
        source_split="test",
        source_profile="anti_shortcut_interleaved_v3",
        source_manifest_sha256=_sha256(manifest_path),
        seed=73,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
        query_suffixes=("-preference-change-delayed",),
        token_distance_thresholds=(100,),
    )

    with pytest.raises(ValueError, match="query gold or causal contract differs"):
        build_evaluation_schedule(dataset, WhitespaceTokenizer(), config)


def test_derivation_preserves_latent_data_and_replaces_surfaces_exactly(
    derived: tuple[Path, Path],
) -> None:
    parent_events = _jsonl(_parent() / "events.jsonl")
    parent_queries = _jsonl(_parent() / "queries.jsonl")
    parent_dialogue = _jsonl(_parent() / "dialogue.jsonl")

    for output in derived:
        manifest = json.loads((output / "generation_manifest.json").read_text())
        mappings = _mapping(manifest)
        child_events = _jsonl(output / "events.jsonl")
        child_queries = _jsonl(output / "queries.jsonl")
        child_dialogue = _jsonl(output / "dialogue.jsonl")
        assert len(child_events) == len(parent_events)
        assert len(child_queries) == len(parent_queries)
        assert len(child_dialogue) == len(parent_dialogue)

        for parent, child in zip(parent_events, child_events, strict=True):
            history_mapping = mappings[parent["history_id"]]
            assert child["model_text"] == _replace_visible_text(
                parent["model_text"], history_mapping
            )
            expected_surface = history_mapping.get(parent["surface_object"], parent["surface_object"])
            assert child["surface_object"] == expected_surface
            assert {key: value for key, value in child.items() if key not in {"model_text", "surface_object"}} == {
                key: value
                for key, value in parent.items()
                if key not in {"model_text", "surface_object"}
            }
        for parent, child in zip(parent_queries, child_queries, strict=True):
            history_mapping = mappings[parent["history_id"]]
            for field in ("query_text", "surface_query_text", "surface_gold"):
                assert child[field] == _replace_visible_text(
                    parent[field], history_mapping
                )
            assert {
                key: value
                for key, value in child.items()
                if key not in {"query_text", "surface_query_text", "surface_gold"}
            } == {
                key: value
                for key, value in parent.items()
                if key not in {"query_text", "surface_query_text", "surface_gold"}
            }
        for parent, child, event in zip(
            parent_dialogue, child_dialogue, child_events, strict=True
        ):
            assert child["text"] == event["model_text"]
            assert child["text"] == _replace_visible_text(
                parent["text"], mappings[event["history_id"]]
            )

        for name in (
            "candidate_updates.jsonl",
            "examples.jsonl",
            "facts.jsonl",
            "raw_responses.jsonl",
            "requests.jsonl",
            "source_documents.jsonl",
        ):
            assert (output / name).read_bytes() == (_parent() / name).read_bytes()


def test_category_uniqueness_collisions_stale_phrases_and_balance(
    derived: tuple[Path, Path],
) -> None:
    parent_queries = _jsonl(_parent() / "queries.jsonl")
    parent_split_composition = Counter(
        (row["split"], json.dumps(row.get("composition"), sort_keys=True))
        for row in parent_queries
    )

    for output in derived:
        manifest = json.loads((output / "generation_manifest.json").read_text())
        mappings = _mapping(manifest)
        rows_by_history: dict[str, list[dict]] = defaultdict(list)
        for row in manifest["surface_mapping"]:
            rows_by_history[row["history_id"]].append(row)
            assert row["target_phrase"] in CATEGORY_VOCABULARIES[row["category"]]
        assert all(
            len({row["target_phrase"] for row in rows}) == len(rows)
            for rows in rows_by_history.values()
        )

        queries = _jsonl(output / "queries.jsonl")
        for suffix in (
            "-preference-change-delayed",
            "-preference-incongruity-delayed",
        ):
            answers = {
                row["surface_gold"]
                for row in queries
                if row["split"] == "test" and row["query_id"].endswith(suffix)
            }
            assert len(answers) == 12
        assert Counter(
            (row["split"], json.dumps(row.get("composition"), sort_keys=True))
            for row in queries
        ) == parent_split_composition

        visible = "\n".join(
            [row["model_text"] for row in _jsonl(output / "events.jsonl")]
            + [row["text"] for row in _jsonl(output / "dialogue.jsonl")]
            + [row["surface_query_text"] for row in queries]
            + [row["surface_gold"] for row in queries]
        )
        for history_mapping in mappings.values():
            assert _stale_role_variants(visible, history_mapping) == []

        for event in _jsonl(output / "events.jsonl"):
            stale = _stale_role_variants(
                event["model_text"], mappings[event["history_id"]]
            )
            assert stale == [], f"{event['event_id']} retains {stale}"
            assert not any(
                source.casefold() in event["model_text"].casefold()
                and target.casefold() in event["model_text"].casefold()
                for source, target in mappings[event["history_id"]].items()
            ), f"{event['event_id']} contradicts its transformed role"
        for event, dialogue in zip(
            _jsonl(output / "events.jsonl"),
            _jsonl(output / "dialogue.jsonl"),
            strict=True,
        ):
            assert _stale_role_variants(
                dialogue["text"], mappings[event["history_id"]]
            ) == []


def test_assignments_rotate_delayed_terms_across_roles_and_are_disjoint(
    derived: tuple[Path, Path],
) -> None:
    surface_a, surface_b = derived
    manifests = [
        json.loads((output / "generation_manifest.json").read_text()) for output in derived
    ]
    target_to_source = [
        {
            (row["category"], row["target_phrase"]): row["source_phrase"]
            for row in manifest["surface_mapping"]
        }
        for manifest in manifests
    ]
    queries = [_jsonl(surface_a / "queries.jsonl"), _jsonl(surface_b / "queries.jsonl")]
    for suffix in (
        "-preference-change-delayed",
        "-preference-incongruity-delayed",
    ):
        answer_sets = [
            {
                row["surface_gold"]
                for row in assignment
                if row["split"] == "test" and row["query_id"].endswith(suffix)
            }
            for assignment in queries
        ]
        assert answer_sets[0].isdisjoint(answer_sets[1])
        category = "tea" if "change" in suffix else "delivery"
        for answer in answer_sets[0] | answer_sets[1]:
            assert target_to_source[0][category, answer] != target_to_source[1][category, answer]


@pytest.mark.parametrize(
    ("directory_name", "seed"),
    (
        ("persona_conflict_conversations_surface_a", 137),
        ("persona_conflict_conversations_surface_b", 911),
    ),
)
def test_materialized_corpora_authenticate_and_schedule_120_conditions(
    directory_name: str, seed: int
) -> None:
    dataset = _root() / "results" / directory_name
    manifest = json.loads((dataset / "generation_manifest.json").read_text())
    assert manifest["derivation"]["seed"] == seed
    config = ScheduleConfig(
        source_split="test",
        source_profile="anti_shortcut_interleaved_v3",
        source_manifest_sha256=_sha256(dataset / "generation_manifest.json"),
        seed=73,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
        query_suffixes=(
            "-preference-change-delayed",
            "-preference-incongruity-delayed",
        ),
        token_distance_thresholds=(100, 500),
    )

    scheduled = build_evaluation_schedule(dataset, WhitespaceTokenizer(), config)

    assert len(scheduled["inputs"]) == 120
    assert scheduled["dataset"]["derivation"]["seed"] == seed
    assert scheduled["dataset"]["checkpoint_policy"] == (
        "authenticated_parent_exact_indices"
    )
    mappings = _mapping(manifest)
    events = _jsonl(dataset / "events.jsonl")
    dialogue = _jsonl(dataset / "dialogue.jsonl")
    for event, turn in zip(events, dialogue, strict=True):
        history_mapping = mappings[event["history_id"]]
        assert _stale_role_variants(event["model_text"], history_mapping) == []
        assert _stale_role_variants(turn["text"], history_mapping) == []


@pytest.mark.parametrize("token_distances", ((100, 500), (4096, 8192)))
def test_surface_assignments_preserve_parent_condition_checkpoints(
    token_distances: tuple[int, int],
) -> None:
    datasets = (
        _parent(),
        _root() / "results" / "persona_conflict_conversations_surface_a",
        _root() / "results" / "persona_conflict_conversations_surface_b",
    )
    schedules = []
    for dataset in datasets:
        config = ScheduleConfig(
            source_split="test",
            source_profile="anti_shortcut_interleaved_v3",
            source_manifest_sha256=_sha256(dataset / "generation_manifest.json"),
            seed=73,
            concurrent_accounts=8,
            min_segment_events=4,
            max_segment_events=8,
            query_suffixes=(
                "-preference-change-delayed",
                "-preference-incongruity-delayed",
            ),
            token_distance_thresholds=token_distances,
        )
        schedules.append(
            build_evaluation_schedule(dataset, WhitespaceTokenizer(), config)
        )

    identities = [
        {
            row["evaluation_input_id"]: (
                row["phase"],
                row["checkpoint_turn_index"],
                row["relevant_update_turn_index"],
                row["relevant_update_event_id"],
                row["internal_gold"],
                row["selected_turn_ids"],
            )
            for row in schedule["inputs"]
        }
        for schedule in schedules
    ]
    assert identities[0] == identities[1] == identities[2]
    assert any(
        parent["actual_token_distance"] != surface["actual_token_distance"]
        for parent, surface in zip(
            schedules[0]["inputs"], schedules[1]["inputs"], strict=True
        )
        if parent["actual_token_distance"] is not None
    )


@pytest.mark.parametrize(
    ("assignment", "manifest_sha256"),
    (
        ("a", "c64c5c4268d93691b9bdf12119fa31a91cc7e9a798182135593c85128fc89c89"),
        ("b", "07dbaa9698e282e67fe992f6f49282c5c2001ec8a99ad6fb472dd04772a5de92"),
    ),
)
def test_benchmark_configs_pin_real_120_condition_schedules(
    tmp_path: Path, assignment: str, manifest_sha256: str
) -> None:
    dataset = _root() / "results" / f"persona_conflict_conversations_surface_{assignment}"
    config = load_benchmark_config(
        _root() / "configs" / f"persona_end_to_end_benchmark_surface_{assignment}.json",
        environ={
            f"PERSONA_SURFACE_{assignment.upper()}_DATASET_DIR": str(dataset),
            f"PERSONA_SURFACE_{assignment.upper()}_BENCHMARK_OUTPUT_DIR": str(tmp_path),
            "PERSONA_QWEN_MODEL_PATH": "/model",
            "PERSONA_QWEN_MODEL_ID": "Qwen/fixture",
            "PERSONA_QWEN_DEVICE": "cuda",
            "PERSONA_SCALLOP_ENDPOINT": "http://scallop.invalid",
        },
    )

    scheduled, _, _ = _load_source_and_rebuild(config, WordChatTokenizer())

    assert config.schedule.seed == 73
    assert config.schedule.source_manifest_sha256 == manifest_sha256
    assert len(scheduled["inputs"]) == 120
    assert [arm.name for arm in config.arms] == [
        "sliding_context_4096",
        "sliding_context_16384",
        "structured_memory_4096",
        "structured_memory_16384",
        "full_qwen_context",
    ]
