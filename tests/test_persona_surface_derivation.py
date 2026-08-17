from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from conftest import FakeKimiClient
from experiments.persona_conversation_generator import LLMResponse
from experiments.persona_interference_schedule import ScheduleConfig, build_evaluation_schedule
from experiments.persona_fixed_assignment_analysis import _authenticate_corpus
from experiments.persona_surface_derivation import (
    SurfacePairConfig,
    _stable_hash,
    authenticate_pair_gate,
    generate_surface_pair,
    load_surface_pair_config,
    validate_kimi_model_identity,
    validate_parent_preservation,
    validate_paired_surface_mappings,
    validate_surface_mapping_response,
)


class WhitespaceTokenizer:
    """Deterministic schedule tokenizer for lineage checks."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def metadata(self) -> dict[str, str]:
        return {"name": "whitespace", "revision": "v1"}


def _root() -> Path:
    return Path(__file__).parents[1]


UNIQUE_PARENT_DIALOGUE_MARKER = "PARENT_KIMI_DIALOGUE_MUST_NOT_ENTER_CHILD_REQUESTS_7F31"


@pytest.fixture(scope="module")
def generated_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, Path]:
    root = tmp_path_factory.mktemp("kimi-pair")
    parent = root / "parent"
    shutil.copytree(_root() / "results" / "persona_conflict_conversations_v1", parent)
    events = [json.loads(line) for line in (parent / "events.jsonl").read_text().splitlines()]
    dialogue = [json.loads(line) for line in (parent / "dialogue.jsonl").read_text().splitlines()]
    events[0]["model_text"] += f"\n{UNIQUE_PARENT_DIALOGUE_MARKER}"
    dialogue[0]["text"] = events[0]["model_text"]
    for name, rows in (("events.jsonl", events), ("dialogue.jsonl", dialogue)):
        (parent / name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    manifest_path = parent / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("events.jsonl", "dialogue.jsonl"):
        manifest["artifact_sha256"][name] = hashlib.sha256(
            (parent / name).read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    output_a = root / "surface-a"
    output_b = root / "surface-b"
    gate = root / "pair-gate.json"
    config = SurfacePairConfig(
        parent_dir=parent,
        output_a=output_a,
        output_b=output_b,
        gate_path=gate,
        endpoint="https://unused.invalid",
        api_key="fixture",
        model="kimi-k3",
        timeout_seconds=1,
        mapping_max_tokens=1,
        dialogue_max_tokens=1,
        mapping_histories_per_request=1,
        events_per_request=416,
        turn_pairs_per_event=1,
        minimum_words_per_turn=1,
        max_validation_attempts=1,
        resume_existing=False,
        enable_thinking=False,
    )
    generate_surface_pair(config, FakeKimiClient())
    return parent, output_a, output_b, gate


EXPECTED = [
    {
        "history_id": "history-001",
        "category": "tea",
        "source_phrase": "cedar tea",
    },
    {
        "history_id": "history-001",
        "category": "seating",
        "source_phrase": "window seating",
    },
]


def _content(targets: tuple[str, str]) -> str:
    rows = [
        {**expected, "target_phrase": target}
        for expected, target in zip(EXPECTED, targets, strict=True)
    ]
    return json.dumps({"mapping": rows})


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (json.dumps({"mapping": []}), "coverage"),
        (
            json.dumps(
                {
                    "mapping": [
                        {**EXPECTED[1], "target_phrase": "garden seating"},
                        {**EXPECTED[0], "target_phrase": "jasmine tea"},
                    ]
                }
            ),
            "order",
        ),
        (_content(("jasmine infusion", "garden seating")), "category"),
        (_content(("jasmine tea", "garden chair")), "category"),
        (_content(("cedar tea", "garden seating")), "parent"),
        (_content(("history-001 tea", "garden seating")), "latent"),
        (_content(("one two three four five tea", "garden seating")), "length"),
    ),
)
def test_mapping_validation_rejects_adversarial_responses(
    content: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_surface_mapping_response(content, EXPECTED, {"cedar tea", "window seating"})


def test_mapping_validation_accepts_exact_complete_category_shaped_response() -> None:
    mapping = validate_surface_mapping_response(
        _content(("jasmine tea", "garden seating")),
        EXPECTED,
        {"cedar tea", "window seating"},
    )

    assert [row["target_phrase"] for row in mapping] == [
        "jasmine tea",
        "garden seating",
    ]


def test_mapping_targets_must_be_unique() -> None:
    expected = [
        EXPECTED[0],
        {
            "history_id": "history-001",
            "category": "tea",
            "source_phrase": "mint tea",
        },
    ]
    content = json.dumps(
        {
            "mapping": [
                {**row, "target_phrase": "jasmine tea"} for row in expected
            ]
        }
    )
    with pytest.raises(ValueError, match="unique"):
        validate_surface_mapping_response(content, expected, set())


@pytest.mark.parametrize(
    "b_targets",
    (
        ("jasmine tea", "courtyard seating"),
        ("iced jasmine tea", "courtyard seating"),
        ("green tea", "courtyard garden seating"),
    ),
)
def test_cross_assignment_normalized_equality_and_containment_are_rejected(
    b_targets: tuple[str, str],
) -> None:
    a = validate_surface_mapping_response(
        _content(("jasmine tea", "garden seating")), EXPECTED, set()
    )
    b = validate_surface_mapping_response(_content(b_targets), EXPECTED, set())

    with pytest.raises(ValueError, match="cross-assignment"):
        validate_paired_surface_mappings(a, b)


def test_requested_and_returned_model_must_both_be_exact_kimi_k3() -> None:
    validate_kimi_model_identity("kimi-k3", "kimi-k3")
    with pytest.raises(ValueError, match="requested model"):
        validate_kimi_model_identity("kimi-k3-preview", "kimi-k3")
    with pytest.raises(ValueError, match="returned model"):
        validate_kimi_model_identity("kimi-k3", "kimi-k3-2026")


def test_parent_preservation_rejects_latent_and_checkpoint_drift() -> None:
    parent_events = [
        {
            "event_id": "event-1",
            "history_id": "history-001",
            "sequence_index": 0,
            "fact": {"object": "value-001-a"},
            "model_text": "old",
            "surface_object": "cedar tea",
        }
    ]
    child_events = [{**parent_events[0], "model_text": "new", "surface_object": "jasmine tea"}]
    parent_queries = [
        {
            "query_id": "query-1",
            "history_id": "history-001",
            "gold": "value-001-a",
            "checkpoint_contract": {"trigger_event_id": "event-1"},
            "query_text": "old?",
            "surface_query_text": "old?",
            "surface_gold": "cedar tea",
        }
    ]
    child_queries = [
        {
            **parent_queries[0],
            "query_text": "new?",
            "surface_query_text": "new?",
            "surface_gold": "jasmine tea",
        }
    ]
    validate_parent_preservation(parent_events, child_events, parent_queries, child_queries)

    child_events[0]["sequence_index"] = 1
    with pytest.raises(ValueError, match="latent"):
        validate_parent_preservation(parent_events, child_events, parent_queries, child_queries)
    child_events[0]["sequence_index"] = 0
    child_queries[0]["checkpoint_contract"] = {"trigger_event_id": "event-2"}
    with pytest.raises(ValueError, match="checkpoint|causal"):
        validate_parent_preservation(parent_events, child_events, parent_queries, child_queries)


def test_mapping_hash_is_canonical_and_sensitive_to_order() -> None:
    mapping = validate_surface_mapping_response(
        _content(("jasmine tea", "garden seating")), EXPECTED, set()
    )
    assert _stable_hash(mapping) != _stable_hash(list(reversed(mapping)))


def test_pair_generation_config_resolves_all_runtime_and_provider_values_from_env() -> None:
    config = load_surface_pair_config(
        _root() / "configs" / "persona_surface_pair_generation.json",
        environ={
            "PERSONA_SURFACE_PARENT_DIR": "/runtime/parent",
            "PERSONA_SURFACE_A_OUTPUT_DIR": "/runtime/a",
            "PERSONA_SURFACE_B_OUTPUT_DIR": "/runtime/b",
            "PERSONA_SURFACE_PAIR_GATE_PATH": "/runtime/pair-gate.json",
            "KIMI_BASE_URL": "https://unused.invalid",
            "KIMI_API_KEY": "fixture",
            "KIMI_MODEL": "kimi-k3",
        },
    )
    assert config.parent_dir == Path("/runtime/parent")
    assert config.output_a == Path("/runtime/a")
    assert config.output_b == Path("/runtime/b")
    assert config.gate_path == Path("/runtime/pair-gate.json")
    assert config.minimum_words_per_turn == 12
    assert config.events_per_request == 2
    assert config.max_validation_attempts == 3
    assert config.dialogue_max_tokens == 16384
    assert config.mapping_max_tokens == 32768
    assert config.timeout_seconds == 600.0
    assert config.resume_existing is True
    assert config.mapping_histories_per_request == 1


def test_pair_generation_config_strictly_validates_mapping_batch_size(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (_root() / "configs" / "persona_surface_pair_generation.json").read_text()
    )
    payload["generation"]["mapping_histories_per_request"] = 0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    environ = {
        "PERSONA_SURFACE_PARENT_DIR": "/runtime/parent",
        "PERSONA_SURFACE_A_OUTPUT_DIR": "/runtime/a",
        "PERSONA_SURFACE_B_OUTPUT_DIR": "/runtime/b",
        "PERSONA_SURFACE_PAIR_GATE_PATH": "/runtime/pair-gate.json",
        "KIMI_BASE_URL": "https://unused.invalid",
        "KIMI_API_KEY": "fixture",
        "KIMI_MODEL": "kimi-k3",
    }
    with pytest.raises(ValueError, match="mapping_histories_per_request"):
        load_surface_pair_config(path, environ=environ)


def test_surface_benchmark_configs_pin_gate_path_hash_and_manifest_via_env() -> None:
    from experiments.persona_end_to_end_benchmark import load_benchmark_config

    for assignment in ("A", "B"):
        config = load_benchmark_config(
            _root()
            / "configs"
            / f"persona_end_to_end_benchmark_surface_{assignment.lower()}.json",
            environ={
                f"PERSONA_SURFACE_{assignment}_DATASET_DIR": f"/runtime/{assignment.lower()}",
                f"PERSONA_SURFACE_{assignment}_BENCHMARK_OUTPUT_DIR": f"/output/{assignment.lower()}",
                f"PERSONA_SURFACE_{assignment}_MANIFEST_SHA256": assignment.lower() * 64,
                "PERSONA_SURFACE_PAIR_GATE_PATH": "/runtime/pair-gate.json",
                "PERSONA_SURFACE_PAIR_GATE_SHA256": "f" * 64,
                "PERSONA_QWEN_MODEL_PATH": "/model",
                "PERSONA_QWEN_MODEL_ID": "Qwen/fixture",
                "PERSONA_QWEN_DEVICE": "cuda",
                "PERSONA_SCALLOP_ENDPOINT": "http://scallop.invalid",
            },
        )
        assert config.schedule.pair_gate_path == Path("/runtime/pair-gate.json")
        assert config.schedule.pair_gate_sha256 == "f" * 64
        assert config.schedule.surface_assignment == assignment


def test_parent_backed_pair_has_fresh_provenance_and_exact_manifest_contracts(
    generated_pair: tuple[Path, Path, Path, Path],
) -> None:
    parent, output_a, output_b, gate = generated_pair
    gate_hash = __import__("hashlib").sha256(gate.read_bytes()).hexdigest()
    authentication = authenticate_pair_gate(gate, gate_hash)
    assert authentication["parent_path"] == parent.resolve()
    parent_events = parent.joinpath("events.jsonl").read_text().splitlines()
    assert len(parent_events) == 416
    for assignment, output in (("A", output_a), ("B", output_b)):
        manifest = json.loads((output / "generation_manifest.json").read_text())
        assert manifest["method"] == "kimi-independent-surface-and-dialogue.v1"
        assert manifest["assignment"] == assignment
        assert manifest["seed"] == {"A": 137, "B": 911}[assignment]
        assert manifest["requested_model"] == manifest["returned_model"] == "kimi-k3"
        assert len((output / "events.jsonl").read_text().splitlines()) == 416
        assert (output / "requests.jsonl").read_bytes() != (parent / "requests.jsonl").read_bytes()
        assert (output / "raw_responses.jsonl").read_bytes() != (
            parent / "raw_responses.jsonl"
        ).read_bytes()
        for name in ("facts.jsonl", "examples.jsonl", "candidate_updates.jsonl"):
            assert (output / name).read_bytes() == (parent / name).read_bytes()


def test_mapping_prompts_are_assignment_local_before_any_response(
    generated_pair: tuple[Path, Path, Path, Path]
) -> None:
    _, output_a, output_b, _ = generated_pair
    requests = {
        label: [
            json.loads(line)
            for line in (output / "requests.jsonl").read_text().splitlines()
        ]
        for label, output in (("A", output_a), ("B", output_b))
    }
    mapping_requests = {
        label: next(row for row in rows if row["stage"] == "mapping")
        for label, rows in requests.items()
    }
    assert mapping_requests["A"]["prompt_sha256"] != mapping_requests["B"][
        "prompt_sha256"
    ]
    assert mapping_requests["A"]["messages"] != mapping_requests["B"]["messages"]
    payloads = {
        label: json.loads(row["messages"][-1]["content"])
        for label, row in mapping_requests.items()
    }
    assert payloads["A"]["realization_id"] != payloads["B"]["realization_id"]
    assert "surface-b" not in payloads["A"]["realization_id"]
    assert "surface-a" not in payloads["B"]["realization_id"]


def test_mapping_is_batched_one_eleven_row_request_per_history(
    generated_pair: tuple[Path, Path, Path, Path]
) -> None:
    _, output_a, output_b, _ = generated_pair
    for output in (output_a, output_b):
        requests = [
            json.loads(line)
            for line in (output / "requests.jsonl").read_text().splitlines()
            if json.loads(line)["stage"] == "mapping"
            and json.loads(line)["accepted"]
        ]
        assert len(requests) == 16
        assert [row["mapping_request_index"] for row in requests] == list(range(16))
        for row in requests:
            payload = json.loads(row["messages"][-1]["content"])
            assert payload["history_ids"] == row["history_ids"]
            assert len(payload["history_ids"]) == 1
            assert len(payload["mapping"]) == 11
            assert {item["history_id"] for item in payload["mapping"]} == set(
                payload["history_ids"]
            )


def test_mapping_batch_size_control_changes_request_grouping(tmp_path: Path) -> None:
    config = replace(
        _launch_order_fixture(tmp_path, attempts=1),
        mapping_histories_per_request=2,
    )
    generate_surface_pair(config, FakeKimiClient())
    for output in (config.output_a, config.output_b):
        requests = [
            json.loads(line)
            for line in (output / "requests.jsonl").read_text().splitlines()
            if json.loads(line)["stage"] == "mapping"
        ]
        assert len(requests) == 8
        for request in requests:
            payload = json.loads(request["messages"][-1]["content"])
            assert len(payload["history_ids"]) == 2
            assert len(payload["mapping"]) == 22


class RepeatUnlessForbiddenClient(FakeKimiClient):
    """Repeat assignment-local targets unless the prompt explicitly forbids them."""

    def complete(self, **kwargs: object) -> LLMResponse:
        messages = kwargs["messages"]
        payload = json.loads(messages[-1]["content"])
        response = super().complete(**kwargs)
        if "mapping" not in payload:
            return response
        assignment_stem = "amber" if "surface-a" in payload["realization_id"] else "cobalt"
        forbidden = set(payload["forbidden_targets"])
        mapping = json.loads(response.content)["mapping"]
        for index, row in enumerate(mapping):
            repeated = {
                "tea": f"{assignment_stem}{index} tea",
                "seating": f"{assignment_stem}{index} seating",
                "food": f"{assignment_stem}{index} curry",
                "delivery": f"{assignment_stem}{index} delivery",
                "receipt": f"{assignment_stem}{index} receipts",
                "private_lineage": f"{assignment_stem}{index} token",
            }[row["category"]]
            if repeated.casefold() not in forbidden:
                row["target_phrase"] = repeated
        return LLMResponse(
            content=json.dumps({"mapping": mapping}),
            finish_reason=response.finish_reason,
            model=response.model,
            usage=response.usage,
        )


def test_prior_targets_prevent_assignment_local_reuse_without_pair_retry(
    tmp_path: Path,
) -> None:
    config = _launch_order_fixture(tmp_path, attempts=2)
    generate_surface_pair(config, RepeatUnlessForbiddenClient())
    prompts = {}
    for assignment, output in (("A", config.output_a), ("B", config.output_b)):
        mapping_requests = [
            row
            for row in (
                json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()
            )
            if row["stage"] == "mapping" and row["attempt_index"] == 0
        ]
        assert len(mapping_requests) == 16
        assert {row["mapping_round"] for row in mapping_requests} == {0}
        payloads = [json.loads(row["messages"][-1]["content"]) for row in mapping_requests]
        assert [len(payload["forbidden_targets"]) for payload in payloads] == [
            index * 11 for index in range(16)
        ]
        prompts[assignment] = payloads
        manifest = json.loads((output / "generation_manifest.json").read_text())
        assert len({row["target_phrase"].casefold() for row in manifest["surface_mapping"]}) == 176
    a_targets = {
        row["target_phrase"].casefold()
        for row in json.loads((config.output_a / "generation_manifest.json").read_text())[
            "surface_mapping"
        ]
    }
    assert all(
        not (a_targets & set(payload["forbidden_targets"]))
        for payload in prompts["B"]
    )


def test_mapping_prompts_bind_distinct_assignment_round_directives(
    generated_pair: tuple[Path, Path, Path, Path]
) -> None:
    _, output_a, output_b, _ = generated_pair
    payloads = {}
    for assignment, output in (("A", output_a), ("B", output_b)):
        request = next(
            json.loads(line)
            for line in (output / "requests.jsonl").read_text().splitlines()
            if json.loads(line)["stage"] == "mapping"
        )
        payloads[assignment] = json.loads(request["messages"][-1]["content"])
        assert payloads[assignment]["lexical_directive"]
    assert payloads["A"]["lexical_directive"] != payloads["B"]["lexical_directive"]


class RepairingMappingClient(FakeKimiClient):
    """Return one live-style invalid term, then repair the same mapping batch."""

    def __init__(self, category: str, invalid_phrase: str) -> None:
        self.category = category
        self.invalid_phrase = invalid_phrase
        self.seen: set[tuple[str, int]] = set()

    def complete(self, **kwargs: object) -> LLMResponse:
        messages = kwargs["messages"]
        payload = json.loads(messages[-1]["content"])
        response = super().complete(**kwargs)
        if "mapping" not in payload or "surface-a" not in payload["realization_id"]:
            return response
        key = (payload["realization_id"], int(payload["mapping_request_index"]))
        if key in self.seen or payload["mapping_request_index"] != 0:
            return response
        self.seen.add(key)
        mapping = json.loads(response.content)["mapping"]
        next(row for row in mapping if row["category"] == self.category)[
            "target_phrase"
        ] = self.invalid_phrase
        return LLMResponse(
            content=json.dumps({"mapping": mapping}),
            finish_reason=response.finish_reason,
            model=response.model,
            usage=response.usage,
        )


@pytest.mark.parametrize(
    ("category", "invalid_phrase"),
    (
        ("tea", "jasmine pearl infusion"),
        ("seating", "quiet corner banquette"),
        ("receipt", "paperless purchase record"),
    ),
)
def test_mapping_batch_repair_prompt_names_exact_validation_failure(
    tmp_path: Path, category: str, invalid_phrase: str
) -> None:
    config = _launch_order_fixture(tmp_path, attempts=2)
    generate_surface_pair(config, RepairingMappingClient(category, invalid_phrase))
    requests = [
        json.loads(line)
        for line in (config.output_a / "requests.jsonl").read_text().splitlines()
        if json.loads(line)["stage"] == "mapping"
        and json.loads(line)["mapping_request_index"] == 0
    ]
    assert [row["attempt_index"] for row in requests] == [0, 1]
    repair_text = " ".join(
        message["content"] for message in requests[1]["messages"] if message["role"] == "system"
    )
    assert f"invalid {category} category shape: {invalid_phrase!r}" in repair_text
    assert "Tea targets must end with 'tea'" in requests[1]["messages"][0]["content"]
    assert "receipt targets must end with the plural 'receipts'" in requests[1][
        "messages"
    ][0]["content"]


class RecordingPairClient(FakeKimiClient):
    """Record mapping/dialogue order and optionally force cross-assignment overlap."""

    def __init__(self, overlap_rounds: set[int]) -> None:
        self.overlap_rounds = overlap_rounds
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, **kwargs: object) -> LLMResponse:
        messages = kwargs["messages"]
        payload = json.loads(messages[-1]["content"])
        if "mapping" not in payload:
            self.calls.append(("dialogue", "", -1))
            return super().complete(**kwargs)
        realization_id = str(payload["realization_id"])
        assignment = "A" if "surface-a" in realization_id else "B"
        mapping_round = int(realization_id.rsplit("round-", 1)[1])
        self.calls.append(("mapping", assignment, mapping_round))
        response = super().complete(**kwargs)
        if mapping_round not in self.overlap_rounds:
            return response
        mapping = json.loads(response.content)["mapping"]
        for index, row in enumerate(mapping):
            history_token = str(row["history_id"]).rsplit("-", 1)[-1]
            stem = f"shared{history_token}{index}"
            row["target_phrase"] = {
                "tea": f"{stem} tea",
                "seating": f"{stem} seating",
                "food": f"{stem} curry",
                "delivery": f"{stem} delivery",
                "receipt": f"{stem} receipts",
                "private_lineage": f"{stem} token",
            }[row["category"]]
        return LLMResponse(
            content=json.dumps({"mapping": mapping}),
            finish_reason=response.finish_reason,
            model=response.model,
            usage=response.usage,
        )


class SimulatedInterruption(BaseException):
    """Model an abrupt process loss that bypasses normal exception cleanup."""


class InterruptingDurableClient(FakeKimiClient):
    """Record successful calls and interrupt before a selected dialogue response."""

    def __init__(self, crash_after_dialogue_responses: int | None) -> None:
        self.crash_after_dialogue_responses = crash_after_dialogue_responses
        self.dialogue_response_count = 0
        self.calls: list[tuple[str, str]] = []

    def complete(self, **kwargs: object) -> LLMResponse:
        messages = kwargs["messages"]
        payload = json.loads(messages[-1]["content"])
        stage = "mapping" if "mapping" in payload else "dialogue"
        prompt_hash = _stable_hash(messages)
        if (
            stage == "dialogue"
            and self.crash_after_dialogue_responses is not None
            and self.dialogue_response_count >= self.crash_after_dialogue_responses
        ):
            raise SimulatedInterruption()
        response = super().complete(**kwargs)
        self.calls.append((stage, prompt_hash))
        if stage == "dialogue":
            self.dialogue_response_count += 1
        return response


class MappingInterruptClient(FakeKimiClient):
    """Interrupt before one mapping response and record calls after resumption."""

    def __init__(self, interrupt_at: int | None) -> None:
        self.interrupt_at = interrupt_at
        self.mapping_calls = 0

    def complete(self, **kwargs: object) -> LLMResponse:
        payload = json.loads(kwargs["messages"][-1]["content"])
        if "mapping" in payload:
            if self.interrupt_at == self.mapping_calls:
                raise SimulatedInterruption()
            self.mapping_calls += 1
        return super().complete(**kwargs)


def _launch_order_fixture(
    tmp_path: Path, *, attempts: int
) -> SurfacePairConfig:
    parent = tmp_path / "parent"
    shutil.copytree(_root() / "results" / "persona_conflict_conversations_v1", parent)
    return SurfacePairConfig(
        parent_dir=parent,
        output_a=tmp_path / "a",
        output_b=tmp_path / "b",
        gate_path=tmp_path / "pair-gate.json",
        endpoint="https://unused.invalid",
        api_key="fixture",
        model="kimi-k3",
        timeout_seconds=1,
        mapping_max_tokens=1,
        dialogue_max_tokens=1,
        mapping_histories_per_request=1,
        events_per_request=416,
        turn_pairs_per_event=1,
        minimum_words_per_turn=1,
        max_validation_attempts=attempts,
        resume_existing=False,
        enable_thinking=False,
    )


def test_cross_pair_validation_precedes_every_dialogue_call(tmp_path: Path) -> None:
    client = RecordingPairClient({0})
    config = _launch_order_fixture(tmp_path, attempts=2)

    generate_surface_pair(config, client)

    assert client.calls[:16] == [("mapping", "A", 0)] * 16
    assert client.calls[16:32] == [("mapping", "B", 0)] * 16
    assert client.calls[32:48] == [("mapping", "A", 1)] * 16
    assert client.calls[48:64] == [("mapping", "B", 1)] * 16
    assert client.calls[64][0] == "dialogue"


def test_overlap_round_retries_both_independently_with_fresh_prompts(
    tmp_path: Path,
) -> None:
    client = RecordingPairClient({0})
    config = _launch_order_fixture(tmp_path, attempts=2)
    generate_surface_pair(config, client)

    by_assignment = {}
    for assignment, output in (("A", config.output_a), ("B", config.output_b)):
        rows = [
            json.loads(line)
            for line in (output / "requests.jsonl").read_text().splitlines()
            if json.loads(line)["stage"] == "mapping"
        ]
        by_assignment[assignment] = rows
        assert [row["mapping_round"] for row in rows] == [0] * 16 + [1] * 16
        assert [row["accepted"] for row in rows] == [False] * 16 + [True] * 16
        assert [row["superseded"] for row in rows] == [True] * 16 + [False] * 16
        assert len({row["prompt_sha256"] for row in rows}) == 32
        sibling = "surface-b" if assignment == "A" else "surface-a"
        assert all(sibling not in json.dumps(row["messages"]) for row in rows)
    assert by_assignment["A"][16]["prompt_sha256"] != by_assignment["B"][16][
        "prompt_sha256"
    ]


def test_persistent_mapping_overlap_fails_without_dialogue_or_gate(
    tmp_path: Path,
) -> None:
    client = RecordingPairClient({0, 1})
    config = _launch_order_fixture(tmp_path, attempts=2)

    with pytest.raises(ValueError, match="cross-assignment"):
        generate_surface_pair(config, client)

    assert all(stage == "mapping" for stage, _, _ in client.calls)
    assert not config.gate_path.exists()
    for output in (config.output_a, config.output_b):
        manifest = json.loads((output / "generation_manifest.json").read_text())
        assert manifest["status"] == "failed"
        assert manifest["generation_controls_sha256"] == _stable_hash(
            manifest["generation_controls"]
        )
        assert manifest["request_parameters"]["mapping_histories_per_request"] == 1
        requests = [
            json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()
        ]
        assert requests and all(row["stage"] == "mapping" for row in requests)
        assert all(not row["accepted"] and row["superseded"] for row in requests)


def test_successful_mapping_pair_then_realizes_all_parent_events(tmp_path: Path) -> None:
    client = RecordingPairClient(set())
    config = _launch_order_fixture(tmp_path, attempts=1)

    generate_surface_pair(config, client)

    assert client.calls[:16] == [("mapping", "A", 0)] * 16
    assert client.calls[16:32] == [("mapping", "B", 0)] * 16
    assert all(stage == "dialogue" for stage, _, _ in client.calls[32:])
    for output in (config.output_a, config.output_b):
        events = (output / "events.jsonl").read_text().splitlines()
        assert len(events) == 416
        manifest = json.loads((output / "generation_manifest.json").read_text())
        assert manifest["event_count"] == 416


def test_generation_authenticates_parent_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.persona_surface_derivation as derivation

    client = RecordingPairClient(set())
    config = _launch_order_fixture(tmp_path, attempts=1)
    original = derivation._authenticate_parent
    calls = 0

    def recording_authentication(parent_dir: Path) -> tuple[dict, str, dict[str, bytes]]:
        nonlocal calls
        calls += 1
        return original(parent_dir)

    monkeypatch.setattr(derivation, "_authenticate_parent", recording_authentication)
    generate_surface_pair(config, client)
    assert calls == 1


def test_resume_after_mapping_reuses_both_mappings_without_duplicate_calls(
    tmp_path: Path,
) -> None:
    config = replace(
        _launch_order_fixture(tmp_path, attempts=2),
        events_per_request=100,
        resume_existing=True,
    )
    interrupted = InterruptingDurableClient(0)
    with pytest.raises(SimulatedInterruption):
        generate_surface_pair(config, interrupted)
    assert [stage for stage, _ in interrupted.calls] == ["mapping"] * 32

    resumed = InterruptingDurableClient(None)
    generate_surface_pair(config, resumed)
    assert all(stage == "dialogue" for stage, _ in resumed.calls)
    assert len(resumed.calls) == 10


def test_resume_mid_dialogue_skips_all_validated_external_calls(tmp_path: Path) -> None:
    config = replace(
        _launch_order_fixture(tmp_path, attempts=2),
        events_per_request=100,
        resume_existing=True,
    )
    interrupted = InterruptingDurableClient(6)
    with pytest.raises(SimulatedInterruption):
        generate_surface_pair(config, interrupted)
    completed_prompt_hashes = {prompt_hash for _, prompt_hash in interrupted.calls}

    resumed = InterruptingDurableClient(None)
    generate_surface_pair(config, resumed)
    resumed_prompt_hashes = {prompt_hash for _, prompt_hash in resumed.calls}
    assert all(stage == "dialogue" for stage, _ in resumed.calls)
    assert len(resumed.calls) == 4
    assert completed_prompt_hashes.isdisjoint(resumed_prompt_hashes)


@pytest.mark.parametrize("artifact", ("request", "response"))
def test_resume_rejects_tampered_cached_mapping_provenance(
    tmp_path: Path, artifact: str
) -> None:
    config = replace(
        _launch_order_fixture(tmp_path, attempts=2),
        events_per_request=100,
        resume_existing=True,
    )
    with pytest.raises(SimulatedInterruption):
        generate_surface_pair(config, InterruptingDurableClient(0))
    if artifact == "request":
        path = config.output_a / "requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["messages"][-1]["content"] += " tampered"
    else:
        path = config.output_b / "raw_responses.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["content"] += " tampered"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    with pytest.raises(ValueError, match="cached (?:request|response)"):
        generate_surface_pair(config, InterruptingDurableClient(None))


def test_resume_reapplies_current_dialogue_validator(tmp_path: Path) -> None:
    config = replace(
        _launch_order_fixture(tmp_path, attempts=2),
        events_per_request=100,
        resume_existing=True,
    )
    with pytest.raises(SimulatedInterruption):
        generate_surface_pair(config, InterruptingDurableClient(1))

    raw_path = config.output_a / "raw_responses.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    target_index = next(
        index for index, row in enumerate(raw_rows) if row["stage"] == "dialogue"
    )
    payload = json.loads(raw_rows[target_index]["content"])
    for event in payload["events"]:
        for turn in event["turns"]:
            turn["content"] = "invalid"
    raw_rows[target_index]["content"] = json.dumps(payload)
    raw_rows[target_index]["response_sha256"] = hashlib.sha256(
        raw_rows[target_index]["content"].encode()
    ).hexdigest()
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows)
    )
    manifest_path = config.output_a / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["response_sha256"][target_index] = raw_rows[target_index]["response_sha256"]
    manifest["provenance_artifact_sha256"]["raw_responses.jsonl"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    resumed = InterruptingDurableClient(None)
    generate_surface_pair(config, resumed)
    assert resumed.calls
    assert json.loads((config.output_a / "generation_manifest.json").read_text())[
        "status"
    ] == "completed"


@pytest.mark.parametrize(
    ("boundary", "completed_dialogue_responses"),
    (("requests", 0), ("raw_responses", 1), ("manifest", 1)),
)
def test_resume_reconciles_logs_ahead_at_each_atomic_replacement(
    tmp_path: Path, boundary: str, completed_dialogue_responses: int
) -> None:
    config = replace(
        _launch_order_fixture(tmp_path, attempts=2),
        events_per_request=100,
        resume_existing=True,
    )
    with pytest.raises(SimulatedInterruption):
        generate_surface_pair(
            config, InterruptingDurableClient(completed_dialogue_responses)
        )
    manifest_path = config.output_a / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if boundary == "requests":
        manifest["provenance_artifact_sha256"]["requests.jsonl"] = "0" * 64
        manifest["request_count"] = max(0, int(manifest["request_count"]) - 1)
    elif boundary == "raw_responses":
        manifest["provenance_artifact_sha256"]["raw_responses.jsonl"] = "0" * 64
        manifest["response_count"] = max(0, int(manifest["response_count"]) - 1)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    generate_surface_pair(config, InterruptingDurableClient(None))
    assert json.loads(manifest_path.read_text())["status"] == "completed"


def test_resume_reconstructs_prior_target_prompt_after_mapping_interruption(
    tmp_path: Path,
) -> None:
    config = replace(_launch_order_fixture(tmp_path, attempts=1), resume_existing=True)
    with pytest.raises(SimulatedInterruption):
        generate_surface_pair(config, MappingInterruptClient(6))
    running_manifest = json.loads(
        (config.output_a / "generation_manifest.json").read_text()
    )
    assert running_manifest["status"] == "running"
    assert running_manifest["generation_controls_sha256"] == _stable_hash(
        running_manifest["generation_controls"]
    )
    requests_before = [
        json.loads(line)
        for line in (config.output_a / "requests.jsonl").read_text().splitlines()
    ]
    assert len(requests_before) == 7
    interrupted_prompt_hash = requests_before[-1]["prompt_sha256"]
    assert len(
        json.loads(requests_before[-1]["messages"][-1]["content"])[
            "forbidden_targets"
        ]
    ) == 66

    resumed = MappingInterruptClient(None)
    generate_surface_pair(config, resumed)
    requests_after = [
        json.loads(line)
        for line in (config.output_a / "requests.jsonl").read_text().splitlines()
    ]
    assert requests_after[6]["prompt_sha256"] == interrupted_prompt_hash
    assert resumed.mapping_calls == 26


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("endpoint", "https://drift.invalid"),
        ("timeout_seconds", 2.0),
        ("mapping_max_tokens", 2),
        ("dialogue_max_tokens", 2),
        ("mapping_histories_per_request", 2),
        ("events_per_request", 17),
        ("turn_pairs_per_event", 2),
        ("minimum_words_per_turn", 2),
        ("max_validation_attempts", 3),
        ("enable_thinking", True),
    ),
)
def test_resume_rejects_generation_control_drift_before_provider_call(
    tmp_path: Path, field: str, replacement: object
) -> None:
    config = replace(
        _launch_order_fixture(tmp_path, attempts=2),
        events_per_request=100,
        resume_existing=True,
    )
    with pytest.raises(SimulatedInterruption):
        generate_surface_pair(config, InterruptingDurableClient(0))
    drifted = replace(config, **{field: replacement})
    client = InterruptingDurableClient(None)
    with pytest.raises(ValueError, match="generation controls"):
        generate_surface_pair(drifted, client)
    assert client.calls == []


def test_completed_manifests_disclose_canonical_generation_controls(
    generated_pair: tuple[Path, Path, Path, Path]
) -> None:
    _, output_a, output_b, gate = generated_pair
    control_hashes = set()
    for output in (output_a, output_b):
        manifest = json.loads((output / "generation_manifest.json").read_text())
        assert manifest["generation_controls"]["mapping_histories_per_request"] == 1
        assert manifest["request_parameters"]["mapping_histories_per_request"] == 1
        assert "api_key" not in json.dumps(manifest["generation_controls"]).casefold()
        assert "https://unused.invalid" not in json.dumps(manifest["generation_controls"])
        assert manifest["generation_controls_sha256"] == _stable_hash(
            manifest["generation_controls"]
        )
        requests = [
            json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()
        ]
        responses = [
            json.loads(line)
            for line in (output / "raw_responses.jsonl").read_text().splitlines()
        ]
        for request, response in zip(requests, responses, strict=True):
            expected_max_tokens = manifest["generation_controls"][
                "mapping_max_tokens"
                if request["stage"] == "mapping"
                else "dialogue_max_tokens"
            ]
            assert request["max_tokens"] == response["max_tokens"] == expected_max_tokens
            assert request["enable_thinking"] == response["enable_thinking"] is False
            assert request["requested_model"] == response["requested_model"] == "kimi-k3"
        control_hashes.add(manifest["generation_controls_sha256"])
    assert len(control_hashes) == 1
    gate_payload = json.loads(gate.read_text())
    assert {
        gate_payload["variants"][assignment]["generation_controls_sha256"]
        for assignment in ("A", "B")
    } == control_hashes


def test_pair_gate_rejects_mixed_generation_controls(
    generated_pair: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    parent, output_a, output_b, gate = generated_pair
    copied = tmp_path / "mixed"
    copied.mkdir()
    for source in (parent, output_a, output_b):
        shutil.copytree(source, copied / source.name)
    copied_gate = copied / gate.name
    shutil.copy2(gate, copied_gate)
    b_manifest_path = copied / output_b.name / "generation_manifest.json"
    b_manifest = json.loads(b_manifest_path.read_text())
    b_manifest["generation_controls"]["timeout_seconds"] += 1
    b_manifest["request_parameters"]["timeout_seconds"] += 1
    b_manifest["generation_controls_sha256"] = _stable_hash(
        b_manifest["generation_controls"]
    )
    b_manifest_path.write_text(json.dumps(b_manifest, indent=2, sort_keys=True) + "\n")
    gate_payload = json.loads(copied_gate.read_text())
    gate_payload["variants"]["B"]["generation_manifest_sha256"] = hashlib.sha256(
        b_manifest_path.read_bytes()
    ).hexdigest()
    gate_payload["variants"]["B"]["generation_controls_sha256"] = b_manifest[
        "generation_controls_sha256"
    ]
    copied_gate.write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="generation controls"):
        authenticate_pair_gate(
            copied_gate, hashlib.sha256(copied_gate.read_bytes()).hexdigest()
        )


def test_child_requests_never_expose_parent_kimi_dialogue(
    generated_pair: tuple[Path, Path, Path, Path]
) -> None:
    parent, output_a, output_b, _ = generated_pair
    assert UNIQUE_PARENT_DIALOGUE_MARKER in (parent / "events.jsonl").read_text()
    for output in (output_a, output_b):
        requests = [
            json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()
        ]
        assert UNIQUE_PARENT_DIALOGUE_MARKER not in json.dumps(requests)


def test_pair_gate_limits_assurance_to_latent_causality_and_defers_checkpoints(
    generated_pair: tuple[Path, Path, Path, Path]
) -> None:
    _, output_a, _, gate = generated_pair
    gate_payload = json.loads(gate.read_text())
    assert "latent_causal_preservation" in gate_payload["assurances"]["proved"]
    assert gate_payload["assurances"]["deferred"]["checkpoint_identity"] == (
        "scheduler_with_configured_tokenizer_before_qwen"
    )
    manifest = json.loads((output_a / "generation_manifest.json").read_text())
    assert manifest["preservation_scope"]["pair_gate"] == "latent_causal_structure"
    assert manifest["preservation_scope"]["scheduler"] == (
        "exact_checkpoint_indices_with_configured_tokenizer"
    )


def test_pair_gate_rejects_resigned_sibling_tampering(
    generated_pair: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    _, _, _, gate = generated_pair
    tampered_path = gate.parent / "tampered-gate.json"
    payload = json.loads(gate.read_text())
    payload["variants"]["B"]["surface_mapping_sha256"] = "0" * 64
    tampered_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    digest = __import__("hashlib").sha256(tampered_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sibling binding"):
        authenticate_pair_gate(tampered_path, digest)


def test_partial_b_failure_leaves_no_completed_pair_gate(tmp_path: Path) -> None:
    class MismatchedBClient(FakeKimiClient):
        def complete(self, **kwargs: object) -> LLMResponse:
            response = super().complete(**kwargs)
            messages = kwargs["messages"]
            seed = kwargs["seed"]
            payload = json.loads(messages[-1]["content"])
            if "mapping" in payload and int(seed) >= 500:
                return LLMResponse(
                    content=response.content,
                    finish_reason=response.finish_reason,
                    model="kimi-k3-wrong",
                    usage=response.usage,
                )
            return response

    parent = tmp_path / "parent"
    shutil.copytree(_root() / "results" / "persona_conflict_conversations_v1", parent)
    gate = tmp_path / "pair-gate.json"
    config = SurfacePairConfig(
        parent_dir=parent,
        output_a=tmp_path / "a",
        output_b=tmp_path / "b",
        gate_path=gate,
        endpoint="https://unused.invalid",
        api_key="fixture",
        model="kimi-k3",
        timeout_seconds=1,
        mapping_max_tokens=1,
        dialogue_max_tokens=1,
        mapping_histories_per_request=1,
        events_per_request=416,
        turn_pairs_per_event=1,
        minimum_words_per_turn=1,
        max_validation_attempts=1,
        resume_existing=False,
        enable_thinking=False,
    )
    with pytest.raises(ValueError, match="returned model"):
        generate_surface_pair(config, MismatchedBClient())
    assert not gate.exists()
    assert json.loads((config.output_b / "generation_manifest.json").read_text())[
        "status"
    ] == "failed"


def test_fixed_assignment_analysis_authenticates_kimi_variant_through_pair_gate(
    generated_pair: tuple[Path, Path, Path, Path]
) -> None:
    _, output_a, _, gate = generated_pair
    digest = __import__("hashlib").sha256(gate.read_bytes()).hexdigest()
    authenticated = _authenticate_corpus(output_a, "A", "A", gate, digest)
    assert authenticated["surface_mapping_by_history"]
    assert authenticated["allowed_targets_by_history"]


def test_scheduler_rejects_legacy_deterministic_surface_children() -> None:
    dataset = _root() / "results" / "persona_conflict_conversations_surface_a"
    manifest_hash = __import__("hashlib").sha256(
        (dataset / "generation_manifest.json").read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="neither a direct v1 Kimi corpus"):
        build_evaluation_schedule(
            dataset,
            WhitespaceTokenizer(),
            ScheduleConfig(
                source_split="test",
                source_profile="anti_shortcut_interleaved_v3",
                source_manifest_sha256=manifest_hash,
                seed=73,
                concurrent_accounts=8,
                min_segment_events=4,
                max_segment_events=8,
                query_suffixes=("-preference-change-delayed",),
                token_distance_thresholds=(100,),
            ),
        )


def test_schedule_preserves_parent_checkpoint_indices_but_allows_visible_distances(
    generated_pair: tuple[Path, Path, Path, Path]
) -> None:
    parent, output_a, _, gate = generated_pair
    hashlib = __import__("hashlib")
    common = dict(
        source_split="test",
        source_profile="anti_shortcut_interleaved_v3",
        seed=73,
        concurrent_accounts=8,
        min_segment_events=4,
        max_segment_events=8,
        query_suffixes=("-preference-change-delayed", "-preference-incongruity-delayed"),
        token_distance_thresholds=(100,),
    )
    parent_schedule = build_evaluation_schedule(
        parent,
        WhitespaceTokenizer(),
        ScheduleConfig(
            source_manifest_sha256=hashlib.sha256(
                (parent / "generation_manifest.json").read_bytes()
            ).hexdigest(),
            **common,
        ),
    )
    child_schedule = build_evaluation_schedule(
        output_a,
        WhitespaceTokenizer(),
        ScheduleConfig(
            source_manifest_sha256=hashlib.sha256(
                (output_a / "generation_manifest.json").read_bytes()
            ).hexdigest(),
            pair_gate_path=gate,
            pair_gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
            surface_assignment="A",
            **common,
        ),
    )
    identity_fields = (
        "evaluation_input_id",
        "checkpoint_turn_index",
        "relevant_update_turn_index",
        "relevant_update_event_id",
        "internal_gold",
        "selected_turn_ids",
    )
    assert [tuple(row[field] if field != "selected_turn_ids" else tuple(row[field]) for field in identity_fields) for row in parent_schedule["inputs"]] == [
        tuple(row[field] if field != "selected_turn_ids" else tuple(row[field]) for field in identity_fields)
        for row in child_schedule["inputs"]
    ]
    assert any(
        left["actual_token_distance"] != right["actual_token_distance"]
        for left, right in zip(parent_schedule["inputs"], child_schedule["inputs"], strict=True)
        if left["actual_token_distance"] is not None
    )
