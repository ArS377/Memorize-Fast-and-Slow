from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from conftest import FakeKimiClient
from experiments.persona_end_to_end_benchmark import _score_short_answer
from experiments.persona_fixed_assignment_analysis import analyze_sources as _analyze_sources
from experiments.persona_conversation_generator import SOURCE_ARTIFACTS
from experiments.persona_surface_derivation import SurfacePairConfig, generate_surface_pair


ARMS = (
    "sliding_context_4096",
    "sliding_context_16384",
    "structured_memory_4096",
    "structured_memory_16384",
    "full_qwen_context",
)
_PAIR_CORPORA: dict[str, Path] = {}
_PAIR_GATE: Path | None = None
_PAIR_GATE_SHA256: str | None = None


def _root() -> Path:
    """Return the repository root independently of the process CWD."""
    return Path(__file__).parents[1]


@pytest.fixture(scope="module", autouse=True)
def _kimi_pair_corpora(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Generate one authenticated fake-client Kimi pair for all analysis tests."""
    global _PAIR_CORPORA, _PAIR_GATE, _PAIR_GATE_SHA256
    root = tmp_path_factory.mktemp("analysis-kimi-pair")
    parent = root / "parent"
    shutil.copytree(_root() / "results" / "persona_conflict_conversations_v1", parent)
    output_a = root / "surface-a"
    output_b = root / "surface-b"
    gate = root / "pair-gate.json"
    generate_surface_pair(
        SurfacePairConfig(
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
            dialogue_assignment_concurrency=1,
            events_per_request=416,
            turn_pairs_per_event=1,
            minimum_words_per_turn=1,
            max_validation_attempts=1,
            resume_existing=False,
            enable_thinking=False,
        ),
        FakeKimiClient(),
    )
    _PAIR_CORPORA = {"A": output_a, "B": output_b}
    _PAIR_GATE = gate
    _PAIR_GATE_SHA256 = hashlib.sha256(gate.read_bytes()).hexdigest()


def analyze_sources(*args: object, **kwargs: object) -> dict:
    """Run analysis with the module-scoped authenticated pair gate."""
    if _PAIR_GATE is None or _PAIR_GATE_SHA256 is None:
        raise RuntimeError("Kimi pair fixture is not initialized")
    kwargs.setdefault("pair_gate_path", _PAIR_GATE)
    kwargs.setdefault("pair_gate_sha256", _PAIR_GATE_SHA256)
    return _analyze_sources(*args, **kwargs)


def _corpus(assignment: str) -> Path:
    """Return one authenticated materialized assignment corpus."""
    return _PAIR_CORPORA[assignment.upper()]


def _assignment_corpora() -> dict[str, Path]:
    """Return the explicit A/B corpus paths used by analysis fixtures."""
    return {label: _corpus(label) for label in ("A", "B")}


def _corpus_dataset(assignment: str) -> dict:
    """Return the result-facing identity of one authenticated corpus fixture."""
    corpus_path = _corpus(assignment)
    manifest_path = corpus_path / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _PAIR_GATE_SHA256 is None:
        raise RuntimeError("Kimi pair fixture is not initialized")
    return {
        "artifact_sha256": {
            name: manifest["artifact_sha256"][name]
            for name in (*SOURCE_ARTIFACTS, "dialogue.jsonl")
        },
        "checkpoint_policy": "authenticated_parent_exact_indices",
        "method": manifest["method"],
        "assignment": assignment.upper(),
        "seed": manifest["seed"],
        "pair_gate_sha256": _PAIR_GATE_SHA256,
        "generation_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "generation_model": manifest["model_identity"],
        "parent_generation_manifest_sha256": manifest["parent"][
            "generation_manifest_sha256"
        ],
        "path": str(corpus_path),
    }


def _write_source(path: Path, assignment: str, *, contaminate: bool = False) -> Path:
    path.mkdir()
    generation_rows = []
    prediction_rows = []
    for history_index in range(5, 17):
        history_id = f"history-{history_index:03d}"
        source_present = (history_index - 5) % 2 == 1
        for arm in ARMS:
            budget = arm.rsplit("_", 1)[-1]
            prompt_identity = (
                f"{assignment}:{history_id}:{budget}:prompt"
                if not source_present
                and arm.startswith(("sliding_context_", "structured_memory_"))
                else f"{assignment}:{history_id}:{arm}:prompt"
            )
            score = (
                1.0
                if contaminate
                and not source_present
                and arm == "structured_memory_4096"
                else 0.0
            )
            generation = {
                "answer": "UNKNOWN" if score else "wrong",
                "arm": arm,
                "condition": "delayed_probe" if source_present else "pre_update",
                "evaluation_input_id": f"{history_id}:probe",
                "gold": "UNKNOWN",
                "generation_seconds": 1.0,
                "history_id": history_id,
                "input_token_count": 100,
                "prompt_sha256": prompt_identity,
                "query_family": "preference_change",
                "selected_turn_count": 10,
                "structured_source_turn_count": (
                    1 if source_present and arm.startswith("structured_memory_") else 0
                ),
            }
            generation_rows.append(generation)
            prediction_rows.append(
                {
                    **generation,
                    "short_answer": generation["answer"],
                    "exact_match": score,
                    "f1": score,
                }
            )
    generations = path / "generations.jsonl"
    generations.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in generation_rows),
        encoding="utf-8",
    )
    predictions = path / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    common_manifest = {
        "arms": [{"name": arm} for arm in ARMS],
        "condition_count": 12,
        "dataset": _corpus_dataset(assignment),
        "decoding": {"do_sample": False, "max_new_tokens": 32},
        "evaluator_script_sha256": "evaluator-script",
        "evaluator_version": "persona_end_to_end_qwen.v1",
        "generation_count": 60,
        "injection_map_sha256": "injection-map",
        "model": {"model_id": "fixture-model", "resolved_revision": "fixture-revision"},
        "prompt_instruction_sha256": "prompt-instruction",
        "scallop": {"rule_version": "preference_stream.v1"},
        "schedule_version": "persona-interference-schedule.v1",
        "status": "completed",
        "tokenizer": {"model_id": "fixture-model", "resolved_revision": "fixture-revision"},
    }
    generation_manifest = {
        **common_manifest,
        "artifact_sha256": {
            "generations.jsonl": hashlib.sha256(generations.read_bytes()).hexdigest()
        },
    }
    generation_manifest_path = path / "generation_manifest.json"
    generation_manifest_path.write_text(
        json.dumps(generation_manifest, sort_keys=True), encoding="utf-8"
    )
    generation_manifest_sha256 = hashlib.sha256(
        generation_manifest_path.read_bytes()
    ).hexdigest()
    manifest = {
        **common_manifest,
        "generation_manifest_sha256": generation_manifest_sha256,
        "artifact_sha256": {
            "generation_manifest.json": generation_manifest_sha256,
            "generations.jsonl": hashlib.sha256(generations.read_bytes()).hexdigest(),
            "predictions.jsonl": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        },
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return path


def _rewrite_predictions(path: Path, rows: list[dict]) -> None:
    """Rewrite and re-authenticate a synthetic predictions artifact."""
    predictions = path / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["predictions.jsonl"] = hashlib.sha256(
        predictions.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _rewrite_generation_bundle(
    path: Path, generation_rows: list[dict], prediction_rows: list[dict]
) -> None:
    """Rewrite and re-authenticate matching generation and prediction rows."""
    generations = path / "generations.jsonl"
    generations.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in generation_rows),
        encoding="utf-8",
    )
    predictions = path / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    generation_manifest_path = path / "generation_manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    generation_manifest["artifact_sha256"]["generations.jsonl"] = hashlib.sha256(
        generations.read_bytes()
    ).hexdigest()
    generation_manifest_path.write_text(
        json.dumps(generation_manifest, sort_keys=True), encoding="utf-8"
    )
    generation_manifest_sha256 = hashlib.sha256(
        generation_manifest_path.read_bytes()
    ).hexdigest()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_manifest_sha256"] = generation_manifest_sha256
    manifest["artifact_sha256"].update(
        {
            "generation_manifest.json": generation_manifest_sha256,
            "generations.jsonl": hashlib.sha256(generations.read_bytes()).hexdigest(),
            "predictions.jsonl": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _relabel_as_b(path: Path) -> None:
    """Relabel copied A manifests as B while preserving copied result content."""
    generation_manifest_path = path / "generation_manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    generation_manifest["dataset"]["assignment"] = "B"
    generation_manifest["dataset"]["seed"] = 911
    generation_manifest_path.write_text(
        json.dumps(generation_manifest, sort_keys=True), encoding="utf-8"
    )
    generation_manifest_sha256 = hashlib.sha256(
        generation_manifest_path.read_bytes()
    ).hexdigest()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["assignment"] = "B"
    manifest["dataset"]["seed"] = 911
    manifest["generation_manifest_sha256"] = generation_manifest_sha256
    manifest["artifact_sha256"]["generation_manifest.json"] = generation_manifest_sha256
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _bind_result_fixture_to_corpus(path: Path, assignment: str) -> None:
    """Re-sign copied result manifests around another corpus identity."""
    dataset = _corpus_dataset(assignment)
    generation_manifest_path = path / "generation_manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    generation_manifest["dataset"] = dataset
    generation_manifest_path.write_text(
        json.dumps(generation_manifest, sort_keys=True), encoding="utf-8"
    )
    generation_manifest_sha256 = hashlib.sha256(
        generation_manifest_path.read_bytes()
    ).hexdigest()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"] = dataset
    manifest["generation_manifest_sha256"] = generation_manifest_sha256
    manifest["artifact_sha256"]["generation_manifest.json"] = generation_manifest_sha256
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _rebind_committed_result_surfaces(path: Path, assignment: str) -> None:
    """Translate committed deterministic result golds onto the fake Kimi mapping."""
    old_manifest = json.loads(
        (
            _root()
            / "results"
            / f"persona_conflict_conversations_surface_{assignment.lower()}"
            / "generation_manifest.json"
        ).read_text()
    )
    new_manifest = json.loads((_corpus(assignment) / "generation_manifest.json").read_text())
    old_source_by_target = {
        (row["history_id"], row["target_phrase"]): row["source_phrase"]
        for row in old_manifest["surface_mapping"]
    }
    new_target_by_source = {
        (row["history_id"], row["source_phrase"]): row["target_phrase"]
        for row in new_manifest["surface_mapping"]
    }
    generations = [
        json.loads(line)
        for line in (path / "generations.jsonl").read_text().splitlines()
    ]
    for row in generations:
        if row["gold"] != "UNKNOWN":
            source = old_source_by_target[(row["history_id"], row["gold"])]
            row["gold"] = new_target_by_source[(row["history_id"], source)]
        row["answer"] = "definitely wrong"
    predictions = [
        {**row, **_score_short_answer(row["answer"], row["gold"])}
        for row in generations
    ]
    _rewrite_generation_bundle(path, generations, predictions)
    _bind_result_fixture_to_corpus(path, assignment)


def _remove_evaluation_input(path: Path, evaluation_input_id: str) -> None:
    """Remove one full arm set and consistently re-sign the result bundle."""
    generation_rows = [
        json.loads(line)
        for line in (path / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prediction_rows = [
        json.loads(line)
        for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    generation_rows = [
        row for row in generation_rows if row["evaluation_input_id"] != evaluation_input_id
    ]
    prediction_rows = [
        row for row in prediction_rows if row["evaluation_input_id"] != evaluation_input_id
    ]
    condition_count = len({row["evaluation_input_id"] for row in generation_rows})
    generation_count = len(generation_rows)
    for name in ("generation_manifest.json", "manifest.json"):
        manifest_path = path / name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["condition_count"] = condition_count
        manifest["generation_count"] = generation_count
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    _rewrite_generation_bundle(path, generation_rows, prediction_rows)


def test_pooled_bootstrap_keeps_both_assignments_in_12_history_clusters(
    tmp_path: Path,
) -> None:
    source_a = _write_source(tmp_path / "a", "A")
    source_b = _write_source(tmp_path / "b", "B")

    report = analyze_sources(
        baseline_path=None,
        assignment_paths={"A": source_a, "B": source_b},
        baseline_corpus_path=None,
        assignment_corpus_paths=_assignment_corpora(),
        bootstrap_samples=50,
        bootstrap_seed=7,
    )

    design = report["design"]
    assert design["bootstrap_cluster_count"] == 12
    assert design["pooled_assignment_history_row_count"] == 24
    assert design["bootstrap_unit"] == "base_history_id"
    assert design["replicas_kept_together"] is True
    assert all(
        assignments == ["A", "B"]
        for assignments in design["cluster_assignments"].values()
    )
    assert not any(
        "Exact A/B gold semantics" in claim for claim in report["bounded_claims"]
    )


def test_source_absent_nonzero_effect_fails_loudly(tmp_path: Path) -> None:
    source_a = _write_source(tmp_path / "a", "A", contaminate=True)
    source_b = _write_source(tmp_path / "b", "B")

    with pytest.raises(ValueError, match="source-absent.*nonzero"):
        analyze_sources(
            baseline_path=None,
            assignment_paths={"A": source_a, "B": source_b},
            baseline_corpus_path=None,
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=20,
            bootstrap_seed=7,
        )


def test_duplicate_assignment_directory_fails_loudly(tmp_path: Path) -> None:
    source_a = _write_source(tmp_path / "a", "A")

    with pytest.raises(ValueError, match="distinct directories"):
        analyze_sources(
            baseline_path=None,
            assignment_paths={"A": source_a, "B": source_a},
            baseline_corpus_path=None,
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=20,
            bootstrap_seed=7,
        )


def test_incomplete_assignment_b_fails_loudly(tmp_path: Path) -> None:
    source_a = _write_source(tmp_path / "a", "A")
    source_b = _write_source(tmp_path / "b", "B")
    rows = [
        json.loads(line)
        for line in (source_b / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    removed = next(
        row
        for row in rows
        if row["evaluation_input_id"] == "history-005:probe"
        and row["arm"] == "full_qwen_context"
    )
    rows.remove(removed)
    generation_rows = [
        json.loads(line)
        for line in (source_b / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    generation_rows.remove(
        next(
            row
            for row in generation_rows
            if row["evaluation_input_id"] == "history-005:probe"
            and row["arm"] == "full_qwen_context"
        )
    )
    _rewrite_generation_bundle(source_b, generation_rows, rows)

    with pytest.raises(ValueError, match="evaluation-input key sets differ|missing arms"):
        analyze_sources(
            baseline_path=None,
            assignment_paths={"A": source_a, "B": source_b},
            baseline_corpus_path=None,
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=20,
            bootstrap_seed=7,
        )


def test_source_absent_prompt_mismatch_fails_loudly(tmp_path: Path) -> None:
    source_a = _write_source(tmp_path / "a", "A")
    source_b = _write_source(tmp_path / "b", "B")
    rows = [
        json.loads(line)
        for line in (source_a / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    structured = next(
        row
        for row in rows
        if row["evaluation_input_id"] == "history-005:probe"
        and row["arm"] == "structured_memory_4096"
    )
    sliding = next(
        row
        for row in rows
        if row["evaluation_input_id"] == "history-005:probe"
        and row["arm"] == "sliding_context_4096"
    )
    structured["prompt_sha256"] = sliding["prompt_sha256"] + ":modified"
    generation_rows = [
        json.loads(line)
        for line in (source_a / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    generation_structured = next(
        row
        for row in generation_rows
        if row["evaluation_input_id"] == "history-005:probe"
        and row["arm"] == "structured_memory_4096"
    )
    generation_structured["prompt_sha256"] = structured["prompt_sha256"]
    _rewrite_generation_bundle(source_a, generation_rows, rows)

    with pytest.raises(ValueError, match="source-absent.*prompt_sha256"):
        analyze_sources(
            baseline_path=None,
            assignment_paths={"A": source_a, "B": source_b},
            baseline_corpus_path=None,
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=20,
            bootstrap_seed=7,
        )


def test_copied_assignment_a_relabelled_as_b_with_timing_drift_fails_loudly(
    tmp_path: Path,
) -> None:
    source_a = _write_source(tmp_path / "a", "A")
    source_b = tmp_path / "b"
    shutil.copytree(source_a, source_b)
    _relabel_as_b(source_b)
    _bind_result_fixture_to_corpus(source_b, "B")
    generation_rows = [
        json.loads(line)
        for line in (source_b / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prediction_rows = [
        json.loads(line)
        for line in (source_b / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    generation_rows[0]["generation_seconds"] = 2.0
    prediction_rows[0]["generation_seconds"] = 2.0
    mappings = {
        assignment: json.loads(
            (_corpus(assignment) / "generation_manifest.json").read_text(encoding="utf-8")
        )["surface_mapping"]
        for assignment in ("A", "B")
    }
    b_targets = {
        row["target_phrase"]
        for row in mappings["B"]
        if row["history_id"] == generation_rows[0]["history_id"]
    }
    copied_a_gold = next(
        row["target_phrase"]
        for row in mappings["A"]
        if row["history_id"] == generation_rows[0]["history_id"]
        and row["target_phrase"] not in b_targets
    )
    generation_rows[0]["gold"] = copied_a_gold
    prediction_rows[0]["gold"] = copied_a_gold
    _rewrite_generation_bundle(source_b, generation_rows, prediction_rows)

    with pytest.raises(ValueError, match="authenticated target set"):
        analyze_sources(
            baseline_path=None,
            assignment_paths={"A": source_a, "B": source_b},
            baseline_corpus_path=None,
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=20,
            bootstrap_seed=7,
        )


def test_resigned_prediction_score_tampering_fails_loudly(tmp_path: Path) -> None:
    source_a = _write_source(tmp_path / "a", "A")
    source_b = _write_source(tmp_path / "b", "B")
    rows = [
        json.loads(line)
        for line in (source_b / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["exact_match"] = 1.0
    rows[0]["f1"] = 1.0
    _rewrite_predictions(source_b, rows)

    with pytest.raises(ValueError, match="canonical score|generation row"):
        analyze_sources(
            baseline_path=None,
            assignment_paths={"A": source_a, "B": source_b},
            baseline_corpus_path=None,
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=20,
            bootstrap_seed=7,
        )


def test_assignment_a_corpus_cannot_be_used_as_v1(tmp_path: Path) -> None:
    source_a = _write_source(tmp_path / "a", "A")
    source_b = _write_source(tmp_path / "b", "B")
    baseline = _write_source(tmp_path / "baseline-a", "A")

    with pytest.raises(ValueError, match="v1 corpus.*Kimi-authored.*non-derived"):
        analyze_sources(
            baseline_path=baseline,
            assignment_paths={"A": source_a, "B": source_b},
            baseline_corpus_path=_corpus("A"),
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=20,
            bootstrap_seed=7,
        )


def test_assignment_gold_must_match_v1_oracle_mapping(tmp_path: Path) -> None:
    root = _root() / "results"
    source_a = tmp_path / "surface-a"
    source_b = tmp_path / "surface-b"
    shutil.copytree(root / "persona_end_to_end_qwen35_4b_surface_a", source_a)
    shutil.copytree(root / "persona_end_to_end_qwen35_4b_surface_b", source_b)
    _rebind_committed_result_surfaces(source_a, "A")
    _rebind_committed_result_surfaces(source_b, "B")
    generation_rows = [
        json.loads(line)
        for line in (source_b / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prediction_rows = [
        json.loads(line)
        for line in (source_b / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target = next(row for row in generation_rows if row["condition"] == "delayed_probe")
    evaluation_input_id = target["evaluation_input_id"]
    history_id = target["history_id"]
    mapping = json.loads(
        (_corpus("B") / "generation_manifest.json").read_text(encoding="utf-8")
    )["surface_mapping"]
    alternate_gold = next(
        row["target_phrase"]
        for row in mapping
        if row["history_id"] == history_id and row["target_phrase"] != target["gold"]
    )
    for generation in generation_rows:
        if generation["evaluation_input_id"] == evaluation_input_id:
            generation["answer"] = "definitely wrong"
            generation["gold"] = alternate_gold
    for prediction in prediction_rows:
        if prediction["evaluation_input_id"] == evaluation_input_id:
            prediction["answer"] = "definitely wrong"
            prediction["gold"] = alternate_gold
            prediction.update(_score_short_answer(prediction["answer"], alternate_gold))
    _rewrite_generation_bundle(source_b, generation_rows, prediction_rows)

    with pytest.raises(ValueError, match="exact gold mismatch"):
        analyze_sources(
            baseline_path=root / "persona_end_to_end_qwen35_4b_v1",
            assignment_paths={
                "A": source_a,
                "B": source_b,
            },
            baseline_corpus_path=root / "persona_conflict_conversations_v1",
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=1,
            bootstrap_seed=7,
        )


def test_coordinated_wrong_v1_and_assignment_golds_fail_corpus_oracle(
    tmp_path: Path,
) -> None:
    root = _root() / "results"
    result_names = {
        "v1": "persona_end_to_end_qwen35_4b_v1",
        "A": "persona_end_to_end_qwen35_4b_surface_a",
        "B": "persona_end_to_end_qwen35_4b_surface_b",
    }
    sources = {}
    for label, name in result_names.items():
        destination = tmp_path / label
        shutil.copytree(root / name, destination)
        sources[label] = destination
    for label in ("A", "B"):
        _rebind_committed_result_surfaces(sources[label], label)
    generation_rows = {
        label: [
            json.loads(line)
            for line in (path / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for label, path in sources.items()
    }
    prediction_rows = {
        label: [
            json.loads(line)
            for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for label, path in sources.items()
    }
    target = next(row for row in generation_rows["v1"] if row["condition"] == "delayed_probe")
    evaluation_input_id = target["evaluation_input_id"]
    history_id = target["history_id"]
    mappings = {
        label: json.loads(
            (_corpus(label) / "generation_manifest.json").read_text(encoding="utf-8")
        )["surface_mapping"]
        for label in ("A", "B")
    }
    wrong_source_gold = next(
        row["source_phrase"]
        for row in mappings["A"]
        if row["history_id"] == history_id and row["source_phrase"] != target["gold"]
    )
    wrong_golds = {"v1": wrong_source_gold}
    for label in ("A", "B"):
        wrong_golds[label] = next(
            row["target_phrase"]
            for row in mappings[label]
            if row["history_id"] == history_id and row["source_phrase"] == wrong_source_gold
        )
    for label in ("v1", "A", "B"):
        for generation in generation_rows[label]:
            if generation["evaluation_input_id"] == evaluation_input_id:
                generation["answer"] = "definitely wrong"
                generation["gold"] = wrong_golds[label]
        for prediction in prediction_rows[label]:
            if prediction["evaluation_input_id"] == evaluation_input_id:
                prediction["answer"] = "definitely wrong"
                prediction["gold"] = wrong_golds[label]
                prediction.update(
                    _score_short_answer(prediction["answer"], wrong_golds[label])
                )
        _rewrite_generation_bundle(
            sources[label], generation_rows[label], prediction_rows[label]
        )

    with pytest.raises(ValueError, match="v1 corpus oracle.*exact gold"):
        analyze_sources(
            baseline_path=sources["v1"],
            assignment_paths={"A": sources["A"], "B": sources["B"]},
            baseline_corpus_path=root / "persona_conflict_conversations_v1",
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=1,
            bootstrap_seed=7,
        )


def test_coordinated_missing_v1_and_assignment_input_fails_complete_design(
    tmp_path: Path,
) -> None:
    root = _root() / "results"
    result_names = {
        "v1": "persona_end_to_end_qwen35_4b_v1",
        "A": "persona_end_to_end_qwen35_4b_surface_a",
        "B": "persona_end_to_end_qwen35_4b_surface_b",
    }
    sources = {}
    for label, name in result_names.items():
        destination = tmp_path / label
        shutil.copytree(root / name, destination)
        sources[label] = destination
    for label in ("A", "B"):
        _rebind_committed_result_surfaces(sources[label], label)
    baseline_rows = [
        json.loads(line)
        for line in (sources["v1"] / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    omitted_input = next(
        row["evaluation_input_id"] for row in baseline_rows if row["condition"] == "pre_update"
    )
    for source in sources.values():
        _remove_evaluation_input(source, omitted_input)

    with pytest.raises(ValueError, match="complete fixed design"):
        analyze_sources(
            baseline_path=sources["v1"],
            assignment_paths={"A": sources["A"], "B": sources["B"]},
            baseline_corpus_path=root / "persona_conflict_conversations_v1",
            assignment_corpus_paths=_assignment_corpora(),
            bootstrap_samples=1,
            bootstrap_seed=7,
        )
