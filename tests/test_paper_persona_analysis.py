from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

import pytest

from experiments import paper_persona_analysis as analysis


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("analysis must not contact any model or service")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_result_counters_may_differ_but_runtime_controls_must_match():
    left = {
        "graphiti_memory": {"build_id": "A", "session": {"driver": "Neo4jDriver", "server_filter_violations_dropped": 0}},
        "hybrid_memory": {"session_id": "A", "top_k": 8},
    }
    right = deepcopy(left)
    right["graphiti_memory"]["build_id"] = "B"
    right["graphiti_memory"]["session"]["server_filter_violations_dropped"] = 2
    right["hybrid_memory"]["session_id"] = "B"
    analysis._compare_runs(left, right)
    right["graphiti_memory"]["session"]["driver"] = "different-driver"
    with pytest.raises(ValueError, match="controls"):
        analysis._compare_runs(left, right)


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(assignment, digest):
    config = json.loads((ROOT / "configs/persona_end_to_end_joint_surface_a.json").read_bytes())
    config["runtime"]["surface_assignment"] = assignment
    config["runtime"]["dataset_dir_env"] = f"DATA_{assignment}"
    config["runtime"]["output_dir_env"] = f"OUTPUT_{assignment}"
    config["schedule"]["source_manifest_sha256"] = digest
    config["analysis"] = {
        "bootstrap_samples": 100, "bootstrap_seed": 73,
        "paired_comparisons": [["hybrid_kg_memory_4096", "sliding_context_4096"]],
    }
    return config


def _queries():
    return [{"history_id": f"history-{history:03d}",
             "query_id": f"history-{history:03d}-preference-{family}-delayed",
             "split": "test", "hardness_profile": "anti_shortcut_interleaved_v3"}
            for history in range(5, 17) for family in ("change", "incongruity")]


def _rows(expected, assignment):
    rows = []
    for index, (identifier, fields) in enumerate(expected.items()):
        for arm in analysis.ARMS:
            correct = arm["kind"] == "hybrid_kg_memory" and ((index // 10 < 6) == (assignment == "A"))
            answer, gold = ("Answer: tea\nignored" if correct else "coffee"), "tea"
            rows.append({**fields, "evaluation_input_id": identifier, "arm": arm["name"],
                         "arm_kind": arm["kind"], "prompt_token_cap": arm["prompt_token_cap"],
                         "answer": answer, "gold": gold,
                         "retrieval_metadata": {"checkpoint_turn_index": index + 1} if arm["kind"] == "graphiti_memory" else {},
                         **analysis._score_short_answer(answer, gold)})
    return rows


def _refresh_run(run, config=None):
    manifest = json.loads((run / "manifest.json").read_bytes())
    if config is not None:
        rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
        summary = analysis.summarize_predictions(rows, config["analysis"])
        metrics = {"condition_count": 120, "generation_count": 960,
                   "aggregates": {**summary["aggregates"], "paired_deltas": summary["paired_deltas"]}}
        _write_json(run / "metrics.json", metrics)
    manifest["artifact_sha256"] = {name: _hash(run / name) for name in ("predictions.jsonl", "metrics.json")}
    _write_json(run / "manifest.json", manifest)


def _make_run(root, assignment):
    corpus = root / "evidence" / assignment
    corpus.mkdir(parents=True)
    _write_rows(corpus / "events.jsonl", [{"event_id": "event-1"}])
    _write_rows(corpus / "queries.jsonl", _queries())
    artifacts = {name: (corpus / name).read_bytes() for name in ("events.jsonl", "queries.jsonl")}
    hashes = {name: _hash(corpus / name) for name in artifacts}
    _write_json(corpus / "generation_manifest.json", {"status": "completed", "artifact_sha256": hashes})
    provenance = {
        "generation_manifest_sha256": _hash(corpus / "generation_manifest.json"),
        "artifact_sha256": hashes, "parent_generation_manifest_sha256": analysis.PARENT_SHA256,
        "assignment": assignment, "pair_gate_sha256": analysis.PAIR_GATE_SHA256,
        "path": str(corpus), "generation_model": "kimi-k3",
        "checkpoint_policy": "authenticated_parent_exact_indices",
    }
    config = _config(assignment, provenance["generation_manifest_sha256"])
    config_path = root / f"config-{assignment}.json"
    _write_json(config_path, config)
    run = root / f"run-{assignment}"
    run.mkdir()
    expected = analysis._expected_conditions(artifacts, config)
    _write_rows(run / "predictions.jsonl", _rows(expected, assignment))
    graphiti = {key: value for key, value in config["graphiti"].items() if not key.endswith("_env")}
    graphiti.update({"build_id": assignment, "episode_failure_count": 0, "episode_failures": [],
                     "embedding_model_id": "bge", "embedding_revision": "embedding-revision",
                     "memory_store_sha256": {assignment: "c" * 64},
                     "timestamp_mapping": {"base": config["graphiti"]["timestamp_base"],
                                           "step_seconds": config["graphiti"]["timestamp_step_seconds"]}})
    manifest = {
        "status": "completed", "dataset": provenance, "config_sha256": _hash(config_path),
        "condition_count": 120, "generation_count": 960, "arms": deepcopy(analysis.ARMS),
        "prompt_instruction_sha256": hashlib.sha256(config["prompt_instruction"].encode()).hexdigest(),
        "decoding": {"do_sample": False, "enable_thinking": False, "batch_size": 1, "max_new_tokens": 32},
        "evaluator_script_sha256": "e" * 64, "evaluator_version": "persona_end_to_end_qwen.v2",
        "schedule_version": "persona-interference-schedule.v1",
        "git": {"git_head": "a" * 40, "dirty": False},
        "source_provenance": {"mode": "git", "git": {"git_head": "a" * 40, "dirty": False}},
        "scallop": {"engine": "scallopy", "rule_version": "preference_stream.v1", "version": "0.2.4"},
        "tokenizer": {"model_id": "Qwen/model", "resolved_revision": "model-revision", "implementation": "Qwen2Tokenizer", "add_special_tokens": False},
        "model": {"model_id": "Qwen/model", "resolved_revision": "model-revision",
                  "dtype": "bfloat16", "attention_implementation": "sdpa",
                  "file_sha256": {"tokenizer.json": "a" * 64, "tokenizer_config.json": "b" * 64, "model.safetensors": "c" * 64}},
        "graphiti_memory": graphiti,
        "hybrid_memory": {"top_k": 8, "hops_requested": 2, "rrf_k": 60, "embedding_batch_size": 32,
                          "embedding_model_id": "bge", "embedding_revision": "embedding-revision",
                          "validator": {"name": "scallop", "scallop_available": True, "rule_version": "rules.v1"}},
    }
    _write_json(run / "manifest.json", manifest)
    _refresh_run(run, config)
    return {"run": run, "config_path": config_path, "config": config, "corpus": corpus,
            "provenance": provenance, "artifacts": artifacts, "expected": expected}


@pytest.fixture
def pair(tmp_path, monkeypatch):
    surfaces = {assignment: _make_run(tmp_path, assignment) for assignment in ("A", "B")}

    def authenticate(evidence_root, mode, assignment, pair_gate_sha256):
        surface = surfaces[assignment]
        return surface["corpus"], surface["provenance"], surface["artifacts"]

    monkeypatch.setattr(analysis, "_authenticate_surface", authenticate)
    return surfaces


def _arguments(pair, tmp_path, mode="kimi_ab"):
    return {"mode": mode, "run_a": pair["A"]["run"], "config_a": pair["A"]["config_path"],
            "evidence_root": tmp_path / "evidence", "output_dir": tmp_path / "analysis-output",
            **({"run_b": pair["B"]["run"], "config_b": pair["B"]["config_path"]} if mode == "kimi_ab" else {})}


def test_ab_keeps_twelve_clusters_and_both_surfaces(pair, tmp_path):
    before = {path: _hash(path) for path in tmp_path.rglob("*") if path.is_file()}
    result = analysis.analyze_paper_persona(**_arguments(pair, tmp_path))
    assert {path: _hash(path) for path in before} == before
    assert result["pooled"]["row_count"] == 1920
    assert result["pooled"]["condition_count"] == 240
    assert result["pooled"]["history_cluster_count"] == 12
    delta = result["pooled"]["paired_deltas"][0]
    assert delta["paired_row_count"] == 240
    assert delta["history_cluster_count"] == 12
    assert delta["exact_match_delta_ci_95"] == [0.5, 0.5]
    assert "not independent" in result["scope"]
    pooled = analysis.pool_surfaces({key: _rows(value["expected"], key) for key, value in pair.items()})
    wrongly_independent = [{**row, "history_id": row["surface_assignment"] + row["history_id"]} for row in pooled]
    incorrect = analysis.summarize_predictions(wrongly_independent, pair["A"]["config"]["analysis"])
    assert incorrect["history_cluster_count"] == 24
    assert incorrect["paired_deltas"][0]["exact_match_delta_ci_95"] != [0.5, 0.5]
    assert len(list((tmp_path / "analysis-output").iterdir())) == 1
    assert json.loads((tmp_path / "analysis-output/analysis.json").read_bytes()) == result


@pytest.mark.parametrize("mode", ["historical_a", "kimi_a"])
def test_single_surface_does_not_claim_robustness(pair, tmp_path, mode):
    result = analysis.analyze_paper_persona(**_arguments(pair, tmp_path, mode))
    assert "pooled" not in result
    assert set(result["surfaces"]) == {"A"}
    assert "no surface-robustness" in result["scope"]
    assert result["surfaces"]["A"]["row_count"] == 960


@pytest.mark.parametrize("change,match", [
    ("config_hash", "config_sha256"), ("dataset_hash", "SHA-256 mismatch"),
    ("assignment", "dataset assignment"), ("gate", "dataset pair_gate_sha256"),
    ("parent", "dataset parent"), ("incomplete", "completed"),
    ("arms", "eight matched arms"), ("model", "A/B model"),
    ("tokenizer", "A/B tokenizer"), ("evaluator", "A/B evaluator"),
    ("source", "A/B source_provenance"), ("git", "A/B git"),
    ("scallop", "A/B scallop"), ("graphiti", "A/B graphiti_memory controls"),
    ("graphiti_missing", "Graphiti control"), ("retrieval", "retrieval top_k"),
    ("tokenizer_hash", "A/B model"), ("condition_count", "condition count"),
    ("historical_as_modern", "generation_manifest"),
])
def test_manifest_mixing_rejected_before_output(pair, tmp_path, change, match):
    path = pair["B"]["run"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    if change == "config_hash":
        manifest["config_sha256"] = "f" * 64
    elif change == "dataset_hash":
        manifest["dataset"]["artifact_sha256"]["events.jsonl"] = "f" * 64
    elif change == "assignment":
        manifest["dataset"]["assignment"] = "A"
    elif change == "gate":
        manifest["dataset"]["pair_gate_sha256"] = "f" * 64
    elif change == "parent":
        manifest["dataset"]["parent_generation_manifest_sha256"] = "f" * 64
    elif change == "incomplete":
        manifest["status"] = "running"
    elif change == "arms":
        manifest["arms"] = manifest["arms"][:4] + [{"name": "full_qwen_context", "kind": "full_qwen_context", "prompt_token_cap": None}]
    elif change == "model":
        manifest["model"]["file_sha256"]["model.safetensors"] = "f" * 64
    elif change == "tokenizer":
        manifest["tokenizer"]["implementation"] = "different"
    elif change == "evaluator":
        manifest["evaluator_script_sha256"] = "f" * 64
    elif change == "source":
        manifest["source_provenance"]["git"]["git_head"] = "b" * 40
    elif change == "git":
        manifest["git"]["git_head"] = "b" * 40
    elif change == "scallop":
        manifest["scallop"]["version"] = "different"
    elif change == "graphiti":
        manifest["graphiti_memory"]["llm_model"] = "different"
    elif change == "graphiti_missing":
        del manifest["graphiti_memory"]["num_results"]
    elif change == "retrieval":
        manifest["hybrid_memory"]["top_k"] = 9
    elif change == "tokenizer_hash":
        manifest["model"]["file_sha256"]["tokenizer.json"] = "f" * 64
    elif change == "condition_count":
        manifest["condition_count"] = 119
    elif change == "historical_as_modern":
        manifest["dataset"]["generation_manifest_sha256"] = analysis.HISTORICAL_SHA256
    _write_json(path, manifest)
    with pytest.raises(ValueError, match=match):
        analysis.analyze_paper_persona(**_arguments(pair, tmp_path))
    assert not (tmp_path / "analysis-output").exists()


@pytest.mark.parametrize("change,match", [
    ("duplicate", "duplicate"), ("missing", "960"), ("arm", "unexpected"),
    ("condition", "unexpected"), ("history", "history_id"), ("family", "query_family"),
    ("phase", "phase"), ("score", "stored score"), ("answer", "stored score"),
    ("checkpoint", "checkpoint identity"), ("checkpoint_missing", "checkpoint identity"),
])
def test_prediction_mixing_rejected(pair, tmp_path, change, match):
    path = pair["B"]["run"] / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "missing":
        rows.pop()
    elif change == "arm":
        rows[0]["arm"] = "full_qwen_context"
    elif change == "condition":
        rows[0]["evaluation_input_id"] = "invented"
    elif change == "history":
        rows[0]["history_id"] = "history-024"
    elif change == "family":
        rows[0]["query_family"] = "other"
    elif change == "phase":
        rows[0]["phase"] = "other"
    elif change == "score":
        rows[0]["exact_match"] = 1.0
    elif change == "answer":
        rows[0]["answer"] = "tea"
    elif change == "checkpoint":
        for row in rows:
            if row["arm_kind"] == "graphiti_memory":
                row["retrieval_metadata"]["checkpoint_turn_index"] += 1
    elif change == "checkpoint_missing":
        rows[6]["retrieval_metadata"] = {}
    _write_rows(path, rows)
    _refresh_run(pair["B"]["run"])
    with pytest.raises(ValueError, match=match):
        analysis.analyze_paper_persona(**_arguments(pair, tmp_path))
    assert not (tmp_path / "analysis-output").exists()


@pytest.mark.parametrize("field", ["prompt_instruction", "retrieval", "graphiti", "generation", "schedule"])
def test_normalization_preserves_scientific_controls(field):
    left = _config("A", analysis.MODERN_SHA256["A"])
    right = _config("B", analysis.MODERN_SHA256["B"])
    assert analysis.normalize_scientific_config(left) == analysis.normalize_scientific_config(right)
    if isinstance(right[field], dict):
        right[field]["scientific_control"] = "altered"
    else:
        right[field] = "altered"
    assert analysis.normalize_scientific_config(left) != analysis.normalize_scientific_config(right)


def test_config_change_with_updated_hash_still_rejected(pair, tmp_path):
    config = pair["B"]["config"]
    config["schedule"]["seed"] += 1
    _write_json(pair["B"]["config_path"], config)
    path = pair["B"]["run"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["config_sha256"] = _hash(pair["B"]["config_path"])
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="scientific config"):
        analysis.analyze_paper_persona(**_arguments(pair, tmp_path))
    assert not (tmp_path / "analysis-output").exists()


def test_different_surface_wording_gold_and_token_distances_allowed(pair, tmp_path):
    path = pair["B"]["run"] / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["gold"] = "mint"
        row["answer"] = "Answer: mint" if row["exact_match"] else "wrong"
        row["query_text"] = "A differently worded query"
        row["actual_token_distance"] = 10001
        row.update(analysis._score_short_answer(row["answer"], row["gold"]))
    _write_rows(path, rows)
    _refresh_run(pair["B"]["run"], pair["B"]["config"])
    assert analysis.analyze_paper_persona(**_arguments(pair, tmp_path))["pooled"]["history_cluster_count"] == 12


@pytest.mark.parametrize("kind", ["predictions", "metrics", "corpus"])
def test_actual_artifact_byte_mutation_is_rejected(pair, tmp_path, kind):
    path = {"predictions": pair["A"]["run"] / "predictions.jsonl",
            "metrics": pair["A"]["run"] / "metrics.json", "corpus": pair["A"]["corpus"] / "events.jsonl"}[kind]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        analysis.analyze_paper_persona(**_arguments(pair, tmp_path))
    assert not (tmp_path / "analysis-output").exists()


def test_aggregate_tampering_with_updated_hash_rejected(pair, tmp_path):
    path = pair["A"]["run"] / "metrics.json"
    metrics = json.loads(path.read_bytes())
    metrics["aggregates"]["by_arm"][0]["exact_match"] = 0.123
    _write_json(path, metrics)
    _refresh_run(pair["A"]["run"])
    with pytest.raises(ValueError, match="stored aggregates"):
        analysis.analyze_paper_persona(**_arguments(pair, tmp_path))


@pytest.mark.parametrize("case", ["missing_b", "duplicate_b", "extra_b", "existing_output", "run_output", "evidence_output", "source_output", "frozen_output"])
def test_modes_and_output_safety(pair, tmp_path, case):
    kwargs = _arguments(pair, tmp_path)
    if case == "missing_b":
        del kwargs["run_b"]
    elif case == "duplicate_b":
        kwargs["run_b"] = kwargs["run_a"]
    elif case == "extra_b":
        kwargs["mode"] = "kimi_a"
    elif case == "existing_output":
        kwargs["output_dir"].mkdir()
    elif case == "run_output":
        kwargs["output_dir"] = kwargs["run_a"] / "analysis"
    elif case == "evidence_output":
        kwargs["output_dir"] = kwargs["evidence_root"] / "analysis"
    elif case == "source_output":
        kwargs["output_dir"] = ROOT / "never-create-paper-analysis"
    elif case == "frozen_output":
        frozen = tmp_path / "frozen"
        frozen.mkdir()
        _write_json(frozen / "freeze_manifest.json", {})
        kwargs["output_dir"] = frozen / "analysis"
    with pytest.raises(ValueError):
        analysis.analyze_paper_persona(**kwargs)
    assert not (kwargs["output_dir"] / "analysis.json").exists()


def test_cli_defaults_and_printed_json(pair, tmp_path, capsys):
    kwargs = _arguments(pair, tmp_path, "historical_a")
    argv = [part for key, value in kwargs.items() if key != "mode" for part in ("--" + key.replace("_", "-"), str(value))]
    assert analysis.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "historical_a"
    assert printed["pooled"] is None
    assert printed["surfaces"]["A"]["history_cluster_count"] == 12


@pytest.fixture(scope="module")
def bundled_corpora(tmp_path_factory):
    destination = tmp_path_factory.mktemp("paper-analysis-bundled-inputs")
    paths = ["results/persona_surface_pair_gate.json"]
    for name in (*analysis.CORPORA.values(), "results/persona_conflict_conversations_v1"):
        manifest_name = f"{name}/generation_manifest.json"
        manifest = json.loads((ROOT / manifest_name).read_bytes())
        paths.extend([manifest_name, *(f"{name}/{artifact}" for artifact in manifest["artifact_sha256"])])
    for name in paths:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    return destination


@pytest.mark.parametrize("mode,assignment", [("historical_a", "A"), ("kimi_a", "A"), ("kimi_ab", "B")])
def test_no_runtime_authentication_of_bundled_inputs(bundled_corpora, mode, assignment):
    corpus, provenance, artifacts = analysis._authenticate_surface(bundled_corpora, mode, assignment, analysis.PAIR_GATE_SHA256)
    assert corpus.is_dir()
    assert provenance["parent_generation_manifest_sha256"] == analysis.PARENT_SHA256
    assert artifacts["queries.jsonl"]
    expected = analysis.HISTORICAL_SHA256 if mode == "historical_a" else analysis.MODERN_SHA256[assignment]
    assert provenance["generation_manifest_sha256"] == expected
    conditions = analysis._expected_conditions(artifacts, _config(assignment, expected))
    assert len(conditions) == 120
    assert len({row["history_id"] for row in conditions.values()}) == 12


def test_real_gate_rejects_wrong_digest(bundled_corpora):
    with pytest.raises(ValueError, match="pair gate hash mismatch"):
        analysis._authenticate_surface(bundled_corpora, "kimi_a", "A", "f" * 64)


def test_historical_corpus_cannot_be_authenticated_as_modern(bundled_corpora):
    corpus = bundled_corpora / analysis.CORPORA["historical_a"]
    gate = bundled_corpora / "results/persona_surface_pair_gate.json"
    with pytest.raises(ValueError, match="gate-bound assignment path"):
        analysis.authenticate_pair_gate(gate, analysis.PAIR_GATE_SHA256, dataset_dir=corpus, expected_assignment="A")
    with pytest.raises(ValueError, match="historical deterministic surfaces"):
        analysis._authenticate_dataset(corpus, analysis.HISTORICAL_SHA256, gate, analysis.PAIR_GATE_SHA256, "A")


def test_gate_rejects_a_dataset_as_b(bundled_corpora):
    with pytest.raises(ValueError, match="gate-bound assignment path"):
        analysis.authenticate_pair_gate(bundled_corpora / "results/persona_surface_pair_gate.json",
                                        analysis.PAIR_GATE_SHA256,
                                        dataset_dir=bundled_corpora / analysis.CORPORA["A"], expected_assignment="B")


@pytest.mark.parametrize("change", ["missing_b", "historical_in_modern_folder"])
def test_real_gate_rejects_corpus_mixing(bundled_corpora, tmp_path, change):
    root = tmp_path / "evidence"
    shutil.copytree(bundled_corpora, root)
    if change == "missing_b":
        shutil.rmtree(root / analysis.CORPORA["B"])
    else:
        shutil.rmtree(root / analysis.CORPORA["A"])
        shutil.copytree(root / analysis.CORPORA["historical_a"], root / analysis.CORPORA["A"])
    with pytest.raises(ValueError):
        analysis._authenticate_surface(root, "kimi_a", "A", analysis.PAIR_GATE_SHA256)


def test_anonymized_historical_fixture_and_canonical_config_contract(bundled_corpora, tmp_path):
    _, provenance, _ = analysis._authenticate_surface(bundled_corpora, "historical_a", "A", analysis.PAIR_GATE_SHA256)
    manifest_bytes = (ROOT / "tests/fixtures/historical_persona_manifest.json").read_bytes()
    # Pin the sanitized test fixture, not the untouched external run manifest.
    assert hashlib.sha256(manifest_bytes).hexdigest() == "9284a94c9a731d0b3a355e16582e2964b9bec704e2825abfb0941ea61dd73f54"
    manifest = json.loads(manifest_bytes)
    assert "git" not in manifest
    assert manifest["anonymization"]["purpose"].startswith("Test fixture only;")
    assert not Path(manifest["dataset"]["path"]).is_absolute()
    assert not Path(manifest["dataset"]["parent_path"]).is_absolute()
    # This test checks the configuration contract, not execution provenance.
    # Supply an explicitly synthetic identity in memory; never ship it as a run.
    manifest["git"] = {"git_head": "0" * 40, "dirty": False}
    config_bytes = (ROOT / "configs/persona_end_to_end_joint_surface_a.json").read_bytes()
    config_path = tmp_path / "canonical-config.json"
    config_path.write_bytes(config_bytes)
    analysis._validate_config(manifest, json.loads(config_bytes), config_path, provenance, "historical_a", "A")


def test_generation_manifest_scientific_mismatch_rejected(pair, tmp_path):
    run = pair["A"]["run"]
    manifest = json.loads((run / "manifest.json").read_bytes())
    generation = deepcopy(manifest)
    generation["prompt_instruction_sha256"] = "f" * 64
    _write_json(run / "generation_manifest.json", generation)
    manifest["generation_manifest_sha256"] = _hash(run / "generation_manifest.json")
    manifest["artifact_sha256"]["generation_manifest.json"] = manifest["generation_manifest_sha256"]
    _write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="generation manifest prompt_instruction"):
        analysis.analyze_paper_persona(**_arguments(pair, tmp_path))
    assert not (tmp_path / "analysis-output").exists()


@pytest.fixture
def ab_table_result():
    result = {"status": "completed", "mode": "kimi_ab", "surfaces": {}}
    kinds = {kind: index for index, kind in enumerate((
        "sliding_context", "structured_memory", "graphiti_memory", "hybrid_kg_memory"
    ))}
    for assignment, offset in (("A", 0.0), ("B", 0.05)):
        rows = []
        for arm in analysis.ARMS:
            score = 0.1 * (kinds[arm["kind"]] + 1) + offset + (0.1 if arm["prompt_token_cap"] == 16384 else 0.0)
            rows.append({"arm": arm["name"], "exact_match": score, "f1": score + 0.025,
                         "row_count": 120, "history_cluster_count": 12})
        result["surfaces"][assignment] = {
            "row_count": 960, "condition_count": 120, "history_cluster_count": 12,
            "aggregates": {"by_arm": rows[::-1]},
        }
    rows = []
    for left, right in zip(result["surfaces"]["A"]["aggregates"]["by_arm"], result["surfaces"]["B"]["aggregates"]["by_arm"]):
        rows.append({"arm": left["arm"], "row_count": 240, "history_cluster_count": 12,
                     **{key: (left[key] + right[key]) / 2 for key in ("exact_match", "f1")}})
    result["pooled"] = {"row_count": 1920, "condition_count": 240, "history_cluster_count": 12,
                        "aggregates": {"by_arm": rows}}
    return result


def test_ab_table_markdown_has_three_panels_and_computed_maxima(ab_table_result):
    before = deepcopy(ab_table_result)
    table = analysis.render_primary_table(ab_table_result)
    rows = [line for line in table.splitlines() if line.startswith("|")]
    assert len(rows) == 6
    assert "Surface A 4K EM" in rows[0] and "Surface B 4K F1" in rows[0] and "Combined 16K F1" in rows[0]
    assert rows[-1] == "| Hybrid KG memory | **40.00** | **42.50** | **50.00** | **52.50** | **45.00** | **47.50** | **55.00** | **57.50** | **42.50** | **45.00** | **52.50** | **55.00** |"
    assert [row.split("|")[1].strip() for row in rows[2:]] == ["Sliding context", "Structured memory", "Graphiti", "Hybrid KG memory"]
    assert "240 conditions" in table and "12 base-history clusters" in table
    assert ab_table_result == before


def test_ab_table_latex_matches_grouped_paper_layout(ab_table_result):
    table = analysis.render_primary_table(ab_table_result, "latex")
    assert table.startswith(r"\begin{table}[H]")
    assert r"\begin{tabular}{lcccccccccccc}" in table
    for label in ("Surface A", "Surface B", "Combined"):
        assert rf"\multicolumn{{4}}{{c}}{{\textbf{{{label}}}}}" in table
    assert r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}" in table
    assert r"Hybrid KG memory & \textbf{40.00} & \textbf{42.50}" in table
    assert table.endswith(r"\end{table}")
    body = table.split(r"\midrule")[1].split(r"\bottomrule")[0]
    assert all(row.count("&") == 12 for row in body.strip().splitlines())


def test_ab_table_combined_rounding_uses_unrounded_scores(ab_table_result):
    for panel, values in (("A", (49 / 120, 63 / 120)), ("B", (48 / 120, 70 / 120)), ("pooled", (97 / 240, 133 / 240))):
        summary = ab_table_result["pooled"] if panel == "pooled" else ab_table_result["surfaces"][panel]
        for budget, value in zip((4096, 16384), values):
            row = next(row for row in summary["aggregates"]["by_arm"] if row["arm"] == f"graphiti_memory_{budget}")
            row["exact_match"] = value
    table = analysis.render_primary_table(ab_table_result)
    cells = next(line for line in table.splitlines() if line.startswith("| Graphiti |")).split("|")
    assert cells[10].strip(" *") == "40.42"
    assert cells[12].strip(" *") == "55.42"


@pytest.mark.parametrize("change", ["missing_b", "missing_pooled", "missing_f1", "duplicate_arm", "extra_arm", "bad_clusters", "bad_count", "nan", "out_of_range", "bad_combined", "historical", "incomplete"])
def test_ab_table_rejects_incomplete_or_inconsistent_summaries(ab_table_result, change):
    if change == "missing_b":
        del ab_table_result["surfaces"]["B"]
    elif change == "missing_pooled":
        del ab_table_result["pooled"]
    elif change == "missing_f1":
        del ab_table_result["surfaces"]["B"]["aggregates"]["by_arm"][0]["f1"]
    elif change in {"duplicate_arm", "extra_arm"}:
        rows = ab_table_result["surfaces"]["B"]["aggregates"]["by_arm"]
        rows.append({**rows[0], **({"arm": "full_qwen_context"} if change == "extra_arm" else {})})
    elif change == "bad_clusters":
        ab_table_result["pooled"]["history_cluster_count"] = 24
    elif change == "bad_count":
        ab_table_result["pooled"]["condition_count"] = 120
    elif change in {"nan", "out_of_range", "bad_combined"}:
        ab_table_result["pooled"]["aggregates"]["by_arm"][0]["f1"] = {"nan": float("nan"), "out_of_range": 1.1, "bad_combined": 0.99}[change]
    elif change == "historical":
        ab_table_result["mode"] = "historical_a"
    else:
        ab_table_result["status"] = "running"
    with pytest.raises((ValueError, KeyError)):
        analysis.render_primary_table(ab_table_result)


@pytest.mark.parametrize("table_format", ["markdown", "latex"])
def test_ab_table_cli_validates_and_formats_results(pair, tmp_path, capsys, table_format):
    kwargs = _arguments(pair, tmp_path)
    argv = [part for key, value in kwargs.items() for part in ("--" + key.replace("_", "-"), str(value))]
    assert analysis.main([*argv, "--table-format", table_format]) == 0
    captured = capsys.readouterr()
    assert "Surface A" in captured.out and "Surface B" in captured.out and "Combined" in captured.out
    assert "Hybrid KG memory" in captured.out and not captured.err
    result = json.loads((tmp_path / "analysis-output/analysis.json").read_bytes())
    assert result["pooled"]["row_count"] == 1920 and result["pooled"]["history_cluster_count"] == 12
    assert len(list((tmp_path / "analysis-output").iterdir())) == 1


def test_ab_table_cli_rejects_mismatched_runs_without_output(pair, tmp_path, capsys):
    path = pair["B"]["run"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["model"]["resolved_revision"] = "different"
    manifest["tokenizer"]["resolved_revision"] = "different"
    _write_json(path, manifest)
    kwargs = _arguments(pair, tmp_path)
    argv = [part for key, value in kwargs.items() for part in ("--" + key.replace("_", "-"), str(value))]
    with pytest.raises(SystemExit) as error:
        analysis.main([*argv, "--table-format", "latex"])
    assert error.value.code == 2
    assert not capsys.readouterr().out
    assert not (tmp_path / "analysis-output").exists()


@pytest.mark.parametrize("mode", ["historical_a", "kimi_a"])
def test_ab_table_cli_requires_both_surfaces_before_analysis(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(analysis, "analyze_paper_persona", lambda **kwargs: pytest.fail("single-surface table request reached analysis"))
    argv = ["--mode", mode, "--table-format", "markdown"]
    for flag in ("run-a", "config-a", "evidence-root", "output-dir"):
        argv.extend(["--" + flag, str(tmp_path / flag)])
    with pytest.raises(SystemExit) as error:
        analysis.main(argv)
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_readme_primary_workflow_uses_both_surfaces():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    primary = readme.split("### Table 1", 1)[1].split("### Continual", 1)[0]
    assert "--mode kimi_ab" in primary and "--run-b" in primary and "--config-b" in primary
    assert "--table-format markdown" in primary and "--table-format latex" in primary
    assert "1,920" in primary and "12 base-history clusters" in primary
    assert "Historical single-surface reference" in primary


def _readme_table_rows(heading):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split(heading + "\n\n", 1)[1].split("\n\n", 1)[0]
    return [[cell.strip().strip("*") for cell in line.strip("|").split("|")]
            for line in block.splitlines()[2:]]


@pytest.mark.parametrize("heading,expected", [
    ("**Exact match (%)**", [
        ["Sliding context", "32.50", "50.00", "28.33", "55.83", "30.42", "52.92"],
        ["Structured memory", "62.50", "60.83", "59.17", "63.33", "60.83", "62.08"],
        ["Graphiti", "40.83", "52.50", "40.00", "58.33", "40.42", "55.42"],
        ["Hybrid KG memory", "69.17", "71.67", "65.83", "66.67", "67.50", "69.17"],
    ]),
    ("**Token F1 (%)**", [
        ["Sliding context", "35.83", "55.64", "31.44", "59.19", "33.64", "57.42"],
        ["Structured memory", "64.50", "64.00", "61.03", "65.19", "62.76", "64.60"],
        ["Graphiti", "47.28", "57.92", "48.15", "61.64", "47.71", "59.78"],
        ["Hybrid KG memory", "71.19", "72.57", "68.86", "67.92", "70.03", "70.24"],
    ]),
])
def test_readme_primary_scores_match_published_ab_results(heading, expected):
    assert _readme_table_rows(heading) == expected


def test_readme_combined_intervals_match_published_analysis():
    assert _readme_table_rows("**Combined paired exact-match effects (percentage points)**") == [
        ["Hybrid - sliding", "4K", "+37.08", "[30.42, 44.58]"],
        ["Hybrid - structured", "4K", "+6.67", "[-0.42, 15.00]"],
        ["Hybrid - Graphiti", "4K", "+27.08", "[19.17, 35.42]"],
        ["Hybrid - sliding", "16K", "+16.25", "[7.08, 25.00]"],
        ["Hybrid - structured", "16K", "+7.08", "[1.25, 13.33]"],
        ["Hybrid - Graphiti", "16K", "+13.75", "[5.42, 21.25]"],
    ]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "667c26a53b8a3455fa0ba0f91f0bff60e30e2bfe/results/persona_joint_kimi_ab_a100_build1/analysis/analysis.json" in readme
    assert "have not been imported" not in readme
    assert "The 4K hybrid-versus-structured EM interval includes zero" in readme
