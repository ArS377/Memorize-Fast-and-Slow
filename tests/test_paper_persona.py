from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from experiments import paper_persona as paper


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    names = [paper.HISTORICAL_PATH, paper.PARENT_PATH, *(row[1] for row in paper.MODERN_SURFACES)]
    root = tmp_path_factory.mktemp("paper bundled evidence")
    source = Path(__file__).resolve().parents[1]
    for name in names:
        shutil.copytree(source / name, root / name)
    (root / paper.PAIR_GATE_PATH).write_bytes((source / paper.PAIR_GATE_PATH).read_bytes())
    assert hashlib.sha256((root / paper.PAIR_GATE_PATH).read_bytes()).hexdigest() == paper.PAIR_GATE_SHA256
    return root


@pytest.fixture
def boundaries(monkeypatch, evidence):
    calls = []

    def gate(path, digest, *, dataset_dir, expected_assignment):
        assert path == evidence / paper.PAIR_GATE_PATH
        assert digest == paper.PAIR_GATE_SHA256
        assignment, relative, expected = next(row for row in paper.MODERN_SURFACES if row[0] == expected_assignment)
        assert dataset_dir == evidence / relative
        assert hashlib.sha256((dataset_dir / "generation_manifest.json").read_bytes()).hexdigest() == expected
        calls.append(("gate", assignment))
        return {
            "parent_generation_manifest_sha256": paper.PARENT_SHA256,
            "parent_path": evidence / paper.PARENT_PATH,
            "assignment_manifest_sha256": {row[0]: row[2] for row in paper.MODERN_SURFACES},
        }

    def dataset(path, digest, *args):
        assert hashlib.sha256((path / "generation_manifest.json").read_bytes()).hexdigest() == digest
        if path == evidence / paper.HISTORICAL_PATH:
            assert digest == paper.HISTORICAL_SHA256
            assert not args
            assignment = "historical"
        else:
            assignment, relative, expected = next(row for row in paper.MODERN_SURFACES if path == evidence / row[1])
            assert digest == expected
            assert args == (evidence / paper.PAIR_GATE_PATH, paper.PAIR_GATE_SHA256, assignment)
        calls.append(("dataset", assignment))
        return {}, {"queries.jsonl": (path / "queries.jsonl").read_bytes()}, {
            "path": evidence / paper.PARENT_PATH,
            "generation_manifest_sha256": paper.PARENT_SHA256,
        }

    monkeypatch.setattr(paper, "authenticate_pair_gate", gate)
    monkeypatch.setattr(paper, "_authenticate_dataset", dataset)
    return calls


def _args(evidence, tmp_path, **extra):
    return {"evidence_root": evidence, "output_root": tmp_path / "new outputs", "run_id": "run_01", **extra}


def _runtime_env():
    template = paper._read_json(paper.ROOT / paper.TEMPLATE_CONFIG)
    env = {
        value: "fixture-value"
        for section in template.values() if isinstance(section, dict)
        for key, value in section.items() if key.endswith("_env")
    }
    env["PERSONA_NEO4J_DATABASE"] = "hybrid-fixture"
    env["PERSONA_GRAPHITI_NEO4J_DATABASE"] = "graphiti-fixture"
    env["PERSONA_NEO4J_PASSWORD"] = "NEVER-PRINT-THIS-SECRET"
    env["PERSONA_GRAPHITI_LLM_API_KEY"] = "NEVER-PRINT-THIS-SECRET"
    env["PERSONA_HYBRID_RETRIEVAL_CACHE"] = "historical-cache-do-not-reuse.json"
    env["PERSONA_GRAPHITI_RETRIEVAL_CACHE"] = "historical-graphiti-do-not-reuse.json"
    return env


def test_registry_pins_and_original_template():
    registry = paper.load_mode_registry()
    assert registry["default_mode"] == "historical_a"
    assert registry["modes"]["historical_a"]["surfaces"][0]["dataset_manifest_sha256"] == paper.HISTORICAL_SHA256
    template = paper._read_json(paper.ROOT / paper.TEMPLATE_CONFIG)
    assert template["schedule"]["source_manifest_sha256"] == paper.HISTORICAL_SHA256
    assert len(template["arms"]) == 8


@pytest.mark.parametrize("mode,count,dispatch", [
    ("historical_a", 960, [("dataset", "historical")]),
    ("kimi_a", 960, [("gate", "A"), ("dataset", "A")]),
    ("kimi_ab", 1920, [("gate", "A"), ("dataset", "A"), ("gate", "B"), ("dataset", "B")]),
])
def test_preview_is_read_only_with_no_runtime_or_subprocess(evidence, tmp_path, boundaries, monkeypatch, mode, count, dispatch):
    monkeypatch.setattr(paper.subprocess, "run", lambda *a, **k: pytest.fail("preview started a subprocess"))
    args = _args(evidence, tmp_path, mode=mode)
    plan = paper.run(**args, environ={})
    assert not args["output_root"].exists()
    assert plan["version"] == 1
    assert plan["generation_count"] == count
    assert boundaries == dispatch
    assert ("pair_gate_sha256" in plan) == (mode != "historical_a")
    for surface in plan["surfaces"]:
        assert surface["condition_count"] == 120
        assert surface["generation_count"] == 960
        assert surface["config_path"] == f"configs/{surface['run_dir']}.json"


@pytest.mark.parametrize("mode", paper.MODES)
def test_actual_bundled_corpora_authenticate_read_only(evidence, tmp_path, mode):
    plan = paper.build_plan(**_args(evidence, tmp_path, mode=mode))
    assert plan["generation_count"] == (1920 if mode == "kimi_ab" else 960)
    assert not (tmp_path / "new outputs").exists()


def test_prepare_only_writes_plan_and_identical_controls(evidence, tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(paper.subprocess, "run", lambda *a, **k: pytest.fail("prepare started a subprocess"))
    args = _args(evidence, tmp_path, mode="kimi_ab")
    plan = paper.run(**args, prepare=True, environ={})
    runset = args["output_root"] / "kimi_ab" / args["run_id"]
    assert sorted(path.relative_to(runset).as_posix() for path in runset.rglob("*") if path.is_file()) == [
        "configs/surface_a.json", "configs/surface_b.json", "plan.json",
    ]
    assert not (runset / "surface_a").exists()
    assert paper._read_json(runset / "plan.json") == plan
    template = paper._read_json(paper.ROOT / paper.TEMPLATE_CONFIG)
    for surface in plan["surfaces"]:
        config = paper._read_json(runset / surface["config_path"])
        assert paper._controls(config) == paper._controls(template)
        assert config["runtime"]["dataset_dir_env"] == "PAPER_DATASET_DIR"
        assert config["runtime"]["output_dir_env"] == "PAPER_OUTPUT_DIR"
        assert config["runtime"]["surface_assignment"] == surface["assignment"]
        assert config["runtime"]["pair_gate_sha256_env"] == "PAPER_PAIR_GATE_SHA256"
        assert config["schedule"]["source_manifest_sha256"] == surface["dataset_manifest_sha256"]
        assert config["generation"] == template["generation"]
        assert config["arms"] == template["arms"]
        assert "fixture-value" not in json.dumps(config)
    with pytest.raises(ValueError, match="already exists"):
        paper.run(**args, prepare=True)


def test_execute_validates_all_surfaces_before_dispatch_and_isolates_paths(evidence, tmp_path, boundaries, monkeypatch):
    from experiments import persona_end_to_end_benchmark as benchmark

    validated = []
    original = benchmark.load_benchmark_config

    def validate(path, *, environ):
        config = original(path, environ=environ)
        validated.append(config)
        return config

    calls = []

    def subprocess_run(command, **kwargs):
        assert len(validated) == 2
        calls.append((command, kwargs))

    monkeypatch.setattr(benchmark, "load_benchmark_config", validate)
    monkeypatch.setattr(paper.subprocess, "run", subprocess_run)
    args = _args(evidence, tmp_path, mode="kimi_ab")
    env = _runtime_env()
    unchanged = dict(env)
    plan = paper.run(**args, execute=True, environ=env)
    assert env == unchanged
    runset = args["output_root"] / "kimi_ab" / args["run_id"]
    assert len(calls) == 2
    for surface, (command, kwargs) in zip(plan["surfaces"], calls, strict=True):
        assert command == [paper.sys.executable, "-m", "experiments.persona_end_to_end_benchmark", "--config", str(runset / surface["config_path"])]
        assert kwargs["cwd"] == paper.ROOT
        assert kwargs["check"] is True
        assert kwargs["stdout"] == subprocess.DEVNULL
        env = kwargs["env"]
        assert env["PAPER_PAIR_GATE_SHA256"] == paper.PAIR_GATE_SHA256
        assert env["PERSONA_NEO4J_PASSWORD"] == "NEVER-PRINT-THIS-SECRET"
        assert env["PAPER_GRAPHITI_BUILD_ID"] == f"kimi_ab__run_01__{surface['run_dir']}"
        for key in ("PAPER_OUTPUT_DIR", "PAPER_RETRIEVAL_INDEX_ROOT"):
            assert Path(env[key]).is_dir()
            assert Path(env[key]).is_relative_to(runset / surface["run_dir"])
        for key in ("PERSONA_HYBRID_RETRIEVAL_CACHE", "PERSONA_GRAPHITI_RETRIEVAL_CACHE"):
            path = Path(env[key])
            assert path.parent.is_dir()
            assert path.is_relative_to(runset / surface["run_dir"])
            assert not path.exists()
    for key in ("PAPER_DATASET_DIR", "PAPER_OUTPUT_DIR", "PAPER_RETRIEVAL_INDEX_ROOT", "PAPER_GRAPHITI_BUILD_ID", "PERSONA_HYBRID_RETRIEVAL_CACHE", "PERSONA_GRAPHITI_RETRIEVAL_CACHE"):
        assert calls[0][1]["env"][key] != calls[1][1]["env"][key]
    assert "NEVER-PRINT-THIS-SECRET" not in (runset / "plan.json").read_text()
    with pytest.raises(ValueError, match="already exists"):
        paper.run(**args, execute=True, environ=_runtime_env())


def test_missing_runtime_variable_prevents_every_subprocess(evidence, tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(paper.subprocess, "run", lambda *a, **k: pytest.fail("runtime validation must precede subprocess"))
    with pytest.raises(ValueError, match="runtime validation failed"):
        paper.run(**_args(evidence, tmp_path, mode="kimi_ab"), execute=True, environ={})


def test_invalid_second_surface_prevents_first_subprocess(evidence, tmp_path, boundaries, monkeypatch):
    original = paper.generated_config

    def malformed(plan, surface):
        config = original(plan, surface)
        if surface["assignment"] == "B":
            config["runtime"]["model_path_env"] = "MISSING_SECOND_SURFACE_MODEL"
        return config

    monkeypatch.setattr(paper, "generated_config", malformed)
    monkeypatch.setattr(paper.subprocess, "run", lambda *a, **k: pytest.fail("surface A ran before validating B"))
    with pytest.raises(ValueError, match="runtime validation failed for surface B"):
        paper.run(**_args(evidence, tmp_path, mode="kimi_ab"), execute=True, environ=_runtime_env())


@pytest.mark.parametrize("run_id", ["", "..", "../escape", "a/b", "a\\b", "a b", "a.b", "a:", "x\n", "é", "a" * 65, "CON", "nul", "COM1", "LPT9"])
def test_invalid_run_ids_are_rejected_before_writes(evidence, tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        paper.run(**_args(evidence, tmp_path, run_id=run_id), prepare=True)
    assert not (tmp_path / "new outputs").exists()


@pytest.mark.parametrize("existing_kind", ["file", "empty", "partial", "completed"])
def test_all_existing_runsets_are_rejected(evidence, tmp_path, existing_kind):
    args = _args(evidence, tmp_path)
    runset = args["output_root"] / "historical_a" / "run_01"
    runset.parent.mkdir(parents=True)
    if existing_kind == "file":
        runset.write_text("existing")
    else:
        runset.mkdir()
        if existing_kind in {"partial", "completed"}:
            (runset / "manifest.json").write_text(json.dumps({"status": existing_kind}))
    with pytest.raises(ValueError, match="already exists|not a directory|completed"):
        paper.run(**args, prepare=True)


@pytest.mark.parametrize("where", [paper.HISTORICAL_PATH, paper.PARENT_PATH, paper.MODERN_SURFACES[0][1], "experiments", "tests", "configs", ".git", "docs"])
def test_output_refuses_evidence_and_source_paths(evidence, tmp_path, where):
    with pytest.raises(ValueError, match="overlaps"):
        paper.run(**_args(evidence, tmp_path, output_root=evidence / where / "new"), prepare=True)


def test_output_refuses_frozen_bundle_and_original_paper_results(evidence, tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "freeze_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen bundle"):
        paper.run(**_args(evidence, tmp_path, output_root=frozen / "new"), prepare=True)
    protected = paper.paper_evidence_roots(paper.ROOT)
    assert protected
    with pytest.raises(ValueError, match="frozen root|completed|input"):
        paper.run(**_args(evidence, tmp_path, output_root=protected[0] / "new"), prepare=True)


@pytest.mark.parametrize("mutation", ["historic_as_modern", "swapped", "path", "gate", "extra_mode", "version"])
def test_registry_refuses_mode_mix_and_substitution(tmp_path, mutation):
    registry = deepcopy(paper.load_mode_registry())
    if mutation == "historic_as_modern":
        registry["modes"]["kimi_a"]["surfaces"] = registry["modes"]["historical_a"]["surfaces"]
    elif mutation == "swapped":
        registry["modes"]["kimi_ab"]["surfaces"].reverse()
    elif mutation == "path":
        registry["modes"]["historical_a"]["surfaces"][0]["dataset_relative_path"] = "../outside"
    elif mutation == "gate":
        registry["pair_gate_sha256"] = "0" * 64
    elif mutation == "version":
        registry["version"] = True
    else:
        registry["modes"]["other"] = registry["modes"]["historical_a"]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="pinned"):
        paper.load_mode_registry(path)


@pytest.mark.parametrize("target", [paper.HISTORICAL_PATH, paper.MODERN_SURFACES[0][1]])
def test_raw_byte_manifest_mutation_is_not_normalized(evidence, tmp_path, target):
    relocated = tmp_path / "changed evidence"
    shutil.copytree(evidence, relocated)
    path = relocated / target / "generation_manifest.json"
    original = path.read_bytes()
    assert b"\r\n" not in original
    path.write_bytes(original.replace(b"\n", b"\r\n"))
    mode = "historical_a" if target == paper.HISTORICAL_PATH else "kimi_a"
    with pytest.raises(ValueError, match="hash mismatch"):
        paper.run(**_args(relocated, tmp_path, mode=mode), prepare=True)
    assert not (tmp_path / "new outputs").exists()


def test_historic_corpus_cannot_replace_modern(evidence, tmp_path):
    relocated = tmp_path / "substituted evidence"
    shutil.copytree(evidence, relocated)
    modern = relocated / paper.MODERN_SURFACES[0][1]
    shutil.rmtree(modern)
    shutil.copytree(relocated / paper.HISTORICAL_PATH, modern)
    with pytest.raises(ValueError, match="hash mismatch|historical"):
        paper.run(**_args(relocated, tmp_path, mode="kimi_a"), prepare=True)


@pytest.mark.parametrize("side,artifact", [(paper.MODERN_SURFACES[1][1], "dialogue.jsonl"), (paper.PARENT_PATH, "events.jsonl")])
def test_modern_authentication_rechecks_sibling_and_parent_bytes(evidence, tmp_path, side, artifact):
    relocated = tmp_path / "tampered pair"
    shutil.copytree(evidence, relocated)
    path = relocated / side / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|differ"):
        paper.run(**_args(relocated, tmp_path, mode="kimi_a"), prepare=True)
    assert not (tmp_path / "new outputs").exists()


def test_cli_defaults_to_preview_and_historical_without_secrets(evidence, tmp_path, boundaries, monkeypatch, capsys):
    monkeypatch.setenv("PERSONA_NEO4J_PASSWORD", "NEVER-PRINT-THIS-SECRET")
    monkeypatch.setattr(paper.subprocess, "run", lambda *a, **k: pytest.fail("default CLI executed a subprocess"))
    output = tmp_path / "cli outputs"
    assert paper.main(["--evidence-root", str(evidence), "--output-root", str(output), "--run-id", "cli-1"]) == 0
    captured = capsys.readouterr()
    assert "NEVER-PRINT-THIS-SECRET" not in captured.out + captured.err
    assert json.loads(captured.out)["mode"] == "historical_a"
    assert not output.exists()
    with pytest.raises(SystemExit):
        paper.main(["--output-root", str(output), "--run-id", "cli-1", "--execute", "--prepare"])


def test_modes_and_run_ids_never_share_builds_or_paths(evidence, tmp_path, boundaries):
    identities = set()
    for mode in paper.MODES:
        for run_id in ("first", "second"):
            args = _args(evidence, tmp_path, mode=mode, run_id=run_id)
            plan = paper.run(**args, prepare=True, environ={})
            runset = args["output_root"] / mode / run_id
            for surface in plan["surfaces"]:
                env = paper.surface_environment(plan, surface, evidence_root=evidence, runset=runset, environ={})
                for key in ("PAPER_GRAPHITI_BUILD_ID", "PAPER_OUTPUT_DIR", "PAPER_RETRIEVAL_INDEX_ROOT", "PERSONA_HYBRID_RETRIEVAL_CACHE", "PERSONA_GRAPHITI_RETRIEVAL_CACHE"):
                    assert env[key] not in identities
                    identities.add(env[key])
                if mode == "historical_a":
                    config = paper._read_json(runset / surface["config_path"])
                    assert "pair_gate_path_env" not in config["runtime"]
                    assert "surface_assignment" not in config["runtime"]
                    assert "PAPER_PAIR_GATE_PATH" not in env


@pytest.mark.parametrize("prepare", [False, True])
def test_optional_model_input_paths_are_protected_without_requiring_secrets(evidence, tmp_path, boundaries, prepare):
    args = _args(evidence, tmp_path)
    with pytest.raises(ValueError, match="overlaps input"):
        paper.run(**args, prepare=prepare, environ={"PERSONA_QWEN_MODEL_PATH": str(args["output_root"])})
    assert not args["output_root"].exists()


def test_configured_freeze_root_is_protected(evidence, tmp_path):
    selection = paper._read_json(paper.ROOT / "configs/paper_artifacts.json")
    with pytest.raises(ValueError, match="frozen root"):
        paper.run(**_args(evidence, tmp_path, output_root=Path(selection["freeze_root"]) / "new"), prepare=True)


@pytest.mark.parametrize("mutation", ["raw_bytes", "paths"])
def test_pair_gate_hash_and_path_bindings_are_strict(evidence, tmp_path, mutation):
    relocated = tmp_path / "modified gate"
    shutil.copytree(evidence, relocated)
    gate_path = relocated / paper.PAIR_GATE_PATH
    if mutation == "raw_bytes":
        gate_path.write_bytes(gate_path.read_bytes() + b" ")
    else:
        gate = paper._read_json(gate_path)
        gate["paths"]["A"] = "./" + gate["paths"]["A"]
        gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="hash mismatch|paths differ"):
        paper.run(**_args(relocated, tmp_path, mode="kimi_a"), prepare=True)
    assert not (tmp_path / "new outputs").exists()


def test_unknown_mode_and_mutually_exclusive_actions_fail_without_writes(evidence, tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        paper.run(**_args(evidence, tmp_path, mode="other"), prepare=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        paper.run(**_args(evidence, tmp_path), prepare=True, execute=True)
    assert not (tmp_path / "new outputs").exists()


def test_failed_evaluator_does_not_start_next_surface(evidence, tmp_path, boundaries, monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(paper.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        paper.run(**_args(evidence, tmp_path, mode="kimi_ab"), execute=True, environ=_runtime_env())
    assert len(calls) == 1


@pytest.mark.platform_specific
def test_output_symlink_is_rejected(evidence, tmp_path):
    link = tmp_path / "linked outputs"
    destination = tmp_path / "destination"
    destination.mkdir()
    try:
        link.symlink_to(destination, target_is_directory=True)
    except OSError:
        pytest.skip("host cannot create unprivileged directory symlinks")
    with pytest.raises(ValueError, match="symlink|junction"):
        paper.run(**_args(evidence, tmp_path, output_root=link), prepare=True)
    assert not list(destination.iterdir())
