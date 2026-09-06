import json
from pathlib import Path
import shutil

import pytest

from neurosym.application.source_provenance import (
    ensure_output_directory,
    paper_evidence_roots,
    source_manifest,
    verify_source_manifest,
    write_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative", [
    "results/persona_conflict_conversations_surface_a",
    "results/persona_conflict_conversations_surface_b",
    "results/persona_surface_pair_gate.json",
    "external-artifacts",
])
def test_all_shipped_and_external_resources_are_output_protected(relative):
    protected = paper_evidence_roots(ROOT)
    with pytest.raises(ValueError, match="frozen root"):
        ensure_output_directory(ROOT / relative, frozen_roots=protected)


def test_all_dependency_specifications_are_source_authenticated():
    paths = {row["path"] for row in source_manifest(ROOT)["files"]}
    specifications = {path.name for path in ROOT.glob("requirements*.txt")}
    assert len(specifications) >= 7
    assert specifications <= paths
    assert {"configs/artifact_resources.json", "configs/dependency_environments.json"} <= paths


def _environment_pins(name, stack=()):
    path = (ROOT / name).resolve()
    assert path not in stack, f"cyclic requirements include: {path}"
    pins = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        incoming = (
            _environment_pins(path.parent / line[3:], (*stack, path))
            if line.startswith("-r ") else dict([line.split("==", 1)])
        )
        assert not pins.keys() & incoming.keys(), f"duplicate pins in {path}"
        pins.update(incoming)
    return pins


def test_primary_environment_matches_recorded_versions():
    manifest = json.loads((ROOT / "tests/fixtures/historical_persona_manifest.json").read_bytes())
    pins = _environment_pins("requirements-generation.txt")
    assert pins == {
        **_environment_pins("requirements.txt"),
        "torch": manifest["model"]["torch_version"].split("+")[0],
        "transformers": manifest["model"]["transformers_version"],
        "sentence-transformers": "6.0.0",
        "graphiti-core": "0.29.3",
    }
    assert 5 <= int(pins["transformers"].split(".")[0]) < 6
    assert tuple(int(part) for part in pins["torch"].split(".")[:2]) >= (2, 2)


def test_primary_embedding_environment_evidence_is_pinned():
    environments = json.loads((ROOT / "configs/dependency_environments.json").read_bytes())["environments"]
    primary = environments["persona_generation"]
    evidence = primary["embedding_version_evidence"]
    resources = json.loads((ROOT / "configs/artifact_resources.json").read_bytes())["resources"]
    assert evidence["resource"] == "reference-persona"
    assert evidence["member"] == "hybrid_retrievals.json"
    assert evidence["sha256"] == resources[evidence["resource"]]["members"][evidence["member"]]
    assert evidence["observed_version"] == _environment_pins(primary["specification"])["sentence-transformers"] == "6.0.0"
    assert evidence["condition_count"] == 120
    assert evidence["field"] == "metadata.dense_index_identity[*].sentence_transformers_version"
    assert primary["recorded_torch_build"] == "2.13.0+cu130"
    assert primary["declared_compatibility"]["transformers"] == ">=5.0.0,<6.0.0"
    assert primary["declared_compatibility"]["torch"] == ">=2.2"


def test_continual_environment_keeps_its_separate_recorded_stack():
    assert _environment_pins("requirements-dense.txt") == {
        **_environment_pins("requirements.txt"),
        "numpy": "2.2.6", "rank-bm25": "0.2.2", "sentence-transformers": "3.4.1",
        "transformers": "4.57.6", "tokenizers": "0.22.2",
    }


@pytest.mark.parametrize("name", ["requirements-generation.txt", "requirements-dense.txt"])
def test_runtime_specifications_include_shared_requirements(name):
    assert (ROOT / name).read_text().splitlines().count("-r requirements.txt") == 1
    assert not {"vllm", "scallopy"} & _environment_pins(name).keys()


def test_graphiti_legacy_filename_aliases_the_primary_environment():
    assert _environment_pins("requirements-graphiti-baseline.txt") == _environment_pins("requirements-generation.txt")
    environments = json.loads((ROOT / "configs/dependency_environments.json").read_bytes())["environments"]
    assert environments["graphiti"]["canonical_specification"] == environments["persona_generation"]["specification"]
    assert environments["persona_generation"]["includes"] == ["requirements.txt"]
    assert environments["continual_dense"]["includes"] == ["requirements.txt"]


def test_supporting_requirements_do_not_import_primary_model_stack():
    assert not {"torch", "graphiti-core"} & _environment_pins("requirements-dense.txt").keys()


def test_readme_uses_consolidated_runtime_entrypoints():
    requirements = (ROOT / "README.md").read_text(encoding="utf-8").split("## Requirements\n", 1)[1].split("## Evaluation", 1)[0]
    assert "pip install -r requirements-generation.txt" in requirements
    assert "pip install -r requirements-dense.txt" in requirements
    assert "-r requirements-graphiti-baseline.txt" not in requirements
    assert "pip install -r requirements.txt" not in requirements


def test_consolidated_requirements_work_in_git_free_source_archive(tmp_path):
    archive = tmp_path / "source archive"
    manifest = tmp_path / "manifest.json"
    digest = write_source_manifest(ROOT, manifest)
    files = source_manifest(ROOT)["files"]
    for entry in files:
        destination = archive / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / entry["path"], destination)
    assert not (archive / ".git").exists()
    assert verify_source_manifest(archive, manifest, digest)["source_file_count"] == len(files)
    for name in ("requirements-generation.txt", "requirements-dense.txt", "requirements-graphiti-baseline.txt"):
        assert _environment_pins(archive / name) == _environment_pins(name)
