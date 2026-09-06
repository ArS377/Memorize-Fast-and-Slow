import json
from pathlib import Path

import pytest

from neurosym.application.source_provenance import (
    ensure_output_directory,
    paper_evidence_roots,
    source_manifest,
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


def _environment_pins(name):
    return dict(line.split("==", 1) for line in (ROOT / name).read_text().splitlines()
                if line.strip() and not line.startswith("#"))


def test_primary_environment_matches_recorded_versions():
    manifest = json.loads((ROOT / "tests/fixtures/historical_persona_manifest.json").read_bytes())
    pins = _environment_pins("requirements-generation.txt")
    assert pins == {
        "torch": manifest["model"]["torch_version"].split("+")[0],
        "transformers": manifest["model"]["transformers_version"],
        "sentence-transformers": "6.0.0",
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
        "numpy": "2.2.6", "rank-bm25": "0.2.2", "sentence-transformers": "3.4.1",
        "transformers": "4.57.6", "tokenizers": "0.22.2",
    }
