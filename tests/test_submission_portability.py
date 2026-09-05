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
