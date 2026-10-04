from __future__ import annotations

import json

from neurosym.application.experiment_io import CELLS, cell_output_path
from experiments.compliance import audit_run


def _write_run(
    tmp_path,
    *,
    mismatch: bool = False,
    missing_hybrid_execution_cell: int | None = None,
):
    metadata = {
        "run_id": "test_run",
        "git_sha": "abc123",
        "input_sha256": "deadbeef",
        "cells": [cell["cell_id"] for cell in CELLS],
        "memory_scope": "example",
        "tool_contract_version": "tools.v1",
        "rule_version": "rules.v1",
        "retrieval_mode": "hybrid",
        "retrieval_config": {"mode": "hybrid"},
        "skip_kg_build": False,
        "kg_artifacts": {"candidate_sha256": "cafe"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    for cell in CELLS:
        path = cell_output_path(cell["cell_id"], cell["label"], tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        example_id = "wrong" if mismatch and cell["cell_id"] == 2 else "ex1"
        row = {
            "example_id": example_id,
            "run_id": "test_run",
            "error": None,
            "orchestration_mode": (
                "qwen_native_tools_inside_rlm" if cell["cell_id"] in {5, 6} else "fixed_context"
            ),
            "validator_backend": "scallop" if cell["cell_id"] == 6 else "n/a",
            "configured_retrieval_mode": (
                "hybrid" if cell["retrieval"] == "kg" else None
            ),
            "effective_retrieval_mode": (
                None
                if cell["cell_id"] == missing_hybrid_execution_cell
                else "hybrid" if cell["retrieval"] == "kg" else None
            ),
            "retrieval_degraded": False if cell["retrieval"] == "kg" else None,
            "dense_index_identity": (
                [{"source_session_id": f"session-{cell['cell_id']}"}]
                if cell["retrieval"] == "kg"
                else None
            ),
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_compliance_accepts_complete_isolated_run(tmp_path) -> None:
    _write_run(tmp_path)
    report = audit_run(tmp_path)
    assert report["passed"] is True


def test_compliance_detects_population_mismatch(tmp_path) -> None:
    _write_run(tmp_path, mismatch=True)
    report = audit_run(tmp_path)
    check = next(item for item in report["checks"] if item["id"] == "identical_example_population")
    assert check["passed"] is False
    assert report["passed"] is False


def test_compliance_rejects_configured_but_unexecuted_hybrid_retrieval(tmp_path) -> None:
    _write_run(tmp_path, missing_hybrid_execution_cell=5)

    report = audit_run(tmp_path)

    check = next(item for item in report["checks"] if item["id"] == "cell_5_retrieval_execution")
    assert check["passed"] is False
    assert report["passed"] is False
