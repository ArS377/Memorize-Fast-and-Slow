from __future__ import annotations

import json

from experiments.common import CELLS, cell_output_path
from experiments.compliance import audit_run


def _write_run(tmp_path, *, mismatch: bool = False):
    metadata = {
        "run_id": "test_run",
        "git_sha": "abc123",
        "input_sha256": "deadbeef",
        "cells": [cell["cell_id"] for cell in CELLS],
        "memory_scope": "example",
        "tool_contract_version": "tools.v1",
        "rule_version": "rules.v1",
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
