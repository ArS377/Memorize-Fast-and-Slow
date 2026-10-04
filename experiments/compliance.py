"""Audit one run directory against the Summer 6/20 experiment contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurosym.application.experiment_io import CELLS, cell_output_path
from neurosym.reporting import read_jsonl


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path, missing_ok=True)


def audit_run(results_dir: Path) -> Dict[str, Any]:
    metadata_path = results_dir / "manifest.json"
    if not metadata_path.exists():
        metadata_path = results_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    requested = {int(cell) for cell in metadata.get("cells", [])}
    checks: List[Dict[str, Any]] = []

    def check(identifier: str, passed: bool, detail: str) -> None:
        checks.append({"id": identifier, "passed": bool(passed), "detail": detail})

    check("manifest", bool(metadata.get("run_id") and metadata.get("git_sha")), "run and Git identity are recorded")
    check("dataset_hash", bool(metadata.get("input_sha256")), "input SHA-256 is recorded")
    check("typed_scope", metadata.get("memory_scope") in {"example", "session", "session_set"}, "memory scope is explicit")
    check("tool_contract", bool(metadata.get("tool_contract_version")), "tool schema version is recorded")
    check("rule_version", bool(metadata.get("rule_version")), "symbolic rule version is recorded")

    retrieval_config = metadata.get("retrieval_config") or {}
    expected_retrieval_mode = metadata.get("retrieval_mode") or retrieval_config.get("mode")

    example_sets: Dict[int, List[str]] = {}
    rows_by_cell: Dict[int, List[Dict[str, Any]]] = {}
    for cell in CELLS:
        cid = cell["cell_id"]
        if cid not in requested:
            continue
        path = cell_output_path(cid, cell["label"], results_dir)
        rows = _jsonl(path)
        rows_by_cell[cid] = rows
        example_sets[cid] = [str(row.get("example_id")) for row in rows]
        check(f"cell_{cid}_output", bool(rows), f"{path} contains {len(rows)} rows")
        check(
            f"cell_{cid}_run_identity",
            bool(rows) and all(row.get("run_id") == metadata.get("run_id") for row in rows),
            "every row belongs to this run",
        )
        check(
            f"cell_{cid}_execution",
            bool(rows) and not any(row.get("error") for row in rows),
            f"{sum(1 for row in rows if row.get('error'))} execution errors",
        )
        if cell["retrieval"] == "kg":
            configured_modes = {row.get("configured_retrieval_mode") for row in rows}
            effective_modes = {row.get("effective_retrieval_mode") for row in rows}
            degraded_rows = sum(bool(row.get("retrieval_degraded")) for row in rows)
            identities_present = all(bool(row.get("dense_index_identity")) for row in rows)
            mode_matches = bool(rows) and bool(expected_retrieval_mode) and configured_modes == {
                expected_retrieval_mode
            }
            execution_matches = bool(rows) and effective_modes == {expected_retrieval_mode}
            if expected_retrieval_mode in {"dense", "hybrid"}:
                execution_matches = execution_matches and identities_present
            check(
                f"cell_{cid}_retrieval_configuration",
                mode_matches,
                (
                    f"expected {expected_retrieval_mode!r}; "
                    f"configured modes: {sorted(str(value) for value in configured_modes)}"
                ),
            )
            check(
                f"cell_{cid}_retrieval_execution",
                execution_matches and degraded_rows == 0,
                (
                    f"effective modes: {sorted(str(value) for value in effective_modes)}; "
                    f"degraded rows: {degraded_rows}; dense identities present: {identities_present}"
                ),
            )

    populations = list(example_sets.values())
    check(
        "identical_example_population",
        bool(populations) and all(population == populations[0] for population in populations[1:]),
        "requested cells use the same ordered example IDs",
    )
    if 5 in rows_by_cell and 6 in rows_by_cell:
        modes = {
            row.get("orchestration_mode")
            for cid in (5, 6)
            for row in rows_by_cell[cid]
        }
        check("rlm_pair_orchestration", modes == {"qwen_native_tools_inside_rlm"}, f"observed modes: {sorted(str(mode) for mode in modes)}")
    if 6 in rows_by_cell:
        backends = {row.get("validator_backend") for row in rows_by_cell[6]}
        check("cell_6_actual_scallop", backends == {"scallop"}, f"observed backends: {sorted(str(value) for value in backends)}")

    kg = metadata.get("kg_artifacts") or {}
    if requested.intersection({2, 3, 5, 6}) and not metadata.get("skip_kg_build"):
        check("frozen_candidate_corpus", bool(kg.get("candidate_sha256")), "both KG variants derive from the recorded frozen candidates")

    requirements = {
        "qwen_first_tool_loop": "cells 5/6 native orchestration and tool traces",
        "persistent_graph_state": "run-specific Neo4j sessions and persisted working-memory transitions",
        "true_cross_session_scope": "typed session_set scope with a trusted allowlist",
        "compiled_working_memory": "update_working_memory artifacts with selected/excluded facts",
        "ranking_confidence_provenance": "structured search scores and compact working memory",
        "temporal_facts": "validity fields and overlap-aware symbolic rules",
        "full_provenance": "fact IDs, document/sentence spans, graph paths, and decision metadata",
        "audit_and_consolidation": "decision ledger, rejection artifacts, manifests, and this compliance report",
        "six_cell_pilot": "run-local six-cell results with identical-population checks",
        "feedback_updates": "Qwen-proposed update_working_memory transitions; Cell 6 is Scallop-gated",
    }
    return {
        "run_id": metadata.get("run_id"),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "requirement_implementation_map": requirements,
    }


def write_report(report: Dict[str, Any], results_dir: Path) -> Dict[str, Path]:
    json_path = results_dir / "compliance_report.json"
    md_path = results_dir / "compliance_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Run compliance: {report.get('run_id')}", "", f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**", ""]
    for item in report["checks"]:
        lines.append(f"- {'PASS' if item['passed'] else 'FAIL'} `{item['id']}` — {item['detail']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def main(argv: Optional[List[str]] = None) -> Dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return write_report(audit_run(args.results_dir), args.results_dir)


if __name__ == "__main__":
    main()
