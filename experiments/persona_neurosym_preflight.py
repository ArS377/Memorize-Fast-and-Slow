"""Run the full Scallop and hybrid-KG persona preflight without Qwen generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.persona_end_to_end_benchmark import (
    _build_arm_specs,
    _derive_condition_sources,
    _git_provenance,
    _load_source_and_rebuild,
    _relation_coverage_matrix,
    _run_scallop_canary,
    _sha256,
    _stable_hash,
    _write_json,
    load_benchmark_config,
    _prepare_hybrid_memory,
)
from experiments.preference_stream_injection import PreferenceStreamInjectionClient


PREFLIGHT_VERSION = "persona_neurosym_preflight.v1"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write stable JSONL rows."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Execute every non-generative layer used by the hybrid benchmark arms."""
    from transformers import AutoTokenizer, __version__ as transformers_version

    config = load_benchmark_config(config_path)
    if config.hybrid_memory is None:
        raise ValueError("persona NeuroSym preflight requires hybrid KG arms")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, local_files_only=config.local_files_only
    )
    scheduled, turns, _ = _load_source_and_rebuild(config, tokenizer)
    injection_client = PreferenceStreamInjectionClient(
        config.scallop_endpoint, config.scallop_timeout_seconds
    )
    canary = _run_scallop_canary(injection_client)
    injections = _derive_condition_sources(scheduled["inputs"], turns, injection_client)
    relation_coverage = _relation_coverage_matrix(scheduled["inputs"], injections)
    retrievals, hybrid_identity, condition_facts, admission_ledgers = (
        _prepare_hybrid_memory(config, scheduled, turns)
    )
    if hybrid_identity is None:
        raise ValueError("hybrid memory identity is unavailable")
    specs = _build_arm_specs(
        scheduled["inputs"],
        turns,
        arms=config.arms,
        injections=injections,
        hybrid_retrievals=retrievals,
        prompt_instruction=config.prompt_instruction,
        tokenizer=tokenizer,
    )
    hybrid_specs = [row for row in specs if row["arm_kind"] == "hybrid_kg_memory"]
    if len(hybrid_specs) != 2 * len(scheduled["inputs"]):
        raise ValueError("hybrid prompt coverage is incomplete")
    retrieval_rows = [
        {
            "evaluation_input_id": evaluation_input_id,
            "seed_entities": value["seed_entities"],
            "retrieved_fact_ids": [
                str(row.get("fact_id", "")) for row in value["rows"]
            ],
            "metadata": value["metadata"],
        }
        for evaluation_input_id, value in sorted(retrievals.items())
    ]
    if any(
        row["metadata"].get("effective_mode") != "hybrid"
        or row["metadata"].get("degraded")
        or row["metadata"].get("branch_counts", {}).get("sparse", 0) < 1
        or row["metadata"].get("branch_counts", {}).get("dense", 0) < 1
        for row in retrieval_rows
    ):
        raise ValueError("one or more conditions did not execute both hybrid branches")
    retrievals_path = config.output_dir / "retrievals.jsonl"
    ledgers_path = config.output_dir / "admission_ledgers.jsonl"
    _write_jsonl(retrievals_path, retrieval_rows)
    _write_jsonl(
        ledgers_path,
        [
            {"evaluation_input_id": key, "decisions": value}
            for key, value in sorted(admission_ledgers.items())
        ],
    )
    manifest = {
        "status": "completed",
        "preflight_version": PREFLIGHT_VERSION,
        "config_sha256": _sha256(config_path),
        "dataset": scheduled["dataset"],
        "schedule_metrics": scheduled["schedule_metrics"],
        "condition_count": len(scheduled["inputs"]),
        "condition_fact_count": len(condition_facts),
        "hybrid_prompt_count": len(hybrid_specs),
        "scallop": {
            "engine": canary["engine"],
            "version": canary["scallopy_version"],
            "rule_version": canary["rule_version"],
        },
        "hybrid_memory": hybrid_identity,
        "relation_coverage_sha256": _stable_hash(relation_coverage),
        "ordered_hybrid_specs_sha256": _stable_hash(
            [{key: value for key, value in row.items() if key != "prompt"} for row in hybrid_specs]
        ),
        "tokenizer": {
            **scheduled["tokenizer"],
            "transformers_version": transformers_version,
        },
        "git": _git_provenance(
            Path(__file__).resolve().parent,
            excluded_untracked_dir=config.output_dir,
        ),
        "artifact_sha256": {
            retrievals_path.name: _sha256(retrievals_path),
            ledgers_path.name: _sha256(ledgers_path),
        },
    }
    _write_json(config.output_dir / "preflight_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Run the configured preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_preflight(args.config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
