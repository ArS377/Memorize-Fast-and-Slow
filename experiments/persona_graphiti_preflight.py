"""Run the Graphiti ingestion and retrieval preflight without Qwen generation.

Mirrors ``persona_neurosym_preflight`` for the external baseline: it executes
every non-generative layer the Graphiti arms depend on (episode rendering,
extraction, write-time revision, store snapshotting, point-in-time retrieval,
prompt fitting) and writes authenticated artifacts.

Its second job is extraction triage. A small extractor can fail three ways, and
only two of them raise: malformed JSON is retried by graphiti-core, a wrongly
shaped response raises ``pydantic.ValidationError``, but a well-formed and
*empty* extraction raises nothing at all. A silently under-extracting Graphiti
would look like a devastating baseline result and be an artifact of the
extractor, so this preflight reads the committed store back and reports
extraction health rather than merely confirming that nothing threw.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.persona_end_to_end_benchmark import (
    _build_arm_specs,
    _git_provenance,
    _load_source_and_rebuild,
    _sha256,
    _stable_hash,
    _write_json,
    load_benchmark_config,
)
from experiments.persona_graphiti_baseline import prepare_graphiti_memory


PREFLIGHT_VERSION = "persona_graphiti_preflight.v1"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write stable JSONL rows."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def select_preflight_conditions(
    conditions: Sequence[Mapping[str, Any]], history_limit: int | None
) -> list[Mapping[str, Any]]:
    """Restrict the preflight to the first N accounts in stable id order.

    A one-account preflight is the cheap extraction smoke test; the full 12
    accounts reproduce the ingestion side of a complete build.
    """
    if history_limit is None:
        return list(conditions)
    if isinstance(history_limit, bool) or not isinstance(history_limit, int):
        raise ValueError("history limit must be an integer")
    if history_limit < 1:
        raise ValueError("history limit must be at least 1")
    ordered = sorted({str(row["history_id"]) for row in conditions})
    keep = set(ordered[:history_limit])
    return [row for row in conditions if str(row["history_id"]) in keep]


def summarize_extraction(
    store_states: Mapping[str, Mapping[str, Any]],
    retrievals: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report whether extraction actually populated the store.

    Every quantity here is descriptive; the caller decides what is fatal. The
    ratios matter more than the totals: a store that ingests thirty episodes and
    holds two facts has not failed loudly, but it has failed.
    """
    checkpoints = [dict(state) for _, state in sorted(store_states.items())]
    episode_total = max((int(row["episode_count"]) for row in checkpoints), default=0)
    facts = [fact for row in checkpoints for fact in row.get("valid_facts", [])]
    dated = [fact for fact in facts if fact.get("valid_at")]
    terminal = [row for row in checkpoints if row.get("valid_fact_count", 0)]
    empty_retrievals = [
        evaluation_input_id
        for evaluation_input_id, value in sorted(retrievals.items())
        if not value.get("rows")
    ]
    per_account: dict[str, dict[str, Any]] = {}
    for row in checkpoints:
        account = per_account.setdefault(
            str(row["history_id"]),
            {"checkpoints": 0, "max_episode_count": 0, "max_valid_fact_count": 0},
        )
        account["checkpoints"] += 1
        account["max_episode_count"] = max(
            account["max_episode_count"], int(row["episode_count"])
        )
        account["max_valid_fact_count"] = max(
            account["max_valid_fact_count"], int(row["valid_fact_count"])
        )
    return {
        "checkpoint_count": len(checkpoints),
        "deepest_episode_count": episode_total,
        "valid_fact_rows_total": len(facts),
        "checkpoints_with_zero_valid_facts": len(checkpoints) - len(terminal),
        "valid_at_resolved_count": len(dated),
        "valid_at_resolved_fraction": (
            round(len(dated) / len(facts), 4) if facts else 0.0
        ),
        "condition_count": len(retrievals),
        "empty_retrieval_count": len(empty_retrievals),
        "empty_retrieval_condition_ids": empty_retrievals[:20],
        "accounts": {name: per_account[name] for name in sorted(per_account)},
    }


def assess_extraction_health(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Separate unambiguous extraction breakage from judgment calls.

    Only conditions that make the run meaningless are fatal. Everything else is
    surfaced for a human, because thresholds on a small preflight would either
    fire constantly or hide real degradation.
    """
    fatal = []
    warnings = []
    if summary["valid_fact_rows_total"] == 0:
        fatal.append("no valid facts were committed by any account")
    if summary["condition_count"] and summary["empty_retrieval_count"] == summary["condition_count"]:
        fatal.append("every condition retrieved zero facts")
    for name, account in summary["accounts"].items():
        if account["max_valid_fact_count"] == 0:
            fatal.append(f"account {name} committed no valid facts")
    if summary["checkpoints_with_zero_valid_facts"]:
        warnings.append(
            f"{summary['checkpoints_with_zero_valid_facts']} checkpoint(s) hold no "
            "valid facts; expected for the earliest checkpoints, suspicious later"
        )
    if summary["valid_at_resolved_fraction"] < 0.8:
        warnings.append(
            f"only {summary['valid_at_resolved_fraction']:.0%} of committed facts carry a "
            "resolved valid_at; point-in-time retrieval degrades toward unfiltered"
        )
    if summary["empty_retrieval_count"]:
        warnings.append(
            f"{summary['empty_retrieval_count']} condition(s) retrieved zero facts"
        )
    return {
        "status": "failed" if fatal else "passed",
        "fatal": fatal,
        "warnings": warnings,
    }


def run_preflight(
    config_path: Path, *, history_limit: int | None = None
) -> dict[str, Any]:
    """Execute every non-generative layer used by the Graphiti arms."""
    from transformers import AutoTokenizer, __version__ as transformers_version

    config = load_benchmark_config(config_path)
    if config.graphiti_memory is None:
        raise ValueError("persona Graphiti preflight requires graphiti memory arms")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, local_files_only=config.local_files_only
    )
    scheduled, turns, _ = _load_source_and_rebuild(config, tokenizer)
    conditions = select_preflight_conditions(scheduled["inputs"], history_limit)
    if not conditions:
        raise ValueError("preflight selected no conditions")
    retrievals, identity, store_states = prepare_graphiti_memory(
        conditions, turns, config.graphiti_memory
    )
    if len(retrievals) != len(conditions):
        raise ValueError("graphiti retrieval coverage is incomplete")
    summary = summarize_extraction(store_states, retrievals)
    health = assess_extraction_health(summary)

    graphiti_arms = [arm for arm in config.arms if arm.kind == "graphiti_memory"]
    specs = _build_arm_specs(
        conditions,
        turns,
        arms=graphiti_arms,
        injections={},
        graphiti_retrievals=retrievals,
        prompt_instruction=config.prompt_instruction,
        tokenizer=tokenizer,
    )
    if len(specs) != len(graphiti_arms) * len(conditions):
        raise ValueError("graphiti prompt coverage is incomplete")

    retrievals_path = config.output_dir / "retrievals.jsonl"
    states_path = config.output_dir / "store_states.jsonl"
    _write_jsonl(
        retrievals_path,
        [
            {
                "evaluation_input_id": evaluation_input_id,
                "retrieved_fact_ids": [
                    str(row.get("fact_id", "")) for row in value["rows"]
                ],
                "retrieved_fact_count": len(value["rows"]),
                "metadata": value["metadata"],
            }
            for evaluation_input_id, value in sorted(retrievals.items())
        ],
    )
    _write_jsonl(
        states_path,
        [dict(state) for _, state in sorted(store_states.items())],
    )
    manifest = {
        "status": "completed" if health["status"] == "passed" else "failed",
        "preflight_version": PREFLIGHT_VERSION,
        "config_sha256": _sha256(config_path),
        "dataset": scheduled["dataset"],
        "schedule_metrics": scheduled["schedule_metrics"],
        "history_limit": history_limit,
        "condition_count": len(conditions),
        "graphiti_prompt_count": len(specs),
        "graphiti_memory": identity,
        "extraction_summary": summary,
        "extraction_health": health,
        "ordered_graphiti_specs_sha256": _stable_hash(
            [{key: value for key, value in row.items() if key != "prompt"} for row in specs]
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
            states_path.name: _sha256(states_path),
        },
    }
    _write_json(config.output_dir / "preflight_manifest.json", manifest)
    if health["status"] != "passed":
        raise ValueError(f"graphiti extraction preflight failed: {health['fatal']}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Run the configured Graphiti preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--histories",
        type=int,
        default=None,
        help="limit the preflight to the first N accounts (1 for a smoke test)",
    )
    args = parser.parse_args(argv)
    manifest = run_preflight(args.config, history_limit=args.histories)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
