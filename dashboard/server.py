"""Dynamic stdlib HTTP server for local NeuroSym experiment results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit


ARTIFACT_NAMES = ("metrics.json", "manifest.json", "report.md")
COMPLETE_STATUSES = {"complete", "completed", "ok", "success", "succeeded"}
CODE_REFERENCES = (
    ("System contracts", "neurosym/architecture.py"),
    ("Cell orchestration", "neurosym/application/cells.py"),
    ("Admission gate", "neurosym/domain/hard_gate_admission.py"),
    ("Experiment runner", "experiments/run_all.py"),
    ("Temporal benchmark", "experiments/synthetic_temporal_benchmark.py"),
    ("Temporal data", "experiments/synthetic_temporal_preferences.py"),
    ("Temporal baselines", "experiments/synthetic_temporal_baselines.py"),
    ("Differentiable admission", "experiments/differentiable_admission.py"),
    ("Multi-task context benchmark", "experiments/multitask_context_benchmark.py"),
    ("Continual memory benchmark", "experiments/continual_memory_benchmark.py"),
)
CODE_REFERENCE_PATHS = {path for _, path in CODE_REFERENCES}


@dataclass(frozen=True)
class DashboardConfig:
    """Filesystem roots used by a dashboard server instance."""

    repository_root: Path
    results_root: Path
    static_root: Path


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    """Read JSON while returning a displayable parse error for incomplete writes."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def _flatten_numbers(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Extract finite numeric leaves with their exact JSON provenance paths."""
    flattened: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.extend(_flatten_numbers(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            flattened.extend(_flatten_numbers(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            flattened.append({"key": prefix or "value", "value": value})
    return flattened


def _score_priority(metric: dict[str, Any]) -> tuple[int, int, int, str]:
    """Rank likely quality metrics without pretending unlike run schemas are identical."""
    key = metric["key"].lower()
    leaf = key.rsplit(".", maxsplit=1)[-1]
    segments = key.split(".")
    priorities = {
        "accuracy": 0,
        "answer_accuracy": 1,
        "grounded_answer_accuracy": 0,
        "candidate_decision_accuracy": 2,
        "f1": 3,
        "f1_score": 3,
        "score": 4,
        "reward": 5,
        "precision": 6,
        "recall": 7,
        "pass_rate": 8,
    }
    priority = 1 if "query_accuracy" in segments else priorities.get(leaf, 20)
    split_priority = 0 if ".test." in key else 1 if ".dev." in key else 2
    return priority, split_priority, key.count("."), key


def _metric_summary(metrics: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return a conservative primary metric plus provenance-rich score leaves."""
    numeric_metrics = _flatten_numbers(metrics)
    scores = [metric for metric in numeric_metrics if _score_priority(metric)[0] < 20]
    scores.sort(key=_score_priority)
    return (scores[0] if scores else None), scores


def _finite_number(value: Any) -> int | float | None:
    """Return a finite JSON number while rejecting booleans and malformed values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _structured_notes(value: Any, path: str) -> list[dict[str, str]]:
    """Copy string leaves from an explicit metrics field with exact source paths."""
    if isinstance(value, str):
        return [{"path": path, "text": value}] if value.strip() else []
    if isinstance(value, dict):
        notes = []
        for key, child in value.items():
            notes.extend(_structured_notes(child, f"{path}.{key}"))
        return notes
    if isinstance(value, list):
        notes = []
        for index, child in enumerate(value):
            notes.extend(_structured_notes(child, f"{path}[{index}]"))
        return notes
    return []


def _stress_scenarios(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract finite summaries for every dynamically named model stress scenario."""
    stress = model.get("stress")
    if not isinstance(stress, dict):
        return []
    scenarios = []
    for name, measurements in sorted(stress.items()):
        if not isinstance(name, str) or not isinstance(measurements, dict):
            continue
        scenarios.append(
            {
                "name": name,
                "stddev": _finite_number(measurements.get("stddev")),
                "repeat_count": _finite_number(measurements.get("repeat_count")),
                "accuracy_mean": _finite_number(measurements.get("accuracy_mean")),
                "accuracy_min": _finite_number(measurements.get("accuracy_min")),
                "accuracy_max": _finite_number(measurements.get("accuracy_max")),
                "eligible_accuracy_mean": _finite_number(
                    measurements.get("eligible_accuracy_mean")
                ),
                "eligible_accuracy_min": _finite_number(
                    measurements.get("eligible_accuracy_min")
                ),
                "eligible_accuracy_max": _finite_number(
                    measurements.get("eligible_accuracy_max")
                ),
                "hard_gate_violations_max": _finite_number(
                    measurements.get("hard_gate_violations_max")
                ),
            }
        )
    return scenarios


def _named_context_scores(value: Any) -> list[dict[str, Any]]:
    """Extract dynamically named context-method score records."""
    if not isinstance(value, dict):
        return []
    scores = []
    for name, measurements in sorted(value.items()):
        if not isinstance(name, str) or not isinstance(measurements, dict):
            continue
        evidence_recall = _finite_number(measurements.get("gold_evidence_recall"))
        if evidence_recall is None:
            evidence_recall = _finite_number(measurements.get("evidence_recall"))
        case_count = _finite_number(measurements.get("case_count"))
        if case_count is None:
            case_count = _finite_number(measurements.get("checkpoint_count"))
        answer_accuracy = _finite_number(measurements.get("answer_accuracy"))
        if answer_accuracy is None:
            answer_accuracy = _finite_number(
                measurements.get("oracle_resolver_answer_accuracy")
            )
        exact_evidence = _finite_number(measurements.get("exact_evidence_hit_rate"))
        if exact_evidence is None:
            exact_evidence = _finite_number(
                measurements.get("complete_provenance_rate")
            )
        scores.append(
            {
                "name": name,
                "answer_accuracy": answer_accuracy,
                "grounded_answer_accuracy": _finite_number(
                    measurements.get("grounded_answer_accuracy")
                ),
                "evidence_recall": evidence_recall,
                "exact_evidence_hit_rate": exact_evidence,
                "cross_task_interference_rate": _finite_number(
                    measurements.get("cross_task_interference_rate")
                ),
                "stale_memory_intrusion_rate": _finite_number(
                    measurements.get("stale_memory_intrusion_rate")
                ),
                "retraction_compliance": _finite_number(
                    measurements.get("retraction_compliance")
                ),
                "thread_precision": _finite_number(
                    measurements.get("target_thread_precision")
                ),
                "case_count": case_count,
            }
        )
    return scores


def _context_breakdown(metrics: dict[str, Any], source: str) -> dict[str, Any] | None:
    """Extract one dynamic context or evidence-position score matrix."""
    value = metrics.get(source)
    if not isinstance(value, dict):
        return None
    groups = []
    for name, measurements in sorted(value.items()):
        if not isinstance(name, str):
            continue
        scores = _named_context_scores(measurements)
        if scores:
            groups.append({"name": name, "scores": scores})
    return {"source": source, "groups": groups} if groups else None


def _realized_context_tiers(value: Any) -> list[dict[str, Any]]:
    """Extract realized context construction diagnostics for every named tier."""
    if not isinstance(value, dict):
        return []
    tiers = []
    for name, measurements in sorted(value.items()):
        if not isinstance(name, str) or not isinstance(measurements, dict):
            continue
        tiers.append(
            {
                "name": name,
                "token_min": _finite_number(measurements.get("model_input_tokens_min")),
                "token_max": _finite_number(measurements.get("model_input_tokens_max")),
                "requested_target_percent": _finite_number(
                    measurements.get("requested_target_thread_percent")
                ),
                "realized_target_percent": _finite_number(
                    measurements.get("target_thread_percent_mean")
                ),
                "max_position_error": _finite_number(
                    measurements.get("gold_position_absolute_error_max")
                ),
                "hard_lexical_count": _finite_number(
                    measurements.get("hard_lexical_distractors")
                ),
                "case_count": _finite_number(measurements.get("case_count")),
            }
        )
    return tiers


def _parameter_tiers(dataset: Any) -> list[dict[str, Any]]:
    """Extract parameter-tier metadata without inferring model evaluation claims."""
    if not isinstance(dataset, dict) or not isinstance(
        dataset.get("model_parameter_tiers"), list
    ):
        return []
    tiers = []
    for tier in dataset["model_parameter_tiers"]:
        if not isinstance(tier, dict) or not isinstance(tier.get("name"), str):
            continue
        tiers.append(
            {
                "name": tier["name"],
                "min_billions": _finite_number(tier.get("min_billions")),
                "max_billions": _finite_number(tier.get("max_billions")),
            }
        )
    return tiers


def _context_benchmark(metrics: Any) -> dict[str, Any] | None:
    """Extract a context benchmark when metrics declares top-level methods."""
    if not isinstance(metrics, dict):
        return None
    methods = _named_context_scores(metrics.get("methods"))
    if not methods:
        horizons = _interleaved_horizons(metrics.get("horizons"))
        if not horizons:
            return None
        online = metrics.get("online_checkpoint_evaluation")
        online_methods = (
            _named_context_scores(online.get("methods"))
            if isinstance(online, dict)
            else []
        )
        interpretation = metrics.get("interpretation")
        return {
            "methods": online_methods,
            "realized_context_tiers": [],
            "windows": [],
            "breakdowns": [],
            "parameter_tiers": [],
            "memory_growth": None,
            "scallop_reasoning_ablation": None,
            "scallop_stream_injection_ablation": None,
            "interleaved_horizons": horizons,
            "contradiction_ledger": metrics.get("contradiction_ledger"),
            "context_horizon_contract": metrics.get("context_horizon_contract"),
            "answer_evaluator": online.get("answer_evaluator") if isinstance(online, dict) else None,
            "interpretation_notes": _structured_notes(
                interpretation, "interpretation"
            )
            if interpretation is not None
            else [],
        }
    breakdowns = []
    for source in (
        "by_context_tier",
        "by_evidence_position",
        "window_truncation_by_context_tier",
        "window_truncation_by_evidence_position",
        "retention_by_interference_tier",
        "retention_by_checkpoint_kind",
        "retention_by_turn_age_bin",
        "retention_by_token_age_bin",
        "retention_by_stream_position_bin",
    ):
        breakdown = _context_breakdown(metrics, source)
        if breakdown is not None:
            breakdowns.append(breakdown)
    interpretation = metrics.get("interpretation")
    return {
        "methods": methods,
        "realized_context_tiers": _realized_context_tiers(
            metrics.get("realized_context_tiers")
        ),
        "windows": _named_context_scores(metrics.get("window_truncation")),
        "breakdowns": breakdowns,
        "parameter_tiers": _parameter_tiers(metrics.get("dataset")),
        "memory_growth": _memory_growth(metrics.get("memory_growth")),
        "scallop_reasoning_ablation": _scallop_reasoning_ablation(
            metrics.get("scallop_reasoning_ablation")
        ),
        "scallop_stream_injection_ablation": _scallop_stream_injection_ablation(
            metrics.get("scallop_stream_injection_ablation")
        ),
        "interleaved_horizons": _interleaved_horizons(metrics.get("horizons")),
        "contradiction_ledger": metrics.get("contradiction_ledger"),
        "context_horizon_contract": metrics.get("context_horizon_contract"),
        "answer_evaluator": None,
        "interpretation_notes": _structured_notes(
            interpretation, "interpretation"
        )
        if interpretation is not None
        else [],
    }


def _interleaved_horizons(value: Any) -> list[dict[str, Any]]:
    """Extract finite-context loss curves from interleaved peer streams."""
    if not isinstance(value, list):
        return []
    rows = []
    for horizon in value:
        if not isinstance(horizon, dict):
            continue
        loss = horizon.get("suffix_contract_loss_rate")
        if not isinstance(loss, dict):
            continue
        loss_by_window = []
        for window, rate in loss.items():
            try:
                window_tokens = int(window)
            except (TypeError, ValueError):
                continue
            loss_by_window.append(
                {
                    "window_tokens": window_tokens,
                    "loss_rate": _finite_number(rate),
                }
            )
        loss_by_window.sort(key=lambda item: item["window_tokens"])
        rows.append(
            {
                "horizon_accounts": _finite_number(horizon.get("horizon_accounts")),
                "stream_token_count": _finite_number(horizon.get("stream_token_count")),
                "loss_by_window": loss_by_window,
            }
        )
    return rows


def _scallop_reasoning_ablation(value: Any) -> dict[str, Any] | None:
    """Extract matched recursive and one-hop Scallop reasoning results."""
    if not isinstance(value, dict) or value.get("enabled") is not True:
        return None
    methods_raw = value.get("methods")
    if not isinstance(methods_raw, dict):
        return None
    methods = []
    for name, measurements in sorted(methods_raw.items()):
        if not isinstance(name, str) or not isinstance(measurements, dict):
            continue
        by_kind = measurements.get("by_checkpoint_kind")
        by_kind = by_kind if isinstance(by_kind, dict) else {}
        methods.append(
            {
                "name": name,
                "accuracy": _finite_number(measurements.get("accuracy")),
                "direct_accuracy": _finite_number(
                    by_kind.get("private", {}).get("accuracy")
                    if isinstance(by_kind.get("private"), dict)
                    else None
                ),
                "positive_accuracy": _finite_number(
                    by_kind.get("private-lineage-positive", {}).get("accuracy")
                    if isinstance(by_kind.get("private-lineage-positive"), dict)
                    else None
                ),
                "recursive_probe_accuracy": _finite_number(
                    by_kind.get("private-lineage", {}).get("accuracy")
                    if isinstance(by_kind.get("private-lineage"), dict)
                    else None
                ),
            }
        )
    if not methods:
        return None
    return {
        "engine": value.get("engine"),
        "version": value.get("scallopy_version"),
        "causal_feature": value.get("causal_feature"),
        "methods": methods,
    }


def _scallop_stream_injection_ablation(value: Any) -> dict[str, Any] | None:
    """Extract matched raw, injected, and reverse-ranked control results."""
    if not isinstance(value, dict) or value.get("enabled") is not True:
        return None
    by_window = value.get("by_window")
    if not isinstance(by_window, dict):
        return None
    rows = []
    for window, tiers in sorted(by_window.items()):
        if not isinstance(window, str) or not isinstance(tiers, dict):
            continue
        for tier, checkpoints in sorted(tiers.items()):
            if not isinstance(tier, str) or not isinstance(checkpoints, dict):
                continue
            for checkpoint, measurements in sorted(checkpoints.items()):
                if not isinstance(checkpoint, str) or not isinstance(measurements, dict):
                    continue
                rows.append(
                    {
                        "window": window,
                        "tier": tier,
                        "checkpoint": checkpoint,
                        "matched_count": _finite_number(measurements.get("matched_count")),
                        "raw_grounded_accuracy": _finite_number(
                            measurements.get("no_injection_grounded_accuracy")
                        ),
                        "scallop_grounded_accuracy": _finite_number(
                            measurements.get("scallop_injected_grounded_accuracy")
                        ),
                        "reverse_ranked_grounded_accuracy": _finite_number(
                            measurements.get("reverse_ranked_control_grounded_accuracy")
                        ),
                        "change_only_grounded_accuracy": _finite_number(
                            measurements.get("change_rule_only_grounded_accuracy")
                        ),
                        "incongruity_only_grounded_accuracy": _finite_number(
                            measurements.get("incongruity_rule_only_grounded_accuracy")
                        ),
                        "paired_delta": _finite_number(measurements.get("paired_delta")),
                    }
                )
    if not rows:
        return None
    return {
        "scallop_query_or_gold_used": value.get("scallop_query_or_gold_used"),
        "capsule_selector": value.get("capsule_selector"),
        "source_grounded": value.get("source_grounded"),
        "engine": value.get("engine"),
        "scallopy_versions": value.get("scallopy_versions"),
        "rule_version": value.get("rule_version"),
        "rows": rows,
    }


def _memory_growth(value: Any) -> dict[str, Any] | None:
    """Extract finite overall and per-tier continual-memory growth measurements."""
    if not isinstance(value, dict):
        return None

    def summary(measurements: Any) -> dict[str, int | float | None]:
        if not isinstance(measurements, dict):
            return {}
        return {
            key: _finite_number(measurements.get(key))
            for key in (
                "checkpoint_count",
                "stored_turns_mean",
                "stored_turns_max",
                "stored_tokens_mean",
                "stored_tokens_max",
            )
        }

    overall = summary(value)
    if not overall:
        return None
    by_tier = value.get("by_interference_tier")
    tiers = []
    if isinstance(by_tier, dict):
        tiers = [
            {"name": name, **summary(measurements)}
            for name, measurements in sorted(by_tier.items())
            if isinstance(name, str) and isinstance(measurements, dict)
        ]
    return {**overall, "by_interference_tier": tiers}


def _model_comparison(metrics: Any) -> dict[str, Any] | None:
    """Extract comparable held-out model measurements from a top-level models object."""
    if not isinstance(metrics, dict) or not isinstance(metrics.get("models"), dict):
        return None
    models = []
    for name, model in sorted(metrics["models"].items()):
        if not isinstance(name, str) or not isinstance(model, dict):
            continue
        test = model.get("test") if isinstance(model.get("test"), dict) else {}
        by_label = test.get("by_label") if isinstance(test.get("by_label"), dict) else {}
        per_label_recall = []
        for label, label_metrics in sorted(by_label.items()):
            if not isinstance(label, str) or not isinstance(label_metrics, dict):
                continue
            recall = _finite_number(label_metrics.get("recall"))
            if recall is not None:
                per_label_recall.append({"label": label, "recall": recall})
        training = model.get("training") if isinstance(model.get("training"), dict) else {}
        training_loss = {
            key: value
            for key, value in (
                ("initial", _finite_number(training.get("initial_loss"))),
                ("final", _finite_number(training.get("final_loss"))),
            )
            if value is not None
        }
        models.append(
            {
                "name": name,
                "test_accuracy": _finite_number(test.get("accuracy")),
                "per_label_recall": per_label_recall,
                "hard_gate_violations": _finite_number(test.get("hard_gate_violations")),
                "training_loss": training_loss or None,
                "stress_scenarios": _stress_scenarios(model),
            }
        )
    if not models:
        return None
    notes = []
    for field in ("interpretation", "limitations"):
        if field in metrics:
            notes.extend(_structured_notes(metrics[field], field))
    return {"models": models, "notes": notes}


def _cell_status(manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    """Summarize per-cell state without converting pending work into success."""
    cells = manifest.get("cell_status")
    if not isinstance(cells, dict) or not cells:
        return None, None
    statuses = [str(value).strip().lower() for value in cells.values()]
    if any(status in {"failed", "error"} for status in statuses):
        return "failed", "manifest.cell_status"
    complete_count = sum(status in COMPLETE_STATUSES for status in statuses)
    if complete_count == len(statuses):
        return "completed", "manifest.cell_status"
    if complete_count:
        return "partial", "manifest.cell_status"
    if any(status in {"running", "active", "in_progress"} for status in statuses):
        return "running", "manifest.cell_status"
    if all(status in {"pending", "queued", "not_started"} for status in statuses):
        return "pending", "manifest.cell_status"
    return "unknown", "manifest.cell_status"


def _run_status(
    manifest: Any, parse_errors: dict[str, str]
) -> tuple[str, str, dict[str, int]]:
    """Resolve declared status and its source, preserving unknown states."""
    if parse_errors:
        return "invalid", "artifact parse error", {}
    if not isinstance(manifest, dict):
        return "unknown", "no manifest status", {}
    declared_status = manifest.get("status")
    if isinstance(declared_status, str) and declared_status.strip():
        return declared_status.strip().lower(), "manifest.status", {}
    status, provenance = _cell_status(manifest)
    cells = manifest.get("cell_status")
    counts = Counter(str(value).strip().lower() for value in cells.values()) if isinstance(cells, dict) else Counter()
    return status or "unknown", provenance or "no manifest status", dict(counts)


def _artifact_record(path: Path, results_root: Path) -> dict[str, Any]:
    """Describe an artifact and create a local server URL for it."""
    relative_path = path.relative_to(results_root)
    stat = path.stat()
    return {
        "name": path.name,
        "path": f"results/{relative_path.as_posix()}",
        "url": f"/files/results/{quote(relative_path.as_posix(), safe='/')}",
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _sha256(path: Path) -> str:
    """Return a file digest for manifest integrity verification."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_artifact_state(
    run_directory: Path, manifest: Any
) -> tuple[list[Path], list[str], dict[str, str]]:
    """Resolve declared artifacts and verify any declared SHA-256 digests."""
    if not isinstance(manifest, dict):
        return [], [], {}
    resolved_run_directory = run_directory.resolve()
    declared = manifest.get("artifacts")
    declared_names = declared if isinstance(declared, list) else []
    paths: list[Path] = []
    missing: list[str] = []
    integrity_errors: dict[str, str] = {}
    for name in declared_names:
        if not isinstance(name, str):
            continue
        path = (run_directory / name).resolve()
        if not path.is_relative_to(resolved_run_directory):
            integrity_errors[name] = "declared artifact escapes its run directory"
        elif path.is_file():
            paths.append(path)
        else:
            missing.append(name)

    hashes = manifest.get("artifact_sha256")
    hashes = hashes if isinstance(hashes, dict) else {}
    for name in declared_names:
        if isinstance(name, str) and name not in hashes and name not in missing:
            integrity_errors[name] = "SHA-256 missing from manifest"
    for name, expected in hashes.items():
        if not isinstance(name, str):
            continue
        path = (run_directory / name).resolve()
        if not path.is_relative_to(resolved_run_directory):
            integrity_errors[name] = "hashed artifact escapes its run directory"
        elif not path.is_file():
            if name not in missing:
                missing.append(name)
        elif not isinstance(expected, str) or len(expected) != 64:
            integrity_errors[name] = "manifest SHA-256 is malformed"
        elif _sha256(path) != expected.lower():
            integrity_errors[name] = "SHA-256 does not match manifest"
    return paths, missing, integrity_errors


def _timestamp(manifest: Any, paths: list[Path]) -> str:
    """Use a declared run time when available, otherwise the newest artifact mtime."""
    if isinstance(manifest, dict):
        for key in ("completed_at", "failed_at", "timestamp", "created_at", "started_at"):
            value = manifest.get(key)
            if isinstance(value, str) and value:
                return value
    newest_mtime = max(path.stat().st_mtime for path in paths)
    return datetime.fromtimestamp(newest_mtime, timezone.utc).isoformat()


def _build_run(run_directory: Path, results_root: Path) -> dict[str, Any]:
    """Build one run record from the recognized artifacts in a directory."""
    paths = [run_directory / name for name in ARTIFACT_NAMES if (run_directory / name).is_file()]
    path_by_name = {path.name: path for path in paths}
    parse_errors: dict[str, str] = {}
    manifest: Any = None
    metrics: Any = None
    for name, destination in (("manifest.json", "manifest"), ("metrics.json", "metrics")):
        path = path_by_name.get(name)
        if path is None:
            continue
        payload, error = _read_json(path)
        if error:
            parse_errors[name] = error
        elif destination == "manifest":
            manifest = payload
        else:
            metrics = payload

    status, status_provenance, cell_counts = _run_status(manifest, parse_errors)
    artifact_paths = set(paths)
    declared_paths, declared_missing, integrity_errors = _declared_artifact_state(
        run_directory, manifest
    )
    artifact_paths.update(declared_paths)
    primary_metric, score_metrics = _metric_summary(metrics)
    model_comparison = _model_comparison(metrics)
    context_benchmark = _context_benchmark(metrics)
    if context_benchmark is not None and len(context_benchmark["methods"]) > 1:
        primary_metric = None
    relative_directory = run_directory.relative_to(results_root)
    run_id = relative_directory.as_posix() or "."
    manifest_id = manifest.get("run_id") if isinstance(manifest, dict) else None
    source = manifest.get("source") if isinstance(manifest, dict) else None
    source = source if isinstance(source, dict) else {}
    git_sha = manifest.get("git_sha", source.get("git_sha")) if isinstance(manifest, dict) else None
    git_dirty = (
        manifest.get("git_dirty", source.get("git_dirty"))
        if isinstance(manifest, dict)
        else None
    )
    missing_artifacts = list(
        dict.fromkeys(
            [name for name in ARTIFACT_NAMES if name not in path_by_name] + declared_missing
        )
    )
    return {
        "id": run_id,
        "name": str(manifest_id) if manifest_id else run_directory.name,
        "kind": "dense index" if "dense_indexes" in relative_directory.parts else "experiment",
        "status": status,
        "status_provenance": status_provenance,
        "incomplete": bool(missing_artifacts or integrity_errors)
        or status not in COMPLETE_STATUSES,
        "missing_artifacts": missing_artifacts,
        "artifact_integrity_errors": integrity_errors,
        "parse_errors": parse_errors,
        "timestamp": _timestamp(manifest, paths),
        "git_sha": str(git_sha) if git_sha else None,
        "git_dirty": git_dirty if isinstance(git_dirty, bool) else None,
        "cell_status_counts": cell_counts,
        "primary_metric": primary_metric,
        "score_metrics": score_metrics,
        "model_comparison": model_comparison,
        "context_benchmark": context_benchmark,
        "artifacts": [_artifact_record(path, results_root) for path in sorted(artifact_paths)],
    }


def _code_references(repository_root: Path) -> list[dict[str, str]]:
    """Return links for implementation surfaces that exist in this checkout."""
    references = []
    for label, relative_path in CODE_REFERENCES:
        if (repository_root / relative_path).is_file():
            references.append(
                {
                    "label": label,
                    "path": relative_path,
                    "url": f"/files/code/{quote(relative_path, safe='/')}",
                }
            )
    return references


def build_dashboard(repository_root: Path, results_root: Path) -> dict[str, Any]:
    """Scan the live results tree and return the complete dashboard payload."""
    run_directories = sorted(
        {path.parent for name in ARTIFACT_NAMES for path in results_root.rglob(name)},
        key=lambda path: path.relative_to(results_root).as_posix(),
    )
    runs = [_build_run(directory, results_root) for directory in run_directories]
    runs.sort(key=lambda run: run["timestamp"], reverse=True)
    status_counts = Counter(run["status"] for run in runs)
    artifact_count = sum(len(run["artifacts"]) for run in runs)
    complete_run_count = sum(not run["incomplete"] for run in runs)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": results_root.relative_to(repository_root).as_posix()
        if results_root.is_relative_to(repository_root)
        else results_root.name,
        "summary": {
            "run_count": len(runs),
            "complete_run_count": complete_run_count,
            "incomplete_run_count": len(runs) - complete_run_count,
            "scored_run_count": sum(run["primary_metric"] is not None for run in runs),
            "artifact_count": artifact_count,
            "status_counts": dict(status_counts),
        },
        "runs": runs,
        "code_references": _code_references(repository_root),
    }


def _safe_child(root: Path, request_path: str) -> Path | None:
    """Resolve a URL path beneath a root and reject traversal or directories."""
    candidate = (root / unquote(request_path)).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
        return None
    return candidate


def _code_reference_path(repository_root: Path, request_path: str) -> Path | None:
    """Resolve only code files explicitly linked by the dashboard payload."""
    relative_path = unquote(request_path)
    if relative_path not in CODE_REFERENCE_PATHS:
        return None
    return _safe_child(repository_root, relative_path)


def _declared_result_path(results_root: Path, request_path: str) -> Path | None:
    """Resolve only canonical or explicitly manifest-declared result artifacts."""
    candidate = _safe_child(results_root, request_path)
    if candidate is None:
        return None
    if candidate.name in ARTIFACT_NAMES:
        return candidate
    resolved_root = results_root.resolve()
    for parent in candidate.parents:
        if not parent.is_relative_to(resolved_root):
            break
        manifest_path = parent / "manifest.json"
        if manifest_path.is_file():
            manifest, error = _read_json(manifest_path)
            declared = manifest.get("artifacts", []) if error is None and isinstance(manifest, dict) else []
            if isinstance(declared, list):
                allowed = {
                    (parent / artifact).resolve()
                    for artifact in declared
                    if isinstance(artifact, str)
                    and (parent / artifact).resolve().is_relative_to(parent.resolve())
                }
                if candidate in allowed:
                    return candidate
        if parent == resolved_root:
            break
    return None


def _handler(config: DashboardConfig) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to one immutable configuration."""

    class DashboardHandler(BaseHTTPRequestHandler):
        """Serve the dashboard shell, dynamic data, and local provenance files."""

        def do_GET(self) -> None:
            """Route a GET request."""
            path = urlsplit(self.path).path
            if path == "/api/health":
                self._send_json({"status": "ok"})
                return
            if path == "/api/dashboard":
                self._send_json(
                    build_dashboard(config.repository_root, config.results_root),
                    no_store=True,
                )
                return
            if path.startswith("/files/results/"):
                self._send_local_file(
                    _declared_result_path(
                        config.results_root, path.removeprefix("/files/results/")
                    )
                )
                return
            if path.startswith("/files/code/"):
                self._send_local_file(
                    _code_reference_path(
                        config.repository_root, path.removeprefix("/files/code/")
                    )
                )
                return
            static_path = "index.html" if path == "/" else path.removeprefix("/")
            self._send_local_file(_safe_child(config.static_root, static_path))

        def _send_json(self, payload: Any, no_store: bool = False) -> None:
            """Serialize and send a JSON response."""
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if no_store:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_local_file(self, path: Path | None) -> None:
            """Send a safe local file or a not-found response."""
            if path is None:
                self.send_error(404, "Not found")
                return
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(404, "Not found")
                return
            content_type, _ = mimetypes.guess_type(path.name)
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type or 'application/octet-stream'}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def create_server(host: str, port: int, config: DashboardConfig) -> ThreadingHTTPServer:
    """Create, but do not start, a threaded dashboard server."""
    return ThreadingHTTPServer((host, port), _handler(config))


def _port(value: str) -> int:
    """Parse a valid TCP port for argparse."""
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard until interrupted."""
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=_port, default=os.environ.get("DASHBOARD_PORT"))
    parser.add_argument("--results", type=Path, default=repository_root / "results")
    arguments = parser.parse_args(argv)
    if arguments.port is None:
        parser.error("--port is required unless DASHBOARD_PORT is set")
    results_root = arguments.results.resolve()
    if not results_root.is_dir():
        parser.error(f"results directory does not exist: {results_root}")
    config = DashboardConfig(repository_root, results_root, package_root / "static")
    server = create_server(arguments.host, arguments.port, config)
    print(f"NeuroSym dashboard: http://{arguments.host}:{arguments.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
