"""Evaluate continual memory under interleaved tasks without online weight updates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments._continual_memory_config import (
    BenchmarkTokenizer,
    ContinualMemoryConfig,
    DenseEmbeddingConfig,
    HuggingFaceTokenizer,
    InterferenceTier,
    MetricBins,
    ModelParameterTier,
    ScallopReasonerConfig,
    SlidingTokenWindow,
    TokenizerConfig,
    _finite_number,
    _increasing,
    _json_object,
    _mapping,
    _nonnegative_int,
    _positive_int,
    _sequence,
    _strict_keys,
    _string,
    _unique_names,
    load_benchmark_config,
)
from experiments._continual_memory_episodes import (
    CHECKPOINT_SPECS,
    QUERY_TEMPLATE,
    TURN_SEPARATOR,
    VISIBLE_QUERY_FIELDS,
    _checkpoint_ages,
    _distractor_text,
    _fit_sliding_window,
    _identifier_groups,
    _read_jsonl,
    _serialized_model_input,
    _stable_index,
    _stream_token_count,
    _suffix,
    _target_text,
    _validate_history_contract,
    _visible_query,
    _visible_query_text,
    build_episodes,
)
from experiments._continual_memory_evaluation import (
    DenseEmbedder,
    _age_grouped_metrics,
    _aggregate,
    _bin_label,
    _bm25_turns,
    _default_dense_embedder,
    _dense_turns,
    _dense_vectors,
    _dot,
    _grouped_metrics,
    _mark_cross_task_interference,
    _memory_growth,
    _method_aggregates,
    _prediction,
    _preference_objects,
    _selected_target_events,
    evaluate_episodes,
)
from experiments.preference_stream_injection import PreferenceStreamInjectionClient


SOURCE_ARTIFACTS = (
    "candidate_updates.jsonl",
    "events.jsonl",
    "examples.jsonl",
    "facts.jsonl",
    "queries.jsonl",
)
SOURCE_FILES = (
    "experiments/_continual_memory_config.py",
    "experiments/_continual_memory_episodes.py",
    "experiments/_continual_memory_evaluation.py",
    "experiments/continual_memory_benchmark.py",
    "experiments/private_lineage_reasoning.py",
    "experiments/preference_stream_injection.py",
    "experiments/synthetic_temporal_baselines.py",
    "experiments/synthetic_temporal_preferences.py",
    "neurosym/adapters/dense_index.py",
    "neurosym/domain/retrieval_config.py",
    "services/scallop_validator_service.py",
)


def _write_json(path: Path, value: Any) -> None:
    """Write stable formatted JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write stable JSONL records in supplied order."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    """Return one artifact's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    """Return an installed package version without failing optional metadata collection."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(repository_root: Path) -> dict[str, Any]:
    """Return current repository commit and effective dirty state."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {"git_sha": sha, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}


def _render_report(metrics: Mapping[str, Any]) -> str:
    """Render benchmark controls, baseline metrics, retention, and memory growth."""
    method_rows = [
        f"| {name} | {values['grounded_answer_accuracy']:.4f} | "
        f"{values['answer_accuracy']:.4f} | "
        f"{values['exact_evidence_hit_rate']:.4f} | {values['evidence_recall']:.4f} | "
        f"{values['stale_memory_intrusion_rate']:.4f} | "
        f"{values['retraction_compliance']:.4f} | "
        f"{values['cross_task_interference_rate']:.4f} |"
        for name, values in metrics["methods"].items()
    ]
    tier_rows = []
    for tier, methods in metrics["retention_by_interference_tier"].items():
        for method, values in methods.items():
            tier_rows.append(
                f"| {tier} | {method} | {values['grounded_answer_accuracy']:.4f} | "
                f"{values['exact_evidence_hit_rate']:.4f} |"
            )
    growth = metrics["memory_growth"]
    reasoning = metrics["scallop_reasoning_ablation"]
    reasoning_rows = []
    if reasoning.get("enabled"):
        for method, values in reasoning["methods"].items():
            reasoning_rows.append(
                f"| {method} | {values['accuracy']:.4f} | "
                f"{values['by_checkpoint_kind']['private']['accuracy']:.4f} | "
                f"{values['by_checkpoint_kind']['private-lineage-positive']['accuracy']:.4f} | "
                f"{values['by_checkpoint_kind']['private-lineage']['accuracy']:.4f} |"
            )
    stream_rows = []
    stream_ablation = metrics["scallop_stream_injection_ablation"]
    if stream_ablation.get("enabled"):
        for window, tiers in stream_ablation["by_window"].items():
            for tier, checkpoint_kinds in tiers.items():
                for checkpoint_kind, values in checkpoint_kinds.items():
                    stream_rows.append(
                        f"| {window} | {tier} | {checkpoint_kind} | "
                        f"{values['no_injection_grounded_accuracy']:.4f} | "
                        f"{values['scallop_injected_grounded_accuracy']:.4f} | "
                        f"{values['reverse_ranked_control_grounded_accuracy']:.4f} | "
                        f"{values['change_rule_only_grounded_accuracy']:.4f} | "
                        f"{values['incongruity_rule_only_grounded_accuracy']:.4f} | "
                        f"{values['paired_delta']:.4f} |"
                    )
    return "\n".join(
        [
            "# Continual Memory Benchmark",
            "",
            metrics["interpretation"]["summary"],
            "",
            metrics["interpretation"]["oracle_reasoning_control"],
            "",
            metrics["interpretation"]["answer_credit"],
            "",
            metrics["interpretation"]["cross_task_interference"],
            "",
            "Model parameter tiers are metadata only and are not evaluated.",
            "No online model weight updates occur.",
            "",
            "## Baselines",
            "",
            "| Method | Grounded accuracy | Raw answer accuracy | Exact evidence hit | Evidence recall | Stale intrusion | Retraction compliance | Causal cross-task interference |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *method_rows,
            "",
            "## Retention By Interference",
            "",
            "| Tier | Method | Grounded accuracy | Exact evidence hit |",
            "|---|---|---:|---:|",
            *tier_rows,
            "",
            "## Scallop Recursive-Reasoning Ablation",
            "",
            "| Method | Overall accuracy | Direct retraction | Unretracted lineage | Two-hop lineage retraction |",
            "|---|---:|---:|---:|---:|",
            *reasoning_rows,
            "",
            "## Scallop Stream-Injection Ablation",
            "",
            "Scallop receives only the causal structured event prefix and returns source event IDs. The injected context copies those original statements under the same token budget; it receives neither the query nor gold answer.",
            "",
            "| Window | Interference tier | Checkpoint | Raw grounded | Scallop grounded | Reverse-ranked grounded | Change rule only | Incongruity rule only | Paired delta |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
            *stream_rows,
            "",
            "## Memory Growth",
            "",
            f"- Checkpoints: {growth['checkpoint_count']}",
            f"- Mean stored turns: {growth['stored_turns_mean']:.2f}",
            f"- Maximum stored turns: {growth['stored_turns_max']}",
            f"- Mean stored tokens: {growth['stored_tokens_mean']:.2f}",
            f"- Maximum stored tokens: {growth['stored_tokens_max']}",
            "",
        ]
    )


def run_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    config_path: Path,
    config: ContinualMemoryConfig,
    tokenizer: BenchmarkTokenizer,
    *,
    embedder: DenseEmbedder | None = None,
) -> dict[str, Any]:
    """Build, evaluate, and write a lifecycle-tracked hashed benchmark bundle."""
    repository_root = Path(__file__).resolve().parents[1]
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    config_path = Path(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = ["episodes.jsonl", "predictions.jsonl", "metrics.json", "report.md"]
    preexisting_artifacts = [
        name for name in artifact_names if (output_dir / name).exists()
    ]
    runtime = {
        "python": sys.version.split()[0],
        "numpy": _package_version("numpy"),
        "rank_bm25": _package_version("rank-bm25"),
        "sentence_transformers": _package_version("sentence-transformers"),
        "tokenizers": _package_version("tokenizers"),
        "transformers": _package_version("transformers"),
    }
    manifest: dict[str, Any] = {
        "run_id": output_dir.name,
        "benchmark_version": config.benchmark_version,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": _git_state(repository_root),
        "dataset": {
            "path": str(dataset_dir),
            "split": config.source_split,
            "history_limit": config.history_limit,
        },
        "config": asdict(config),
        "config_path": str(config_path),
        "runtime": runtime,
        "tokenizer": dict(tokenizer.metadata()),
        "dense": {
            "configuration": asdict(config.dense_embedding),
            "provider": None,
        },
        "preexisting_artifacts": preexisting_artifacts,
    }
    _write_json(output_dir / "manifest.json", manifest)
    try:
        manifest["source_sha256"] = {
            name: _sha256(repository_root / name) for name in SOURCE_FILES
        }
        manifest["config_sha256"] = _sha256(config_path)
        manifest["dataset"]["sha256"] = {
            name: _sha256(dataset_dir / name) for name in SOURCE_ARTIFACTS
        }
        _write_json(output_dir / "manifest.json", manifest)
        episodes = build_episodes(dataset_dir, config, tokenizer)
        stream_injection_client = (
            PreferenceStreamInjectionClient(
                config.scallop_reasoner.endpoint,
                config.scallop_reasoner.timeout_seconds,
            )
            if config.scallop_reasoner.enabled
            and config.scallop_reasoner.endpoint is not None
            else None
        )
        metrics, predictions = evaluate_episodes(
            episodes,
            config,
            embedder=embedder,
            stream_injection_client=stream_injection_client,
            tokenizer=tokenizer if stream_injection_client is not None else None,
        )
        metrics["benchmark_version"] = config.benchmark_version
        metrics["tokenizer"] = dict(tokenizer.metadata())
        metrics["runtime"] = runtime
        metrics["dense"] = {
            "configuration": asdict(config.dense_embedding),
            "provider": metrics.get("dense_metadata"),
        }
        _write_jsonl(output_dir / "episodes.jsonl", episodes)
        _write_jsonl(output_dir / "predictions.jsonl", predictions)
        _write_json(output_dir / "metrics.json", metrics)
        (output_dir / "report.md").write_text(_render_report(metrics), encoding="utf-8")
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "dense": metrics["dense"],
                "artifacts": artifact_names,
                "artifact_sha256": {
                    name: _sha256(output_dir / name) for name in artifact_names
                },
            }
        )
        _write_json(output_dir / "manifest.json", manifest)
        return metrics
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)},
                "artifacts_present_at_failure": [
                    name for name in artifact_names if (output_dir / name).exists()
                ],
            }
        )
        _write_json(output_dir / "manifest.json", manifest)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured continual-memory benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = load_benchmark_config(arguments.config)
    metrics = run_benchmark(
        arguments.dataset,
        arguments.output_dir,
        arguments.config,
        config,
        HuggingFaceTokenizer(config.tokenizer),
    )
    for method, values in metrics["methods"].items():
        print(
            f"{method}: grounded_answer_accuracy="
            f"{values['grounded_answer_accuracy']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
