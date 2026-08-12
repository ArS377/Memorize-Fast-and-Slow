"""Run the synthetic temporal benchmark against an actual Scallop HTTP validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from experiments.synthetic_temporal_baselines import (
    answer_bm25_raw_events,
    answer_full_history,
    answer_no_history,
    answer_recency,
    evaluate_batched_dense_raw_events,
    group_records_by_history,
)
from experiments.synthetic_temporal_preferences import (
    DATASET_VERSION,
    DEFAULT_HARDNESS_PROFILE,
    HARDNESS_PROFILES,
    RENDERING_CONDITIONS,
    generate_dataset,
    preference_rule_parameters,
)
from neurosym.adapters.validation_backend import HttpScallopValidatorBackend
from neurosym.domain.retrieval_config import DEFAULT_EMBEDDING_MODEL


POLICIES = ("accept_all", "scallop_fail_closed")
BASELINES = ("no_history", "recency", "full_history", "bm25")
QUERY_KINDS = (
    "preference",
    "recommendation",
    "private_recall",
    "ambiguity",
    "private_lineage",
)
DenseEvaluator = Callable[..., tuple[list[str | None], dict[str, Any]]]


def _write_json(path: Path, value: Any) -> None:
    """Write one deterministic JSON document, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write JSONL records in their supplied deterministic order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records from a benchmark artifact."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_scallop_health(backend: HttpScallopValidatorBackend) -> dict[str, Any]:
    """Verify the live HTTP service is backed by actual Scallop."""
    health = backend._request("GET", "/health", None)
    if health.get("engine") != "scallopy" or health.get("scallop_available") is not True:
        raise RuntimeError("validator health does not confirm an available scallopy engine")
    return health


def replay_candidates(
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    policy: str,
    backend: HttpScallopValidatorBackend,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay candidate writes locally, retaining validator-directed replacements."""
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    replayed = list(events)
    facts = [event["fact"] for event in replayed]
    decisions: list[dict[str, Any]] = []
    rule_params = preference_rule_parameters()
    for candidate in candidates:
        if policy == "accept_all":
            decision = "accept"
            reason = "accept_all_policy"
            replace_fact_id = None
            rule_params_version = rule_params.version
            rejection_label = None
        else:
            validated = backend.validate(facts, candidate["fact"], rule_params)
            if validated.rule_params_version != rule_params.version:
                raise RuntimeError(
                    "validator response rule version does not match synthetic benchmark: "
                    f"expected {rule_params.version!r}, received {validated.rule_params_version!r}"
                )
            decision = validated.decision
            reason = validated.reason
            replace_fact_id = validated.replace_fact_id
            rule_params_version = validated.rule_params_version
            rejection_label = (
                validated.rejection_label.to_dict() if validated.rejection_label is not None else None
            )
        record = {
            "candidate_id": candidate["candidate_id"],
            "history_id": candidate["history_id"],
            "policy": policy,
            "decision": decision,
            "reason": reason,
            "replace_fact_id": replace_fact_id,
            "rejection_label": rejection_label,
            "rule_params_version": rule_params_version,
            "gold_decision": candidate["gold_decision"],
            "admitted": decision in {"accept", "replace"},
            "correct": decision == candidate["gold_decision"],
        }
        decisions.append(record)
        if decision == "reject":
            continue
        if decision not in {"accept", "replace"}:
            raise RuntimeError(f"validator returned unsupported decision {decision!r}")
        if decision == "replace" and replace_fact_id is None:
            raise RuntimeError("validator returned replace without replace_fact_id")
        if replace_fact_id is not None:
            if not any(fact["fact_id"] == replace_fact_id for fact in facts):
                raise RuntimeError(f"validator requested replacement of absent fact {replace_fact_id!r}")
            facts = [fact for fact in facts if fact["fact_id"] != replace_fact_id]
            replayed = [event for event in replayed if event["fact"]["fact_id"] != replace_fact_id]
        facts.append(candidate["fact"])
        replayed.append({
            "event_id": f"replay-{candidate['candidate_id']}",
            "history_id": candidate["history_id"],
            "session_id": f"{candidate['history_id']}-candidate-replay",
            "turn_index": 1,
            "operation": "add",
            "fact": candidate["fact"],
        })
    return replayed, decisions


def _evaluate_dense_raw_events(
    replayed_by_history: dict[str, list[dict[str, Any]]],
    queries: list[dict[str, Any]],
    *,
    k: int,
    model: str,
    revision: str | None,
    batch_size: int,
) -> tuple[list[str | None], dict[str, Any]]:
    """Evaluate existing BGE raw-event retrieval and expose its run metadata."""
    from neurosym.adapters.dense_index import SentenceTransformerEmbedder
    from neurosym.domain.retrieval_config import EmbeddingConfig

    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(
            model=model,
            requested_revision=revision,
            device="cpu",
            batch_size=batch_size,
        )
    )
    return evaluate_batched_dense_raw_events(
        replayed_by_history,
        queries,
        k=k,
        embedder=embedder,
        model=model,
        revision=revision,
        batch_size=batch_size,
    )


def _evaluate_policy(
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    condition: str,
    policy: str,
    backend: HttpScallopValidatorBackend,
    bm25_k: int,
    include_dense: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_revision: str | None = None,
    embedding_batch_size: int = 32,
    dense_evaluator: DenseEvaluator = _evaluate_dense_raw_events,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Admit candidates per history, then produce predictions for every baseline."""
    event_groups = group_records_by_history(events)
    candidate_groups = group_records_by_history(candidates)
    query_groups = group_records_by_history(queries)
    history_ids = sorted(event_groups.keys() | candidate_groups.keys() | query_groups.keys())
    replayed_by_history: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    for history_id in history_ids:
        replayed, history_decisions = replay_candidates(
            event_groups.get(history_id, []),
            candidate_groups.get(history_id, []),
            policy=policy,
            backend=backend,
        )
        replayed_by_history[history_id] = replayed
        decisions.extend(history_decisions)

    methods: dict[str, Callable[[list[dict[str, Any]], dict[str, Any]], str | None]] = {
        "no_history": answer_no_history,
        "recency": answer_recency,
        "full_history": answer_full_history,
        "bm25": lambda history_events, query: answer_bm25_raw_events(history_events, query, k=bm25_k),
    }
    predictions = {baseline: [] for baseline in BASELINES}
    for query in queries:
        visible_query = {key: value for key, value in query.items() if key != "gold"}
        for baseline, method in methods.items():
            prediction = method(replayed_by_history[query["history_id"]], visible_query)
            predictions[baseline].append({
                "query_id": query["query_id"],
                "history_id": query["history_id"],
                "condition": condition,
                "policy": policy,
                "baseline": baseline,
                "kind": query["kind"],
                "prediction": prediction,
                "gold": query["gold"],
                "correct": prediction == query["gold"],
            })
    dense_metadata = None
    if include_dense:
        dense_answers, dense_metadata = dense_evaluator(
            replayed_by_history,
            queries,
            k=bm25_k,
            model=embedding_model,
            revision=embedding_revision,
            batch_size=embedding_batch_size,
        )
        if len(dense_answers) != len(queries):
            raise RuntimeError("dense evaluator returned a prediction count that does not match queries")
        predictions["dense"] = [
            {
                "query_id": query["query_id"],
                "history_id": query["history_id"],
                "condition": condition,
                "policy": policy,
                "baseline": "dense",
                "kind": query["kind"],
                "prediction": prediction,
                "gold": query["gold"],
                "correct": prediction == query["gold"],
            }
            for query, prediction in zip(queries, dense_answers)
        ]
    metrics = {
        "candidate_decision_accuracy": (
            sum(record["correct"] for record in decisions) / len(decisions) if decisions else 0.0
        ),
        "candidate_accept_count": sum(record["admitted"] for record in decisions),
        "candidate_reject_count": sum(record["decision"] == "reject" for record in decisions),
        "query_accuracy": {
            baseline: sum(record["correct"] for record in records) / len(records) if records else 0.0
            for baseline, records in predictions.items()
        },
        "query_accuracy_by_kind": {
            baseline: {
                kind: (
                    sum(record["correct"] for record in records if record["kind"] == kind)
                    / sum(record["kind"] == kind for record in records)
                    if any(record["kind"] == kind for record in records)
                    else 0.0
                )
                for kind in QUERY_KINDS
            }
            for baseline, records in predictions.items()
        },
    }
    if dense_metadata is not None:
        metrics["dense_metadata"] = dense_metadata
    return decisions, predictions, metrics


def _render_report(manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Render a compact Markdown result table from complete benchmark artifacts."""
    dense_enabled = bool(manifest.get("dense", {}).get("enabled", False))
    lines = [
        "# Synthetic Temporal Benchmark",
        "",
        f"Status: {manifest['status']}",
        f"Rule version: {manifest['rule_parameters']['version']}",
        "",
        "| Condition | Policy | Decision accuracy | no_history | recency | full_history | bm25"
        + (" | dense |" if dense_enabled else " |"),
        "| --- | --- | ---: | ---: | ---: | ---: | ---:"
        + (" | ---: |" if dense_enabled else " |"),
    ]
    for condition in RENDERING_CONDITIONS:
        for policy in POLICIES:
            result = metrics["conditions"][condition][policy]
            accuracy = result["query_accuracy"]
            row = (
                f"| {condition} | {policy} | {result['candidate_decision_accuracy']:.3f} | "
                f"{accuracy['no_history']:.3f} | {accuracy['recency']:.3f} | "
                f"{accuracy['full_history']:.3f} | {accuracy['bm25']:.3f}"
            )
            lines.append(row + (f" | {accuracy['dense']:.3f} |" if dense_enabled else " |"))
    query_kinds = sorted(
        {
            kind
            for condition in metrics["conditions"].values()
            for policy in condition.values()
            for by_kind in policy["query_accuracy_by_kind"].values()
            for kind in by_kind
        }
    )
    lines.extend([
        "",
        "## Query Accuracy by Kind",
        "",
        "| Condition | Policy | Baseline | Aggregate | "
        + " | ".join(query_kinds)
        + " |",
        "| --- | --- | --- | ---: | " + " | ".join("---:" for _ in query_kinds) + " |",
    ])
    for condition in RENDERING_CONDITIONS:
        for policy in POLICIES:
            result = metrics["conditions"][condition][policy]
            for baseline in manifest["baselines"]:
                by_kind = result["query_accuracy_by_kind"][baseline]
                values = " | ".join(f"{by_kind[kind]:.3f}" for kind in query_kinds)
                lines.append(
                    f"| {condition} | {policy} | {baseline} | "
                    f"{result['query_accuracy'][baseline]:.3f} | {values} |"
                )
    return "\n".join(lines) + "\n"


def run_benchmark(
    output_dir: Path,
    *,
    scallop_validator_url: str,
    history_count: int = 1,
    timeout: float = 30.0,
    bm25_k: int = 5,
    include_dense: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_revision: str | None = None,
    embedding_batch_size: int = 32,
    dense_evaluator: DenseEvaluator = _evaluate_dense_raw_events,
    hardness_profile: str = DEFAULT_HARDNESS_PROFILE,
) -> dict[str, Any]:
    """Generate both renderings and evaluate candidate policies and selected baselines."""
    if history_count < 1:
        raise ValueError(f"history_count must be positive, got {history_count}")
    if bm25_k < 1:
        raise ValueError(f"bm25_k must be positive, got {bm25_k}")
    if embedding_batch_size < 1:
        raise ValueError(f"embedding_batch_size must be positive, got {embedding_batch_size}")
    output_dir = Path(output_dir)
    rule_params = preference_rule_parameters()
    manifest: dict[str, Any] = {
        "benchmark_version": DATASET_VERSION,
        "status": "running",
        "history_count": history_count,
        "policies": list(POLICIES),
        "baselines": list(BASELINES) + (["dense"] if include_dense else []),
        "rule_parameters": rule_params.to_dict(),
        "scallop_validator_url": scallop_validator_url,
        "timeout_seconds": timeout,
        "bm25_k": bm25_k,
        "hardness_profile": hardness_profile,
    }
    if include_dense:
        manifest["dense"] = {
            "enabled": True,
            "embedding_requested_model": embedding_model,
            "embedding_requested_revision": embedding_revision,
            "embedding_batch_size": embedding_batch_size,
            "retrieval_k": bm25_k,
            "retrieval_unit": "raw_event_support_text",
        }
    _write_json(output_dir / "manifest.json", manifest)
    try:
        backend = HttpScallopValidatorBackend(scallop_validator_url, timeout=timeout)
        health = validate_scallop_health(backend)
        manifest["validator"] = backend.info.to_dict()
        manifest["validator_health"] = health
        metrics: dict[str, Any] = {"conditions": {}}
        for condition in RENDERING_CONDITIONS:
            paths = generate_dataset(
                output_dir / "datasets" / condition,
                history_count,
                condition,
                hardness_profile=hardness_profile,
            )
            events = _read_jsonl(paths["events"])
            candidates = _read_jsonl(paths["candidates"])
            queries = _read_jsonl(paths["queries"])
            metrics["conditions"][condition] = {}
            for policy in POLICIES:
                decisions, predictions, result = _evaluate_policy(
                    events,
                    candidates,
                    queries,
                    condition=condition,
                    policy=policy,
                    backend=backend,
                    bm25_k=bm25_k,
                    include_dense=include_dense,
                    embedding_model=embedding_model,
                    embedding_revision=embedding_revision,
                    embedding_batch_size=embedding_batch_size,
                    dense_evaluator=dense_evaluator,
                )
                _write_jsonl(output_dir / "decisions" / condition / f"{policy}.jsonl", decisions)
                for baseline, records in predictions.items():
                    _write_jsonl(
                        output_dir / "predictions" / condition / policy / f"{baseline}.jsonl", records
                    )
                metrics["conditions"][condition][policy] = result
        if include_dense:
            manifest["dense"]["metadata"] = metrics["conditions"][
                RENDERING_CONDITIONS[0]
            ][POLICIES[0]]["dense_metadata"]
        _write_json(output_dir / "metrics.json", metrics)
        manifest["status"] = "completed"
        _write_json(output_dir / "manifest.json", manifest)
        (output_dir / "report.md").write_text(_render_report(manifest, metrics), encoding="utf-8")
        return {"manifest": manifest, "metrics": metrics}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output_dir / "manifest.json", manifest)
        raise


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the benchmark from its command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scallop-validator-url", required=True)
    parser.add_argument("--history-count", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--bm25-k", type=int, default=5)
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--hardness-profile",
        choices=HARDNESS_PROFILES,
        default=DEFAULT_HARDNESS_PROFILE,
    )
    args = parser.parse_args(argv)
    result = run_benchmark(
        args.output_dir,
        scallop_validator_url=args.scallop_validator_url,
        history_count=args.history_count,
        timeout=args.timeout,
        bm25_k=args.bm25_k,
        include_dense=args.include_dense,
        embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision,
        embedding_batch_size=args.embedding_batch_size,
        hardness_profile=args.hardness_profile,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
