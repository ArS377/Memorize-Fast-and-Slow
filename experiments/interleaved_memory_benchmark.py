"""Evaluate finite-suffix context rot in one evolving multi-task conversation."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments._continual_memory_config import HuggingFaceTokenizer, TokenizerConfig
from experiments._continual_memory_episodes import (
    QUERY_TEMPLATE,
    _visible_query,
    _visible_query_text,
)
from experiments.contradiction_ledger import (
    derive_contradiction_ledger,
    ContradictionLedgerClient,
)
from experiments.interleaved_conversation import (
    build_interleaved_schedule,
    lifecycle_distribution,
    schedule_metrics,
    suffix_event_ids_by_window,
)
from experiments.synthetic_temporal_preferences import resolve_query


BENCHMARK_VERSION = "interleaved_continual_memory.v2"
HORIZONS = (8, 16, 32, 64, 128, 256, 512, 1024)
WINDOWS = (4096, 16384, 65536, 131072, 262144, 1048576)
DEFAULT_CONTEXT_MULTIPLIER = 2.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read object-valued JSONL."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: Any) -> None:
    """Write stable formatted JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write stable JSONL."""
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Return one file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_context_horizon(
    stream_token_count: int,
    *,
    declared_context_limit: int,
    multiplier: float,
) -> float:
    """Require the generated stream to exceed the declared finite context horizon."""
    if declared_context_limit < 1 or multiplier <= 1.0:
        raise ValueError("declared context limit must be positive and multiplier must exceed 1")
    required_tokens = math.ceil(declared_context_limit * multiplier)
    if stream_token_count < required_tokens:
        raise ValueError(
            f"stream has {stream_token_count} tokens but declared horizon requires "
            f"at least {required_tokens} ({multiplier}x {declared_context_limit})"
        )
    return stream_token_count / declared_context_limit


def _fit_online_window(
    turns: Sequence[Mapping[str, Any]],
    cumulative_tokens: Sequence[int],
    query_text: str,
    max_tokens: int,
    tokenizer: Any,
) -> dict[str, int]:
    """Fit an exact complete-turn suffix using precomputed stream token totals."""
    prompt = QUERY_TEMPLATE.format(query_text=query_text)
    prompt_token_count = len(tokenizer.encode(prompt))
    if prompt_token_count > max_tokens:
        raise ValueError(
            f"checkpoint query prompt requires {prompt_token_count} tokens but window allows "
            f"{max_tokens}"
        )
    if not turns:
        return {
            "start_turn_index": 0,
            "model_input_token_count": prompt_token_count,
        }
    if len(cumulative_tokens) != len(turns):
        raise ValueError("cumulative token count must align with the checkpoint prefix")
    last_text = str(turns[-1]["text"])
    prompt_increment = len(tokenizer.encode(f"{last_text}{prompt}")) - len(
        tokenizer.encode(last_text)
    )

    def exact_count(start: int) -> int:
        if start == len(turns):
            return prompt_token_count
        internal_increments = cumulative_tokens[-1] - cumulative_tokens[start]
        return int(turns[start]["token_count"]) + internal_increments + prompt_increment

    low = 0
    high = len(turns)
    while low < high:
        midpoint = (low + high) // 2
        if exact_count(midpoint) <= max_tokens:
            high = midpoint
        else:
            low = midpoint + 1
    return {
        "evaluation_family": "deterministic_oracle_resolver",
        "start_turn_index": low,
        "model_input_token_count": exact_count(low),
    }


def _historical_values(
    history_prefix: Sequence[Mapping[str, Any]], query: Mapping[str, Any]
) -> set[str]:
    """Return prior values that make an incorrect prediction a stale intrusion."""
    return {
        str(turn["stream_event"]["fact"]["object"])
        for turn in history_prefix
        if isinstance(turn["stream_event"].get("fact"), Mapping)
        and turn["stream_event"]["fact"].get("subject") == query.get("subject")
        and (
            query.get("kind") != "preference"
            or (
                isinstance(turn["stream_event"]["fact"].get("qualifiers"), Mapping)
                and turn["stream_event"]["fact"]["qualifiers"].get("scope", "default")
                == query.get("scope", "default")
            )
        )
    }


def _retention_contracts(history_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Return final-sweep contracts for long-running state changes."""
    contracts = []
    for history_id in history_ids:
        contracts.extend(
            [
                {
                    "history_id": history_id,
                    "relation": "preference_change",
                    "required_event_id_groups": [
                        [f"{history_id}-add"],
                        [f"{history_id}-transition"],
                    ],
                },
                {
                    "history_id": history_id,
                    "relation": "authority_incongruity",
                    "required_event_id_groups": [
                        [f"{history_id}-indirect-source"],
                        [f"{history_id}-direct-correction"],
                    ],
                },
                {
                    "history_id": history_id,
                    "relation": "contradiction_rectification",
                    "required_event_id_groups": [
                        [f"{history_id}-conflict-left"],
                        [f"{history_id}-conflict-right"],
                        [f"{history_id}-conflict-resolution"],
                    ],
                },
                {
                    "history_id": history_id,
                    "relation": "backdated_correction",
                    "required_event_id_groups": [
                        [f"{history_id}-add"],
                        [f"{history_id}-backdated"],
                    ],
                },
            ]
        )
    return contracts


def _online_prediction(
    query: Mapping[str, Any],
    history_prefix: Sequence[Mapping[str, Any]],
    selected_history_turns: Sequence[Mapping[str, Any]],
    *,
    method: str,
    checkpoint_turn_index: int,
    selected_turn_count: int,
    selected_start_turn_id: str,
    selected_end_turn_id: str,
    turn_age: int,
    token_age: int,
    stream_position_percentage: float,
) -> dict[str, Any]:
    """Resolve and score one memory view at one causal checkpoint."""
    contract = query.get("checkpoint_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"query {query.get('query_id')} lacks checkpoint_contract")
    event_groups = [
        [str(identifier) for identifier in group]
        for group in contract.get("required_event_id_groups", [])
    ]
    fact_groups = [
        [str(identifier) for identifier in group]
        for group in contract.get("required_fact_id_groups", [])
    ]
    if not event_groups or not fact_groups:
        raise ValueError(f"query {query.get('query_id')} has an empty evidence contract")
    selected_events = [dict(turn["stream_event"]) for turn in selected_history_turns]
    selected_event_ids = {str(event["event_id"]) for event in selected_events}
    selected_fact_ids = {
        str(event["fact"]["fact_id"])
        for event in selected_events
        if isinstance(event.get("fact"), Mapping)
    }
    event_hits = [bool(selected_event_ids & set(group)) for group in event_groups]
    fact_hits = [bool(selected_fact_ids & set(group)) for group in fact_groups]
    evidence_hits = event_hits + fact_hits
    answer = resolve_query(selected_events, dict(_visible_query(query)))
    gold = query.get("gold")
    answer_correct = answer == gold
    complete_provenance = all(evidence_hits)
    historical_values = _historical_values(history_prefix, query)
    return {
        "history_id": str(query["history_id"]),
        "query_id": str(query["query_id"]),
        "query_kind": str(query.get("kind", "preference")),
        "trigger_event_id": str(contract["trigger_event_id"]),
        "method": method,
        "checkpoint_turn_index": checkpoint_turn_index,
        "selected_through_turn_index": checkpoint_turn_index,
        "stream_position_percentage": stream_position_percentage,
        "turn_age": turn_age,
        "token_age": token_age,
        "selected_turn_count": selected_turn_count,
        "selected_history_turn_count": len(selected_history_turns),
        "selected_start_turn_id": selected_start_turn_id,
        "selected_end_turn_id": selected_end_turn_id,
        "selected_event_ids": sorted(selected_event_ids),
        "selected_fact_ids": sorted(selected_fact_ids),
        "prediction": answer,
        "gold": gold,
        "answer_correct": answer_correct,
        "complete_provenance": complete_provenance,
        "grounded_answer_correct": answer_correct and complete_provenance,
        "evidence_recall": sum(evidence_hits) / len(evidence_hits),
        "stale_memory_intrusion": answer != gold and answer in historical_values,
    }


def evaluate_online_checkpoints(
    schedule: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    windows: Sequence[int],
) -> list[dict[str, Any]]:
    """Compare structured memory and finite context at each query's causal trigger."""
    turns = list(schedule["turns"])
    turn_index_by_event_id = {
        str(turn["stream_event"]["event_id"]): index
        for index, turn in enumerate(turns)
    }
    cumulative_tokens = []
    token_count = 0
    for turn in turns:
        token_count += int(turn["serialized_token_count"])
        cumulative_tokens.append(token_count)
    turns_by_history: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, turn in enumerate(turns):
        turns_by_history.setdefault(str(turn["history_id"]), []).append((index, turn))
    checkpoint_queries = [
        query for query in queries if isinstance(query.get("checkpoint_contract"), Mapping)
    ]
    rows = []
    for query in sorted(checkpoint_queries, key=lambda item: str(item["query_id"])):
        history_id = str(query["history_id"])
        contract = query["checkpoint_contract"]
        trigger_event_id = str(contract["trigger_event_id"])
        trigger_index = turn_index_by_event_id.get(trigger_event_id)
        if trigger_index is None:
            raise ValueError(
                f"query {query.get('query_id')} references unscheduled trigger {trigger_event_id}"
            )
        prefix = turns[: trigger_index + 1]
        indexed_history_turns = turns_by_history.get(history_id, [])
        history_positions = [position for position, _ in indexed_history_turns]
        history_prefix_end = bisect_left(history_positions, trigger_index + 1)
        structured = [turn for _, turn in indexed_history_turns[:history_prefix_end]]
        event_groups = [
            [str(identifier) for identifier in group]
            for group in contract.get("required_event_id_groups", [])
        ]
        evidence_positions = []
        for group in event_groups:
            positions = [turn_index_by_event_id[event_id] for event_id in group if event_id in turn_index_by_event_id and turn_index_by_event_id[event_id] <= trigger_index]
            if not positions:
                raise ValueError(
                    f"query {query.get('query_id')} has unavailable evidence group {group}"
                )
            evidence_positions.append(max(positions))
        turn_age = max(trigger_index - position for position in evidence_positions)
        token_age = max(
            cumulative_tokens[trigger_index] - cumulative_tokens[position]
            for position in evidence_positions
        )
        stream_position = 100.0 * (trigger_index + 1) / len(turns)
        full_row = _online_prediction(
            query,
            structured,
            structured,
            method="full_structured_memory",
            checkpoint_turn_index=trigger_index + 1,
            selected_turn_count=len(structured),
            selected_start_turn_id=str(structured[0]["turn_id"]),
            selected_end_turn_id=str(structured[-1]["turn_id"]),
            turn_age=turn_age,
            token_age=token_age,
            stream_position_percentage=stream_position,
        )
        if not full_row["grounded_answer_correct"]:
            raise RuntimeError(
                f"full structured memory failed checkpoint {query.get('query_id')}: "
                f"prediction={full_row['prediction']!r}, gold={full_row['gold']!r}, "
                f"complete_provenance={full_row['complete_provenance']}"
            )
        rows.append(full_row)
        query_text = str(query.get("query_text") or "")
        if not query_text:
            raise ValueError(f"query {query.get('query_id')} lacks model-visible query_text")
        for window in windows:
            window_size = int(window)
            fitted = _fit_online_window(
                prefix,
                cumulative_tokens[: trigger_index + 1],
                query_text,
                window_size,
                tokenizer,
            )
            start = int(fitted["start_turn_index"])
            history_start = bisect_left(history_positions, start)
            selected_history_turns = [
                turn
                for _, turn in indexed_history_turns[history_start:history_prefix_end]
            ]
            row = _online_prediction(
                query,
                structured,
                selected_history_turns,
                method=f"sliding_context:{window_size}",
                checkpoint_turn_index=trigger_index + 1,
                selected_turn_count=len(prefix) - start,
                selected_start_turn_id=(
                    str(prefix[start]["turn_id"]) if start < len(prefix) else ""
                ),
                selected_end_turn_id=str(prefix[-1]["turn_id"]),
                turn_age=turn_age,
                token_age=token_age,
                stream_position_percentage=stream_position,
            )
            row["max_tokens"] = window_size
            row["model_input_token_count"] = int(fitted["model_input_token_count"])
            rows.append(row)
    return rows


def _aggregate_online_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate answer-sufficient and provenance-complete checkpoint metrics."""
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    return {
        method: {
            "checkpoint_count": len(method_rows),
            "oracle_resolver_answer_accuracy": sum(
                bool(row["answer_correct"]) for row in method_rows
            ) / len(method_rows),
            "complete_provenance_rate": sum(
                bool(row["complete_provenance"]) for row in method_rows
            )
            / len(method_rows),
            "grounded_answer_accuracy": sum(
                bool(row["grounded_answer_correct"]) for row in method_rows
            )
            / len(method_rows),
            "evidence_recall": sum(float(row["evidence_recall"]) for row in method_rows)
            / len(method_rows),
            "stale_memory_intrusion_rate": sum(
                bool(row["stale_memory_intrusion"]) for row in method_rows
            )
            / len(method_rows),
        }
        for method, method_rows in sorted(by_method.items())
    }


def _render_report(metrics: Mapping[str, Any]) -> str:
    """Render a concise report with the benchmark's interpretation limits."""
    horizon = metrics["horizons"][-1]
    methods = metrics["online_checkpoint_evaluation"]["methods"]
    context = metrics["context_horizon_contract"]
    ledger = metrics["contradiction_ledger"]
    return "\n".join(
        [
            "# Interleaved Continual Memory Benchmark",
            "",
            f"- Benchmark version: `{metrics['benchmark_version']}`",
            f"- Evolving peer tasks: {horizon['horizon_accounts']:,}",
            f"- Stream tokens: {horizon['stream_token_count']:,}",
            f"- Declared context coverage: {context['realized_multiplier']:.3f}x",
            f"- Canonical contradiction pairs: {ledger['pair_count']:,} "
            f"({ledger['resolved_pair_count']:,} resolved)",
            "",
            "## Final Transcript-Suffix Retention",
            "",
            "These loss rates reserve the declared budget for transcript turns only; they do not include a query prompt.",
            "",
            *[
                f"- {int(window):,} tokens: {rate:.2%} complete-provenance loss"
                for window, rate in sorted(
                    horizon["suffix_contract_loss_rate"].items(),
                    key=lambda item: int(item[0]),
                )
            ],
            "",
            "## Online Causal Checkpoints",
            "",
            "Online windows include the exact query prompt in their token budget.",
            "Answer sufficiency uses the disclosed deterministic oracle resolver and is not LLM generation accuracy.",
            "",
            *[
                f"- {name}: oracle answer {values['oracle_resolver_answer_accuracy']:.2%}; "
                f"complete provenance {values['complete_provenance_rate']:.2%}; "
                f"grounded answer {values['grounded_answer_accuracy']:.2%}"
                for name, values in sorted(methods.items())
            ],
            "",
        ]
    )


def run_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    *,
    use_scallop: bool = False,
    declared_context_limit: int = max(WINDOWS),
    context_multiplier: float = DEFAULT_CONTEXT_MULTIPLIER,
) -> dict[str, Any]:
    """Generate horizon schedules, ledgers, suffix-loss curves, and witnesses."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = HuggingFaceTokenizer(
        TokenizerConfig(
            name="Qwen/Qwen3-4B",
            revision="1cfa9a7208912126459214e8b04321603b3df60c",
            local_files_only=True,
        )
    )
    all_events = [
        event
        for event in _read_jsonl(dataset_dir / "events.jsonl")
        if event.get("split") == "test"
        and event.get("hardness_profile") == "anti_shortcut_interleaved_v3"
    ]
    history_ids = sorted({str(event["history_id"]) for event in all_events})
    if len(history_ids) < max(HORIZONS):
        raise ValueError(
            f"interleaved benchmark needs {max(HORIZONS)} test histories, found {len(history_ids)}"
        )
    horizon_metrics = []
    final_schedule = None
    final_ledgers = []
    final_online_predictions = []
    witnesses = []
    all_queries = [
        query
        for query in _read_jsonl(dataset_dir / "queries.jsonl")
        if query.get("split") == "test"
        and query.get("hardness_profile") == "anti_shortcut_interleaved_v3"
    ]
    for horizon in HORIZONS:
        selected_ids = history_ids[:horizon]
        selected_events = [
            event for event in all_events if event["history_id"] in set(selected_ids)
        ]
        schedule = build_interleaved_schedule(
            selected_events,
            tokenizer=tokenizer,
            seed=47,
            concurrent_accounts=min(8, horizon),
            min_segment_events=4,
            max_segment_events=8,
        )
        contracts = _retention_contracts(selected_ids)
        suffix_event_ids = suffix_event_ids_by_window(
            schedule["turns"], windows=WINDOWS
        )
        available_counts = {window: 0 for window in WINDOWS}
        relation_counts: dict[str, dict[int, int]] = {}
        for contract in contracts:
            availability = {
                window: all(
                    set(group) & retained_ids
                    for group in contract["required_event_id_groups"]
                )
                for window, retained_ids in suffix_event_ids.items()
            }
            relation_counts.setdefault(
                contract["relation"], {window: 0 for window in WINDOWS}
            )
            for window, available in availability.items():
                available_counts[window] += available
                relation_counts[contract["relation"]][window] += available
                if not available and len(witnesses) < 100:
                    witnesses.append(
                        {
                            "horizon": horizon,
                            "window": window,
                            **contract,
                        }
                    )
        schedule_summary = schedule_metrics(schedule)
        distribution = lifecycle_distribution(schedule)
        if schedule_summary["adjacent_same_account_segments"] != 0:
            raise RuntimeError(
                f"horizon {horizon} produced adjacent same-account segments"
            )
        if schedule_summary["causal_order_violations"] != 0:
            raise RuntimeError(f"horizon {horizon} violated account event order")
        if schedule_summary["minimum_intervening_accounts_per_resume"] < 1:
            raise RuntimeError(
                f"horizon {horizon} contains a resumption without intervening work"
            )
        if horizon >= 32:
            missing_thirds = sorted(
                family
                for family, values in distribution.items()
                if values["total"] and not values["covers_all_stream_thirds"]
            )
            if missing_thirds:
                raise RuntimeError(
                    f"horizon {horizon} clusters lifecycle families outside stream thirds: "
                    f"{missing_thirds}"
                )
        total_contracts = len(contracts)
        horizon_metrics.append(
            {
                "horizon_accounts": horizon,
                "stream_token_count": sum(
                    int(turn["serialized_token_count"]) for turn in schedule["turns"]
                ),
                "schedule": schedule_summary,
                "lifecycle_distribution": distribution,
                "suffix_contract_loss_rate": {
                    str(window): 1.0 - available_counts[window] / total_contracts
                    for window in WINDOWS
                },
                "suffix_contract_loss_by_relation": {
                    relation: {
                        str(window): 1.0
                        - available / sum(
                            contract["relation"] == relation for contract in contracts
                        )
                        for window, available in by_window.items()
                    }
                    for relation, by_window in relation_counts.items()
                },
            }
        )
        if horizon == max(HORIZONS):
            final_schedule = schedule
            final_online_predictions = evaluate_online_checkpoints(
                schedule,
                [query for query in all_queries if query["history_id"] in set(selected_ids)],
                tokenizer=tokenizer,
                windows=WINDOWS,
            )
            ledger_client = None
            if use_scallop:
                endpoint = os.environ.get("SCALLOP_VALIDATOR_URL")
                if not endpoint:
                    raise ValueError(
                        "--scallop requires SCALLOP_VALIDATOR_URL"
                    )
                ledger_client = ContradictionLedgerClient(endpoint)
            final_ledgers = [
                (
                    ledger_client.derive(history_events)
                    if ledger_client is not None
                    else derive_contradiction_ledger(history_events)
                )
                | {"history_id": history_id}
                for history_id in selected_ids
                for history_events in [[
                        event
                        for event in selected_events
                        if event["history_id"] == history_id
                    ]]
            ]
    if final_schedule is None:
        raise RuntimeError("interleaved benchmark produced no final schedule")
    final_stream_tokens = sum(
        int(turn["serialized_token_count"]) for turn in final_schedule["turns"]
    )
    realized_context_multiplier = validate_context_horizon(
        final_stream_tokens,
        declared_context_limit=declared_context_limit,
        multiplier=context_multiplier,
    )
    unexpected = [
        pair_id
        for ledger in final_ledgers
        for pair_id in ledger["unexpected_unresolved_pair_ids"]
    ]
    metrics = {
        "benchmark_version": BENCHMARK_VERSION,
        "generator": "custom deterministic Python",
        "external_scheduler_library": None,
        "tokenizer": dict(tokenizer.metadata()),
        "horizons": horizon_metrics,
        "online_checkpoint_evaluation": {
            "evaluation_horizon_accounts": max(HORIZONS),
            "causal_timing": "immediately after each query trigger event",
            "structured_memory_scope": "all observed events for the queried history",
            "answer_evaluator": (
                "deterministic oracle resolver over selected structured events; "
                "not LLM generation accuracy"
            ),
            "methods": _aggregate_online_predictions(final_online_predictions),
        },
        "context_horizon_contract": {
            "declared_context_limit": declared_context_limit,
            "required_multiplier": context_multiplier,
            "realized_multiplier": realized_context_multiplier,
            "stream_token_count": final_stream_tokens,
        },
        "contradiction_ledger": {
            "engine": (
                final_ledgers[0].get("engine", "python_reference")
                if final_ledgers
                else None
            ),
            "scallopy_version": (
                final_ledgers[0].get("scallopy_version")
                if final_ledgers
                else None
            ),
            "history_count": len(final_ledgers),
            "pair_count": sum(len(ledger["pairs"]) for ledger in final_ledgers),
            "resolved_pair_count": sum(
                pair["status"] != "unresolved"
                for ledger in final_ledgers
                for pair in ledger["pairs"]
            ),
            "unexpected_unresolved_pair_ids": unexpected,
        },
        "interpretation": {
            "summary": (
                "All accounts are peer tasks in one causal conversation. Finite suffix loss is "
                "measured at final retention sweeps as the account horizon grows."
            ),
            "limitation": (
                "Text is deterministic synthetic dialogue, not human conversation or an LLM "
                "generation evaluation. Online answer scores use the same deterministic resolver "
                "as source gold and measure context availability, not independent reasoning."
            ),
        },
    }
    schedule_path = output_dir / "schedule.jsonl"
    ledger_path = output_dir / "contradiction_ledger.jsonl"
    witness_path = output_dir / "suffix_loss_witnesses.jsonl"
    online_path = output_dir / "online_checkpoint_predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    _write_jsonl(schedule_path, final_schedule["turns"])
    _write_jsonl(ledger_path, final_ledgers)
    _write_jsonl(witness_path, witnesses)
    _write_jsonl(online_path, final_online_predictions)
    _write_json(metrics_path, metrics)
    report_path.write_text(_render_report(metrics), encoding="utf-8")
    artifact_paths = (
        schedule_path,
        ledger_path,
        witness_path,
        online_path,
        metrics_path,
        report_path,
    )
    manifest = {
        "status": "completed",
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": str(dataset_dir),
        "dataset_sha256": {
            name: _sha256(dataset_dir / name)
            for name in ("events.jsonl", "queries.jsonl")
        },
        "artifacts": [path.name for path in artifact_paths],
        "artifact_sha256": {
            path.name: _sha256(path)
            for path in artifact_paths
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Run the interleaved benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scallop", action="store_true")
    parser.add_argument("--declared-context-limit", type=int, default=max(WINDOWS))
    parser.add_argument(
        "--context-multiplier", type=float, default=DEFAULT_CONTEXT_MULTIPLIER
    )
    args = parser.parse_args(argv)
    metrics = run_benchmark(
        args.dataset,
        args.output_dir,
        use_scallop=args.scallop,
        declared_context_limit=args.declared_context_limit,
        context_multiplier=args.context_multiplier,
    )
    for horizon in metrics["horizons"]:
        print(
            f"horizon={horizon['horizon_accounts']} tokens={horizon['stream_token_count']} "
            f"loss_128k={horizon['suffix_contract_loss_rate']['131072']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
