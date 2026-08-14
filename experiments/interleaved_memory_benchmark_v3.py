"""Evaluate delayed peer-task memory with imperfect query-blind structured retrieval."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from rank_bm25 import BM25Okapi

from experiments._continual_memory_config import HuggingFaceTokenizer, TokenizerConfig
from experiments._continual_memory_episodes import (
    QUERY_TEMPLATE,
    TURN_SEPARATOR,
    _visible_query,
)
from experiments.interleaved_conversation import build_interleaved_schedule
from experiments.interleaved_memory_benchmark import (
    _aggregate_online_predictions,
    _online_prediction,
)
from experiments.preference_stream_injection import PreferenceStreamInjectionClient
from experiments.synthetic_temporal_baselines import _tokenize
from experiments.synthetic_temporal_preferences import resolve_query


BENCHMARK_VERSION = "interleaved_delayed_retrieval.v3"
CHECKPOINT_KINDS = {
    "preference-change-delayed",
    "preference-incongruity-delayed",
}
DEFAULT_WINDOWS = (65536, 131072)
DEFAULT_SWEEP_INTERVAL = 262144
DEFAULT_CAPSULE_CAPACITY = 1024


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


def _cumulative_tokens(turns: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return cumulative serialized stream tokens after every turn."""
    cumulative = []
    total = 0
    for turn in turns:
        total += int(turn["serialized_token_count"])
        cumulative.append(total)
    return cumulative


def _checkpoint_kind(query: Mapping[str, Any]) -> str | None:
    """Return the supported delayed checkpoint kind encoded by a query ID."""
    query_id = str(query.get("query_id", ""))
    for kind in CHECKPOINT_KINDS:
        if query_id.endswith(f"-{kind}"):
            return kind
    return None


def delayed_checkpoint_specs(
    schedule: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    *,
    sweep_interval_tokens: int,
) -> list[dict[str, Any]]:
    """Schedule eligible queries at the next periodic or terminal global sweep."""
    if sweep_interval_tokens < 1:
        raise ValueError("sweep_interval_tokens must be positive")
    turns = list(schedule["turns"])
    cumulative = _cumulative_tokens(turns)
    event_turn = {
        str(turn["stream_event"]["event_id"]): index
        for index, turn in enumerate(turns)
    }
    specs = []
    for query in sorted(queries, key=lambda item: str(item.get("query_id", ""))):
        kind = _checkpoint_kind(query)
        contract = query.get("checkpoint_contract")
        if kind is None or not isinstance(contract, Mapping):
            continue
        trigger_event_id = str(contract.get("trigger_event_id", ""))
        trigger_index = event_turn.get(trigger_event_id)
        if trigger_index is None:
            raise ValueError(
                f"query {query.get('query_id')} references unscheduled trigger {trigger_event_id}"
            )
        trigger_tokens = cumulative[trigger_index]
        next_sweep = (trigger_tokens // sweep_interval_tokens + 1) * sweep_interval_tokens
        checkpoint_index = bisect_left(cumulative, next_sweep, lo=trigger_index + 1)
        if checkpoint_index >= len(turns):
            checkpoint_index = len(turns) - 1
        if checkpoint_index <= trigger_index:
            continue
        event_groups = [
            [str(identifier) for identifier in group]
            for group in contract.get("required_event_id_groups", [])
        ]
        if not event_groups:
            raise ValueError(f"query {query.get('query_id')} has no event evidence groups")
        evidence_positions = []
        for group in event_groups:
            positions = [
                event_turn[event_id]
                for event_id in group
                if event_id in event_turn and event_turn[event_id] <= trigger_index
            ]
            if not positions:
                raise ValueError(
                    f"query {query.get('query_id')} has unavailable evidence group {group}"
                )
            evidence_positions.append(max(positions))
        specs.append(
            {
                "checkpoint_kind": kind,
                "query": dict(query),
                "trigger_turn_index": trigger_index + 1,
                "checkpoint_turn_index": checkpoint_index + 1,
                "sweep_kind": (
                    "periodic" if cumulative[checkpoint_index] >= next_sweep else "terminal"
                ),
                "scheduled_sweep_token": next_sweep,
                "turn_age": max(checkpoint_index - position for position in evidence_positions),
                "token_age": max(
                    cumulative[checkpoint_index] - cumulative[position]
                    for position in evidence_positions
                ),
                "stream_position_percentage": 100.0
                * (checkpoint_index + 1)
                / len(turns),
            }
        )
    return specs


def build_preference_capsules(
    schedule: Mapping[str, Any],
    injections_by_history: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build causal source capsules without exposing relation labels to ranking."""
    turns = list(schedule["turns"])
    turn_by_event_id = {
        str(turn["stream_event"]["event_id"]): (index, turn)
        for index, turn in enumerate(turns)
    }
    capsules = []
    relation_descriptions = {
        "preference_change": "An earlier standing preference was later replaced.",
        "preference_incongruity": "An indirect report and a direct correction disagreed.",
    }
    for history_id, result in sorted(injections_by_history.items()):
        injections = result.get("injections")
        if not isinstance(injections, list):
            raise ValueError(f"history {history_id} has malformed preference injections")
        for injection in injections:
            source_ids = injection.get("source_event_ids")
            if not isinstance(source_ids, list) or len(source_ids) != 3:
                raise ValueError(f"history {history_id} has malformed capsule sources")
            missing = [event_id for event_id in source_ids if event_id not in turn_by_event_id]
            if missing:
                raise ValueError(f"history {history_id} capsule sources are absent: {missing}")
            source_turns = [turn_by_event_id[event_id] for event_id in source_ids]
            if any(str(turn["history_id"]) != history_id for _, turn in source_turns):
                raise ValueError(f"history {history_id} capsule crosses account boundaries")
            capsules.append(
                {
                    "kind": str(injection["kind"]),
                    "history_id": history_id,
                    "release_turn_index": max(index for index, _ in source_turns) + 1,
                    "source_event_ids": list(source_ids),
                    "text": " ".join(
                        [
                            relation_descriptions[str(injection["kind"])],
                            *[
                                str(
                                    turn["stream_event"].get("model_text")
                                    or turn["stream_event"]["fact"]["support_text"]
                                )
                                for _, turn in source_turns
                            ],
                        ]
                    ),
                }
            )
    for index, turn in enumerate(turns):
        event = turn["stream_event"]
        if event.get("event_family") != "same_entity_hard_negative":
            continue
        text = str(event.get("model_text") or event["fact"]["support_text"])
        capsules.append(
            {
                "kind": "hard_negative",
                "history_id": str(turn["history_id"]),
                "release_turn_index": index + 1,
                "source_event_ids": [str(event["event_id"])],
                "text": text,
            }
        )
    capsules = sorted(
        capsules,
        key=lambda capsule: (
            int(capsule["release_turn_index"]),
            str(capsule["text"]),
        ),
    )
    duplicate_texts = {
        str(capsule["text"])
        for capsule in capsules
        if sum(str(candidate["text"]) == str(capsule["text"]) for candidate in capsules) > 1
    }
    if duplicate_texts:
        raise ValueError("preference capsule text must be globally unique")
    return capsules


def rank_global_capsule(
    capsules: Sequence[Mapping[str, Any]], query_text: str
) -> dict[str, Any]:
    """Select one capsule globally using only model-visible query text."""
    if not capsules:
        raise ValueError("global capsule ranking requires at least one capsule")
    ordered = sorted(capsules, key=lambda capsule: str(capsule["text"]))
    index = BM25Okapi([_tokenize(str(capsule["text"])) for capsule in ordered])
    scores = index.get_scores(_tokenize(query_text))
    selected_index = min(
        range(len(ordered)),
        key=lambda candidate: (-float(scores[candidate]), candidate),
    )
    return dict(ordered[selected_index])


def compact_valid_capsules_to_capacity(
    capsules: Sequence[Mapping[str, Any]], capacity: int
) -> list[dict[str, Any]]:
    """Retain a stable content-hash sample of valid capsules under a fixed capacity."""
    if capacity < 1:
        raise ValueError("capsule capacity must be positive")
    valid = [
        capsule
        for capsule in capsules
        if str(capsule.get("kind")) in {
            "preference_change",
            "preference_incongruity",
        }
    ]
    return sorted(
        (
            dict(capsule)
            for capsule in sorted(
                valid,
                key=lambda capsule: hashlib.sha256(
                    str(capsule["text"]).encode("utf-8")
                ).digest(),
            )[:capacity]
        ),
        key=lambda capsule: (int(capsule["release_turn_index"]), str(capsule["text"])),
    )


def _expected_capsule_kind(checkpoint_kind: str) -> str:
    """Map one model-visible delayed query family to its capsule relation."""
    mapping = {
        "preference-change-delayed": "preference_change",
        "preference-incongruity-delayed": "preference_incongruity",
    }
    return mapping[checkpoint_kind]


def _prompt_increment(last_text: str, query_text: str, tokenizer: Any) -> int:
    """Return exact token growth from appending the shared query prompt."""
    prompt = QUERY_TEMPLATE.format(query_text=query_text)
    return len(tokenizer.encode(f"{last_text}{prompt}")) - len(
        tokenizer.encode(last_text)
    )


def _fit_sliding(
    prefix: Sequence[Mapping[str, Any]],
    cumulative: Sequence[int],
    query_text: str,
    max_tokens: int,
    tokenizer: Any,
) -> tuple[int, int]:
    """Fit the maximal complete-turn raw suffix under an exact shared prompt cap."""
    prompt = QUERY_TEMPLATE.format(query_text=query_text)
    prompt_tokens = len(tokenizer.encode(prompt))
    if prompt_tokens > max_tokens:
        raise ValueError("query prompt exceeds the context window")

    def exact_count(start: int) -> int:
        if start == len(prefix):
            return prompt_tokens
        internal = cumulative[-1] - cumulative[start]
        return (
            int(prefix[start]["token_count"])
            + internal
            + _prompt_increment(str(prefix[-1]["text"]), query_text, tokenizer)
        )

    low = 0
    high = len(prefix)
    while low < high:
        midpoint = (low + high) // 2
        if exact_count(midpoint) <= max_tokens:
            high = midpoint
        else:
            low = midpoint + 1
    return low, exact_count(low)


def _capsule_turns(
    capsule: Mapping[str, Any],
    turn_by_event_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Render compact source turns cited by one capsule."""
    turns = []
    for event_id in capsule["source_event_ids"]:
        source = turn_by_event_id[str(event_id)]
        event = source["stream_event"]
        text = str(event.get("model_text") or event["fact"]["support_text"])
        turns.append(
            {
                "turn_id": f"capsule:{event_id}",
                "history_id": source["history_id"],
                "text": text,
                "token_count": 0,
                "stream_event": event,
            }
        )
    return turns


def _fit_structured(
    prefix: Sequence[Mapping[str, Any]],
    cumulative: Sequence[int],
    capsule: Mapping[str, Any],
    turn_by_event_id: Mapping[str, Mapping[str, Any]],
    query_text: str,
    max_tokens: int,
    tokenizer: Any,
) -> tuple[list[Mapping[str, Any]], int, int]:
    """Fit one deduplicated source capsule and recent suffix exactly."""
    source_turns = _capsule_turns(capsule, turn_by_event_id)
    source_positions = {
        str(turn["stream_event"]["event_id"]): index
        for index, turn in enumerate(prefix)
        if str(turn["stream_event"]["event_id"])
        in {str(event_id) for event_id in capsule["source_event_ids"]}
    }
    if len(source_positions) != len(source_turns):
        raise ValueError("capsule sources must all occur in the causal prefix")
    prompt = QUERY_TEMPLATE.format(query_text=query_text)
    prompt_tokens = len(tokenizer.encode(prompt))

    def injections_for(start: int) -> list[Mapping[str, Any]]:
        return [
            turn
            for turn in source_turns
            if source_positions[str(turn["stream_event"]["event_id"])] < start
        ]

    def exact_count(start: int) -> int:
        injections = injections_for(start)
        has_raw = start < len(prefix)
        if not injections and not has_raw:
            return prompt_tokens
        if injections:
            injection_text = TURN_SEPARATOR.join(str(turn["text"]) for turn in injections)
            total = len(tokenizer.encode(injection_text))
            previous_text = str(injections[-1]["text"])
        else:
            total = int(prefix[start]["token_count"])
            previous_text = str(prefix[start]["text"])
        if has_raw:
            if injections:
                first_raw = str(prefix[start]["text"])
                total += len(
                    tokenizer.encode(f"{previous_text}{TURN_SEPARATOR}{first_raw}")
                ) - len(tokenizer.encode(previous_text))
            total += cumulative[-1] - cumulative[start]
            previous_text = str(prefix[-1]["text"])
        return total + _prompt_increment(previous_text, query_text, tokenizer)

    boundaries = sorted({0, len(prefix), *(position + 1 for position in source_positions.values())})
    candidates = []
    for lower, upper in zip(boundaries, boundaries[1:]):
        low = lower
        high = upper
        while low < high:
            midpoint = (low + high) // 2
            if exact_count(midpoint) <= max_tokens:
                high = midpoint
            else:
                low = midpoint + 1
        if low <= upper and exact_count(low) <= max_tokens:
            candidates.append(low)
    if exact_count(len(prefix)) <= max_tokens:
        candidates.append(len(prefix))
    if not candidates:
        raise ValueError("structured capsule and query exceed the context window")
    start = min(candidates)
    return injections_for(start), start, exact_count(start)


def _target_history_turns(
    selected: Sequence[Mapping[str, Any]],
    history_id: str,
    event_order: Mapping[str, int],
) -> list[Mapping[str, Any]]:
    """Return deduplicated selected events for deterministic oracle resolution."""
    by_event_id = {}
    for turn in selected:
        if str(turn["history_id"]) != history_id:
            continue
        event_id = str(turn["stream_event"]["event_id"])
        by_event_id[event_id] = turn
    return sorted(
        by_event_id.values(),
        key=lambda turn: event_order[str(turn["stream_event"]["event_id"])],
    )


def _capsule_ranker(
    capsules: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], BM25Okapi]:
    """Build one content-ordered BM25 index for a causal capsule pool."""
    ordered = sorted(capsules, key=lambda capsule: str(capsule["text"]))
    return ordered, BM25Okapi([_tokenize(str(capsule["text"])) for capsule in ordered])


def _rank_with_index(
    ordered: Sequence[Mapping[str, Any]], index: BM25Okapi, query_text: str
) -> dict[str, Any]:
    """Select one content-tie-broken capsule from a prebuilt index."""
    scores = index.get_scores(_tokenize(query_text))
    selected_index = min(
        range(len(ordered)), key=lambda candidate: (-float(scores[candidate]), candidate)
    )
    return dict(ordered[selected_index])


def evaluate_delayed_preference_checkpoints(
    schedule: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    capsules: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    windows: Sequence[int],
    sweep_interval_tokens: int,
    capsule_capacity: int = DEFAULT_CAPSULE_CAPACITY,
) -> list[dict[str, Any]]:
    """Evaluate matched sliding and top-1 structured contexts at delayed sweeps."""
    turns = list(schedule["turns"])
    cumulative = _cumulative_tokens(turns)
    turn_by_event_id = {
        str(turn["stream_event"]["event_id"]): turn for turn in turns
    }
    event_order = {
        str(turn["stream_event"]["event_id"]): index for index, turn in enumerate(turns)
    }
    turns_by_history: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, turn in enumerate(turns):
        turns_by_history.setdefault(str(turn["history_id"]), []).append((index, turn))
    specs = delayed_checkpoint_specs(
        schedule, queries, sweep_interval_tokens=sweep_interval_tokens
    )
    rankers = {}
    rows = []
    for spec in specs:
        query = spec["query"]
        history_id = str(query["history_id"])
        checkpoint_index = int(spec["checkpoint_turn_index"]) - 1
        prefix = turns[: checkpoint_index + 1]
        prefix_cumulative = cumulative[: checkpoint_index + 1]
        indexed_history = turns_by_history[history_id]
        history_end = bisect_left(
            [position for position, _ in indexed_history], checkpoint_index + 1
        )
        history_prefix = [turn for _, turn in indexed_history[:history_end]]
        resolved = resolve_query(
            [dict(turn["stream_event"]) for turn in history_prefix],
            dict(_visible_query(query)),
        )
        if resolved != query.get("gold"):
            raise ValueError(
                f"delayed checkpoint changes gold for {query.get('query_id')}: {resolved!r}"
            )
        query_text = str(query["query_text"])
        ranker = rankers.get(checkpoint_index)
        if ranker is None:
            causal_candidates = [
                capsule
                for capsule in capsules
                if int(capsule["release_turn_index"]) <= checkpoint_index + 1
            ]
            causal_capsules = compact_valid_capsules_to_capacity(
                causal_candidates, capsule_capacity
            )
            ranker = _capsule_ranker(causal_capsules)
            rankers[checkpoint_index] = ranker
        selected_capsule = _rank_with_index(*ranker, query_text)
        for window in windows:
            max_tokens = int(window)
            raw_start, raw_count = _fit_sliding(
                prefix, prefix_cumulative, query_text, max_tokens, tokenizer
            )
            raw_selected = prefix[raw_start:]
            raw_row = _online_prediction(
                query,
                history_prefix,
                _target_history_turns(raw_selected, history_id, event_order),
                method=f"sliding_context:{max_tokens}",
                checkpoint_turn_index=checkpoint_index + 1,
                selected_turn_count=len(raw_selected),
                selected_start_turn_id=(
                    str(raw_selected[0]["turn_id"]) if raw_selected else ""
                ),
                selected_end_turn_id=str(prefix[-1]["turn_id"]),
                turn_age=int(spec["turn_age"]),
                token_age=int(spec["token_age"]),
                stream_position_percentage=float(spec["stream_position_percentage"]),
            )
            raw_row.update(
                {
                    "checkpoint_kind": spec["checkpoint_kind"],
                    "trigger_turn_index": spec["trigger_turn_index"],
                    "max_tokens": max_tokens,
                    "model_input_token_count": raw_count,
                    "selected_capsule_count": 0,
                    "retrieval_query_text": query_text,
                    "sweep_kind": spec["sweep_kind"],
                    "scheduled_sweep_token": spec["scheduled_sweep_token"],
                }
            )
            rows.append(raw_row)

            injected_turns, structured_start, structured_count = _fit_structured(
                prefix,
                prefix_cumulative,
                selected_capsule,
                turn_by_event_id,
                query_text,
                max_tokens,
                tokenizer,
            )
            history_start = bisect_left(
                [position for position, _ in indexed_history], structured_start
            )
            structured_selected = [
                *injected_turns,
                *(turn for _, turn in indexed_history[history_start:history_end]),
            ]
            structured_row = _online_prediction(
                query,
                history_prefix,
                _target_history_turns(structured_selected, history_id, event_order),
                method=f"structured_capacity_top1:{max_tokens}",
                checkpoint_turn_index=checkpoint_index + 1,
                selected_turn_count=(
                    len(injected_turns) + len(prefix) - structured_start
                ),
                selected_start_turn_id=(
                    str(structured_selected[0]["turn_id"])
                    if structured_selected
                    else ""
                ),
                selected_end_turn_id=(
                    str(structured_selected[-1]["turn_id"])
                    if structured_selected
                    else ""
                ),
                turn_age=int(spec["turn_age"]),
                token_age=int(spec["token_age"]),
                stream_position_percentage=float(spec["stream_position_percentage"]),
            )
            structured_row.update(
                {
                    "checkpoint_kind": spec["checkpoint_kind"],
                    "trigger_turn_index": spec["trigger_turn_index"],
                    "max_tokens": max_tokens,
                    "model_input_token_count": structured_count,
                    "selected_capsule_count": 1,
                    "selected_capsule_history_id": selected_capsule["history_id"],
                    "selected_capsule_kind": selected_capsule["kind"],
                    "selected_capsule_family_correct": selected_capsule["kind"]
                    == _expected_capsule_kind(str(spec["checkpoint_kind"])),
                    "selected_capsule_release_turn_index": selected_capsule[
                        "release_turn_index"
                    ],
                    "selected_capsule_source_event_ids": selected_capsule[
                        "source_event_ids"
                    ],
                    "retrieval_query_text": query_text,
                    "sweep_kind": spec["sweep_kind"],
                    "scheduled_sweep_token": spec["scheduled_sweep_token"],
                }
            )
            rows.append(structured_row)
    return rows


def _age_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize unique delayed checkpoint ages and tested-window coverage."""
    unique = {
        str(row["query_id"]): int(row["token_age"])
        for row in rows
    }
    ages = sorted(unique.values())
    return {
        "checkpoint_count": len(ages),
        "minimum_token_age": min(ages),
        "median_token_age": ages[len(ages) // 2],
        "maximum_token_age": max(ages),
        "fraction_above_64k": sum(age > 65536 for age in ages) / len(ages),
        "fraction_above_128k": sum(age > 131072 for age in ages) / len(ages),
    }


def _aggregate_by_checkpoint_kind(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Aggregate every method separately for each delayed relation family."""
    kinds = sorted({str(row["checkpoint_kind"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    return {
        kind: {
            method: _aggregate_online_predictions(
                [
                    row
                    for row in rows
                    if row["checkpoint_kind"] == kind and row["method"] == method
                ]
            )[method]
            for method in methods
        }
        for kind in kinds
    }


def _paired_grounded_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 47,
) -> dict[str, dict[str, Any]]:
    """Return paired grounded deltas with deterministic history-clustered intervals."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    by_window: dict[int, dict[tuple[str, str], dict[str, bool]]] = {}
    for row in rows:
        window = int(row["max_tokens"])
        key = (str(row["history_id"]), str(row["query_id"]))
        family = "structured" if str(row["method"]).startswith("structured_") else "sliding"
        slot = by_window.setdefault(window, {}).setdefault(key, {})
        if family in slot:
            raise ValueError(f"duplicate {family} row for window={window}, checkpoint={key}")
        slot[family] = bool(row["grounded_answer_correct"])
    result = {}
    for window, checkpoints in sorted(by_window.items()):
        incomplete = [key for key, values in checkpoints.items() if set(values) != {"sliding", "structured"}]
        if incomplete:
            raise ValueError(f"unmatched paired checkpoints at window {window}: {incomplete[:5]}")
        by_history: dict[str, list[float]] = {}
        for (history_id, _), values in checkpoints.items():
            by_history.setdefault(history_id, []).append(
                float(values["structured"]) - float(values["sliding"])
            )
        histories = sorted(by_history)
        history_means = {
            history_id: sum(values) / len(values)
            for history_id, values in by_history.items()
        }
        delta = sum(history_means.values()) / len(history_means)
        rng = random.Random(f"{seed}:{window}")
        bootstrap = sorted(
            sum(history_means[rng.choice(histories)] for _ in histories) / len(histories)
            for _ in range(bootstrap_samples)
        )
        lower_index = max(0, math.floor(0.025 * bootstrap_samples))
        upper_index = min(bootstrap_samples - 1, math.ceil(0.975 * bootstrap_samples) - 1)
        result[str(window)] = {
            "matched_checkpoint_count": len(checkpoints),
            "history_cluster_count": len(histories),
            "paired_delta": delta,
            "confidence_interval_95": [bootstrap[lower_index], bootstrap[upper_index]],
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        }
    return result


def _selector_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one selector decision per delayed checkpoint."""
    structured = {}
    for row in rows:
        if not str(row["method"]).startswith("structured_"):
            continue
        structured.setdefault(str(row["query_id"]), row)
    selected = list(structured.values())
    errors = [row for row in selected if not row["grounded_answer_correct"]]
    return {
        "checkpoint_count": len(selected),
        "target_history_selection_rate": sum(
            row["selected_capsule_history_id"] == row["history_id"] for row in selected
        )
        / len(selected),
        "expected_family_selection_rate": sum(
            bool(row["selected_capsule_family_correct"]) for row in selected
        )
        / len(selected),
        "target_history_and_family_selection_rate": sum(
            row["selected_capsule_history_id"] == row["history_id"]
            and bool(row["selected_capsule_family_correct"])
            for row in selected
        )
        / len(selected),
        "structured_error_count_64k": sum(
            row["method"] == "structured_capacity_top1:65536"
            and not row["grounded_answer_correct"]
            for row in rows
        ),
        "stale_given_structured_error_64k": (
            sum(
                row["method"] == "structured_capacity_top1:65536"
                and row["stale_memory_intrusion"]
                for row in rows
            )
            / sum(
                row["method"] == "structured_capacity_top1:65536"
                and not row["grounded_answer_correct"]
                for row in rows
            )
        ),
        "structured_error_count_128k": sum(
            row["method"] == "structured_capacity_top1:131072"
            and not row["grounded_answer_correct"]
            for row in rows
        ),
        "stale_given_structured_error_128k": (
            sum(
                row["method"] == "structured_capacity_top1:131072"
                and row["stale_memory_intrusion"]
                for row in rows
            )
            / sum(
                row["method"] == "structured_capacity_top1:131072"
                and not row["grounded_answer_correct"]
                for row in rows
            )
        ),
    }
def _render_report(metrics: Mapping[str, Any]) -> str:
    """Render an auditable report that keeps retrieval and reasoning claims separate."""
    dataset = metrics["dataset"]
    ages = metrics["age_distribution"]
    paired = metrics["paired_grounded_comparison"]
    selector = metrics["selector_diagnostics"]
    lines = [
        "# Delayed Interleaved Memory Benchmark",
        "",
        f"- Benchmark version: `{metrics['benchmark_version']}`",
        f"- Eligible queries: {dataset['query_count']:,}",
        f"- Evaluated queries: {dataset['evaluated_checkpoint_count']:,}",
        f"- Excluded queries: {len(dataset['excluded_query_ids']):,} (`{dataset['excluded_query_ids'][0]}`)",
        f"- Median oldest-required-evidence age: {ages['median_token_age']:,} tokens",
        f"- Checkpoints with an oldest required item beyond 64K: {ages['fraction_above_64k']:.2%}",
        f"- Checkpoints with an oldest required item beyond 128K: {ages['fraction_above_128k']:.2%}",
        "",
        "## Matched Methods",
        "",
        "Both methods use the same causal checkpoint, original dataset query text without augmentation, exact tokenizer, and prompt-inclusive token cap.",
        "The structured method selects one capsule globally with BM25 and receives no history filter, gold answer, or evidence contract during ranking.",
        "Validity admission uses Scallop-derived relation kind before query-blind, history-blind, stable-hash capacity compaction within admitted capsules.",
        "Answers are scored with a deterministic oracle resolver, so results measure context availability rather than LLM reasoning.",
        "",
    ]
    for method, values in metrics["methods"].items():
        lines.append(
            f"- {method}: grounded {values['grounded_answer_accuracy']:.2%}; "
            f"oracle answer {values['oracle_resolver_answer_accuracy']:.2%}; "
            f"complete provenance {values['complete_provenance_rate']:.2%}; "
            f"stale intrusion {values['stale_memory_intrusion_rate']:.2%}"
        )
    lines.extend(["", "## Paired Grounded Comparison", ""])
    for window, values in sorted(paired.items(), key=lambda item: int(item[0])):
        lower, upper = values["confidence_interval_95"]
        lines.append(
            f"- {int(window):,} tokens: structured minus sliding is "
            f"{values['paired_delta']:.2%}, with deterministic history-clustered "
            f"95% interval [{lower:.2%}, {upper:.2%}]."
        )
    lines.extend(
        [
            "",
            "Each interval uses 2,000 bootstrap samples over 1,024 history clusters with seed 47.",
            "",
            "## Family-Stratified Grounded Availability",
            "",
        ]
    )
    for kind, methods in metrics["by_checkpoint_kind"].items():
        lines.append(f"### {kind}")
        lines.append("")
        for method, values in methods.items():
            lines.append(
                f"- {method}: grounded {values['grounded_answer_accuracy']:.2%}; "
                f"stale intrusion {values['stale_memory_intrusion_rate']:.2%}."
            )
        lines.append("")
    lines.extend(
        [
            "## Selector and Error Diagnostics",
            "",
            f"- Target-history selection rate: {selector['target_history_selection_rate']:.2%}.",
            f"- Expected-family selection rate: {selector['expected_family_selection_rate']:.2%}.",
            f"- Target-history and expected-family selection rate: {selector['target_history_and_family_selection_rate']:.2%}.",
            f"- Stale among structured 64K errors: {selector['stale_given_structured_error_64k']:.2%}.",
            f"- Stale among structured 128K errors: {selector['stale_given_structured_error_128k']:.2%}.",
            "",
            "The supported claim is bounded deterministic context availability under this synthetic protocol.",
            "This is not LLM answer accuracy, extraction quality, independent reasoning, deployment safety, or a general memory-system superiority claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    *,
    endpoint: str,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    sweep_interval_tokens: int = DEFAULT_SWEEP_INTERVAL,
    capsule_capacity: int = DEFAULT_CAPSULE_CAPACITY,
) -> dict[str, Any]:
    """Run the full delayed checkpoint benchmark and write reproducible artifacts."""
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
    events = [
        event
        for event in _read_jsonl(dataset_dir / "events.jsonl")
        if event.get("split") == "test"
        and event.get("hardness_profile") == "anti_shortcut_interleaved_v3"
    ]
    queries = [
        query
        for query in _read_jsonl(dataset_dir / "queries.jsonl")
        if query.get("split") == "test"
        and query.get("hardness_profile") == "anti_shortcut_interleaved_v3"
    ]
    history_ids = sorted({str(event["history_id"]) for event in events})
    schedule = build_interleaved_schedule(
        events,
        tokenizer=tokenizer,
        seed=47,
        concurrent_accounts=min(8, len(history_ids)),
        min_segment_events=4,
        max_segment_events=8,
    )
    client = PreferenceStreamInjectionClient(endpoint, timeout_seconds=10.0)
    injections = {
        history_id: client.derive(
            [event for event in events if event["history_id"] == history_id]
        )
        for history_id in history_ids
    }
    candidates = build_preference_capsules(schedule, injections)
    capsules = compact_valid_capsules_to_capacity(candidates, capsule_capacity)
    predictions = evaluate_delayed_preference_checkpoints(
        schedule,
        queries,
        candidates,
        tokenizer=tokenizer,
        windows=windows,
        sweep_interval_tokens=sweep_interval_tokens,
        capsule_capacity=capsule_capacity,
    )
    metrics = {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": {
            "history_count": len(history_ids),
            "event_count": len(events),
            "query_count": len(
                [query for query in queries if _checkpoint_kind(query) is not None]
            ),
            "evaluated_checkpoint_count": len(
                delayed_checkpoint_specs(
                    schedule,
                    queries,
                    sweep_interval_tokens=sweep_interval_tokens,
                )
            ),
            "excluded_query_ids": sorted(
                {
                    str(query["query_id"])
                    for query in queries
                    if _checkpoint_kind(query) is not None
                }
                - {
                    str(row["query_id"])
                    for row in predictions
                }
            ),
        },
        "schedule": {
            "seed": 47,
            "concurrent_accounts": min(8, len(history_ids)),
            "turn_count": len(schedule["turns"]),
            "token_count": sum(
                int(turn["serialized_token_count"]) for turn in schedule["turns"]
            ),
        },
        "sweep_interval_tokens": sweep_interval_tokens,
        "windows": list(windows),
        "capsules": {
            "candidate_count": len(candidates),
            "retained_count": len(capsules),
            "capacity": capsule_capacity,
            "compaction_policy": "lowest stable content hashes among causal Scallop-derived valid capsules",
            "query_blind_compaction": True,
            "history_blind_compaction": True,
            "validity_admission_uses_scallop_relation_kind": True,
            "compaction_within_admitted_capsules_is_relation_blind": True,
            "selector": "global BM25 top-1 over capsule text using only visible query text",
            "structured_relation_label_field_visible_to_selector": False,
            "natural_relation_description_visible_in_capsule_text": True,
            "history_id_visible_to_selector": False,
            "gold_or_evidence_contract_visible_to_selector": False,
        },
        "age_distribution": _age_summary(predictions),
        "methods": _aggregate_online_predictions(predictions),
        "by_checkpoint_kind": _aggregate_by_checkpoint_kind(predictions),
        "paired_grounded_comparison": _paired_grounded_comparison(predictions),
        "selector_diagnostics": _selector_diagnostics(predictions),
        "tokenizer": dict(tokenizer.metadata()),
        "interpretation": {
            "claim": "grounded context availability under delayed peer-task interference",
            "non_claim": "not LLM generation, extraction, or independent reasoning accuracy",
        },
    }
    candidates_path = output_dir / "memory_candidates.jsonl"
    capsules_path = output_dir / "retained_capsules.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    _write_jsonl(candidates_path, candidates)
    _write_jsonl(capsules_path, capsules)
    _write_jsonl(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    report_path.write_text(_render_report(metrics), encoding="utf-8")
    artifacts = (
        candidates_path,
        capsules_path,
        predictions_path,
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
        "artifacts": [path.name for path in artifacts],
        "artifact_sha256": {path.name: _sha256(path) for path in artifacts},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Run the delayed interleaved memory CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scallop-endpoint",
        default=os.environ.get("SCALLOP_VALIDATOR_URL"),
    )
    parser.add_argument("--windows", default=",".join(str(value) for value in DEFAULT_WINDOWS))
    parser.add_argument("--sweep-interval-tokens", type=int, default=DEFAULT_SWEEP_INTERVAL)
    parser.add_argument(
        "--capsule-capacity", type=int, default=DEFAULT_CAPSULE_CAPACITY
    )
    args = parser.parse_args(argv)
    if not args.scallop_endpoint:
        raise ValueError("SCALLOP_VALIDATOR_URL or --scallop-endpoint is required")
    windows = tuple(int(value) for value in args.windows.split(","))
    metrics = run_benchmark(
        args.dataset,
        args.output_dir,
        endpoint=args.scallop_endpoint,
        windows=windows,
        sweep_interval_tokens=args.sweep_interval_tokens,
        capsule_capacity=args.capsule_capacity,
    )
    print(json.dumps(metrics["methods"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
