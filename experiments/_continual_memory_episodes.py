"""Deterministic online episode construction for continual-memory benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments._continual_memory_config import (
    BenchmarkTokenizer,
    ContinualMemoryConfig,
    _mapping,
    _positive_int,
    _string,
)
from experiments.synthetic_temporal_preferences import resolve_query


TURN_SEPARATOR = "\n"
QUERY_TEMPLATE = "\nQuestion: {query_text}\nAnswer:"
VISIBLE_QUERY_FIELDS = (
    "query_id",
    "history_id",
    "kind",
    "subject",
    "date",
    "scope",
    "candidate",
    "fact_id",
    "query_text",
)
BASE_CHECKPOINT_SPECS: dict[str, dict[str, tuple[tuple[str, ...], ...] | str]] = {
    "current": {
        "trigger": "transition",
        "event_groups": (("transition",),),
        "fact_groups": (("current",),),
    },
    "scope": {
        "trigger": "scope",
        "event_groups": (("scope",),),
        "fact_groups": (("scoped",),),
    },
    "constraint": {
        "trigger": "constraint",
        "event_groups": (("constraint",),),
        "fact_groups": (("constraint",),),
    },
    "private": {
        "trigger": "retract",
        "event_groups": (("retract",),),
        "fact_groups": (("private",),),
    },
    "ambiguity": {
        "trigger": "ambiguity",
        "event_groups": (("ambiguity",),),
        "fact_groups": (("ambiguity",),),
    },
    "backdated": {
        "trigger": "backdated",
        "event_groups": (("backdated",),),
        "fact_groups": (("backdated",),),
    },
    "duplicate": {
        "trigger": "duplicate",
        "event_groups": (("transition", "duplicate"),),
        "fact_groups": (("current",),),
    },
    "authority": {
        "trigger": "direct-correction",
        "event_groups": (("direct-correction",),),
        "fact_groups": (("direct-correction",),),
    },
    "scope-leakage": {
        "trigger": "scope-leakage",
        "event_groups": (("transition", "duplicate"), ("scope-leakage",)),
        "fact_groups": (("current",), ("scope-leakage",)),
    },
}
LINEAGE_CHECKPOINT_SPECS: dict[str, dict[str, tuple[tuple[str, ...], ...] | str]] = {
    "private-lineage-positive": {
        "trigger": "lineage-copy-2",
        "event_groups": (),
        "fact_groups": (),
    },
    "private-lineage": {
        "trigger": "lineage-retract",
        "event_groups": (),
        "fact_groups": (),
    },
    "preference-change-delayed": {
        "trigger": "preference-change-probe",
        "event_groups": (("add",), ("transition",)),
        "fact_groups": (("initial",), ("current",)),
    },
    "preference-incongruity-delayed": {
        "trigger": "preference-incongruity-probe",
        "event_groups": (("indirect-source",), ("direct-correction",)),
        "fact_groups": (("indirect-source",), ("direct-correction",)),
    },
}
CHECKPOINT_SPECS = {**BASE_CHECKPOINT_SPECS, **LINEAGE_CHECKPOINT_SPECS}


def _checkpoint_specs(
    source_profile: str,
) -> dict[str, dict[str, tuple[tuple[str, ...], ...] | str]]:
    """Return checkpoint contracts enabled by one source-data profile."""
    if source_profile == "base":
        return BASE_CHECKPOINT_SPECS
    if source_profile == "anti_shortcut_stream_v2":
        return CHECKPOINT_SPECS
    raise ValueError(f"unsupported continual-memory source profile: {source_profile}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load object-valued JSONL rows with line-specific failures."""
    rows = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read dataset artifact {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _suffix(identifier: str, history_id: str) -> str:
    """Return a history-local identifier suffix or reject a mismatched identifier."""
    prefix = f"{history_id}-"
    if not identifier.startswith(prefix):
        raise ValueError(f"identifier {identifier!r} does not belong to history {history_id!r}")
    return identifier[len(prefix) :]


def _target_text(event: Mapping[str, Any]) -> str:
    """Render one target turn, preferring natural text with no benchmark labels."""
    model_text = event.get("model_text")
    if isinstance(model_text, str) and model_text.strip():
        return model_text
    fact = _mapping(event.get("fact"), f"event {event.get('event_id')} fact")
    fields = [
        "role=target",
        "task_id=temporal_preference_memory",
        f"thread_id={event['history_id']}",
        f"event_id={event['event_id']}",
        f"operation={event['operation']}",
        f"support={fact.get('support_text', '')}",
    ]
    if event.get("retracts") is not None:
        fields.append(f"retracts={event['retracts']}")
    if event.get("supersedes") is not None:
        fields.append(f"supersedes={event['supersedes']}")
    if event.get("duplicate_of") is not None:
        fields.append(f"duplicate_of={event['duplicate_of']}")
    return " | ".join(fields)


def _distractor_text(event: Mapping[str, Any]) -> str:
    """Render one distractor without exposing its hidden relevance annotation."""
    model_text = event.get("model_text")
    if isinstance(model_text, str) and model_text.strip():
        if event.get("hardness_profile") == "anti_shortcut_stream_v2":
            fact = _mapping(event.get("fact"), f"distractor event {event.get('event_id')} fact")
            value = str(fact.get("object", "the recorded option"))
            return " ".join(
                [
                    model_text,
                    f"A separate archive entry compared {value} with several nearby options before a later review.",
                    "Calendar notes recorded shifting plans across morning, afternoon, and evening sessions.",
                    "A shopping discussion revisited delivery timing, substitutions, and standing account choices.",
                    "Travel planning added temporary exceptions for lodging, meals, and local transportation.",
                    "Another conversation reconsidered earlier selections after availability and scheduling changed.",
                    "The account history also included revisions to notification, privacy, and sharing settings.",
                    "A later message repeated portions of the record while comparing alternative arrangements.",
                    "The final archive page retained the surrounding discussion for a future consistency review.",
                ]
            )
        return model_text
    fact = _mapping(event.get("fact"), f"distractor event {event.get('event_id')} fact")
    return " | ".join(
        [
            "role=distractor",
            "task_id=distractor_temporal_preference",
            f"thread_id={event['history_id']}",
            "distractor_type=cross_task_event",
            f"operation={event['operation']}",
            f"support={fact.get('support_text', '')}",
        ]
    )


def _stable_index(seed: int, parts: Sequence[Any], size: int) -> int:
    """Return a deterministic pool index without process-randomized hashing."""
    if size < 1:
        raise ValueError("cannot select from an empty distractor pool")
    material = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def _identifier_groups(history_id: str, suffix_groups: Sequence[Sequence[str]]) -> list[list[str]]:
    """Expand history-local suffix groups to complete identifiers."""
    return [[f"{history_id}-{suffix}" for suffix in group] for group in suffix_groups]


def _visible_query(query: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields available to retrieval and deterministic resolution."""
    return {field: query[field] for field in VISIBLE_QUERY_FIELDS if field in query}


def _visible_query_text(query: Mapping[str, Any]) -> str:
    """Return only the natural-language query, excluding IDs and structured labels."""
    return _string(query.get("query_text"), "query.query_text")


def _serialized_model_input(
    turns: Sequence[Mapping[str, Any]], query_text: str
) -> str:
    """Serialize one exact model-visible suffix and its gold-free query prompt."""
    context = TURN_SEPARATOR.join(str(turn["text"]) for turn in turns)
    prompt = QUERY_TEMPLATE.format(query_text=query_text)
    return f"{context}{prompt}" if context else prompt


def _stream_token_count(
    turns: Sequence[Mapping[str, Any]], tokenizer: BenchmarkTokenizer
) -> int:
    """Count an exact serialized stream prefix including turn separators."""
    return len(tokenizer.encode(TURN_SEPARATOR.join(str(turn["text"]) for turn in turns)))


def _serialized_turn_increment(
    previous_text: str | None,
    text: str,
    tokenizer: BenchmarkTokenizer,
) -> int:
    """Return the exact token increase from appending one serialized turn."""
    if previous_text is None:
        return len(tokenizer.encode(text))
    previous_count = len(tokenizer.encode(previous_text))
    combined_count = len(tokenizer.encode(f"{previous_text}{TURN_SEPARATOR}{text}"))
    increment = combined_count - previous_count
    if increment < 1:
        raise ValueError("serialized turn must increase the stream token count")
    return increment


def _fit_sliding_window(
    turns: Sequence[Mapping[str, Any]],
    query_text: str,
    max_tokens: int,
    tokenizer: BenchmarkTokenizer,
) -> dict[str, int]:
    """Fit the maximal complete-turn suffix using exact serialized tokenization."""
    prompt_token_count = len(tokenizer.encode(QUERY_TEMPLATE.format(query_text=query_text)))
    if prompt_token_count > max_tokens:
        raise ValueError(
            f"checkpoint query prompt requires {prompt_token_count} tokens but window allows "
            f"{max_tokens}"
        )
    if not turns:
        return {
            "max_tokens": max_tokens,
            "start_turn_index": 0,
            "selected_turn_count": 0,
            "query_prompt_token_count": prompt_token_count,
            "model_input_token_count": prompt_token_count,
        }
    prompt = QUERY_TEMPLATE.format(query_text=query_text)
    last_text = str(turns[-1]["text"])
    prompt_increment = len(tokenizer.encode(f"{last_text}{prompt}")) - len(
        tokenizer.encode(last_text)
    )
    suffix_increments = [0] * (len(turns) + 1)
    for index in range(len(turns) - 1, -1, -1):
        suffix_increments[index] = suffix_increments[index + 1] + int(
            turns[index].get("serialized_token_count", turns[index]["token_count"])
        )

    def exact_count(candidate_start: int) -> int:
        if candidate_start == len(turns):
            return prompt_token_count
        internal_increments = suffix_increments[candidate_start + 1]
        return (
            int(turns[candidate_start]["token_count"])
            + internal_increments
            + prompt_increment
        )

    low = 0
    high = len(turns)
    while low < high:
        midpoint = (low + high) // 2
        if exact_count(midpoint) <= max_tokens:
            high = midpoint
        else:
            low = midpoint + 1
    start = low
    model_input_token_count = exact_count(start)
    return {
        "max_tokens": max_tokens,
        "start_turn_index": start,
        "selected_turn_count": len(turns) - start,
        "query_prompt_token_count": prompt_token_count,
        "model_input_token_count": model_input_token_count,
    }


def _checkpoint_ages(
    event_groups: Sequence[Sequence[str]],
    event_turns: Mapping[str, int],
    token_ends: Mapping[int, int],
    seen_turn_count: int,
    seen_token_count: int,
) -> tuple[int, int]:
    """Measure age from the newest valid alternative in each required evidence group."""
    group_turns = []
    for group in event_groups:
        visible_turns = [event_turns[event_id] for event_id in group if event_id in event_turns]
        if not visible_turns:
            raise ValueError(f"checkpoint evidence group is not yet visible: {list(group)}")
        group_turns.append(max(visible_turns))
    return (
        max(seen_turn_count - turn_index for turn_index in group_turns),
        max(seen_token_count - token_ends[turn_index] for turn_index in group_turns),
    )


def _validate_history_contract(
    history_id: str,
    events: Sequence[Mapping[str, Any]],
    queries_by_suffix: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate event causality and source gold at each query's causal prefix."""
    seen_event_ids: set[str] = set()
    seen_fact_ids: set[str] = set()
    seen_facts: dict[str, Mapping[str, Any]] = {}
    closed_sessions: set[str] = set()
    current_session: str | None = None
    prior_turn_index = 0
    for event in events:
        event_id = _string(event.get("event_id"), "event event_id")
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate event_id {event_id!r} in history {history_id}")
        seen_event_ids.add(event_id)

        session_id = _string(event.get("session_id"), f"event {event_id} session_id")
        turn_index = _positive_int(event.get("turn_index"), f"event {event_id} turn_index")
        if session_id != current_session:
            if current_session is not None:
                closed_sessions.add(current_session)
            if session_id in closed_sessions:
                raise ValueError(
                    f"event chronology reopens session {session_id!r} in history {history_id}"
                )
            if turn_index != 1:
                raise ValueError(
                    f"event chronology for session {session_id!r} starts at turn {turn_index}, expected 1"
                )
            current_session = session_id
            prior_turn_index = 0
        if turn_index != prior_turn_index + 1:
            raise ValueError(
                f"event chronology for session {session_id!r} has turn {turn_index} "
                f"after {prior_turn_index}"
            )

        fact = event.get("fact")
        if not isinstance(fact, Mapping):
            raise ValueError(f"event {event_id} fact must be an object")
        fact_id = _string(fact.get("fact_id"), f"event {event_id} fact_id")
        for reference_field in ("supersedes", "retracts", "duplicate_of"):
            reference = event.get(reference_field)
            if reference is None:
                continue
            reference_id = _string(reference, f"event {event_id} {reference_field}")
            if reference_id not in seen_fact_ids:
                raise ValueError(
                    f"forward causal reference in event {event_id}: "
                    f"{reference_field}={reference_id!r} has not been observed"
                )
            if reference_field == "duplicate_of" and reference_id in seen_facts:
                prior = seen_facts[reference_id]
                if (
                    fact.get("subject") != prior.get("subject")
                    or fact.get("object") != prior.get("object")
                    or fact.get("qualifiers", {}).get("scope")
                    != prior.get("qualifiers", {}).get("scope")
                ):
                    raise ValueError(
                        f"duplicate lineage mismatch in event {event_id}: {reference_id!r}"
                    )
        seen_fact_ids.add(fact_id)
        seen_facts[fact_id] = fact
        prior_turn_index = turn_index

    for suffix, source_query in queries_by_suffix.items():
        source_gold = source_query.get("gold")
        query_events = list(events)
        checkpoint_contract = source_query.get("checkpoint_contract")
        if checkpoint_contract is not None:
            trigger_event_id = _string(
                checkpoint_contract.get("trigger_event_id"),
                f"query {history_id}-{suffix} trigger_event_id",
            )
            trigger_index = next(
                (
                    index
                    for index, event in enumerate(events)
                    if event.get("event_id") == trigger_event_id
                ),
                None,
            )
            if trigger_index is None:
                raise ValueError(
                    f"query {history_id}-{suffix} references missing trigger event "
                    f"{trigger_event_id}"
                )
            query_events = list(events[: trigger_index + 1])
        resolved = resolve_query(query_events, dict(_visible_query(source_query)))
        if resolved != source_gold:
            raise ValueError(
                f"causal-prefix gold mismatch for {history_id}-{suffix}: "
                f"resolved {resolved!r}, source has {source_gold!r}"
            )


def build_episodes(
    dataset_dir: Path,
    config: ContinualMemoryConfig,
    tokenizer: BenchmarkTokenizer,
) -> list[dict[str, Any]]:
    """Build deterministic online episodes and validate every checkpoint's gold answer."""
    dataset_dir = Path(dataset_dir)
    events = _read_jsonl(dataset_dir / "events.jsonl")
    queries = _read_jsonl(dataset_dir / "queries.jsonl")
    checkpoint_specs = _checkpoint_specs(config.source_profile)
    events_by_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    queries_by_history: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        if (
            event.get("split") == config.source_split
            and event.get("hardness_profile") == config.source_profile
        ):
            history_id = _string(event.get("history_id"), "event history_id")
            events_by_history[history_id].append(event)
    for query in queries:
        if (
            query.get("split") == config.source_split
            and query.get("hardness_profile") == config.source_profile
        ):
            history_id = _string(query.get("history_id"), "query history_id")
            query_id = _string(query.get("query_id"), "query query_id")
            suffix = _suffix(query_id, history_id)
            if suffix in queries_by_history[history_id]:
                raise ValueError(f"duplicate query suffix {suffix!r} for {history_id}")
            queries_by_history[history_id][suffix] = query

    eligible_histories = sorted(
        (
            history_id
            for history_id in events_by_history
            if set(queries_by_history[history_id]) == set(checkpoint_specs)
        ),
        key=lambda history_id: int(history_id.rsplit("-", maxsplit=1)[-1]),
    )
    if len(eligible_histories) < config.history_limit:
        raise ValueError(
            f"requested {config.history_limit} complete histories from {config.source_split}, "
            f"found {len(eligible_histories)}"
        )
    history_ids = eligible_histories[: config.history_limit]
    for history_id in history_ids:
        _validate_history_contract(
            history_id,
            events_by_history[history_id],
            queries_by_history[history_id],
        )
        suffix_map = {}
        for event in events_by_history[history_id]:
            event_id = _string(event.get("event_id"), "event event_id")
            suffix = _suffix(event_id, history_id)
            if suffix in suffix_map:
                raise ValueError(f"duplicate event suffix {suffix!r} for {history_id}")
            suffix_map[suffix] = event
        missing_triggers = sorted(
            str(spec["trigger"])
            for spec in checkpoint_specs.values()
            if spec["trigger"] not in suffix_map
        )
        if missing_triggers:
            raise ValueError(f"history {history_id} lacks checkpoint events {missing_triggers}")

    distractor_pool = [
        event
        for history_id in eligible_histories
        for event in events_by_history[history_id]
        if history_id not in history_ids or len(eligible_histories) > 1
    ]
    event_order_by_id = {
        str(event["event_id"]): event_order
        for members in events_by_history.values()
        for event_order, event in enumerate(members)
    }
    episodes = []
    for history_id in history_ids:
        history_events = events_by_history[history_id]
        other_events = [event for event in distractor_pool if event["history_id"] != history_id]
        for tier in config.interference_tiers:
            if tier.distractor_turns_per_target_event and not other_events:
                raise ValueError(
                    f"interference tier {tier.name} requires another {config.source_split} history"
                )
            total_turns = len(history_events) * (tier.distractor_turns_per_target_event + 1)
            turns: list[dict[str, Any]] = []
            checkpoints: list[dict[str, Any]] = []
            event_turns: dict[str, int] = {}
            token_ends: dict[int, int] = {}
            seen_events: list[dict[str, Any]] = []
            seen_stream_tokens = 0
            checkpoint_by_trigger = {
                str(spec["trigger"]): kind for kind, spec in checkpoint_specs.items()
            }
            for event_order, event in enumerate(history_events):
                target_text = _target_text(event)
                target_turn_index = len(turns) + 1
                target_turn = {
                    "turn_id": f"{history_id}:{tier.name}:turn-{target_turn_index}",
                    "role": "target",
                    "task_id": "temporal_preference_memory",
                    "thread_id": history_id,
                    "distractor_type": None,
                    "text": target_text,
                    "token_count": len(tokenizer.encode(target_text)),
                    "serialized_token_count": _serialized_turn_increment(
                        str(turns[-1]["text"]) if turns else None,
                        target_text,
                        tokenizer,
                    ),
                    "target_event_order": event_order,
                    "target_event": event,
                    "stream_event_order": event_order,
                    "stream_event": event,
                }
                turns.append(target_turn)
                seen_stream_tokens += int(target_turn["serialized_token_count"])
                seen_events.append(event)
                event_turns[str(event["event_id"])] = target_turn_index
                token_ends[target_turn_index] = seen_stream_tokens

                for distractor_index in range(tier.distractor_turns_per_target_event):
                    pool_index = _stable_index(
                        config.seed,
                        (history_id, tier.name, event_order, distractor_index),
                        len(other_events),
                    )
                    distractor_event = other_events[pool_index]
                    distractor_text = _distractor_text(distractor_event)
                    distractor_turn_index = len(turns) + 1
                    distractor_turn = {
                        "turn_id": f"{history_id}:{tier.name}:turn-{distractor_turn_index}",
                        "role": "distractor",
                        "task_id": "distractor_temporal_preference",
                        "thread_id": distractor_event["history_id"],
                        "distractor_type": "cross_task_event",
                        "text": distractor_text,
                        "token_count": len(tokenizer.encode(distractor_text)),
                        "serialized_token_count": _serialized_turn_increment(
                            str(turns[-1]["text"]), distractor_text, tokenizer
                        ),
                        "stream_event_order": event_order_by_id[str(distractor_event["event_id"])],
                        "stream_event": distractor_event,
                    }
                    turns.append(distractor_turn)
                    seen_stream_tokens += int(distractor_turn["serialized_token_count"])

                event_suffix = _suffix(str(event["event_id"]), history_id)
                checkpoint_kind = checkpoint_by_trigger.get(event_suffix)
                if checkpoint_kind is None:
                    continue
                source_query = queries_by_history[history_id][checkpoint_kind]
                visible_query = _visible_query(source_query)
                gold = resolve_query(list(seen_events), dict(visible_query))
                if gold is None:
                    raise ValueError(
                        f"unanswerable checkpoint {history_id}-{checkpoint_kind}: "
                        "causal target-event prefix resolved no answer"
                    )
                spec = checkpoint_specs[checkpoint_kind]
                seen_tokens = seen_stream_tokens
                contract = source_query.get("checkpoint_contract")
                if isinstance(contract, Mapping):
                    event_groups = [
                        [str(identifier) for identifier in group]
                        for group in contract.get("required_event_id_groups", [])
                    ]
                    fact_groups = [
                        [str(identifier) for identifier in group]
                        for group in contract.get("required_fact_id_groups", [])
                    ]
                else:
                    event_groups = _identifier_groups(
                        history_id, spec["event_groups"]  # type: ignore[arg-type]
                    )
                    fact_groups = _identifier_groups(
                        history_id, spec["fact_groups"]  # type: ignore[arg-type]
                    )
                if not event_groups or not fact_groups:
                    raise ValueError(
                        f"checkpoint {history_id}-{checkpoint_kind} has an empty evidence contract"
                    )
                turn_age, token_age = _checkpoint_ages(
                    event_groups,
                    event_turns,
                    token_ends,
                    len(turns),
                    seen_tokens,
                )
                query_text = _visible_query_text(visible_query)
                sliding_context = {
                    window.name: _fit_sliding_window(
                        turns, query_text, window.max_tokens, tokenizer
                    )
                    for window in config.sliding_token_windows
                }
                checkpoints.append(
                    {
                        "checkpoint_id": f"{history_id}:{tier.name}:{checkpoint_kind}",
                        "checkpoint_kind": checkpoint_kind,
                        "after_event_id": event["event_id"],
                        "stream_turn_index": len(turns),
                        "stream_token_count": seen_tokens,
                        "stream_position_percentage": 100.0 * len(turns) / total_turns,
                        "turn_age": turn_age,
                        "token_age": token_age,
                        "query": visible_query,
                        "query_token_count": len(tokenizer.encode(query_text)),
                        "sliding_context": sliding_context,
                        "gold": gold,
                        "composition_id": (
                            source_query.get("composition", {}).get("id")
                            if isinstance(source_query.get("composition"), Mapping)
                            else None
                        ),
                        "evidence_event_ids": sorted(
                            {identifier for group in event_groups for identifier in group}
                        ),
                        "evidence_fact_ids": sorted(
                            {identifier for group in fact_groups for identifier in group}
                        ),
                        "valid_alternative_event_id_groups": event_groups,
                        "valid_alternative_fact_id_groups": fact_groups,
                    }
                )

            if len(checkpoints) != len(checkpoint_specs):
                raise ValueError(
                    f"history {history_id} produced {len(checkpoints)} checkpoints, "
                    f"expected {len(checkpoint_specs)}"
                )
            distractor_count = sum(turn["role"] == "distractor" for turn in turns)
            realized_percentage = 100.0 * distractor_count / len(turns)
            if not math.isclose(
                realized_percentage,
                tier.distractor_turn_percentage,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"episode {history_id}:{tier.name} realized distractor percentage "
                    f"{realized_percentage}, expected {tier.distractor_turn_percentage}"
                )
            episodes.append(
                {
                    "episode_id": f"{history_id}:{tier.name}",
                    "history_id": history_id,
                    "interference_tier": tier.name,
                    "distractor_turns_per_target_event": tier.distractor_turns_per_target_event,
                    "distractor_turn_percentage": tier.distractor_turn_percentage,
                    "turns": turns,
                    "checkpoints": checkpoints,
                    "stream": {
                        "target_turn_count": len(history_events),
                        "distractor_turn_count": distractor_count,
                        "total_turn_count": len(turns),
                        "total_token_count": _stream_token_count(turns, tokenizer),
                    },
                }
            )
    return episodes
