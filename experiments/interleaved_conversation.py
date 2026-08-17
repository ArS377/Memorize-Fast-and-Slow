"""Deterministically schedule causal account histories into one peer conversation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Protocol, Sequence

from experiments._continual_memory_episodes import TURN_SEPARATOR, _serialized_turn_increment


class Tokenizer(Protocol):
    """Minimal tokenizer boundary for exact schedule accounting."""

    def encode(self, text: str) -> list[Any]:
        """Encode model-visible text."""


def _rank(seed: int, *parts: object) -> bytes:
    """Return a stable pseudorandom rank without mutable PRNG state."""
    material = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).digest()


def _conversation_text(event: Mapping[str, Any]) -> str:
    """Render one semantic update as a substantial human-like task resumption."""
    text = str(event.get("model_text") or event["fact"]["support_text"])
    if event.get("preserve_model_text"):
        return text
    if event.get("hardness_profile") != "anti_shortcut_interleaved_v3":
        return text
    value = str(event.get("surface_object") or "the current choice")
    closing_variants = (
        "They closed by listing the next decision that would be needed when this work resumed.",
        "The remaining uncertainty was recorded so a later session could continue without guessing.",
        "Before pausing, they separated the settled choice from details that still needed confirmation.",
        "The session ended with a short handoff covering dependencies and the next practical check.",
        "They noted which assumption should be revisited if timing or available resources changed.",
        "A final recap distinguished the durable preference from the temporary implementation plan.",
        "The follow-up was left with a concrete question to answer after the other work progressed.",
        "They documented the unresolved tradeoff rather than silently treating it as settled.",
        "The handoff identified what evidence would justify changing this choice in a later session.",
        "They paused after confirming which detail should remain stable across future task switches.",
        "The closing note captured the dependency most likely to affect the next resumption.",
        "They finished by recording both the agreed action and the condition that could revise it.",
        "The next session was left a concise checkpoint instead of a generic reminder to continue.",
        "They marked one follow-up as time-sensitive and kept the rest as background context.",
        "The recap preserved the rationale so the choice would not look arbitrary when revisited.",
        "They ended with a clear boundary between this task and the neighboring work it depended on.",
    )
    variant_index = int.from_bytes(
        _rank(0, "dialogue", event.get("event_id", text))[:4], "big"
    ) % len(closing_variants)
    return " ".join(
        [
            text,
            f"The discussion then worked through practical details involving {value}, timing, and tradeoffs.",
            "Several implementation questions came up, including scheduling, available resources, and what should happen if the original plan changed.",
            "The user compared alternatives, revisited assumptions from an earlier session, and clarified which parts were temporary versus intended to persist.",
            closing_variants[variant_index],
        ]
    )


def _segment_lengths(
    events: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    history_id: str,
    minimum: int,
    maximum: int,
) -> list[int]:
    """Partition one causal history into deterministic bounded segments."""
    event_count = len(events)
    if minimum < 1 or maximum < minimum:
        raise ValueError("segment bounds must satisfy 1 <= minimum <= maximum")
    if event_count < minimum:
        raise ValueError(
            f"history {history_id} has {event_count} events, below minimum segment {minimum}"
        )
    contradiction_positions = {
        str(event.get("event_family")): index
        for index, event in enumerate(events)
        if str(event.get("event_family"))
        in {"contradiction_opening", "contradiction_rectification"}
    }
    opening_positions = [
        index
        for index, event in enumerate(events)
        if event.get("event_family") == "contradiction_opening"
    ]
    resolution_position = contradiction_positions.get("contradiction_rectification")
    candidates: list[list[int]] = []

    def visit(remaining: int, prefix: list[int]) -> None:
        if remaining == 0:
            boundaries = []
            offset = 0
            for length in prefix:
                boundaries.append((offset, offset + length - 1))
                offset += length
            if len(opening_positions) == 2 and resolution_position is not None:
                segment_indices = [
                    next(
                        index
                        for index, (start, end) in enumerate(boundaries)
                        if start <= position <= end
                    )
                    for position in [*opening_positions, resolution_position]
                ]
                if len(set(segment_indices)) != 3 or segment_indices[-1] == len(prefix) - 1:
                    return
            candidates.append(list(prefix))
            return
        for length in range(minimum, min(maximum, remaining) + 1):
            tail = remaining - length
            if tail and tail < minimum:
                continue
            visit(tail, [*prefix, length])

    visit(event_count, [])
    if not candidates:
        raise ValueError(
            f"cannot partition {history_id} into {minimum}-{maximum} event segments"
        )
    return min(
        candidates,
        key=lambda lengths: _rank(seed, "partition", history_id, *lengths),
    )


def build_interleaved_schedule(
    events: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Tokenizer,
    seed: int,
    concurrent_accounts: int,
    min_segment_events: int,
    max_segment_events: int,
) -> dict[str, Any]:
    """Build one globally interleaved stream while preserving each account's order."""
    by_history: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        history_id = str(event.get("history_id", ""))
        if not history_id:
            raise ValueError("scheduled events require history_id")
        by_history[history_id].append(event)
    if concurrent_accounts < 2 or concurrent_accounts > len(by_history):
        raise ValueError("concurrent_accounts must be between two and the history count")

    segment_queues: dict[str, list[list[Mapping[str, Any]]]] = {}
    for history_id, history_events in sorted(by_history.items()):
        lengths = _segment_lengths(
            history_events,
            seed=seed,
            history_id=history_id,
            minimum=min_segment_events,
            maximum=max_segment_events,
        )
        offset = 0
        segment_queues[history_id] = []
        for length in lengths:
            segment_queues[history_id].append(history_events[offset : offset + length])
            offset += length

    pending = sorted(
        segment_queues,
        key=lambda history_id: _rank(seed, "admission", history_id),
    )
    active = pending[:concurrent_accounts]
    pending = pending[concurrent_accounts:]
    segment_ordinals = defaultdict(int)
    account_event_indices = defaultdict(int)
    waiting_segments = defaultdict(int)
    segments = []
    turns = []
    schedule_step = 0
    prior_history: str | None = None
    while active:
        eligible = [history_id for history_id in active if history_id != prior_history]
        if not eligible:
            eligible = list(active)
        if pending:
            forced = [
                history_id
                for history_id in eligible
                if waiting_segments[history_id] >= 2 * len(active)
            ]
            pool = forced or eligible
            history_id = min(
                pool,
                key=lambda candidate: (
                    -waiting_segments[candidate] if forced else 0,
                    _rank(seed, "step", schedule_step, candidate),
                ),
            )
        else:
            history_id = min(
                eligible,
                key=lambda candidate: (
                    -len(segment_queues[candidate]),
                    -waiting_segments[candidate],
                    _rank(seed, "drain", schedule_step, candidate),
                ),
            )
        members = segment_queues[history_id].pop(0)
        segment_id = f"segment-{len(segments) + 1:06d}"
        segments.append(
            {
                "segment_id": segment_id,
                "history_id": history_id,
                "segment_ordinal": segment_ordinals[history_id],
                "schedule_step": schedule_step,
                "event_count": len(members),
                "start_turn_index": len(turns),
            }
        )
        for event in members:
            text = _conversation_text(event)
            turns.append(
                {
                    "turn_id": f"turn-{len(turns) + 1:07d}",
                    "segment_id": segment_id,
                    "history_id": history_id,
                    "segment_ordinal": segment_ordinals[history_id],
                    "account_event_index": account_event_indices[history_id],
                    "text": text,
                    "token_count": len(tokenizer.encode(text)),
                    "serialized_token_count": _serialized_turn_increment(
                        str(turns[-1]["text"]) if turns else None,
                        text,
                        tokenizer,
                    ),
                    "stream_event": dict(event),
                }
            )
            account_event_indices[history_id] += 1
        segments[-1]["end_turn_index"] = len(turns)
        segment_ordinals[history_id] += 1
        for candidate in active:
            waiting_segments[candidate] += 1
        waiting_segments[history_id] = 0
        prior_history = history_id
        if not segment_queues[history_id]:
            active.remove(history_id)
            if pending:
                admitted = pending.pop(0)
                active.append(admitted)
                waiting_segments[admitted] = 0
        schedule_step += 1

    return {
        "schedule_version": "interleaved_conversation.v1",
        "seed": seed,
        "concurrent_accounts": concurrent_accounts,
        "turn_separator": TURN_SEPARATOR,
        "segments": segments,
        "turns": turns,
    }


def schedule_metrics(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Return fail-loud interleaving and causal-order measurements."""
    segments = list(schedule["segments"])
    turns = list(schedule["turns"])
    adjacent = sum(
        left["history_id"] == right["history_id"]
        for left, right in zip(segments, segments[1:])
    )
    causal_violations = 0
    prior_event_index: dict[str, int] = {}
    for turn in turns:
        history_id = str(turn["history_id"])
        event_index = int(turn["account_event_index"])
        expected = prior_event_index.get(history_id, -1) + 1
        causal_violations += event_index != expected
        prior_event_index[history_id] = event_index
    segment_positions: dict[str, list[int]] = defaultdict(list)
    for index, segment in enumerate(segments):
        segment_positions[str(segment["history_id"])].append(index)
    intervening_counts = [
        len(
            {
                str(segments[index]["history_id"])
                for index in range(left + 1, right)
            }
        )
        for positions in segment_positions.values()
        for left, right in zip(positions, positions[1:])
    ]
    return {
        "account_count": len(segment_positions),
        "segment_count": len(segments),
        "turn_count": len(turns),
        "stream_token_count": sum(
            int(turn["serialized_token_count"]) for turn in turns
        ),
        "resume_count": sum(max(0, len(positions) - 1) for positions in segment_positions.values()),
        "adjacent_same_account_segments": adjacent,
        "causal_order_violations": causal_violations,
        "minimum_intervening_accounts_per_resume": min(intervening_counts, default=0),
        "maximum_intervening_accounts_per_resume": max(intervening_counts, default=0),
    }


def lifecycle_distribution(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Count state-changing event families across stream thirds."""
    turns = list(schedule["turns"])
    families = {
        "non_overlap_transition",
        "backdated_correction",
        "source_authority_conflict",
        "contradiction_opening",
        "contradiction_rectification",
        "retraction",
    }
    counts = {
        family: {"first": 0, "middle": 0, "final": 0}
        for family in sorted(families)
    }
    for index, turn in enumerate(turns):
        family = str(turn["stream_event"].get("event_family", ""))
        if family not in counts:
            continue
        fraction = index / max(1, len(turns))
        third = "first" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "final"
        counts[family][third] += 1
    return {
        family: {
            "by_stream_third": by_third,
            "total": sum(by_third.values()),
            "covers_all_stream_thirds": all(by_third.values()),
        }
        for family, by_third in counts.items()
    }


def suffix_contract_availability(
    turns: Sequence[Mapping[str, Any]],
    required_event_id_groups: Sequence[Sequence[str]],
    *,
    windows: Sequence[int],
) -> dict[int, bool]:
    """Report whether every required event group survives each token suffix."""
    start_by_window = {int(window): len(turns) for window in windows}
    trailing_increments = 0
    for index in range(len(turns) - 1, -1, -1):
        suffix_tokens = int(turns[index]["token_count"]) + trailing_increments
        for window in windows:
            if suffix_tokens <= window:
                start_by_window[int(window)] = index
        trailing_increments += int(turns[index]["serialized_token_count"])
    return {
        window: all(
            set(group)
            & {
                str(turn["stream_event"]["event_id"])
                for turn in turns[start_by_window[window] :]
            }
            for group in required_event_id_groups
        )
        for window in start_by_window
    }


def suffix_event_ids_by_window(
    turns: Sequence[Mapping[str, Any]], *, windows: Sequence[int]
) -> dict[int, set[str]]:
    """Return exact retained event IDs for several complete-turn suffix budgets."""
    start_by_window = {int(window): len(turns) for window in windows}
    trailing_increments = 0
    for index in range(len(turns) - 1, -1, -1):
        suffix_tokens = int(turns[index]["token_count"]) + trailing_increments
        for window in windows:
            if suffix_tokens <= window:
                start_by_window[int(window)] = index
        trailing_increments += int(turns[index]["serialized_token_count"])
    return {
        window: {
            str(turn["stream_event"]["event_id"])
            for turn in turns[start_index:]
        }
        for window, start_index in start_by_window.items()
    }
