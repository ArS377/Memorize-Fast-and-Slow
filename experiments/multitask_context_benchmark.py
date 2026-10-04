"""Generate and evaluate multi-task distractor contexts across context tiers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from rank_bm25 import BM25Okapi

from experiments.synthetic_temporal_baselines import _tokenize
from experiments.synthetic_temporal_preferences import resolve_query


DISTRACTOR_TYPES = (
    "lexical_overlap",
    "obsolete_task",
    "same_user_other_task",
    "unrelated_task",
)
SOURCE_ARTIFACTS = ("events.jsonl", "queries.jsonl")
SOURCE_FILES = (
    "configs/multitask_context_benchmark.json",
    "experiments/multitask_context_benchmark.py",
    "experiments/synthetic_temporal_baselines.py",
    "experiments/synthetic_temporal_preferences.py",
)
CONTEXT_SEPARATOR = "\n\n"
QUERY_TEMPLATE = "\n\nQuestion: {query_text}\nAnswer:"


class ContextTokenizer(Protocol):
    """Tokenizer boundary needed for auditable context accounting."""

    def encode(self, text: str) -> list[Any]:
        """Return model token IDs without special tokens."""

    def decode(self, tokens: list[Any]) -> str:
        """Decode a token prefix for exact budget fitting."""

    def encode_with_offsets(self, text: str) -> tuple[list[Any], list[tuple[int, int]]]:
        """Return token IDs and character offsets for executable input accounting."""

    def metadata(self) -> dict[str, Any]:
        """Return requested and resolved tokenizer identity."""


@dataclass(frozen=True)
class ContextTier:
    """One context-capacity and target-thread-density condition."""

    name: str
    max_tokens: int
    target_thread_percent: float
    hard_lexical_distractors: int


@dataclass(frozen=True)
class WindowTier:
    """One model-visible head-window truncation budget."""

    name: str
    max_tokens: int


@dataclass(frozen=True)
class ModelParameterTier:
    """Reporting-only parameter-count tier, independent of context capacity."""

    name: str
    min_billions: float
    max_billions: float | None


@dataclass(frozen=True)
class TokenizerConfig:
    """Pinned local tokenizer request."""

    name: str
    revision: str
    local_files_only: bool


@dataclass(frozen=True)
class ContextBenchmarkConfig:
    """Validated multi-task context benchmark configuration."""

    benchmark_version: str
    context_tiers: tuple[ContextTier, ...]
    context_window_tiers: tuple[WindowTier, ...]
    distractor_block_tokens: int
    distractor_mix_percent_by_block: dict[str, float]
    evidence_positions_percent: tuple[float, ...]
    history_limit: int
    model_parameter_tiers: tuple[ModelParameterTier, ...]
    retrieval_k: int
    seed: int
    source_split: str
    tokenizer: TokenizerConfig


class HuggingFaceTokenizer:
    """Pinned Hugging Face tokenizer adapter with resolved snapshot identity."""

    def __init__(self, config: TokenizerConfig) -> None:
        from transformers import AutoTokenizer, __version__ as transformers_version
        from transformers.utils import cached_file

        config_path = cached_file(
            config.name,
            "tokenizer_config.json",
            revision=config.revision,
            local_files_only=config.local_files_only,
        )
        if config_path is None:
            raise RuntimeError(
                f"tokenizer {config.name}@{config.revision} is not available in the local cache"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            revision=config.revision,
            local_files_only=config.local_files_only,
        )
        snapshot_path = Path(config_path)
        try:
            snapshot_index = snapshot_path.parts.index("snapshots")
            resolved_revision = snapshot_path.parts[snapshot_index + 1]
        except (ValueError, IndexError) as error:
            raise RuntimeError(f"cannot resolve tokenizer snapshot from {config_path}") from error
        self._metadata = {
            "requested_name": config.name,
            "requested_revision": config.revision,
            "resolved_revision": resolved_revision,
            "implementation": type(self._tokenizer).__name__,
            "transformers_version": transformers_version,
            "add_special_tokens": False,
        }

    def encode(self, text: str) -> list[int]:
        """Tokenize without model-added boundary tokens."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        """Decode a token prefix without cleanup that changes accounting."""
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Tokenize model-visible text with exact character offsets."""
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return list(encoded["input_ids"]), [tuple(offset) for offset in encoded["offset_mapping"]]

    def metadata(self) -> dict[str, Any]:
        """Return pinned tokenizer provenance."""
        return dict(self._metadata)


def _json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object and name malformed configuration at the boundary."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _positive_int(value: Any, name: str) -> int:
    """Parse a positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    """Parse a finite number without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def load_context_config(path: Path) -> ContextBenchmarkConfig:
    """Load and validate context tiers, percentages, and independent model tiers."""
    raw = _json_object(Path(path))
    required = {
        "benchmark_version",
        "context_tiers",
        "context_window_tiers",
        "distractor_block_tokens",
        "distractor_mix_percent_by_block",
        "evidence_positions_percent",
        "history_limit",
        "model_parameter_tiers",
        "retrieval_k",
        "seed",
        "source_split",
        "tokenizer",
    }
    missing = sorted(required - raw.keys())
    unknown = sorted(raw.keys() - required)
    if missing or unknown:
        raise ValueError(f"config keys mismatch: missing={missing}, unknown={unknown}")

    context_tiers = tuple(
        ContextTier(
            name=str(tier["name"]),
            max_tokens=_positive_int(tier["max_tokens"], "context tier max_tokens"),
            target_thread_percent=_finite_number(
                tier["target_thread_percent"], "target_thread_percent"
            ),
            hard_lexical_distractors=_positive_int(
                tier["hard_lexical_distractors"], "hard_lexical_distractors"
            ),
        )
        for tier in raw["context_tiers"]
    )
    if not context_tiers or any(not 0 < tier.target_thread_percent <= 100 for tier in context_tiers):
        raise ValueError("context tiers require target_thread_percent in (0, 100]")
    window_tiers = tuple(
        WindowTier(
            name=str(tier["name"]),
            max_tokens=_positive_int(tier["max_tokens"], "window tier max_tokens"),
        )
        for tier in raw["context_window_tiers"]
    )
    if not window_tiers:
        raise ValueError("context_window_tiers must be non-empty")
    positions = tuple(
        _finite_number(position, "evidence position")
        for position in raw["evidence_positions_percent"]
    )
    if not positions or any(not 0 <= position <= 100 for position in positions):
        raise ValueError("evidence positions must be percentages in [0, 100]")
    distractor_mix = {
        str(name): _finite_number(value, f"distractor mix {name}")
        for name, value in raw["distractor_mix_percent_by_block"].items()
    }
    if set(distractor_mix) != set(DISTRACTOR_TYPES):
        raise ValueError(f"distractor mix must contain exactly {DISTRACTOR_TYPES}")
    if any(value < 0 for value in distractor_mix.values()) or not math.isclose(
        sum(distractor_mix.values()), 100.0, abs_tol=1e-9
    ):
        raise ValueError("distractor mix percentages must be non-negative and sum to 100")
    model_tiers = tuple(
        ModelParameterTier(
            name=str(tier["name"]),
            min_billions=_finite_number(tier["min_billions"], "model min_billions"),
            max_billions=(
                None
                if tier["max_billions"] is None
                else _finite_number(tier["max_billions"], "model max_billions")
            ),
        )
        for tier in raw["model_parameter_tiers"]
    )
    if not model_tiers:
        raise ValueError("model_parameter_tiers must be non-empty")
    tokenizer_raw = raw["tokenizer"]
    if not isinstance(tokenizer_raw, dict):
        raise ValueError("tokenizer must be an object")
    tokenizer = TokenizerConfig(
        name=str(tokenizer_raw.get("name") or ""),
        revision=str(tokenizer_raw.get("revision") or ""),
        local_files_only=tokenizer_raw.get("local_files_only"),
    )
    if not tokenizer.name or not tokenizer.revision or not isinstance(tokenizer.local_files_only, bool):
        raise ValueError("tokenizer requires name, revision, and boolean local_files_only")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    source_split = str(raw["source_split"])
    if source_split not in {"train", "dev", "test"}:
        raise ValueError("source_split must be train, dev, or test")
    return ContextBenchmarkConfig(
        benchmark_version=str(raw["benchmark_version"]),
        context_tiers=context_tiers,
        context_window_tiers=window_tiers,
        distractor_block_tokens=_positive_int(
            raw["distractor_block_tokens"], "distractor_block_tokens"
        ),
        distractor_mix_percent_by_block=distractor_mix,
        evidence_positions_percent=positions,
        history_limit=_positive_int(raw["history_limit"], "history_limit"),
        model_parameter_tiers=model_tiers,
        retrieval_k=_positive_int(raw["retrieval_k"], "retrieval_k"),
        seed=seed,
        source_split=source_split,
        tokenizer=tokenizer,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read object-only JSONL with line-specific failures."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read required artifact {path}: {error}") from error
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} at line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"expected object in {path} at line {line_number}")
        rows.append(row)
    return rows


def _weighted_distractor_cycle(mix: Mapping[str, float]) -> list[str]:
    """Build a smooth 100-slot schedule whose prefixes track configured shares."""
    used = {name: 0 for name in DISTRACTOR_TYPES}
    cycle: list[str] = []
    for slot in range(100):
        name = max(
            DISTRACTOR_TYPES,
            key=lambda item: (((slot + 1) * mix[item] / 100.0) - used[item], item),
        )
        cycle.append(name)
        used[name] += 1
    return cycle


def _distractor_text(
    distractor_type: str,
    *,
    hard_lexical: bool,
    index: int,
    query_date: str,
    query_scope: str,
    subject: str,
) -> str:
    """Render a plausible distractor without revealing its hidden relevance label."""
    common = (
        f"Record {index:05d}. The discussion moved between planning, status checks, ownership, "
        "deadlines, follow-up questions, and notes from another part of the conversation. "
    )
    if distractor_type == "unrelated_task":
        return common + (
            f"Team-{index % 17:02d} reviewed project-{index % 29:02d}, assigned a deployment task, "
            "and recorded that the unrelated job remains open pending a budget review."
        )
    if distractor_type == "same_user_other_task":
        return common + (
            f"{subject} discussed a calendar invitation, code review, document revision, and handoff "
            "while deciding which option the team preferred for the next review."
        )
    if distractor_type == "lexical_overlap":
        if hard_lexical:
            return (
                f"Record {index:05d}. On {query_date}, {subject} catalogued an item in "
                f"{query_scope} scope. The same minutes recorded a preferred current option for "
                "a recommendation discussed by another group."
            )
        return common + (
            f"A different participant discussed {subject} while using the terms preferred, scope, "
            "date, current, option, and recommendation while revising a shared planning record."
        )
    if distractor_type == "obsolete_task":
        return common + (
            f"A task for {subject} was superseded, retracted, archived, and closed after its status "
            "record named a preferred option for the current recommendation."
        )
    raise ValueError(f"unsupported distractor type: {distractor_type}")


def _target_items(
    history_id: str,
    events: Sequence[Mapping[str, Any]],
    tokenizer: ContextTokenizer,
) -> list[dict[str, Any]]:
    """Convert target history events into labeled, uniquely positioned context items."""
    gold_fact_id = f"{history_id}-current"
    items = []
    for event_order, event in enumerate(events):
        fact = event.get("fact")
        if not isinstance(fact, Mapping):
            continue
        text = str(fact.get("support_text") or "")
        if not text:
            continue
        event_id = str(event.get("event_id") or "")
        items.append(
            {
                "context_item_id": "",
                "event_id": event_id,
                "event_order": event_order,
                "fact_id": str(fact.get("fact_id") or ""),
                "thread_id": history_id,
                "task_id": f"memory:{history_id}",
                "role": (
                    "gold_evidence"
                    if str(fact.get("fact_id") or "") == gold_fact_id
                    else "target_thread_context"
                ),
                "distractor_type": None,
                "distractor_difficulty": None,
                "text": text,
                "token_count": len(tokenizer.encode(text)),
                "event": dict(event),
            }
        )
    if sum(item["role"] == "gold_evidence" for item in items) < 1:
        raise ValueError(f"history {history_id} must contain answer-bearing current evidence")
    return items


def _distractor_items(
    *,
    case_id: str,
    token_budget: int,
    cycle: Sequence[str],
    cycle_offset: int,
    block_tokens: int,
    hard_lexical_distractors: int,
    query_date: str,
    query_scope: str,
    subject: str,
    tokenizer: ContextTokenizer,
) -> list[dict[str, Any]]:
    """Generate labeled distractors up to an exact tokenizer-defined budget."""
    items = []
    used_tokens = 0
    index = 0
    lexical_count = 0
    while used_tokens < token_budget:
        distractor_type = cycle[(cycle_offset + index) % len(cycle)]
        hard_lexical = (
            distractor_type == "lexical_overlap"
            and lexical_count < hard_lexical_distractors
        )
        text = _distractor_text(
            distractor_type,
            hard_lexical=hard_lexical,
            index=cycle_offset + index,
            query_date=query_date,
            query_scope=query_scope,
            subject=subject,
        )
        if distractor_type == "lexical_overlap":
            lexical_count += 1
        tokens = tokenizer.encode(text)
        remaining = token_budget - used_tokens
        keep = min(len(tokens), block_tokens, remaining)
        if keep < 1:
            break
        text = tokenizer.decode(tokens[:keep])
        actual_tokens = tokenizer.encode(text)
        if not actual_tokens:
            raise ValueError("tokenizer produced an empty distractor after decoding")
        if len(actual_tokens) > remaining:
            raise ValueError("decoded distractor exceeds its remaining token budget")
        items.append(
            {
                "context_item_id": "",
                "event_id": None,
                "event_order": None,
                "fact_id": None,
                "thread_id": f"{case_id}-distractor-thread-{index:05d}",
                "task_id": f"{case_id}-distractor-task-{index:05d}",
                "role": "distractor",
                "distractor_type": distractor_type,
                "distractor_difficulty": (
                    "hard_lexical_near_miss" if hard_lexical else "standard"
                ),
                "text": text,
                "token_count": len(actual_tokens),
                "event": None,
            }
        )
        used_tokens += len(actual_tokens)
        index += 1
    return items


def _serialize_model_input(
    items: Sequence[dict[str, Any]],
    query_text: str,
    tokenizer: ContextTokenizer,
) -> tuple[str, dict[str, int]]:
    """Serialize executable model input and assign exact tokenizer-coordinate spans."""
    context_parts: list[str] = []
    character_spans: list[tuple[int, int]] = []
    cursor = 0
    for item in items:
        if context_parts:
            context_parts.append(CONTEXT_SEPARATOR)
            cursor += len(CONTEXT_SEPARATOR)
        start = cursor
        text = str(item["text"])
        context_parts.append(text)
        cursor += len(text)
        character_spans.append((start, cursor))
    context_text = "".join(context_parts)
    query_suffix = QUERY_TEMPLATE.format(query_text=query_text)
    model_input_text = context_text + query_suffix
    input_ids, offsets = tokenizer.encode_with_offsets(model_input_text)
    context_token_count = len(tokenizer.encode(context_text))
    standalone_query_tokens = len(tokenizer.encode(query_suffix))
    for item, (character_start, character_end) in zip(items, character_spans):
        token_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > character_start and token_start < character_end
        ]
        if not token_indices:
            raise ValueError(f"context item {item['context_item_id']} has no serialized tokens")
        item["token_start"] = token_indices[0]
        item["token_end"] = token_indices[-1] + 1
        item["token_count"] = len(token_indices)
    return model_input_text, {
        "context_token_count": context_token_count,
        "query_prompt_token_count": len(input_ids) - context_token_count,
        "standalone_query_prompt_token_count": standalone_query_tokens,
        "model_input_token_count": len(input_ids),
    }


def _build_case(
    *,
    history_id: str,
    history_index: int,
    tier: ContextTier,
    tier_index: int,
    events: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
    config: ContextBenchmarkConfig,
    tokenizer: ContextTokenizer,
    distractor_cycle: Sequence[str],
) -> dict[str, Any]:
    """Build one context with a contiguous target thread at a configured position."""
    target_items = _target_items(history_id, events, tokenizer)
    target_tokens = sum(item["token_count"] for item in target_items)
    requested_total = math.ceil(target_tokens * 100.0 / tier.target_thread_percent)
    if requested_total > tier.max_tokens:
        raise ValueError(
            f"tier {tier.name} cannot honor {tier.target_thread_percent}% target density: "
            f"requires {requested_total} tokens but max_tokens={tier.max_tokens}"
        )
    distractor_budget = requested_total - target_tokens
    case_id = f"{history_id}-{tier.name}"
    case_number = history_index * len(config.context_tiers) + tier_index
    cycle_offset = case_number * 37
    distractors = _distractor_items(
        case_id=case_id,
        token_budget=distractor_budget,
        cycle=distractor_cycle,
        cycle_offset=cycle_offset,
        block_tokens=config.distractor_block_tokens,
        hard_lexical_distractors=tier.hard_lexical_distractors,
        query_date=str(query.get("date") or "unspecified-date"),
        query_scope=str(query.get("scope") or "default"),
        subject=str(query["subject"]),
        tokenizer=tokenizer,
    )
    requested_position = config.evidence_positions_percent[
        (history_index + tier_index) % len(config.evidence_positions_percent)
    ]
    target_prefix_tokens = 0
    for item in target_items:
        if item["role"] == "gold_evidence":
            break
        target_prefix_tokens += item["token_count"]
    target_offset = max(
        0.0,
        requested_total * requested_position / 100.0 - target_prefix_tokens,
    )
    before = []
    before_tokens = 0
    while distractors and before_tokens < target_offset:
        item = distractors.pop(0)
        before.append(item)
        before_tokens += item["token_count"]
    ordered_items = [*before, *target_items, *distractors]
    for item_index, item in enumerate(ordered_items):
        item["context_item_id"] = f"{case_id}-item-{item_index:05d}"
    model_input_text, token_counts = _serialize_model_input(
        ordered_items,
        str(query["query_text"]),
        tokenizer,
    )
    if token_counts["model_input_token_count"] > tier.max_tokens:
        raise ValueError(
            f"tier {tier.name} serialized input has {token_counts['model_input_token_count']} "
            f"tokens, exceeding max_tokens={tier.max_tokens}"
        )
    gold_items = [item for item in ordered_items if item["role"] == "gold_evidence"]
    if not gold_items:
        raise ValueError(f"case {case_id} has no gold evidence item")
    total_tokens = token_counts["model_input_token_count"]
    target_tokens = sum(
        item["token_count"] for item in ordered_items if item["role"] != "distractor"
    )
    gold_tokens = sum(item["token_count"] for item in gold_items)
    first_gold_item = min(gold_items, key=lambda item: item["token_start"])
    distractor_counts = Counter(
        item["distractor_type"] for item in ordered_items if item["role"] == "distractor"
    )
    visible_query = {
        key: value for key, value in query.items() if key not in {"gold", "split"}
    }
    return {
        "case_id": case_id,
        "benchmark_version": config.benchmark_version,
        "split": config.source_split,
        "task_kind": str(query["kind"]),
        "target_history_id": history_id,
        "query": visible_query,
        "gold": {
            "answer": query["gold"],
            "evidence_context_item_ids": [item["context_item_id"] for item in gold_items],
            "evidence_context_item_groups": [
                [item["context_item_id"] for item in gold_items]
            ],
            "evidence_event_ids": [item["event_id"] for item in gold_items],
            "evidence_fact_ids": [f"{history_id}-current"],
        },
        "context": ordered_items,
        "model_input_text": model_input_text,
        "context_metadata": {
            **token_counts,
            "context_token_tier": tier.name,
            "context_token_limit": tier.max_tokens,
            "requested_target_thread_percentage": tier.target_thread_percent,
            "target_thread_token_count": target_tokens,
            "target_thread_percentage": 100.0 * target_tokens / total_tokens,
            "gold_evidence_token_count": gold_tokens,
            "gold_evidence_percentage": 100.0 * gold_tokens / total_tokens,
            "gold_evidence_token_offset": first_gold_item["token_start"],
            "requested_evidence_position_percent": requested_position,
            "gold_evidence_position_percent": 100.0
            * first_gold_item["token_start"]
            / max(total_tokens - 1, 1),
            "distractor_counts": dict(sorted(distractor_counts.items())),
            "hard_lexical_distractor_count": sum(
                item.get("distractor_difficulty") == "hard_lexical_near_miss"
                for item in ordered_items
            ),
            "distractor_token_counts": dict(
                sorted(
                    {
                        name: sum(
                            item["token_count"]
                            for item in ordered_items
                            if item["distractor_type"] == name
                        )
                        for name in DISTRACTOR_TYPES
                    }.items()
                )
            ),
        },
    }


def build_context_cases(
    dataset_dir: Path,
    config: ContextBenchmarkConfig,
    tokenizer: ContextTokenizer,
) -> list[dict[str, Any]]:
    """Build deterministic tiered contexts from one frozen source split."""
    dataset_dir = Path(dataset_dir)
    events = _read_jsonl(dataset_dir / "events.jsonl")
    queries = _read_jsonl(dataset_dir / "queries.jsonl")
    events_by_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("split") == config.source_split:
            events_by_history[str(event["history_id"])].append(event)
    current_queries = {
        str(query["history_id"]): query
        for query in queries
        if query.get("split") == config.source_split
        and str(query.get("query_id") or "").endswith("-current")
    }
    history_ids = sorted(set(events_by_history) & set(current_queries))[: config.history_limit]
    if len(history_ids) < config.history_limit:
        raise ValueError(
            f"requested {config.history_limit} histories from {config.source_split}, found {len(history_ids)}"
        )
    distractor_cycle = _weighted_distractor_cycle(config.distractor_mix_percent_by_block)
    cases = []
    for history_index, history_id in enumerate(history_ids):
        for tier_index, tier in enumerate(config.context_tiers):
            cases.append(
                _build_case(
                    history_id=history_id,
                    history_index=history_index,
                    tier=tier,
                    tier_index=tier_index,
                    events=events_by_history[history_id],
                    query=current_queries[history_id],
                    config=config,
                    tokenizer=tokenizer,
                    distractor_cycle=distractor_cycle,
                )
            )
    return cases


def _selection_result(case: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Resolve one selected context subset and score answer and evidence retrieval."""
    target_events = sorted(
        (item for item in selected if isinstance(item.get("event"), Mapping)),
        key=lambda item: int(item["event_order"]),
    )
    answer = resolve_query([item["event"] for item in target_events], dict(case["query"]))
    selected_ids = {str(item["context_item_id"]) for item in selected}
    evidence_groups = case["gold"]["evidence_context_item_groups"]
    evidence_recall = sum(
        bool(selected_ids & set(group)) for group in evidence_groups
    ) / len(evidence_groups)
    target_count = sum(item.get("role") != "distractor" for item in selected)
    return {
        "answer": answer,
        "answer_correct": answer == case["gold"]["answer"],
        "gold_evidence_recall": evidence_recall,
        "target_thread_precision": target_count / len(selected) if selected else 0.0,
        "selected_context_item_ids": [item["context_item_id"] for item in selected],
    }


def _bm25_selection(case: Mapping[str, Any], k: int) -> list[Mapping[str, Any]]:
    """Return top-k context items from visible query text only."""
    context = case["context"]
    index = BM25Okapi([_tokenize(str(item["text"])) for item in context])
    scores = index.get_scores(_tokenize(str(case["query"]["query_text"])))
    ranked = sorted(range(len(context)), key=lambda item: (-float(scores[item]), item))[:k]
    return [context[index] for index in ranked]


def _random_selection(case: Mapping[str, Any], k: int, seed: int) -> list[Mapping[str, Any]]:
    """Return a stable random selection keyed by case identity."""
    digest = hashlib.sha256(f"{seed}:{case['case_id']}".encode("utf-8")).digest()
    randomizer = random.Random(int.from_bytes(digest[:8], "big"))
    indices = randomizer.sample(range(len(case["context"])), k=min(k, len(case["context"])))
    return [case["context"][index] for index in indices]


def _head_window(case: Mapping[str, Any], max_tokens: int) -> list[Mapping[str, Any]]:
    """Return complete context items visible while preserving the query prompt at the tail."""
    query_tokens = int(case["context_metadata"]["query_prompt_token_count"])
    available_context_tokens = max(max_tokens - query_tokens, 0)
    return [
        item for item in case["context"] if int(item["token_end"]) <= available_context_tokens
    ]


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Aggregate answer and evidence metrics for one matched slice."""
    if not results:
        return {
            "case_count": 0,
            "answer_accuracy": 0.0,
            "gold_evidence_recall": 0.0,
            "target_thread_precision": 0.0,
        }
    return {
        "case_count": len(results),
        "answer_accuracy": sum(bool(result["answer_correct"]) for result in results) / len(results),
        "gold_evidence_recall": sum(float(result["gold_evidence_recall"]) for result in results)
        / len(results),
        "target_thread_precision": sum(float(result["target_thread_precision"]) for result in results)
        / len(results),
    }


def evaluate_context_cases(
    cases: Sequence[Mapping[str, Any]],
    config: ContextBenchmarkConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate retrieval, semantic controls, and context-window truncation."""
    predictions = []
    for case in cases:
        context = case["context"]
        selections = {
            "recency": context[-config.retrieval_k :],
            "random": _random_selection(case, config.retrieval_k, config.seed),
            "bm25": _bm25_selection(case, config.retrieval_k),
            "oracle_thread_filter": [item for item in context if item["role"] != "distractor"],
            "full_structured_history": list(context),
        }
        method_results = {
            name: _selection_result(case, selected) for name, selected in selections.items()
        }
        window_results = {
            tier.name: _selection_result(case, _head_window(case, tier.max_tokens))
            for tier in config.context_window_tiers
        }
        predictions.append(
            {
                "case_id": case["case_id"],
                "context_token_tier": case["context_metadata"]["context_token_tier"],
                "requested_evidence_position_percent": case["context_metadata"][
                    "requested_evidence_position_percent"
                ],
                "realized_evidence_position_percent": case["context_metadata"][
                    "gold_evidence_position_percent"
                ],
                "gold_answer": case["gold"]["answer"],
                "methods": method_results,
                "window_truncation": window_results,
            }
        )

    method_names = tuple(predictions[0]["methods"]) if predictions else ()
    methods = {
        name: _aggregate([prediction["methods"][name] for prediction in predictions])
        for name in method_names
    }
    by_context_tier = {}
    for tier in config.context_tiers:
        members = [
            prediction
            for prediction in predictions
            if prediction["context_token_tier"] == tier.name
        ]
        by_context_tier[tier.name] = {
            name: _aggregate([member["methods"][name] for member in members])
            for name in method_names
        }
    by_position = {}
    for position in config.evidence_positions_percent:
        key = f"{position:g}"
        members = [
            prediction
            for prediction in predictions
            if min(
                config.evidence_positions_percent,
                key=lambda candidate: abs(
                    candidate - prediction["realized_evidence_position_percent"]
                ),
            )
            == position
        ]
        by_position[key] = {
            name: _aggregate([member["methods"][name] for member in members])
            for name in method_names
        }
    window_metrics = {
        tier.name: _aggregate(
            [prediction["window_truncation"][tier.name] for prediction in predictions]
        )
        for tier in config.context_window_tiers
    }
    window_by_context_tier = {
        context_tier.name: {
            window_tier.name: _aggregate(
                [
                    prediction["window_truncation"][window_tier.name]
                    for prediction in predictions
                    if prediction["context_token_tier"] == context_tier.name
                ]
            )
            for window_tier in config.context_window_tiers
        }
        for context_tier in config.context_tiers
    }
    window_by_position = {
        f"{position:g}": {
            window_tier.name: _aggregate(
                [
                    prediction["window_truncation"][window_tier.name]
                    for prediction in predictions
                    if min(
                        config.evidence_positions_percent,
                        key=lambda candidate: abs(
                            candidate - prediction["realized_evidence_position_percent"]
                        ),
                    )
                    == position
                ]
            )
            for window_tier in config.context_window_tiers
        }
        for position in config.evidence_positions_percent
    }
    realized_context_tiers = {}
    for tier in config.context_tiers:
        tier_cases = [
            case for case in cases if case["context_metadata"]["context_token_tier"] == tier.name
        ]
        metadata_rows = [case["context_metadata"] for case in tier_cases]
        block_counts = Counter()
        token_counts = Counter()
        for metadata in metadata_rows:
            block_counts.update(metadata["distractor_counts"])
            token_counts.update(metadata["distractor_token_counts"])
        total_blocks = sum(block_counts.values())
        total_distractor_tokens = sum(token_counts.values())
        realized_context_tiers[tier.name] = {
            "case_count": len(tier_cases),
            "requested_target_thread_percent": tier.target_thread_percent,
            "target_thread_percent_mean": sum(
                row["target_thread_percentage"] for row in metadata_rows
            )
            / len(metadata_rows),
            "model_input_tokens_min": min(row["model_input_token_count"] for row in metadata_rows),
            "model_input_tokens_max": max(row["model_input_token_count"] for row in metadata_rows),
            "gold_position_absolute_error_max": max(
                abs(
                    row["gold_evidence_position_percent"]
                    - row["requested_evidence_position_percent"]
                )
                for row in metadata_rows
            ),
            "hard_lexical_distractors": tier.hard_lexical_distractors,
            "distractor_block_percent": {
                name: 100.0 * block_counts[name] / total_blocks for name in DISTRACTOR_TYPES
            },
            "distractor_token_percent": {
                name: 100.0 * token_counts[name] / total_distractor_tokens
                for name in DISTRACTOR_TYPES
            },
        }
    metrics = {
        "methods": methods,
        "window_truncation": window_metrics,
        "window_truncation_by_context_tier": window_by_context_tier,
        "window_truncation_by_evidence_position": window_by_position,
        "by_context_tier": by_context_tier,
        "by_evidence_position": by_position,
        "realized_context_tiers": realized_context_tiers,
        "dataset": {
            "case_count": len(cases),
            "context_token_tiers": [asdict(tier) for tier in config.context_tiers],
            "context_window_tiers": [asdict(tier) for tier in config.context_window_tiers],
            "model_parameter_tiers": [asdict(tier) for tier in config.model_parameter_tiers],
            "model_parameter_axis_evaluated": False,
            "distractor_mix_percent_by_block": config.distractor_mix_percent_by_block,
        },
        "interpretation": {
            "summary": (
                "This benchmark isolates retrieval and context-availability failures under labeled "
                "multi-task distraction; deterministic answer resolution is an oracle reasoning control."
            ),
            "limitations": [
                "No language model is evaluated in this first context-layer run.",
                "Parameter-count tiers are reporting metadata and are independent of context-window tiers.",
                "BM25 and full-history scores use deterministic structured answer resolution after selection.",
            ],
        },
        "runtime": {
            "numpy": importlib.metadata.version("numpy"),
            "python": sys.version.split()[0],
            "rank_bm25": importlib.metadata.version("rank-bm25"),
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
    }
    return metrics, predictions


def _write_json(path: Path, payload: Any) -> None:
    """Write stable formatted JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write stable JSONL records in supplied order."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    """Return an artifact SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repository_root: Path) -> tuple[str | None, bool | None]:
    """Return current commit and effective dirty state."""
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
        return sha, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def _render_report(metrics: Mapping[str, Any], tokenizer: Mapping[str, Any]) -> str:
    """Render tiered retrieval and context-window results."""
    rows = [
        f"| {name} | {values['answer_accuracy']:.4f} | {values['gold_evidence_recall']:.4f} | "
        f"{values['target_thread_precision']:.4f} |"
        for name, values in metrics["methods"].items()
    ]
    window_rows = [
        f"| {name} | {values['answer_accuracy']:.4f} | {values['gold_evidence_recall']:.4f} |"
        for name, values in metrics["window_truncation"].items()
    ]
    tier_rows = [
        f"| {name} | {values['model_input_tokens_min']:,}-{values['model_input_tokens_max']:,} | "
        f"{values['requested_target_thread_percent']:.3f}% | "
        f"{values['target_thread_percent_mean']:.3f}% | "
        f"{values['gold_position_absolute_error_max']:.3f} pp | "
        f"{values['hard_lexical_distractors']} |"
        for name, values in metrics["realized_context_tiers"].items()
    ]
    return "\n".join(
        [
            "# Multi-task Context Distractor Benchmark",
            "",
            metrics["interpretation"]["summary"],
            "",
            "| Method | Answer accuracy | Gold evidence recall | Target-thread precision |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "## Realized context tiers",
            "",
            "| Context tier | Model-input tokens | Requested target thread | Realized target thread | Max position error | Hard lexical distractors |",
            "|---|---:|---:|---:|---:|---:|",
            *tier_rows,
            "",
            "## Head-window truncation",
            "",
            "| Window tier | Answer accuracy | Gold evidence recall |",
            "|---|---:|---:|",
            *window_rows,
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in metrics["interpretation"]["limitations"]],
            "",
            f"Tokenizer: `{tokenizer}`.",
            "",
        ]
    )


def run_context_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    config: ContextBenchmarkConfig,
    tokenizer: ContextTokenizer,
) -> dict[str, Any]:
    """Generate cases, evaluate controls, and write a hashed result bundle."""
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    git_sha, git_dirty = _git_state(repository_root)
    manifest = {
        "run_id": output_dir.name,
        "benchmark_version": config.benchmark_version,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config": asdict(config),
        "tokenizer": tokenizer.metadata(),
        "runtime": {
            "numpy": importlib.metadata.version("numpy"),
            "python": sys.version.split()[0],
            "rank_bm25": importlib.metadata.version("rank-bm25"),
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
        "dataset": str(dataset_dir),
        "dataset_sha256": {
            name: _sha256(Path(dataset_dir) / name) for name in SOURCE_ARTIFACTS
        },
        "source_sha256": {
            name: _sha256(repository_root / name) for name in SOURCE_FILES
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    try:
        cases = build_context_cases(dataset_dir, config, tokenizer)
        metrics, predictions = evaluate_context_cases(cases, config)
        metrics["tokenizer"] = tokenizer.metadata()
        _write_jsonl(output_dir / "cases.jsonl", cases)
        _write_jsonl(output_dir / "predictions.jsonl", predictions)
        _write_json(output_dir / "metrics.json", metrics)
        (output_dir / "report.md").write_text(
            _render_report(metrics, tokenizer.metadata()), encoding="utf-8"
        )
        artifact_names = ["cases.jsonl", "predictions.jsonl", "metrics.json", "report.md"]
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
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
            }
        )
        _write_json(output_dir / "manifest.json", manifest)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured context benchmark with a pinned local tokenizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = load_context_config(arguments.config)
    metrics = run_context_benchmark(
        arguments.dataset,
        arguments.output_dir,
        config,
        HuggingFaceTokenizer(config.tokenizer),
    )
    for method, values in metrics["methods"].items():
        print(f"{method}: answer_accuracy={values['answer_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
