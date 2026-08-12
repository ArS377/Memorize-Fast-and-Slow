"""Retrieval, scoring, and metric evaluation for continual-memory episodes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Mapping, Protocol, Sequence

from rank_bm25 import BM25Okapi

from experiments._continual_memory_config import (
    BenchmarkTokenizer,
    ContinualMemoryConfig,
    DenseEmbeddingConfig,
)
from experiments._continual_memory_episodes import (
    QUERY_TEMPLATE,
    _serialized_model_input,
    _serialized_turn_increment,
    _visible_query_text,
)
from experiments.synthetic_temporal_baselines import _tokenize
from experiments.synthetic_temporal_preferences import resolve_query
from experiments.private_lineage_reasoning import (
    ScallopLineageClient,
    resolve_private_lineage_reference,
)
from experiments.preference_stream_injection import PreferenceStreamInjectionClient


DELAYED_PREFERENCE_CHECKPOINTS = {
    "preference-change-delayed",
    "preference-incongruity-delayed",
}


class DenseEmbedder(Protocol):
    """Batch-capable embedding boundary shared with the repository embedder."""

    def encode_documents(self, texts: Sequence[str]) -> Any:
        """Encode model-visible stream turns."""

    def encode_queries(self, texts: Sequence[str]) -> Any:
        """Encode all model-visible checkpoint queries in one call."""

    def encode_query(self, text: str) -> Any:
        """Encode one model-visible checkpoint query."""

    def metadata(self) -> Mapping[str, Any]:
        """Return dense model provenance."""


def _selected_target_events(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return every selected stream event without hidden-role filtering."""
    selected: dict[str, tuple[str, int, dict[str, Any]]] = {}
    for turn in turns:
        stream_event = turn.get("stream_event", turn.get("target_event"))
        if not isinstance(stream_event, Mapping):
            continue
        event = dict(stream_event)
        event_id = str(event.get("event_id", ""))
        if not event_id:
            raise ValueError("selected stream event is missing event_id")
        selected[event_id] = (
            str(event.get("history_id", "")),
            int(turn.get("stream_event_order", turn.get("target_event_order", 0))),
            event,
        )
    return [
        event
        for _, _, event in sorted(
            selected.values(), key=lambda item: (item[0], item[1], item[2]["event_id"])
        )
    ]


def _bm25_turns(
    turns: Sequence[Mapping[str, Any]], query: Mapping[str, Any], k: int
) -> list[Mapping[str, Any]]:
    """Retrieve top-k seen target and distractor turns from gold-free query text."""
    if not turns:
        return []
    index = BM25Okapi([_tokenize(str(turn["text"])) for turn in turns])
    scores = index.get_scores(_tokenize(_visible_query_text(query)))
    ranked = sorted(range(len(turns)), key=lambda item: (-float(scores[item]), item))[:k]
    return [turns[item] for item in ranked]


def _dot(left: Any, right: Any) -> float:
    """Compute a checked dot product for repository or fixture embedding arrays."""
    left_values = list(left)
    right_values = list(right)
    if not left_values or len(left_values) != len(right_values):
        raise ValueError(
            f"dense vector shape mismatch: document={len(left_values)}, query={len(right_values)}"
        )
    return sum(float(a) * float(b) for a, b in zip(left_values, right_values))


def _dense_vectors(
    episodes: Sequence[Mapping[str, Any]], embedder: DenseEmbedder
) -> tuple[dict[tuple[str, int], Any], dict[tuple[str, str], Any]]:
    """Encode all turn occurrences and checkpoint queries in two backend calls."""
    document_key_to_text: dict[tuple[str, int], str] = {}
    query_key_to_text: dict[tuple[str, str], str] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        for turn_index, turn in enumerate(episode["turns"]):
            document_key_to_text[(episode_id, turn_index)] = str(turn["text"])
        for checkpoint in episode["checkpoints"]:
            checkpoint_id = str(checkpoint["checkpoint_id"])
            query_key_to_text[(episode_id, checkpoint_id)] = _visible_query_text(
                checkpoint["query"]
            )
    document_texts = list(dict.fromkeys(document_key_to_text.values()))
    raw_query_texts = list(dict.fromkeys(query_key_to_text.values()))
    from neurosym.adapters.dense_index import query_text_v1

    query_texts = [query_text_v1(text) for text in raw_query_texts]
    document_vectors = embedder.encode_documents(document_texts)
    encode_queries = getattr(embedder, "encode_queries", None)
    if not callable(encode_queries):
        raise ValueError("dense embedder must provide batched encode_queries")
    query_vectors = encode_queries(query_texts)
    if len(document_vectors) != len(document_texts):
        raise ValueError("dense embedder returned the wrong document vector count")
    if len(query_vectors) != len(query_texts):
        raise ValueError("dense embedder returned the wrong query vector count")
    document_by_text = dict(zip(document_texts, document_vectors))
    query_by_text = dict(zip(raw_query_texts, query_vectors))
    return (
        {key: document_by_text[text] for key, text in document_key_to_text.items()},
        {key: query_by_text[text] for key, text in query_key_to_text.items()},
    )


def _dense_turns(
    episode_id: str,
    checkpoint_id: str,
    turns: Sequence[Mapping[str, Any]],
    k: int,
    document_vectors: Mapping[tuple[str, int], Any],
    query_vectors: Mapping[tuple[str, str], Any],
) -> list[Mapping[str, Any]]:
    """Rank seen turns using precomputed batched vectors and stable index tie-breaking."""
    query_vector = query_vectors[(episode_id, checkpoint_id)]
    scores = [
        _dot(document_vectors[(episode_id, index)], query_vector) for index in range(len(turns))
    ]
    ranked = sorted(range(len(turns)), key=lambda item: (-scores[item], item))[:k]
    return [turns[item] for item in ranked]


def _preference_objects(
    turns: Sequence[Mapping[str, Any]], query: Mapping[str, Any]
) -> set[str]:
    """Return historical values whose incorrect return constitutes stale intrusion."""
    if query.get("kind") != "preference":
        return set()
    query_scope = query.get("scope", "default")
    preference_operations = {
        "add",
        "supersede",
        "temporary_exception",
        "backdated_correction",
        "duplicate_delivery",
        "direct_user_correction",
    }
    values = set()
    for event in _selected_target_events(turns):
        fact = event.get("fact")
        if (
            event.get("operation") not in preference_operations
            or not isinstance(fact, Mapping)
            or fact.get("subject") != query.get("subject")
            or not isinstance(fact.get("qualifiers"), Mapping)
            or fact["qualifiers"].get("scope", "default") != query_scope
        ):
            continue
        value = fact.get("object")
        if isinstance(value, str):
            values.add(value)
    return values


def _prediction(
    episode: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    method: str,
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve and score one selected memory view with deterministic oracle reasoning."""
    query = dict(checkpoint["query"])
    selected_events = _selected_target_events(selected)
    answer = resolve_query(selected_events, query)
    selected_event_ids = {str(event["event_id"]) for event in selected_events}
    selected_fact_ids = {
        str(event["fact"]["fact_id"])
        for event in selected_events
        if isinstance(event.get("fact"), Mapping)
    }
    event_group_hits = [
        bool(selected_event_ids & set(group))
        for group in checkpoint["valid_alternative_event_id_groups"]
    ]
    fact_group_hits = [
        bool(selected_fact_ids & set(group))
        for group in checkpoint["valid_alternative_fact_id_groups"]
    ]
    group_hits = event_group_hits + fact_group_hits
    historical_values = _preference_objects(
        episode["turns"][: int(checkpoint["stream_turn_index"])], query
    )
    gold = checkpoint["gold"]
    selected_distractors = sum(turn.get("role") == "distractor" for turn in selected)
    exact_evidence_hit = all(group_hits)
    answer_match = answer == gold
    grounded_correct = answer_match and exact_evidence_hit
    return {
        "evaluation_family": "retrieval",
        "episode_id": episode["episode_id"],
        "history_id": episode["history_id"],
        "interference_tier": episode["interference_tier"],
        "distractor_turns_per_target_event": episode[
            "distractor_turns_per_target_event"
        ],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "method": method,
        "retrieval_query": {"query_text": _visible_query_text(query)},
        "prediction": answer,
        "gold": gold,
        "answer_correct": answer_match,
        "grounded_answer_correct": grounded_correct,
        "exact_evidence_hit": exact_evidence_hit,
        "evidence_recall": sum(group_hits) / len(group_hits),
        "stale_memory_intrusion": answer != gold and answer in historical_values,
        "retraction_compliant": (
            grounded_correct
            if checkpoint["checkpoint_kind"] in {"private", "private-lineage"}
            else None
        ),
        "cross_task_interference": False,
        "cross_task_interference_eligible": False,
        "selected_distractor_turn_count": selected_distractors,
        "selected_turn_count": len(selected),
        "selected_turn_ids": [turn["turn_id"] for turn in selected],
        "selected_event_ids": sorted(selected_event_ids),
        "selected_fact_ids": sorted(selected_fact_ids),
        "turn_age": checkpoint["turn_age"],
        "token_age": checkpoint["token_age"],
        "stream_position_percentage": checkpoint["stream_position_percentage"],
        "stored_turn_count": checkpoint["stream_turn_index"],
        "stored_token_count": checkpoint["stream_token_count"],
        "composition_id": checkpoint.get("composition_id"),
    }


def _source_injection_turns(
    episode: Mapping[str, Any],
    result: Mapping[str, Any],
    query_text: str,
    kinds: set[str] | None = None,
    reverse_rank: bool = False,
) -> list[dict[str, Any]]:
    """Rank Scallop-derived source capsules using only visible query text."""
    stream_turns = {
        str(turn["stream_event"]["event_id"]): turn
        for turn in episode["turns"]
        if isinstance(turn.get("stream_event"), Mapping)
    }
    enabled_kinds = kinds or set(DELAYED_PREFERENCE_CHECKPOINTS)
    kind_mapping = {
        "preference-change-delayed": "preference_change",
        "preference-incongruity-delayed": "preference_incongruity",
    }
    enabled_relations = {
        kind_mapping.get(kind, kind) for kind in enabled_kinds
    }
    selected_capsules = []
    for relation in sorted(enabled_relations):
        capsules = [
            injection
            for injection in result["injections"]
            if injection["kind"] == relation
        ]
        if not capsules:
            continue
        capsule_texts = [
            " ".join(
                str(
                    stream_turns[event_id]["stream_event"].get("model_text")
                    or stream_turns[event_id]["stream_event"]["fact"]["support_text"]
                )
                for event_id in injection["source_event_ids"]
            )
            for injection in capsules
        ]
        capsule_index = BM25Okapi([_tokenize(text) for text in capsule_texts])
        capsule_scores = capsule_index.get_scores(_tokenize(query_text))
        ranked = sorted(
            range(len(capsules)),
            key=lambda index: (float(capsule_scores[index]), index)
            if reverse_rank
            else (-float(capsule_scores[index]), index),
        )
        selected_capsules.append(capsules[ranked[0]])
    ordered_ids = []
    for injection in selected_capsules:
        for event_id in injection["source_event_ids"]:
            if event_id not in ordered_ids:
                ordered_ids.append(event_id)
    missing = [event_id for event_id in ordered_ids if event_id not in stream_turns]
    if missing:
        raise ValueError(f"Scallop injection references events outside its episode: {missing}")
    return [
        {
            "turn_id": f"{episode['episode_id']}:scallop:{event_id}",
            "role": "scallop_injection",
            "task_id": "preference_consistency_memory",
            "thread_id": episode["history_id"],
            "distractor_type": None,
            "text": str(
                stream_turns[event_id]["stream_event"].get("model_text")
                or stream_turns[event_id]["stream_event"]["fact"]["support_text"]
            ),
            "token_count": 0,
            "stream_event_order": int(stream_turns[event_id]["stream_event_order"]),
            "stream_event": stream_turns[event_id]["stream_event"],
        }
        for event_id in ordered_ids
    ]


def _fit_injected_context(
    turns: Sequence[Mapping[str, Any]],
    injections: Sequence[Mapping[str, Any]],
    query_text: str,
    max_tokens: int,
    tokenizer: BenchmarkTokenizer,
) -> tuple[list[Mapping[str, Any]], int]:
    """Fit a maximal raw suffix after reserving the same budget for source injections."""
    injection_only = list(injections)
    minimum = len(tokenizer.encode(_serialized_model_input(injection_only, query_text)))
    if minimum > max_tokens:
        raise ValueError(
            f"Scallop source injection needs {minimum} tokens, exceeding {max_tokens}"
        )
    for turn in injection_only:
        turn["token_count"] = len(tokenizer.encode(str(turn["text"])))
    injection_count = int(injection_only[0]["token_count"])
    for previous, current in zip(injection_only, injection_only[1:]):
        injection_count += _serialized_turn_increment(
            str(previous["text"]), str(current["text"]), tokenizer
        )
    injection_last_text = str(injection_only[-1]["text"])
    raw_suffix_increments = [0] * (len(turns) + 1)
    for index in range(len(turns) - 1, -1, -1):
        raw_suffix_increments[index] = raw_suffix_increments[index + 1] + int(
            turns[index].get("serialized_token_count", turns[index]["token_count"])
        )
    prompt = QUERY_TEMPLATE.format(query_text=query_text)

    def exact_count(candidate_start: int) -> int:
        if candidate_start == len(turns):
            prompt_increment = len(
                tokenizer.encode(f"{injection_last_text}{prompt}")
            ) - len(tokenizer.encode(injection_last_text))
            return injection_count + prompt_increment
        first_raw_text = str(turns[candidate_start]["text"])
        boundary_increment = _serialized_turn_increment(
            injection_last_text, first_raw_text, tokenizer
        )
        last_raw_text = str(turns[-1]["text"])
        prompt_increment = len(tokenizer.encode(f"{last_raw_text}{prompt}")) - len(
            tokenizer.encode(last_raw_text)
        )
        return (
            injection_count
            + boundary_increment
            + raw_suffix_increments[candidate_start + 1]
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
    selected = [*injection_only, *turns[start:]]
    token_count = exact_count(start)
    return selected, token_count


def _stream_injection_ablation(
    predictions: Sequence[Mapping[str, Any]],
    window_names: Sequence[str],
    scallopy_versions: Sequence[str],
) -> dict[str, Any]:
    """Aggregate matched raw, Scallop, and reverse-ranked context arms."""
    by_key = {
        (str(row["episode_id"]), str(row["checkpoint_id"]), str(row["method"])): row
        for row in predictions
        if row["evaluation_family"] == "retrieval"
    }
    by_window: dict[str, Any] = {}
    for window_name in window_names:
        raw_method = f"sliding_context:{window_name}"
        injected_method = f"{raw_method}+scallop_injection"
        reverse_method = f"{raw_method}+reverse_ranked_injection"
        change_only_method = f"{raw_method}+scallop_change_only"
        incongruity_only_method = f"{raw_method}+scallop_incongruity_only"
        groups: dict[
            tuple[str, str],
            list[tuple[Mapping[str, Any], ...]],
        ] = defaultdict(list)
        for row in predictions:
            if row.get("method") != injected_method:
                continue
            key = (str(row["episode_id"]), str(row["checkpoint_id"]))
            raw = by_key.get((*key, raw_method))
            reverse = by_key.get((*key, reverse_method))
            change_only = by_key.get((*key, change_only_method))
            incongruity_only = by_key.get((*key, incongruity_only_method))
            if (
                raw is None
                or reverse is None
                or change_only is None
                or incongruity_only is None
            ):
                raise RuntimeError(f"incomplete Scallop stream ablation for {key}")
            groups[(str(row["interference_tier"]), str(row["checkpoint_kind"]))].append(
                (raw, row, reverse, change_only, incongruity_only)
            )
        by_window[window_name] = {
            tier: {
                kind: {
                    "matched_count": len(rows),
                    "no_injection_grounded_accuracy": sum(
                        bool(raw["grounded_answer_correct"]) for raw, _, _, _, _ in rows
                    )
                    / len(rows),
                    "scallop_injected_grounded_accuracy": sum(
                        bool(injected["grounded_answer_correct"])
                        for _, injected, _, _, _ in rows
                    )
                    / len(rows),
                    "reverse_ranked_control_grounded_accuracy": sum(
                        bool(reverse["grounded_answer_correct"])
                        for _, _, reverse, _, _ in rows
                    )
                    / len(rows),
                    "change_rule_only_grounded_accuracy": sum(
                        bool(change_only["grounded_answer_correct"])
                        for _, _, _, change_only, _ in rows
                    )
                    / len(rows),
                    "incongruity_rule_only_grounded_accuracy": sum(
                        bool(incongruity_only["grounded_answer_correct"])
                        for _, _, _, _, incongruity_only in rows
                    )
                    / len(rows),
                    "paired_delta": sum(
                        bool(injected["grounded_answer_correct"])
                        - bool(raw["grounded_answer_correct"])
                        for raw, injected, _, _, _ in rows
                    )
                    / len(rows),
                }
                for kind in sorted({group_kind for group_tier, group_kind in groups if group_tier == tier})
                for rows in [groups[(tier, kind)]]
            }
            for tier in sorted({group_tier for group_tier, _ in groups})
        }
    return {
        "enabled": True,
        "scallop_query_or_gold_used": False,
        "capsule_selector": "BM25 over visible query text, one slot per enabled relation",
        "source_grounded": True,
        "engine": "scallopy",
        "scallopy_versions": list(scallopy_versions),
        "rule_version": "preference_stream.v1",
        "by_window": by_window,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Aggregate accuracy, evidence, privacy, staleness, and interference metrics."""
    if not rows:
        return {
            "checkpoint_count": 0,
            "answer_accuracy": 0.0,
            "grounded_answer_accuracy": 0.0,
            "exact_evidence_hit_rate": 0.0,
            "evidence_recall": 0.0,
            "stale_memory_intrusion_rate": 0.0,
            "retraction_compliance": 0.0,
            "cross_task_interference_rate": 0.0,
            "retrieved_distractor_turn_rate": 0.0,
        }
    private_rows = [row for row in rows if row["retraction_compliant"] is not None]
    total_selected = sum(int(row["selected_turn_count"]) for row in rows)
    return {
        "checkpoint_count": len(rows),
        "answer_accuracy": sum(bool(row["answer_correct"]) for row in rows) / len(rows),
        "grounded_answer_accuracy": sum(
            bool(row["grounded_answer_correct"]) for row in rows
        )
        / len(rows),
        "exact_evidence_hit_rate": sum(bool(row["exact_evidence_hit"]) for row in rows)
        / len(rows),
        "evidence_recall": sum(float(row["evidence_recall"]) for row in rows) / len(rows),
        "stale_memory_intrusion_rate": sum(
            bool(row["stale_memory_intrusion"]) for row in rows
        )
        / len(rows),
        "retraction_compliance": (
            sum(bool(row["retraction_compliant"]) for row in private_rows) / len(private_rows)
            if private_rows
            else 0.0
        ),
        "cross_task_interference_rate": (
            sum(bool(row["cross_task_interference"]) for row in rows)
            / sum(bool(row["cross_task_interference_eligible"]) for row in rows)
            if any(row["cross_task_interference_eligible"] for row in rows)
            else 0.0
        ),
        "retrieved_distractor_turn_rate": (
            sum(int(row["selected_distractor_turn_count"]) for row in rows) / total_selected
            if total_selected
            else 0.0
        ),
    }


def _mark_cross_task_interference(predictions: Sequence[dict[str, Any]]) -> None:
    """Mark failures induced relative to each matched zero-distractor prediction."""
    references: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in predictions:
        if int(row["distractor_turns_per_target_event"]) != 0:
            continue
        key = (str(row["history_id"]), str(row["checkpoint_kind"]), str(row["method"]))
        if key in references:
            raise ValueError(f"duplicate zero-distractor prediction for {key}")
        references[key] = row
    for row in predictions:
        key = (str(row["history_id"]), str(row["checkpoint_kind"]), str(row["method"]))
        reference = references.get(key)
        if reference is None:
            raise ValueError(f"missing zero-distractor prediction for {key}")
        treated = int(row["distractor_turns_per_target_event"]) != 0
        row["cross_task_interference_eligible"] = treated and bool(
            reference["grounded_answer_correct"]
        )
        row["cross_task_interference"] = bool(
            row["cross_task_interference_eligible"]
            and not row["grounded_answer_correct"]
        )


def _lineage_reasoning_ablation(
    episodes: Sequence[Mapping[str, Any]],
    client: ScallopLineageClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare recursive Scallop with a one-hop rule on matched full evidence."""
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if int(episode["distractor_turns_per_target_event"]) != 0:
            continue
        for checkpoint in episode["checkpoints"]:
            if checkpoint["checkpoint_kind"] not in {
                "private",
                "private-lineage-positive",
                "private-lineage",
            }:
                continue
            seen_turns = episode["turns"][: int(checkpoint["stream_turn_index"])]
            events = _selected_target_events(seen_turns)
            query = dict(checkpoint["query"])
            responses = {
                "python_gold_resolver_replay": resolve_private_lineage_reference(
                    events, query, strict=True
                ),
                "scallop_recursive": client.resolve(events, query, mode="recursive"),
                "scallop_one_hop": client.resolve(events, query, mode="one_hop"),
            }
            for method, response in responses.items():
                rows.append(
                    {
                        "evaluation_family": "scallop_reasoning_ablation",
                        "episode_id": episode["episode_id"],
                        "history_id": episode["history_id"],
                        "interference_tier": episode["interference_tier"],
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "checkpoint_kind": checkpoint["checkpoint_kind"],
                        "method": method,
                        "prediction": response["answer"],
                        "gold": checkpoint["gold"],
                        "answer_correct": response["answer"] == checkpoint["gold"],
                        "retracted_fact_ids": response["retracted_fact_ids"],
                        "same_memory_edges": response.get("same_memory_edges", []),
                        "engine": response["engine"],
                        "mode": response.get("mode"),
                        "scallopy_version": response.get("scallopy_version"),
                    }
                )

    methods: dict[str, Any] = {}
    for method in ("python_gold_resolver_replay", "scallop_recursive", "scallop_one_hop"):
        method_rows = [row for row in rows if row["method"] == method]
        by_checkpoint = {
            kind: {
                "count": len(kind_rows),
                "accuracy": sum(row["answer_correct"] for row in kind_rows) / len(kind_rows),
            }
            for kind in dict.fromkeys(row["checkpoint_kind"] for row in method_rows)
            if (kind_rows := [row for row in method_rows if row["checkpoint_kind"] == kind])
        }
        methods[method] = {
            "count": len(method_rows),
            "accuracy": (
                sum(row["answer_correct"] for row in method_rows) / len(method_rows)
                if method_rows
                else 0.0
            ),
            "by_checkpoint_kind": by_checkpoint,
        }

    recursive = methods["scallop_recursive"]["by_checkpoint_kind"]
    one_hop = methods["scallop_one_hop"]["by_checkpoint_kind"]
    required = {"private", "private-lineage-positive", "private-lineage"}
    if set(recursive) != required or set(one_hop) != required:
        raise RuntimeError("Scallop reasoning ablation lacks required control checkpoints")
    if any(recursive[kind]["accuracy"] != 1.0 for kind in required):
        raise RuntimeError("recursive Scallop failed a private-lineage checkpoint")
    if one_hop["private"]["accuracy"] != 1.0 or one_hop["private-lineage-positive"]["accuracy"] != 1.0:
        raise RuntimeError("one-hop Scallop failed a non-recursive control")
    if one_hop["private-lineage"]["accuracy"] != 0.0:
        raise RuntimeError("one-hop Scallop unexpectedly solved the recursive lineage probe")
    return {
        "enabled": True,
        "engine": "scallopy",
        "scallopy_version": next(
            (
                row["scallopy_version"]
                for row in rows
                if row["method"] == "scallop_recursive"
                and row.get("scallopy_version")
            ),
            None,
        ),
        "methods": methods,
        "causal_feature": "recursive transitive closure over duplicate lineage",
        "treatment_checkpoint": "private-lineage",
        "recursive_accuracy": recursive["private-lineage"]["accuracy"],
        "one_hop_accuracy": one_hop["private-lineage"]["accuracy"],
    }, rows


def _method_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Aggregate a prediction slice independently for each evaluated method."""
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    return {
        method: _aggregate([row for row in rows if row["method"] == method]) for method in methods
    }


def _bin_label(value: float, bounds: Sequence[float], unit: str) -> str:
    """Assign one value to a configured upper-bound bin."""
    lower = 0.0
    for upper in bounds:
        if value <= upper:
            return f"{lower:g}-{upper:g}{unit}"
        lower = upper
    return f">{bounds[-1]:g}{unit}"


def _grouped_metrics(
    predictions: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Group retention metrics by a categorical prediction field."""
    keys = list(dict.fromkeys(str(row[field]) for row in predictions))
    return {
        key: _method_aggregates([row for row in predictions if str(row[field]) == key])
        for key in keys
    }


def _age_grouped_metrics(
    predictions: Sequence[Mapping[str, Any]],
    field: str,
    bounds: Sequence[float],
    unit: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Group retention metrics by configured numeric bins."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[_bin_label(float(row[field]), bounds, unit)].append(row)
    return {label: _method_aggregates(rows) for label, rows in grouped.items()}


def _memory_growth(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize stored turn and token growth once per checkpoint, not per baseline."""
    rows = [
        {
            "tier": episode["interference_tier"],
            "turns": checkpoint["stream_turn_index"],
            "tokens": checkpoint["stream_token_count"],
        }
        for episode in episodes
        for checkpoint in episode["checkpoints"]
    ]

    def summarize(members: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {
            "checkpoint_count": len(members),
            "stored_turns_mean": sum(int(row["turns"]) for row in members) / len(members),
            "stored_turns_max": max(int(row["turns"]) for row in members),
            "stored_tokens_mean": sum(int(row["tokens"]) for row in members) / len(members),
            "stored_tokens_max": max(int(row["tokens"]) for row in members),
        }

    overall = summarize(rows)
    overall["by_interference_tier"] = {
        tier: summarize([row for row in rows if row["tier"] == tier])
        for tier in dict.fromkeys(str(row["tier"]) for row in rows)
    }
    return overall


def evaluate_episodes(
    episodes: Sequence[Mapping[str, Any]],
    config: ContinualMemoryConfig,
    *,
    embedder: DenseEmbedder | None = None,
    lineage_client: ScallopLineageClient | None = None,
    stream_injection_client: PreferenceStreamInjectionClient | None = None,
    tokenizer: BenchmarkTokenizer | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate retrieval and context controls at every online checkpoint."""
    if not episodes:
        raise ValueError("cannot evaluate an empty episode collection")
    if config.dense_embedding.enabled and embedder is None:
        embedder = _default_dense_embedder(config.dense_embedding)
    if not config.dense_embedding.enabled and embedder is not None:
        raise ValueError("dense embedder was supplied while dense_embedding.enabled is false")
    if config.scallop_reasoner.enabled and lineage_client is None:
        if config.scallop_reasoner.endpoint is None:
            raise ValueError("enabled Scallop reasoner has no resolved endpoint")
        lineage_client = ScallopLineageClient(
            config.scallop_reasoner.endpoint,
            config.scallop_reasoner.timeout_seconds,
        )
    if not config.scallop_reasoner.enabled and lineage_client is not None:
        raise ValueError("lineage client supplied while scallop_reasoner.enabled is false")
    if not config.scallop_reasoner.enabled and stream_injection_client is not None:
        raise ValueError(
            "stream injection client supplied while scallop_reasoner.enabled is false"
        )
    if stream_injection_client is not None and tokenizer is None:
        raise ValueError("Scallop stream injection requires the benchmark tokenizer")
    dense_documents: Mapping[tuple[str, int], Any] = {}
    dense_queries: Mapping[tuple[str, str], Any] = {}
    if embedder is not None:
        dense_documents, dense_queries = _dense_vectors(episodes, embedder)

    injection_windows = list(config.sliding_token_windows)
    injection_results: dict[tuple[str, str], Mapping[str, Any]] = {}
    if stream_injection_client is not None:
        for episode in episodes:
            for checkpoint in episode["checkpoints"]:
                if checkpoint["checkpoint_kind"] not in DELAYED_PREFERENCE_CHECKPOINTS:
                    continue
                cache_key = (
                    str(episode["episode_id"]),
                    str(checkpoint["checkpoint_kind"]),
                )
                if cache_key not in injection_results:
                    seen_turns = episode["turns"][: int(checkpoint["stream_turn_index"])]
                    injection_results[cache_key] = stream_injection_client.derive(
                        _selected_target_events(seen_turns)
                    )

    predictions = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        for checkpoint in episode["checkpoints"]:
            seen_turns = episode["turns"][: int(checkpoint["stream_turn_index"])]
            selections: dict[str, Sequence[Mapping[str, Any]]] = {
                "full_structured_memory": [
                    turn for turn in seen_turns if turn["role"] == "target"
                ],
                "recency": seen_turns[-config.retrieval_k :],
                "bm25": _bm25_turns(seen_turns, checkpoint["query"], config.retrieval_k),
            }
            selection_metadata: dict[str, dict[str, Any]] = {}
            for window in config.sliding_token_windows:
                window_metadata = checkpoint["sliding_context"].get(window.name)
                if not isinstance(window_metadata, Mapping):
                    raise ValueError(
                        f"checkpoint {checkpoint['checkpoint_id']} lacks window {window.name}"
                    )
                start = int(window_metadata["start_turn_index"])
                method = f"sliding_context:{window.name}"
                selections[method] = seen_turns[start:]
                selection_metadata[method] = {
                    "max_model_input_tokens": window.max_tokens,
                    "model_input_token_count": int(
                        window_metadata["model_input_token_count"]
                    ),
                    "injected_source_event_ids": [],
                }
            if (
                stream_injection_client is not None
                and checkpoint["checkpoint_kind"] in DELAYED_PREFERENCE_CHECKPOINTS
            ):
                cache_key = (
                    str(episode["episode_id"]),
                    str(checkpoint["checkpoint_kind"]),
                )
                own_result = injection_results[cache_key]
                query_text = _visible_query_text(checkpoint["query"])
                own_injections = _source_injection_turns(
                    episode, own_result, query_text
                )
                change_only_injections = _source_injection_turns(
                    episode,
                    own_result,
                    query_text,
                    {"preference_change"},
                )
                incongruity_only_injections = _source_injection_turns(
                    episode,
                    own_result,
                    query_text,
                    {"preference_incongruity"},
                )
                reverse_injections = _source_injection_turns(
                    episode, own_result, query_text, reverse_rank=True
                )
                for window in injection_windows:
                    raw_method = f"sliding_context:{window.name}"
                    for suffix, injection_turns in (
                        ("scallop_injection", own_injections),
                        ("reverse_ranked_injection", reverse_injections),
                        ("scallop_change_only", change_only_injections),
                        ("scallop_incongruity_only", incongruity_only_injections),
                    ):
                        method = f"{raw_method}+{suffix}"
                        selected, token_count = _fit_injected_context(
                            seen_turns,
                            injection_turns,
                            query_text,
                            window.max_tokens,
                            tokenizer,
                        )
                        selections[method] = selected
                        selection_metadata[method] = {
                            "max_model_input_tokens": window.max_tokens,
                            "model_input_token_count": token_count,
                            "injected_source_event_ids": sorted(
                                {
                                    str(turn["stream_event"]["event_id"])
                                    for turn in injection_turns
                                }
                            ),
                        }
            if embedder is not None:
                selections["dense"] = _dense_turns(
                    episode_id,
                    str(checkpoint["checkpoint_id"]),
                    seen_turns,
                    config.retrieval_k,
                    dense_documents,
                    dense_queries,
                )
            for method, selected in selections.items():
                row = _prediction(episode, checkpoint, method, selected)
                row.update(selection_metadata.get(method, {}))
                predictions.append(row)

    _mark_cross_task_interference(predictions)
    full_memory = [
        row for row in predictions if row["method"] == "full_structured_memory"
    ]
    if not all(row["answer_correct"] for row in full_memory):
        failures = [row["checkpoint_id"] for row in full_memory if not row["answer_correct"]]
        raise RuntimeError(f"full structured memory failed oracle checkpoints: {failures}")
    if not all(row["exact_evidence_hit"] for row in full_memory):
        failures = [
            row["checkpoint_id"] for row in full_memory if not row["exact_evidence_hit"]
        ]
        raise RuntimeError(
            f"full structured memory failed evidence contract: {failures}"
        )
    if not all(row["grounded_answer_correct"] for row in full_memory):
        failures = [
            row["checkpoint_id"]
            for row in full_memory
            if not row["grounded_answer_correct"]
        ]
        raise RuntimeError(f"full structured memory failed grounded checkpoints: {failures}")
    reasoning_metrics: dict[str, Any] = {"enabled": False}
    reasoning_rows: list[dict[str, Any]] = []
    if lineage_client is not None:
        reasoning_metrics, reasoning_rows = _lineage_reasoning_ablation(
            episodes, lineage_client
        )
    stream_injection_metrics: dict[str, Any] = {"enabled": False}
    if stream_injection_client is not None:
        stream_injection_metrics = _stream_injection_ablation(
            predictions,
            [window.name for window in injection_windows],
            sorted(
                {
                    str(result["scallopy_version"])
                    for result in injection_results.values()
                }
            ),
        )
    metrics = {
        "methods": _method_aggregates(predictions),
        "retention_by_interference_tier": _grouped_metrics(
            predictions, "interference_tier"
        ),
        "retention_by_checkpoint_kind": _grouped_metrics(predictions, "checkpoint_kind"),
        "retention_by_composition": _grouped_metrics(
            [row for row in predictions if row.get("composition_id")], "composition_id"
        ),
        "retention_by_turn_age_bin": _age_grouped_metrics(
            predictions,
            "turn_age",
            config.metric_bins.turn_age_upper_bounds,
            " turns",
        ),
        "retention_by_token_age_bin": _age_grouped_metrics(
            predictions,
            "token_age",
            config.metric_bins.token_age_upper_bounds,
            " tokens",
        ),
        "retention_by_stream_position_bin": _age_grouped_metrics(
            predictions,
            "stream_position_percentage",
            config.metric_bins.stream_position_upper_bounds_percent,
            "%",
        ),
        "memory_growth": _memory_growth(episodes),
        "scallop_reasoning_ablation": reasoning_metrics,
        "scallop_stream_injection_ablation": stream_injection_metrics,
        "dataset": {
            "episode_count": len(episodes),
            "checkpoint_count": sum(len(episode["checkpoints"]) for episode in episodes),
            "interference_tiers": [asdict(tier) for tier in config.interference_tiers],
            "sliding_token_windows": [asdict(window) for window in config.sliding_token_windows],
            "model_parameter_tiers": [
                asdict(tier) for tier in config.model_parameter_tiers
            ],
        },
        "interpretation": {
            "summary": (
                "This is continual memory under interleaved tasks without online parameter "
                "learning, not generic long-context question answering."
            ),
            "oracle_reasoning_control": (
                "All baselines use deterministic structured resolution after memory selection; "
                "this is an oracle reasoning control, not a language-model reasoning score."
            ),
            "answer_credit": (
                "Grounded answer accuracy requires every declared event and fact evidence "
                "group for both known and UNKNOWN answers."
            ),
            "cross_task_interference": (
                "Cross-task interference counts only failures absent from the same method, "
                "history, and checkpoint under the matched zero-distractor tier, divided by "
                "treated comparisons whose zero-distractor control was correct."
            ),
            "model_parameter_axis_evaluated": False,
            "online_parameter_updates": False,
            "limitations": [
                "Model parameter tiers are metadata only and are not evaluated.",
                "No online model weight updates occur.",
                "Dense retrieval is omitted unless the pinned local embedding configuration is enabled.",
            ],
        },
    }
    if embedder is not None:
        metrics["dense_metadata"] = dict(embedder.metadata())
    return metrics, [*predictions, *reasoning_rows]


def _default_dense_embedder(config: DenseEmbeddingConfig) -> DenseEmbedder:
    """Create one repository embedder after a cache-only snapshot preflight."""
    from transformers.utils import cached_file

    try:
        cached = cached_file(
            config.model,
            "config.json",
            revision=config.revision,
            local_files_only=config.local_files_only,
        )
    except Exception as error:
        raise RuntimeError(
            f"dense model {config.model}@{config.revision} is not available locally"
        ) from error
    if cached is None:
        raise RuntimeError(f"dense model {config.model}@{config.revision} is not available locally")
    from neurosym.adapters.dense_index import SentenceTransformerEmbedder
    from neurosym.domain.retrieval_config import EmbeddingConfig

    return SentenceTransformerEmbedder(
        EmbeddingConfig(
            model=config.model,
            requested_revision=config.revision,
            device=config.device,
            batch_size=config.batch_size,
        ),
        local_files_only=config.local_files_only,
    )
