"""Evaluate persona-conditioned Qwen recall over generated natural conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments._continual_memory_config import BenchmarkTokenizer, ContinualMemoryConfig
from experiments.continual_memory_benchmark import (
    HuggingFaceTokenizer,
    build_episodes,
    evaluate_episodes,
    load_benchmark_config,
)
from experiments.persona_conversation_generator import (
    CompletionClient,
    LLMResponse,
    OpenAICompletionClient,
    Persona,
    SOURCE_ARTIFACTS,
    _nonempty,
    _positive_int,
    _positive_number,
    _strict_object,
    default_personas,
)


@dataclass(frozen=True)
class ContextTier:
    """Maximum model-visible token budget for one complete-turn conversation suffix."""

    name: str
    max_tokens: int

    def __post_init__(self) -> None:
        """Validate a named positive token tier."""
        _nonempty(self.name, "context_tier.name")
        _positive_int(self.max_tokens, "context_tier.max_tokens")


@dataclass(frozen=True)
class RecallConfig:
    """Validated Qwen-only recall provider and context configuration."""

    endpoint: str
    api_key: str
    model: str
    timeout_seconds: float
    seed: int
    max_tokens: int
    history_limit: int
    max_concurrent_requests: int
    interference_tiers: tuple[str, ...]
    query_suffixes: tuple[str, ...]
    context_tiers: tuple[ContextTier, ...]
    prompt_schema_version: str

    def __post_init__(self) -> None:
        """Reject incomplete provider settings and ambiguous context tiers."""
        _nonempty(self.endpoint, "evaluation.endpoint")
        _nonempty(self.api_key, "evaluation.api_key")
        _nonempty(self.model, "evaluation.model")
        if not self.model.casefold().startswith("qwen"):
            raise ValueError("evaluation.model must identify Qwen")
        _positive_number(self.timeout_seconds, "evaluation.timeout_seconds")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("evaluation.seed must be an integer")
        _positive_int(self.max_tokens, "evaluation.max_tokens")
        _positive_int(self.history_limit, "evaluation.history_limit")
        _positive_int(
            self.max_concurrent_requests, "evaluation.max_concurrent_requests"
        )
        if not self.interference_tiers or any(
            not isinstance(value, str) or not value.strip()
            for value in self.interference_tiers
        ):
            raise ValueError("evaluation.interference_tiers must contain non-empty names")
        if len(set(self.interference_tiers)) != len(self.interference_tiers):
            raise ValueError("evaluation.interference_tiers must be unique")
        if not self.query_suffixes or any(
            not isinstance(value, str) or not value.startswith("-")
            for value in self.query_suffixes
        ):
            raise ValueError("evaluation.query_suffixes must contain hyphen-prefixed names")
        if len(set(self.query_suffixes)) != len(self.query_suffixes):
            raise ValueError("evaluation.query_suffixes must be unique")
        if not self.context_tiers:
            raise ValueError("evaluation.context_tiers must not be empty")
        if len({tier.name for tier in self.context_tiers}) != len(self.context_tiers):
            raise ValueError("evaluation.context_tier names must be unique")
        if any(left.max_tokens >= right.max_tokens for left, right in zip(self.context_tiers, self.context_tiers[1:])):
            raise ValueError("evaluation.context_tiers must be strictly increasing")
        _nonempty(self.prompt_schema_version, "evaluation.prompt_schema_version")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read object-valued JSONL and reject malformed boundary input."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    """Write stable formatted JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write stable JSONL records in supplied order."""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Return one artifact's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_provenance(dataset_dir: Path) -> dict[str, Any]:
    """Verify and bind evaluation to one completed Kimi generation bundle."""
    manifest_path = dataset_dir / "generation_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load generation manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, Mapping) or manifest.get("status") != "completed":
        raise ValueError("persona recall requires a completed generation manifest")
    model_identity = manifest.get("model_identity")
    if not isinstance(model_identity, str) or "kimi" not in model_identity.casefold():
        raise ValueError(f"generation manifest is not Kimi-authored: {model_identity!r}")
    recorded_hashes = manifest.get("artifact_sha256")
    if not isinstance(recorded_hashes, Mapping):
        raise ValueError("generation manifest lacks artifact_sha256")
    current_hashes = {}
    for name in SOURCE_ARTIFACTS:
        current = _sha256(dataset_dir / name)
        if recorded_hashes.get(name) != current:
            raise ValueError(f"generated dataset artifact hash mismatch for {name}")
        current_hashes[name] = current
    return {
        "path": str(dataset_dir),
        "generation_model": model_identity,
        "generation_manifest_sha256": _sha256(manifest_path),
        "artifact_sha256": current_hashes,
    }


def _validate_personas(personas: Sequence[Persona]) -> None:
    """Require one no-persona arm and matched pairs differing on exactly their named axis."""
    if len({persona.persona_id for persona in personas}) != len(personas):
        raise ValueError("persona IDs must be unique")
    if sum(persona.persona_id == "no-persona" for persona in personas) != 1:
        raise ValueError("personas must contain exactly one no-persona control")
    pairs: dict[str, list[Persona]] = defaultdict(list)
    for persona in personas:
        if persona.pair_id is not None:
            pairs[persona.pair_id].append(persona)
    if not pairs:
        raise ValueError("personas must contain matched counterfactual pairs")
    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"counterfactual pair {pair_id} must contain exactly two personas")
        left, right = pair
        if left.counterfactual_axis != right.counterfactual_axis:
            raise ValueError(f"counterfactual pair {pair_id} has inconsistent axes")
        axis = left.counterfactual_axis
        stable = (
            "name", "age", "ethnicity", "education", "gender_identity", "disability",
            "religion", "socioeconomic_background", "locale", "occupation", "languages",
            "interests",
        )
        differences = [field for field in stable if getattr(left, field) != getattr(right, field)]
        if differences != [axis]:
            raise ValueError(
                f"counterfactual pair {pair_id} must differ only on {axis}, got {differences}"
            )


def _normalize_short_answer(value: str) -> str:
    """Normalize harmless punctuation and casing in one short answer."""
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _answer_state(raw: str, gold: str) -> tuple[str | None, bool, str]:
    """Conservatively parse an exact short answer, refusal, or invalid response."""

    stripped = raw.strip()
    if not stripped:
        return None, False, "invalid"
    if _normalize_short_answer(stripped) == _normalize_short_answer(gold):
        return stripped, True, "answered"
    lowered = stripped.casefold()
    refusal_markers = ("i don't know", "i do not know", "cannot answer", "can't answer", "unknown")
    if any(marker in lowered for marker in refusal_markers):
        return None, False, "refusal"
    if len(stripped.split()) > 12:
        return None, False, "invalid"
    return stripped, False, "answered"


def _surface_queries(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """Index generated natural query fields while requiring complete surface annotations."""
    indexed = {}
    for query in _read_jsonl(dataset_dir / "queries.jsonl"):
        query_id = _nonempty(query.get("query_id"), "query.query_id")
        surface_query = _nonempty(query.get("surface_query_text"), f"query {query_id} surface_query_text")
        surface_gold = query.get("surface_gold")
        if not isinstance(surface_gold, str) or not surface_gold.strip():
            raise ValueError(f"query {query_id} surface_gold must be a non-empty string")
        indexed[query_id] = {
            "surface_query_text": surface_query,
            "surface_gold": surface_gold,
        }
    return indexed


def _messages(persona: Persona, context: str, query_text: str) -> list[dict[str, str]]:
    """Build a fixed exact-answer prompt where only the prepended persona biography varies."""
    system = "Answer the recall question with only the exact short answer."
    if persona.biography:
        system = f"Persona biography: {persona.biography}\n\n{system}"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Conversation:\n{context}\n\nQuestion:\n{query_text}"
                "\n\nAnswer with only the exact short answer:"
            ),
        },
    ]


def _serialized_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """Serialize chat roles deterministically for declared context-tier accounting."""
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _chat_token_count(
    tokenizer: BenchmarkTokenizer, messages: Sequence[Mapping[str, str]]
) -> int:
    """Count the exact provider chat template when the tokenizer exposes it."""
    encode_chat = getattr(tokenizer, "encode_chat", None)
    if callable(encode_chat):
        return len(encode_chat(messages))
    return len(tokenizer.encode(_serialized_messages(messages)))


def _fit_shared_persona_window(
    turns: Sequence[Mapping[str, Any]],
    query_text: str,
    personas: Sequence[Persona],
    max_tokens: int,
    tokenizer: BenchmarkTokenizer,
) -> tuple[int, dict[str, int]]:
    """Fit one complete-turn suffix that stays identical across all persona arms."""
    start = len(turns)
    context_parts: list[str] = []
    counts = {
        persona.persona_id: _chat_token_count(
            tokenizer, _messages(persona, "", query_text)
        )
        for persona in personas
    }
    if not counts or max(counts.values()) > max_tokens:
        raise ValueError(
            f"persona prompt and query exceed the declared context limit of {max_tokens} tokens"
        )
    for candidate in range(len(turns) - 1, -1, -1):
        candidate_parts = [_visible_turn_text(turns[candidate]), *context_parts]
        context = "\n".join(candidate_parts)
        counts = {
            persona.persona_id: _chat_token_count(
                tokenizer, _messages(persona, context, query_text)
            )
            for persona in personas
        }
        if max(counts.values()) > max_tokens:
            retained_context = "\n".join(context_parts)
            retained_counts = {
                persona.persona_id: _chat_token_count(
                    tokenizer,
                    _messages(persona, retained_context, query_text),
                )
                for persona in personas
            }
            return start, retained_counts
        start = candidate
        context_parts = candidate_parts
    return start, counts


def _visible_turn_text(turn: Mapping[str, Any]) -> str:
    """Label target and cross-history dialogue so speakers cannot collapse together."""
    text = str(turn["text"])
    if turn.get("role") == "target":
        return f"Account holder conversation:\n{text}"
    return f"Unrelated person's conversation, not the account holder:\n{text}"


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate correctness and response-state rates for one non-empty row slice."""
    if not rows:
        return {"count": 0, "accuracy": 0.0, "refusal_rate": 0.0, "invalid_rate": 0.0}
    return {
        "count": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "refusal_rate": sum(row["response_state"] == "refusal" for row in rows) / len(rows),
        "invalid_rate": sum(row["response_state"] == "invalid" for row in rows) / len(rows),
    }


def _paired_deltas(rows: Sequence[Mapping[str, Any]], personas: Sequence[Persona]) -> dict[str, Any]:
    """Aggregate matched condition deltas without overstating independent sample size."""
    by_pair = {
        pair_id: sorted(
            [persona for persona in personas if persona.pair_id == pair_id],
            key=lambda persona: persona.persona_id,
        )
        for pair_id in sorted({persona.pair_id for persona in personas if persona.pair_id})
    }
    indexed = {
        (
            str(row["episode_id"]), str(row["context_tier"]), str(row["query_id"]),
            str(row["persona_id"]),
        ): row
        for row in rows
    }
    results = {}
    item_keys = sorted(
        {
            (str(row["episode_id"]), str(row["context_tier"]), str(row["query_id"]))
            for row in rows
        }
    )
    for pair_id, pair in by_pair.items():
        left, right = pair
        deltas = []
        response_disagreements = []
        for item in item_keys:
            left_row = indexed.get((*item, left.persona_id))
            right_row = indexed.get((*item, right.persona_id))
            if left_row is None or right_row is None:
                raise ValueError(f"matched persona rows are incomplete for {pair_id} at {item}")
            deltas.append(int(bool(right_row["correct"])) - int(bool(left_row["correct"])))
            response_disagreements.append(
                _normalize_short_answer(str(left_row["raw_response"]))
                != _normalize_short_answer(str(right_row["raw_response"]))
            )
        independent_items = {
            (str(row["history_id"]), str(row["query_id"]))
            for row in rows
            if row["persona_id"] in {left.persona_id, right.persona_id}
        }
        results[pair_id] = {
            "axis": left.counterfactual_axis,
            "left_persona_id": left.persona_id,
            "right_persona_id": right.persona_id,
            "matched_condition_count": len(deltas),
            "independent_history_query_count": len(independent_items),
            "mean_accuracy_delta": sum(deltas) / len(deltas),
            "accuracy_disagreement_rate": sum(delta != 0 for delta in deltas) / len(deltas),
            "response_disagreement_rate": sum(response_disagreements) / len(deltas),
        }
    return results


def _control_summary(
    predictions: Sequence[Mapping[str, Any]],
    checkpoint_ids: set[str],
    context_tiers: Sequence[ContextTier],
) -> dict[str, Any]:
    """Aggregate deterministic controls over exactly the Qwen query population."""
    allowed_methods = {
        "full_structured_memory",
        *(f"sliding_context:{tier.name}" for tier in context_tiers),
    }
    selected = [
        row
        for row in predictions
        if row.get("checkpoint_id") in checkpoint_ids
        and row.get("method") in allowed_methods
    ]
    controls = {}
    for method in sorted(allowed_methods):
        method_rows = [row for row in selected if row["method"] == method]
        if not method_rows:
            raise ValueError(f"deterministic control {method} has no matched rows")
        values = {
            "count": len(method_rows),
            "answer_accuracy": sum(bool(row["answer_correct"]) for row in method_rows)
            / len(method_rows),
            "grounded_answer_accuracy": sum(
                bool(row["grounded_answer_correct"]) for row in method_rows
            )
            / len(method_rows),
        }
        if method == "full_structured_memory":
            controls[method] = {
                "label": "oracle control",
                **values,
            }
        else:
            controls[method] = {
                "label": "deterministic sliding-window control",
                **values,
            }
    return controls


def evaluate_persona_recall(
    dataset_dir: Path,
    output_dir: Path,
    benchmark_config: ContinualMemoryConfig,
    tokenizer: BenchmarkTokenizer,
    recall_config: RecallConfig,
    client: CompletionClient,
    *,
    personas: Sequence[Persona] | None = None,
) -> dict[str, Any]:
    """Run Qwen recall arms and existing deterministic controls over shared task inputs."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    personas = tuple(personas or default_personas())
    _validate_personas(personas)
    dataset_provenance = _dataset_provenance(dataset_dir)
    manifest_path = output_dir / "evaluation_manifest.json"
    predictions_path = output_dir / "qwen_predictions.jsonl"
    report_path = output_dir / "persona_recall_report.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "evaluator_model_family": "Qwen only; Kimi is not used in evaluation",
        "provider_identity_assurance": (
            "endpoint-self-reported and checked against configured Qwen model; artifact hashes "
            "are cryptographic"
        ),
        "prompt_schema_version": recall_config.prompt_schema_version,
        "request_parameters": {
            "model": recall_config.model,
            "timeout_seconds": recall_config.timeout_seconds,
            "seed": recall_config.seed,
            "max_tokens": recall_config.max_tokens,
            "history_limit": recall_config.history_limit,
            "max_concurrent_requests": recall_config.max_concurrent_requests,
            "interference_tiers": list(recall_config.interference_tiers),
            "query_suffixes": list(recall_config.query_suffixes),
            "context_tiers": [asdict(tier) for tier in recall_config.context_tiers],
            "temperature": 0,
        },
        "personas": [asdict(persona) for persona in personas],
        "dataset": dataset_provenance,
        "controls": {"full_structured_memory": "oracle control"},
    }
    _write_json(manifest_path, manifest)
    rows: list[dict[str, Any]] = []
    executor: ThreadPoolExecutor | None = None
    try:
        surface_queries = _surface_queries(dataset_dir)
        available_tiers = {
            tier.name: tier for tier in benchmark_config.interference_tiers
        }
        missing_tiers = sorted(
            set(recall_config.interference_tiers) - set(available_tiers)
        )
        if missing_tiers:
            raise ValueError(
                f"requested interference tiers are absent from benchmark config: {missing_tiers}"
            )
        benchmark_config = replace(
            benchmark_config,
            history_limit=recall_config.history_limit,
            interference_tiers=tuple(
                available_tiers[name] for name in recall_config.interference_tiers
            ),
        )
        episodes = build_episodes(dataset_dir, benchmark_config, tokenizer)
        control_config = replace(
            benchmark_config,
            dense_embedding=replace(benchmark_config.dense_embedding, enabled=False),
            scallop_reasoner=replace(
                benchmark_config.scallop_reasoner, enabled=False, endpoint=None
            ),
        )
        selected_checkpoint_ids = {
            str(checkpoint["checkpoint_id"])
            for episode in episodes
            for checkpoint in episode["checkpoints"]
            if str(checkpoint["query"]["query_id"]).endswith(
                recall_config.query_suffixes
            )
        }
        if not selected_checkpoint_ids:
            raise ValueError("configured query suffixes selected no checkpoints")
        _, control_predictions = evaluate_episodes(episodes, control_config)
        executor = ThreadPoolExecutor(
            max_workers=recall_config.max_concurrent_requests
        )
        for episode in episodes:
            for checkpoint in episode["checkpoints"]:
                query_id = str(checkpoint["query"]["query_id"])
                if not query_id.endswith(recall_config.query_suffixes):
                    continue
                if query_id not in surface_queries:
                    raise ValueError(f"checkpoint references query without surface fields: {query_id}")
                surface = surface_queries[query_id]
                query_text = str(surface["surface_query_text"])
                gold = str(surface["surface_gold"])
                seen_turns = episode["turns"][: int(checkpoint["stream_turn_index"])]
                for tier in recall_config.context_tiers:
                    start, model_input_counts = _fit_shared_persona_window(
                        seen_turns, query_text, personas, tier.max_tokens, tokenizer
                    )
                    selected_turns = seen_turns[start:]
                    context = "\n".join(_visible_turn_text(turn) for turn in selected_turns)
                    conversation_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
                    pending: list[
                        tuple[Persona, Future[LLMResponse]]
                    ] = []
                    for persona in personas:
                        messages = _messages(persona, context, query_text)
                        pending.append((persona, executor.submit(
                            client.complete,
                            messages=messages,
                            model=recall_config.model,
                            timeout=recall_config.timeout_seconds,
                            seed=recall_config.seed,
                            max_tokens=recall_config.max_tokens,
                        )))
                    for persona, future in pending:
                        response = future.result()
                        if response.model != recall_config.model:
                            raise ValueError(
                                f"Qwen endpoint returned model {response.model!r}, expected "
                                f"{recall_config.model!r}"
                            )
                        provider_prompt_tokens = response.usage.get("prompt_tokens")
                        if not isinstance(provider_prompt_tokens, int):
                            raise ValueError("Qwen response omitted integer prompt_tokens usage")
                        if provider_prompt_tokens > tier.max_tokens:
                            raise ValueError(
                                f"provider counted {provider_prompt_tokens} prompt tokens for "
                                f"{tier.name}, exceeding {tier.max_tokens}"
                            )
                        if response.finish_reason != "stop":
                            prediction, correct, state = None, False, "invalid"
                        else:
                            prediction, correct, state = _answer_state(response.content, gold)
                        rows.append(
                            {
                                "episode_id": episode["episode_id"],
                                "history_id": episode["history_id"],
                                "interference_tier": episode["interference_tier"],
                                "distractor_turns_per_target_event": episode[
                                    "distractor_turns_per_target_event"
                                ],
                                "checkpoint_kind": checkpoint["checkpoint_kind"],
                                "query_id": query_id,
                                "query_text": query_text,
                                "gold": gold,
                                "persona_id": persona.persona_id,
                                "persona_pair_id": persona.pair_id,
                                "persona_axis": persona.counterfactual_axis,
                                "persona_bio": persona.biography,
                                "ethnicity": persona.ethnicity,
                                "education": persona.education,
                                "context_tier": tier.name,
                                "context_max_tokens": tier.max_tokens,
                                "context_start_turn": start,
                                "selected_turn_count": len(selected_turns),
                                "model_input_token_count": model_input_counts[persona.persona_id],
                                "provider_prompt_token_count": provider_prompt_tokens,
                                "conversation_sha256": conversation_sha256,
                                "model": response.model,
                                "raw_response": response.content,
                                "response_sha256": hashlib.sha256(
                                    response.content.encode("utf-8")
                                ).hexdigest(),
                                "finish_reason": response.finish_reason,
                                "usage": dict(response.usage),
                                "prediction": prediction,
                                "correct": correct,
                                "response_state": state,
                            }
                        )
                        _write_jsonl(predictions_path, rows)
        executor.shutdown(wait=True)
        executor = None
        subgroup_fields = ("persona_id", "ethnicity", "education", "context_tier", "interference_tier")
        subgroup_summaries = {
            field: {
                str(value): _aggregate([row for row in rows if row[field] == value])
                for value in sorted({row[field] for row in rows}, key=lambda item: str(item))
            }
            for field in subgroup_fields
        }
        intersection_summaries = {
            f"{ethnicity} | {education}": _aggregate(
                [
                    row for row in rows
                    if row["ethnicity"] == ethnicity and row["education"] == education
                ]
            )
            for ethnicity, education in sorted(
                {(row["ethnicity"], row["education"]) for row in rows},
                key=lambda item: (str(item[0]), str(item[1])),
            )
        }
        length_distractor = {
            tier: {
                interference: _aggregate(
                    [
                        row for row in rows
                        if row["context_tier"] == tier
                        and row["interference_tier"] == interference
                    ]
                )
                for interference in sorted({str(row["interference_tier"]) for row in rows})
            }
            for tier in sorted({str(row["context_tier"]) for row in rows})
        }
        report = {
            "qwen_overall": _aggregate(rows),
            "paired_persona_deltas": _paired_deltas(rows, personas),
            "subgroup_summaries": subgroup_summaries,
            "intersection_summaries": intersection_summaries,
            "length_distractor_summaries": length_distractor,
            "controls": _control_summary(
                control_predictions,
                selected_checkpoint_ids,
                recall_config.context_tiers,
            ),
            "interpretation": {
                "persona_effect": "Matched arms change only the prepended identity biography.",
                "persona_axes_evaluated": ["ethnicity", "education"],
                "other_declared_persona_fields_evaluated": False,
                "statistical_scope": (
                    "Exploratory pipeline smoke only; independent history-query counts, not "
                    "condition copies, determine statistical power."
                ),
                "length_distractor_effect": "Context-token and distractor tiers are summarized separately from persona identity.",
                "protected_attributes_determine_preferences": False,
                "structured_memory": "full_structured_memory is an oracle control, not a Qwen arm.",
            },
        }
        _write_json(report_path, report)
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "model_identities": sorted({str(row["model"]) for row in rows}),
                "artifact_sha256": {
                    predictions_path.name: _sha256(predictions_path),
                    report_path.name: _sha256(report_path),
                },
            }
        )
        _write_json(manifest_path, manifest)
        return report
    except Exception as error:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if not predictions_path.exists():
            _write_jsonl(predictions_path, rows)
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)},
                "artifact_sha256": {
                    path.name: _sha256(path)
                    for path in (predictions_path, report_path)
                    if path.exists()
                },
            }
        )
        _write_json(manifest_path, manifest)
        raise


def load_recall_config(path: Path) -> RecallConfig:
    """Load strict evaluation config and resolve the Qwen endpoint identity from env vars."""
    try:
        root = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load recall config {path}: {error}") from error
    if not isinstance(root, Mapping) or "evaluation" not in root:
        raise ValueError("config must contain an evaluation object")
    raw = _strict_object(
        root["evaluation"],
        {
            "endpoint_env", "api_key_env", "model_env", "timeout_seconds", "seed",
            "max_tokens", "history_limit", "max_concurrent_requests",
            "interference_tiers", "query_suffixes", "context_tiers",
            "prompt_schema_version",
        },
        "evaluation",
    )
    endpoint_env = _nonempty(raw["endpoint_env"], "evaluation.endpoint_env")
    api_key_env = _nonempty(raw["api_key_env"], "evaluation.api_key_env")
    model_env = _nonempty(raw["model_env"], "evaluation.model_env")
    missing_env = [name for name in (endpoint_env, api_key_env, model_env) if not os.environ.get(name)]
    if missing_env:
        raise ValueError(f"required evaluation environment variables are unset: {missing_env}")
    tiers = raw["context_tiers"]
    if not isinstance(tiers, list):
        raise ValueError("evaluation.context_tiers must be an array")
    parsed_tiers = []
    for index, tier in enumerate(tiers):
        value = _strict_object(tier, {"name", "max_tokens"}, f"context_tiers[{index}]")
        parsed_tiers.append(ContextTier(value["name"], value["max_tokens"]))
    interference_tiers = raw["interference_tiers"]
    if not isinstance(interference_tiers, list):
        raise ValueError("evaluation.interference_tiers must be an array")
    query_suffixes = raw["query_suffixes"]
    if not isinstance(query_suffixes, list):
        raise ValueError("evaluation.query_suffixes must be an array")
    return RecallConfig(
        endpoint=os.environ[endpoint_env],
        api_key=os.environ[api_key_env],
        model=os.environ[model_env],
        timeout_seconds=raw["timeout_seconds"],
        seed=raw["seed"],
        max_tokens=raw["max_tokens"],
        history_limit=raw["history_limit"],
        max_concurrent_requests=raw["max_concurrent_requests"],
        interference_tiers=tuple(interference_tiers),
        query_suffixes=tuple(query_suffixes),
        context_tiers=tuple(parsed_tiers),
        prompt_schema_version=raw["prompt_schema_version"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured Qwen persona recall and deterministic control evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    args = parser.parse_args(argv)
    recall_config = load_recall_config(args.config)
    benchmark_config = load_benchmark_config(args.benchmark_config)
    evaluate_persona_recall(
        args.dataset,
        args.output_dir,
        benchmark_config,
        HuggingFaceTokenizer(benchmark_config.tokenizer),
        recall_config,
        OpenAICompletionClient(
            recall_config.endpoint,
            recall_config.api_key,
            recall_config.timeout_seconds,
            enable_thinking=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
