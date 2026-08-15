"""Run the five-arm Qwen persona-interference generation benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import mean
import subprocess
import time
from typing import Any, Mapping, Protocol, Sequence

from experiments.answer_eval import exact_match, f1_score
from experiments.interleaved_conversation import build_interleaved_schedule
from experiments.interleaved_memory_qwen35_eval import (
    _model_context_limit,
    _model_file_hashes,
)
from experiments.persona_interference_schedule import (
    SCHEDULE_VERSION,
    ScheduleConfig,
    _authenticate_dataset,
    _read_jsonl_bytes,
    build_evaluation_schedule,
)
from experiments.preference_stream_injection import PreferenceStreamInjectionClient


EVALUATOR_VERSION = "persona_end_to_end_qwen.v1"
_EXPECTED_ARMS = (
    ("sliding_context_4096", "sliding_context", 4096),
    ("sliding_context_16384", "sliding_context", 16384),
    ("structured_memory_4096", "structured_memory", 4096),
    ("structured_memory_16384", "structured_memory", 16384),
    ("full_qwen_context", "full_qwen_context", None),
)
_ANSWER_PREFIX = re.compile(r"^\s*(?:final\s+)?answer\s*:\s*", re.IGNORECASE)


class ChatTokenizer(Protocol):
    """Tokenizer operations required for exact Qwen prompt accounting."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[Any]:
        """Encode text without adding another special-token layer."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        """Apply the model's non-thinking generation template."""


class InjectionClient(Protocol):
    """Query-blind preference-source derivation boundary."""

    def derive(self, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Return source event IDs derived only from a causal event prefix."""


@dataclass(frozen=True)
class ArmConfig:
    """One allowed context-construction arm."""

    name: str
    kind: str
    prompt_token_cap: int | None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved benchmark runtime and experimental controls."""

    dataset_dir: Path
    output_dir: Path
    model_path: Path
    model_id: str
    device: str
    scallop_endpoint: str
    scallop_timeout_seconds: float
    schedule: ScheduleConfig
    arms: tuple[ArmConfig, ...]
    prompt_instruction: str
    max_new_tokens: int
    bootstrap_samples: int
    bootstrap_seed: int
    dtype: str
    attention_implementation: str
    local_files_only: bool
    use_kernels: bool
    comparisons: tuple[tuple[str, str], ...]


class _ScheduleTokenizerAdapter:
    """Expose model tokenization and provenance to the authenticated scheduler."""

    def __init__(self, tokenizer: ChatTokenizer, model_id: str, model_path: Path) -> None:
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._model_path = model_path

    def encode(self, text: str) -> list[Any]:
        """Encode schedule-visible text without special tokens."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def metadata(self) -> Mapping[str, Any]:
        """Return resolved tokenizer identity without embedding a runtime path."""
        return {
            "model_id": self._model_id,
            "resolved_revision": (
                self._model_path.name
                if self._model_path.parent.name == "snapshots"
                else None
            ),
            "implementation": type(self._tokenizer).__name__,
            "add_special_tokens": False,
        }


def _required_env(environ: Mapping[str, str], name: str) -> str:
    """Resolve one required non-empty environment value."""
    value = environ.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"required environment variable {name} is not set")
    return value


def _positive_int(value: Any, name: str) -> int:
    """Validate one positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    """Validate one non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def load_benchmark_config(
    path: Path, *, environ: Mapping[str, str] | None = None
) -> BenchmarkConfig:
    """Load strict controls and resolve every runtime path or endpoint from env."""
    environ = os.environ if environ is None else environ
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load benchmark config {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark config must be a JSON object")
    runtime = payload.get("runtime")
    schedule_payload = payload.get("schedule")
    generation = payload.get("generation")
    arms_payload = payload.get("arms")
    scallop = payload.get("scallop")
    analysis = payload.get("analysis")
    if not all(
        isinstance(value, Mapping)
        for value in (runtime, schedule_payload, generation, scallop, analysis)
    ) or not isinstance(arms_payload, list):
        raise ValueError("benchmark config sections are incomplete")

    arms = tuple(
        ArmConfig(
            name=_nonempty_string(row.get("name"), "arms.name"),
            kind=_nonempty_string(row.get("kind"), "arms.kind"),
            prompt_token_cap=row.get("prompt_token_cap"),
        )
        for row in arms_payload
        if isinstance(row, Mapping)
    )
    if len(arms) != len(arms_payload):
        raise ValueError("every arms entry must be an object")
    for arm in arms:
        if arm.prompt_token_cap is not None:
            _positive_int(arm.prompt_token_cap, f"{arm.name}.prompt_token_cap")
    if tuple((arm.name, arm.kind, arm.prompt_token_cap) for arm in arms) != _EXPECTED_ARMS:
        raise ValueError("arms must declare exactly the two matched caps and full Qwen context")

    suffixes = schedule_payload.get("query_suffixes")
    distances = schedule_payload.get("token_distance_thresholds")
    if not isinstance(suffixes, list) or not all(
        isinstance(value, str) and value for value in suffixes
    ):
        raise ValueError("schedule.query_suffixes must be non-empty strings")
    if not isinstance(distances, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in distances
    ):
        raise ValueError("schedule.token_distance_thresholds must be integers")
    schedule = ScheduleConfig(
        source_split=_nonempty_string(schedule_payload.get("source_split"), "schedule.source_split"),
        source_profile=_nonempty_string(schedule_payload.get("source_profile"), "schedule.source_profile"),
        source_manifest_sha256=_nonempty_string(
            schedule_payload.get("source_manifest_sha256"),
            "schedule.source_manifest_sha256",
        ),
        seed=_positive_int(schedule_payload.get("seed"), "schedule.seed"),
        concurrent_accounts=_positive_int(
            schedule_payload.get("concurrent_accounts"),
            "schedule.concurrent_accounts",
        ),
        min_segment_events=_positive_int(
            schedule_payload.get("min_segment_events"),
            "schedule.min_segment_events",
        ),
        max_segment_events=_positive_int(
            schedule_payload.get("max_segment_events"),
            "schedule.max_segment_events",
        ),
        query_suffixes=tuple(suffixes),
        token_distance_thresholds=tuple(distances),
    )
    endpoint_env = _nonempty_string(scallop.get("endpoint_env"), "scallop.endpoint_env")
    timeout = scallop.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("scallop.timeout_seconds must be positive")
    comparisons_payload = analysis.get("paired_comparisons")
    if not isinstance(comparisons_payload, list) or not all(
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(name, str) and name for name in pair)
        for pair in comparisons_payload
    ):
        raise ValueError("analysis.paired_comparisons must contain arm-name pairs")
    arm_names = {arm.name for arm in arms}
    comparisons = tuple((pair[0], pair[1]) for pair in comparisons_payload)
    if any(left not in arm_names or right not in arm_names for left, right in comparisons):
        raise ValueError("paired comparison references an unknown arm")
    local_files_only = generation.get("local_files_only")
    use_kernels = generation.get("use_kernels")
    if not isinstance(local_files_only, bool) or not isinstance(use_kernels, bool):
        raise ValueError("generation local_files_only and use_kernels must be booleans")

    dataset_env = _nonempty_string(runtime.get("dataset_dir_env"), "runtime.dataset_dir_env")
    output_env = _nonempty_string(runtime.get("output_dir_env"), "runtime.output_dir_env")
    model_path_env = _nonempty_string(runtime.get("model_path_env"), "runtime.model_path_env")
    model_id_env = _nonempty_string(runtime.get("model_id_env"), "runtime.model_id_env")
    device_env = _nonempty_string(runtime.get("device_env"), "runtime.device_env")
    return BenchmarkConfig(
        dataset_dir=Path(_required_env(environ, dataset_env)),
        output_dir=Path(_required_env(environ, output_env)),
        model_path=Path(_required_env(environ, model_path_env)),
        model_id=_required_env(environ, model_id_env),
        device=_required_env(environ, device_env),
        scallop_endpoint=_required_env(environ, endpoint_env),
        scallop_timeout_seconds=float(timeout),
        schedule=schedule,
        arms=arms,
        prompt_instruction=_nonempty_string(
            payload.get("prompt_instruction"), "prompt_instruction"
        ),
        max_new_tokens=_positive_int(
            generation.get("max_new_tokens"), "generation.max_new_tokens"
        ),
        bootstrap_samples=_positive_int(
            analysis.get("bootstrap_samples"), "analysis.bootstrap_samples"
        ),
        bootstrap_seed=_positive_int(
            analysis.get("bootstrap_seed"), "analysis.bootstrap_seed"
        ),
        dtype=_nonempty_string(generation.get("dtype"), "generation.dtype"),
        attention_implementation=_nonempty_string(
            generation.get("attention_implementation"),
            "generation.attention_implementation",
        ),
        local_files_only=local_files_only,
        use_kernels=use_kernels,
        comparisons=comparisons,
    )


def _render_turns(turns: Sequence[Mapping[str, Any]], history_id: str) -> str:
    """Render natural turns with model-safe account-holder labels."""
    rendered = []
    for turn in turns:
        label = (
            "Account holder conversation"
            if str(turn["history_id"]) == history_id
            else "Unrelated person's conversation, not the account holder"
        )
        rendered.append(f"{label}:\n{turn['text']}")
    return "\n".join(rendered)


def _chat_prompt(
    context: str,
    query_text: str,
    prompt_instruction: str,
    tokenizer: ChatTokenizer,
) -> tuple[str, int]:
    """Apply and count the exact non-thinking model prompt."""
    content = (
        f"{prompt_instruction}\n\nConversation record:\n{context}"
        f"\n\nQuestion: {query_text}\nAnswer:"
    )
    prompt = str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))


def _fit_sliding_prompt(
    prefix: Sequence[Mapping[str, Any]],
    *,
    history_id: str,
    query_text: str,
    prompt_instruction: str,
    cap: int,
    tokenizer: ChatTokenizer,
) -> tuple[list[Mapping[str, Any]], str, int]:
    """Fit the exact maximal complete-turn suffix under one templated prompt cap."""
    _positive_int(cap, "sliding prompt cap")
    start = len(prefix)
    prompt, count = _chat_prompt("", query_text, prompt_instruction, tokenizer)
    if count > cap:
        raise ValueError(f"query and chat template require {count} tokens, exceeding cap {cap}")
    while start > 0:
        candidate = prefix[start - 1 :]
        candidate_prompt, candidate_count = _chat_prompt(
            _render_turns(candidate, history_id), query_text, prompt_instruction, tokenizer
        )
        if candidate_count > cap:
            break
        start -= 1
        prompt, count = candidate_prompt, candidate_count
    return list(prefix[start:]), prompt, count


def _fit_structured_prompt(
    prefix: Sequence[Mapping[str, Any]],
    *,
    source_event_ids: Sequence[str],
    history_id: str,
    query_text: str,
    prompt_instruction: str,
    cap: int,
    tokenizer: ChatTokenizer,
) -> tuple[list[Mapping[str, Any]], str, int]:
    """Fit causal natural source turns plus a maximal recent suffix, without IDs."""
    _positive_int(cap, "structured prompt cap")
    by_event_id = {
        str(turn["stream_event"]["event_id"]): turn for turn in prefix
    }
    source_ids = list(dict.fromkeys(str(identifier) for identifier in source_event_ids))
    if not source_ids:
        return _fit_sliding_prompt(
            prefix,
            history_id=history_id,
            query_text=query_text,
            prompt_instruction=prompt_instruction,
            cap=cap,
            tokenizer=tokenizer,
        )
    missing = [identifier for identifier in source_ids if identifier not in by_event_id]
    if missing:
        raise ValueError(f"structured source is absent from causal prefix: {missing}")
    source_turns = [by_event_id[identifier] for identifier in source_ids]
    if any(str(turn["history_id"]) != history_id for turn in source_turns):
        raise ValueError("structured source crosses histories")
    prefix_position = {str(turn["turn_id"]): index for index, turn in enumerate(prefix)}
    source_turns.sort(key=lambda turn: prefix_position[str(turn["turn_id"])])
    source_turn_ids = {str(turn["turn_id"]) for turn in source_turns}

    def context_for(suffix: Sequence[Mapping[str, Any]]) -> str:
        recent = [turn for turn in suffix if str(turn["turn_id"]) not in source_turn_ids]
        sections = []
        if source_turns:
            sections.append("Relevant account-holder source turns:\n" + _render_turns(source_turns, history_id))
        if recent:
            sections.append("Recent conversation suffix:\n" + _render_turns(recent, history_id))
        return "\n\n".join(sections)

    start = len(prefix)
    prompt, count = _chat_prompt(context_for(()), query_text, prompt_instruction, tokenizer)
    if count > cap:
        raise ValueError(
            f"structured sources, query, and chat template require {count} tokens, exceeding cap {cap}"
        )
    while start > 0:
        candidate = prefix[start - 1 :]
        candidate_prompt, candidate_count = _chat_prompt(
            context_for(candidate), query_text, prompt_instruction, tokenizer
        )
        if candidate_count > cap:
            break
        start -= 1
        prompt, count = candidate_prompt, candidate_count
    return list(prefix[start:]), prompt, count


def _canary_events() -> list[dict[str, Any]]:
    """Return fixed facts that exercise both Scallop preference rules."""
    def preference(
        event_id: str,
        fact_id: str,
        value: str,
        valid_from: str,
        authority: str,
        supersedes: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": event_id,
            "fact": {
                "fact_id": fact_id,
                "subject": "canary-subject",
                "predicate": "PREFERS",
                "object": value,
                "temporal": {"valid_from": valid_from},
                "qualifiers": {"scope": "default", "source_authority": authority},
            },
        }
        if supersedes is not None:
            event["supersedes"] = supersedes
        return event

    return [
        {
            "event_id": "canary-alias",
            "fact": {
                "fact_id": "canary-alias-fact",
                "subject": "canary-subject",
                "predicate": "SAME_ACCOUNT",
                "object": "canary-alias-object",
            },
        },
        preference("canary-old-change", "canary-old-change-fact", "tea", "2025-01-01", "direct"),
        preference(
            "canary-new-change",
            "canary-new-change-fact",
            "coffee",
            "2025-02-01",
            "direct",
            "canary-old-change-fact",
        ),
        preference("canary-old-conflict", "canary-old-conflict-fact", "aisle", "2025-03-01", "inferred"),
        preference(
            "canary-new-conflict",
            "canary-new-conflict-fact",
            "window",
            "2025-03-01",
            "direct",
            "canary-old-conflict-fact",
        ),
    ]


def _run_scallop_canary(client: InjectionClient) -> dict[str, Any]:
    """Require exact source semantics for both Scallop rules before evaluation."""
    result = client.derive(_canary_events())
    actual = {
        (str(injection.get("kind")), tuple(injection.get("source_event_ids", [])))
        for injection in result.get("injections", [])
        if isinstance(injection, Mapping)
    }
    expected = {
        (
            "preference_change",
            ("canary-old-change", "canary-new-change", "canary-alias"),
        ),
        (
            "preference_incongruity",
            ("canary-old-conflict", "canary-new-conflict", "canary-alias"),
        ),
    }
    if actual != expected:
        raise ValueError(f"Scallop semantic canary failed: expected {sorted(expected)}, got {sorted(actual)}")
    return result


def _query_family(query_id: str) -> str:
    """Map the two configured delayed query suffixes to report families."""
    for suffix, family in (
        ("-preference-change-delayed", "preference_change"),
        ("-preference-incongruity-delayed", "preference_incongruity"),
    ):
        if query_id.endswith(suffix):
            return family
    raise ValueError(f"unsupported persona query family: {query_id}")


def _condition_name(row: Mapping[str, Any]) -> str:
    """Return one stable phase/distance condition label."""
    if row.get("phase") != "token_distance":
        return str(row["phase"])
    return f"token_distance_{int(row['requested_token_distance'])}"


def _arm_tuple(arm: ArmConfig | Sequence[Any]) -> tuple[str, str, int | None]:
    """Normalize configured and fixture arm values."""
    if isinstance(arm, ArmConfig):
        return arm.name, arm.kind, arm.prompt_token_cap
    if len(arm) != 3:
        raise ValueError(f"invalid arm tuple: {arm}")
    return str(arm[0]), str(arm[1]), arm[2]


def _build_arm_specs(
    conditions: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[ArmConfig | Sequence[Any]],
    injections: Mapping[str, Mapping[str, Any]],
    prompt_instruction: str,
    tokenizer: ChatTokenizer,
) -> list[dict[str, Any]]:
    """Build every model prompt from the same scheduler conditions and prompt."""
    specs = []
    for condition in conditions:
        checkpoint = int(condition["checkpoint_turn_index"])
        if checkpoint < 1 or checkpoint > len(turns):
            raise ValueError(f"condition {condition.get('evaluation_input_id')} has invalid checkpoint")
        prefix = turns[:checkpoint]
        history_id = str(condition["history_id"])
        query_text = str(condition["query_text"])
        for arm in arms:
            arm_name, kind, cap = _arm_tuple(arm)
            source_count = 0
            if kind == "sliding_context":
                if cap is None:
                    raise ValueError(f"sliding arm {arm_name} lacks a cap")
                selected, prompt, token_count = _fit_sliding_prompt(
                    prefix,
                    history_id=history_id,
                    query_text=query_text,
                    prompt_instruction=prompt_instruction,
                    cap=int(cap),
                    tokenizer=tokenizer,
                )
            elif kind == "structured_memory":
                if cap is None:
                    raise ValueError(f"structured arm {arm_name} lacks a cap")
                injection = injections.get(str(condition["evaluation_input_id"]), {})
                source_ids = injection.get("source_event_ids", ())
                source_count = len(set(source_ids))
                selected, prompt, token_count = _fit_structured_prompt(
                    prefix,
                    source_event_ids=source_ids,
                    history_id=history_id,
                    query_text=query_text,
                    prompt_instruction=prompt_instruction,
                    cap=int(cap),
                    tokenizer=tokenizer,
                )
            elif kind == "full_qwen_context":
                if cap is not None:
                    raise ValueError("full_qwen_context must not declare a truncation cap")
                selected = list(prefix)
                prompt, token_count = _chat_prompt(
                    _render_turns(prefix, history_id),
                    query_text,
                    prompt_instruction,
                    tokenizer,
                )
            else:
                raise ValueError(f"unsupported arm kind: {kind}")
            unrelated_gold_occurrences = sum(
                str(turn["text"]).casefold().count(str(condition["gold"]).casefold())
                for turn in selected
                if str(turn["history_id"]) != history_id
            )
            specs.append(
                {
                    "evaluation_input_id": str(condition["evaluation_input_id"]),
                    "history_id": history_id,
                    "query_id": str(condition["query_id"]),
                    "query_family": _query_family(str(condition["query_id"])),
                    "condition": _condition_name(condition),
                    "phase": str(condition["phase"]),
                    "requested_token_distance": condition.get("requested_token_distance"),
                    "gold": str(condition["gold"]),
                    "arm": arm_name,
                    "arm_kind": kind,
                    "prompt_token_cap": cap,
                    "input_token_count": token_count,
                    "selected_turn_count": len(selected),
                    "structured_source_turn_count": source_count,
                    "unrelated_gold_occurrence_count": unrelated_gold_occurrences,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt": prompt,
                }
            )
    expected = len(conditions) * len(arms)
    keys = {(row["evaluation_input_id"], row["arm"]) for row in specs}
    if len(specs) != expected or len(keys) != expected:
        raise ValueError("arm specifications do not cover every condition exactly once")
    return specs


def _assert_full_context_fits(
    *, input_tokens: int, max_new_tokens: int, model_capacity: int
) -> None:
    """Fail rather than truncate a full-context input beyond model capacity."""
    required = input_tokens + max_new_tokens
    if required > model_capacity:
        raise ValueError(
            f"full_qwen_context requires {required} positions, exceeding model capacity {model_capacity}"
        )


def _score_short_answer(answer: str, gold: str) -> dict[str, Any]:
    """Score the first non-empty completion line as a short free-form answer."""
    lines = [line.strip() for line in str(answer).splitlines() if line.strip()]
    short_answer = _ANSWER_PREFIX.sub("", lines[0]).strip() if lines else ""
    return {
        "short_answer": short_answer,
        "exact_match": exact_match(short_answer, gold),
        "f1": f1_score(short_answer, gold),
    }


def _percentile_interval(values: Sequence[float]) -> list[float]:
    """Return a deterministic two-sided percentile interval."""
    ordered = sorted(values)
    lower = max(0, math.floor(0.025 * len(ordered)))
    upper = min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)
    return [ordered[lower], ordered[upper]]


def _group_metrics(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    """Aggregate EM and F1 over explicit report dimensions."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[field] for field in fields), []).append(row)
    return [
        {
            **dict(zip(fields, key)),
            "row_count": len(group),
            "history_cluster_count": len({str(row["history_id"]) for row in group}),
            "exact_match": mean(float(row["exact_match"]) for row in group),
            "f1": mean(float(row["f1"]) for row in group),
        }
        for key, group in sorted(groups.items())
    ]


def _paired_delta(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute paired row deltas with history-cluster resampling."""
    by_arm = {
        arm: {
            str(row["evaluation_input_id"]): row
            for row in rows
            if str(row["arm"]) == arm
        }
        for arm in (left, right)
    }
    if set(by_arm[left]) != set(by_arm[right]) or not by_arm[left]:
        raise ValueError(f"paired comparison {left} vs {right} lacks identical conditions")
    pairs = [(by_arm[left][key], by_arm[right][key]) for key in sorted(by_arm[left])]
    if any(str(a["history_id"]) != str(b["history_id"]) for a, b in pairs):
        raise ValueError(f"paired comparison {left} vs {right} crosses histories")
    by_history: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for pair in pairs:
        by_history.setdefault(str(pair[0]["history_id"]), []).append(pair)
    histories = sorted(by_history)
    rng = random.Random(f"{seed}:{left}:{right}")
    bootstrap_em = []
    bootstrap_f1 = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(histories) for _ in histories]
        sampled_pairs = [pair for history in sampled for pair in by_history[history]]
        bootstrap_em.append(
            mean(float(a["exact_match"]) - float(b["exact_match"]) for a, b in sampled_pairs)
        )
        bootstrap_f1.append(mean(float(a["f1"]) - float(b["f1"]) for a, b in sampled_pairs))
    return {
        "left_arm": left,
        "right_arm": right,
        "paired_row_count": len(pairs),
        "history_cluster_count": len(histories),
        "exact_match_delta": mean(
            float(a["exact_match"]) - float(b["exact_match"]) for a, b in pairs
        ),
        "exact_match_delta_ci_95": _percentile_interval(bootstrap_em),
        "f1_delta": mean(float(a["f1"]) - float(b["f1"]) for a, b in pairs),
        "f1_delta_ci_95": _percentile_interval(bootstrap_f1),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def _stratified_paired_deltas(
    rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[tuple[str, str]],
    fields: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compute paired clustered deltas independently within each requested stratum."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[field] for field in fields), []).append(row)
    results = []
    for values, group in sorted(groups.items()):
        for left, right in comparisons:
            delta = _paired_delta(
                group,
                left,
                right,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            results.append({**dict(zip(fields, values)), **delta})
    return results


def _gold_occurrence_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate observed target-gold occurrences in unrelated model-visible turns."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["arm"]), str(row["condition"])), []).append(row)
    diagnostics = []
    for (arm, condition), group in sorted(groups.items()):
        counts = [int(row.get("unrelated_gold_occurrence_count", 0)) for row in group]
        diagnostics.append(
            {
                "arm": arm,
                "condition": condition,
                "row_count": len(group),
                "total_occurrences": sum(counts),
                "rows_with_occurrences": sum(count > 0 for count in counts),
                "mean_occurrences": mean(counts),
                "maximum_occurrences": max(counts),
                "contexts_filtered": False,
            }
        )
    return diagnostics


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparisons: Sequence[tuple[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Aggregate by requested dimensions and report paired arm deltas."""
    if not rows:
        raise ValueError("cannot aggregate an empty benchmark")
    _positive_int(bootstrap_samples, "bootstrap_samples")
    return {
        "by_arm": _group_metrics(rows, ("arm",)),
        "by_arm_condition": _group_metrics(rows, ("arm", "condition")),
        "by_arm_query_family": _group_metrics(rows, ("arm", "query_family")),
        "by_arm_condition_query_family": _group_metrics(
            rows, ("arm", "condition", "query_family")
        ),
        "paired_deltas": [
            _paired_delta(
                rows,
                left,
                right,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            for left, right in comparisons
        ],
        "paired_deltas_by_condition": _stratified_paired_deltas(
            rows,
            comparisons,
            ("condition",),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "paired_deltas_by_query_family": _stratified_paired_deltas(
            rows,
            comparisons,
            ("query_family",),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "paired_deltas_by_condition_query_family": _stratified_paired_deltas(
            rows,
            comparisons,
            ("condition", "query_family"),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "unrelated_gold_occurrences_by_arm_condition": _gold_occurrence_diagnostics(rows),
    }


def _sha256(path: Path) -> str:
    """Return one artifact SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    """Hash one JSON-compatible value canonically."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_resumable_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load JSONL, truncating only one malformed unterminated final record."""
    if not path.exists():
        return [], {"recovered": False}
    payload = path.read_bytes()
    records = payload.splitlines(keepends=True)
    rows = []
    for index, record in enumerate(records):
        terminated = record.endswith((b"\n", b"\r"))
        body = record.rstrip(b"\r\n")
        try:
            row = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            is_final = index == len(records) - 1
            if not is_final or terminated:
                raise ValueError(
                    f"malformed interior or terminated JSONL record {index + 1}: {error}"
                ) from error
            retained_bytes = len(payload) - len(record)
            removed_sha256 = hashlib.sha256(record).hexdigest()
            with path.open("r+b") as handle:
                handle.truncate(retained_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            return rows, {
                "recovered": True,
                "record_number": index + 1,
                "removed_byte_count": len(record),
                "removed_sha256": removed_sha256,
                "retained_byte_count": retained_bytes,
            }
        if not isinstance(row, dict):
            raise ValueError(f"JSONL record {index + 1} must be an object")
        rows.append(row)
    return rows, {"recovered": False}


def _git_provenance(
    path: Path, *, excluded_untracked_dir: Path | None = None
) -> dict[str, Any]:
    """Hash HEAD plus tracked and untracked source changes for immutable resume state."""
    def run(*args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=path,
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ValueError(f"cannot resolve git provenance with {' '.join(args)}: {error}") from error

    root = Path(run("rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    head = run("rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = run("diff", "--binary", "HEAD", "--")
    untracked_names = [
        name.decode("utf-8")
        for name in run("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if name
    ]
    excluded = excluded_untracked_dir.resolve() if excluded_untracked_dir is not None else None
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    included_untracked = []
    for name in sorted(untracked_names):
        source = (root / name).resolve()
        if excluded is not None and (source == excluded or excluded in source.parents):
            continue
        if not source.is_file():
            continue
        included_untracked.append(name)
        digest.update(b"\0untracked\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
    return {
        "git_head": head,
        "dirty": bool(tracked_diff or included_untracked),
        "dirty_diff_sha256": digest.hexdigest(),
        "untracked_file_count": len(included_untracked),
    }


def _write_json(path: Path, value: Any) -> None:
    """Write and fsync one stable JSON artifact."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_source_and_rebuild(
    config: BenchmarkConfig,
    tokenizer: ChatTokenizer,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Authenticate source artifacts and independently verify the rebuilt schedule."""
    adapter = _ScheduleTokenizerAdapter(tokenizer, config.model_id, config.model_path)
    scheduled = build_evaluation_schedule(config.dataset_dir, adapter, config.schedule)
    provenance, artifacts = _authenticate_dataset(
        config.dataset_dir, config.schedule.source_manifest_sha256
    )
    events = [
        row
        for row in _read_jsonl_bytes(
            artifacts["events.jsonl"], config.dataset_dir / "events.jsonl"
        )
        if row.get("split") == config.schedule.source_split
        and row.get("hardness_profile") == config.schedule.source_profile
    ]
    rebuilt = build_interleaved_schedule(
        events,
        tokenizer=adapter,
        seed=config.schedule.seed,
        concurrent_accounts=config.schedule.concurrent_accounts,
        min_segment_events=config.schedule.min_segment_events,
        max_segment_events=config.schedule.max_segment_events,
    )
    turns = list(rebuilt["turns"])
    safe_identity = [
        (str(turn["turn_id"]), str(turn["history_id"]), str(turn["text"]))
        for turn in turns
    ]
    scheduled_identity = [
        (str(turn["turn_id"]), str(turn["history_id"]), str(turn["text"]))
        for turn in scheduled["turns"]
    ]
    if safe_identity != scheduled_identity:
        raise ValueError("independently rebuilt schedule differs from authenticated scheduler")
    if len(scheduled["inputs"]) != 120:
        raise ValueError(f"persona benchmark requires exactly 120 conditions, got {len(scheduled['inputs'])}")
    scheduled["dataset"] = provenance
    return scheduled, turns, events


def _derive_condition_sources(
    conditions: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]],
    client: InjectionClient,
) -> dict[str, dict[str, Any]]:
    """Derive query-blind relations from each target's causal same-history prefix."""
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    by_condition = {}
    for condition in conditions:
        history_id = str(condition["history_id"])
        checkpoint = int(condition["checkpoint_turn_index"])
        cache_key = (history_id, checkpoint)
        if cache_key not in cache:
            events = [
                dict(turn["stream_event"])
                for turn in turns[:checkpoint]
                if str(turn["history_id"]) == history_id
            ]
            result = client.derive(events)
            relations: dict[str, list[str]] = {}
            for injection in result["injections"]:
                kind = str(injection["kind"])
                relations[kind] = list(
                    dict.fromkeys(
                        [
                            *relations.get(kind, []),
                            *(
                                str(identifier)
                                for identifier in injection["source_event_ids"]
                            ),
                        ]
                    )
                )
            source_ids = list(
                dict.fromkeys(
                    identifier
                    for kind in sorted(relations)
                    for identifier in relations[kind]
                )
            )
            cache[cache_key] = {
                "relations": relations,
                "source_event_ids": source_ids,
            }
        by_condition[str(condition["evaluation_input_id"])] = dict(cache[cache_key])
    return by_condition


def _relation_coverage_matrix(
    conditions: Sequence[Mapping[str, Any]],
    injections: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Record relation coverage and fail delayed probes missing their expected rule."""
    expected_by_family = {
        "preference_change": "preference_change",
        "preference_incongruity": "preference_incongruity",
    }
    matrix = []
    for condition in conditions:
        evaluation_input_id = str(condition["evaluation_input_id"])
        query_family = _query_family(str(condition["query_id"]))
        expected_relation = expected_by_family[query_family]
        injection = injections.get(evaluation_input_id)
        if not isinstance(injection, Mapping):
            raise ValueError(f"condition {evaluation_input_id} lacks Scallop preflight data")
        relations = injection.get("relations")
        source_ids = injection.get("source_event_ids")
        if not isinstance(relations, Mapping) or not isinstance(source_ids, list):
            raise ValueError(f"condition {evaluation_input_id} has malformed Scallop preflight data")
        phase = str(condition["phase"])
        expected_present = bool(relations.get(expected_relation))
        row = {
            "evaluation_input_id": evaluation_input_id,
            "condition": _condition_name(condition),
            "phase": phase,
            "query_family": query_family,
            "expected_relation": expected_relation,
            "expected_relation_present": expected_present,
            "zero_sources_allowed": phase in {"pre_update", "post_update"},
            "source_event_count": len(source_ids),
            "relation_source_counts": {
                str(kind): len(identifiers)
                for kind, identifiers in sorted(relations.items())
            },
        }
        if phase == "delayed_probe" and not expected_present:
            raise ValueError(
                f"delayed_probe {evaluation_input_id} is missing expected Scallop relation {expected_relation}"
            )
        matrix.append(row)
    return matrix


def _resume_manifest_fields(
    config: BenchmarkConfig,
    scheduled: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    config_sha256: str,
    *,
    scallop_identity: Mapping[str, Any],
    injections: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    relation_coverage: Sequence[Mapping[str, Any]],
    evaluator_script_sha256: str,
    git_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return immutable generation provenance checked on every resume."""
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "schedule_version": SCHEDULE_VERSION,
        "config_sha256": config_sha256,
        "dataset": scheduled["dataset"],
        "tokenizer": scheduled["tokenizer"],
        "model": dict(model_metadata),
        "scallop": dict(scallop_identity),
        "injection_map_sha256": _stable_hash(injections),
        "ordered_spec_sha256": _stable_hash(list(specs)),
        "evaluator_script_sha256": evaluator_script_sha256,
        "git": dict(git_provenance),
        "relation_coverage_matrix": list(relation_coverage),
        "arms": [asdict(arm) for arm in config.arms],
        "prompt_instruction_sha256": hashlib.sha256(
            config.prompt_instruction.encode("utf-8")
        ).hexdigest(),
        "decoding": {
            "do_sample": False,
            "batch_size": 1,
            "max_new_tokens": config.max_new_tokens,
            "enable_thinking": False,
        },
        "condition_count": len(scheduled["inputs"]),
        "generation_count": len(scheduled["inputs"]) * len(config.arms),
        "schedule_sha256": _stable_hash(
            {
                "turns": scheduled["turns"],
                "inputs": scheduled["inputs"],
                "metrics": scheduled["schedule_metrics"],
            }
        ),
    }


def _validate_resume_manifest(
    path: Path, immutable: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Validate every immutable provenance field before any generation row is read."""
    if not path.exists():
        return None
    try:
        old_manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read resume manifest {path}: {error}") from error
    if not isinstance(old_manifest, Mapping):
        raise ValueError("resume manifest must be a JSON object")
    for field, expected in immutable.items():
        if old_manifest.get(field) != expected:
            raise ValueError(f"cannot resume because manifest field {field} differs")
    return old_manifest


def run_benchmark(config: BenchmarkConfig, *, config_sha256: str) -> dict[str, Any]:
    """Run resumable local Qwen generation and write authenticated benchmark artifacts."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    evaluator_script_sha256 = _sha256(Path(__file__).resolve())
    git_provenance = _git_provenance(
        Path(__file__).resolve().parent,
        excluded_untracked_dir=config.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, local_files_only=config.local_files_only
    )
    scheduled, turns, _ = _load_source_and_rebuild(config, tokenizer)
    client = PreferenceStreamInjectionClient(
        config.scallop_endpoint, config.scallop_timeout_seconds
    )
    canary = _run_scallop_canary(client)
    injections = _derive_condition_sources(scheduled["inputs"], turns, client)
    relation_coverage = _relation_coverage_matrix(scheduled["inputs"], injections)
    specs = _build_arm_specs(
        scheduled["inputs"],
        turns,
        arms=config.arms,
        injections=injections,
        prompt_instruction=config.prompt_instruction,
        tokenizer=tokenizer,
    )

    torch_dtype = getattr(torch, config.dtype, None)
    if torch_dtype is None:
        raise ValueError(f"torch has no configured dtype {config.dtype!r}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=torch_dtype,
        attn_implementation=config.attention_implementation,
        local_files_only=config.local_files_only,
        use_kernels=config.use_kernels,
    ).to(config.device).eval()
    capacity = _model_context_limit(model.config)
    for spec in specs:
        required = int(spec["input_token_count"]) + config.max_new_tokens
        if spec["arm_kind"] == "full_qwen_context":
            _assert_full_context_fits(
                input_tokens=int(spec["input_token_count"]),
                max_new_tokens=config.max_new_tokens,
                model_capacity=capacity,
            )
        elif required > capacity:
            raise ValueError(
                f"{spec['arm']} requires {required} positions, exceeding model capacity {capacity}"
            )
    model_metadata = {
        "model_id": config.model_id,
        "resolved_revision": (
            config.model_path.name
            if config.model_path.parent.name == "snapshots"
            else None
        ),
        "file_sha256": _model_file_hashes(config.model_path),
        "architecture": type(model).__name__,
        "model_type": str(model.config.model_type),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": config.dtype,
        "attention_implementation": config.attention_implementation,
        "device": config.device,
        "context_capacity": capacity,
    }
    immutable = _resume_manifest_fields(
        config,
        scheduled,
        model_metadata,
        config_sha256,
        scallop_identity={
            "engine": canary["engine"],
            "version": canary["scallopy_version"],
            "rule_version": canary["rule_version"],
        },
        injections=injections,
        specs=specs,
        relation_coverage=relation_coverage,
        evaluator_script_sha256=evaluator_script_sha256,
        git_provenance=git_provenance,
    )
    generation_manifest_path = config.output_dir / "generation_manifest.json"
    generations_path = config.output_dir / "generations.jsonl"
    manifest_path = config.output_dir / "manifest.json"
    if generations_path.exists() and not generation_manifest_path.exists():
        raise ValueError("cannot resume generations without generation_manifest.json")
    old_manifest = _validate_resume_manifest(generation_manifest_path, immutable)
    generation_manifest = {
        **immutable,
        "status": "running",
        "scallop_canary": {
            "engine": canary["engine"],
            "version": canary["scallopy_version"],
            "rule_version": canary["rule_version"],
            "semantic_relations": sorted(
                injection["kind"] for injection in canary["injections"]
            ),
        },
    }
    prior_recoveries = (
        old_manifest.get("resume_recoveries", []) if old_manifest is not None else []
    )
    if not isinstance(prior_recoveries, list):
        raise ValueError("resume manifest resume_recoveries must be an array")
    recovery_history = list(prior_recoveries)
    if recovery_history:
        generation_manifest["resume_recoveries"] = recovery_history
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    resumed_rows, resume_recovery = _load_resumable_jsonl(generations_path)
    if resume_recovery["recovered"]:
        recovery_history.append(resume_recovery)
        generation_manifest["resume_recoveries"] = recovery_history
        _write_json(generation_manifest_path, generation_manifest)
    if resumed_rows:
        for row in resumed_rows:
            key = (str(row["evaluation_input_id"]), str(row["arm"]))
            if key in completed:
                raise ValueError(f"duplicate resumed generation {key}")
            completed[key] = row
    specs_by_key = {
        (str(spec["evaluation_input_id"]), str(spec["arm"])): spec for spec in specs
    }
    for key, row in completed.items():
        spec = specs_by_key.get(key)
        if spec is None:
            raise ValueError(f"resumed generation {key} is outside this benchmark")
        for field, expected in spec.items():
            if field != "prompt" and row.get(field) != expected:
                raise ValueError(f"cannot resume {key}: field {field} differs")
    _write_json(generation_manifest_path, generation_manifest)
    try:
        with generations_path.open("a", encoding="utf-8") as handle:
            for spec in specs:
                key = (str(spec["evaluation_input_id"]), str(spec["arm"]))
                if key in completed:
                    continue
                inputs = tokenizer(
                    spec["prompt"], return_tensors="pt", add_special_tokens=False
                ).to(config.device)
                input_count = int(inputs.input_ids.shape[1])
                if input_count != int(spec["input_token_count"]):
                    raise ValueError(
                        f"exact token count changed for {key}: built {spec['input_token_count']}, model saw {input_count}"
                    )
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=False,
                    )
                if config.device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                generated = output[0, input_count:]
                row = {name: value for name, value in spec.items() if name != "prompt"}
                row.update(
                    {
                        "answer": tokenizer.decode(
                            generated, skip_special_tokens=True
                        ).strip(),
                        "generated_token_count": int(generated.shape[0]),
                        "hit_max_new_tokens": int(generated.shape[0])
                        == config.max_new_tokens,
                        "generation_seconds": elapsed,
                    }
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed[key] = row
        ordered_generations = [
            completed[(str(spec["evaluation_input_id"]), str(spec["arm"]))]
            for spec in specs
        ]
        predictions = []
        for row in ordered_generations:
            scored = dict(row)
            scored.update(_score_short_answer(str(row["answer"]), str(row["gold"])))
            predictions.append(scored)
        predictions_path = config.output_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        metrics = {
            "evaluator_version": EVALUATOR_VERSION,
            "condition_count": len(scheduled["inputs"]),
            "generation_count": len(predictions),
            "aggregates": _aggregate_rows(
                predictions,
                comparisons=config.comparisons,
                bootstrap_samples=config.bootstrap_samples,
                seed=config.bootstrap_seed,
            ),
        }
        metrics_path = config.output_dir / "metrics.json"
        _write_json(metrics_path, metrics)
        generation_manifest.update(
            {
                "status": "completed",
                "artifact_sha256": {
                    generations_path.name: _sha256(generations_path),
                },
            }
        )
        _write_json(generation_manifest_path, generation_manifest)
        manifest = {
            **immutable,
            "status": "completed",
            "generation_manifest_sha256": _sha256(generation_manifest_path),
            "artifact_sha256": {
                generation_manifest_path.name: _sha256(generation_manifest_path),
                generations_path.name: _sha256(generations_path),
                predictions_path.name: _sha256(predictions_path),
                metrics_path.name: _sha256(metrics_path),
            },
        }
        _write_json(manifest_path, manifest)
        return metrics
    except Exception as error:
        generation_manifest.update(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        )
        _write_json(generation_manifest_path, generation_manifest)
        _write_json(manifest_path, generation_manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    """Run the configured benchmark without command-line runtime defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_benchmark_config(args.config)
    config_sha256 = _sha256(args.config)
    metrics = run_benchmark(config, config_sha256=config_sha256)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
