"""Evaluate direct local Qwen3.5 generation on v3 sliding-window checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import mean
import time
from typing import Any, Mapping, Sequence

from experiments._continual_memory_episodes import (
    QUERY_TEMPLATE,
    TURN_SEPARATOR,
    _visible_query,
)
from experiments.answer_eval import exact_match, f1_score
from experiments.interleaved_conversation import build_interleaved_schedule
from experiments.interleaved_memory_benchmark_v3 import BENCHMARK_VERSION, _read_jsonl
from experiments.synthetic_temporal_preferences import resolve_query


EVALUATOR_VERSION = "interleaved_qwen35_generation.v1"
_SYNTHETIC_ANSWER = re.compile(r"(?<![A-Za-z0-9-])value-[0-9]+-[a-z](?![A-Za-z0-9-])")


class _ScheduleTokenizer:
    """Minimal schedule tokenizer because saved checkpoints define all token boundaries."""

    def encode(self, text: str) -> list[str]:
        """Return deterministic placeholder tokens unused by schedule ordering."""
        return text.split()


def _sha256(path: Path) -> str:
    """Return one file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    """Write stable formatted JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_dataset_hashes(
    dataset_dir: Path, declared_hashes: Mapping[str, Any]
) -> None:
    """Fail if source data differs from the v3 manifest."""
    for name in ("events.jsonl", "queries.jsonl"):
        expected = str(declared_hashes.get(name, ""))
        if not expected:
            raise ValueError(f"v3 manifest does not declare {name} hash")
        actual = _sha256(dataset_dir / name)
        if actual != expected:
            raise ValueError(f"{name} hash mismatch: expected {expected}, found {actual}")


def _validate_declared_artifact(path: Path, declared_hashes: Mapping[str, Any]) -> str:
    """Validate one v3 artifact and return its authenticated digest."""
    expected = str(declared_hashes.get(path.name, ""))
    if not expected:
        raise ValueError(f"v3 manifest does not declare {path.name} hash")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{path.name} hash mismatch: expected {expected}, found {actual}")
    return actual


def _chat_prompt(
    turns: Sequence[Mapping[str, Any]], query_text: str, tokenizer: Any
) -> str:
    """Serialize the original gold-free prompt inside Qwen's non-thinking chat template."""
    context = TURN_SEPARATOR.join(str(turn["text"]) for turn in turns)
    question = QUERY_TEMPLATE.format(query_text=query_text)
    content = f"{context}{question}" if context else question
    return str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def _prompt_token_count(prompt: str, tokenizer: Any) -> int:
    """Count one already-templated model input without adding more special tokens."""
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def _fit_chat_sliding(
    prefix: Sequence[Mapping[str, Any]],
    query_text: str,
    max_tokens: int,
    tokenizer: Any,
    turn_token_counts: Mapping[str, int] | None = None,
) -> tuple[list[Mapping[str, Any]], str, int]:
    """Fit the maximal suffix without tokenizing arbitrarily overlong candidates."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    def candidate(start: int) -> tuple[str, int]:
        prompt = _chat_prompt(prefix[start:], query_text, tokenizer)
        return prompt, _prompt_token_count(prompt, tokenizer)

    empty_prompt, empty_count = candidate(len(prefix))
    if empty_count > max_tokens:
        raise ValueError(
            f"query and chat template require {empty_count} tokens, exceeding {max_tokens}"
        )
    start = len(prefix)
    estimated_count = empty_count
    while start > 0:
        turn = prefix[start - 1]
        turn_id = str(turn["turn_id"])
        standalone_count = (
            int(turn_token_counts[turn_id])
            if turn_token_counts is not None
            else len(tokenizer.encode(str(turn["text"]), add_special_tokens=False))
        )
        if estimated_count + standalone_count + 1 > max_tokens:
            break
        start -= 1
        estimated_count += standalone_count + 1

    prompt, count = candidate(start)
    while count > max_tokens:
        start += 1
        prompt, count = candidate(start)
    while start > 0:
        longer_prompt, longer_count = candidate(start - 1)
        if longer_count > max_tokens:
            break
        start -= 1
        prompt, count = longer_prompt, longer_count
    return list(prefix[start:]), prompt if start < len(prefix) else empty_prompt, count


def _select_history_ids(
    rows: Sequence[Mapping[str, Any]], *, history_limit: int | None, seed: int
) -> set[str]:
    """Select a stable content-ranked set of complete histories."""
    history_ids = sorted({str(row["history_id"]) for row in rows})
    if history_limit is None:
        return set(history_ids)
    if history_limit < 1:
        raise ValueError("history_limit must be positive")
    if history_limit >= len(history_ids):
        return set(history_ids)
    ranked = sorted(
        history_ids,
        key=lambda history_id: hashlib.sha256(
            f"{seed}:{history_id}".encode("utf-8")
        ).digest(),
    )
    return set(ranked[:history_limit])


def _extract_synthetic_answer(text: str) -> str | None:
    """Return one unique benchmark label, rejecting absent or ambiguous completions."""
    labels = set(_SYNTHETIC_ANSWER.findall(text))
    return next(iter(labels)) if len(labels) == 1 else None


def _score_generation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive all answer fields from immutable raw completions."""
    scored = []
    for row in rows:
        extracted_answer = _extract_synthetic_answer(str(row.get("answer", "")))
        updated = dict(row)
        updated.update(
            {
                "extracted_answer": extracted_answer,
                "exact_match": exact_match(extracted_answer or "", str(row["gold"])),
                "f1": f1_score(extracted_answer or "", str(row["gold"])),
            }
        )
        scored.append(updated)
    return scored


def _percentile_interval(values: Sequence[float]) -> list[float]:
    """Return a deterministic percentile interval from sorted bootstrap values."""
    ordered = sorted(values)
    lower = max(0, math.floor(0.025 * len(ordered)))
    upper = min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)
    return [ordered[lower], ordered[upper]]


def _aggregate_generation_rows(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int = 2000, seed: int = 47
) -> dict[str, Any]:
    """Aggregate generation accuracy with history-clustered bootstrap intervals."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    result = {}
    for window in sorted({int(row["window"]) for row in rows}):
        window_rows = [row for row in rows if int(row["window"]) == window]
        by_history: dict[str, list[Mapping[str, Any]]] = {}
        for row in window_rows:
            by_history.setdefault(str(row["history_id"]), []).append(row)
        histories = sorted(by_history)
        rng = random.Random(f"{seed}:{window}")
        em_bootstrap = []
        f1_bootstrap = []
        for _ in range(bootstrap_samples):
            sampled = [rng.choice(histories) for _ in histories]
            sampled_rows = [row for history in sampled for row in by_history[history]]
            em_bootstrap.append(mean(float(row["exact_match"]) for row in sampled_rows))
            f1_bootstrap.append(mean(float(row["f1"]) for row in sampled_rows))
        result[str(window)] = {
            "checkpoint_count": len(window_rows),
            "history_cluster_count": len(histories),
            "exact_match": mean(float(row["exact_match"]) for row in window_rows),
            "exact_match_ci_95": _percentile_interval(em_bootstrap),
            "f1": mean(float(row["f1"]) for row in window_rows),
            "f1_ci_95": _percentile_interval(f1_bootstrap),
            "oracle_context_answer_available": mean(
                float(bool(row["oracle_context_answer_available"])) for row in window_rows
            ),
            "unique_label_extraction_rate": mean(
                float(row.get("extracted_answer") is not None) for row in window_rows
            ),
            "hit_max_new_tokens_rate": mean(
                float(bool(row.get("hit_max_new_tokens"))) for row in window_rows
            ),
            "mean_generation_seconds": mean(
                float(row["generation_seconds"]) for row in window_rows
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
        }
    return result


def _load_v3_inputs(
    dataset_dir: Path, v3_results_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Validate and load the v3 source rows and sliding checkpoint contract."""
    manifest = json.loads((v3_results_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((v3_results_dir / "metrics.json").read_text(encoding="utf-8"))
    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError(f"unsupported v3 benchmark: {manifest.get('benchmark_version')!r}")
    _validate_dataset_hashes(dataset_dir, manifest.get("dataset_sha256", {}))
    artifact_hashes = manifest.get("artifact_sha256", {})
    source_artifact_sha256 = {
        name: _validate_declared_artifact(v3_results_dir / name, artifact_hashes)
        for name in ("metrics.json", "predictions.jsonl")
    }
    events = [
        row
        for row in _read_jsonl(dataset_dir / "events.jsonl")
        if row.get("split") == "test"
        and row.get("hardness_profile") == "anti_shortcut_interleaved_v3"
    ]
    queries = [
        row
        for row in _read_jsonl(dataset_dir / "queries.jsonl")
        if row.get("split") == "test"
        and row.get("hardness_profile") == "anti_shortcut_interleaved_v3"
    ]
    checkpoint_rows = [
        row
        for row in _read_jsonl(v3_results_dir / "predictions.jsonl")
        if str(row.get("method", "")).startswith("sliding_context:")
    ]
    if not checkpoint_rows:
        raise ValueError("v3 predictions contain no sliding checkpoints")
    return events, queries, metrics, {
        "manifest": manifest,
        "rows": checkpoint_rows,
        "source_artifact_sha256": source_artifact_sha256,
    }


def _build_generation_specs(
    events: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    history_limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Rebuild saved checkpoints and fit each model-specific complete-turn suffix."""
    schedule_config = metrics.get("schedule", {})
    history_ids = sorted({str(event["history_id"]) for event in events})
    schedule = build_interleaved_schedule(
        events,
        tokenizer=_ScheduleTokenizer(),
        seed=int(schedule_config["seed"]),
        concurrent_accounts=int(schedule_config["concurrent_accounts"]),
        min_segment_events=4,
        max_segment_events=8,
    )
    turns = list(schedule["turns"])
    if len(turns) != int(schedule_config["turn_count"]):
        raise ValueError("rebuilt schedule turn count differs from v3 metrics")
    schedule_sha256 = hashlib.sha256(
        "\n".join(
            f"{turn['stream_event']['event_id']}\t{turn['text']}" for turn in turns
        ).encode("utf-8")
    ).hexdigest()
    turn_token_counts = {
        str(turn["turn_id"]): len(
            tokenizer.encode(str(turn["text"]), add_special_tokens=False)
        )
        for turn in turns
    }
    query_by_id = {str(query["query_id"]): query for query in queries}
    selected_histories = _select_history_ids(
        checkpoint_rows, history_limit=history_limit, seed=seed
    )
    specs = []
    for row in checkpoint_rows:
        if str(row["history_id"]) not in selected_histories:
            continue
        query_id = str(row["query_id"])
        query = query_by_id.get(query_id)
        if query is None:
            raise ValueError(f"checkpoint references unknown query {query_id}")
        checkpoint = int(row["checkpoint_turn_index"])
        if checkpoint < 1 or checkpoint > len(turns):
            raise ValueError(f"checkpoint {query_id} has invalid turn index {checkpoint}")
        prefix = turns[:checkpoint]
        if str(prefix[-1]["turn_id"]) != str(row["selected_end_turn_id"]):
            raise ValueError(f"checkpoint {query_id} end turn differs from rebuilt schedule")
        selected, prompt, token_count = _fit_chat_sliding(
            prefix,
            str(row["retrieval_query_text"]),
            int(row["max_tokens"]),
            tokenizer,
            turn_token_counts,
        )
        selected_target_events = [
            dict(turn["stream_event"])
            for turn in selected
            if str(turn["history_id"]) == str(row["history_id"])
        ]
        resolved = resolve_query(selected_target_events, dict(_visible_query(query)))
        specs.append(
            {
                "history_id": str(row["history_id"]),
                "query_id": query_id,
                "query_kind": str(row["checkpoint_kind"]),
                "checkpoint_turn_index": checkpoint,
                "window": int(row["max_tokens"]),
                "gold": str(row["gold"]),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_token_count": token_count,
                "selected_turn_count": len(selected),
                "selected_start_turn_id": str(selected[0]["turn_id"]) if selected else "",
                "selected_end_turn_id": str(prefix[-1]["turn_id"]),
                "oracle_context_answer_available": resolved == row["gold"],
                "schedule_sha256": schedule_sha256,
            }
        )
    return sorted(specs, key=lambda item: (item["history_id"], item["query_id"], item["window"]))


def _build_rescore_specs(
    events: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    preserved_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    history_limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Rebuild and validate preserved model-specific suffixes without binary fitting."""
    schedule_config = metrics.get("schedule", {})
    schedule = build_interleaved_schedule(
        events,
        tokenizer=_ScheduleTokenizer(),
        seed=int(schedule_config["seed"]),
        concurrent_accounts=int(schedule_config["concurrent_accounts"]),
        min_segment_events=4,
        max_segment_events=8,
    )
    turns = list(schedule["turns"])
    if len(turns) != int(schedule_config["turn_count"]):
        raise ValueError("rebuilt schedule turn count differs from v3 metrics")
    schedule_sha256 = hashlib.sha256(
        "\n".join(
            f"{turn['stream_event']['event_id']}\t{turn['text']}" for turn in turns
        ).encode("utf-8")
    ).hexdigest()
    turn_index = {str(turn["turn_id"]): index for index, turn in enumerate(turns)}
    query_by_id = {str(query["query_id"]): query for query in queries}
    checkpoint_by_key = {
        (str(row["query_id"]), int(row["max_tokens"])): row for row in checkpoint_rows
    }
    selected_histories = _select_history_ids(
        checkpoint_rows, history_limit=history_limit, seed=seed
    )
    specs = []
    for preserved in preserved_rows:
        if str(preserved["history_id"]) not in selected_histories:
            raise ValueError("preserved predictions contain a history outside the requested sample")
        key = (str(preserved["query_id"]), int(preserved["window"]))
        checkpoint_row = checkpoint_by_key.get(key)
        if checkpoint_row is None:
            raise ValueError(f"preserved prediction {key} has no v3 checkpoint")
        query = query_by_id.get(key[0])
        if query is None:
            raise ValueError(f"preserved prediction references unknown query {key[0]}")
        checkpoint = int(checkpoint_row["checkpoint_turn_index"])
        start_id = str(preserved["selected_start_turn_id"])
        start = turn_index.get(start_id)
        if start is None or start >= checkpoint:
            raise ValueError(f"preserved prediction {key} has invalid start turn {start_id}")
        prefix = turns[:checkpoint]
        if str(prefix[-1]["turn_id"]) != str(checkpoint_row["selected_end_turn_id"]):
            raise ValueError(f"checkpoint {key[0]} end turn differs from rebuilt schedule")
        selected = prefix[start:]
        prompt = _chat_prompt(selected, str(checkpoint_row["retrieval_query_text"]), tokenizer)
        token_count = _prompt_token_count(prompt, tokenizer)
        if token_count != int(preserved["input_token_count"]):
            raise ValueError(f"preserved prediction {key} input token count differs")
        if start > 0:
            longer = _chat_prompt(prefix[start - 1 :], str(checkpoint_row["retrieval_query_text"]), tokenizer)
            if _prompt_token_count(longer, tokenizer) <= int(preserved["window"]):
                raise ValueError(f"preserved prediction {key} suffix is not maximal")
        selected_target_events = [
            dict(turn["stream_event"])
            for turn in selected
            if str(turn["history_id"]) == str(checkpoint_row["history_id"])
        ]
        resolved = resolve_query(selected_target_events, dict(_visible_query(query)))
        specs.append(
            {
                "history_id": str(checkpoint_row["history_id"]),
                "query_id": key[0],
                "query_kind": str(checkpoint_row["checkpoint_kind"]),
                "checkpoint_turn_index": checkpoint,
                "window": key[1],
                "gold": str(checkpoint_row["gold"]),
                "input_token_count": token_count,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "selected_turn_count": len(selected),
                "selected_start_turn_id": start_id,
                "selected_end_turn_id": str(prefix[-1]["turn_id"]),
                "oracle_context_answer_available": resolved == checkpoint_row["gold"],
                "schedule_sha256": schedule_sha256,
            }
        )
    return sorted(specs, key=lambda item: (item["history_id"], item["query_id"], item["window"]))


def _model_file_hashes(model_path: Path) -> dict[str, str]:
    """Hash the local files that determine weights, tokenization, and prompting."""
    names = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    )
    paths = [model_path / name for name in names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"model snapshot is missing required files: {missing}")

    # Prompt rendering must be hashed, but releases place the chat template
    # differently: some ship a standalone chat_template.jinja, others embed a
    # chat_template field inside tokenizer_config.json. Require one or the
    # other so the prompt source is always covered, and hash the standalone
    # file whenever it exists.
    template_file = model_path / "chat_template.jinja"
    if template_file.is_file():
        paths.append(template_file)
    else:
        try:
            tokenizer_config = json.loads(
                (model_path / "tokenizer_config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read tokenizer_config.json for prompt provenance: {error}"
            ) from error
        if not str(tokenizer_config.get("chat_template", "")).strip():
            raise ValueError(
                "model snapshot has neither chat_template.jinja nor a "
                "chat_template embedded in tokenizer_config.json, so prompt "
                "rendering cannot be authenticated"
            )

    paths.extend(sorted(model_path.glob("model.safetensors-*.safetensors")))
    return {path.name: _sha256(path) for path in paths}


def _model_metadata(
    model_path: Path, model_id: str, tokenizer: Any, model: Any
) -> dict[str, Any]:
    """Record resolved local model and tokenizer provenance."""
    return {
        "model_id": model_id,
        "resolved_revision": model_path.name if model_path.parent.name == "snapshots" else None,
        "file_sha256": _model_file_hashes(model_path),
        "model_type": str(model.config.model_type),
        "architecture": type(model).__name__,
        "tokenizer": type(tokenizer).__name__,
    }


def _model_context_limit(config: Any) -> int:
    """Return the context limit from text-only or nested multimodal config."""
    text_config = getattr(config, "text_config", config)
    limit = getattr(text_config, "max_position_embeddings", None)
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("model config does not declare positive max_position_embeddings")
    return limit


def _validate_resume_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    """Reject completed rows created under different source or model provenance."""
    if not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "evaluator_version",
        "source_benchmark_version",
        "dataset_sha256",
        "model",
        "evaluation_config",
        "source_artifact_sha256",
        "schedule_sha256",
    ):
        if existing.get(field) != expected.get(field):
            raise ValueError(f"cannot resume because manifest field {field} differs")


def _validate_resumed_row(row: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    """Reject a cached generation whose immutable checkpoint fields have drifted."""
    fields = (
        "history_id",
        "query_id",
        "window",
        "gold",
        "checkpoint_turn_index",
        "input_token_count",
        "selected_turn_count",
        "selected_start_turn_id",
        "selected_end_turn_id",
        "prompt_sha256",
    )
    for field in fields:
        if row.get(field) != spec.get(field):
            raise ValueError(f"cannot resume {spec.get('query_id')}: field {field} differs")


def run_evaluation(
    dataset_dir: Path,
    v3_results_dir: Path,
    output_dir: Path,
    model_path: Path,
    model_id: str,
    *,
    device: str,
    history_limit: int | None,
    seed: int,
    max_new_tokens: int,
    bootstrap_samples: int,
    use_kernels: bool,
) -> dict[str, Any]:
    """Run resumable, greedy, batch-one local generation and write artifacts."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    dataset_dir = Path(dataset_dir)
    v3_results_dir = Path(v3_results_dir)
    output_dir = Path(output_dir)
    model_path = Path(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    events, queries, v3_metrics, v3_data = _load_v3_inputs(dataset_dir, v3_results_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    specs = _build_generation_specs(
        events,
        queries,
        v3_metrics,
        v3_data["rows"],
        tokenizer,
        history_limit=history_limit,
        seed=seed,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        use_kernels=use_kernels,
    ).to(device).eval()
    metadata = _model_metadata(model_path, model_id, tokenizer, model)
    metadata.update(
        {
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "device": device,
            "do_sample": False,
            "batch_size": 1,
            "enable_thinking": False,
            "max_new_tokens": max_new_tokens,
            "hub_kernels_requested": use_kernels,
        }
    )
    manifest_path = output_dir / "manifest.json"
    generation_manifest_path = output_dir / "generation_manifest.json"
    generations_path = output_dir / "generations.jsonl"
    evaluation_config = {
        "history_limit": history_limit,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "hub_kernels_requested": use_kernels,
    }
    generation_manifest = {
        "status": "running",
        "evaluator_version": EVALUATOR_VERSION,
        "source_benchmark_version": BENCHMARK_VERSION,
        "dataset_sha256": dict(v3_data["manifest"]["dataset_sha256"]),
        "source_artifact_sha256": dict(v3_data["source_artifact_sha256"]),
        "schedule_sha256": specs[0]["schedule_sha256"] if specs else None,
        "model": metadata,
        "evaluation_config": evaluation_config,
        "requested_checkpoint_count": len(specs),
    }
    if generations_path.exists() and not generation_manifest_path.exists():
        raise ValueError("cannot resume generations without generation_manifest.json")
    _validate_resume_manifest(generation_manifest_path, generation_manifest)
    completed = {}
    if generations_path.exists():
        for row in _read_jsonl(generations_path):
            key = (str(row["query_id"]), int(row["window"]))
            if key in completed:
                raise ValueError(f"duplicate resumed prediction for {key}")
            completed[key] = row
        spec_by_key = {
            (str(spec["query_id"]), int(spec["window"])): spec for spec in specs
        }
        for key, row in completed.items():
            spec = spec_by_key.get(key)
            if spec is None:
                raise ValueError(f"resumed prediction {key} is outside the requested sample")
            _validate_resumed_row(row, spec)
    _write_json(generation_manifest_path, generation_manifest)
    try:
        with generations_path.open("a", encoding="utf-8") as handle:
            for spec in specs:
                key = (str(spec["query_id"]), int(spec["window"]))
                if key in completed:
                    continue
                inputs = tokenizer(
                    spec["prompt"],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(device)
                input_count = int(inputs.input_ids.shape[1])
                if input_count != int(spec["input_token_count"]):
                    raise ValueError(
                        f"token count changed for {spec['query_id']}: "
                        f"fit {spec['input_token_count']}, model input {input_count}"
                    )
                context_limit = _model_context_limit(model.config)
                if input_count + max_new_tokens > context_limit:
                    raise ValueError(
                        f"{spec['query_id']} requires {input_count + max_new_tokens} positions, "
                        f"exceeding model limit {context_limit}"
                    )
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                generated = output[0, input_count:]
                answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
                row = {
                    key: value for key, value in spec.items() if key != "prompt"
                }
                row.update(
                    {
                        "answer": answer,
                        "generated_token_count": int(generated.shape[0]),
                        "hit_max_new_tokens": int(generated.shape[0]) == max_new_tokens,
                        "generation_seconds": elapsed,
                    }
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed[key] = row
        generation_rows = [
            completed[(str(spec["query_id"]), int(spec["window"]))] for spec in specs
        ]
        generations_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in generation_rows),
            encoding="utf-8",
        )
        generation_manifest.update(
            {
                "status": "completed",
                "artifact_sha256": {
                    generations_path.name: _sha256(generations_path),
                },
            }
        )
        _write_json(generation_manifest_path, generation_manifest)
        scored_rows = _score_generation_rows(generation_rows)
        predictions_path = output_dir / "predictions.jsonl"
        predictions_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored_rows),
            encoding="utf-8",
        )
        metrics = {
            "evaluator_version": EVALUATOR_VERSION,
            "source_benchmark_version": BENCHMARK_VERSION,
            "model": metadata,
            "sampling": {
                "history_limit": history_limit,
                "seed": seed,
                "history_count": len({row["history_id"] for row in scored_rows}),
                "checkpoint_count": len(scored_rows),
            },
            "generation_accuracy": _aggregate_generation_rows(
                scored_rows, bootstrap_samples=bootstrap_samples, seed=seed
            ),
            "interpretation": {
                "claim": "direct local Qwen3.5 synthetic-label extraction accuracy on a deterministic pilot sample",
                "oracle_context_answer_available_is_separate": True,
                "window_definition": "prompt-input token cap including chat template and question; generated tokens are additional",
                "structured_context_generation_evaluated": False,
            },
        }
        metrics_path = output_dir / "metrics.json"
        _write_json(metrics_path, metrics)
        manifest = {
            **generation_manifest,
            "status": "completed",
            "generation_manifest_sha256": _sha256(generation_manifest_path),
            "artifacts": [
                generation_manifest_path.name,
                generations_path.name,
                predictions_path.name,
                metrics_path.name,
            ],
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


def rescore_evaluation(
    dataset_dir: Path,
    v3_results_dir: Path,
    output_dir: Path,
    model_path: Path,
    model_id: str,
    *,
    history_limit: int | None,
    seed: int,
    max_new_tokens: int,
    bootstrap_samples: int,
    use_kernels: bool,
) -> dict[str, Any]:
    """Authenticate and rescore preserved raw completions without model inference."""
    from transformers import AutoTokenizer

    dataset_dir = Path(dataset_dir)
    v3_results_dir = Path(v3_results_dir)
    output_dir = Path(output_dir)
    model_path = Path(model_path)
    manifest_path = output_dir / "manifest.json"
    generation_manifest_path = output_dir / "generation_manifest.json"
    generations_path = output_dir / "generations.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    if generation_manifest.get("status") != "completed":
        raise ValueError("cannot rescore an incomplete generation artifact")
    expected_generation_hash = str(
        generation_manifest.get("artifact_sha256", {}).get(generations_path.name, "")
    )
    if _sha256(generations_path) != expected_generation_hash:
        raise ValueError("cannot rescore because generations.jsonl hash differs")
    events, queries, v3_metrics, v3_data = _load_v3_inputs(dataset_dir, v3_results_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    expected_config = {
        "history_limit": history_limit,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "hub_kernels_requested": use_kernels,
    }
    if generation_manifest.get("dataset_sha256") != v3_data["manifest"]["dataset_sha256"]:
        raise ValueError("cannot rescore because dataset provenance differs")
    if generation_manifest.get("source_artifact_sha256") != v3_data["source_artifact_sha256"]:
        raise ValueError("cannot rescore because v3 artifact provenance differs")
    if generation_manifest.get("evaluation_config") != expected_config:
        raise ValueError("cannot rescore because evaluation_config differs")
    old_model = generation_manifest.get("model", {})
    if old_model.get("model_id") != model_id:
        raise ValueError("cannot rescore because model ID differs")
    if old_model.get("resolved_revision") != (
        model_path.name if model_path.parent.name == "snapshots" else None
    ):
        raise ValueError("cannot rescore because model revision differs")
    current_model_hashes = _model_file_hashes(model_path)
    old_model_hashes = old_model.get("file_sha256", {})
    if any(current_model_hashes.get(name) != digest for name, digest in old_model_hashes.items()):
        raise ValueError("cannot rescore because authenticated model file hashes differ")
    added_hashes = sorted(set(current_model_hashes) - set(old_model_hashes))
    if added_hashes:
        old_model = dict(old_model)
        old_model["file_sha256"] = current_model_hashes
        generation_manifest["model"] = old_model
        generation_manifest["provenance_augmentation"] = {
            "added_file_hashes": added_hashes,
            "raw_generations_unchanged": True,
        }
        _write_json(generation_manifest_path, generation_manifest)
    rows_by_key = {}
    preserved_rows = _read_jsonl(generations_path)
    selected_histories = _select_history_ids(
        v3_data["rows"], history_limit=history_limit, seed=seed
    )
    expected_keys = {
        (str(row["query_id"]), int(row["max_tokens"]))
        for row in v3_data["rows"]
        if str(row["history_id"]) in selected_histories
    }
    actual_keys = {
        (str(row["query_id"]), int(row["window"])) for row in preserved_rows
    }
    if actual_keys != expected_keys:
        raise ValueError("generation rows do not match the complete requested sample")
    specs = _build_rescore_specs(
        events,
        queries,
        v3_metrics,
        v3_data["rows"],
        preserved_rows,
        tokenizer,
        history_limit=history_limit,
        seed=seed,
    )
    if generation_manifest.get("schedule_sha256") != (
        specs[0]["schedule_sha256"] if specs else None
    ):
        raise ValueError("cannot rescore because schedule digest differs")
    for row in preserved_rows:
        key = (str(row["query_id"]), int(row["window"]))
        if key in rows_by_key:
            raise ValueError(f"duplicate prediction for {key}")
        rows_by_key[key] = row
    generation_rows = []
    for spec in specs:
        key = (str(spec["query_id"]), int(spec["window"]))
        row = rows_by_key.pop(key, None)
        if row is None:
            raise ValueError(f"missing prediction for {key}")
        _validate_resumed_row(row, spec)
        updated = dict(row)
        updated.update({key: value for key, value in spec.items() if key != "prompt"})
        generation_rows.append(updated)
    if rows_by_key:
        raise ValueError(f"predictions contain rows outside requested sample: {sorted(rows_by_key)}")
    generation_rows.sort(
        key=lambda row: (row["history_id"], row["query_id"], row["window"])
    )
    rescored = _score_generation_rows(generation_rows)
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rescored),
        encoding="utf-8",
    )
    metrics = {
        "evaluator_version": EVALUATOR_VERSION,
        "source_benchmark_version": BENCHMARK_VERSION,
        "model": old_model,
        "sampling": {
            "history_limit": history_limit,
            "seed": seed,
            "history_count": len({row["history_id"] for row in rescored}),
            "checkpoint_count": len(rescored),
        },
        "generation_accuracy": _aggregate_generation_rows(
            rescored, bootstrap_samples=bootstrap_samples, seed=seed
        ),
        "interpretation": {
            "claim": "direct local Qwen3.5 unique synthetic-label extraction accuracy on a deterministic pilot sample",
            "raw_completions_preserved_and_rescored": True,
            "oracle_context_answer_available_is_separate": True,
            "window_definition": "prompt-input token cap including chat template and question; generated tokens are additional",
            "structured_context_generation_evaluated": False,
        },
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    manifest = {
        "status": "completed",
        "evaluator_version": EVALUATOR_VERSION,
        "source_benchmark_version": BENCHMARK_VERSION,
        "dataset_sha256": dict(v3_data["manifest"]["dataset_sha256"]),
        "source_artifact_sha256": dict(v3_data["source_artifact_sha256"]),
        "schedule_sha256": specs[0]["schedule_sha256"] if specs else None,
        "model": old_model,
        "evaluation_config": expected_config,
        "requested_checkpoint_count": len(specs),
        "generation_manifest_sha256": _sha256(generation_manifest_path),
        "artifacts": [
            generation_manifest_path.name,
            generations_path.name,
            predictions_path.name,
            metrics_path.name,
        ],
        "artifact_sha256": {
            generation_manifest_path.name: _sha256(generation_manifest_path),
            generations_path.name: _sha256(generations_path),
            predictions_path.name: _sha256(predictions_path),
            metrics_path.name: _sha256(metrics_path),
        },
    }
    _write_json(manifest_path, manifest)
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Run the local Qwen3.5 evaluator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--v3-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default=os.environ.get("QWEN_EVAL_DEVICE", "cuda"))
    parser.add_argument("--history-limit", type=int)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--use-kernels", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--rescore-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.rescore_existing:
        metrics = rescore_evaluation(
            args.dataset,
            args.v3_results,
            args.output_dir,
            args.model,
            args.model_id,
            history_limit=args.history_limit,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            bootstrap_samples=args.bootstrap_samples,
            use_kernels=args.use_kernels,
        )
    else:
        metrics = run_evaluation(
            args.dataset,
            args.v3_results,
            args.output_dir,
            args.model,
            args.model_id,
            device=args.device,
            history_limit=args.history_limit,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            bootstrap_samples=args.bootstrap_samples,
            use_kernels=args.use_kernels,
        )
    print(json.dumps(metrics["generation_accuracy"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
