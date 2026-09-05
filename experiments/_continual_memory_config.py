"""Validated configuration and tokenizer boundaries for continual-memory benchmarks."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class BenchmarkTokenizer(Protocol):
    """Tokenizer boundary used for exact stream accounting."""

    def encode(self, text: str) -> list[Any]:
        """Return token IDs without model-added special tokens."""

    def metadata(self) -> Mapping[str, Any]:
        """Return requested and resolved tokenizer provenance."""


@dataclass(frozen=True)
class TokenizerConfig:
    """Pinned local tokenizer identity."""

    name: str
    revision: str
    local_files_only: bool


@dataclass(frozen=True)
class InterferenceTier:
    """Number and derived share of distractor turns after each target event."""

    name: str
    distractor_turns_per_target_event: int
    distractor_turn_percentage: float


@dataclass(frozen=True)
class SlidingTokenWindow:
    """Model-visible suffix budget including the checkpoint query."""

    name: str
    max_tokens: int


@dataclass(frozen=True)
class ModelParameterTier:
    """Reporting-only model-size axis that this benchmark does not evaluate."""

    name: str
    min_billions: float
    max_billions: float | None


@dataclass(frozen=True)
class MetricBins:
    """Configured upper bounds for retention slices."""

    turn_age_upper_bounds: tuple[int, ...]
    token_age_upper_bounds: tuple[int, ...]
    stream_position_upper_bounds_percent: tuple[float, ...]


@dataclass(frozen=True)
class DenseEmbeddingConfig:
    """Optional pinned dense baseline configuration."""

    enabled: bool
    model: str
    revision: str
    device: str
    batch_size: int
    local_files_only: bool


@dataclass(frozen=True)
class ScallopReasonerConfig:
    """Resolved HTTP boundary for recursive Scallop lineage reasoning."""

    enabled: bool
    endpoint_env: str
    endpoint: str | None
    timeout_seconds: float


@dataclass(frozen=True)
class ContinualMemoryConfig:
    """Validated continual-memory benchmark configuration."""

    benchmark_version: str
    source_split: str
    source_profile: str
    history_limit: int
    retrieval_k: int
    seed: int
    tokenizer: TokenizerConfig
    interference_tiers: tuple[InterferenceTier, ...]
    sliding_token_windows: tuple[SlidingTokenWindow, ...]
    model_parameter_tiers: tuple[ModelParameterTier, ...]
    metric_bins: MetricBins
    dense_embedding: DenseEmbeddingConfig
    scallop_reasoner: ScallopReasonerConfig


class HuggingFaceTokenizer:
    """Pinned Hugging Face tokenizer with resolved local snapshot metadata."""

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
            "local_files_only": config.local_files_only,
            "add_special_tokens": False,
        }

    def encode(self, text: str) -> list[int]:
        """Tokenize text without model-added special tokens."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def encode_chat(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        """Tokenize the exact non-thinking chat template submitted to Qwen."""
        return self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def metadata(self) -> dict[str, Any]:
        """Return pinned tokenizer provenance."""
        return dict(self._metadata)


def _json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object and identify malformed input at the boundary."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require a mapping-valued configuration field."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    """Require a non-string sequence-valued configuration field."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _string(value: Any, name: str) -> str:
    """Require a non-empty string-valued configuration field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    """Require a positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    """Require a non-negative integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    """Require a finite numeric value without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _strict_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    """Reject missing and unknown keys so benchmark intent cannot drift silently."""
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"{name} keys mismatch: missing={missing}, unknown={unknown}")


def _unique_names(values: Sequence[Any], name: str) -> None:
    """Require unique non-empty names across a configured axis."""
    names = [value.name for value in values]
    if len(names) != len(set(names)):
        raise ValueError(f"{name} names must be unique")


def _increasing(values: Sequence[float | int], name: str) -> None:
    """Require positive strictly increasing bin boundaries."""
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive values")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")


def load_benchmark_config(path: Path) -> ContinualMemoryConfig:
    """Load and validate every benchmark, retrieval, and reporting axis."""
    raw = _json_object(Path(path))
    expected = {
        "benchmark_version",
        "source_split",
        "source_profile",
        "history_limit",
        "retrieval_k",
        "seed",
        "tokenizer",
        "interference_tiers",
        "sliding_token_windows",
        "model_parameter_tiers",
        "metric_bins",
        "dense_embedding",
        "scallop_reasoner",
    }
    _strict_keys(raw, expected, "config")

    tokenizer_raw = _mapping(raw["tokenizer"], "tokenizer")
    _strict_keys(tokenizer_raw, {"name", "revision", "local_files_only"}, "tokenizer")
    if tokenizer_raw["local_files_only"] is not True:
        raise ValueError("tokenizer.local_files_only must be true")
    tokenizer = TokenizerConfig(
        name=_string(tokenizer_raw["name"], "tokenizer.name"),
        revision=_string(tokenizer_raw["revision"], "tokenizer.revision"),
        local_files_only=True,
    )

    interference_tiers = []
    for index, value in enumerate(_sequence(raw["interference_tiers"], "interference_tiers")):
        tier = _mapping(value, f"interference_tiers[{index}]")
        _strict_keys(
            tier,
            {"name", "distractor_turns_per_target_event", "distractor_turn_percentage"},
            f"interference_tiers[{index}]",
        )
        distractor_count = _nonnegative_int(
            tier["distractor_turns_per_target_event"],
            f"interference_tiers[{index}].distractor_turns_per_target_event",
        )
        percentage = _finite_number(
            tier["distractor_turn_percentage"],
            f"interference_tiers[{index}].distractor_turn_percentage",
        )
        derived = 100.0 * distractor_count / (distractor_count + 1)
        if not math.isclose(percentage, derived, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "interference tier derived distractor-turn percentage mismatch: "
                f"configured={percentage}, derived={derived}, distractor_turns={distractor_count}"
            )
        interference_tiers.append(
            InterferenceTier(
                name=_string(tier["name"], f"interference_tiers[{index}].name"),
                distractor_turns_per_target_event=distractor_count,
                distractor_turn_percentage=percentage,
            )
        )
    if not interference_tiers:
        raise ValueError("interference_tiers must not be empty")
    _unique_names(interference_tiers, "interference tier")
    if sum(tier.distractor_turns_per_target_event == 0 for tier in interference_tiers) != 1:
        raise ValueError("interference_tiers must contain exactly one zero-distractor tier")

    windows = []
    for index, value in enumerate(
        _sequence(raw["sliding_token_windows"], "sliding_token_windows")
    ):
        window = _mapping(value, f"sliding_token_windows[{index}]")
        _strict_keys(window, {"name", "max_tokens"}, f"sliding_token_windows[{index}]")
        windows.append(
            SlidingTokenWindow(
                name=_string(window["name"], f"sliding_token_windows[{index}].name"),
                max_tokens=_positive_int(
                    window["max_tokens"], f"sliding_token_windows[{index}].max_tokens"
                ),
            )
        )
    if not windows:
        raise ValueError("sliding_token_windows must not be empty")
    _unique_names(windows, "sliding token window")
    if any(left.max_tokens >= right.max_tokens for left, right in zip(windows, windows[1:])):
        raise ValueError("sliding_token_windows must be strictly increasing")

    parameter_tiers = []
    for index, value in enumerate(
        _sequence(raw["model_parameter_tiers"], "model_parameter_tiers")
    ):
        tier = _mapping(value, f"model_parameter_tiers[{index}]")
        _strict_keys(
            tier, {"name", "min_billions", "max_billions"}, f"model_parameter_tiers[{index}]"
        )
        minimum = _finite_number(tier["min_billions"], "model tier min_billions")
        maximum = (
            None
            if tier["max_billions"] is None
            else _finite_number(tier["max_billions"], "model tier max_billions")
        )
        if minimum < 0 or (maximum is not None and maximum <= minimum):
            raise ValueError(f"invalid model parameter tier bounds at index {index}")
        parameter_tiers.append(
            ModelParameterTier(
                name=_string(tier["name"], f"model_parameter_tiers[{index}].name"),
                min_billions=minimum,
                max_billions=maximum,
            )
        )
    if not parameter_tiers:
        raise ValueError("model_parameter_tiers must not be empty")
    _unique_names(parameter_tiers, "model parameter tier")

    bins_raw = _mapping(raw["metric_bins"], "metric_bins")
    _strict_keys(
        bins_raw,
        {
            "turn_age_upper_bounds",
            "token_age_upper_bounds",
            "stream_position_upper_bounds_percent",
        },
        "metric_bins",
    )
    turn_bounds = tuple(
        _positive_int(value, "turn age bound")
        for value in _sequence(bins_raw["turn_age_upper_bounds"], "turn_age_upper_bounds")
    )
    token_bounds = tuple(
        _positive_int(value, "token age bound")
        for value in _sequence(bins_raw["token_age_upper_bounds"], "token_age_upper_bounds")
    )
    position_bounds = tuple(
        _finite_number(value, "stream position bound")
        for value in _sequence(
            bins_raw["stream_position_upper_bounds_percent"],
            "stream_position_upper_bounds_percent",
        )
    )
    _increasing(turn_bounds, "turn_age_upper_bounds")
    _increasing(token_bounds, "token_age_upper_bounds")
    _increasing(position_bounds, "stream_position_upper_bounds_percent")
    if position_bounds[-1] >= 100.0:
        raise ValueError("stream position upper bounds must be below 100")

    dense_raw = _mapping(raw["dense_embedding"], "dense_embedding")
    _strict_keys(
        dense_raw,
        {"enabled", "model", "revision", "device", "batch_size", "local_files_only"},
        "dense_embedding",
    )
    if not isinstance(dense_raw["enabled"], bool):
        raise ValueError("dense_embedding.enabled must be boolean")
    if dense_raw["device"] != "cpu":
        raise ValueError("dense_embedding.device must be cpu")
    if dense_raw["local_files_only"] is not True:
        raise ValueError("dense_embedding.local_files_only must be true")
    dense = DenseEmbeddingConfig(
        enabled=dense_raw["enabled"],
        model=_string(dense_raw["model"], "dense_embedding.model"),
        revision=_string(dense_raw["revision"], "dense_embedding.revision"),
        device="cpu",
        batch_size=_positive_int(dense_raw["batch_size"], "dense_embedding.batch_size"),
        local_files_only=True,
    )

    scallop_raw = _mapping(raw["scallop_reasoner"], "scallop_reasoner")
    _strict_keys(
        scallop_raw,
        {"enabled", "endpoint_env", "timeout_seconds"},
        "scallop_reasoner",
    )
    if not isinstance(scallop_raw["enabled"], bool):
        raise ValueError("scallop_reasoner.enabled must be boolean")
    endpoint_env = _string(scallop_raw["endpoint_env"], "scallop_reasoner.endpoint_env")
    endpoint = os.environ.get(endpoint_env)
    if scallop_raw["enabled"] and not endpoint:
        raise ValueError(
            f"scallop_reasoner is enabled but environment variable {endpoint_env} is unset"
        )
    timeout_seconds = _finite_number(
        scallop_raw["timeout_seconds"], "scallop_reasoner.timeout_seconds"
    )
    if timeout_seconds <= 0:
        raise ValueError("scallop_reasoner.timeout_seconds must be positive")
    scallop_reasoner = ScallopReasonerConfig(
        enabled=scallop_raw["enabled"],
        endpoint_env=endpoint_env,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
    )

    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return ContinualMemoryConfig(
        benchmark_version=_string(raw["benchmark_version"], "benchmark_version"),
        source_split=_string(raw["source_split"], "source_split"),
        source_profile=_string(raw["source_profile"], "source_profile"),
        history_limit=_positive_int(raw["history_limit"], "history_limit"),
        retrieval_k=_positive_int(raw["retrieval_k"], "retrieval_k"),
        seed=seed,
        tokenizer=tokenizer,
        interference_tiers=tuple(interference_tiers),
        sliding_token_windows=tuple(windows),
        model_parameter_tiers=tuple(parameter_tiers),
        metric_bins=MetricBins(turn_bounds, token_bounds, position_bounds),
        dense_embedding=dense,
        scallop_reasoner=scallop_reasoner,
    )
