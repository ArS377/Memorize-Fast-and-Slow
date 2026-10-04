"""Train and evaluate hard-gated neural and differentiable Scallop admission models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import scallopy
import torch

from neurosym.domain.hard_gate_admission import admit_candidate, derive_candidate_features


DECISION_LABELS = ("accept", "replace", "reject")
EVIDENCE_CODES = (
    "no_conflict",
    "resolves_ambiguity",
    "higher_evidence_conflict",
    "direct_supersedes",
    "low_evidence_conflict",
    "equal_evidence_conflict",
)
FEATURE_NAMES = (
    "candidate_confidence_score",
    "candidate_source_authority",
    "candidate_has_supersedes",
    "active_conflict_count",
    "max_active_conflict_confidence_score",
    "equal_evidence_conflict",
    "resolves_ambiguity",
)
DATASET_ARTIFACTS = ("candidate_updates.jsonl", "events.jsonl")
SOURCE_FILES = (
    "configs/differentiable_admission.json",
    "experiments/differentiable_admission.py",
    "experiments/synthetic_temporal_preferences.py",
    "neurosym/domain/hard_gate_admission.py",
)


@dataclass(frozen=True)
class AdmissionExample:
    """One candidate update encoded for hard-gated admission evaluation."""

    candidate_id: str
    history_id: str
    event_family: str
    split: str
    features: tuple[float, ...]
    target: int
    hard_eligible: bool
    hard_reason: str


@dataclass(frozen=True)
class TrainingConfig:
    """Validated training controls loaded from a run configuration file."""

    batch_size: int
    confidence_noise_stddevs: tuple[float, ...]
    epochs: int
    hidden_dim: int
    learning_rate: float
    seed: int
    stress_repeats: int


class DifferentiableScallopLayer(torch.nn.Module):
    """Map learned rationale probabilities to admission decisions with Scallop rules."""

    def __init__(self) -> None:
        super().__init__()
        context = scallopy.ScallopContext(provenance="diffaddmultprob2")
        context.add_relation(
            "evidence",
            (str,),
            input_mapping=[(code,) for code in EVIDENCE_CODES],
        )
        context.add_rule('decision("accept") = evidence("no_conflict")')
        context.add_rule('decision("accept") = evidence("resolves_ambiguity")')
        context.add_rule('decision("replace") = evidence("higher_evidence_conflict")')
        context.add_rule('decision("replace") = evidence("direct_supersedes")')
        context.add_rule('decision("reject") = evidence("low_evidence_conflict")')
        context.add_rule('decision("reject") = evidence("equal_evidence_conflict")')
        self._forward = context.forward_function(
            "decision",
            output_mapping=[(label,) for label in DECISION_LABELS],
            dispatch="serial",
        )

    def forward(self, evidence_probabilities: torch.Tensor) -> torch.Tensor:
        """Return normalized decision probabilities for a batch of rationale scores."""
        decisions = self._forward(evidence=evidence_probabilities)
        denominator = decisions.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return decisions / denominator


class LinearAdmissionModel(torch.nn.Module):
    """Multinomial logistic-regression admission baseline."""

    returns_probabilities = False

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(feature_count, len(DECISION_LABELS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return unnormalized decision logits."""
        return self.classifier(features)


class MLPAdmissionModel(torch.nn.Module):
    """Direct neural admission baseline without symbolic reasoning."""

    returns_probabilities = False

    def __init__(self, feature_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(feature_count, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(DECISION_LABELS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return unnormalized decision logits."""
        return self.classifier(features)


class NeuroSymbolicAdmissionModel(torch.nn.Module):
    """Learn rationale probabilities and execute admission rules through Scallop."""

    returns_probabilities = True

    def __init__(self, feature_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.evidence_scorer = torch.nn.Sequential(
            torch.nn.Linear(feature_count, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(EVIDENCE_CODES)),
        )
        self.reasoner = DifferentiableScallopLayer()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return Scallop-derived decision probabilities."""
        evidence_probabilities = torch.softmax(self.evidence_scorer(features), dim=1)
        return self.reasoner(evidence_probabilities)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact and fail at the exact malformed line."""
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read required dataset artifact {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} at line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"expected object in {path} at line {line_number}")
        rows.append(row)
    return rows


def _numeric_features(raw: Mapping[str, Any], candidate_id: str) -> tuple[float, ...]:
    """Validate and encode admission-time features without reading gold metadata."""
    missing = [name for name in FEATURE_NAMES if name not in raw]
    if missing:
        raise ValueError(f"candidate {candidate_id} is missing features: {', '.join(missing)}")
    authority = raw["candidate_source_authority"]
    if authority not in {"inferred", "direct_user"}:
        raise ValueError(f"candidate {candidate_id} has unsupported source authority: {authority!r}")
    encoded = (
        raw["candidate_confidence_score"],
        1.0 if authority == "direct_user" else 0.0,
        raw["candidate_has_supersedes"],
        raw["active_conflict_count"],
        raw["max_active_conflict_confidence_score"],
        raw["equal_evidence_conflict"],
        raw["resolves_ambiguity"],
    )
    try:
        return tuple(float(value) for value in encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"candidate {candidate_id} contains a non-numeric feature") from error


def load_admission_dataset(dataset_dir: Path) -> dict[str, list[AdmissionExample]]:
    """Load candidates and independently apply the benchmark hard-gate oracle."""
    dataset_dir = Path(dataset_dir)
    events_by_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_by_history: dict[str, str] = {}
    for event in _read_jsonl(dataset_dir / "events.jsonl"):
        history_id = str(event.get("history_id") or "")
        split = str(event.get("split") or "")
        if not history_id:
            raise ValueError("event is missing history_id")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"event for {history_id} has unsupported split: {split!r}")
        prior_split = split_by_history.setdefault(history_id, split)
        if prior_split != split:
            raise ValueError(f"history {history_id} appears in both {prior_split} and {split}")
        events_by_history[history_id].append(event)

    examples_by_split = {"train": [], "dev": [], "test": []}
    for candidate in _read_jsonl(dataset_dir / "candidate_updates.jsonl"):
        candidate_id = str(candidate.get("candidate_id") or "")
        history_id = str(candidate.get("history_id") or "")
        split = str(candidate.get("split") or "")
        if not candidate_id or not history_id:
            raise ValueError("candidate is missing candidate_id or history_id")
        if split not in examples_by_split:
            raise ValueError(f"candidate {candidate_id} has unsupported split: {split!r}")
        if history_id not in events_by_history:
            raise ValueError(f"candidate {candidate_id} has no history events")
        expected_split = split_by_history[history_id]
        if split != expected_split:
            raise ValueError(f"history {history_id} appears in both {expected_split} and {split}")
        raw_features = candidate.get("candidate_features")
        if not isinstance(raw_features, Mapping):
            raise ValueError(f"candidate {candidate_id} is missing candidate_features")
        derived_features = derive_candidate_features(events_by_history[history_id], candidate)
        if dict(raw_features) != derived_features:
            raise ValueError(f"candidate {candidate_id} features disagree with history-derived evidence")

        hard_decision = admit_candidate(events_by_history[history_id], candidate)
        declared_hard_gate = candidate.get("gold_hard_gate")
        if declared_hard_gate is not None and declared_hard_gate is not hard_decision.eligible:
            raise ValueError(
                f"candidate {candidate_id} gold_hard_gate={declared_hard_gate!r} "
                f"contradicts oracle eligibility={hard_decision.eligible!r}"
            )
        label = str(candidate.get("gold_soft_label") or "")
        if label not in DECISION_LABELS:
            raise ValueError(f"candidate {candidate_id} has unsupported soft label: {label!r}")
        if not hard_decision.eligible and label != "reject":
            raise ValueError(f"hard-rejected candidate {candidate_id} must have reject as its target")
        examples_by_split[split].append(
            AdmissionExample(
                candidate_id=candidate_id,
                history_id=history_id,
                event_family=str(candidate.get("event_family") or "unknown"),
                split=split,
                features=_numeric_features(derived_features, candidate_id),
                target=DECISION_LABELS.index(label),
                hard_eligible=hard_decision.eligible,
                hard_reason=hard_decision.reason_code,
            )
        )
    if not examples_by_split["train"] or not examples_by_split["test"]:
        raise ValueError("dataset must contain non-empty train and test splits")
    return examples_by_split


def load_training_config(path: Path) -> TrainingConfig:
    """Load training controls from JSON and reject invalid or silent defaults."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load training config {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("training config must be a JSON object")
    required = {
        "batch_size",
        "confidence_noise_stddevs",
        "epochs",
        "hidden_dim",
        "learning_rate",
        "seed",
        "stress_repeats",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"training config is missing keys: {', '.join(missing)}")
    unknown = sorted(raw.keys() - required)
    if unknown:
        raise ValueError(f"training config has unknown keys: {', '.join(unknown)}")
    integer_fields = ("batch_size", "epochs", "hidden_dim", "seed", "stress_repeats")
    if any(isinstance(raw[name], bool) or not isinstance(raw[name], int) for name in integer_fields):
        raise ValueError(f"{', '.join(integer_fields)} must be integers")
    noise_values = raw["confidence_noise_stddevs"]
    if not isinstance(noise_values, list) or not noise_values:
        raise ValueError("confidence_noise_stddevs must be a non-empty list")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in noise_values
    ):
        raise ValueError("confidence_noise_stddevs must contain positive finite numbers")
    learning_rate = raw["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0
    ):
        raise ValueError("learning_rate must be positive")
    config = TrainingConfig(
        batch_size=raw["batch_size"],
        confidence_noise_stddevs=tuple(float(value) for value in noise_values),
        epochs=raw["epochs"],
        hidden_dim=raw["hidden_dim"],
        learning_rate=float(learning_rate),
        seed=raw["seed"],
        stress_repeats=raw["stress_repeats"],
    )
    if config.batch_size < 1 or config.epochs < 1 or config.hidden_dim < 1 or config.stress_repeats < 1:
        raise ValueError("batch_size, epochs, hidden_dim, and stress_repeats must be positive")
    return config


def perturb_confidence_features(
    examples: Sequence[AdmissionExample],
    *,
    stddev: float,
    seed: int,
    repeat: int,
) -> list[AdmissionExample]:
    """Perturb observed confidence evidence while preserving latent truth and hard gates."""
    if not math.isfinite(stddev) or stddev < 0:
        raise ValueError(f"stddev must be a finite non-negative number, got {stddev}")
    if repeat < 0:
        raise ValueError(f"repeat must be non-negative, got {repeat}")
    if stddev == 0:
        return list(examples)
    generator = torch.Generator().manual_seed(seed + repeat * 1_000_003)
    noise = torch.randn((len(examples), 2), generator=generator) * stddev
    stressed: list[AdmissionExample] = []
    for index, example in enumerate(examples):
        features = list(example.features)
        features[0] = min(1.0, max(0.0, features[0] + float(noise[index, 0])))
        features[4] = min(1.0, max(0.0, features[4] + float(noise[index, 1])))
        features[5] = float(math.isclose(features[0], features[4], abs_tol=1e-9))
        stressed.append(replace(example, features=tuple(features)))
    return stressed


def _eligible_tensors(examples: Sequence[AdmissionExample]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build feature and target tensors for candidates that passed the hard gate."""
    eligible = [example for example in examples if example.hard_eligible]
    if not eligible:
        raise ValueError("split has no hard-gate-eligible examples")
    return (
        torch.tensor([example.features for example in eligible], dtype=torch.float32),
        torch.tensor([example.target for example in eligible], dtype=torch.long),
    )


def train_model(
    model: torch.nn.Module,
    examples: Sequence[AdmissionExample],
    config: TrainingConfig,
) -> list[float]:
    """Train a model on soft decisions for hard-gate-eligible candidates only."""
    features, targets = _eligible_tensors(examples)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        epoch_loss = 0.0
        permutation = torch.randperm(len(features), generator=generator)
        for start in range(0, len(features), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            output = model(features[indices])
            if getattr(model, "returns_probabilities", False):
                loss = torch.nn.functional.nll_loss(output.clamp_min(1e-8).log(), targets[indices])
            else:
                loss = torch.nn.functional.cross_entropy(output, targets[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indices)
        losses.append(epoch_loss / len(features))
    return losses


def _model_predictions(
    model: torch.nn.Module, examples: Sequence[AdmissionExample]
) -> list[int]:
    """Predict all examples while forcing hard-gate failures to reject."""
    reject = DECISION_LABELS.index("reject")
    predictions = [reject] * len(examples)
    eligible_indices = [index for index, example in enumerate(examples) if example.hard_eligible]
    if not eligible_indices:
        return predictions
    features = torch.tensor(
        [examples[index].features for index in eligible_indices], dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        output = model(features)
        eligible_predictions = output.argmax(dim=1).tolist()
    for index, prediction in zip(eligible_indices, eligible_predictions):
        predictions[index] = int(prediction)
    return predictions


def _fixed_symbolic_predictions(examples: Sequence[AdmissionExample]) -> list[int]:
    """Execute a fixed feature policy in Scallop after the immutable hard gate."""
    reject = DECISION_LABELS.index("reject")
    predictions = [reject] * len(examples)
    feature_rows: list[tuple[int, bool, bool, bool, bool]] = []
    for index, example in enumerate(examples):
        if not example.hard_eligible:
            continue
        confidence, direct, has_supersedes, conflicts, max_confidence, equal, resolves = example.features
        feature_rows.append(
            (
                index,
                bool(resolves),
                conflicts == 0,
                bool(direct and has_supersedes),
                bool(not equal and confidence > max_confidence),
            )
        )
    context = scallopy.ScallopContext()
    context.add_relation("features", (int, bool, bool, bool, bool))
    context.add_facts("features", feature_rows)
    context.add_rule('decision(i, "accept") = features(i, true, _, _, _)')
    context.add_rule('decision(i, "accept") = features(i, false, true, _, _)')
    context.add_rule('decision(i, "replace") = features(i, false, false, true, _)')
    context.add_rule('decision(i, "replace") = features(i, false, false, false, true)')
    context.add_rule('decision(i, "reject") = features(i, false, false, false, false)')
    context.run()
    for index, label in context.relation("decision"):
        predictions[index] = DECISION_LABELS.index(label)
    return predictions


def _oracle_predictions(examples: Sequence[AdmissionExample]) -> list[int]:
    """Return gold soft labels as an explicitly labeled upper bound."""
    return [example.target for example in examples]


def _predictions_for_model(
    model_name: str,
    model: torch.nn.Module | None,
    examples: Sequence[AdmissionExample],
) -> list[int]:
    """Dispatch one named baseline without changing its hard-gate wrapper."""
    if model_name == "fixed_symbolic_scallop":
        return _fixed_symbolic_predictions(examples)
    if model_name == "oracle_soft_upper_bound":
        return _oracle_predictions(examples)
    if model is None:
        raise RuntimeError(f"model {model_name} has no prediction implementation")
    return _model_predictions(model, examples)


def _evaluate_confidence_stress(
    models: Mapping[str, torch.nn.Module | None],
    examples: Sequence[AdmissionExample],
    config: TrainingConfig,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate paired confidence-noise scenarios and retain per-candidate evidence."""
    repeat_metrics: dict[str, dict[str, list[dict[str, Any]]]] = {
        model_name: {} for model_name in models
    }
    scenario_stddevs: dict[str, float] = {}
    prediction_rows: list[dict[str, Any]] = []
    for stddev in config.confidence_noise_stddevs:
        scenario = f"confidence_noise_stddev_{stddev:g}"
        scenario_stddevs[scenario] = stddev
        for model_metrics in repeat_metrics.values():
            model_metrics[scenario] = []
        for repeat in range(config.stress_repeats):
            stressed = perturb_confidence_features(
                examples,
                stddev=stddev,
                seed=config.seed,
                repeat=repeat,
            )
            predictions_by_model = {
                model_name: _predictions_for_model(model_name, model, stressed)
                for model_name, model in models.items()
            }
            for model_name, predictions in predictions_by_model.items():
                repeat_metrics[model_name][scenario].append(
                    _classification_metrics(stressed, predictions)
                )
            for index, (original, observed) in enumerate(zip(examples, stressed)):
                prediction_rows.append(
                    {
                        "scenario": scenario,
                        "stddev": stddev,
                        "repeat": repeat,
                        "candidate_id": original.candidate_id,
                        "history_id": original.history_id,
                        "event_family": original.event_family,
                        "hard_eligible": original.hard_eligible,
                        "hard_reason": original.hard_reason,
                        "target": DECISION_LABELS[original.target],
                        "clean_candidate_confidence": original.features[0],
                        "observed_candidate_confidence": observed.features[0],
                        "clean_conflict_confidence": original.features[4],
                        "observed_conflict_confidence": observed.features[4],
                        "observed_equal_evidence": bool(observed.features[5]),
                        "predictions": {
                            model_name: DECISION_LABELS[predictions[index]]
                            for model_name, predictions in predictions_by_model.items()
                        },
                    }
                )

    scenarios_by_model: dict[str, dict[str, Any]] = {}
    for model_name, model_scenarios in repeat_metrics.items():
        scenarios_by_model[model_name] = {}
        for scenario, measurements in model_scenarios.items():
            accuracies = [measurement["accuracy"] for measurement in measurements]
            eligible_accuracies = [
                measurement["eligible_accuracy"] for measurement in measurements
            ]
            violations = [measurement["hard_gate_violations"] for measurement in measurements]
            scenarios_by_model[model_name][scenario] = {
                "stddev": scenario_stddevs[scenario],
                "repeat_count": len(measurements),
                "repeat_accuracies": accuracies,
                "repeat_eligible_accuracies": eligible_accuracies,
                "accuracy_mean": sum(accuracies) / len(accuracies),
                "accuracy_min": min(accuracies),
                "accuracy_max": max(accuracies),
                "eligible_accuracy_mean": sum(eligible_accuracies) / len(eligible_accuracies),
                "eligible_accuracy_min": min(eligible_accuracies),
                "eligible_accuracy_max": max(eligible_accuracies),
                "hard_gate_violations_max": max(violations),
            }
    return scenarios_by_model, prediction_rows


def _classification_metrics(
    examples: Sequence[AdmissionExample], predictions: Sequence[int]
) -> dict[str, Any]:
    """Compute accuracy and per-label recall from complete hard-gated predictions."""
    if len(examples) != len(predictions):
        raise ValueError("prediction count does not match example count")
    correct = sum(example.target == prediction for example, prediction in zip(examples, predictions))
    by_label: dict[str, dict[str, float | int]] = {}
    for label_index, label in enumerate(DECISION_LABELS):
        members = [index for index, example in enumerate(examples) if example.target == label_index]
        label_correct = sum(predictions[index] == label_index for index in members)
        by_label[label] = {
            "count": len(members),
            "recall": label_correct / len(members) if members else 0.0,
        }
    hard_rejected = [index for index, example in enumerate(examples) if not example.hard_eligible]
    eligible = [index for index, example in enumerate(examples) if example.hard_eligible]
    eligible_correct = sum(examples[index].target == predictions[index] for index in eligible)
    hard_gate_violations = sum(
        predictions[index] != DECISION_LABELS.index("reject") for index in hard_rejected
    )
    return {
        "accuracy": correct / len(examples) if examples else 0.0,
        "count": len(examples),
        "eligible_accuracy": eligible_correct / len(eligible) if eligible else 0.0,
        "eligible_count": len(eligible),
        "by_label": by_label,
        "hard_rejected_count": len(hard_rejected),
        "hard_gate_violations": hard_gate_violations,
    }


def _git_state(repository_root: Path) -> tuple[str | None, bool | None]:
    """Return the current commit and whether the effective checkout is dirty."""
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


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of an exact source or dataset artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_predictions(
    path: Path,
    test_examples: Sequence[AdmissionExample],
    predictions_by_model: Mapping[str, Sequence[int]],
) -> None:
    """Write per-candidate predictions for error analysis and dashboard provenance."""
    with path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(test_examples):
            row = {
                "candidate_id": example.candidate_id,
                "history_id": example.history_id,
                "event_family": example.event_family,
                "hard_eligible": example.hard_eligible,
                "hard_reason": example.hard_reason,
                "target": DECISION_LABELS[example.target],
                "predictions": {
                    model_name: DECISION_LABELS[predictions[index]]
                    for model_name, predictions in predictions_by_model.items()
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_stress_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write per-candidate perturbed evidence and paired model predictions."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_report(path: Path, metrics: Mapping[str, Any]) -> None:
    """Write a compact experiment report with model comparisons and safety checks."""
    first_model = next(iter(metrics["models"].values()))
    noise_scenarios = sorted(
        first_model["stress"],
        key=lambda name: first_model["stress"][name]["stddev"],
    )
    rows = []
    for model_name, model_metrics in metrics["models"].items():
        stress_cells = []
        for scenario in noise_scenarios:
            scenario_metrics = model_metrics["stress"][scenario]
            stress_cells.append(
                f"{scenario_metrics['eligible_accuracy_mean']:.4f} / "
                f"{scenario_metrics['eligible_accuracy_min']:.4f}"
            )
        max_violations = max(
            model_metrics["test"]["hard_gate_violations"],
            *(scenario["hard_gate_violations_max"] for scenario in model_metrics["stress"].values()),
        )
        rows.append(
            f"| {model_name} | {model_metrics['dev']['accuracy']:.4f} | "
            f"{model_metrics['test']['accuracy']:.4f} | "
            f"{' | '.join(stress_cells)} | {max_violations} |"
        )
    stress_headers = [
        f"Noise {first_model['stress'][scenario]['stddev']:g} eligible mean / min"
        for scenario in noise_scenarios
    ]
    interpretation = metrics["interpretation"]
    path.write_text(
        "\n".join(
            [
                "# Differentiable Admission Benchmark",
                "",
                "All learned models score only candidates that pass the immutable hard gate.",
                "Hard-rejected updates are forced to `reject` and audited separately.",
                "The oracle uses gold soft labels and is an upper bound, not a deployable baseline.",
                interpretation["summary"],
                interpretation["stress_protocol"],
                "",
                f"| Model | Dev accuracy | Clean test accuracy | {' | '.join(stress_headers)} | Max hard-gate violations |",
                f"|---|---:|---:|{'|'.join('---:' for _ in stress_headers)}|---:|",
                *rows,
                "",
                "## Limitations",
                "",
                *[f"- {limitation}" for limitation in interpretation["limitations"]],
                "",
                f"Scallop version: `{metrics['runtime']['scallopy']}`.",
                f"PyTorch version: `{metrics['runtime']['torch']}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_experiment(
    dataset_dir: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Run all admission comparisons and persist dashboard-discoverable evidence."""
    torch.manual_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[1]
    git_sha, git_dirty = _git_state(repository_root)
    dataset_dir = Path(dataset_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": output_dir.name,
        "status": "running",
        "started_at": started_at,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "dataset": str(dataset_dir),
        "dataset_sha256": {
            name: _sha256(dataset_dir / name) for name in DATASET_ARTIFACTS
        },
        "source_sha256": {
            name: _sha256(repository_root / name) for name in SOURCE_FILES
        },
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "config": asdict(config),
    }
    _write_json(output_dir / "manifest.json", manifest)

    dataset = load_admission_dataset(dataset_dir)
    feature_count = len(FEATURE_NAMES)
    models: dict[str, torch.nn.Module] = {
        "logistic_regression": LinearAdmissionModel(feature_count),
        "mlp": MLPAdmissionModel(feature_count, config.hidden_dim),
        "differentiable_scallop": NeuroSymbolicAdmissionModel(feature_count, config.hidden_dim),
    }
    training_losses = {
        model_name: train_model(model, dataset["train"], config)
        for model_name, model in models.items()
    }

    prediction_functions = {
        **{model_name: model for model_name, model in models.items()},
        "fixed_symbolic_scallop": None,
        "oracle_soft_upper_bound": None,
    }
    metrics_by_model: dict[str, Any] = {}
    test_predictions: dict[str, list[int]] = {}
    for model_name, model in prediction_functions.items():
        split_predictions: dict[str, list[int]] = {}
        for split in ("dev", "test"):
            predictions = _predictions_for_model(model_name, model, dataset[split])
            split_predictions[split] = predictions
        metrics_by_model[model_name] = {
            "dev": _classification_metrics(dataset["dev"], split_predictions["dev"]),
            "test": _classification_metrics(dataset["test"], split_predictions["test"]),
        }
        if model_name in training_losses:
            metrics_by_model[model_name]["training"] = {
                "initial_loss": training_losses[model_name][0],
                "final_loss": training_losses[model_name][-1],
            }
        test_predictions[model_name] = split_predictions["test"]

    stress_metrics, stress_prediction_rows = _evaluate_confidence_stress(
        prediction_functions,
        dataset["test"],
        config,
    )
    for model_name, model_stress_metrics in stress_metrics.items():
        metrics_by_model[model_name]["stress"] = model_stress_metrics

    metrics = {
        "models": metrics_by_model,
        "interpretation": {
            "summary": (
                "The clean split remains a ceiling result because histories repeat deterministic "
                "feature templates; clean accuracy verifies execution rather than model advantage."
            ),
            "stress_protocol": (
                "Confidence-noise scenarios add deterministic Gaussian noise only to observed "
                "candidate and active-conflict confidence scores. Gold decisions and hard-gate "
                "truth remain fixed to measure robustness to extraction uncertainty. Noise draws "
                "are paired across standard deviations, and stress headlines use eligible-only "
                "accuracy so invariant hard rejects cannot dilute errors."
            ),
            "limitations": [
                "The noise model is synthetic and does not substitute for natural-language extraction evaluation.",
                "The oracle upper bound is invariant by construction and is not deployable.",
                "History identities are held out, but event and candidate-family templates repeat across splits.",
            ],
        },
        "dataset": {
            "split_counts": {split: len(examples) for split, examples in dataset.items()},
            "hard_gate_counts": {
                split: dict(Counter(example.hard_reason for example in examples))
                for split, examples in dataset.items()
            },
            "feature_names": list(FEATURE_NAMES),
        },
        "runtime": {
            "torch": torch.__version__,
            "scallopy": importlib.metadata.version("scallopy"),
        },
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_predictions(output_dir / "predictions.jsonl", dataset["test"], test_predictions)
    _write_stress_predictions(output_dir / "stress_predictions.jsonl", stress_prediction_rows)
    _write_report(output_dir / "report.md", metrics)
    artifact_names = [
        "metrics.json",
        "predictions.jsonl",
        "stress_predictions.jsonl",
        "report.md",
    ]
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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse explicit run inputs and execute the differentiable admission benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    metrics = run_experiment(
        arguments.dataset,
        arguments.output_dir,
        load_training_config(arguments.config),
    )
    for model_name, model_metrics in metrics["models"].items():
        print(f"{model_name}: test_accuracy={model_metrics['test']['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
