from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scallopy")

from experiments.differentiable_admission import (
    DECISION_LABELS,
    DifferentiableScallopLayer,
    _classification_metrics,
    perturb_confidence_features,
    load_admission_dataset,
)
from experiments.synthetic_temporal_preferences import generate_dataset


def test_loader_keeps_histories_disjoint_and_applies_the_hard_gate(tmp_path: Path) -> None:
    """Loaded examples must preserve splits and independently derive hard eligibility."""
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=4, split_counts=(2, 1, 1))

    dataset = load_admission_dataset(dataset_dir)

    histories_by_split = {
        split: {example.history_id for example in examples}
        for split, examples in dataset.items()
    }
    assert histories_by_split == {
        "train": {"history-001", "history-002"},
        "dev": {"history-003"},
        "test": {"history-004"},
    }
    assert all(len(examples) == count for examples, count in zip(dataset.values(), (18, 9, 9)))
    direct_conflict = next(
        example
        for example in dataset["train"]
        if example.event_family == "direct_user_conflict_lacking_supersession_reject"
    )
    assert direct_conflict.hard_eligible is False
    assert direct_conflict.target == DECISION_LABELS.index("reject")


def test_differentiable_scallop_aggregates_evidence_and_backpropagates() -> None:
    """Scallop must combine rationale probabilities into decisions with gradients."""
    layer = DifferentiableScallopLayer()
    evidence = torch.tensor(
        [[0.10, 0.20, 0.30, 0.15, 0.05, 0.20]],
        dtype=torch.float32,
        requires_grad=True,
    )

    decisions = layer(evidence)

    assert decisions.shape == (1, 3)
    assert torch.allclose(decisions.sum(dim=1), torch.ones(1), atol=1e-5)
    decisions[0, DECISION_LABELS.index("replace")].backward()
    assert evidence.grad is not None
    assert evidence.grad[0, 2] > 0
    assert evidence.grad[0, 3] > 0


def test_generated_hard_gate_labels_match_the_independent_oracle(tmp_path: Path) -> None:
    """Generated hard-gate labels cannot contradict the benchmark admission oracle."""
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=2, split_counts=(1, 0, 1))

    dataset = load_admission_dataset(dataset_dir)
    candidates = {
        row["candidate_id"]: row
        for row in (
            json.loads(line)
            for line in (dataset_dir / "candidate_updates.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }

    for examples in dataset.values():
        for example in examples:
            assert candidates[example.candidate_id]["gold_hard_gate"] is example.hard_eligible


def test_loader_rejects_history_leakage_across_splits(tmp_path: Path) -> None:
    """A history assigned to multiple splits must fail before training."""
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=2, split_counts=(1, 0, 1))
    candidate_path = dataset_dir / "candidate_updates.jsonl"
    candidates = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    candidates[0]["split"] = "test"
    candidate_path.write_text(
        "".join(json.dumps(candidate) + "\n" for candidate in candidates),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="appears in both"):
        load_admission_dataset(dataset_dir)


def test_loader_rejects_features_that_disagree_with_history(tmp_path: Path) -> None:
    """Precomputed feature fields cannot silently act as labels in disguise."""
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=2, split_counts=(1, 0, 1))
    candidate_path = dataset_dir / "candidate_updates.jsonl"
    candidates = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    candidates[0]["candidate_features"]["active_conflict_count"] = 99
    candidate_path.write_text(
        "".join(json.dumps(candidate) + "\n" for candidate in candidates),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="features disagree"):
        load_admission_dataset(dataset_dir)


def test_confidence_stress_is_deterministic_and_changes_only_observed_scores(
    tmp_path: Path,
) -> None:
    """Stress noise must preserve labels, hard-gate truth, and non-confidence evidence."""
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=2, split_counts=(1, 0, 1))
    examples = load_admission_dataset(dataset_dir)["test"]

    first = perturb_confidence_features(examples, stddev=0.15, seed=17, repeat=2)
    second = perturb_confidence_features(examples, stddev=0.15, seed=17, repeat=2)
    other_repeat = perturb_confidence_features(examples, stddev=0.15, seed=17, repeat=3)

    assert first == second
    assert first != other_repeat
    for original, stressed in zip(examples, first):
        assert stressed.candidate_id == original.candidate_id
        assert stressed.target == original.target
        assert stressed.hard_eligible is original.hard_eligible
        assert stressed.hard_reason == original.hard_reason
        assert stressed.features[1:4] == original.features[1:4]
        assert stressed.features[5] == float(
            math.isclose(stressed.features[0], stressed.features[4], abs_tol=1e-9)
        )
        assert stressed.features[6:] == original.features[6:]
        assert 0.0 <= stressed.features[0] <= 1.0
        assert 0.0 <= stressed.features[4] <= 1.0


def test_zero_confidence_noise_is_identity(tmp_path: Path) -> None:
    """A zero-noise control must not alter the effective evaluation examples."""
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=2, split_counts=(1, 0, 1))
    examples = load_admission_dataset(dataset_dir)["test"]

    assert perturb_confidence_features(examples, stddev=0.0, seed=17, repeat=0) == examples


def test_metrics_report_soft_eligible_accuracy_separately(tmp_path: Path) -> None:
    """Invariant hard rejects must not dilute the soft-admission stress headline."""
    dataset_dir = tmp_path / "dataset"
    generate_dataset(dataset_dir, history_count=2, split_counts=(1, 0, 1))
    examples = load_admission_dataset(dataset_dir)["test"]
    reject_predictions = [DECISION_LABELS.index("reject")] * len(examples)

    metrics = _classification_metrics(examples, reject_predictions)

    assert metrics["count"] == 9
    assert metrics["eligible_count"] == 6
    assert metrics["accuracy"] == pytest.approx(5 / 9)
    assert metrics["eligible_accuracy"] == pytest.approx(2 / 6)
