from __future__ import annotations

from pathlib import Path

import pytest

from experiments.preference_stream_injection import (
    _preference_rows,
    derive_preference_injections_with_scallop,
    validate_preference_injection_result,
)
from experiments.synthetic_temporal_preferences import generate_dataset


def _rows(path: Path) -> list[dict]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_actual_scallop_derives_source_grounded_preference_pairs(tmp_path: Path) -> None:
    pytest.importorskip("scallopy")
    paths = generate_dataset(
        tmp_path,
        history_count=3,
        split_counts=(1, 1, 1),
        hardness_profile="anti_shortcut_stream_v2",
    )
    events = [
        event
        for event in _rows(paths["events"])
        if event["history_id"] == "history-003"
    ]

    result = derive_preference_injections_with_scallop(events)

    assert result["engine"] == "scallopy"
    assert result["rule_version"] == "preference_stream.v1"
    assert result["injections"] == [
        {
            "kind": "preference_change",
            "source_event_ids": [
                "history-003-add",
                "history-003-transition",
                "history-003-lineage-alias",
            ],
        },
        {
            "kind": "preference_incongruity",
            "source_event_ids": [
                "history-003-indirect-source",
                "history-003-direct-correction",
                "history-003-lineage-alias",
            ],
        },
    ]
    assert not any(
        key in injection
        for injection in result["injections"]
        for key in {"answer", "gold", "prediction", "object", "text"}
    )


def test_typed_identity_is_separate_from_preference_rows(tmp_path: Path) -> None:
    paths = generate_dataset(
        tmp_path,
        history_count=1,
        split_counts=(0, 0, 1),
        hardness_profile="anti_shortcut_interleaved_v3",
    )
    events = _rows(paths["events"])

    preference_rows, _, identity_rows = _preference_rows(events)

    assert identity_rows == [("history-001-lineage-alias", "subject-001")]
    preference_event_ids = {row[0] for row in preference_rows}
    assert "history-001-lineage-alias" not in preference_event_ids
    assert "history-001-preference-change-probe" not in preference_event_ids


@pytest.mark.parametrize("forbidden_key", ["answer", "gold", "prediction", "object", "text"])
def test_injection_contract_rejects_answer_bearing_payloads(forbidden_key: str) -> None:
    result = {
        "engine": "scallopy",
        "scallopy_version": "0.2.4",
        "rule_version": "preference_stream.v1",
        "injections": [
            {
                "kind": "preference_change",
                "source_event_ids": [
                    "history-001-add",
                    "history-001-transition",
                    "history-001-lineage-alias",
                ],
                forbidden_key: "leak",
            }
        ],
    }

    with pytest.raises(ValueError, match="forbidden response keys"):
        validate_preference_injection_result(
            result,
            causal_event_ids={
                "history-001-add",
                "history-001-transition",
                "history-001-lineage-alias",
            },
        )
