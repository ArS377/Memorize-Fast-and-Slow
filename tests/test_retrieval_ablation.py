from pathlib import Path

import pytest

from experiments.retrieval_ablation import mode_command


def test_mode_command_uses_shared_sessions_and_controlled_modes(tmp_path: Path) -> None:
    sparse = mode_command(
        mode="sparse",
        output_root=tmp_path,
        ablation_id="ablation-1",
        run_all_arguments=["--input", "fixture.jsonl", "--limit", "5"],
        skip_kg_build=False,
    )
    hybrid = mode_command(
        mode="hybrid",
        output_root=tmp_path,
        ablation_id="ablation-1",
        run_all_arguments=["--input", "fixture.jsonl", "--limit", "5"],
        skip_kg_build=True,
    )

    assert sparse[sparse.index("--retrieval-mode") + 1] == "sparse"
    assert hybrid[hybrid.index("--retrieval-mode") + 1] == "hybrid"
    assert sparse[sparse.index("--kg-session-noscallop") + 1] == "ablation-1_noscallop"
    assert hybrid[hybrid.index("--kg-session-noscallop") + 1] == "ablation-1_noscallop"
    assert sparse[sparse.index("--dense-failure-policy") + 1] == "error"
    assert "--skip-kg-build" not in sparse
    assert "--skip-kg-build" in hybrid
    assert sparse[sparse.index("--cells") + 1] == "2,3,5,6"


def test_mode_command_rejects_overrides_of_controlled_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="controlled options"):
        mode_command(
            mode="dense",
            output_root=tmp_path,
            ablation_id="ablation-1",
            run_all_arguments=["--retrieval-mode", "sparse"],
            skip_kg_build=True,
        )
