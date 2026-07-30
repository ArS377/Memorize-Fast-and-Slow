from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from neurosym.domain.retrieval_config import RetrievalConfig
from experiments.run_all import _common_cell_args, _materialize_pilot_input


def test_materialize_pilot_input_freezes_one_sorted_shared_slice(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {"_id": "c", "context": "third"},
        {"_id": "a", "context": "first"},
        {"_id": "b", "context": "second"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    frozen = tmp_path / "run" / "pilot_input.jsonl"

    metadata = _materialize_pilot_input(source, frozen, limit=2, seed=0)

    selected = [json.loads(line) for line in frozen.read_text().splitlines()]
    assert [row["_id"] for row in selected] == ["a", "b"]
    assert metadata["example_count"] == 2
    assert metadata["example_ids"] == ["a", "b"]
    assert metadata["path"] == str(frozen)
    assert len(metadata["sha256"]) == 64


def test_cell6_arguments_default_to_dense_ppr(tmp_path: Path) -> None:
    args = SimpleNamespace(
        input=tmp_path / "input.jsonl",
        pilot_input=tmp_path / "pilot.jsonl",
        limit=1,
        seed=0,
        model="Qwen/Qwen3-4B",
        vllm_base_url="http://localhost:8000/v1",
        results_dir=tmp_path / "results",
        run_id="run",
        neo4j_uri=None,
        neo4j_user=None,
        kg_sessions={5: "session", 6: "session"},
        hops=2,
        limit_triples=50,
        memory_scope="example",
        retrieval_mode="hybrid",
        embedding_model="fake/bge",
        embedding_device="cpu",
        embedding_batch_size=32,
        dense_index_root=tmp_path / "indexes",
        dense_failure_policy="error",
        rrf_k=60,
        embedding_revision=None,
        source_session=[],
        scallop_validator_url=None,
        max_depth=2,
        max_iterations=3,
        max_tokens=64000,
        fixed_kg_retrieval=True,
    )

    cell6 = _common_cell_args(args, 6)
    other_cell = _common_cell_args(args, 5)

    assert cell6[cell6.index("--retrieval-mode") + 1] == "dense_ppr"
    assert "--ppr-seed-count" in cell6
    assert other_cell[other_cell.index("--retrieval-mode") + 1] == "hybrid"
    assert not any(value.startswith("--ppr-") for value in other_cell)
    assert "ppr" not in RetrievalConfig().to_dict()
    assert RetrievalConfig(mode="dense_ppr").to_dict()["ppr"]["seed_count"] == 20
