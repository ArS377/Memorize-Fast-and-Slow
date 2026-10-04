from __future__ import annotations

from inspect import signature
from pathlib import Path

import experiments.build_kg_facts_file as module


def test_facts_file_builder_api_defaults_to_exhaustive_chunks() -> None:
    assert (
        signature(module.build_kg_facts_file)
        .parameters["max_chunks_per_example"]
        .default
        is None
    )


def test_facts_file_builder_cli_defaults_to_exhaustive_chunks(monkeypatch) -> None:
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return Path("unused.jsonl")

    monkeypatch.setattr(module, "build_kg_facts_file", fake_build)

    module.main(["--session", "audit"])

    assert captured["max_chunks_per_example"] is None
