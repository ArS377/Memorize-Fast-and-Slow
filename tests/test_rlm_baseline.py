from __future__ import annotations

import ast
from pathlib import Path


def test_rlm_baseline_imports_json_for_jsonl_output() -> None:
    path = Path(__file__).parents[1] / "legacy" / "rlm_baseline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    assert any(
        isinstance(node, ast.Import) and any(alias.name == "json" for alias in node.names)
        for node in tree.body
    )
