#!/usr/bin/env python3
"""Audit the repository for duplicate helper logic that should be consolidated."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

DUPLICATE_HELPERS = [
    "extract_letter",
    "format_question",
    "extract_seed_entities",
    "load_examples",
    "load_facts_from_jsonl",
    "format_facts_from_jsonl",
]

DEFAULT_TARGET_FILES = [
    Path("experiments/common.py"),
    Path("experiments/graph_context.py"),
    Path("rlm_baseline.py"),
    Path("rlm_graph_baseline.py"),
    Path("evaluate.py"),
]


def _resolve_targets(root: Path) -> List[Path]:
    targets: List[Path] = []
    for rel_path in DEFAULT_TARGET_FILES:
        path = root / rel_path
        if path.exists():
            targets.append(path)
    return targets


def _find_function(path: Path, name: str) -> bool:
    text = path.read_text(encoding="utf-8")
    pattern = rf"^\s*def\s+{re.escape(name)}\s*\("  # noqa: W605
    return bool(re.search(pattern, text, flags=re.MULTILINE))


def audit_duplicates(root: Path | str | None = None) -> Dict[str, Any]:
    repo_root = Path(root or Path(__file__).resolve().parent.parent).resolve()
    targets = _resolve_targets(repo_root)
    duplicate_helpers: List[Dict[str, Any]] = []

    for helper_name in DUPLICATE_HELPERS:
        files = [str(path.resolve()) for path in targets if _find_function(path, helper_name)]
        if len(files) > 1:
            duplicate_helpers.append({
                "name": helper_name,
                "files": files,
            })

    return {
        "root": str(repo_root),
        "target_files": [str(path.resolve()) for path in targets],
        "duplicate_helpers": duplicate_helpers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--json", action="store_true", help="Print the audit report as JSON")
    args = parser.parse_args()

    report = audit_duplicates(args.root)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Repo audit for {report['root']}")
    print("-" * 72)
    if not report["duplicate_helpers"]:
        print("No duplicate helper definitions found.")
        return

    for item in report["duplicate_helpers"]:
        print(f"{item['name']}:")
        for path in item["files"]:
            print(f"  - {path}")
        print()


if __name__ == "__main__":
    main()
