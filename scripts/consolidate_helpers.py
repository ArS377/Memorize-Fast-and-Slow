#!/usr/bin/env python3
"""Generate a consolidation plan for duplicated helper logic in the repo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_repo import audit_duplicates


def build_consolidation_plan(root: Path | str | None = None) -> Dict[str, Any]:
    repo_root = Path(root or Path(__file__).resolve().parent.parent).resolve()
    audit = audit_duplicates(repo_root)
    helpers: List[Dict[str, Any]] = []

    for item in audit["duplicate_helpers"]:
        helpers.append({
            "name": item["name"],
            "locations": item["files"],
            "target": "experiments/common.py",
            "action": "import from shared helper module",
        })

    return {
        "root": str(repo_root),
        "recommended_module": "experiments/common.py",
        "helpers": helpers,
        "notes": [
            "Prefer the shared helpers in experiments/common.py for question formatting, seed extraction, and fact formatting.",
            "Keep any file-specific logic in the calling script or module rather than re-implementing it in multiple entrypoints.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--json", action="store_true", help="Print the consolidation plan as JSON")
    args = parser.parse_args()

    plan = build_consolidation_plan(args.root)
    if args.json:
        print(json.dumps(plan, indent=2))
        return

    print(f"Consolidation plan for {plan['root']}")
    print("-" * 72)
    print(f"Recommended module: {plan['recommended_module']}")
    for helper in plan["helpers"]:
        print(f"- {helper['name']}: {', '.join(helper['locations'])}")
    print()
    for note in plan["notes"]:
        print(f"Note: {note}")


if __name__ == "__main__":
    main()
