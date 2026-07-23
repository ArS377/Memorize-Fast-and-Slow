from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from experiments.retrieval_ablation import mode_command
from experiments.retrieval_report import main as retrieval_report_main


MODES = ("sparse", "dense", "hybrid", "dense_ppr")


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not safe:
        raise ValueError("ablation id must contain a safe character")
    return safe


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the opt-in dense-PPR experiment without changing the standard "
            "three-mode retrieval ablation."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ablation-id", default=None)
    parser.add_argument("run_all_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    default_id = datetime.now(timezone.utc).strftime("dense_ppr_%Y%m%dT%H%M%SZ")
    try:
        ablation_id = _safe_id(args.ablation_id or default_id)
    except ValueError as exc:
        parser.error(str(exc))

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    statuses = {}
    report_inputs: List[Path] = []
    for index, mode in enumerate(MODES):
        try:
            command = mode_command(
                mode=mode,
                output_root=output_root,
                ablation_id=ablation_id,
                run_all_arguments=args.run_all_arguments,
                skip_kg_build=index > 0,
            )
        except ValueError as exc:
            parser.error(str(exc))
        completed = subprocess.run(command, check=False)
        statuses[mode] = completed.returncode
        if completed.returncode != 0:
            break
        retrieval_eval = output_root / mode / "retrieval_eval.jsonl"
        if retrieval_eval.exists():
            report_inputs.append(retrieval_eval)

    metadata = {
        "schema_version": "dense_ppr_ablation.v1",
        "ablation_id": ablation_id,
        "modes": list(MODES),
        "shared_kg_sessions": {
            "noscallop": f"{ablation_id}_noscallop",
            "scallop": f"{ablation_id}_scallop",
        },
        "statuses": statuses,
    }
    (output_root / "ablation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if len(report_inputs) == len(MODES):
        report_args: List[str] = []
        for path in report_inputs:
            report_args.extend(["--input", str(path)])
        report_args.extend(["--output-dir", str(output_root)])
        retrieval_report_main(report_args)
    if any(code != 0 for code in statuses.values()) or len(statuses) != len(MODES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
