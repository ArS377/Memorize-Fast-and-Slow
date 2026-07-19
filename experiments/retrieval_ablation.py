from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

from experiments.retrieval_report import main as retrieval_report_main


MODES = ("sparse", "dense", "hybrid")
_FORBIDDEN_PASSTHROUGH = {
    "--retrieval-mode",
    "--results-dir",
    "--run-id",
    "--kg-session-noscallop",
    "--kg-session-scallop",
    "--dense-failure-policy",
}


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not safe:
        raise ValueError("ablation id must contain a safe character")
    return safe


def _contains_option(arguments: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def mode_command(
    *,
    mode: str,
    output_root: Path,
    ablation_id: str,
    run_all_arguments: Sequence[str],
    skip_kg_build: bool,
) -> List[str]:
    if mode not in MODES:
        raise ValueError(f"unsupported retrieval mode: {mode}")
    conflicting = sorted(
        option for option in _FORBIDDEN_PASSTHROUGH if _contains_option(run_all_arguments, option)
    )
    if conflicting:
        raise ValueError(
            "run_all passthrough cannot override controlled options: "
            + ", ".join(conflicting)
        )
    arguments = list(run_all_arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not _contains_option(arguments, "--cells"):
        arguments.extend(["--cells", "2,3,5,6"])
    command = [
        sys.executable,
        "-m",
        "experiments.run_all",
        *arguments,
        "--retrieval-mode",
        mode,
        "--dense-failure-policy",
        "error",
        "--results-dir",
        str(output_root / mode),
        "--run-id",
        f"{ablation_id}_{mode}",
        "--kg-session-noscallop",
        f"{ablation_id}_noscallop",
        "--kg-session-scallop",
        f"{ablation_id}_scallop",
    ]
    if skip_kg_build:
        command.append("--skip-kg-build")
    return command


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ablation-id", default=None)
    parser.add_argument("run_all_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    default_id = datetime.now(timezone.utc).strftime("retrieval_%Y%m%dT%H%M%SZ")
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
        "schema_version": "retrieval_ablation.v1",
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
