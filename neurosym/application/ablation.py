from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class AblationRunResult:
    statuses: Dict[str, int]
    report_inputs: List[Path]

    @property
    def completed(self) -> bool:
        return bool(self.statuses) and all(code == 0 for code in self.statuses.values())


class AblationRunner:
    def __init__(
        self,
        *,
        modes: Sequence[str],
        output_root: Path,
        command_factory: Callable[[str, bool], Sequence[str]],
        process_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.modes = tuple(modes)
        self.output_root = Path(output_root)
        self.command_factory = command_factory
        self.process_runner = process_runner

    def run(self) -> AblationRunResult:
        statuses: Dict[str, int] = {}
        report_inputs: List[Path] = []
        for index, mode in enumerate(self.modes):
            command = list(self.command_factory(mode, index > 0))
            completed = self.process_runner(command, check=False)
            statuses[mode] = int(completed.returncode)
            if completed.returncode != 0:
                break
            retrieval_eval = self.output_root / mode / "retrieval_eval.jsonl"
            if retrieval_eval.exists():
                report_inputs.append(retrieval_eval)
        return AblationRunResult(statuses=statuses, report_inputs=report_inputs)


def safe_ablation_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not safe:
        raise ValueError("ablation id must contain a safe character")
    return safe


def contains_option(arguments: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def build_mode_command(
    *,
    mode: str,
    allowed_modes: Sequence[str],
    output_root: Path,
    ablation_id: str,
    run_all_arguments: Sequence[str],
    skip_kg_build: bool,
    forbidden_passthrough: Sequence[str],
) -> List[str]:
    if mode not in set(allowed_modes):
        raise ValueError(f"unsupported retrieval mode: {mode}")
    conflicting = sorted(
        option
        for option in forbidden_passthrough
        if contains_option(run_all_arguments, option)
    )
    if conflicting:
        raise ValueError(
            "run_all passthrough cannot override controlled options: "
            + ", ".join(conflicting)
        )
    arguments = list(run_all_arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not contains_option(arguments, "--cells"):
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
        str(Path(output_root) / mode),
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


def execute_ablation(
    *,
    modes: Sequence[str],
    output_root: Path,
    ablation_id: str,
    schema_version: str,
    command_factory: Callable[[str, bool], Sequence[str]],
    report: Callable[[List[str]], Any],
    process_runner: Callable[..., Any] = subprocess.run,
) -> AblationRunResult:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    result = AblationRunner(
        modes=modes,
        output_root=root,
        command_factory=command_factory,
        process_runner=process_runner,
    ).run()
    metadata = {
        "schema_version": schema_version,
        "ablation_id": ablation_id,
        "modes": list(modes),
        "shared_kg_sessions": {
            "noscallop": f"{ablation_id}_noscallop",
            "scallop": f"{ablation_id}_scallop",
        },
        "statuses": result.statuses,
    }
    (root / "ablation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if len(result.report_inputs) == len(modes):
        report_arguments: List[str] = []
        for path in result.report_inputs:
            report_arguments.extend(["--input", str(path)])
        report_arguments.extend(["--output-dir", str(root)])
        report(report_arguments)
    return result
