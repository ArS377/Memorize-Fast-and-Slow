from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

from experiments.retrieval_report import main as retrieval_report_main
from neurosym.application.ablation import (
    build_mode_command,
    contains_option,
    execute_ablation,
    safe_ablation_id,
)


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
    return safe_ablation_id(value)


def _contains_option(arguments: Sequence[str], option: str) -> bool:
    return contains_option(arguments, option)


def mode_command(
    *,
    mode: str,
    output_root: Path,
    ablation_id: str,
    run_all_arguments: Sequence[str],
    skip_kg_build: bool,
) -> List[str]:
    return build_mode_command(
        mode=mode,
        allowed_modes=(*MODES, "dense_ppr"),
        output_root=output_root,
        ablation_id=ablation_id,
        run_all_arguments=run_all_arguments,
        skip_kg_build=skip_kg_build,
        forbidden_passthrough=tuple(_FORBIDDEN_PASSTHROUGH),
    )


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
    try:
        result = execute_ablation(
            modes=MODES,
            output_root=output_root,
            ablation_id=ablation_id,
            schema_version="retrieval_ablation.v1",
            command_factory=lambda mode, skip: mode_command(
                mode=mode,
                output_root=output_root,
                ablation_id=ablation_id,
                run_all_arguments=args.run_all_arguments,
                skip_kg_build=skip,
            ),
            report=retrieval_report_main,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if any(code != 0 for code in result.statuses.values()) or len(result.statuses) != len(MODES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
