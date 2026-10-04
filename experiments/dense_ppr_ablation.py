from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from experiments.retrieval_ablation import mode_command
from experiments.retrieval_report import main as retrieval_report_main
from neurosym.application.ablation import execute_ablation, safe_ablation_id


MODES = ("sparse", "dense", "hybrid", "dense_ppr")


def _safe_id(value: str) -> str:
    return safe_ablation_id(value)


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
    try:
        result = execute_ablation(
            modes=MODES,
            output_root=output_root,
            ablation_id=ablation_id,
            schema_version="dense_ppr_ablation.v1",
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
