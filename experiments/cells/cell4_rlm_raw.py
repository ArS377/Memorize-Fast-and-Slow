"""Cell 4: RLM over raw LongBench-v2 context."""

from __future__ import annotations

from experiments._cli import run_cell

CELL_ID = 4
LABEL = "rlm_raw"


def main(argv=None):
    return run_cell(
        cell_id=CELL_ID, label=LABEL, kind="rlm", retrieval="raw",
        session_id=None, argv=argv,
    )


if __name__ == "__main__":
    main()
