"""Cell 8: Flat LLM over dense-only chunk retrieval (no KG, no graph)."""

from __future__ import annotations

import sys

from experiments._cli import run_cell

CELL_ID = 8
LABEL = "flat_dense"


def main(argv=None):
    argv = list(argv) if argv is not None else sys.argv[1:]
    if "--chunk-retrieval-mode" not in argv:
        argv = ["--chunk-retrieval-mode", "dense", *argv]
    return run_cell(
        cell_id=CELL_ID, label=LABEL, kind="flat", retrieval="chunk",
        session_id=None, argv=argv,
    )


if __name__ == "__main__":
    main()
